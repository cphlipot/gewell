#include "gewell/fp8_linear.h"
#include "gewell/bf16_primitives.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using gewell::fp8::Plan;
using gewell::fp8::Weight;

void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct Buffer {
  explicit Buffer(std::size_t bytes) { cuda_check(cudaMalloc(&data, bytes)); }
  ~Buffer() { cudaFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  template <typename T> T* as() { return static_cast<T*>(data); }
  void* data{};
};

// Independent E4M3FN decoding and exhaustive nearest-even conversion. No
// CUDA FP8 conversion intrinsics or production quantization code in the oracle.
float decode(unsigned bits) {
  const unsigned magnitude = bits & 127;
  const unsigned exponent = magnitude / 8, mantissa = magnitude % 8;
  const float value = exponent == 0
      ? std::ldexp(float(mantissa), -9)
      : std::ldexp(1.0F + float(mantissa) / 8, int(exponent) - 7);
  return bits & 128 ? -value : value;
}

float nearest(float value) {
  const bool negative = std::signbit(value);
  value = std::abs(value);
  unsigned best = 0;
  if (value >= 448.0F) return negative ? -448.0F : 448.0F;
  for (unsigned i = 1; i <= 126; ++i) {
    const float distance = std::abs(value - decode(i));
    const float previous = std::abs(value - decode(best));
    if (distance < previous || (distance == previous && !(i & 1))) best = i;
  }
  return negative ? -decode(best) : decode(best);
}

void verify_guard(const Buffer& buffer, std::size_t offset) {
  std::array<unsigned char, 256> guard{};
  cuda_check(cudaMemcpy(guard.data(), static_cast<unsigned char*>(buffer.data) + offset,
                         guard.size(), cudaMemcpyDeviceToHost));
  if (!std::all_of(guard.begin(), guard.end(), [](unsigned char x) { return x == 0xCD; }))
    throw std::runtime_error("FP8 wrote beyond the caller's buffer");
}

struct Matrix {
  Matrix(unsigned k, unsigned n)
      : k(k), n(n), bytes(std::size_t(k) * n), device(bytes.size()) {
    for (unsigned row = 0; row < n; ++row)
      for (unsigned col = 0; col < k; ++col) {
        const unsigned code = 16 + (col * 13 + row * 7 + col / 17) % 89;
        bytes[std::size_t(row) * k + col] = code | (((col * 3 + row + col / 31) & 1) << 7);
      }
    upload();
  }
  void upload() {
    cuda_check(cudaMemcpy(device.data, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
  }
  unsigned k, n;
  std::vector<unsigned char> bytes;
  Buffer device;
};

void verify_output(Matrix& matrix, const std::vector<__nv_bfloat16>& input,
                   Buffer& output, unsigned rows, float input_scale,
                   float weight_scale, bool exact = false,
                   const std::vector<float>* channel_scales = nullptr) {
  const unsigned k = matrix.k, n = matrix.n;
  std::vector<unsigned> checked_rows{0, rows / 2, rows - 1};
  std::vector<unsigned> checked_cols{0, 15, n / 2, n - 1};
  if (k <= 256) {
    checked_rows.clear();
    checked_cols.clear();
    for (unsigned row = 0; row < rows; ++row) checked_rows.push_back(row);
    for (unsigned col = 0; col < n; ++col) checked_cols.push_back(col);
  }
  for (unsigned row : checked_rows) {
    std::vector<float> quantized(k);
    for (unsigned j = 0; j < k; ++j)
      quantized[j] = nearest(float(input[std::size_t(row) * k + j]) * (1.0F / input_scale));
    std::vector<__nv_bfloat16> actual(n);
    cuda_check(cudaMemcpy(actual.data(), output.as<__nv_bfloat16>() + std::size_t(row) * n,
                           n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    for (unsigned col : checked_cols) {
      double sum = 0;
      for (unsigned j = 0; j < k; ++j)
        sum += double(quantized[j]) * decode(matrix.bytes[std::size_t(col) * k + j]);
      const float alpha = channel_scales ? (*channel_scales)[col] : input_scale * weight_scale;
      const float expected = float(__float2bfloat16(float(sum * alpha)));
      const float got = float(actual[col]);
      // One BF16 ulp permits FP32 accumulation orders differing at a BF16 tie.
      const float tolerance = exact ? 0.0F : std::max(0.008F * std::abs(expected), 0.001F);
      if (!std::isfinite(got) || std::abs(got - expected) > tolerance)
        throw std::runtime_error("FP8 oracle mismatch M=" + std::to_string(rows) +
            " K=" + std::to_string(k) + " N=" + std::to_string(n) +
            " row=" + std::to_string(row) + " col=" + std::to_string(col) +
            " got=" + std::to_string(got) + " expected=" + std::to_string(expected));
    }
  }
}

void check_shape(cublasLtHandle_t handle, Matrix& matrix, unsigned rows,
                  bool graph = false, bool joined = false) {
  const unsigned k = matrix.k, n = matrix.n;
  constexpr float input_scale = 0.0113F, weight_scale = 0.0737F;
  std::vector<__nv_bfloat16> input(std::size_t(rows) * k);
  auto fill = [&](unsigned iteration) {
    for (std::size_t i = 0; i < input.size(); ++i) {
      float value = float(int((i * 31 + i / 13 + iteration * 17) % 127) - 63) / 8;
      if ((i / 16) % 17 == 0) value = 0;
      if ((i / 16) % 17 == 1) value *= 0.000001F;
      if ((i / 16) % 17 == 2) value *= 128;
      input[i] = __float2bfloat16(value);
    }
  };
  Buffer device_input(input.size() * sizeof(__nv_bfloat16));
  const std::size_t output_bytes = std::size_t(rows) * n * sizeof(__nv_bfloat16);
  Buffer output(output_bytes + 256);
  Plan plan(handle, rows, k, n, joined);
  Buffer device_scales(n * sizeof(float));
  std::vector<float> scales(n);
  for (unsigned col = 0; col < n; ++col)
    scales[col] = input_scale * (weight_scale * (0.5F + 0.375F * (col / (n / 3 + 1))));
  cuda_check(cudaMemcpy(device_scales.data, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice));
  if (plan.scratch_bytes() > gewell::fp8::scratch_upper_bound_bytes(rows, k))
    throw std::runtime_error("FP8 plan exceeds advertised scratch bound");
  Buffer scratch(plan.scratch_bytes() + 256);
  cuda_check(cudaMemset(output.data, 0xCD, output_bytes + 256));
  cuda_check(cudaMemset(scratch.data, 0xCD, plan.scratch_bytes() + 256));
  Weight weight{matrix.device.as<std::uint8_t>(), input_scale, weight_scale};
  if (joined) weight.channel_scales = device_scales.as<float>();
  cudaStream_t stream;
  cuda_check(cudaStreamCreate(&stream));
  cudaGraph_t captured{};
  cudaGraphExec_t executable{};
  if (graph) {
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    plan.run(device_input.as<__nv_bfloat16>(), weight, output.as<__nv_bfloat16>(),
             scratch.data, plan.scratch_bytes(), stream);
    cuda_check(cudaStreamEndCapture(stream, &captured));
    cuda_check(cudaGraphInstantiate(&executable, captured, 0));
  }
  for (unsigned iteration = 0; iteration < (graph ? 3u : 1u); ++iteration) {
    fill(iteration);
    if (iteration) {
      for (auto& byte : matrix.bytes) byte ^= 128;
      matrix.upload();
    }
    cuda_check(cudaMemcpy(device_input.data, input.data(), input.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    if (graph) cuda_check(cudaGraphLaunch(executable, stream));
    else plan.run(device_input.as<__nv_bfloat16>(), weight, output.as<__nv_bfloat16>(),
                  scratch.data, plan.scratch_bytes(), stream);
    cuda_check(cudaStreamSynchronize(stream));
    verify_output(matrix, input, output, rows, input_scale, weight_scale, false,
                   joined ? &scales : nullptr);
    verify_guard(output, output_bytes);
    verify_guard(scratch, plan.scratch_bytes());
  }
  if (graph) {
    cuda_check(cudaGraphExecDestroy(executable));
    cuda_check(cudaGraphDestroy(captured));
  }
  cuda_check(cudaStreamDestroy(stream));
  std::cout << "native FP8 M=" << rows << " K=" << k << " N=" << n
            << (joined ? " joined channel scales" : "")
            << (graph ? " changed-input/weight graph replay" : "") << ": PASS\n";
}

void check_fused_gelu(cublasLtHandle_t handle) {
  constexpr unsigned k = 256, n = 256;
  Matrix matrix(k, n);
  for (unsigned rows : {1u, 3u, 32u, 1024u}) {
    Plan plan(handle, rows, k, n);
    Buffer input(std::size_t(rows) * 2 * k * 2), product(std::size_t(rows) * k * 2);
    Buffer separate(std::size_t(rows) * n * 2), fused(std::size_t(rows) * n * 2);
    Buffer scratch(plan.scratch_bytes()), reference(plan.scratch_bytes());
    Weight w{matrix.device.as<std::uint8_t>(), 0.0737F, 0.0113F};
    cudaStream_t stream; cuda_check(cudaStreamCreate(&stream));
    cudaGraph_t graph; cudaGraphExec_t executable;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    plan.run(input.as<__nv_bfloat16>(), w, fused.as<__nv_bfloat16>(),
             scratch.data, plan.scratch_bytes(), stream, gewell::fp8::InputTransform::gelu_tanh_multiply);
    cuda_check(cudaStreamEndCapture(stream, &graph));
    cuda_check(cudaGraphInstantiate(&executable, graph, 0));
    for (unsigned iteration = 0; iteration < 3; ++iteration) {
      std::vector<__nv_bfloat16> host(std::size_t(rows) * 2 * k);
      for (std::size_t i = 0; i < host.size(); ++i) {
        float x = float(int((i * 31 + iteration * 17) % 127) - 63) / 8;
        if (i % 7 == 0) x *= 100;
        if (i % 11 == 0) x = 0;
        host[i] = __float2bfloat16(x);
      }
      cuda_check(cudaMemcpy(input.data, host.data(), host.size() * 2, cudaMemcpyHostToDevice));
      gewell::bf16_primitives::gelu_tanh_multiply_interleaved(
          input.as<__nv_bfloat16>(), product.as<__nv_bfloat16>(), rows, k, stream);
      plan.run(product.as<__nv_bfloat16>(), w, separate.as<__nv_bfloat16>(),
               reference.data, plan.scratch_bytes(), stream);
      cuda_check(cudaGraphLaunch(executable, stream));
      cuda_check(cudaStreamSynchronize(stream));
      for (bool packed : {true, false}) {
        const auto bytes = std::size_t(rows) * (packed ? k : n * 2);
        std::vector<unsigned char> a(bytes), b(bytes);
        cuda_check(cudaMemcpy(a.data(), packed ? reference.data : separate.data, bytes, cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(b.data(), packed ? scratch.data : fused.data, bytes, cudaMemcpyDeviceToHost));
        if (a != b) throw std::runtime_error("FP8 fused GELU changed packing/output");
      }
    }
    cuda_check(cudaGraphExecDestroy(executable));cuda_check(cudaGraphDestroy(graph));
    cuda_check(cudaStreamDestroy(stream));
  }
  std::cout << "FP8 GELU/product packing exact, changed-input graph replay: PASS\n";
}

void check_rounding_and_scales(cublasLtHandle_t handle) {
  constexpr unsigned k = 256, n = 256;
  Matrix matrix(k, n);
  std::fill(matrix.bytes.begin(), matrix.bytes.end(), 0);
  for (unsigned i = 0; i < n; ++i) matrix.bytes[i * k + i] = 0x38;  // Identity.
  matrix.upload();
  Plan plan(handle, 1, k, n);
  Buffer scratch(plan.scratch_bytes()), device_input(k * 2), output(n * 2);
  const std::array<float, 8> extremes{
      0.0F, -0.0F, 1.0F / 2048, 3.0F / 2048, 464.0F, -464.0F,
      std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
  for (float scale : {1.0F, 0.125F, 0.0737F, 8.0F}) {
    std::vector<__nv_bfloat16> input(k);
    for (unsigned i = 0; i < k; ++i) {
      const unsigned code = (i / 2) % 126;
      float value = (decode(code) + decode(code + 1)) / 2;
      if (i & 1) value = -value;
      // Multiplication by a power of two preserves exact midpoint cases.
      input[i] = __float2bfloat16(value * scale);
    }
    for (unsigned i = 0; i < extremes.size(); ++i)
      input[i] = __float2bfloat16(extremes[i]);
    cuda_check(cudaMemcpy(device_input.data, input.data(), k * 2, cudaMemcpyHostToDevice));
    plan.run(device_input.as<__nv_bfloat16>(), {matrix.device.as<std::uint8_t>(), scale, 0.25F},
             output.as<__nv_bfloat16>(), scratch.data, plan.scratch_bytes());
    verify_output(matrix, input, output, 1, scale, 0.25F, true);
  }
  // Every signed finite weight code, with a unit input selecting one weight.
  std::vector<__nv_bfloat16> input(k, __float2bfloat16(0));
  input[0] = __float2bfloat16(1);
  for (unsigned row = 0; row < n; ++row)
    matrix.bytes[row * k] = (row % 127) | ((row / 127) % 2 * 128);
  matrix.upload();
  cuda_check(cudaMemcpy(device_input.data, input.data(), k * 2, cudaMemcpyHostToDevice));
  plan.run(device_input.as<__nv_bfloat16>(), {matrix.device.as<std::uint8_t>(), 1.0F, 0.0737F},
           output.as<__nv_bfloat16>(), scratch.data, plan.scratch_bytes());
  verify_output(matrix, input, output, 1, 1.0F, 0.0737F, true);
  cuda_check(cudaMemset(device_input.data, 0, k * 2));
  std::fill(input.begin(), input.end(), __float2bfloat16(0));
  plan.run(device_input.as<__nv_bfloat16>(), {matrix.device.as<std::uint8_t>(), 0.0113F, 0.0737F},
           output.as<__nv_bfloat16>(), scratch.data, plan.scratch_bytes());
  verify_output(matrix, input, output, 1, 0.0113F, 0.0737F, true);
  std::cout << "FP8 nearest-even boundaries, subnormals, finite saturation, all weight codes, scales, zero: PASS\n";
}

template <typename F> void rejects(F&& function) {
  try { function(); } catch (const std::invalid_argument&) { return; }
  throw std::runtime_error("FP8 accepted invalid parameters");
}

void check_alignment(cublasLtHandle_t handle) {
  // Inputs promise only BF16 alignment, even with paired FP8 conversion.
  // Weights/output promise 16B alignment at every row count.
  for (const auto& shape : std::array<std::array<unsigned, 3>, 6>{{
           {1, 16, 16}, {3, 16, 16}, {16, 16, 16}, {17, 16, 16},
           {1024, 5376, 21504}, {2048, 5376, 21504}}}) {
    const unsigned m = shape[0], k = shape[1], n = shape[2];
    Buffer input(m * k * 2 + 2), weights(n * k + 16), output(m * n * 2 + 16);
    std::vector<__nv_bfloat16> values(m * k, __float2bfloat16(1));
    cuda_check(cudaMemcpy(input.as<__nv_bfloat16>() + 1, values.data(), values.size() * 2, cudaMemcpyHostToDevice));
    cuda_check(cudaMemset(weights.data, 0x38, n * k + 16));
    Plan plan(handle, m, k, n);
    Buffer scratch(plan.scratch_bytes());
    plan.run(input.as<__nv_bfloat16>() + 1, {weights.as<std::uint8_t>() + 16, 1, 1},
             output.as<__nv_bfloat16>() + 8, scratch.data, plan.scratch_bytes());
    std::vector<__nv_bfloat16> actual(m * n);
    cuda_check(cudaMemcpy(actual.data(), output.as<__nv_bfloat16>() + 8,
                           actual.size() * 2, cudaMemcpyDeviceToHost));
    for (const auto value : actual)
      if (float(value) != float(k)) throw std::runtime_error("FP8 16B-aligned buffers failed");
  }
  std::cout << "FP8 2B-aligned input and 16B-aligned weight/output pointers: PASS\n";
}

void check_invalid(cublasLtHandle_t handle) {
  rejects([&] { Plan p(nullptr, 1, 16, 16); });
  rejects([&] { Plan p(handle, 0, 16, 16); });
  rejects([&] { Plan p(handle, std::numeric_limits<unsigned>::max(), 16, 16); });
  rejects([&] { Plan p(handle, 1, 0, 16); });
  rejects([&] { Plan p(handle, 1, 17, 16); });
  rejects([&] { Plan p(handle, 1, 16, 0); });
  rejects([&] { Plan p(handle, 1, 16, 17); });
  Plan plan(handle, 1, 16, 16);
  Buffer input(32), weights(256), output(32), scratch(plan.scratch_bytes() + 256);
  const Weight valid{weights.as<std::uint8_t>(), 1.0F, 1.0F};
  auto run = [&](Weight weight, void* storage, std::size_t capacity) {
    plan.run(input.as<__nv_bfloat16>(), weight, output.as<__nv_bfloat16>(), storage, capacity);
  };
  rejects([&] { run(valid, scratch.data, plan.scratch_bytes() - 1); });
  rejects([&] { run(valid, scratch.as<char>() + 1, plan.scratch_bytes()); });
  rejects([&] { run(valid, nullptr, plan.scratch_bytes()); });
  rejects([&] { run({nullptr, 1, 1}, scratch.data, plan.scratch_bytes()); });
  rejects([&] { run({weights.as<std::uint8_t>() + 1, 1, 1}, scratch.data, plan.scratch_bytes()); });
  for (float scale : {0.0F, -1.0F, std::nanf(""), std::numeric_limits<float>::infinity()}) {
    rejects([&] { run({valid.data, scale, 1}, scratch.data, plan.scratch_bytes()); });
    rejects([&] { run({valid.data, 1, scale}, scratch.data, plan.scratch_bytes()); });
  }
  rejects([&] { run({valid.data, std::numeric_limits<float>::denorm_min(), 1}, scratch.data, plan.scratch_bytes()); });
  rejects([&] { run({valid.data, std::numeric_limits<float>::max(), 2}, scratch.data, plan.scratch_bytes()); });
  rejects([&] { run({valid.data, std::numeric_limits<float>::min(), std::numeric_limits<float>::min()}, scratch.data, plan.scratch_bytes()); });
  rejects([&] { plan.run(nullptr, valid, output.as<__nv_bfloat16>(), scratch.data, plan.scratch_bytes()); });
  rejects([&] { plan.run(input.as<__nv_bfloat16>(), valid, nullptr, scratch.data, plan.scratch_bytes()); });
  std::cout << "FP8 invalid dimensions, buffers, scratch, scales: PASS\n";
}

}  // namespace

int main() {
  try {
    int device = 0;
    const auto status = cudaGetDevice(&device);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) return 77;
    cuda_check(status);
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device));
    if (properties.major < 9 && !(properties.major == 8 && properties.minor >= 9)) return 77;
    cublasLtHandle_t handle;
    if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("create cuBLASLt handle");
    check_invalid(handle);
    check_alignment(handle);
    check_rounding_and_scales(handle);
    check_fused_gelu(handle);
    {
      Matrix smallest(16, 16);
      check_shape(handle, smallest, 1);
      Matrix small(256, 256);
      for (unsigned rows : {1u, 3u, 15u, 16u, 17u, 31u, 32u, 33u, 129u})
        check_shape(handle, small, rows, rows == 3);
      for (unsigned rows : {1u, 3u, 17u, 129u})
        check_shape(handle, small, rows, true, true);
    }
    for (unsigned n : {16384u, 18432u, 43008u}) {
      Matrix joined(5376, n);
      for (unsigned rows : {1u, 3u, 32u, 64u, 65u, 128u, 256u, 257u,
                            1023u, 1024u, 1025u, 2048u, 4096u})
        check_shape(handle, joined, rows, rows == 3 || rows == 1024, true);
    }
    // All distinct local/global text q/k/v/o and gate/up/down dimensions.
    for (const auto& shape : std::array<std::array<unsigned, 2>, 8>{{
             {5376, 8192}, {5376, 16384}, {5376, 4096}, {5376, 2048},
             {8192, 5376}, {16384, 5376}, {5376, 21504}, {21504, 5376}}}) {
      Matrix matrix(shape[0], shape[1]);
      for (unsigned rows : {1u, 3u, 16u, 17u, 32u, 64u, 128u, 1024u})
        check_shape(handle, matrix, rows, rows == 1 || rows == 1024);
      if (shape[0] == 5376 && shape[1] == 8192) {
        for (unsigned rows : {1023u, 1025u}) check_shape(handle, matrix, rows);
      }
    }
    cublasLtDestroy(handle);
    std::cout << "FP8 native quantization/matmul, unaligned row counts, guards, graph capture: PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
