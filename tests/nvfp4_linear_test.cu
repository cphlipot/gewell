#include "gewell/nvfp4_linear.h"
#include "gewell/models/gemma4/31b/sm120/nvfp4_projections.h"
#include "gewell/bf16_primitives.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using gewell::nvfp4::Plan;
using gewell::nvfp4::Weight;

void cuda_check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

struct Buffer {
  explicit Buffer(std::size_t bytes) { cuda_check(cudaMalloc(&data, bytes)); }
  ~Buffer() { cudaFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  template <typename T> T* as() { return static_cast<T*>(data); }
  void* data{};
};

float fp4(unsigned bits) {
  constexpr float values[]{0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
  return (bits & 8) ? -values[bits & 7] : values[bits & 7];
}

float fp8(unsigned bits) {
  const unsigned exponent = bits / 8;
  const unsigned mantissa = bits % 8;
  return exponent == 0 ? std::ldexp(float(mantissa), -9)
                       : std::ldexp(1.0F + float(mantissa) / 8, int(exponent) - 7);
}

// Independent nearest-even references, enumerating the representable values.
unsigned nearest_fp8(float value) {
  unsigned best = 0;
  for (unsigned i = 1; i <= 126; ++i) {
    const float distance = std::abs(value - fp8(i));
    const float previous = std::abs(value - fp8(best));
    if (distance < previous || (distance == previous && !(i & 1))) best = i;
  }
  return best;
}

float nearest_fp4(float value) {
  const bool negative = std::signbit(value);
  value = std::abs(value);
  unsigned best = 0;
  for (unsigned i = 1; i < 8; ++i) {
    const float distance = std::abs(value - fp4(i));
    const float previous = std::abs(value - fp4(best));
    if (distance < previous || (distance == previous && !(i & 1))) best = i;
  }
  return negative ? -fp4(best) : fp4(best);
}

void verify_guard(const Buffer& buffer, std::size_t offset) {
  std::array<unsigned char, 256> guard{};
  cuda_check(cudaMemcpy(guard.data(), static_cast<unsigned char*>(buffer.data) + offset,
                         guard.size(), cudaMemcpyDeviceToHost));
  if (!std::all_of(guard.begin(), guard.end(), [](unsigned char x) { return x == 0xCD; })) {
    throw std::runtime_error("NVFP4 wrote beyond the caller's buffer");
  }
}

struct Matrix {
  Matrix(unsigned k, unsigned n)
      : k(k), n(n), packed(std::size_t(k) * n / 2),
        scales(gewell::nvfp4::scale_storage_bytes(n, k)),
        device_packed(packed.size()), device_scales(scales.size()) {
    for (unsigned row = 0; row < n; ++row) {
      for (unsigned column = 0; column < k / 2; ++column) {
        const unsigned lo = (column * 13 + row * 7 + column / 17) % 16;
        const unsigned hi = (column * 3 + row * 11 + column / 31 + 5) % 16;
        packed[std::size_t(row) * (k / 2) + column] = lo | (hi << 4);
      }
      for (unsigned block = 0; block < k / 16; ++block) {
        scales[gewell::nvfp4::scale_offset(row, block, k)] = 16 + (row * 11 + block * 7) % 72;
      }
    }
    cuda_check(cudaMemcpy(device_packed.data, packed.data(), packed.size(), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(device_scales.data, scales.data(), scales.size(), cudaMemcpyHostToDevice));
  }
  unsigned k, n;
  std::vector<unsigned char> packed, scales;
  Buffer device_packed, device_scales;
};

void check_dequantize(bool graph) {
  constexpr unsigned k = 256, n = 256;
  Matrix matrix(k, n);
  // Cover every positive finite E4M3 code, zero/subnormal scales, both row
  // tiles, and every E2M1 nibble. Activation globals must have no effect.
  for (unsigned row = 0; row < n; ++row)
    for (unsigned block = 0; block < k / 16; ++block)
      matrix.scales[gewell::nvfp4::scale_offset(row, block, k)] = (row * 17 + block) % 127;
  cuda_check(cudaMemcpy(matrix.device_scales.data, matrix.scales.data(), matrix.scales.size(), cudaMemcpyHostToDevice));
  Buffer output(std::size_t(k) * n * 2 + 256);
  cuda_check(cudaMemset(output.data, 0xCD, std::size_t(k) * n * 2 + 256));
  Weight weight{matrix.device_packed.as<std::uint8_t>(), matrix.device_scales.as<std::uint8_t>(),
                std::nanf(""), 0.0737F};
  cudaStream_t stream;
  cuda_check(cudaStreamCreate(&stream));
  if (graph) {
    cudaGraph_t captured;
    cudaGraphExec_t executable;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    gewell::nvfp4::dequantize(weight, k, n, output.as<__nv_bfloat16>(), stream);
    cuda_check(cudaStreamEndCapture(stream, &captured));
    cuda_check(cudaGraphInstantiate(&executable, captured, 0));
    cuda_check(cudaGraphLaunch(executable, stream));
    cuda_check(cudaGraphLaunch(executable, stream));
    cuda_check(cudaStreamSynchronize(stream));
    cuda_check(cudaGraphExecDestroy(executable));
    cuda_check(cudaGraphDestroy(captured));
  } else {
    gewell::nvfp4::dequantize(weight, k, n, output.as<__nv_bfloat16>(), stream);
    cuda_check(cudaStreamSynchronize(stream));
  }
  cuda_check(cudaStreamDestroy(stream));
  std::vector<__nv_bfloat16> actual(std::size_t(k) * n);
  cuda_check(cudaMemcpy(actual.data(), output.data, actual.size() * 2, cudaMemcpyDeviceToHost));
  for (unsigned row = 0; row < n; ++row)
    for (unsigned column = 0; column < k; ++column) {
      const auto packed = matrix.packed[std::size_t(row) * (k / 2) + column / 2];
      const auto scale = matrix.scales[gewell::nvfp4::scale_offset(row, column / 16, k)];
      const float decoded_scale = fp8(scale) * weight.weight_scale;
      const auto expected = __float2bfloat16(fp4((packed >> ((column % 2) * 4)) & 15) * decoded_scale);
      if (std::memcmp(&actual[std::size_t(row) * k + column], &expected, sizeof(expected)))
        throw std::runtime_error("NVFP4 BF16 reconstruction differs from stored nibble/scale oracle");
    }
  verify_guard(output, actual.size() * 2);
  std::cout << "NVFP4 BF16 reconstruction exact, activation global ignored, guards" << (graph ? ", graph" : "") << ": PASS\n";
}

void check_vllm_rounding_boundaries(cublasLtHandle_t handle) {
  // Frozen outputs of installed vLLM 0.23.0 (91df0fad4) scaled_fp4_quant.
  // NVIDIA's layer 17/59 down globals expose FP8/FP4 rounding boundaries that
  // mathematically equivalent division and reassociation do not preserve.
  struct Fixture {
    float global;
    std::array<float, 16> input;
    std::array<unsigned char, 8> packed;
    unsigned char scale;
  };
  const std::array<Fixture, 2> fixtures{{
      {0.2708333432674408F,
       {-3.859375F, -2.296875F, -1.53125F, 2.765625F,
        2.046875F, -2.203125F, 0.08544921875F, 1.1484375F,
        -1.8984375F, 2.3125F, 3.25F, 1.3203125F,
        -0.154296875F, -0.2890625F, -2.8125F, -0.875F},
       {223, 108, 213, 48, 93, 70, 152, 190}, 66},
      {0.0424107164144516F,
       {2.375F, 0.416015625F, 0.94921875F, 0.33984375F,
        1.984375F, -1.234375F, 1.0703125F, 1.71875F,
        0.0194091796875F, 1.0546875F, -3.0625F, -2.328125F,
        -0.1552734375F, -0.15234375F, -0.890625F, -0.01531982421875F},
       {38, 20, 198, 84, 64, 239, 153, 139}, 84},
  }};
  constexpr unsigned k = 64, n = 128;
  // Each output independently selects one quantized input element, making the
  // FP4/scale fixture observable through the public linear API.
  std::vector<unsigned char> packed(n * k / 2, 0);
  for (unsigned col = 0; col < n; ++col) {
    const unsigned selected = col % k;
    packed[col * (k / 2) + selected / 2] = 2 << ((selected % 2) * 4);
  }
  Buffer weights(packed.size());
  Buffer scales(gewell::nvfp4::scale_storage_bytes(n, k));
  Buffer input(k * sizeof(__nv_bfloat16)), output(n * sizeof(__nv_bfloat16));
  cuda_check(cudaMemcpy(weights.data, packed.data(), packed.size(), cudaMemcpyHostToDevice));
  cuda_check(cudaMemset(scales.data, 0x38, gewell::nvfp4::scale_storage_bytes(n, k)));
  Plan plan(handle, 1, k, n);
  Buffer scratch(plan.scratch_bytes());
  for (const auto& fixture : fixtures) {
    std::array<__nv_bfloat16, k> values;
    for (unsigned j = 0; j < k; ++j) values[j] = __float2bfloat16(fixture.input[j % 16]);
    cuda_check(cudaMemcpy(input.data, values.data(), sizeof(values), cudaMemcpyHostToDevice));
    plan.run(input.as<__nv_bfloat16>(),
             {weights.as<std::uint8_t>(), scales.as<std::uint8_t>(), fixture.global, 1.0F},
             output.as<__nv_bfloat16>(), scratch.data, plan.scratch_bytes());
    std::array<__nv_bfloat16, n> actual;
    cuda_check(cudaMemcpy(actual.data(), output.data, sizeof(actual), cudaMemcpyDeviceToHost));
    for (unsigned col = 0; col < n; ++col) {
      const unsigned j = col % 16;
      const auto bits = (fixture.packed[j / 2] >> ((j % 2) * 4)) & 15;
      const float expected = float(__float2bfloat16(fp4(bits) * fp8(fixture.scale) * fixture.global));
      if (float(actual[col]) != expected) {
        throw std::runtime_error("NVFP4 differs from frozen vLLM quantization boundary fixture");
      }
    }
  }
  std::cout << "vLLM calibrated-global FP8/FP4 rounding fixtures: PASS\n";
}

void check_shape(cublasLtHandle_t handle, Matrix& matrix, unsigned rows,
                 bool graph = false) {
  const unsigned k = matrix.k, n = matrix.n;
  const float input_global = 0.0113F, weight_global = 0.0737F;
  std::vector<__nv_bfloat16> input(std::size_t(rows) * k);
  for (unsigned row = 0; row < rows; ++row) {
    for (unsigned col = 0; col < k; ++col) {
      float value = float(int((col * 31 + row * 7 + col / 13) % 127) - 63) / 8;
      if ((col / 16) % 17 == 0) value = 0.0F;
      if ((col / 16) % 17 == 1) value *= 0.000001F;  // Underflowed scale.
      if ((col / 16) % 17 == 2) value *= 128;       // Saturated scale/FP4.
      input[std::size_t(row) * k + col] = __float2bfloat16(value);
    }
  }
  Buffer device_input(input.size() * sizeof(__nv_bfloat16));
  Buffer output(std::size_t(rows) * n * sizeof(__nv_bfloat16) + 256);
  Plan plan(handle, rows, k, n);
  if (plan.scratch_bytes() >
      gewell::nvfp4::scratch_upper_bound_bytes(rows, k, n)) {
    throw std::runtime_error("NVFP4 plan exceeds advertised scratch bound");
  }
  Buffer scratch(plan.scratch_bytes() + 256);
  cuda_check(cudaMemcpy(device_input.data, input.data(), input.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  cuda_check(cudaMemset(output.data, 0xCD, std::size_t(rows) * n * sizeof(__nv_bfloat16) + 256));
  cuda_check(cudaMemset(scratch.data, 0xCD, plan.scratch_bytes() + 256));
  Weight weight{matrix.device_packed.as<std::uint8_t>(), matrix.device_scales.as<std::uint8_t>(), input_global, weight_global};
  cudaStream_t stream;
  cuda_check(cudaStreamCreate(&stream));
  if (graph) {
    cudaGraph_t captured;
    cudaGraphExec_t executable;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    plan.run(device_input.as<__nv_bfloat16>(), weight, output.as<__nv_bfloat16>(), scratch.data, plan.scratch_bytes(), stream);
    cuda_check(cudaStreamEndCapture(stream, &captured));
    cuda_check(cudaGraphInstantiate(&executable, captured, 0));
    cuda_check(cudaGraphLaunch(executable, stream));
    cuda_check(cudaGraphLaunch(executable, stream));
    cuda_check(cudaStreamSynchronize(stream));
    cuda_check(cudaGraphExecDestroy(executable));
    cuda_check(cudaGraphDestroy(captured));
  } else {
    plan.run(device_input.as<__nv_bfloat16>(), weight, output.as<__nv_bfloat16>(), scratch.data, plan.scratch_bytes(), stream);
    cuda_check(cudaStreamSynchronize(stream));
  }
  cuda_check(cudaStreamDestroy(stream));
  verify_guard(output, std::size_t(rows) * n * sizeof(__nv_bfloat16));
  verify_guard(scratch, plan.scratch_bytes());

  std::vector<unsigned> selected_rows{0, rows / 2, rows - 1};
  std::vector<unsigned> selected_columns{0, 31, 32, 127, n / 2, n - 1};
  if (k <= 256) {
    selected_rows.clear();
    selected_columns.clear();
    for (unsigned row = 0; row < rows; ++row) selected_rows.push_back(row);
    for (unsigned col = 0; col < n; ++col) selected_columns.push_back(col);
  }
  double worst = 0;
  for (unsigned row : selected_rows) {
    std::vector<float> quantized(k);
    const float inverse_global = 1.0F / input_global;
    for (unsigned block = 0; block < k / 16; ++block) {
      float amax = 0;
      for (unsigned j = 0; j < 16; ++j) amax = std::max(amax, std::abs(float(input[std::size_t(row) * k + block * 16 + j])));
      const float scale = fp8(nearest_fp8(amax * (inverse_global / 6.0F)));
      const float factor = scale == 0 ? 0 : inverse_global / scale;
      for (unsigned j = 0; j < 16; ++j) quantized[block * 16 + j] = nearest_fp4(float(input[std::size_t(row) * k + block * 16 + j]) * factor) * scale;
    }
    std::vector<__nv_bfloat16> actual(n);
    cuda_check(cudaMemcpy(actual.data(), output.as<__nv_bfloat16>() + std::size_t(row) * n,
                           n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    for (unsigned col : selected_columns) {
      double reference = 0;
      for (unsigned j = 0; j < k; ++j) {
        const auto byte = matrix.packed[std::size_t(col) * (k / 2) + j / 2];
        const float value = fp4((byte >> ((j % 2) * 4)) & 15);
        const float scale = fp8(matrix.scales[gewell::nvfp4::scale_offset(col, j / 16, k)]);
        reference += double(quantized[j]) * value * scale;
      }
      reference *= float(input_global * weight_global);
      const float rounded = float(__float2bfloat16(float(reference)));
      const double error = std::abs(double(float(actual[col])) - rounded);
      worst = std::max(worst, error);
      // One BF16 ulp permits different FP32 accumulation orders at a BF16 tie.
      if (!std::isfinite(float(actual[col])) || error > std::max(0.008 * std::abs(double(rounded)), 0.001)) {
        throw std::runtime_error("NVFP4 oracle mismatch rows=" + std::to_string(rows) + " K=" + std::to_string(k) + " N=" + std::to_string(n) + " row=" + std::to_string(row) + " col=" + std::to_string(col) + " got=" + std::to_string(float(actual[col])) + " expected=" + std::to_string(rounded));
      }
    }
  }
  std::cout << "native NVFP4 rows=" << rows << " K=" << k << " N=" << n
            << " max BF16 absolute error=" << worst << (graph ? " graph" : "") << '\n';
}
void check_gate_up(cublasLtHandle_t handle) {
  constexpr unsigned k = 5376, n = 21504;
  Matrix joined(k, 2 * n);
  Matrix down(n, k);
  Weight gate{joined.device_packed.as<std::uint8_t>(), joined.device_scales.as<std::uint8_t>(),
               0.03125F, 9.736560605233535e-05F};
  Weight up{gate.data + std::size_t(n) * k / 2,
             gate.scales + gewell::nvfp4::scale_storage_bytes(n, k),
             gate.input_scale, gate.weight_scale};
  const auto fused = gewell::nvfp4::gate_up_weight(gate, up);
  if (!fused.data) throw std::runtime_error("contiguous gate/up fusion rejected");
  auto incompatible = up;
  incompatible.input_scale *= 2;
  if (gewell::nvfp4::gate_up_weight(gate, incompatible).data)
    throw std::runtime_error("gate/up fused different activation globals");
  incompatible = up; incompatible.weight_scale *= 2;
  if (gewell::nvfp4::gate_up_weight(gate, incompatible).data)
    throw std::runtime_error("gate/up fused different weight globals");
  incompatible = up; incompatible.scales += 512;
  if (gewell::nvfp4::gate_up_weight(gate, incompatible).data)
    throw std::runtime_error("gate/up fused nonconsecutive scale tiles");
  for (unsigned rows : {1u, 3u, 32u, 128u}) {
    std::vector<__nv_bfloat16> input(std::size_t(rows) * k);
    for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = __float2bfloat16(float(int((i * 17 + i / k) % 127) - 63) / 16);
    Buffer x(input.size() * 2), g(std::size_t(rows) * n * 2), u(std::size_t(rows) * n * 2);
    Buffer gu(std::size_t(rows) * 2 * n * 2 + 256);
    cuda_check(cudaMemcpy(x.data, input.data(), input.size() * 2, cudaMemcpyHostToDevice));
    cuda_check(cudaMemset(gu.data, 0xCD, std::size_t(rows) * 2 * n * 2 + 256));
    Plan separate(handle, rows, k, n), combined(handle, rows, k, 2 * n);
    Buffer scratch(std::max(separate.scratch_bytes(), combined.scratch_bytes()));
    separate.run(x.as<__nv_bfloat16>(), gate, g.as<__nv_bfloat16>(), scratch.data, separate.scratch_bytes());
    separate.run(x.as<__nv_bfloat16>(), up, u.as<__nv_bfloat16>(), scratch.data, separate.scratch_bytes());
    combined.run(x.as<__nv_bfloat16>(), fused, gu.as<__nv_bfloat16>(), scratch.data, combined.scratch_bytes());
    std::vector<__nv_bfloat16> a(std::size_t(rows) * n), b(a.size()), both(2 * a.size());
    cuda_check(cudaMemcpy(a.data(), g.data, a.size() * 2, cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(b.data(), u.data, b.size() * 2, cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(both.data(), gu.data, both.size() * 2, cudaMemcpyDeviceToHost));
    for (unsigned row = 0; row < rows; ++row)
      for (unsigned col = 0; col < n; ++col)
        if (float(a[row * n + col]) != float(both[row * 2 * n + col]) ||
            float(b[row * n + col]) != float(both[row * 2 * n + n + col]))
          throw std::runtime_error("fused gate/up differs from separate GEMMs");
    verify_guard(gu, both.size() * 2);
    std::cout << "fused NVFP4 gate/up rows=" << rows << " exact, guard: PASS\n";
    Buffer product(std::size_t(rows) * n * 2), reference(std::size_t(rows) * k * 2);
    Buffer result(std::size_t(rows) * k * 2 + 256);
    cuda_check(cudaMemset(result.data, 0xCD, std::size_t(rows) * k * 2 + 256));
    gewell::bf16_primitives::gelu_tanh_multiply_interleaved(
        gu.as<__nv_bfloat16>(), product.as<__nv_bfloat16>(), rows, n);
    Plan down_plan(handle, rows, n, k);
    Buffer down_scratch(down_plan.scratch_bytes());
    Weight dw{down.device_packed.as<std::uint8_t>(), down.device_scales.as<std::uint8_t>(),
               rows == 1 ? 0.2708333432674408F : 0.0424107164144516F, 0.0737F};
    down_plan.run(product.as<__nv_bfloat16>(), dw, reference.as<__nv_bfloat16>(),
                   down_scratch.data, down_plan.scratch_bytes());
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    cuda_check(cudaStreamCreate(&stream));
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    down_plan.run(gu.as<__nv_bfloat16>(), dw, result.as<__nv_bfloat16>(),
                   down_scratch.data, down_plan.scratch_bytes(), stream,
                   gewell::nvfp4::InputTransform::gelu_tanh_multiply);
    cuda_check(cudaStreamEndCapture(stream, &graph));
    cuda_check(cudaGraphInstantiate(&executable, graph, 0));
    cuda_check(cudaGraphLaunch(executable, stream));
    cuda_check(cudaGraphLaunch(executable, stream));
    cuda_check(cudaStreamSynchronize(stream));
    std::vector<__nv_bfloat16> expected(std::size_t(rows) * k), actual(expected.size());
    cuda_check(cudaMemcpy(expected.data(), reference.data, expected.size() * 2, cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(actual.data(), result.data, actual.size() * 2, cudaMemcpyDeviceToHost));
    if (std::memcmp(expected.data(), actual.data(), actual.size() * 2))
      throw std::runtime_error("fused GELU/FP4 packing differs from separately rounded product");
    verify_guard(result, actual.size() * 2);
    cudaGraphExecDestroy(executable); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
    std::cout << "fused GELU/FP4 packing rows=" << rows << " exact, graph replay, guard: PASS\n";
  }
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
    if (properties.major < 10) return 77;
    check_dequantize(false);
    check_dequantize(true);
    cublasLtHandle_t handle;
    if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("create cuBLASLt handle");
    check_vllm_rounding_boundaries(handle);
    check_gate_up(handle);
    {
      Matrix small(256, 256);
      for (unsigned rows : {1u, 3u, 31u, 32u, 33u, 127u, 128u, 129u}) check_shape(handle, small, rows, rows == 3);
    }
    {
      Matrix gate(5376, 21504);
      for (unsigned rows : {1u, 17u, 32u, 129u, 1024u, 1280u}) check_shape(handle, gate, rows);
    }
    {
      Matrix down(21504, 5376);
      for (unsigned rows : {1u, 17u, 32u, 129u, 511u, 512u, 513u, 1024u, 1280u, 2048u, 2592u, 4096u})
        check_shape(handle, down, rows, rows == 512);
    }
    {
      Matrix joined(5376, 43008);
      for (unsigned rows : {511u, 512u, 513u, 1024u, 2048u, 2848u, 4096u})
        check_shape(handle, joined, rows, rows == 513);
    }
    // Local/global Q, K/V and O have distinct widths. Exercise decode,
    // padded batches, prefill tails and graph capture for every shape.
    for (const auto [k, n] : std::array<std::pair<unsigned, unsigned>, 6>{{
             {5376, 8192}, {5376, 16384}, {5376, 4096},
             {5376, 2048}, {8192, 5376}, {16384, 5376}}}) {
      Matrix attention(k, n);
      for (unsigned rows : {1u, 3u, 33u, 129u, 1024u})
        check_shape(handle, attention, rows, rows == 3);
    }
    bool rejected = false;
    try { Plan invalid(handle, 0, 5376, 21504); } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("NVFP4 accepted zero rows");
    cublasLtDestroy(handle);
    std::cout << "NVFP4 native quantization/matmul, row padding, guards, graph capture: PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
