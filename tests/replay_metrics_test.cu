#include "gewell/replay_metrics.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace metrics = gewell::replay_metrics;

void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

double value(__nv_bfloat16 item) { return __bfloat162float(item); }

std::uint32_t argmax(const __nv_bfloat16* row, std::uint32_t vocabulary) {
  std::uint32_t chosen = 0;
  for (std::uint32_t id = 1; id < vocabulary; ++id) {
    if (value(row[id]) > value(row[chosen])) chosen = id;
  }
  return chosen;
}

// Independent long-double normalized-probability oracle, intentionally unlike
// the kernel's two-pass moment formula and reduction tree.
long double kl(const __nv_bfloat16* a, const __nv_bfloat16* b,
               std::uint32_t vocabulary) {
  const long double maximum_a = value(a[argmax(a, vocabulary)]);
  const long double maximum_b = value(b[argmax(b, vocabulary)]);
  long double mass_a = 0, mass_b = 0;
  for (std::uint32_t id = 0; id < vocabulary; ++id) {
    mass_a += std::exp(static_cast<long double>(value(a[id])) - maximum_a);
    mass_b += std::exp(static_cast<long double>(value(b[id])) - maximum_b);
  }
  const long double normalizer_a = std::log(mass_a);
  const long double normalizer_b = std::log(mass_b);
  long double result = 0;
  for (std::uint32_t id = 0; id < vocabulary; ++id) {
    const long double logp = value(a[id]) - maximum_a - normalizer_a;
    const long double logq = value(b[id]) - maximum_b - normalizer_b;
    result += std::exp(logp) * (logp - logq);
  }
  return result;
}

void run(std::uint32_t vocabulary, std::uint32_t rows) {
  const std::size_t elements = static_cast<std::size_t>(rows) * vocabulary;
  std::vector<__nv_bfloat16> reference(elements), replay(elements);
  std::vector<std::uint32_t> tokens(rows);
  std::vector<std::uint32_t> flags(rows, 0);
  for (std::uint32_t row = 0; row < rows; ++row) {
    const std::size_t start = static_cast<std::size_t>(row) * vocabulary;
    for (std::uint32_t id = 0; id < vocabulary; ++id) {
      reference[start + id] = __float2bfloat16_rn(
          static_cast<float>(static_cast<int>(id % 241) - 120) / 4.0F);
      replay[start + id] = row == 0 ? reference[start + id] :
          __float2bfloat16_rn(
              static_cast<float>(static_cast<int>((id * 13) % 193) - 96) / 4.0F);
    }
    if (row == 2 || row == 3) {
      std::fill_n(reference.data() + start, vocabulary, __float2bfloat16_rn(-30));
      std::fill_n(replay.data() + start, vocabulary, __float2bfloat16_rn(-30));
      reference[start + 100] = __float2bfloat16_rn(7);
      reference[start + 101] = __float2bfloat16_rn(row == 2 ? 7 : 8);
      replay[start + 100] = __float2bfloat16_rn(row == 2 ? 6.875F : 8);
      replay[start + 101] = __float2bfloat16_rn(row == 2 ? 7 : 8);
    } else if (row == 4) {
      std::fill_n(reference.data() + start, vocabulary, __float2bfloat16_rn(0));
      std::fill_n(replay.data() + start, vocabulary, __float2bfloat16_rn(0));
      replay[start + vocabulary - 1] = __float2bfloat16_rn(2);
    } else if (row == 5) {
      replay[start + 37] = __float2bfloat16_rn(std::numeric_limits<float>::quiet_NaN());
      flags[row] = metrics::nonfinite;
    } else if (row == 6) {
      reference[start + 3] = __float2bfloat16_rn(31);
      flags[row] = metrics::outside_softcap;
    } else if (row == 9) {
      std::fill_n(reference.data() + start, vocabulary, __float2bfloat16_rn(-30));
      std::fill_n(replay.data() + start, vocabulary, __float2bfloat16_rn(30));
      reference[start] = __float2bfloat16_rn(30);
      replay[start] = __float2bfloat16_rn(-30);
    }
    tokens[row] = argmax(reference.data() + start, vocabulary);
    if (row == 7) {
      tokens[row] = (tokens[row] + 1) % vocabulary;
      flags[row] = metrics::reference_argmax_mismatch;
    } else if (row == 8) {
      tokens[row] = vocabulary;
      flags[row] = metrics::invalid_recorded_token;
    }
  }

  __nv_bfloat16 *device_reference = nullptr, *device_replay = nullptr;
  std::uint32_t* device_tokens = nullptr;
  metrics::Row* device_rows = nullptr;
  cudaStream_t stream = nullptr;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  check(cudaMalloc(&device_reference, elements * sizeof(__nv_bfloat16)));
  check(cudaMalloc(&device_replay, elements * sizeof(__nv_bfloat16)));
  check(cudaMalloc(&device_tokens, rows * sizeof(std::uint32_t)));
  check(cudaMalloc(&device_rows, rows * sizeof(metrics::Row)));
  check(cudaMemcpyAsync(device_reference, reference.data(), elements * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice, stream));
  check(cudaMemcpyAsync(device_replay, replay.data(), elements * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice, stream));
  check(cudaMemcpyAsync(device_tokens, tokens.data(), rows * sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice, stream));
  metrics::compare_rows(device_reference, device_replay, device_tokens, rows,
                        vocabulary, device_rows, stream);
  std::vector<metrics::Row> result(rows);
  check(cudaMemcpyAsync(result.data(), device_rows, rows * sizeof(metrics::Row),
                        cudaMemcpyDeviceToHost, stream));
  check(cudaStreamSynchronize(stream));
  check(cudaFree(device_reference));
  check(cudaFree(device_replay));
  check(cudaFree(device_tokens));
  check(cudaFree(device_rows));
  check(cudaStreamDestroy(stream));

  for (std::uint32_t row = 0; row < rows; ++row) {
    const std::string label = "vocab=" + std::to_string(vocabulary) +
                              " row=" + std::to_string(row);
    require(result[row].flags == flags[row], label + ": validation flags differ");
    require(result[row].recorded_token == tokens[row], label + ": token differs");
    if (flags[row]) continue;
    const std::size_t start = static_cast<std::size_t>(row) * vocabulary;
    const auto* a = reference.data() + start;
    const auto* b = replay.data() + start;
    require(result[row].reference_argmax == argmax(a, vocabulary), label + ": reference argmax differs");
    require(result[row].replay_argmax == argmax(b, vocabulary), label + ": replay argmax differs");
    require(result[row].reference_margin == 0, label + ": reference margin differs");
    require(result[row].replay_margin == value(b[argmax(b, vocabulary)]) - value(b[tokens[row]]),
            label + ": replay margin differs");
    const long double forward = kl(a, b, vocabulary);
    const long double reverse = kl(b, a, vocabulary);
    require(std::abs(result[row].kl_ref_replay - forward) < 2e-10L,
            label + ": forward KL differs from long-double oracle");
    require(std::abs(result[row].kl_replay_ref - reverse) < 2e-10L,
            label + ": reverse KL differs from long-double oracle");
    if (row == 0) {
      require(result[row].kl_ref_replay == 0 && result[row].kl_replay_ref == 0,
              label + ": identical rows do not have exact zero KL");
    }
  }
}

}  // namespace

int main() {
  try {
    bool rejected = false;
    try {
      metrics::compare_rows(nullptr, nullptr, nullptr, 1, 1, nullptr);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "null device pointers were accepted");
    run(4097, 10);
    run(262144, 5);
    std::cout << "replay metrics tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "replay metrics test failed: " << error.what() << '\n';
    return 1;
  }
}
