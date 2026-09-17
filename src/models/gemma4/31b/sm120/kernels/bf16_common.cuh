#pragma once

#include "kv_storage.cuh"

#include "gewell/bf16_primitives.h"

#include <cuda_runtime.h>
#include <climits>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace gewell::bf16_primitives::detail {

constexpr unsigned kThreads = 256;
constexpr unsigned kWarpThreads = 32;
[[noreturn]] inline void fail(std::string_view operation, std::string_view detail) {
  throw std::runtime_error(std::string(operation) + ": " +
                           std::string(detail));
}

inline void check_cuda(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    fail(operation, cudaGetErrorString(status));
  }
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (unsigned offset = kWarpThreads / 2; offset != 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (unsigned offset = kWarpThreads / 2; offset != 0; offset /= 2) {
    value = fmaxf(value,
                  __shfl_down_sync(0xffffffffU, value, offset));
  }
  return value;
}

inline void check_pointer(const void* pointer, std::string_view name) {
  if (pointer == nullptr) {
    fail(name, "null device pointer");
  }
}

inline void check_decode_rows(std::uint32_t rows, std::string_view operation) {
  if (rows == 0 || rows > static_cast<std::uint32_t>(INT_MAX)) {
    fail(operation, "row count must fit a positive int");
  }
}

inline unsigned blocks_for(std::size_t elements) {
  const std::size_t blocks = (elements + kThreads - 1) / kThreads;
  if (blocks > std::numeric_limits<unsigned>::max()) {
    fail("kernel launch", "element count exceeds the CUDA grid contract");
  }
  return static_cast<unsigned>(blocks);
}

}  // namespace gewell::bf16_primitives::detail
