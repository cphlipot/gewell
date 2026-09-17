#pragma once

#include "gewell/weight_qdq.h"
#include "gewell/bf16_primitives.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace gewell::weight_qdq::detail {

namespace model = gemma4_31b;
using BFloat16 = bf16_primitives::BFloat16;

constexpr std::size_t kMaximumMaskBytes = 1U << 20;
constexpr int kThreads = 256;
constexpr int kMaximumReductionBlocks = 4'096;
constexpr float kE4m3Maximum = 448.0F;
constexpr float kE2m1Maximum = 6.0F;
constexpr float kNvfp4MinimumBlockScale = 0x1p-9F;

static_assert(static_cast<std::uint8_t>(Type::bf16) == 0);

[[noreturn]] inline void fail(const std::string& message) {
  throw std::runtime_error(message);
}

inline void check_cuda(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    fail(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

[[nodiscard]] inline std::optional<std::size_t> projection_index(
    model::TensorRole role) {
  switch (role) {
    case model::TensorRole::q_proj:
      return 0;
    case model::TensorRole::k_proj:
      return 1;
    case model::TensorRole::v_proj:
      return 2;
    case model::TensorRole::o_proj:
      return 3;
    case model::TensorRole::gate_proj:
      return 4;
    case model::TensorRole::up_proj:
      return 5;
    case model::TensorRole::down_proj:
      return 6;
    default:
      return std::nullopt;
  }
}

__host__ __device__ inline float round_e4m3(float value) {
  return static_cast<float>(__nv_fp8_e4m3(value));
}

__host__ __device__ inline float round_e2m1(float value) {
  return static_cast<float>(__nv_fp4_e2m1(value));
}

__host__ __device__ inline BFloat16 round_bf16(float value) {
  return __float2bfloat16_rn(value);
}

__host__ __device__ inline float nvfp4_block_scale(float block_amax,
                                             float global_scale) {
  if (block_amax == 0.0F) {
    return 1.0F;
  }
  const float unrounded = block_amax / (kE2m1Maximum * global_scale);
  const float clamped =
      fminf(kE4m3Maximum, fmaxf(kNvfp4MinimumBlockScale, unrounded));
  return round_e4m3(clamped);
}

struct DeviceReductionState {
  unsigned int amax_bits;
  unsigned int nonfinite;
};

class DeviceState {
 public:
  DeviceState() {
    check_cuda(cudaMalloc(&pointer_, sizeof(DeviceReductionState)),
               "cudaMalloc QDQ reduction state");
  }

  ~DeviceState() {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
  }

  DeviceState(const DeviceState&) = delete;
  DeviceState& operator=(const DeviceState&) = delete;

  [[nodiscard]] DeviceReductionState* get() const { return pointer_; }

 private:
  DeviceReductionState* pointer_{};
};

void apply_tensor(DeviceState& state, Type type, BFloat16* values,
                  std::uint64_t elements);

}  // namespace gewell::weight_qdq::detail
