#pragma once

#include "gewell/models/gemma4/31b/model.h"
#include <cuda_bf16.h>

#include <cmath>
#include <cstddef>

namespace gewell::mtp_target::detail {
using BFloat16 = __nv_bfloat16;

// Preserve the generic norm's 256-thread reduction and BF16 boundaries while
// keeping each residual row in registers across the two normalizations.
constexpr unsigned kNormThreads = 256;
constexpr unsigned kHidden = gemma4_31b::kHiddenSize;
constexpr unsigned kNormValues = kHidden / kNormThreads;
static_assert(kHidden % kNormThreads == 0);

__device__ inline float inverse_rms(const float (&values)[kNormValues], float* sums) {
  float sum = 0.0F;
#pragma unroll
  for (unsigned i = 0; i < kNormValues; ++i) sum += values[i] * values[i];
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned offset = kNormThreads / 2; offset; offset /= 2) {
    if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
    __syncthreads();
  }
  const float result = powf(sums[0] / float(kHidden) + 1e-6F, -0.5F);
  __syncthreads();  // All threads consume sums[0] before another reduction.
  return result;
}

template <bool Scale>
__global__ void residual_norm(BFloat16* branch, const BFloat16* post_weight,
                              const BFloat16* residual, const BFloat16* scalar,
                              const BFloat16* next_weight, BFloat16* normalized) {
  __shared__ float sums[kNormThreads];
  const std::size_t row = std::size_t(blockIdx.x) * kHidden;
  float values[kNormValues];
#pragma unroll
  for (unsigned i = 0; i < kNormValues; ++i)
    values[i] = __bfloat162float(branch[row + threadIdx.x + i * kNormThreads]);
  const float post_inverse = inverse_rms(values, sums);
#pragma unroll
  for (unsigned i = 0; i < kNormValues; ++i) {
    const unsigned d = threadIdx.x + i * kNormThreads;
    const BFloat16 post = __float2bfloat16_rn(
        values[i] * post_inverse * __bfloat162float(post_weight[d]));
    BFloat16 value = __float2bfloat16_rn(
        __bfloat162float(residual[row + d]) + __bfloat162float(post));
    if constexpr (Scale)
      value = __float2bfloat16_rn(__bfloat162float(value) * __bfloat162float(*scalar));
    branch[row + d] = value;
    values[i] = __bfloat162float(value);
  }
  const float next_inverse = inverse_rms(values, sums);
#pragma unroll
  for (unsigned i = 0; i < kNormValues; ++i) {
    const unsigned d = threadIdx.x + i * kNormThreads;
    normalized[row + d] = __float2bfloat16_rn(
        values[i] * next_inverse * __bfloat162float(next_weight[d]));
  }
}

}  // namespace gewell::mtp_target::detail
