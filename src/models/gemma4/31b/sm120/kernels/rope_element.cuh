#pragma once

#include "bf16_common.cuh"

namespace gewell::bf16_primitives::detail {

// Preserve the BF16 rounding of both products before their sum.
template<unsigned D>
__device__ __forceinline__ BFloat16 rope_element(const BFloat16* input,
    const BFloat16* cosine, const BFloat16* sine, unsigned dimension) {
  const unsigned paired = dimension < D / 2 ? dimension + D / 2 : dimension - D / 2;
  float rotated = __bfloat162float(input[paired]);
  if (dimension < D / 2) rotated = -rotated;
  const BFloat16 direct = __float2bfloat16_rn(
      __bfloat162float(input[dimension]) * __bfloat162float(cosine[dimension]));
  const BFloat16 cross = __float2bfloat16_rn(rotated * __bfloat162float(sine[dimension]));
  return __float2bfloat16_rn(__bfloat162float(direct) + __bfloat162float(cross));
}

}  // namespace gewell::bf16_primitives::detail
