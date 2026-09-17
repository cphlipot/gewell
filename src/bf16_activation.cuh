#pragma once

#include <cuda_bf16.h>
#include <cmath>

namespace gewell::detail {

// Preserve both BF16 boundaries even when the consumer immediately quantizes
// the product. These constants and FP32 operation order match Gemma's GELU.
__device__ __forceinline__ float gelu_tanh_multiply_bf16(float gate, float up) {
  const float cube = gate * gate * gate;
  const float inner = 0.7978845608028654F * (gate + 0.044715F * cube);
  const float activated = 0.5F * gate * (1.0F + tanhf(inner));
  const auto rounded = __float2bfloat16_rn(activated);
  return __bfloat162float(__float2bfloat16_rn(__bfloat162float(rounded) * up));
}

}  // namespace gewell::detail
