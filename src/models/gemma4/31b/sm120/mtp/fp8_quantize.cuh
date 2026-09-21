#pragma once

#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace gewell::mtp_attention::detail {

__device__ __forceinline__ float fp8_scale(float maximum) {
  const float scale = maximum / 448.0F;
  return scale > 0.0F ? scale : 1.0F;
}

__device__ __noinline__ inline float fp8_tiny_scale_ratio(float x, float scale) {
  return x / scale;
}

__device__ __forceinline__ float fp8_ratio(float x, float scale, float inverse) {
  // Amortize the reciprocal over the vector. Correct the rounded product
  // before E4M3 rounding. Tiny scales can underflow the residual even when
  // the scale itself is normal; keep direct division outside the safe range.
  if (scale >= 0x1p-64F) {
    const float quotient = x * inverse;
    return fmaf(fmaf(-quotient, scale, x), inverse, quotient);
  }
  return fp8_tiny_scale_ratio(x, scale);
}

__device__ __forceinline__ unsigned char fp8_quantize(float x, float scale, float inverse) {
  return __nv_cvt_float_to_fp8(fp8_ratio(x, scale, inverse), __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ unsigned fp8_pack_four(const float* x, float scale, float inverse) {
  const auto lo = __nv_cvt_float2_to_fp8x2(make_float2(
      fp8_ratio(x[0], scale, inverse), fp8_ratio(x[1], scale, inverse)), __NV_SATFINITE, __NV_E4M3);
  const auto hi = __nv_cvt_float2_to_fp8x2(make_float2(
      fp8_ratio(x[2], scale, inverse), fp8_ratio(x[3], scale, inverse)), __NV_SATFINITE, __NV_E4M3);
  return unsigned(lo) | (unsigned(hi) << 16);
}

}  // namespace gewell::mtp_attention::detail
