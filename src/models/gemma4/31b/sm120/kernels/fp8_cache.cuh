#pragma once

#include "kv_storage.cuh"

namespace gewell::fp8_cache {

__device__ __forceinline__ bool rotated(unsigned d) {
  return d < 64 || (d >= 256 && d < 320);
}

// Each lane owns consecutive groups of four dimensions. This bound depends
// only on the layer weights, not on the cached tokens or future queries.
__device__ __forceinline__ float norm_bound(const kv_storage::BF16* norm) {
  const unsigned lane = threadIdx.x % 32;
  float maximum = 0;
  for (unsigned d = lane; d < 512; d += 32)
    if (!rotated(d)) maximum = fmaxf(maximum, fabsf(float(norm[d])));
  for (unsigned delta = 16; delta; delta /= 2)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, delta));
  return maximum;
}

struct Scales {
  float key, value, rotated_ratio, value_ratio;
};

__device__ __forceinline__ Scales scales(const kv_storage::BF16* record,
                                         float norm_maximum) {
  const auto* stored = reinterpret_cast<const float*>(
      reinterpret_cast<const unsigned char*>(record) + 640);
  const float key = fmaxf(stored[0], stored[1] * norm_maximum);
  return {key, stored[1], stored[0] / key, stored[1] / key};
}

__device__ __forceinline__ unsigned scale_four(unsigned packed, float ratio,
                                               const kv_storage::BF16* norm) {
  unsigned result = 0;
#pragma unroll
  for (unsigned i = 0; i < 2; ++i) {
    const auto h = __nv_cvt_fp8x2_to_halfraw2(packed >> (i * 16), __NV_E4M3);
    auto x = __half22float2(static_cast<__half2>(h));
    if (norm) {
      x.x *= float(norm[i * 2]);
      x.y *= float(norm[i * 2 + 1]);
    }
    x.x *= ratio; x.y *= ratio;
    result |= unsigned(__nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3)) << (i * 16);
  }
  return result;
}

// V bytes are used unchanged. K has a single direct FP8 rounding, without
// restoring/scaling to BF16 first. The derived scale bounds both K regions.
__device__ __forceinline__ uint2 compact_four(const kv_storage::BF16* record,
    const kv_storage::BF16* norm, unsigned d, const Scales& s) {
  const auto* bytes = reinterpret_cast<const unsigned char*>(record);
  const unsigned v = *reinterpret_cast<const unsigned*>(bytes + 128 + d);
  const unsigned k = rotated(d)
      ? scale_four(*reinterpret_cast<const unsigned*>(bytes + (d < 64 ? d : d - 192)), s.rotated_ratio, nullptr)
      : scale_four(v, s.value_ratio, norm + d);
  return {k, v};
}

__device__ __forceinline__ bool finite_four(unsigned word) {
  return (word & 0x7fU) != 0x7fU && ((word >> 8) & 0x7fU) != 0x7fU &&
         ((word >> 16) & 0x7fU) != 0x7fU && ((word >> 24) & 0x7fU) != 0x7fU;
}

}  // namespace gewell::fp8_cache
