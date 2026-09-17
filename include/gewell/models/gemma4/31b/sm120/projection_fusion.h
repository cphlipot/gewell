#pragma once

#include "gewell/models/gemma4/31b/model.h"
#include <cuda_bf16.h>
#include <array>
#include <cstdint>

namespace gewell::gemma4_31b::sm120::fusion {

inline constexpr std::uint32_t qkv_width(bool global) {
  return global ? 18'432 : 16'384;
}
inline constexpr auto kMaxQkvWidth = qkv_width(true);

inline bool adjacent(const __nv_bfloat16* first, const __nv_bfloat16* second,
                     std::size_t elements) {
  return first && second && reinterpret_cast<std::uintptr_t>(second) ==
      reinterpret_cast<std::uintptr_t>(first) + elements * sizeof(__nv_bfloat16);
}

// BF16 artifact matrices are consecutive row-major output-channel slices.
// Mixed storage and separately allocated caller weights use separate GEMMs.
inline const __nv_bfloat16* qkv_weight(
    const std::array<const __nv_bfloat16*, kLogicalTensorCount>& weights,
    std::uint32_t layer) {
  const auto q = 2 + 14 * layer - layer / 6;
  const bool global = is_global_layer(layer);
  if (!adjacent(weights[q], weights[q + 1],
                std::size_t(global ? 16'384 : 8'192) * kHiddenSize) ||
      (!global && !adjacent(weights[q + 1], weights[q + 2],
                            std::size_t(4'096) * kHiddenSize))) return nullptr;
  return weights[q];
}

inline const __nv_bfloat16* gate_up_weight(const __nv_bfloat16* gate,
                                          const __nv_bfloat16* up,
                                          std::uint32_t rows) {
  // cuBLASLt's fused BF16 shape is slower at M=1 and large prefill sizes.
  // Retain the join for the measured small-batch decode range.
  if (rows < 2 || rows > 128) return nullptr;
  return adjacent(gate, up, std::size_t(kMlpSize) * kHiddenSize) ? gate : nullptr;
}

}  // namespace gewell::gemma4_31b::sm120::fusion
