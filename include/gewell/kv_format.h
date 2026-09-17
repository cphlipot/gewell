#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace gewell::kv_cache {

enum class Format { bf16, fp8 };

constexpr std::string_view format_name(Format format) {
  return format == Format::fp8 ? "fp8-e4m3-token-head" : "bf16";
}

// FP8 records contain E4M3 payload, FP32 dequantization scales, and padding
// to keep every row aligned for vector loads. Global compact rows have two
// scales (K128 and V512); each local K or V row has one.
// FP8 record bases must be 16-byte aligned, including page and layer offsets.
#ifdef __CUDACC__
__host__ __device__
#endif
constexpr std::size_t row_bytes(std::size_t elements, Format format,
                                std::size_t scales = 1) {
  return format == Format::bf16 ? elements * 2
      : (elements + scales * sizeof(float) + 15) / 16 * 16;
}

// Cache addresses and page offsets use 16-bit storage words for both formats.
// An FP8 record is opaque through the BF16 pointer; readers must decode it.
#ifdef __CUDACC__
__host__ __device__
#endif
constexpr std::size_t row_words(std::size_t elements, Format format,
                                std::size_t scales = 1) {
  return row_bytes(elements, format, scales) / 2;
}

}  // namespace gewell::kv_cache
