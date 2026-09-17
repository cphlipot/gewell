#pragma once

#include "kv_storage.cuh"

#include "gewell/prefill_primitives.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace gewell::prefill_primitives::detail {

constexpr unsigned kThreads = 256;
constexpr unsigned kWarpSize = 32;
constexpr unsigned kWarpsPerBlock = kThreads / kWarpSize;

static_assert(kWarpsPerBlock == 8);
static_assert(kTokenCount == gemma4_31b::kLocalWindowSize);
static_assert(kGlobalCompactRotatedKeyElements == 2 * 64);
static_assert(kGlobalCompactCacheRowElements == 640);

[[noreturn]] inline void fail(std::string_view operation, std::string_view detail) {
  throw std::runtime_error(std::string(operation) + ": " +
                           std::string(detail));
}

inline void check_cuda(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    fail(operation, cudaGetErrorString(status));
  }
}

inline void check_cublas(cublasStatus_t status, std::string_view operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    fail(operation, "cuBLAS status " + std::to_string(status));
  }
}

inline void check_pointer(const void* pointer, std::string_view name) {
  if (pointer == nullptr) {
    fail(name, "null device pointer");
  }
}

inline void check_kind(gemma4_31b::AttentionKind kind, std::string_view operation) {
  if (kind != gemma4_31b::AttentionKind::local &&
      kind != gemma4_31b::AttentionKind::global) {
    fail(operation, "invalid Gemma 4 attention kind");
  }
}

inline unsigned blocks_for(std::size_t elements) {
  const std::size_t blocks = (elements + kThreads - 1) / kThreads;
  if (blocks > std::numeric_limits<unsigned>::max()) {
    fail("kernel launch", "element count exceeds the CUDA grid contract");
  }
  return static_cast<unsigned>(blocks);
}

inline void check_chunk_range(std::uint32_t base_position,
                       std::uint32_t token_count,
                       std::string_view operation) {
  if (token_count == 0 || token_count > kMaxChunkTokenCount) {
    fail(operation, "token count must be in [1,4096]");
  }
  if (base_position >
      std::numeric_limits<std::uint32_t>::max() - (token_count - 1)) {
    fail(operation, "absolute chunk position overflows uint32");
  }
}

inline void check_tensor_chunk_range(std::uint32_t base_position,
                             std::uint32_t token_count,
                             std::string_view operation) {
  if (token_count == 0 || token_count > kTensorAttentionMaximumQueryRows) {
    fail(operation, "token count must be in [1,4096]");
  }
  if (base_position >
      std::numeric_limits<std::uint32_t>::max() - token_count) {
    fail(operation, "exclusive chunk end overflows uint32");
  }
}

inline void check_chunk_cache(std::uint32_t base_position,
                       std::uint32_t token_count,
                       std::uint32_t cache_capacity,
                       gemma4_31b::AttentionKind kind,
                       std::string_view operation) {
  check_chunk_range(base_position, token_count, operation);
  if (kind == gemma4_31b::AttentionKind::local) {
    if (cache_capacity != gemma4_31b::kLocalWindowSize) {
      fail(operation, "local cache capacity must equal 1024");
    }
    return;
  }
  if (base_position > cache_capacity ||
      token_count > cache_capacity - base_position) {
    fail(operation, "global cache does not cover the chunk end");
  }
}

inline void check_image_block(std::uint32_t base_position,
                       std::uint32_t token_count,
                       std::uint32_t image_begin,
                       std::uint32_t image_end,
                       std::string_view operation) {
  const std::uint32_t chunk_end = base_position + token_count;
  if (image_begin < base_position || image_begin >= image_end ||
      image_end > chunk_end) {
    fail(operation, "image block must be nonempty and contained in the chunk");
  }
  if (image_end - image_begin > gemma4_31b::kVisionMaxSoftTokenCount) {
    fail(operation, "image block exceeds 1120 soft tokens");
  }
}

__device__ __forceinline__ bool is_compact_global_key_dimension(
    std::uint32_t dimension) {
  return dimension < 64 || (dimension >= 256 && dimension < 320);
}

__device__ __forceinline__ std::uint32_t compact_global_key_index(
    std::uint32_t dimension) {
  return dimension < 64 ? dimension : 64 + (dimension - 256);
}

__device__ __forceinline__ BFloat16 compact_global_reconstructed_key(
    BFloat16 value, BFloat16 scale) {
  return __float2bfloat16_rn(__bfloat162float(value) *
                             __bfloat162float(scale));
}

template <bool Paged, typename Element>
__device__ Element* compact_global_cache_row(
    Element* cache, const std::uint64_t* page_offsets,
    std::uint32_t page_tokens, std::size_t layer_offset_elements,
    std::uint32_t kv_head, std::uint32_t absolute_position,
    std::uint32_t cache_capacity, kv_cache::Format format = kv_cache::Format::bf16) {
  if constexpr (Paged) {
    return compact_global_cache::paged_row(
        cache, page_offsets, page_tokens, layer_offset_elements, kv_head,
        absolute_position, format);
  }
  return cache +
         (static_cast<std::size_t>(kv_head) * cache_capacity +
          absolute_position) *
             kv_cache::row_words(kGlobalCompactCacheRowElements, format, 2);
}

inline void check_paged_compact_global_cache(
    const CompactGlobalPagedCache& cache, std::uint32_t base_position,
    std::uint32_t token_count,
    std::string_view operation, bool prefix_only = false) {
  check_pointer(cache.page_pool, std::string(operation) + " page pool");
  check_pointer(cache.page_offsets,
                std::string(operation) + " page table");
  check_chunk_range(base_position, token_count, operation);
  if (cache.page_tokens == 0 || cache.page_count == 0 ||
      cache.page_tokens != 256) {
    fail(operation, "page table must use nonempty 256-token pages");
  }
  const std::uint64_t end = static_cast<std::uint64_t>(base_position) +
                            (prefix_only ? 0 : token_count);
  const std::uint64_t required_pages =
      (end + cache.page_tokens - 1) / cache.page_tokens;
  if (required_pages > cache.page_count) {
    fail(operation, "page table does not cover the chunk");
  }
  const std::size_t layer_elements =
      static_cast<std::size_t>(gemma4_31b::kGlobalKvHeadCount) *
      cache.page_tokens * kv_cache::row_words(kGlobalCompactCacheRowElements, cache.format, 2);
  if (cache.layer_offset_elements >
          std::numeric_limits<std::size_t>::max() - layer_elements ||
      cache.page_stride_elements < cache.layer_offset_elements +
                                        layer_elements) {
    fail(operation, "page stride does not cover one global layer");
  }
}

}  // namespace gewell::prefill_primitives::detail
