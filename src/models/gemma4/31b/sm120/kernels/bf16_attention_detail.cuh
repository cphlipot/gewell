#pragma once

#include "bf16_common.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

// Internal attention definitions shared with the primitive reference tests.
namespace gewell::bf16_primitives::detail {

constexpr unsigned kGraphAttentionFusedFinalizeTiles = 4;
constexpr std::uint32_t kRuntimeAttentionTokensPerSplit = 32;
constexpr std::uint32_t kRuntimeGlobalCoarseGroupSize = 64;
constexpr std::uint32_t kRuntimeGlobalCoarseMinimumSplitCount = 1'024;

__device__ __forceinline__ void store_runtime_partial(
    float* partial_context, std::size_t index, float value) {
  partial_context[index] = value;
}

__device__ __forceinline__ void store_runtime_partial(
    __half* partial_context, std::size_t index, float value) {
  partial_context[index] = __float2half_rn(value);
}

__device__ __forceinline__ float load_runtime_partial(
    const float* partial_context, std::size_t index) {
  return partial_context[index];
}

__device__ __forceinline__ float load_runtime_partial(
    const __half* partial_context, std::size_t index) {
  return __half2float(partial_context[index]);
}

inline std::size_t runtime_fused_scratch_bytes_for_splits(
    std::size_t split_count, std::uint32_t head_size, bool global) {
  constexpr std::size_t kQueryHeads = gemma4_31b::kQueryHeadCount;
  const std::size_t partial_element_bytes =
      global ? sizeof(__half) : sizeof(float);
  const std::size_t bytes_per_query_split =
      static_cast<std::size_t>(head_size) * partial_element_bytes +
      2 * sizeof(float);
  if (split_count > std::numeric_limits<std::size_t>::max() /
                        (kQueryHeads * bytes_per_query_split)) {
    fail("runtime fused attention scratch", "scratch size overflows size_t");
  }
  return kQueryHeads * split_count * bytes_per_query_split;
}

inline std::size_t runtime_global_coarse_scratch_bytes_for_splits(
    std::size_t split_count) {
  if (split_count < kRuntimeGlobalCoarseMinimumSplitCount) {
    return 0;
  }
  constexpr std::size_t kQueryHeads = gemma4_31b::kQueryHeadCount;
  constexpr std::size_t kBytesPerQueryGroup =
      static_cast<std::size_t>(gemma4_31b::kGlobalHeadSize) * sizeof(float) +
      2 * sizeof(float);
  const std::size_t group_count =
      (split_count + kRuntimeGlobalCoarseGroupSize - 1) /
      kRuntimeGlobalCoarseGroupSize;
  if (group_count > std::numeric_limits<std::size_t>::max() /
                        (kQueryHeads * kBytesPerQueryGroup)) {
    fail("runtime global coarse attention scratch",
         "scratch size overflows size_t");
  }
  return kQueryHeads * group_count * kBytesPerQueryGroup;
}

struct RuntimeFusedScratchLayout {
  void* partial_context{};
  float* split_maximum{};
  float* split_denominator{};
  std::size_t bytes{};
};

struct RuntimeGlobalCoarseScratchLayout {
  float* partial_context{};
  float* split_maximum{};
  float* split_denominator{};
  std::uint32_t split_count{};
  std::size_t bytes{};
};

inline RuntimeFusedScratchLayout runtime_fused_scratch_layout(
    void* scratch, std::uint32_t split_count, std::uint32_t head_size,
    bool global) {
  const std::size_t partial_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * split_count *
      head_size;
  const std::size_t metadata_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * split_count;
  const std::size_t partial_bytes =
      partial_elements * (global ? sizeof(__half) : sizeof(float));
  auto* const bytes = static_cast<std::uint8_t*>(scratch);
  auto* const split_maximum =
      reinterpret_cast<float*>(bytes + partial_bytes);
  if (reinterpret_cast<std::uintptr_t>(split_maximum) % alignof(float) != 0) {
    fail("runtime fused attention scratch",
         "metadata is not float-aligned");
  }
  return {
      scratch,
      split_maximum,
      split_maximum + metadata_elements,
      runtime_fused_scratch_bytes_for_splits(split_count, head_size, global),
  };
}

inline RuntimeGlobalCoarseScratchLayout runtime_global_coarse_scratch_layout(
    void* scratch, std::size_t fine_bytes, std::uint32_t fine_split_count) {
  const std::uint32_t coarse_split_count =
      (fine_split_count + kRuntimeGlobalCoarseGroupSize - 1) /
      kRuntimeGlobalCoarseGroupSize;
  const std::size_t partial_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) *
      coarse_split_count * gemma4_31b::kGlobalHeadSize;
  const std::size_t metadata_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) *
      coarse_split_count;
  auto* const bytes = static_cast<std::uint8_t*>(scratch);
  auto* const partial_context =
      reinterpret_cast<float*>(bytes + fine_bytes);
  auto* const split_maximum = partial_context + partial_elements;
  if (reinterpret_cast<std::uintptr_t>(partial_context) % alignof(float) !=
      0) {
    fail("runtime global coarse attention scratch",
         "coarse scratch is not float-aligned");
  }
  return {
      partial_context,
      split_maximum,
      split_maximum + metadata_elements,
      coarse_split_count,
      fine_bytes + runtime_global_coarse_scratch_bytes_for_splits(
                       fine_split_count),
  };
}

template <typename PartialContext, std::uint32_t HeadSize,
          std::uint32_t KvHeads, bool Local>
__device__ __forceinline__
void causal_gqa_attention_cached_m1_fused_split_body(
    const BFloat16* query, const BFloat16* key_cache,
    const BFloat16* value_cache, std::uint32_t absolute_position,
    std::uint32_t capacity, std::uint32_t split_count,
    PartialContext* partial_context, float* split_maximum,
    float* split_denominator, unsigned grid_x, unsigned grid_y, kv_cache::Format format = kv_cache::Format::bf16) {
  static_assert(HeadSize == gemma4_31b::kLocalHeadSize ||
                HeadSize == gemma4_31b::kGlobalHeadSize);
  static_assert(gemma4_31b::kQueryHeadCount % KvHeads == 0);
  static_assert(kThreads % kWarpThreads == 0);
  constexpr std::uint32_t kWarps = kThreads / kWarpThreads;
  constexpr std::uint32_t kQueriesPerKv =
      gemma4_31b::kQueryHeadCount / KvHeads;
  constexpr std::uint32_t kWarpsPerQuery = kWarps / kQueriesPerKv;
  static_assert(kWarps % kQueriesPerKv == 0);
  static_assert(HeadSize % kWarpThreads == 0);
  static_assert(HeadSize % (kWarpsPerQuery * kWarpThreads) == 0);
  static_assert(kRuntimeAttentionTokensPerSplit * HeadSize *
                            sizeof(BFloat16) +
                        kQueriesPerKv * kRuntimeAttentionTokensPerSplit *
                            sizeof(float) <=
                    48U * 1'024U);

  __shared__ __align__(16)
      BFloat16 cache_tile[kRuntimeAttentionTokensPerSplit * HeadSize];
  __shared__
      float scores[kQueriesPerKv * kRuntimeAttentionTokensPerSplit];

  const std::uint32_t first_visible_position =
      Local && absolute_position >= kGraphAttentionLocalCapacity - 1
          ? absolute_position - (kGraphAttentionLocalCapacity - 1)
          : 0;
  const std::uint32_t visible_tokens =
      absolute_position - first_visible_position + 1;
  const std::uint32_t split = grid_y;
  const std::uint32_t split_begin = static_cast<std::uint32_t>(
      static_cast<std::uint64_t>(visible_tokens) * split / split_count);
  const std::uint32_t split_end = static_cast<std::uint32_t>(
      static_cast<std::uint64_t>(visible_tokens) * (split + 1) /
      split_count);
  const std::uint32_t split_tokens = split_end - split_begin;
  const std::uint32_t first_absolute_token =
      first_visible_position + split_begin;
  const std::uint32_t kv_head = grid_x;

  const std::size_t tile_elements =
      static_cast<std::size_t>(split_tokens) * HeadSize;
  for (std::size_t index = threadIdx.x * 8; index < tile_elements;
       index += blockDim.x * 8) {
    const std::uint32_t token =
        static_cast<std::uint32_t>(index / HeadSize);
    const std::uint32_t dimension =
        static_cast<std::uint32_t>(index % HeadSize);
    const std::uint32_t absolute_token = first_absolute_token + token;
    const std::uint32_t slot =
        Local ? absolute_token % kGraphAttentionLocalCapacity
              : absolute_token;
    const std::size_t cache_index =
        (static_cast<std::size_t>(kv_head) * capacity + slot) * HeadSize +
        dimension;
    *reinterpret_cast<uint4*>(cache_tile + index) = kv_storage::load_eight(
        kv_storage::row(key_cache, cache_index / HeadSize, HeadSize, format),
        dimension, HeadSize, format);
  }
  __syncthreads();

  const std::uint32_t warp = threadIdx.x / kWarpThreads;
  const std::uint32_t lane = threadIdx.x % kWarpThreads;
  const std::uint32_t query_within_kv = warp / kWarpsPerQuery;
  const std::uint32_t warp_within_query = warp % kWarpsPerQuery;
  const std::uint32_t query_head =
      kv_head * kQueriesPerKv + query_within_kv;
  const std::size_t query_offset =
      static_cast<std::size_t>(query_head) * HeadSize;
  constexpr std::uint32_t kQueryValuesPerLane = HeadSize / kWarpThreads;
  float query_values[kQueryValuesPerLane];
#pragma unroll
  for (std::uint32_t index = 0; index < kQueryValuesPerLane; ++index) {
    query_values[index] = __bfloat162float(
        query[query_offset + lane + index * kWarpThreads]);
  }

  for (std::uint32_t token = warp_within_query; token < split_tokens;
       token += kWarpsPerQuery) {
    float score = 0.0F;
#pragma unroll
    for (std::uint32_t index = 0; index < kQueryValuesPerLane; ++index) {
      const std::uint32_t dimension = lane + index * kWarpThreads;
      score = fmaf(
          query_values[index],
          __bfloat162float(
              cache_tile[static_cast<std::size_t>(token) * HeadSize +
                         dimension]),
          score);
    }
    score = warp_sum(score);
    if (lane == 0) {
      scores[query_within_kv * kRuntimeAttentionTokensPerSplit + token] =
          score;
    }
  }
  __syncthreads();

  if (warp_within_query == 0) {
    float maximum = -__int_as_float(0x7f800000);
    for (std::uint32_t token = lane; token < split_tokens;
         token += kWarpThreads) {
      maximum = fmaxf(
          maximum,
          scores[query_within_kv * kRuntimeAttentionTokensPerSplit + token]);
    }
    maximum = warp_max(maximum);
    maximum = __shfl_sync(0xffffffffU, maximum, 0);

    float denominator = 0.0F;
    for (std::uint32_t token = lane; token < split_tokens;
         token += kWarpThreads) {
      const std::size_t score_index =
          static_cast<std::size_t>(query_within_kv) *
              kRuntimeAttentionTokensPerSplit +
          token;
      const float exponential = __expf(scores[score_index] - maximum);
      scores[score_index] = exponential;
      denominator += exponential;
    }
    denominator = warp_sum(denominator);
    if (lane == 0) {
      const std::size_t metadata_index =
          static_cast<std::size_t>(query_head) * split_count + split;
      split_maximum[metadata_index] = maximum;
      split_denominator[metadata_index] = denominator;
    }
  }
  __syncthreads();

  for (std::size_t index = threadIdx.x * 8; index < tile_elements;
       index += blockDim.x * 8) {
    const std::uint32_t token =
        static_cast<std::uint32_t>(index / HeadSize);
    const std::uint32_t dimension =
        static_cast<std::uint32_t>(index % HeadSize);
    const std::uint32_t absolute_token = first_absolute_token + token;
    const std::uint32_t slot =
        Local ? absolute_token % kGraphAttentionLocalCapacity
              : absolute_token;
    const std::size_t cache_index =
        (static_cast<std::size_t>(kv_head) * capacity + slot) * HeadSize +
        dimension;
    *reinterpret_cast<uint4*>(cache_tile + index) = kv_storage::load_eight(
        kv_storage::row(value_cache, cache_index / HeadSize, HeadSize, format),
        dimension, HeadSize, format);
  }
  __syncthreads();

  constexpr std::uint32_t kContextValuesPerThread =
      HeadSize / (kWarpsPerQuery * kWarpThreads);
#pragma unroll
  for (std::uint32_t index = 0; index < kContextValuesPerThread; ++index) {
    const std::uint32_t dimension =
        warp_within_query * kWarpThreads + lane +
        index * kWarpsPerQuery * kWarpThreads;
    float sum = 0.0F;
    for (std::uint32_t token = 0; token < split_tokens; ++token) {
      sum = fmaf(
          scores[query_within_kv * kRuntimeAttentionTokensPerSplit + token],
          __bfloat162float(
              cache_tile[static_cast<std::size_t>(token) * HeadSize +
                         dimension]),
          sum);
    }
    const std::size_t partial_index =
        (static_cast<std::size_t>(query_head) * split_count + split) *
            HeadSize +
        dimension;
    store_runtime_partial(partial_context, partial_index, sum);
  }
}

template <typename PartialContext, std::uint32_t HeadSize,
          std::uint32_t KvHeads, bool Local>
__global__ __launch_bounds__(kThreads, 2)
void causal_gqa_attention_cached_m1_fused_split_kernel(
    const BFloat16* query, const BFloat16* key_cache,
    const BFloat16* value_cache, std::uint32_t absolute_position,
    std::uint32_t capacity, std::uint32_t split_count,
    PartialContext* partial_context, float* split_maximum,
    float* split_denominator, kv_cache::Format format = kv_cache::Format::bf16) {
  causal_gqa_attention_cached_m1_fused_split_body<PartialContext, HeadSize, KvHeads, Local>(query, key_cache, value_cache, absolute_position, capacity, split_count, partial_context, split_maximum, split_denominator, blockIdx.x, blockIdx.y, format);
}

template <std::uint32_t TileStride, bool Paged, bool PadTokens>
__device__ __forceinline__ void stage_compact_global_tile(
    const BFloat16* cache, const std::uint64_t* page_offsets,
    std::size_t layer_offset_elements, std::uint32_t kv_head,
    std::uint32_t capacity, std::uint32_t split_begin,
    std::uint32_t split_tokens, BFloat16* tile, kv_cache::Format format = kv_cache::Format::bf16) {
  constexpr std::uint32_t kVectorElements = sizeof(int4) / sizeof(BFloat16);
  constexpr std::uint32_t kRowVectors = kGlobalCompactKvSize / kVectorElements;
  static_assert(kGlobalCompactKvSize % kVectorElements == 0);
  static_assert(TileStride % kVectorElements == 0);
  const std::uint32_t vectors =
      (PadTokens ? kRuntimeAttentionTokensPerSplit : split_tokens) * kRowVectors;
  for (std::uint32_t index = threadIdx.x; index < vectors;
       index += blockDim.x) {
    const std::uint32_t token = index / kRowVectors;
    const std::uint32_t dimension = (index % kRowVectors) * kVectorElements;
    BFloat16* const destination =
        tile + static_cast<std::size_t>(token) * TileStride + dimension;
    if (token >= split_tokens) {
      *reinterpret_cast<int4*>(destination) = int4{};
      continue;
    }
    const std::uint32_t position = split_begin + token;
    const BFloat16* record;
    if constexpr (Paged) {
      record = compact_global_cache::paged_row(cache, page_offsets, 256,
          layer_offset_elements, kv_head, position, format);
    } else {
      record = kv_storage::row(cache, std::size_t(kv_head) * capacity + position,
                               640, format, 2);
    }
    if (format == kv_cache::Format::fp8) {
      *reinterpret_cast<uint4*>(destination) =
          kv_storage::load_eight(record, dimension, 640, format, 128);
      continue;
    }
    const BFloat16* source = record + dimension;
    if ((reinterpret_cast<std::uintptr_t>(source) & (alignof(int4) - 1)) == 0) {
      *reinterpret_cast<int4*>(destination) = *reinterpret_cast<const int4*>(source);
    } else {
      // Cache views permit BF16-aligned offsets; only the shared tile has a
      // guaranteed vector alignment.
#pragma unroll
      for (std::uint32_t d = 0; d < kVectorElements; ++d)
        destination[d] = source[d];
    }
  }
}

template <typename PartialContext, bool Paged>
__device__ __forceinline__
void causal_gqa_attention_cached_m1_fused_global_compact_scalar_split_body(
    const BFloat16* query, const BFloat16* compact_kv_cache,
    const std::uint64_t* page_offsets, const BFloat16* k_norm_scale,
    std::uint32_t page_tokens, std::size_t layer_offset_elements,
    std::uint32_t absolute_position, std::uint32_t capacity,
    std::uint32_t split_count,
    PartialContext* partial_context, float* split_maximum,
    float* split_denominator, unsigned grid_x, unsigned grid_y) {
  constexpr std::uint32_t kHeadSize = gemma4_31b::kGlobalHeadSize;
  constexpr std::uint32_t kKvHeads = gemma4_31b::kGlobalKvHeadCount;
  constexpr std::uint32_t kQueriesPerKv =
      gemma4_31b::kQueryHeadCount / kKvHeads;
  constexpr std::uint32_t kWarps = kThreads / kWarpThreads;
  constexpr std::uint32_t kWarpsPerQuery = kWarps / kQueriesPerKv;
  static_assert(kGlobalCompactKeySize == 128);
  static_assert(kGlobalCompactKvSize == 640);
  static_assert(kWarpsPerQuery == 1);
  static_assert(kHeadSize % kWarpThreads == 0);
  static_assert(kRuntimeAttentionTokensPerSplit * kGlobalCompactKvSize *
                            sizeof(BFloat16) +
                        kQueriesPerKv * kRuntimeAttentionTokensPerSplit *
                            sizeof(float) <=
                    48U * 1'024U);

  // A complete compact tile remains resident across QK, softmax, and P*V.
  // Unlike the separate/full layouts, the value pass performs no cache load.
  __shared__ __align__(16) BFloat16
      compact_tile[kRuntimeAttentionTokensPerSplit * kGlobalCompactKvSize];
  __shared__ float scores[kQueriesPerKv *
                          kRuntimeAttentionTokensPerSplit];

  const std::uint32_t visible_tokens = absolute_position + 1;
  const std::uint32_t split = grid_y;
  const std::uint32_t split_begin = static_cast<std::uint32_t>(
      static_cast<std::uint64_t>(visible_tokens) * split / split_count);
  const std::uint32_t split_end = static_cast<std::uint32_t>(
      static_cast<std::uint64_t>(visible_tokens) * (split + 1) /
      split_count);
  const std::uint32_t split_tokens = split_end - split_begin;
  const std::uint32_t kv_head = grid_x;

  stage_compact_global_tile<kGlobalCompactKvSize, Paged, false>(
      compact_kv_cache, page_offsets, layer_offset_elements, kv_head, capacity,
      split_begin, split_tokens, compact_tile);
  __syncthreads();

  const std::uint32_t warp = threadIdx.x / kWarpThreads;
  const std::uint32_t lane = threadIdx.x % kWarpThreads;
  const std::uint32_t query_within_kv = warp;
  const std::uint32_t query_head =
      kv_head * kQueriesPerKv + query_within_kv;
  const std::size_t query_offset =
      static_cast<std::size_t>(query_head) * kHeadSize;
  constexpr std::uint32_t kQueryValuesPerLane =
      kHeadSize / kWarpThreads;
  float query_values[kQueryValuesPerLane];
#pragma unroll
  for (std::uint32_t index = 0; index < kQueryValuesPerLane; ++index) {
    const std::uint32_t dimension = lane + index * kWarpThreads;
    float query_value =
        __bfloat162float(query[query_offset + dimension]);
    const bool rotated = dimension < 64 ||
                         (dimension >= 256 && dimension < 320);
    if (!rotated) {
      // K[d] = V[d] * KNorm[d] for these dimensions. Moving KNorm onto Q
      // avoids one multiply and one BF16 rounding for every cached token.
      query_value *= __bfloat162float(k_norm_scale[dimension]);
    }
    query_values[index] = query_value;
  }

  for (std::uint32_t token = 0; token < split_tokens; ++token) {
    const std::size_t compact_row =
        static_cast<std::size_t>(token) * kGlobalCompactKvSize;
    float score = 0.0F;
#pragma unroll
    for (std::uint32_t index = 0; index < kQueryValuesPerLane; ++index) {
      const std::uint32_t dimension = lane + index * kWarpThreads;
      BFloat16 key_value;
      if (dimension < 64) {
        key_value = compact_tile[compact_row + dimension];
      } else if (dimension < 256) {
        key_value = compact_tile[compact_row + kGlobalCompactKeySize +
                                 dimension];
      } else if (dimension < 320) {
        key_value = compact_tile[compact_row + 64 + dimension - 256];
      } else {
        key_value = compact_tile[compact_row + kGlobalCompactKeySize +
                                 dimension];
      }
      score = fmaf(query_values[index], __bfloat162float(key_value), score);
    }
    score = warp_sum(score);
    if (lane == 0) {
      scores[query_within_kv * kRuntimeAttentionTokensPerSplit + token] =
          score;
    }
  }
  __syncthreads();

  float maximum = -__int_as_float(0x7f800000);
  for (std::uint32_t token = lane; token < split_tokens;
       token += kWarpThreads) {
    maximum = fmaxf(
        maximum,
        scores[query_within_kv * kRuntimeAttentionTokensPerSplit + token]);
  }
  maximum = warp_max(maximum);
  maximum = __shfl_sync(0xffffffffU, maximum, 0);

  float denominator = 0.0F;
  for (std::uint32_t token = lane; token < split_tokens;
       token += kWarpThreads) {
    const std::size_t score_index =
        static_cast<std::size_t>(query_within_kv) *
            kRuntimeAttentionTokensPerSplit +
        token;
    const float exponential = __expf(scores[score_index] - maximum);
    scores[score_index] = exponential;
    denominator += exponential;
  }
  denominator = warp_sum(denominator);
  if (lane == 0) {
    const std::size_t metadata_index =
        static_cast<std::size_t>(query_head) * split_count + split;
    split_maximum[metadata_index] = maximum;
    split_denominator[metadata_index] = denominator;
  }
  __syncthreads();

#pragma unroll
  for (std::uint32_t index = 0; index < kQueryValuesPerLane; ++index) {
    const std::uint32_t dimension = lane + index * kWarpThreads;
    float sum = 0.0F;
    for (std::uint32_t token = 0; token < split_tokens; ++token) {
      const std::size_t value_index =
          static_cast<std::size_t>(token) * kGlobalCompactKvSize +
          kGlobalCompactKeySize + dimension;
      sum = fmaf(
          scores[query_within_kv * kRuntimeAttentionTokensPerSplit + token],
          __bfloat162float(compact_tile[value_index]), sum);
    }
    const std::size_t partial_index =
        (static_cast<std::size_t>(query_head) * split_count + split) *
            kHeadSize +
        dimension;
    store_runtime_partial(partial_context, partial_index, sum);
  }
}

template <typename PartialContext, bool Paged>
__global__ __launch_bounds__(kThreads, 1)
void causal_gqa_attention_cached_m1_fused_global_compact_scalar_split_kernel(
    const BFloat16* query, const BFloat16* compact_kv_cache,
    const std::uint64_t* page_offsets, const BFloat16* k_norm_scale,
    std::uint32_t page_tokens, std::size_t layer_offset_elements,
    std::uint32_t absolute_position, std::uint32_t capacity,
    std::uint32_t split_count,
    PartialContext* partial_context, float* split_maximum,
    float* split_denominator) {
  causal_gqa_attention_cached_m1_fused_global_compact_scalar_split_body<PartialContext, Paged>(query, compact_kv_cache, page_offsets, k_norm_scale, page_tokens, layer_offset_elements, absolute_position, capacity, split_count, partial_context, split_maximum, split_denominator, blockIdx.x, blockIdx.y);
}

// Two BF16 components approximate the FP32
// scaled query and probability used by the scalar kernel.
__device__ __forceinline__ void decode_mma(float (&c)[4], unsigned a0,
    unsigned a1, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%4,%5,%5}, {%6,%7}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

template<bool Transpose>
__device__ __forceinline__ void decode_load(unsigned (&b)[2], const BFloat16* p) {
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(p));
  if constexpr (Transpose)
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
        : "=r"(b[0]), "=r"(b[1]) : "r"(address));
  else
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
        : "=r"(b[0]), "=r"(b[1]) : "r"(address));
}

template <bool Paged>
__device__ __forceinline__
void causal_gqa_attention_cached_m1_fused_global_compact_tensor_split_body(
    const BFloat16* query, const BFloat16* compact_kv_cache,
    const std::uint64_t* page_offsets, const BFloat16* k_norm_scale,
    std::uint32_t page_tokens, std::size_t layer_offset_elements,
    std::uint32_t absolute_position, std::uint32_t capacity,
    std::uint32_t split_count,
    __half* partial_context, float* split_maximum,
    float* split_denominator, unsigned grid_x, unsigned grid_y, kv_cache::Format format = kv_cache::Format::bf16) {
  constexpr unsigned Stride = 648;
  __shared__ __align__(32) BFloat16 tile[32 * Stride];
  __shared__ float warp_maxima[4 * 8], warp_sums[4 * 8];
  const unsigned visible = absolute_position + 1, split = grid_y;
  const unsigned begin = std::uint64_t(visible) * split / split_count;
  const unsigned end = std::uint64_t(visible) * (split + 1) / split_count;
  const unsigned count = end - begin;
  stage_compact_global_tile<Stride, Paged, true>(compact_kv_cache, page_offsets,
      layer_offset_elements, grid_x, capacity, begin, count, tile, format);
  __syncthreads();
  const unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const unsigned head = grid_x * 8 + lane / 4;
  // All eight warps stage KV; four warps own the 8-query tensor product.
  float score[4]{};
  if (warp < 4) {
#pragma unroll
    for (unsigned k = 0; k < 32; ++k) {
      unsigned hi[2], lo[2];
#pragma unroll
      for (unsigned part = 0; part < 2; ++part) {
        const unsigned d = k * 16 + (lane % 4) * 2 + part * 8;
        const bool rotated = d < 64 || (d >= 256 && d < 320);
        float x = __bfloat162float(query[head * 512 + d]);
        float y = __bfloat162float(query[head * 512 + d + 1]);
        if (!rotated) {
          x *= __bfloat162float(k_norm_scale[d]);
          y *= __bfloat162float(k_norm_scale[d + 1]);
        }
        const auto h = __floats2bfloat162_rn(x, y);
        const auto l = __floats2bfloat162_rn(x - __low2float(h), y - __high2float(h));
        hi[part] = reinterpret_cast<const unsigned&>(h);
        lo[part] = reinterpret_cast<const unsigned&>(l);
      }
      const unsigned d = k * 16;
      const unsigned compact = d < 64 ? d : d < 256 ? d + 128 : d < 320 ? d - 192 : d + 128;
      unsigned key[2];
      decode_load<false>(key, tile + (warp * 8 + lane % 8) * Stride + compact + ((lane / 8) % 2) * 8);
      decode_mma(score, hi[0], hi[1], key[0], key[1]);
      decode_mma(score, lo[0], lo[1], key[0], key[1]);
    }
#pragma unroll
    for (unsigned c = 0; c < 2; ++c)
      if (warp * 8 + (lane % 4) * 2 + c >= count) score[c] = -__int_as_float(0x7f800000);
    float maximum = fmaxf(score[0], score[1]);
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, 1));
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, 2));
    if (lane % 4 == 0) warp_maxima[warp * 8 + lane / 4] = maximum;
  }
  __syncthreads();
  float maximum = -__int_as_float(0x7f800000);
  if (warp < 4) {
#pragma unroll
    for (unsigned w = 0; w < 4; ++w) maximum = fmaxf(maximum, warp_maxima[w * 8 + lane / 4]);
    float sum = 0;
#pragma unroll
    for (unsigned c = 0; c < 2; ++c) {
      const unsigned token = warp * 8 + (lane % 4) * 2 + c;
      const float p = token < count ? __expf(score[c] - maximum) : 0;
      const auto hi = __float2bfloat16_rn(p), lo = __float2bfloat16_rn(p - __bfloat162float(hi));
      tile[token * Stride + lane / 4] = hi;
      tile[token * Stride + 8 + lane / 4] = lo;
      sum += p;
    }
    sum += __shfl_xor_sync(0xffffffffU, sum, 1);
    sum += __shfl_xor_sync(0xffffffffU, sum, 2);
    if (lane % 4 == 0) warp_sums[warp * 8 + lane / 4] = sum;
  }
  __syncthreads();
  if (warp < 4) {
    if (warp == 0 && lane % 4 == 0) {
      float sum = 0;
#pragma unroll
      for (unsigned w = 0; w < 4; ++w) sum += warp_sums[w * 8 + lane / 4];
      split_maximum[std::size_t(head) * split_count + split] = maximum;
      split_denominator[std::size_t(head) * split_count + split] = sum;
    }
    unsigned p[2][2], low[2][2];
#pragma unroll
    for (unsigned k = 0; k < 2; ++k) {
      decode_load<true>(p[k], tile + (k * 16 + lane % 16) * Stride);
      decode_load<true>(low[k], tile + (k * 16 + lane % 16) * Stride + 8);
    }
#pragma unroll
    for (unsigned n = 0; n < 16; ++n) {
      float numerator[4]{};
#pragma unroll
      for (unsigned k = 0; k < 2; ++k) {
        unsigned value[2];
        decode_load<true>(value, tile + (k * 16 + lane % 16) * Stride + 128 + warp * 8 + n * 32);
        decode_mma(numerator, p[k][0], p[k][1], value[0], value[1]);
        decode_mma(numerator, low[k][0], low[k][1], value[0], value[1]);
      }
      const unsigned d = warp * 8 + n * 32 + (lane % 4) * 2;
      const std::size_t output = (std::size_t(head) * split_count + split) * 512 + d;
      store_runtime_partial(partial_context, output, numerator[0]);
      store_runtime_partial(partial_context, output + 1, numerator[1]);
    }
  }
}

template <bool Paged>
__global__ __launch_bounds__(kThreads, 2)
void causal_gqa_attention_cached_m1_fused_global_compact_tensor_split_kernel(
    const BFloat16* query, const BFloat16* compact_kv_cache,
    const std::uint64_t* page_offsets, const BFloat16* k_norm_scale,
    std::uint32_t page_tokens, std::size_t layer_offset_elements,
    std::uint32_t absolute_position, std::uint32_t capacity,
    std::uint32_t split_count,
    __half* partial_context, float* split_maximum,
    float* split_denominator, kv_cache::Format format = kv_cache::Format::bf16) {
  causal_gqa_attention_cached_m1_fused_global_compact_tensor_split_body<Paged>(query, compact_kv_cache, page_offsets, k_norm_scale, page_tokens, layer_offset_elements, absolute_position, capacity, split_count, partial_context, split_maximum, split_denominator, blockIdx.x, blockIdx.y, format);
}

template <typename PartialContext, std::uint32_t HeadSize,
          std::uint32_t DimensionTile>
__device__ __forceinline__ void causal_gqa_attention_cached_m1_fused_finalize_body(
    const PartialContext* partial_context, const float* split_maximum,
    const float* split_denominator, std::uint32_t split_count,
    BFloat16* context, unsigned grid_x, unsigned grid_y) {
  static_assert(HeadSize % DimensionTile == 0);
  static_assert(HeadSize / DimensionTile ==
                kGraphAttentionFusedFinalizeTiles);
  extern __shared__ float split_scale[];
  __shared__ float inverse_denominator;

  const std::uint32_t query_head = grid_x;
  const std::size_t metadata_offset =
      static_cast<std::size_t>(query_head) * split_count;
  if (threadIdx.x < kWarpThreads) {
    float maximum = -__int_as_float(0x7f800000);
    for (std::uint32_t split = threadIdx.x; split < split_count;
         split += kWarpThreads) {
      maximum = fmaxf(maximum, split_maximum[metadata_offset + split]);
    }
    maximum = warp_max(maximum);
    maximum = __shfl_sync(0xffffffffU, maximum, 0);

    float denominator = 0.0F;
    for (std::uint32_t split = threadIdx.x; split < split_count;
         split += kWarpThreads) {
      const float scale =
          __expf(split_maximum[metadata_offset + split] - maximum);
      split_scale[split] = scale;
      denominator = fmaf(split_denominator[metadata_offset + split], scale,
                         denominator);
    }
    denominator = warp_sum(denominator);
    if (threadIdx.x == 0) {
      inverse_denominator = 1.0F / denominator;
    }
  }
  __syncthreads();

  const std::uint32_t dimension =
      static_cast<std::uint32_t>(grid_y) * DimensionTile + threadIdx.x;
  float sum = 0.0F;
  for (std::uint32_t split = 0; split < split_count; ++split) {
    const std::size_t partial_index =
        (metadata_offset + split) * HeadSize + dimension;
    sum = fmaf(load_runtime_partial(partial_context, partial_index),
               split_scale[split], sum);
  }
  context[static_cast<std::size_t>(query_head) * HeadSize + dimension] =
      __float2bfloat16_rn(sum * inverse_denominator);
}

template <typename PartialContext, std::uint32_t HeadSize,
          std::uint32_t DimensionTile>
__global__ void causal_gqa_attention_cached_m1_fused_finalize_kernel(
    const PartialContext* partial_context, const float* split_maximum,
    const float* split_denominator, std::uint32_t split_count,
    BFloat16* context) {
  causal_gqa_attention_cached_m1_fused_finalize_body<PartialContext, HeadSize, DimensionTile>(partial_context, split_maximum, split_denominator, split_count, context, blockIdx.x, blockIdx.y);
}

template <typename PartialContext, std::uint32_t HeadSize>
__device__ __forceinline__ void causal_gqa_attention_cached_m1_fused_coarse_reduce_body(
    const PartialContext* partial_context, const float* split_maximum,
    const float* split_denominator, std::uint32_t split_count,
    std::uint32_t coarse_split_count, float* coarse_partial_context,
    float* coarse_split_maximum, float* coarse_split_denominator, unsigned grid_x, unsigned grid_y) {
  static_assert(kRuntimeGlobalCoarseGroupSize == 2 * kWarpThreads);
  static_assert(HeadSize % kThreads == 0);
  __shared__ float split_scale[kRuntimeGlobalCoarseGroupSize];

  const std::uint32_t query_head = grid_x;
  const std::uint32_t coarse_split = grid_y;
  const std::uint32_t group_begin =
      coarse_split * kRuntimeGlobalCoarseGroupSize;
  const std::uint32_t group_end =
      min(group_begin + kRuntimeGlobalCoarseGroupSize, split_count);
  const std::uint32_t group_size = group_end - group_begin;
  const std::size_t fine_metadata_offset =
      static_cast<std::size_t>(query_head) * split_count;
  const std::size_t coarse_metadata_index =
      static_cast<std::size_t>(query_head) * coarse_split_count +
      coarse_split;

  if (threadIdx.x < kWarpThreads) {
    const std::uint32_t lane = threadIdx.x;
    float maximum = -__int_as_float(0x7f800000);
    if (lane < group_size) {
      maximum = split_maximum[fine_metadata_offset + group_begin + lane];
    }
    if (lane + kWarpThreads < group_size) {
      maximum = fmaxf(
          maximum,
          split_maximum[fine_metadata_offset + group_begin + lane +
                        kWarpThreads]);
    }
    maximum = warp_max(maximum);
    maximum = __shfl_sync(0xffffffffU, maximum, 0);

    float denominator = 0.0F;
    if (lane < group_size) {
      const std::size_t split_index =
          fine_metadata_offset + group_begin + lane;
      const float scale =
          __expf(split_maximum[split_index] - maximum);
      split_scale[lane] = scale;
      denominator = split_denominator[split_index] * scale;
    }
    if (lane + kWarpThreads < group_size) {
      const std::size_t split_index =
          fine_metadata_offset + group_begin + lane + kWarpThreads;
      const float scale =
          __expf(split_maximum[split_index] - maximum);
      split_scale[lane + kWarpThreads] = scale;
      denominator = fmaf(split_denominator[split_index], scale,
                         denominator);
    }
    denominator = warp_sum(denominator);
    if (lane == 0) {
      coarse_split_maximum[coarse_metadata_index] = maximum;
      coarse_split_denominator[coarse_metadata_index] = denominator;
    }
  }
  __syncthreads();

  for (std::uint32_t dimension = threadIdx.x; dimension < HeadSize;
       dimension += blockDim.x) {
    float sum = 0.0F;
    for (std::uint32_t split_within_group = 0;
         split_within_group < group_size; ++split_within_group) {
      const std::uint32_t split = group_begin + split_within_group;
      const std::size_t partial_index =
          (fine_metadata_offset + split) * HeadSize + dimension;
      sum = fmaf(load_runtime_partial(partial_context, partial_index),
                 split_scale[split_within_group], sum);
    }
    const std::size_t coarse_partial_index =
        coarse_metadata_index * HeadSize + dimension;
    coarse_partial_context[coarse_partial_index] = sum;
  }
}

template <typename PartialContext, std::uint32_t HeadSize>
__global__ void causal_gqa_attention_cached_m1_fused_coarse_reduce_kernel(
    const PartialContext* partial_context, const float* split_maximum,
    const float* split_denominator, std::uint32_t split_count,
    std::uint32_t coarse_split_count, float* coarse_partial_context,
    float* coarse_split_maximum, float* coarse_split_denominator) {
  causal_gqa_attention_cached_m1_fused_coarse_reduce_body<PartialContext, HeadSize>(partial_context, split_maximum, split_denominator, split_count, coarse_split_count, coarse_partial_context, coarse_split_maximum, coarse_split_denominator, blockIdx.x, blockIdx.y);
}

}  // namespace gewell::bf16_primitives::detail
