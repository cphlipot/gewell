#include "bf16_common.cuh"
#include "bf16_attention_detail.cuh"
#include "rope_element.cuh"
#include <algorithm>
#include <cmath>

namespace gewell::bf16_primitives {
namespace {

using namespace detail;

constexpr unsigned kDecodeBatchEntries = 16;
constexpr unsigned kFrozenLocalBatchEntries = 32;

struct FrozenLocalAttentionJob {
  FrozenLocalAttentionInput input;
  RuntimeFusedScratchLayout scratch;
};
struct FrozenLocalAttentionBatch {
  FrozenLocalAttentionJob jobs[kFrozenLocalBatchEntries];
};

__global__ void frozen_local_attention_batch_partial(
    const __grid_constant__ FrozenLocalAttentionBatch batch) {
  const unsigned job = blockIdx.y / kGraphAttentionFusedLocalSplitCount;
  const unsigned split = blockIdx.y % kGraphAttentionFusedLocalSplitCount;
  const auto& entry = batch.jobs[job];
  causal_gqa_attention_cached_m1_fused_split_body<float, 256, 16, true>(
      entry.input.query, entry.input.key_cache, entry.input.value_cache,
      entry.input.absolute_position, 1024,
      kGraphAttentionFusedLocalSplitCount,
      static_cast<float*>(entry.scratch.partial_context),
      entry.scratch.split_maximum, entry.scratch.split_denominator,
      blockIdx.x, split, entry.input.format);
}

__global__ void frozen_local_attention_batch_finalize(
    const __grid_constant__ FrozenLocalAttentionBatch batch) {
  const auto& entry = batch.jobs[blockIdx.z];
  causal_gqa_attention_cached_m1_fused_finalize_body<float, 256, 64>(
      static_cast<const float*>(entry.scratch.partial_context),
      entry.scratch.split_maximum, entry.scratch.split_denominator,
      kGraphAttentionFusedLocalSplitCount, entry.input.context,
      blockIdx.x, blockIdx.y);
}

void launch_frozen_local_attention_batch(const FrozenLocalAttentionBatch& batch,
                                         unsigned count,
                                         cudaStream_t stream) {
  frozen_local_attention_batch_partial<<<
      dim3(16, count * kGraphAttentionFusedLocalSplitCount), kThreads, 0,
      stream>>>(batch);
  check_cuda(cudaGetLastError(), "frozen local attention batch partial");
  frozen_local_attention_batch_finalize<<<dim3(32, 4, count), 64,
      kGraphAttentionFusedLocalSplitCount * sizeof(float), stream>>>(batch);
  check_cuda(cudaGetLastError(), "frozen local attention batch finalize");
}

struct DecodeAttentionJob {
  DecodeAttentionInput input;
  RuntimeFusedScratchLayout fine;
  RuntimeGlobalCoarseScratchLayout coarse;
  unsigned splits{}, end_splits{}, end_coarse{};
};
struct DecodeAttentionBatch {
  DecodeAttentionJob jobs[kDecodeBatchEntries];
};

template<bool Global>
__global__ void prepare_decode_attention_batch(
    const __grid_constant__ DecodeAttentionBatch batch) {
  constexpr unsigned D = Global ? 512 : 256;
  const auto& input = batch.jobs[blockIdx.y].input;
  const unsigned head = blockIdx.x;
  if (head < 32) {
    for (unsigned d = threadIdx.x; d < D; d += blockDim.x)
      input.rotated_query[head * D + d] = rope_element<D>(
          input.query + head * D, input.cosine, input.sine, d);
  } else {
    const unsigned kv_head = head - 32;
    const auto format = Global ? input.global_cache.format : input.format;
    if (format == kv_cache::Format::fp8) {
      if constexpr (Global) {
        const auto& cache = input.global_cache;
        auto* record = compact_global_cache::paged_row(cache.page_pool,
            cache.page_offsets, cache.page_tokens, cache.layer_offset_elements,
            kv_head, input.position, format);
        float values[3]{};
#pragma unroll
        for (unsigned i = 0; i < 3; ++i) {
          const unsigned d = threadIdx.x + i * 256;
          if (d < 128) values[i] = __bfloat162float(rope_element<D>(
              input.key + kv_head * D, input.cosine, input.sine, d < 64 ? d : d + 192));
          else if (d < 640) values[i] = __bfloat162float(input.value[kv_head * D + d - 128]);
        }
        kv_storage::store_fp8<640, 128>(record, values);
      } else {
        const auto index = std::size_t(kv_head) * 1024 + input.position % 1024;
        float k[1]{__bfloat162float(rope_element<D>(input.key + kv_head * D,
                   input.cosine, input.sine, threadIdx.x))};
        float v[1]{__bfloat162float(input.value[kv_head * D + threadIdx.x])};
        kv_storage::store_fp8<256>(kv_storage::row(input.local_key, index, D, format), k);
        kv_storage::store_fp8<256>(kv_storage::row(input.local_value, index, D, format), v);
      }
      return;
    }

    if constexpr (Global) {
      const auto& cache = input.global_cache;
      auto* row = compact_global_cache::paged_row(cache.page_pool,
          cache.page_offsets, cache.page_tokens, cache.layer_offset_elements,
          kv_head, input.position);
      if (threadIdx.x < 128) {
        const unsigned d = threadIdx.x < 64 ? threadIdx.x : threadIdx.x + 192;
        row[threadIdx.x] = rope_element<D>(input.key + kv_head * D,
            input.cosine, input.sine, d);
      }
      for (unsigned d = threadIdx.x; d < D; d += blockDim.x)
        row[128 + d] = input.value[kv_head * D + d];
    } else {
      const auto offset = (std::size_t(kv_head) * 1024 + input.position % 1024) * D;
      for (unsigned d = threadIdx.x; d < D; d += blockDim.x) {
        input.local_key[offset + d] = rope_element<D>(input.key + kv_head * D,
            input.cosine, input.sine, d);
        input.local_value[offset + d] = input.value[kv_head * D + d];
      }
    }
  }
}

template<bool Global, bool Coarse>
__global__ __launch_bounds__(kThreads, (Global && !Coarse) ? 1 : 2)
void decode_attention_batch_partial(const __grid_constant__ DecodeAttentionBatch batch,
                                    const BFloat16* k_norm_scale) {
  unsigned job = 0;
  while (blockIdx.y >= batch.jobs[job].end_splits) ++job;
  const auto& entry = batch.jobs[job];
  const auto& input = entry.input;
  const unsigned split = blockIdx.y - (job ? batch.jobs[job - 1].end_splits : 0);
  if constexpr (Global) {
    const auto& cache = input.global_cache;
    causal_gqa_attention_cached_m1_fused_global_compact_tensor_split_body<true>(
        input.rotated_query, cache.page_pool, cache.page_offsets, k_norm_scale,
        cache.page_tokens, cache.layer_offset_elements, input.position,
        cache.page_count * cache.page_tokens, entry.splits,
        static_cast<__half*>(entry.fine.partial_context), entry.fine.split_maximum,
        entry.fine.split_denominator, blockIdx.x, split, cache.format);
  } else {
    causal_gqa_attention_cached_m1_fused_split_body<float, 256, 16, true>(
        input.rotated_query, input.local_key, input.local_value, input.position,
        1024, entry.splits, static_cast<float*>(entry.fine.partial_context),
        entry.fine.split_maximum, entry.fine.split_denominator, blockIdx.x, split, input.format);
  }
}

__global__ void decode_attention_batch_coarse(
    const __grid_constant__ DecodeAttentionBatch batch) {
  unsigned job = 0;
  while (blockIdx.y >= batch.jobs[job].end_coarse) ++job;
  const auto& entry = batch.jobs[job];
  const unsigned split = blockIdx.y - (job ? batch.jobs[job - 1].end_coarse : 0);
  causal_gqa_attention_cached_m1_fused_coarse_reduce_body<__half, 512>(
      static_cast<__half*>(entry.fine.partial_context), entry.fine.split_maximum,
      entry.fine.split_denominator, entry.splits, entry.coarse.split_count,
      entry.coarse.partial_context, entry.coarse.split_maximum,
      entry.coarse.split_denominator, blockIdx.x, split);
}

template<bool Global, bool Coarse>
__global__ void decode_attention_batch_finalize(
    const __grid_constant__ DecodeAttentionBatch batch) {
  const auto& entry = batch.jobs[blockIdx.z];
  if constexpr (Coarse)
    causal_gqa_attention_cached_m1_fused_finalize_body<float, 512, 128>(
        entry.coarse.partial_context, entry.coarse.split_maximum,
        entry.coarse.split_denominator, entry.coarse.split_count,
        entry.input.context, blockIdx.x, blockIdx.y);
  else if constexpr (Global)
    causal_gqa_attention_cached_m1_fused_finalize_body<__half, 512, 128>(
        static_cast<__half*>(entry.fine.partial_context), entry.fine.split_maximum,
        entry.fine.split_denominator, entry.splits, entry.input.context,
        blockIdx.x, blockIdx.y);
  else
    causal_gqa_attention_cached_m1_fused_finalize_body<float, 256, 64>(
        static_cast<float*>(entry.fine.partial_context), entry.fine.split_maximum,
        entry.fine.split_denominator, entry.splits, entry.input.context,
        blockIdx.x, blockIdx.y);
}

template<bool Global, bool Coarse>
void launch_decode_attention_batch(const DecodeAttentionBatch& batch, unsigned count,
                                   unsigned max_splits, const BFloat16* k_norm_scale,
                                   cudaStream_t stream) {
  const auto& last = batch.jobs[count - 1];
  prepare_decode_attention_batch<Global><<<dim3(Global ? 36 : 48, count), kThreads, 0, stream>>>(batch);
  check_cuda(cudaGetLastError(), "prepare batched decode attention");
  decode_attention_batch_partial<Global, Coarse>
      <<<dim3(Global ? 4 : 16, last.end_splits), kThreads, 0, stream>>>(batch, k_norm_scale);
  check_cuda(cudaGetLastError(), "batched decode attention partial");
  if constexpr (Coarse) {
    decode_attention_batch_coarse<<<dim3(32, last.end_coarse), kThreads, 0, stream>>>(batch);
    check_cuda(cudaGetLastError(), "batched decode attention coarse reduction");
  }
  const unsigned final_splits = Coarse ? (max_splits + 63) / 64 : max_splits;
  decode_attention_batch_finalize<Global, Coarse>
      <<<dim3(32, 4, count), Global ? 128 : 64, final_splits * sizeof(float), stream>>>(batch);
  check_cuda(cudaGetLastError(), "batched decode attention finalize");
}

constexpr unsigned kGraphAttentionScoreStripes = 128;
constexpr unsigned kGraphAttentionContextThreads = 32;
constexpr unsigned kGraphAttentionFusedLocalMaxTokens =
    (kGraphAttentionLocalCapacity + kGraphAttentionFusedLocalSplitCount - 1) /
    kGraphAttentionFusedLocalSplitCount;
constexpr unsigned kGraphAttentionFusedGlobalMaxTokens =
    (kGraphAttentionGlobalCapacity +
     kGraphAttentionFusedGlobalSplitCount - 1) /
    kGraphAttentionFusedGlobalSplitCount;
template <std::uint32_t HeadSize, std::uint32_t KvHeads>
__global__ void causal_gqa_attention_cached_m1_kernel(
    const BFloat16* query, const BFloat16* key_cache,
    const BFloat16* value_cache, std::uint32_t absolute_position,
    std::uint32_t capacity, BFloat16* probabilities, BFloat16* context) {
  static_assert(HeadSize == gemma4_31b::kLocalHeadSize ||
                HeadSize == gemma4_31b::kGlobalHeadSize);
  static_assert(HeadSize % kThreads == 0);
  static_assert(gemma4_31b::kQueryHeadCount % KvHeads == 0);
  __shared__ float key_zero_sums[kThreads];
  __shared__ float key_one_sums[kThreads];
  __shared__ BFloat16 probability_values[2];

  const std::uint32_t query_head = blockIdx.x;
  const std::uint32_t repeats = gemma4_31b::kQueryHeadCount / KvHeads;
  const std::uint32_t kv_head = query_head / repeats;
  const std::size_t query_offset =
      static_cast<std::size_t>(query_head) * HeadSize;
  const std::size_t key_zero_offset =
      (static_cast<std::size_t>(kv_head) * capacity) * HeadSize;
  const std::size_t key_one_offset = key_zero_offset + HeadSize;

  float key_zero_sum = 0.0F;
  float key_one_sum = 0.0F;
  for (std::uint32_t dimension = threadIdx.x; dimension < HeadSize;
       dimension += blockDim.x) {
    const float query_value =
        __bfloat162float(query[query_offset + dimension]);
    key_zero_sum = fmaf(
        query_value,
        __bfloat162float(key_cache[key_zero_offset + dimension]),
        key_zero_sum);
    if (absolute_position == 1) {
      key_one_sum = fmaf(
          query_value,
          __bfloat162float(key_cache[key_one_offset + dimension]),
          key_one_sum);
    }
  }
  key_zero_sums[threadIdx.x] = key_zero_sum;
  key_one_sums[threadIdx.x] = key_one_sum;
  __syncthreads();

  for (unsigned offset = kThreads / 2; offset != 0; offset /= 2) {
    if (threadIdx.x < offset) {
      key_zero_sums[threadIdx.x] += key_zero_sums[threadIdx.x + offset];
      key_one_sums[threadIdx.x] += key_one_sums[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const BFloat16 score_zero = __float2bfloat16_rn(key_zero_sums[0]);
    const BFloat16 score_one = __float2bfloat16_rn(key_one_sums[0]);
    if (absolute_position == 0) {
      probability_values[0] = __float2bfloat16_rn(1.0F);
      probability_values[1] = __float2bfloat16_rn(0.0F);
    } else {
      const float score_zero_fp32 = __bfloat162float(score_zero);
      const float score_one_fp32 = __bfloat162float(score_one);
      const float maximum = fmaxf(score_zero_fp32, score_one_fp32);
      const float exponential_zero = expf(score_zero_fp32 - maximum);
      const float exponential_one = expf(score_one_fp32 - maximum);
      const float denominator = exponential_zero + exponential_one;
      probability_values[0] =
          __float2bfloat16_rn(exponential_zero / denominator);
      probability_values[1] =
          __float2bfloat16_rn(exponential_one / denominator);
    }
    const std::size_t probability_offset =
        static_cast<std::size_t>(query_head) * 2;
    probabilities[probability_offset] = probability_values[0];
    probabilities[probability_offset + 1] = probability_values[1];
  }
  __syncthreads();

  const float probability_zero = __bfloat162float(probability_values[0]);
  const float probability_one = __bfloat162float(probability_values[1]);
  const std::size_t value_zero_offset =
      (static_cast<std::size_t>(kv_head) * capacity) * HeadSize;
  const std::size_t value_one_offset = value_zero_offset + HeadSize;
  const std::size_t context_offset =
      static_cast<std::size_t>(query_head) * HeadSize;
  for (std::uint32_t dimension = threadIdx.x; dimension < HeadSize;
       dimension += blockDim.x) {
    const float product_zero =
        probability_zero *
        __bfloat162float(value_cache[value_zero_offset + dimension]);
    float sum = product_zero;
    if (absolute_position == 1) {
      sum = fmaf(
          probability_one,
          __bfloat162float(value_cache[value_one_offset + dimension]),
          product_zero);
    }
    context[context_offset + dimension] = __float2bfloat16_rn(sum);
  }
}

template <std::uint32_t HeadSize, std::uint32_t KvHeads>
__global__ void causal_gqa_attention_cached_m1_24_kernel(
    const BFloat16* query, const BFloat16* key_cache,
    const BFloat16* value_cache, std::uint32_t absolute_position,
    std::uint32_t capacity, BFloat16* probabilities, BFloat16* context) {
  static_assert(HeadSize == gemma4_31b::kLocalHeadSize ||
                HeadSize == gemma4_31b::kGlobalHeadSize);
  static_assert(HeadSize % kThreads == 0);
  static_assert(gemma4_31b::kQueryHeadCount % KvHeads == 0);
  __shared__ float score_sums[kCachedAttentionM1ShortCapacity][kThreads];
  __shared__ BFloat16 probability_values[kCachedAttentionM1ShortCapacity];

  const std::uint32_t query_head = blockIdx.x;
  const std::uint32_t repeats = gemma4_31b::kQueryHeadCount / KvHeads;
  const std::uint32_t kv_head = query_head / repeats;
  const std::size_t query_offset =
      static_cast<std::size_t>(query_head) * HeadSize;

  for (std::uint32_t token = 0; token <= absolute_position; ++token) {
    const std::uint32_t slot = token % capacity;
    const std::size_t key_offset =
        (static_cast<std::size_t>(kv_head) * capacity + slot) * HeadSize;
    float sum = 0.0F;
    for (std::uint32_t dimension = threadIdx.x; dimension < HeadSize;
         dimension += blockDim.x) {
      sum = fmaf(__bfloat162float(query[query_offset + dimension]),
                 __bfloat162float(key_cache[key_offset + dimension]), sum);
    }
    score_sums[token][threadIdx.x] = sum;
  }
  __syncthreads();

  for (unsigned offset = kThreads / 2; offset != 0; offset /= 2) {
    if (threadIdx.x < offset) {
      for (std::uint32_t token = 0; token <= absolute_position; ++token) {
        score_sums[token][threadIdx.x] +=
            score_sums[token][threadIdx.x + offset];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float maximum = -__int_as_float(0x7f800000);
    for (std::uint32_t token = 0; token <= absolute_position; ++token) {
      const BFloat16 score = __float2bfloat16_rn(score_sums[token][0]);
      score_sums[token][0] = __bfloat162float(score);
      maximum = fmaxf(maximum, score_sums[token][0]);
    }

    float denominator = 0.0F;
    for (std::uint32_t token = 0; token <= absolute_position; ++token) {
      const float exponential = expf(score_sums[token][0] - maximum);
      score_sums[token][0] = exponential;
      denominator += exponential;
    }

    const std::size_t probability_offset =
        static_cast<std::size_t>(query_head) *
        kCachedAttentionM1ShortCapacity;
    for (std::uint32_t token = 0; token < kCachedAttentionM1ShortCapacity;
         ++token) {
      const BFloat16 probability =
          token <= absolute_position
              ? __float2bfloat16_rn(score_sums[token][0] / denominator)
              : __float2bfloat16_rn(0.0F);
      probability_values[token] = probability;
      probabilities[probability_offset + token] = probability;
    }
  }
  __syncthreads();

  const std::size_t value_zero_offset =
      (static_cast<std::size_t>(kv_head) * capacity) * HeadSize;
  const std::size_t context_offset =
      static_cast<std::size_t>(query_head) * HeadSize;
  for (std::uint32_t dimension = threadIdx.x; dimension < HeadSize;
       dimension += blockDim.x) {
    float sum = __bfloat162float(probability_values[0]) *
                __bfloat162float(value_cache[value_zero_offset + dimension]);
    for (std::uint32_t token = 1; token <= absolute_position; ++token) {
      const std::uint32_t slot = token % capacity;
      const std::size_t value_offset =
          (static_cast<std::size_t>(kv_head) * capacity + slot) * HeadSize;
      sum = fmaf(__bfloat162float(probability_values[token]),
                 __bfloat162float(value_cache[value_offset + dimension]),
                 sum);
    }
    context[context_offset + dimension] = __float2bfloat16_rn(sum);
  }
}

template <std::uint32_t HeadSize, std::uint32_t KvHeads,
          std::uint32_t Capacity>
__global__ void causal_gqa_attention_cached_m1_boundary_score_kernel(
    const BFloat16* query, const BFloat16* key_cache,
    std::uint32_t first_visible_position, std::uint32_t absolute_position,
    BFloat16* score_scratch) {
  static_assert(HeadSize == gemma4_31b::kLocalHeadSize ||
                HeadSize == gemma4_31b::kGlobalHeadSize);
  static_assert(HeadSize % kThreads == 0);
  static_assert(gemma4_31b::kQueryHeadCount % KvHeads == 0);
  static_assert(Capacity == kCachedAttentionM1BoundaryLocalCapacity ||
                Capacity == kCachedAttentionM1BoundaryGlobalCapacity);
  __shared__ float score_sums[kThreads];

  const std::uint32_t query_head = blockIdx.x;
  const std::uint32_t repeats = gemma4_31b::kQueryHeadCount / KvHeads;
  const std::uint32_t kv_head = query_head / repeats;
  const std::size_t query_offset =
      static_cast<std::size_t>(query_head) * HeadSize;

  for (std::uint32_t absolute_token = first_visible_position;
       absolute_token <= absolute_position; ++absolute_token) {
    const std::uint32_t slot = absolute_token % Capacity;
    const std::size_t key_offset =
        (static_cast<std::size_t>(kv_head) * Capacity + slot) * HeadSize;
    float sum = 0.0F;
    for (std::uint32_t dimension = threadIdx.x; dimension < HeadSize;
         dimension += blockDim.x) {
      sum = fmaf(__bfloat162float(query[query_offset + dimension]),
                 __bfloat162float(key_cache[key_offset + dimension]), sum);
    }
    score_sums[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned offset = kThreads / 2; offset != 0; offset /= 2) {
      if (threadIdx.x < offset) {
        score_sums[threadIdx.x] += score_sums[threadIdx.x + offset];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const std::size_t score_index =
          static_cast<std::size_t>(query_head) *
              kCachedAttentionM1BoundaryPositionCount +
          absolute_token;
      score_scratch[score_index] = __float2bfloat16_rn(score_sums[0]);
    }
  }
}

__global__ void causal_gqa_attention_cached_m1_boundary_softmax_kernel(
    const BFloat16* score_scratch, std::uint32_t first_visible_position,
    std::uint32_t absolute_position, BFloat16* probabilities) {
  if (threadIdx.x != 0) {
    return;
  }
  const std::uint32_t query_head = blockIdx.x;
  const std::size_t row =
      static_cast<std::size_t>(query_head) *
      kCachedAttentionM1BoundaryPositionCount;

  float maximum = -__int_as_float(0x7f800000);
  for (std::uint32_t token = first_visible_position;
       token <= absolute_position; ++token) {
    maximum = fmaxf(maximum, __bfloat162float(score_scratch[row + token]));
  }
  float denominator = 0.0F;
  for (std::uint32_t token = first_visible_position;
       token <= absolute_position; ++token) {
    denominator +=
        expf(__bfloat162float(score_scratch[row + token]) - maximum);
  }
  for (std::uint32_t token = 0;
       token < kCachedAttentionM1BoundaryPositionCount; ++token) {
    if (token < first_visible_position || token > absolute_position) {
      probabilities[row + token] = __float2bfloat16_rn(0.0F);
    } else {
      const float exponential =
          expf(__bfloat162float(score_scratch[row + token]) - maximum);
      probabilities[row + token] =
          __float2bfloat16_rn(exponential / denominator);
    }
  }
}

template <std::uint32_t HeadSize, std::uint32_t KvHeads,
          std::uint32_t Capacity>
__global__ void causal_gqa_attention_cached_m1_boundary_context_kernel(
    const BFloat16* value_cache, std::uint32_t first_visible_position,
    std::uint32_t absolute_position, const BFloat16* probabilities,
    BFloat16* context) {
  static_assert(HeadSize == gemma4_31b::kLocalHeadSize ||
                HeadSize == gemma4_31b::kGlobalHeadSize);
  static_assert(HeadSize % kThreads == 0);
  static_assert(gemma4_31b::kQueryHeadCount % KvHeads == 0);
  static_assert(Capacity == kCachedAttentionM1BoundaryLocalCapacity ||
                Capacity == kCachedAttentionM1BoundaryGlobalCapacity);

  const std::uint32_t query_head = blockIdx.x;
  const std::uint32_t repeats = gemma4_31b::kQueryHeadCount / KvHeads;
  const std::uint32_t kv_head = query_head / repeats;
  const std::size_t probability_row =
      static_cast<std::size_t>(query_head) *
      kCachedAttentionM1BoundaryPositionCount;
  const std::size_t context_offset =
      static_cast<std::size_t>(query_head) * HeadSize;
  const std::uint32_t first_slot = first_visible_position % Capacity;
  const std::size_t first_value_offset =
      (static_cast<std::size_t>(kv_head) * Capacity + first_slot) * HeadSize;

  for (std::uint32_t dimension = threadIdx.x; dimension < HeadSize;
       dimension += blockDim.x) {
    float sum =
        __bfloat162float(probabilities[probability_row +
                                        first_visible_position]) *
        __bfloat162float(value_cache[first_value_offset + dimension]);
    for (std::uint32_t token = first_visible_position + 1;
         token <= absolute_position; ++token) {
      const std::uint32_t slot = token % Capacity;
      const std::size_t value_offset =
          (static_cast<std::size_t>(kv_head) * Capacity + slot) * HeadSize;
      sum = fmaf(__bfloat162float(probabilities[probability_row + token]),
                 __bfloat162float(value_cache[value_offset + dimension]),
                 sum);
    }
    context[context_offset + dimension] = __float2bfloat16_rn(sum);
  }
}

template <std::uint32_t HeadSize, std::uint32_t KvHeads,
          std::uint32_t Capacity>
__global__ void causal_gqa_attention_cached_m1_device_position_score_kernel(
    const BFloat16* query, const BFloat16* key_cache,
    const std::uint32_t* absolute_position, BFloat16* score_scratch) {
  static_assert(HeadSize == gemma4_31b::kLocalHeadSize ||
                HeadSize == gemma4_31b::kGlobalHeadSize);
  static_assert(HeadSize % kThreads == 0);
  static_assert(gemma4_31b::kQueryHeadCount % KvHeads == 0);
  static_assert(Capacity == kGraphAttentionLocalCapacity ||
                Capacity == kGraphAttentionGlobalCapacity);
  static_assert(Capacity % kGraphAttentionScoreStripes == 0);
  __shared__ float score_sums[kThreads];

  const std::uint32_t position = absolute_position[0];
  if (position < kGraphDecodeFirstPosition ||
      position > kGraphDecodeLastPosition) {
    return;
  }
  const std::uint32_t first_visible_position =
      Capacity == kGraphAttentionLocalCapacity
          ? position - (kGraphAttentionLocalCapacity - 1)
          : 0;
  const std::uint32_t query_head = blockIdx.x;
  const std::uint32_t token_stripe = blockIdx.y;
  const std::uint32_t repeats = gemma4_31b::kQueryHeadCount / KvHeads;
  const std::uint32_t kv_head = query_head / repeats;
  const std::size_t query_offset =
      static_cast<std::size_t>(query_head) * HeadSize;
  constexpr std::uint32_t kDimensionsPerThread = HeadSize / kThreads;
  float query_values[kDimensionsPerThread];
#pragma unroll
  for (std::uint32_t index = 0; index < kDimensionsPerThread; ++index) {
    query_values[index] = __bfloat162float(
        query[query_offset + threadIdx.x + index * kThreads]);
  }

  for (std::uint32_t absolute_token =
           first_visible_position + token_stripe;
       absolute_token <= position;
       absolute_token += kGraphAttentionScoreStripes) {
    const std::uint32_t slot =
        Capacity == kGraphAttentionGlobalCapacity
            ? absolute_token
            : absolute_token % kGraphAttentionLocalCapacity;
    const std::size_t key_offset =
        (static_cast<std::size_t>(kv_head) * Capacity + slot) * HeadSize;
    float sum = 0.0F;
#pragma unroll
    for (std::uint32_t index = 0; index < kDimensionsPerThread; ++index) {
      const std::uint32_t dimension = threadIdx.x + index * kThreads;
      sum = fmaf(query_values[index],
                 __bfloat162float(key_cache[key_offset + dimension]), sum);
    }
    score_sums[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned offset = kThreads / 2; offset >= kWarpThreads;
         offset /= 2) {
      if (threadIdx.x < offset) {
        score_sums[threadIdx.x] += score_sums[threadIdx.x + offset];
      }
      __syncthreads();
    }

    if (threadIdx.x < kWarpThreads) {
      float reduced_score = score_sums[threadIdx.x];
#pragma unroll
      for (unsigned offset = kWarpThreads / 2; offset != 0; offset /= 2) {
        reduced_score +=
            __shfl_down_sync(0xffffffffU, reduced_score, offset);
      }
      if (threadIdx.x == 0) {
        const std::size_t score_index =
            static_cast<std::size_t>(query_head) *
                kGraphAttentionPositionCount +
            absolute_token;
        score_scratch[score_index] = __float2bfloat16_rn(reduced_score);
      }
    }
  }
}

template <std::uint32_t Capacity>
__global__ void
causal_gqa_attention_cached_m1_device_position_softmax_kernel(
    const BFloat16* score_scratch,
    const std::uint32_t* absolute_position, BFloat16* probabilities) {
  static_assert(Capacity == kGraphAttentionLocalCapacity ||
                Capacity == kGraphAttentionGlobalCapacity);
  __shared__ float exponentials[kGraphAttentionPositionCount];
  __shared__ float maximum_value;
  __shared__ float denominator_value;

  const std::uint32_t position = absolute_position[0];
  if (position < kGraphDecodeFirstPosition ||
      position > kGraphDecodeLastPosition) {
    return;
  }
  const std::uint32_t first_visible_position =
      Capacity == kGraphAttentionLocalCapacity
          ? position - (kGraphAttentionLocalCapacity - 1)
          : 0;
  const std::uint32_t query_head = blockIdx.x;
  const std::size_t row =
      static_cast<std::size_t>(query_head) * kGraphAttentionPositionCount;

  if (threadIdx.x == 0) {
    float maximum = -__int_as_float(0x7f800000);
    for (std::uint32_t token = first_visible_position; token <= position;
         ++token) {
      maximum =
          fmaxf(maximum, __bfloat162float(score_scratch[row + token]));
    }
    maximum_value = maximum;
  }
  __syncthreads();

  for (std::uint32_t token = threadIdx.x;
       token < kGraphAttentionPositionCount; token += blockDim.x) {
    bool masked = token > position;
    if constexpr (Capacity == kGraphAttentionLocalCapacity) {
      masked = masked || token < first_visible_position;
    }
    exponentials[token] =
        masked ? 0.0F
               : expf(__bfloat162float(score_scratch[row + token]) -
                      maximum_value);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float denominator = 0.0F;
    for (std::uint32_t token = first_visible_position; token <= position;
         ++token) {
      denominator += exponentials[token];
    }
    denominator_value = denominator;
  }
  __syncthreads();

  for (std::uint32_t token = threadIdx.x;
       token < kGraphAttentionPositionCount; token += blockDim.x) {
    bool masked = token > position;
    if constexpr (Capacity == kGraphAttentionLocalCapacity) {
      masked = masked || token < first_visible_position;
    }
    if (masked) {
      probabilities[row + token] = __float2bfloat16_rn(0.0F);
    } else {
      probabilities[row + token] =
          __float2bfloat16_rn(exponentials[token] / denominator_value);
    }
  }
}

template <std::uint32_t HeadSize, std::uint32_t KvHeads,
          std::uint32_t Capacity>
__global__ void
causal_gqa_attention_cached_m1_device_position_context_kernel(
    const BFloat16* value_cache, const std::uint32_t* absolute_position,
    const BFloat16* probabilities, BFloat16* context) {
  static_assert(HeadSize == gemma4_31b::kLocalHeadSize ||
                HeadSize == gemma4_31b::kGlobalHeadSize);
  static_assert(HeadSize % kGraphAttentionContextThreads == 0);
  static_assert(gemma4_31b::kQueryHeadCount % KvHeads == 0);
  static_assert(Capacity == kGraphAttentionLocalCapacity ||
                Capacity == kGraphAttentionGlobalCapacity);

  const std::uint32_t position = absolute_position[0];
  if (position < kGraphDecodeFirstPosition ||
      position > kGraphDecodeLastPosition) {
    return;
  }
  const std::uint32_t first_visible_position =
      Capacity == kGraphAttentionLocalCapacity
          ? position - (kGraphAttentionLocalCapacity - 1)
          : 0;
  const std::uint32_t query_head = blockIdx.x;
  const std::uint32_t repeats = gemma4_31b::kQueryHeadCount / KvHeads;
  const std::uint32_t kv_head = query_head / repeats;
  const std::size_t probability_row =
      static_cast<std::size_t>(query_head) * kGraphAttentionPositionCount;
  const std::size_t context_offset =
      static_cast<std::size_t>(query_head) * HeadSize;
  const std::uint32_t first_slot = first_visible_position % Capacity;
  const std::size_t first_value_offset =
      (static_cast<std::size_t>(kv_head) * Capacity + first_slot) * HeadSize;

  const std::uint32_t dimension =
      static_cast<std::uint32_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (dimension < HeadSize) {
    float sum =
        __bfloat162float(
            probabilities[probability_row + first_visible_position]) *
        __bfloat162float(value_cache[first_value_offset + dimension]);
    for (std::uint32_t token = first_visible_position + 1;
         token <= position; ++token) {
      const std::uint32_t slot = token % Capacity;
      const std::size_t value_offset =
          (static_cast<std::size_t>(kv_head) * Capacity + slot) * HeadSize;
      sum = fmaf(__bfloat162float(probabilities[probability_row + token]),
                 __bfloat162float(value_cache[value_offset + dimension]),
                 sum);
    }
    context[context_offset + dimension] = __float2bfloat16_rn(sum);
  }
}

template <std::uint32_t HeadSize, std::uint32_t KvHeads,
          std::uint32_t Capacity, std::uint32_t SplitCount,
          std::uint32_t MaxSplitTokens>
__global__ __launch_bounds__(kThreads, 2)
void causal_gqa_attention_cached_m1_device_position_fused_split_kernel(
    const BFloat16* query, const BFloat16* key_cache,
    const BFloat16* value_cache,
    const std::uint32_t* absolute_position, float* partial_context,
    float* split_maximum, float* split_denominator) {
  static_assert(HeadSize == gemma4_31b::kLocalHeadSize ||
                HeadSize == gemma4_31b::kGlobalHeadSize);
  static_assert(gemma4_31b::kQueryHeadCount % KvHeads == 0);
  static_assert(Capacity == kGraphAttentionLocalCapacity ||
                Capacity == kGraphAttentionGlobalCapacity);
  static_assert(kThreads % kWarpThreads == 0);
  constexpr std::uint32_t kWarps = kThreads / kWarpThreads;
  constexpr std::uint32_t kQueriesPerKv =
      gemma4_31b::kQueryHeadCount / KvHeads;
  constexpr std::uint32_t kWarpsPerQuery = kWarps / kQueriesPerKv;
  static_assert(kWarps % kQueriesPerKv == 0);
  static_assert(HeadSize % kWarpThreads == 0);
  static_assert(HeadSize % (kWarpsPerQuery * kWarpThreads) == 0);
  static_assert(MaxSplitTokens * HeadSize * sizeof(BFloat16) +
                        kQueriesPerKv * MaxSplitTokens * sizeof(float) <=
                    48U * 1'024U);

  __shared__ __align__(16)
      BFloat16 cache_tile[MaxSplitTokens * HeadSize];
  __shared__ float scores[kQueriesPerKv * MaxSplitTokens];

  const std::uint32_t position = absolute_position[0];
  if (position < kGraphDecodeFirstPosition ||
      position > kGraphDecodeLastPosition) {
    return;
  }

  const std::uint32_t first_visible_position =
      Capacity == kGraphAttentionLocalCapacity
          ? position - (kGraphAttentionLocalCapacity - 1)
          : 0;
  const std::uint32_t visible_tokens =
      position - first_visible_position + 1;
  const std::uint32_t split = blockIdx.y;
  const std::uint32_t split_begin =
      static_cast<std::uint32_t>(
          static_cast<std::uint64_t>(visible_tokens) * split / SplitCount);
  const std::uint32_t split_end =
      static_cast<std::uint32_t>(
          static_cast<std::uint64_t>(visible_tokens) * (split + 1) /
          SplitCount);
  const std::uint32_t split_tokens = split_end - split_begin;
  const std::uint32_t first_absolute_token =
      first_visible_position + split_begin;
  const std::uint32_t kv_head = blockIdx.x;

  const std::size_t tile_elements =
      static_cast<std::size_t>(split_tokens) * HeadSize;
  for (std::size_t index = threadIdx.x; index < tile_elements;
       index += blockDim.x) {
    const std::uint32_t token =
        static_cast<std::uint32_t>(index / HeadSize);
    const std::uint32_t dimension =
        static_cast<std::uint32_t>(index % HeadSize);
    const std::uint32_t absolute_token = first_absolute_token + token;
    const std::uint32_t slot =
        Capacity == kGraphAttentionGlobalCapacity
            ? absolute_token
            : absolute_token % kGraphAttentionLocalCapacity;
    const std::size_t cache_index =
        (static_cast<std::size_t>(kv_head) * Capacity + slot) * HeadSize +
        dimension;
    cache_tile[index] = key_cache[cache_index];
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
  constexpr std::uint32_t kQueryValuesPerLane =
      HeadSize / kWarpThreads;
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
      scores[query_within_kv * MaxSplitTokens + token] = score;
    }
  }
  __syncthreads();

  if (warp_within_query == 0) {
    float maximum = -__int_as_float(0x7f800000);
    for (std::uint32_t token = lane; token < split_tokens;
         token += kWarpThreads) {
      maximum = fmaxf(
          maximum,
          scores[query_within_kv * MaxSplitTokens + token]);
    }
    maximum = warp_max(maximum);
    maximum = __shfl_sync(0xffffffffU, maximum, 0);

    float denominator = 0.0F;
    for (std::uint32_t token = lane; token < split_tokens;
         token += kWarpThreads) {
      const std::size_t score_index =
          static_cast<std::size_t>(query_within_kv) * MaxSplitTokens +
          token;
      const float exponential = __expf(scores[score_index] - maximum);
      scores[score_index] = exponential;
      denominator += exponential;
    }
    denominator = warp_sum(denominator);
    if (lane == 0) {
      const std::size_t metadata_index =
          static_cast<std::size_t>(query_head) * SplitCount + split;
      split_maximum[metadata_index] = maximum;
      split_denominator[metadata_index] = denominator;
    }
  }
  __syncthreads();

  for (std::size_t index = threadIdx.x; index < tile_elements;
       index += blockDim.x) {
    const std::uint32_t token =
        static_cast<std::uint32_t>(index / HeadSize);
    const std::uint32_t dimension =
        static_cast<std::uint32_t>(index % HeadSize);
    const std::uint32_t absolute_token = first_absolute_token + token;
    const std::uint32_t slot =
        Capacity == kGraphAttentionGlobalCapacity
            ? absolute_token
            : absolute_token % kGraphAttentionLocalCapacity;
    const std::size_t cache_index =
        (static_cast<std::size_t>(kv_head) * Capacity + slot) * HeadSize +
        dimension;
    cache_tile[index] = value_cache[cache_index];
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
          scores[query_within_kv * MaxSplitTokens + token],
          __bfloat162float(
              cache_tile[static_cast<std::size_t>(token) * HeadSize +
                         dimension]),
          sum);
    }
    const std::size_t partial_index =
        (static_cast<std::size_t>(query_head) * SplitCount + split) *
            HeadSize +
        dimension;
    partial_context[partial_index] = sum;
  }
}

template <std::uint32_t HeadSize, std::uint32_t SplitCount,
          std::uint32_t DimensionTile>
__global__ void
causal_gqa_attention_cached_m1_device_position_fused_finalize_kernel(
    const std::uint32_t* absolute_position, const float* partial_context,
    const float* split_maximum, const float* split_denominator,
    BFloat16* context) {
  static_assert(HeadSize % DimensionTile == 0);
  static_assert(HeadSize / DimensionTile ==
                kGraphAttentionFusedFinalizeTiles);
  __shared__ float split_scale[SplitCount];
  __shared__ float inverse_denominator;

  const std::uint32_t position = absolute_position[0];
  if (position < kGraphDecodeFirstPosition ||
      position > kGraphDecodeLastPosition) {
    return;
  }

  const std::uint32_t query_head = blockIdx.x;
  const std::size_t metadata_offset =
      static_cast<std::size_t>(query_head) * SplitCount;
  if (threadIdx.x < kWarpThreads) {
    float maximum = -__int_as_float(0x7f800000);
    for (std::uint32_t split = threadIdx.x; split < SplitCount;
         split += kWarpThreads) {
      maximum = fmaxf(maximum, split_maximum[metadata_offset + split]);
    }
    maximum = warp_max(maximum);
    maximum = __shfl_sync(0xffffffffU, maximum, 0);

    float denominator = 0.0F;
    for (std::uint32_t split = threadIdx.x; split < SplitCount;
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
      static_cast<std::uint32_t>(blockIdx.y) * DimensionTile + threadIdx.x;
  float sum = 0.0F;
  for (std::uint32_t split = 0; split < SplitCount; ++split) {
    const std::size_t partial_index =
        (metadata_offset + split) * HeadSize + dimension;
    sum = fmaf(partial_context[partial_index], split_scale[split], sum);
  }
  context[static_cast<std::size_t>(query_head) * HeadSize + dimension] =
      __float2bfloat16_rn(sum * inverse_denominator);
}

template <bool Paged>
void launch_causal_gqa_attention_cached_m1_fused_global_compact(
    const BFloat16* query, const BFloat16* compact_kv_cache,
    const std::uint64_t* page_offsets, const BFloat16* k_norm_scale,
    std::uint32_t page_tokens, std::size_t layer_offset_elements,
    std::uint32_t absolute_position, std::uint32_t capacity,
    std::size_t scratch_bytes, void* scratch, BFloat16* context,
    cudaStream_t stream, std::string_view operation, kv_cache::Format format) {
  const std::uint32_t split_count =
      (absolute_position + 1 + kRuntimeAttentionTokensPerSplit - 1) /
      kRuntimeAttentionTokensPerSplit;
  const RuntimeFusedScratchLayout storage = runtime_fused_scratch_layout(
      scratch, split_count, gemma4_31b::kGlobalHeadSize, true);
  if (storage.bytes > scratch_bytes) {
    fail(operation, "internal scratch layout mismatch");
  }
  auto* const partial_context =
      static_cast<__half*>(storage.partial_context);

  const dim3 split_grid{gemma4_31b::kGlobalKvHeadCount, split_count};
  causal_gqa_attention_cached_m1_fused_global_compact_tensor_split_kernel<
      Paged><<<split_grid, kThreads, 0, stream>>>(
      query, compact_kv_cache, page_offsets, k_norm_scale, page_tokens,
      layer_offset_elements, absolute_position, capacity, split_count,
      partial_context, storage.split_maximum, storage.split_denominator, format);
  check_cuda(cudaGetLastError(),
             std::string(operation) + " split kernel launch");

  constexpr dim3 kFinalizeGrid{gemma4_31b::kQueryHeadCount,
                               kGraphAttentionFusedFinalizeTiles};
  constexpr std::uint32_t kDimensionTile =
      gemma4_31b::kGlobalHeadSize / kGraphAttentionFusedFinalizeTiles;
  if (split_count >= kRuntimeGlobalCoarseMinimumSplitCount) {
    const RuntimeGlobalCoarseScratchLayout coarse =
        runtime_global_coarse_scratch_layout(scratch, storage.bytes,
                                             split_count);
    if (coarse.bytes > scratch_bytes) {
      fail(operation, "coarse scratch layout exceeds allocation");
    }
    const dim3 coarse_grid{gemma4_31b::kQueryHeadCount,
                           coarse.split_count};
    causal_gqa_attention_cached_m1_fused_coarse_reduce_kernel<
        __half, gemma4_31b::kGlobalHeadSize>
        <<<coarse_grid, kThreads, 0, stream>>>(
            partial_context, storage.split_maximum,
            storage.split_denominator, split_count, coarse.split_count,
            coarse.partial_context, coarse.split_maximum,
            coarse.split_denominator);
    check_cuda(cudaGetLastError(),
               std::string(operation) + " coarse reduction kernel launch");
    const std::size_t shared_bytes =
        static_cast<std::size_t>(coarse.split_count) * sizeof(float);
    causal_gqa_attention_cached_m1_fused_finalize_kernel<
        float, gemma4_31b::kGlobalHeadSize, kDimensionTile>
        <<<kFinalizeGrid, kDimensionTile, shared_bytes, stream>>>(
            coarse.partial_context, coarse.split_maximum,
            coarse.split_denominator, coarse.split_count, context);
  } else {
    const std::size_t shared_bytes =
        static_cast<std::size_t>(split_count) * sizeof(float);
    causal_gqa_attention_cached_m1_fused_finalize_kernel<
        __half, gemma4_31b::kGlobalHeadSize, kDimensionTile>
        <<<kFinalizeGrid, kDimensionTile, shared_bytes, stream>>>(
            partial_context, storage.split_maximum,
            storage.split_denominator, split_count, context);
  }
  check_cuda(cudaGetLastError(),
             std::string(operation) + " finalize kernel launch");
}

template <std::uint32_t HeadSize, std::uint32_t KvHeads>
__global__ void causal_gqa_attention_m2_kernel(
    const BFloat16* query, const BFloat16* key,
    const BFloat16* value_token_major, BFloat16* probabilities,
    BFloat16* context_token_major) {
  static_assert(HeadSize == gemma4_31b::kLocalHeadSize ||
                HeadSize == gemma4_31b::kGlobalHeadSize);
  static_assert(HeadSize % kThreads == 0);
  static_assert(gemma4_31b::kQueryHeadCount % KvHeads == 0);
  __shared__ float key_zero_sums[kThreads];
  __shared__ float key_one_sums[kThreads];
  __shared__ BFloat16 probability_values[2];

  const std::uint32_t query_head = blockIdx.x / 2;
  const std::uint32_t query_token = blockIdx.x % 2;
  const std::uint32_t repeats = gemma4_31b::kQueryHeadCount / KvHeads;
  const std::uint32_t kv_head = query_head / repeats;
  const std::size_t query_offset =
      (static_cast<std::size_t>(query_head) * 2 + query_token) * HeadSize;
  const std::size_t key_zero_offset =
      (static_cast<std::size_t>(kv_head) * 2) * HeadSize;
  const std::size_t key_one_offset = key_zero_offset + HeadSize;

  float key_zero_sum = 0.0F;
  float key_one_sum = 0.0F;
  for (std::uint32_t dimension = threadIdx.x; dimension < HeadSize;
       dimension += blockDim.x) {
    const float query_value =
        __bfloat162float(query[query_offset + dimension]);
    key_zero_sum = fmaf(query_value,
                        __bfloat162float(key[key_zero_offset + dimension]),
                        key_zero_sum);
    key_one_sum = fmaf(query_value,
                       __bfloat162float(key[key_one_offset + dimension]),
                       key_one_sum);
  }
  key_zero_sums[threadIdx.x] = key_zero_sum;
  key_one_sums[threadIdx.x] = key_one_sum;
  __syncthreads();

  for (unsigned offset = kThreads / 2; offset != 0; offset /= 2) {
    if (threadIdx.x < offset) {
      key_zero_sums[threadIdx.x] += key_zero_sums[threadIdx.x + offset];
      key_one_sums[threadIdx.x] += key_one_sums[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    // The matmul result is BF16 before the mask and FP32 softmax in eager
    // attention. For query token zero, the second key is causally masked and
    // therefore has exactly zero probability.
    const BFloat16 score_zero = __float2bfloat16_rn(key_zero_sums[0]);
    const BFloat16 score_one = __float2bfloat16_rn(key_one_sums[0]);
    if (query_token == 0) {
      probability_values[0] = __float2bfloat16_rn(1.0F);
      probability_values[1] = __float2bfloat16_rn(0.0F);
    } else {
      const float score_zero_fp32 = __bfloat162float(score_zero);
      const float score_one_fp32 = __bfloat162float(score_one);
      const float maximum = fmaxf(score_zero_fp32, score_one_fp32);
      const float exponential_zero = expf(score_zero_fp32 - maximum);
      const float exponential_one = expf(score_one_fp32 - maximum);
      const float denominator = exponential_zero + exponential_one;
      probability_values[0] =
          __float2bfloat16_rn(exponential_zero / denominator);
      probability_values[1] =
          __float2bfloat16_rn(exponential_one / denominator);
    }
    const std::size_t probability_offset =
        (static_cast<std::size_t>(query_head) * 2 + query_token) * 2;
    probabilities[probability_offset] = probability_values[0];
    probabilities[probability_offset + 1] = probability_values[1];
  }
  __syncthreads();

  const float probability_zero = __bfloat162float(probability_values[0]);
  const float probability_one = __bfloat162float(probability_values[1]);
  const std::size_t value_zero_offset =
      static_cast<std::size_t>(kv_head) * HeadSize;
  const std::size_t value_one_offset =
      (static_cast<std::size_t>(KvHeads) + kv_head) * HeadSize;
  const std::size_t context_offset =
      (static_cast<std::size_t>(query_token) *
           gemma4_31b::kQueryHeadCount +
       query_head) *
      HeadSize;
  for (std::uint32_t dimension = threadIdx.x; dimension < HeadSize;
       dimension += blockDim.x) {
    const float product_zero =
        probability_zero *
        __bfloat162float(value_token_major[value_zero_offset + dimension]);
    const float sum = fmaf(
        probability_one,
        __bfloat162float(value_token_major[value_one_offset + dimension]),
        product_zero);
    context_token_major[context_offset + dimension] =
        __float2bfloat16_rn(sum);
  }
}

}  // namespace

void causal_gqa_attention_m2(const BFloat16* query, const BFloat16* key,
                             const BFloat16* value_token_major,
                             BFloat16* probabilities,
                             BFloat16* context_token_major,
                             gemma4_31b::AttentionKind kind,
                             cudaStream_t stream) {
  check_pointer(query, "causal_gqa_attention_m2 query");
  check_pointer(key, "causal_gqa_attention_m2 key");
  check_pointer(value_token_major, "causal_gqa_attention_m2 value");
  check_pointer(probabilities, "causal_gqa_attention_m2 probabilities");
  check_pointer(context_token_major, "causal_gqa_attention_m2 context");
  if (kind != gemma4_31b::AttentionKind::local &&
      kind != gemma4_31b::AttentionKind::global) {
    fail("causal_gqa_attention_m2", "invalid Gemma 4 attention kind");
  }

  constexpr unsigned kBlocks = 2 * gemma4_31b::kQueryHeadCount;
  if (kind == gemma4_31b::AttentionKind::global) {
    causal_gqa_attention_m2_kernel<gemma4_31b::kGlobalHeadSize,
                                   gemma4_31b::kGlobalKvHeadCount>
        <<<kBlocks, kThreads, 0, stream>>>(query, key, value_token_major,
                                           probabilities,
                                           context_token_major);
  } else {
    causal_gqa_attention_m2_kernel<gemma4_31b::kLocalHeadSize,
                                   gemma4_31b::kLocalKvHeadCount>
        <<<kBlocks, kThreads, 0, stream>>>(query, key, value_token_major,
                                           probabilities,
                                           context_token_major);
  }
  check_cuda(cudaGetLastError(), "causal_gqa_attention_m2 kernel launch");
}

void causal_gqa_attention_cached_m1(
    const BFloat16* query, const BFloat16* key_cache,
    const BFloat16* value_cache, std::uint32_t absolute_position,
    std::uint32_t capacity, BFloat16* probabilities, BFloat16* context,
    gemma4_31b::AttentionKind kind, cudaStream_t stream) {
  check_pointer(query, "causal_gqa_attention_cached_m1 query");
  check_pointer(key_cache, "causal_gqa_attention_cached_m1 key cache");
  check_pointer(value_cache, "causal_gqa_attention_cached_m1 value cache");
  check_pointer(probabilities,
                "causal_gqa_attention_cached_m1 probabilities");
  check_pointer(context, "causal_gqa_attention_cached_m1 context");
  if (kind != gemma4_31b::AttentionKind::local &&
      kind != gemma4_31b::AttentionKind::global) {
    fail("causal_gqa_attention_cached_m1",
         "invalid Gemma 4 attention kind");
  }
  if (absolute_position > 1) {
    fail("causal_gqa_attention_cached_m1",
         "only absolute positions zero and one are implemented");
  }
  if (capacity <= absolute_position) {
    fail("causal_gqa_attention_cached_m1",
         "cache capacity cannot retain every causal position");
  }

  constexpr unsigned kBlocks = gemma4_31b::kQueryHeadCount;
  if (kind == gemma4_31b::AttentionKind::global) {
    causal_gqa_attention_cached_m1_kernel<
        gemma4_31b::kGlobalHeadSize, gemma4_31b::kGlobalKvHeadCount>
        <<<kBlocks, kThreads, 0, stream>>>(
            query, key_cache, value_cache, absolute_position, capacity,
            probabilities, context);
  } else {
    causal_gqa_attention_cached_m1_kernel<
        gemma4_31b::kLocalHeadSize, gemma4_31b::kLocalKvHeadCount>
        <<<kBlocks, kThreads, 0, stream>>>(
            query, key_cache, value_cache, absolute_position, capacity,
            probabilities, context);
  }
  check_cuda(cudaGetLastError(),
             "causal_gqa_attention_cached_m1 kernel launch");
}

void causal_gqa_attention_cached_m1_24(
    const BFloat16* query, const BFloat16* key_cache,
    const BFloat16* value_cache, std::uint32_t absolute_position,
    std::uint32_t capacity, BFloat16* probabilities, BFloat16* context,
    gemma4_31b::AttentionKind kind, cudaStream_t stream) {
  check_pointer(query, "causal_gqa_attention_cached_m1_24 query");
  check_pointer(key_cache, "causal_gqa_attention_cached_m1_24 key cache");
  check_pointer(value_cache,
                "causal_gqa_attention_cached_m1_24 value cache");
  check_pointer(probabilities,
                "causal_gqa_attention_cached_m1_24 probabilities");
  check_pointer(context, "causal_gqa_attention_cached_m1_24 context");
  if (kind != gemma4_31b::AttentionKind::local &&
      kind != gemma4_31b::AttentionKind::global) {
    fail("causal_gqa_attention_cached_m1_24",
         "invalid Gemma 4 attention kind");
  }
  if (absolute_position >= kCachedAttentionM1ShortCapacity) {
    fail("causal_gqa_attention_cached_m1_24",
         "absolute position exceeds the fixed short-decode range");
  }
  if (capacity <= absolute_position) {
    fail("causal_gqa_attention_cached_m1_24",
         "cache capacity cannot retain every causal position");
  }

  constexpr unsigned kBlocks = gemma4_31b::kQueryHeadCount;
  if (kind == gemma4_31b::AttentionKind::global) {
    causal_gqa_attention_cached_m1_24_kernel<
        gemma4_31b::kGlobalHeadSize, gemma4_31b::kGlobalKvHeadCount>
        <<<kBlocks, kThreads, 0, stream>>>(
            query, key_cache, value_cache, absolute_position, capacity,
            probabilities, context);
  } else {
    causal_gqa_attention_cached_m1_24_kernel<
        gemma4_31b::kLocalHeadSize, gemma4_31b::kLocalKvHeadCount>
        <<<kBlocks, kThreads, 0, stream>>>(
            query, key_cache, value_cache, absolute_position, capacity,
            probabilities, context);
  }
  check_cuda(cudaGetLastError(),
             "causal_gqa_attention_cached_m1_24 kernel launch");
}

void causal_gqa_attention_cached_m1_boundary(
    const BFloat16* query, const BFloat16* key_cache,
    const BFloat16* value_cache, std::uint32_t absolute_position,
    BFloat16* score_scratch, BFloat16* probabilities, BFloat16* context,
    gemma4_31b::AttentionKind kind, cudaStream_t stream) {
  check_pointer(query, "causal_gqa_attention_cached_m1_boundary query");
  check_pointer(key_cache,
                "causal_gqa_attention_cached_m1_boundary key cache");
  check_pointer(value_cache,
                "causal_gqa_attention_cached_m1_boundary value cache");
  check_pointer(score_scratch,
                "causal_gqa_attention_cached_m1_boundary score scratch");
  check_pointer(probabilities,
                "causal_gqa_attention_cached_m1_boundary probabilities");
  check_pointer(context, "causal_gqa_attention_cached_m1_boundary context");
  if (kind != gemma4_31b::AttentionKind::local &&
      kind != gemma4_31b::AttentionKind::global) {
    fail("causal_gqa_attention_cached_m1_boundary",
         "invalid Gemma 4 attention kind");
  }
  if (absolute_position >= kCachedAttentionM1BoundaryPositionCount) {
    fail("causal_gqa_attention_cached_m1_boundary",
         "absolute position exceeds the fixed boundary-proof range");
  }

  const bool global = kind == gemma4_31b::AttentionKind::global;
  const std::uint32_t first_visible_position =
      global || absolute_position < kCachedAttentionM1BoundaryLocalCapacity
          ? 0
          : absolute_position -
                (kCachedAttentionM1BoundaryLocalCapacity - 1);
  constexpr unsigned kHeadBlocks = gemma4_31b::kQueryHeadCount;
  if (global) {
    causal_gqa_attention_cached_m1_boundary_score_kernel<
        gemma4_31b::kGlobalHeadSize, gemma4_31b::kGlobalKvHeadCount,
        kCachedAttentionM1BoundaryGlobalCapacity>
        <<<kHeadBlocks, kThreads, 0, stream>>>(
            query, key_cache, first_visible_position, absolute_position,
            score_scratch);
  } else {
    causal_gqa_attention_cached_m1_boundary_score_kernel<
        gemma4_31b::kLocalHeadSize, gemma4_31b::kLocalKvHeadCount,
        kCachedAttentionM1BoundaryLocalCapacity>
        <<<kHeadBlocks, kThreads, 0, stream>>>(
            query, key_cache, first_visible_position, absolute_position,
            score_scratch);
  }
  check_cuda(cudaGetLastError(),
             "causal_gqa_attention_cached_m1_boundary score kernel launch");

  causal_gqa_attention_cached_m1_boundary_softmax_kernel
      <<<kHeadBlocks, 1, 0, stream>>>(score_scratch, first_visible_position,
                                      absolute_position, probabilities);
  check_cuda(cudaGetLastError(),
             "causal_gqa_attention_cached_m1_boundary softmax kernel launch");

  if (global) {
    causal_gqa_attention_cached_m1_boundary_context_kernel<
        gemma4_31b::kGlobalHeadSize, gemma4_31b::kGlobalKvHeadCount,
        kCachedAttentionM1BoundaryGlobalCapacity>
        <<<kHeadBlocks, kThreads, 0, stream>>>(
            value_cache, first_visible_position, absolute_position,
            probabilities, context);
  } else {
    causal_gqa_attention_cached_m1_boundary_context_kernel<
        gemma4_31b::kLocalHeadSize, gemma4_31b::kLocalKvHeadCount,
        kCachedAttentionM1BoundaryLocalCapacity>
        <<<kHeadBlocks, kThreads, 0, stream>>>(
            value_cache, first_visible_position, absolute_position,
            probabilities, context);
  }
  check_cuda(cudaGetLastError(),
             "causal_gqa_attention_cached_m1_boundary context kernel launch");
}

void causal_gqa_attention_cached_m1_device_position(
    const BFloat16* query, const BFloat16* key_cache,
    const BFloat16* value_cache,
    const std::uint32_t* absolute_position, BFloat16* score_scratch,
    BFloat16* probabilities, BFloat16* context,
    gemma4_31b::AttentionKind kind, cudaStream_t stream) {
  check_pointer(query,
                "causal_gqa_attention_cached_m1_device_position query");
  check_pointer(
      key_cache,
      "causal_gqa_attention_cached_m1_device_position key cache");
  check_pointer(
      value_cache,
      "causal_gqa_attention_cached_m1_device_position value cache");
  check_pointer(
      absolute_position,
      "causal_gqa_attention_cached_m1_device_position position");
  check_pointer(
      score_scratch,
      "causal_gqa_attention_cached_m1_device_position score scratch");
  check_pointer(
      probabilities,
      "causal_gqa_attention_cached_m1_device_position probabilities");
  check_pointer(context,
                "causal_gqa_attention_cached_m1_device_position context");
  if (kind != gemma4_31b::AttentionKind::local &&
      kind != gemma4_31b::AttentionKind::global) {
    fail("causal_gqa_attention_cached_m1_device_position",
         "invalid Gemma 4 attention kind");
  }

  constexpr unsigned kHeadBlocks = gemma4_31b::kQueryHeadCount;
  constexpr dim3 kScoreGrid{kHeadBlocks, kGraphAttentionScoreStripes};
  const bool global = kind == gemma4_31b::AttentionKind::global;
  if (global) {
    causal_gqa_attention_cached_m1_device_position_score_kernel<
        gemma4_31b::kGlobalHeadSize, gemma4_31b::kGlobalKvHeadCount,
        kGraphAttentionGlobalCapacity>
        <<<kScoreGrid, kThreads, 0, stream>>>(
            query, key_cache, absolute_position, score_scratch);
  } else {
    causal_gqa_attention_cached_m1_device_position_score_kernel<
        gemma4_31b::kLocalHeadSize, gemma4_31b::kLocalKvHeadCount,
        kGraphAttentionLocalCapacity>
        <<<kScoreGrid, kThreads, 0, stream>>>(
            query, key_cache, absolute_position, score_scratch);
  }
  check_cuda(
      cudaGetLastError(),
      "causal_gqa_attention_cached_m1_device_position score kernel launch");

  if (global) {
    causal_gqa_attention_cached_m1_device_position_softmax_kernel<
        kGraphAttentionGlobalCapacity>
        <<<kHeadBlocks, kThreads, 0, stream>>>(
            score_scratch, absolute_position, probabilities);
  } else {
    causal_gqa_attention_cached_m1_device_position_softmax_kernel<
        kGraphAttentionLocalCapacity>
        <<<kHeadBlocks, kThreads, 0, stream>>>(
            score_scratch, absolute_position, probabilities);
  }
  check_cuda(
      cudaGetLastError(),
      "causal_gqa_attention_cached_m1_device_position softmax kernel launch");

  if (global) {
    constexpr dim3 kContextGrid{
        kHeadBlocks,
        gemma4_31b::kGlobalHeadSize / kGraphAttentionContextThreads};
    causal_gqa_attention_cached_m1_device_position_context_kernel<
        gemma4_31b::kGlobalHeadSize, gemma4_31b::kGlobalKvHeadCount,
        kGraphAttentionGlobalCapacity>
        <<<kContextGrid, kGraphAttentionContextThreads, 0, stream>>>(
            value_cache, absolute_position, probabilities, context);
  } else {
    constexpr dim3 kContextGrid{
        kHeadBlocks,
        gemma4_31b::kLocalHeadSize / kGraphAttentionContextThreads};
    causal_gqa_attention_cached_m1_device_position_context_kernel<
        gemma4_31b::kLocalHeadSize, gemma4_31b::kLocalKvHeadCount,
        kGraphAttentionLocalCapacity>
        <<<kContextGrid, kGraphAttentionContextThreads, 0, stream>>>(
            value_cache, absolute_position, probabilities, context);
  }
  check_cuda(
      cudaGetLastError(),
      "causal_gqa_attention_cached_m1_device_position context kernel launch");
}

void causal_gqa_attention_cached_m1_device_position_fused(
    const BFloat16* query, const BFloat16* key_cache,
    const BFloat16* value_cache,
    const std::uint32_t* absolute_position, void* scratch,
    BFloat16* context, gemma4_31b::AttentionKind kind,
    cudaStream_t stream) {
  check_pointer(
      query,
      "causal_gqa_attention_cached_m1_device_position_fused query");
  check_pointer(
      key_cache,
      "causal_gqa_attention_cached_m1_device_position_fused key cache");
  check_pointer(
      value_cache,
      "causal_gqa_attention_cached_m1_device_position_fused value cache");
  check_pointer(
      absolute_position,
      "causal_gqa_attention_cached_m1_device_position_fused position");
  check_pointer(
      scratch,
      "causal_gqa_attention_cached_m1_device_position_fused scratch");
  check_pointer(
      context,
      "causal_gqa_attention_cached_m1_device_position_fused context");
  if (kind != gemma4_31b::AttentionKind::local &&
      kind != gemma4_31b::AttentionKind::global) {
    fail("causal_gqa_attention_cached_m1_device_position_fused",
         "invalid Gemma 4 attention kind");
  }
  if (reinterpret_cast<std::uintptr_t>(scratch) % alignof(float) != 0) {
    fail("causal_gqa_attention_cached_m1_device_position_fused",
         "scratch is not float-aligned");
  }

  float* const partial_context = static_cast<float*>(scratch);
  constexpr dim3 kFinalizeGrid{
      gemma4_31b::kQueryHeadCount,
      kGraphAttentionFusedFinalizeTiles};
  const bool global = kind == gemma4_31b::AttentionKind::global;
  if (global) {
    constexpr std::uint32_t kSplitCount =
        kGraphAttentionFusedGlobalSplitCount;
    constexpr std::size_t kPartialElements =
        static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) *
        kSplitCount * gemma4_31b::kGlobalHeadSize;
    constexpr std::size_t kMetadataElements =
        static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * kSplitCount;
    static_assert((kPartialElements + 2 * kMetadataElements) *
                      sizeof(float) ==
                  kGraphAttentionFusedScratchBytes);
    float* const split_maximum = partial_context + kPartialElements;
    float* const split_denominator = split_maximum + kMetadataElements;
    constexpr dim3 kSplitGrid{gemma4_31b::kGlobalKvHeadCount,
                              kSplitCount};
    causal_gqa_attention_cached_m1_device_position_fused_split_kernel<
        gemma4_31b::kGlobalHeadSize, gemma4_31b::kGlobalKvHeadCount,
        kGraphAttentionGlobalCapacity, kSplitCount,
        kGraphAttentionFusedGlobalMaxTokens>
        <<<kSplitGrid, kThreads, 0, stream>>>(
            query, key_cache, value_cache, absolute_position,
            partial_context, split_maximum, split_denominator);
    check_cuda(
        cudaGetLastError(),
        "causal_gqa_attention_cached_m1_device_position_fused global split "
        "kernel launch");
    constexpr std::uint32_t kDimensionTile =
        gemma4_31b::kGlobalHeadSize /
        kGraphAttentionFusedFinalizeTiles;
    causal_gqa_attention_cached_m1_device_position_fused_finalize_kernel<
        gemma4_31b::kGlobalHeadSize, kSplitCount, kDimensionTile>
        <<<kFinalizeGrid, kDimensionTile, 0, stream>>>(
            absolute_position, partial_context, split_maximum,
            split_denominator, context);
  } else {
    constexpr std::uint32_t kSplitCount =
        kGraphAttentionFusedLocalSplitCount;
    constexpr std::size_t kPartialElements =
        static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) *
        kSplitCount * gemma4_31b::kLocalHeadSize;
    constexpr std::size_t kMetadataElements =
        static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * kSplitCount;
    static_assert((kPartialElements + 2 * kMetadataElements) *
                      sizeof(float) <=
                  kGraphAttentionFusedScratchBytes);
    float* const split_maximum = partial_context + kPartialElements;
    float* const split_denominator = split_maximum + kMetadataElements;
    constexpr dim3 kSplitGrid{gemma4_31b::kLocalKvHeadCount,
                              kSplitCount};
    causal_gqa_attention_cached_m1_device_position_fused_split_kernel<
        gemma4_31b::kLocalHeadSize, gemma4_31b::kLocalKvHeadCount,
        kGraphAttentionLocalCapacity, kSplitCount,
        kGraphAttentionFusedLocalMaxTokens>
        <<<kSplitGrid, kThreads, 0, stream>>>(
            query, key_cache, value_cache, absolute_position,
            partial_context, split_maximum, split_denominator);
    check_cuda(
        cudaGetLastError(),
        "causal_gqa_attention_cached_m1_device_position_fused local split "
        "kernel launch");
    constexpr std::uint32_t kDimensionTile =
        gemma4_31b::kLocalHeadSize /
        kGraphAttentionFusedFinalizeTiles;
    causal_gqa_attention_cached_m1_device_position_fused_finalize_kernel<
        gemma4_31b::kLocalHeadSize, kSplitCount, kDimensionTile>
        <<<kFinalizeGrid, kDimensionTile, 0, stream>>>(
            absolute_position, partial_context, split_maximum,
            split_denominator, context);
  }
  check_cuda(
      cudaGetLastError(),
      "causal_gqa_attention_cached_m1_device_position_fused finalize kernel "
      "launch");
}

std::size_t causal_gqa_attention_cached_m1_fused_scratch_bytes(
    std::uint32_t capacity, gemma4_31b::AttentionKind kind) {
  if (kind != gemma4_31b::AttentionKind::local &&
      kind != gemma4_31b::AttentionKind::global) {
    fail("causal_gqa_attention_cached_m1_fused_scratch_bytes",
         "invalid Gemma 4 attention kind");
  }
  const bool global = kind == gemma4_31b::AttentionKind::global;
  if ((!global && capacity != kGraphAttentionLocalCapacity) ||
      (global && (capacity == 0 || capacity > kMaxContextTokenCount))) {
    fail("causal_gqa_attention_cached_m1_fused_scratch_bytes",
         global ? "global capacity must be in 1..262144"
                : "local capacity must be exactly 1024");
  }
  const std::size_t split_count =
      (static_cast<std::size_t>(capacity) +
       kRuntimeAttentionTokensPerSplit - 1) /
      kRuntimeAttentionTokensPerSplit;
  const std::uint32_t head_size =
      global ? gemma4_31b::kGlobalHeadSize : gemma4_31b::kLocalHeadSize;
  return runtime_fused_scratch_bytes_for_splits(split_count, head_size,
                                                 global) +
         (global ? runtime_global_coarse_scratch_bytes_for_splits(split_count)
                 : 0);
}

void causal_gqa_attention_cached_m1_fused(
    const BFloat16* query, const BFloat16* key_cache,
    const BFloat16* value_cache, std::uint32_t absolute_position,
    std::uint32_t capacity, void* scratch, BFloat16* context,
    gemma4_31b::AttentionKind kind, cudaStream_t stream,
    kv_cache::Format format) {
  check_pointer(query, "causal_gqa_attention_cached_m1_fused query");
  check_pointer(key_cache,
                "causal_gqa_attention_cached_m1_fused key cache");
  check_pointer(value_cache,
                "causal_gqa_attention_cached_m1_fused value cache");
  check_pointer(scratch, "causal_gqa_attention_cached_m1_fused scratch");
  check_pointer(context, "causal_gqa_attention_cached_m1_fused context");
  if (reinterpret_cast<std::uintptr_t>(scratch) % alignof(float) != 0) {
    fail("causal_gqa_attention_cached_m1_fused",
         "scratch is not float-aligned");
  }
  if (absolute_position >= kMaxContextTokenCount) {
    fail("causal_gqa_attention_cached_m1_fused",
         "absolute position must be in 0..262143");
  }
  const std::size_t scratch_bytes =
      causal_gqa_attention_cached_m1_fused_scratch_bytes(capacity, kind);
  const bool global = kind == gemma4_31b::AttentionKind::global;
  if (global && absolute_position >= capacity) {
    fail("causal_gqa_attention_cached_m1_fused",
         "global capacity must retain the absolute position");
  }

  const std::uint32_t split_count =
      global
          ? (absolute_position + 1 + kRuntimeAttentionTokensPerSplit - 1) /
                kRuntimeAttentionTokensPerSplit
          : kGraphAttentionFusedLocalSplitCount;
  const std::uint32_t head_size =
      global ? gemma4_31b::kGlobalHeadSize : gemma4_31b::kLocalHeadSize;
  const RuntimeFusedScratchLayout storage = runtime_fused_scratch_layout(
      scratch, split_count, head_size, global);
  if (storage.bytes > scratch_bytes) {
    fail("causal_gqa_attention_cached_m1_fused",
         "internal scratch layout mismatch");
  }

  const dim3 split_grid{
      global ? gemma4_31b::kGlobalKvHeadCount
             : gemma4_31b::kLocalKvHeadCount,
      split_count};
  if (global) {
    auto* const partial_context =
        static_cast<__half*>(storage.partial_context);
    causal_gqa_attention_cached_m1_fused_split_kernel<
        __half, gemma4_31b::kGlobalHeadSize,
        gemma4_31b::kGlobalKvHeadCount, false>
        <<<split_grid, kThreads, 0, stream>>>(
            query, key_cache, value_cache, absolute_position, capacity,
            split_count, partial_context, storage.split_maximum,
            storage.split_denominator, format);
  } else {
    auto* const partial_context =
        static_cast<float*>(storage.partial_context);
    causal_gqa_attention_cached_m1_fused_split_kernel<
        float, gemma4_31b::kLocalHeadSize,
        gemma4_31b::kLocalKvHeadCount, true>
        <<<split_grid, kThreads, 0, stream>>>(
            query, key_cache, value_cache, absolute_position, capacity,
            split_count, partial_context, storage.split_maximum,
            storage.split_denominator, format);
  }
  check_cuda(cudaGetLastError(),
             "causal_gqa_attention_cached_m1_fused split kernel launch");

  constexpr dim3 kFinalizeGrid{gemma4_31b::kQueryHeadCount,
                               kGraphAttentionFusedFinalizeTiles};
  const std::size_t shared_bytes =
      static_cast<std::size_t>(split_count) * sizeof(float);
  if (global) {
    const auto* const partial_context =
        static_cast<const __half*>(storage.partial_context);
    constexpr std::uint32_t kDimensionTile =
        gemma4_31b::kGlobalHeadSize /
        kGraphAttentionFusedFinalizeTiles;
    causal_gqa_attention_cached_m1_fused_finalize_kernel<
        __half, gemma4_31b::kGlobalHeadSize, kDimensionTile>
        <<<kFinalizeGrid, kDimensionTile, shared_bytes, stream>>>(
            partial_context, storage.split_maximum,
            storage.split_denominator, split_count, context);
  } else {
    const auto* const partial_context =
        static_cast<const float*>(storage.partial_context);
    constexpr std::uint32_t kDimensionTile =
        gemma4_31b::kLocalHeadSize /
        kGraphAttentionFusedFinalizeTiles;
    causal_gqa_attention_cached_m1_fused_finalize_kernel<
        float, gemma4_31b::kLocalHeadSize, kDimensionTile>
        <<<kFinalizeGrid, kDimensionTile, shared_bytes, stream>>>(
            partial_context, storage.split_maximum,
            storage.split_denominator, split_count, context);
  }
  check_cuda(cudaGetLastError(),
             "causal_gqa_attention_cached_m1_fused finalize kernel launch");
}

void causal_gqa_attention_cached_m1_fused_local_batch(
    const std::vector<FrozenLocalAttentionInput>& inputs, void* scratch,
    std::size_t scratch_size, cudaStream_t stream) {
  constexpr auto operation =
      "causal_gqa_attention_cached_m1_fused_local_batch";
  if (inputs.empty()) fail(operation, "empty batch");
  check_pointer(scratch, operation);
  if (reinterpret_cast<std::uintptr_t>(scratch) % alignof(float))
    fail(operation, "scratch is not float-aligned");
  const auto request_bytes =
      causal_gqa_attention_cached_m1_fused_scratch_bytes(
          1024, gemma4_31b::AttentionKind::local);
  if (scratch_size < request_bytes)
    fail(operation, "scratch cannot fit one request");
  for (const auto& input : inputs) {
    check_pointer(input.query, operation);
    check_pointer(input.key_cache, operation);
    check_pointer(input.value_cache, operation);
    check_pointer(input.context, operation);
    if (input.absolute_position >= kMaxContextTokenCount)
      fail(operation, "absolute position exceeds context limit");
  }
  if (inputs.size() == 1) {
    const auto& input = inputs.front();
    causal_gqa_attention_cached_m1_fused(
        input.query, input.key_cache, input.value_cache,
        input.absolute_position, 1024, scratch, input.context,
        gemma4_31b::AttentionKind::local, stream, input.format);
    return;
  }

  const auto group_capacity = std::min<std::size_t>(
      kFrozenLocalBatchEntries, scratch_size / request_bytes);
  FrozenLocalAttentionBatch batch{};
  for (std::size_t first = 0; first < inputs.size();
       first += group_capacity) {
    const auto count = static_cast<unsigned>(
        std::min(group_capacity, inputs.size() - first));
    for (unsigned i = 0; i < count; ++i) {
      auto& job = batch.jobs[i];
      job.input = inputs[first + i];
      job.scratch = runtime_fused_scratch_layout(
          static_cast<std::uint8_t*>(scratch) + i * request_bytes,
          kGraphAttentionFusedLocalSplitCount, 256, false);
    }
    launch_frozen_local_attention_batch(batch, count, stream);
  }
}

void causal_gqa_attention_cached_m1_fused_global_compact(
    const BFloat16* query, const BFloat16* compact_kv_cache,
    const BFloat16* k_norm_scale, std::uint32_t absolute_position,
    std::uint32_t capacity, void* scratch, BFloat16* context,
    cudaStream_t stream,
    kv_cache::Format format) {
  check_pointer(
      query, "causal_gqa_attention_cached_m1_fused_global_compact query");
  check_pointer(
      compact_kv_cache,
      "causal_gqa_attention_cached_m1_fused_global_compact cache");
  check_pointer(k_norm_scale,
                "causal_gqa_attention_cached_m1_fused_global_compact scale");
  check_pointer(
      scratch,
      "causal_gqa_attention_cached_m1_fused_global_compact scratch");
  check_pointer(
      context,
      "causal_gqa_attention_cached_m1_fused_global_compact context");
  if (reinterpret_cast<std::uintptr_t>(scratch) % alignof(float) != 0) {
    fail("causal_gqa_attention_cached_m1_fused_global_compact",
         "scratch is not float-aligned");
  }
  if (absolute_position >= kMaxContextTokenCount) {
    fail("causal_gqa_attention_cached_m1_fused_global_compact",
         "absolute position must be in 0..262143");
  }
  const std::size_t scratch_bytes =
      causal_gqa_attention_cached_m1_fused_scratch_bytes(
          capacity, gemma4_31b::AttentionKind::global);
  if (absolute_position >= capacity) {
    fail("causal_gqa_attention_cached_m1_fused_global_compact",
         "capacity must retain the absolute position");
  }

  launch_causal_gqa_attention_cached_m1_fused_global_compact<false>(
      query, compact_kv_cache, nullptr, k_norm_scale, 0, 0,
      absolute_position, capacity, scratch_bytes, scratch, context, stream,
      "causal_gqa_attention_cached_m1_fused_global_compact", format);
}

void causal_gqa_attention_cached_m1_fused_global_compact_paged(
    const BFloat16* query,
    const compact_global_cache::PagedView<BFloat16>& cache,
    const BFloat16* k_norm_scale, std::uint32_t absolute_position,
    void* scratch, BFloat16* context, cudaStream_t stream) {
  constexpr std::string_view kOperation =
      "causal_gqa_attention_cached_m1_fused_global_compact_paged";
  check_pointer(query, "paged compact-global attention query");
  check_pointer(cache.page_pool, "paged compact-global attention cache");
  check_pointer(cache.page_offsets,
                "paged compact-global attention page table");
  check_pointer(k_norm_scale, "paged compact-global attention scale");
  check_pointer(scratch, "paged compact-global attention scratch");
  check_pointer(context, "paged compact-global attention context");
  if (reinterpret_cast<std::uintptr_t>(scratch) % alignof(float) != 0) {
    fail(kOperation, "scratch is not float-aligned");
  }
  if (absolute_position >= kMaxContextTokenCount) {
    fail(kOperation, "absolute position must be in 0..262143");
  }
  if (cache.page_tokens != 256 || cache.page_count == 0) {
    fail(kOperation, "page table must use nonempty 256-token pages");
  }
  const std::uint64_t capacity =
      static_cast<std::uint64_t>(cache.page_count) * cache.page_tokens;
  if (capacity > kMaxContextTokenCount || absolute_position >= capacity) {
    fail(kOperation, "page table must cover the absolute position");
  }
  const std::size_t layer_elements =
      static_cast<std::size_t>(gemma4_31b::kGlobalKvHeadCount) *
      cache.page_tokens * kv_cache::row_words(kGlobalCompactKvSize, cache.format, 2);
  if (cache.layer_offset_elements >
          std::numeric_limits<std::size_t>::max() - layer_elements ||
      cache.page_stride_elements < cache.layer_offset_elements +
                                        layer_elements) {
    fail(kOperation, "page stride does not cover one global layer");
  }
  const auto logical_capacity = static_cast<std::uint32_t>(capacity);
  const std::size_t scratch_bytes =
      causal_gqa_attention_cached_m1_fused_scratch_bytes(
          absolute_position + 1, gemma4_31b::AttentionKind::global);
  launch_causal_gqa_attention_cached_m1_fused_global_compact<true>(
      query, cache.page_pool, cache.page_offsets, k_norm_scale,
      cache.page_tokens, cache.layer_offset_elements, absolute_position,
      logical_capacity, scratch_bytes, scratch, context, stream, kOperation, cache.format);
}

void decode_attention_batch(const std::vector<DecodeAttentionInput>& inputs,
    const BFloat16* k_norm_scale, void* scratch, std::size_t scratch_size,
    gemma4_31b::AttentionKind kind, cudaStream_t stream) {
  constexpr auto operation = "decode_attention_batch";
  const bool global = kind == gemma4_31b::AttentionKind::global;
  if (!global && kind != gemma4_31b::AttentionKind::local)
    fail(operation, "invalid attention kind");
  if (inputs.empty()) fail(operation, "empty batch");
  check_pointer(scratch, operation);
  if (reinterpret_cast<std::uintptr_t>(scratch) % alignof(float))
    fail(operation, "scratch is not float-aligned");
  if (global) check_pointer(k_norm_scale, operation);
  // Validate every request before any cache writes are enqueued.
  for (const auto& input : inputs) {
    for (const auto* pointer : {input.query, input.key, input.value, input.cosine,
                                input.sine, static_cast<const BFloat16*>(input.rotated_query),
                                static_cast<const BFloat16*>(input.context)})
      check_pointer(pointer, operation);
    if (input.position >= kMaxContextTokenCount) fail(operation, "position exceeds context limit");
    if (global) {
      const auto& cache = input.global_cache;
      check_pointer(cache.page_pool, operation);
      check_pointer(cache.page_offsets, operation);
      const std::uint64_t capacity = std::uint64_t(cache.page_count) * cache.page_tokens;
      const std::size_t layer_elements = 4 * 256 *
          kv_cache::row_words(kGlobalCompactKvSize, cache.format, 2);
      if (cache.page_tokens != 256 || capacity > kMaxContextTokenCount ||
          input.position >= capacity)
        fail(operation, "page table must cover the position with 256-token pages");
      if (cache.layer_offset_elements > std::numeric_limits<std::size_t>::max() - layer_elements ||
          cache.page_stride_elements < cache.layer_offset_elements + layer_elements)
        fail(operation, "page stride does not cover one global layer");
    } else {
      check_pointer(input.local_key, operation);
      check_pointer(input.local_value, operation);
    }
    if (scratch_size < causal_gqa_attention_cached_m1_fused_scratch_bytes(
            global ? input.position + 1 : 1024, kind))
      fail(operation, "scratch cannot fit one request");
  }

  DecodeAttentionBatch batch{};
  unsigned count = 0, splits_total = 0, coarse_total = 0, max_splits = 0;
  std::size_t used = 0;
  bool coarse = false;
  const auto flush = [&] {
    if (!count) return;
    if (!global) launch_decode_attention_batch<false, false>(batch, count, max_splits, nullptr, stream);
    else if (coarse) launch_decode_attention_batch<true, true>(batch, count, max_splits, k_norm_scale, stream);
    else launch_decode_attention_batch<true, false>(batch, count, max_splits, k_norm_scale, stream);
    count = splits_total = coarse_total = max_splits = 0;
    used = 0;
  };
  for (const auto& input : inputs) {
    const unsigned splits = global ? (input.position + 32) / 32 : 32;
    const bool use_coarse = global && splits >= kRuntimeGlobalCoarseMinimumSplitCount;
    const auto bytes = causal_gqa_attention_cached_m1_fused_scratch_bytes(
        global ? input.position + 1 : 1024, kind);
    if (count == kDecodeBatchEntries || bytes > scratch_size - used ||
        splits_total + splits > 65535 || (count && use_coarse != coarse)) flush();
    coarse = use_coarse;
    auto* storage = static_cast<std::uint8_t*>(scratch) + used;
    auto& job = batch.jobs[count++];
    job.input = input;
    job.fine = runtime_fused_scratch_layout(storage, splits, global ? 512 : 256, global);
    job.coarse = use_coarse ? runtime_global_coarse_scratch_layout(storage, job.fine.bytes, splits)
                           : RuntimeGlobalCoarseScratchLayout{};
    job.splits = splits;
    job.end_splits = splits_total += splits;
    job.end_coarse = coarse_total += job.coarse.split_count;
    max_splits = std::max(max_splits, splits);
    used += bytes;
  }
  flush();
}

}  // namespace gewell::bf16_primitives
