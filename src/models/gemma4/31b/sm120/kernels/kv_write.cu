#include "bf16_common.cuh"
#include "gewell/prefill_primitives.h"
#include <algorithm>
#include <cmath>

namespace gewell::bf16_primitives {
namespace {

using namespace detail;

__global__ void expand_value_heads_kernel(const BFloat16* compact,
                                          BFloat16* expanded,
                                          std::uint32_t kv_heads,
                                          std::uint32_t repeats,
                                          std::uint32_t head_size,
                                          std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const std::uint32_t output_head =
        static_cast<std::uint32_t>(index / head_size);
    const std::uint32_t element =
        static_cast<std::uint32_t>(index % head_size);
    const std::uint32_t input_head = output_head / repeats;
    if (input_head < kv_heads) {
      expanded[index] = compact[input_head * head_size + element];
    }
  }
}

template <std::uint32_t HeadSize, std::uint32_t KvHeads>
__global__ void write_kv_cache_m1_kernel(
    const BFloat16* key, const BFloat16* value, BFloat16* key_cache,
    BFloat16* value_cache, std::uint32_t slot, std::uint32_t capacity,
    std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }

  const std::uint32_t dimension =
      static_cast<std::uint32_t>(index % HeadSize);
  const std::uint32_t head =
      static_cast<std::uint32_t>(index / HeadSize);
  if (head < KvHeads) {
    const std::size_t cache_index =
        (static_cast<std::size_t>(head) * capacity + slot) * HeadSize +
        dimension;
    key_cache[cache_index] = key[index];
    value_cache[cache_index] = value[index];
  }
}

__global__ void write_kv_cache_m1_global_compact_kernel(
    const BFloat16* key, const BFloat16* value,
    BFloat16* compact_kv_cache, std::uint32_t slot,
    std::uint32_t capacity, std::size_t elements) {
  static_assert(kGlobalCompactKeySize == 128);
  static_assert(kGlobalCompactKvSize == 640);
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }

  const std::uint32_t compact_dimension =
      static_cast<std::uint32_t>(index % kGlobalCompactKvSize);
  const std::uint32_t head =
      static_cast<std::uint32_t>(index / kGlobalCompactKvSize);
  if (head < gemma4_31b::kGlobalKvHeadCount) {
    const std::size_t cache_index =
        (static_cast<std::size_t>(head) * capacity + slot) *
            kGlobalCompactKvSize +
        compact_dimension;
    const std::size_t source_row =
        static_cast<std::size_t>(head) * gemma4_31b::kGlobalHeadSize;
    if (compact_dimension < 64) {
      compact_kv_cache[cache_index] = key[source_row + compact_dimension];
    } else if (compact_dimension < kGlobalCompactKeySize) {
      compact_kv_cache[cache_index] =
          key[source_row + 256 + compact_dimension - 64];
    } else {
      compact_kv_cache[cache_index] =
          value[source_row + compact_dimension - kGlobalCompactKeySize];
    }
  }
}

template <std::uint32_t HeadSize, std::uint32_t KvHeads,
          std::uint32_t Capacity>
__global__ void write_kv_cache_m1_device_position_kernel(
    const BFloat16* key, const BFloat16* value, BFloat16* key_cache,
    BFloat16* value_cache, const std::uint32_t* absolute_position,
    std::size_t elements) {
  static_assert(Capacity == kGraphAttentionLocalCapacity ||
                Capacity == kGraphAttentionGlobalCapacity);
  const std::uint32_t position = absolute_position[0];
  if (position < kGraphDecodeFirstPosition ||
      position > kGraphDecodeLastPosition) {
    return;
  }
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }

  const std::uint32_t dimension =
      static_cast<std::uint32_t>(index % HeadSize);
  const std::uint32_t head =
      static_cast<std::uint32_t>(index / HeadSize);
  if (head < KvHeads) {
    const std::uint32_t slot = position % Capacity;
    const std::size_t cache_index =
        (static_cast<std::size_t>(head) * Capacity + slot) * HeadSize +
        dimension;
    key_cache[cache_index] = key[index];
    value_cache[cache_index] = value[index];
  }
}

}  // namespace

void expand_value_heads(const BFloat16* compact, BFloat16* expanded,
                        gemma4_31b::AttentionKind kind,
                        cudaStream_t stream) {
  check_pointer(compact, "expand_value_heads compact");
  check_pointer(expanded, "expand_value_heads expanded");
  if (kind != gemma4_31b::AttentionKind::local &&
      kind != gemma4_31b::AttentionKind::global) {
    fail("expand_value_heads", "invalid Gemma 4 attention kind");
  }
  const bool global = kind == gemma4_31b::AttentionKind::global;
  const std::uint32_t kv_heads = global ? gemma4_31b::kGlobalKvHeadCount
                                        : gemma4_31b::kLocalKvHeadCount;
  const std::uint32_t head_size = global ? gemma4_31b::kGlobalHeadSize
                                         : gemma4_31b::kLocalHeadSize;
  const std::uint32_t repeats = gemma4_31b::kQueryHeadCount / kv_heads;
  const std::size_t elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * head_size;
  expand_value_heads_kernel<<<blocks_for(elements), kThreads, 0, stream>>>(
      compact, expanded, kv_heads, repeats, head_size, elements);
  check_cuda(cudaGetLastError(), "expand_value_heads kernel launch");
}

void write_kv_cache_m1(const BFloat16* key, const BFloat16* value,
                       BFloat16* key_cache, BFloat16* value_cache,
                       std::uint32_t absolute_position,
                       std::uint32_t capacity,
                       gemma4_31b::AttentionKind kind,
                       cudaStream_t stream,
    kv_cache::Format format) {
  if (format == kv_cache::Format::fp8) {
    prefill_primitives::write_kv_cache_chunk(key, value, key_cache, value_cache, absolute_position, 1, capacity, kind, stream, format);
    return;
  }

  check_pointer(key, "write_kv_cache_m1 key");
  check_pointer(value, "write_kv_cache_m1 value");
  check_pointer(key_cache, "write_kv_cache_m1 key cache");
  check_pointer(value_cache, "write_kv_cache_m1 value cache");
  if (kind != gemma4_31b::AttentionKind::local &&
      kind != gemma4_31b::AttentionKind::global) {
    fail("write_kv_cache_m1", "invalid Gemma 4 attention kind");
  }
  if (capacity == 0) {
    fail("write_kv_cache_m1", "cache capacity must be positive");
  }
  const bool global = kind == gemma4_31b::AttentionKind::global;
  const std::uint32_t kv_heads = global ? gemma4_31b::kGlobalKvHeadCount
                                        : gemma4_31b::kLocalKvHeadCount;
  const std::uint32_t head_size = global ? gemma4_31b::kGlobalHeadSize
                                         : gemma4_31b::kLocalHeadSize;
  const std::size_t elements =
      static_cast<std::size_t>(kv_heads) * head_size;
  const std::uint32_t slot = absolute_position % capacity;
  if (global) {
    write_kv_cache_m1_kernel<gemma4_31b::kGlobalHeadSize,
                             gemma4_31b::kGlobalKvHeadCount>
        <<<blocks_for(elements), kThreads, 0, stream>>>(
            key, value, key_cache, value_cache, slot, capacity, elements);
  } else {
    write_kv_cache_m1_kernel<gemma4_31b::kLocalHeadSize,
                             gemma4_31b::kLocalKvHeadCount>
        <<<blocks_for(elements), kThreads, 0, stream>>>(
            key, value, key_cache, value_cache, slot, capacity, elements);
  }
  check_cuda(cudaGetLastError(), "write_kv_cache_m1 kernel launch");
}

void write_kv_cache_m1_global_compact(
    const BFloat16* key, const BFloat16* value,
    BFloat16* compact_kv_cache, std::uint32_t absolute_position,
    std::uint32_t capacity, cudaStream_t stream,
    kv_cache::Format format) {
  if (format == kv_cache::Format::fp8) {
    prefill_primitives::write_kv_cache_chunk_global_compact(key, value, compact_kv_cache, absolute_position, 1, capacity, stream, format);
    return;
  }

  check_pointer(key, "write_kv_cache_m1_global_compact key");
  check_pointer(value, "write_kv_cache_m1_global_compact value");
  check_pointer(compact_kv_cache,
                "write_kv_cache_m1_global_compact cache");
  if (capacity == 0) {
    fail("write_kv_cache_m1_global_compact",
         "cache capacity must be positive");
  }
  constexpr std::size_t kElements =
      static_cast<std::size_t>(gemma4_31b::kGlobalKvHeadCount) *
      kGlobalCompactKvSize;
  const std::uint32_t slot = absolute_position % capacity;
  write_kv_cache_m1_global_compact_kernel
      <<<blocks_for(kElements), kThreads, 0, stream>>>(
          key, value, compact_kv_cache, slot, capacity, kElements);
  check_cuda(cudaGetLastError(),
             "write_kv_cache_m1_global_compact kernel launch");
}

void write_kv_cache_m1_device_position(
    const BFloat16* key, const BFloat16* value, BFloat16* key_cache,
    BFloat16* value_cache, const std::uint32_t* absolute_position,
    gemma4_31b::AttentionKind kind, cudaStream_t stream) {
  check_pointer(key, "write_kv_cache_m1_device_position key");
  check_pointer(value, "write_kv_cache_m1_device_position value");
  check_pointer(key_cache,
                "write_kv_cache_m1_device_position key cache");
  check_pointer(value_cache,
                "write_kv_cache_m1_device_position value cache");
  check_pointer(absolute_position,
                "write_kv_cache_m1_device_position position");
  if (kind != gemma4_31b::AttentionKind::local &&
      kind != gemma4_31b::AttentionKind::global) {
    fail("write_kv_cache_m1_device_position",
         "invalid Gemma 4 attention kind");
  }
  const bool global = kind == gemma4_31b::AttentionKind::global;
  const std::uint32_t kv_heads = global ? gemma4_31b::kGlobalKvHeadCount
                                        : gemma4_31b::kLocalKvHeadCount;
  const std::uint32_t head_size = global ? gemma4_31b::kGlobalHeadSize
                                         : gemma4_31b::kLocalHeadSize;
  const std::size_t elements =
      static_cast<std::size_t>(kv_heads) * head_size;
  if (global) {
    write_kv_cache_m1_device_position_kernel<
        gemma4_31b::kGlobalHeadSize, gemma4_31b::kGlobalKvHeadCount,
        kGraphAttentionGlobalCapacity>
        <<<blocks_for(elements), kThreads, 0, stream>>>(
            key, value, key_cache, value_cache, absolute_position, elements);
  } else {
    write_kv_cache_m1_device_position_kernel<
        gemma4_31b::kLocalHeadSize, gemma4_31b::kLocalKvHeadCount,
        kGraphAttentionLocalCapacity>
        <<<blocks_for(elements), kThreads, 0, stream>>>(
            key, value, key_cache, value_cache, absolute_position, elements);
  }
  check_cuda(cudaGetLastError(),
             "write_kv_cache_m1_device_position kernel launch");
}

}  // namespace gewell::bf16_primitives
