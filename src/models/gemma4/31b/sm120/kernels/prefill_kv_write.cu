#include "prefill_common.cuh"

namespace gewell::prefill_primitives {
namespace {

using namespace detail;

template <unsigned D, unsigned Heads>
__global__ void write_fp8_separate_chunk(const BFloat16* key, const BFloat16* value,
    BFloat16* key_cache, BFloat16* value_cache, unsigned base, unsigned rows,
    unsigned capacity, bool local) {
  const unsigned head = blockIdx.x, token = blockIdx.y;
  if (local && token + capacity < rows) return;
  const unsigned slot = local ? (base + token) % capacity : base + token;
  const auto index = std::size_t(head) * capacity + slot;
  kv_storage::store_separate<D>(
      kv_storage::row(key_cache, index, D, kv_cache::Format::fp8),
      kv_storage::row(value_cache, index, D, kv_cache::Format::fp8),
      key + (std::size_t(head) * rows + token) * D,
      value + (std::size_t(token) * Heads + head) * D);
}

__global__ void write_fp8_compact_chunk(const BFloat16* key, const BFloat16* value,
    BFloat16* cache, const std::uint64_t* offsets, unsigned page_tokens,
    std::size_t layer_offset, unsigned base, unsigned rows, unsigned capacity) {
  const unsigned head = blockIdx.x, token = blockIdx.y, position = base + token;
  auto* record = offsets ? compact_global_cache::paged_row(cache, offsets,
      page_tokens, layer_offset, head, position, kv_cache::Format::fp8)
      : kv_storage::row(cache, std::size_t(head) * capacity + position,
                        640, kv_cache::Format::fp8, 2);
  kv_storage::store_compact(record, key + (std::size_t(head) * rows + token) * 512,
      value + (std::size_t(token) * 4 + head) * 512);
}


template <std::uint32_t HeadSize, std::uint32_t KvHeads>
__global__ void write_kv_cache_kernel(
    const BFloat16* key_head_major, const BFloat16* value_token_major,
    BFloat16* key_cache, BFloat16* value_cache, std::uint32_t capacity,
    std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }

  const std::uint32_t dimension =
      static_cast<std::uint32_t>(index % HeadSize);
  const std::size_t row = index / HeadSize;
  const std::uint32_t position =
      static_cast<std::uint32_t>(row % kTokenCount);
  const std::uint32_t kv_head =
      static_cast<std::uint32_t>(row / kTokenCount);
  if (kv_head >= KvHeads) {
    return;
  }

  const std::size_t cache_index =
      (static_cast<std::size_t>(kv_head) * capacity + position) * HeadSize +
      dimension;
  const std::size_t key_index =
      (static_cast<std::size_t>(kv_head) * kTokenCount + position) * HeadSize +
      dimension;
  const std::size_t value_index =
      (static_cast<std::size_t>(position) * KvHeads + kv_head) * HeadSize +
      dimension;
  key_cache[cache_index] = key_head_major[key_index];
  value_cache[cache_index] = value_token_major[value_index];
}

template <std::uint32_t HeadSize, std::uint32_t KvHeads, bool Local>
__global__ void write_kv_cache_chunk_kernel(
    const BFloat16* key_head_major, const BFloat16* value_token_major,
    BFloat16* key_cache, BFloat16* value_cache,
    std::uint32_t base_position, std::uint32_t token_count,
    std::uint32_t capacity, std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }

  const std::uint32_t dimension =
      static_cast<std::uint32_t>(index % HeadSize);
  const std::size_t row = index / HeadSize;
  const std::uint32_t position =
      static_cast<std::uint32_t>(row % token_count);
  const std::uint32_t kv_head =
      static_cast<std::uint32_t>(row / token_count);
  if constexpr (Local) {
    // A runtime chunk can be wider than the local ring. Only its final
    // window belongs in the cache; writing older rows as well would race on
    // the same wrapped slots.
    if (token_count > gemma4_31b::kLocalWindowSize &&
        position < token_count - gemma4_31b::kLocalWindowSize) {
      return;
    }
  }
  const std::uint32_t absolute_position = base_position + position;
  const std::uint32_t cache_position =
      Local ? absolute_position % gemma4_31b::kLocalWindowSize
            : absolute_position;
  const std::size_t cache_index =
      (static_cast<std::size_t>(kv_head) * capacity + cache_position) *
          HeadSize +
      dimension;
  const std::size_t key_index =
      (static_cast<std::size_t>(kv_head) * token_count + position) * HeadSize +
      dimension;
  const std::size_t value_index =
      (static_cast<std::size_t>(position) * KvHeads + kv_head) * HeadSize +
      dimension;
  key_cache[cache_index] = key_head_major[key_index];
  value_cache[cache_index] = value_token_major[value_index];
}

__global__ void write_kv_cache_chunk_global_compact_kernel(
    const BFloat16* key_head_major, const BFloat16* value_token_major,
    BFloat16* compact_kv_cache, std::uint32_t base_position,
    std::uint32_t token_count, std::uint32_t capacity,
    std::size_t elements) {
  constexpr std::uint32_t kHeadSize = gemma4_31b::kGlobalHeadSize;
  constexpr std::uint32_t kKvHeads = gemma4_31b::kGlobalKvHeadCount;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }

  const std::uint32_t compact_dimension =
      static_cast<std::uint32_t>(index % kGlobalCompactCacheRowElements);
  const std::size_t row = index / kGlobalCompactCacheRowElements;
  const std::uint32_t position =
      static_cast<std::uint32_t>(row % token_count);
  const std::uint32_t kv_head =
      static_cast<std::uint32_t>(row / token_count);
  const std::uint32_t cache_position = base_position + position;
  const std::size_t cache_index =
      (static_cast<std::size_t>(kv_head) * capacity + cache_position) *
          kGlobalCompactCacheRowElements +
      compact_dimension;
  if (compact_dimension < kGlobalCompactRotatedKeyElements) {
    const std::uint32_t source_dimension =
        compact_dimension < 64 ? compact_dimension
                               : 256 + (compact_dimension - 64);
    const std::size_t key_index =
        (static_cast<std::size_t>(kv_head) * token_count + position) *
            kHeadSize +
        source_dimension;
    compact_kv_cache[cache_index] = key_head_major[key_index];
    return;
  }

  const std::uint32_t value_dimension =
      compact_dimension - kGlobalCompactRotatedKeyElements;
  const std::size_t value_index =
      (static_cast<std::size_t>(position) * kKvHeads + kv_head) * kHeadSize +
      value_dimension;
  compact_kv_cache[cache_index] = value_token_major[value_index];
}

__global__ void write_kv_cache_chunk_global_compact_paged_kernel(
    const BFloat16* key_head_major, const BFloat16* value_token_major,
    BFloat16* page_pool, const std::uint64_t* page_offsets,
    std::uint32_t page_tokens, std::size_t layer_offset_elements,
    std::uint32_t base_position, std::uint32_t token_count,
    std::size_t elements) {
  constexpr std::uint32_t kHeadSize = gemma4_31b::kGlobalHeadSize;
  constexpr std::uint32_t kKvHeads = gemma4_31b::kGlobalKvHeadCount;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }

  const std::uint32_t compact_dimension =
      static_cast<std::uint32_t>(index % kGlobalCompactCacheRowElements);
  const std::size_t row = index / kGlobalCompactCacheRowElements;
  const std::uint32_t position =
      static_cast<std::uint32_t>(row % token_count);
  const std::uint32_t kv_head =
      static_cast<std::uint32_t>(row / token_count);
  const std::uint32_t absolute_position = base_position + position;
  BFloat16* const cache_row = compact_global_cache_row<true>(
      page_pool, page_offsets, page_tokens, layer_offset_elements, kv_head,
      absolute_position, 0);
  if (compact_dimension < kGlobalCompactRotatedKeyElements) {
    const std::uint32_t source_dimension =
        compact_dimension < 64 ? compact_dimension
                               : 256 + (compact_dimension - 64);
    const std::size_t key_index =
        (static_cast<std::size_t>(kv_head) * token_count + position) *
            kHeadSize +
        source_dimension;
    cache_row[compact_dimension] = key_head_major[key_index];
    return;
  }

  const std::uint32_t value_dimension =
      compact_dimension - kGlobalCompactRotatedKeyElements;
  const std::size_t value_index =
      (static_cast<std::size_t>(position) * kKvHeads + kv_head) * kHeadSize +
      value_dimension;
  cache_row[compact_dimension] = value_token_major[value_index];
}

}  // namespace

void write_kv_cache_m1024(
    const BFloat16* key_head_major, const BFloat16* value_token_major,
    BFloat16* key_cache, BFloat16* value_cache, std::uint32_t capacity,
    gemma4_31b::AttentionKind kind, cudaStream_t stream) {
  check_pointer(key_head_major, "write_kv_cache_m1024 key");
  check_pointer(value_token_major, "write_kv_cache_m1024 value");
  check_pointer(key_cache, "write_kv_cache_m1024 key cache");
  check_pointer(value_cache, "write_kv_cache_m1024 value cache");
  check_kind(kind, "write_kv_cache_m1024");

  const bool global = kind == gemma4_31b::AttentionKind::global;
  if ((!global && capacity != kTokenCount) ||
      (global && capacity < kGlobalCacheMinimumCapacity)) {
    fail("write_kv_cache_m1024", global
                                       ? "global cache capacity is below 1026"
                                       : "local cache capacity must equal 1024");
  }
  const std::uint32_t kv_heads = global ? gemma4_31b::kGlobalKvHeadCount
                                        : gemma4_31b::kLocalKvHeadCount;
  const std::uint32_t head_size = global ? gemma4_31b::kGlobalHeadSize
                                         : gemma4_31b::kLocalHeadSize;
  const std::size_t elements =
      static_cast<std::size_t>(kv_heads) * kTokenCount * head_size;
  if (global) {
    write_kv_cache_kernel<gemma4_31b::kGlobalHeadSize,
                          gemma4_31b::kGlobalKvHeadCount>
        <<<blocks_for(elements), kThreads, 0, stream>>>(
            key_head_major, value_token_major, key_cache, value_cache,
            capacity, elements);
  } else {
    write_kv_cache_kernel<gemma4_31b::kLocalHeadSize,
                          gemma4_31b::kLocalKvHeadCount>
        <<<blocks_for(elements), kThreads, 0, stream>>>(
            key_head_major, value_token_major, key_cache, value_cache,
            capacity, elements);
  }
  check_cuda(cudaGetLastError(), "write M=1024 KV cache kernel launch");
}

void write_kv_cache_chunk(
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major, BFloat16* key_cache,
    BFloat16* value_cache, std::uint32_t base_position,
    std::uint32_t token_count, std::uint32_t cache_capacity,
    gemma4_31b::AttentionKind kind, cudaStream_t stream,
    kv_cache::Format format) {
  check_pointer(current_key_head_major, "write_kv_cache_chunk current key");
  check_pointer(current_value_token_major,
                "write_kv_cache_chunk current value");
  check_pointer(key_cache, "write_kv_cache_chunk key cache");
  check_pointer(value_cache, "write_kv_cache_chunk value cache");
  check_kind(kind, "write_kv_cache_chunk");
  check_chunk_cache(base_position, token_count, cache_capacity, kind,
                    "write_kv_cache_chunk");

  const bool global = kind == gemma4_31b::AttentionKind::global;
  const std::uint32_t kv_heads = global ? gemma4_31b::kGlobalKvHeadCount
                                        : gemma4_31b::kLocalKvHeadCount;
  const std::uint32_t head_size = global ? gemma4_31b::kGlobalHeadSize
                                         : gemma4_31b::kLocalHeadSize;
  const std::size_t elements =
      static_cast<std::size_t>(kv_heads) * token_count * head_size;
  if (format == kv_cache::Format::fp8) {
    if (global) write_fp8_separate_chunk<512, 4><<<dim3(4, token_count), 256, 0, stream>>>(
        current_key_head_major, current_value_token_major, key_cache, value_cache,
        base_position, token_count, cache_capacity, false);
    else write_fp8_separate_chunk<256, 16><<<dim3(16, token_count), 256, 0, stream>>>(
        current_key_head_major, current_value_token_major, key_cache, value_cache,
        base_position, token_count, cache_capacity, true);
    check_cuda(cudaGetLastError(), "write FP8 separate KV chunk");
    return;
  }
  if (global) {
    write_kv_cache_chunk_kernel<gemma4_31b::kGlobalHeadSize,
                                gemma4_31b::kGlobalKvHeadCount, false>
        <<<blocks_for(elements), kThreads, 0, stream>>>(
            current_key_head_major, current_value_token_major, key_cache,
            value_cache, base_position, token_count, cache_capacity, elements);
  } else {
    write_kv_cache_chunk_kernel<gemma4_31b::kLocalHeadSize,
                                gemma4_31b::kLocalKvHeadCount, true>
        <<<blocks_for(elements), kThreads, 0, stream>>>(
            current_key_head_major, current_value_token_major, key_cache,
            value_cache, base_position, token_count, cache_capacity, elements);
  }
  check_cuda(cudaGetLastError(), "write chunk KV cache kernel launch");
}

void write_kv_cache_chunk_global_compact(
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major, BFloat16* compact_kv_cache,
    std::uint32_t base_position, std::uint32_t token_count,
    std::uint32_t cache_capacity, cudaStream_t stream,
    kv_cache::Format format) {
  check_pointer(current_key_head_major,
                "write compact-global cache current key");
  check_pointer(current_value_token_major,
                "write compact-global cache current value");
  check_pointer(compact_kv_cache, "write compact-global cache");
  check_chunk_cache(base_position, token_count, cache_capacity,
                    gemma4_31b::AttentionKind::global,
                    "write_kv_cache_chunk_global_compact");

  const std::size_t elements =
      static_cast<std::size_t>(gemma4_31b::kGlobalKvHeadCount) *
      token_count * kGlobalCompactCacheRowElements;
  if (format == kv_cache::Format::fp8) {
    write_fp8_compact_chunk<<<dim3(4, token_count), 256, 0, stream>>>(
        current_key_head_major, current_value_token_major, compact_kv_cache,
        nullptr, 0, 0, base_position, token_count, cache_capacity);
    check_cuda(cudaGetLastError(), "write FP8 compact KV chunk");
    return;
  }
  write_kv_cache_chunk_global_compact_kernel
      <<<blocks_for(elements), kThreads, 0, stream>>>(
          current_key_head_major, current_value_token_major,
          compact_kv_cache, base_position, token_count, cache_capacity,
          elements);
  check_cuda(cudaGetLastError(),
             "write compact-global chunk KV cache kernel launch");
}

void write_kv_cache_chunk_global_compact_paged(
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major,
    const CompactGlobalPagedCache& cache, std::uint32_t base_position,
    std::uint32_t token_count, cudaStream_t stream) {
  check_pointer(current_key_head_major,
                "write paged compact-global current key");
  check_pointer(current_value_token_major,
                "write paged compact-global current value");
  check_paged_compact_global_cache(
      cache, base_position, token_count,
      "write_kv_cache_chunk_global_compact_paged");

  const std::size_t elements =
      static_cast<std::size_t>(gemma4_31b::kGlobalKvHeadCount) * token_count *
      kGlobalCompactCacheRowElements;
  if (cache.format == kv_cache::Format::fp8) {
    write_fp8_compact_chunk<<<dim3(4, token_count), 256, 0, stream>>>(
        current_key_head_major, current_value_token_major, cache.page_pool,
        cache.page_offsets, cache.page_tokens, cache.layer_offset_elements,
        base_position, token_count, 0);
    check_cuda(cudaGetLastError(), "write FP8 paged compact KV chunk");
    return;
  }
  write_kv_cache_chunk_global_compact_paged_kernel
      <<<blocks_for(elements), kThreads, 0, stream>>>(
          current_key_head_major, current_value_token_major, cache.page_pool,
          cache.page_offsets, cache.page_tokens, cache.layer_offset_elements,
          base_position, token_count, elements);
  check_cuda(cudaGetLastError(),
             "write paged compact-global chunk KV cache kernel launch");
}

}  // namespace gewell::prefill_primitives
