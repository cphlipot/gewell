#pragma once

#include "gewell/kv_cache.h"

namespace gewell::gemma4_31b::sm120 {

inline constexpr std::uint32_t kGlobalPageTokens = 256;
inline constexpr std::uint32_t kLocalWindowTokens = 1'024;
inline constexpr std::uint32_t kMaximumContextTokens = 262'144;

// The ledger accounts for these explicit physical sizes without model knowledge.
kv_cache::PoolConfig compact_pool_config(
    std::size_t gpu_bytes, std::size_t cpu_bytes = 0,
    std::size_t index_bytes = kv_cache::kDefaultIndexBytes,
    kv_cache::Format local = kv_cache::Format::bf16,
    kv_cache::Format global = kv_cache::Format::bf16);

}  // namespace gewell::gemma4_31b::sm120
