#include "cache_config.h"

#include "gewell/compact_global_cache.h"
#include "gewell/models/gemma4/31b/model.h"

namespace gewell::gemma4_31b::sm120 {

kv_cache::PoolConfig compact_pool_config(std::size_t gpu_bytes,
                                    std::size_t cpu_bytes,
                                    std::size_t index_bytes,
                                    kv_cache::Format local, kv_cache::Format global) {
  // Every sixth Gemma layer is global. A compact global row stores the 128
  // position-dependent K elements followed by the complete V vector. Local
  // rows retain both K and V.
  static_assert(kLocalWindowTokens == gemma4_31b::kLocalWindowSize);

  kv_cache::PoolConfig config;
  config.local_format = local;
  config.global_format = global;
  config.global_page_tokens = kGlobalPageTokens;
  config.local_window_tokens = kLocalWindowTokens;
  config.maximum_context_tokens = kMaximumContextTokens;
  config.gpu_bytes = gpu_bytes;
  config.cpu_bytes = cpu_bytes;
  config.index_bytes = index_bytes;
  config.global_page_bytes =
      gemma4_31b::kGlobalLayerCount * gemma4_31b::kGlobalKvHeadCount *
      kGlobalPageTokens * kv_cache::row_bytes(compact_global_cache::kRowElements, global, 2);
  config.local_ring_bytes =
      gemma4_31b::kLocalLayerCount * gemma4_31b::kLocalKvHeadCount *
      kLocalWindowTokens * 2 * kv_cache::row_bytes(gemma4_31b::kLocalHeadSize, local);
  config.local_bytes_per_token =
      gemma4_31b::kLocalLayerCount * gemma4_31b::kLocalKvHeadCount *
      2 * kv_cache::row_bytes(gemma4_31b::kLocalHeadSize, local);
  config.terminal_hidden_bytes =
      gemma4_31b::kHiddenSize * gemma4_31b::kBf16Bytes;
  config.page_table_bytes =
      ((static_cast<std::size_t>(kMaximumContextTokens) +
        kGlobalPageTokens - 1) /
       kGlobalPageTokens) *
      sizeof(std::uint64_t);
  config.validate();
  return config;
}

}  // namespace gewell::gemma4_31b::sm120
