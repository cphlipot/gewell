#pragma once

#include "gewell/runtime/cache_storage.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace gewell::runtime::test {

inline kv_cache::PoolConfig small_cache_config(std::size_t gpu_bytes = 4096,
                                              std::size_t cpu_bytes = 0) {
  kv_cache::PoolConfig config;
  config.gpu_bytes = gpu_bytes;
  config.cpu_bytes = cpu_bytes;
  config.index_bytes = 64 * 1024;
  config.global_page_tokens = 4;
  config.local_window_tokens = 8;
  config.maximum_context_tokens = 64;
  config.global_page_bytes = 256;
  config.local_ring_bytes = 256;
  config.local_bytes_per_token = 32;
  config.terminal_hidden_bytes = 16;
  config.page_table_bytes = 16 * sizeof(std::uint64_t);
  config.validate();
  return config;
}

// A byte storage implementation for exercising the real policy and allocator.
// It records physical boundaries and permits deterministic transfer failures.
class ByteStorage final : public CacheStorage {
 public:
  ByteStorage(const kv_cache::PoolConfig& config, kv_cache::CacheLedger& ledger)
      : config(config), ledger(ledger), gpu(config.gpu_bytes), cpu(config.cpu_bytes) {}

  std::uint8_t* pointer(const kv_cache::Allocation& allocation) const {
    auto& pool = allocation.tier == kv_cache::Tier::gpu ? gpu : cpu;
    const auto& allocator = allocation.tier == kv_cache::Tier::gpu
        ? ledger.gpu_pool() : ledger.cpu_pool();
    if (!allocator.owns(allocation) || allocation.offset > pool.size() ||
        allocation.bytes > pool.size() - allocation.offset)
      throw std::runtime_error("test storage: invalid allocation");
    return pool.data() + allocation.offset;
  }
  void boundary(const char* event) const {
    events.emplace_back(event);
    if (observe) observe(event);
    if (fail_at == event) throw std::runtime_error(std::string("injected ") + event);
  }
  void clear_page_table(const kv_cache::ExecutionInfo& info) override {
    boundary("clear");
    std::memset(pointer(info.page_table), 0, info.page_table.bytes);
  }
  void upload_page_table(const kv_cache::ExecutionInfo&, CompletionContext) override {
    boundary("upload");
  }
  void fork_local(const kv_cache::ExecutionInfo& from,
                  const kv_cache::ExecutionInfo& to, CompletionContext) override {
    boundary("fork");
    std::memcpy(pointer(to.local_ring), pointer(from.local_ring), config.local_ring_bytes);
  }
  void restore_local(const kv_cache::ExecutionInfo& destination,
                     const kv_cache::Allocation& snapshot, std::uint32_t,
                     std::uint32_t local_tokens, CompletionContext) override {
    boundary("restore_local");
    if (local_tokens)
      std::memcpy(pointer(destination.local_ring), pointer(snapshot),
                  local_tokens * config.local_bytes_per_token);
  }
  void copy_local_to_snapshot(const kv_cache::ExecutionInfo& source,
                              std::uint32_t local_tokens,
                              const kv_cache::Allocation& destination,
                              CompletionContext) override {
    boundary("snapshot_local");
    if (local_tokens)
      std::memcpy(pointer(destination), pointer(source.local_ring),
                  local_tokens * config.local_bytes_per_token);
  }
  void restore_terminal(const kv_cache::Allocation& source, TerminalState destination,
                        CompletionContext) const override {
    boundary("restore_terminal");
    std::memcpy(destination.value, pointer(source), config.terminal_hidden_bytes);
  }
  void snapshot_terminal(TerminalState source, const kv_cache::Allocation& destination,
                         CompletionContext, std::string_view) override {
    boundary("snapshot_terminal");
    std::memcpy(pointer(destination), source.value, config.terminal_hidden_bytes);
  }
  void copy_global_page(const kv_cache::Allocation& source,
                        const kv_cache::Allocation& destination,
                        CompletionContext) override {
    boundary("cow");
    std::memcpy(pointer(destination), pointer(source), config.global_page_bytes);
  }
  void restore_global_page(const kv_cache::Allocation& source,
                           const kv_cache::Allocation& destination,
                           CompletionContext) override {
    boundary("restore_page");
    std::memcpy(pointer(destination), pointer(source), config.global_page_bytes);
  }
  void spill(const kv_cache::Allocation& source, const kv_cache::Allocation& destination,
             std::size_t bytes, std::string_view) override {
    boundary("spill");
    std::memcpy(pointer(destination), pointer(source), bytes);
  }
  void synchronize(CompletionContext, std::string_view) const override {
    boundary("synchronize");
  }
  void wait(CompletionContext) const noexcept override { events.emplace_back("wait"); }

  const kv_cache::PoolConfig config;
  kv_cache::CacheLedger& ledger;
  mutable std::vector<std::uint8_t> gpu, cpu;
  mutable std::vector<std::string> events;
  std::string fail_at;
  std::function<void(const std::string&)> observe;
};

inline CacheStorageFactory byte_storage_factory(ByteStorage** observed = nullptr) {
  return [observed](const kv_cache::PoolConfig& config, kv_cache::CacheLedger& ledger,
                    std::size_t) {
    auto result = std::make_unique<ByteStorage>(config, ledger);
    if (observed) *observed = result.get();
    return result;
  };
}

}  // namespace gewell::runtime::test
