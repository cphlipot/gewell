#pragma once

#include "gewell/kv_cache.h"
#include "gewell/runtime/state.h"

#include <functional>
#include <memory>
#include <string_view>

namespace gewell::runtime {

// CacheLedger remains the sole allocator. Storage interprets payloads and
// executes already-reserved copies; the manager publishes only after completion.
class CacheStorage {
 public:
  virtual ~CacheStorage() = default;
  virtual void clear_page_table(const kv_cache::ExecutionInfo& info) = 0;
  virtual void upload_page_table(const kv_cache::ExecutionInfo& info,
                                 CompletionContext completion) = 0;
  virtual void fork_local(const kv_cache::ExecutionInfo& from,
                          const kv_cache::ExecutionInfo& to,
                          CompletionContext completion) = 0;
  virtual void restore_local(const kv_cache::ExecutionInfo& destination,
                             const kv_cache::Allocation& snapshot,
                             std::uint32_t local_start, std::uint32_t local_tokens,
                             CompletionContext completion) = 0;
  virtual void copy_local_to_snapshot(const kv_cache::ExecutionInfo& source,
                                      std::uint32_t local_tokens,
                                      const kv_cache::Allocation& destination,
                                      CompletionContext completion) = 0;
  virtual void restore_terminal(const kv_cache::Allocation& source,
                                TerminalState destination,
                                CompletionContext completion) const = 0;
  virtual void snapshot_terminal(TerminalState source,
                                 const kv_cache::Allocation& destination,
                                 CompletionContext completion,
                                 std::string_view operation) = 0;
  virtual void copy_global_page(const kv_cache::Allocation& source,
                                const kv_cache::Allocation& destination,
                                CompletionContext completion) = 0;
  virtual void restore_global_page(const kv_cache::Allocation& source,
                                   const kv_cache::Allocation& destination,
                                   CompletionContext completion) = 0;
  virtual void spill(const kv_cache::Allocation& source,
                      const kv_cache::Allocation& destination, std::size_t bytes,
                      std::string_view operation) = 0;
  virtual void synchronize(CompletionContext completion,
                            std::string_view operation) const = 0;
  virtual void wait(CompletionContext completion) const noexcept = 0;
};

using CacheStorageFactory = std::function<std::unique_ptr<CacheStorage>(
    const kv_cache::PoolConfig&, kv_cache::CacheLedger&, std::size_t)>;

}  // namespace gewell::runtime
