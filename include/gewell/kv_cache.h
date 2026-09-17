#pragma once

#include "gewell/kv_format.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace gewell::kv_cache {

inline constexpr std::size_t kMib = 1U << 20;

enum class Tier : std::uint8_t { gpu, cpu };

struct PoolConfig {
  Format local_format{Format::bf16}, global_format{Format::bf16};
  std::size_t gpu_bytes{};
  std::size_t cpu_bytes{};
  std::size_t index_bytes{};
  std::uint32_t global_page_tokens{};
  std::uint32_t local_window_tokens{};
  std::uint32_t maximum_context_tokens{};

  // Physical sizes come from the backend. The ledger accounts for shared
  // pages, private local rings and terminal state without interpreting their
  // payloads; host tests can use an explicit small representation.
  std::size_t global_page_bytes{};
  std::size_t local_ring_bytes{};
  std::size_t local_bytes_per_token{};
  std::size_t terminal_hidden_bytes{};
  std::size_t page_table_bytes{};

  void validate() const;
};

struct Allocation {
  std::uint64_t id{};
  Tier tier{Tier::gpu};
  std::size_t offset{};
  std::size_t bytes{};

  [[nodiscard]] bool valid() const { return id != 0; }
};

struct PoolStats {
  std::size_t capacity{};
  std::size_t used{};
  std::size_t peak_used{};
  std::size_t free{};
};

// A fixed-range allocator used by the cache manager. It never grows and does
// not fall back to an unrelated allocation API. The implementation is small
// enough to use directly in manager tests as the fake byte pool.
class BytePool {
 public:
  BytePool(std::size_t capacity, Tier tier);

  Allocation allocate(std::size_t bytes, std::size_t alignment = 256);
  // Non-throwing variant: returns false when no free range fits instead of
  // failing, so callers can apply cache policy before giving up.
  bool try_allocate(std::size_t bytes, Allocation* out,
                    std::size_t alignment = 256);
  // Checks whether this exact allocation order fits without changing pool
  // state. Callers use it to reserve a serial execution plan across fragmented
  // free ranges before beginning work.
  [[nodiscard]] bool can_allocate_sequence(
      const std::vector<std::size_t>& bytes,
      std::size_t alignment = 256) const;
  void release(const Allocation& allocation);

  [[nodiscard]] PoolStats stats() const;
  [[nodiscard]] bool owns(const Allocation& allocation) const;

 private:
  struct FreeRange {
    std::size_t offset{};
    std::size_t bytes{};
  };

  struct LiveAllocation {
    Allocation allocation;
  };

  std::size_t capacity_{};
  Tier tier_{Tier::gpu};
  std::size_t used_{};
  std::size_t peak_used_{};
  std::uint64_t next_id_{1};
  std::vector<FreeRange> free_ranges_;
  std::vector<LiveAllocation> live_;
};

using PageId = std::uint64_t;
using CheckpointId = std::uint64_t;
using ExecutionId = std::uint64_t;

struct PageInfo {
  PageId id{};
  Allocation storage;
  std::uint32_t first_token{};
  std::uint32_t valid_tokens{};
  std::size_t references{};
};

struct CheckpointInfo {
  CheckpointId id{};
  std::uint32_t processed_tokens{};
  std::uint32_t local_start{};
  std::uint32_t local_tokens{};
  // Exact token histories live in the bounded shared prefix index rather than
  // being duplicated in every physical checkpoint record.
  std::vector<PageId> global_pages;
  Allocation local_snapshot;
  Allocation terminal_hidden;
  std::size_t execution_references{};
  // Queued handoffs can retain an exact checkpoint before admitting a borrower.
  std::size_t dependency_pins{};
};

struct CheckpointCapture {
  Allocation local_snapshot;
  Allocation terminal_hidden;
};

struct ExecutionInfo {
  ExecutionId id{};
  std::uint32_t processed_tokens{};
  // Nonzero for batch executions, including retained-prefix continuations.
  // Their additional private growth through this boundary is reserved.
  std::uint32_t batch_max_processed_tokens{};
  std::vector<PageId> global_pages;
  Allocation local_ring;
  Allocation page_table;
  CheckpointId source_checkpoint{};
  // Ledger-owned allocations and page references until publication or abort.
  std::optional<CheckpointCapture> pending_checkpoint;
};

struct CopyOnWrite {
  PageId source{};
  PageId destination{};
};

struct WritePlan {
  std::uint32_t first_token{};
  std::uint32_t token_count{};
  std::vector<CopyOnWrite> copies;
  std::vector<PageId> new_pages;
};

struct CacheStats {
  PoolStats gpu;
  PoolStats cpu;
  std::size_t index_bytes{};
  std::size_t index_used{};
  std::size_t page_count{};
  std::size_t checkpoint_count{};
  std::size_t execution_count{};
  std::size_t copy_on_write_pages{};
};

// Disjoint physical allocations sum to pool usage. Shared pages and slack are
// subsets of global_page_bytes; reservations are future allocations, not usage.
struct PoolMemoryStats {
  std::size_t global_page_bytes{};
  std::size_t checkpoint_state_bytes{};
  std::size_t execution_buffer_bytes{};
  std::size_t pending_capture_bytes{};
  std::size_t other_bytes{};
  std::size_t shared_page_bytes{};
  std::size_t reclaimable_bytes{};
  std::size_t nonreclaimable_bytes{};
  std::size_t reserved_growth_bytes{};
  std::size_t page_slack_bytes{};
};

// Host-side lifecycle ledger for persistent compact-global state. CUDA code
// owns the bytes at Allocation::offset; this class owns all references and
// fixed-budget decisions. As a result it can be exhaustively tested without a
// model artifact or a CUDA device.
enum class EvictionCause : std::uint8_t {
  gpu_pressure,
  index_pressure,
};

class CacheLedger {
 public:
  using EvictionSelector = std::function<CheckpointId(
      CheckpointId protected_checkpoint,
      const std::vector<CheckpointId>& eligible_checkpoints,
      EvictionCause cause)>;
  // Runs only for pressure eviction, before the ledger drops the physical
  // checkpoint. Returning true means the caller retained an external backing
  // copy (for example in a cold tier); the ledger still reclaims its own GPU
  // state and reports that fact to CheckpointRemoved.
  using PressureEviction = std::function<bool(CheckpointId)>;
  using CheckpointRemoved = std::function<void(
      CheckpointId checkpoint, bool externally_preserved)>;

  explicit CacheLedger(PoolConfig config);

  [[nodiscard]] const PoolConfig& config() const { return config_; }

  ExecutionId begin_execution(CheckpointId source = 0);
  // Admits a GPU execution from an existing checkpoint, or empty state when
  // source is zero. The limit is absolute and cannot precede the checkpoint.
  // Shares inherited pages and protects the source while evicting idle state.
  // A source plus request exceeding fixed byte/index budgets throws; pool/index
  // pressure or fragmentation returns nullopt. Allocates private rings/tables
  // now and reserves future pages and tail COW collectively, including fragmentation.
  // The CUDA caller restores local state and releases after the last GPU use.
  [[nodiscard]] std::optional<ExecutionId> try_begin_batch(
      CheckpointId source, std::size_t maximum_processed_tokens);
  // Forks an execution's completed state, sharing global pages and reserving
  // both executions' private growth/COW. The caller copies valid local-ring
  // state and uploads the fork's page table before using it on the GPU. After
  // those copies complete, the fork does not depend on the source's lifetime.
  // Byte/index pressure or fragmentation returns nullopt, after idle eviction.
  [[nodiscard]] std::optional<ExecutionId> try_fork_batch(
      ExecutionId source_execution, std::size_t maximum_processed_tokens);
  // Atomically adjusts the reservation without changing state or allocations.
  // Shrinking to at least processed_tokens always fits; expansion returns false
  // when current free space cannot preserve all other admitted reservations.
  // Invalid ranges throw. On a capacity miss, `failure_cause` identifies the
  // budget that must be relieved. This operation does not evict checkpoints.
  [[nodiscard]] bool try_resize_batch(
      ExecutionId execution, std::size_t maximum_processed_tokens,
      EvictionCause* failure_cause = nullptr);
  // Whether this source/request can fit in isolation. Temporary pressure and
  // live allocations are handled separately by try_begin_batch.
  [[nodiscard]] bool batch_request_fits(
      CheckpointId source, std::size_t maximum_processed_tokens) const;
  WritePlan prepare_write(ExecutionId execution, std::uint32_t first_token,
                          std::uint32_t token_count);
  CheckpointId publish_checkpoint(ExecutionId execution,
                                  const std::vector<std::uint8_t>&
                                      local_snapshot_bytes,
                                  const std::vector<std::uint8_t>&
                                      terminal_hidden_bytes);
  CheckpointId publish_checkpoint_allocations(
      ExecutionId execution, Allocation local_snapshot,
      Allocation terminal_hidden);
  // Optional capture never evicts state or spends admitted batch growth. It
  // reserves checkpoint metadata and any newly shared partial-tail COW before
  // returning destinations for CUDA copies. Other executions may keep growing;
  // this execution cannot write until publish_checkpoint_allocations or abort.
  // Returned allocations remain ledger-owned; do not release them directly.
  [[nodiscard]] std::optional<CheckpointCapture> try_begin_checkpoint_capture(
      ExecutionId execution, EvictionCause* failure_cause = nullptr);
  // The caller must finish outstanding GPU copies before aborting or releasing
  // the execution. Releasing an execution also aborts its pending capture.
  void abort_checkpoint_capture(ExecutionId execution);
  // Allocates from the GPU pool without disturbing retained state. Returns an
  // invalid allocation when the bytes do not currently fit.
  [[nodiscard]] Allocation try_allocate_gpu(std::size_t bytes,
                                            std::size_t alignment = 256);
  [[nodiscard]] Allocation try_allocate_cpu(std::size_t bytes,
                                            std::size_t alignment = 256);
  // Allocates from the GPU pool, evicting idle checkpoints (oldest first,
  // never a checkpoint borrowed by an execution, dependency-pinned, or `protected`)
  // while the bytes do not fit. Fails when nothing evictable remains.
  Allocation allocate_gpu_releasing_idle(std::size_t bytes,
                                         std::size_t alignment,
                                         CheckpointId protected_checkpoint);
  // Evicts a policy-selected checkpoint without active borrowers or pins. Returns
  // false when no checkpoint is evictable.
  bool evict_idle_checkpoint(
      CheckpointId protected_checkpoint = 0,
      EvictionCause cause = EvictionCause::gpu_pressure);
  // The automatic prefix index supplies a policy selector. It is invoked only
  // for idle checkpoints; active borrowers, pins, and `protected_checkpoint` are
  // omitted. Before pressure physically reclaims a selected checkpoint, the
  // optional preservation callback can retain it outside this ledger. The
  // removal notification then receives whether that happened, so an external
  // prefix index can retain or remove its record accordingly. Explicit
  // release never invokes the preservation callback.
  void set_eviction_callbacks(EvictionSelector selector,
                              PressureEviction pressure_eviction,
                              CheckpointRemoved removed);
  void release_gpu(const Allocation& allocation);
  void release_cpu(const Allocation& allocation);
  void release_execution(ExecutionId execution);
  // A pin excludes both pressure eviction and explicit checkpoint release.
  // Dropping the final pin restores eligibility without automatically releasing.
  void pin_checkpoint(CheckpointId checkpoint);
  void unpin_checkpoint(CheckpointId checkpoint);
  void release_checkpoint(CheckpointId checkpoint);

  [[nodiscard]] const PageInfo& page(PageId page) const;
  [[nodiscard]] PageInfo& page(PageId page);
  [[nodiscard]] const CheckpointInfo& checkpoint(CheckpointId checkpoint) const;
  [[nodiscard]] CheckpointInfo& checkpoint(CheckpointId checkpoint);
  [[nodiscard]] const ExecutionInfo& execution(ExecutionId execution) const;
  [[nodiscard]] ExecutionInfo& execution(ExecutionId execution);
  [[nodiscard]] const BytePool& gpu_pool() const { return gpu_pool_; }
  [[nodiscard]] const BytePool& cpu_pool() const { return cpu_pool_; }
  [[nodiscard]] CacheStats stats() const;
  // Captured by the cache owner between mutations; these visits neither copy
  // state nor retain references after the visitor returns. Visitors must not
  // mutate the ledger. Network threads must consume a copied snapshot instead.
  void visit_pages(const std::function<void(const PageInfo&)>& visitor) const;
  void visit_checkpoints(
      const std::function<void(const CheckpointInfo&)>& visitor) const;
  void visit_executions(
      const std::function<void(const ExecutionInfo&)>& visitor) const;
  // Includes collective reclamation: pages shared only by idle checkpoints
  // become free after all of those checkpoints are removed.
  [[nodiscard]] PoolMemoryStats gpu_memory_stats() const;
  // Bytes reserved in live GPU global pages beyond their populated token rows.
  // Local-ring capacity is execution scratch, not retained-page slack.
  [[nodiscard]] std::size_t global_page_slack_bytes() const;
  [[nodiscard]] bool has_checkpoint(CheckpointId checkpoint) const;

 private:
  struct PageRecord {
    PageInfo info;
  };
  struct CheckpointRecord {
    CheckpointInfo info;
  };
  struct ExecutionRecord {
    ExecutionInfo info;
  };

  static std::size_t checked_mul(std::size_t left, std::size_t right,
                                 const char* label);
  static std::size_t checkpoint_metadata_bytes(std::size_t global_page_count);
  static std::size_t execution_page_capacity(const PoolConfig& config);
  static std::size_t execution_metadata_bytes(
      std::size_t global_page_capacity);
  [[nodiscard]] std::size_t batch_reserved_pages() const;
  [[nodiscard]] bool has_batch_executions() const;
  ExecutionId begin_execution_impl(CheckpointId source,
                                  std::uint32_t batch_max_processed_tokens,
                                  ExecutionId source_execution = 0);
  void reserve_index(std::size_t bytes, const char* label);
  void release_index(std::size_t bytes);
  PageId create_page(std::uint32_t first_token, bool batch = false);
  void retain_page(PageId page);
  void release_page(PageId page);
  void require_checkpoint(CheckpointId checkpoint) const;
  // Releases without re-checking invariants; pressure eviction runs while an
  // execution's page list can be mid-mutation inside prepare_write.
  void erase_checkpoint(CheckpointId checkpoint, bool externally_preserved);
  void check_invariants() const;

  PoolConfig config_;
  BytePool gpu_pool_;
  BytePool cpu_pool_;
  std::size_t index_used_{};
  PageId next_page_id_{1};
  CheckpointId next_checkpoint_id_{1};
  ExecutionId next_execution_id_{1};
  std::vector<PageRecord> pages_;
  std::vector<CheckpointRecord> checkpoints_;
  std::vector<ExecutionRecord> executions_;
  EvictionSelector eviction_selector_;
  PressureEviction pressure_eviction_;
  CheckpointRemoved checkpoint_removed_;
  std::size_t copy_on_write_pages_{};
};

}  // namespace gewell::kv_cache
