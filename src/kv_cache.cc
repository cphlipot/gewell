#include "gewell/kv_cache.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace gewell::kv_cache {
namespace {

[[noreturn]] void fail(std::string_view operation, std::string_view detail) {
  throw std::runtime_error(std::string(operation) + ": " +
                           std::string(detail));
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    fail("KV byte pool", "alignment must be a nonzero power of two");
  }
  if (value > std::numeric_limits<std::size_t>::max() - alignment + 1) {
    fail("KV byte pool", "aligned allocation overflows size_t");
  }
  return (value + alignment - 1) & ~(alignment - 1);
}

std::size_t local_ring_bytes_per_token(const PoolConfig& config) {
  if (config.local_window_tokens == 0 || config.local_ring_bytes == 0) {
    fail("KV cache configuration", "local representation is empty");
  }
  if (config.local_ring_bytes % config.local_window_tokens != 0) {
    fail("KV cache configuration", "local ring is not token divisible");
  }
  return config.local_ring_bytes / config.local_window_tokens;
}

template <typename T>
decltype(auto) find_by_id(std::vector<T>& values, std::uint64_t id,
                          const char* label) {
  const auto found = std::find_if(
      values.begin(), values.end(),
      [id](const T& value) { return value.info.id == id; });
  if (found == values.end()) {
    fail(label, "unknown identifier");
  }
  return (found->info);
}

template <typename T>
decltype(auto) find_by_id(const std::vector<T>& values, std::uint64_t id,
                          const char* label) {
  const auto found = std::find_if(
      values.begin(), values.end(),
      [id](const T& value) { return value.info.id == id; });
  if (found == values.end()) {
    fail(label, "unknown identifier");
  }
  return (found->info);
}

}  // namespace

void PoolConfig::validate() const {
  if (gpu_bytes == 0) {
    fail("KV cache configuration", "GPU budget must be positive");
  }
  if (index_bytes == 0) {
    fail("KV cache configuration", "index budget must be positive");
  }
  if (global_page_tokens == 0 || local_window_tokens == 0 ||
      maximum_context_tokens == 0 ||
      global_page_tokens > maximum_context_tokens ||
      local_window_tokens > maximum_context_tokens) {
    fail("KV cache configuration", "token capacities are invalid");
  }
  if (global_page_bytes == 0 || local_ring_bytes == 0 ||
      local_bytes_per_token == 0 || terminal_hidden_bytes == 0 ||
      page_table_bytes == 0) {
    fail("KV cache configuration", "representation byte sizes are empty");
  }
  if (local_ring_bytes !=
      static_cast<std::size_t>(local_window_tokens) * local_bytes_per_token) {
    fail("KV cache configuration", "local ring/token sizes disagree");
  }
  const std::size_t page_count =
      (static_cast<std::size_t>(maximum_context_tokens) +
       global_page_tokens - 1) /
      global_page_tokens;
  if (page_table_bytes < page_count * sizeof(std::uint64_t)) {
    fail("KV cache configuration", "page table cannot cover context");
  }
}

BytePool::BytePool(std::size_t capacity, Tier tier)
    : capacity_(capacity), tier_(tier) {
  if (capacity != 0) {
    free_ranges_.push_back({0, capacity});
  }
}

Allocation BytePool::allocate(std::size_t bytes, std::size_t alignment) {
  Allocation allocation;
  if (!try_allocate(bytes, &allocation, alignment)) {
    fail("KV byte pool", "fixed budget exhausted");
  }
  return allocation;
}

bool BytePool::try_allocate(std::size_t bytes, Allocation* out,
                            std::size_t alignment) {
  if (bytes == 0) {
    fail("KV byte pool", "zero-sized allocation");
  }
  if (out == nullptr) {
    fail("KV byte pool", "allocation output is null");
  }
  for (std::size_t index = 0; index < free_ranges_.size(); ++index) {
    const FreeRange range = free_ranges_[index];
    const std::size_t aligned = align_up(range.offset, alignment);
    if (aligned < range.offset || aligned - range.offset > range.bytes ||
        bytes > range.bytes - (aligned - range.offset)) {
      continue;
    }
    const std::size_t prefix = aligned - range.offset;
    const std::size_t suffix = range.bytes - prefix - bytes;
    free_ranges_.erase(free_ranges_.begin() + static_cast<std::ptrdiff_t>(index));
    if (suffix != 0) {
      free_ranges_.insert(free_ranges_.begin() +
                              static_cast<std::ptrdiff_t>(index),
                          {aligned + bytes, suffix});
    }
    if (prefix != 0) {
      free_ranges_.insert(free_ranges_.begin() +
                              static_cast<std::ptrdiff_t>(index),
                          {range.offset, prefix});
    }

    Allocation allocation{next_id_++, tier_, aligned, bytes};
    live_.push_back({allocation});
    used_ += bytes;
    peak_used_ = std::max(peak_used_, used_);
    *out = allocation;
    return true;
  }
  return false;
}

bool BytePool::can_allocate_sequence(
    const std::vector<std::size_t>& bytes, std::size_t alignment) const {
  std::vector<FreeRange> ranges = free_ranges_;
  for (const std::size_t requested : bytes) {
    if (requested == 0) {
      fail("KV byte pool", "zero-sized allocation");
    }
    bool allocated = false;
    for (std::size_t index = 0; index < ranges.size(); ++index) {
      const FreeRange range = ranges[index];
      const std::size_t aligned = align_up(range.offset, alignment);
      if (aligned < range.offset || aligned - range.offset > range.bytes ||
          requested > range.bytes - (aligned - range.offset)) {
        continue;
      }
      const std::size_t prefix = aligned - range.offset;
      const std::size_t suffix = range.bytes - prefix - requested;
      ranges.erase(ranges.begin() + static_cast<std::ptrdiff_t>(index));
      if (suffix != 0) {
        ranges.insert(ranges.begin() + static_cast<std::ptrdiff_t>(index),
                      {aligned + requested, suffix});
      }
      if (prefix != 0) {
        ranges.insert(ranges.begin() + static_cast<std::ptrdiff_t>(index),
                      {range.offset, prefix});
      }
      allocated = true;
      break;
    }
    if (!allocated) {
      return false;
    }
  }
  return true;
}

void BytePool::release(const Allocation& allocation) {
  const auto found = std::find_if(
      live_.begin(), live_.end(), [&allocation](const LiveAllocation& value) {
        return value.allocation.id == allocation.id;
      });
  if (found == live_.end() || found->allocation.offset != allocation.offset ||
      found->allocation.bytes != allocation.bytes ||
      found->allocation.tier != allocation.tier) {
    fail("KV byte pool", "release does not match a live allocation");
  }
  const Allocation released = found->allocation;
  live_.erase(found);
  used_ -= released.bytes;
  free_ranges_.push_back({released.offset, released.bytes});
  std::sort(free_ranges_.begin(), free_ranges_.end(),
            [](const FreeRange& left, const FreeRange& right) {
              return left.offset < right.offset;
            });
  std::vector<FreeRange> merged;
  merged.reserve(free_ranges_.size());
  for (const FreeRange range : free_ranges_) {
    if (!merged.empty() &&
        merged.back().offset + merged.back().bytes == range.offset) {
      merged.back().bytes += range.bytes;
    } else {
      merged.push_back(range);
    }
  }
  free_ranges_.swap(merged);
}

PoolStats BytePool::stats() const {
  return {capacity_, used_, peak_used_, capacity_ - used_};
}

bool BytePool::owns(const Allocation& allocation) const {
  return std::any_of(live_.begin(), live_.end(), [&allocation](const auto& live) {
    return live.allocation.id == allocation.id &&
           live.allocation.offset == allocation.offset &&
           live.allocation.bytes == allocation.bytes &&
           live.allocation.tier == allocation.tier;
  });
}

CacheLedger::CacheLedger(PoolConfig config)
    : config_(std::move(config)),
      gpu_pool_(config_.gpu_bytes, Tier::gpu),
      cpu_pool_(config_.cpu_bytes, Tier::cpu) {
  config_.validate();
}

std::size_t CacheLedger::checked_mul(std::size_t left, std::size_t right,
                                     const char* label) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    fail(label, "size multiplication overflows size_t");
  }
  return left * right;
}

std::size_t CacheLedger::checkpoint_metadata_bytes(
    std::size_t global_page_count) {
  const std::size_t page_bytes = checked_mul(
      global_page_count, sizeof(PageId), "checkpoint metadata accounting");
  if (page_bytes > std::numeric_limits<std::size_t>::max() -
                       sizeof(CheckpointInfo)) {
    fail("checkpoint metadata accounting", "size addition overflows size_t");
  }
  return sizeof(CheckpointInfo) + page_bytes;
}

std::size_t CacheLedger::execution_page_capacity(const PoolConfig& config) {
  // Reserve the complete bounded page-ID table with the execution, rather
  // than letting std::vector grow outside the index budget during prefill.
  return (static_cast<std::size_t>(config.maximum_context_tokens) - 1) /
             config.global_page_tokens +
         1;
}

std::size_t CacheLedger::execution_metadata_bytes(
    std::size_t global_page_capacity) {
  const std::size_t page_bytes = checked_mul(
      global_page_capacity, sizeof(PageId), "execution metadata accounting");
  if (page_bytes > std::numeric_limits<std::size_t>::max() -
                       sizeof(ExecutionInfo)) {
    fail("execution metadata accounting", "size addition overflows size_t");
  }
  return sizeof(ExecutionInfo) + page_bytes;
}

void CacheLedger::reserve_index(std::size_t bytes, const char* label) {
  if (index_used_ > config_.index_bytes ||
      bytes > config_.index_bytes - index_used_) {
    fail(label, "metadata index budget exhausted");
  }
  index_used_ += bytes;
}

void CacheLedger::release_index(std::size_t bytes) {
  if (bytes > index_used_) {
    fail("KV cache index", "metadata release exceeds usage");
  }
  index_used_ -= bytes;
}

PageId CacheLedger::create_page(std::uint32_t first_token, bool batch) {
  reserve_index(sizeof(PageInfo), "create KV page");
  Allocation storage;
  try {
    PageRecord record;
    record.info.id = next_page_id_++;
    // Active executions hold references on their source checkpoints, so
    // evicting idle checkpoints here can never displace borrowed state.
    storage = batch ? gpu_pool_.allocate(config_.global_page_bytes)
                    : allocate_gpu_releasing_idle(
                          config_.global_page_bytes, 256, 0);
    record.info.storage = storage;
    record.info.first_token = first_token;
    record.info.valid_tokens = 0;
    record.info.references = 1;
    pages_.push_back(std::move(record));
    return pages_.back().info.id;
  } catch (...) {
    if (storage.valid() && gpu_pool_.owns(storage)) {
      gpu_pool_.release(storage);
    }
    release_index(sizeof(PageInfo));
    throw;
  }
}

void CacheLedger::retain_page(PageId page_id) {
  PageInfo& value = find_by_id(pages_, page_id, "retain KV page");
  ++value.references;
}

void CacheLedger::release_page(PageId page_id) {
  const auto found = std::find_if(
      pages_.begin(), pages_.end(),
      [page_id](const PageRecord& value) { return value.info.id == page_id; });
  if (found == pages_.end() || found->info.references == 0) {
    fail("release KV page", "unknown or unreferenced page");
  }
  --found->info.references;
  if (found->info.references == 0) {
    gpu_pool_.release(found->info.storage);
    release_index(sizeof(PageInfo));
    pages_.erase(found);
  }
}

ExecutionId CacheLedger::begin_execution(CheckpointId source) {
  if (has_batch_executions()) {
    fail("begin KV execution", "serial execution cannot overlap batch execution");
  }
  return begin_execution_impl(source, 0);
}

bool CacheLedger::has_batch_executions() const {
  return std::any_of(executions_.begin(), executions_.end(),
                     [](const ExecutionRecord& record) {
                       return record.info.batch_max_processed_tokens != 0;
                     });
}

std::size_t CacheLedger::batch_reserved_pages() const {
  std::size_t remaining = 0;
  for (const ExecutionRecord& record : executions_) {
    const ExecutionInfo& info = record.info;
    if (info.batch_max_processed_tokens == 0) {
      continue;
    }
    const std::size_t total =
        (static_cast<std::size_t>(info.batch_max_processed_tokens) - 1) /
            config_.global_page_tokens +
        1;
    if (info.global_pages.size() > total) {
      fail("batch KV reservation", "reserved page accounting is invalid");
    }
    // A borrowed partial tail needs an extra physical page on its first write.
    // Once copied, its private replacement no longer consumes a reservation.
    const bool tail_copy = info.processed_tokens < info.batch_max_processed_tokens &&
        info.processed_tokens % config_.global_page_tokens != 0 &&
        page(info.global_pages.back()).references > 1;
    const std::size_t pages = total - info.global_pages.size() + tail_copy;
    if (pages > std::numeric_limits<std::size_t>::max() - remaining) {
      fail("batch KV reservation", "reserved page accounting is invalid");
    }
    remaining += pages;
  }
  return remaining;
}

bool CacheLedger::batch_request_fits(
    CheckpointId source, std::size_t maximum_processed_tokens) const {
  if (maximum_processed_tokens == 0 ||
      maximum_processed_tokens > config_.maximum_context_tokens) {
    fail("batch KV admission", "processed-token limit is outside the context");
  }
  const CheckpointInfo* const source_info =
      source == 0 ? nullptr : &checkpoint(source);
  const std::size_t resumed =
      source_info == nullptr ? 0 : source_info->processed_tokens;
  if (maximum_processed_tokens < resumed) {
    fail("batch KV admission", "processed-token limit precedes the source checkpoint");
  }
  const std::size_t source_pages =
      source_info == nullptr ? 0 : source_info->global_pages.size();
  const std::size_t total_pages =
      (maximum_processed_tokens - 1) / config_.global_page_tokens + 1;
  const bool tail_copy = resumed < maximum_processed_tokens &&
                        resumed % config_.global_page_tokens != 0;
  const std::size_t pages = total_pages - source_pages + tail_copy;
  const std::size_t metadata =
      execution_metadata_bytes(execution_page_capacity(config_));
  const std::size_t page_metadata =
      checked_mul(pages, sizeof(PageInfo), "batch KV admission");
  std::vector<std::size_t> allocation_plan{
      config_.local_ring_bytes, config_.page_table_bytes};
  allocation_plan.insert(allocation_plan.end(), pages, config_.global_page_bytes);
  // Inherited snapshots may have smaller alignment than new allocations.
  // Reject impossible payload budgets here; admission checks actual alignment
  // and fragmentation without repacking the source.
  std::size_t intrinsic_free = config_.gpu_bytes;
  const auto charge_intrinsic = [&](std::size_t bytes) {
    if (bytes > intrinsic_free) {
      return false;
    }
    intrinsic_free -= bytes;
    return true;
  };
  if (!charge_intrinsic(config_.local_ring_bytes) ||
      !charge_intrinsic(config_.page_table_bytes) ||
      !charge_intrinsic(checked_mul(total_pages + tail_copy,
                                   config_.global_page_bytes, "batch KV admission"))) {
    return false;
  }
  std::size_t source_metadata = 0;
  if (source_info != nullptr) {
    source_metadata = checkpoint_metadata_bytes(source_pages) +
        checked_mul(source_pages, sizeof(PageInfo), "batch KV admission");
    if (!charge_intrinsic(source_info->local_snapshot.bytes) ||
        !charge_intrinsic(source_info->terminal_hidden.bytes)) {
      return false;
    }
  }
  if (source_metadata > config_.index_bytes ||
      metadata > config_.index_bytes - source_metadata ||
      page_metadata > config_.index_bytes - source_metadata - metadata ||
      (source == 0 && !BytePool(config_.gpu_bytes, Tier::gpu).can_allocate_sequence(
                         allocation_plan))) {
    return false;
  }
  return true;
}

std::optional<ExecutionId> CacheLedger::try_begin_batch(
    CheckpointId source, std::size_t maximum_processed_tokens) {
  if (!batch_request_fits(source, maximum_processed_tokens)) {
    fail("batch KV admission", "request working set exceeds the fixed budget");
  }
  if (std::any_of(executions_.begin(), executions_.end(),
                  [](const ExecutionRecord& record) {
                    return record.info.batch_max_processed_tokens == 0;
                  })) {
    fail("batch KV admission", "batch execution cannot overlap serial execution");
  }
  const CheckpointInfo* const source_info =
      source == 0 ? nullptr : &checkpoint(source);
  const std::size_t resumed =
      source_info == nullptr ? 0 : source_info->processed_tokens;
  if (maximum_processed_tokens < resumed) {
    fail("batch KV admission", "processed-token limit precedes the source checkpoint");
  }
  const std::size_t source_pages =
      source_info == nullptr ? 0 : source_info->global_pages.size();
  const std::size_t total_pages =
      (maximum_processed_tokens - 1) / config_.global_page_tokens + 1;
  const bool tail_copy = resumed < maximum_processed_tokens &&
                        resumed % config_.global_page_tokens != 0;
  const std::size_t pages = total_pages - source_pages + tail_copy;
  const std::size_t metadata =
      execution_metadata_bytes(execution_page_capacity(config_));
  const std::size_t page_metadata =
      checked_mul(pages, sizeof(PageInfo), "batch KV admission");
  std::vector<std::size_t> allocation_plan{
      config_.local_ring_bytes, config_.page_table_bytes};
  allocation_plan.insert(allocation_plan.end(), pages, config_.global_page_bytes);

  const std::size_t reserved_pages = batch_reserved_pages();
  allocation_plan.insert(allocation_plan.end(), reserved_pages,
                         config_.global_page_bytes);
  const std::size_t reserved_metadata =
      checked_mul(reserved_pages, sizeof(PageInfo), "batch KV admission");
  const auto pressure_cause = [&]() -> std::optional<EvictionCause> {
    const std::size_t free_index = config_.index_bytes - index_used_;
    if (metadata > free_index || page_metadata > free_index - metadata ||
        reserved_metadata > free_index - metadata - page_metadata) {
      return EvictionCause::index_pressure;
    }
    if (!gpu_pool_.can_allocate_sequence(allocation_plan)) {
      return EvictionCause::gpu_pressure;
    }
    return std::nullopt;
  };
  for (;;) {
    const auto cause = pressure_cause();
    if (!cause.has_value()) {
      break;
    }
    if (!evict_idle_checkpoint(source, *cause)) {
      return std::nullopt;
    }
  }
  // Every remaining growth allocation has the same size/alignment. Any
  // interleaving of admitted requests therefore consumes this proven sequence
  // in the same physical order. No future KV pages are allocated up front.
  return begin_execution_impl(
      source, static_cast<std::uint32_t>(maximum_processed_tokens));
}

std::optional<ExecutionId> CacheLedger::try_fork_batch(
    ExecutionId source_execution, std::size_t maximum_processed_tokens) {
  const ExecutionInfo& source = execution(source_execution);
  if (source.batch_max_processed_tokens == 0 || maximum_processed_tokens == 0 ||
      maximum_processed_tokens < source.processed_tokens ||
      maximum_processed_tokens > config_.maximum_context_tokens) {
    fail("fork batch KV execution", "source or processed-token limit is invalid");
  }
  const std::size_t metadata =
      execution_metadata_bytes(execution_page_capacity(config_));
  const std::size_t total_pages =
      (maximum_processed_tokens - 1) / config_.global_page_tokens + 1;
  const bool partial_tail =
      source.processed_tokens % config_.global_page_tokens != 0;
  const std::size_t fork_pages = total_pages - source.global_pages.size() +
      (partial_tail && source.processed_tokens < maximum_processed_tokens);
  const auto pressure_cause = [&]() -> std::optional<EvictionCause> {
    // Forking a private partial tail also creates a COW obligation for the
    // source when it may continue. Recompute after eviction: an idle snapshot
    // disappearing can make the source tail private again.
    const bool source_tail_copy = partial_tail &&
        source.processed_tokens < source.batch_max_processed_tokens &&
        page(source.global_pages.back()).references == 1;
    const std::size_t reserved_pages =
        batch_reserved_pages() + fork_pages + source_tail_copy;
    const std::size_t page_metadata = checked_mul(
        reserved_pages, sizeof(PageInfo), "fork batch KV execution");
    const std::size_t free_index = config_.index_bytes - index_used_;
    if (metadata > free_index || page_metadata > free_index - metadata) {
      return EvictionCause::index_pressure;
    }
    std::vector<std::size_t> allocation_plan{
        config_.local_ring_bytes, config_.page_table_bytes};
    allocation_plan.insert(allocation_plan.end(), reserved_pages,
                           config_.global_page_bytes);
    return gpu_pool_.can_allocate_sequence(allocation_plan)
               ? std::nullopt
               : std::optional<EvictionCause>(EvictionCause::gpu_pressure);
  };
  for (;;) {
    const auto cause = pressure_cause();
    if (!cause.has_value()) {
      break;
    }
    if (!evict_idle_checkpoint(0, *cause)) {
      return std::nullopt;
    }
  }
  return begin_execution_impl(
      0, static_cast<std::uint32_t>(maximum_processed_tokens), source_execution);
}

bool CacheLedger::try_resize_batch(
    ExecutionId execution_id, std::size_t maximum_processed_tokens,
    EvictionCause* failure_cause) {
  ExecutionInfo& info = execution(execution_id);
  if (info.batch_max_processed_tokens == 0 || maximum_processed_tokens == 0 ||
      maximum_processed_tokens < info.processed_tokens ||
      maximum_processed_tokens > config_.maximum_context_tokens) {
    fail("resize batch KV execution", "execution or processed-token limit is invalid");
  }
  const std::uint32_t previous = info.batch_max_processed_tokens;
  info.batch_max_processed_tokens =
      static_cast<std::uint32_t>(maximum_processed_tokens);
  if (maximum_processed_tokens <= previous) {
    return true;
  }
  try {
    const std::size_t reserved_pages = batch_reserved_pages();
    const std::size_t page_metadata = checked_mul(
        reserved_pages, sizeof(PageInfo), "resize batch KV execution");
    const bool index_fits =
        page_metadata <= config_.index_bytes - index_used_;
    if (!index_fits) {
      if (failure_cause != nullptr) {
        *failure_cause = EvictionCause::index_pressure;
      }
    } else if (gpu_pool_.can_allocate_sequence(std::vector<std::size_t>(
                   reserved_pages, config_.global_page_bytes))) {
      return true;
    } else if (failure_cause != nullptr) {
      *failure_cause = EvictionCause::gpu_pressure;
    }
  } catch (...) {
    info.batch_max_processed_tokens = previous;
    throw;
  }
  info.batch_max_processed_tokens = previous;
  return false;
}

ExecutionId CacheLedger::begin_execution_impl(
    CheckpointId source, std::uint32_t batch_max_processed_tokens,
    ExecutionId source_execution) {
  if (source != 0) {
    require_checkpoint(source);
  }
  const std::size_t page_capacity = execution_page_capacity(config_);
  const std::size_t metadata_bytes = execution_metadata_bytes(page_capacity);
  reserve_index(metadata_bytes, "begin KV execution");
  ExecutionRecord record;
  bool committed = false;
  std::size_t retained_pages = 0;
  bool source_reference = false;
  try {
    record.info.id = next_execution_id_++;
    record.info.batch_max_processed_tokens = batch_max_processed_tokens;
    record.info.global_pages.reserve(page_capacity);
    if (record.info.global_pages.capacity() != page_capacity) {
      fail("begin KV execution",
           "page-ID vector did not honor its bounded capacity");
    }
    record.info.local_ring = batch_max_processed_tokens != 0
        ? gpu_pool_.allocate(config_.local_ring_bytes)
        : allocate_gpu_releasing_idle(config_.local_ring_bytes, 256, source);
    record.info.page_table = batch_max_processed_tokens != 0
        ? gpu_pool_.allocate(config_.page_table_bytes)
        : allocate_gpu_releasing_idle(config_.page_table_bytes, 256, source);
    record.info.source_checkpoint = source;
    if (source != 0) {
      const CheckpointInfo& checkpoint_info = checkpoint(source);
      record.info.processed_tokens = checkpoint_info.processed_tokens;
      record.info.global_pages.insert(record.info.global_pages.end(),
                                      checkpoint_info.global_pages.begin(),
                                      checkpoint_info.global_pages.end());
      for (const PageId page_id : record.info.global_pages) {
        retain_page(page_id);
        ++retained_pages;
      }
      // Keep the source checkpoint alive until this borrower releases it.
      checkpoint(source).execution_references++;
      source_reference = true;
    } else if (source_execution != 0) {
      const ExecutionInfo& source_info = execution(source_execution);
      record.info.processed_tokens = source_info.processed_tokens;
      record.info.global_pages.insert(record.info.global_pages.end(),
                                      source_info.global_pages.begin(),
                                      source_info.global_pages.end());
      for (const PageId page_id : record.info.global_pages) {
        retain_page(page_id);
        ++retained_pages;
      }
    }
    executions_.reserve(executions_.size() + 1);
    executions_.push_back(std::move(record));
    committed = true;
    check_invariants();
    return executions_.back().info.id;
  } catch (...) {
    if (committed) {
      ExecutionInfo& value = executions_.back().info;
      for (const PageId page_id : value.global_pages) {
        release_page(page_id);
      }
      if (source_reference && value.source_checkpoint != 0) {
        CheckpointInfo& checkpoint_info = checkpoint(value.source_checkpoint);
        if (checkpoint_info.execution_references != 0) {
          --checkpoint_info.execution_references;
        }
      }
      gpu_pool_.release(value.local_ring);
      gpu_pool_.release(value.page_table);
      executions_.pop_back();
    } else {
      for (std::size_t index = 0; index < retained_pages; ++index) {
        release_page(record.info.global_pages[index]);
      }
      if (record.info.page_table.valid() &&
          gpu_pool_.owns(record.info.page_table)) {
        gpu_pool_.release(record.info.page_table);
      }
      if (record.info.local_ring.valid() &&
          gpu_pool_.owns(record.info.local_ring)) {
        gpu_pool_.release(record.info.local_ring);
      }
      if (source_reference && source != 0 && has_checkpoint(source)) {
        CheckpointInfo& checkpoint_info = checkpoint(source);
        if (checkpoint_info.execution_references != 0) {
          --checkpoint_info.execution_references;
        }
      }
    }
    release_index(metadata_bytes);
    throw;
  }
}

WritePlan CacheLedger::prepare_write(ExecutionId execution_id,
                                     std::uint32_t first_token,
                                     std::uint32_t token_count) {
  if (token_count == 0) {
    fail("prepare KV write", "token count must be positive");
  }
  ExecutionInfo& execution_info = this->execution(execution_id);
  if (execution_info.pending_checkpoint) {
    fail("prepare KV write", "execution checkpoint capture is still pending");
  }
  if (first_token != execution_info.processed_tokens) {
    fail("prepare KV write", "write is not contiguous with execution state");
  }
  const std::uint64_t end = static_cast<std::uint64_t>(first_token) +
                            static_cast<std::uint64_t>(token_count);
  if (end > config_.maximum_context_tokens) {
    fail("prepare KV write", "write exceeds configured context");
  }
  if (execution_info.batch_max_processed_tokens != 0 &&
      end > execution_info.batch_max_processed_tokens) {
    fail("prepare KV write", "write exceeds the admitted batch reservation");
  }

  WritePlan plan{first_token, token_count, {}, {}};
  const std::uint32_t page_tokens = config_.global_page_tokens;
  const std::uint32_t first_page = first_token / page_tokens;
  const std::uint32_t last_page =
      static_cast<std::uint32_t>((end - 1) / page_tokens);
  const std::vector<PageId> original_pages = execution_info.global_pages;
  std::vector<std::uint32_t> original_valid_tokens;
  original_valid_tokens.reserve(original_pages.size());
  for (const PageId page_id : original_pages) {
    original_valid_tokens.push_back(page(page_id).valid_tokens);
  }
  const std::uint32_t original_processed = execution_info.processed_tokens;
  struct Replacement {
    std::uint32_t index{};
    PageId old_page{};
    PageId new_page{};
  };
  std::vector<Replacement> replacements;
  std::vector<PageId> created_pages;
  try {
    const std::size_t touched_pages =
        static_cast<std::size_t>(last_page - first_page) + 1;
    created_pages.reserve(touched_pages);
    replacements.reserve(touched_pages);
    plan.new_pages.reserve(touched_pages);
    plan.copies.reserve(touched_pages);
    if (last_page >= execution_info.global_pages.capacity()) {
      fail("prepare KV write", "execution page-ID reservation is exhausted");
    }
    for (std::uint32_t page_index = first_page; page_index <= last_page;
         ++page_index) {
      if (page_index >= execution_info.global_pages.size()) {
        const PageId page_id = create_page(
            page_index * page_tokens,
            execution_info.batch_max_processed_tokens != 0);
        created_pages.push_back(page_id);
        execution_info.global_pages.push_back(page_id);
        plan.new_pages.push_back(page_id);
        continue;
      }
      const PageId old_page_id = execution_info.global_pages[page_index];
      const PageInfo& old_page = page(old_page_id);
      if (old_page.references > 1) {
        // create_page() may grow pages_ or pressure-evict another checkpoint,
        // either of which can invalidate references into pages_. Preserve the
        // source metadata before allocating the replacement.
        const std::uint32_t old_first_token = old_page.first_token;
        const std::uint32_t old_valid_tokens = old_page.valid_tokens;
        const PageId new_page_id = create_page(
            old_first_token, execution_info.batch_max_processed_tokens != 0);
        if (page(old_page_id).references == 1) {
          // Allocation pressure may have evicted the last idle checkpoint
          // sharing this page. The execution can then safely keep writing its
          // now-exclusive page, and no source copy is needed.
          release_page(new_page_id);
          continue;
        }
        created_pages.push_back(new_page_id);
        PageInfo& new_page = page(new_page_id);
        new_page.valid_tokens = old_valid_tokens;
        execution_info.global_pages[page_index] = new_page_id;
        release_page(old_page_id);
        replacements.push_back({page_index, old_page_id, new_page_id});
        plan.copies.push_back({old_page_id, new_page_id});
        ++copy_on_write_pages_;
      }
    }
    execution_info.processed_tokens = static_cast<std::uint32_t>(end);
    for (std::uint32_t page_index = first_page; page_index <= last_page;
         ++page_index) {
      PageInfo& page_info = page(execution_info.global_pages[page_index]);
      const std::uint32_t valid = static_cast<std::uint32_t>(std::min<
          std::uint64_t>(page_tokens,
                         end - static_cast<std::uint64_t>(page_info.first_token)));
      page_info.valid_tokens = std::max(page_info.valid_tokens, valid);
    }
    check_invariants();
    return plan;
  } catch (...) {
    // Restore the logical vector first, then undo each physical reference. The
    // old page remains alive because a retained checkpoint owns the shared
    // page whenever this path is reachable.
    execution_info.global_pages = original_pages;
    execution_info.processed_tokens = original_processed;
    for (std::size_t index = 0; index < original_pages.size(); ++index) {
      page(original_pages[index]).valid_tokens = original_valid_tokens[index];
    }
    for (const Replacement& replacement : replacements) {
      retain_page(replacement.old_page);
      release_page(replacement.new_page);
    }
    for (const PageId page_id : created_pages) {
      const bool was_replacement = std::any_of(
          replacements.begin(), replacements.end(),
          [page_id](const Replacement& replacement) {
            return replacement.new_page == page_id;
          });
      if (!was_replacement) {
        release_page(page_id);
      }
    }
    copy_on_write_pages_ -= replacements.size();
    throw;
  }
}

std::optional<CheckpointCapture> CacheLedger::try_begin_checkpoint_capture(
    ExecutionId execution_id, EvictionCause* failure_cause) {
  ExecutionInfo& info = execution(execution_id);
  if (info.processed_tokens == 0 || info.pending_checkpoint) {
    fail("begin KV checkpoint", "execution is empty or already capturing");
  }
  const std::size_t local_tokens =
      std::min<std::size_t>(info.processed_tokens, config_.local_window_tokens);
  const std::size_t local_bytes = checked_mul(
      local_tokens, config_.local_bytes_per_token, "begin KV checkpoint");
  const std::size_t metadata_bytes =
      checkpoint_metadata_bytes(info.global_pages.size());
  // Retaining a currently private partial tail introduces another future page
  // allocation. A tail already shared with a checkpoint has its COW reserved.
  const bool new_tail_copy = info.batch_max_processed_tokens != 0 &&
      info.processed_tokens < info.batch_max_processed_tokens &&
      info.processed_tokens % config_.global_page_tokens != 0 &&
      page(info.global_pages.back()).references == 1;
  const std::size_t reserved_pages = batch_reserved_pages() + new_tail_copy;
  const std::size_t reserved_metadata = checked_mul(
      reserved_pages, sizeof(PageInfo), "begin KV checkpoint");
  std::vector<std::size_t> allocation_plan{
      local_bytes, config_.terminal_hidden_bytes};
  allocation_plan.insert(allocation_plan.end(), reserved_pages,
                         config_.global_page_bytes);
  const std::size_t free_index = config_.index_bytes - index_used_;
  if (metadata_bytes > free_index ||
      reserved_metadata > free_index - metadata_bytes) {
    if (failure_cause != nullptr) {
      *failure_cause = EvictionCause::index_pressure;
    }
    return std::nullopt;
  }
  if (!gpu_pool_.can_allocate_sequence(allocation_plan)) {
    if (failure_cause != nullptr) {
      *failure_cause = EvictionCause::gpu_pressure;
    }
    return std::nullopt;
  }

  reserve_index(metadata_bytes, "begin KV checkpoint");
  CheckpointCapture capture;
  std::size_t retained_pages = 0;
  try {
    capture.local_snapshot = gpu_pool_.allocate(local_bytes);
    capture.terminal_hidden = gpu_pool_.allocate(config_.terminal_hidden_bytes);
    for (const PageId page_id : info.global_pages) {
      retain_page(page_id);
      ++retained_pages;
    }
    info.pending_checkpoint = capture;
    check_invariants();
    return capture;
  } catch (...) {
    info.pending_checkpoint.reset();
    for (std::size_t index = 0; index < retained_pages; ++index) {
      release_page(info.global_pages[index]);
    }
    if (capture.local_snapshot.valid() && gpu_pool_.owns(capture.local_snapshot)) {
      gpu_pool_.release(capture.local_snapshot);
    }
    if (capture.terminal_hidden.valid() && gpu_pool_.owns(capture.terminal_hidden)) {
      gpu_pool_.release(capture.terminal_hidden);
    }
    release_index(metadata_bytes);
    throw;
  }
}

void CacheLedger::abort_checkpoint_capture(ExecutionId execution_id) {
  ExecutionInfo& info = execution(execution_id);
  if (!info.pending_checkpoint) {
    return;
  }
  for (const PageId page_id : info.global_pages) {
    release_page(page_id);
  }
  gpu_pool_.release(info.pending_checkpoint->local_snapshot);
  gpu_pool_.release(info.pending_checkpoint->terminal_hidden);
  release_index(checkpoint_metadata_bytes(info.global_pages.size()));
  info.pending_checkpoint.reset();
  check_invariants();
}

CheckpointId CacheLedger::publish_checkpoint(
    ExecutionId execution_id,
    const std::vector<std::uint8_t>& local_snapshot_bytes,
    const std::vector<std::uint8_t>& terminal_hidden_bytes) {
  const ExecutionInfo& info = execution(execution_id);
  const std::size_t expected_local = checked_mul(
      std::min<std::size_t>(info.processed_tokens, config_.local_window_tokens),
      config_.local_bytes_per_token, "publish KV checkpoint");
  if (local_snapshot_bytes.size() != expected_local ||
      terminal_hidden_bytes.size() != config_.terminal_hidden_bytes) {
    fail("publish KV checkpoint", "snapshot payload has the wrong size");
  }
  const auto capture = try_begin_checkpoint_capture(execution_id);
  if (!capture) {
    fail("publish KV checkpoint", "snapshot exceeds unreserved capacity");
  }
  try {
    return publish_checkpoint_allocations(
        execution_id, capture->local_snapshot, capture->terminal_hidden);
  } catch (...) {
    abort_checkpoint_capture(execution_id);
    throw;
  }
}

CheckpointId CacheLedger::publish_checkpoint_allocations(
    ExecutionId execution_id, Allocation local_snapshot,
    Allocation terminal_hidden) {
  ExecutionInfo& execution_info = execution(execution_id);
  const bool pending = execution_info.pending_checkpoint.has_value();
  if (has_batch_executions() && !pending) {
    fail("publish KV checkpoint", "batch publication requires a reserved capture");
  }
  if (execution_info.processed_tokens == 0) {
    fail("publish KV checkpoint", "execution state is empty");
  }
  const std::uint32_t local_tokens = static_cast<std::uint32_t>(std::min<
      std::size_t>(execution_info.processed_tokens,
                   config_.local_window_tokens));
  const std::uint32_t local_start =
      execution_info.processed_tokens - local_tokens;
  const std::size_t expected_local = checked_mul(
      local_tokens, config_.local_bytes_per_token, "publish KV checkpoint");
  if (!gpu_pool_.owns(local_snapshot) ||
      !gpu_pool_.owns(terminal_hidden) ||
      local_snapshot.bytes != expected_local ||
      terminal_hidden.bytes != config_.terminal_hidden_bytes) {
    fail("publish KV checkpoint", "snapshot allocations are invalid");
  }
  if (pending &&
      (local_snapshot.id != execution_info.pending_checkpoint->local_snapshot.id ||
       terminal_hidden.id != execution_info.pending_checkpoint->terminal_hidden.id)) {
    fail("publish KV checkpoint", "allocations do not belong to pending capture");
  }
  const std::size_t metadata_bytes =
      checkpoint_metadata_bytes(execution_info.global_pages.size());
  if (!pending) {
    reserve_index(metadata_bytes, "publish KV checkpoint");
  }
  std::size_t retained_pages = 0;
  try {
    CheckpointRecord record;
    record.info.id = next_checkpoint_id_++;
    record.info.processed_tokens = execution_info.processed_tokens;
    record.info.local_start = local_start;
    record.info.local_tokens = local_tokens;
    record.info.global_pages = execution_info.global_pages;
    record.info.local_snapshot = local_snapshot;
    record.info.terminal_hidden = terminal_hidden;
    checkpoints_.reserve(checkpoints_.size() + 1);
    if (!pending) {
      for (const PageId page_id : execution_info.global_pages) {
        retain_page(page_id);
        ++retained_pages;
      }
    }
    checkpoints_.push_back(std::move(record));
    execution_info.pending_checkpoint.reset();
    try {
      check_invariants();
    } catch (...) {
      if (pending) {
        execution_info.pending_checkpoint =
            CheckpointCapture{local_snapshot, terminal_hidden};
      } else {
        for (const PageId page_id : checkpoints_.back().info.global_pages) {
          release_page(page_id);
        }
      }
      checkpoints_.pop_back();
      retained_pages = 0;
      throw;
    }
    return checkpoints_.back().info.id;
  } catch (...) {
    if (retained_pages != 0) {
      for (std::size_t index = 0; index < retained_pages; ++index) {
        release_page(execution_info.global_pages[index]);
      }
    }
    if (!pending) {
      release_index(metadata_bytes);
    }
    throw;
  }
}

Allocation CacheLedger::try_allocate_gpu(std::size_t bytes,
                                         std::size_t alignment) {
  Allocation allocation;
  if (has_batch_executions()) {
    return allocation;
  }
  (void)gpu_pool_.try_allocate(bytes, &allocation, alignment);
  return allocation;
}

Allocation CacheLedger::try_allocate_cpu(std::size_t bytes,
                                         std::size_t alignment) {
  Allocation allocation;
  (void)cpu_pool_.try_allocate(bytes, &allocation, alignment);
  return allocation;
}

Allocation CacheLedger::allocate_gpu_releasing_idle(
    std::size_t bytes, std::size_t alignment,
    CheckpointId protected_checkpoint) {
  if (has_batch_executions()) {
    fail("KV byte pool", "offline batch reservations exclude unrelated allocations");
  }
  Allocation allocation;
  while (!gpu_pool_.try_allocate(bytes, &allocation, alignment)) {
    if (!evict_idle_checkpoint(protected_checkpoint)) {
      fail("KV byte pool", "fixed budget exhausted");
    }
  }
  return allocation;
}

bool CacheLedger::evict_idle_checkpoint(CheckpointId protected_checkpoint,
                                        EvictionCause cause) {
  std::vector<CheckpointId> eligible;
  eligible.reserve(checkpoints_.size());
  for (const CheckpointRecord& record : checkpoints_) {
    if (record.info.execution_references != 0 || record.info.dependency_pins != 0 ||
        record.info.id == protected_checkpoint) {
      continue;
    }
    eligible.push_back(record.info.id);
  }
  if (eligible.empty()) {
    return false;
  }
  if (eviction_selector_) {
    const CheckpointId selected =
        eviction_selector_(protected_checkpoint, eligible, cause);
    if (selected == 0 ||
        std::find(eligible.begin(), eligible.end(), selected) ==
            eligible.end()) {
      return false;
    }
    const bool externally_preserved =
        pressure_eviction_ && pressure_eviction_(selected);
    erase_checkpoint(selected, externally_preserved);
    return true;
  }
  const CheckpointId victim = eligible.front();
  const bool externally_preserved =
      pressure_eviction_ && pressure_eviction_(victim);
  erase_checkpoint(victim, externally_preserved);
  return true;
}

void CacheLedger::set_eviction_callbacks(EvictionSelector selector,
                                         PressureEviction pressure_eviction,
                                         CheckpointRemoved removed) {
  eviction_selector_ = std::move(selector);
  pressure_eviction_ = std::move(pressure_eviction);
  checkpoint_removed_ = std::move(removed);
}

void CacheLedger::release_gpu(const Allocation& allocation) {
  gpu_pool_.release(allocation);
}

void CacheLedger::release_cpu(const Allocation& allocation) {
  cpu_pool_.release(allocation);
}

void CacheLedger::release_execution(ExecutionId execution_id) {
  abort_checkpoint_capture(execution_id);
  const auto found = std::find_if(
      executions_.begin(), executions_.end(), [execution_id](const auto& value) {
        return value.info.id == execution_id;
      });
  if (found == executions_.end()) {
    fail("release KV execution", "unknown execution");
  }
  const std::size_t metadata_bytes =
      execution_metadata_bytes(found->info.global_pages.capacity());
  const ExecutionInfo& value = found->info;
  for (const PageId page_id : value.global_pages) {
    release_page(page_id);
  }
  if (value.source_checkpoint != 0) {
    CheckpointInfo& source = checkpoint(value.source_checkpoint);
    if (source.execution_references == 0) {
      fail("release KV execution", "source execution reference underflow");
    }
    --source.execution_references;
  }
  gpu_pool_.release(value.local_ring);
  gpu_pool_.release(value.page_table);
  release_index(metadata_bytes);
  executions_.erase(found);
  check_invariants();
}

void CacheLedger::release_checkpoint(CheckpointId checkpoint_id) {
  erase_checkpoint(checkpoint_id, false);
  check_invariants();
}

void CacheLedger::pin_checkpoint(CheckpointId checkpoint_id) {
  CheckpointInfo& info = checkpoint(checkpoint_id);
  if (info.dependency_pins == std::numeric_limits<std::size_t>::max()) {
    fail("pin KV checkpoint", "dependency pin count overflows");
  }
  ++info.dependency_pins;
}

void CacheLedger::unpin_checkpoint(CheckpointId checkpoint_id) {
  CheckpointInfo& info = checkpoint(checkpoint_id);
  if (info.dependency_pins == 0) {
    fail("unpin KV checkpoint", "checkpoint has no dependency pin");
  }
  --info.dependency_pins;
}

void CacheLedger::erase_checkpoint(CheckpointId checkpoint_id,
                                   bool externally_preserved) {
  const auto found = std::find_if(
      checkpoints_.begin(), checkpoints_.end(), [checkpoint_id](const auto& value) {
        return value.info.id == checkpoint_id;
      });
  if (found == checkpoints_.end()) {
    fail("release KV checkpoint", "unknown checkpoint");
  }
  if (found->info.execution_references != 0 || found->info.dependency_pins != 0) {
    fail("release KV checkpoint", "checkpoint is borrowed or pinned by a dependency");
  }
  const CheckpointInfo value = found->info;
  for (const PageId page_id : value.global_pages) {
    release_page(page_id);
  }
  gpu_pool_.release(value.local_snapshot);
  gpu_pool_.release(value.terminal_hidden);
  release_index(checkpoint_metadata_bytes(value.global_pages.size()));
  checkpoints_.erase(found);
  if (checkpoint_removed_) {
    checkpoint_removed_(checkpoint_id, externally_preserved);
  }
}

const PageInfo& CacheLedger::page(PageId page_id) const {
  return find_by_id(pages_, page_id, "KV page");
}

PageInfo& CacheLedger::page(PageId page_id) {
  return find_by_id(pages_, page_id, "KV page");
}

const CheckpointInfo& CacheLedger::checkpoint(CheckpointId checkpoint_id) const {
  return find_by_id(checkpoints_, checkpoint_id, "KV checkpoint");
}

CheckpointInfo& CacheLedger::checkpoint(CheckpointId checkpoint_id) {
  return find_by_id(checkpoints_, checkpoint_id, "KV checkpoint");
}

const ExecutionInfo& CacheLedger::execution(ExecutionId execution_id) const {
  return find_by_id(executions_, execution_id, "KV execution");
}

ExecutionInfo& CacheLedger::execution(ExecutionId execution_id) {
  return find_by_id(executions_, execution_id, "KV execution");
}

CacheStats CacheLedger::stats() const {
  return {gpu_pool_.stats(),       cpu_pool_.stats(),
          config_.index_bytes,     index_used_,
          pages_.size(),           checkpoints_.size(),
          executions_.size(),      copy_on_write_pages_};
}

void CacheLedger::visit_pages(
    const std::function<void(const PageInfo&)>& visitor) const {
  for (const auto& record : pages_) visitor(record.info);
}

void CacheLedger::visit_checkpoints(
    const std::function<void(const CheckpointInfo&)>& visitor) const {
  for (const auto& record : checkpoints_) visitor(record.info);
}

void CacheLedger::visit_executions(
    const std::function<void(const ExecutionInfo&)>& visitor) const {
  for (const auto& record : executions_) visitor(record.info);
}

PoolMemoryStats CacheLedger::gpu_memory_stats() const {
  PoolMemoryStats result;
  std::unordered_map<PageId, std::size_t> idle_references;
  idle_references.reserve(pages_.size());
  for (const auto& record : checkpoints_) {
    const auto& checkpoint = record.info;
    const auto state_bytes =
        checkpoint.local_snapshot.bytes + checkpoint.terminal_hidden.bytes;
    result.checkpoint_state_bytes += state_bytes;
    if (checkpoint.execution_references || checkpoint.dependency_pins) continue;
    result.reclaimable_bytes += state_bytes;
    for (const auto id : checkpoint.global_pages) ++idle_references[id];
  }
  for (const auto& record : pages_) {
    const auto& page = record.info;
    result.global_page_bytes += page.storage.bytes;
    if (page.references > 1) result.shared_page_bytes += page.storage.bytes;
    const auto found = idle_references.find(page.id);
    if (found != idle_references.end() && found->second == page.references)
      result.reclaimable_bytes += page.storage.bytes;
  }
  for (const auto& record : executions_) {
    const auto& execution = record.info;
    result.execution_buffer_bytes +=
        execution.local_ring.bytes + execution.page_table.bytes;
    if (execution.pending_checkpoint)
      result.pending_capture_bytes += execution.pending_checkpoint->local_snapshot.bytes +
                                      execution.pending_checkpoint->terminal_hidden.bytes;
  }
  const auto used = gpu_pool_.stats().used;
  const auto accounted = result.global_page_bytes + result.checkpoint_state_bytes +
                         result.execution_buffer_bytes + result.pending_capture_bytes;
  if (accounted > used || result.reclaimable_bytes > used)
    fail("KV memory statistics", "accounted bytes exceed pool usage");
  // Includes explicit caller allocations, such as serial speculative staging.
  result.other_bytes = used - accounted;
  result.nonreclaimable_bytes = used - result.reclaimable_bytes;
  result.reserved_growth_bytes =
      checked_mul(batch_reserved_pages(), config_.global_page_bytes,
                  "KV memory statistics");
  result.page_slack_bytes = global_page_slack_bytes();
  return result;
}

std::size_t CacheLedger::global_page_slack_bytes() const {
  if (config_.global_page_tokens == 0 ||
      config_.global_page_bytes % config_.global_page_tokens != 0) {
    fail("KV cache statistics", "global page layout is not token divisible");
  }
  const std::size_t bytes_per_token =
      config_.global_page_bytes / config_.global_page_tokens;
  std::size_t result = 0;
  for (const PageRecord& record : pages_) {
    if (record.info.valid_tokens > config_.global_page_tokens) {
      fail("KV cache statistics", "page has too many valid tokens");
    }
    const std::size_t slack_tokens =
        config_.global_page_tokens - record.info.valid_tokens;
    const std::size_t slack_bytes = checked_mul(
        slack_tokens, bytes_per_token, "KV cache statistics");
    if (slack_bytes > std::numeric_limits<std::size_t>::max() - result) {
      fail("KV cache statistics", "page slack byte count overflows size_t");
    }
    result += slack_bytes;
  }
  return result;
}

bool CacheLedger::has_checkpoint(CheckpointId checkpoint_id) const {
  return std::any_of(
      checkpoints_.begin(), checkpoints_.end(),
      [checkpoint_id](const CheckpointRecord& value) {
        return value.info.id == checkpoint_id;
      });
}

void CacheLedger::require_checkpoint(CheckpointId checkpoint_id) const {
  (void)checkpoint(checkpoint_id);
}

void CacheLedger::check_invariants() const {
  std::size_t page_references = 0;
  for (const PageRecord& record : pages_) {
    if (record.info.references == 0 ||
        record.info.valid_tokens > config_.global_page_tokens ||
        record.info.first_token % config_.global_page_tokens != 0) {
      fail("KV cache invariant", "invalid page state");
    }
    page_references += record.info.references;
  }
  for (const ExecutionRecord& record : executions_) {
    if (record.info.processed_tokens > config_.maximum_context_tokens ||
        (record.info.batch_max_processed_tokens != 0 &&
         (record.info.batch_max_processed_tokens > config_.maximum_context_tokens ||
          record.info.processed_tokens > record.info.batch_max_processed_tokens)) ||
        record.info.global_pages.capacity() != execution_page_capacity(config_) ||
        record.info.global_pages.size() !=
            (static_cast<std::size_t>(record.info.processed_tokens) +
             config_.global_page_tokens - 1) /
                config_.global_page_tokens) {
      fail("KV cache invariant", "execution pages do not cover state");
    }
    if (page_references < record.info.global_pages.size()) {
      fail("KV cache invariant", "execution references exceed page references");
    }
    page_references -= record.info.global_pages.size();
    if (record.info.pending_checkpoint) {
      if (page_references < record.info.global_pages.size() ||
          !gpu_pool_.owns(record.info.pending_checkpoint->local_snapshot) ||
          !gpu_pool_.owns(record.info.pending_checkpoint->terminal_hidden)) {
        fail("KV cache invariant", "pending checkpoint state is incomplete");
      }
      page_references -= record.info.global_pages.size();
    }
  }
  for (const CheckpointRecord& record : checkpoints_) {
    if (record.info.processed_tokens > config_.maximum_context_tokens ||
        record.info.local_tokens > config_.local_window_tokens ||
        record.info.local_start + record.info.local_tokens !=
            record.info.processed_tokens ||
        record.info.global_pages.size() !=
            (static_cast<std::size_t>(record.info.processed_tokens) +
             config_.global_page_tokens - 1) /
                config_.global_page_tokens) {
      fail("KV cache invariant", "checkpoint state is incomplete");
    }
    if (page_references < record.info.global_pages.size()) {
      fail("KV cache invariant", "checkpoint references exceed page references");
    }
    page_references -= record.info.global_pages.size();
  }
  if (page_references != 0) {
    fail("KV cache invariant", "page reference count disagrees with users");
  }
  if (index_used_ > config_.index_bytes ||
      gpu_pool_.stats().used > config_.gpu_bytes) {
    fail("KV cache invariant", "budget usage exceeds configured capacity");
  }
}

}  // namespace gewell::kv_cache
