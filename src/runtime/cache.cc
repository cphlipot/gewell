#include "gewell/runtime/cache.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <utility>

namespace gewell::runtime {
namespace {
[[noreturn]] void fail(std::string_view operation, std::string_view detail) {
  throw std::runtime_error(std::string(operation) + ": " + std::string(detail));
}
[[noreturn]] void fail_invalid_capacity(std::string_view operation,
                                       std::string_view detail) {
  throw CacheCapacityError(std::string(operation) + ": " + std::string(detail), true);
}
[[noreturn]] void fail_cache_capacity(std::string_view operation,
                                     std::string_view detail) {
  throw CacheCapacityError(std::string(operation) + ": " + std::string(detail), false);
}
}  // namespace

[[nodiscard]] CaptureTailCowHeadroom capture_tail_cow_headroom(
    bool reserve_future_writes, bool partial_tail,
    kv_cache::PageId tail_page, bool base_tail_cow_reserved,
    kv_cache::PageId pending_tail_page) {
  if (!reserve_future_writes) {
    return {};
  }
  if (pending_tail_page != 0 &&
      (!partial_tail || pending_tail_page != tail_page)) {
    fail("persistent KV checkpoint",
         "pending capture tail does not match the next write");
  }
  const bool keeps_pending_tail = pending_tail_page != 0;
  const bool reserves_new_tail =
      partial_tail && !base_tail_cow_reserved && !keeps_pending_tail;
  return {keeps_pending_tail, reserves_new_tail};
}

[[nodiscard]] std::size_t checked_telemetry_delta(std::size_t after,
                                                   std::size_t before,
                                                   std::string_view label) {
  if (after < before) {
    fail("server cache telemetry",
         std::string(label) + " counter moved backwards");
  }
  return after - before;
}

[[nodiscard]] double checked_telemetry_delta(double after, double before,
                                              std::string_view label) {
  if (!std::isfinite(after) || !std::isfinite(before) || after < before) {
    fail("server cache telemetry",
         std::string(label) + " elapsed time moved backwards");
  }
  return after - before;
}

[[nodiscard]] CacheTelemetry telemetry_delta(const CacheTelemetry& after,
                                              const CacheTelemetry& before) {
  CacheTelemetry result;
  for (std::size_t index = 0; index < result.checkpoint_buckets.size();
       ++index) {
    const CheckpointLifecycleBucket& total = after.checkpoint_buckets[index];
    const CheckpointLifecycleBucket& prior = before.checkpoint_buckets[index];
    result.checkpoint_buckets[index] = {
        checked_telemetry_delta(total.admissions, prior.admissions,
                                "checkpoint admissions"),
        checked_telemetry_delta(total.hits, prior.hits, "checkpoint hits"),
        checked_telemetry_delta(total.removals, prior.removals,
                                "checkpoint removals"),
        checked_telemetry_delta(total.admitted_bytes, prior.admitted_bytes,
                                "checkpoint admitted bytes"),
        checked_telemetry_delta(total.reclaimed_gpu_bytes,
                                prior.reclaimed_gpu_bytes,
                                "checkpoint GPU reclaimed bytes"),
        checked_telemetry_delta(total.reclaimed_cpu_bytes,
                                prior.reclaimed_cpu_bytes,
                                "checkpoint CPU reclaimed bytes"),
    };
  }
  result.gpu_reclaimed_bytes = checked_telemetry_delta(
      after.gpu_reclaimed_bytes, before.gpu_reclaimed_bytes,
      "GPU reclaimed bytes");
  result.cpu_reclaimed_bytes = checked_telemetry_delta(
      after.cpu_reclaimed_bytes, before.cpu_reclaimed_bytes,
      "CPU reclaimed bytes");
  result.cold_spill_bytes = checked_telemetry_delta(
      after.cold_spill_bytes, before.cold_spill_bytes, "cold spill bytes");
  result.cold_restore_bytes = checked_telemetry_delta(
      after.cold_restore_bytes, before.cold_restore_bytes,
      "cold restore bytes");
  result.cold_spill_count = checked_telemetry_delta(
      after.cold_spill_count, before.cold_spill_count, "cold spill count");
  result.cold_restore_count = checked_telemetry_delta(
      after.cold_restore_count, before.cold_restore_count,
      "cold restore count");
  result.cold_spill_avoided_rewrite_bytes = checked_telemetry_delta(
      after.cold_spill_avoided_rewrite_bytes,
      before.cold_spill_avoided_rewrite_bytes,
      "cold spill avoided rewrite bytes");
  result.cold_spill_wall_milliseconds = checked_telemetry_delta(
      after.cold_spill_wall_milliseconds,
      before.cold_spill_wall_milliseconds, "cold spill wall milliseconds");
  result.cold_restore_wall_milliseconds = checked_telemetry_delta(
      after.cold_restore_wall_milliseconds,
      before.cold_restore_wall_milliseconds,
      "cold restore wall milliseconds");
  return result;
}

[[nodiscard]] std::size_t lifecycle_bucket_index(
    prefix_index::CheckpointSources sources,
    prefix_index::RetentionPriority priority,
    prefix_index::CacheClass cache_class) {
  const std::size_t source = static_cast<std::size_t>(sources);
  const std::size_t priority_index = static_cast<std::size_t>(priority);
  const std::size_t class_index = static_cast<std::size_t>(cache_class);
  if (source >= kLifecycleSourceMaskCount ||
      priority_index >= kLifecyclePriorityCount ||
      class_index >= kLifecycleClassCount) {
    fail("server cache telemetry", "checkpoint metadata has an invalid bucket");
  }
  return (source * kLifecyclePriorityCount + priority_index) *
             kLifecycleClassCount +
         class_index;
}

void add_telemetry(std::size_t* destination, std::size_t value,
                   std::string_view label) {
  if (destination == nullptr ||
      value > std::numeric_limits<std::size_t>::max() - *destination) {
    fail("server cache telemetry", std::string(label) + " overflows");
  }
  *destination += value;
}

void add_telemetry(double* destination, double value, std::string_view label) {
  if (destination == nullptr || !std::isfinite(value) || value < 0.0 ||
      !std::isfinite(*destination) ||
      *destination > std::numeric_limits<double>::max() - value) {
    fail("server cache telemetry", std::string(label) + " is invalid");
  }
  *destination += value;
}

[[nodiscard]] const char* retention_priority_name(
    prefix_index::RetentionPriority priority) {
  switch (priority) {
    case prefix_index::RetentionPriority::low:
      return "low";
    case prefix_index::RetentionPriority::normal:
      return "normal";
    case prefix_index::RetentionPriority::high:
      return "high";
  }
  fail("server cache telemetry", "checkpoint priority is unknown");
}

[[nodiscard]] const char* cache_class_name(prefix_index::CacheClass value) {
  switch (value) {
    case prefix_index::CacheClass::probationary:
      return "probationary";
    case prefix_index::CacheClass::reused:
      return "reused";
  }
  fail("server cache telemetry", "checkpoint cache class is unknown");
}

void write_checkpoint_sources(std::ostream& output,
                              prefix_index::CheckpointSources sources) {
  struct SourceName {
    prefix_index::CheckpointSource source;
    const char* name;
  };
  constexpr std::array<SourceName, 4> names{{
      {prefix_index::CheckpointSource::input_endpoint, "input_endpoint"},
      {prefix_index::CheckpointSource::continuation_endpoint,
       "continuation_endpoint"},
      {prefix_index::CheckpointSource::learned_branch, "learned_branch"},
      {prefix_index::CheckpointSource::periodic, "periodic"},
  }};
  bool first = true;
  for (const SourceName& entry : names) {
    if (!prefix_index::has_checkpoint_source(sources, entry.source)) {
      continue;
    }
    output << (first ? "" : "|") << entry.name;
    first = false;
  }
  if (first) {
    output << "none";
  }
}

[[nodiscard]] std::string checkpoint_sources_name(
    prefix_index::CheckpointSources sources) {
  std::ostringstream output;
  write_checkpoint_sources(output, sources);
  return output.str();
}

void write_checkpoint_buckets(std::ostream& output,
                              const CacheTelemetry& telemetry) {
  output << '[';
  bool first = true;
  for (std::size_t source = 0; source < kLifecycleSourceMaskCount; ++source) {
    for (std::size_t priority = 0; priority < kLifecyclePriorityCount;
         ++priority) {
      for (std::size_t cache_class = 0; cache_class < kLifecycleClassCount;
           ++cache_class) {
        const std::size_t index =
            (source * kLifecyclePriorityCount + priority) *
                kLifecycleClassCount +
            cache_class;
        const CheckpointLifecycleBucket& bucket =
            telemetry.checkpoint_buckets[index];
        if (bucket.empty()) {
          continue;
        }
        output << (first ? "" : ",") << "{\"sources\":\"";
        write_checkpoint_sources(
            output, static_cast<prefix_index::CheckpointSources>(source));
        output << "\",\"priority\":\""
               << retention_priority_name(
                      static_cast<prefix_index::RetentionPriority>(priority))
               << "\",\"cache_class\":\""
               << cache_class_name(
                      static_cast<prefix_index::CacheClass>(cache_class))
               << "\",\"admissions\":" << bucket.admissions
               << ",\"hits\":" << bucket.hits
               << ",\"removals\":" << bucket.removals
               << ",\"admitted_bytes\":" << bucket.admitted_bytes
               << ",\"reclaimed_gpu_bytes\":"
               << bucket.reclaimed_gpu_bytes
               << ",\"reclaimed_cpu_bytes\":"
               << bucket.reclaimed_cpu_bytes << '}';
        first = false;
      }
    }
  }
  output << ']';
}

class PersistentCacheManager::Impl {
 private:
  struct DemandSlot {
    kv_cache::CheckpointId checkpoint{};
    std::array<char, kMaximumCacheOwnerBytes> owner{};
    std::uint8_t owner_bytes{};
    prefix_index::RetentionPriority priority{
        prefix_index::RetentionPriority::normal};
    bool automatic{};
  };

  struct ColdPage {
    kv_cache::PageId source_page{};
    kv_cache::Allocation storage;
    std::uint32_t valid_tokens{};
    std::size_t references{};
    bool pending_copy{};
  };

  struct ColdCheckpoint {
    kv_cache::CheckpointId checkpoint{};
    std::uint32_t processed_tokens{};
    std::uint32_t local_start{};
    std::uint32_t local_tokens{};
    std::size_t source_page_count{};
    kv_cache::Allocation local_snapshot;
    kv_cache::Allocation terminal_hidden;
    std::size_t execution_references{};
    std::size_t dependency_pins{};
  };

  struct ColdExecution {
    kv_cache::ExecutionId execution{};
    kv_cache::CheckpointId checkpoint{};
  };

 public:  Impl(kv_cache::PoolConfig config, CacheStorageFactory storage_factory,
       CacheEventSink events)
      : config_(config),
        page_offsets_count_(page_offsets_count(config_)),
        page_offsets_index_bytes_(page_offsets_index_bytes(config_)),
        cold_index_bytes_(cold_index_bytes(config_)),
        cold_page_capacity_(cold_page_capacity(config_)),
        cold_checkpoint_capacity_(cold_checkpoint_capacity(config_)),
        cold_execution_capacity_(cold_execution_capacity(config_)),
        cold_page_id_slots_per_checkpoint_(maximum_global_page_count(config_)),
        prefix_index_(prefix_index_bytes(config_)),
        demand_slot_capacity_(demand_slot_capacity(config_)),
        ledger_(ledger_config(config_)),
        physical_(storage_factory(config_, ledger_, page_offsets_count_)),
        events_(std::move(events)),
        cold_pages_(cold_page_capacity_ == 0
                        ? nullptr
                        : std::make_unique<ColdPage[]>(cold_page_capacity_)),
        cold_checkpoints_(cold_checkpoint_capacity_ == 0
                              ? nullptr
                              : std::make_unique<ColdCheckpoint[]>(
                                    cold_checkpoint_capacity_)),
        cold_page_ids_(cold_page_id_count(config_) == 0
                           ? nullptr
                           : std::make_unique<kv_cache::PageId[]>(
                                 cold_page_id_count(config_))),
        cold_executions_(cold_execution_capacity_ == 0
                             ? nullptr
                             : std::make_unique<ColdExecution[]>(cold_execution_capacity_)) {
    if (demand_slot_capacity_ == 0) {
      fail("persistent KV cache", "index budget cannot hold an owner demand");
    }
    // Reserve the bounded storage once; scans visit only populated demands.
    demand_slots_.reserve(demand_slot_capacity_);
    if (config_.cpu_bytes != 0 &&
        (cold_page_capacity_ == 0 || cold_checkpoint_capacity_ == 0)) {
      fail("persistent KV cache", "index budget cannot hold cold metadata");
    }
    ledger_.set_eviction_callbacks(
        [this](kv_cache::CheckpointId,
               const std::vector<kv_cache::CheckpointId>& eligible,
               kv_cache::EvictionCause cause) {
          return select_pressure_victim(eligible, cause);
        },
        [this](kv_cache::CheckpointId checkpoint) {
          return spill_checkpoint(checkpoint);
        },
        [this](kv_cache::CheckpointId checkpoint, bool externally_preserved) {
          on_hot_checkpoint_removed(checkpoint, externally_preserved);
        });
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  CacheStorage& storage() { return *physical_; }
  [[nodiscard]] const kv_cache::PoolConfig& config() const { return config_; }

  // A quarter of the configured metadata budget is reserved for bounded,
  // request-local checkpoint scheduling and endpoint bookkeeping.  Persistent
  // trie metadata receives only the remainder, so the two cannot overcommit
  // the advertised index budget.
  [[nodiscard]] std::size_t active_checkpoint_capacity(bool named) const {
    const std::size_t fixed = active_checkpoint_fixed_bytes(named);
    const std::size_t budget = active_checkpoint_index_bytes(config_);
    if (budget < fixed) {
      return 0;
    }
    return (budget - fixed) / active_checkpoint_entry_bytes(named);
  }

  [[nodiscard]] kv_cache::CacheStats stats() const {
    kv_cache::CacheStats result = ledger_.stats();
    const prefix_index::MetadataStats index_stats = prefix_index_.stats();
    result.index_bytes = config_.index_bytes;
    result.index_used += index_stats.used_bytes +
                         active_demand_count() * sizeof(DemandSlot) +
                         page_offsets_index_bytes_ + cold_metadata_used();
    result.checkpoint_count += cold_checkpoint_count_;
    result.page_count += cold_page_count_;
    return result;
  }

  [[nodiscard]] prefix_index::MetadataStats prefix_stats() const {
    return prefix_index_.stats();
  }

  // The scheduler owns this state. Build a value snapshot at a stable boundary
  // and pass that value to the network thread; inspection never borrows KV.
  [[nodiscard]] nlohmann::json observability_snapshot(
      bool include_entries = true) const {
    using json = nlohmann::json;
    const auto totals = stats();
    const auto hot = ledger_.stats();
    const auto prefix = prefix_index_.stats();
    auto gpu_memory = ledger_.gpu_memory_stats();
    if (reservation_execution_) {
      const auto remaining = remaining_reservation(reservation_execution_);
      gpu_memory.reserved_growth_bytes = remaining.gpu_bytes - remaining.speculative_bytes;
    }
    kv_cache::PoolMemoryStats cpu_memory;
    std::map<kv_cache::PageId, const ColdPage*> cpu_pages;
    std::map<kv_cache::PageId, std::size_t> idle_cpu_references;
    for (std::size_t i = 0; i < cold_page_count_; ++i) {
      const auto& page = cold_pages_[i];
      cpu_pages.emplace(page.source_page, &page);
      cpu_memory.global_page_bytes += page.storage.bytes;
      if (page.references > 1) cpu_memory.shared_page_bytes += page.storage.bytes;
    }
    for (std::size_t i = 0; i < cold_checkpoint_count_; ++i) {
      const auto& checkpoint = cold_checkpoints_[i];
      const auto bytes = checkpoint.local_snapshot.bytes + checkpoint.terminal_hidden.bytes;
      cpu_memory.checkpoint_state_bytes += bytes;
      if (checkpoint.execution_references || checkpoint.dependency_pins ||
          checkpoint.checkpoint == cold_reservation_checkpoint_) continue;
      cpu_memory.reclaimable_bytes += bytes;
      const auto* pages = cold_page_ids_at(i);
      for (std::size_t p = 0; p < checkpoint.source_page_count; ++p)
        ++idle_cpu_references[pages[p]];
    }
    for (const auto& item : cpu_pages) {
      const auto found = idle_cpu_references.find(item.first);
      if (!item.second->pending_copy && found != idle_cpu_references.end() &&
          found->second == item.second->references)
        cpu_memory.reclaimable_bytes += item.second->storage.bytes;
    }
    const auto cpu_accounted = cpu_memory.global_page_bytes + cpu_memory.checkpoint_state_bytes;
    if (cpu_accounted > totals.cpu.used || cpu_memory.reclaimable_bytes > totals.cpu.used)
      fail("cache snapshot", "CPU accounting exceeds pool usage");
    cpu_memory.other_bytes = totals.cpu.used - cpu_accounted;
    cpu_memory.nonreclaimable_bytes = totals.cpu.used - cpu_memory.reclaimable_bytes;
    cpu_memory.page_slack_bytes = cpu_page_slack_bytes();

    const auto pool_json = [](const kv_cache::PoolStats& pool,
                              const kv_cache::PoolMemoryStats& memory,
                              std::size_t pages, std::size_t checkpoints) {
      return json{{"capacity_bytes", pool.capacity}, {"used_bytes", pool.used},
          {"free_bytes", pool.free}, {"peak_used_bytes", pool.peak_used},
          {"reclaimable_bytes", memory.reclaimable_bytes},
          {"nonreclaimable_bytes", memory.nonreclaimable_bytes},
          {"reserved_growth_bytes", memory.reserved_growth_bytes},
          {"page_slack_bytes", memory.page_slack_bytes},
          {"global_page_bytes", memory.global_page_bytes},
          {"checkpoint_state_bytes", memory.checkpoint_state_bytes},
          {"execution_buffer_bytes", memory.execution_buffer_bytes},
          {"pending_capture_bytes", memory.pending_capture_bytes},
          {"other_bytes", memory.other_bytes},
          {"shared_page_bytes", memory.shared_page_bytes},
          {"page_count", pages}, {"checkpoint_count", checkpoints}};
    };
    const auto counters = telemetry();
    std::size_t admissions = 0, hits = 0, removals = 0;
    for (const auto& bucket : counters.checkpoint_buckets) {
      add_telemetry(&admissions, bucket.admissions, "checkpoint admissions");
      add_telemetry(&hits, bucket.hits, "checkpoint hits");
      add_telemetry(&removals, bucket.removals, "checkpoint removals");
    }
    json result{{"gpu", pool_json(totals.gpu, gpu_memory, hot.page_count, hot.checkpoint_count)},
        {"cpu", pool_json(totals.cpu, cpu_memory, cold_page_count_, cold_checkpoint_count_)},
        {"index", {{"capacity_bytes", totals.index_bytes}, {"used_bytes", totals.index_used},
            {"prefix_capacity_bytes", prefix.capacity_bytes}, {"prefix_used_bytes", prefix.used_bytes},
            {"node_count", prefix.node_count}, {"checkpoint_count", prefix.checkpoint_count},
            {"unpromoted_periodic_count", prefix.unpromoted_periodic_count}}},
        {"page_count", totals.page_count}, {"checkpoint_count", totals.checkpoint_count},
        {"execution_count", totals.execution_count}, {"copy_on_write_pages", totals.copy_on_write_pages},
        {"telemetry", {{"gpu_reclaimed_bytes", counters.gpu_reclaimed_bytes},
            {"cpu_reclaimed_bytes", counters.cpu_reclaimed_bytes},
            {"cold_spill_bytes", counters.cold_spill_bytes},
            {"cold_restore_bytes", counters.cold_restore_bytes},
            {"cold_spill_count", counters.cold_spill_count},
            {"cold_restore_count", counters.cold_restore_count},
            {"cold_spill_avoided_rewrite_bytes", counters.cold_spill_avoided_rewrite_bytes},
            {"cold_spill_seconds", counters.cold_spill_wall_milliseconds / 1000.0},
            {"cold_restore_seconds", counters.cold_restore_wall_milliseconds / 1000.0},
            {"checkpoint_admissions", admissions}, {"checkpoint_hits", hits},
            {"checkpoint_removals", removals}}}};
    if (!include_entries) return result;
    result["telemetry"]["checkpoint_buckets"] = json::array();
    for (std::size_t source = 0; source < kLifecycleSourceMaskCount; ++source) {
      for (std::size_t priority = 0; priority < kLifecyclePriorityCount; ++priority) {
        for (std::size_t cache_class = 0; cache_class < kLifecycleClassCount;
             ++cache_class) {
          const std::size_t index =
              (source * kLifecyclePriorityCount + priority) *
                  kLifecycleClassCount +
              cache_class;
          const CheckpointLifecycleBucket& bucket =
              counters.checkpoint_buckets[index];
          if (bucket.empty()) continue;
          std::ostringstream source_name;
          write_checkpoint_sources(
              source_name,
              static_cast<prefix_index::CheckpointSources>(source));
          result["telemetry"]["checkpoint_buckets"].push_back(
              {{"sources", source_name.str()},
               {"priority", retention_priority_name(
                                static_cast<prefix_index::RetentionPriority>(
                                    priority))},
               {"cache_class", cache_class_name(
                                   static_cast<prefix_index::CacheClass>(
                                       cache_class))},
               {"admissions", bucket.admissions},
               {"hits", bucket.hits},
               {"removals", bucket.removals},
               {"admitted_bytes", bucket.admitted_bytes},
               {"reclaimed_gpu_bytes", bucket.reclaimed_gpu_bytes},
               {"reclaimed_cpu_bytes", bucket.reclaimed_cpu_bytes}});
        }
      }
    }
    result["checkpoints"] = json::array();
    result["executions"] = json::array();
    std::map<kv_cache::PageId, const kv_cache::PageInfo*> gpu_pages;
    ledger_.visit_pages([&](const auto& page) { gpu_pages.emplace(page.id, &page); });
    const auto checkpoint_json = [&](kv_cache::CheckpointId id, std::uint32_t tokens,
        const char* tier, std::uint32_t local_start, std::uint32_t local_tokens,
        std::size_t references, std::size_t pins, std::size_t page_count,
        std::size_t referenced, std::size_t reclaimable) {
      const auto metadata = prefix_index_.metadata(id);
      if (!metadata) fail("cache snapshot", "resident checkpoint has no prefix metadata");
      json sources = json::array();
      for (const auto& source : std::array<std::pair<prefix_index::CheckpointSource, const char*>, 4>{{
          {prefix_index::CheckpointSource::input_endpoint, "input_endpoint"},
          {prefix_index::CheckpointSource::continuation_endpoint, "continuation_endpoint"},
          {prefix_index::CheckpointSource::learned_branch, "learned_branch"},
          {prefix_index::CheckpointSource::periodic, "periodic"}}}) {
        if (prefix_index::has_checkpoint_source(metadata->sources, source.first))
          sources.push_back(source.second);
      }
      const auto ancestor = prefix_index_.nearest_ancestor(id);
      if (ancestor.has_value() && ancestor->prefix_tokens >= tokens)
        fail("cache snapshot", "checkpoint ancestor is not a strict prefix");
      return json{{"id", std::to_string(id)}, {"tokens", tokens}, {"tier", tier},
          {"local_start", local_start}, {"local_tokens", local_tokens},
          {"sources", std::move(sources)}, {"priority", retention_priority_name(metadata->priority)},
          {"cache_class", cache_class_name(metadata->cache_class)},
          {"reference_bit", metadata->reference_bit},
          {"admission_order", metadata->admission_order},
          {"ancestor_checkpoint_id", ancestor.has_value()
              ? json(std::to_string(ancestor->checkpoint)) : json(nullptr)},
          {"distance_from_ancestor_tokens", ancestor.has_value()
              ? json(tokens - ancestor->prefix_tokens) : json(nullptr)},
          {"reuse_count", metadata->reuse_count}, {"execution_references", references},
          {"dependency_pins", pins}, {"global_page_count", page_count},
          {"referenced_bytes", referenced}, {"reclaimable_bytes", reclaimable}};
    };
    ledger_.visit_checkpoints([&](const auto& checkpoint) {
      std::size_t referenced = checkpoint.local_snapshot.bytes + checkpoint.terminal_hidden.bytes;
      std::size_t reclaimable = referenced;
      for (const auto id : checkpoint.global_pages) {
        const auto& page = *gpu_pages.at(id);
        referenced += page.storage.bytes;
        if (page.references == 1) reclaimable += page.storage.bytes;
      }
      if (checkpoint.execution_references || checkpoint.dependency_pins) reclaimable = 0;
      result["checkpoints"].push_back(checkpoint_json(checkpoint.id, checkpoint.processed_tokens,
          "gpu", checkpoint.local_start, checkpoint.local_tokens, checkpoint.execution_references,
          checkpoint.dependency_pins, checkpoint.global_pages.size(), referenced, reclaimable));
    });
    for (std::size_t i = 0; i < cold_checkpoint_count_; ++i) {
      const auto& checkpoint = cold_checkpoints_[i];
      std::size_t referenced = checkpoint.local_snapshot.bytes + checkpoint.terminal_hidden.bytes;
      std::size_t reclaimable = referenced;
      const auto* pages = cold_page_ids_at(i);
      for (std::size_t p = 0; p < checkpoint.source_page_count; ++p) {
        const auto& page = *cpu_pages.at(pages[p]);
        referenced += page.storage.bytes;
        if (page.references == 1 && !page.pending_copy) reclaimable += page.storage.bytes;
      }
      if (checkpoint.execution_references || checkpoint.dependency_pins ||
          checkpoint.checkpoint == cold_reservation_checkpoint_) reclaimable = 0;
      result["checkpoints"].push_back(checkpoint_json(checkpoint.checkpoint, checkpoint.processed_tokens,
          "cpu", checkpoint.local_start, checkpoint.local_tokens, checkpoint.execution_references,
          checkpoint.dependency_pins, checkpoint.source_page_count, referenced, reclaimable));
    }
    ledger_.visit_executions([&](const auto& execution) {
      const auto source = execution.source_checkpoint
          ? execution.source_checkpoint : cold_execution_source(execution.id).value_or(0);
      const auto buffers = execution.local_ring.bytes + execution.page_table.bytes;
      const auto pending = execution.pending_checkpoint
          ? execution.pending_checkpoint->local_snapshot.bytes +
            execution.pending_checkpoint->terminal_hidden.bytes : 0;
      std::size_t referenced = buffers + pending;
      for (const auto id : execution.global_pages) referenced += gpu_pages.at(id)->storage.bytes;
      result["executions"].push_back({{"id", std::to_string(execution.id)},
          {"tokens", execution.processed_tokens}, {"tier", "gpu"},
          {"source_checkpoint_id", source ? json(std::to_string(source)) : json(nullptr)},
          {"reserved_tokens", execution.batch_max_processed_tokens
              ? execution.batch_max_processed_tokens : reservation_max_processed_tokens_},
          {"global_page_count", execution.global_pages.size()},
          {"referenced_bytes", referenced}, {"execution_buffer_bytes", buffers},
          {"pending_capture_bytes", pending}});
    });
    return result;
  }

  [[nodiscard]] CacheTelemetry telemetry() const {
    CacheTelemetry result;
    result.checkpoint_buckets = checkpoint_buckets_;
    result.gpu_reclaimed_bytes = gpu_reclaimed_bytes_;
    result.cpu_reclaimed_bytes = cpu_reclaimed_bytes_;
    result.cold_spill_bytes = cold_spill_bytes_;
    result.cold_restore_bytes = cold_restore_bytes_;
    result.cold_spill_count = cold_spill_count_;
    result.cold_restore_count = cold_restore_count_;
    result.cold_spill_avoided_rewrite_bytes =
        cold_spill_avoided_rewrite_bytes_;
    result.cold_spill_wall_milliseconds = cold_spill_wall_milliseconds_;
    result.cold_restore_wall_milliseconds =
        cold_restore_wall_milliseconds_;
    return result;
  }

  [[nodiscard]] bool is_cold_checkpoint(
      kv_cache::CheckpointId checkpoint) const {
    return find_cold(checkpoint) != nullptr;
  }

  [[nodiscard]] std::size_t gpu_page_slack_bytes() const {
    return ledger_.global_page_slack_bytes();
  }

  [[nodiscard]] std::size_t cpu_page_slack_bytes() const {
    if (config_.global_page_tokens == 0 ||
        config_.global_page_bytes % config_.global_page_tokens != 0) {
      fail("persistent KV cold tier", "global page layout is not token divisible");
    }
    const std::size_t bytes_per_token =
        config_.global_page_bytes / config_.global_page_tokens;
    std::size_t slack = 0;
    for (std::size_t index = 0; index < cold_page_count_; ++index) {
      const ColdPage& page = cold_pages_[index];
      if (page.valid_tokens > config_.global_page_tokens) {
        fail("persistent KV cold tier", "cold page token count is invalid");
      }
      const std::size_t unused =
          config_.global_page_tokens - page.valid_tokens;
      if (unused > std::numeric_limits<std::size_t>::max() / bytes_per_token ||
          unused * bytes_per_token >
              std::numeric_limits<std::size_t>::max() - slack) {
        fail("persistent KV cold tier", "cold page slack overflows size_t");
      }
      slack += unused * bytes_per_token;
    }
    return slack;
  }

  [[nodiscard]] prefix_index::LookupResult find_longest(
      const std::vector<std::uint32_t>& tokens,
      const std::vector<prefix_index::ImageSpan>& images) const {
    const prefix_index::LookupResult any = prefix_index_.find_longest(
        tokens, [this](prefix_index::CheckpointId checkpoint) {
          return has_checkpoint(checkpoint);
        }, images);
    const prefix_index::LookupResult gpu = prefix_index_.find_longest(
        tokens, [this](prefix_index::CheckpointId checkpoint) {
          return ledger_.has_checkpoint(checkpoint);
        }, images);
    return gpu.has_checkpoint() &&
                   gpu.checkpoint_tokens == any.checkpoint_tokens
               ? gpu
               : any;
  }

  [[nodiscard]] bool checkpoint_matches(
      kv_cache::CheckpointId checkpoint,
      const std::vector<std::uint32_t>& tokens,
      const std::vector<prefix_index::ImageSpan>& images) const {
    if (!has_checkpoint(checkpoint) ||
        tokens.size() != checkpoint_tokens(checkpoint)) {
      return false;
    }
    const prefix_index::LookupResult result = prefix_index_.find_longest(
        tokens, [checkpoint](prefix_index::CheckpointId candidate) {
          return candidate == checkpoint;
        }, images);
    return result.checkpoint == checkpoint &&
           result.checkpoint_tokens == checkpoint_tokens(checkpoint);
  }

  [[nodiscard]] prefix_index::LookupResult find_batch_prefix(
      const std::vector<std::uint32_t>& tokens,
      std::size_t maximum_processed_tokens,
      const std::vector<prefix_index::ImageSpan>& images) const {
    const auto gpu = prefix_index_.find_longest(tokens,
        [this, maximum_processed_tokens](prefix_index::CheckpointId checkpoint) {
          return ledger_.has_checkpoint(checkpoint) &&
                 ledger_.batch_request_fits(checkpoint, maximum_processed_tokens);
        }, images);
    const auto cold = prefix_index_.find_longest(tokens,
        [this, maximum_processed_tokens](prefix_index::CheckpointId checkpoint) {
          const auto* source = find_cold(checkpoint);
          return source && source->processed_tokens <= maximum_processed_tokens &&
                 ledger_.batch_request_fits(0, maximum_processed_tokens);
        }, images);
    return cold.checkpoint_tokens > gpu.checkpoint_tokens ? cold : gpu;
  }

  [[nodiscard]] bool has_checkpoint(kv_cache::CheckpointId checkpoint) const {
    return ledger_.has_checkpoint(checkpoint) || find_cold(checkpoint) != nullptr;
  }

  [[nodiscard]] std::uint32_t checkpoint_tokens(
      kv_cache::CheckpointId checkpoint) const {
    if (ledger_.has_checkpoint(checkpoint)) {
      return ledger_.checkpoint(checkpoint).processed_tokens;
    }
    const ColdCheckpoint* const cold = find_cold(checkpoint);
    if (cold == nullptr) {
      fail("persistent KV cache", "checkpoint is not resident");
    }
    return cold->processed_tokens;
  }

  void mark_used(kv_cache::CheckpointId checkpoint) {
    if (prefix_index_.mark_used(checkpoint)) {
      const std::optional<prefix_index::CheckpointMetadata> metadata =
          prefix_index_.metadata(checkpoint);
      if (!metadata.has_value()) {
        fail("server cache telemetry", "used checkpoint lost its metadata");
      }
      CheckpointLifecycleBucket& bucket = checkpoint_buckets_[
          lifecycle_bucket_index(metadata->sources, metadata->priority,
                                 metadata->cache_class)];
      add_telemetry(&bucket.hits, 1, "checkpoint hit count");
      emit_checkpoint_event("hit", checkpoint, metadata,
                            ledger_.has_checkpoint(checkpoint) ? "gpu" : "cpu",
                            "lookup", 0, 0);
    }
  }

  void add_checkpoint_source(kv_cache::CheckpointId checkpoint,
                             const std::vector<std::uint32_t>& tokens,
                             prefix_index::CheckpointSource source,
                             const std::vector<prefix_index::ImageSpan>& images) {
    if (checkpoint == 0 || !has_checkpoint(checkpoint)) {
      return;
    }
    const std::optional<prefix_index::CheckpointMetadata> before =
        prefix_index_.metadata(checkpoint);
    const prefix_index::AdmissionResult admission =
        prefix_index_.admit(checkpoint, tokens, source,
            [this](auto id) { return checkpoint_idle(id); }, images);
    if (!admission.retained()) {
      return;
    }
    if (admission.status == prefix_index::AdmissionStatus::admitted) {
      record_admission(checkpoint);
    } else {
      const std::optional<prefix_index::CheckpointMetadata> after =
          prefix_index_.metadata(checkpoint);
      if (before.has_value() && after.has_value() &&
          before->sources != after->sources) {
        emit_checkpoint_event("update", checkpoint, after,
                              ledger_.has_checkpoint(checkpoint) ? "gpu" : "cpu",
                              "source_merge", 0, 0);
      }
    }
    for (const prefix_index::ThinnedCheckpoint& thinned :
         admission.thinned_periodic) {
      if (ledger_.has_checkpoint(thinned.checkpoint)) {
        release_hot_checkpoint(thinned.checkpoint, "periodic_thinning",
                               thinned.metadata);
      } else {
        discard_cold(thinned.checkpoint, "periodic_thinning",
                     thinned.metadata);
      }
    }
  }

  [[nodiscard]] bool add_automatic_demand(
      kv_cache::CheckpointId checkpoint,
      prefix_index::RetentionPriority priority =
          prefix_index::RetentionPriority::normal) {
    return add_demand(checkpoint, {}, priority, true);
  }

  [[nodiscard]] bool add_owner_demands(
      const std::vector<kv_cache::CheckpointId>& checkpoints,
      std::string_view owner, prefix_index::RetentionPriority priority,
      OwnerDemandReservation* reservation = nullptr,
      OwnerDemandCommit* commit = nullptr) {
    if (owner.empty() || owner.size() > kMaximumCacheOwnerBytes) {
      fail("persistent KV demand", "owner demand is malformed");
    }
    if (commit != nullptr && !commit->changes.empty()) {
      fail("persistent KV demand", "owner demand commit is not empty");
    }
    if (!std::is_sorted(checkpoints.begin(), checkpoints.end()) ||
        std::adjacent_find(checkpoints.begin(), checkpoints.end()) !=
            checkpoints.end()) {
      fail("persistent KV demand", "owner demand checkpoints are not canonical");
    }
    std::size_t required_slots = 0;
    for (const kv_cache::CheckpointId checkpoint : checkpoints) {
      if (checkpoint == 0 || !has_checkpoint(checkpoint)) {
        continue;
      }
      const bool existing = std::any_of(
          demand_slots_.begin(), demand_slots_.end(),
          [checkpoint, owner](const DemandSlot& slot) {
            return !slot.automatic &&
                   slot.checkpoint == checkpoint && same_owner(slot, owner);
          });
      if (!existing) {
        ++required_slots;
      }
    }
    const std::size_t available_slots = demand_slot_capacity_ -
        active_demand_count();
    if ((reservation != nullptr && required_slots > reservation->slots) ||
        (reservation == nullptr && required_slots > available_slots)) {
      return false;
    }
    OwnerDemandCommit local_commit;
    OwnerDemandCommit* const active_commit =
        commit == nullptr ? &local_commit : commit;
    if (commit != nullptr) {
      if (active_commit->changes.capacity() < checkpoints.size()) {
        fail("persistent KV demand",
             "owner demand commit was not pre-reserved");
      }
    } else {
      active_commit->changes.reserve(checkpoints.size());
    }
    try {
      for (const kv_cache::CheckpointId checkpoint : checkpoints) {
        if (checkpoint == 0 || !has_checkpoint(checkpoint)) {
          continue;
        }
        const auto existing = std::find_if(
            demand_slots_.begin(), demand_slots_.end(),
            [checkpoint, owner](const DemandSlot& slot) {
              return !slot.automatic &&
                     slot.checkpoint == checkpoint && same_owner(slot, owner);
            });
        active_commit->changes.push_back(
            {checkpoint,
             existing == demand_slots_.end()
                 ? prefix_index::RetentionPriority::normal
                 : existing->priority,
             existing != demand_slots_.end()});
        if (!add_demand(checkpoint, owner, priority, false,
                        reservation != nullptr)) {
          fail("persistent KV demand",
               "preflighted owner demand allocation failed");
        }
      }
    } catch (...) {
      rollback_owner_demands(owner, active_commit);
      throw;
    }
    if (reservation != nullptr) {
      if (required_slots > reserved_owner_demand_slots_) {
        fail("persistent KV demand", "owner demand reservation underflow");
      }
      reservation->slots -= required_slots;
      reserved_owner_demand_slots_ -= required_slots;
    }
    return true;
  }

  void rollback_owner_demands(std::string_view owner,
                              OwnerDemandCommit* commit) {
    if (owner.empty() || commit == nullptr) {
      fail("persistent KV demand", "owner demand rollback is malformed");
    }
    for (const OwnerDemandCommit::Change& change : commit->changes) {
      const auto found = std::find_if(
          demand_slots_.begin(), demand_slots_.end(),
          [&change, owner](const DemandSlot& slot) {
            return !slot.automatic &&
                   slot.checkpoint == change.checkpoint &&
                   same_owner(slot, owner);
          });
      if (found == demand_slots_.end()) {
        continue;
      }
      if (change.existed) {
        found->priority = change.previous_priority;
      } else {
        *found = std::move(demand_slots_.back());
        demand_slots_.pop_back();
      }
    }
    for (const OwnerDemandCommit::Change& change : commit->changes) {
      refresh_priority(change.checkpoint);
      if (!change.existed) {
        reclaim_if_undemanded(change.checkpoint);
      }
    }
    commit->changes.clear();
  }

  [[nodiscard]] bool reserve_owner_demand_slots(
      std::size_t slots, OwnerDemandReservation* reservation) {
    if (reservation == nullptr || reservation->slots != 0) {
      fail("persistent KV demand", "owner demand reservation is malformed");
    }
    const std::size_t active = active_demand_count();
    if (active > demand_slot_capacity_ ||
        reserved_owner_demand_slots_ > demand_slot_capacity_ - active ||
        slots > demand_slot_capacity_ - active - reserved_owner_demand_slots_) {
      return false;
    }
    reservation->slots = slots;
    reserved_owner_demand_slots_ += slots;
    return true;
  }

  [[nodiscard]] std::size_t owner_demand_slot_capacity() const {
    return demand_slot_capacity_;
  }

  void release_owner_demand_reservation(OwnerDemandReservation* reservation) {
    if (reservation == nullptr) {
      return;
    }
    if (reservation->slots > reserved_owner_demand_slots_) {
      fail("persistent KV demand", "owner demand reservation underflow");
    }
    reserved_owner_demand_slots_ -= reservation->slots;
    reservation->slots = 0;
  }

  void reclaim_undemanded(kv_cache::CheckpointId checkpoint) {
    reclaim_if_undemanded(checkpoint);
  }

  void release_owner(std::string_view owner) {
    if (owner.empty()) {
      return;
    }
    // Each (owner, checkpoint) pair occupies at most one demand slot.
    // Reclaim it immediately after removing the slot, avoiding a temporary
    // vector when a finished streaming request releases its owner.
    for (std::size_t index = 0; index < demand_slots_.size();) {
      const DemandSlot& slot = demand_slots_[index];
      if (slot.automatic || !same_owner(slot, owner)) {
        ++index;
        continue;
      }
      const kv_cache::CheckpointId checkpoint = slot.checkpoint;
      demand_slots_[index] = std::move(demand_slots_.back());
      demand_slots_.pop_back();
      refresh_priority(checkpoint);
      reclaim_if_undemanded(checkpoint);
    }
  }

  // Reserves a serial execution's maximum private growth before computation.
  // The ledger's fixed allocator has no other concurrent borrower, so evicting
  // idle state until this amount is free is an actual reservation until the
  // execution releases its ring and pages. A shared partial tail needs one
  // additional copy-on-write page even though its logical page count is
  // unchanged.
  void ensure_capacity(kv_cache::CheckpointId source,
                       std::size_t total_processed_tokens,
                       std::size_t speculative_bytes = 0) {
    if (ledger_.stats().execution_count != 0) {
      fail("persistent KV cache",
           "cache reservations require serial execution");
    }
    const bool source_on_gpu = source != 0 && ledger_.has_checkpoint(source);
    const ColdCheckpoint* const cold_source =
        source == 0 ? nullptr : find_cold(source);
    if (source != 0 && !source_on_gpu && cold_source == nullptr) {
      fail("persistent KV cache", "source checkpoint is not resident");
    }
    const std::uint32_t resumed =
        source == 0 ? 0 : source_on_gpu
                            ? ledger_.checkpoint(source).processed_tokens
                            : cold_source->processed_tokens;
    const std::size_t page_tokens = config_.global_page_tokens;
    const std::size_t source_pages =
        (static_cast<std::size_t>(resumed) + page_tokens - 1) / page_tokens;
    const std::size_t total_pages =
        (total_processed_tokens + page_tokens - 1) / page_tokens;
    const std::size_t execution_page_capacity =
        (static_cast<std::size_t>(config_.maximum_context_tokens) +
         page_tokens - 1) /
        page_tokens;
    const std::size_t new_pages =
        cold_source != nullptr
            ? total_pages
            : total_pages > source_pages ? total_pages - source_pages : 0;
    const std::size_t copy_on_write_pages =
        source_on_gpu && resumed % page_tokens != 0 &&
                total_processed_tokens > resumed
            ? 1
            : 0;
    if (new_pages > std::numeric_limits<std::size_t>::max() -
                        copy_on_write_pages) {
      fail_invalid_capacity("persistent KV cache",
                            "request page count overflows size_t");
    }
    const std::size_t required_page_allocations =
        new_pages + copy_on_write_pages;
    const std::size_t target_bytes =
        config_.local_ring_bytes + config_.page_table_bytes +
        required_page_allocations * config_.global_page_bytes;
    if (speculative_bytes >
        std::numeric_limits<std::size_t>::max() - target_bytes) {
      fail_invalid_capacity("persistent KV cache",
                            "speculative working set overflows size_t");
    }
    const std::size_t required_bytes = target_bytes + speculative_bytes;
    const std::size_t required_index_bytes =
        sizeof(kv_cache::ExecutionInfo) +
        execution_page_capacity * sizeof(kv_cache::PageId) +
        required_page_allocations * sizeof(kv_cache::PageInfo);
    if (required_bytes >
        config_.gpu_bytes) {
      fail_invalid_capacity(
          "persistent KV cache",
          "request working set exceeds the configured GPU KV budget");
    }
    if (required_index_bytes > ledger_.config().index_bytes) {
      fail_invalid_capacity(
          "persistent KV cache",
          "request working set exceeds the configured KV metadata budget");
    }
    std::vector<std::size_t> allocation_plan;
    allocation_plan.reserve(required_page_allocations + 3);
    allocation_plan.push_back(config_.local_ring_bytes);
    allocation_plan.push_back(config_.page_table_bytes);
    // begin() restores cold pages first. Staging is acquired immediately
    // afterwards, before any new target writes, so preserve this exact order
    // when proving the plan fits a fragmented byte pool.
    const std::size_t restore_pages = cold_source != nullptr ? source_pages : 0;
    if (restore_pages > required_page_allocations) {
      fail_invalid_capacity("persistent KV cache",
                            "request ends before its source checkpoint");
    }
    allocation_plan.insert(allocation_plan.end(), restore_pages,
                           config_.global_page_bytes);
    if (speculative_bytes != 0) {
      allocation_plan.push_back(speculative_bytes);
    }
    allocation_plan.insert(allocation_plan.end(),
                           required_page_allocations - restore_pages,
                           config_.global_page_bytes);
    cold_reservation_checkpoint_ = cold_source == nullptr ? 0 : source;
    try {
      const auto pressure_cause = [&]() -> std::optional<kv_cache::EvictionCause> {
        const kv_cache::CacheStats stats = ledger_.stats();
        if (stats.index_used > stats.index_bytes ||
            required_index_bytes > stats.index_bytes - stats.index_used) {
          return kv_cache::EvictionCause::index_pressure;
        }
        if (stats.gpu.free < required_bytes ||
            !ledger_.gpu_pool().can_allocate_sequence(allocation_plan)) {
          return kv_cache::EvictionCause::gpu_pressure;
        }
        return std::nullopt;
      };
      for (;;) {
        const auto cause = pressure_cause();
        if (!cause.has_value()) {
          break;
        }
        if (!ledger_.evict_idle_checkpoint(source, *cause)) {
          fail_cache_capacity(
              "persistent KV cache",
              "working-set reservation cannot displace active cache state");
        }
      }
    } catch (...) {
      cold_reservation_checkpoint_ = 0;
      reservation_max_processed_tokens_ = 0;
      pending_capture_tail_page_ = 0;
      throw;
    }
    execution_reservation_bytes_ = required_bytes;
    speculative_reservation_bytes_ = speculative_bytes;
    execution_reservation_index_bytes_ = required_index_bytes;
    reservation_max_processed_tokens_ =
        static_cast<std::uint32_t>(total_processed_tokens);
    pending_capture_tail_page_ = 0;
    reservation_execution_ = 0;
  }

  // Each borrower shares the source's global pages and restores a private
  // local ring. The ledger reserves collective growth before any GPU work.
  [[nodiscard]] std::optional<kv_cache::ExecutionId> try_begin_batch(
      kv_cache::CheckpointId source, std::size_t max_processed_tokens,
      CompletionContext stream) {
    if (execution_reservation_bytes_ != 0) {
      fail("batch KV admission", "serial serving has an active reservation");
    }
    const bool cold = source != 0 && !ledger_.has_checkpoint(source) && find_cold(source);
    if (cold && checkpoint_tokens(source) > max_processed_tokens)
      fail("batch KV admission", "processed-token limit precedes the cold checkpoint");
    // Admission may spill other GPU checkpoints and evict CPU state. Pin the
    // chosen cold source before any such pressure, then hand off to an execution
    // reference only after its bounded restore completes.
    if (cold) pin_batch_checkpoint(source);
    std::optional<kv_cache::ExecutionId> execution;
    try {
      execution = ledger_.try_begin_batch(cold ? 0 : source, max_processed_tokens);
      if (!execution) {
        if (cold) unpin_batch_checkpoint(source);
        return std::nullopt;
      }
      if (cold) {
        restore_cold(*execution, source, stream);
        add_cold_execution(*execution, source);
        unpin_batch_checkpoint(source);
      } else if (source != 0) {
        const auto& info = ledger_.execution(*execution);
        restore_local(info, ledger_.checkpoint(source), stream);
        physical_->upload_page_table(info, stream);
      }
    } catch (...) {
      physical_->wait(stream);
      if (execution) {
        ledger_.release_execution(*execution);
        release_cold_execution(*execution);
      }
      if (cold) unpin_batch_checkpoint(source);
      throw;
    }
    // Empty executions upload their table on the first prepare_write.
    return execution;
  }

  std::optional<kv_cache::ExecutionId> try_fork_batch(
      kv_cache::ExecutionId source, std::size_t horizon, CompletionContext stream) {
    const auto child = ledger_.try_fork_batch(source, horizon);
    if (!child) return std::nullopt;
    try {
      const auto& from = ledger_.execution(source);
      const auto& to = ledger_.execution(*child);
      physical_->fork_local(from, to, stream);
      physical_->upload_page_table(to, stream);
      if (const auto cold = cold_execution_source(source)) add_cold_execution(*child, *cold);
    } catch (...) {
      physical_->wait(stream);
      ledger_.release_execution(*child);
      release_cold_execution(*child);
      throw;
    }
    return child;
  }

  bool try_resize_batch(kv_cache::ExecutionId execution, std::size_t horizon) {
    kv_cache::EvictionCause cause = kv_cache::EvictionCause::gpu_pressure;
    while (!ledger_.try_resize_batch(execution, horizon, &cause))
      if (!ledger_.evict_idle_checkpoint(
              ledger_.execution(execution).source_checkpoint, cause))
        return false;
    return true;
  }

  void pin_batch_checkpoint(kv_cache::CheckpointId checkpoint) {
    if (ledger_.has_checkpoint(checkpoint)) ledger_.pin_checkpoint(checkpoint);
    else {
      auto* cold = find_cold(checkpoint);
      if (!cold) fail("persistent KV pin", "checkpoint is not resident");
      ++cold->dependency_pins;
    }
  }
  void unpin_batch_checkpoint(kv_cache::CheckpointId checkpoint) {
    if (ledger_.has_checkpoint(checkpoint)) ledger_.unpin_checkpoint(checkpoint);
    else {
      auto* cold = find_cold(checkpoint);
      if (!cold || !cold->dependency_pins) fail("persistent KV pin", "cold pin underflow");
      --cold->dependency_pins;
    }
    reclaim_if_undemanded(checkpoint);
  }

  kv_cache::ExecutionId begin(kv_cache::CheckpointId source = 0) {
    kv_cache::ExecutionId execution = 0;
    bool cold_execution = false;
    try {
      const bool source_is_cold = source != 0 && find_cold(source) != nullptr;
      execution = ledger_.begin_execution(source_is_cold ? 0 : source);
      reservation_execution_ = execution;
      const kv_cache::ExecutionInfo& info = ledger_.execution(execution);
      physical_->clear_page_table(info);
      if (source_is_cold) {
        restore_cold(execution, source, nullptr);
        add_cold_execution(execution, source);
        cold_execution = true;
      } else if (source != 0) {
        restore_local(info, ledger_.checkpoint(source));
        physical_->upload_page_table(info, nullptr);
      }
      cold_reservation_checkpoint_ = 0;
    } catch (...) {
      physical_->wait(nullptr);
      if (execution != 0) {
        release_speculative_staging(execution);
        ledger_.release_execution(execution);
      }
      if (cold_execution) release_cold_execution(execution);
      clear_reservation();
      throw;
    }
    return execution;
  }

  [[nodiscard]] kv_cache::Allocation acquire_speculative_staging(
      kv_cache::ExecutionId execution, std::size_t bytes) {
    static_cast<void>(ledger_.execution(execution));
    if (reservation_execution_ != execution ||
        bytes != speculative_reservation_bytes_ ||
        speculative_staging_.valid()) {
      fail("persistent speculative KV",
           "staging allocation does not match the execution reservation");
    }
    if (bytes == 0) {
      return {};
    }
    speculative_staging_ = ledger_.try_allocate_gpu(bytes);
    if (!speculative_staging_.valid()) {
      fail("persistent speculative KV", "reserved staging allocation failed");
    }
    return speculative_staging_;
  }

  void release_speculative_staging(kv_cache::ExecutionId execution) {
    if (reservation_execution_ != execution) {
      fail("persistent speculative KV", "staging belongs to another execution");
    }
    if (speculative_staging_.valid()) {
      ledger_.release_gpu(speculative_staging_);
      speculative_staging_ = {};
    }
  }

  void release(kv_cache::ExecutionId execution) {
    const auto& info = ledger_.execution(execution);
    const bool batch = info.batch_max_processed_tokens != 0;
    const auto hot_source = info.source_checkpoint;
    if (!batch) release_speculative_staging(execution);
    ledger_.release_execution(execution);
    release_cold_execution(execution);
    if (hot_source != 0) reclaim_if_undemanded(hot_source);
    if (!batch) clear_reservation();
  }

  [[nodiscard]] std::uint32_t processed_tokens(
      kv_cache::ExecutionId execution) const {
    return ledger_.execution(execution).processed_tokens;
  }

  void restore_terminal_hidden(kv_cache::ExecutionId execution,
                               TerminalState destination,
                               CompletionContext stream) const {
    if (!destination) {
      fail("persistent KV restore", "terminal destination is null");
    }
    const kv_cache::ExecutionInfo& info = ledger_.execution(execution);
    if (info.source_checkpoint != 0) {
      physical_->restore_terminal(
          ledger_.checkpoint(info.source_checkpoint).terminal_hidden, destination, stream);
      return;
    }
    const std::optional<kv_cache::CheckpointId> cold_source =
        cold_execution_source(execution);
    if (!cold_source.has_value()) {
      fail("persistent KV restore", "complete hit has no checkpoint source");
    }
    const ColdCheckpoint* const cold = find_cold(*cold_source);
    if (cold == nullptr) {
      fail("persistent KV restore", "cold terminal hidden state is missing");
    }
    physical_->restore_terminal(cold->terminal_hidden, destination, stream);
  }

  kv_cache::WritePlan prepare_write(kv_cache::ExecutionId execution,
                                    std::uint32_t first_token,
                                    std::uint32_t token_count,
                                    CompletionContext stream) {
    kv_cache::WritePlan plan =
        ledger_.prepare_write(execution, first_token, token_count);
    if (reservation_execution_ == execution) {
      // The next contiguous write either copies this captured partial tail or
      // makes it exclusive after pressure removes its checkpoint.
      pending_capture_tail_page_ = 0;
    }
    try {
      for (const kv_cache::CopyOnWrite& copy : plan.copies) {
        const kv_cache::Allocation& source =
            ledger_.page(copy.source).storage;
        const kv_cache::Allocation& destination =
            ledger_.page(copy.destination).storage;
        physical_->copy_global_page(source, destination, stream);
      }
      if (!plan.copies.empty() || !plan.new_pages.empty()) {
        physical_->upload_page_table(ledger_.execution(execution), stream);
      }
    } catch (...) {
      // The caller cannot safely continue this execution after a failed device
      // copy. The ledger still owns the private pages and will release them
      // with the execution; retained checkpoints remain untouched.
      throw;
    }
    return plan;
  }

  kv_cache::CheckpointId try_capture(
      kv_cache::ExecutionId execution, const std::vector<std::uint32_t>& tokens,
      TerminalState terminal, CompletionContext stream,
      prefix_index::CheckpointSource source =
          prefix_index::CheckpointSource::input_endpoint,
      bool automatic_demand = true,
      prefix_index::RetentionPriority retention_priority =
          prefix_index::RetentionPriority::normal,
      bool may_write_after_capture = true,
      bool required_retention = false,
      const std::vector<prefix_index::ImageSpan>& images = {}) {
    const kv_cache::ExecutionInfo& info = ledger_.execution(execution);
    if (tokens.size() != info.processed_tokens || !terminal) {
      fail("persistent KV checkpoint", "endpoint state does not reach execution");
    }
    std::uint32_t previous_end = 0;
    for (const auto& image : images) {
      if (image.begin < previous_end || image.begin >= image.end ||
          (image.begin < tokens.size() && tokens.size() < image.end)) {
        fail("persistent KV checkpoint", "invalid image spans or partial image checkpoint");
      }
      previous_end = image.end;
    }
    const std::optional<kv_cache::CheckpointId> source_checkpoint =
        info.source_checkpoint != 0
            ? std::optional<kv_cache::CheckpointId>(info.source_checkpoint)
            : cold_execution_source(execution);
    if (source_checkpoint.has_value() &&
        checkpoint_matches(*source_checkpoint, tokens, images)) {
      // A complete source hit already has the exact local snapshot and hidden
      // endpoint needed by this state. Reuse it instead of requiring a second
      // full local snapshot from the fixed pool.
      add_checkpoint_source(*source_checkpoint, tokens, source, images);
      if (automatic_demand) {
        (void)add_automatic_demand(*source_checkpoint,
                                   retention_priority);
      }
      return *source_checkpoint;
    }
    const std::uint32_t local_tokens = static_cast<std::uint32_t>(std::min<
        std::size_t>(info.processed_tokens, config_.local_window_tokens));
    if (automatic_demand && !required_retention &&
        local_tokens == config_.local_window_tokens) {
      const prefix_index::LookupResult retained = prefix_index_.find_longest(
          tokens, [this](prefix_index::CheckpointId checkpoint) {
            return has_checkpoint(checkpoint);
          }, images);
      if (retained.has_checkpoint() &&
          retained.checkpoint_tokens == tokens.size()) {
        add_checkpoint_source(retained.checkpoint, tokens, source, images);
        (void)add_automatic_demand(retained.checkpoint, retention_priority);
        return retained.checkpoint;
      }
      if (retained.has_checkpoint() &&
          tokens.size() - retained.checkpoint_tokens <
              config_.local_window_tokens) {
        emit_checkpoint_rejection(
            0, tokens.size(), source, "minimum_checkpoint_spacing",
            prefix_index_.metadata(retained.checkpoint),
            ledger_.has_checkpoint(retained.checkpoint) ? "gpu" : "cpu");
        return 0;
      }
    }
    if (info.batch_max_processed_tokens != 0) {
      return try_capture_batch(execution, tokens, terminal, stream, source,
                               automatic_demand, retention_priority, images);
    }
    const std::size_t local_bytes =
        static_cast<std::size_t>(local_tokens) * config_.local_bytes_per_token;
    if (info.global_pages.size() >
        (std::numeric_limits<std::size_t>::max() -
         sizeof(kv_cache::CheckpointInfo)) /
            sizeof(kv_cache::PageId)) {
      fail("persistent KV checkpoint", "checkpoint metadata byte count overflows");
    }
    const std::size_t metadata_bytes =
        sizeof(kv_cache::CheckpointInfo) +
        info.global_pages.size() * sizeof(kv_cache::PageId);
    const bool reserve_future_writes =
        may_write_after_capture && reservation_execution_ == execution &&
        info.processed_tokens < reservation_max_processed_tokens_;
    const bool partial_tail =
        info.processed_tokens % config_.global_page_tokens != 0;
    const kv_cache::PageId tail_page =
        info.global_pages.empty() ? 0 : info.global_pages.back();
    bool base_tail_cow_reserved = false;
    if (reserve_future_writes && partial_tail &&
        info.source_checkpoint != 0) {
      const kv_cache::CheckpointInfo& source_info =
          ledger_.checkpoint(info.source_checkpoint);
      base_tail_cow_reserved =
          source_info.processed_tokens == info.processed_tokens &&
          !source_info.global_pages.empty() && tail_page != 0 &&
          source_info.global_pages.back() == tail_page;
    }
    const CaptureTailCowHeadroom tail_cow = capture_tail_cow_headroom(
        reserve_future_writes, partial_tail, tail_page,
        base_tail_cow_reserved, pending_capture_tail_page_);
    const std::size_t tail_cow_pages = tail_cow.page_count();
    if (tail_cow_pages > 1) {
      fail("persistent KV checkpoint", "capture tail COW reservation is invalid");
    }
    const std::size_t tail_cow_bytes =
        tail_cow_pages * config_.global_page_bytes;
    const std::size_t tail_cow_index_bytes =
        tail_cow_pages * sizeof(kv_cache::PageInfo);
    const ExecutionReservation remaining =
        remaining_reservation(execution, reserve_future_writes);
    if (local_bytes > std::numeric_limits<std::size_t>::max() -
                          config_.terminal_hidden_bytes) {
      fail("persistent KV checkpoint", "snapshot byte count overflows");
    }
    const std::size_t checkpoint_bytes =
        local_bytes + config_.terminal_hidden_bytes;

    // A new endpoint may replace idle state through normal CLOCK pressure, but
    // it never borrows the active execution's reserved future pages or index
    // records. If nothing evictable can create opportunistic room, retaining
    // this endpoint is optional and the request still succeeds.
    const auto capture_pressure_cause =
        [&]() -> std::optional<kv_cache::EvictionCause> {
      const kv_cache::CacheStats stats = ledger_.stats();
      if (stats.index_used > stats.index_bytes ||
          remaining.index_bytes > stats.index_bytes - stats.index_used) {
        return kv_cache::EvictionCause::index_pressure;
      }
      const std::size_t index_free =
          stats.index_bytes - stats.index_used - remaining.index_bytes;
      if (metadata_bytes > index_free ||
          tail_cow_index_bytes > index_free - metadata_bytes) {
        return kv_cache::EvictionCause::index_pressure;
      }
      if (stats.gpu.free < remaining.gpu_bytes ||
          checkpoint_bytes > stats.gpu.free - remaining.gpu_bytes ||
          tail_cow_bytes >
              stats.gpu.free - remaining.gpu_bytes - checkpoint_bytes ||
          remaining.speculative_bytes > remaining.gpu_bytes ||
          (remaining.gpu_bytes - remaining.speculative_bytes) %
                  config_.global_page_bytes !=
              0) {
        return kv_cache::EvictionCause::gpu_pressure;
      }
      const std::size_t remaining_page_allocations =
          (remaining.gpu_bytes - remaining.speculative_bytes) /
          config_.global_page_bytes;
      std::vector<std::size_t> allocation_plan;
      allocation_plan.reserve(remaining_page_allocations + tail_cow_pages + 3);
      allocation_plan.push_back(local_bytes);
      allocation_plan.push_back(config_.terminal_hidden_bytes);
      if (remaining.speculative_bytes != 0) {
        allocation_plan.push_back(remaining.speculative_bytes);
      }
      allocation_plan.insert(allocation_plan.end(), tail_cow_pages,
                             config_.global_page_bytes);
      allocation_plan.insert(allocation_plan.end(),
                             remaining_page_allocations,
                             config_.global_page_bytes);
      return ledger_.gpu_pool().can_allocate_sequence(allocation_plan)
                 ? std::nullopt
                 : std::optional<kv_cache::EvictionCause>(
                       kv_cache::EvictionCause::gpu_pressure);
    };
    for (;;) {
      const auto cause = capture_pressure_cause();
      if (!cause.has_value()) {
        break;
      }
      if (!ledger_.evict_idle_checkpoint(info.source_checkpoint, *cause)) {
        emit_checkpoint_rejection(0, tokens.size(), source,
                                  "capture_headroom");
        return 0;
      }
    }

    kv_cache::Allocation local_snapshot = ledger_.try_allocate_gpu(local_bytes);
    if (!local_snapshot.valid()) {
      emit_checkpoint_rejection(0, tokens.size(), source,
                                "local_snapshot_allocation");
      return 0;
    }
    kv_cache::Allocation terminal_hidden_allocation =
        ledger_.try_allocate_gpu(config_.terminal_hidden_bytes);
    if (!terminal_hidden_allocation.valid()) {
      ledger_.release_gpu(local_snapshot);
      emit_checkpoint_rejection(0, tokens.size(), source,
                                "terminal_snapshot_allocation");
      return 0;
    }
    kv_cache::CheckpointId published_checkpoint = 0;
    try {
      physical_->copy_local_to_snapshot(info, local_tokens, local_snapshot, stream);
      physical_->snapshot_terminal(terminal, terminal_hidden_allocation, stream,
                                  "copy persistent KV terminal hidden state");
      physical_->synchronize(stream, "synchronize persistent KV checkpoint");
      published_checkpoint = ledger_.publish_checkpoint_allocations(
              execution, local_snapshot, terminal_hidden_allocation);
      local_snapshot = {};
      terminal_hidden_allocation = {};
      if (!retain_checkpoint(published_checkpoint, tokens, source,
                             automatic_demand, retention_priority, images)) return 0;
      if (tail_cow.reserves_new_tail) {
        pending_capture_tail_page_ = tail_page;
      }
      return published_checkpoint;
    } catch (...) {
      if (published_checkpoint != 0 &&
          ledger_.has_checkpoint(published_checkpoint)) {
        release_hot_checkpoint(published_checkpoint, "capture_exception");
      }
      if (local_snapshot.valid() && ledger_.gpu_pool().owns(local_snapshot)) {
        ledger_.release_gpu(local_snapshot);
      }
      if (terminal_hidden_allocation.valid() &&
          ledger_.gpu_pool().owns(terminal_hidden_allocation)) {
        ledger_.release_gpu(terminal_hidden_allocation);
      }
      throw;
    }
  }

  [[nodiscard]] std::size_t execution_reservation_bytes() const {
    return execution_reservation_bytes_;
  }

  [[nodiscard]] std::size_t execution_reservation_index_bytes() const {
    return execution_reservation_index_bytes_;
  }

  [[nodiscard]] std::size_t cold_spill_bytes() const {
    return cold_spill_bytes_;
  }

  [[nodiscard]] std::size_t cold_restore_bytes() const {
    return cold_restore_bytes_;
  }

  [[nodiscard]] std::size_t cold_spill_count() const {
    return cold_spill_count_;
  }

  [[nodiscard]] std::size_t cold_restore_count() const {
    return cold_restore_count_;
  }

 private:
  bool retain_checkpoint(kv_cache::CheckpointId checkpoint,
                        const std::vector<std::uint32_t>& tokens,
                        prefix_index::CheckpointSource source,
                        bool automatic_demand,
                        prefix_index::RetentionPriority priority,
                        const std::vector<prefix_index::ImageSpan>& images) {
    const auto eligible = [this](auto id) { return checkpoint_idle(id); };
    auto admission = prefix_index_.admit(checkpoint, tokens, source, eligible, images);
    while (!admission.retained()) {
      if (admission.status == prefix_index::AdmissionStatus::protected_periodic ||
          !reclaim_prefix_index(checkpoint)) {
        emit_checkpoint_rejection(
            checkpoint, tokens.size(), source,
            admission.status == prefix_index::AdmissionStatus::protected_periodic
                ? "protected_periodic"
                : "index_capacity");
        release_hot_checkpoint(checkpoint, "admission_rejected");
        return false;
      }
      admission = prefix_index_.admit(checkpoint, tokens, source, eligible, images);
    }
    for (const auto& thinned : admission.thinned_periodic) {
      if (ledger_.has_checkpoint(thinned.checkpoint))
        release_hot_checkpoint(thinned.checkpoint, "periodic_thinning",
                               thinned.metadata);
      else
        discard_cold(thinned.checkpoint, "periodic_thinning",
                     thinned.metadata);
    }
    if (!prefix_index_.set_priority(checkpoint, priority))
      fail("persistent KV checkpoint", "published checkpoint is not indexed");
    record_admission(checkpoint);
    if (automatic_demand && !add_automatic_demand(checkpoint, priority)) {
      release_hot_checkpoint(checkpoint, "automatic_demand_capacity");
      return false;
    }
    return true;
  }

  kv_cache::CheckpointId try_capture_batch(
      kv_cache::ExecutionId execution, const std::vector<std::uint32_t>& tokens,
      TerminalState terminal, CompletionContext stream,
      prefix_index::CheckpointSource source, bool automatic_demand,
      prefix_index::RetentionPriority priority,
      const std::vector<prefix_index::ImageSpan>& images) {
    const auto& info = ledger_.execution(execution);
    kv_cache::EvictionCause cause = kv_cache::EvictionCause::gpu_pressure;
    auto capture = ledger_.try_begin_checkpoint_capture(execution, &cause);
    while (!capture) {
      if (!ledger_.evict_idle_checkpoint(info.source_checkpoint, cause)) {
        emit_checkpoint_rejection(0, tokens.size(), source,
                                  "batch_capture_headroom");
        return 0;
      }
      capture = ledger_.try_begin_checkpoint_capture(execution, &cause);
    }
    kv_cache::CheckpointId checkpoint = 0;
    try {
      physical_->copy_local_to_snapshot(info,
          std::min(info.processed_tokens, config_.local_window_tokens),
          capture->local_snapshot, stream);
      physical_->snapshot_terminal(terminal, capture->terminal_hidden, stream,
                                  "copy batch checkpoint terminal hidden");
      physical_->synchronize(stream, "complete batch checkpoint");
      checkpoint = ledger_.publish_checkpoint_allocations(
          execution, capture->local_snapshot, capture->terminal_hidden);
      if (!retain_checkpoint(checkpoint, tokens, source, automatic_demand, priority, images))
        return 0;
      return checkpoint;
    } catch (...) {
      physical_->wait(stream);
      ledger_.abort_checkpoint_capture(execution);
      if (checkpoint != 0 && ledger_.has_checkpoint(checkpoint))
        release_hot_checkpoint(checkpoint, "capture_exception");
      throw;
    }
  }

  struct ExecutionReservation {
    std::size_t gpu_bytes{};
    std::size_t index_bytes{};
    std::size_t speculative_bytes{};
  };

  struct PendingHotReclaim {
    kv_cache::CheckpointId checkpoint{};
    std::optional<prefix_index::CheckpointMetadata> metadata;
    std::size_t gpu_bytes{};
    bool redundant_spacing{};
    bool index_pressure{};
  };



  [[nodiscard]] std::size_t checkpoint_admitted_bytes(
      kv_cache::CheckpointId checkpoint) const {
    if (!ledger_.has_checkpoint(checkpoint)) {
      return 0;
    }
    const kv_cache::CheckpointInfo& info = ledger_.checkpoint(checkpoint);
    std::size_t bytes = info.local_snapshot.bytes;
    const auto add = [&bytes](std::size_t value) {
      if (value > std::numeric_limits<std::size_t>::max() - bytes) {
        fail("persistent KV checkpoint", "checkpoint admission bytes overflow");
      }
      bytes += value;
    };
    add(info.terminal_hidden.bytes);
    for (const kv_cache::PageId page_id : info.global_pages) {
      const kv_cache::PageInfo& page = ledger_.page(page_id);
      // Publishing retains pages already held by the active execution. A page
      // with exactly those two references becomes physically retained here;
      // inherited pages already belong to an earlier checkpoint.
      if (page.references == 2) {
        add(page.storage.bytes);
      }
    }
    return bytes;
  }

  [[nodiscard]] std::uint64_t next_checkpoint_event_sequence() {
    if (checkpoint_event_sequence_ ==
        std::numeric_limits<std::uint64_t>::max()) {
      fail("server cache telemetry", "checkpoint event sequence overflows");
    }
    return ++checkpoint_event_sequence_;
  }

  void add_checkpoint_event_state(nlohmann::json* event) const {
    if (event == nullptr) {
      fail("server cache telemetry", "checkpoint event is null");
    }
    const kv_cache::CacheStats cache = ledger_.stats();
    const prefix_index::MetadataStats index = prefix_index_.stats();
    (*event)["gpu_used_bytes"] = cache.gpu.used;
    (*event)["gpu_free_bytes"] = cache.gpu.free;
    (*event)["resident_checkpoint_count"] = index.checkpoint_count;
    (*event)["unpromoted_periodic_count"] =
        index.unpromoted_periodic_count;
  }

  void emit_checkpoint_event(
      std::string_view action, kv_cache::CheckpointId checkpoint,
      const std::optional<prefix_index::CheckpointMetadata>& metadata,
      std::string_view tier, std::string_view reason,
      std::size_t gpu_bytes, std::size_t cpu_bytes) {
    if (!bool(events_) || !metadata.has_value()) {
      return;
    }
    const auto ancestor = prefix_index_.nearest_ancestor(checkpoint);
    nlohmann::json event{{"sequence", next_checkpoint_event_sequence()},
               {"monotonic_seconds",
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count()},
               {"action", action},
               {"checkpoint", checkpoint == 0
                                  ? nlohmann::json(nullptr)
                                  : nlohmann::json(std::to_string(checkpoint))},
               {"tokens", metadata->prefix_tokens},
               {"sources", checkpoint_sources_name(metadata->sources)},
               {"priority", retention_priority_name(metadata->priority)},
               {"cache_class", cache_class_name(metadata->cache_class)},
               {"reference_bit", metadata->reference_bit},
               {"admission_order", metadata->admission_order},
               {"reuse_count", metadata->reuse_count},
               {"tier", tier},
               {"reason", reason},
               {"gpu_bytes", gpu_bytes},
               {"cpu_bytes", cpu_bytes},
               {"ancestor_checkpoint",
                ancestor.has_value()
                    ? nlohmann::json(std::to_string(ancestor->checkpoint))
                    : nlohmann::json(nullptr)},
               {"distance_from_ancestor_tokens",
                ancestor.has_value()
                    ? nlohmann::json(
                          metadata->prefix_tokens - ancestor->prefix_tokens)
                    : nlohmann::json(nullptr)}};
    add_checkpoint_event_state(&event);
    events_(event);
  }

  void emit_checkpoint_rejection(
      kv_cache::CheckpointId checkpoint, std::size_t tokens,
      prefix_index::CheckpointSource source, std::string_view reason,
      const std::optional<prefix_index::CheckpointMetadata>& ancestor =
          std::nullopt,
      std::string_view tier = "gpu") {
    if (!bool(events_)) {
      return;
    }
    nlohmann::json event{{"sequence", next_checkpoint_event_sequence()},
               {"monotonic_seconds",
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count()},
               {"action", "reject"},
               {"checkpoint", checkpoint == 0
                                  ? nlohmann::json(nullptr)
                                  : nlohmann::json(std::to_string(checkpoint))},
               {"tokens", tokens},
               {"sources", checkpoint_sources_name(
                   prefix_index::checkpoint_source_mask(source))},
               {"tier", tier},
               {"reason", reason},
               {"gpu_bytes", 0},
               {"cpu_bytes", 0},
               {"ancestor_checkpoint",
                ancestor.has_value()
                    ? nlohmann::json(std::to_string(ancestor->checkpoint))
                    : nlohmann::json(nullptr)},
               {"distance_from_ancestor_tokens",
                ancestor.has_value()
                    ? nlohmann::json(tokens - ancestor->prefix_tokens)
                    : nlohmann::json(nullptr)}};
    add_checkpoint_event_state(&event);
    events_(event);
  }

  void record_admission(kv_cache::CheckpointId checkpoint) {
    const std::optional<prefix_index::CheckpointMetadata> metadata =
        prefix_index_.metadata(checkpoint);
    if (!metadata.has_value()) {
      return;
    }
    const std::size_t admitted_bytes = checkpoint_admitted_bytes(checkpoint);
    CheckpointLifecycleBucket& bucket = checkpoint_buckets_[
        lifecycle_bucket_index(metadata->sources, metadata->priority,
                               metadata->cache_class)];
    add_telemetry(&bucket.admissions, 1, "checkpoint admission count");
    add_telemetry(&bucket.admitted_bytes, admitted_bytes,
                  "checkpoint admitted bytes");
    emit_checkpoint_event("admit", checkpoint, metadata, "gpu", "capture",
                          admitted_bytes, 0);
  }

  void record_reclamation(
      kv_cache::CheckpointId checkpoint,
      const std::optional<prefix_index::CheckpointMetadata>& metadata,
      std::size_t gpu_bytes, std::size_t cpu_bytes, bool removed,
      std::string_view tier, std::string_view reason) {
    add_telemetry(&gpu_reclaimed_bytes_, gpu_bytes, "GPU reclaimed bytes");
    add_telemetry(&cpu_reclaimed_bytes_, cpu_bytes, "CPU reclaimed bytes");
    if (!metadata.has_value()) {
      return;
    }
    CheckpointLifecycleBucket& bucket = checkpoint_buckets_[
        lifecycle_bucket_index(metadata->sources, metadata->priority,
                               metadata->cache_class)];
    if (removed) {
      add_telemetry(&bucket.removals, 1, "checkpoint removal count");
    }
    add_telemetry(&bucket.reclaimed_gpu_bytes, gpu_bytes,
                  "checkpoint GPU reclaimed bytes");
    add_telemetry(&bucket.reclaimed_cpu_bytes, cpu_bytes,
                  "checkpoint CPU reclaimed bytes");
    emit_checkpoint_event(removed ? "remove" : "spill", checkpoint, metadata,
                          tier, reason, gpu_bytes, cpu_bytes);
  }

  void on_hot_checkpoint_removed(kv_cache::CheckpointId checkpoint,
                                 bool externally_preserved) {
    std::optional<prefix_index::CheckpointMetadata> metadata =
        prefix_index_.metadata(checkpoint);
    std::size_t gpu_bytes = 0;
    bool record_pressure_reclamation = false;
    bool redundant_spacing = false;
    bool index_pressure = false;
    if (pending_hot_reclaim_.has_value() &&
        pending_hot_reclaim_->checkpoint == checkpoint) {
      metadata = pending_hot_reclaim_->metadata;
      gpu_bytes = pending_hot_reclaim_->gpu_bytes;
      redundant_spacing = pending_hot_reclaim_->redundant_spacing;
      index_pressure = pending_hot_reclaim_->index_pressure;
      pending_hot_reclaim_.reset();
      record_pressure_reclamation = true;
    }
    if (!externally_preserved) {
      discard_demands(checkpoint);
      (void)prefix_index_.remove(checkpoint);
    }
    if (record_pressure_reclamation) {
      // A successful spill preserves the checkpoint in CPU residence. Its GPU
      // bytes were reclaimed, but the lifecycle removal occurs only if the
      // checkpoint itself is discarded.
      record_reclamation(checkpoint, metadata, gpu_bytes, 0,
                         !externally_preserved,
                         externally_preserved ? "gpu_to_cpu" : "gpu",
                         redundant_spacing
                             ? (externally_preserved
                                    ? "hierarchical_spacing_spill"
                                    : "hierarchical_spacing")
                             : index_pressure
                                   ? (externally_preserved
                                          ? "index_pressure_spill"
                                          : "index_pressure")
                                   : (externally_preserved ? "clock_spill"
                                                           : "clock_pressure"));
    }
  }

  void release_hot_checkpoint(
      kv_cache::CheckpointId checkpoint,
      std::string_view reason,
      std::optional<prefix_index::CheckpointMetadata> metadata_override =
          std::nullopt) {
    const std::optional<prefix_index::CheckpointMetadata> metadata =
        metadata_override.has_value() ? metadata_override
                                      : prefix_index_.metadata(checkpoint);
    const std::size_t before = ledger_.stats().gpu.used;
    pending_hot_reclaim_.reset();
    ledger_.release_checkpoint(checkpoint);
    const std::size_t after = ledger_.stats().gpu.used;
    if (after > before) {
      fail("persistent KV checkpoint", "checkpoint release grew the GPU pool");
    }
    record_reclamation(checkpoint, metadata, before - after, 0, true, "gpu",
                       reason);
  }

  void clear_reservation() {
    if (speculative_staging_.valid()) {
      fail("persistent speculative KV", "execution released with live staging");
    }
    execution_reservation_bytes_ = 0;
    execution_reservation_index_bytes_ = 0;
    speculative_reservation_bytes_ = 0;
    reservation_max_processed_tokens_ = 0;
    reservation_execution_ = 0;
    pending_capture_tail_page_ = 0;
    cold_reservation_checkpoint_ = 0;
  }

  [[nodiscard]] ExecutionReservation remaining_reservation(
      kv_cache::ExecutionId execution,
      bool retain_future_reservation = true) const {
    if (reservation_execution_ != execution) {
      return {};
    }
    const kv_cache::ExecutionInfo& state = ledger_.execution(execution);
    if (speculative_staging_.bytes > speculative_reservation_bytes_) {
      fail("persistent speculative KV", "staging exceeded its reservation");
    }
    std::size_t private_pages = 0;
    if (state.source_checkpoint == 0) {
      private_pages = state.global_pages.size();
    } else {
      const std::vector<kv_cache::PageId>& source_pages =
          ledger_.checkpoint(state.source_checkpoint).global_pages;
      for (std::size_t index = 0; index < state.global_pages.size(); ++index) {
        if (index >= source_pages.size() ||
            state.global_pages[index] != source_pages[index]) {
          ++private_pages;
        }
      }
    }
    const std::size_t consumed_gpu =
        state.local_ring.bytes + state.page_table.bytes +
        private_pages * config_.global_page_bytes + speculative_staging_.bytes;
    const std::size_t consumed_index =
        sizeof(kv_cache::ExecutionInfo) +
        state.global_pages.capacity() * sizeof(kv_cache::PageId) +
        private_pages * sizeof(kv_cache::PageInfo);
    if (consumed_gpu > execution_reservation_bytes_ ||
        consumed_index > execution_reservation_index_bytes_) {
      fail("persistent KV cache",
           "execution exceeded its reserved working set");
    }
    if (!retain_future_reservation) {
      return {};
    }
    return {execution_reservation_bytes_ - consumed_gpu,
            execution_reservation_index_bytes_ - consumed_index,
            speculative_reservation_bytes_ - speculative_staging_.bytes};
  }

  static std::size_t ledger_index_bytes(const kv_cache::PoolConfig& config) {
    return config.index_bytes / 4;
  }

  static std::size_t demand_index_bytes(const kv_cache::PoolConfig& config) {
    return config.index_bytes / 8;
  }

  static std::size_t active_checkpoint_index_bytes(
      const kv_cache::PoolConfig& config) {
    return config.index_bytes / 4;
  }

  static std::size_t active_checkpoint_entry_bytes(bool named) {
    return sizeof(CheckpointTrigger) + sizeof(CapturedCheckpoint) +
           sizeof(kv_cache::CheckpointId) +
           (named ? sizeof(kv_cache::CheckpointId) +
                        sizeof(OwnerDemandCommit::Change)
                  : 0);
  }

  static std::size_t active_checkpoint_fixed_bytes(bool named) {
    // Source/input/completion can contribute two IDs beyond scheduled
    // captures. Named requests also stage one endpoint and retain rollback
    // information for those two extra owner-demand changes.
    return 2 * sizeof(kv_cache::CheckpointId) +
           (named ? sizeof(kv_cache::CheckpointId) +
                        2 * sizeof(OwnerDemandCommit::Change)
                  : 0);
  }

  static std::size_t cold_index_bytes(const kv_cache::PoolConfig& config) {
    return config.cpu_bytes == 0 ? 0 : config.index_bytes / 8;
  }

  static std::size_t maximum_global_page_count(
      const kv_cache::PoolConfig& config) {
    return (static_cast<std::size_t>(config.maximum_context_tokens) +
            config.global_page_tokens - 1) /
           config.global_page_tokens;
  }

  static std::size_t page_offsets_count(
      const kv_cache::PoolConfig& config) {
    if (config.page_table_bytes % sizeof(std::uint64_t) != 0) {
      fail("persistent KV cache", "page-table bytes are not uint64-aligned");
    }
    const std::size_t count = config.page_table_bytes / sizeof(std::uint64_t);
    if (count < maximum_global_page_count(config)) {
      fail("persistent KV cache", "page-table metadata cannot cover context");
    }
    return count;
  }

  static std::size_t page_offsets_index_bytes(
      const kv_cache::PoolConfig& config) {
    return page_offsets_count(config) * sizeof(std::uint64_t);
  }

  static std::size_t cold_fixed_index_bytes(
      const kv_cache::PoolConfig& config) {
    const auto count = cold_execution_capacity(config);
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(ColdExecution))
      fail("persistent KV cache", "cold execution metadata overflows");
    return count * sizeof(ColdExecution);
  }

  static std::size_t cold_execution_capacity(const kv_cache::PoolConfig& config) {
    return config.cpu_bytes == 0 ? 0 : config.gpu_bytes / config.local_ring_bytes;
  }

  static std::size_t cold_dynamic_index_bytes(
      const kv_cache::PoolConfig& config) {
    const std::size_t total = cold_index_bytes(config);
    const std::size_t fixed = cold_fixed_index_bytes(config);
    if (fixed > total) {
      fail("persistent KV cache", "index budget cannot hold cold bookkeeping");
    }
    return total - fixed;
  }

  static std::size_t cold_page_record_bytes(
      const kv_cache::PoolConfig& config) {
    return cold_dynamic_index_bytes(config) / 4;
  }

  static std::size_t cold_checkpoint_record_bytes(
      const kv_cache::PoolConfig& config) {
    return cold_dynamic_index_bytes(config) / 4;
  }

  static std::size_t cold_page_id_bytes(const kv_cache::PoolConfig& config) {
    return cold_dynamic_index_bytes(config) - cold_page_record_bytes(config) -
           cold_checkpoint_record_bytes(config);
  }

  static std::size_t cold_page_capacity(const kv_cache::PoolConfig& config) {
    return cold_page_record_bytes(config) / sizeof(ColdPage);
  }

  static std::size_t cold_checkpoint_capacity(
      const kv_cache::PoolConfig& config) {
    const std::size_t record_capacity =
        cold_checkpoint_record_bytes(config) / sizeof(ColdCheckpoint);
    const std::size_t pages_per_checkpoint = maximum_global_page_count(config);
    if (pages_per_checkpoint == 0 ||
        pages_per_checkpoint > std::numeric_limits<std::size_t>::max() /
                                   sizeof(kv_cache::PageId)) {
      fail("persistent KV cache", "cold page-ID capacity is invalid");
    }
    const std::size_t page_id_capacity =
        cold_page_id_bytes(config) /
        (pages_per_checkpoint * sizeof(kv_cache::PageId));
    return std::min(record_capacity, page_id_capacity);
  }

  static std::size_t cold_page_id_count(
      const kv_cache::PoolConfig& config) {
    const std::size_t checkpoints = cold_checkpoint_capacity(config);
    const std::size_t pages_per_checkpoint = maximum_global_page_count(config);
    if (checkpoints > std::numeric_limits<std::size_t>::max() /
                          pages_per_checkpoint) {
      fail("persistent KV cache", "cold page-ID count overflows");
    }
    return checkpoints * pages_per_checkpoint;
  }

  static std::size_t prefix_index_bytes(const kv_cache::PoolConfig& config) {
    const std::size_t reserved = ledger_index_bytes(config) +
        demand_index_bytes(config) + active_checkpoint_index_bytes(config) +
        cold_index_bytes(config) +
        page_offsets_index_bytes(config);
    if (reserved >= config.index_bytes) {
      fail("persistent KV cache", "index budget cannot hold cache metadata");
    }
    return config.index_bytes - reserved;
  }

  static std::size_t demand_slot_capacity(const kv_cache::PoolConfig& config) {
    return demand_index_bytes(config) / sizeof(DemandSlot);
  }

  static kv_cache::PoolConfig ledger_config(kv_cache::PoolConfig config) {
    config.index_bytes = ledger_index_bytes(config);
    config.validate();
    return config;
  }

  [[nodiscard]] std::size_t active_demand_count() const {
    return demand_slots_.size();
  }

  [[nodiscard]] static bool same_owner(const DemandSlot& slot,
                                       std::string_view owner) {
    return slot.owner_bytes == owner.size() &&
           std::memcmp(slot.owner.data(), owner.data(), owner.size()) == 0;
  }

  [[nodiscard]] bool has_demand(kv_cache::CheckpointId checkpoint) const {
    return std::any_of(demand_slots_.begin(), demand_slots_.end(),
                       [checkpoint](const DemandSlot& slot) {
                         return slot.checkpoint == checkpoint;
                       });
  }

  [[nodiscard]] bool add_demand(
      kv_cache::CheckpointId checkpoint, std::string_view owner,
      prefix_index::RetentionPriority priority, bool automatic,
      bool consume_reserved_slot = false) {
    if (checkpoint == 0 || !has_checkpoint(checkpoint)) {
      return false;
    }
    if (automatic ? !owner.empty()
                  : owner.empty() || owner.size() > kMaximumCacheOwnerBytes) {
      fail("persistent KV demand", "owner demand is malformed");
    }
    for (DemandSlot& slot : demand_slots_) {
      if (slot.checkpoint == checkpoint && slot.automatic == automatic &&
          (automatic || same_owner(slot, owner))) {
        if (!automatic || static_cast<unsigned>(priority) >
                              static_cast<unsigned>(slot.priority)) {
          slot.priority = priority;
        }
        refresh_priority(checkpoint);
        return true;
      }
    }
    const std::size_t free_slots = demand_slot_capacity_ - active_demand_count();
    if (free_slots == 0 ||
        (!consume_reserved_slot && free_slots <= reserved_owner_demand_slots_)) {
      return false;
    }
    demand_slots_.emplace_back();
    DemandSlot& slot = demand_slots_.back();
    slot.checkpoint = checkpoint;
    slot.owner_bytes = static_cast<std::uint8_t>(owner.size());
    slot.priority = priority;
    slot.automatic = automatic;
    if (!automatic) {
      std::copy(owner.begin(), owner.end(), slot.owner.begin());
    }
    refresh_priority(checkpoint);
    return true;
  }

  void discard_demands(kv_cache::CheckpointId checkpoint) {
    demand_slots_.erase(std::remove_if(demand_slots_.begin(), demand_slots_.end(),
        [checkpoint](const DemandSlot& slot) { return slot.checkpoint == checkpoint; }),
        demand_slots_.end());
  }

  void refresh_priority(kv_cache::CheckpointId checkpoint) {
    if (!has_checkpoint(checkpoint)) {
      return;
    }
    prefix_index::RetentionPriority strongest =
        prefix_index::RetentionPriority::low;
    bool found = false;
    for (const DemandSlot& slot : demand_slots_) {
      if (slot.checkpoint != checkpoint) {
        continue;
      }
      if (!found || static_cast<unsigned>(slot.priority) >
                        static_cast<unsigned>(strongest)) {
        strongest = slot.priority;
        found = true;
      }
    }
    if (found) {
      (void)prefix_index_.set_priority(checkpoint, strongest);
    }
  }

  void reclaim_if_undemanded(kv_cache::CheckpointId checkpoint) {
    if (checkpoint == 0 || has_demand(checkpoint)) {
      return;
    }
    if (ledger_.has_checkpoint(checkpoint)) {
      const kv_cache::CheckpointInfo& info = ledger_.checkpoint(checkpoint);
      if (info.execution_references == 0 && info.dependency_pins == 0) {
        release_hot_checkpoint(checkpoint, "undemanded");
      }
      return;
    }
    ColdCheckpoint* const cold = find_cold(checkpoint);
    if (cold != nullptr && cold->execution_references == 0 && cold->dependency_pins == 0) {
      discard_cold(checkpoint, "undemanded");
    }
  }

  [[nodiscard]] bool checkpoint_idle(kv_cache::CheckpointId checkpoint) const {
    if (ledger_.has_checkpoint(checkpoint)) {
      const auto& source = ledger_.checkpoint(checkpoint);
      return !source.execution_references && !source.dependency_pins;
    }
    const auto* cold = find_cold(checkpoint);
    return cold && !cold->execution_references && !cold->dependency_pins;
  }

  [[nodiscard]] std::size_t cold_checkpoint_index(
      const ColdCheckpoint& checkpoint) const {
    for (std::size_t index = 0; index < cold_checkpoint_count_; ++index) {
      if (&cold_checkpoints_[index] == &checkpoint) {
        return index;
      }
    }
    fail("persistent KV cold tier", "checkpoint is outside the cold index");
  }

  [[nodiscard]] kv_cache::PageId* cold_page_ids_at(
      std::size_t checkpoint_index) {
    if (cold_page_ids_ == nullptr ||
        checkpoint_index >= cold_checkpoint_capacity_) {
      fail("persistent KV cold tier", "cold page-ID slot is invalid");
    }
    return cold_page_ids_.get() +
           checkpoint_index * cold_page_id_slots_per_checkpoint_;
  }

  [[nodiscard]] const kv_cache::PageId* cold_page_ids_at(
      std::size_t checkpoint_index) const {
    if (cold_page_ids_ == nullptr ||
        checkpoint_index >= cold_checkpoint_capacity_) {
      fail("persistent KV cold tier", "cold page-ID slot is invalid");
    }
    return cold_page_ids_.get() +
           checkpoint_index * cold_page_id_slots_per_checkpoint_;
  }

  [[nodiscard]] const kv_cache::PageId* cold_page_ids(
      const ColdCheckpoint& checkpoint) const {
    return cold_page_ids_at(cold_checkpoint_index(checkpoint));
  }

  [[nodiscard]] std::size_t cold_metadata_used() const {
    if (config_.cpu_bytes == 0) {
      return 0;
    }
    std::size_t used = cold_fixed_index_bytes(config_);
    const auto add = [&used](std::size_t bytes) {
      if (bytes > std::numeric_limits<std::size_t>::max() - used) {
        fail("persistent KV cold tier", "metadata byte count overflows");
      }
      used += bytes;
    };
    if (cold_page_count_ > std::numeric_limits<std::size_t>::max() /
                               sizeof(ColdPage) ||
        cold_checkpoint_count_ > std::numeric_limits<std::size_t>::max() /
                                     sizeof(ColdCheckpoint)) {
      fail("persistent KV cold tier", "metadata record count overflows");
    }
    add(cold_page_count_ * sizeof(ColdPage));
    add(cold_checkpoint_count_ * sizeof(ColdCheckpoint));
    for (std::size_t index = 0; index < cold_checkpoint_count_; ++index) {
      const ColdCheckpoint& checkpoint = cold_checkpoints_[index];
      if (checkpoint.source_page_count > cold_page_id_slots_per_checkpoint_ ||
          checkpoint.source_page_count >
              std::numeric_limits<std::size_t>::max() /
                  sizeof(kv_cache::PageId)) {
        fail("persistent KV cold tier", "cold page-ID count is invalid");
      }
      add(checkpoint.source_page_count * sizeof(kv_cache::PageId));
    }
    if (used > cold_index_bytes_) {
      fail("persistent KV cold tier", "metadata exceeds its fixed budget");
    }
    return used;
  }

  [[nodiscard]] const ColdPage* find_cold_page(
      kv_cache::PageId source_page) const {
    for (std::size_t index = 0; index < cold_page_count_; ++index) {
      if (cold_pages_[index].source_page == source_page) {
        return &cold_pages_[index];
      }
    }
    return nullptr;
  }

  [[nodiscard]] ColdPage* find_cold_page(kv_cache::PageId source_page) {
    for (std::size_t index = 0; index < cold_page_count_; ++index) {
      if (cold_pages_[index].source_page == source_page) {
        return &cold_pages_[index];
      }
    }
    return nullptr;
  }

  [[nodiscard]] const ColdCheckpoint* find_cold(
      kv_cache::CheckpointId checkpoint) const {
    for (std::size_t index = 0; index < cold_checkpoint_count_; ++index) {
      if (cold_checkpoints_[index].checkpoint == checkpoint) {
        return &cold_checkpoints_[index];
      }
    }
    return nullptr;
  }

  [[nodiscard]] ColdCheckpoint* find_cold(kv_cache::CheckpointId checkpoint) {
    for (std::size_t index = 0; index < cold_checkpoint_count_; ++index) {
      if (cold_checkpoints_[index].checkpoint == checkpoint) {
        return &cold_checkpoints_[index];
      }
    }
    return nullptr;
  }

  [[nodiscard]] std::optional<kv_cache::CheckpointId> cold_execution_source(
      kv_cache::ExecutionId execution) const {
    for (std::size_t i = 0; i < cold_execution_capacity_; ++i)
      if (cold_executions_[i].execution == execution) return cold_executions_[i].checkpoint;
    return std::nullopt;
  }

  void add_cold_execution(kv_cache::ExecutionId execution, kv_cache::CheckpointId checkpoint) {
    auto* cold = find_cold(checkpoint);
    if (!execution || !cold || cold_execution_source(execution))
      fail("persistent KV restore", "cold execution source is invalid");
    for (std::size_t i = 0; i < cold_execution_capacity_; ++i) {
      if (cold_executions_[i].execution) continue;
      cold_executions_[i] = {execution, checkpoint};
      ++cold->execution_references;
      return;
    }
    fail("persistent KV restore", "cold execution slots exhausted");
  }

  void release_cold_execution(kv_cache::ExecutionId execution) {
    for (std::size_t i = 0; i < cold_execution_capacity_; ++i) {
      if (cold_executions_[i].execution != execution) continue;
      const auto checkpoint = cold_executions_[i].checkpoint;
      auto* cold = find_cold(checkpoint);
      if (!cold || !cold->execution_references)
        fail("persistent KV release", "cold execution reference underflow");
      --cold->execution_references;
      cold_executions_[i] = {};
      reclaim_if_undemanded(checkpoint);
      return;
    }
  }

  void restore_cold(kv_cache::ExecutionId execution, kv_cache::CheckpointId checkpoint,
                    CompletionContext stream) {
    const auto started = std::chrono::steady_clock::now();
    auto* cold = find_cold(checkpoint);
    if (!cold) fail("persistent KV restore", "cold checkpoint disappeared");
    const auto plan = ledger_.prepare_write(execution, 0, cold->processed_tokens);
    // Serial allocation can spill and compact cold metadata during prepare_write.
    cold = find_cold(checkpoint);
    if (!cold || !plan.copies.empty() || plan.new_pages.size() != cold->source_page_count)
      fail("persistent KV restore", "cold checkpoint page plan is invalid");
    const auto& restored = ledger_.execution(execution);
    const auto* source_pages = cold_page_ids(*cold);
    for (std::size_t i = 0; i < cold->source_page_count; ++i) {
      const auto* page = find_cold_page(source_pages[i]);
      if (!page) fail("persistent KV restore", "cold checkpoint page is missing");
      physical_->restore_global_page(page->storage,
          ledger_.page(restored.global_pages[i]).storage, stream);
    }
    restore_local(restored, *cold, stream);
    physical_->upload_page_table(restored, stream);
    std::size_t bytes = cold->local_snapshot.bytes;
    add_telemetry(&bytes, cold->terminal_hidden.bytes, "cold restore byte count");
    if (cold->source_page_count > std::numeric_limits<std::size_t>::max() / config_.global_page_bytes)
      fail("persistent KV restore", "cold restore byte count overflows");
    add_telemetry(&bytes, cold->source_page_count * config_.global_page_bytes, "cold restore byte count");
    add_telemetry(&cold_restore_bytes_, bytes, "cold restore bytes");
    add_telemetry(&cold_restore_count_, 1, "cold restore count");
    add_telemetry(&cold_restore_wall_milliseconds_,
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(),
        "cold restore wall milliseconds");
  }

  [[nodiscard]] std::size_t cold_checkpoint_bytes(
      const ColdCheckpoint& checkpoint) const {
    std::size_t bytes = checkpoint.local_snapshot.bytes +
                        checkpoint.terminal_hidden.bytes;
    const kv_cache::PageId* const page_ids = cold_page_ids(checkpoint);
    for (std::size_t index = 0; index < checkpoint.source_page_count; ++index) {
      const kv_cache::PageId page_id = page_ids[index];
      const ColdPage* const page = find_cold_page(page_id);
      if (page == nullptr) {
        fail("persistent KV cold tier", "checkpoint page is missing");
      }
      if (page->references == 1) {
        if (page->storage.bytes > std::numeric_limits<std::size_t>::max() -
                                      bytes) {
          fail("persistent KV cold tier", "checkpoint byte count overflows");
        }
        bytes += page->storage.bytes;
      }
    }
    return bytes;
  }

  [[nodiscard]] static bool lower_cold_value(
      const prefix_index::CheckpointMetadata& candidate,
      const prefix_index::CheckpointMetadata& incoming) {
    if (candidate.priority != incoming.priority) {
      return static_cast<unsigned>(candidate.priority) <
             static_cast<unsigned>(incoming.priority);
    }
    return candidate.cache_class == prefix_index::CacheClass::probationary &&
           incoming.cache_class == prefix_index::CacheClass::reused;
  }

  [[nodiscard]] kv_cache::CheckpointId select_cold_clock_victim(
      kv_cache::CheckpointId protected_checkpoint,
      const prefix_index::CheckpointMetadata& incoming) {
    const std::optional<prefix_index::CheckpointId> victim =
        prefix_index_.select_clock_victim(
            [this, protected_checkpoint, &incoming](
                prefix_index::CheckpointId candidate) {
              const ColdCheckpoint* const cold = find_cold(candidate);
              const std::optional<prefix_index::CheckpointMetadata> metadata =
                  prefix_index_.metadata(candidate);
              return cold != nullptr && metadata.has_value() &&
                     candidate != protected_checkpoint &&
                     candidate != cold_reservation_checkpoint_ &&
                     cold->execution_references == 0 &&
                     cold->dependency_pins == 0 &&
                     lower_cold_value(*metadata, incoming);
            },
            [this](prefix_index::CheckpointId candidate) {
              const ColdCheckpoint* const cold = find_cold(candidate);
              return cold == nullptr ? std::size_t{0}
                                     : cold_checkpoint_bytes(*cold);
            });
    return victim.value_or(0);
  }

  [[nodiscard]] kv_cache::Allocation allocate_cold(std::size_t bytes,
                                                    kv_cache::CheckpointId protected_checkpoint,
                                                    const prefix_index::CheckpointMetadata& incoming) {
    kv_cache::Allocation allocation = ledger_.try_allocate_cpu(bytes);
    while (!allocation.valid()) {
      const kv_cache::CheckpointId victim =
          select_cold_clock_victim(protected_checkpoint, incoming);
      if (victim == 0) {
        return {};
      }
      discard_cold(victim, "cold_capacity");
      allocation = ledger_.try_allocate_cpu(bytes);
    }
    return allocation;
  }

  [[nodiscard]] std::size_t count_missing_cold_pages(
      const std::vector<kv_cache::PageId>& source_pages) const {
    std::size_t missing = 0;
    for (const kv_cache::PageId page_id : source_pages) {
      if (find_cold_page(page_id) == nullptr) {
        ++missing;
      }
    }
    return missing;
  }

  [[nodiscard]] bool cold_spill_fits_cpu(
      const kv_cache::CheckpointInfo& source) const {
    std::size_t required = source.local_snapshot.bytes;
    const auto add = [&required](std::size_t bytes) {
      if (bytes > std::numeric_limits<std::size_t>::max() - required) {
        return false;
      }
      required += bytes;
      return true;
    };
    if (!add(source.terminal_hidden.bytes) ||
        source.global_pages.size() >
            std::numeric_limits<std::size_t>::max() /
                config_.global_page_bytes ||
        !add(source.global_pages.size() * config_.global_page_bytes)) {
      return false;
    }
    return required <= ledger_.cpu_pool().stats().capacity;
  }

  [[nodiscard]] bool ensure_cold_metadata_capacity(
      const std::vector<kv_cache::PageId>& source_pages,
      kv_cache::CheckpointId protected_checkpoint,
      const prefix_index::CheckpointMetadata& incoming) {
    if (source_pages.size() > cold_page_id_slots_per_checkpoint_ ||
        cold_page_capacity_ == 0 || cold_checkpoint_capacity_ == 0) {
      return false;
    }
    while (true) {
      if (cold_page_count_ > cold_page_capacity_ ||
          cold_checkpoint_count_ > cold_checkpoint_capacity_) {
        fail("persistent KV cold tier", "metadata count exceeds capacity");
      }
      const std::size_t missing = count_missing_cold_pages(source_pages);
      if (cold_checkpoint_count_ < cold_checkpoint_capacity_ &&
          missing <= cold_page_capacity_ - cold_page_count_) {
        return true;
      }
      const kv_cache::CheckpointId victim =
          select_cold_clock_victim(protected_checkpoint, incoming);
      if (victim == 0) {
        return false;
      }
      discard_cold(victim, "cold_metadata_capacity");
    }
  }

  void release_cold_page(kv_cache::PageId source_page) {
    ColdPage* const found = find_cold_page(source_page);
    if (found == nullptr || found->references == 0) {
      fail("persistent KV cold tier", "cold page reference underflow");
    }
    if (--found->references == 0) {
      ledger_.release_cpu(found->storage);
      const std::size_t index =
          static_cast<std::size_t>(found - cold_pages_.get());
      const std::size_t last = cold_page_count_ - 1;
      if (index != last) {
        cold_pages_[index] = cold_pages_[last];
      }
      --cold_page_count_;
    }
  }

  void discard_cold(
      kv_cache::CheckpointId checkpoint,
      std::string_view reason,
      std::optional<prefix_index::CheckpointMetadata> metadata_override =
          std::nullopt) {
    ColdCheckpoint* const found = find_cold(checkpoint);
    if (found == nullptr) {
      return;
    }
    if (found->execution_references != 0 || found->dependency_pins != 0) {
      fail("persistent KV cold tier", "cannot discard a borrowed checkpoint");
    }
    const std::optional<prefix_index::CheckpointMetadata> metadata =
        metadata_override.has_value() ? metadata_override
                                      : prefix_index_.metadata(checkpoint);
    const std::size_t cpu_before = ledger_.stats().cpu.used;
    const std::size_t index = cold_checkpoint_index(*found);
    const ColdCheckpoint value = *found;
    const kv_cache::PageId* const page_ids = cold_page_ids_at(index);
    for (std::size_t page_index = 0;
         page_index < value.source_page_count; ++page_index) {
      release_cold_page(page_ids[page_index]);
    }
    ledger_.release_cpu(value.local_snapshot);
    ledger_.release_cpu(value.terminal_hidden);
    const std::size_t last = cold_checkpoint_count_ - 1;
    if (index != last) {
      const ColdCheckpoint replacement = cold_checkpoints_[last];
      std::copy_n(cold_page_ids_at(last), replacement.source_page_count,
                  cold_page_ids_at(index));
      cold_checkpoints_[index] = replacement;
    }
    --cold_checkpoint_count_;
    discard_demands(checkpoint);
    (void)prefix_index_.remove(checkpoint);
    const std::size_t cpu_after = ledger_.stats().cpu.used;
    if (cpu_after > cpu_before) {
      fail("persistent KV cold tier", "cold discard grew the CPU pool");
    }
    record_reclamation(checkpoint, metadata, 0, cpu_before - cpu_after, true,
                       "cpu", reason);
  }

  [[nodiscard]] bool spill_checkpoint(kv_cache::CheckpointId checkpoint) {
    if (config_.cpu_bytes == 0 || !has_demand(checkpoint) ||
        !ledger_.has_checkpoint(checkpoint)) {
      return false;
    }
    const kv_cache::CheckpointInfo& source = ledger_.checkpoint(checkpoint);
    if (source.execution_references != 0 || source.dependency_pins != 0 ||
        find_cold(checkpoint) != nullptr) {
      return false;
    }
    const std::optional<prefix_index::CheckpointMetadata> incoming =
        prefix_index_.metadata(checkpoint);
    if (!incoming.has_value()) {
      return false;
    }
    if (!cold_spill_fits_cpu(source)) {
      return false;
    }
    if (!ensure_cold_metadata_capacity(source.global_pages, checkpoint,
                                        *incoming)) {
      return false;
    }
    const auto spill_started = std::chrono::steady_clock::now();
    ColdCheckpoint cold;
    std::size_t retained_page_count = 0;
    std::size_t newly_copied_page_count = 0;
    std::size_t avoided_rewrite_bytes = 0;
    try {
      cold.checkpoint = checkpoint;
      cold.processed_tokens = source.processed_tokens;
      cold.local_start = source.local_start;
      cold.local_tokens = source.local_tokens;
      cold.source_page_count = source.global_pages.size();
      cold.local_snapshot =
          allocate_cold(source.local_snapshot.bytes, checkpoint, *incoming);
      if (!cold.local_snapshot.valid()) {
        return false;
      }
      cold.terminal_hidden =
          allocate_cold(source.terminal_hidden.bytes, checkpoint, *incoming);
      if (!cold.terminal_hidden.valid()) {
        ledger_.release_cpu(cold.local_snapshot);
        return false;
      }
      for (std::size_t index = 0; index < cold.source_page_count; ++index) {
        const kv_cache::PageId page_id = source.global_pages[index];
        ColdPage* page = find_cold_page(page_id);
        if (page != nullptr) {
          if (page->valid_tokens != ledger_.page(page_id).valid_tokens) {
            fail("persistent KV cold tier",
                 "cold page token count differs from its GPU source");
          }
          add_telemetry(&avoided_rewrite_bytes, config_.global_page_bytes,
                        "cold spill avoided rewrite bytes");
          ++page->references;
          ++retained_page_count;
          continue;
        }
        if (!ensure_cold_metadata_capacity(source.global_pages, checkpoint,
                                            *incoming) ||
            cold_page_count_ == cold_page_capacity_) {
          throw std::bad_alloc();
        }
        kv_cache::Allocation storage =
            allocate_cold(config_.global_page_bytes, checkpoint, *incoming);
        if (!storage.valid()) {
          throw std::bad_alloc();
        }
        cold_pages_[cold_page_count_] = {
            page_id, storage, ledger_.page(page_id).valid_tokens, 1, true};
        ++cold_page_count_;
        ++retained_page_count;
        ++newly_copied_page_count;
      }
      for (std::size_t index = 0; index < cold.source_page_count; ++index) {
        const kv_cache::PageId page_id = source.global_pages[index];
        ColdPage* const page = find_cold_page(page_id);
        if (page == nullptr) {
          fail("persistent KV cold tier", "new cold page disappeared");
        }
        if (!page->pending_copy) {
          continue;
        }
        physical_->spill(ledger_.page(page_id).storage, page->storage,
                        config_.global_page_bytes, "spill persistent global KV page");
        page->pending_copy = false;
      }
      physical_->spill(source.local_snapshot, cold.local_snapshot,
                      source.local_snapshot.bytes, "spill persistent local KV snapshot");
      physical_->spill(source.terminal_hidden, cold.terminal_hidden,
                      source.terminal_hidden.bytes, "spill persistent terminal hidden state");
      if (cold_checkpoint_count_ >= cold_checkpoint_capacity_) {
        fail("persistent KV cold tier", "checkpoint slot disappeared");
      }
      const std::size_t cold_index = cold_checkpoint_count_;
      std::copy(source.global_pages.begin(), source.global_pages.end(),
                cold_page_ids_at(cold_index));
      cold_checkpoints_[cold_index] = cold;
      ++cold_checkpoint_count_;
      if (newly_copied_page_count >
          std::numeric_limits<std::size_t>::max() /
              config_.global_page_bytes) {
        fail("persistent KV cold tier", "cold spill byte count overflows");
      }
      std::size_t copied_bytes = source.local_snapshot.bytes;
      add_telemetry(&copied_bytes, source.terminal_hidden.bytes,
                    "cold spill byte count");
      add_telemetry(&copied_bytes,
                    newly_copied_page_count * config_.global_page_bytes,
                    "cold spill byte count");
      add_telemetry(&cold_spill_bytes_, copied_bytes, "cold spill bytes");
      add_telemetry(&cold_spill_count_, 1, "cold spill count");
      add_telemetry(&cold_spill_avoided_rewrite_bytes_, avoided_rewrite_bytes,
                    "cold spill avoided rewrite bytes");
      add_telemetry(
          &cold_spill_wall_milliseconds_,
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - spill_started)
              .count(),
          "cold spill wall milliseconds");
      return true;
    } catch (...) {
      for (std::size_t index = 0; index < retained_page_count; ++index) {
        const kv_cache::PageId page_id = source.global_pages[index];
        ColdPage* const page = find_cold_page(page_id);
        if (page != nullptr) {
          release_cold_page(page_id);
        }
      }
      if (cold.local_snapshot.valid() && ledger_.cpu_pool().owns(cold.local_snapshot)) {
        ledger_.release_cpu(cold.local_snapshot);
      }
      if (cold.terminal_hidden.valid() &&
          ledger_.cpu_pool().owns(cold.terminal_hidden)) {
        ledger_.release_cpu(cold.terminal_hidden);
      }
      return false;
    }
  }

  [[nodiscard]] std::size_t checkpoint_reclaimable_gpu_bytes(
      kv_cache::CheckpointId checkpoint) const {
    const kv_cache::CheckpointInfo& info = ledger_.checkpoint(checkpoint);
    std::size_t bytes = info.local_snapshot.bytes;
    const auto add_bytes = [&bytes](std::size_t value) {
      if (value > std::numeric_limits<std::size_t>::max() - bytes) {
        fail("persistent KV cache", "checkpoint byte count overflows size_t");
      }
      bytes += value;
    };
    add_bytes(info.terminal_hidden.bytes);
    for (const kv_cache::PageId page_id : info.global_pages) {
      const kv_cache::PageInfo& page = ledger_.page(page_id);
      // A shared page remains resident when this checkpoint is removed, so it
      // cannot contribute to the current idle allocation or its class target.
      if (page.references == 1) {
        add_bytes(page.storage.bytes);
      }
    }
    return bytes;
  }

  [[nodiscard]] bool reclaim_prefix_index(
      kv_cache::CheckpointId protected_checkpoint) {
    // Spilling frees ledger/GPU storage but keeps the trie entry. Index
    // admission must instead remove idle retained state, in either tier.
    const auto victim = prefix_index_.select_clock_victim(
        [this, protected_checkpoint](auto checkpoint) {
          return checkpoint != protected_checkpoint &&
                 checkpoint != cold_reservation_checkpoint_ &&
                 checkpoint_idle(checkpoint);
        },
        [this](auto checkpoint) {
          return ledger_.has_checkpoint(checkpoint)
              ? checkpoint_reclaimable_gpu_bytes(checkpoint)
              : cold_checkpoint_bytes(*find_cold(checkpoint));
        });
    if (!victim) return false;
    if (ledger_.has_checkpoint(*victim))
      release_hot_checkpoint(*victim, "prefix_index_pressure");
    else
      discard_cold(*victim, "prefix_index_pressure");
    return true;
  }

  [[nodiscard]] kv_cache::CheckpointId select_pressure_victim(
      const std::vector<kv_cache::CheckpointId>& eligible,
      kv_cache::EvictionCause cause) {
    pending_hot_reclaim_.reset();
    const auto is_eligible =
        [this, &eligible](prefix_index::CheckpointId candidate) {
          return prefix_index_.contains(candidate) &&
                 std::find(eligible.begin(), eligible.end(), candidate) !=
                     eligible.end();
        };
    std::optional<prefix_index::CheckpointId> victim;
    if (cause == kv_cache::EvictionCause::gpu_pressure) {
      victim = prefix_index_.select_redundant_victim(
          8 * config_.local_window_tokens,
          [this, &is_eligible](prefix_index::CheckpointId candidate) {
            const auto metadata = prefix_index_.metadata(candidate);
            return is_eligible(candidate) && metadata.has_value() &&
                   metadata->prefix_tokens >= config_.local_window_tokens;
          });
    }
    const bool redundant_spacing = victim.has_value();
    if (!victim.has_value()) {
      victim = prefix_index_.select_clock_victim(
          is_eligible,
            [this](prefix_index::CheckpointId candidate) {
              return checkpoint_reclaimable_gpu_bytes(candidate);
            });
    }
    if (!victim.has_value()) {
      return 0;
    }
    const std::optional<prefix_index::CheckpointMetadata> metadata =
        prefix_index_.metadata(*victim);
    if (!metadata.has_value()) {
      fail("persistent KV eviction", "selected checkpoint has no metadata");
    }
    pending_hot_reclaim_ =
        PendingHotReclaim{*victim, metadata,
                          checkpoint_reclaimable_gpu_bytes(*victim),
                          redundant_spacing,
                          cause == kv_cache::EvictionCause::index_pressure};
    return *victim;
  }

  void restore_local(const kv_cache::ExecutionInfo& destination,
                     const kv_cache::CheckpointInfo& source,
                     CompletionContext stream = {}) {
    physical_->restore_local(destination, source.local_snapshot, source.local_start,
                            source.local_tokens, stream);
  }

  void restore_local(const kv_cache::ExecutionInfo& destination,
                     const ColdCheckpoint& source,
                     CompletionContext stream = {}) {
    physical_->restore_local(destination, source.local_snapshot, source.local_start,
                            source.local_tokens, stream);
  }

  kv_cache::PoolConfig config_;
  std::size_t page_offsets_count_{};
  std::size_t page_offsets_index_bytes_{};
  std::size_t cold_index_bytes_{};
  std::size_t cold_page_capacity_{};
  std::size_t cold_checkpoint_capacity_{};
  std::size_t cold_execution_capacity_{};
  std::size_t cold_page_id_slots_per_checkpoint_{};
  prefix_index::PrefixIndex prefix_index_;
  std::size_t demand_slot_capacity_{};
  std::vector<DemandSlot> demand_slots_;
  std::size_t reserved_owner_demand_slots_{};
  kv_cache::CacheLedger ledger_;
  std::unique_ptr<CacheStorage> physical_;
  CacheEventSink events_;
  std::unique_ptr<ColdPage[]> cold_pages_;
  std::unique_ptr<ColdCheckpoint[]> cold_checkpoints_;
  std::unique_ptr<kv_cache::PageId[]> cold_page_ids_;
  std::size_t cold_page_count_{};
  std::size_t cold_checkpoint_count_{};
  std::unique_ptr<ColdExecution[]> cold_executions_;
  std::size_t cold_spill_bytes_{};
  std::size_t cold_restore_bytes_{};
  std::size_t cold_spill_count_{};
  std::size_t cold_restore_count_{};
  std::size_t cold_spill_avoided_rewrite_bytes_{};
  double cold_spill_wall_milliseconds_{};
  double cold_restore_wall_milliseconds_{};
  std::array<CheckpointLifecycleBucket, kLifecycleBucketCount>
      checkpoint_buckets_{};
  std::uint64_t checkpoint_event_sequence_{};
  std::size_t gpu_reclaimed_bytes_{};
  std::size_t cpu_reclaimed_bytes_{};
  std::optional<PendingHotReclaim> pending_hot_reclaim_;
  std::size_t execution_reservation_bytes_{};
  std::size_t execution_reservation_index_bytes_{};
  std::size_t speculative_reservation_bytes_{};
  kv_cache::Allocation speculative_staging_;
  std::uint32_t reservation_max_processed_tokens_{};
  kv_cache::ExecutionId reservation_execution_{};
  kv_cache::PageId pending_capture_tail_page_{};
  kv_cache::CheckpointId cold_reservation_checkpoint_{};
};

PersistentCacheManager::PersistentCacheManager(kv_cache::PoolConfig config,
    CacheStorageFactory storage_factory, CacheEventSink events)
    : impl_(std::make_unique<Impl>(config, std::move(storage_factory), std::move(events))) {}
PersistentCacheManager::~PersistentCacheManager() = default;

CacheStorage& PersistentCacheManager::storage() {
  return impl_->storage();
}

const kv_cache::PoolConfig& PersistentCacheManager::config() const {
  return impl_->config();
}

std::size_t PersistentCacheManager::active_checkpoint_capacity(bool named) const {
  return impl_->active_checkpoint_capacity(named);
}

kv_cache::CacheStats PersistentCacheManager::stats() const {
  return impl_->stats();
}

prefix_index::MetadataStats PersistentCacheManager::prefix_stats() const {
  return impl_->prefix_stats();
}

nlohmann::json PersistentCacheManager::observability_snapshot(bool include_entries) const {
  return impl_->observability_snapshot(include_entries);
}

CacheTelemetry PersistentCacheManager::telemetry() const {
  return impl_->telemetry();
}

bool PersistentCacheManager::is_cold_checkpoint(kv_cache::CheckpointId checkpoint) const {
  return impl_->is_cold_checkpoint(checkpoint);
}

std::size_t PersistentCacheManager::gpu_page_slack_bytes() const {
  return impl_->gpu_page_slack_bytes();
}

std::size_t PersistentCacheManager::cpu_page_slack_bytes() const {
  return impl_->cpu_page_slack_bytes();
}

prefix_index::LookupResult PersistentCacheManager::find_longest(const std::vector<std::uint32_t>& tokens, const std::vector<prefix_index::ImageSpan>& images) const {
  return impl_->find_longest(tokens, images);
}

bool PersistentCacheManager::checkpoint_matches(kv_cache::CheckpointId checkpoint, const std::vector<std::uint32_t>& tokens, const std::vector<prefix_index::ImageSpan>& images) const {
  return impl_->checkpoint_matches(checkpoint, tokens, images);
}

prefix_index::LookupResult PersistentCacheManager::find_batch_prefix(const std::vector<std::uint32_t>& tokens, std::size_t maximum_processed_tokens, const std::vector<prefix_index::ImageSpan>& images) const {
  return impl_->find_batch_prefix(tokens, maximum_processed_tokens, images);
}

bool PersistentCacheManager::has_checkpoint(kv_cache::CheckpointId checkpoint) const {
  return impl_->has_checkpoint(checkpoint);
}

std::uint32_t PersistentCacheManager::checkpoint_tokens(kv_cache::CheckpointId checkpoint) const {
  return impl_->checkpoint_tokens(checkpoint);
}

void PersistentCacheManager::mark_used(kv_cache::CheckpointId checkpoint) {
  return impl_->mark_used(checkpoint);
}

void PersistentCacheManager::add_checkpoint_source(kv_cache::CheckpointId checkpoint, const std::vector<std::uint32_t>& tokens, prefix_index::CheckpointSource source, const std::vector<prefix_index::ImageSpan>& images) {
  return impl_->add_checkpoint_source(checkpoint, tokens, source, images);
}

bool PersistentCacheManager::add_automatic_demand(kv_cache::CheckpointId checkpoint, prefix_index::RetentionPriority priority) {
  return impl_->add_automatic_demand(checkpoint, priority);
}

bool PersistentCacheManager::add_owner_demands(const std::vector<kv_cache::CheckpointId>& checkpoints, std::string_view owner, prefix_index::RetentionPriority priority, OwnerDemandReservation* reservation, OwnerDemandCommit* commit) {
  return impl_->add_owner_demands(checkpoints, owner, priority, reservation, commit);
}

void PersistentCacheManager::rollback_owner_demands(std::string_view owner, OwnerDemandCommit* commit) {
  return impl_->rollback_owner_demands(owner, commit);
}

bool PersistentCacheManager::reserve_owner_demand_slots(std::size_t slots, OwnerDemandReservation* reservation) {
  return impl_->reserve_owner_demand_slots(slots, reservation);
}

std::size_t PersistentCacheManager::owner_demand_slot_capacity() const {
  return impl_->owner_demand_slot_capacity();
}

void PersistentCacheManager::release_owner_demand_reservation(OwnerDemandReservation* reservation) {
  return impl_->release_owner_demand_reservation(reservation);
}

void PersistentCacheManager::reclaim_undemanded(kv_cache::CheckpointId checkpoint) {
  return impl_->reclaim_undemanded(checkpoint);
}

void PersistentCacheManager::release_owner(std::string_view owner) {
  return impl_->release_owner(owner);
}

void PersistentCacheManager::ensure_capacity(kv_cache::CheckpointId source, std::size_t total_processed_tokens, std::size_t speculative_bytes) {
  return impl_->ensure_capacity(source, total_processed_tokens, speculative_bytes);
}

std::optional<kv_cache::ExecutionId> PersistentCacheManager::try_begin_batch(kv_cache::CheckpointId source, std::size_t max_processed_tokens, CompletionContext stream) {
  return impl_->try_begin_batch(source, max_processed_tokens, stream);
}

std::optional<kv_cache::ExecutionId> PersistentCacheManager::try_fork_batch(kv_cache::ExecutionId source, std::size_t horizon, CompletionContext stream) {
  return impl_->try_fork_batch(source, horizon, stream);
}

bool PersistentCacheManager::try_resize_batch(kv_cache::ExecutionId execution, std::size_t horizon) {
  return impl_->try_resize_batch(execution, horizon);
}

void PersistentCacheManager::pin_batch_checkpoint(kv_cache::CheckpointId checkpoint) {
  return impl_->pin_batch_checkpoint(checkpoint);
}

void PersistentCacheManager::unpin_batch_checkpoint(kv_cache::CheckpointId checkpoint) {
  return impl_->unpin_batch_checkpoint(checkpoint);
}

kv_cache::ExecutionId PersistentCacheManager::begin(kv_cache::CheckpointId source) {
  return impl_->begin(source);
}

kv_cache::Allocation PersistentCacheManager::acquire_speculative_staging(kv_cache::ExecutionId execution, std::size_t bytes) {
  return impl_->acquire_speculative_staging(execution, bytes);
}

void PersistentCacheManager::release_speculative_staging(kv_cache::ExecutionId execution) {
  return impl_->release_speculative_staging(execution);
}

void PersistentCacheManager::release(kv_cache::ExecutionId execution) {
  return impl_->release(execution);
}

std::uint32_t PersistentCacheManager::processed_tokens(kv_cache::ExecutionId execution) const {
  return impl_->processed_tokens(execution);
}

void PersistentCacheManager::restore_terminal_hidden(kv_cache::ExecutionId execution, TerminalState destination, CompletionContext stream) const {
  return impl_->restore_terminal_hidden(execution, destination, stream);
}

kv_cache::WritePlan PersistentCacheManager::prepare_write(kv_cache::ExecutionId execution, std::uint32_t first_token, std::uint32_t token_count, CompletionContext stream) {
  return impl_->prepare_write(execution, first_token, token_count, stream);
}

kv_cache::CheckpointId PersistentCacheManager::try_capture(kv_cache::ExecutionId execution, const std::vector<std::uint32_t>& tokens, TerminalState terminal, CompletionContext stream, prefix_index::CheckpointSource source, bool automatic_demand, prefix_index::RetentionPriority retention_priority, bool may_write_after_capture, bool required_retention, const std::vector<prefix_index::ImageSpan>& images) {
  return impl_->try_capture(execution, tokens, terminal, stream, source, automatic_demand, retention_priority, may_write_after_capture, required_retention, images);
}

std::size_t PersistentCacheManager::execution_reservation_bytes() const {
  return impl_->execution_reservation_bytes();
}

std::size_t PersistentCacheManager::execution_reservation_index_bytes() const {
  return impl_->execution_reservation_index_bytes();
}

std::size_t PersistentCacheManager::cold_spill_bytes() const {
  return impl_->cold_spill_bytes();
}

std::size_t PersistentCacheManager::cold_restore_bytes() const {
  return impl_->cold_restore_bytes();
}

std::size_t PersistentCacheManager::cold_spill_count() const {
  return impl_->cold_spill_count();
}

std::size_t PersistentCacheManager::cold_restore_count() const {
  return impl_->cold_restore_count();
}

int checkpoint_source_rank(CheckpointSource source) {
  switch (source) {
    case CheckpointSource::input:
      return 3;
    case CheckpointSource::learned_branch:
      return 2;
    case CheckpointSource::periodic:
      return 1;
  }
  return 0;
}

std::vector<CheckpointTrigger> make_checkpoint_triggers(
    std::uint32_t prompt_tokens, std::uint32_t maximum_processed_tokens,
    std::uint32_t resumed_tokens, std::uint32_t longest_matching_tokens,
    std::uint32_t checkpoint_interval_tokens,
    const std::vector<std::uint32_t>& nominated_offsets,
    std::size_t capacity) {
  if (nominated_offsets.size() > kMaximumCheckpointOffsets) {
    fail("generation checkpoint", "too many nominated checkpoint offsets");
  }
  std::array<CheckpointTrigger, kMaximumCheckpointOffsets + 2> extras{};
  std::size_t extra_count = 0;
  const auto add_extra = [&extras, &extra_count](std::uint32_t position,
                                                  CheckpointSource source) {
    if (position == 0) {
      return;
    }
    if (extra_count == extras.size()) {
      fail("generation checkpoint", "checkpoint trigger extras overflow");
    }
    extras[extra_count++] = {position, source};
  };

  // Every ordinary prompt has an input endpoint. A shared boundary discovered
  // while replaying is retained too, unless it coincides with that endpoint.
  add_extra(prompt_tokens, CheckpointSource::input);
  if (longest_matching_tokens > resumed_tokens &&
      longest_matching_tokens <= prompt_tokens) {
    add_extra(longest_matching_tokens, CheckpointSource::learned_branch);
  }
  for (const std::uint32_t offset : nominated_offsets) {
    if (offset <= prompt_tokens) {
      add_extra(offset, CheckpointSource::learned_branch);
    }
  }

  std::sort(extras.begin(), extras.begin() + extra_count,
            [](const CheckpointTrigger& left, const CheckpointTrigger& right) {
              return left.processed_tokens < right.processed_tokens;
            });
  std::size_t unique_extras = 0;
  for (std::size_t index = 0; index < extra_count; ++index) {
    const CheckpointTrigger candidate = extras[index];
    if (unique_extras != 0 &&
        extras[unique_extras - 1].processed_tokens ==
            candidate.processed_tokens) {
      if (checkpoint_source_rank(candidate.source) >
          checkpoint_source_rank(extras[unique_extras - 1].source)) {
        extras[unique_extras - 1].source = candidate.source;
      }
      continue;
    }
    extras[unique_extras++] = candidate;
  }

  std::uint64_t first_periodic = 0;
  std::size_t periodic_count = 0;
  if (checkpoint_interval_tokens != 0) {
    first_periodic =
        (static_cast<std::uint64_t>(resumed_tokens) /
             checkpoint_interval_tokens +
         1) *
        checkpoint_interval_tokens;
    if (first_periodic <= maximum_processed_tokens) {
      periodic_count = static_cast<std::size_t>(
          (maximum_processed_tokens - first_periodic) /
              checkpoint_interval_tokens +
          1);
    }
  }

  std::size_t extra_index = 0;
  std::size_t periodic_index = 0;
  std::size_t trigger_count = 0;
  while (extra_index < unique_extras || periodic_index < periodic_count) {
    const std::uint64_t extra_position =
        extra_index == unique_extras
            ? std::numeric_limits<std::uint64_t>::max()
            : extras[extra_index].processed_tokens;
    const std::uint64_t periodic_position =
        periodic_index == periodic_count
            ? std::numeric_limits<std::uint64_t>::max()
            : first_periodic +
                  static_cast<std::uint64_t>(periodic_index) *
                      checkpoint_interval_tokens;
    if (trigger_count == capacity) {
      fail_invalid_capacity(
          "server cache",
          "declared checkpoint bookkeeping exceeds the configured metadata budget");
    }
    ++trigger_count;
    if (extra_position <= periodic_position) {
      ++extra_index;
    }
    if (periodic_position <= extra_position) {
      ++periodic_index;
    }
  }

  std::vector<CheckpointTrigger> triggers;
  triggers.reserve(trigger_count);
  extra_index = 0;
  periodic_index = 0;
  while (extra_index < unique_extras || periodic_index < periodic_count) {
    const std::uint64_t extra_position =
        extra_index == unique_extras
            ? std::numeric_limits<std::uint64_t>::max()
            : extras[extra_index].processed_tokens;
    const std::uint64_t periodic_position =
        periodic_index == periodic_count
            ? std::numeric_limits<std::uint64_t>::max()
            : first_periodic +
                  static_cast<std::uint64_t>(periodic_index) *
                      checkpoint_interval_tokens;
    if (extra_position <= periodic_position) {
      triggers.push_back(extras[extra_index++]);
      if (periodic_position == extra_position) {
        ++periodic_index;
      }
    } else {
      triggers.push_back(
          {static_cast<std::uint32_t>(periodic_position),
           CheckpointSource::periodic});
      ++periodic_index;
    }
  }
  return triggers;
}


}  // namespace gewell::runtime
