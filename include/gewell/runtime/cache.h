#pragma once

#include "gewell/kv_cache.h"
#include "gewell/prefix_index.h"
#include "gewell/runtime/cache_storage.h"
#include "json.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gewell::runtime {

inline constexpr std::size_t kMaximumCacheOwnerBytes = 128;

enum class CheckpointSource : std::uint8_t { input, periodic, learned_branch };
struct CheckpointTrigger {
  std::uint32_t processed_tokens{};
  CheckpointSource source{CheckpointSource::input};
};
struct CapturedCheckpoint {
  std::uint32_t processed_tokens{};
  kv_cache::CheckpointId checkpoint{};
};

inline constexpr std::size_t kMaximumCheckpointOffsets = 1'024;
int checkpoint_source_rank(CheckpointSource source);
std::vector<CheckpointTrigger> make_checkpoint_triggers(
    std::uint32_t prompt_tokens, std::uint32_t maximum_processed_tokens,
    std::uint32_t resumed_tokens, std::uint32_t longest_matching_tokens,
    std::uint32_t checkpoint_interval_tokens,
    const std::vector<std::uint32_t>& nominated_offsets,
    std::size_t capacity);

class CacheCapacityError : public std::runtime_error {
 public:
  CacheCapacityError(std::string message, bool request_invalid)
      : std::runtime_error(std::move(message)), request_invalid(request_invalid) {}
  bool request_invalid;
};
using CacheEventSink = std::function<void(const nlohmann::json&)>;

enum class CacheRequestMode : std::uint8_t { automatic, reuse_only };

struct CacheRequestControls {
  CacheRequestMode mode{CacheRequestMode::automatic};
  std::string prompt_id;
  prefix_index::RetentionPriority priority{
      prefix_index::RetentionPriority::normal};
  bool finished{};

  [[nodiscard]] bool named() const { return !prompt_id.empty(); }
  [[nodiscard]] bool admits_retained_state() const {
    return mode == CacheRequestMode::automatic;
  }
};

struct CaptureTailCowHeadroom {
  bool keeps_pending_tail{};
  bool reserves_new_tail{};

  [[nodiscard]] std::size_t page_count() const {
    return static_cast<std::size_t>(keeps_pending_tail) +
           static_cast<std::size_t>(reserves_new_tail);
  }
};

[[nodiscard]] CaptureTailCowHeadroom capture_tail_cow_headroom(
    bool reserve_future_writes, bool partial_tail,
    kv_cache::PageId tail_page, bool base_tail_cow_reserved,
    kv_cache::PageId pending_tail_page);

// Lifecycle telemetry stays fixed-size. A checkpoint may have multiple sources.
inline constexpr std::size_t kLifecycleSourceMaskCount = 16;
inline constexpr std::size_t kLifecyclePriorityCount = 3;
inline constexpr std::size_t kLifecycleClassCount = 2;
inline constexpr std::size_t kLifecycleBucketCount =
    kLifecycleSourceMaskCount * kLifecyclePriorityCount *
    kLifecycleClassCount;

struct CheckpointLifecycleBucket {
  std::size_t admissions{};
  std::size_t hits{};
  std::size_t removals{};
  std::size_t admitted_bytes{};
  std::size_t reclaimed_gpu_bytes{};
  std::size_t reclaimed_cpu_bytes{};

  [[nodiscard]] bool empty() const {
    return admissions == 0 && hits == 0 && removals == 0 &&
           admitted_bytes == 0 && reclaimed_gpu_bytes == 0 &&
           reclaimed_cpu_bytes == 0;
  }
};

struct CacheTelemetry {
  std::array<CheckpointLifecycleBucket, kLifecycleBucketCount>
      checkpoint_buckets{};
  std::size_t gpu_reclaimed_bytes{};
  std::size_t cpu_reclaimed_bytes{};
  std::size_t cold_spill_bytes{};
  std::size_t cold_restore_bytes{};
  std::size_t cold_spill_count{};
  std::size_t cold_restore_count{};
  std::size_t cold_spill_avoided_rewrite_bytes{};
  double cold_spill_wall_milliseconds{};
  double cold_restore_wall_milliseconds{};
};

[[nodiscard]] std::size_t checked_telemetry_delta(std::size_t after,
                                                   std::size_t before,
                                                   std::string_view label);

[[nodiscard]] double checked_telemetry_delta(double after, double before,
                                              std::string_view label);

[[nodiscard]] CacheTelemetry telemetry_delta(const CacheTelemetry& after,
                                              const CacheTelemetry& before);

[[nodiscard]] std::size_t lifecycle_bucket_index(
    prefix_index::CheckpointSources sources,
    prefix_index::RetentionPriority priority,
    prefix_index::CacheClass cache_class);

void add_telemetry(std::size_t* destination, std::size_t value,
                   std::string_view label);

void add_telemetry(double* destination, double value, std::string_view label);

[[nodiscard]] const char* retention_priority_name(
    prefix_index::RetentionPriority priority);

[[nodiscard]] const char* cache_class_name(prefix_index::CacheClass value);

void write_checkpoint_sources(std::ostream& output,
                              prefix_index::CheckpointSources sources);

[[nodiscard]] std::string checkpoint_sources_name(
    prefix_index::CheckpointSources sources);

void write_checkpoint_buckets(std::ostream& output,
                              const CacheTelemetry& telemetry);

class PersistentCacheManager {
 public:

  struct OwnerDemandReservation {
    std::size_t slots{};
  };

  struct OwnerDemandCommit {
    struct Change {
      kv_cache::CheckpointId checkpoint{};
      prefix_index::RetentionPriority previous_priority{
          prefix_index::RetentionPriority::normal};
      bool existed{};
    };

    std::vector<Change> changes;
  };

  PersistentCacheManager(kv_cache::PoolConfig config,
                         CacheStorageFactory storage_factory,
                         CacheEventSink events = {});
  ~PersistentCacheManager();
  PersistentCacheManager(const PersistentCacheManager&) = delete;
  PersistentCacheManager& operator=(const PersistentCacheManager&) = delete;

  CacheStorage& storage();
  [[nodiscard]] const kv_cache::PoolConfig& config() const;
  [[nodiscard]] std::size_t active_checkpoint_capacity(bool named) const;
  [[nodiscard]] kv_cache::CacheStats stats() const;
  [[nodiscard]] prefix_index::MetadataStats prefix_stats() const;
  [[nodiscard]] nlohmann::json observability_snapshot(
      bool include_entries = true) const;
  [[nodiscard]] CacheTelemetry telemetry() const;
  [[nodiscard]] bool is_cold_checkpoint(
      kv_cache::CheckpointId checkpoint) const;
  [[nodiscard]] std::size_t gpu_page_slack_bytes() const;
  [[nodiscard]] std::size_t cpu_page_slack_bytes() const;
  [[nodiscard]] prefix_index::LookupResult find_longest(
      const std::vector<std::uint32_t>& tokens,
      const std::vector<prefix_index::ImageSpan>& images = {}) const;
  [[nodiscard]] bool checkpoint_matches(
      kv_cache::CheckpointId checkpoint,
      const std::vector<std::uint32_t>& tokens,
      const std::vector<prefix_index::ImageSpan>& images = {}) const;
  [[nodiscard]] prefix_index::LookupResult find_batch_prefix(
      const std::vector<std::uint32_t>& tokens,
      std::size_t maximum_processed_tokens,
      const std::vector<prefix_index::ImageSpan>& images = {}) const;
  [[nodiscard]] bool has_checkpoint(kv_cache::CheckpointId checkpoint) const;
  [[nodiscard]] std::uint32_t checkpoint_tokens(
      kv_cache::CheckpointId checkpoint) const;
  void mark_used(kv_cache::CheckpointId checkpoint);
  void add_checkpoint_source(kv_cache::CheckpointId checkpoint,
                             const std::vector<std::uint32_t>& tokens,
                             prefix_index::CheckpointSource source,
                             const std::vector<prefix_index::ImageSpan>& images = {});
  [[nodiscard]] bool add_automatic_demand(
      kv_cache::CheckpointId checkpoint,
      prefix_index::RetentionPriority priority =
          prefix_index::RetentionPriority::normal);
  [[nodiscard]] bool add_owner_demands(
      const std::vector<kv_cache::CheckpointId>& checkpoints,
      std::string_view owner, prefix_index::RetentionPriority priority,
      OwnerDemandReservation* reservation = nullptr,
      OwnerDemandCommit* commit = nullptr);
  void rollback_owner_demands(std::string_view owner,
                              OwnerDemandCommit* commit);
  [[nodiscard]] bool reserve_owner_demand_slots(
      std::size_t slots, OwnerDemandReservation* reservation);
  [[nodiscard]] std::size_t owner_demand_slot_capacity() const;
  void release_owner_demand_reservation(OwnerDemandReservation* reservation);
  void reclaim_undemanded(kv_cache::CheckpointId checkpoint);
  void release_owner(std::string_view owner);
  void ensure_capacity(kv_cache::CheckpointId source,
                       std::size_t total_processed_tokens,
                       std::size_t speculative_bytes = 0);
  [[nodiscard]] std::optional<kv_cache::ExecutionId> try_begin_batch(
      kv_cache::CheckpointId source, std::size_t max_processed_tokens,
      CompletionContext stream);
  std::optional<kv_cache::ExecutionId> try_fork_batch(
      kv_cache::ExecutionId source, std::size_t horizon, CompletionContext stream);
  bool try_resize_batch(kv_cache::ExecutionId execution, std::size_t horizon);
  void pin_batch_checkpoint(kv_cache::CheckpointId checkpoint);
  void unpin_batch_checkpoint(kv_cache::CheckpointId checkpoint);
  kv_cache::ExecutionId begin(kv_cache::CheckpointId source = 0);
  [[nodiscard]] kv_cache::Allocation acquire_speculative_staging(
      kv_cache::ExecutionId execution, std::size_t bytes);
  void release_speculative_staging(kv_cache::ExecutionId execution);
  void release(kv_cache::ExecutionId execution);
  [[nodiscard]] std::uint32_t processed_tokens(
      kv_cache::ExecutionId execution) const;
  void restore_terminal_hidden(kv_cache::ExecutionId execution,
                               TerminalState destination,
                               CompletionContext stream) const;
  kv_cache::WritePlan prepare_write(kv_cache::ExecutionId execution,
                                    std::uint32_t first_token,
                                    std::uint32_t token_count,
                                    CompletionContext stream);
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
      const std::vector<prefix_index::ImageSpan>& images = {});
  [[nodiscard]] std::size_t execution_reservation_bytes() const;
  [[nodiscard]] std::size_t execution_reservation_index_bytes() const;
  [[nodiscard]] std::size_t cold_spill_bytes() const;
  [[nodiscard]] std::size_t cold_restore_bytes() const;
  [[nodiscard]] std::size_t cold_spill_count() const;
  [[nodiscard]] std::size_t cold_restore_count() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gewell::runtime
