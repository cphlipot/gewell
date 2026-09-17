#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace gewell::prefix_index {

// This intentionally mirrors the cache ledger's identifier representation
// without depending on the ledger. The server can pass kv_cache::CheckpointId
// values directly because both are uint64_t.
using CheckpointId = std::uint64_t;

// Image feature tokens retain their real positions. The digest identifies the
// prepared encoder input; the span is indivisible for checkpoint purposes.
struct ImageSpan {
  std::uint32_t begin{}, end{};
  std::array<std::uint8_t, 32> digest{};
};

inline constexpr std::size_t kMaxUnpromotedPeriodicPerPath = 8;

// More than one trigger may name the same checkpoint. This matters when, for
// example, an input endpoint is also a periodic boundary: it remains an
// endpoint and must not be thinned as a periodic-only candidate.
enum class CheckpointSource : std::uint8_t {
  input_endpoint = 1U << 0,
  continuation_endpoint = 1U << 1,
  learned_branch = 1U << 2,
  periodic = 1U << 3,
};

using CheckpointSources = std::uint8_t;

constexpr CheckpointSources checkpoint_source_mask(CheckpointSource source) {
  return static_cast<CheckpointSources>(source);
}

constexpr bool has_checkpoint_source(CheckpointSources sources,
                                     CheckpointSource source) {
  return (sources & checkpoint_source_mask(source)) != 0;
}

// Explicit retention demands are ordered from least to most protected. They
// influence eviction only; none of these levels pins a checkpoint.
enum class RetentionPriority : std::uint8_t {
  low = 0,
  normal = 1,
  high = 2,
};

enum class CacheClass : std::uint8_t { probationary, reused };

struct CheckpointMetadata {
  CheckpointId checkpoint{};
  std::uint32_t prefix_tokens{};
  CheckpointSources sources{};
  RetentionPriority priority{RetentionPriority::normal};
  CacheClass cache_class{CacheClass::probationary};
  bool reference_bit{};
  std::uint64_t admission_order{};
  // Successful execution borrows; metadata lookups do not count as reuse.
  std::uint64_t reuse_count{};
};

struct ThinnedCheckpoint {
  CheckpointId checkpoint{};
  CheckpointMetadata metadata{};
};

class PrefixIndex;

// Periodic thinning preserves at most eight probationary periodic records per
// path. Keep its result inline so automatic captures never allocate host
// bookkeeping after a streaming response has started.
class ThinnedPeriodic {
 public:
  [[nodiscard]] bool empty() const noexcept { return count_ == 0; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }
  [[nodiscard]] const ThinnedCheckpoint& front() const noexcept {
    return entries_.front();
  }
  [[nodiscard]] ThinnedCheckpoint* begin() noexcept {
    return entries_.data();
  }
  [[nodiscard]] ThinnedCheckpoint* end() noexcept {
    return entries_.data() + count_;
  }
  [[nodiscard]] const ThinnedCheckpoint* begin() const noexcept {
    return entries_.data();
  }
  [[nodiscard]] const ThinnedCheckpoint* end() const noexcept {
    return entries_.data() + count_;
  }

 private:
  friend class PrefixIndex;

  [[nodiscard]] bool append(ThinnedCheckpoint checkpoint) noexcept {
    if (count_ == entries_.size()) {
      return false;
    }
    entries_[count_++] = checkpoint;
    return true;
  }

  std::array<ThinnedCheckpoint, kMaxUnpromotedPeriodicPerPath> entries_{};
  std::size_t count_{};
};

struct LookupResult {
  // Deepest matching token boundary outside an image span, regardless of
  // whether a usable checkpoint existed there. This is useful for replay
  // diagnostics and learned checkpoints.
  std::uint32_t longest_matching_tokens{};

  // The deepest caller-approved checkpoint. Zero means that no usable
  // checkpoint was found. A caller should call mark_used() only after it has
  // successfully borrowed this state for execution.
  CheckpointId checkpoint{};
  std::uint32_t checkpoint_tokens{};

  [[nodiscard]] bool has_checkpoint() const { return checkpoint != 0; }
};

enum class AdmissionStatus : std::uint8_t {
  admitted,
  already_present,
  insufficient_metadata,
  protected_periodic,
};

struct AdmissionResult {
  AdmissionStatus status{AdmissionStatus::insufficient_metadata};

  // Records removed from this index by periodic thinning. The caller owns the
  // corresponding cache lifecycle and can release their physical state after
  // receiving these IDs. Metadata is captured before index removal.
  ThinnedPeriodic thinned_periodic;

  [[nodiscard]] bool retained() const {
    return status == AdmissionStatus::admitted || status == AdmissionStatus::already_present;
  }
};

struct MetadataStats {
  std::size_t capacity_bytes{};
  std::size_t used_bytes{};
  std::size_t node_count{};
  std::size_t checkpoint_count{};
  std::size_t unpromoted_periodic_count{};
};

// A bounded host-side trie of exact token and image histories. It owns no KV
// state: the supplied usability predicate determines whether a checkpoint is currently
// valid/resident, and callers remove entries when their cache manager drops
// the matching state.
class PrefixIndex {
 public:
  using UsableCheckpoint = std::function<bool(CheckpointId)>;
  // The manager supplies the bytes that would become reclaimable if an idle
  // checkpoint were removed, so each priority's 75/25 reused/probation target
  // follows physical storage rather than checkpoint count.
  using ReclaimableBytes = std::function<std::size_t(CheckpointId)>;

  explicit PrefixIndex(std::size_t metadata_bytes);
  ~PrefixIndex();

  PrefixIndex(const PrefixIndex&) = delete;
  PrefixIndex& operator=(const PrefixIndex&) = delete;

  // Adds an automatic normal-priority checkpoint at an exact nonempty token
  // prefix. Existing token nodes are shared. Re-admitting the same ID/prefix
  // merges its source trigger without allocating another record. Reusing an
  // ID for a different prefix is an error.
  // A thinning predicate protects borrowed or pinned periodic checkpoints.
  // Admission can refuse another periodic candidate while all older ones are
  // protected, preserving the bounded number of candidates on an idle path.
  // Ordered image spans may include later images beyond this prefix; the
  // endpoint must not split an image. Image identity branches at feature begin.
  [[nodiscard]] AdmissionResult admit(
      CheckpointId checkpoint, const std::vector<std::uint32_t>& prefix,
      CheckpointSource source, const UsableCheckpoint& thinning_eligible = {},
      const std::vector<ImageSpan>& images = {});

  // Traverses the submitted token history once and returns its longest exact
  // trie prefix plus the deepest checkpoint accepted by `usable`.
  [[nodiscard]] LookupResult find_longest(
      const std::vector<std::uint32_t>& tokens,
      const UsableCheckpoint& usable,
      const std::vector<ImageSpan>& images = {}) const;

  // A real cache use promotes a candidate from probationary to reused and
  // sets its CLOCK reference bit. Lookup itself deliberately has no side
  // effect, so metadata traversal cannot manufacture a cache hit.
  [[nodiscard]] bool mark_used(CheckpointId checkpoint);

  // Changes the eviction priority for a retained checkpoint. Returns false
  // when the checkpoint is no longer indexed. The cache class and reference
  // bit are deliberately preserved: reprioritizing is not a new admission or
  // a cache use.
  [[nodiscard]] bool set_priority(CheckpointId checkpoint,
                                  RetentionPriority priority);

  // Selects, but does not remove, a CLOCK victim. It considers low, normal,
  // then high priority; within each priority it scans probationary entries
  // before reused entries and clears reference bits as it passes them. The
  // caller must release physical state and then call remove.
  [[nodiscard]] std::optional<CheckpointId> select_clock_victim();

  // The filtered form is for a cache manager with active borrowers or a
  // protected source checkpoint. Ineligible records are skipped before their
  // CLOCK state is touched; in particular, a scan cannot clear a reference bit
  // for state that the caller is not currently allowed to evict.
  [[nodiscard]] std::optional<CheckpointId> select_clock_victim(
      const UsableCheckpoint& eligible);

  // As above, but balances eligible GPU storage independently within each
  // priority toward the documented 75% reused / 25% probationary soft target
  // before choosing a victim. Reused entries above a priority's target are
  // demoted with CLOCK; remaining probationary entries retain eviction
  // precedence at that priority.
  [[nodiscard]] std::optional<CheckpointId> select_clock_victim(
      const UsableCheckpoint& eligible,
      const ReclaimableBytes& reclaimable_bytes);

  // Selects the oldest eligible probationary checkpoint whose closest
  // retained strict ancestor is no farther away than `maximum_distance`.
  // Priority ordering matches CLOCK. The selection is read-only so falling
  // back to CLOCK retains its existing second-chance behavior.
  [[nodiscard]] std::optional<CheckpointId> select_redundant_victim(
      std::uint32_t maximum_distance,
      const UsableCheckpoint& eligible) const;

  // Removes an advertised checkpoint. Empty token nodes are pruned and become
  // reusable, so deleting metadata makes room for later automatic admissions.
  [[nodiscard]] bool remove(CheckpointId checkpoint);

  // Applies periodic thinning to an existing exact trie path. Only
  // periodic-only probationary entries count toward the eight-entry limit.
  // The returned IDs have already been removed from this index.
  [[nodiscard]] ThinnedPeriodic thin_periodic(
      const std::vector<std::uint32_t>& prefix,
      const std::vector<ImageSpan>& images = {});

  [[nodiscard]] bool contains(CheckpointId checkpoint) const;
  [[nodiscard]] std::optional<CheckpointMetadata> metadata(
      CheckpointId checkpoint) const;
  // Returns the closest retained checkpoint on the same strict prefix path.
  // This exposes resident spacing without copying token histories.
  [[nodiscard]] std::optional<CheckpointMetadata> nearest_ancestor(
      CheckpointId checkpoint) const;
  [[nodiscard]] MetadataStats stats() const;

 private:
  struct TrieNode;
  struct Record;
  struct Slot;
  enum class SlotKind : std::uint8_t;

  static constexpr std::uint32_t kNoIndex = 0xffff'ffffU;

  [[nodiscard]] std::uint32_t find_child(std::uint32_t parent,
                                         std::uint32_t token,
                                         const ImageSpan* image) const;
  [[nodiscard]] std::uint32_t find_node(
      const std::vector<std::uint32_t>& prefix,
      const std::vector<ImageSpan>& images) const;
  [[nodiscard]] std::uint32_t find_record(CheckpointId checkpoint) const;
  [[nodiscard]] std::uint32_t allocate_node(std::uint32_t token,
                                             std::uint32_t parent,
                                             const ImageSpan* image);
  [[nodiscard]] std::uint32_t allocate_record(
      CheckpointId checkpoint, std::uint32_t node, CheckpointSource source);
  [[nodiscard]] std::uint32_t take_slot(SlotKind kind);
  void release_slot(std::uint32_t slot);
  [[nodiscard]] TrieNode& node_ref(std::uint32_t index);
  [[nodiscard]] const TrieNode& node_ref(std::uint32_t index) const;
  [[nodiscard]] Record& record_ref(std::uint32_t index);
  [[nodiscard]] const Record& record_ref(std::uint32_t index) const;
  [[nodiscard]] std::uint32_t next_clock_record();
  void rebalance_reused(RetentionPriority priority,
                        const UsableCheckpoint& eligible,
                        const ReclaimableBytes& reclaimable_bytes);
  void erase_record(std::uint32_t record);
  void prune_empty_nodes(std::uint32_t node);
  [[nodiscard]] bool is_unpromoted_periodic(const Record& record) const;
  [[nodiscard]] ThinnedPeriodic thin_periodic_from_node(
      std::uint32_t node, const UsableCheckpoint& eligible);
  [[nodiscard]] bool has_room(std::size_t nodes,
                              std::size_t records) const;
  [[nodiscard]] static std::size_t checked_add(std::size_t left,
                                               std::size_t right,
                                               const char* label);
  [[nodiscard]] static std::size_t checked_mul(std::size_t left,
                                               std::size_t right,
                                               const char* label);
  [[nodiscard]] static std::uint32_t checked_tokens(std::size_t count,
                                                     const char* label);

  std::size_t metadata_bytes_{};
  std::size_t used_bytes_{};
  std::size_t active_nodes_{};
  std::size_t active_records_{};
  std::uint64_t next_admission_order_{1};
  std::size_t slot_capacity_{};
  std::uint32_t clock_hand_{kNoIndex};
  std::uint32_t free_slot_{kNoIndex};
  std::uint32_t active_record_head_{kNoIndex};
  std::unique_ptr<Slot[]> slots_;
};

}  // namespace gewell::prefix_index
