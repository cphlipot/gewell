#include "gewell/prefix_index.h"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace gewell::prefix_index {
namespace {

[[noreturn]] void fail(std::string_view operation, std::string_view detail) {
  throw std::runtime_error(std::string(operation) + ": " +
                           std::string(detail));
}

bool is_known_source(CheckpointSource source) {
  switch (source) {
    case CheckpointSource::input_endpoint:
    case CheckpointSource::continuation_endpoint:
    case CheckpointSource::learned_branch:
    case CheckpointSource::periodic:
      return true;
  }
  return false;
}

bool is_known_priority(RetentionPriority priority) {
  switch (priority) {
    case RetentionPriority::low:
    case RetentionPriority::normal:
    case RetentionPriority::high:
      return true;
  }
  return false;
}

void validate_images(const std::vector<ImageSpan>& images,
                     std::size_t prefix_tokens, bool checkpoint) {
  std::uint32_t previous_end = 0;
  for (const auto& image : images) {
    if (image.begin < previous_end || image.begin >= image.end) {
      fail("prefix index image", "image spans must be nonempty and ordered");
    }
    if (checkpoint && image.begin < prefix_tokens && prefix_tokens < image.end) {
      fail("prefix index image", "checkpoint cannot end inside an image span");
    }
    previous_end = image.end;
  }
}

const ImageSpan* image_at(const std::vector<ImageSpan>& images,
                         std::size_t position, std::size_t* index) {
  while (*index < images.size() && images[*index].end <= position) ++*index;
  return *index < images.size() && images[*index].begin == position
             ? &images[*index] : nullptr;
}

}  // namespace

struct PrefixIndex::TrieNode {
  std::uint32_t token{};
  std::uint32_t image_slot{kNoIndex};
  std::uint32_t parent{kNoIndex};
  std::uint32_t first_child{kNoIndex};
  std::uint32_t next_sibling{kNoIndex};
  std::uint32_t first_record{kNoIndex};
  std::uint32_t depth{};
};

struct PrefixIndex::Record {
  CheckpointId checkpoint{};
  std::uint32_t node{kNoIndex};
  std::uint32_t next_at_node{kNoIndex};
  std::uint32_t next_active{kNoIndex};
  std::uint32_t previous_active{kNoIndex};
  CheckpointSources sources{};
  RetentionPriority priority{RetentionPriority::normal};
  CacheClass cache_class{CacheClass::probationary};
  bool reference_bit{};
  std::uint64_t admission_order{};
  std::uint64_t reuse_count{};
};

enum class PrefixIndex::SlotKind : std::uint8_t { free, node, record, image };

struct PrefixIndex::Slot {
  // Image identities use their own budgeted slot without growing text slots.
  static_assert(sizeof(TrieNode) <= sizeof(Record));
  static_assert(sizeof(ImageSpan) <= sizeof(Record));
  union Storage {
    TrieNode node;
    Record record;
    ImageSpan image;
    std::uint32_t next_free;

    Storage() {}
    ~Storage() {}
  } storage;
  SlotKind kind{SlotKind::free};
};

PrefixIndex::PrefixIndex(std::size_t metadata_bytes)
    : metadata_bytes_(metadata_bytes),
      slot_capacity_(metadata_bytes_ / sizeof(Slot)) {
  if (slot_capacity_ == 0) {
    fail("prefix index configuration", "metadata budget cannot hold a slot");
  }
  if (slot_capacity_ > kNoIndex) {
    fail("prefix index configuration", "metadata budget exceeds index space");
  }
  slots_ = std::make_unique<Slot[]>(slot_capacity_);
  for (std::size_t index = 0; index < slot_capacity_; ++index) {
    Slot& slot = slots_[index];
    slot.kind = SlotKind::free;
    slot.storage.next_free =
        index + 1 == slot_capacity_ ? kNoIndex
                                    : static_cast<std::uint32_t>(index + 1);
  }
  free_slot_ = 0;

  const std::uint32_t root_index = take_slot(SlotKind::node);
  if (root_index != 0) {
    fail("prefix index configuration", "root did not receive the first slot");
  }
  TrieNode root;
  root.parent = kNoIndex;
  root.first_child = kNoIndex;
  root.next_sibling = kNoIndex;
  root.first_record = kNoIndex;
  root.depth = 0;
  new (&slots_[root_index].storage.node) TrieNode(root);
  active_nodes_ = 1;
  used_bytes_ = sizeof(Slot);
}

PrefixIndex::~PrefixIndex() = default;

AdmissionResult PrefixIndex::admit(
    CheckpointId checkpoint, const std::vector<std::uint32_t>& prefix,
    CheckpointSource source, const UsableCheckpoint& thinning_eligible,
    const std::vector<ImageSpan>& images) {
  if (checkpoint == 0) {
    fail("prefix index admission", "checkpoint ID must be positive");
  }
  if (prefix.empty()) {
    fail("prefix index admission", "checkpoint prefix must be nonempty");
  }
  (void)checked_tokens(prefix.size(), "prefix index admission");
  validate_images(images, prefix.size(), true);
  if (!is_known_source(source)) {
    fail("prefix index admission", "checkpoint source is invalid");
  }

  std::uint32_t node = 0;
  std::size_t matched = 0;
  std::size_t image_index = 0;
  while (matched < prefix.size()) {
    const std::uint32_t child = find_child(
        node, prefix[matched], image_at(images, matched, &image_index));
    if (child == kNoIndex) {
      break;
    }
    node = child;
    ++matched;
  }

  const std::uint32_t existing = find_record(checkpoint);
  if (existing != kNoIndex) {
    Record& record = record_ref(existing);
    if (matched != prefix.size() || record.node != node) {
      fail("prefix index admission",
           "checkpoint ID already names a different token prefix");
    }
    record.sources = static_cast<CheckpointSources>(
        record.sources | checkpoint_source_mask(source));
    AdmissionResult result;
    result.status = AdmissionStatus::already_present;
    if (source == CheckpointSource::periodic) {
      result.thinned_periodic = thin_periodic_from_node(node, thinning_eligible);
    }
    return result;
  }

  // Do not admit a ninth candidate that cannot be thinned without releasing
  // an active source. Refusing it here leaves every existing record unchanged.
  if (source == CheckpointSource::periodic && thinning_eligible) {
    std::size_t candidates = 0;
    bool can_thin = false;
    for (auto ancestor = node;; ancestor = node_ref(ancestor).parent) {
      for (auto index = node_ref(ancestor).first_record; index != kNoIndex;
           index = record_ref(index).next_at_node) {
        const auto& record = record_ref(index);
        if (!is_unpromoted_periodic(record)) continue;
        ++candidates;
        can_thin |= thinning_eligible(record.checkpoint);
      }
      if (ancestor == 0) break;
    }
    if (candidates >= kMaxUnpromotedPeriodicPerPath && !can_thin) {
      AdmissionResult result;
      result.status = AdmissionStatus::protected_periodic;
      return result;
    }
  }
  const std::size_t new_nodes = prefix.size() - matched;
  const auto new_images = static_cast<std::size_t>(std::count_if(
      images.begin(), images.end(), [&](const ImageSpan& image) {
        return image.begin >= matched && image.begin < prefix.size();
      }));
  if (!has_room(checked_add(new_nodes, new_images, "prefix index admission"), 1)) {
    return {};
  }
  for (; matched < prefix.size(); ++matched) {
    node = allocate_node(prefix[matched], node,
                         image_at(images, matched, &image_index));
  }
  (void)allocate_record(checkpoint, node, source);

  AdmissionResult result;
  result.status = AdmissionStatus::admitted;
  if (source == CheckpointSource::periodic) {
    result.thinned_periodic = thin_periodic_from_node(node, thinning_eligible);
  }
  return result;
}

LookupResult PrefixIndex::find_longest(
    const std::vector<std::uint32_t>& tokens,
    const UsableCheckpoint& usable,
    const std::vector<ImageSpan>& images) const {
  if (!usable) {
    fail("prefix index lookup", "usable-checkpoint predicate is empty");
  }
  validate_images(images, tokens.size(), false);

  LookupResult result;
  std::uint32_t node = 0;
  std::size_t image_index = 0;
  for (std::size_t position = 0; position < tokens.size(); ++position) {
    const std::uint32_t child = find_child(
        node, tokens[position], image_at(images, position, &image_index));
    if (child == kNoIndex) {
      break;
    }
    node = child;
    const TrieNode& current = node_ref(node);
    if (image_index < images.size() &&
        images[image_index].begin < current.depth &&
        current.depth < images[image_index].end) continue;
    result.longest_matching_tokens = current.depth;
    for (std::uint32_t index = current.first_record; index != kNoIndex;
         index = record_ref(index).next_at_node) {
      const Record& record = record_ref(index);
      if (usable(record.checkpoint)) {
        // Records attach at the head of a node list, so this selects the most
        // recently admitted usable state when several histories reach an
        // identical token boundary.
        result.checkpoint = record.checkpoint;
        result.checkpoint_tokens = current.depth;
        break;
      }
    }
  }
  return result;
}

bool PrefixIndex::mark_used(CheckpointId checkpoint) {
  const std::uint32_t index = find_record(checkpoint);
  if (index == kNoIndex) {
    return false;
  }
  Record& record = record_ref(index);
  if (record.reuse_count == std::numeric_limits<std::uint64_t>::max()) {
    fail("prefix index use", "checkpoint reuse count overflow");
  }
  ++record.reuse_count;
  record.cache_class = CacheClass::reused;
  record.reference_bit = true;
  return true;
}

bool PrefixIndex::set_priority(CheckpointId checkpoint,
                               RetentionPriority priority) {
  if (!is_known_priority(priority)) {
    fail("prefix index priority", "retention priority is invalid");
  }
  const std::uint32_t index = find_record(checkpoint);
  if (index == kNoIndex) {
    return false;
  }
  record_ref(index).priority = priority;
  return true;
}

std::optional<CheckpointId> PrefixIndex::select_clock_victim() {
  return select_clock_victim(
      [](CheckpointId) { return true; },
      [](CheckpointId) { return std::size_t{1}; });
}

std::optional<CheckpointId> PrefixIndex::select_clock_victim(
    const UsableCheckpoint& eligible) {
  return select_clock_victim(
      eligible, [](CheckpointId) { return std::size_t{1}; });
}

std::optional<CheckpointId> PrefixIndex::select_clock_victim(
    const UsableCheckpoint& eligible,
    const ReclaimableBytes& reclaimable_bytes) {
  if (!eligible) {
    fail("prefix index CLOCK", "eligibility predicate is empty");
  }
  if (!reclaimable_bytes) {
    fail("prefix index CLOCK", "reclaimable-bytes callback is empty");
  }
  if (active_records_ == 0) {
    return std::nullopt;
  }

  for (const RetentionPriority priority : {RetentionPriority::low,
                                           RetentionPriority::normal,
                                           RetentionPriority::high}) {
    rebalance_reused(priority, eligible, reclaimable_bytes);

    for (const CacheClass cache_class :
         {CacheClass::probationary, CacheClass::reused}) {
      // A full pass clears every reference bit in this class. A second pass
      // can then select an entry even when every candidate was recently used.
      for (std::size_t pass = 0; pass < 2; ++pass) {
        for (std::size_t scanned = 0; scanned < active_records_; ++scanned) {
          Record& record = record_ref(next_clock_record());
          if (record.priority != priority ||
              record.cache_class != cache_class ||
              !eligible(record.checkpoint)) {
            continue;
          }
          if (record.reference_bit) {
            record.reference_bit = false;
            continue;
          }
          return record.checkpoint;
        }
      }
    }
  }
  return std::nullopt;
}

std::optional<CheckpointId> PrefixIndex::select_redundant_victim(
    std::uint32_t maximum_distance,
    const UsableCheckpoint& eligible) const {
  if (!eligible) {
    fail("prefix index redundant selection", "eligibility predicate is empty");
  }

  for (const RetentionPriority priority : {RetentionPriority::low,
                                           RetentionPriority::normal,
                                           RetentionPriority::high}) {
    std::optional<CheckpointId> victim;
    std::uint32_t victim_distance = std::numeric_limits<std::uint32_t>::max();
    std::uint64_t victim_order = std::numeric_limits<std::uint64_t>::max();
    for (std::uint32_t index = active_record_head_; index != kNoIndex;
         index = record_ref(index).next_active) {
      const Record& candidate = record_ref(index);
      if (candidate.priority != priority ||
          candidate.cache_class != CacheClass::probationary ||
          !eligible(candidate.checkpoint)) {
        continue;
      }

      const std::uint32_t candidate_depth = node_ref(candidate.node).depth;
      std::uint32_t node = candidate.node;
      while (node != 0) {
        node = node_ref(node).parent;
        const std::uint32_t ancestor_depth = node_ref(node).depth;
        const std::uint32_t distance = candidate_depth - ancestor_depth;
        if (distance > maximum_distance) {
          break;
        }
        if (node_ref(node).first_record == kNoIndex) {
          continue;
        }
        if (distance < victim_distance ||
            (distance == victim_distance &&
             candidate.admission_order < victim_order)) {
          victim = candidate.checkpoint;
          victim_distance = distance;
          victim_order = candidate.admission_order;
        }
        break;
      }
    }
    if (victim.has_value()) {
      return victim;
    }
  }
  return std::nullopt;
}

void PrefixIndex::rebalance_reused(
    RetentionPriority priority,
    const UsableCheckpoint& eligible,
    const ReclaimableBytes& reclaimable_bytes) {
  std::size_t total_bytes = 0;
  std::size_t reused_bytes = 0;
  for (std::uint32_t index = active_record_head_; index != kNoIndex;
       index = record_ref(index).next_active) {
    const Record& record = record_ref(index);
    if (record.priority != priority || !eligible(record.checkpoint)) {
      continue;
    }
    const std::size_t bytes = reclaimable_bytes(record.checkpoint);
    total_bytes = checked_add(total_bytes, bytes,
                              "prefix index CLOCK byte accounting");
    if (record.cache_class == CacheClass::reused) {
      reused_bytes = checked_add(reused_bytes, bytes,
                                 "prefix index CLOCK byte accounting");
    }
  }
  if (total_bytes == 0) {
    return;
  }

  // Keep the target calculation overflow-safe: floor(total * 3 / 4).
  const std::size_t reused_target =
      (total_bytes / 4) * 3 + (total_bytes % 4) * 3 / 4;
  if (reused_bytes <= reused_target) {
    return;
  }

  // CLOCK's first observation of a referenced candidate grants it a second
  // chance; an unreferenced candidate is demoted to probationary. The next
  // eviction then still gives probationary state precedence.
  for (std::size_t pass = 0; pass < 2 && reused_bytes > reused_target;
       ++pass) {
    for (std::size_t scanned = 0;
         scanned < active_records_ && reused_bytes > reused_target;
         ++scanned) {
      Record& record = record_ref(next_clock_record());
      if (record.priority != priority ||
          record.cache_class != CacheClass::reused ||
          !eligible(record.checkpoint)) {
        continue;
      }
      if (record.reference_bit) {
        record.reference_bit = false;
        continue;
      }
      const std::size_t bytes = reclaimable_bytes(record.checkpoint);
      if (bytes == 0) {
        continue;
      }
      if (bytes > reused_bytes) {
        fail("prefix index CLOCK", "reclaimable-byte accounting underflow");
      }
      const std::size_t after_demotion = reused_bytes - bytes;
      // The target is soft: do not turn a single oversized reused entry into
      // an empty reused class when that would move farther from 75/25.
      if (after_demotion < reused_target &&
          reused_target - after_demotion > reused_bytes - reused_target) {
        continue;
      }
      record.cache_class = CacheClass::probationary;
      reused_bytes = after_demotion;
    }
  }
}

bool PrefixIndex::remove(CheckpointId checkpoint) {
  const std::uint32_t index = find_record(checkpoint);
  if (index == kNoIndex) {
    return false;
  }
  erase_record(index);
  return true;
}

ThinnedPeriodic PrefixIndex::thin_periodic(
    const std::vector<std::uint32_t>& prefix,
    const std::vector<ImageSpan>& images) {
  validate_images(images, prefix.size(), true);
  const std::uint32_t node = find_node(prefix, images);
  return node == kNoIndex ? ThinnedPeriodic{} : thin_periodic_from_node(node, {});
}

bool PrefixIndex::contains(CheckpointId checkpoint) const {
  return find_record(checkpoint) != kNoIndex;
}

std::optional<CheckpointMetadata> PrefixIndex::metadata(
    CheckpointId checkpoint) const {
  const std::uint32_t index = find_record(checkpoint);
  if (index == kNoIndex) {
    return std::nullopt;
  }
  const Record& record = record_ref(index);
  return CheckpointMetadata{record.checkpoint,
                            node_ref(record.node).depth,
                            record.sources,
                            record.priority,
                            record.cache_class,
                            record.reference_bit,
                            record.admission_order,
                            record.reuse_count};
}

std::optional<CheckpointMetadata> PrefixIndex::nearest_ancestor(
    CheckpointId checkpoint) const {
  const std::uint32_t index = find_record(checkpoint);
  if (index == kNoIndex) {
    return std::nullopt;
  }
  std::uint32_t node = record_ref(index).node;
  while (node != 0) {
    node = node_ref(node).parent;
    const std::uint32_t ancestor = node_ref(node).first_record;
    if (ancestor == kNoIndex) continue;
    const Record& record = record_ref(ancestor);
    return CheckpointMetadata{record.checkpoint,
                              node_ref(record.node).depth,
                              record.sources,
                              record.priority,
                              record.cache_class,
                              record.reference_bit,
                              record.admission_order,
                              record.reuse_count};
  }
  return std::nullopt;
}

MetadataStats PrefixIndex::stats() const {
  std::size_t unpromoted_periodic = 0;
  for (std::uint32_t index = active_record_head_; index != kNoIndex;
       index = record_ref(index).next_active) {
    if (is_unpromoted_periodic(record_ref(index))) {
      ++unpromoted_periodic;
    }
  }
  return {metadata_bytes_, used_bytes_, active_nodes_, active_records_,
          unpromoted_periodic};
}

std::uint32_t PrefixIndex::find_child(std::uint32_t parent,
                                      std::uint32_t token,
                                      const ImageSpan* image) const {
  const TrieNode& parent_node = node_ref(parent);
  for (std::uint32_t child = parent_node.first_child; child != kNoIndex;
       child = node_ref(child).next_sibling) {
    const auto& node = node_ref(child);
    if (node.token != token || (node.image_slot != kNoIndex) != (image != nullptr))
      continue;
    if (!image) return child;
    const auto& stored = slots_[node.image_slot].storage.image;
    if (stored.end == image->end && stored.digest == image->digest) return child;
  }
  return kNoIndex;
}

std::uint32_t PrefixIndex::find_node(
    const std::vector<std::uint32_t>& prefix,
    const std::vector<ImageSpan>& images) const {
  std::uint32_t node = 0;
  std::size_t image_index = 0;
  for (std::size_t position = 0; position < prefix.size(); ++position) {
    node = find_child(node, prefix[position], image_at(images, position, &image_index));
    if (node == kNoIndex) {
      return kNoIndex;
    }
  }
  return node;
}

std::uint32_t PrefixIndex::find_record(CheckpointId checkpoint) const {
  if (checkpoint == 0) {
    return kNoIndex;
  }
  for (std::uint32_t index = active_record_head_; index != kNoIndex;
       index = record_ref(index).next_active) {
    if (record_ref(index).checkpoint == checkpoint) {
      return index;
    }
  }
  return kNoIndex;
}

std::uint32_t PrefixIndex::allocate_node(std::uint32_t token,
                                         std::uint32_t parent,
                                         const ImageSpan* image) {
  TrieNode& parent_node = node_ref(parent);
  if (parent_node.depth == std::numeric_limits<std::uint32_t>::max()) {
    fail("prefix index node", "token prefix length overflows uint32");
  }

  TrieNode node;
  node.token = token;
  if (image) {
    node.image_slot = take_slot(SlotKind::image);
    new (&slots_[node.image_slot].storage.image) ImageSpan(*image);
    used_bytes_ = checked_add(used_bytes_, sizeof(Slot),
                              "prefix index image accounting");
  }
  node.parent = parent;
  node.first_child = kNoIndex;
  node.next_sibling = parent_node.first_child;
  node.first_record = kNoIndex;
  node.depth = parent_node.depth + 1;
  const std::uint32_t index = take_slot(SlotKind::node);
  new (&slots_[index].storage.node) TrieNode(node);
  parent_node.first_child = index;
  ++active_nodes_;
  used_bytes_ = checked_add(used_bytes_, sizeof(Slot),
                            "prefix index node accounting");
  return index;
}

std::uint32_t PrefixIndex::allocate_record(CheckpointId checkpoint,
                                           std::uint32_t node,
                                           CheckpointSource source) {
  TrieNode& target = node_ref(node);
  if (next_admission_order_ == std::numeric_limits<std::uint64_t>::max()) {
    fail("prefix index checkpoint", "admission order is exhausted");
  }

  Record record;
  record.checkpoint = checkpoint;
  record.node = node;
  record.next_at_node = target.first_record;
  record.next_active = active_record_head_;
  record.previous_active = kNoIndex;
  record.sources = checkpoint_source_mask(source);
  record.priority = RetentionPriority::normal;
  record.cache_class = CacheClass::probationary;
  record.reference_bit = false;
  record.admission_order = next_admission_order_++;
  const std::uint32_t index = take_slot(SlotKind::record);
  new (&slots_[index].storage.record) Record(record);
  if (active_record_head_ != kNoIndex) {
    record_ref(active_record_head_).previous_active = index;
  }
  active_record_head_ = index;
  target.first_record = index;
  ++active_records_;
  used_bytes_ = checked_add(used_bytes_, sizeof(Slot),
                            "prefix index checkpoint accounting");
  return index;
}

std::uint32_t PrefixIndex::take_slot(SlotKind kind) {
  if (free_slot_ == kNoIndex) {
    fail("prefix index allocation", "metadata budget is exhausted");
  }
  const std::uint32_t index = free_slot_;
  Slot& slot = slots_[index];
  if (slot.kind != SlotKind::free) {
    fail("prefix index allocation", "free-list slot is not free");
  }
  free_slot_ = slot.storage.next_free;
  slot.kind = kind;
  return index;
}

void PrefixIndex::release_slot(std::uint32_t index) {
  if (index >= slot_capacity_ || slots_[index].kind == SlotKind::free) {
    fail("prefix index release", "slot is invalid or already free");
  }
  Slot& slot = slots_[index];
  if (slot.kind == SlotKind::node) {
    slot.storage.node.~TrieNode();
  } else if (slot.kind == SlotKind::record) {
    slot.storage.record.~Record();
  } else {
    slot.storage.image.~ImageSpan();
  }
  slot.kind = SlotKind::free;
  slot.storage.next_free = free_slot_;
  free_slot_ = index;
}

PrefixIndex::TrieNode& PrefixIndex::node_ref(std::uint32_t index) {
  if (index >= slot_capacity_ || slots_[index].kind != SlotKind::node) {
    fail("prefix index trie", "node slot is invalid");
  }
  return slots_[index].storage.node;
}

const PrefixIndex::TrieNode& PrefixIndex::node_ref(std::uint32_t index) const {
  if (index >= slot_capacity_ || slots_[index].kind != SlotKind::node) {
    fail("prefix index trie", "node slot is invalid");
  }
  return slots_[index].storage.node;
}

PrefixIndex::Record& PrefixIndex::record_ref(std::uint32_t index) {
  if (index >= slot_capacity_ || slots_[index].kind != SlotKind::record) {
    fail("prefix index record", "record slot is invalid");
  }
  return slots_[index].storage.record;
}

const PrefixIndex::Record& PrefixIndex::record_ref(std::uint32_t index) const {
  if (index >= slot_capacity_ || slots_[index].kind != SlotKind::record) {
    fail("prefix index record", "record slot is invalid");
  }
  return slots_[index].storage.record;
}

std::uint32_t PrefixIndex::next_clock_record() {
  if (active_record_head_ == kNoIndex) {
    fail("prefix index CLOCK", "active record list is empty");
  }
  if (clock_hand_ == kNoIndex || clock_hand_ >= slot_capacity_ ||
      slots_[clock_hand_].kind != SlotKind::record) {
    clock_hand_ = active_record_head_;
    return clock_hand_;
  }
  const std::uint32_t next = record_ref(clock_hand_).next_active;
  clock_hand_ = next == kNoIndex ? active_record_head_ : next;
  return clock_hand_;
}

void PrefixIndex::erase_record(std::uint32_t index) {
  const Record removed = record_ref(index);
  TrieNode& node = node_ref(removed.node);
  if (node.first_record == index) {
    node.first_record = removed.next_at_node;
  } else {
    std::uint32_t previous = node.first_record;
    while (previous != kNoIndex && record_ref(previous).next_at_node != index) {
      previous = record_ref(previous).next_at_node;
    }
    if (previous == kNoIndex) {
      fail("prefix index removal", "checkpoint record is not linked");
    }
    record_ref(previous).next_at_node = removed.next_at_node;
  }

  if (removed.previous_active == kNoIndex) {
    active_record_head_ = removed.next_active;
  } else {
    record_ref(removed.previous_active).next_active = removed.next_active;
  }
  if (removed.next_active != kNoIndex) {
    record_ref(removed.next_active).previous_active = removed.previous_active;
  }
  if (clock_hand_ == index) {
    clock_hand_ = kNoIndex;
  }
  release_slot(index);
  --active_records_;
  if (used_bytes_ < sizeof(Slot)) {
    fail("prefix index removal", "checkpoint accounting underflow");
  }
  used_bytes_ -= sizeof(Slot);
  prune_empty_nodes(removed.node);
}

void PrefixIndex::prune_empty_nodes(std::uint32_t node) {
  while (node != 0) {
    const TrieNode removed = node_ref(node);
    if (removed.first_child != kNoIndex || removed.first_record != kNoIndex) {
      return;
    }
    TrieNode& parent = node_ref(removed.parent);
    if (parent.first_child == node) {
      parent.first_child = removed.next_sibling;
    } else {
      std::uint32_t previous = parent.first_child;
      while (previous != kNoIndex && node_ref(previous).next_sibling != node) {
        previous = node_ref(previous).next_sibling;
      }
      if (previous == kNoIndex) {
        fail("prefix index pruning", "node is not linked to its parent");
      }
      node_ref(previous).next_sibling = removed.next_sibling;
    }
    release_slot(node);
    if (removed.image_slot != kNoIndex) {
      release_slot(removed.image_slot);
      if (used_bytes_ < sizeof(Slot)) {
        fail("prefix index pruning", "image accounting underflow");
      }
      used_bytes_ -= sizeof(Slot);
    }
    --active_nodes_;
    if (used_bytes_ < sizeof(Slot)) {
      fail("prefix index pruning", "node accounting underflow");
    }
    used_bytes_ -= sizeof(Slot);
    node = removed.parent;
  }
}

bool PrefixIndex::is_unpromoted_periodic(const Record& record) const {
  return record.cache_class == CacheClass::probationary &&
         record.sources == checkpoint_source_mask(CheckpointSource::periodic);
}

ThinnedPeriodic PrefixIndex::thin_periodic_from_node(
    std::uint32_t node, const UsableCheckpoint& eligible) {
  struct Candidate {
    CheckpointId checkpoint{};
    std::uint32_t depth{};
    std::uint64_t admission_order{};
  };

  // This method runs immediately after every periodic admission, which has
  // already kept the path at eight candidates. The just-admitted record is
  // the only possible excess.
  std::array<Candidate, kMaxUnpromotedPeriodicPerPath + 1> candidates{};
  std::size_t candidate_count = 0;
  while (true) {
    const TrieNode& current = node_ref(node);
    for (std::uint32_t index = current.first_record; index != kNoIndex;
         index = record_ref(index).next_at_node) {
      const Record& record = record_ref(index);
      if (is_unpromoted_periodic(record)) {
        if (candidate_count == candidates.size()) {
          fail("prefix index thinning", "periodic candidate invariant exceeded");
        }
        candidates[candidate_count++] =
            {record.checkpoint, current.depth, record.admission_order};
      }
    }
    if (node == 0) {
      break;
    }
    node = current.parent;
  }
  if (candidate_count <= kMaxUnpromotedPeriodicPerPath) {
    return {};
  }

  std::sort(candidates.begin(), candidates.begin() + candidate_count,
            [](const Candidate& left, const Candidate& right) {
              if (left.depth != right.depth) {
                return left.depth < right.depth;
              }
              return left.admission_order < right.admission_order;
            });
  const std::size_t excess =
      candidate_count - kMaxUnpromotedPeriodicPerPath;
  ThinnedPeriodic removed;
  const auto remove_candidate = [this, &removed, &eligible](const Candidate& candidate) {
    if (eligible && !eligible(candidate.checkpoint)) return;
    const std::optional<CheckpointMetadata> snapshot =
        metadata(candidate.checkpoint);
    if (snapshot.has_value() && remove(candidate.checkpoint) &&
        !removed.append({candidate.checkpoint, *snapshot})) {
      fail("prefix index thinning", "periodic removal capacity exceeded");
    }
  };

  // Thin alternating older entries first, preserving the latest candidate and
  // keeping a coarser history instead of simply retaining the last eight.
  for (std::size_t index = 0;
       index + 1 < candidate_count && removed.size() < excess; index += 2) {
    remove_candidate(candidates[index]);
  }
  for (std::size_t index = 1;
       index + 1 < candidate_count && removed.size() < excess; index += 2) {
    remove_candidate(candidates[index]);
  }
  return removed;
}

bool PrefixIndex::has_room(std::size_t nodes, std::size_t records) const {
  const std::size_t slots =
      checked_add(nodes, records, "prefix index admission accounting");
  const std::size_t required = checked_mul(slots, sizeof(Slot),
                                           "prefix index admission accounting");
  return used_bytes_ <= metadata_bytes_ &&
         required <= metadata_bytes_ - used_bytes_;
}

std::size_t PrefixIndex::checked_add(std::size_t left, std::size_t right,
                                     const char* label) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    fail(label, "size addition overflows size_t");
  }
  return left + right;
}

std::size_t PrefixIndex::checked_mul(std::size_t left, std::size_t right,
                                     const char* label) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    fail(label, "size multiplication overflows size_t");
  }
  return left * right;
}

std::uint32_t PrefixIndex::checked_tokens(std::size_t count,
                                          const char* label) {
  if (count > std::numeric_limits<std::uint32_t>::max()) {
    fail(label, "token count exceeds uint32 range");
  }
  return static_cast<std::uint32_t>(count);
}

}  // namespace gewell::prefix_index
