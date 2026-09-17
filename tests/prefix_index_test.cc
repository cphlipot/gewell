#include "gewell/prefix_index.h"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <string_view>
#include <utility>


#include <iostream>

namespace gewell::prefix_index {
namespace {

[[noreturn]] void fail(std::string_view operation, std::string_view detail) {
  throw std::runtime_error(std::string(operation) + ": " +
                           std::string(detail));
}

void image_prefix_tests() {
  const auto require = [](bool value, const char* message) {
    if (!value) fail("image prefix self-test", message);
  };
  const auto usable = [](CheckpointId) { return true; };
  const auto endpoint = CheckpointSource::input_endpoint;
  const std::vector<std::uint32_t> tokens{1, 2, 9, 9, 3, 4, 9, 9, 5};
  std::vector<ImageSpan> images{{2, 4, {}}, {6, 8, {}}};
  images[1].digest[31] = 1;
  const auto prefix = [&](std::size_t count) {
    return std::vector<std::uint32_t>(tokens.begin(), tokens.begin() + count);
  };
  PrefixIndex index(8 * 1024);
  const auto root_bytes = index.stats().used_bytes;
  require(index.admit(1, prefix(2), endpoint, {}, images).retained() &&
          index.admit(2, prefix(4), endpoint, {}, images).retained() &&
          index.admit(3, prefix(6), endpoint, {}, images).retained() &&
          index.admit(4, tokens, endpoint, {}, images).retained(),
          "image checkpoints were not admitted");
  const auto hit = index.find_longest(tokens, usable, images);
  require(hit.checkpoint == 4 && hit.checkpoint_tokens == tokens.size() &&
          hit.longest_matching_tokens == tokens.size() &&
          index.metadata(2)->prefix_tokens == 4 &&
          index.nearest_ancestor(4)->checkpoint == 3,
          "image keys changed physical token positions or ancestry");
  auto changed = images;
  changed[0].digest[0] = 2;
  require(index.find_longest(tokens, usable, changed).checkpoint == 1 &&
          index.find_longest(tokens, usable, changed).longest_matching_tokens == 2,
          "changed first image reused image state");
  changed = images;
  changed[1].digest[0] = 2;
  require(index.find_longest(tokens, usable, changed).checkpoint == 3 &&
          index.find_longest(tokens, usable, changed).longest_matching_tokens == 6,
          "changed later image lost or crossed its earlier prefix");
  std::swap(changed[0].digest, changed[1].digest);
  require(index.find_longest(tokens, usable, changed).checkpoint == 1,
          "reordered image identities reused image state");
  changed = images;
  changed[0].end = 5;
  require(index.find_longest(tokens, usable, changed).checkpoint == 1,
          "image span length was not part of its identity");
  require(index.find_longest(tokens, usable).checkpoint == 1,
          "zero image digest collided with a text edge");
  require(index.admit(5, tokens, endpoint).retained() &&
          index.find_longest(tokens, usable).checkpoint == 5 &&
          index.find_longest(tokens, usable, images).checkpoint == 4,
          "text and image histories did not remain distinct");
  require(index.find_longest(prefix(3), usable, images).longest_matching_tokens == 2,
          "partial image lookup advertised an unsafe learned boundary");
  auto invalid_feature = tokens;
  invalid_feature[3] = 10;
  require(index.find_longest(invalid_feature, usable, images).longest_matching_tokens == 2,
          "feature mismatch advertised a partial image prefix");
  const auto before = index.stats();
  for (const auto& invalid : std::vector<std::vector<ImageSpan>>{
           images, {{2, 4, {}}, {3, 5, {}}}, {{4, 4, {}}}}) {
    bool rejected = false;
    try { (void)index.admit(6, prefix(3), endpoint, {}, invalid); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected && index.stats().used_bytes == before.used_bytes &&
            index.stats().checkpoint_count == before.checkpoint_count,
            "invalid or partial image admission mutated metadata");
  }
  changed = images;
  changed[1].digest[0] = 7;
  bool rejected = false;
  try { (void)index.admit(4, tokens, endpoint, {}, changed); }
  catch (const std::runtime_error&) { rejected = true; }
  require(rejected && index.find_longest(tokens, usable, images).checkpoint == 4,
          "checkpoint ID was reassigned to different image content");
  for (CheckpointId checkpoint = 1; checkpoint <= 5; ++checkpoint)
    require(index.remove(checkpoint), "image fixture checkpoint disappeared");
  require(index.stats().used_bytes == root_bytes && index.stats().node_count == 1,
          "image metadata slots leaked after pruning");

  // Root, token nodes and checkpoint fit; the second image slot does not.
  PrefixIndex limited(root_bytes * (tokens.size() + 3));
  require(limited.admit(7, tokens, endpoint, {}, images).status ==
              AdmissionStatus::insufficient_metadata &&
          limited.stats().used_bytes == root_bytes &&
          limited.stats().node_count == 1,
          "image slot capacity failure partially admitted a path");
  for (int cycle = 0; cycle < 16; ++cycle) {
    require(index.admit(8, tokens, endpoint, {}, images).retained() &&
            index.stats().used_bytes == root_bytes * (tokens.size() + 4) &&
            index.remove(8) && index.stats().used_bytes == root_bytes,
            "image slots were not charged or recycled exactly once");
  }
}

bool run_self_tests(std::string* failure) {
  try {
    image_prefix_tests();
    PrefixIndex index(4 * 1024);
    if (index.admit(11, {2, 3, 4}, CheckpointSource::input_endpoint).status !=
            AdmissionStatus::admitted ||
        index.admit(12, {2, 3, 5},
                    CheckpointSource::continuation_endpoint)
                .status != AdmissionStatus::admitted) {
      fail("prefix index self-test", "basic checkpoint admission failed");
    }
    const MetadataStats shared_stats = index.stats();
    if (shared_stats.node_count != 5 || shared_stats.checkpoint_count != 2 ||
        shared_stats.used_bytes > shared_stats.capacity_bytes) {
      fail("prefix index self-test", "trie did not share token prefix nodes");
    }

    const LookupResult hit = index.find_longest(
        {2, 3, 4, 9}, [](CheckpointId checkpoint) { return checkpoint == 11; });
    if (hit.longest_matching_tokens != 3 || hit.checkpoint != 11 ||
        hit.checkpoint_tokens != 3) {
      fail("prefix index self-test", "longest usable checkpoint is wrong");
    }
    const LookupResult stale = index.find_longest(
        {2, 3, 4}, [](CheckpointId) { return false; });
    if (stale.longest_matching_tokens != 3 || stale.has_checkpoint()) {
      fail("prefix index self-test", "lookup accepted an unusable checkpoint");
    }
    if (index.metadata(11)->reuse_count != 0)
      fail("prefix index self-test", "lookup manufactured a checkpoint reuse");
    if (!index.mark_used(11)) {
      fail("prefix index self-test", "checkpoint hit was not recorded");
    }
    const std::optional<CheckpointMetadata> reused = index.metadata(11);
    if (!reused.has_value() ||
        reused->priority != RetentionPriority::normal ||
        reused->cache_class != CacheClass::reused || !reused->reference_bit ||
        reused->reuse_count != 1) {
      fail("prefix index self-test", "CLOCK metadata was not promoted");
    }
    if (!index.mark_used(11) || index.mark_used(999) ||
        !index.admit(11, {2, 3, 4}, CheckpointSource::periodic).retained() ||
        index.metadata(11)->reuse_count != 2 ||
        !index.set_priority(11, RetentionPriority::high) ||
        index.metadata(11)->reuse_count != 2 ||
        !index.set_priority(11, RetentionPriority::normal))
      fail("prefix index self-test", "checkpoint reuse count was lost or fabricated");
    const std::optional<CheckpointId> victim = index.select_clock_victim();
    if (!victim.has_value() || *victim != 12) {
      fail("prefix index self-test", "CLOCK did not prefer probationary state");
    }
    if (!index.remove(12) || index.contains(12) ||
        index.find_longest({2, 3, 5},
                           [](CheckpointId) { return true; })
                .longest_matching_tokens != 2) {
      fail("prefix index self-test", "removal did not prune dead trie nodes");
    }

    PrefixIndex ancestry(4 * 1024);
    (void)ancestry.admit(101, {7}, CheckpointSource::periodic);
    (void)ancestry.admit(102, {7, 8, 9}, CheckpointSource::periodic);
    (void)ancestry.admit(103, {7, 8, 9, 10}, CheckpointSource::input_endpoint);
    (void)ancestry.admit(104, {11}, CheckpointSource::input_endpoint);
    const auto parent = ancestry.nearest_ancestor(103);
    const auto grandparent = ancestry.nearest_ancestor(102);
    const bool removed_parent = ancestry.remove(102);
    const auto replacement_parent = ancestry.nearest_ancestor(103);
    if (!parent.has_value() || parent->checkpoint != 102 ||
        parent->prefix_tokens != 3 || !grandparent.has_value() ||
        grandparent->checkpoint != 101 || grandparent->prefix_tokens != 1 ||
        ancestry.nearest_ancestor(101).has_value() ||
        ancestry.nearest_ancestor(104).has_value() ||
        ancestry.nearest_ancestor(999).has_value() ||
        !removed_parent || !replacement_parent.has_value() ||
        replacement_parent->checkpoint != 101) {
      fail("prefix index self-test", "nearest checkpoint ancestry is wrong");
    }

    PrefixIndex redundant(8 * 1024);
    (void)redundant.admit(201, {1, 2}, CheckpointSource::periodic);
    (void)redundant.admit(202, {1, 2, 3},
                         CheckpointSource::input_endpoint);
    (void)redundant.admit(203, {1, 2, 3, 4},
                         CheckpointSource::continuation_endpoint);
    (void)redundant.admit(204, {9}, CheckpointSource::periodic);
    (void)redundant.admit(205, {9, 8}, CheckpointSource::input_endpoint);
    (void)redundant.set_priority(205, RetentionPriority::low);
    const auto any_redundant = [](CheckpointId) { return true; };
    const auto low_redundant =
        redundant.select_redundant_victim(2, any_redundant);
    if (!low_redundant.has_value() || *low_redundant != 205 ||
        !redundant.remove(205)) {
      fail("prefix index self-test",
           "redundant selection did not preserve priority ordering");
    }
    const auto oldest_close =
        redundant.select_redundant_victim(2, any_redundant);
    if (!oldest_close.has_value() || *oldest_close != 202 ||
        !redundant.remove(202)) {
      fail("prefix index self-test",
           "redundant selection did not prefer the oldest closest state");
    }
    const auto coarsened =
        redundant.select_redundant_victim(2, any_redundant);
    if (!coarsened.has_value() || *coarsened != 203 ||
        !redundant.mark_used(203) ||
        redundant.select_redundant_victim(2, any_redundant).has_value() ||
        redundant.select_redundant_victim(
                     2, [](CheckpointId) { return false; })
            .has_value()) {
      fail("prefix index self-test",
           "redundant selection did not coarsen or protect reused state");
    }

    PrefixIndex filtered_clock(4 * 1024);
    (void)filtered_clock.admit(21, {20}, CheckpointSource::input_endpoint);
    (void)filtered_clock.admit(22, {21}, CheckpointSource::input_endpoint);
    (void)filtered_clock.mark_used(21);
    (void)filtered_clock.mark_used(22);
    if (filtered_clock.select_clock_victim(
            [](CheckpointId) { return false; })
            .has_value() ||
        !filtered_clock.metadata(21)->reference_bit ||
        !filtered_clock.metadata(22)->reference_bit) {
      fail("prefix index self-test",
           "ineligible CLOCK candidates had their state mutated");
    }
    const std::optional<CheckpointId> filtered_victim =
        filtered_clock.select_clock_victim(
            [](CheckpointId checkpoint) { return checkpoint == 21; });
    if (!filtered_victim.has_value() || *filtered_victim != 21 ||
        filtered_clock.metadata(21)->reference_bit ||
        !filtered_clock.metadata(22)->reference_bit) {
      fail("prefix index self-test",
           "filtered CLOCK selection touched an ineligible candidate");
    }

    PrefixIndex weighted_clock(4 * 1024);
    (void)weighted_clock.admit(31, {31}, CheckpointSource::input_endpoint);
    (void)weighted_clock.admit(32, {32}, CheckpointSource::input_endpoint);
    (void)weighted_clock.admit(33, {33}, CheckpointSource::input_endpoint);
    (void)weighted_clock.mark_used(31);
    (void)weighted_clock.mark_used(32);
    (void)weighted_clock.mark_used(33);
    const auto reclaimable_bytes = [](CheckpointId checkpoint) {
      switch (checkpoint) {
        case 31:
          return std::size_t{400};
        case 32:
        case 33:
          return std::size_t{100};
        default:
          return std::size_t{0};
      }
    };
    const std::optional<CheckpointId> weighted_victim =
        weighted_clock.select_clock_victim(
            [](CheckpointId) { return true; }, reclaimable_bytes);
    std::size_t weighted_reused_bytes = 0;
    for (const CheckpointId checkpoint : {CheckpointId{31}, CheckpointId{32},
                                           CheckpointId{33}}) {
      const std::optional<CheckpointMetadata> metadata =
          weighted_clock.metadata(checkpoint);
      if (!metadata.has_value()) {
        fail("prefix index self-test", "weighted CLOCK lost a record");
      }
      if (metadata->cache_class == CacheClass::reused) {
        weighted_reused_bytes += reclaimable_bytes(checkpoint);
      }
    }
    if (!weighted_victim.has_value() ||
        weighted_clock.metadata(*weighted_victim)->cache_class !=
            CacheClass::probationary ||
        weighted_reused_bytes > 450) {
      fail("prefix index self-test",
           "CLOCK did not demote reused storage to the 75 percent target");
    }

    PrefixIndex singleton_clock(4 * 1024);
    (void)singleton_clock.admit(41, {41}, CheckpointSource::input_endpoint);
    (void)singleton_clock.mark_used(41);
    (void)singleton_clock.select_clock_victim(
        [](CheckpointId) { return true; },
        [](CheckpointId) { return std::size_t{400}; });
    if (singleton_clock.metadata(41)->cache_class != CacheClass::reused) {
      fail("prefix index self-test",
           "soft CLOCK target demoted the only reused checkpoint");
    }

    PrefixIndex priorities(4 * 1024);
    (void)priorities.admit(61, {61}, CheckpointSource::input_endpoint);
    (void)priorities.admit(62, {62}, CheckpointSource::input_endpoint);
    (void)priorities.admit(63, {63}, CheckpointSource::input_endpoint);
    if (!priorities.mark_used(61) ||
        !priorities.set_priority(61, RetentionPriority::high) ||
        !priorities.set_priority(62, RetentionPriority::normal) ||
        !priorities.set_priority(63, RetentionPriority::low) ||
        priorities.set_priority(99, RetentionPriority::low)) {
      fail("prefix index self-test", "priority update failed");
    }
    const std::optional<CheckpointMetadata> high_metadata =
        priorities.metadata(61);
    if (!high_metadata.has_value() ||
        high_metadata->priority != RetentionPriority::high ||
        high_metadata->cache_class != CacheClass::reused ||
        !high_metadata->reference_bit) {
      fail("prefix index self-test",
           "reprioritizing changed checkpoint CLOCK state");
    }
    const auto equal_bytes = [](CheckpointId) { return std::size_t{100}; };
    for (const CheckpointId expected :
         {CheckpointId{63}, CheckpointId{62}, CheckpointId{61}}) {
      const std::optional<CheckpointId> priority_victim =
          priorities.select_clock_victim(
              [](CheckpointId) { return true; }, equal_bytes);
      if (!priority_victim.has_value() || *priority_victim != expected ||
          !priorities.remove(*priority_victim)) {
        fail("prefix index self-test", "CLOCK priority ordering is wrong");
      }
    }

    PrefixIndex tiered_clock(4 * 1024);
    (void)tiered_clock.admit(71, {71}, CheckpointSource::input_endpoint);
    (void)tiered_clock.admit(72, {72}, CheckpointSource::input_endpoint);
    (void)tiered_clock.admit(73, {73}, CheckpointSource::input_endpoint);
    if (!tiered_clock.mark_used(71) || !tiered_clock.mark_used(72) ||
        !tiered_clock.set_priority(71, RetentionPriority::low) ||
        !tiered_clock.set_priority(72, RetentionPriority::low)) {
      fail("prefix index self-test", "tiered CLOCK setup failed");
    }
    const auto tiered_bytes = [](CheckpointId checkpoint) {
      return checkpoint == 73 ? std::size_t{800} : std::size_t{100};
    };
    const std::optional<CheckpointId> tiered_victim =
        tiered_clock.select_clock_victim(
            [](CheckpointId) { return true; }, tiered_bytes);
    if (!tiered_victim.has_value() ||
        tiered_clock.metadata(*tiered_victim)->priority !=
            RetentionPriority::low ||
        tiered_clock.metadata(*tiered_victim)->cache_class !=
            CacheClass::probationary ||
        tiered_clock.metadata(73)->cache_class != CacheClass::probationary) {
      fail("prefix index self-test",
           "CLOCK did not preserve priority-local class balancing");
    }
    std::size_t tiered_low_reused_bytes = 0;
    for (const CheckpointId checkpoint :
         {CheckpointId{71}, CheckpointId{72}}) {
      if (tiered_clock.metadata(checkpoint)->cache_class ==
          CacheClass::reused) {
        tiered_low_reused_bytes += tiered_bytes(checkpoint);
      }
    }
    if (tiered_low_reused_bytes > 150) {
      fail("prefix index self-test",
           "CLOCK did not apply the 75 percent target per priority");
    }

    PrefixIndex periodic(8 * 1024);
    std::vector<std::uint32_t> prefix;
    AdmissionResult ninth;
    for (std::uint32_t length = 1; length <= 9; ++length) {
      prefix.push_back(100 + length);
      AdmissionResult result = periodic.admit(
          500 + length, prefix, CheckpointSource::periodic);
      if (result.status != AdmissionStatus::admitted) {
        fail("prefix index self-test", "periodic checkpoint admission failed");
      }
      if (length == 9) {
        ninth = std::move(result);
      }
    }
    if (ninth.thinned_periodic.size() != 1 ||
        ninth.thinned_periodic.front().checkpoint != 501 ||
        ninth.thinned_periodic.front().metadata.checkpoint != 501 ||
        !has_checkpoint_source(
            ninth.thinned_periodic.front().metadata.sources,
            CheckpointSource::periodic) ||
        periodic.contains(501) ||
        periodic.stats().unpromoted_periodic_count !=
            kMaxUnpromotedPeriodicPerPath) {
      fail("prefix index self-test", "periodic thinning did not preserve limit");
    }
    if (!periodic.mark_used(502)) {
      fail("prefix index self-test", "periodic checkpoint was not promoted");
    }
    prefix.push_back(110);
    (void)periodic.admit(510, prefix, CheckpointSource::periodic);
    prefix.push_back(111);
    const AdmissionResult promoted_safe =
        periodic.admit(511, prefix, CheckpointSource::periodic);
    if (!periodic.contains(502) || promoted_safe.thinned_periodic.empty()) {
      fail("prefix index self-test",
           "periodic thinning removed promoted state or did not thin");
    }

    PrefixIndex pinned_periodic(64 * 1024);
    std::vector<std::uint32_t> pinned_path;
    for (std::uint64_t i = 1; i <= kMaxUnpromotedPeriodicPerPath; ++i) {
      pinned_path.push_back(200 + i);
      if (!pinned_periodic.admit(600 + i, pinned_path, CheckpointSource::periodic).retained())
        fail("prefix index self-test", "protected periodic setup did not fit");
    }
    const auto pinned_used = pinned_periodic.stats().used_bytes;
    pinned_path.push_back(209);
    const auto refused = pinned_periodic.admit(609, pinned_path, CheckpointSource::periodic,
        [](CheckpointId) { return false; });
    if (refused.retained() || refused.status != AdmissionStatus::protected_periodic ||
        !refused.thinned_periodic.empty() || pinned_periodic.contains(609) ||
        pinned_periodic.stats().used_bytes != pinned_used)
      fail("prefix index self-test", "protected periodic admission changed live state");
    const auto unpinned = pinned_periodic.admit(609, pinned_path, CheckpointSource::periodic,
        [](CheckpointId id) { return id == 602; });
    if (!unpinned.retained() || unpinned.thinned_periodic.size() != 1 ||
        unpinned.thinned_periodic.front().checkpoint != 602 ||
        !pinned_periodic.contains(601) || !pinned_periodic.contains(609) ||
        pinned_periodic.stats().unpromoted_periodic_count != kMaxUnpromotedPeriodicPerPath)
      fail("prefix index self-test", "periodic thinning ignored source eligibility");

    PrefixIndex limited(256);
    const std::vector<std::uint32_t> oversized_prefix(64, 7);
    if (limited.admit(900, oversized_prefix, CheckpointSource::input_endpoint)
            .status != AdmissionStatus::insufficient_metadata ||
        limited.stats().used_bytes > limited.stats().capacity_bytes ||
        limited.stats().checkpoint_count != 0 ||
        limited.find_longest(oversized_prefix,
                             [](CheckpointId) { return true; })
                .longest_matching_tokens != 0) {
      fail("prefix index self-test", "bounded admission mutated the trie");
    }

    PrefixIndex recycled(512);
    for (std::uint32_t cycle = 0; cycle < 32; ++cycle) {
      const std::vector<std::uint32_t> cycle_prefix{cycle + 1, cycle + 2};
      if (recycled.admit(1'000 + cycle, cycle_prefix,
                          CheckpointSource::input_endpoint)
              .status != AdmissionStatus::admitted ||
          !recycled.remove(1'000 + cycle) ||
          recycled.stats().used_bytes > recycled.stats().capacity_bytes ||
          recycled.stats().node_count != 1 ||
          recycled.stats().checkpoint_count != 0) {
        fail("prefix index self-test",
             "removal did not recycle fixed metadata slots");
      }
    }
    return true;
  } catch (const std::exception& error) {
    if (failure != nullptr) {
      *failure = error.what();
    }
    return false;
  }
}

}  // namespace
}  // namespace gewell::prefix_index

int main() {
  std::string failure;
  if (!gewell::prefix_index::run_self_tests(&failure)) {
    std::cerr << failure << "\n";
    return 1;
  }
  std::cout << "prefix_index tests: ok\n";
  return 0;
}
