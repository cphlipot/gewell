#include "gewell/runtime/cache.h"
#include "runtime_test_storage.h"

#include <array>
#include <iostream>

namespace {
using namespace gewell;
using namespace gewell::runtime;
using namespace gewell::runtime::test;

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

void capture_order_and_failure() {
  for (const std::string failure : {"snapshot_terminal", "synchronize", ""}) {
    ByteStorage* storage = nullptr;
    PersistentCacheManager cache(small_cache_config(), byte_storage_factory(&storage));
    const auto execution = cache.try_begin_batch(0, 8, {});
    require(bool(execution), "capture fixture admission failed");
    cache.prepare_write(*execution, 0, 2, {});
    const auto before = cache.stats();
    std::array<std::uint8_t, 16> terminal{};
    terminal.fill(0x65);
    storage->events.clear();
    storage->fail_at = failure;
    storage->observe = [&](const std::string&) {
      require(cache.stats().checkpoint_count == 0,
              "checkpoint published before physical completion");
      require(storage->ledger.execution(*execution).pending_checkpoint.has_value() &&
              cache.stats().gpu.used > before.gpu.used,
              "physical copy started before capture allocations were reserved");
      require(!cache.find_longest({7, 8}).has_checkpoint(),
              "incomplete checkpoint became reusable");
    };
    kv_cache::CheckpointId checkpoint = 0;
    bool threw = false;
    try {
      checkpoint = cache.try_capture(*execution, {7, 8}, TerminalState{terminal.data()}, {});
    } catch (const std::runtime_error&) { threw = true; }
    storage->observe = {};
    require(threw == !failure.empty(), "capture failure did not propagate");
    if (failure.empty()) {
      require(checkpoint != 0 && cache.find_longest({7, 8}).checkpoint == checkpoint,
              "completed snapshot was not published");
      require(storage->events == std::vector<std::string>{
          "snapshot_local", "snapshot_terminal", "synchronize"},
          "capture physical boundaries changed order");
      const auto& saved = storage->ledger.checkpoint(checkpoint);
      require(std::equal(terminal.begin(), terminal.end(), storage->pointer(saved.terminal_hidden)),
              "terminal snapshot differs");
    } else {
      require(cache.stats().checkpoint_count == 0 && cache.stats().gpu.used == before.gpu.used,
              "failed capture retained allocations or metadata");
      require(storage->events.back() == "wait", "failed capture did not await pending work");
    }
    storage->fail_at.clear();
    cache.release(*execution);
  }
}

void live_cow_and_owner_release() {
  ByteStorage* storage = nullptr;
  PersistentCacheManager cache(small_cache_config(), byte_storage_factory(&storage));
  const auto execution = cache.try_begin_batch(0, 8, {});
  require(bool(execution), "COW fixture admission failed");
  cache.prepare_write(*execution, 0, 2, {});
  const auto original_page = storage->ledger.execution(*execution).global_pages.front();
  const auto original_storage = storage->ledger.page(original_page).storage;
  std::fill_n(storage->pointer(original_storage), original_storage.bytes, 0x31);
  std::array<std::uint8_t, 16> terminal{};
  const auto checkpoint = cache.try_capture(*execution, {9, 10},
      TerminalState{terminal.data()}, {}, prefix_index::CheckpointSource::input_endpoint, false);
  require(checkpoint != 0, "COW checkpoint admission failed");
  require(cache.add_owner_demands({checkpoint}, "owner", prefix_index::RetentionPriority::normal),
          "owner demand admission failed");
  const auto borrower = cache.try_begin_batch(checkpoint, 8, {});
  require(bool(borrower), "live borrower admission failed");
  const auto fork = cache.try_fork_batch(*execution, 8, {});
  require(bool(fork), "live fork admission failed");
  const auto write = cache.prepare_write(*execution, 2, 1, {});
  require(write.copies.size() == 1 && write.copies.front().source == original_page,
          "shared tail was not copied before write");
  const auto changed_storage = storage->ledger.page(write.copies.front().destination).storage;
  require(std::all_of(storage->pointer(changed_storage),
                      storage->pointer(changed_storage) + changed_storage.bytes,
                      [](std::uint8_t value) { return value == 0x31; }),
          "COW did not preserve original bytes");
  storage->pointer(changed_storage)[0] = 0x90;
  require(storage->pointer(original_storage)[0] == 0x31,
          "private write changed retained/shared state");
  cache.release_owner("owner");
  require(cache.has_checkpoint(checkpoint), "owner release evicted a live borrower");
  cache.release(*borrower);
  require(!cache.has_checkpoint(checkpoint), "last borrower did not release undemanded checkpoint");
  cache.release(*fork);
  cache.release(*execution);
  require(cache.stats().gpu.used == 0 && cache.stats().execution_count == 0,
          "COW fixture leaked allocations");
}

void cold_restore_and_owner_release(bool multimodal) {
  ByteStorage* storage = nullptr;
  PersistentCacheManager cache(small_cache_config(2048, 2048), byte_storage_factory(&storage));
  const auto execution = cache.try_begin_batch(0, 4, {});
  require(bool(execution), "cold fixture admission failed");
  cache.prepare_write(*execution, 0, 4, {});
  std::array<std::uint8_t, 16> terminal{};
  terminal.fill(0x42);
  const std::vector<std::uint32_t> prompt{10, 11, 12, 13};
  const std::vector<prefix_index::ImageSpan> images = multimodal
      ? std::vector<prefix_index::ImageSpan>{{1, 3, {}}}
      : std::vector<prefix_index::ImageSpan>{};
  const auto checkpoint = cache.try_capture(*execution, prompt, TerminalState{terminal.data()}, {},
      prefix_index::CheckpointSource::input_endpoint, false,
      prefix_index::RetentionPriority::normal, true, false, images);
  require(checkpoint != 0, "cold checkpoint admission failed");
  require(cache.add_owner_demands({checkpoint}, "keep", prefix_index::RetentionPriority::normal),
          "cold owner admission failed");
  cache.release(*execution);
  cache.ensure_capacity(0, 24);
  require(cache.is_cold_checkpoint(checkpoint) && cache.cold_spill_count() == 1,
          "pressure did not retain checkpoint in cold tier");
  const auto pressure = cache.begin();
  cache.prepare_write(pressure, 0, 24, {});
  cache.release(pressure);
  require(cache.find_longest(prompt, images).checkpoint == checkpoint &&
          cache.find_batch_prefix(prompt, 5, images).checkpoint == checkpoint &&
          cache.checkpoint_matches(checkpoint, prompt, images),
          "cold prefix became invisible");
  if (multimodal) {
    auto changed = images;
    changed[0].digest[0] = 1;
    require(!cache.find_longest(prompt, changed).has_checkpoint() &&
            !cache.find_batch_prefix(prompt, 5, changed).has_checkpoint() &&
            !cache.checkpoint_matches(checkpoint, prompt, changed) &&
            !cache.find_longest(prompt).has_checkpoint(),
            "cold image state matched different image or text content");
  }
  cache.ensure_capacity(checkpoint, 5);
  const auto restored = cache.begin(checkpoint);
  std::array<std::uint8_t, 16> actual{};
  cache.restore_terminal_hidden(restored, TerminalState{actual.data()}, {});
  require(actual == terminal && cache.processed_tokens(restored) == prompt.size(),
          "cold restore lost terminal state or position");
  require(cache.cold_restore_count() == 1, "cold restore accounting changed");
  cache.release_owner("keep");
  require(cache.has_checkpoint(checkpoint), "cold source reclaimed while execution borrowed it");
  cache.release(restored);
  require(!cache.has_checkpoint(checkpoint) && cache.stats().gpu.used == 0 &&
          cache.stats().cpu.used == 0, "cold owner release leaked state");
}

kv_cache::PoolConfig index_pressure_config(bool offload) {
  auto config = small_cache_config(65536, offload ? 65536 : 0);
  config.global_page_tokens = 64;
  config.local_ring_bytes = 1024;
  config.local_bytes_per_token = 128;
  return config;
}

void index_pressure_keeps_admitting(bool offload) {
  std::size_t removals = 0;
  PersistentCacheManager cache(index_pressure_config(offload), byte_storage_factory(),
      [&](const nlohmann::json& event) {
        if (event.at("reason") == "prefix_index_pressure") ++removals;
      });
  std::array<std::uint8_t, 16> terminal{};
  kv_cache::CheckpointId pinned = 0, borrowed = 0;
  kv_cache::ExecutionId borrower = 0;
  for (std::uint32_t i = 0; i < 12; ++i) {
    const std::vector<std::uint32_t> prompt(64, i + 1);
    const auto execution = cache.try_begin_batch(0, prompt.size(), {});
    require(bool(execution), "index pressure execution did not fit");
    cache.prepare_write(*execution, 0, prompt.size(), {});
    const auto checkpoint = cache.try_capture(*execution, prompt, TerminalState{terminal.data()}, {});
    require(checkpoint != 0 && cache.find_longest(prompt).checkpoint == checkpoint,
            "index pressure prevented a new checkpoint despite idle victims");
    cache.release(*execution);
    if (i == 0) {
      pinned = checkpoint;
      cache.pin_batch_checkpoint(pinned);
    } else if (i == 1) {
      borrowed = checkpoint;
      const auto resumed = cache.try_begin_batch(borrowed, prompt.size(), {});
      require(bool(resumed), "index pressure borrower did not fit");
      borrower = *resumed;
    }
    require(cache.has_checkpoint(pinned) && (!borrowed || cache.has_checkpoint(borrowed)),
            "index pressure evicted a pinned or borrowed GPU checkpoint");
    require(cache.stats().index_used <= cache.config().index_bytes,
            "index pressure exceeded the metadata budget");
  }
  require(removals != 0 && cache.cold_spill_count() == 0 && cache.stats().cpu.used == 0,
          "prefix index pressure spilled instead of reclaiming entries");
  cache.unpin_batch_checkpoint(pinned);
  cache.release(borrower);
}

void index_pressure_reclaims_cold_entries() {
  ByteStorage* storage = nullptr;
  PersistentCacheManager cache(index_pressure_config(true), byte_storage_factory(&storage));
  std::array<std::uint8_t, 16> terminal{};
  std::vector<kv_cache::CheckpointId> checkpoints;
  std::size_t entry_bytes = 0;
  do {
    const std::vector<std::uint32_t> prompt(64, checkpoints.size() + 1);
    const auto execution = cache.try_begin_batch(0, prompt.size(), {});
    require(bool(execution), "cold index fixture admission failed");
    cache.prepare_write(*execution, 0, prompt.size(), {});
    const auto before = cache.prefix_stats().used_bytes;
    const auto checkpoint = cache.try_capture(*execution, prompt, TerminalState{terminal.data()}, {});
    require(checkpoint != 0, "cold index fixture capture failed");
    entry_bytes = cache.prefix_stats().used_bytes - before;
    cache.release(*execution);
    // Exercise the normal GPU-pressure spill path before applying trie pressure.
    require(storage->ledger.evict_idle_checkpoint() && cache.is_cold_checkpoint(checkpoint),
            "GPU pressure did not preserve the fixture in RAM");
    checkpoints.push_back(checkpoint);
    cache.pin_batch_checkpoint(checkpoint);
  } while (entry_bytes <= cache.prefix_stats().capacity_bytes - cache.prefix_stats().used_bytes);
  require(checkpoints.size() >= 3, "cold index fixture has insufficient protected entries");
  const auto borrowed = checkpoints.front();
  const auto borrower = cache.try_begin_batch(borrowed, 64, {});
  require(bool(borrower), "cold index borrower did not fit");
  cache.unpin_batch_checkpoint(borrowed);
  const auto before = cache.stats();
  const auto spills = cache.cold_spill_count();
  const std::vector<std::uint32_t> incoming(64, 100);
  const auto execution = cache.try_begin_batch(0, incoming.size(), {});
  require(bool(execution), "cold index incoming execution did not fit");
  cache.prepare_write(*execution, 0, incoming.size(), {});
  require(cache.try_capture(*execution, incoming, TerminalState{terminal.data()}, {}) == 0,
          "index pressure displaced pinned or borrowed RAM state");
  require(cache.stats().checkpoint_count == checkpoints.size() && cache.stats().cpu.used == before.cpu.used,
          "failed index admission changed protected RAM state");
  const auto victim = checkpoints.back();
  cache.unpin_batch_checkpoint(victim);
  const auto checkpoint = cache.try_capture(*execution, incoming, TerminalState{terminal.data()}, {});
  require(checkpoint != 0 && !cache.has_checkpoint(victim) && !cache.is_cold_checkpoint(checkpoint),
          "index pressure failed to replace an idle RAM entry with a GPU checkpoint");
  for (std::size_t i = 0; i + 1 < checkpoints.size(); ++i) {
    require(cache.has_checkpoint(checkpoints[i]), "index reclamation lost protected RAM state");
    if (i != 0) cache.unpin_batch_checkpoint(checkpoints[i]);
  }
  require(cache.stats().cpu.used < before.cpu.used && cache.cold_spill_count() == spills,
          "index reclamation did not release RAM without further spilling");
  cache.release(*execution);
  cache.release(*borrower);
}

void owner_demand_reuse_and_rollback() {
  PersistentCacheManager cache(small_cache_config(16384), byte_storage_factory());
  std::array<std::uint8_t, 16> terminal{};
  std::vector<kv_cache::CheckpointId> checkpoints;
  for (std::uint32_t i = 0; i < 3; ++i) {
    const std::vector<std::uint32_t> prompt(4, i + 1);
    const auto execution = cache.try_begin_batch(0, prompt.size(), {});
    require(bool(execution), "owner fixture admission failed");
    cache.prepare_write(*execution, 0, prompt.size(), {});
    const auto checkpoint = cache.try_capture(*execution, prompt, TerminalState{terminal.data()}, {},
        prefix_index::CheckpointSource::input_endpoint, false);
    require(checkpoint && cache.add_owner_demands({checkpoint}, "base", prefix_index::RetentionPriority::normal),
            "owner fixture capture failed");
    checkpoints.push_back(checkpoint);
    cache.release(*execution);
  }
  const auto shared = checkpoints[1];
  require(cache.add_owner_demands({shared}, "shared", prefix_index::RetentionPriority::normal),
          "shared owner admission failed");
  const auto before = cache.stats().index_used;
  PersistentCacheManager::OwnerDemandCommit commit;
  commit.changes.reserve(checkpoints.size());
  require(cache.add_owner_demands(checkpoints, "shared", prefix_index::RetentionPriority::high, nullptr, &commit),
          "owner rollback setup failed");
  cache.rollback_owner_demands("shared", &commit);
  require(commit.changes.empty() && cache.stats().index_used == before,
          "owner rollback retained new demands");
  cache.release_owner("base");
  require(!cache.has_checkpoint(checkpoints[0]) && cache.has_checkpoint(shared) &&
          !cache.has_checkpoint(checkpoints[2]), "owner release lost shared state or skipped demands");

  PersistentCacheManager::OwnerDemandReservation reservation;
  require(cache.reserve_owner_demand_slots(1, &reservation), "owner slot reservation failed");
  const auto filler_count = cache.owner_demand_slot_capacity() - 2;
  for (std::size_t i = 0; i < filler_count; ++i)
    require(cache.add_owner_demands({shared}, std::to_string(i), prefix_index::RetentionPriority::normal),
            "owner capacity was lost after release");
  require(!cache.add_automatic_demand(shared), "automatic demand consumed a reserved slot");
  require(cache.add_owner_demands({shared}, "reserved", prefix_index::RetentionPriority::normal, &reservation) &&
          reservation.slots == 0 && !cache.add_automatic_demand(shared), "owner capacity bound was not enforced");
  cache.release_owner("0");
  require(cache.add_automatic_demand(shared), "released owner slot was not reusable");
  for (std::size_t i = 1; i < filler_count; ++i) cache.release_owner(std::to_string(i));
  cache.release_owner("shared");
  cache.release_owner("reserved");
  require(cache.has_checkpoint(shared) && cache.stats().checkpoint_count == 1,
          "owner cleanup removed automatic retention");
}

void image_checkpoint_identity() {
  for (const std::uint32_t count : {4U, 8U}) {
    ByteStorage* storage = nullptr;
    PersistentCacheManager cache(small_cache_config(16 * 1024), byte_storage_factory(&storage));
    const std::vector<std::uint32_t> prompt(count, 9);
    const std::vector<prefix_index::ImageSpan> images{{1, 3, {}}};
    auto changed = images;
    changed[0].digest[0] = 1;
    std::array<std::uint8_t, 16> terminal{};
    terminal.fill(0x31);
    const auto capture = [&](kv_cache::ExecutionId execution,
                             const std::vector<prefix_index::ImageSpan>& spans) {
      return cache.try_capture(execution, prompt, TerminalState{terminal.data()}, {},
          prefix_index::CheckpointSource::input_endpoint, true,
          prefix_index::RetentionPriority::normal, true, false, spans);
    };
    const auto first = cache.try_begin_batch(0, count, {});
    require(bool(first), "image capture admission failed");
    cache.prepare_write(*first, 0, count, {});
    const auto checkpoint = capture(*first, images);
    require(checkpoint && cache.checkpoint_tokens(checkpoint) == count &&
            cache.find_longest(prompt, images).checkpoint == checkpoint &&
            cache.find_batch_prefix(prompt, count, images).checkpoint == checkpoint &&
            cache.checkpoint_matches(checkpoint, prompt, images) &&
            !cache.checkpoint_matches(checkpoint, prompt, changed) &&
            !cache.find_longest(prompt).has_checkpoint(),
            "image checkpoint identity or token count was lost");
    cache.release(*first);

    const auto resumed = cache.try_begin_batch(checkpoint, count, {});
    require(bool(resumed), "image source resume failed");
    storage->events.clear();
    require(capture(*resumed, images) == checkpoint && storage->events.empty(),
            "exact image source did not reuse its existing snapshot");
    cache.release(*resumed);

    const auto second = cache.try_begin_batch(0, count, {});
    require(bool(second), "changed image admission failed");
    cache.prepare_write(*second, 0, count, {});
    terminal.fill(0x42);
    const auto other = capture(*second, changed);
    require(other && other != checkpoint &&
            cache.find_longest(prompt, images).checkpoint == checkpoint &&
            cache.find_longest(prompt, changed).checkpoint == other,
            "capture deduplication conflated different images");
    cache.release(*second);

    bool rejected = false;
    try {
      cache.add_checkpoint_source(checkpoint, prompt,
          prefix_index::CheckpointSource::periodic, changed);
    } catch (const std::runtime_error&) { rejected = true; }
    require(rejected && cache.checkpoint_matches(checkpoint, prompt, images),
            "source merge reassigned a checkpoint to another image");

    if (count == cache.config().local_window_tokens) {
      const auto duplicate = cache.try_begin_batch(0, count, {});
      require(bool(duplicate), "duplicate image admission failed");
      cache.prepare_write(*duplicate, 0, count, {});
      storage->events.clear();
      require(capture(*duplicate, changed) == other && storage->events.empty(),
              "retained image deduplication lost the existing snapshot");
      cache.release(*duplicate);
    }
    const auto partial = cache.try_begin_batch(0, count, {});
    require(bool(partial), "partial image fixture admission failed");
    cache.prepare_write(*partial, 0, 2, {});
    storage->events.clear();
    const auto before = cache.stats();
    rejected = false;
    try {
      (void)cache.try_capture(*partial, {9, 9}, TerminalState{terminal.data()}, {},
          prefix_index::CheckpointSource::periodic, true,
          prefix_index::RetentionPriority::normal, true, false, images);
    } catch (const std::runtime_error&) { rejected = true; }
    require(rejected && storage->events.empty() &&
            cache.stats().gpu.used == before.gpu.used &&
            cache.stats().checkpoint_count == before.checkpoint_count,
            "partial image capture performed physical work or published state");
    cache.release(*partial);
  }
}

void capacity_errors_and_metadata() {
  PersistentCacheManager cache(small_cache_config(2048), byte_storage_factory());
  bool invalid = false;
  try { cache.ensure_capacity(0, 64); }
  catch (const CacheCapacityError& error) { invalid = error.request_invalid; }
  require(invalid && cache.stats().gpu.used == 0,
          "oversized request did not fail before physical allocation");
  const auto execution = cache.try_begin_batch(0, 4, {});
  require(bool(execution), "metadata fixture admission failed");
  cache.prepare_write(*execution, 0, 4, {});
  std::array<std::uint8_t, 16> terminal{};
  const auto checkpoint = cache.try_capture(*execution, {1, 2, 3, 4},
      TerminalState{terminal.data()}, {}, prefix_index::CheckpointSource::input_endpoint, false);
  require(checkpoint != 0, "metadata fixture checkpoint failed");
  PersistentCacheManager::OwnerDemandReservation reservation;
  const auto metadata_before = cache.stats().index_used;
  require(!cache.reserve_owner_demand_slots(cache.owner_demand_slot_capacity() + 1,
                                           &reservation) &&
          reservation.slots == 0 && cache.stats().index_used == metadata_before,
          "oversized owner reservation mutated metadata accounting");
  require(cache.reserve_owner_demand_slots(1, &reservation), "owner reservation failed");
  PersistentCacheManager::OwnerDemandCommit commit;
  commit.changes.reserve(1);
  require(cache.add_owner_demands({checkpoint}, "rollback", prefix_index::RetentionPriority::high,
                                  &reservation, &commit), "reserved owner commit failed");
  require(reservation.slots == 0 && cache.stats().index_used <= cache.stats().index_bytes,
          "owner demand exceeded metadata accounting");
  cache.rollback_owner_demands("rollback", &commit);
  cache.reclaim_undemanded(checkpoint);
  require(!cache.has_checkpoint(checkpoint), "owner rollback kept an idle checkpoint");
  cache.release(*execution);
  bool checkpoint_capacity = false;
  try { (void)make_checkpoint_triggers(4, 64, 0, 0, 1, {}, 3); }
  catch (const CacheCapacityError& error) { checkpoint_capacity = error.request_invalid; }
  require(checkpoint_capacity, "checkpoint bookkeeping overflow lost request-error semantics");
}
}  // namespace

int main() {
  try {
    capture_order_and_failure();
    live_cow_and_owner_release();
    cold_restore_and_owner_release(false);
    cold_restore_and_owner_release(true);
    index_pressure_keeps_admitting(true);
    index_pressure_keeps_admitting(false);
    index_pressure_reclaims_cold_entries();
    owner_demand_reuse_and_rollback();
    image_checkpoint_identity();
    capacity_errors_and_metadata();
    std::cout << "runtime cache tests: ok\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "runtime cache tests: " << error.what() << '\n';
    return 1;
  }
}
