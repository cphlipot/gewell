#include "gewell/kv_cache.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <iostream>

namespace gewell::kv_cache {
namespace {

[[noreturn]] void fail(std::string_view operation, std::string_view detail) {
  throw std::runtime_error(std::string(operation) + ": " +
                           std::string(detail));
}

bool run_self_tests(std::string* failure) {
  try {
    // Explicit small geometry tests page/snapshot relationships independently
    // of the model's physical representation.
    PoolConfig config;
    config.gpu_bytes = 64 * kMib;
    config.index_bytes = 64 * kMib;
    config.global_page_tokens = 256;
    config.local_window_tokens = 1'024;
    config.global_page_bytes = 256;
    config.local_ring_bytes = 1'024;
    config.local_bytes_per_token = 1;
    config.terminal_hidden_bytes = 16;
    config.page_table_bytes = 8 * sizeof(std::uint64_t);
    config.maximum_context_tokens = 2'048;
    config.validate();

    // A shared page is reclaimable collectively even though deleting either
    // idle checkpoint alone would retain it. Snapshots must not create borrows
    // or change allocation/eviction state.
    CacheLedger observed(config);
    const auto observed_execution = observed.try_begin_batch(0, 768);
    if (!observed_execution) fail("KV snapshot self-test", "execution admission failed");
    (void)observed.prepare_write(*observed_execution, 0, 256);
    const auto capture_observed = [&]() {
      return observed.publish_checkpoint(*observed_execution,
          std::vector<std::uint8_t>(256), std::vector<std::uint8_t>(16));
    };
    const auto observed_first = capture_observed();
    const auto observed_second = capture_observed();
    const auto observed_before = observed.stats();
    std::size_t visited_pages = 0, visited_checkpoints = 0, visited_executions = 0;
    observed.visit_pages([&](const PageInfo& page) {
      ++visited_pages;
      if (page.references != 3) fail("KV snapshot self-test", "shared reference count changed");
    });
    observed.visit_checkpoints([&](const CheckpointInfo& checkpoint) {
      ++visited_checkpoints;
      if (checkpoint.processed_tokens != 256 || checkpoint.global_pages.size() != 1)
        fail("KV snapshot self-test", "checkpoint enumeration lost its prefix");
    });
    observed.visit_executions([&](const ExecutionInfo& execution) {
      ++visited_executions;
      if (execution.id != *observed_execution || execution.batch_max_processed_tokens != 768)
        fail("KV snapshot self-test", "execution enumeration lost its reservation");
    });
    const auto observed_active = observed.gpu_memory_stats();
    if (visited_pages != 1 || visited_checkpoints != 2 || visited_executions != 1 ||
        observed.stats().gpu.used != observed_before.gpu.used ||
        observed.stats().gpu.peak_used != observed_before.gpu.peak_used ||
        observed_active.global_page_bytes != 256 || observed_active.shared_page_bytes != 256 ||
        observed_active.checkpoint_state_bytes != 544 ||
        observed_active.execution_buffer_bytes != config.local_ring_bytes + config.page_table_bytes ||
        observed_active.reclaimable_bytes != 544 || observed_active.reserved_growth_bytes != 512 ||
        observed_active.page_slack_bytes != 0 || observed_active.other_bytes != 0 ||
        observed_active.global_page_bytes + observed_active.checkpoint_state_bytes +
            observed_active.execution_buffer_bytes != observed_before.gpu.used)
      fail("KV snapshot self-test", "active memory accounting is inconsistent");
    observed.release_execution(*observed_execution);
    const auto observed_idle = observed.gpu_memory_stats();
    if (observed_idle.reclaimable_bytes != observed.stats().gpu.used ||
        observed_idle.nonreclaimable_bytes != 0 || observed_idle.reserved_growth_bytes != 0)
      fail("KV snapshot self-test", "idle shared page was not collectively reclaimable");
    observed.pin_checkpoint(observed_first);
    const auto observed_pinned = observed.gpu_memory_stats();
    if (observed_pinned.reclaimable_bytes != 272 || observed_pinned.nonreclaimable_bytes != 528)
      fail("KV snapshot self-test", "pin did not protect checkpoint and shared page");
    const auto borrower = observed.begin_execution(observed_second);
    if (observed.gpu_memory_stats().reclaimable_bytes != 0)
      fail("KV snapshot self-test", "borrowed checkpoint was counted as reclaimable");
    observed.release_execution(borrower);
    observed.unpin_checkpoint(observed_first);
    observed.release_checkpoint(observed_first);
    if (observed.gpu_memory_stats().shared_page_bytes != 0)
      fail("KV snapshot self-test", "released reference remained shared");
    observed.release_checkpoint(observed_second);

    // Private caller allocations and incomplete captures occupy the pool but
    // are not advertised as retained checkpoints or as reusable free space.
    const auto private_buffer = observed.try_allocate_gpu(128);
    const auto observed_capturing = observed.try_begin_batch(0, 512);
    if (!observed_capturing) fail("KV snapshot self-test", "capture execution admission failed");
    (void)observed.prepare_write(*observed_capturing, 0, 1);
    const auto pending = observed.try_begin_checkpoint_capture(*observed_capturing);
    if (!pending) fail("KV snapshot self-test", "capture admission failed");
    const auto observed_pending = observed.gpu_memory_stats();
    if (observed_pending.other_bytes != 128 || observed_pending.pending_capture_bytes != 17 ||
        observed_pending.checkpoint_state_bytes != 0 || observed_pending.reclaimable_bytes != 0 ||
        observed_pending.nonreclaimable_bytes != observed.stats().gpu.used ||
        observed_pending.reserved_growth_bytes != 512 || observed_pending.page_slack_bytes != 255)
      fail("KV snapshot self-test", "pending capture or private buffer accounting is wrong");
    observed.abort_checkpoint_capture(*observed_capturing);
    if (observed.gpu_memory_stats().reserved_growth_bytes != 256)
      fail("KV snapshot self-test", "aborted capture kept its COW reservation");
    observed.release_execution(*observed_capturing);
    observed.release_gpu(private_buffer);
    if (observed.gpu_memory_stats().nonreclaimable_bytes != 0 || observed.stats().gpu.used != 0)
      fail("KV snapshot self-test", "snapshot fixture leaked state");

    // CPU allocations are separately tagged and cannot be mistaken for the
    // GPU arena merely because both pools start their allocation IDs at one.
    BytePool direct_gpu_pool(1'024, Tier::gpu);
    BytePool direct_cpu_pool(1'024, Tier::cpu);
    const Allocation direct_gpu = direct_gpu_pool.allocate(256);
    const Allocation direct_cpu = direct_cpu_pool.allocate(256);
    if (direct_gpu.tier != Tier::gpu || direct_cpu.tier != Tier::cpu ||
        !direct_gpu_pool.owns(direct_gpu) ||
        !direct_cpu_pool.owns(direct_cpu) ||
        direct_gpu_pool.owns(direct_cpu) ||
        direct_cpu_pool.owns(direct_gpu)) {
      fail("KV cache self-test", "byte pools did not preserve tier identity");
    }
    direct_gpu_pool.release(direct_gpu);
    direct_cpu_pool.release(direct_cpu);

    // Aggregate free bytes are insufficient when the required allocation is
    // larger than every free range. The non-mutating sequence check protects
    // admission from this fragmentation case.
    BytePool fragmented_pool(768, Tier::gpu);
    const Allocation fragmented_first = fragmented_pool.allocate(256);
    const Allocation fragmented_middle = fragmented_pool.allocate(256);
    const Allocation fragmented_last = fragmented_pool.allocate(256);
    fragmented_pool.release(fragmented_first);
    fragmented_pool.release(fragmented_last);
    const PoolStats fragmented_stats = fragmented_pool.stats();
    if (fragmented_stats.free != 512 ||
        fragmented_pool.can_allocate_sequence({512}) ||
        !fragmented_pool.can_allocate_sequence({256, 256}) ||
        fragmented_pool.stats().used != fragmented_stats.used ||
        fragmented_pool.stats().peak_used != fragmented_stats.peak_used) {
      fail("KV cache self-test",
           "fragmented allocation preflight changed pool state or fit a gap");
    }
    fragmented_pool.release(fragmented_middle);

    PoolConfig cpu_pool_config = config;
    cpu_pool_config.cpu_bytes = 1'024;
    cpu_pool_config.validate();
    CacheLedger cpu_pool_ledger(cpu_pool_config);
    const Allocation cpu_allocation = cpu_pool_ledger.try_allocate_cpu(256);
    if (!cpu_allocation.valid() || cpu_allocation.tier != Tier::cpu ||
        !cpu_pool_ledger.cpu_pool().owns(cpu_allocation) ||
        cpu_pool_ledger.gpu_pool().owns(cpu_allocation) ||
        cpu_pool_ledger.stats().cpu.used != cpu_allocation.bytes) {
      fail("KV cache self-test", "ledger CPU allocation was not isolated");
    }
    cpu_pool_ledger.release_cpu(cpu_allocation);
    if (cpu_pool_ledger.stats().cpu.used != 0) {
      fail("KV cache self-test", "ledger CPU allocation leaked");
    }

    // Two resident requests reserve all future pages without allocating them.
    // Arbitrarily interleaved growth must still reach both declared limits;
    // finishing a request immediately makes its complete reservation reusable.
    PoolConfig batch_config = config;
    batch_config.gpu_bytes = 4'608;
    batch_config.page_table_bytes = 256;
    CacheLedger batch(batch_config);
    const auto batch_first = batch.try_begin_batch(0, 1'024);
    const auto batch_second = batch.try_begin_batch(0, 1'024);
    if (!batch_first || !batch_second ||
        batch.stats().gpu.used != 2 * (1'024 + 256) ||
        batch.stats().page_count != 0 || batch.try_begin_batch(0, 256) ||
        batch.execution(*batch_first).local_ring.id ==
            batch.execution(*batch_second).local_ring.id ||
        batch.try_allocate_gpu(256).valid()) {
      fail("KV cache self-test", "batch admission did not reserve private growth");
    }
    bool rejected_mixed_execution = false;
    try {
      (void)batch.begin_execution();
    } catch (const std::runtime_error&) {
      rejected_mixed_execution = true;
    }
    (void)batch.prepare_write(*batch_first, 0, 257);
    (void)batch.prepare_write(*batch_second, 0, 256);
    bool rejected_batch_overrun = false;
    try {
      (void)batch.prepare_write(*batch_first, 257, 768);
    } catch (const std::runtime_error&) {
      rejected_batch_overrun = true;
    }
    bool rejected_checkpoint_pressure = false;
    try {
      (void)batch.publish_checkpoint(
          *batch_first, std::vector<std::uint8_t>(257),
          std::vector<std::uint8_t>(16));
    } catch (const std::runtime_error&) {
      rejected_checkpoint_pressure = true;
    }
    if (!rejected_mixed_execution || !rejected_batch_overrun ||
        !rejected_checkpoint_pressure ||
        batch.try_begin_checkpoint_capture(*batch_first) ||
        batch.execution(*batch_first).processed_tokens != 257) {
      fail("KV cache self-test", "batch reservation boundaries were not enforced");
    }
    (void)batch.prepare_write(*batch_second, 256, 513);
    (void)batch.prepare_write(*batch_first, 257, 767);
    (void)batch.prepare_write(*batch_second, 769, 255);
    if (batch.stats().gpu.used != batch_config.gpu_bytes) {
      fail("KV cache self-test", "interleaved batches did not reach reserved growth");
    }
    batch.release_execution(*batch_first);
    const auto batch_replacement = batch.try_begin_batch(0, 1'024);
    if (!batch_replacement) {
      fail("KV cache self-test", "completed batch capacity was not reusable");
    }
    (void)batch.prepare_write(*batch_replacement, 0, 1);
    batch.release_execution(*batch_replacement);
    batch.release_execution(*batch_second);
    if (batch.stats().gpu.used != 0 || batch.stats().index_used != 0) {
      fail("KV cache self-test", "batch completion/cancellation leaked capacity");
    }

    // Admission must reserve page metadata before any pages are populated.
    PoolConfig batch_index_config = batch_config;
    const std::size_t batch_execution_metadata =
        sizeof(ExecutionInfo) + 8 * sizeof(PageId);
    batch_index_config.index_bytes =
        2 * batch_execution_metadata + 3 * sizeof(PageInfo);
    CacheLedger batch_index(batch_index_config);
    const auto index_first = batch_index.try_begin_batch(0, 512);
    if (!index_first || batch_index.try_begin_batch(0, 512)) {
      fail("KV cache self-test", "batch admission ignored future page metadata");
    }
    (void)batch_index.prepare_write(*index_first, 0, 257);
    batch_index.release_execution(*index_first);
    const auto index_replacement = batch_index.try_begin_batch(0, 512);
    if (!index_replacement) {
      fail("KV cache self-test", "batch metadata reservation was not released");
    }
    batch_index.release_execution(*index_replacement);
    PoolConfig batch_too_small_config = batch_config;
    batch_too_small_config.gpu_bytes = 1'536;
    CacheLedger batch_too_small(batch_too_small_config);
    bool rejected_intrinsic_batch = false;
    try {
      (void)batch_too_small.try_begin_batch(0, 512);
    } catch (const std::runtime_error&) {
      rejected_intrinsic_batch = true;
    }
    if (!rejected_intrinsic_batch || batch_too_small.stats().gpu.used != 0) {
      fail("KV cache self-test", "intrinsically oversized batch was not rejected");
    }

    // Sufficient total free bytes can still lack the contiguous ring. Once a
    // suitable hole exists, leave space for every remaining page allocation.
    PoolConfig batch_fragmented_config = batch_config;
    batch_fragmented_config.gpu_bytes = 4'096;
    CacheLedger batch_fragmented(batch_fragmented_config);
    std::vector<Allocation> occupied;
    for (std::size_t index = 0; index < 8; ++index) {
      occupied.push_back(batch_fragmented.try_allocate_gpu(512));
    }
    for (std::size_t index = 0; index < 8; index += 2) {
      batch_fragmented.release_gpu(occupied[index]);
    }
    if (batch_fragmented.stats().gpu.free != 2'048 ||
        batch_fragmented.try_begin_batch(0, 512)) {
      fail("KV cache self-test", "batch admission ignored ring fragmentation");
    }
    batch_fragmented.release_gpu(occupied[1]);
    const auto fragmented_batch = batch_fragmented.try_begin_batch(0, 512);
    if (!fragmented_batch) {
      fail("KV cache self-test", "batch admission rejected a valid fragmented plan");
    }
    (void)batch_fragmented.prepare_write(*fragmented_batch, 0, 512);
    batch_fragmented.release_execution(*fragmented_batch);
    for (std::size_t index = 3; index < 8; index += 2) {
      batch_fragmented.release_gpu(occupied[index]);
    }
    if (batch_fragmented.stats().gpu.used != 0) {
      fail("KV cache self-test", "fragmented batch lifecycle leaked allocations");
    }

    const auto retain_prefix = [](CacheLedger& cache, std::uint32_t tokens) {
      const ExecutionId producer = cache.begin_execution();
      (void)cache.prepare_write(producer, 0, tokens);
      const CheckpointId checkpoint = cache.publish_checkpoint(
          producer,
          std::vector<std::uint8_t>(std::min(
              tokens, cache.config().local_window_tokens), 0x5a),
          std::vector<std::uint8_t>(cache.config().terminal_hidden_bytes, 0xa5));
      cache.release_execution(producer);
      return checkpoint;
    };
    const auto same_usage = [](const CacheStats& left, const CacheStats& right) {
      return left.gpu.used == right.gpu.used &&
             left.index_used == right.index_used &&
             left.page_count == right.page_count &&
             left.checkpoint_count == right.checkpoint_count &&
             left.execution_count == right.execution_count;
    };

    // Two continuations fit only by borrowing the same 4K global prefix.
    // The ring/table and both future pages remain private to each execution.
    PoolConfig prefix_config = batch_config;
    prefix_config.maximum_context_tokens = 8'192;
    prefix_config.terminal_hidden_bytes = 256;
    prefix_config.gpu_bytes = 8'960;
    CacheLedger shared(prefix_config);
    const CheckpointId shared_source = retain_prefix(shared, 4'096);
    const CacheStats shared_baseline = shared.stats();
    const std::vector<PageId> shared_pages =
        shared.checkpoint(shared_source).global_pages;
    const auto shared_first = shared.try_begin_batch(shared_source, 4'608);
    const auto shared_second = shared.try_begin_batch(shared_source, 4'608);
    if (!shared_first || !shared_second ||
        shared.try_begin_batch(shared_source, 4'096) ||
        shared.execution(*shared_first).global_pages != shared_pages ||
        shared.execution(*shared_second).global_pages != shared_pages ||
        shared.execution(*shared_first).processed_tokens != 4'096 ||
        shared.execution(*shared_second).source_checkpoint != shared_source ||
        shared.execution(*shared_first).local_ring.id ==
            shared.execution(*shared_second).local_ring.id ||
        shared.execution(*shared_first).page_table.id ==
            shared.execution(*shared_second).page_table.id ||
        shared.stats().gpu.used != shared_baseline.gpu.used + 2 * 1'280 ||
        shared.checkpoint(shared_source).execution_references != 2 ||
        shared.evict_idle_checkpoint()) {
      fail("KV cache self-test", "batch prefix admission did not share protected state");
    }
    bool rejected_borrowed_release = false;
    try {
      shared.release_checkpoint(shared_source);
    } catch (const std::runtime_error&) {
      rejected_borrowed_release = true;
    }
    if (!rejected_borrowed_release) {
      fail("KV cache self-test", "batch admission did not protect its source");
    }
    for (const PageId page_id : shared_pages) {
      if (shared.page(page_id).references != 3) {
        fail("KV cache self-test", "shared batch prefix has incorrect references");
      }
    }
    (void)shared.prepare_write(*shared_first, 4'096, 256);
    (void)shared.prepare_write(*shared_second, 4'096, 257);
    (void)shared.prepare_write(*shared_first, 4'352, 256);
    if (shared.stats().gpu.used != prefix_config.gpu_bytes ||
        shared.stats().copy_on_write_pages != 0) {
      fail("KV cache self-test", "aligned prefix growth exceeded private reservations");
    }
    shared.release_execution(*shared_first);
    (void)shared.prepare_write(*shared_second, 4'353, 255);
    for (std::size_t index = 0; index < shared_pages.size(); ++index) {
      if (shared.execution(*shared_second).global_pages[index] != shared_pages[index] ||
          shared.page(shared_pages[index]).references != 2) {
        fail("KV cache self-test", "finishing a borrower damaged its sibling");
      }
    }
    shared.release_execution(*shared_second);
    if (!same_usage(shared.stats(), shared_baseline) ||
        shared.checkpoint(shared_source).execution_references != 0) {
      fail("KV cache self-test", "shared batch cleanup missed the retained baseline");
    }
    shared.release_checkpoint(shared_source);
    if (shared.stats().gpu.used != 0 || shared.stats().index_used != 0) {
      fail("KV cache self-test", "shared batch prefix leaked after release");
    }

    // A short final allocation need not occupy its trailing alignment gap.
    // A complete hit can reuse the producer's exact ring/table holes.
    PoolConfig short_hidden_config = prefix_config;
    short_hidden_config.terminal_hidden_bytes = 16;
    short_hidden_config.gpu_bytes = 6'416;
    CacheLedger short_hidden(short_hidden_config);
    const CheckpointId short_hidden_source = retain_prefix(short_hidden, 4'096);
    const auto complete_hit = short_hidden.try_begin_batch(short_hidden_source, 4'096);
    if (!complete_hit || short_hidden.stats().gpu.used != short_hidden_config.gpu_bytes) {
      fail("KV cache self-test", "complete prefix hit charged trailing alignment padding");
    }
    short_hidden.release_execution(*complete_hit);

    // Externally supplied snapshots may occupy alignment gaps. Admission must
    // retain their existing placement instead of assuming new 256-byte slots.
    PoolConfig packed_config = prefix_config;
    packed_config.global_page_tokens = packed_config.local_window_tokens =
        packed_config.maximum_context_tokens = 1;
    packed_config.local_ring_bytes = packed_config.local_bytes_per_token =
        packed_config.terminal_hidden_bytes = 1;
    packed_config.page_table_bytes = 8;
    packed_config.gpu_bytes = 768;
    CacheLedger packed(packed_config);
    const ExecutionId packed_producer = packed.begin_execution();
    (void)packed.prepare_write(packed_producer, 0, 1);
    const Allocation packed_snapshot = packed.try_allocate_gpu(1, 1);
    const Allocation packed_hidden = packed.try_allocate_gpu(1, 1);
    const CheckpointId packed_source = packed.publish_checkpoint_allocations(
        packed_producer, packed_snapshot, packed_hidden);
    packed.release_execution(packed_producer);
    const CacheStats packed_baseline = packed.stats();
    const auto packed_hit = packed.try_begin_batch(packed_source, 1);
    if (!packed_hit || packed_snapshot.offset != 1 || packed_hidden.offset != 2 ||
        packed.execution(*packed_hit).local_ring.offset != 0 ||
        packed.execution(*packed_hit).page_table.offset != 256) {
      fail("KV cache self-test", "prefix admission rejected snapshots in alignment gaps");
    }
    packed.release_execution(*packed_hit);
    if (!same_usage(packed.stats(), packed_baseline)) {
      fail("KV cache self-test", "packed prefix admission leaked private state");
    }

    // A partial inherited tail needs one COW reservation even if no page is
    // appended. Complete hits need none; crossing the next boundary needs both.
    for (const std::uint32_t tokens : {4'095U, 4'097U}) {
      for (const std::uint32_t growth : {0U, 1U, 257U}) {
        const std::size_t source_pages = (tokens + 255) / 256;
        const std::size_t private_pages =
            (tokens + growth + 255) / 256 - source_pages + (growth != 0);
        PoolConfig tail_config = prefix_config;
        tail_config.gpu_bytes = source_pages * 256 + 1'280 +
                                2 * (1'280 + private_pages * 256);
        CacheLedger tails(tail_config);
        const CheckpointId source = retain_prefix(tails, tokens);
        const CacheStats baseline = tails.stats();
        const PageId source_tail = tails.checkpoint(source).global_pages.back();
        const auto first = tails.try_begin_batch(source, tokens + growth);
        if (!first) {
          fail("KV cache self-test", "partial prefix admission rejected reserved growth");
        }
        WritePlan first_cow;
        if (growth != 0) {
          first_cow = tails.prepare_write(*first, tokens, 1);
        }
        // A consumed COW reservation must not be charged again when admitting
        // another borrower before the first execution finishes growing.
        const auto second = tails.try_begin_batch(source, tokens + growth);
        if (!second || tails.try_begin_batch(source, tokens)) {
          fail("KV cache self-test", "partial prefix reservation admitted wrong capacity");
        }
        if (growth != 0) {
          const WritePlan second_cow = tails.prepare_write(*second, tokens, 1);
          if (first_cow.copies.size() != 1 || second_cow.copies.size() != 1 ||
              !first_cow.new_pages.empty() || !second_cow.new_pages.empty() ||
              first_cow.copies.front().source != source_tail ||
              second_cow.copies.front().source != source_tail ||
              first_cow.copies.front().destination ==
                  second_cow.copies.front().destination ||
              tails.page(source_tail).references != 1 ||
              tails.page(source_tail).valid_tokens != tokens % 256) {
            fail("KV cache self-test", "partial prefix COW did not isolate both tails");
          }
          if (growth > 1) {
            (void)tails.prepare_write(*second, tokens + 1, growth - 1);
            (void)tails.prepare_write(*first, tokens + 1, growth - 1);
          }
        }
        if (tails.stats().gpu.used != tail_config.gpu_bytes ||
            tails.stats().copy_on_write_pages != (growth == 0 ? 0 : 2)) {
          fail("KV cache self-test", "partial prefix growth escaped its reservation");
        }
        tails.release_execution(*first);
        tails.release_execution(*second);
        if (!same_usage(tails.stats(), baseline)) {
          fail("KV cache self-test", "partial prefix batch cleanup leaked capacity");
        }
      }
    }

    // Enough raw free space for a second ring/table still cannot spend the
    // first borrower's not-yet-copied tail reservation.
    PoolConfig pending_tail_config = prefix_config;
    pending_tail_config.gpu_bytes = 8'192;
    CacheLedger pending_tail(pending_tail_config);
    const CheckpointId pending_source = retain_prefix(pending_tail, 4'097);
    const CacheStats pending_baseline = pending_tail.stats();
    const auto pending_first = pending_tail.try_begin_batch(pending_source, 4'098);
    if (!pending_first || pending_tail.stats().gpu.free != 1'280 ||
        pending_tail.try_begin_batch(pending_source, 4'097)) {
      fail("KV cache self-test", "admission spent another borrower's pending tail COW");
    }
    const WritePlan pending_copy = pending_tail.prepare_write(*pending_first, 4'097, 1);
    if (pending_copy.copies.size() != 1 || !pending_copy.new_pages.empty()) {
      fail("KV cache self-test", "reserved tail COW was not available after rejection");
    }
    pending_tail.release_execution(*pending_first);
    if (!same_usage(pending_tail.stats(), pending_baseline)) {
      fail("KV cache self-test", "pending tail reservation leaked after release");
    }

    CacheLedger invalid_prefix(prefix_config);
    const CheckpointId valid_source = retain_prefix(invalid_prefix, 4'096);
    const CacheStats invalid_baseline = invalid_prefix.stats();
    for (const auto& request :
         std::vector<std::pair<CheckpointId, std::size_t>>{
             {valid_source + 1, 4'608}, {valid_source, 0},
             {valid_source, 4'095}, {valid_source, 8'193}}) {
      bool rejected = false;
      try {
        (void)invalid_prefix.try_begin_batch(request.first, request.second);
      } catch (const std::runtime_error&) {
        rejected = true;
      }
      if (!rejected || !same_usage(invalid_prefix.stats(), invalid_baseline) ||
          invalid_prefix.stats().gpu.peak_used != invalid_baseline.gpu.peak_used ||
          invalid_prefix.checkpoint(valid_source).execution_references != 0) {
        fail("KV cache self-test", "invalid prefix admission changed retained state");
      }
    }

    // The selected source is part of the unavoidable working set. Pressure
    // from a live allocation returns nullopt without sacrificing that source.
    for (const bool temporary : {false, true}) {
      PoolConfig pressure_config = prefix_config;
      pressure_config.gpu_bytes = 5'376 + (temporary ? 1'792 : 1'280);
      CacheLedger pressure(pressure_config);
      const CheckpointId source = retain_prefix(pressure, 4'096);
      const Allocation blocker = temporary ? pressure.try_allocate_gpu(256)
                                           : Allocation{};
      const CacheStats baseline = pressure.stats();
      if (pressure.batch_request_fits(source, 4'608) != temporary ||
          !pressure.batch_request_fits(0, 4'608) ||
          !same_usage(pressure.stats(), baseline)) {
        fail("KV cache self-test", "intrinsic fit did not distinguish prefix overhead from pressure");
      }
      bool rejected = false;
      std::optional<ExecutionId> admitted;
      try {
        admitted = pressure.try_begin_batch(source, 4'608);
      } catch (const std::runtime_error&) {
        rejected = true;
      }
      if (admitted || rejected == temporary ||
          !same_usage(pressure.stats(), baseline) ||
          pressure.stats().gpu.peak_used != baseline.gpu.peak_used ||
          !pressure.has_checkpoint(source) ||
          pressure.checkpoint(source).execution_references != 0) {
        fail("KV cache self-test", "failed admission sacrificed its prefix source");
      }
      if (temporary) {
        pressure.release_gpu(blocker);
        const auto retry = pressure.try_begin_batch(source, 4'608);
        if (!retry) {
          fail("KV cache self-test", "prefix admission did not recover after pressure");
        }
        pressure.release_execution(*retry);
      }
    }

    // Admission may reclaim other idle checkpoints, even when the selected
    // source is older. It must protect that source before borrowing begins.
    CacheLedger selective_pressure(prefix_config);
    const CheckpointId protected_source = retain_prefix(selective_pressure, 4'096);
    const CacheStats protected_baseline = selective_pressure.stats();
    const CheckpointId idle_victim = retain_prefix(selective_pressure, 256);
    const Allocation pressure_blocker = selective_pressure.try_allocate_gpu(1'280);
    const auto protected_batch = selective_pressure.try_begin_batch(protected_source, 4'608);
    if (!pressure_blocker.valid() || !protected_batch ||
        !selective_pressure.has_checkpoint(protected_source) ||
        selective_pressure.has_checkpoint(idle_victim) ||
        selective_pressure.checkpoint(protected_source).execution_references != 1) {
      fail("KV cache self-test", "batch pressure evicted its selected source");
    }
    (void)selective_pressure.prepare_write(*protected_batch, 4'096, 512);
    selective_pressure.release_execution(*protected_batch);
    selective_pressure.release_gpu(pressure_blocker);
    if (!same_usage(selective_pressure.stats(), protected_baseline)) {
      fail("KV cache self-test", "selective batch eviction leaked retained state");
    }

    // Source pages already have metadata; only future private pages need new
    // records. Existing borrowers' reservations must still be counted.
    PoolConfig prefix_index_config = prefix_config;
    const std::size_t prefix_execution_metadata =
        sizeof(ExecutionInfo) + 32 * sizeof(PageId);
    prefix_index_config.index_bytes = sizeof(CheckpointInfo) +
        16 * sizeof(PageId) + 16 * sizeof(PageInfo) +
        2 * prefix_execution_metadata + 3 * sizeof(PageInfo);
    CacheLedger prefix_index(prefix_index_config);
    const CheckpointId index_source = retain_prefix(prefix_index, 4'096);
    const CacheStats prefix_index_baseline = prefix_index.stats();
    const auto prefix_index_first = prefix_index.try_begin_batch(index_source, 4'608);
    if (!prefix_index_first || prefix_index.try_begin_batch(index_source, 4'608)) {
      fail("KV cache self-test", "shared batch metadata reservation was incorrect");
    }
    (void)prefix_index.prepare_write(*prefix_index_first, 4'096, 512);
    prefix_index.release_execution(*prefix_index_first);
    if (!same_usage(prefix_index.stats(), prefix_index_baseline)) {
      fail("KV cache self-test", "shared batch metadata reservation leaked");
    }

    // Fragment the free ranges around a retained source into 256-byte holes.
    // A ring cannot fit until adjacent holes are joined, even with ample bytes.
    PoolConfig prefix_fragmented_config = prefix_config;
    prefix_fragmented_config.gpu_bytes = 10'496;
    CacheLedger prefix_fragmented(prefix_fragmented_config);
    const CheckpointId fragmented_source = retain_prefix(prefix_fragmented, 4'096);
    const CacheStats prefix_fragmented_baseline = prefix_fragmented.stats();
    std::vector<Allocation> prefix_occupied;
    for (std::size_t index = 0; index < 20; ++index) {
      const Allocation allocation = prefix_fragmented.try_allocate_gpu(256);
      if (!allocation.valid()) {
        fail("KV cache self-test", "prefix fragmentation setup exceeded its budget");
      }
      prefix_occupied.push_back(allocation);
    }
    for (std::size_t index = 0; index < 20; index += 2) {
      prefix_fragmented.release_gpu(prefix_occupied[index]);
    }
    const CacheStats prefix_fragmented_pressure = prefix_fragmented.stats();
    if (prefix_fragmented.try_begin_batch(fragmented_source, 4'352) ||
        !same_usage(prefix_fragmented.stats(), prefix_fragmented_pressure) ||
        !prefix_fragmented.has_checkpoint(fragmented_source)) {
      fail("KV cache self-test", "prefix admission ignored ring fragmentation");
    }
    prefix_fragmented.release_gpu(prefix_occupied[1]);
    prefix_fragmented.release_gpu(prefix_occupied[3]);
    const auto prefix_fragmented_batch =
        prefix_fragmented.try_begin_batch(fragmented_source, 4'352);
    if (!prefix_fragmented_batch) {
      fail("KV cache self-test", "prefix admission rejected a valid fragmented plan");
    }
    (void)prefix_fragmented.prepare_write(*prefix_fragmented_batch, 4'096, 256);
    prefix_fragmented.release_execution(*prefix_fragmented_batch);
    for (std::size_t index = 5; index < 20; index += 2) {
      prefix_fragmented.release_gpu(prefix_occupied[index]);
    }
    if (!same_usage(prefix_fragmented.stats(), prefix_fragmented_baseline)) {
      fail("KV cache self-test", "fragmented shared batch leaked allocations");
    }

    // Pre-existing raw allocations cannot bypass collective reservations by
    // publishing a batch snapshot without a pending capture.
    CacheLedger raw_capture(config);
    const Allocation raw_local = raw_capture.try_allocate_gpu(257);
    const Allocation raw_hidden = raw_capture.try_allocate_gpu(16);
    const auto raw_execution = raw_capture.try_begin_batch(0, 512);
    (void)raw_capture.prepare_write(*raw_execution, 0, 257);
    const CacheStats raw_baseline = raw_capture.stats();
    bool rejected_raw_capture = false;
    try {
      (void)raw_capture.publish_checkpoint_allocations(
          *raw_execution, raw_local, raw_hidden);
    } catch (const std::runtime_error&) {
      rejected_raw_capture = true;
    }
    if (!rejected_raw_capture || !same_usage(raw_capture.stats(), raw_baseline)) {
      fail("KV cache self-test", "raw batch publication bypassed capture reservation");
    }
    raw_capture.release_execution(*raw_execution);
    raw_capture.release_gpu(raw_local);
    raw_capture.release_gpu(raw_hidden);

    // Queued handoffs pin their checkpoint without paying for an execution
    // ring. Pressure can evict other idle state, but neither explicit release
    // nor borrower completion may discard the pinned exact boundary.
    CacheLedger pinned(prefix_config);
    const CheckpointId pinned_source = retain_prefix(pinned, 4'096);
    pinned.pin_checkpoint(pinned_source);
    pinned.pin_checkpoint(pinned_source);
    bool rejected_pinned_release = false;
    try {
      pinned.release_checkpoint(pinned_source);
    } catch (const std::runtime_error&) {
      rejected_pinned_release = true;
    }
    if (!rejected_pinned_release || pinned.evict_idle_checkpoint() ||
        pinned.checkpoint(pinned_source).execution_references != 0 ||
        pinned.checkpoint(pinned_source).dependency_pins != 2) {
      fail("KV cache self-test", "queued checkpoint pin did not protect exact state");
    }
    const CheckpointId pin_victim = retain_prefix(pinned, 256);
    const auto pin_pressure = pinned.try_begin_batch(0, 2'048);
    if (!pin_pressure || !pinned.has_checkpoint(pinned_source) ||
        pinned.has_checkpoint(pin_victim)) {
      fail("KV cache self-test", "admission pressure evicted a queued handoff");
    }
    pinned.release_execution(*pin_pressure);
    const auto pinned_borrower = pinned.try_begin_batch(pinned_source, 4'352);
    if (!pinned_borrower ||
        pinned.checkpoint(pinned_source).execution_references != 1 ||
        pinned.checkpoint(pinned_source).dependency_pins != 2) {
      fail("KV cache self-test", "checkpoint pin was confused with an active borrower");
    }
    pinned.release_execution(*pinned_borrower);
    if (pinned.try_begin_batch(0, 4'096) ||
        pinned.checkpoint(pinned_source).execution_references != 0) {
      fail("KV cache self-test", "borrower release dropped a waiting handoff pin");
    }
    pinned.unpin_checkpoint(pinned_source);
    if (pinned.evict_idle_checkpoint() ||
        pinned.checkpoint(pinned_source).dependency_pins != 1) {
      fail("KV cache self-test", "cancelling one dependent discarded another's checkpoint");
    }
    pinned.unpin_checkpoint(pinned_source);
    bool rejected_pin_underflow = false;
    try {
      pinned.unpin_checkpoint(pinned_source);
    } catch (const std::runtime_error&) {
      rejected_pin_underflow = true;
    }
    if (!rejected_pin_underflow || !pinned.evict_idle_checkpoint() ||
        pinned.stats().gpu.used != 0 || pinned.stats().index_used != 0) {
      fail("KV cache self-test", "final handoff unpin did not restore eviction eligibility");
    }

    // A live 4K trunk forks without a retained checkpoint or duplicate global
    // pages. Its successor remains valid after the original execution leaves.
    PoolConfig live_fork_config = prefix_config;
    live_fork_config.gpu_bytes = 7'680;
    CacheLedger live_fork(live_fork_config);
    const auto live_source = live_fork.try_begin_batch(0, 4'608);
    (void)live_fork.prepare_write(*live_source, 0, 4'096);
    const std::vector<PageId> live_pages = live_fork.execution(*live_source).global_pages;
    const auto live_child = live_fork.try_fork_batch(*live_source, 4'608);
    if (!live_child || live_fork.stats().checkpoint_count != 0 ||
        live_fork.execution(*live_child).processed_tokens != 4'096 ||
        live_fork.execution(*live_child).global_pages != live_pages ||
        live_fork.execution(*live_child).source_checkpoint != 0 ||
        live_fork.execution(*live_child).local_ring.id ==
            live_fork.execution(*live_source).local_ring.id ||
        live_fork.execution(*live_child).page_table.id ==
            live_fork.execution(*live_source).page_table.id ||
        live_fork.try_begin_checkpoint_capture(*live_source) ||
        live_fork.try_fork_batch(*live_source, 4'096)) {
      fail("KV cache self-test", "live fork duplicated its trunk or spent reserved growth");
    }
    (void)live_fork.prepare_write(*live_source, 4'096, 256);
    (void)live_fork.prepare_write(*live_child, 4'096, 257);
    live_fork.release_execution(*live_source);
    for (const PageId page_id : live_pages) {
      if (live_fork.page(page_id).references != 1) {
        fail("KV cache self-test", "live fork retained an execution lifetime dependency");
      }
    }
    (void)live_fork.prepare_write(*live_child, 4'353, 255);
    live_fork.release_execution(*live_child);
    if (live_fork.stats().gpu.used != 0 || live_fork.stats().index_used != 0) {
      fail("KV cache self-test", "live fork cancellation leaked capacity");
    }

    // Sharing a private partial tail introduces COW obligations for both the
    // source and the fork when either may grow. Cover GPU and metadata limits
    // one byte below, and exactly at, that collective reservation.
    for (const bool source_grows : {false, true}) {
      for (const bool child_grows : {false, true}) {
        for (const bool enough : {false, true}) {
          for (const bool constrain_index : {false, true}) {
            PoolConfig fork_tail_config = batch_config;
            const std::size_t cow_pages = source_grows + child_grows;
            fork_tail_config.gpu_bytes = 3'072 + cow_pages * 256 -
                (!enough && !constrain_index);
            fork_tail_config.index_bytes = 2 * batch_execution_metadata +
                (2 + cow_pages) * sizeof(PageInfo) -
                (!enough && constrain_index);
            CacheLedger fork_tail(fork_tail_config);
            const auto producer = fork_tail.try_begin_batch(0, 257 + source_grows);
            (void)fork_tail.prepare_write(*producer, 0, 257);
            const PageId tail = fork_tail.execution(*producer).global_pages.back();
            const CacheStats baseline = fork_tail.stats();
            const auto child = fork_tail.try_fork_batch(*producer, 257 + child_grows);
            if (child.has_value() != enough ||
                (!child && (!same_usage(fork_tail.stats(), baseline) ||
                            fork_tail.page(tail).references != 1))) {
              fail("KV cache self-test", "fork ignored source or borrower tail COW reservation");
            }
            if (source_grows) {
              const WritePlan growth = fork_tail.prepare_write(*producer, 257, 1);
              if (growth.copies.size() != (child ? 1U : 0U)) {
                fail("KV cache self-test", "fork failed to preserve the source's private tail");
              }
            }
            fork_tail.release_execution(*producer);
            if (child) {
              if (fork_tail.page(tail).valid_tokens != 1) {
                fail("KV cache self-test", "producer changed a fork's partial tail");
              }
              if (child_grows) {
                (void)fork_tail.prepare_write(*child, 257, 1);
              }
              fork_tail.release_execution(*child);
            }
            if (fork_tail.stats().gpu.used != 0 || fork_tail.stats().index_used != 0) {
              fail("KV cache self-test", "partial-tail fork leaked resources");
            }
          }
        }
      }
    }

    // Reparenting a frozen trunk changes only its growth limit. A failed
    // expansion preserves that limit and every allocation/page identity; after
    // another execution leaves, the same trunk can expand without replay.
    CacheLedger reparent(batch_config);
    const auto trunk = reparent.try_begin_batch(0, 512);
    (void)reparent.prepare_write(*trunk, 0, 257);
    const ExecutionInfo trunk_state = reparent.execution(*trunk);
    if (!reparent.try_resize_batch(*trunk, 257)) {
      fail("KV cache self-test", "frozen trunk could not release unused growth");
    }
    const auto sibling = reparent.try_fork_batch(*trunk, 1'024);
    if (!sibling) {
      fail("KV cache self-test", "reparent setup could not fork a sibling");
    }
    const CacheStats before_reparent = reparent.stats();
    if (reparent.try_resize_batch(*trunk, 2'048) ||
        reparent.execution(*trunk).batch_max_processed_tokens != 257 ||
        !same_usage(reparent.stats(), before_reparent)) {
      fail("KV cache self-test", "failed horizon expansion changed a frozen trunk");
    }
    if (!reparent.try_resize_batch(*trunk, 768) ||
        reparent.execution(*trunk).id != trunk_state.id ||
        reparent.execution(*trunk).processed_tokens != trunk_state.processed_tokens ||
        reparent.execution(*trunk).global_pages != trunk_state.global_pages ||
        reparent.execution(*trunk).local_ring.id != trunk_state.local_ring.id ||
        reparent.execution(*trunk).page_table.id != trunk_state.page_table.id ||
        !same_usage(reparent.stats(), before_reparent)) {
      fail("KV cache self-test", "reparenting copied state or lost the existing reservation");
    }
    (void)reparent.prepare_write(*sibling, 257, 767);
    reparent.release_execution(*sibling);
    if (!reparent.try_resize_batch(*trunk, 2'048)) {
      fail("KV cache self-test", "surviving trunk could not reuse cancelled capacity");
    }
    (void)reparent.prepare_write(*trunk, 257, 1'791);
    reparent.release_execution(*trunk);
    if (reparent.stats().gpu.used != 0 || reparent.stats().index_used != 0) {
      fail("KV cache self-test", "reparented trunk leaked capacity");
    }

    PoolConfig resize_index_config = batch_config;
    resize_index_config.index_bytes =
        2 * batch_execution_metadata + 3 * sizeof(PageInfo);
    CacheLedger resize_index(resize_index_config);
    const auto resize_first = resize_index.try_begin_batch(0, 256);
    const auto resize_second = resize_index.try_begin_batch(0, 256);
    (void)resize_index.prepare_write(*resize_first, 0, 256);
    const CacheStats resize_baseline = resize_index.stats();
    EvictionCause resize_cause = EvictionCause::gpu_pressure;
    if (resize_index.try_resize_batch(*resize_first, 768, &resize_cause) ||
        resize_cause != EvictionCause::index_pressure ||
        resize_index.execution(*resize_first).batch_max_processed_tokens != 256 ||
        !same_usage(resize_index.stats(), resize_baseline) ||
        !resize_index.try_resize_batch(*resize_first, 512)) {
      fail("KV cache self-test", "horizon expansion spent another execution's page metadata");
    }
    (void)resize_index.prepare_write(*resize_first, 256, 256);
    (void)resize_index.prepare_write(*resize_second, 0, 256);
    resize_index.release_execution(*resize_first);
    resize_index.release_execution(*resize_second);

    // Expansion must prove a physical layout, not only sum the free bytes.
    PoolConfig resize_fragmented_config = batch_config;
    resize_fragmented_config.gpu_bytes = 3'840;
    resize_fragmented_config.global_page_bytes = 512;
    CacheLedger resize_fragmented(resize_fragmented_config);
    const Allocation resize_ring_gap = resize_fragmented.try_allocate_gpu(1'280);
    std::vector<Allocation> resize_blockers;
    for (std::size_t index = 0; index < 10; ++index) {
      resize_blockers.push_back(resize_fragmented.try_allocate_gpu(256));
    }
    resize_fragmented.release_gpu(resize_ring_gap);
    for (const std::size_t index : {0U, 1U, 3U, 5U, 7U, 9U}) {
      resize_fragmented.release_gpu(resize_blockers[index]);
    }
    const auto resize_fragmented_execution = resize_fragmented.try_begin_batch(0, 256);
    (void)resize_fragmented.prepare_write(*resize_fragmented_execution, 0, 256);
    const CacheStats resize_fragmented_baseline = resize_fragmented.stats();
    if (resize_fragmented_baseline.gpu.free < 512 ||
        resize_fragmented.try_resize_batch(*resize_fragmented_execution, 512) ||
        !same_usage(resize_fragmented.stats(), resize_fragmented_baseline)) {
      fail("KV cache self-test", "horizon expansion ignored global-page fragmentation");
    }
    resize_fragmented.release_gpu(resize_blockers[4]);
    if (!resize_fragmented.try_resize_batch(*resize_fragmented_execution, 512)) {
      fail("KV cache self-test", "horizon expansion did not recover after holes joined");
    }
    (void)resize_fragmented.prepare_write(*resize_fragmented_execution, 256, 256);
    resize_fragmented.release_execution(*resize_fragmented_execution);
    for (const std::size_t index : {2U, 6U, 8U}) {
      resize_fragmented.release_gpu(resize_blockers[index]);
    }

    // Capturing an endpoint during two active executions must preserve every
    // admitted page, including the COW newly introduced by its partial tail.
    PoolConfig capture_config = batch_config;
    capture_config.gpu_bytes = 5'632;
    CacheLedger capturing(capture_config);
    const auto capturing_first = capturing.try_begin_batch(0, 1'024);
    const auto capturing_second = capturing.try_begin_batch(0, 1'024);
    if (!capturing_first || !capturing_second) {
      fail("KV cache self-test", "concurrent capture setup failed");
    }
    (void)capturing.prepare_write(*capturing_first, 0, 257);
    (void)capturing.prepare_write(*capturing_second, 0, 256);
    const PageId captured_tail = capturing.execution(*capturing_first).global_pages.back();
    const CacheStats before_capture = capturing.stats();
    const std::size_t captured_metadata =
        sizeof(CheckpointInfo) + 2 * sizeof(PageId);
    const auto abandoned_capture =
        capturing.try_begin_checkpoint_capture(*capturing_first);
    if (!abandoned_capture ||
        capturing.stats().gpu.used != before_capture.gpu.used + 257 + 16 ||
        capturing.stats().index_used != before_capture.index_used + captured_metadata ||
        capturing.page(captured_tail).references != 2) {
      fail("KV cache self-test", "pending capture did not own its allocations and pages");
    }
    bool rejected_pending_write = false;
    try {
      (void)capturing.prepare_write(*capturing_first, 257, 1);
    } catch (const std::runtime_error&) {
      rejected_pending_write = true;
    }
    bool rejected_wrong_capture = false;
    try {
      (void)capturing.publish_checkpoint_allocations(
          *capturing_first, abandoned_capture->terminal_hidden,
          abandoned_capture->local_snapshot);
    } catch (const std::runtime_error&) {
      rejected_wrong_capture = true;
    }
    if (!rejected_pending_write || !rejected_wrong_capture ||
        !capturing.execution(*capturing_first).pending_checkpoint) {
      fail("KV cache self-test", "pending snapshot was mutable or lost on failed publish");
    }
    capturing.abort_checkpoint_capture(*capturing_first);
    if (!same_usage(capturing.stats(), before_capture) ||
        capturing.page(captured_tail).references != 1 ||
        capturing.gpu_pool().owns(abandoned_capture->local_snapshot) ||
        capturing.gpu_pool().owns(abandoned_capture->terminal_hidden)) {
      fail("KV cache self-test", "aborted capture leaked bytes, metadata, or references");
    }
    const auto capture = capturing.try_begin_checkpoint_capture(*capturing_first);
    if (!capture || capturing.try_begin_batch(0, 256) ||
        capturing.try_begin_checkpoint_capture(*capturing_second)) {
      fail("KV cache self-test", "admission spent a pending capture's COW reservation");
    }
    (void)capturing.prepare_write(*capturing_second, 256, 256);
    const CacheStats before_publish = capturing.stats();
    const CheckpointId captured = capturing.publish_checkpoint_allocations(
        *capturing_first, capture->local_snapshot, capture->terminal_hidden);
    if (capturing.execution(*capturing_first).pending_checkpoint ||
        capturing.stats().gpu.used != before_publish.gpu.used ||
        capturing.stats().index_used != before_publish.index_used ||
        capturing.page(captured_tail).references != 2) {
      fail("KV cache self-test", "publication duplicated its pending capture resources");
    }
    const WritePlan capture_growth =
        capturing.prepare_write(*capturing_first, 257, 767);
    (void)capturing.prepare_write(*capturing_second, 512, 512);
    if (capture_growth.copies.size() != 1 ||
        capture_growth.copies.front().source != captured_tail ||
        capturing.page(captured_tail).valid_tokens != 1) {
      fail("KV cache self-test", "publication failed to protect its partial tail");
    }
    capturing.release_execution(*capturing_first);
    const auto captured_borrower = capturing.try_begin_batch(captured, 512);
    if (!captured_borrower ||
        capturing.execution(*captured_borrower).global_pages !=
            capturing.checkpoint(captured).global_pages) {
      fail("KV cache self-test", "published batch checkpoint was not reusable");
    }
    capturing.release_execution(*capturing_second);
    (void)capturing.prepare_write(*captured_borrower, 257, 255);
    capturing.release_execution(*captured_borrower);
    capturing.release_checkpoint(captured);
    if (capturing.stats().gpu.used != 0 || capturing.stats().index_used != 0) {
      fail("KV cache self-test", "concurrent capture cleanup leaked resources");
    }

    // Raw snapshot bytes fit here, but the additional private partial-tail COW
    // does not. Capture must fail before any allocations or references change.
    PoolConfig capture_pressure_config = capture_config;
    capture_pressure_config.gpu_bytes -= 256;
    CacheLedger capture_pressure(capture_pressure_config);
    const auto capture_pressure_first = capture_pressure.try_begin_batch(0, 1'024);
    const auto capture_pressure_second = capture_pressure.try_begin_batch(0, 1'024);
    (void)capture_pressure.prepare_write(*capture_pressure_first, 0, 257);
    (void)capture_pressure.prepare_write(*capture_pressure_second, 0, 256);
    const CacheStats capture_pressure_baseline = capture_pressure.stats();
    EvictionCause capture_pressure_cause = EvictionCause::index_pressure;
    if (capture_pressure.try_begin_checkpoint_capture(
            *capture_pressure_first, &capture_pressure_cause) ||
        capture_pressure_cause != EvictionCause::gpu_pressure ||
        !same_usage(capture_pressure.stats(), capture_pressure_baseline)) {
      fail("KV cache self-test", "capture ignored new tail COW byte headroom");
    }
    (void)capture_pressure.prepare_write(*capture_pressure_first, 257, 767);
    (void)capture_pressure.prepare_write(*capture_pressure_second, 256, 768);
    capture_pressure.release_execution(*capture_pressure_first);
    capture_pressure.release_execution(*capture_pressure_second);

    // The same constraint applies to future PageInfo records, not just bytes.
    // At the exact boundary a pending capture can be cancelled by releasing
    // its execution; no explicit abort from the caller is required.
    for (const bool enough_index : {false, true}) {
      PoolConfig capture_index_config = capture_config;
      capture_index_config.index_bytes = 2 * batch_execution_metadata +
          9 * sizeof(PageInfo) + captured_metadata - !enough_index;
      CacheLedger capture_index(capture_index_config);
      const auto capture_index_first = capture_index.try_begin_batch(0, 1'024);
      const auto capture_index_second = capture_index.try_begin_batch(0, 1'024);
      (void)capture_index.prepare_write(*capture_index_first, 0, 257);
      (void)capture_index.prepare_write(*capture_index_second, 0, 256);
      const CacheStats capture_index_baseline = capture_index.stats();
      EvictionCause capture_index_cause = EvictionCause::gpu_pressure;
      const auto pending = capture_index.try_begin_checkpoint_capture(
          *capture_index_first, &capture_index_cause);
      if (pending.has_value() != enough_index ||
          (!enough_index &&
           capture_index_cause != EvictionCause::index_pressure) ||
          (!pending &&
           !same_usage(capture_index.stats(), capture_index_baseline))) {
        fail("KV cache self-test", "capture ignored tail COW metadata headroom");
      }
      capture_index.release_execution(*capture_index_first);
      (void)capture_index.prepare_write(*capture_index_second, 256, 768);
      capture_index.release_execution(*capture_index_second);
      if (capture_index.stats().gpu.used != 0 ||
          capture_index.stats().index_used != 0) {
        fail("KV cache self-test", "release did not abort its pending checkpoint");
      }
    }

    // Aligned checkpoints and partial tails already at their admitted horizon
    // do not need an extra COW page. Verify both at a tight physical boundary.
    for (const std::uint32_t tokens : {256U, 257U}) {
      PoolConfig endpoint_config = batch_config;
      endpoint_config.gpu_bytes = tokens == 256 ? 2'304 : 2'320;
      CacheLedger endpoint(endpoint_config);
      const std::uint32_t horizon = 257;
      const auto request = endpoint.try_begin_batch(0, horizon);
      (void)endpoint.prepare_write(*request, 0, tokens);
      const auto endpoint_capture = endpoint.try_begin_checkpoint_capture(*request);
      if (!endpoint_capture) {
        fail("KV cache self-test", "capture reserved an unnecessary tail COW page");
      }
      const CheckpointId endpoint_checkpoint = endpoint.publish_checkpoint_allocations(
          *request, endpoint_capture->local_snapshot, endpoint_capture->terminal_hidden);
      if (tokens < horizon) {
        const WritePlan growth = endpoint.prepare_write(*request, tokens, horizon - tokens);
        if (!growth.copies.empty()) {
          fail("KV cache self-test", "aligned capture copied a complete page");
        }
      }
      endpoint.release_execution(*request);
      endpoint.release_checkpoint(endpoint_checkpoint);
    }

    // Enough aggregate bytes in isolated page-sized holes cannot hold a local
    // snapshot crossing a page boundary. Failed capture leaves growth usable.
    PoolConfig capture_fragmented_config = batch_config;
    capture_fragmented_config.gpu_bytes = 3'840;
    CacheLedger capture_fragmented(capture_fragmented_config);
    const Allocation ring_gap = capture_fragmented.try_allocate_gpu(1'280);
    std::vector<Allocation> capture_blockers;
    for (std::size_t index = 0; index < 10; ++index) {
      capture_blockers.push_back(capture_fragmented.try_allocate_gpu(256));
    }
    capture_fragmented.release_gpu(ring_gap);
    for (std::size_t index = 0; index < 10; index += 2) {
      capture_fragmented.release_gpu(capture_blockers[index]);
    }
    const auto fragmented_capture_request = capture_fragmented.try_begin_batch(0, 512);
    (void)capture_fragmented.prepare_write(*fragmented_capture_request, 0, 257);
    const CacheStats before_fragmented_capture = capture_fragmented.stats();
    if (before_fragmented_capture.gpu.free < 257 + 16 + 256 ||
        capture_fragmented.try_begin_checkpoint_capture(*fragmented_capture_request) ||
        !same_usage(capture_fragmented.stats(), before_fragmented_capture)) {
      fail("KV cache self-test", "capture ignored fragmented snapshot allocation");
    }
    capture_fragmented.release_gpu(capture_blockers[5]);
    const auto joined_capture =
        capture_fragmented.try_begin_checkpoint_capture(*fragmented_capture_request);
    if (!joined_capture) {
      fail("KV cache self-test", "capture did not recover after fragmented holes joined");
    }
    capture_fragmented.abort_checkpoint_capture(*fragmented_capture_request);
    (void)capture_fragmented.prepare_write(*fragmented_capture_request, 257, 255);
    capture_fragmented.release_execution(*fragmented_capture_request);
    for (std::size_t index = 1; index < 10; index += 2) {
      if (index != 5) {
        capture_fragmented.release_gpu(capture_blockers[index]);
      }
    }
    if (capture_fragmented.stats().gpu.used != 0 ||
        capture_fragmented.stats().index_used != 0) {
      fail("KV cache self-test", "fragmented capture leaked resources");
    }

    CacheLedger ledger(config);
    const ExecutionId first = ledger.begin_execution();
    const WritePlan first_write = ledger.prepare_write(first, 0, 300);
    if (first_write.new_pages.size() != 2 || !first_write.copies.empty()) {
      fail("KV cache self-test", "initial append did not allocate two pages");
    }
    const std::size_t page_bytes_per_token =
        ledger.config().global_page_bytes / ledger.config().global_page_tokens;
    if (ledger.global_page_slack_bytes() !=
        (2 * ledger.config().global_page_tokens - 300) *
            page_bytes_per_token) {
      fail("KV cache self-test", "global page slack accounting is wrong");
    }
    const std::vector<std::uint8_t> local(300, 0x5a);
    const std::vector<std::uint8_t> hidden(16, 0xa5);
    const CheckpointId base =
        ledger.publish_checkpoint(first, local, hidden);

    const ExecutionId fork = ledger.begin_execution(base);
    const WritePlan fork_write = ledger.prepare_write(fork, 300, 12);
    if (fork_write.copies.size() != 1 || fork_write.new_pages.size() != 0 ||
        ledger.stats().copy_on_write_pages != 1) {
      fail("KV cache self-test", "shared partial page was not copied on write");
    }
    if (ledger.stats().gpu.used <= ledger.config().global_page_bytes * 2) {
      fail("KV cache self-test", "copy-on-write did not charge destination");
    }
    ledger.release_execution(fork);
    ledger.release_checkpoint(base);
    ledger.release_execution(first);
    if (ledger.stats().gpu.used != 0 || ledger.stats().page_count != 0 ||
        ledger.stats().checkpoint_count != 0 ||
        ledger.stats().execution_count != 0 ||
        ledger.global_page_slack_bytes() != 0) {
      fail("KV cache self-test", "release leaked physical state");
    }

    bool rejected_noncontiguous = false;
    const ExecutionId rejected = ledger.begin_execution();
    try {
      (void)ledger.prepare_write(rejected, 1, 1);
    } catch (const std::runtime_error&) {
      rejected_noncontiguous = true;
    }
    ledger.release_execution(rejected);
    if (!rejected_noncontiguous) {
      fail("KV cache self-test", "noncontiguous append was accepted");
    }

    // Active executions reserve their complete, bounded page-ID vector before
    // prefill. A tiny index budget must therefore reject the first page record
    // rather than silently letting the vector grow outside the accounting.
    PoolConfig metadata_bound = config;
    const std::size_t execution_page_slots =
        (static_cast<std::size_t>(metadata_bound.maximum_context_tokens) - 1) /
            metadata_bound.global_page_tokens +
        1;
    const std::size_t execution_metadata =
        sizeof(ExecutionInfo) + execution_page_slots * sizeof(PageId);
    metadata_bound.index_bytes =
        execution_metadata + sizeof(PageInfo) - 1;
    metadata_bound.validate();
    CacheLedger metadata_limited(metadata_bound);
    const ExecutionId metadata_execution = metadata_limited.begin_execution();
    if (metadata_limited.stats().index_used != execution_metadata ||
        metadata_limited.execution(metadata_execution).global_pages.capacity() !=
            execution_page_slots) {
      fail("KV cache self-test",
           "execution page-ID reservation was not fully accounted");
    }
    bool rejected_page_metadata = false;
    try {
      (void)metadata_limited.prepare_write(metadata_execution, 0, 1);
    } catch (const std::runtime_error&) {
      rejected_page_metadata = true;
    }
    if (!rejected_page_metadata || metadata_limited.stats().page_count != 0 ||
        metadata_limited.stats().index_used != execution_metadata) {
      fail("KV cache self-test",
           "page metadata escaped the bounded execution index reservation");
    }
    metadata_limited.release_execution(metadata_execution);
    if (metadata_limited.stats().index_used != 0) {
      fail("KV cache self-test", "execution metadata reservation leaked");
    }

    // Fixed-budget pressure evicts the oldest idle checkpoint instead of
    // failing, never a checkpoint borrowed by an execution. All sizes are
    // multiples of the pool alignment so the accounting is exact.
    PoolConfig pressured = config;
    pressured.gpu_bytes = 4'096;
    pressured.page_table_bytes = 256;
    pressured.terminal_hidden_bytes = 256;
    pressured.validate();
    CacheLedger busy(pressured);
    const std::vector<std::uint8_t> busy_local(256, 0x5a);
    const std::vector<std::uint8_t> busy_hidden(256, 0xa5);
    const ExecutionId first_busy = busy.begin_execution();
    (void)busy.prepare_write(first_busy, 0, 256);
    const CheckpointId oldest =
        busy.publish_checkpoint(first_busy, busy_local, busy_hidden);
    const ExecutionId second_busy = busy.begin_execution();
    (void)busy.prepare_write(second_busy, 0, 256);
    const CheckpointId newest =
        busy.publish_checkpoint(second_busy, busy_local, busy_hidden);
    busy.release_execution(first_busy);
    busy.release_execution(second_busy);
    const ExecutionId third_busy = busy.begin_execution();
    const WritePlan pressured_write = busy.prepare_write(third_busy, 0, 2'048);
    if (pressured_write.new_pages.size() != 8 ||
        !busy.has_checkpoint(newest) || busy.has_checkpoint(oldest)) {
      fail("KV cache self-test",
           "pressure did not evict the oldest idle checkpoint");
    }
    bool rejected_oversized = false;
    try {
      (void)busy.begin_execution(newest);
    } catch (const std::runtime_error&) {
      rejected_oversized = true;
    }
    if (!rejected_oversized || !busy.has_checkpoint(newest)) {
      fail("KV cache self-test",
           "pressure failed without sacrificing a protected source");
    }
    busy.release_execution(third_busy);

    // A cold-tier manager can preserve the contents of a pressure victim
    // before the ledger reclaims its physical GPU state. The removal callback
    // sees that preservation decision; explicit release never calls the
    // preservation hook and reports false instead.
    CacheLedger pressure_callbacks(config);
    const std::vector<std::uint8_t> callback_local(256, 0x5a);
    const std::vector<std::uint8_t> callback_hidden(16, 0xa5);
    const auto make_idle_checkpoint = [&pressure_callbacks, &callback_local,
                                       &callback_hidden]() {
      const ExecutionId execution = pressure_callbacks.begin_execution();
      (void)pressure_callbacks.prepare_write(execution, 0, 256);
      const CheckpointId checkpoint = pressure_callbacks.publish_checkpoint(
          execution, callback_local, callback_hidden);
      pressure_callbacks.release_execution(execution);
      return checkpoint;
    };
    const CheckpointId externally_preserved = make_idle_checkpoint();
    std::size_t pressure_calls = 0;
    bool pressure_saw_live_checkpoint = false;
    std::vector<std::pair<CheckpointId, bool>> removals;
    bool removal_saw_erased_checkpoint = false;
    pressure_callbacks.set_eviction_callbacks(
        {},
        [&](CheckpointId checkpoint) {
          ++pressure_calls;
          pressure_saw_live_checkpoint =
              pressure_callbacks.has_checkpoint(checkpoint);
          return checkpoint == externally_preserved;
        },
        [&](CheckpointId checkpoint, bool externally_preserved_flag) {
          removals.push_back({checkpoint, externally_preserved_flag});
          removal_saw_erased_checkpoint =
              !pressure_callbacks.has_checkpoint(checkpoint);
        });
    if (!pressure_callbacks.evict_idle_checkpoint() || pressure_calls != 1 ||
        !pressure_saw_live_checkpoint || !removal_saw_erased_checkpoint ||
        pressure_callbacks.has_checkpoint(externally_preserved) ||
        removals.size() != 1 || removals.front().first != externally_preserved ||
        !removals.front().second) {
      fail("KV cache self-test",
           "pressure preservation callback did not bracket physical removal");
    }

    const CheckpointId pressure_discarded = make_idle_checkpoint();
    if (!pressure_callbacks.evict_idle_checkpoint() || pressure_calls != 2 ||
        pressure_callbacks.has_checkpoint(pressure_discarded) ||
        removals.size() != 2 || removals.back().first != pressure_discarded ||
        removals.back().second) {
      fail("KV cache self-test",
           "unpreserved pressure victim had the wrong removal state");
    }

    const CheckpointId explicitly_released = make_idle_checkpoint();
    pressure_callbacks.release_checkpoint(explicitly_released);
    if (pressure_calls != 2 || removals.size() != 3 ||
        removals.back().first != explicitly_released || removals.back().second) {
      fail("KV cache self-test",
           "explicit checkpoint release invoked pressure preservation");
    }

    // A policy selector may intentionally protect every eligible checkpoint.
    // In that case pressure must fail instead of silently falling back to an
    // arbitrary idle victim.
    const CheckpointId policy_protected = make_idle_checkpoint();
    pressure_callbacks.set_eviction_callbacks(
        [](CheckpointId, const std::vector<CheckpointId>&, EvictionCause) {
          return CheckpointId{0};
        },
        {}, {});
    if (pressure_callbacks.evict_idle_checkpoint() ||
        !pressure_callbacks.has_checkpoint(policy_protected)) {
      fail("KV cache self-test",
           "empty eviction selection reclaimed a protected checkpoint");
    }

    // A checkpoint captured from the currently running execution may be the
    // only other owner of its partial tail page. If allocating a COW page
    // evicts that checkpoint, the original page becomes exclusive and must be
    // kept in place rather than named as a copy source after its last retained
    // reference is gone.
    PoolConfig cow_pressure = config;
    cow_pressure.gpu_bytes = 2'048;
    cow_pressure.page_table_bytes = 256;
    cow_pressure.terminal_hidden_bytes = 256;
    cow_pressure.validate();
    CacheLedger cow(cow_pressure);
    const ExecutionId cow_execution = cow.begin_execution();
    (void)cow.prepare_write(cow_execution, 0, 1);
    const PageId original_tail =
        cow.execution(cow_execution).global_pages.front();
    const CheckpointId transient = cow.publish_checkpoint(
        cow_execution, std::vector<std::uint8_t>{0x5a}, busy_hidden);
    const WritePlan exclusive_write =
        cow.prepare_write(cow_execution, 1, 1);
    if (cow.has_checkpoint(transient) || !exclusive_write.copies.empty() ||
        !exclusive_write.new_pages.empty() ||
        cow.execution(cow_execution).global_pages.front() != original_tail ||
        cow.page(original_tail).references != 1) {
      fail("KV cache self-test",
           "pressure COW did not preserve the newly exclusive tail page");
    }
    cow.release_execution(cow_execution);
    if (cow.stats().gpu.used != 0 || cow.stats().page_count != 0) {
      fail("KV cache self-test", "pressure COW cleanup leaked state");
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
}  // namespace gewell::kv_cache

int main() {
  std::string failure;
  if (!gewell::kv_cache::run_self_tests(&failure)) {
    std::cerr << failure << "\n";
    return 1;
  }
  std::cout << "kv_cache tests: ok\n";
  return 0;
}
