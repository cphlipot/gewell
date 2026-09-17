// This contract suite exercises private fixture and runtime helpers together.
// Include their implementations only here; neither application includes tests.
#include "../src/app/runner.cu"
#include "../src/diagnostics/pair_fixtures.cu"

namespace gewell::app {
namespace {
using namespace gewell::diagnostics;
using namespace gewell::runtime;
std::size_t generation_logits_output_bytes(std::uint32_t new_tokens) {
  if (new_tokens >
      std::numeric_limits<std::size_t>::max() / kGenerationLogitRowBytes) {
    fail("generation logits output", "requested byte count overflows size_t");
  }
  return static_cast<std::size_t>(new_tokens) * kGenerationLogitRowBytes;
}

constexpr std::size_t compact_global_row_offset(
    std::uint32_t head, std::uint32_t position, std::uint32_t capacity) {
  return (static_cast<std::size_t>(head) * capacity + position) *
         primitives::kGlobalCompactKvSize;
}
}  // namespace

bool run_self_tests(std::string* failure) {
  try {
    {
      BatchRequest request;
      request.id = "operation-record";
      request.prompt = std::make_shared<const std::vector<std::uint32_t>>(1, 2);
      request.cursor = 1;
      const std::array<Operation, 4> operations{{Operation::generate, Operation::prefill,
                                                Operation::finish, Operation::stats}};
      for (std::size_t i = 0; i < operations.size(); ++i) {
        request.operation = operations[i];
        std::ostringstream records;
        write_batch_result(records, request, 0);
        if (nlohmann::json::parse(records.str()).at("operation") != i + 1)
          fail("batch operation record", "internal enum escaped into output schema");
      }
    }

    if (mtp_accepted_median({}).has_value() ||
        mtp_accepted_median({0, 0, 0}).has_value() ||
        mtp_accepted_median({3, 0, 0}) != std::optional<double>{0.0} ||
        mtp_accepted_median({1, 0, 3, 1}) != std::optional<double>{2.0} ||
        mtp_accepted_median({1, 1, 1, 1}) != std::optional<double>{1.5} ||
        mtp_accepted_median({2, 0, 0, 2}) != std::optional<double>{1.5} ||
        mtp_accepted_median({1, 4, 1}) != std::optional<double>{1.0}) {
      fail("MTP acceptance median", "empty, zero, odd, or even round median is incorrect");
    }
    {
      const prefix_index::CheckpointSources sources =
          static_cast<prefix_index::CheckpointSources>(
              prefix_index::checkpoint_source_mask(
                  prefix_index::CheckpointSource::input_endpoint) |
              prefix_index::checkpoint_source_mask(
                  prefix_index::CheckpointSource::periodic));
      const std::size_t bucket_index = lifecycle_bucket_index(
          sources, prefix_index::RetentionPriority::high,
          prefix_index::CacheClass::reused);
      CacheTelemetry before;
      CacheTelemetry after;
      before.checkpoint_buckets[bucket_index] = {2, 3, 5, 7, 11, 13};
      after = before;
      after.checkpoint_buckets[bucket_index] = {5, 7, 10, 13, 18, 23};
      before.gpu_reclaimed_bytes = 29;
      before.cpu_reclaimed_bytes = 31;
      before.cold_spill_bytes = 37;
      before.cold_restore_bytes = 41;
      before.cold_spill_count = 43;
      before.cold_restore_count = 47;
      before.cold_spill_avoided_rewrite_bytes = 53;
      before.cold_spill_wall_milliseconds = 59.0;
      before.cold_restore_wall_milliseconds = 61.0;
      after.gpu_reclaimed_bytes = 67;
      after.cpu_reclaimed_bytes = 71;
      after.cold_spill_bytes = 73;
      after.cold_restore_bytes = 79;
      after.cold_spill_count = 83;
      after.cold_restore_count = 89;
      after.cold_spill_avoided_rewrite_bytes = 97;
      after.cold_spill_wall_milliseconds = 101.0;
      after.cold_restore_wall_milliseconds = 103.0;
      const CacheTelemetry delta = telemetry_delta(after, before);
      const CheckpointLifecycleBucket& bucket =
          delta.checkpoint_buckets[bucket_index];
      if (bucket.admissions != 3 || bucket.hits != 4 ||
          bucket.removals != 5 || bucket.admitted_bytes != 6 ||
          bucket.reclaimed_gpu_bytes != 7 ||
          bucket.reclaimed_cpu_bytes != 10 ||
          delta.gpu_reclaimed_bytes != 38 ||
          delta.cpu_reclaimed_bytes != 40 || delta.cold_spill_bytes != 36 ||
          delta.cold_restore_bytes != 38 || delta.cold_spill_count != 40 ||
          delta.cold_restore_count != 42 ||
          delta.cold_spill_avoided_rewrite_bytes != 44 ||
          delta.cold_spill_wall_milliseconds != 42.0 ||
          delta.cold_restore_wall_milliseconds != 42.0) {
        fail("server cache telemetry", "fixed bucket delta is incorrect");
      }
      std::ostringstream encoded;
      write_checkpoint_buckets(encoded, delta);
      const std::string expected =
          "[{\"sources\":\"input_endpoint|periodic\",\"priority\":\"high\","
          "\"cache_class\":\"reused\",\"admissions\":3,\"hits\":4,"
          "\"removals\":5,\"admitted_bytes\":6,\"reclaimed_gpu_bytes\":7,"
          "\"reclaimed_cpu_bytes\":10}]";
      if (encoded.str() != expected) {
        fail("server cache telemetry", "fixed bucket encoding is incorrect");
      }
    }
    if (!weight_ids_are_valid()) {
      fail("weight map", "compile-time role/layer map is invalid");
    }
    if (ScratchLayout::kBytes != 1'721'856) {
      fail("scratch layout", "byte count changed");
    }
    if (prefill::kTensorAttentionScratchBytes != 285'474'816 ||
        prefill::kTensorAttentionStagedBytes != 8'388'608 ||
        prefill::kTensorAttentionScoreBytes != 134'217'728 ||
        prefill::kTensorAttentionProbabilityBytes != 67'108'864 ||
        prefill::kTensorAttentionNumeratorBytes != 67'108'864 ||
        prefill::kTensorAttentionStateBytes != 131'072) {
      fail("runtime tensor prefill attention scratch",
           "byte contract changed");
    }
    for (const auto cap : {1U, 256U, 512U, 1024U, 2048U, 4096U}) {
      const RuntimeGenerationScratchLayout layout(std::max(kMultimodalChunkTokens, cap));
      if (generation_prefill_chunk_tokens(8193, cap) != cap ||
          generation_prefill_chunk_tokens(17, cap) != std::min(17U, cap) ||
          layout.kRows < cap || layout.kRows < kMaxBatchRows ||
          layout.kArgmax + sizeof(std::uint32_t) > layout.kBytes)
        fail("runtime chunk configuration", "chunk or workspace bounds are incorrect");
    }
    for (const auto cap : {0U, kMaxPrefillChunkTokens + 1}) {
      bool rejected = false;
      try { checked_prefill_chunk_tokens(cap); }
      catch (const std::runtime_error&) { rejected = true; }
      if (!rejected) fail("runtime chunk configuration", "invalid cap was accepted");
    }
    if (CachedScratchLayout::kBytes != 1'388'288 ||
        2 * kLocalCacheBytesPerKind != 838'860'800 ||
        2 * kGlobalCacheBytesPerKind != 163'840) {
      fail("cached-pair memory contract", "byte count changed");
    }
    std::array<bool, model::kLocalLayerCount> local_slots{};
    std::array<bool, model::kGlobalLayerCount> global_slots{};
    for (std::uint32_t layer = 0; layer < model::kLayerCount; ++layer) {
      if (model::is_global_layer(layer)) {
        const std::size_t slot = layer / 6;
        if (slot >= global_slots.size() || global_slots[slot]) {
          fail("cached-pair layer map", "invalid global cache slot");
        }
        global_slots[slot] = true;
      } else {
        const std::size_t slot = layer - layer / 6;
        if (slot >= local_slots.size() || local_slots[slot]) {
          fail("cached-pair layer map", "invalid local cache slot");
        }
        local_slots[slot] = true;
      }
    }
    if (!std::all_of(local_slots.begin(), local_slots.end(),
                     [](bool used) { return used; }) ||
        !std::all_of(global_slots.begin(), global_slots.end(),
                     [](bool used) { return used; })) {
      fail("cached-pair layer map", "cache slots are not exhaustive");
    }
    const std::vector<CaptureSpec> specs = make_capture_specs();
    if (specs.size() != kCaptureCount) {
      fail("capture inventory", "expected 114 tensors");
    }
    for (std::size_t index = 0; index < specs.size(); ++index) {
      if (specs[index].elements == 0 ||
          (index != 0 && specs[index - 1].name >= specs[index].name)) {
        fail("capture inventory", "names are duplicated or not canonical");
      }
    }
    if (ShortDecodeScratchLayout::kBytes != 1'457'408 ||
        2 * kLocalCacheBytesPerKind != 838'860'800 ||
        2 * kShortDecodeGlobalCacheBytesPerKind != 1'966'080 ||
        kShortDecodeGlobalCacheCapacity != 24 ||
        kShortDecodeFedTokens != 23) {
      fail("short-decode memory contract", "fixed byte count changed");
    }
    if (kShortDecodePrompt.size() != 16 ||
        kShortDecodeM1Expected.size() != 8 ||
        kShortDecodePrompt.front() != 2 ||
        kShortDecodePrompt.back() != 506 ||
        kShortDecodeM1Expected.front() != 1638 ||
        kShortDecodeM1Expected.back() != 1852) {
      fail("short-decode token contract", "pinned token sequence changed");
    }
    const std::vector<CaptureSpec> short_specs =
        make_short_decode_capture_specs();
    if (short_specs.size() != kShortDecodeCaptureCount) {
      fail("short-decode capture inventory", "expected 572 tensors");
    }
    std::size_t prefill_count = 0;
    std::array<std::size_t, kShortDecodeDecisionCount> step_counts{};
    for (std::size_t index = 0; index < short_specs.size(); ++index) {
      const CaptureSpec& spec = short_specs[index];
      if (spec.elements == 0 ||
          (index != 0 && short_specs[index - 1].name >= spec.name) ||
          spec.name.find(".full.") != std::string::npos) {
        fail("short-decode capture inventory",
             "name is duplicate, empty, or oracle-only");
      }
      if (spec.name.rfind("prefill.", 0) == 0) {
        ++prefill_count;
      }
      for (std::uint32_t step = 0; step < kShortDecodeDecisionCount; ++step) {
        char prefix[32];
        const int length = std::snprintf(prefix, sizeof(prefix),
                                         "generation.step.%02u.", step);
        if (length < 0 || static_cast<std::size_t>(length) >= sizeof(prefix)) {
          fail("short-decode capture inventory", "step prefix is too long");
        }
        if (spec.name.rfind(
                std::string_view(prefix, static_cast<std::size_t>(length)),
                0) == 0) {
          ++step_counts[step];
        }
      }
    }
    if (prefill_count != 68 ||
        !std::all_of(step_counts.begin(), step_counts.end(),
                     [](std::size_t count) { return count == 63; })) {
      fail("short-decode capture inventory",
           "expected 68 prefill and 63 tensors per decision");
    }
    const auto require_short_capture = [&](std::string_view name,
                                           std::size_t elements) {
      const auto found = std::find_if(
          short_specs.begin(), short_specs.end(),
          [&](const CaptureSpec& spec) { return spec.name == name; });
      if (found == short_specs.end() || found->elements != elements) {
        fail("short-decode capture inventory",
             std::string(name) + " has the wrong shape");
      }
    };
    require_short_capture("prefill.embedding",
                          kShortDecodePromptTokens * model::kHiddenSize);
    require_short_capture("generation.step.00.cached.layer.00.output",
                          model::kHiddenSize);
    require_short_capture("generation.step.07.cached.layer.59.output",
                          model::kHiddenSize);
    require_short_capture("generation.step.07.logits.cached.post_softcap",
                          model::kVocabSize);
    constexpr auto encoded_tokens =
        encode_short_decode_tokens(kShortDecodeM1Expected);
    static_assert(encoded_tokens.size() == 32);
    for (std::size_t index = 0; index < kShortDecodeM1Expected.size(); ++index) {
      const std::uint32_t decoded =
          static_cast<std::uint32_t>(encoded_tokens[4 * index]) |
          (static_cast<std::uint32_t>(encoded_tokens[4 * index + 1]) << 8) |
          (static_cast<std::uint32_t>(encoded_tokens[4 * index + 2]) << 16) |
          (static_cast<std::uint32_t>(encoded_tokens[4 * index + 3]) << 24);
      if (decoded != kShortDecodeM1Expected[index]) {
        fail("short-decode token encoding", "little-endian round trip failed");
      }
    }
    if (BoundaryScratchLayout::kBytes != 1'516'800 ||
        primitives::kCachedAttentionM1BoundaryScoreScratchBytes != 65'664 ||
        kLocalCacheCapacity != 1'024 ||
        kBoundaryGlobalCacheCapacity != 1'026 ||
        2 * kLocalCacheBytesPerKind != 838'860'800 ||
        2 * kBoundaryGlobalCacheBytesPerKind != 84'049'920) {
      fail("local-boundary memory contract", "fixed byte count changed");
    }
    if (kBoundaryPrefixTokens != 1'026 ||
        kBoundaryCaptureFirstPosition != 1'022 ||
        kBoundaryCaptureRows != 4 || kBoundaryDecisionPosition != 1'025 ||
        kBoundaryExpectedToken != 121'160 ||
        kBoundaryInputTokenIdsSha256.size() != 64 ||
        boundary_fixture_fingerprint() != 0xa266c2be06f28bbcULL) {
      fail("local-boundary fixture contract", "pinned fixture changed");
    }
    for (std::uint32_t position = 0; position < kBoundaryPrefixTokens;
         ++position) {
      if (boundary_input_token(position) >= model::kVocabSize ||
          (position > 90 &&
           boundary_input_token(position) !=
               boundary_input_token(position - 90))) {
        fail("local-boundary fixture contract",
             "90-token repetition or vocabulary bound failed");
      }
    }

    const std::vector<CaptureSpec> boundary_specs =
        make_boundary_capture_specs();
    if (boundary_specs.size() != kBoundaryCaptureCount) {
      fail("local-boundary capture inventory", "expected 90 tensors");
    }
    std::size_t boundary_elements = 0;
    std::size_t boundary_row_tensors = 0;
    for (std::size_t index = 0; index < boundary_specs.size(); ++index) {
      const CaptureSpec& spec = boundary_specs[index];
      boundary_elements += spec.elements;
      if (spec.elements == 0 ||
          spec.name.rfind("boundary.cached.", 0) != 0 ||
          spec.name.find("boundary.hybrid.") != std::string::npos ||
          spec.name.find("boundary.full.") != std::string::npos ||
          (index != 0 && boundary_specs[index - 1].name >= spec.name)) {
        fail("local-boundary capture inventory",
             "name is duplicate, empty, or not native cached evidence");
      }
      if (spec.name.rfind("boundary.cached.rows_1022_1025.", 0) == 0) {
        ++boundary_row_tensors;
      }
    }
    if (boundary_elements != 2'685'440 ||
        boundary_elements * sizeof(BFloat16) != 5'370'880 ||
        boundary_row_tensors != 88) {
      fail("local-boundary capture inventory",
           "element, byte, or retained-row count changed");
    }
    const auto require_boundary_capture = [&](std::string_view name,
                                              std::size_t elements) {
      const auto found = std::find_if(
          boundary_specs.begin(), boundary_specs.end(),
          [&](const CaptureSpec& spec) { return spec.name == name; });
      if (found == boundary_specs.end() || found->elements != elements) {
        fail("local-boundary capture inventory",
             std::string(name) + " has the wrong shape");
      }
    };
    require_boundary_capture(
        "boundary.cached.rows_1022_1025.embedding",
        kBoundaryCaptureRows * model::kHiddenSize);
    require_boundary_capture(
        "boundary.cached.rows_1022_1025.rotary.sliding_attention.cos",
        kBoundaryCaptureRows * model::kLocalHeadSize);
    require_boundary_capture(
        "boundary.cached.rows_1022_1025.rotary.full_attention.sin",
        kBoundaryCaptureRows * model::kGlobalHeadSize);
    for (std::uint32_t layer = 0; layer < model::kLayerCount; ++layer) {
      require_boundary_capture(boundary_layer_capture_name(layer, "output"),
                               kBoundaryCaptureRows * model::kHiddenSize);
    }
    require_boundary_capture(
        "boundary.cached.rows_1022_1025.layer.00.q_rope",
        kBoundaryCaptureRows * model::kQueryHeadCount *
            model::kLocalHeadSize);
    require_boundary_capture(
        "boundary.cached.rows_1022_1025.layer.00.k_raw",
        kBoundaryCaptureRows * model::kLocalKvHeadCount *
            model::kLocalHeadSize);
    require_boundary_capture(
        "boundary.cached.rows_1022_1025.layer.00.attention_probabilities",
        kBoundaryCaptureRows * model::kQueryHeadCount *
            kBoundaryPrefixTokens);
    require_boundary_capture(
        "boundary.cached.rows_1022_1025.layer.05.q_rope",
        kBoundaryCaptureRows * model::kQueryHeadCount *
            model::kGlobalHeadSize);
    require_boundary_capture(
        "boundary.cached.rows_1022_1025.layer.05.v_norm",
        kBoundaryCaptureRows * model::kGlobalKvHeadCount *
            model::kGlobalHeadSize);
    require_boundary_capture(
        "boundary.cached.rows_1022_1025.layer.05.attention_probabilities",
        kBoundaryCaptureRows * model::kQueryHeadCount *
            kBoundaryPrefixTokens);
    require_boundary_capture(
        "boundary.cached.position_1025.logits.pre_softcap",
        model::kVocabSize);
    require_boundary_capture(
        "boundary.cached.position_1025.logits.post_softcap",
        model::kVocabSize);

    if (HybridBoundaryScratchLayout::kBytes != 479'789'568 ||
        HybridBoundaryScratchLayout::kGatherElements != 32'832 ||
        HybridBoundaryScratchLayout::kArgmax -
                HybridBoundaryScratchLayout::kGatherStaging !=
            65'792 ||
        prefill::kAttentionScoreScratchBytes != 67'108'864 ||
        prefill::kAttentionProbabilityBytes != 67'108'864 ||
        2 * kLocalCacheBytesPerKind != 838'860'800 ||
        2 * kBoundaryGlobalCacheBytesPerKind != 84'049'920 ||
        HybridBoundaryScratchLayout::kBytes +
                2 * kLocalCacheBytesPerKind +
                2 * kBoundaryGlobalCacheBytesPerKind !=
            1'402'700'288 ||
        artifact::kPayloadBytes + HybridBoundaryScratchLayout::kBytes +
                2 * kLocalCacheBytesPerKind +
                2 * kBoundaryGlobalCacheBytesPerKind !=
            62'798'426'624ULL) {
      fail("hybrid local-boundary memory contract",
           "fixed scratch, gather, or cache byte count changed");
    }
    if (kHybridBoundaryScheduleRows !=
            std::array<std::uint32_t, 3>{{1'024, 1, 1}} ||
        kHybridBoundaryDecisionPositions !=
            std::array<std::uint32_t, 3>{{1'023, 1'024, 1'025}} ||
        kHybridBoundaryExpected !=
            std::array<std::uint32_t, 3>{{236'764, 532, 121'160}} ||
        kHybridBoundaryExpected[0] != boundary_input_token(1'024) ||
        kHybridBoundaryExpected[1] != boundary_input_token(1'025)) {
      fail("hybrid local-boundary fixture contract",
           "fixed prefill/decode schedule or predictions changed");
    }

    constexpr std::size_t kGlobalHeadRowBytes =
        model::kGlobalHeadSize * sizeof(BFloat16);
    constexpr std::size_t kGlobalHeadSourcePitch =
        static_cast<std::size_t>(kHybridPrefillTokens) *
        kGlobalHeadRowBytes;
    constexpr std::size_t kLocalHeadRowBytes =
        model::kLocalHeadSize * sizeof(BFloat16);
    constexpr std::size_t kLocalHeadSourcePitch =
        static_cast<std::size_t>(kHybridPrefillTokens) * kLocalHeadRowBytes;
    constexpr std::size_t kProbabilitySourcePitch =
        static_cast<std::size_t>(kHybridPrefillTokens) *
        kHybridPrefillTokens * sizeof(BFloat16);
    constexpr std::size_t kProbabilityDestinationPitch =
        static_cast<std::size_t>(kBoundaryPrefixTokens) * sizeof(BFloat16);
    constexpr std::size_t kLastGlobalHeadSource =
        ((static_cast<std::size_t>(model::kQueryHeadCount - 1) *
              kHybridPrefillTokens +
          (kHybridPrefillTokens - 1)) *
             model::kGlobalHeadSize) +
        (model::kGlobalHeadSize - 1);
    constexpr std::size_t kLastProbabilitySource =
        ((static_cast<std::size_t>(model::kQueryHeadCount - 1) *
              kHybridPrefillTokens +
          (kHybridPrefillTokens - 1)) *
             kHybridPrefillTokens) +
        (kHybridPrefillTokens - 1);
    if (kGlobalHeadRowBytes != 1'024 ||
        kGlobalHeadSourcePitch != 1'048'576 ||
        kLocalHeadRowBytes != 512 ||
        kLocalHeadSourcePitch != 524'288 ||
        kProbabilitySourcePitch != 2'097'152 ||
        kProbabilityDestinationPitch != 2'052 ||
        kLastGlobalHeadSource + 1 !=
            HybridBoundaryScratchLayout::kPrefillQueryElements ||
        kLastProbabilitySource + 1 != prefill::kAttentionMatrixElements ||
        HybridBoundaryScratchLayout::kGatherElements !=
            static_cast<std::size_t>(model::kQueryHeadCount) *
                kBoundaryPrefixTokens) {
      fail("hybrid local-boundary gather contract",
           "head-major or absolute-probability row layout changed");
    }

    const std::vector<CaptureSpec> hybrid_specs =
        make_hybrid_boundary_capture_specs();
    if (hybrid_specs.size() != kBoundaryCaptureCount ||
        hybrid_specs.size() != boundary_specs.size()) {
      fail("hybrid local-boundary capture inventory", "expected 90 tensors");
    }
    std::size_t hybrid_elements = 0;
    std::size_t hybrid_row_tensors = 0;
    constexpr std::string_view kCachedPrefix = "boundary.cached";
    constexpr std::string_view kHybridPrefix = "boundary.hybrid";
    for (std::size_t index = 0; index < hybrid_specs.size(); ++index) {
      const CaptureSpec& spec = hybrid_specs[index];
      const CaptureSpec& cached = boundary_specs[index];
      hybrid_elements += spec.elements;
      if (spec.name.rfind("boundary.hybrid.rows_1022_1025.", 0) == 0) {
        ++hybrid_row_tensors;
      }
      if (spec.elements == 0 ||
          spec.name.rfind("boundary.hybrid.", 0) != 0 ||
          spec.name.find("boundary.cached.") != std::string::npos ||
          spec.name.find("boundary.full.") != std::string::npos ||
          (index != 0 && hybrid_specs[index - 1].name >= spec.name) ||
          cached.name.rfind(kCachedPrefix, 0) != 0 ||
          spec.name.rfind(kHybridPrefix, 0) != 0 ||
          cached.name.substr(kCachedPrefix.size()) !=
              spec.name.substr(kHybridPrefix.size()) ||
          cached.elements != spec.elements) {
        fail("hybrid local-boundary capture inventory",
             "name, ordering, suffix, or shape differs from the oracle");
      }
    }
    if (hybrid_elements != 2'685'440 ||
        hybrid_elements * sizeof(BFloat16) != 5'370'880 ||
        hybrid_row_tensors != 88) {
      fail("hybrid local-boundary capture inventory",
           "element, byte, or retained-row count changed");
    }
    const auto require_hybrid_capture = [&](std::string_view name,
                                            std::size_t elements) {
      const auto found = std::find_if(
          hybrid_specs.begin(), hybrid_specs.end(),
          [&](const CaptureSpec& spec) { return spec.name == name; });
      if (found == hybrid_specs.end() || found->elements != elements) {
        fail("hybrid local-boundary capture inventory",
             std::string(name) + " has the wrong shape");
      }
    };
    require_hybrid_capture(
        "boundary.hybrid.rows_1022_1025.embedding",
        kBoundaryCaptureRows * model::kHiddenSize);
    require_hybrid_capture(
        "boundary.hybrid.rows_1022_1025.layer.00.q_rope",
        kBoundaryCaptureRows * model::kQueryHeadCount *
            model::kLocalHeadSize);
    require_hybrid_capture(
        "boundary.hybrid.rows_1022_1025.layer.00.attention_probabilities",
        kBoundaryCaptureRows * model::kQueryHeadCount *
            kBoundaryPrefixTokens);
    require_hybrid_capture(
        "boundary.hybrid.rows_1022_1025.layer.05.k_rope",
        kBoundaryCaptureRows * model::kGlobalKvHeadCount *
            model::kGlobalHeadSize);
    require_hybrid_capture(
        "boundary.hybrid.rows_1022_1025.layer.05.attention_probabilities",
        kBoundaryCaptureRows * model::kQueryHeadCount *
            kBoundaryPrefixTokens);
    require_hybrid_capture(
        "boundary.hybrid.rows_1022_1025.layer.59.output",
        kBoundaryCaptureRows * model::kHiddenSize);
    require_hybrid_capture(
        "boundary.hybrid.position_1025.logits.post_softcap",
        model::kVocabSize);

    constexpr std::array<std::uint8_t, 4> kHybridFinalTokenBytes{{
        static_cast<std::uint8_t>(kHybridBoundaryExpected[2]),
        static_cast<std::uint8_t>(kHybridBoundaryExpected[2] >> 8),
        static_cast<std::uint8_t>(kHybridBoundaryExpected[2] >> 16),
        static_cast<std::uint8_t>(kHybridBoundaryExpected[2] >> 24),
    }};
    const std::uint32_t decoded_hybrid_token =
        static_cast<std::uint32_t>(kHybridFinalTokenBytes[0]) |
        (static_cast<std::uint32_t>(kHybridFinalTokenBytes[1]) << 8) |
        (static_cast<std::uint32_t>(kHybridFinalTokenBytes[2]) << 16) |
        (static_cast<std::uint32_t>(kHybridFinalTokenBytes[3]) << 24);
    if (decoded_hybrid_token != kHybridBoundaryExpected[2]) {
      fail("hybrid local-boundary token encoding",
           "little-endian final-token round trip failed");
    }

    constexpr std::uint32_t kFirstCommitOutputIndex =
        primitives::kGraphDecodeFirstPosition -
        (primitives::kGraphDecodeFirstPosition - 1);
    constexpr std::uint32_t kLastCommitOutputIndex =
        primitives::kGraphDecodeLastPosition -
        (primitives::kGraphDecodeFirstPosition - 1);
    if (kGraphPromptTokens != 1'024 || kGraphOutputTokens != 512 ||
        kGraphDecodeReplays != 511 ||
        primitives::kGraphDecodeFirstPosition != 1'024 ||
        primitives::kGraphDecodeLastPosition != 1'534 ||
        primitives::kGraphDecodeFinalPosition != 1'535 ||
        kFirstCommitOutputIndex != 1 || kLastCommitOutputIndex != 511 ||
        kGraphExpectedFirstThree !=
            std::array<std::uint32_t, 3>{{236'764, 532, 121'160}}) {
      fail("graph-decode schedule contract",
           "prompt, replay, commit index, or first-token contract changed");
    }
    if (kProfileDecodeReplays != 1 ||
        primitives::kGraphDecodeFirstPosition != 1'024 ||
        kProfileDecodeFinalPosition != 1'025 ||
        kProfileDecodeExpectedOutputs !=
            std::array<std::uint32_t, 2>{{236'764, 532}} ||
        kProfileDecodeExpectedOutputs[0] >= model::kVocabSize ||
        kProfileDecodeExpectedOutputs[1] >= model::kVocabSize) {
      fail("profile-decode schedule contract",
           "single-replay position, output, or vocabulary contract changed");
    }
    if (kGraphGlobalCacheCapacity != 1'536 ||
        2 * kLocalCacheBytesPerKind != 838'860'800 ||
        2 * kGraphGlobalCacheBytesPerKind != 125'829'120 ||
        2 * kLocalCacheBytesPerKind +
                2 * kGraphGlobalCacheBytesPerKind !=
            964'689'920 ||
        primitives::kGraphAttentionScoreScratchBytes != 98'304 ||
        primitives::kGraphAttentionFusedLocalSplitCount != 32 ||
        primitives::kGraphAttentionFusedGlobalSplitCount != 48 ||
        primitives::kGraphAttentionFusedScratchBytes != 3'158'016 ||
        kGraphExpectedNodeCount != 1'078 ||
        kGraphPersistentStateBytes != 2'056 || kGraphOutputBytes != 2'048 ||
        HybridBoundaryScratchLayout::kBytes +
                2 * kLocalCacheBytesPerKind +
                2 * kGraphGlobalCacheBytesPerKind +
                kGraphPersistentStateBytes !=
            1'444'481'544 ||
        artifact::kPayloadBytes + HybridBoundaryScratchLayout::kBytes +
                2 * kLocalCacheBytesPerKind +
                2 * kGraphGlobalCacheBytesPerKind +
                kGraphPersistentStateBytes !=
            62'840'207'880ULL) {
      fail("graph-decode memory contract",
           "fixed scratch, cache, state, or output byte count changed");
    }
    if (HybridBoundaryScratchLayout::kScoreScratch +
                primitives::kGraphAttentionFusedScratchBytes >
            HybridBoundaryScratchLayout::kProbabilities ||
        HybridBoundaryScratchLayout::kProbabilities +
                primitives::kGraphAttentionScoreScratchBytes >
            HybridBoundaryScratchLayout::kGatherStaging ||
        HybridBoundaryScratchLayout::kArgmax + sizeof(std::uint32_t) >
            HybridBoundaryScratchLayout::kBytes) {
      fail("graph-decode alias contract",
           "fused attention, prefill probabilities, or argmax overlap "
           "illegally");
    }
    if (kGraphWarmupRequests != 2 || kGraphMeasuredRequests != 7 ||
        kGraphPersistentTimingEventCount != 4 ||
        kGraphRequestCudaObjectCreations != 0) {
      fail("graph-decode benchmark contract",
           "warmup, sample, or fixed timing-resource count changed");
    }
    if (generation_cache_capacity(1, 1) != 1 ||
        generation_cache_capacity(1'024, 512) != 1'535 ||
        generation_cache_capacity(primitives::kMaxContextTokenCount, 1) !=
            primitives::kMaxContextTokenCount ||
        kGenerationLogitRowBytes != 524'288 ||
        generation_logits_output_bytes(1) != 524'288 ||
        generation_logits_output_bytes(8) != 4'194'304 ||
        generation_logits_output_bytes(512) != 268'435'456 ||
        generation_logits_output_bytes(
            primitives::kMaxContextTokenCount) != 137'438'953'472ULL ||
        primitives::kGlobalCompactKeySize != 128 ||
        primitives::kGlobalCompactKvSize != 640 ||
        runtime_global_compact_cache_bytes(
            primitives::kMaxContextTokenCount) != 13'421'772'800ULL ||
        runtime_global_compact_cache_bytes(
            primitives::kMaxContextTokenCount) +
                2 * kLocalCacheBytesPerKind !=
            14'260'633'600ULL) {
      fail("generation capacity contract",
           "context horizon, logits size, or BF16 KV byte count changed");
    }
    const std::vector<std::uint32_t> valid_image_prompt{
        model::kBeginImageTokenId, model::kImageTokenId,
        model::kImageTokenId, model::kEndImageTokenId, 2};
    const ImagePromptSpan valid_image_span =
        find_image_prompt_span(valid_image_prompt);
    const auto rejects_image_prompt = [](std::vector<std::uint32_t> value) {
      try {
        (void)find_image_prompt_span(value);
        return false;
      } catch (const std::runtime_error&) {
        return true;
      }
    };
    std::vector<std::uint32_t> maximum_image_prompt{
        model::kBeginImageTokenId};
    maximum_image_prompt.insert(maximum_image_prompt.end(),
                                model::kVisionMaxSoftTokenCount,
                                model::kImageTokenId);
    maximum_image_prompt.push_back(model::kEndImageTokenId);
    maximum_image_prompt.push_back(2);
    const ImagePromptSpan maximum_image_span =
        find_image_prompt_span(maximum_image_prompt);
    std::vector<std::uint32_t> over_capacity_image_prompt{
        model::kBeginImageTokenId};
    over_capacity_image_prompt.insert(
        over_capacity_image_prompt.end(),
        model::kVisionMaxSoftTokenCount + 1, model::kImageTokenId);
    over_capacity_image_prompt.push_back(model::kEndImageTokenId);
    over_capacity_image_prompt.push_back(2);
    std::vector<std::uint32_t> over_length_image_prompt =
        maximum_image_prompt;
    over_length_image_prompt.resize(kMultimodalChunkTokens + 1, 2);
    if (valid_image_span.begin != 1 || valid_image_span.end != 3 ||
        maximum_image_span.begin != 1 ||
        maximum_image_span.end !=
            1 + model::kVisionMaxSoftTokenCount ||
        generation_prefill_chunk_tokens(kMultimodalChunkTokens) !=
            kRuntimeChunkTokens ||
        !rejects_image_prompt(over_capacity_image_prompt) ||
        rejects_image_prompt(over_length_image_prompt) ||
        !rejects_image_prompt({2, 3}) ||
        !rejects_image_prompt({model::kBeginImageTokenId,
                               model::kImageTokenId, 2,
                               model::kImageTokenId,
                               model::kEndImageTokenId}) ||
        !rejects_image_prompt({2, model::kImageTokenId,
                               model::kEndImageTokenId}) ||
        !rejects_image_prompt({model::kBeginImageTokenId,
                               model::kImageTokenId, 2})) {
      fail("caption prompt contract",
           "image placeholder span extraction or rejection changed");
    }
    for (const auto cap : {1U, 4U, 256U, 1024U, 4096U}) {
      const std::array<ImagePromptSpan, 3> images{{{17, 1137}, {2200, 2203}, {2203, 2205}}};
      std::uint32_t base = 0;
      std::size_t next_image = 0;
      while (base < 2400) {
        const auto image = next_image < images.size() ? images[next_image] : ImagePromptSpan{};
        const auto rows = multimodal_prefill_chunk_rows(base, 2400, cap, image.begin, image.end);
        if (!rows || base + rows > 2400) fail("multimodal chunks", "invalid progress");
        if (next_image < images.size() && base == image.begin) {
          if (base + rows != image.end) fail("multimodal chunks", "split complete image");
          ++next_image;
        } else if (rows > cap || (next_image < images.size() && base + rows > image.begin)) {
          fail("multimodal chunks", "text chunk crossed an image boundary or cap");
        }
        base += rows;
      }
      if (next_image != images.size()) fail("multimodal chunks", "lost an image");
    }
    for (const auto image : {ImagePromptSpan{10, 20}, ImagePromptSpan{30, 41}, ImagePromptSpan{31, 30}}) {
      bool rejected = false;
      try { (void)multimodal_prefill_chunk_rows(15, 40, 4, image.begin, image.end); }
      catch (const std::runtime_error&) { rejected = true; }
      if (!rejected) fail("multimodal chunks", "invalid next image accepted");
    }
    if (!is_generation_stop_token(1) || !is_generation_stop_token(50) ||
        !is_generation_stop_token(106) || is_generation_stop_token(2))
      fail("stop token contract", "generation stop set changed");
    const CaptureTailCowHeadroom new_tail_cow =
        capture_tail_cow_headroom(true, true, 7, false, 0);
    const CaptureTailCowHeadroom pending_tail_cow =
        capture_tail_cow_headroom(true, true, 7, false, 7);
    const CaptureTailCowHeadroom source_tail_cow =
        capture_tail_cow_headroom(true, true, 7, true, 0);
    const CaptureTailCowHeadroom terminal_tail_cow =
        capture_tail_cow_headroom(false, true, 7, false, 7);
    if (!new_tail_cow.reserves_new_tail || new_tail_cow.page_count() != 1 ||
        !pending_tail_cow.keeps_pending_tail ||
        pending_tail_cow.reserves_new_tail ||
        pending_tail_cow.page_count() != 1 ||
        source_tail_cow.page_count() != 0 ||
        terminal_tail_cow.page_count() != 0) {
      fail("cache lifecycle contract",
           "partial-checkpoint COW headroom changed");
    }
    const std::vector<CheckpointTrigger> bounded_triggers =
        runtime::make_checkpoint_triggers(8, 10, 0, 4, 2, {4, 6}, 5);
    bool rejected_trigger_capacity = false;
    try {
      (void)runtime::make_checkpoint_triggers(8, 10, 0, 4, 2, {4, 6}, 4);
    } catch (const runtime::CacheCapacityError& error) {
      rejected_trigger_capacity = error.request_invalid;
    }
    if (bounded_triggers.size() != 5 ||
        bounded_triggers[0].processed_tokens != 2 ||
        bounded_triggers[0].source != CheckpointSource::periodic ||
        bounded_triggers[1].processed_tokens != 4 ||
        bounded_triggers[1].source != CheckpointSource::learned_branch ||
        bounded_triggers[2].processed_tokens != 6 ||
        bounded_triggers[2].source != CheckpointSource::learned_branch ||
        bounded_triggers[3].processed_tokens != 8 ||
        bounded_triggers[3].source != CheckpointSource::input ||
        bounded_triggers[4].processed_tokens != 10 ||
        bounded_triggers[4].source != CheckpointSource::periodic ||
        !rejected_trigger_capacity) {
      fail("cache lifecycle contract",
           "checkpoint bookkeeping capacity or source ordering changed");
    }
    constexpr std::uint32_t kLayoutTestCapacity = 7;
    constexpr std::uint32_t kLayoutTestHead = 1;
    constexpr std::uint32_t kLayoutTestPosition = 2;
    constexpr std::size_t kCompactTestRow = compact_global_row_offset(
        kLayoutTestHead, kLayoutTestPosition, kLayoutTestCapacity);
    if (compact_global_layer_elements(kLayoutTestCapacity) !=
            static_cast<std::size_t>(model::kGlobalKvHeadCount) *
                kLayoutTestCapacity * 640 ||
        compact_global_row_offset(kLayoutTestHead, kLayoutTestPosition + 1,
                                  kLayoutTestCapacity) -
                kCompactTestRow !=
            640 ||
        kCompactTestRow + primitives::kGlobalCompactKeySize !=
            (static_cast<std::size_t>(kLayoutTestHead) *
                 kLayoutTestCapacity +
             kLayoutTestPosition) *
                    640 +
                128 ||
        kGenerationKvLayoutId !=
            "compact-global-k128-v512-v1") {
      fail("generation KV layout contract",
           "row offsets, compact partition, or stable identity changed");
    }
    bool rejected_overflow = false;
    try {
      (void)generation_cache_capacity(
          primitives::kMaxContextTokenCount, 2);
    } catch (const std::runtime_error&) {
      rejected_overflow = true;
    }
    if (!rejected_overflow) {
      fail("generation capacity contract",
           "request beyond 262144 fed tokens was accepted");
    }
    return true;
  } catch (const std::exception& error) {
    if (failure != nullptr) {
      *failure = error.what();
    }
    return false;
  }
}


}  // namespace gewell::app
int main() {
  std::string failure;
  if (gewell::app::run_self_tests(&failure)) return 0;
  std::cerr << failure << '\n';
  return 1;
}
