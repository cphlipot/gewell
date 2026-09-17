#include "models/gemma4/31b/sm120/runner_support.cuh"
#include "fixtures.h"

namespace gewell::diagnostics {
namespace {
using namespace gewell::gemma4_31b::sm120;
constexpr std::array<std::uint32_t, 2> kInputTokens{{2, 902}};
constexpr std::uint32_t kExpectedToken = 902;
constexpr std::uint32_t kTokens = 2;
constexpr std::size_t kCaptureCount = 114;
constexpr std::array<std::uint32_t, 16> kShortDecodePrompt{{
    2,      902,  2172,  18362, 2490,   496,  7501, 529,
    8369,   104026, 1131, 9948,  107, 199986, 1024, 506,
}};
// This runner deliberately builds the entire prompt as M=1 cache updates.  Its
// reference sequence is therefore the eager BF16 sequential-M1 result, not the
// M=16 cache-building branch stored as the oracle pack's canonical ranking.
constexpr std::array<std::uint32_t, 8> kShortDecodeM1Expected{{
    1638, 1076, 529, 21739, 236761, 107, 236772, 1852,
}};
constexpr std::uint32_t kShortDecodePromptTokens =
    static_cast<std::uint32_t>(kShortDecodePrompt.size());
constexpr std::uint32_t kShortDecodeDecisionCount =
    static_cast<std::uint32_t>(kShortDecodeM1Expected.size());
constexpr std::uint32_t kShortDecodeFedTokens =
    kShortDecodePromptTokens + kShortDecodeDecisionCount - 1;
constexpr std::size_t kShortDecodeCaptureCount = 572;
constexpr std::array<std::uint32_t, 90> kBoundaryRepeatedBody{{
    902,    2172,   18362,  2490,   496,    7501,   529,    8369,   104026,
    1131,   9948,   107,    199986, 1024,   506,    2148,   8369,   236761,
    562,    6684,   30998,  15033,  506,    2028,   236764, 107,    32061,
    236764, 8299,   3861,   236764, 18756,  4957,   236764, 532,    121160,
    6530,   236761, 12263,  532,    107,    13214,  5700,   735,    1607,
    15612,  6868,   236764, 1651,   19707,  35719,  659,    107,    47896,
    27222,  684,    6013,   18710,  657,    12539,  886,    236761, 66017,
    1374,   2072,   107,    27413,  11172,  8487,   532,    16932,  91102,
    106430, 53255,  236764, 11015,  1418,   10445,  107,    13271,  236764,
    532,    23227,  11652,  3938,   21739,  699,    62632,  25398,  236761,
}};
constexpr std::uint32_t kBoundaryPrefixTokens =
    primitives::kCachedAttentionM1BoundaryPositionCount;
constexpr std::uint32_t kBoundaryCaptureFirstPosition = 1'022;
constexpr std::uint32_t kBoundaryCaptureRows = 4;
constexpr std::uint32_t kBoundaryDecisionPosition = 1'025;
constexpr std::uint32_t kBoundaryExpectedToken = 121'160;
constexpr std::size_t kBoundaryCaptureCount = 90;
constexpr std::uint32_t kHybridPrefillTokens = prefill::kTokenCount;
constexpr std::array<std::uint32_t, 3> kHybridBoundaryExpected{{
    236'764,
    532,
    121'160,
}};
constexpr std::array<std::uint32_t, 3> kHybridBoundaryScheduleRows{{
    1'024,
    1,
    1,
}};
constexpr std::array<std::uint32_t, 3> kHybridBoundaryDecisionPositions{{
    1'023,
    1'024,
    1'025,
}};
constexpr std::string_view kBoundaryInputTokenIdsSha256 =
    "df25e40aaa35f85b2e9e3b258ef6a09309ddb012a31ee0a530668f669140cfae";

constexpr std::uint32_t kGlobalCacheCapacity = kTokens;
constexpr std::uint32_t kShortDecodeGlobalCacheCapacity =
    primitives::kCachedAttentionM1ShortCapacity;
constexpr std::uint32_t kBoundaryGlobalCacheCapacity =
    primitives::kCachedAttentionM1BoundaryGlobalCapacity;
constexpr std::uint32_t kGraphPromptTokens =
    primitives::kGraphPromptTokenCount;
constexpr std::uint32_t kGraphOutputTokens =
    primitives::kGraphOutputTokenCount;
constexpr std::uint32_t kGraphDecodeReplays = kGraphOutputTokens - 1;
constexpr std::uint32_t kGraphGlobalCacheCapacity =
    primitives::kGraphAttentionGlobalCapacity;
constexpr std::array<std::uint32_t, 3> kGraphExpectedFirstThree{{
    236'764,
    532,
    121'160,
}};
constexpr std::size_t kGraphWarmupRequests = 2;
constexpr std::size_t kGraphMeasuredRequests = 7;
constexpr std::size_t kGraphPersistentTimingEventCount = 4;
constexpr std::size_t kGraphRequestCudaObjectCreations = 0;
constexpr std::size_t kGraphExpectedNodeCount = 1'078;
constexpr std::size_t kGraphOutputBytes =
    static_cast<std::size_t>(kGraphOutputTokens) * sizeof(std::uint32_t);
constexpr std::uint32_t kProfileDecodeReplays = 1;
constexpr std::uint32_t kProfileDecodeFinalPosition =
    primitives::kGraphDecodeFirstPosition + kProfileDecodeReplays;
constexpr std::array<std::uint32_t, 2> kProfileDecodeExpectedOutputs{{
    236'764,
    532,
}};

constexpr std::uint32_t boundary_input_token(std::uint32_t position) {
  return position == 0
             ? 2
             : kBoundaryRepeatedBody[(position - 1) %
                                     kBoundaryRepeatedBody.size()];
}

constexpr std::uint64_t boundary_fixture_fingerprint() {
  std::uint64_t hash = 14'695'981'039'346'656'037ULL;
  for (std::uint32_t position = 0; position < kBoundaryPrefixTokens;
       ++position) {
    const std::uint32_t token = boundary_input_token(position);
    for (unsigned shift = 0; shift < 32; shift += 8) {
      hash ^= static_cast<std::uint8_t>(token >> shift);
      hash *= 1'099'511'628'211ULL;
    }
  }
  return hash;
}


constexpr std::size_t kGlobalCacheBytesPerKind =
    static_cast<std::size_t>(model::kGlobalLayerCount) *
    model::kGlobalKvHeadCount * kGlobalCacheCapacity * model::kGlobalHeadSize *
    sizeof(BFloat16);
constexpr std::size_t kShortDecodeGlobalCacheBytesPerKind =
    static_cast<std::size_t>(model::kGlobalLayerCount) *
    model::kGlobalKvHeadCount * kShortDecodeGlobalCacheCapacity *
    model::kGlobalHeadSize * sizeof(BFloat16);
constexpr std::size_t kBoundaryGlobalCacheBytesPerKind =
    static_cast<std::size_t>(model::kGlobalLayerCount) *
    model::kGlobalKvHeadCount * kBoundaryGlobalCacheCapacity *
    model::kGlobalHeadSize * sizeof(BFloat16);
constexpr std::size_t kGraphGlobalCacheBytesPerKind =
    static_cast<std::size_t>(model::kGlobalLayerCount) *
    model::kGlobalKvHeadCount * kGraphGlobalCacheCapacity *
    model::kGlobalHeadSize * sizeof(BFloat16);
constexpr std::size_t kGraphPersistentStateBytes =
    2 * sizeof(std::uint32_t) + kGraphOutputBytes;

static_assert(kGlobalCacheBytesPerKind == 81'920);
static_assert(kShortDecodeGlobalCacheBytesPerKind == 983'040);
static_assert(kBoundaryGlobalCacheBytesPerKind == 42'024'960);
static_assert(kGraphGlobalCacheBytesPerKind == 62'914'560);
static_assert(kGraphPromptTokens == kHybridPrefillTokens);
static_assert(kGraphOutputTokens == 512);
static_assert(kGraphDecodeReplays == 511);
static_assert(primitives::kGraphDecodeFirstPosition == 1'024);
static_assert(primitives::kGraphDecodeLastPosition == 1'534);
static_assert(primitives::kGraphDecodeFinalPosition == 1'535);
static_assert(primitives::kGraphDecodeLastPosition -
                      primitives::kGraphDecodeFirstPosition + 1 ==
                  kGraphDecodeReplays);
static_assert(kGraphPersistentStateBytes == 2'056);
static_assert(kShortDecodeFedTokens == 23);
static_assert(kBoundaryPrefixTokens == 1'026);
static_assert(kHybridPrefillTokens == 1'024);
static_assert(kHybridBoundaryExpected[0] ==
              boundary_input_token(kHybridPrefillTokens));
static_assert(kHybridBoundaryExpected[1] ==
              boundary_input_token(kHybridPrefillTokens + 1));
static_assert(kHybridBoundaryExpected[2] == kBoundaryExpectedToken);
static_assert(kBoundaryRepeatedBody.size() == 90);
static_assert(boundary_input_token(0) == 2);
static_assert(boundary_input_token(1) == 902);
static_assert(boundary_input_token(91) == 902);
static_assert(boundary_input_token(1'022) == 18'756);
static_assert(boundary_input_token(1'023) == 4'957);
static_assert(boundary_input_token(1'024) == 236'764);
static_assert(boundary_input_token(1'025) == 532);
static_assert(boundary_fixture_fingerprint() == 0xa266c2be06f28bbcULL);


struct ScratchLayout {
  static constexpr std::size_t kH0 = 0;
  static constexpr std::size_t kH1 =
      align_up(kH0 + kTokens * model::kHiddenSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kH2 =
      align_up(kH1 + kTokens * model::kHiddenSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kQueryRaw =
      align_up(kH2 + kTokens * model::kHiddenSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kQueryNorm =
      align_up(kQueryRaw + kTokens * model::kQueryHeadCount *
                               model::kGlobalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kQueryRope =
      align_up(kQueryNorm + kTokens * model::kQueryHeadCount *
                                model::kGlobalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kKeyRaw =
      align_up(kQueryRope + kTokens * model::kQueryHeadCount *
                                model::kGlobalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kKeyNorm =
      align_up(kKeyRaw + kTokens * model::kLocalKvHeadCount *
                              model::kLocalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kKeyRope =
      align_up(kKeyNorm + kTokens * model::kLocalKvHeadCount *
                               model::kLocalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kValueRaw =
      align_up(kKeyRope + kTokens * model::kLocalKvHeadCount *
                               model::kLocalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kValueNorm =
      align_up(kValueRaw + kTokens * model::kLocalKvHeadCount *
                                model::kLocalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kContext =
      align_up(kValueNorm + kTokens * model::kLocalKvHeadCount *
                                 model::kLocalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kGate =
      align_up(kContext + kTokens * model::kQueryHeadCount *
                              model::kGlobalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kUp =
      align_up(kGate + kTokens * model::kMlpSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kProduct =
      align_up(kUp + kTokens * model::kMlpSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kLogits =
      align_up(kProduct + kTokens * model::kMlpSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kCappedLogits =
      align_up(kLogits + model::kVocabSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kLocalCos =
      align_up(kCappedLogits + model::kVocabSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kLocalSin =
      align_up(kLocalCos + kTokens * model::kLocalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kGlobalCos =
      align_up(kLocalSin + kTokens * model::kLocalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kGlobalSin =
      align_up(kGlobalCos + kTokens * model::kGlobalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kProbabilities =
      align_up(kGlobalSin + kTokens * model::kGlobalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kArgmax =
      kProbabilities + model::kQueryHeadCount * kTokens * kTokens *
                           sizeof(BFloat16);
  static constexpr std::size_t kBytes =
      align_up(kArgmax + sizeof(std::uint32_t), kScratchAlignment);
};

static_assert(ScratchLayout::kH1 == 21'504);
static_assert(ScratchLayout::kQueryRaw == 64'512);
static_assert(ScratchLayout::kQueryRope == 195'584);
static_assert(ScratchLayout::kKeyRaw == 261'120);
static_assert(ScratchLayout::kContext == 343'040);
static_assert(ScratchLayout::kLogits == 666'624);
static_assert(ScratchLayout::kProbabilities == 1'721'344);
static_assert(ScratchLayout::kBytes == 1'721'856);

struct CachedScratchLayout {
  static constexpr std::size_t kH0 = 0;
  static constexpr std::size_t kH1 =
      align_up(kH0 + model::kHiddenSize * sizeof(BFloat16), kScratchAlignment);
  static constexpr std::size_t kH2 =
      align_up(kH1 + model::kHiddenSize * sizeof(BFloat16), kScratchAlignment);
  static constexpr std::size_t kQueryRaw =
      align_up(kH2 + model::kHiddenSize * sizeof(BFloat16), kScratchAlignment);
  static constexpr std::size_t kQueryNorm =
      align_up(kQueryRaw + model::kQueryHeadCount * model::kGlobalHeadSize *
                               sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kQueryRope =
      align_up(kQueryNorm + model::kQueryHeadCount * model::kGlobalHeadSize *
                                sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kKeyRaw =
      align_up(kQueryRope + model::kQueryHeadCount * model::kGlobalHeadSize *
                                sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kKeyNorm =
      align_up(kKeyRaw + model::kLocalKvHeadCount * model::kLocalHeadSize *
                              sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kKeyRope =
      align_up(kKeyNorm + model::kLocalKvHeadCount * model::kLocalHeadSize *
                               sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kValueRaw =
      align_up(kKeyRope + model::kLocalKvHeadCount * model::kLocalHeadSize *
                               sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kValueNorm =
      align_up(kValueRaw + model::kLocalKvHeadCount * model::kLocalHeadSize *
                                sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kContext =
      align_up(kValueNorm + model::kLocalKvHeadCount * model::kLocalHeadSize *
                                 sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kGate =
      align_up(kContext + model::kQueryHeadCount * model::kGlobalHeadSize *
                              sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kUp =
      align_up(kGate + model::kMlpSize * sizeof(BFloat16), kScratchAlignment);
  static constexpr std::size_t kProduct =
      align_up(kUp + model::kMlpSize * sizeof(BFloat16), kScratchAlignment);
  static constexpr std::size_t kLogits =
      align_up(kProduct + model::kMlpSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kCappedLogits =
      align_up(kLogits + model::kVocabSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kLocalCos =
      align_up(kCappedLogits + model::kVocabSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kLocalSin =
      align_up(kLocalCos + kTokens * model::kLocalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kGlobalCos =
      align_up(kLocalSin + kTokens * model::kLocalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kGlobalSin =
      align_up(kGlobalCos + kTokens * model::kGlobalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kProbabilities =
      align_up(kGlobalSin + kTokens * model::kGlobalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kArgmax =
      kProbabilities + model::kQueryHeadCount * kTokens * sizeof(BFloat16);
  static constexpr std::size_t kBytes =
      align_up(kArgmax + sizeof(std::uint32_t), kScratchAlignment);
};

static_assert(CachedScratchLayout::kQueryRaw == 32'256);
static_assert(CachedScratchLayout::kQueryRope == 97'792);
static_assert(CachedScratchLayout::kKeyRaw == 130'560);
static_assert(CachedScratchLayout::kContext == 171'520);
static_assert(CachedScratchLayout::kLogits == 333'312);
static_assert(CachedScratchLayout::kProbabilities == 1'388'032);
static_assert(CachedScratchLayout::kBytes == 1'388'288);

struct ShortDecodeScratchLayout {
  static constexpr std::size_t kH0 = CachedScratchLayout::kH0;
  static constexpr std::size_t kH1 = CachedScratchLayout::kH1;
  static constexpr std::size_t kH2 = CachedScratchLayout::kH2;
  static constexpr std::size_t kQueryRaw = CachedScratchLayout::kQueryRaw;
  static constexpr std::size_t kQueryNorm = CachedScratchLayout::kQueryNorm;
  static constexpr std::size_t kQueryRope = CachedScratchLayout::kQueryRope;
  static constexpr std::size_t kKeyRaw = CachedScratchLayout::kKeyRaw;
  static constexpr std::size_t kKeyNorm = CachedScratchLayout::kKeyNorm;
  static constexpr std::size_t kKeyRope = CachedScratchLayout::kKeyRope;
  static constexpr std::size_t kValueRaw = CachedScratchLayout::kValueRaw;
  static constexpr std::size_t kValueNorm = CachedScratchLayout::kValueNorm;
  static constexpr std::size_t kContext = CachedScratchLayout::kContext;
  static constexpr std::size_t kGate = CachedScratchLayout::kGate;
  static constexpr std::size_t kUp = CachedScratchLayout::kUp;
  static constexpr std::size_t kProduct = CachedScratchLayout::kProduct;
  static constexpr std::size_t kLogits = CachedScratchLayout::kLogits;
  static constexpr std::size_t kCappedLogits =
      CachedScratchLayout::kCappedLogits;
  static constexpr std::size_t kLocalCos = CachedScratchLayout::kLocalCos;
  static constexpr std::size_t kLocalSin =
      align_up(kLocalCos + kShortDecodeGlobalCacheCapacity *
                               model::kLocalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kGlobalCos =
      align_up(kLocalSin + kShortDecodeGlobalCacheCapacity *
                               model::kLocalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kGlobalSin =
      align_up(kGlobalCos + kShortDecodeGlobalCacheCapacity *
                                model::kGlobalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kProbabilities =
      align_up(kGlobalSin + kShortDecodeGlobalCacheCapacity *
                                model::kGlobalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kArgmax =
      kProbabilities + model::kQueryHeadCount *
                           primitives::kCachedAttentionM1ShortCapacity *
                           sizeof(BFloat16);
  static constexpr std::size_t kBytes =
      align_up(kArgmax + sizeof(std::uint32_t), kScratchAlignment);
};

static_assert(ShortDecodeScratchLayout::kLocalCos == 1'381'888);
static_assert(ShortDecodeScratchLayout::kLocalSin == 1'394'176);
static_assert(ShortDecodeScratchLayout::kGlobalCos == 1'406'464);
static_assert(ShortDecodeScratchLayout::kGlobalSin == 1'431'040);
static_assert(ShortDecodeScratchLayout::kProbabilities == 1'455'616);
static_assert(ShortDecodeScratchLayout::kBytes == 1'457'408);

struct BoundaryScratchLayout {
  static constexpr std::size_t kH0 = CachedScratchLayout::kH0;
  static constexpr std::size_t kH1 = CachedScratchLayout::kH1;
  static constexpr std::size_t kH2 = CachedScratchLayout::kH2;
  static constexpr std::size_t kQueryRaw = CachedScratchLayout::kQueryRaw;
  static constexpr std::size_t kQueryNorm = CachedScratchLayout::kQueryNorm;
  static constexpr std::size_t kQueryRope = CachedScratchLayout::kQueryRope;
  static constexpr std::size_t kKeyRaw = CachedScratchLayout::kKeyRaw;
  static constexpr std::size_t kKeyNorm = CachedScratchLayout::kKeyNorm;
  static constexpr std::size_t kKeyRope = CachedScratchLayout::kKeyRope;
  static constexpr std::size_t kValueRaw = CachedScratchLayout::kValueRaw;
  static constexpr std::size_t kValueNorm = CachedScratchLayout::kValueNorm;
  static constexpr std::size_t kContext = CachedScratchLayout::kContext;
  static constexpr std::size_t kGate = CachedScratchLayout::kGate;
  static constexpr std::size_t kUp = CachedScratchLayout::kUp;
  static constexpr std::size_t kProduct = CachedScratchLayout::kProduct;
  static constexpr std::size_t kLogits = CachedScratchLayout::kLogits;
  static constexpr std::size_t kCappedLogits =
      CachedScratchLayout::kCappedLogits;
  static constexpr std::size_t kLocalCos = CachedScratchLayout::kLocalCos;
  static constexpr std::size_t kLocalSin =
      align_up(kLocalCos + model::kLocalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kGlobalCos =
      align_up(kLocalSin + model::kLocalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kGlobalSin =
      align_up(kGlobalCos + model::kGlobalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kScoreScratch =
      align_up(kGlobalSin + model::kGlobalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kProbabilities =
      align_up(kScoreScratch +
                   primitives::kCachedAttentionM1BoundaryScoreScratchBytes,
               kScratchAlignment);
  static constexpr std::size_t kArgmax =
      align_up(kProbabilities + model::kQueryHeadCount *
                                    kBoundaryPrefixTokens * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kBytes =
      align_up(kArgmax + sizeof(std::uint32_t), kScratchAlignment);
};

static_assert(BoundaryScratchLayout::kLocalCos == 1'381'888);
static_assert(BoundaryScratchLayout::kLocalSin == 1'382'400);
static_assert(BoundaryScratchLayout::kGlobalCos == 1'382'912);
static_assert(BoundaryScratchLayout::kGlobalSin == 1'383'936);
static_assert(BoundaryScratchLayout::kScoreScratch == 1'384'960);
static_assert(BoundaryScratchLayout::kProbabilities == 1'450'752);
static_assert(BoundaryScratchLayout::kArgmax == 1'516'544);
static_assert(BoundaryScratchLayout::kBytes == 1'516'800);

struct HybridBoundaryScratchLayout {
  static constexpr std::size_t kPrefillHiddenElements =
      static_cast<std::size_t>(kHybridPrefillTokens) * model::kHiddenSize;
  static constexpr std::size_t kPrefillQueryElements =
      static_cast<std::size_t>(kHybridPrefillTokens) *
      model::kQueryHeadCount * model::kGlobalHeadSize;
  static constexpr std::size_t kPrefillKvElements =
      static_cast<std::size_t>(kHybridPrefillTokens) *
      model::kLocalKvHeadCount * model::kLocalHeadSize;
  static constexpr std::size_t kPrefillContextElements =
      kPrefillQueryElements;
  static constexpr std::size_t kPrefillMlpElements =
      static_cast<std::size_t>(kHybridPrefillTokens) * model::kMlpSize;
  static constexpr std::size_t kGatherElements =
      static_cast<std::size_t>(model::kQueryHeadCount) *
      kBoundaryPrefixTokens;

  static constexpr std::size_t kH0 = 0;
  static constexpr std::size_t kH1 =
      align_up(kH0 + kPrefillHiddenElements * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kH2 =
      align_up(kH1 + kPrefillHiddenElements * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kQueryRaw =
      align_up(kH2 + kPrefillHiddenElements * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kQueryNorm =
      align_up(kQueryRaw + kPrefillQueryElements * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kQueryRope =
      align_up(kQueryNorm + kPrefillQueryElements * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kKeyRaw =
      align_up(kQueryRope + kPrefillQueryElements * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kKeyNorm =
      align_up(kKeyRaw + kPrefillKvElements * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kKeyRope =
      align_up(kKeyNorm + kPrefillKvElements * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kValueRaw =
      align_up(kKeyRope + kPrefillKvElements * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kValueNorm =
      align_up(kValueRaw + kPrefillKvElements * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kContext =
      align_up(kValueNorm + kPrefillKvElements * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kGate =
      align_up(kContext + kPrefillContextElements * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kUp =
      align_up(kGate + kPrefillMlpElements * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kProduct =
      align_up(kUp + kPrefillMlpElements * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kLogits =
      align_up(kProduct + kPrefillMlpElements * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kCappedLogits =
      align_up(kLogits + model::kVocabSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kLocalCos =
      align_up(kCappedLogits + model::kVocabSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kLocalSin =
      align_up(kLocalCos + static_cast<std::size_t>(kHybridPrefillTokens) *
                                model::kLocalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kGlobalCos =
      align_up(kLocalSin + static_cast<std::size_t>(kHybridPrefillTokens) *
                                model::kLocalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kGlobalSin =
      align_up(kGlobalCos + static_cast<std::size_t>(kHybridPrefillTokens) *
                                 model::kGlobalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kScoreScratch =
      align_up(kGlobalSin + static_cast<std::size_t>(kHybridPrefillTokens) *
                                 model::kGlobalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kProbabilities =
      align_up(kScoreScratch + prefill::kAttentionScoreScratchBytes,
               kScratchAlignment);
  static constexpr std::size_t kGatherStaging =
      align_up(kProbabilities + prefill::kAttentionProbabilityBytes,
               kScratchAlignment);
  static constexpr std::size_t kArgmax =
      align_up(kGatherStaging + kGatherElements * sizeof(BFloat16),
               kScratchAlignment);
  static constexpr std::size_t kBytes =
      align_up(kArgmax + sizeof(std::uint32_t), kScratchAlignment);
};

static_assert(HybridBoundaryScratchLayout::kH1 == 11'010'048);
static_assert(HybridBoundaryScratchLayout::kQueryRaw == 33'030'144);
static_assert(HybridBoundaryScratchLayout::kQueryRope == 100'139'008);
static_assert(HybridBoundaryScratchLayout::kKeyRaw == 133'693'440);
static_assert(HybridBoundaryScratchLayout::kContext == 175'636'480);
static_assert(HybridBoundaryScratchLayout::kLogits == 341'311'488);
static_assert(HybridBoundaryScratchLayout::kScoreScratch == 345'505'792);
static_assert(HybridBoundaryScratchLayout::kProbabilities == 412'614'656);
static_assert(HybridBoundaryScratchLayout::kGatherStaging == 479'723'520);
static_assert(HybridBoundaryScratchLayout::kArgmax == 479'789'312);
static_assert(HybridBoundaryScratchLayout::kBytes == 479'789'568);
static_assert(HybridBoundaryScratchLayout::kGatherElements *
                      sizeof(BFloat16) ==
                  primitives::kCachedAttentionM1BoundaryScoreScratchBytes);
static_assert(HybridBoundaryScratchLayout::kScoreScratch +
                      primitives::kGraphAttentionFusedScratchBytes <=
                  HybridBoundaryScratchLayout::kProbabilities);
static_assert(HybridBoundaryScratchLayout::kProbabilities +
                      model::kQueryHeadCount *
                          primitives::kGraphAttentionPositionCount *
                          sizeof(BFloat16) <=
                  HybridBoundaryScratchLayout::kGatherStaging);
static_assert(HybridBoundaryScratchLayout::kArgmax + sizeof(std::uint32_t) <=
              HybridBoundaryScratchLayout::kBytes);


struct CaptureSpec {
  std::string name;
  std::size_t elements{};
};

std::string layer_capture_name(std::uint32_t layer, std::string_view field) {
  char buffer[96];
  const int length = std::snprintf(buffer, sizeof(buffer),
                                   "prefill.layer.%02u.%.*s", layer,
                                   static_cast<int>(field.size()), field.data());
  if (length < 0 || static_cast<std::size_t>(length) >= sizeof(buffer)) {
    fail("capture name", "formatted name exceeds fixed buffer");
  }
  return std::string(buffer, static_cast<std::size_t>(length));
}

void add_deep_capture_specs(std::vector<CaptureSpec>* specs,
                            std::uint32_t layer) {
  const bool global = model::is_global_layer(layer);
  const std::size_t head_size =
      global ? model::kGlobalHeadSize : model::kLocalHeadSize;
  const std::size_t kv_heads =
      global ? model::kGlobalKvHeadCount : model::kLocalKvHeadCount;
  const std::size_t q_width = model::kQueryHeadCount * head_size;
  const std::size_t kv_width = kv_heads * head_size;
  const auto add = [&](std::string_view field, std::size_t elements) {
    specs->push_back({layer_capture_name(layer, field), elements});
  };
  add("block_input", kTokens * model::kHiddenSize);
  add("layer_scalar", 1);
  add("input_norm", kTokens * model::kHiddenSize);
  add("q_raw", kTokens * q_width);
  add("q_norm", kTokens * q_width);
  add("q_rope", kTokens * q_width);
  add("k_raw", kTokens * kv_width);
  add("k_norm", kTokens * kv_width);
  add("k_rope", kTokens * kv_width);
  add("v_raw", kTokens * kv_width);
  add("v_norm", kTokens * kv_width);
  add("attention_probabilities",
      model::kQueryHeadCount * kTokens * kTokens);
  add("attention_context", kTokens * q_width);
  add("attention_output", kTokens * model::kHiddenSize);
  add("post_attention_norm", kTokens * model::kHiddenSize);
  add("post_attention_residual", kTokens * model::kHiddenSize);
  add("pre_feedforward_norm", kTokens * model::kHiddenSize);
  add("mlp_gate", kTokens * model::kMlpSize);
  add("mlp_up", kTokens * model::kMlpSize);
  add("mlp_product", kTokens * model::kMlpSize);
  add("mlp_down", kTokens * model::kHiddenSize);
  add("post_feedforward_norm", kTokens * model::kHiddenSize);
  add("pre_scalar_output", kTokens * model::kHiddenSize);
}

std::vector<CaptureSpec> make_capture_specs() {
  std::vector<CaptureSpec> specs;
  specs.reserve(kCaptureCount);
  specs.push_back({"prefill.embedding", kTokens * model::kHiddenSize});
  specs.push_back({"prefill.final_norm", kTokens * model::kHiddenSize});
  specs.push_back({"prefill.logits.pre_softcap", model::kVocabSize});
  specs.push_back({"prefill.logits.post_softcap", model::kVocabSize});
  specs.push_back({"prefill.rotary.sliding_attention.cos",
                   kTokens * model::kLocalHeadSize});
  specs.push_back({"prefill.rotary.sliding_attention.sin",
                   kTokens * model::kLocalHeadSize});
  specs.push_back({"prefill.rotary.full_attention.cos",
                   kTokens * model::kGlobalHeadSize});
  specs.push_back({"prefill.rotary.full_attention.sin",
                   kTokens * model::kGlobalHeadSize});
  for (std::uint32_t layer = 0; layer < model::kLayerCount; ++layer) {
    specs.push_back({layer_capture_name(layer, "output"),
                     kTokens * model::kHiddenSize});
  }
  add_deep_capture_specs(&specs, 0);
  add_deep_capture_specs(&specs, 5);
  std::sort(specs.begin(), specs.end(),
            [](const CaptureSpec& left, const CaptureSpec& right) {
              return left.name < right.name;
            });
  return specs;
}

struct CaptureRecord {
  std::string name;
  std::size_t elements{};
  std::size_t byte_offset{};
  bool captured{};
  std::uint8_t partial_mask{};
};


class CapturePlan {
 public:
  explicit CapturePlan(std::filesystem::path directory)
      : directory_(std::move(directory)) {
    std::error_code error;
    if (std::filesystem::exists(directory_, error) || error) {
      fail("capture directory",
           error ? error.message() : "path already exists (overwrite refused)");
    }
    const std::vector<CaptureSpec> specs = make_capture_specs();
    if (specs.size() != kCaptureCount) {
      fail("capture plan", "wrong capture count");
    }
    records_.reserve(specs.size());
    std::size_t bytes = 0;
    for (const CaptureSpec& spec : specs) {
      records_.push_back({spec.name, spec.elements, bytes, false, 0});
      bytes += spec.elements * sizeof(BFloat16);
    }
    check_cuda(cudaMallocHost(&host_arena_, bytes),
               "cudaMallocHost pair captures");
  }
  ~CapturePlan() {
    if (host_arena_ != nullptr) {
      cudaFreeHost(host_arena_);
    }
  }
  CapturePlan(const CapturePlan&) = delete;
  CapturePlan& operator=(const CapturePlan&) = delete;

  void copy_device(std::string_view name, const BFloat16* source,
                   cudaStream_t stream = nullptr) {
    CaptureRecord& record = find(name);
    if (record.captured || record.partial_mask != 0) {
      fail("capture", std::string(name) + " was produced twice");
    }
    auto* destination =
        static_cast<std::uint8_t*>(host_arena_) + record.byte_offset;
    check_cuda(cudaMemcpyAsync(destination, source,
                               record.elements * sizeof(BFloat16),
                               cudaMemcpyDeviceToHost, stream),
               "capture device-to-host copy");
    record.captured = true;
  }

  // Assemble a token-major oracle tensor from two independent M=1 steps.
  void copy_token(std::string_view name, std::uint32_t position,
                  const BFloat16* source, std::size_t elements,
                  cudaStream_t stream = nullptr) {
    CaptureRecord& record = find(name);
    if (position >= kTokens || record.elements != kTokens * elements ||
        record.captured || (record.partial_mask & (1U << position)) != 0) {
      fail("capture token slice", std::string(name) + " has an invalid slice");
    }
    auto* destination = static_cast<std::uint8_t*>(host_arena_) +
                        record.byte_offset +
                        position * elements * sizeof(BFloat16);
    check_cuda(cudaMemcpyAsync(destination, source, elements * sizeof(BFloat16),
                               cudaMemcpyDeviceToHost, stream),
               "capture token-slice device-to-host copy");
    record.partial_mask |= static_cast<std::uint8_t>(1U << position);
    record.captured = record.partial_mask == 0x3;
  }

  // Assemble oracle head-major [heads,2,width] tensors from M=1 head-major
  // [heads,width] steps. This also covers [32,2,2] probabilities.
  void copy_head_token(std::string_view name, std::uint32_t position,
                       const BFloat16* source, std::uint32_t heads,
                       std::uint32_t width,
                       cudaStream_t stream = nullptr) {
    CaptureRecord& record = find(name);
    const std::size_t step_elements =
        static_cast<std::size_t>(heads) * width;
    if (position >= kTokens ||
        record.elements != kTokens * step_elements || record.captured ||
        (record.partial_mask & (1U << position)) != 0) {
      fail("capture head/token slice",
           std::string(name) + " has an invalid slice");
    }
    auto* destination = static_cast<std::uint8_t*>(host_arena_) +
                        record.byte_offset;
    const std::size_t row_bytes = width * sizeof(BFloat16);
    for (std::uint32_t head = 0; head < heads; ++head) {
      const std::size_t destination_row =
          (static_cast<std::size_t>(head) * kTokens + position) * width;
      check_cuda(cudaMemcpyAsync(destination +
                                     destination_row * sizeof(BFloat16),
                                 source + static_cast<std::size_t>(head) * width,
                                 row_bytes, cudaMemcpyDeviceToHost, stream),
                 "capture head/token device-to-host copy");
    }
    record.partial_mask |= static_cast<std::uint8_t>(1U << position);
    record.captured = record.partial_mask == 0x3;
  }

  void write(std::uint32_t token) {
    check_cuda(cudaDeviceSynchronize(), "synchronize pair captures");
    for (const CaptureRecord& record : records_) {
      if (!record.captured) {
        fail("capture", record.name + " was not produced");
      }
    }
    std::error_code error;
    if (!std::filesystem::create_directory(directory_, error)) {
      fail("capture directory",
           error ? error.message() : "path already exists (overwrite refused)");
    }
    for (const CaptureRecord& record : records_) {
      const auto* source = static_cast<const std::uint8_t*>(host_arena_) +
                           record.byte_offset;
      write_exclusive(directory_ / (record.name + ".bf16"), source,
                      record.elements * sizeof(BFloat16));
    }
    const std::array<std::uint8_t, 4> encoded_token{{
        static_cast<std::uint8_t>(token),
        static_cast<std::uint8_t>(token >> 8),
        static_cast<std::uint8_t>(token >> 16),
        static_cast<std::uint8_t>(token >> 24),
    }};
    write_exclusive(directory_ / "argmax.u32", encoded_token.data(),
                    encoded_token.size());
  }

  [[nodiscard]] std::size_t size() const { return records_.size(); }
  [[nodiscard]] const std::filesystem::path& directory() const {
    return directory_;
  }

 private:
  CaptureRecord& find(std::string_view name) {
    for (CaptureRecord& record : records_) {
      if (record.name == name) {
        return record;
      }
    }
    fail("capture", std::string(name) + " is not in the pair inventory");
  }

  std::filesystem::path directory_;
  std::vector<CaptureRecord> records_;
  void* host_arena_{nullptr};
};

std::string short_decode_step_capture_name(std::uint32_t step,
                                           std::string_view suffix) {
  char buffer[128];
  const int length = std::snprintf(
      buffer, sizeof(buffer), "generation.step.%02u.%.*s", step,
      static_cast<int>(suffix.size()), suffix.data());
  if (length < 0 || static_cast<std::size_t>(length) >= sizeof(buffer)) {
    fail("short-decode capture name", "formatted name exceeds fixed buffer");
  }
  return std::string(buffer, static_cast<std::size_t>(length));
}

std::vector<CaptureSpec> make_short_decode_capture_specs() {
  std::vector<CaptureSpec> specs;
  specs.reserve(kShortDecodeCaptureCount);
  specs.push_back({"prefill.embedding",
                   kShortDecodePromptTokens * model::kHiddenSize});
  specs.push_back({"prefill.final_norm",
                   kShortDecodePromptTokens * model::kHiddenSize});
  specs.push_back({"prefill.logits.pre_softcap", model::kVocabSize});
  specs.push_back({"prefill.logits.post_softcap", model::kVocabSize});
  specs.push_back({"prefill.rotary.sliding_attention.cos",
                   kShortDecodePromptTokens * model::kLocalHeadSize});
  specs.push_back({"prefill.rotary.sliding_attention.sin",
                   kShortDecodePromptTokens * model::kLocalHeadSize});
  specs.push_back({"prefill.rotary.full_attention.cos",
                   kShortDecodePromptTokens * model::kGlobalHeadSize});
  specs.push_back({"prefill.rotary.full_attention.sin",
                   kShortDecodePromptTokens * model::kGlobalHeadSize});
  for (std::uint32_t layer = 0; layer < model::kLayerCount; ++layer) {
    specs.push_back({layer_capture_name(layer, "output"),
                     kShortDecodePromptTokens * model::kHiddenSize});
  }
  for (std::uint32_t step = 0; step < kShortDecodeDecisionCount; ++step) {
    for (std::uint32_t layer = 0; layer < model::kLayerCount; ++layer) {
      char suffix[96];
      const int length = std::snprintf(suffix, sizeof(suffix),
                                       "cached.layer.%02u.output", layer);
      if (length < 0 || static_cast<std::size_t>(length) >= sizeof(suffix)) {
        fail("short-decode capture name",
             "formatted layer suffix exceeds fixed buffer");
      }
      specs.push_back(
          {short_decode_step_capture_name(
               step, std::string_view(suffix, static_cast<std::size_t>(length))),
           model::kHiddenSize});
    }
    specs.push_back({short_decode_step_capture_name(step, "cached.final_norm"),
                     model::kHiddenSize});
    specs.push_back({short_decode_step_capture_name(
                         step, "logits.cached.pre_softcap"),
                     model::kVocabSize});
    specs.push_back({short_decode_step_capture_name(
                         step, "logits.cached.post_softcap"),
                     model::kVocabSize});
  }
  std::sort(specs.begin(), specs.end(),
            [](const CaptureSpec& left, const CaptureSpec& right) {
              return left.name < right.name;
            });
  return specs;
}

struct ShortDecodeCaptureRecord {
  std::string name;
  std::size_t elements{};
  std::size_t byte_offset{};
  bool captured{};
  std::uint32_t prefill_mask{};
};

constexpr std::array<std::uint8_t, kShortDecodeDecisionCount * 4>
encode_short_decode_tokens(
    const std::array<std::uint32_t, kShortDecodeDecisionCount>& generated) {
  std::array<std::uint8_t, kShortDecodeDecisionCount * 4> encoded{};
  for (std::size_t index = 0; index < generated.size(); ++index) {
    const std::uint32_t token = generated[index];
    encoded[4 * index] = static_cast<std::uint8_t>(token);
    encoded[4 * index + 1] = static_cast<std::uint8_t>(token >> 8);
    encoded[4 * index + 2] = static_cast<std::uint8_t>(token >> 16);
    encoded[4 * index + 3] = static_cast<std::uint8_t>(token >> 24);
  }
  return encoded;
}

class ShortDecodeCapturePlan {
 public:
  explicit ShortDecodeCapturePlan(std::filesystem::path directory)
      : directory_(std::move(directory)) {
    std::error_code error;
    if (std::filesystem::exists(directory_, error) || error) {
      fail("short-decode capture directory",
           error ? error.message() : "path already exists (overwrite refused)");
    }
    const std::vector<CaptureSpec> specs =
        make_short_decode_capture_specs();
    if (specs.size() != kShortDecodeCaptureCount) {
      fail("short-decode capture plan", "wrong capture count");
    }
    records_.reserve(specs.size());
    std::size_t bytes = 0;
    for (const CaptureSpec& spec : specs) {
      records_.push_back({spec.name, spec.elements, bytes, false, 0});
      bytes += spec.elements * sizeof(BFloat16);
    }
    check_cuda(cudaMallocHost(&host_arena_, bytes),
               "cudaMallocHost short-decode captures");
  }

  ~ShortDecodeCapturePlan() {
    if (host_arena_ != nullptr) {
      cudaFreeHost(host_arena_);
    }
  }
  ShortDecodeCapturePlan(const ShortDecodeCapturePlan&) = delete;
  ShortDecodeCapturePlan& operator=(const ShortDecodeCapturePlan&) = delete;

  void copy_device(std::string_view name, const BFloat16* source,
                   cudaStream_t stream = nullptr) {
    ShortDecodeCaptureRecord& record = find(name);
    if (record.captured || record.prefill_mask != 0) {
      fail("short-decode capture", std::string(name) + " was produced twice");
    }
    auto* destination =
        static_cast<std::uint8_t*>(host_arena_) + record.byte_offset;
    check_cuda(cudaMemcpyAsync(destination, source,
                               record.elements * sizeof(BFloat16),
                               cudaMemcpyDeviceToHost, stream),
               "short-decode capture device-to-host copy");
    record.captured = true;
  }

  void copy_prefill_row(std::string_view name, std::uint32_t position,
                        const BFloat16* source, std::size_t elements,
                        cudaStream_t stream = nullptr) {
    ShortDecodeCaptureRecord& record = find(name);
    const std::uint32_t bit = 1U << position;
    constexpr std::uint32_t kCompleteMask =
        (1U << kShortDecodePromptTokens) - 1U;
    if (position >= kShortDecodePromptTokens ||
        record.elements != kShortDecodePromptTokens * elements ||
        record.captured || (record.prefill_mask & bit) != 0) {
      fail("short-decode prefill capture",
           std::string(name) + " has an invalid row");
    }
    auto* destination = static_cast<std::uint8_t*>(host_arena_) +
                        record.byte_offset +
                        position * elements * sizeof(BFloat16);
    check_cuda(cudaMemcpyAsync(destination, source, elements * sizeof(BFloat16),
                               cudaMemcpyDeviceToHost, stream),
               "short-decode prefill row device-to-host copy");
    record.prefill_mask |= bit;
    record.captured = record.prefill_mask == kCompleteMask;
  }

  void write(const std::array<std::uint32_t, kShortDecodeDecisionCount>&
                 generated) {
    check_cuda(cudaDeviceSynchronize(),
               "synchronize short-decode captures");
    for (const ShortDecodeCaptureRecord& record : records_) {
      if (!record.captured) {
        fail("short-decode capture", record.name + " was not produced");
      }
    }
    std::error_code error;
    if (!std::filesystem::create_directory(directory_, error)) {
      fail("short-decode capture directory",
           error ? error.message() : "path already exists (overwrite refused)");
    }
    for (const ShortDecodeCaptureRecord& record : records_) {
      const auto* source = static_cast<const std::uint8_t*>(host_arena_) +
                           record.byte_offset;
      write_exclusive(directory_ / (record.name + ".bf16"), source,
                      record.elements * sizeof(BFloat16));
    }
    const auto encoded = encode_short_decode_tokens(generated);
    write_exclusive(directory_ / "generated_tokens.u32", encoded.data(),
                    encoded.size());
  }

  [[nodiscard]] std::size_t size() const { return records_.size(); }
  [[nodiscard]] const std::filesystem::path& directory() const {
    return directory_;
  }

 private:
  ShortDecodeCaptureRecord& find(std::string_view name) {
    for (ShortDecodeCaptureRecord& record : records_) {
      if (record.name == name) {
        return record;
      }
    }
    fail("short-decode capture",
         std::string(name) + " is not in the runtime inventory");
  }

  std::filesystem::path directory_;
  std::vector<ShortDecodeCaptureRecord> records_;
  void* host_arena_{nullptr};
};

std::string boundary_rows_capture_name(std::string_view field) {
  constexpr std::string_view prefix =
      "boundary.cached.rows_1022_1025.";
  std::string result;
  result.reserve(prefix.size() + field.size());
  result.append(prefix);
  result.append(field);
  return result;
}

std::string boundary_layer_capture_name(std::uint32_t layer,
                                        std::string_view field) {
  char buffer[160];
  const int length = std::snprintf(
      buffer, sizeof(buffer),
      "boundary.cached.rows_1022_1025.layer.%02u.%.*s", layer,
      static_cast<int>(field.size()), field.data());
  if (length < 0 || static_cast<std::size_t>(length) >= sizeof(buffer)) {
    fail("boundary capture name", "formatted name exceeds fixed buffer");
  }
  return std::string(buffer, static_cast<std::size_t>(length));
}

void add_boundary_deep_capture_specs(std::vector<CaptureSpec>* specs,
                                     std::uint32_t layer) {
  const bool global = model::is_global_layer(layer);
  const std::size_t head_size =
      global ? model::kGlobalHeadSize : model::kLocalHeadSize;
  const std::size_t kv_heads =
      global ? model::kGlobalKvHeadCount : model::kLocalKvHeadCount;
  const std::size_t q_width = model::kQueryHeadCount * head_size;
  const std::size_t kv_width = kv_heads * head_size;
  const auto add = [&](std::string_view field, std::size_t row_elements) {
    specs->push_back({boundary_layer_capture_name(layer, field),
                      kBoundaryCaptureRows * row_elements});
  };
  add("q_raw", q_width);
  add("q_norm", q_width);
  add("q_rope", q_width);
  add("k_raw", kv_width);
  add("k_norm", kv_width);
  add("k_rope", kv_width);
  add("v_raw", kv_width);
  add("v_norm", kv_width);
  add("attention_probabilities",
      model::kQueryHeadCount * kBoundaryPrefixTokens);
  add("attention_context", q_width);
  add("attention_output", model::kHiddenSize);
}

std::vector<CaptureSpec> make_boundary_capture_specs() {
  std::vector<CaptureSpec> specs;
  specs.reserve(kBoundaryCaptureCount);
  const auto add_rows = [&](std::string_view field,
                            std::size_t row_elements) {
    specs.push_back({boundary_rows_capture_name(field),
                     kBoundaryCaptureRows * row_elements});
  };
  add_rows("embedding", model::kHiddenSize);
  add_rows("final_norm", model::kHiddenSize);
  add_rows("rotary.sliding_attention.cos", model::kLocalHeadSize);
  add_rows("rotary.sliding_attention.sin", model::kLocalHeadSize);
  add_rows("rotary.full_attention.cos", model::kGlobalHeadSize);
  add_rows("rotary.full_attention.sin", model::kGlobalHeadSize);
  for (std::uint32_t layer = 0; layer < model::kLayerCount; ++layer) {
    specs.push_back({boundary_layer_capture_name(layer, "output"),
                     kBoundaryCaptureRows * model::kHiddenSize});
  }
  add_boundary_deep_capture_specs(&specs, 0);
  add_boundary_deep_capture_specs(&specs, 5);
  specs.push_back(
      {"boundary.cached.position_1025.logits.pre_softcap",
       model::kVocabSize});
  specs.push_back(
      {"boundary.cached.position_1025.logits.post_softcap",
       model::kVocabSize});
  std::sort(specs.begin(), specs.end(),
            [](const CaptureSpec& left, const CaptureSpec& right) {
              return left.name < right.name;
            });
  return specs;
}

std::string hybrid_boundary_rows_capture_name(std::string_view field) {
  constexpr std::string_view prefix =
      "boundary.hybrid.rows_1022_1025.";
  std::string result;
  result.reserve(prefix.size() + field.size());
  result.append(prefix);
  result.append(field);
  return result;
}

std::string hybrid_boundary_layer_capture_name(std::uint32_t layer,
                                               std::string_view field) {
  char buffer[160];
  const int length = std::snprintf(
      buffer, sizeof(buffer),
      "boundary.hybrid.rows_1022_1025.layer.%02u.%.*s", layer,
      static_cast<int>(field.size()), field.data());
  if (length < 0 || static_cast<std::size_t>(length) >= sizeof(buffer)) {
    fail("hybrid boundary capture name", "formatted name exceeds fixed buffer");
  }
  return std::string(buffer, static_cast<std::size_t>(length));
}

void add_hybrid_boundary_deep_capture_specs(std::vector<CaptureSpec>* specs,
                                            std::uint32_t layer) {
  const bool global = model::is_global_layer(layer);
  const std::size_t head_size =
      global ? model::kGlobalHeadSize : model::kLocalHeadSize;
  const std::size_t kv_heads =
      global ? model::kGlobalKvHeadCount : model::kLocalKvHeadCount;
  const std::size_t q_width = model::kQueryHeadCount * head_size;
  const std::size_t kv_width = kv_heads * head_size;
  const auto add = [&](std::string_view field, std::size_t row_elements) {
    specs->push_back({hybrid_boundary_layer_capture_name(layer, field),
                      kBoundaryCaptureRows * row_elements});
  };
  add("q_raw", q_width);
  add("q_norm", q_width);
  add("q_rope", q_width);
  add("k_raw", kv_width);
  add("k_norm", kv_width);
  add("k_rope", kv_width);
  add("v_raw", kv_width);
  add("v_norm", kv_width);
  add("attention_probabilities",
      model::kQueryHeadCount * kBoundaryPrefixTokens);
  add("attention_context", q_width);
  add("attention_output", model::kHiddenSize);
}

std::vector<CaptureSpec> make_hybrid_boundary_capture_specs() {
  std::vector<CaptureSpec> specs;
  specs.reserve(kBoundaryCaptureCount);
  const auto add_rows = [&](std::string_view field,
                            std::size_t row_elements) {
    specs.push_back({hybrid_boundary_rows_capture_name(field),
                     kBoundaryCaptureRows * row_elements});
  };
  add_rows("embedding", model::kHiddenSize);
  add_rows("final_norm", model::kHiddenSize);
  add_rows("rotary.sliding_attention.cos", model::kLocalHeadSize);
  add_rows("rotary.sliding_attention.sin", model::kLocalHeadSize);
  add_rows("rotary.full_attention.cos", model::kGlobalHeadSize);
  add_rows("rotary.full_attention.sin", model::kGlobalHeadSize);
  for (std::uint32_t layer = 0; layer < model::kLayerCount; ++layer) {
    specs.push_back({hybrid_boundary_layer_capture_name(layer, "output"),
                     kBoundaryCaptureRows * model::kHiddenSize});
  }
  add_hybrid_boundary_deep_capture_specs(&specs, 0);
  add_hybrid_boundary_deep_capture_specs(&specs, 5);
  specs.push_back(
      {"boundary.hybrid.position_1025.logits.pre_softcap",
       model::kVocabSize});
  specs.push_back(
      {"boundary.hybrid.position_1025.logits.post_softcap",
       model::kVocabSize});
  std::sort(specs.begin(), specs.end(),
            [](const CaptureSpec& left, const CaptureSpec& right) {
              return left.name < right.name;
            });
  return specs;
}

class BoundaryCapturePlan {
 public:
  explicit BoundaryCapturePlan(std::filesystem::path directory)
      : BoundaryCapturePlan(std::move(directory),
                            make_boundary_capture_specs()) {}

  BoundaryCapturePlan(std::filesystem::path directory,
                      std::vector<CaptureSpec> specs)
      : directory_(std::move(directory)) {
    std::error_code error;
    if (std::filesystem::exists(directory_, error) || error) {
      fail("boundary capture directory",
           error ? error.message() : "path already exists (overwrite refused)");
    }
    if (specs.size() != kBoundaryCaptureCount) {
      fail("boundary capture plan", "wrong capture count");
    }
    records_.reserve(specs.size());
    for (const CaptureSpec& spec : specs) {
      records_.push_back({spec.name, spec.elements, host_bytes_, false, 0});
      host_bytes_ += spec.elements * sizeof(BFloat16);
    }
    check_cuda(cudaMallocHost(&host_arena_, host_bytes_),
               "cudaMallocHost boundary captures");
  }

  ~BoundaryCapturePlan() {
    if (host_arena_ != nullptr) {
      cudaFreeHost(host_arena_);
    }
  }
  BoundaryCapturePlan(const BoundaryCapturePlan&) = delete;
  BoundaryCapturePlan& operator=(const BoundaryCapturePlan&) = delete;

  void copy_row(std::string_view name, std::uint32_t absolute_position,
                const BFloat16* source, std::size_t row_elements,
                cudaStream_t stream = nullptr) {
    CaptureRecord& record = find(name);
    if (absolute_position < kBoundaryCaptureFirstPosition ||
        absolute_position > kBoundaryDecisionPosition ||
        record.elements != kBoundaryCaptureRows * row_elements ||
        record.captured) {
      fail("boundary capture row", std::string(name) + " has an invalid row");
    }
    const std::uint32_t row =
        absolute_position - kBoundaryCaptureFirstPosition;
    const std::uint8_t bit = static_cast<std::uint8_t>(1U << row);
    if ((record.partial_mask & bit) != 0) {
      fail("boundary capture row", std::string(name) + " was produced twice");
    }
    auto* destination = static_cast<std::uint8_t*>(host_arena_) +
                        record.byte_offset +
                        row * row_elements * sizeof(BFloat16);
    check_cuda(cudaMemcpyAsync(destination, source,
                               row_elements * sizeof(BFloat16),
                               cudaMemcpyDeviceToHost, stream),
               "boundary capture row device-to-host copy");
    record.partial_mask |= bit;
    record.captured = record.partial_mask == 0x0f;
  }

  void copy_device(std::string_view name, const BFloat16* source,
                   cudaStream_t stream = nullptr) {
    CaptureRecord& record = find(name);
    if (record.captured || record.partial_mask != 0) {
      fail("boundary capture", std::string(name) + " was produced twice");
    }
    auto* destination =
        static_cast<std::uint8_t*>(host_arena_) + record.byte_offset;
    check_cuda(cudaMemcpyAsync(destination, source,
                               record.elements * sizeof(BFloat16),
                               cudaMemcpyDeviceToHost, stream),
               "boundary capture device-to-host copy");
    record.captured = true;
  }

  void write(std::uint32_t token) {
    check_cuda(cudaDeviceSynchronize(), "synchronize boundary captures");
    for (const CaptureRecord& record : records_) {
      if (!record.captured) {
        fail("boundary capture", record.name + " was not produced");
      }
    }
    std::error_code error;
    if (!std::filesystem::create_directory(directory_, error)) {
      fail("boundary capture directory",
           error ? error.message() : "path already exists (overwrite refused)");
    }
    for (const CaptureRecord& record : records_) {
      const auto* source = static_cast<const std::uint8_t*>(host_arena_) +
                           record.byte_offset;
      write_exclusive(directory_ / (record.name + ".bf16"), source,
                      record.elements * sizeof(BFloat16));
    }
    const std::array<std::uint8_t, 4> encoded_token{{
        static_cast<std::uint8_t>(token),
        static_cast<std::uint8_t>(token >> 8),
        static_cast<std::uint8_t>(token >> 16),
        static_cast<std::uint8_t>(token >> 24),
    }};
    write_exclusive(directory_ / "argmax.u32", encoded_token.data(),
                    encoded_token.size());
  }

  [[nodiscard]] std::size_t size() const { return records_.size(); }
  [[nodiscard]] std::size_t host_bytes() const { return host_bytes_; }
  [[nodiscard]] const std::filesystem::path& directory() const {
    return directory_;
  }

 private:
  CaptureRecord& find(std::string_view name) {
    for (CaptureRecord& record : records_) {
      if (record.name == name) {
        return record;
      }
    }
    fail("boundary capture",
         std::string(name) + " is not in the native boundary inventory");
  }

  std::filesystem::path directory_;
  std::vector<CaptureRecord> records_;
  void* host_arena_{nullptr};
  std::size_t host_bytes_{};
};


class KvCaches {
 public:
  KvCaches()
      : local_key_(kLocalCacheBytesPerKind),
        local_value_(kLocalCacheBytesPerKind),
        global_key_(kGlobalCacheBytesPerKind),
        global_value_(kGlobalCacheBytesPerKind) {
    const std::array<std::uintptr_t, 4> addresses{{
        reinterpret_cast<std::uintptr_t>(local_key_.data()),
        reinterpret_cast<std::uintptr_t>(local_value_.data()),
        reinterpret_cast<std::uintptr_t>(global_key_.data()),
        reinterpret_cast<std::uintptr_t>(global_value_.data()),
    }};
    for (std::size_t left = 0; left < addresses.size(); ++left) {
      for (std::size_t right = left + 1; right < addresses.size(); ++right) {
        if (addresses[left] == addresses[right]) {
          fail("KV caches", "distinct cache arenas alias");
        }
      }
    }
  }

  [[nodiscard]] LayerCacheView layer(std::uint32_t layer_index) const {
    if (layer_index >= model::kLayerCount) {
      fail("KV cache view", "layer is outside the model");
    }
    if (model::is_global_layer(layer_index)) {
      const std::size_t slot = layer_index / 6;
      const std::size_t elements_per_layer =
          static_cast<std::size_t>(model::kGlobalKvHeadCount) *
          kGlobalCacheCapacity * model::kGlobalHeadSize;
      return {
          static_cast<BFloat16*>(global_key_.data()) +
              slot * elements_per_layer,
          static_cast<BFloat16*>(global_value_.data()) +
              slot * elements_per_layer,
          kGlobalCacheCapacity,
      };
    }
    const std::size_t slot = layer_index - layer_index / 6;
    const std::size_t elements_per_layer =
        static_cast<std::size_t>(model::kLocalKvHeadCount) *
        kLocalCacheCapacity * model::kLocalHeadSize;
    return {
        static_cast<BFloat16*>(local_key_.data()) + slot * elements_per_layer,
        static_cast<BFloat16*>(local_value_.data()) + slot * elements_per_layer,
        kLocalCacheCapacity,
    };
  }

  [[nodiscard]] std::size_t local_bytes() const {
    return local_key_.size() + local_value_.size();
  }
  [[nodiscard]] std::size_t global_bytes() const {
    return global_key_.size() + global_value_.size();
  }

 private:
  DeviceAllocation local_key_;
  DeviceAllocation local_value_;
  DeviceAllocation global_key_;
  DeviceAllocation global_value_;
};

class ShortDecodeKvCaches {
 public:
  ShortDecodeKvCaches()
      : local_key_(kLocalCacheBytesPerKind),
        local_value_(kLocalCacheBytesPerKind),
        global_key_(kShortDecodeGlobalCacheBytesPerKind),
        global_value_(kShortDecodeGlobalCacheBytesPerKind) {}

  [[nodiscard]] LayerCacheView layer(std::uint32_t layer_index) const {
    if (layer_index >= model::kLayerCount) {
      fail("short-decode KV cache view", "layer is outside the model");
    }
    if (model::is_global_layer(layer_index)) {
      const std::size_t slot = layer_index / 6;
      const std::size_t elements_per_layer =
          static_cast<std::size_t>(model::kGlobalKvHeadCount) *
          kShortDecodeGlobalCacheCapacity * model::kGlobalHeadSize;
      return {
          static_cast<BFloat16*>(global_key_.data()) +
              slot * elements_per_layer,
          static_cast<BFloat16*>(global_value_.data()) +
              slot * elements_per_layer,
          kShortDecodeGlobalCacheCapacity,
      };
    }
    const std::size_t slot = layer_index - layer_index / 6;
    const std::size_t elements_per_layer =
        static_cast<std::size_t>(model::kLocalKvHeadCount) *
        kLocalCacheCapacity * model::kLocalHeadSize;
    return {
        static_cast<BFloat16*>(local_key_.data()) + slot * elements_per_layer,
        static_cast<BFloat16*>(local_value_.data()) + slot * elements_per_layer,
        kLocalCacheCapacity,
    };
  }

  [[nodiscard]] std::size_t local_bytes() const {
    return local_key_.size() + local_value_.size();
  }
  [[nodiscard]] std::size_t global_bytes() const {
    return global_key_.size() + global_value_.size();
  }

 private:
  DeviceAllocation local_key_;
  DeviceAllocation local_value_;
  DeviceAllocation global_key_;
  DeviceAllocation global_value_;
};

class BoundaryKvCaches {
 public:
  BoundaryKvCaches()
      : local_key_(kLocalCacheBytesPerKind),
        local_value_(kLocalCacheBytesPerKind),
        global_key_(kBoundaryGlobalCacheBytesPerKind),
        global_value_(kBoundaryGlobalCacheBytesPerKind) {}

  [[nodiscard]] LayerCacheView layer(std::uint32_t layer_index) const {
    if (layer_index >= model::kLayerCount) {
      fail("boundary KV cache view", "layer is outside the model");
    }
    if (model::is_global_layer(layer_index)) {
      const std::size_t slot = layer_index / 6;
      const std::size_t elements_per_layer =
          static_cast<std::size_t>(model::kGlobalKvHeadCount) *
          kBoundaryGlobalCacheCapacity * model::kGlobalHeadSize;
      return {
          static_cast<BFloat16*>(global_key_.data()) +
              slot * elements_per_layer,
          static_cast<BFloat16*>(global_value_.data()) +
              slot * elements_per_layer,
          kBoundaryGlobalCacheCapacity,
      };
    }
    const std::size_t slot = layer_index - layer_index / 6;
    const std::size_t elements_per_layer =
        static_cast<std::size_t>(model::kLocalKvHeadCount) *
        kLocalCacheCapacity * model::kLocalHeadSize;
    return {
        static_cast<BFloat16*>(local_key_.data()) + slot * elements_per_layer,
        static_cast<BFloat16*>(local_value_.data()) + slot * elements_per_layer,
        kLocalCacheCapacity,
    };
  }

  [[nodiscard]] std::size_t local_bytes() const {
    return local_key_.size() + local_value_.size();
  }
  [[nodiscard]] std::size_t global_bytes() const {
    return global_key_.size() + global_value_.size();
  }

 private:
  DeviceAllocation local_key_;
  DeviceAllocation local_value_;
  DeviceAllocation global_key_;
  DeviceAllocation global_value_;
};

class GraphDecodeKvCaches {
 public:
  GraphDecodeKvCaches()
      : local_key_(kLocalCacheBytesPerKind),
        local_value_(kLocalCacheBytesPerKind),
        global_key_(kGraphGlobalCacheBytesPerKind),
        global_value_(kGraphGlobalCacheBytesPerKind) {
    const std::array<const void*, 4> arenas{{
        local_key_.data(),
        local_value_.data(),
        global_key_.data(),
        global_value_.data(),
    }};
    for (std::size_t left = 0; left < arenas.size(); ++left) {
      for (std::size_t right = left + 1; right < arenas.size(); ++right) {
        if (arenas[left] == arenas[right]) {
          fail("graph-decode KV caches", "distinct cache arenas alias");
        }
      }
    }
  }

  [[nodiscard]] LayerCacheView layer(std::uint32_t layer_index) const {
    if (layer_index >= model::kLayerCount) {
      fail("graph-decode KV cache view", "layer is outside the model");
    }
    if (model::is_global_layer(layer_index)) {
      const std::size_t slot = layer_index / 6;
      const std::size_t elements_per_layer =
          static_cast<std::size_t>(model::kGlobalKvHeadCount) *
          kGraphGlobalCacheCapacity * model::kGlobalHeadSize;
      return {
          static_cast<BFloat16*>(global_key_.data()) +
              slot * elements_per_layer,
          static_cast<BFloat16*>(global_value_.data()) +
              slot * elements_per_layer,
          kGraphGlobalCacheCapacity,
      };
    }
    const std::size_t slot = layer_index - layer_index / 6;
    const std::size_t elements_per_layer =
        static_cast<std::size_t>(model::kLocalKvHeadCount) *
        kLocalCacheCapacity * model::kLocalHeadSize;
    return {
        static_cast<BFloat16*>(local_key_.data()) + slot * elements_per_layer,
        static_cast<BFloat16*>(local_value_.data()) +
            slot * elements_per_layer,
        kLocalCacheCapacity,
    };
  }

  [[nodiscard]] std::size_t local_bytes() const {
    return local_key_.size() + local_value_.size();
  }
  [[nodiscard]] std::size_t global_bytes() const {
    return global_key_.size() + global_value_.size();
  }

 private:
  DeviceAllocation local_key_;
  DeviceAllocation local_value_;
  DeviceAllocation global_key_;
  DeviceAllocation global_value_;
};


class PairEngine {
 public:
  explicit PairEngine(const WeightArena& weights)
      : weights_(weights),
        scratch_(ScratchLayout::kBytes),
        local_q_(kTokens, model::kHiddenSize,
                 model::kQueryHeadCount * model::kLocalHeadSize),
        local_kv_(kTokens, model::kHiddenSize,
                  model::kLocalKvHeadCount * model::kLocalHeadSize),
        global_q_(kTokens, model::kHiddenSize,
                  model::kQueryHeadCount * model::kGlobalHeadSize),
        global_kv_(kTokens, model::kHiddenSize,
                   model::kGlobalKvHeadCount * model::kGlobalHeadSize),
        local_o_(kTokens,
                 model::kQueryHeadCount * model::kLocalHeadSize,
                 model::kHiddenSize),
        global_o_(kTokens,
                  model::kQueryHeadCount * model::kGlobalHeadSize,
                  model::kHiddenSize),
        hidden_to_mlp_(kTokens, model::kHiddenSize, model::kMlpSize),
        mlp_to_hidden_(kTokens, model::kMlpSize, model::kHiddenSize),
        lm_head_(1, model::kHiddenSize, model::kVocabSize) {
    for (std::size_t layer = 0; layer < model::kLayerCount; ++layer) {
      const LayerWeightIds& ids = kWeightIds[layer];
      layers_[layer] = {
          weights.pointer(ids.input_norm),
          weights.pointer(ids.q_proj),
          weights.pointer(ids.k_proj),
          model::is_global_layer(layer) ? nullptr : weights.pointer(ids.v_proj),
          weights.pointer(ids.q_norm),
          weights.pointer(ids.k_norm),
          weights.pointer(ids.o_proj),
          weights.pointer(ids.post_attention_norm),
          weights.pointer(ids.pre_feedforward_norm),
          weights.pointer(ids.gate_proj),
          weights.pointer(ids.up_proj),
          weights.pointer(ids.down_proj),
          weights.pointer(ids.post_feedforward_norm),
          weights.pointer(ids.layer_scalar),
      };
    }
  }

  [[nodiscard]] std::size_t scratch_bytes() const { return scratch_.size(); }

  std::uint32_t run(CapturePlan* captures) {
    BFloat16* const h0 = at<BFloat16>(ScratchLayout::kH0);
    BFloat16* const h1 = at<BFloat16>(ScratchLayout::kH1);
    BFloat16* const h2 = at<BFloat16>(ScratchLayout::kH2);
    BFloat16* const q_raw = at<BFloat16>(ScratchLayout::kQueryRaw);
    BFloat16* const q_norm = at<BFloat16>(ScratchLayout::kQueryNorm);
    BFloat16* const q_rope = at<BFloat16>(ScratchLayout::kQueryRope);
    BFloat16* const k_raw = at<BFloat16>(ScratchLayout::kKeyRaw);
    BFloat16* const k_norm = at<BFloat16>(ScratchLayout::kKeyNorm);
    BFloat16* const k_rope = at<BFloat16>(ScratchLayout::kKeyRope);
    BFloat16* const v_raw = at<BFloat16>(ScratchLayout::kValueRaw);
    BFloat16* const v_norm = at<BFloat16>(ScratchLayout::kValueNorm);
    BFloat16* const context = at<BFloat16>(ScratchLayout::kContext);
    BFloat16* const gate = at<BFloat16>(ScratchLayout::kGate);
    BFloat16* const up = at<BFloat16>(ScratchLayout::kUp);
    BFloat16* const product = at<BFloat16>(ScratchLayout::kProduct);
    BFloat16* const logits = at<BFloat16>(ScratchLayout::kLogits);
    BFloat16* const capped_logits =
        at<BFloat16>(ScratchLayout::kCappedLogits);
    BFloat16* const local_cos = at<BFloat16>(ScratchLayout::kLocalCos);
    BFloat16* const local_sin = at<BFloat16>(ScratchLayout::kLocalSin);
    BFloat16* const global_cos = at<BFloat16>(ScratchLayout::kGlobalCos);
    BFloat16* const global_sin = at<BFloat16>(ScratchLayout::kGlobalSin);
    BFloat16* const probabilities =
        at<BFloat16>(ScratchLayout::kProbabilities);
    std::uint32_t* const argmax = at<std::uint32_t>(ScratchLayout::kArgmax);

    primitives::generate_rope_factors_m2(local_cos, local_sin, global_cos,
                                         global_sin);
    capture(captures, "prefill.rotary.sliding_attention.cos", local_cos);
    capture(captures, "prefill.rotary.sliding_attention.sin", local_sin);
    capture(captures, "prefill.rotary.full_attention.cos", global_cos);
    capture(captures, "prefill.rotary.full_attention.sin", global_sin);

    primitives::embedding_lookup(
        weights_.pointer(model::kEmbeddingPhysicalId), kInputTokens[0], h0);
    primitives::embedding_lookup(
        weights_.pointer(model::kEmbeddingPhysicalId), kInputTokens[1],
        h0 + model::kHiddenSize);
    capture(captures, "prefill.embedding", h0);

    for (std::uint32_t layer = 0; layer < model::kLayerCount; ++layer) {
      const bool global = model::is_global_layer(layer);
      const model::AttentionKind kind =
          global ? model::AttentionKind::global : model::AttentionKind::local;
      const std::uint32_t head_size =
          global ? model::kGlobalHeadSize : model::kLocalHeadSize;
      const std::uint32_t kv_heads =
          global ? model::kGlobalKvHeadCount : model::kLocalKvHeadCount;
      const LayerWeights& weight = layers_[layer];
      const bool deep = layer == 0 || layer == 5;
      BFloat16* const cos = global ? global_cos : local_cos;
      BFloat16* const sin = global ? global_sin : local_sin;

      if (deep) {
        capture_layer(captures, layer, "block_input", h0);
        capture_layer(captures, layer, "layer_scalar", weight.layer_scalar);
      }

      primitives::rms_norm(h0, weight.input_norm, h1, kTokens,
                           model::kHiddenSize);
      if (deep) {
        capture_layer(captures, layer, "input_norm", h1);
      }

      const LinearPlan& q_plan = global ? global_q_ : local_q_;
      const LinearPlan& kv_plan = global ? global_kv_ : local_kv_;
      q_plan.run(handle_.get(), h1, weight.q_proj, q_raw);
      kv_plan.run(handle_.get(), h1, weight.k_proj, k_raw);
      if (!global) {
        kv_plan.run(handle_.get(), h1, weight.v_proj, v_raw);
      }
      if (deep) {
        capture_layer(captures, layer, "q_raw", q_raw);
        capture_layer(captures, layer, "k_raw", k_raw);
        capture_layer(captures, layer, "v_raw", global ? k_raw : v_raw);
      }

      primitives::rms_norm(q_raw, weight.q_norm, q_norm,
                           kTokens * model::kQueryHeadCount, head_size);
      primitives::rms_norm(k_raw, weight.k_norm, k_norm,
                           kTokens * kv_heads, head_size);
      primitives::rms_norm_unscaled(global ? k_raw : v_raw, v_norm,
                                    kTokens * kv_heads, head_size);
      if (deep) {
        capture_layer(captures, layer, "q_norm", q_norm);
        capture_layer(captures, layer, "k_norm", k_norm);
        capture_layer(captures, layer, "v_norm", v_norm);
      }

      primitives::apply_rope_transpose_m2(
          q_norm, cos, sin, q_rope, model::kQueryHeadCount, kind);
      primitives::apply_rope_transpose_m2(k_norm, cos, sin, k_rope,
                                          kv_heads, kind);
      if (deep) {
        capture_layer(captures, layer, "q_rope", q_rope);
        capture_layer(captures, layer, "k_rope", k_rope);
      }

      primitives::causal_gqa_attention_m2(q_rope, k_rope, v_norm,
                                          probabilities, context, kind);
      if (deep) {
        capture_layer(captures, layer, "attention_probabilities",
                      probabilities);
        capture_layer(captures, layer, "attention_context", context);
      }

      const LinearPlan& o_plan = global ? global_o_ : local_o_;
      o_plan.run(handle_.get(), context, weight.o_proj, h2);
      if (deep) {
        capture_layer(captures, layer, "attention_output", h2);
      }

      primitives::rms_norm(h2, weight.post_attention_norm, h1, kTokens,
                           model::kHiddenSize);
      if (deep) {
        capture_layer(captures, layer, "post_attention_norm", h1);
      }
      primitives::residual_add(h0, h1, h2,
                               kTokens * model::kHiddenSize);
      if (deep) {
        capture_layer(captures, layer, "post_attention_residual", h2);
      }

      primitives::rms_norm(h2, weight.pre_feedforward_norm, h1, kTokens,
                           model::kHiddenSize);
      if (deep) {
        capture_layer(captures, layer, "pre_feedforward_norm", h1);
      }
      hidden_to_mlp_.run(handle_.get(), h1, weight.gate_proj, gate);
      hidden_to_mlp_.run(handle_.get(), h1, weight.up_proj, up);
      if (deep) {
        capture_layer(captures, layer, "mlp_gate", gate);
        capture_layer(captures, layer, "mlp_up", up);
      }
      primitives::gelu_tanh_multiply(gate, up, product,
                                     kTokens * model::kMlpSize);
      if (deep) {
        capture_layer(captures, layer, "mlp_product", product);
      }
      mlp_to_hidden_.run(handle_.get(), product, weight.down_proj, h0);
      if (deep) {
        capture_layer(captures, layer, "mlp_down", h0);
      }
      primitives::rms_norm(h0, weight.post_feedforward_norm, h1, kTokens,
                           model::kHiddenSize);
      if (deep) {
        capture_layer(captures, layer, "post_feedforward_norm", h1);
      }
      primitives::residual_add(h2, h1, h0,
                               kTokens * model::kHiddenSize);
      if (deep) {
        capture_layer(captures, layer, "pre_scalar_output", h0);
      }
      primitives::trained_scalar(h0, weight.layer_scalar,
                                 kTokens * model::kHiddenSize);
      capture_layer(captures, layer, "output", h0);
    }

    primitives::rms_norm(
        h0, weights_.pointer(model::kFinalNormPhysicalId), h1, kTokens,
        model::kHiddenSize);
    capture(captures, "prefill.final_norm", h1);
    lm_head_.run(handle_.get(), h1 + model::kHiddenSize,
                 weights_.pointer(model::kLmHeadLogicalId), logits);
    capture(captures, "prefill.logits.pre_softcap", logits);
    primitives::softcap_and_argmax(logits, capped_logits, argmax,
                                   model::kVocabSize, 30.0F);
    capture(captures, "prefill.logits.post_softcap", capped_logits);

    std::uint32_t result = 0;
    check_cuda(cudaMemcpy(&result, argmax, sizeof(result),
                          cudaMemcpyDeviceToHost),
               "copy pair argmax");
    return result;
  }

 private:
  template <typename T>
  T* at(std::size_t byte_offset) const {
    return reinterpret_cast<T*>(static_cast<std::uint8_t*>(scratch_.data()) +
                                byte_offset);
  }

  static void capture(CapturePlan* captures, std::string_view name,
                      const BFloat16* values) {
    if (captures != nullptr) {
      captures->copy_device(name, values);
    }
  }

  static void capture_layer(CapturePlan* captures, std::uint32_t layer,
                            std::string_view field,
                            const BFloat16* values) {
    if (captures == nullptr) {
      return;
    }
    char name[96];
    const int length = std::snprintf(
        name, sizeof(name), "prefill.layer.%02u.%.*s", layer,
        static_cast<int>(field.size()), field.data());
    if (length < 0 || static_cast<std::size_t>(length) >= sizeof(name)) {
      fail("capture name", "formatted name exceeds fixed buffer");
    }
    captures->copy_device(
        std::string_view(name, static_cast<std::size_t>(length)), values);
  }

  const WeightArena& weights_;
  DeviceAllocation scratch_;
  LtHandle handle_;
  LinearPlan local_q_;
  LinearPlan local_kv_;
  LinearPlan global_q_;
  LinearPlan global_kv_;
  LinearPlan local_o_;
  LinearPlan global_o_;
  LinearPlan hidden_to_mlp_;
  LinearPlan mlp_to_hidden_;
  LinearPlan lm_head_;
  std::array<LayerWeights, model::kLayerCount> layers_{};
};

class CachedPairEngine {
 public:
  explicit CachedPairEngine(const WeightArena& weights)
      : weights_(weights),
        scratch_(CachedScratchLayout::kBytes),
        local_q_(1, model::kHiddenSize,
                 model::kQueryHeadCount * model::kLocalHeadSize),
        local_kv_(1, model::kHiddenSize,
                  model::kLocalKvHeadCount * model::kLocalHeadSize),
        global_q_(1, model::kHiddenSize,
                  model::kQueryHeadCount * model::kGlobalHeadSize),
        global_kv_(1, model::kHiddenSize,
                   model::kGlobalKvHeadCount * model::kGlobalHeadSize),
        local_o_(1, model::kQueryHeadCount * model::kLocalHeadSize,
                 model::kHiddenSize),
        global_o_(1, model::kQueryHeadCount * model::kGlobalHeadSize,
                  model::kHiddenSize),
        hidden_to_mlp_(1, model::kHiddenSize, model::kMlpSize),
        mlp_to_hidden_(1, model::kMlpSize, model::kHiddenSize),
        lm_head_(1, model::kHiddenSize, model::kVocabSize) {
    for (std::size_t layer = 0; layer < model::kLayerCount; ++layer) {
      const LayerWeightIds& ids = kWeightIds[layer];
      layers_[layer] = {
          weights.pointer(ids.input_norm),
          weights.pointer(ids.q_proj),
          weights.pointer(ids.k_proj),
          model::is_global_layer(layer) ? nullptr : weights.pointer(ids.v_proj),
          weights.pointer(ids.q_norm),
          weights.pointer(ids.k_norm),
          weights.pointer(ids.o_proj),
          weights.pointer(ids.post_attention_norm),
          weights.pointer(ids.pre_feedforward_norm),
          weights.pointer(ids.gate_proj),
          weights.pointer(ids.up_proj),
          weights.pointer(ids.down_proj),
          weights.pointer(ids.post_feedforward_norm),
          weights.pointer(ids.layer_scalar),
      };
    }
  }

  [[nodiscard]] std::size_t scratch_bytes() const { return scratch_.size(); }
  [[nodiscard]] std::size_t local_cache_bytes() const {
    return caches_.local_bytes();
  }
  [[nodiscard]] std::size_t global_cache_bytes() const {
    return caches_.global_bytes();
  }

  std::uint32_t run(CapturePlan* captures) {
    BFloat16* const h0 = at<BFloat16>(CachedScratchLayout::kH0);
    BFloat16* const h1 = at<BFloat16>(CachedScratchLayout::kH1);
    BFloat16* const h2 = at<BFloat16>(CachedScratchLayout::kH2);
    BFloat16* const q_raw = at<BFloat16>(CachedScratchLayout::kQueryRaw);
    BFloat16* const q_norm = at<BFloat16>(CachedScratchLayout::kQueryNorm);
    BFloat16* const q_rope = at<BFloat16>(CachedScratchLayout::kQueryRope);
    BFloat16* const k_raw = at<BFloat16>(CachedScratchLayout::kKeyRaw);
    BFloat16* const k_norm = at<BFloat16>(CachedScratchLayout::kKeyNorm);
    BFloat16* const k_rope = at<BFloat16>(CachedScratchLayout::kKeyRope);
    BFloat16* const v_raw = at<BFloat16>(CachedScratchLayout::kValueRaw);
    BFloat16* const v_norm = at<BFloat16>(CachedScratchLayout::kValueNorm);
    BFloat16* const context = at<BFloat16>(CachedScratchLayout::kContext);
    BFloat16* const gate = at<BFloat16>(CachedScratchLayout::kGate);
    BFloat16* const up = at<BFloat16>(CachedScratchLayout::kUp);
    BFloat16* const product = at<BFloat16>(CachedScratchLayout::kProduct);
    BFloat16* const logits = at<BFloat16>(CachedScratchLayout::kLogits);
    BFloat16* const capped_logits =
        at<BFloat16>(CachedScratchLayout::kCappedLogits);
    BFloat16* const local_cos = at<BFloat16>(CachedScratchLayout::kLocalCos);
    BFloat16* const local_sin = at<BFloat16>(CachedScratchLayout::kLocalSin);
    BFloat16* const global_cos = at<BFloat16>(CachedScratchLayout::kGlobalCos);
    BFloat16* const global_sin = at<BFloat16>(CachedScratchLayout::kGlobalSin);
    BFloat16* const probabilities =
        at<BFloat16>(CachedScratchLayout::kProbabilities);
    std::uint32_t* const argmax =
        at<std::uint32_t>(CachedScratchLayout::kArgmax);

    primitives::generate_rope_factors_m2(local_cos, local_sin, global_cos,
                                         global_sin);
    capture(captures, "prefill.rotary.sliding_attention.cos", local_cos);
    capture(captures, "prefill.rotary.sliding_attention.sin", local_sin);
    capture(captures, "prefill.rotary.full_attention.cos", global_cos);
    capture(captures, "prefill.rotary.full_attention.sin", global_sin);

    for (std::uint32_t position = 0; position < kTokens; ++position) {
      primitives::embedding_lookup(
          weights_.pointer(model::kEmbeddingPhysicalId),
          kInputTokens[position], h0);
      capture_token(captures, "prefill.embedding", position, h0,
                    model::kHiddenSize);

      for (std::uint32_t layer = 0; layer < model::kLayerCount; ++layer) {
        const bool global = model::is_global_layer(layer);
        const model::AttentionKind kind =
            global ? model::AttentionKind::global
                   : model::AttentionKind::local;
        const std::uint32_t head_size =
            global ? model::kGlobalHeadSize : model::kLocalHeadSize;
        const std::uint32_t kv_heads =
            global ? model::kGlobalKvHeadCount : model::kLocalKvHeadCount;
        const std::uint32_t q_width = model::kQueryHeadCount * head_size;
        const std::uint32_t kv_width = kv_heads * head_size;
        const LayerWeights& weight = layers_[layer];
        const LayerCacheView cache = caches_.layer(layer);
        const bool deep = layer == 0 || layer == 5;
        const BFloat16* const cos =
            (global ? global_cos : local_cos) + position * head_size;
        const BFloat16* const sin =
            (global ? global_sin : local_sin) + position * head_size;

        if (deep) {
          capture_layer_token(captures, layer, "block_input", position, h0,
                              model::kHiddenSize);
          if (position == 0) {
            capture_layer(captures, layer, "layer_scalar",
                          weight.layer_scalar);
          }
        }

        primitives::rms_norm(h0, weight.input_norm, h1, 1,
                             model::kHiddenSize);
        if (deep) {
          capture_layer_token(captures, layer, "input_norm", position, h1,
                              model::kHiddenSize);
        }

        const LinearPlan& q_plan = global ? global_q_ : local_q_;
        const LinearPlan& kv_plan = global ? global_kv_ : local_kv_;
        q_plan.run(handle_.get(), h1, weight.q_proj, q_raw);
        kv_plan.run(handle_.get(), h1, weight.k_proj, k_raw);
        if (!global) {
          kv_plan.run(handle_.get(), h1, weight.v_proj, v_raw);
        }
        if (deep) {
          capture_layer_token(captures, layer, "q_raw", position, q_raw,
                              q_width);
          capture_layer_token(captures, layer, "k_raw", position, k_raw,
                              kv_width);
          capture_layer_token(captures, layer, "v_raw", position,
                              global ? k_raw : v_raw, kv_width);
        }

        primitives::rms_norm(q_raw, weight.q_norm, q_norm,
                             model::kQueryHeadCount, head_size);
        primitives::rms_norm(k_raw, weight.k_norm, k_norm, kv_heads,
                             head_size);
        primitives::rms_norm_unscaled(global ? k_raw : v_raw, v_norm,
                                      kv_heads, head_size);
        if (deep) {
          capture_layer_token(captures, layer, "q_norm", position, q_norm,
                              q_width);
          capture_layer_token(captures, layer, "k_norm", position, k_norm,
                              kv_width);
          capture_layer_token(captures, layer, "v_norm", position, v_norm,
                              kv_width);
        }

        primitives::apply_rope_m1(q_norm, cos, sin, q_rope,
                                  model::kQueryHeadCount, kind);
        primitives::apply_rope_m1(k_norm, cos, sin, k_rope, kv_heads, kind);
        if (deep) {
          capture_layer_head_token(captures, layer, "q_rope", position,
                                   q_rope, model::kQueryHeadCount, head_size);
          capture_layer_head_token(captures, layer, "k_rope", position,
                                   k_rope, kv_heads, head_size);
        }

        primitives::write_kv_cache_m1(
            k_rope, v_norm, cache.key, cache.value, position, cache.capacity,
            kind);
        primitives::causal_gqa_attention_cached_m1(
            q_rope, cache.key, cache.value, position, cache.capacity,
            probabilities, context, kind);
        if (deep) {
          capture_layer_head_token(captures, layer,
                                   "attention_probabilities", position,
                                   probabilities, model::kQueryHeadCount,
                                   kTokens);
          capture_layer_token(captures, layer, "attention_context", position,
                              context, q_width);
        }

        const LinearPlan& o_plan = global ? global_o_ : local_o_;
        o_plan.run(handle_.get(), context, weight.o_proj, h2);
        if (deep) {
          capture_layer_token(captures, layer, "attention_output", position,
                              h2, model::kHiddenSize);
        }

        primitives::rms_norm(h2, weight.post_attention_norm, h1, 1,
                             model::kHiddenSize);
        if (deep) {
          capture_layer_token(captures, layer, "post_attention_norm",
                              position, h1, model::kHiddenSize);
        }
        primitives::residual_add(h0, h1, h2, model::kHiddenSize);
        if (deep) {
          capture_layer_token(captures, layer, "post_attention_residual",
                              position, h2, model::kHiddenSize);
        }

        primitives::rms_norm(h2, weight.pre_feedforward_norm, h1, 1,
                             model::kHiddenSize);
        if (deep) {
          capture_layer_token(captures, layer, "pre_feedforward_norm",
                              position, h1, model::kHiddenSize);
        }
        hidden_to_mlp_.run(handle_.get(), h1, weight.gate_proj, gate);
        hidden_to_mlp_.run(handle_.get(), h1, weight.up_proj, up);
        if (deep) {
          capture_layer_token(captures, layer, "mlp_gate", position, gate,
                              model::kMlpSize);
          capture_layer_token(captures, layer, "mlp_up", position, up,
                              model::kMlpSize);
        }
        primitives::gelu_tanh_multiply(gate, up, product, model::kMlpSize);
        if (deep) {
          capture_layer_token(captures, layer, "mlp_product", position,
                              product, model::kMlpSize);
        }
        mlp_to_hidden_.run(handle_.get(), product, weight.down_proj, h0);
        if (deep) {
          capture_layer_token(captures, layer, "mlp_down", position, h0,
                              model::kHiddenSize);
        }
        primitives::rms_norm(h0, weight.post_feedforward_norm, h1, 1,
                             model::kHiddenSize);
        if (deep) {
          capture_layer_token(captures, layer, "post_feedforward_norm",
                              position, h1, model::kHiddenSize);
        }
        primitives::residual_add(h2, h1, h0, model::kHiddenSize);
        if (deep) {
          capture_layer_token(captures, layer, "pre_scalar_output", position,
                              h0, model::kHiddenSize);
        }
        primitives::trained_scalar(h0, weight.layer_scalar,
                                   model::kHiddenSize);
        capture_layer_token(captures, layer, "output", position, h0,
                            model::kHiddenSize);
      }

      primitives::rms_norm(
          h0, weights_.pointer(model::kFinalNormPhysicalId), h1, 1,
          model::kHiddenSize);
      capture_token(captures, "prefill.final_norm", position, h1,
                    model::kHiddenSize);

      if (position == kTokens - 1) {
        lm_head_.run(handle_.get(), h1,
                     weights_.pointer(model::kLmHeadLogicalId), logits);
        capture(captures, "prefill.logits.pre_softcap", logits);
        primitives::softcap_and_argmax(logits, capped_logits, argmax,
                                       model::kVocabSize, 30.0F);
        capture(captures, "prefill.logits.post_softcap", capped_logits);
      }
    }

    std::uint32_t result = 0;
    check_cuda(cudaMemcpy(&result, argmax, sizeof(result),
                          cudaMemcpyDeviceToHost),
               "copy cached-pair argmax");
    return result;
  }

 private:
  template <typename T>
  T* at(std::size_t byte_offset) const {
    return reinterpret_cast<T*>(static_cast<std::uint8_t*>(scratch_.data()) +
                                byte_offset);
  }

  static void capture(CapturePlan* captures, std::string_view name,
                      const BFloat16* values) {
    if (captures != nullptr) {
      captures->copy_device(name, values);
    }
  }

  static void capture_token(CapturePlan* captures, std::string_view name,
                            std::uint32_t position, const BFloat16* values,
                            std::size_t elements) {
    if (captures != nullptr) {
      captures->copy_token(name, position, values, elements);
    }
  }

  static void capture_layer(CapturePlan* captures, std::uint32_t layer,
                            std::string_view field,
                            const BFloat16* values) {
    if (captures == nullptr) {
      return;
    }
    char name[96];
    const int length = std::snprintf(
        name, sizeof(name), "prefill.layer.%02u.%.*s", layer,
        static_cast<int>(field.size()), field.data());
    if (length < 0 || static_cast<std::size_t>(length) >= sizeof(name)) {
      fail("capture name", "formatted name exceeds fixed buffer");
    }
    captures->copy_device(
        std::string_view(name, static_cast<std::size_t>(length)), values);
  }

  static void capture_layer_token(CapturePlan* captures, std::uint32_t layer,
                                  std::string_view field,
                                  std::uint32_t position,
                                  const BFloat16* values,
                                  std::size_t elements) {
    if (captures == nullptr) {
      return;
    }
    char name[96];
    const int length = std::snprintf(
        name, sizeof(name), "prefill.layer.%02u.%.*s", layer,
        static_cast<int>(field.size()), field.data());
    if (length < 0 || static_cast<std::size_t>(length) >= sizeof(name)) {
      fail("capture name", "formatted name exceeds fixed buffer");
    }
    captures->copy_token(
        std::string_view(name, static_cast<std::size_t>(length)), position,
        values, elements);
  }

  static void capture_layer_head_token(
      CapturePlan* captures, std::uint32_t layer, std::string_view field,
      std::uint32_t position, const BFloat16* values, std::uint32_t heads,
      std::uint32_t width) {
    if (captures == nullptr) {
      return;
    }
    char name[96];
    const int length = std::snprintf(
        name, sizeof(name), "prefill.layer.%02u.%.*s", layer,
        static_cast<int>(field.size()), field.data());
    if (length < 0 || static_cast<std::size_t>(length) >= sizeof(name)) {
      fail("capture name", "formatted name exceeds fixed buffer");
    }
    captures->copy_head_token(
        std::string_view(name, static_cast<std::size_t>(length)), position,
        values, heads, width);
  }

  const WeightArena& weights_;
  DeviceAllocation scratch_;
  KvCaches caches_;
  LtHandle handle_;
  LinearPlan local_q_;
  LinearPlan local_kv_;
  LinearPlan global_q_;
  LinearPlan global_kv_;
  LinearPlan local_o_;
  LinearPlan global_o_;
  LinearPlan hidden_to_mlp_;
  LinearPlan mlp_to_hidden_;
  LinearPlan lm_head_;
  std::array<LayerWeights, model::kLayerCount> layers_{};
};

class ShortDecodeEngine {
 public:
  explicit ShortDecodeEngine(const WeightArena& weights)
      : weights_(weights),
        scratch_(ShortDecodeScratchLayout::kBytes),
        local_q_(1, model::kHiddenSize,
                 model::kQueryHeadCount * model::kLocalHeadSize),
        local_kv_(1, model::kHiddenSize,
                  model::kLocalKvHeadCount * model::kLocalHeadSize),
        global_q_(1, model::kHiddenSize,
                  model::kQueryHeadCount * model::kGlobalHeadSize),
        global_kv_(1, model::kHiddenSize,
                   model::kGlobalKvHeadCount * model::kGlobalHeadSize),
        local_o_(1, model::kQueryHeadCount * model::kLocalHeadSize,
                 model::kHiddenSize),
        global_o_(1, model::kQueryHeadCount * model::kGlobalHeadSize,
                  model::kHiddenSize),
        hidden_to_mlp_(1, model::kHiddenSize, model::kMlpSize),
        mlp_to_hidden_(1, model::kMlpSize, model::kHiddenSize),
        lm_head_(1, model::kHiddenSize, model::kVocabSize) {
    for (std::size_t layer = 0; layer < model::kLayerCount; ++layer) {
      const LayerWeightIds& ids = kWeightIds[layer];
      layers_[layer] = {
          weights.pointer(ids.input_norm),
          weights.pointer(ids.q_proj),
          weights.pointer(ids.k_proj),
          model::is_global_layer(layer) ? nullptr : weights.pointer(ids.v_proj),
          weights.pointer(ids.q_norm),
          weights.pointer(ids.k_norm),
          weights.pointer(ids.o_proj),
          weights.pointer(ids.post_attention_norm),
          weights.pointer(ids.pre_feedforward_norm),
          weights.pointer(ids.gate_proj),
          weights.pointer(ids.up_proj),
          weights.pointer(ids.down_proj),
          weights.pointer(ids.post_feedforward_norm),
          weights.pointer(ids.layer_scalar),
      };
    }
  }

  [[nodiscard]] std::size_t scratch_bytes() const { return scratch_.size(); }
  [[nodiscard]] std::size_t local_cache_bytes() const {
    return caches_.local_bytes();
  }
  [[nodiscard]] std::size_t global_cache_bytes() const {
    return caches_.global_bytes();
  }

  std::array<std::uint32_t, kShortDecodeDecisionCount> run(
      ShortDecodeCapturePlan* captures) {
    BFloat16* const h0 = at<BFloat16>(ShortDecodeScratchLayout::kH0);
    BFloat16* const h1 = at<BFloat16>(ShortDecodeScratchLayout::kH1);
    BFloat16* const h2 = at<BFloat16>(ShortDecodeScratchLayout::kH2);
    BFloat16* const q_raw =
        at<BFloat16>(ShortDecodeScratchLayout::kQueryRaw);
    BFloat16* const q_norm =
        at<BFloat16>(ShortDecodeScratchLayout::kQueryNorm);
    BFloat16* const q_rope =
        at<BFloat16>(ShortDecodeScratchLayout::kQueryRope);
    BFloat16* const k_raw =
        at<BFloat16>(ShortDecodeScratchLayout::kKeyRaw);
    BFloat16* const k_norm =
        at<BFloat16>(ShortDecodeScratchLayout::kKeyNorm);
    BFloat16* const k_rope =
        at<BFloat16>(ShortDecodeScratchLayout::kKeyRope);
    BFloat16* const v_raw =
        at<BFloat16>(ShortDecodeScratchLayout::kValueRaw);
    BFloat16* const v_norm =
        at<BFloat16>(ShortDecodeScratchLayout::kValueNorm);
    BFloat16* const context =
        at<BFloat16>(ShortDecodeScratchLayout::kContext);
    BFloat16* const gate = at<BFloat16>(ShortDecodeScratchLayout::kGate);
    BFloat16* const up = at<BFloat16>(ShortDecodeScratchLayout::kUp);
    BFloat16* const product =
        at<BFloat16>(ShortDecodeScratchLayout::kProduct);
    BFloat16* const logits = at<BFloat16>(ShortDecodeScratchLayout::kLogits);
    BFloat16* const capped_logits =
        at<BFloat16>(ShortDecodeScratchLayout::kCappedLogits);
    BFloat16* const local_cos =
        at<BFloat16>(ShortDecodeScratchLayout::kLocalCos);
    BFloat16* const local_sin =
        at<BFloat16>(ShortDecodeScratchLayout::kLocalSin);
    BFloat16* const global_cos =
        at<BFloat16>(ShortDecodeScratchLayout::kGlobalCos);
    BFloat16* const global_sin =
        at<BFloat16>(ShortDecodeScratchLayout::kGlobalSin);
    BFloat16* const probabilities =
        at<BFloat16>(ShortDecodeScratchLayout::kProbabilities);
    std::uint32_t* const argmax =
        at<std::uint32_t>(ShortDecodeScratchLayout::kArgmax);

    primitives::generate_rope_factors_24(local_cos, local_sin, global_cos,
                                         global_sin);
    capture(captures, "prefill.rotary.sliding_attention.cos", local_cos);
    capture(captures, "prefill.rotary.sliding_attention.sin", local_sin);
    capture(captures, "prefill.rotary.full_attention.cos", global_cos);
    capture(captures, "prefill.rotary.full_attention.sin", global_sin);

    std::array<std::uint32_t, kShortDecodeDecisionCount> generated{};
    for (std::uint32_t position = 0; position < kShortDecodeFedTokens;
         ++position) {
      const std::uint32_t input_token =
          position < kShortDecodePromptTokens
              ? kShortDecodePrompt[position]
              : generated[position - kShortDecodePromptTokens];
      primitives::embedding_lookup(
          weights_.pointer(model::kEmbeddingPhysicalId), input_token, h0);
      if (position < kShortDecodePromptTokens) {
        capture_prefill_row(captures, "prefill.embedding", position, h0,
                            model::kHiddenSize);
      }

      for (std::uint32_t layer = 0; layer < model::kLayerCount; ++layer) {
        const bool global = model::is_global_layer(layer);
        const model::AttentionKind kind =
            global ? model::AttentionKind::global
                   : model::AttentionKind::local;
        const std::uint32_t head_size =
            global ? model::kGlobalHeadSize : model::kLocalHeadSize;
        const std::uint32_t kv_heads =
            global ? model::kGlobalKvHeadCount : model::kLocalKvHeadCount;
        const LayerWeights& weight = layers_[layer];
        const LayerCacheView cache = caches_.layer(layer);
        const BFloat16* const cos =
            (global ? global_cos : local_cos) + position * head_size;
        const BFloat16* const sin =
            (global ? global_sin : local_sin) + position * head_size;

        primitives::rms_norm(h0, weight.input_norm, h1, 1,
                             model::kHiddenSize);
        const LinearPlan& q_plan = global ? global_q_ : local_q_;
        const LinearPlan& kv_plan = global ? global_kv_ : local_kv_;
        q_plan.run(handle_.get(), h1, weight.q_proj, q_raw);
        kv_plan.run(handle_.get(), h1, weight.k_proj, k_raw);
        if (!global) {
          kv_plan.run(handle_.get(), h1, weight.v_proj, v_raw);
        }
        primitives::rms_norm(q_raw, weight.q_norm, q_norm,
                             model::kQueryHeadCount, head_size);
        primitives::rms_norm(k_raw, weight.k_norm, k_norm, kv_heads,
                             head_size);
        primitives::rms_norm_unscaled(global ? k_raw : v_raw, v_norm,
                                      kv_heads, head_size);
        primitives::apply_rope_m1(q_norm, cos, sin, q_rope,
                                  model::kQueryHeadCount, kind);
        primitives::apply_rope_m1(k_norm, cos, sin, k_rope, kv_heads, kind);
        primitives::write_kv_cache_m1(
            k_rope, v_norm, cache.key, cache.value, position, cache.capacity,
            kind);
        primitives::causal_gqa_attention_cached_m1_24(
            q_rope, cache.key, cache.value, position, cache.capacity,
            probabilities, context, kind);

        const LinearPlan& o_plan = global ? global_o_ : local_o_;
        o_plan.run(handle_.get(), context, weight.o_proj, h2);
        primitives::rms_norm(h2, weight.post_attention_norm, h1, 1,
                             model::kHiddenSize);
        primitives::residual_add(h0, h1, h2, model::kHiddenSize);
        primitives::rms_norm(h2, weight.pre_feedforward_norm, h1, 1,
                             model::kHiddenSize);
        hidden_to_mlp_.run(handle_.get(), h1, weight.gate_proj, gate);
        hidden_to_mlp_.run(handle_.get(), h1, weight.up_proj, up);
        primitives::gelu_tanh_multiply(gate, up, product, model::kMlpSize);
        mlp_to_hidden_.run(handle_.get(), product, weight.down_proj, h0);
        primitives::rms_norm(h0, weight.post_feedforward_norm, h1, 1,
                             model::kHiddenSize);
        primitives::residual_add(h2, h1, h0, model::kHiddenSize);
        primitives::trained_scalar(h0, weight.layer_scalar,
                                   model::kHiddenSize);

        if (position < kShortDecodePromptTokens) {
          capture_prefill_layer_row(captures, layer, position, h0);
        }
        if (position >= kShortDecodePromptTokens - 1) {
          capture_step_layer(captures,
                             position - (kShortDecodePromptTokens - 1), layer,
                             h0);
        }
      }

      primitives::rms_norm(
          h0, weights_.pointer(model::kFinalNormPhysicalId), h1, 1,
          model::kHiddenSize);
      if (position < kShortDecodePromptTokens) {
        capture_prefill_row(captures, "prefill.final_norm", position, h1,
                            model::kHiddenSize);
      }
      if (position >= kShortDecodePromptTokens - 1) {
        const std::uint32_t step =
            position - (kShortDecodePromptTokens - 1);
        capture_step(captures, step, "cached.final_norm", h1);
        lm_head_.run(handle_.get(), h1,
                     weights_.pointer(model::kLmHeadLogicalId), logits);
        if (position == kShortDecodePromptTokens - 1) {
          capture(captures, "prefill.logits.pre_softcap", logits);
        }
        capture_step(captures, step, "logits.cached.pre_softcap", logits);
        primitives::softcap_and_argmax(logits, capped_logits, argmax,
                                       model::kVocabSize, 30.0F);
        if (position == kShortDecodePromptTokens - 1) {
          capture(captures, "prefill.logits.post_softcap", capped_logits);
        }
        capture_step(captures, step, "logits.cached.post_softcap",
                     capped_logits);
        check_cuda(cudaMemcpy(&generated[step], argmax,
                              sizeof(generated[step]), cudaMemcpyDeviceToHost),
                   "copy short-decode generated token");
      }
    }
    return generated;
  }

 private:
  template <typename T>
  T* at(std::size_t byte_offset) const {
    return reinterpret_cast<T*>(static_cast<std::uint8_t*>(scratch_.data()) +
                                byte_offset);
  }

  static void capture(ShortDecodeCapturePlan* captures, std::string_view name,
                      const BFloat16* values) {
    if (captures != nullptr) {
      captures->copy_device(name, values);
    }
  }

  static void capture_prefill_row(ShortDecodeCapturePlan* captures,
                                  std::string_view name,
                                  std::uint32_t position,
                                  const BFloat16* values,
                                  std::size_t elements) {
    if (captures != nullptr) {
      captures->copy_prefill_row(name, position, values, elements);
    }
  }

  static void capture_prefill_layer_row(ShortDecodeCapturePlan* captures,
                                        std::uint32_t layer,
                                        std::uint32_t position,
                                        const BFloat16* values) {
    if (captures == nullptr) {
      return;
    }
    char name[96];
    const int length = std::snprintf(name, sizeof(name),
                                     "prefill.layer.%02u.output", layer);
    if (length < 0 || static_cast<std::size_t>(length) >= sizeof(name)) {
      fail("short-decode capture name", "prefill layer name is too long");
    }
    captures->copy_prefill_row(
        std::string_view(name, static_cast<std::size_t>(length)), position,
        values, model::kHiddenSize);
  }

  static void capture_step_layer(ShortDecodeCapturePlan* captures,
                                 std::uint32_t step, std::uint32_t layer,
                                 const BFloat16* values) {
    if (captures == nullptr) {
      return;
    }
    char name[128];
    const int length = std::snprintf(
        name, sizeof(name),
        "generation.step.%02u.cached.layer.%02u.output", step, layer);
    if (length < 0 || static_cast<std::size_t>(length) >= sizeof(name)) {
      fail("short-decode capture name", "step layer name is too long");
    }
    captures->copy_device(
        std::string_view(name, static_cast<std::size_t>(length)), values);
  }

  static void capture_step(ShortDecodeCapturePlan* captures,
                           std::uint32_t step, std::string_view suffix,
                           const BFloat16* values) {
    if (captures == nullptr) {
      return;
    }
    char name[128];
    const int length = std::snprintf(
        name, sizeof(name), "generation.step.%02u.%.*s", step,
        static_cast<int>(suffix.size()), suffix.data());
    if (length < 0 || static_cast<std::size_t>(length) >= sizeof(name)) {
      fail("short-decode capture name", "step capture name is too long");
    }
    captures->copy_device(
        std::string_view(name, static_cast<std::size_t>(length)), values);
  }

  const WeightArena& weights_;
  DeviceAllocation scratch_;
  ShortDecodeKvCaches caches_;
  LtHandle handle_;
  LinearPlan local_q_;
  LinearPlan local_kv_;
  LinearPlan global_q_;
  LinearPlan global_kv_;
  LinearPlan local_o_;
  LinearPlan global_o_;
  LinearPlan hidden_to_mlp_;
  LinearPlan mlp_to_hidden_;
  LinearPlan lm_head_;
  std::array<LayerWeights, model::kLayerCount> layers_{};
};

class BoundaryEngine {
 public:
  explicit BoundaryEngine(const WeightArena& weights)
      : weights_(weights),
        scratch_(BoundaryScratchLayout::kBytes),
        local_q_(1, model::kHiddenSize,
                 model::kQueryHeadCount * model::kLocalHeadSize),
        local_kv_(1, model::kHiddenSize,
                  model::kLocalKvHeadCount * model::kLocalHeadSize),
        global_q_(1, model::kHiddenSize,
                  model::kQueryHeadCount * model::kGlobalHeadSize),
        global_kv_(1, model::kHiddenSize,
                   model::kGlobalKvHeadCount * model::kGlobalHeadSize),
        local_o_(1, model::kQueryHeadCount * model::kLocalHeadSize,
                 model::kHiddenSize),
        global_o_(1, model::kQueryHeadCount * model::kGlobalHeadSize,
                  model::kHiddenSize),
        hidden_to_mlp_(1, model::kHiddenSize, model::kMlpSize),
        mlp_to_hidden_(1, model::kMlpSize, model::kHiddenSize),
        lm_head_(1, model::kHiddenSize, model::kVocabSize) {
    for (std::size_t layer = 0; layer < model::kLayerCount; ++layer) {
      const LayerWeightIds& ids = kWeightIds[layer];
      layers_[layer] = {
          weights.pointer(ids.input_norm),
          weights.pointer(ids.q_proj),
          weights.pointer(ids.k_proj),
          model::is_global_layer(layer) ? nullptr : weights.pointer(ids.v_proj),
          weights.pointer(ids.q_norm),
          weights.pointer(ids.k_norm),
          weights.pointer(ids.o_proj),
          weights.pointer(ids.post_attention_norm),
          weights.pointer(ids.pre_feedforward_norm),
          weights.pointer(ids.gate_proj),
          weights.pointer(ids.up_proj),
          weights.pointer(ids.down_proj),
          weights.pointer(ids.post_feedforward_norm),
          weights.pointer(ids.layer_scalar),
      };
    }
  }

  [[nodiscard]] std::size_t scratch_bytes() const { return scratch_.size(); }
  [[nodiscard]] std::size_t local_cache_bytes() const {
    return caches_.local_bytes();
  }
  [[nodiscard]] std::size_t global_cache_bytes() const {
    return caches_.global_bytes();
  }

  std::uint32_t run(BoundaryCapturePlan* captures) {
    BFloat16* const h0 = at<BFloat16>(BoundaryScratchLayout::kH0);
    BFloat16* const h1 = at<BFloat16>(BoundaryScratchLayout::kH1);
    BFloat16* const h2 = at<BFloat16>(BoundaryScratchLayout::kH2);
    BFloat16* const q_raw = at<BFloat16>(BoundaryScratchLayout::kQueryRaw);
    BFloat16* const q_norm = at<BFloat16>(BoundaryScratchLayout::kQueryNorm);
    BFloat16* const q_rope = at<BFloat16>(BoundaryScratchLayout::kQueryRope);
    BFloat16* const k_raw = at<BFloat16>(BoundaryScratchLayout::kKeyRaw);
    BFloat16* const k_norm = at<BFloat16>(BoundaryScratchLayout::kKeyNorm);
    BFloat16* const k_rope = at<BFloat16>(BoundaryScratchLayout::kKeyRope);
    BFloat16* const v_raw = at<BFloat16>(BoundaryScratchLayout::kValueRaw);
    BFloat16* const v_norm = at<BFloat16>(BoundaryScratchLayout::kValueNorm);
    BFloat16* const context = at<BFloat16>(BoundaryScratchLayout::kContext);
    BFloat16* const gate = at<BFloat16>(BoundaryScratchLayout::kGate);
    BFloat16* const up = at<BFloat16>(BoundaryScratchLayout::kUp);
    BFloat16* const product = at<BFloat16>(BoundaryScratchLayout::kProduct);
    BFloat16* const logits = at<BFloat16>(BoundaryScratchLayout::kLogits);
    BFloat16* const capped_logits =
        at<BFloat16>(BoundaryScratchLayout::kCappedLogits);
    BFloat16* const local_cos = at<BFloat16>(BoundaryScratchLayout::kLocalCos);
    BFloat16* const local_sin = at<BFloat16>(BoundaryScratchLayout::kLocalSin);
    BFloat16* const global_cos =
        at<BFloat16>(BoundaryScratchLayout::kGlobalCos);
    BFloat16* const global_sin =
        at<BFloat16>(BoundaryScratchLayout::kGlobalSin);
    BFloat16* const score_scratch =
        at<BFloat16>(BoundaryScratchLayout::kScoreScratch);
    BFloat16* const probabilities =
        at<BFloat16>(BoundaryScratchLayout::kProbabilities);
    std::uint32_t* const argmax =
        at<std::uint32_t>(BoundaryScratchLayout::kArgmax);

    std::uint32_t result = 0;
    for (std::uint32_t position = 0; position < kBoundaryPrefixTokens;
         ++position) {
      const bool retained = position >= kBoundaryCaptureFirstPosition;
      primitives::generate_rope_factors_m1(
          local_cos, local_sin, global_cos, global_sin, position);
      if (retained) {
        capture_row(captures,
                    "boundary.cached.rows_1022_1025.rotary."
                    "sliding_attention.cos",
                    position, local_cos, model::kLocalHeadSize);
        capture_row(captures,
                    "boundary.cached.rows_1022_1025.rotary."
                    "sliding_attention.sin",
                    position, local_sin, model::kLocalHeadSize);
        capture_row(captures,
                    "boundary.cached.rows_1022_1025.rotary."
                    "full_attention.cos",
                    position, global_cos, model::kGlobalHeadSize);
        capture_row(captures,
                    "boundary.cached.rows_1022_1025.rotary."
                    "full_attention.sin",
                    position, global_sin, model::kGlobalHeadSize);
      }

      primitives::embedding_lookup(
          weights_.pointer(model::kEmbeddingPhysicalId),
          boundary_input_token(position), h0);
      if (retained) {
        capture_row(captures,
                    "boundary.cached.rows_1022_1025.embedding", position, h0,
                    model::kHiddenSize);
      }

      for (std::uint32_t layer = 0; layer < model::kLayerCount; ++layer) {
        const bool global = model::is_global_layer(layer);
        const model::AttentionKind kind =
            global ? model::AttentionKind::global
                   : model::AttentionKind::local;
        const std::uint32_t head_size =
            global ? model::kGlobalHeadSize : model::kLocalHeadSize;
        const std::uint32_t kv_heads =
            global ? model::kGlobalKvHeadCount : model::kLocalKvHeadCount;
        const std::uint32_t q_width = model::kQueryHeadCount * head_size;
        const std::uint32_t kv_width = kv_heads * head_size;
        const LayerWeights& weight = layers_[layer];
        const LayerCacheView cache = caches_.layer(layer);
        const BFloat16* const cos = global ? global_cos : local_cos;
        const BFloat16* const sin = global ? global_sin : local_sin;
        const bool deep = retained && (layer == 0 || layer == 5);

        primitives::rms_norm(h0, weight.input_norm, h1, 1,
                             model::kHiddenSize);
        const LinearPlan& q_plan = global ? global_q_ : local_q_;
        const LinearPlan& kv_plan = global ? global_kv_ : local_kv_;
        q_plan.run(handle_.get(), h1, weight.q_proj, q_raw);
        kv_plan.run(handle_.get(), h1, weight.k_proj, k_raw);
        if (!global) {
          kv_plan.run(handle_.get(), h1, weight.v_proj, v_raw);
        }
        if (deep) {
          capture_layer_row(captures, layer, "q_raw", position, q_raw,
                            q_width);
          capture_layer_row(captures, layer, "k_raw", position, k_raw,
                            kv_width);
          capture_layer_row(captures, layer, "v_raw", position,
                            global ? k_raw : v_raw, kv_width);
        }

        primitives::rms_norm(q_raw, weight.q_norm, q_norm,
                             model::kQueryHeadCount, head_size);
        primitives::rms_norm(k_raw, weight.k_norm, k_norm, kv_heads,
                             head_size);
        primitives::rms_norm_unscaled(global ? k_raw : v_raw, v_norm,
                                      kv_heads, head_size);
        if (deep) {
          capture_layer_row(captures, layer, "q_norm", position, q_norm,
                            q_width);
          capture_layer_row(captures, layer, "k_norm", position, k_norm,
                            kv_width);
          capture_layer_row(captures, layer, "v_norm", position, v_norm,
                            kv_width);
        }

        primitives::apply_rope_m1(q_norm, cos, sin, q_rope,
                                  model::kQueryHeadCount, kind);
        primitives::apply_rope_m1(k_norm, cos, sin, k_rope, kv_heads, kind);
        if (deep) {
          capture_layer_row(captures, layer, "q_rope", position, q_rope,
                            q_width);
          capture_layer_row(captures, layer, "k_rope", position, k_rope,
                            kv_width);
        }

        primitives::write_kv_cache_m1(
            k_rope, v_norm, cache.key, cache.value, position, cache.capacity,
            kind);
        primitives::causal_gqa_attention_cached_m1_boundary(
            q_rope, cache.key, cache.value, position, score_scratch,
            probabilities, context, kind);
        if (deep) {
          capture_layer_row(captures, layer, "attention_probabilities",
                            position, probabilities,
                            model::kQueryHeadCount * kBoundaryPrefixTokens);
          capture_layer_row(captures, layer, "attention_context", position,
                            context, q_width);
        }

        const LinearPlan& o_plan = global ? global_o_ : local_o_;
        o_plan.run(handle_.get(), context, weight.o_proj, h2);
        if (deep) {
          capture_layer_row(captures, layer, "attention_output", position, h2,
                            model::kHiddenSize);
        }
        primitives::rms_norm(h2, weight.post_attention_norm, h1, 1,
                             model::kHiddenSize);
        primitives::residual_add(h0, h1, h2, model::kHiddenSize);
        primitives::rms_norm(h2, weight.pre_feedforward_norm, h1, 1,
                             model::kHiddenSize);
        hidden_to_mlp_.run(handle_.get(), h1, weight.gate_proj, gate);
        hidden_to_mlp_.run(handle_.get(), h1, weight.up_proj, up);
        primitives::gelu_tanh_multiply(gate, up, product, model::kMlpSize);
        mlp_to_hidden_.run(handle_.get(), product, weight.down_proj, h0);
        primitives::rms_norm(h0, weight.post_feedforward_norm, h1, 1,
                             model::kHiddenSize);
        primitives::residual_add(h2, h1, h0, model::kHiddenSize);
        primitives::trained_scalar(h0, weight.layer_scalar,
                                   model::kHiddenSize);
        if (retained) {
          capture_layer_row(captures, layer, "output", position, h0,
                            model::kHiddenSize);
        }
      }

      if (retained) {
        primitives::rms_norm(
            h0, weights_.pointer(model::kFinalNormPhysicalId), h1, 1,
            model::kHiddenSize);
        capture_row(captures,
                    "boundary.cached.rows_1022_1025.final_norm", position, h1,
                    model::kHiddenSize);
      }
      if (position == kBoundaryDecisionPosition) {
        lm_head_.run(handle_.get(), h1,
                     weights_.pointer(model::kLmHeadLogicalId), logits);
        capture_device(
            captures, "boundary.cached.position_1025.logits.pre_softcap",
            logits);
        primitives::softcap_and_argmax(logits, capped_logits, argmax,
                                       model::kVocabSize, 30.0F);
        capture_device(
            captures, "boundary.cached.position_1025.logits.post_softcap",
            capped_logits);
        check_cuda(cudaMemcpy(&result, argmax, sizeof(result),
                              cudaMemcpyDeviceToHost),
                   "copy local-boundary argmax");
      }
    }
    return result;
  }

 private:
  template <typename T>
  T* at(std::size_t byte_offset) const {
    return reinterpret_cast<T*>(static_cast<std::uint8_t*>(scratch_.data()) +
                                byte_offset);
  }

  static void capture_row(BoundaryCapturePlan* captures,
                          std::string_view name, std::uint32_t position,
                          const BFloat16* values, std::size_t elements) {
    if (captures != nullptr) {
      captures->copy_row(name, position, values, elements);
    }
  }

  static void capture_device(BoundaryCapturePlan* captures,
                             std::string_view name,
                             const BFloat16* values) {
    if (captures != nullptr) {
      captures->copy_device(name, values);
    }
  }

  static void capture_layer_row(BoundaryCapturePlan* captures,
                                std::uint32_t layer, std::string_view field,
                                std::uint32_t position,
                                const BFloat16* values,
                                std::size_t elements) {
    if (captures == nullptr) {
      return;
    }
    char name[160];
    const int length = std::snprintf(
        name, sizeof(name),
        "boundary.cached.rows_1022_1025.layer.%02u.%.*s", layer,
        static_cast<int>(field.size()), field.data());
    if (length < 0 || static_cast<std::size_t>(length) >= sizeof(name)) {
      fail("boundary capture name", "formatted layer name exceeds buffer");
    }
    captures->copy_row(
        std::string_view(name, static_cast<std::size_t>(length)), position,
        values, elements);
  }

  const WeightArena& weights_;
  DeviceAllocation scratch_;
  BoundaryKvCaches caches_;
  LtHandle handle_;
  LinearPlan local_q_;
  LinearPlan local_kv_;
  LinearPlan global_q_;
  LinearPlan global_kv_;
  LinearPlan local_o_;
  LinearPlan global_o_;
  LinearPlan hidden_to_mlp_;
  LinearPlan mlp_to_hidden_;
  LinearPlan lm_head_;
  std::array<LayerWeights, model::kLayerCount> layers_{};
};

class HybridBoundaryEngine {
 public:
  explicit HybridBoundaryEngine(const WeightArena& weights)
      : weights_(weights),
        scratch_(HybridBoundaryScratchLayout::kBytes),
        prefill_local_q_(kHybridPrefillTokens, model::kHiddenSize,
                         model::kQueryHeadCount * model::kLocalHeadSize),
        prefill_local_kv_(kHybridPrefillTokens, model::kHiddenSize,
                          model::kLocalKvHeadCount * model::kLocalHeadSize),
        prefill_global_q_(kHybridPrefillTokens, model::kHiddenSize,
                          model::kQueryHeadCount * model::kGlobalHeadSize),
        prefill_global_kv_(kHybridPrefillTokens, model::kHiddenSize,
                           model::kGlobalKvHeadCount * model::kGlobalHeadSize),
        prefill_local_o_(kHybridPrefillTokens,
                         model::kQueryHeadCount * model::kLocalHeadSize,
                         model::kHiddenSize),
        prefill_global_o_(kHybridPrefillTokens,
                          model::kQueryHeadCount * model::kGlobalHeadSize,
                          model::kHiddenSize),
        prefill_hidden_to_mlp_(kHybridPrefillTokens, model::kHiddenSize,
                               model::kMlpSize),
        prefill_mlp_to_hidden_(kHybridPrefillTokens, model::kMlpSize,
                               model::kHiddenSize),
        decode_local_q_(1, model::kHiddenSize,
                        model::kQueryHeadCount * model::kLocalHeadSize),
        decode_local_kv_(1, model::kHiddenSize,
                         model::kLocalKvHeadCount * model::kLocalHeadSize),
        decode_global_q_(1, model::kHiddenSize,
                         model::kQueryHeadCount * model::kGlobalHeadSize),
        decode_global_kv_(1, model::kHiddenSize,
                          model::kGlobalKvHeadCount * model::kGlobalHeadSize),
        decode_local_o_(1,
                        model::kQueryHeadCount * model::kLocalHeadSize,
                        model::kHiddenSize),
        decode_global_o_(1,
                         model::kQueryHeadCount * model::kGlobalHeadSize,
                         model::kHiddenSize),
        decode_hidden_to_mlp_(1, model::kHiddenSize, model::kMlpSize),
        decode_mlp_to_hidden_(1, model::kMlpSize, model::kHiddenSize),
        lm_head_(1, model::kHiddenSize, model::kVocabSize) {
    for (std::size_t layer = 0; layer < model::kLayerCount; ++layer) {
      const LayerWeightIds& ids = kWeightIds[layer];
      layers_[layer] = {
          weights.pointer(ids.input_norm),
          weights.pointer(ids.q_proj),
          weights.pointer(ids.k_proj),
          model::is_global_layer(layer) ? nullptr : weights.pointer(ids.v_proj),
          weights.pointer(ids.q_norm),
          weights.pointer(ids.k_norm),
          weights.pointer(ids.o_proj),
          weights.pointer(ids.post_attention_norm),
          weights.pointer(ids.pre_feedforward_norm),
          weights.pointer(ids.gate_proj),
          weights.pointer(ids.up_proj),
          weights.pointer(ids.down_proj),
          weights.pointer(ids.post_feedforward_norm),
          weights.pointer(ids.layer_scalar),
      };
    }
  }

  [[nodiscard]] std::size_t scratch_bytes() const { return scratch_.size(); }
  [[nodiscard]] std::size_t local_cache_bytes() const {
    return caches_.local_bytes();
  }
  [[nodiscard]] std::size_t global_cache_bytes() const {
    return caches_.global_bytes();
  }

  std::array<std::uint32_t, 3> run(BoundaryCapturePlan* captures) {
    std::array<std::uint32_t, 3> predictions{};
    predictions[0] = run_prefill(captures);
    predictions[1] = run_decode_position(kHybridPrefillTokens, captures);
    predictions[2] = run_decode_position(kHybridPrefillTokens + 1, captures);
    return predictions;
  }

 private:
  std::uint32_t run_prefill(BoundaryCapturePlan* captures) {
    BFloat16* const h0 = at<BFloat16>(HybridBoundaryScratchLayout::kH0);
    BFloat16* const h1 = at<BFloat16>(HybridBoundaryScratchLayout::kH1);
    BFloat16* const h2 = at<BFloat16>(HybridBoundaryScratchLayout::kH2);
    BFloat16* const q_raw =
        at<BFloat16>(HybridBoundaryScratchLayout::kQueryRaw);
    BFloat16* const q_norm =
        at<BFloat16>(HybridBoundaryScratchLayout::kQueryNorm);
    BFloat16* const q_rope =
        at<BFloat16>(HybridBoundaryScratchLayout::kQueryRope);
    BFloat16* const k_raw =
        at<BFloat16>(HybridBoundaryScratchLayout::kKeyRaw);
    BFloat16* const k_norm =
        at<BFloat16>(HybridBoundaryScratchLayout::kKeyNorm);
    BFloat16* const k_rope =
        at<BFloat16>(HybridBoundaryScratchLayout::kKeyRope);
    BFloat16* const v_raw =
        at<BFloat16>(HybridBoundaryScratchLayout::kValueRaw);
    BFloat16* const v_norm =
        at<BFloat16>(HybridBoundaryScratchLayout::kValueNorm);
    BFloat16* const context =
        at<BFloat16>(HybridBoundaryScratchLayout::kContext);
    BFloat16* const gate = at<BFloat16>(HybridBoundaryScratchLayout::kGate);
    BFloat16* const up = at<BFloat16>(HybridBoundaryScratchLayout::kUp);
    BFloat16* const product =
        at<BFloat16>(HybridBoundaryScratchLayout::kProduct);
    BFloat16* const local_cos =
        at<BFloat16>(HybridBoundaryScratchLayout::kLocalCos);
    BFloat16* const local_sin =
        at<BFloat16>(HybridBoundaryScratchLayout::kLocalSin);
    BFloat16* const global_cos =
        at<BFloat16>(HybridBoundaryScratchLayout::kGlobalCos);
    BFloat16* const global_sin =
        at<BFloat16>(HybridBoundaryScratchLayout::kGlobalSin);
    BFloat16* const score_scratch =
        at<BFloat16>(HybridBoundaryScratchLayout::kScoreScratch);
    BFloat16* const probabilities =
        at<BFloat16>(HybridBoundaryScratchLayout::kProbabilities);

    prefill::generate_rope_factors_m1024(local_cos, local_sin, global_cos,
                                         global_sin);
    capture_prefill_rows(
        captures,
        "boundary.hybrid.rows_1022_1025.rotary.sliding_attention.cos",
        local_cos, model::kLocalHeadSize);
    capture_prefill_rows(
        captures,
        "boundary.hybrid.rows_1022_1025.rotary.sliding_attention.sin",
        local_sin, model::kLocalHeadSize);
    capture_prefill_rows(
        captures,
        "boundary.hybrid.rows_1022_1025.rotary.full_attention.cos",
        global_cos, model::kGlobalHeadSize);
    capture_prefill_rows(
        captures,
        "boundary.hybrid.rows_1022_1025.rotary.full_attention.sin",
        global_sin, model::kGlobalHeadSize);

    for (std::uint32_t position = 0; position < kHybridPrefillTokens;
         ++position) {
      primitives::embedding_lookup(
          weights_.pointer(model::kEmbeddingPhysicalId),
          boundary_input_token(position),
          h0 + static_cast<std::size_t>(position) * model::kHiddenSize);
    }
    capture_prefill_rows(captures,
                         "boundary.hybrid.rows_1022_1025.embedding", h0,
                         model::kHiddenSize);

    for (std::uint32_t layer = 0; layer < model::kLayerCount; ++layer) {
      const bool global = model::is_global_layer(layer);
      const model::AttentionKind kind =
          global ? model::AttentionKind::global : model::AttentionKind::local;
      const std::uint32_t head_size =
          global ? model::kGlobalHeadSize : model::kLocalHeadSize;
      const std::uint32_t kv_heads =
          global ? model::kGlobalKvHeadCount : model::kLocalKvHeadCount;
      const std::uint32_t q_width = model::kQueryHeadCount * head_size;
      const std::uint32_t kv_width = kv_heads * head_size;
      const LayerWeights& weight = layers_[layer];
      const LayerCacheView cache = caches_.layer(layer);
      const BFloat16* const cosine = global ? global_cos : local_cos;
      const BFloat16* const sine = global ? global_sin : local_sin;
      const bool deep = layer == 0 || layer == 5;

      primitives::rms_norm(h0, weight.input_norm, h1,
                           kHybridPrefillTokens, model::kHiddenSize);
      const LinearPlan& q_plan =
          global ? prefill_global_q_ : prefill_local_q_;
      const LinearPlan& kv_plan =
          global ? prefill_global_kv_ : prefill_local_kv_;
      q_plan.run(handle_.get(), h1, weight.q_proj, q_raw);
      kv_plan.run(handle_.get(), h1, weight.k_proj, k_raw);
      if (!global) {
        kv_plan.run(handle_.get(), h1, weight.v_proj, v_raw);
      }
      if (deep) {
        capture_prefill_layer_rows(captures, layer, "q_raw", q_raw,
                                   q_width);
        capture_prefill_layer_rows(captures, layer, "k_raw", k_raw,
                                   kv_width);
        capture_prefill_layer_rows(captures, layer, "v_raw",
                                   global ? k_raw : v_raw, kv_width);
      }

      primitives::rms_norm(q_raw, weight.q_norm, q_norm,
                           kHybridPrefillTokens * model::kQueryHeadCount,
                           head_size);
      primitives::rms_norm(k_raw, weight.k_norm, k_norm,
                           kHybridPrefillTokens * kv_heads, head_size);
      primitives::rms_norm_unscaled(
          global ? k_raw : v_raw, v_norm,
          kHybridPrefillTokens * kv_heads, head_size);
      if (deep) {
        capture_prefill_layer_rows(captures, layer, "q_norm", q_norm,
                                   q_width);
        capture_prefill_layer_rows(captures, layer, "k_norm", k_norm,
                                   kv_width);
        capture_prefill_layer_rows(captures, layer, "v_norm", v_norm,
                                   kv_width);
      }

      prefill::apply_rope_transpose_m1024(
          q_norm, cosine, sine, q_rope, model::kQueryHeadCount, kind);
      prefill::apply_rope_transpose_m1024(k_norm, cosine, sine, k_rope,
                                          kv_heads, kind);
      if (deep) {
        capture_prefill_head_major_layer_rows(
            captures, layer, "q_rope", q_rope, model::kQueryHeadCount,
            head_size);
        capture_prefill_head_major_layer_rows(
            captures, layer, "k_rope", k_rope, kv_heads, head_size);
      }

      prefill::write_kv_cache_m1024(k_rope, v_norm, cache.key, cache.value,
                                    cache.capacity, kind);
      prefill::causal_gqa_attention_m1024(
          q_rope, k_rope, v_norm, score_scratch, probabilities, context,
          kind);
      if (deep) {
        capture_prefill_probability_rows(captures, layer, probabilities);
        capture_prefill_layer_rows(captures, layer, "attention_context",
                                   context, q_width);
      }

      const LinearPlan& o_plan =
          global ? prefill_global_o_ : prefill_local_o_;
      o_plan.run(handle_.get(), context, weight.o_proj, h2);
      if (deep) {
        capture_prefill_layer_rows(captures, layer, "attention_output", h2,
                                   model::kHiddenSize);
      }
      primitives::rms_norm(h2, weight.post_attention_norm, h1,
                           kHybridPrefillTokens, model::kHiddenSize);
      primitives::residual_add(
          h0, h1, h2,
          HybridBoundaryScratchLayout::kPrefillHiddenElements);
      primitives::rms_norm(h2, weight.pre_feedforward_norm, h1,
                           kHybridPrefillTokens, model::kHiddenSize);
      prefill_hidden_to_mlp_.run(handle_.get(), h1, weight.gate_proj, gate);
      prefill_hidden_to_mlp_.run(handle_.get(), h1, weight.up_proj, up);
      primitives::gelu_tanh_multiply(
          gate, up, product,
          HybridBoundaryScratchLayout::kPrefillMlpElements);
      prefill_mlp_to_hidden_.run(handle_.get(), product, weight.down_proj, h0);
      primitives::rms_norm(h0, weight.post_feedforward_norm, h1,
                           kHybridPrefillTokens, model::kHiddenSize);
      primitives::residual_add(
          h2, h1, h0,
          HybridBoundaryScratchLayout::kPrefillHiddenElements);
      primitives::trained_scalar(
          h0, weight.layer_scalar,
          HybridBoundaryScratchLayout::kPrefillHiddenElements);
      capture_prefill_layer_rows(captures, layer, "output", h0,
                                 model::kHiddenSize);
    }

    primitives::rms_norm(
        h0, weights_.pointer(model::kFinalNormPhysicalId), h1,
        kHybridPrefillTokens, model::kHiddenSize);
    capture_prefill_rows(captures,
                         "boundary.hybrid.rows_1022_1025.final_norm", h1,
                         model::kHiddenSize);
    return predict(h1 + static_cast<std::size_t>(kHybridPrefillTokens - 1) *
                            model::kHiddenSize,
                   nullptr, nullptr);
  }

  std::uint32_t run_decode_position(std::uint32_t position,
                                    BoundaryCapturePlan* captures) {
    BFloat16* const h0 = at<BFloat16>(HybridBoundaryScratchLayout::kH0);
    BFloat16* const h1 = at<BFloat16>(HybridBoundaryScratchLayout::kH1);
    BFloat16* const h2 = at<BFloat16>(HybridBoundaryScratchLayout::kH2);
    BFloat16* const q_raw =
        at<BFloat16>(HybridBoundaryScratchLayout::kQueryRaw);
    BFloat16* const q_norm =
        at<BFloat16>(HybridBoundaryScratchLayout::kQueryNorm);
    BFloat16* const q_rope =
        at<BFloat16>(HybridBoundaryScratchLayout::kQueryRope);
    BFloat16* const k_raw =
        at<BFloat16>(HybridBoundaryScratchLayout::kKeyRaw);
    BFloat16* const k_norm =
        at<BFloat16>(HybridBoundaryScratchLayout::kKeyNorm);
    BFloat16* const k_rope =
        at<BFloat16>(HybridBoundaryScratchLayout::kKeyRope);
    BFloat16* const v_raw =
        at<BFloat16>(HybridBoundaryScratchLayout::kValueRaw);
    BFloat16* const v_norm =
        at<BFloat16>(HybridBoundaryScratchLayout::kValueNorm);
    BFloat16* const context =
        at<BFloat16>(HybridBoundaryScratchLayout::kContext);
    BFloat16* const gate = at<BFloat16>(HybridBoundaryScratchLayout::kGate);
    BFloat16* const up = at<BFloat16>(HybridBoundaryScratchLayout::kUp);
    BFloat16* const product =
        at<BFloat16>(HybridBoundaryScratchLayout::kProduct);
    BFloat16* const local_cos =
        at<BFloat16>(HybridBoundaryScratchLayout::kLocalCos);
    BFloat16* const local_sin =
        at<BFloat16>(HybridBoundaryScratchLayout::kLocalSin);
    BFloat16* const global_cos =
        at<BFloat16>(HybridBoundaryScratchLayout::kGlobalCos);
    BFloat16* const global_sin =
        at<BFloat16>(HybridBoundaryScratchLayout::kGlobalSin);
    BFloat16* const score_scratch =
        at<BFloat16>(HybridBoundaryScratchLayout::kScoreScratch);
    BFloat16* const probabilities =
        at<BFloat16>(HybridBoundaryScratchLayout::kProbabilities);

    primitives::generate_rope_factors_m1(local_cos, local_sin, global_cos,
                                         global_sin, position);
    capture_row(captures,
                "boundary.hybrid.rows_1022_1025.rotary."
                "sliding_attention.cos",
                position, local_cos, model::kLocalHeadSize);
    capture_row(captures,
                "boundary.hybrid.rows_1022_1025.rotary."
                "sliding_attention.sin",
                position, local_sin, model::kLocalHeadSize);
    capture_row(captures,
                "boundary.hybrid.rows_1022_1025.rotary.full_attention.cos",
                position, global_cos, model::kGlobalHeadSize);
    capture_row(captures,
                "boundary.hybrid.rows_1022_1025.rotary.full_attention.sin",
                position, global_sin, model::kGlobalHeadSize);

    primitives::embedding_lookup(
        weights_.pointer(model::kEmbeddingPhysicalId),
        boundary_input_token(position), h0);
    capture_row(captures, "boundary.hybrid.rows_1022_1025.embedding",
                position, h0, model::kHiddenSize);

    for (std::uint32_t layer = 0; layer < model::kLayerCount; ++layer) {
      const bool global = model::is_global_layer(layer);
      const model::AttentionKind kind =
          global ? model::AttentionKind::global : model::AttentionKind::local;
      const std::uint32_t head_size =
          global ? model::kGlobalHeadSize : model::kLocalHeadSize;
      const std::uint32_t kv_heads =
          global ? model::kGlobalKvHeadCount : model::kLocalKvHeadCount;
      const std::uint32_t q_width = model::kQueryHeadCount * head_size;
      const std::uint32_t kv_width = kv_heads * head_size;
      const LayerWeights& weight = layers_[layer];
      const LayerCacheView cache = caches_.layer(layer);
      const BFloat16* const cosine = global ? global_cos : local_cos;
      const BFloat16* const sine = global ? global_sin : local_sin;
      const bool deep = layer == 0 || layer == 5;

      primitives::rms_norm(h0, weight.input_norm, h1, 1,
                           model::kHiddenSize);
      const LinearPlan& q_plan = global ? decode_global_q_ : decode_local_q_;
      const LinearPlan& kv_plan =
          global ? decode_global_kv_ : decode_local_kv_;
      q_plan.run(handle_.get(), h1, weight.q_proj, q_raw);
      kv_plan.run(handle_.get(), h1, weight.k_proj, k_raw);
      if (!global) {
        kv_plan.run(handle_.get(), h1, weight.v_proj, v_raw);
      }
      if (deep) {
        capture_layer_row(captures, layer, "q_raw", position, q_raw,
                          q_width);
        capture_layer_row(captures, layer, "k_raw", position, k_raw,
                          kv_width);
        capture_layer_row(captures, layer, "v_raw", position,
                          global ? k_raw : v_raw, kv_width);
      }

      primitives::rms_norm(q_raw, weight.q_norm, q_norm,
                           model::kQueryHeadCount, head_size);
      primitives::rms_norm(k_raw, weight.k_norm, k_norm, kv_heads,
                           head_size);
      primitives::rms_norm_unscaled(global ? k_raw : v_raw, v_norm,
                                    kv_heads, head_size);
      if (deep) {
        capture_layer_row(captures, layer, "q_norm", position, q_norm,
                          q_width);
        capture_layer_row(captures, layer, "k_norm", position, k_norm,
                          kv_width);
        capture_layer_row(captures, layer, "v_norm", position, v_norm,
                          kv_width);
      }

      primitives::apply_rope_m1(q_norm, cosine, sine, q_rope,
                                model::kQueryHeadCount, kind);
      primitives::apply_rope_m1(k_norm, cosine, sine, k_rope, kv_heads,
                                kind);
      if (deep) {
        capture_layer_row(captures, layer, "q_rope", position, q_rope,
                          q_width);
        capture_layer_row(captures, layer, "k_rope", position, k_rope,
                          kv_width);
      }

      primitives::write_kv_cache_m1(k_rope, v_norm, cache.key, cache.value,
                                    position, cache.capacity, kind);
      primitives::causal_gqa_attention_cached_m1_boundary(
          q_rope, cache.key, cache.value, position, score_scratch,
          probabilities, context, kind);
      if (deep) {
        capture_layer_row(captures, layer, "attention_probabilities",
                          position, probabilities,
                          model::kQueryHeadCount * kBoundaryPrefixTokens);
        capture_layer_row(captures, layer, "attention_context", position,
                          context, q_width);
      }

      const LinearPlan& o_plan = global ? decode_global_o_ : decode_local_o_;
      o_plan.run(handle_.get(), context, weight.o_proj, h2);
      if (deep) {
        capture_layer_row(captures, layer, "attention_output", position, h2,
                          model::kHiddenSize);
      }
      primitives::rms_norm(h2, weight.post_attention_norm, h1, 1,
                           model::kHiddenSize);
      primitives::residual_add(h0, h1, h2, model::kHiddenSize);
      primitives::rms_norm(h2, weight.pre_feedforward_norm, h1, 1,
                           model::kHiddenSize);
      decode_hidden_to_mlp_.run(handle_.get(), h1, weight.gate_proj, gate);
      decode_hidden_to_mlp_.run(handle_.get(), h1, weight.up_proj, up);
      primitives::gelu_tanh_multiply(gate, up, product, model::kMlpSize);
      decode_mlp_to_hidden_.run(handle_.get(), product, weight.down_proj, h0);
      primitives::rms_norm(h0, weight.post_feedforward_norm, h1, 1,
                           model::kHiddenSize);
      primitives::residual_add(h2, h1, h0, model::kHiddenSize);
      primitives::trained_scalar(h0, weight.layer_scalar,
                                 model::kHiddenSize);
      capture_layer_row(captures, layer, "output", position, h0,
                        model::kHiddenSize);
    }

    primitives::rms_norm(
        h0, weights_.pointer(model::kFinalNormPhysicalId), h1, 1,
        model::kHiddenSize);
    capture_row(captures, "boundary.hybrid.rows_1022_1025.final_norm",
                position, h1, model::kHiddenSize);
    const bool final = position == kBoundaryDecisionPosition;
    return predict(
        h1, final ? "boundary.hybrid.position_1025.logits.pre_softcap"
                  : nullptr,
        final ? "boundary.hybrid.position_1025.logits.post_softcap"
                  : nullptr,
        captures);
  }

  std::uint32_t predict(const BFloat16* final_norm,
                        const char* pre_softcap_name,
                        const char* post_softcap_name,
                        BoundaryCapturePlan* captures = nullptr) {
    BFloat16* const logits =
        at<BFloat16>(HybridBoundaryScratchLayout::kLogits);
    BFloat16* const capped_logits =
        at<BFloat16>(HybridBoundaryScratchLayout::kCappedLogits);
    std::uint32_t* const argmax =
        at<std::uint32_t>(HybridBoundaryScratchLayout::kArgmax);
    lm_head_.run(handle_.get(), final_norm,
                 weights_.pointer(model::kLmHeadLogicalId), logits);
    if (pre_softcap_name != nullptr && captures != nullptr) {
      captures->copy_device(pre_softcap_name, logits);
    }
    primitives::softcap_and_argmax(logits, capped_logits, argmax,
                                   model::kVocabSize, 30.0F);
    if (post_softcap_name != nullptr && captures != nullptr) {
      captures->copy_device(post_softcap_name, capped_logits);
    }
    std::uint32_t result = 0;
    check_cuda(cudaMemcpy(&result, argmax, sizeof(result),
                          cudaMemcpyDeviceToHost),
               "copy hybrid local-boundary argmax");
    return result;
  }

  template <typename T>
  T* at(std::size_t byte_offset) const {
    return reinterpret_cast<T*>(static_cast<std::uint8_t*>(scratch_.data()) +
                                byte_offset);
  }

  void capture_prefill_rows(BoundaryCapturePlan* captures,
                            std::string_view name, const BFloat16* values,
                            std::size_t row_elements) {
    if (captures == nullptr) {
      return;
    }
    for (std::uint32_t position = kBoundaryCaptureFirstPosition;
         position < kHybridPrefillTokens; ++position) {
      captures->copy_row(
          name, position,
          values + static_cast<std::size_t>(position) * row_elements,
          row_elements);
    }
  }

  void capture_prefill_layer_rows(BoundaryCapturePlan* captures,
                                  std::uint32_t layer,
                                  std::string_view field,
                                  const BFloat16* values,
                                  std::size_t row_elements) {
    if (captures == nullptr) {
      return;
    }
    char name[160];
    const int length = std::snprintf(
        name, sizeof(name),
        "boundary.hybrid.rows_1022_1025.layer.%02u.%.*s", layer,
        static_cast<int>(field.size()), field.data());
    if (length < 0 || static_cast<std::size_t>(length) >= sizeof(name)) {
      fail("hybrid boundary capture name",
           "formatted prefill layer name exceeds buffer");
    }
    capture_prefill_rows(
        captures, std::string_view(name, static_cast<std::size_t>(length)),
        values, row_elements);
  }

  void capture_prefill_head_major_layer_rows(
      BoundaryCapturePlan* captures, std::uint32_t layer,
      std::string_view field, const BFloat16* values, std::uint32_t heads,
      std::uint32_t head_size) {
    if (captures == nullptr) {
      return;
    }
    char name[160];
    const int length = std::snprintf(
        name, sizeof(name),
        "boundary.hybrid.rows_1022_1025.layer.%02u.%.*s", layer,
        static_cast<int>(field.size()), field.data());
    if (length < 0 || static_cast<std::size_t>(length) >= sizeof(name)) {
      fail("hybrid boundary capture name",
           "formatted head-major layer name exceeds buffer");
    }
    BFloat16* const staging =
        at<BFloat16>(HybridBoundaryScratchLayout::kGatherStaging);
    const std::size_t row_bytes = head_size * sizeof(BFloat16);
    const std::size_t source_pitch =
        static_cast<std::size_t>(kHybridPrefillTokens) * row_bytes;
    for (std::uint32_t position = kBoundaryCaptureFirstPosition;
         position < kHybridPrefillTokens; ++position) {
      check_cuda(
          cudaMemcpy2DAsync(
              staging, row_bytes,
              values + static_cast<std::size_t>(position) * head_size,
              source_pitch, row_bytes, heads, cudaMemcpyDeviceToDevice),
          "gather hybrid prefill head-major capture row");
      captures->copy_row(
          std::string_view(name, static_cast<std::size_t>(length)), position,
          staging, static_cast<std::size_t>(heads) * head_size);
    }
  }

  void capture_prefill_probability_rows(BoundaryCapturePlan* captures,
                                        std::uint32_t layer,
                                        const BFloat16* probabilities) {
    if (captures == nullptr) {
      return;
    }
    char name[160];
    const int length = std::snprintf(
        name, sizeof(name),
        "boundary.hybrid.rows_1022_1025.layer.%02u."
        "attention_probabilities",
        layer);
    if (length < 0 || static_cast<std::size_t>(length) >= sizeof(name)) {
      fail("hybrid boundary capture name",
           "formatted probability name exceeds buffer");
    }
    BFloat16* const staging =
        at<BFloat16>(HybridBoundaryScratchLayout::kGatherStaging);
    constexpr std::size_t destination_pitch =
        static_cast<std::size_t>(kBoundaryPrefixTokens) * sizeof(BFloat16);
    constexpr std::size_t source_pitch =
        static_cast<std::size_t>(kHybridPrefillTokens) *
        kHybridPrefillTokens * sizeof(BFloat16);
    constexpr std::size_t copied_width =
        static_cast<std::size_t>(kHybridPrefillTokens) * sizeof(BFloat16);
    constexpr std::size_t padded_width =
        static_cast<std::size_t>(kBoundaryPrefixTokens -
                                 kHybridPrefillTokens) *
        sizeof(BFloat16);
    for (std::uint32_t position = kBoundaryCaptureFirstPosition;
         position < kHybridPrefillTokens; ++position) {
      check_cuda(
          cudaMemcpy2DAsync(
              staging, destination_pitch,
              probabilities + static_cast<std::size_t>(position) *
                                  kHybridPrefillTokens,
              source_pitch, copied_width, model::kQueryHeadCount,
              cudaMemcpyDeviceToDevice),
          "gather hybrid prefill probability capture row");
      check_cuda(
          cudaMemset2DAsync(staging + kHybridPrefillTokens,
                            destination_pitch, 0, padded_width,
                            model::kQueryHeadCount),
          "zero-pad hybrid prefill probability capture row");
      captures->copy_row(
          std::string_view(name, static_cast<std::size_t>(length)), position,
          staging, HybridBoundaryScratchLayout::kGatherElements);
    }
  }

  static void capture_row(BoundaryCapturePlan* captures,
                          std::string_view name, std::uint32_t position,
                          const BFloat16* values, std::size_t elements) {
    if (captures != nullptr) {
      captures->copy_row(name, position, values, elements);
    }
  }

  static void capture_layer_row(BoundaryCapturePlan* captures,
                                std::uint32_t layer, std::string_view field,
                                std::uint32_t position,
                                const BFloat16* values,
                                std::size_t elements) {
    if (captures == nullptr) {
      return;
    }
    char name[160];
    const int length = std::snprintf(
        name, sizeof(name),
        "boundary.hybrid.rows_1022_1025.layer.%02u.%.*s", layer,
        static_cast<int>(field.size()), field.data());
    if (length < 0 || static_cast<std::size_t>(length) >= sizeof(name)) {
      fail("hybrid boundary capture name",
           "formatted decode layer name exceeds buffer");
    }
    captures->copy_row(
        std::string_view(name, static_cast<std::size_t>(length)), position,
        values, elements);
  }

  const WeightArena& weights_;
  DeviceAllocation scratch_;
  BoundaryKvCaches caches_;
  LtHandle handle_;
  LinearPlan prefill_local_q_;
  LinearPlan prefill_local_kv_;
  LinearPlan prefill_global_q_;
  LinearPlan prefill_global_kv_;
  LinearPlan prefill_local_o_;
  LinearPlan prefill_global_o_;
  LinearPlan prefill_hidden_to_mlp_;
  LinearPlan prefill_mlp_to_hidden_;
  LinearPlan decode_local_q_;
  LinearPlan decode_local_kv_;
  LinearPlan decode_global_q_;
  LinearPlan decode_global_kv_;
  LinearPlan decode_local_o_;
  LinearPlan decode_global_o_;
  LinearPlan decode_hidden_to_mlp_;
  LinearPlan decode_mlp_to_hidden_;
  LinearPlan lm_head_;
  std::array<LayerWeights, model::kLayerCount> layers_{};
};


struct GraphBuildMetrics {
  double capture_wall_seconds{};
  double instantiate_wall_seconds{};
  double upload_wall_seconds{};
  std::size_t node_count{};
  std::size_t free_before_upload{};
  std::size_t free_after_upload{};
  std::size_t total_device_bytes{};
};

struct GraphRequestResult {
  std::array<std::uint32_t, kGraphOutputTokens> outputs{};
  std::uint32_t final_position{};
  float prefill_gpu_milliseconds{};
  double prefill_wall_seconds{};
  float decode_gpu_milliseconds{};
  double decode_wall_seconds{};
};

struct GraphProfileResult {
  std::array<std::uint32_t, kProfileDecodeReplays + 1> outputs{};
  std::uint32_t initial_position{};
  std::uint32_t final_position{};
  float prefill_gpu_milliseconds{};
  double prefill_wall_seconds{};
  float decode_gpu_milliseconds{};
  double decode_wall_seconds{};
};

class GraphDecodeEngine {
 public:
  explicit GraphDecodeEngine(const WeightArena& weights)
      : weights_(weights),
        scratch_(HybridBoundaryScratchLayout::kBytes),
        current_token_(sizeof(std::uint32_t)),
        absolute_position_(sizeof(std::uint32_t)),
        outputs_(kGraphOutputBytes),
        prefill_local_q_(kGraphPromptTokens, model::kHiddenSize,
                         model::kQueryHeadCount * model::kLocalHeadSize),
        prefill_local_kv_(kGraphPromptTokens, model::kHiddenSize,
                          model::kLocalKvHeadCount * model::kLocalHeadSize),
        prefill_global_q_(kGraphPromptTokens, model::kHiddenSize,
                          model::kQueryHeadCount * model::kGlobalHeadSize),
        prefill_global_kv_(kGraphPromptTokens, model::kHiddenSize,
                           model::kGlobalKvHeadCount * model::kGlobalHeadSize),
        prefill_local_o_(kGraphPromptTokens,
                         model::kQueryHeadCount * model::kLocalHeadSize,
                         model::kHiddenSize),
        prefill_global_o_(kGraphPromptTokens,
                          model::kQueryHeadCount * model::kGlobalHeadSize,
                          model::kHiddenSize),
        prefill_hidden_to_mlp_(kGraphPromptTokens, model::kHiddenSize,
                               model::kMlpSize),
        prefill_mlp_to_hidden_(kGraphPromptTokens, model::kMlpSize,
                               model::kHiddenSize),
        decode_local_q_(1, model::kHiddenSize,
                        model::kQueryHeadCount * model::kLocalHeadSize),
        decode_local_kv_(1, model::kHiddenSize,
                         model::kLocalKvHeadCount * model::kLocalHeadSize),
        decode_global_q_(1, model::kHiddenSize,
                         model::kQueryHeadCount * model::kGlobalHeadSize),
        decode_global_kv_(1, model::kHiddenSize,
                          model::kGlobalKvHeadCount * model::kGlobalHeadSize),
        decode_local_o_(1,
                        model::kQueryHeadCount * model::kLocalHeadSize,
                        model::kHiddenSize),
        decode_global_o_(1,
                         model::kQueryHeadCount * model::kGlobalHeadSize,
                         model::kHiddenSize),
        decode_hidden_to_mlp_(1, model::kHiddenSize, model::kMlpSize),
        decode_mlp_to_hidden_(1, model::kMlpSize, model::kHiddenSize),
        lm_head_(1, model::kHiddenSize, model::kVocabSize) {
    for (std::size_t layer = 0; layer < model::kLayerCount; ++layer) {
      const LayerWeightIds& ids = kWeightIds[layer];
      layers_[layer] = {
          weights.pointer(ids.input_norm),
          weights.pointer(ids.q_proj),
          weights.pointer(ids.k_proj),
          model::is_global_layer(layer) ? nullptr : weights.pointer(ids.v_proj),
          weights.pointer(ids.q_norm),
          weights.pointer(ids.k_norm),
          weights.pointer(ids.o_proj),
          weights.pointer(ids.post_attention_norm),
          weights.pointer(ids.pre_feedforward_norm),
          weights.pointer(ids.gate_proj),
          weights.pointer(ids.up_proj),
          weights.pointer(ids.down_proj),
          weights.pointer(ids.post_feedforward_norm),
          weights.pointer(ids.layer_scalar),
      };
    }
    const std::array<const void*, 4> arenas{{
        scratch_.data(),
        current_token_.data(),
        absolute_position_.data(),
        outputs_.data(),
    }};
    for (std::size_t left = 0; left < arenas.size(); ++left) {
      for (std::size_t right = left + 1; right < arenas.size(); ++right) {
        if (arenas[left] == arenas[right]) {
          fail("graph-decode arenas", "scratch or persistent state aliases");
        }
      }
    }
  }

  ~GraphDecodeEngine() {
    if (executable_ != nullptr) {
      cudaGraphExecDestroy(executable_);
    }
    if (graph_ != nullptr) {
      cudaGraphDestroy(graph_);
    }
  }

  GraphDecodeEngine(const GraphDecodeEngine&) = delete;
  GraphDecodeEngine& operator=(const GraphDecodeEngine&) = delete;

  [[nodiscard]] std::size_t scratch_bytes() const { return scratch_.size(); }
  [[nodiscard]] std::size_t local_cache_bytes() const {
    return caches_.local_bytes();
  }
  [[nodiscard]] std::size_t global_cache_bytes() const {
    return caches_.global_bytes();
  }
  [[nodiscard]] std::size_t persistent_state_bytes() const {
    return current_token_.size() + absolute_position_.size() + outputs_.size();
  }

  GraphBuildMetrics build_graph() {
    if (graph_ != nullptr || executable_ != nullptr) {
      fail("graph-decode capture", "graph was already built");
    }

    // Warm every M=1024 and M=1 cuBLASLt plan before stream capture. The warm
    // decode mutates position 1024, so rebuild the prompt caches and seed state
    // again before capturing the actual body.
    prefill_and_seed();
    decode_one();
    check_cuda(cudaStreamSynchronize(stream_.get()),
               "synchronize graph-decode cuBLASLt warmup");
    prefill_and_seed();
    check_cuda(cudaStreamSynchronize(stream_.get()),
               "synchronize graph-decode capture seed");

    GraphBuildMetrics metrics;
    const auto capture_started = std::chrono::steady_clock::now();
    check_cuda(cudaStreamBeginCapture(stream_.get(),
                                      cudaStreamCaptureModeThreadLocal),
               "begin graph-decode stream capture");
    decode_one();
    check_cuda(cudaStreamEndCapture(stream_.get(), &graph_),
               "end graph-decode stream capture");
    metrics.capture_wall_seconds = seconds_since(capture_started);
    if (graph_ == nullptr) {
      fail("graph-decode capture", "CUDA returned a null graph");
    }
    check_cuda(cudaGraphGetNodes(graph_, nullptr, &metrics.node_count),
               "count graph-decode nodes");
    if (metrics.node_count != kGraphExpectedNodeCount) {
      fail("graph-decode capture",
           "captured graph node count is not the fixed 1078");
    }

    const auto instantiate_started = std::chrono::steady_clock::now();
    check_cuda(cudaGraphInstantiate(&executable_, graph_, 0),
               "instantiate graph-decode graph");
    metrics.instantiate_wall_seconds = seconds_since(instantiate_started);

    check_cuda(cudaMemGetInfo(&metrics.free_before_upload,
                              &metrics.total_device_bytes),
               "cudaMemGetInfo before graph-decode upload");
    const auto upload_started = std::chrono::steady_clock::now();
    check_cuda(cudaGraphUpload(executable_, stream_.get()),
               "upload graph-decode graph");
    check_cuda(cudaStreamSynchronize(stream_.get()),
               "synchronize graph-decode upload");
    metrics.upload_wall_seconds = seconds_since(upload_started);
    std::size_t total_after = 0;
    check_cuda(cudaMemGetInfo(&metrics.free_after_upload, &total_after),
               "cudaMemGetInfo after graph-decode upload");
    if (total_after != metrics.total_device_bytes) {
      fail("graph-decode upload memory", "CUDA total memory changed");
    }
    return metrics;
  }

  GraphRequestResult run_uncaptured_request() {
    return run_request(false);
  }

  GraphRequestResult run_graphed_request() {
    if (executable_ == nullptr) {
      fail("graph-decode request", "graph has not been built");
    }
    return run_request(true);
  }

  GraphProfileResult run_profiled_step() {
    if (executable_ == nullptr) {
      fail("profile-decode request", "graph has not been built");
    }

    GraphProfileResult result;
    const auto prefill_wall_started = std::chrono::steady_clock::now();
    check_cuda(cudaEventRecord(prefill_begin_.get(), stream_.get()),
               "record profile-decode prefill start");
    prefill_and_seed();
    check_cuda(cudaEventRecord(prefill_end_.get(), stream_.get()),
               "record profile-decode prefill end");
    check_cuda(cudaEventSynchronize(prefill_end_.get()),
               "synchronize profile-decode prefill");
    result.prefill_wall_seconds = seconds_since(prefill_wall_started);
    check_cuda(cudaEventElapsedTime(&result.prefill_gpu_milliseconds,
                                    prefill_begin_.get(), prefill_end_.get()),
               "measure profile-decode prefill");
    check_cuda(cudaMemcpyAsync(&result.initial_position,
                               absolute_position_.data(),
                               sizeof(result.initial_position),
                               cudaMemcpyDeviceToHost, stream_.get()),
               "copy profile-decode initial position");
    check_cuda(cudaStreamSynchronize(stream_.get()),
               "synchronize profile-decode initial position");

    // Keep setup, result copies, and every allocation outside the profiler
    // range. CUDA graph-node tracing can distort event elapsed time, so the
    // profiler's kernel span remains the authoritative profiled GPU interval.
    check_cuda(cudaProfilerStart(), "start profile-decode profiler range");
    check_cuda(cudaEventRecord(decode_begin_.get(), stream_.get()),
               "record profile-decode graph start");
    const auto decode_wall_started = std::chrono::steady_clock::now();
    check_cuda(cudaGraphLaunch(executable_, stream_.get()),
               "launch profiled graph-decode step");
    check_cuda(cudaEventRecord(decode_end_.get(), stream_.get()),
               "record profile-decode graph end");
    check_cuda(cudaEventSynchronize(decode_end_.get()),
               "synchronize profiled graph-decode step");
    result.decode_wall_seconds = seconds_since(decode_wall_started);
    check_cuda(cudaProfilerStop(), "stop profile-decode profiler range");
    check_cuda(cudaEventElapsedTime(&result.decode_gpu_milliseconds,
                                    decode_begin_.get(), decode_end_.get()),
               "measure profiled graph-decode step");

    check_cuda(cudaMemcpyAsync(result.outputs.data(), outputs_.data(),
                               sizeof(result.outputs), cudaMemcpyDeviceToHost,
                               stream_.get()),
               "copy profile-decode output IDs");
    check_cuda(cudaMemcpyAsync(&result.final_position,
                               absolute_position_.data(),
                               sizeof(result.final_position),
                               cudaMemcpyDeviceToHost, stream_.get()),
               "copy profile-decode final position");
    check_cuda(cudaStreamSynchronize(stream_.get()),
               "synchronize profile-decode result copies");
    return result;
  }

 private:
  GraphRequestResult run_request(bool graphed) {
    GraphRequestResult result;
    const auto prefill_wall_started = std::chrono::steady_clock::now();
    check_cuda(cudaEventRecord(prefill_begin_.get(), stream_.get()),
               "record graph-decode prefill start");
    prefill_and_seed();
    check_cuda(cudaEventRecord(prefill_end_.get(), stream_.get()),
               "record graph-decode prefill end");
    check_cuda(cudaEventSynchronize(prefill_end_.get()),
               "synchronize graph-decode prefill");
    result.prefill_wall_seconds = seconds_since(prefill_wall_started);
    check_cuda(cudaEventElapsedTime(&result.prefill_gpu_milliseconds,
                                    prefill_begin_.get(), prefill_end_.get()),
               "measure graph-decode prefill");

    const auto decode_wall_started = std::chrono::steady_clock::now();
    check_cuda(cudaEventRecord(decode_begin_.get(), stream_.get()),
               "record graph-decode decode start");
    if (graphed) {
      for (std::uint32_t replay = 0; replay < kGraphDecodeReplays; ++replay) {
        check_cuda(cudaGraphLaunch(executable_, stream_.get()),
                   "launch graph-decode graph");
      }
    } else {
      for (std::uint32_t step = 0; step < kGraphDecodeReplays; ++step) {
        decode_one();
      }
    }
    check_cuda(cudaEventRecord(decode_end_.get(), stream_.get()),
               "record graph-decode decode end");
    check_cuda(cudaEventSynchronize(decode_end_.get()),
               "synchronize graph-decode decode");
    result.decode_wall_seconds = seconds_since(decode_wall_started);
    check_cuda(cudaEventElapsedTime(&result.decode_gpu_milliseconds,
                                    decode_begin_.get(), decode_end_.get()),
               "measure graph-decode decode");

    check_cuda(cudaMemcpyAsync(result.outputs.data(), outputs_.data(),
                               kGraphOutputBytes, cudaMemcpyDeviceToHost,
                               stream_.get()),
               "copy graph-decode output IDs");
    check_cuda(cudaMemcpyAsync(&result.final_position,
                               absolute_position_.data(),
                               sizeof(result.final_position),
                               cudaMemcpyDeviceToHost, stream_.get()),
               "copy graph-decode final position");
    check_cuda(cudaStreamSynchronize(stream_.get()),
               "synchronize graph-decode result copies");
    return result;
  }

  void prefill_and_seed() {
    BFloat16* const h0 = at<BFloat16>(HybridBoundaryScratchLayout::kH0);
    BFloat16* const h1 = at<BFloat16>(HybridBoundaryScratchLayout::kH1);
    BFloat16* const h2 = at<BFloat16>(HybridBoundaryScratchLayout::kH2);
    BFloat16* const q_raw =
        at<BFloat16>(HybridBoundaryScratchLayout::kQueryRaw);
    BFloat16* const q_norm =
        at<BFloat16>(HybridBoundaryScratchLayout::kQueryNorm);
    BFloat16* const q_rope =
        at<BFloat16>(HybridBoundaryScratchLayout::kQueryRope);
    BFloat16* const k_raw =
        at<BFloat16>(HybridBoundaryScratchLayout::kKeyRaw);
    BFloat16* const k_norm =
        at<BFloat16>(HybridBoundaryScratchLayout::kKeyNorm);
    BFloat16* const k_rope =
        at<BFloat16>(HybridBoundaryScratchLayout::kKeyRope);
    BFloat16* const v_raw =
        at<BFloat16>(HybridBoundaryScratchLayout::kValueRaw);
    BFloat16* const v_norm =
        at<BFloat16>(HybridBoundaryScratchLayout::kValueNorm);
    BFloat16* const context =
        at<BFloat16>(HybridBoundaryScratchLayout::kContext);
    BFloat16* const gate = at<BFloat16>(HybridBoundaryScratchLayout::kGate);
    BFloat16* const up = at<BFloat16>(HybridBoundaryScratchLayout::kUp);
    BFloat16* const product =
        at<BFloat16>(HybridBoundaryScratchLayout::kProduct);
    BFloat16* const logits =
        at<BFloat16>(HybridBoundaryScratchLayout::kLogits);
    BFloat16* const capped_logits =
        at<BFloat16>(HybridBoundaryScratchLayout::kCappedLogits);
    BFloat16* const local_cos =
        at<BFloat16>(HybridBoundaryScratchLayout::kLocalCos);
    BFloat16* const local_sin =
        at<BFloat16>(HybridBoundaryScratchLayout::kLocalSin);
    BFloat16* const global_cos =
        at<BFloat16>(HybridBoundaryScratchLayout::kGlobalCos);
    BFloat16* const global_sin =
        at<BFloat16>(HybridBoundaryScratchLayout::kGlobalSin);
    BFloat16* const score_scratch =
        at<BFloat16>(HybridBoundaryScratchLayout::kScoreScratch);
    BFloat16* const probabilities =
        at<BFloat16>(HybridBoundaryScratchLayout::kProbabilities);
    std::uint32_t* const argmax =
        at<std::uint32_t>(HybridBoundaryScratchLayout::kArgmax);
    const cudaStream_t stream = stream_.get();

    prefill::generate_rope_factors_m1024(local_cos, local_sin, global_cos,
                                         global_sin, stream);
    for (std::uint32_t position = 0; position < kGraphPromptTokens;
         ++position) {
      primitives::embedding_lookup(
          weights_.pointer(model::kEmbeddingPhysicalId),
          boundary_input_token(position),
          h0 + static_cast<std::size_t>(position) * model::kHiddenSize,
          stream);
    }

    for (std::uint32_t layer = 0; layer < model::kLayerCount; ++layer) {
      const bool global = model::is_global_layer(layer);
      const model::AttentionKind kind =
          global ? model::AttentionKind::global : model::AttentionKind::local;
      const std::uint32_t head_size =
          global ? model::kGlobalHeadSize : model::kLocalHeadSize;
      const std::uint32_t kv_heads =
          global ? model::kGlobalKvHeadCount : model::kLocalKvHeadCount;
      const LayerWeights& weight = layers_[layer];
      const LayerCacheView cache = caches_.layer(layer);
      const BFloat16* const cosine = global ? global_cos : local_cos;
      const BFloat16* const sine = global ? global_sin : local_sin;

      primitives::rms_norm(h0, weight.input_norm, h1, kGraphPromptTokens,
                           model::kHiddenSize, 1.0e-6F, stream);
      const LinearPlan& q_plan =
          global ? prefill_global_q_ : prefill_local_q_;
      const LinearPlan& kv_plan =
          global ? prefill_global_kv_ : prefill_local_kv_;
      q_plan.run(handle_.get(), h1, weight.q_proj, q_raw, stream);
      kv_plan.run(handle_.get(), h1, weight.k_proj, k_raw, stream);
      if (!global) {
        kv_plan.run(handle_.get(), h1, weight.v_proj, v_raw, stream);
      }
      primitives::rms_norm(q_raw, weight.q_norm, q_norm,
                           kGraphPromptTokens * model::kQueryHeadCount,
                           head_size, 1.0e-6F, stream);
      primitives::rms_norm(k_raw, weight.k_norm, k_norm,
                           kGraphPromptTokens * kv_heads, head_size,
                           1.0e-6F, stream);
      primitives::rms_norm_unscaled(global ? k_raw : v_raw, v_norm,
                                    kGraphPromptTokens * kv_heads, head_size,
                                    1.0e-6F, stream);
      prefill::apply_rope_transpose_m1024(
          q_norm, cosine, sine, q_rope, model::kQueryHeadCount, kind, stream);
      prefill::apply_rope_transpose_m1024(
          k_norm, cosine, sine, k_rope, kv_heads, kind, stream);
      prefill::write_kv_cache_m1024(k_rope, v_norm, cache.key, cache.value,
                                    cache.capacity, kind, stream);
      prefill::causal_gqa_attention_m1024(
          q_rope, k_rope, v_norm, score_scratch, probabilities, context, kind,
          stream);

      const LinearPlan& o_plan =
          global ? prefill_global_o_ : prefill_local_o_;
      o_plan.run(handle_.get(), context, weight.o_proj, h2, stream);
      primitives::rms_norm(h2, weight.post_attention_norm, h1,
                           kGraphPromptTokens, model::kHiddenSize, 1.0e-6F,
                           stream);
      primitives::residual_add(
          h0, h1, h2, HybridBoundaryScratchLayout::kPrefillHiddenElements,
          stream);
      primitives::rms_norm(h2, weight.pre_feedforward_norm, h1,
                           kGraphPromptTokens, model::kHiddenSize, 1.0e-6F,
                           stream);
      prefill_hidden_to_mlp_.run(handle_.get(), h1, weight.gate_proj, gate,
                                 stream);
      prefill_hidden_to_mlp_.run(handle_.get(), h1, weight.up_proj, up,
                                 stream);
      primitives::gelu_tanh_multiply(
          gate, up, product,
          HybridBoundaryScratchLayout::kPrefillMlpElements, stream);
      prefill_mlp_to_hidden_.run(handle_.get(), product, weight.down_proj, h0,
                                 stream);
      primitives::rms_norm(h0, weight.post_feedforward_norm, h1,
                           kGraphPromptTokens, model::kHiddenSize, 1.0e-6F,
                           stream);
      primitives::residual_add(
          h2, h1, h0, HybridBoundaryScratchLayout::kPrefillHiddenElements,
          stream);
      primitives::trained_scalar(
          h0, weight.layer_scalar,
          HybridBoundaryScratchLayout::kPrefillHiddenElements, stream);
    }

    primitives::rms_norm(
        h0, weights_.pointer(model::kFinalNormPhysicalId), h1,
        kGraphPromptTokens, model::kHiddenSize, 1.0e-6F, stream);
    const BFloat16* const final_prompt_row =
        h1 + static_cast<std::size_t>(kGraphPromptTokens - 1) *
                 model::kHiddenSize;
    lm_head_.run(handle_.get(), final_prompt_row,
                 weights_.pointer(model::kLmHeadLogicalId), logits, stream);
    primitives::softcap_and_argmax(logits, capped_logits, argmax,
                                   model::kVocabSize, 30.0F, stream);
    primitives::seed_graph_decode_state(
        argmax, static_cast<std::uint32_t*>(current_token_.data()),
        static_cast<std::uint32_t*>(absolute_position_.data()),
        static_cast<std::uint32_t*>(outputs_.data()), stream);
  }

  void decode_one() {
    BFloat16* const h0 = at<BFloat16>(HybridBoundaryScratchLayout::kH0);
    BFloat16* const h1 = at<BFloat16>(HybridBoundaryScratchLayout::kH1);
    BFloat16* const h2 = at<BFloat16>(HybridBoundaryScratchLayout::kH2);
    BFloat16* const q_raw =
        at<BFloat16>(HybridBoundaryScratchLayout::kQueryRaw);
    BFloat16* const q_norm =
        at<BFloat16>(HybridBoundaryScratchLayout::kQueryNorm);
    BFloat16* const q_rope =
        at<BFloat16>(HybridBoundaryScratchLayout::kQueryRope);
    BFloat16* const k_raw =
        at<BFloat16>(HybridBoundaryScratchLayout::kKeyRaw);
    BFloat16* const k_norm =
        at<BFloat16>(HybridBoundaryScratchLayout::kKeyNorm);
    BFloat16* const k_rope =
        at<BFloat16>(HybridBoundaryScratchLayout::kKeyRope);
    BFloat16* const v_raw =
        at<BFloat16>(HybridBoundaryScratchLayout::kValueRaw);
    BFloat16* const v_norm =
        at<BFloat16>(HybridBoundaryScratchLayout::kValueNorm);
    BFloat16* const context =
        at<BFloat16>(HybridBoundaryScratchLayout::kContext);
    BFloat16* const gate = at<BFloat16>(HybridBoundaryScratchLayout::kGate);
    BFloat16* const up = at<BFloat16>(HybridBoundaryScratchLayout::kUp);
    BFloat16* const product =
        at<BFloat16>(HybridBoundaryScratchLayout::kProduct);
    BFloat16* const logits =
        at<BFloat16>(HybridBoundaryScratchLayout::kLogits);
    BFloat16* const capped_logits =
        at<BFloat16>(HybridBoundaryScratchLayout::kCappedLogits);
    BFloat16* const local_cos =
        at<BFloat16>(HybridBoundaryScratchLayout::kLocalCos);
    BFloat16* const local_sin =
        at<BFloat16>(HybridBoundaryScratchLayout::kLocalSin);
    BFloat16* const global_cos =
        at<BFloat16>(HybridBoundaryScratchLayout::kGlobalCos);
    BFloat16* const global_sin =
        at<BFloat16>(HybridBoundaryScratchLayout::kGlobalSin);
    void* const attention_scratch = static_cast<void*>(
        at<BFloat16>(HybridBoundaryScratchLayout::kScoreScratch));
    std::uint32_t* const argmax =
        at<std::uint32_t>(HybridBoundaryScratchLayout::kArgmax);
    auto* const current_token =
        static_cast<std::uint32_t*>(current_token_.data());
    auto* const absolute_position =
        static_cast<std::uint32_t*>(absolute_position_.data());
    auto* const outputs = static_cast<std::uint32_t*>(outputs_.data());
    const cudaStream_t stream = stream_.get();

    primitives::embedding_lookup_device_token(
        weights_.pointer(model::kEmbeddingPhysicalId), current_token, h0,
        stream);
    primitives::generate_rope_factors_m1_device_position(
        local_cos, local_sin, global_cos, global_sin, absolute_position,
        stream);
    primitives::rms_norm_hidden_m1(h0, layers_[0].input_norm, h1, stream);

    for (std::uint32_t layer = 0; layer < model::kLayerCount; ++layer) {
      const bool global = model::is_global_layer(layer);
      const model::AttentionKind kind =
          global ? model::AttentionKind::global : model::AttentionKind::local;
      const std::uint32_t head_size =
          global ? model::kGlobalHeadSize : model::kLocalHeadSize;
      const std::uint32_t kv_heads =
          global ? model::kGlobalKvHeadCount : model::kLocalKvHeadCount;
      const LayerWeights& weight = layers_[layer];
      const LayerCacheView cache = caches_.layer(layer);
      const BFloat16* const cosine = global ? global_cos : local_cos;
      const BFloat16* const sine = global ? global_sin : local_sin;

      const LinearPlan& q_plan = global ? decode_global_q_ : decode_local_q_;
      const LinearPlan& kv_plan =
          global ? decode_global_kv_ : decode_local_kv_;
      q_plan.run(handle_.get(), h1, weight.q_proj, q_raw, stream);
      kv_plan.run(handle_.get(), h1, weight.k_proj, k_raw, stream);
      if (!global) {
        kv_plan.run(handle_.get(), h1, weight.v_proj, v_raw, stream);
      }
      primitives::rms_norm(q_raw, weight.q_norm, q_norm,
                           model::kQueryHeadCount, head_size, 1.0e-6F,
                           stream);
      primitives::rms_norm(k_raw, weight.k_norm, k_norm, kv_heads, head_size,
                           1.0e-6F, stream);
      primitives::rms_norm_unscaled(global ? k_raw : v_raw, v_norm, kv_heads,
                                    head_size, 1.0e-6F, stream);
      primitives::apply_rope_m1(q_norm, cosine, sine, q_rope,
                                model::kQueryHeadCount, kind, stream);
      primitives::apply_rope_m1(k_norm, cosine, sine, k_rope, kv_heads, kind,
                                stream);
      primitives::write_kv_cache_m1_device_position(
          k_rope, v_norm, cache.key, cache.value, absolute_position, kind,
          stream);
      primitives::causal_gqa_attention_cached_m1_device_position_fused(
          q_rope, cache.key, cache.value, absolute_position,
          attention_scratch, context, kind, stream);

      const LinearPlan& o_plan = global ? decode_global_o_ : decode_local_o_;
      o_plan.run(handle_.get(), context, weight.o_proj, h2, stream);
      primitives::post_attention_residual_pre_feedforward_norm_m1(
          h2, weight.post_attention_norm, h0,
          weight.pre_feedforward_norm, h1, stream);
      decode_hidden_to_mlp_.run(handle_.get(), h1, weight.gate_proj, gate,
                                stream);
      decode_hidden_to_mlp_.run(handle_.get(), h1, weight.up_proj, up,
                                stream);
      primitives::gelu_tanh_multiply(gate, up, product, model::kMlpSize,
                                     stream);
      decode_mlp_to_hidden_.run(handle_.get(), product, weight.down_proj, h0,
                                stream);
      const BFloat16* const next_norm_weight =
          layer + 1 < model::kLayerCount
              ? layers_[layer + 1].input_norm
              : weights_.pointer(model::kFinalNormPhysicalId);
      primitives::post_feedforward_residual_scalar_next_norm_m1(
          h0, weight.post_feedforward_norm, h2, weight.layer_scalar,
          next_norm_weight, h1, stream);
    }

    lm_head_.run(handle_.get(), h1,
                 weights_.pointer(model::kLmHeadLogicalId), logits, stream);
    primitives::softcap_and_argmax(logits, capped_logits, argmax,
                                   model::kVocabSize, 30.0F, stream);
    primitives::commit_graph_decode_state(argmax, current_token,
                                          absolute_position, outputs, stream);
  }

  template <typename T>
  T* at(std::size_t byte_offset) const {
    return reinterpret_cast<T*>(static_cast<std::uint8_t*>(scratch_.data()) +
                                byte_offset);
  }

  const WeightArena& weights_;
  NonblockingCudaStream stream_;
  DeviceAllocation scratch_;
  GraphDecodeKvCaches caches_;
  DeviceAllocation current_token_;
  DeviceAllocation absolute_position_;
  DeviceAllocation outputs_;
  CudaEvent prefill_begin_;
  CudaEvent prefill_end_;
  CudaEvent decode_begin_;
  CudaEvent decode_end_;
  LtHandle handle_;
  LinearPlan prefill_local_q_;
  LinearPlan prefill_local_kv_;
  LinearPlan prefill_global_q_;
  LinearPlan prefill_global_kv_;
  LinearPlan prefill_local_o_;
  LinearPlan prefill_global_o_;
  LinearPlan prefill_hidden_to_mlp_;
  LinearPlan prefill_mlp_to_hidden_;
  LinearPlan decode_local_q_;
  LinearPlan decode_local_kv_;
  LinearPlan decode_global_q_;
  LinearPlan decode_global_kv_;
  LinearPlan decode_local_o_;
  LinearPlan decode_global_o_;
  LinearPlan decode_hidden_to_mlp_;
  LinearPlan decode_mlp_to_hidden_;
  LinearPlan lm_head_;
  std::array<LayerWeights, model::kLayerCount> layers_{};
  cudaGraph_t graph_{nullptr};
  cudaGraphExec_t executable_{nullptr};
};


void validate_graph_request(const GraphRequestResult& result,
                            std::string_view label) {
  if (result.final_position != primitives::kGraphDecodeFinalPosition) {
    fail(label, "final device position is not 1535");
  }
  if (!std::equal(kGraphExpectedFirstThree.begin(),
                  kGraphExpectedFirstThree.end(), result.outputs.begin())) {
    fail(label, "first three output IDs do not match [236764,532,121160]");
  }
  if (!std::all_of(result.outputs.begin(), result.outputs.end(),
                   [](std::uint32_t token) {
                     return token < model::kVocabSize;
                   })) {
    fail(label, "an output ID is outside the Gemma 4 vocabulary");
  }
  if (!std::isfinite(result.prefill_gpu_milliseconds) ||
      result.prefill_gpu_milliseconds <= 0.0F ||
      !std::isfinite(result.prefill_wall_seconds) ||
      result.prefill_wall_seconds <= 0.0 ||
      !std::isfinite(result.decode_gpu_milliseconds) ||
      result.decode_gpu_milliseconds <= 0.0F ||
      !std::isfinite(result.decode_wall_seconds) ||
      result.decode_wall_seconds <= 0.0) {
    fail(label, "CUDA-event or wall timing is not finite and positive");
  }
}

void validate_graph_profile(const GraphProfileResult& result) {
  if (result.initial_position != primitives::kGraphDecodeFirstPosition) {
    fail("profile-decode validation", "initial device position is not 1024");
  }
  if (result.final_position != kProfileDecodeFinalPosition) {
    fail("profile-decode validation", "final device position is not 1025");
  }
  if (result.outputs != kProfileDecodeExpectedOutputs) {
    fail("profile-decode validation",
         "output IDs do not match [236764,532]");
  }
  if (!std::all_of(result.outputs.begin(), result.outputs.end(),
                   [](std::uint32_t token) {
                     return token < model::kVocabSize;
                   })) {
    fail("profile-decode validation",
         "an output ID is outside the Gemma 4 vocabulary");
  }
  if (!std::isfinite(result.prefill_gpu_milliseconds) ||
      result.prefill_gpu_milliseconds <= 0.0F ||
      !std::isfinite(result.prefill_wall_seconds) ||
      result.prefill_wall_seconds <= 0.0 ||
      !std::isfinite(result.decode_gpu_milliseconds) ||
      result.decode_gpu_milliseconds <= 0.0F ||
      !std::isfinite(result.decode_wall_seconds) ||
      result.decode_wall_seconds <= 0.0) {
    fail("profile-decode validation",
         "CUDA-event or wall timing is not finite and positive");
  }
}

void write_graph_outputs_exclusive(
    const std::string& output_directory,
    const std::array<std::uint32_t, kGraphOutputTokens>& outputs) {
  if (output_directory == "-") {
    return;
  }
  const std::filesystem::path directory(output_directory);
  if (directory.empty()) {
    fail("inspect graph-decode output directory", "path is empty");
  }
  std::error_code error;
  if (!std::filesystem::create_directory(directory, error)) {
    fail("create graph-decode output directory",
         error ? directory.string() + ": " + error.message()
               : directory.string() + " already exists");
  }
  write_exclusive(directory / "outputs.u32", outputs.data(),
                  kGraphOutputBytes);
}

void preflight_graph_output_directory(const std::string& output_directory) {
  if (output_directory == "-") {
    return;
  }
  const std::filesystem::path directory(output_directory);
  if (directory.empty()) {
    fail("inspect graph-decode output directory", "path is empty");
  }
  std::error_code error;
  const bool exists = std::filesystem::exists(directory, error);
  if (error) {
    fail("inspect graph-decode output directory",
         directory.string() + ": " + error.message());
  }
  if (exists) {
    fail("inspect graph-decode output directory",
         directory.string() + " already exists");
  }
  const std::filesystem::path parent =
      directory.has_parent_path() ? directory.parent_path()
                                  : std::filesystem::path(".");
  if (!std::filesystem::is_directory(parent, error)) {
    fail("inspect graph-decode output parent",
         error ? parent.string() + ": " + error.message()
               : parent.string() + " is not an existing directory");
  }
}

template <std::size_t Count>
double median(std::array<double, Count> values) {
  static_assert(Count % 2 == 1);
  std::sort(values.begin(), values.end());
  return values[Count / 2];
}

template <typename T, std::size_t Count>
void print_values(std::string_view name, const std::array<T, Count>& values) {
  console::field(name, values);
}

void print_device_resident_delta(std::string_view name,
                                 std::size_t free_before,
                                 std::size_t free_after) {
  const auto delta = free_before >= free_after
      ? static_cast<std::int64_t>(free_before - free_after)
      : -static_cast<std::int64_t>(free_after - free_before);
  console::field(name, delta);
}


}  // namespace

int run(const std::string& artifact_path, const std::string& capture_directory) {
  console::section("Pair inference diagnostic");
  console::field("pair_tokens", kInputTokens);
  console::field("artifact", artifact_path);
  console::field("verification", "disabled (header+table only)");
  artifact::ArtifactFile file = artifact::ArtifactFile::Open(artifact_path);
  console::field("payload_sha256", artifact::digest_hex(file.header().payload_sha256));

  validate_cuda_device();
  std::size_t free_before = 0;
  std::size_t total = 0;
  check_cuda(cudaMemGetInfo(&free_before, &total),
             "cudaMemGetInfo before pair load");

  const auto load_started = std::chrono::steady_clock::now();
  WeightArena weights(file);
  const double load_seconds = seconds_since(load_started);
  PairEngine engine(weights);
  std::size_t free_after = 0;
  std::size_t total_after = 0;
  check_cuda(cudaMemGetInfo(&free_after, &total_after),
             "cudaMemGetInfo after pair initialization");
  if (total_after != total || free_after > free_before) {
    fail("CUDA memory", "inconsistent cudaMemGetInfo result");
  }

  console::section("Weight load and GPU memory");
  console::field("weight_copy_seconds", load_seconds);
  console::field("device_weight_arena_bytes", weights.size());
  console::field("device_weight_arena_address_mod_4096", weights.address_mod_4096());
  console::field("device_scratch_arena_bytes", engine.scratch_bytes());
  console::field("gpu_total_bytes", total);
  console::field("gpu_free_before_bytes", free_before);
  console::field("gpu_free_after_initialization_bytes", free_after);
  console::field("gpu_initialization_delta_bytes", free_before - free_after);

  CudaEvent begin;
  CudaEvent end;
  const auto cold_started = std::chrono::steady_clock::now();
  check_cuda(cudaEventRecord(begin.get()), "record cold pair start event");
  const std::uint32_t token = engine.run(nullptr);
  check_cuda(cudaEventRecord(end.get()), "record cold pair end event");
  check_cuda(cudaEventSynchronize(end.get()), "synchronize cold pair event");
  float cold_milliseconds = 0.0F;
  check_cuda(cudaEventElapsedTime(&cold_milliseconds, begin.get(), end.get()),
             "measure cold pair inference");
  const double cold_wall_seconds = seconds_since(cold_started);

  const auto steady_started = std::chrono::steady_clock::now();
  check_cuda(cudaEventRecord(begin.get()), "record steady pair start event");
  const std::uint32_t steady_token = engine.run(nullptr);
  check_cuda(cudaEventRecord(end.get()), "record steady pair end event");
  check_cuda(cudaEventSynchronize(end.get()),
             "synchronize steady pair event");
  float steady_milliseconds = 0.0F;
  check_cuda(cudaEventElapsedTime(&steady_milliseconds, begin.get(), end.get()),
             "measure steady pair inference");
  const double steady_wall_seconds = seconds_since(steady_started);
  if (steady_token != token) {
    fail("steady pair replay", "argmax differs from cold run");
  }

  console::section("Inference timing and validation");
  console::field("cold_inference_gpu_milliseconds", cold_milliseconds);
  console::field("cold_inference_wall_seconds", cold_wall_seconds);
  console::field("steady_state_inference_gpu_milliseconds", steady_milliseconds);
  console::field("steady_state_inference_wall_seconds", steady_wall_seconds);
  console::field("argmax_token", token);
  console::field("expected_argmax_token", kExpectedToken);
  console::field("argmax_match", token == kExpectedToken);

  if (capture_directory != "-") {
    CapturePlan captures(capture_directory);
    const auto capture_started = std::chrono::steady_clock::now();
    const std::uint32_t capture_token = engine.run(&captures);
    const double capture_seconds = seconds_since(capture_started);
    if (capture_token != token) {
      fail("pair capture replay", "argmax differs from timed run");
    }
    captures.write(capture_token);
    console::section("Capture files");
    console::field("capture_directory", captures.directory().string());
    console::field("capture_tensor_count", captures.size());
    console::field("capture_replay_wall_seconds", capture_seconds);
  }

  return token == kExpectedToken ? 0 : 1;
}

int run_cached(const std::string& artifact_path,
               const std::string& capture_directory) {
  console::section("Cached pair diagnostic");
  console::field("cached_pair_tokens", kInputTokens);
  console::field("artifact", artifact_path);
  console::field("verification", "disabled (header+table only)");
  artifact::ArtifactFile file = artifact::ArtifactFile::Open(artifact_path);
  console::field("payload_sha256", artifact::digest_hex(file.header().payload_sha256));

  validate_cuda_device();
  std::size_t free_before = 0;
  std::size_t total = 0;
  check_cuda(cudaMemGetInfo(&free_before, &total),
             "cudaMemGetInfo before cached-pair load");

  const auto load_started = std::chrono::steady_clock::now();
  WeightArena weights(file);
  const double load_seconds = seconds_since(load_started);
  CachedPairEngine engine(weights);
  std::size_t free_after = 0;
  std::size_t total_after = 0;
  check_cuda(cudaMemGetInfo(&free_after, &total_after),
             "cudaMemGetInfo after cached-pair initialization");
  if (total_after != total || free_after > free_before) {
    fail("CUDA memory", "inconsistent cudaMemGetInfo result");
  }

  console::section("Weight load and GPU memory");
  console::field("weight_copy_seconds", load_seconds);
  console::field("device_weight_arena_bytes", weights.size());
  console::field("device_weight_arena_address_mod_4096", weights.address_mod_4096());
  console::field("device_scratch_arena_bytes", engine.scratch_bytes());
  console::field("local_kv_cache_capacity_tokens", kLocalCacheCapacity);
  console::field("local_kv_cache_bytes", engine.local_cache_bytes());
  console::field("global_kv_cache_capacity_tokens", kGlobalCacheCapacity);
  console::field("global_kv_cache_bytes", engine.global_cache_bytes());
  console::field("total_kv_cache_bytes", engine.local_cache_bytes() + engine.global_cache_bytes());
  console::field("gpu_total_bytes", total);
  console::field("gpu_free_before_bytes", free_before);
  console::field("gpu_free_after_initialization_bytes", free_after);
  console::field("gpu_initialization_delta_bytes", free_before - free_after);

  CudaEvent begin;
  CudaEvent end;
  const auto cold_started = std::chrono::steady_clock::now();
  check_cuda(cudaEventRecord(begin.get()),
             "record cold cached-pair start event");
  const std::uint32_t token = engine.run(nullptr);
  check_cuda(cudaEventRecord(end.get()), "record cold cached-pair end event");
  check_cuda(cudaEventSynchronize(end.get()),
             "synchronize cold cached-pair event");
  float cold_milliseconds = 0.0F;
  check_cuda(cudaEventElapsedTime(&cold_milliseconds, begin.get(), end.get()),
             "measure cold cached-pair inference");
  const double cold_wall_seconds = seconds_since(cold_started);

  const auto steady_started = std::chrono::steady_clock::now();
  check_cuda(cudaEventRecord(begin.get()),
             "record steady cached-pair start event");
  const std::uint32_t steady_token = engine.run(nullptr);
  check_cuda(cudaEventRecord(end.get()),
             "record steady cached-pair end event");
  check_cuda(cudaEventSynchronize(end.get()),
             "synchronize steady cached-pair event");
  float steady_milliseconds = 0.0F;
  check_cuda(cudaEventElapsedTime(&steady_milliseconds, begin.get(), end.get()),
             "measure steady cached-pair inference");
  const double steady_wall_seconds = seconds_since(steady_started);
  if (steady_token != token) {
    fail("steady cached-pair replay", "argmax differs from cold run");
  }

  console::section("Inference timing and validation");
  console::field("cold_inference_gpu_milliseconds", cold_milliseconds);
  console::field("cold_inference_wall_seconds", cold_wall_seconds);
  console::field("steady_state_inference_gpu_milliseconds", steady_milliseconds);
  console::field("steady_state_inference_wall_seconds", steady_wall_seconds);
  console::field("argmax_token", token);
  console::field("expected_argmax_token", kExpectedToken);
  console::field("argmax_match", token == kExpectedToken);

  if (capture_directory != "-") {
    CapturePlan captures(capture_directory);
    const auto capture_started = std::chrono::steady_clock::now();
    const std::uint32_t capture_token = engine.run(&captures);
    const double capture_seconds = seconds_since(capture_started);
    if (capture_token != token) {
      fail("cached-pair capture replay", "argmax differs from timed run");
    }
    captures.write(capture_token);
    console::section("Capture files");
    console::field("capture_directory", captures.directory().string());
    console::field("capture_tensor_count", captures.size());
    console::field("capture_replay_wall_seconds", capture_seconds);
  }

  return token == kExpectedToken ? 0 : 1;
}

int run_short_decode(const std::string& artifact_path,
                     const std::string& capture_directory) {
  console::section("Short decode diagnostic");
  console::field("short_decode_prompt_tokens", kShortDecodePrompt);
  console::field("short_decode_schedule", "sequential_m1");
  console::field("short_decode_m1_expected_tokens", kShortDecodeM1Expected);
  console::field("artifact", artifact_path);
  console::field("verification", "disabled (header+table only)");

  artifact::ArtifactFile file = artifact::ArtifactFile::Open(artifact_path);
  console::field("payload_sha256", artifact::digest_hex(file.header().payload_sha256));

  validate_cuda_device();
  std::size_t free_before = 0;
  std::size_t total = 0;
  check_cuda(cudaMemGetInfo(&free_before, &total),
             "cudaMemGetInfo before short-decode load");

  const auto load_started = std::chrono::steady_clock::now();
  WeightArena weights(file);
  const double load_seconds = seconds_since(load_started);
  ShortDecodeEngine engine(weights);
  std::size_t free_after = 0;
  std::size_t total_after = 0;
  check_cuda(cudaMemGetInfo(&free_after, &total_after),
             "cudaMemGetInfo after short-decode initialization");
  if (total_after != total || free_after > free_before) {
    fail("CUDA memory", "inconsistent cudaMemGetInfo result");
  }

  console::section("Weight load and GPU memory");
  console::field("weight_copy_seconds", load_seconds);
  console::field("device_weight_arena_bytes", weights.size());
  console::field("device_weight_arena_address_mod_4096", weights.address_mod_4096());
  console::field("device_scratch_arena_bytes", engine.scratch_bytes());
  console::field("local_kv_cache_capacity_tokens", kLocalCacheCapacity);
  console::field("local_kv_cache_bytes", engine.local_cache_bytes());
  console::field("global_kv_cache_capacity_tokens", kShortDecodeGlobalCacheCapacity);
  console::field("global_kv_cache_bytes", engine.global_cache_bytes());
  console::field("total_kv_cache_bytes", engine.local_cache_bytes() + engine.global_cache_bytes());
  console::field("gpu_total_bytes", total);
  console::field("gpu_free_before_bytes", free_before);
  console::field("gpu_free_after_initialization_bytes", free_after);
  console::field("gpu_initialization_delta_bytes", free_before - free_after);

  CudaEvent begin;
  CudaEvent end;
  const auto cold_started = std::chrono::steady_clock::now();
  check_cuda(cudaEventRecord(begin.get()),
             "record cold short-decode start event");
  const auto generated = engine.run(nullptr);
  check_cuda(cudaEventRecord(end.get()),
             "record cold short-decode end event");
  check_cuda(cudaEventSynchronize(end.get()),
             "synchronize cold short-decode event");
  float cold_milliseconds = 0.0F;
  check_cuda(cudaEventElapsedTime(&cold_milliseconds, begin.get(), end.get()),
             "measure cold short-decode inference");
  const double cold_wall_seconds = seconds_since(cold_started);

  const auto steady_started = std::chrono::steady_clock::now();
  check_cuda(cudaEventRecord(begin.get()),
             "record steady short-decode start event");
  const auto steady_generated = engine.run(nullptr);
  check_cuda(cudaEventRecord(end.get()),
             "record steady short-decode end event");
  check_cuda(cudaEventSynchronize(end.get()),
             "synchronize steady short-decode event");
  float steady_milliseconds = 0.0F;
  check_cuda(cudaEventElapsedTime(&steady_milliseconds, begin.get(), end.get()),
             "measure steady short-decode inference");
  const double steady_wall_seconds = seconds_since(steady_started);
  if (steady_generated != generated) {
    fail("steady short-decode replay",
         "generated sequence differs from cold run");
  }

  console::section("Inference timing and validation");
  console::field("cold_inference_gpu_milliseconds", cold_milliseconds);
  console::field("cold_inference_wall_seconds", cold_wall_seconds);
  console::field("steady_state_inference_gpu_milliseconds", steady_milliseconds);
  console::field("steady_state_inference_wall_seconds", steady_wall_seconds);
  console::field("generated_tokens", generated);
  const bool sequence_matches = generated == kShortDecodeM1Expected;
  console::field("generated_tokens_match_m1_reference", sequence_matches);

  if (capture_directory != "-") {
    ShortDecodeCapturePlan captures(capture_directory);
    const auto capture_started = std::chrono::steady_clock::now();
    const auto capture_generated = engine.run(&captures);
    const double capture_seconds = seconds_since(capture_started);
    if (capture_generated != generated) {
      fail("short-decode capture replay",
           "generated sequence differs from timed run");
    }
    captures.write(capture_generated);
    console::section("Capture files");
    console::field("capture_directory", captures.directory().string());
    console::field("capture_tensor_count", captures.size());
    console::field("generated_tokens_file_bytes",
                   kShortDecodeDecisionCount * sizeof(std::uint32_t));
    console::field("capture_replay_wall_seconds", capture_seconds);
  }

  return sequence_matches ? 0 : 1;
}

int run_local_boundary(const std::string& artifact_path,
                       const std::string& capture_directory) {
  console::section("Local boundary diagnostic");
  console::field("local_boundary_schedule", "sequential_m1");
  console::field("local_boundary_fixture", "bos_plus_repeated_90_token_body");
  console::field("local_boundary_prefix_tokens", kBoundaryPrefixTokens);
  console::field("local_boundary_input_token_ids_sha256", kBoundaryInputTokenIdsSha256);
  console::field("local_boundary_capture_positions", nlohmann::json::array({1022, 1023, 1024, 1025}));
  console::field("local_boundary_decision_position", kBoundaryDecisionPosition);
  console::field("artifact", artifact_path);
  console::field("verification", "disabled (header+table only)");

  artifact::ArtifactFile file = artifact::ArtifactFile::Open(artifact_path);
  console::field("payload_sha256", artifact::digest_hex(file.header().payload_sha256));

  validate_cuda_device();
  std::size_t free_before = 0;
  std::size_t total = 0;
  check_cuda(cudaMemGetInfo(&free_before, &total),
             "cudaMemGetInfo before local-boundary load");

  const auto load_started = std::chrono::steady_clock::now();
  WeightArena weights(file);
  const double load_seconds = seconds_since(load_started);
  BoundaryEngine engine(weights);
  std::size_t free_after = 0;
  std::size_t total_after = 0;
  check_cuda(cudaMemGetInfo(&free_after, &total_after),
             "cudaMemGetInfo after local-boundary initialization");
  if (total_after != total || free_after > free_before) {
    fail("CUDA memory", "inconsistent cudaMemGetInfo result");
  }

  console::section("Weight load and GPU memory");
  console::field("weight_copy_seconds", load_seconds);
  console::field("device_weight_arena_bytes", weights.size());
  console::field("device_weight_arena_address_mod_4096", weights.address_mod_4096());
  console::field("device_scratch_arena_bytes", engine.scratch_bytes());
  console::field("attention_score_scratch_bytes",
                 primitives::kCachedAttentionM1BoundaryScoreScratchBytes);
  console::field("attention_probability_bytes", model::kQueryHeadCount * kBoundaryPrefixTokens *
                   sizeof(BFloat16));
  console::field("local_kv_cache_capacity_tokens", kLocalCacheCapacity);
  console::field("local_kv_cache_bytes", engine.local_cache_bytes());
  console::field("global_kv_cache_capacity_tokens", kBoundaryGlobalCacheCapacity);
  console::field("global_kv_cache_bytes", engine.global_cache_bytes());
  console::field("total_kv_cache_bytes", engine.local_cache_bytes() + engine.global_cache_bytes());
  console::field("gpu_total_bytes", total);
  console::field("gpu_free_before_bytes", free_before);
  console::field("gpu_free_after_initialization_bytes", free_after);
  console::field("gpu_initialization_delta_bytes", free_before - free_after);

  CudaEvent begin;
  CudaEvent end;
  const auto cold_started = std::chrono::steady_clock::now();
  check_cuda(cudaEventRecord(begin.get()),
             "record cold local-boundary start event");
  const std::uint32_t token = engine.run(nullptr);
  check_cuda(cudaEventRecord(end.get()),
             "record cold local-boundary end event");
  check_cuda(cudaEventSynchronize(end.get()),
             "synchronize cold local-boundary event");
  float cold_milliseconds = 0.0F;
  check_cuda(cudaEventElapsedTime(&cold_milliseconds, begin.get(), end.get()),
             "measure cold local-boundary inference");
  const double cold_wall_seconds = seconds_since(cold_started);

  const auto steady_started = std::chrono::steady_clock::now();
  check_cuda(cudaEventRecord(begin.get()),
             "record steady local-boundary start event");
  const std::uint32_t steady_token = engine.run(nullptr);
  check_cuda(cudaEventRecord(end.get()),
             "record steady local-boundary end event");
  check_cuda(cudaEventSynchronize(end.get()),
             "synchronize steady local-boundary event");
  float steady_milliseconds = 0.0F;
  check_cuda(cudaEventElapsedTime(&steady_milliseconds, begin.get(), end.get()),
             "measure steady local-boundary inference");
  const double steady_wall_seconds = seconds_since(steady_started);
  if (steady_token != token) {
    fail("steady local-boundary replay", "argmax differs from cold run");
  }

  console::section("Inference timing and validation");
  console::field("cold_inference_gpu_milliseconds", cold_milliseconds);
  console::field("cold_inference_wall_seconds", cold_wall_seconds);
  console::field("steady_state_inference_gpu_milliseconds", steady_milliseconds);
  console::field("steady_state_inference_wall_seconds", steady_wall_seconds);
  console::field("argmax_token", token);
  console::field("expected_argmax_token", kBoundaryExpectedToken);
  console::field("argmax_match", token == kBoundaryExpectedToken);

  if (capture_directory != "-") {
    BoundaryCapturePlan captures(capture_directory);
    const auto capture_started = std::chrono::steady_clock::now();
    check_cuda(cudaEventRecord(begin.get()),
               "record capture local-boundary start event");
    const std::uint32_t capture_token = engine.run(&captures);
    check_cuda(cudaEventRecord(end.get()),
               "record capture local-boundary end event");
    check_cuda(cudaEventSynchronize(end.get()),
               "synchronize capture local-boundary event");
    float capture_milliseconds = 0.0F;
    check_cuda(
        cudaEventElapsedTime(&capture_milliseconds, begin.get(), end.get()),
        "measure capture local-boundary inference");
    const double capture_seconds = seconds_since(capture_started);
    if (capture_token != token) {
      fail("local-boundary capture replay", "argmax differs from timed run");
    }
    captures.write(capture_token);
    console::section("Capture files");
    console::field("capture_directory", captures.directory().string());
    console::field("capture_tensor_count", captures.size());
    console::field("capture_host_arena_bytes", captures.host_bytes());
    console::field("argmax_file_bytes", sizeof(std::uint32_t));
    console::field("capture_replay_gpu_milliseconds", capture_milliseconds);
    console::field("capture_replay_wall_seconds", capture_seconds);
  }

  return token == kBoundaryExpectedToken ? 0 : 1;
}

int run_local_boundary_prefill(const std::string& artifact_path,
                               const std::string& capture_directory) {
  console::section("Local boundary diagnostic");
  console::field("local_boundary_schedule", "m1024_prefill_then_m1_m1");
  console::field("local_boundary_fixture", "bos_plus_repeated_90_token_body");
  console::field("local_boundary_schedule_rows", nlohmann::json::array({1024, 1, 1}));
  console::field("local_boundary_prefix_tokens", kBoundaryPrefixTokens);
  console::field("local_boundary_input_token_ids_sha256", kBoundaryInputTokenIdsSha256);
  console::field("local_boundary_capture_source", "hybrid");
  console::field("local_boundary_capture_positions", nlohmann::json::array({1022, 1023, 1024, 1025}));
  console::field("local_boundary_decision_positions", nlohmann::json::array({1023, 1024, 1025}));
  console::field("local_boundary_expected_predictions", nlohmann::json::array({236764, 532, 121160}));
  console::field("artifact", artifact_path);
  console::field("verification", "disabled (header+table only)");

  artifact::ArtifactFile file = artifact::ArtifactFile::Open(artifact_path);
  console::field("payload_sha256", artifact::digest_hex(file.header().payload_sha256));

  validate_cuda_device();
  std::size_t free_before = 0;
  std::size_t total = 0;
  check_cuda(cudaMemGetInfo(&free_before, &total),
             "cudaMemGetInfo before hybrid local-boundary load");

  const auto load_started = std::chrono::steady_clock::now();
  WeightArena weights(file);
  const double load_seconds = seconds_since(load_started);
  HybridBoundaryEngine engine(weights);
  std::size_t free_after = 0;
  std::size_t total_after = 0;
  check_cuda(cudaMemGetInfo(&free_after, &total_after),
             "cudaMemGetInfo after hybrid local-boundary initialization");
  if (total_after != total || free_after > free_before) {
    fail("CUDA memory", "inconsistent cudaMemGetInfo result");
  }

  const std::size_t planned_cache_bytes =
      engine.local_cache_bytes() + engine.global_cache_bytes();
  const std::size_t planned_device_arena_bytes =
      weights.size() + engine.scratch_bytes() + planned_cache_bytes;
  const std::size_t observed_initialization_bytes = free_before - free_after;
  console::section("Weight load and GPU memory");
  console::field("weight_copy_seconds", load_seconds);
  console::field("device_weight_arena_bytes", weights.size());
  console::field("device_weight_arena_address_mod_4096", weights.address_mod_4096());
  console::field("device_scratch_arena_bytes", engine.scratch_bytes());
  console::field("prefill_attention_score_scratch_bytes", prefill::kAttentionScoreScratchBytes);
  console::field("prefill_attention_probability_bytes", prefill::kAttentionProbabilityBytes);
  console::field("capture_gather_staging_payload_bytes",
                 HybridBoundaryScratchLayout::kGatherElements *
                   sizeof(BFloat16));
  console::field("capture_gather_staging_arena_span_bytes", HybridBoundaryScratchLayout::kArgmax -
                   HybridBoundaryScratchLayout::kGatherStaging);
  console::field("local_kv_cache_capacity_tokens", kLocalCacheCapacity);
  console::field("local_kv_cache_bytes", engine.local_cache_bytes());
  console::field("global_kv_cache_capacity_tokens", kBoundaryGlobalCacheCapacity);
  console::field("global_kv_cache_bytes", engine.global_cache_bytes());
  console::field("total_kv_cache_bytes", planned_cache_bytes);
  console::field("planned_device_arena_bytes", planned_device_arena_bytes);
  console::field("gpu_total_bytes", total);
  console::field("gpu_free_before_bytes", free_before);
  console::field("gpu_free_after_initialization_bytes", free_after);
  console::field("observed_gpu_initialization_delta_bytes", observed_initialization_bytes);
  print_device_resident_delta("observed_minus_planned_device_bytes",
                              observed_initialization_bytes,
                              planned_device_arena_bytes);


  CudaEvent begin;
  CudaEvent end;
  const auto cold_started = std::chrono::steady_clock::now();
  check_cuda(cudaEventRecord(begin.get()),
             "record cold hybrid local-boundary start event");
  const auto predictions = engine.run(nullptr);
  check_cuda(cudaEventRecord(end.get()),
             "record cold hybrid local-boundary end event");
  check_cuda(cudaEventSynchronize(end.get()),
             "synchronize cold hybrid local-boundary event");
  float cold_milliseconds = 0.0F;
  check_cuda(
      cudaEventElapsedTime(&cold_milliseconds, begin.get(), end.get()),
      "measure cold hybrid local-boundary inference");
  const double cold_wall_seconds = seconds_since(cold_started);

  const auto steady_started = std::chrono::steady_clock::now();
  check_cuda(cudaEventRecord(begin.get()),
             "record steady hybrid local-boundary start event");
  const auto steady_predictions = engine.run(nullptr);
  check_cuda(cudaEventRecord(end.get()),
             "record steady hybrid local-boundary end event");
  check_cuda(cudaEventSynchronize(end.get()),
             "synchronize steady hybrid local-boundary event");
  float steady_milliseconds = 0.0F;
  check_cuda(
      cudaEventElapsedTime(&steady_milliseconds, begin.get(), end.get()),
      "measure steady hybrid local-boundary inference");
  const double steady_wall_seconds = seconds_since(steady_started);
  if (steady_predictions != predictions) {
    fail("steady hybrid local-boundary replay",
         "prediction sequence differs from cold run");
  }

  const bool predictions_match = predictions == kHybridBoundaryExpected;
  console::section("Inference timing and validation");
  console::field("cold_inference_gpu_milliseconds", cold_milliseconds);
  console::field("cold_inference_wall_seconds", cold_wall_seconds);
  console::field("steady_state_inference_gpu_milliseconds", steady_milliseconds);
  console::field("steady_state_inference_wall_seconds", steady_wall_seconds);
  console::field("predictions", predictions);
  console::field("expected_predictions", kHybridBoundaryExpected);
  console::field("predictions_match", predictions_match);
  if (!predictions_match) {
    return 1;
  }

  if (capture_directory != "-") {
    BoundaryCapturePlan captures(capture_directory,
                                 make_hybrid_boundary_capture_specs());
    const auto capture_started = std::chrono::steady_clock::now();
    check_cuda(cudaEventRecord(begin.get()),
               "record capture hybrid local-boundary start event");
    const auto capture_predictions = engine.run(&captures);
    check_cuda(cudaEventRecord(end.get()),
               "record capture hybrid local-boundary end event");
    check_cuda(cudaEventSynchronize(end.get()),
               "synchronize capture hybrid local-boundary event");
    float capture_milliseconds = 0.0F;
    check_cuda(
        cudaEventElapsedTime(&capture_milliseconds, begin.get(), end.get()),
        "measure capture hybrid local-boundary inference");
    const double capture_seconds = seconds_since(capture_started);
    if (capture_predictions != predictions ||
        capture_predictions != kHybridBoundaryExpected) {
      fail("hybrid local-boundary capture replay",
           "prediction sequence differs from the timed reference");
    }
    captures.write(capture_predictions[2]);
    console::section("Capture files");
    console::field("capture_directory", captures.directory().string());
    console::field("capture_tensor_count", captures.size());
    console::field("capture_host_arena_bytes", captures.host_bytes());
    console::field("argmax_file_bytes", sizeof(std::uint32_t));
    console::field("capture_replay_gpu_milliseconds", capture_milliseconds);
    console::field("capture_replay_wall_seconds", capture_seconds);
  }

  return 0;
}

int run_graph_decode(const std::string& artifact_path,
                     const std::string& output_directory) {
  preflight_graph_output_directory(output_directory);
  console::section("Graph decode benchmark");
  console::field("graph_decode_schedule", "m1024_prefill_then_511_graph_m1");
  console::field("graph_decode_fixture", "bos_plus_repeated_90_token_body");
  console::field("graph_decode_prompt_tokens", kGraphPromptTokens);
  console::field("graph_decode_output_tokens", kGraphOutputTokens);
  console::field("graph_decode_replays", kGraphDecodeReplays);
  console::field("graph_decode_positions", nlohmann::json::array({1024, 1534}));
  console::field("graph_decode_final_position", primitives::kGraphDecodeFinalPosition);
  console::field("graph_decode_expected_first_three", nlohmann::json::array({236764, 532, 121160}));
  console::field("graph_decode_stream", "explicit_nonblocking");
  console::field("graph_decode_request_cuda_object_creations", kGraphRequestCudaObjectCreations);
  console::field("artifact", artifact_path);
  console::field("verification", "disabled (header+table only)");

  artifact::ArtifactFile file = artifact::ArtifactFile::Open(artifact_path);
  console::field("payload_sha256", artifact::digest_hex(file.header().payload_sha256));

  validate_cuda_device();
  std::size_t free_before = 0;
  std::size_t total = 0;
  check_cuda(cudaMemGetInfo(&free_before, &total),
             "cudaMemGetInfo before graph-decode load");

  const auto load_started = std::chrono::steady_clock::now();
  WeightArena weights(file);
  const double load_seconds = seconds_since(load_started);
  GraphDecodeEngine engine(weights);
  std::size_t free_after_initialization = 0;
  std::size_t total_after_initialization = 0;
  check_cuda(cudaMemGetInfo(&free_after_initialization,
                            &total_after_initialization),
             "cudaMemGetInfo after graph-decode initialization");
  if (total_after_initialization != total ||
      free_after_initialization > free_before) {
    fail("graph-decode CUDA memory", "inconsistent initialization memory");
  }

  const std::size_t cache_bytes =
      engine.local_cache_bytes() + engine.global_cache_bytes();
  const std::size_t planned_device_arena_bytes =
      weights.size() + engine.scratch_bytes() + cache_bytes +
      engine.persistent_state_bytes();
  console::section("Weight load and GPU memory");
  console::field("weight_copy_seconds", load_seconds);
  console::field("device_weight_arena_bytes", weights.size());
  console::field("device_weight_arena_address_mod_4096", weights.address_mod_4096());
  console::field("device_scratch_arena_bytes", engine.scratch_bytes());
  console::field("prefill_attention_score_scratch_bytes", prefill::kAttentionScoreScratchBytes);
  console::field("prefill_attention_probability_bytes", prefill::kAttentionProbabilityBytes);
  console::field("decode_attention_fused_scratch_bytes",
                 primitives::kGraphAttentionFusedScratchBytes);
  console::field("local_kv_cache_capacity_tokens", kLocalCacheCapacity);
  console::field("local_kv_cache_bytes", engine.local_cache_bytes());
  console::field("global_kv_cache_capacity_tokens", kGraphGlobalCacheCapacity);
  console::field("global_kv_cache_bytes", engine.global_cache_bytes());
  console::field("total_kv_cache_bytes", cache_bytes);
  console::field("persistent_device_state_bytes", engine.persistent_state_bytes());
  console::field("persistent_timing_event_count", kGraphPersistentTimingEventCount);
  console::field("planned_device_arena_bytes_before_graph", planned_device_arena_bytes);
  console::field("gpu_total_bytes", total);
  console::field("gpu_free_before_bytes", free_before);
  console::field("gpu_free_after_initialization_bytes", free_after_initialization);
  console::field("observed_gpu_initialization_delta_bytes",
                 free_before - free_after_initialization);
  const std::size_t observed_initialization_bytes =
      free_before - free_after_initialization;
  print_device_resident_delta("observed_minus_planned_device_bytes",
                              observed_initialization_bytes,
                              planned_device_arena_bytes);


  const GraphBuildMetrics graph = engine.build_graph();
  console::section("CUDA graph build");
  console::field("graph_capture_wall_seconds", graph.capture_wall_seconds);
  console::field("graph_instantiate_wall_seconds", graph.instantiate_wall_seconds);
  console::field("graph_upload_wall_seconds", graph.upload_wall_seconds);
  console::field("graph_node_count", graph.node_count);
  console::field("gpu_free_before_graph_upload_bytes", graph.free_before_upload);
  console::field("gpu_free_after_graph_upload_bytes", graph.free_after_upload);
  print_device_resident_delta("graph_build_preupload_device_delta_bytes",
                              free_after_initialization,
                              graph.free_before_upload);
  print_device_resident_delta("graph_upload_device_delta_bytes",
                              graph.free_before_upload,
                              graph.free_after_upload);
  print_device_resident_delta("graph_total_resident_device_delta_bytes",
                              free_before, graph.free_after_upload);

  const GraphRequestResult uncaptured = engine.run_uncaptured_request();
  validate_graph_request(uncaptured, "uncaptured graph-decode request");
  const GraphRequestResult graphed = engine.run_graphed_request();
  validate_graph_request(graphed, "graphed graph-decode request");
  if (std::memcmp(uncaptured.outputs.data(), graphed.outputs.data(),
                  kGraphOutputBytes) != 0) {
    fail("graph-decode validation",
         "uncaptured and graphed 2048-byte outputs differ");
  }
  const GraphRequestResult repeated = engine.run_graphed_request();
  validate_graph_request(repeated, "repeated graph-decode request");
  if (std::memcmp(graphed.outputs.data(), repeated.outputs.data(),
                  kGraphOutputBytes) != 0) {
    fail("graph-decode validation", "repeated graph output differs");
  }

  const std::string output_sha256 =
      artifact::digest_hex(sha256_bytes(graphed.outputs.data(),
                                        kGraphOutputBytes));
  console::section("Graph validation");
  console::field("validation_uncaptured_prefill_gpu_milliseconds",
                 uncaptured.prefill_gpu_milliseconds);
  console::field("validation_uncaptured_prefill_wall_seconds", uncaptured.prefill_wall_seconds);
  console::field("validation_uncaptured_decode_gpu_milliseconds",
                 uncaptured.decode_gpu_milliseconds);
  console::field("validation_uncaptured_decode_wall_seconds", uncaptured.decode_wall_seconds);
  console::field("validation_graph_prefill_gpu_milliseconds", graphed.prefill_gpu_milliseconds);
  console::field("validation_graph_prefill_wall_seconds", graphed.prefill_wall_seconds);
  console::field("validation_graph_decode_gpu_milliseconds", graphed.decode_gpu_milliseconds);
  console::field("validation_graph_decode_wall_seconds", graphed.decode_wall_seconds);
  console::field("validation_graph_repeat_prefill_gpu_milliseconds",
                 repeated.prefill_gpu_milliseconds);
  console::field("validation_graph_repeat_decode_gpu_milliseconds",
                 repeated.decode_gpu_milliseconds);
  console::field("uncaptured_graph_outputs_byte_identical", true);
  console::field("graph_repeat_outputs_byte_identical", true);
  console::field("graph_output_first_three",
                 nlohmann::json::array({graphed.outputs[0], graphed.outputs[1], graphed.outputs[2]}));
  console::field("graph_output_final_position", graphed.final_position);
  console::field("graph_output_bytes", kGraphOutputBytes);
  console::field("graph_output_sha256", output_sha256);

  for (std::size_t warmup = 0; warmup < kGraphWarmupRequests; ++warmup) {
    const GraphRequestResult result = engine.run_graphed_request();
    validate_graph_request(result, "graph-decode benchmark warmup");
    if (std::memcmp(result.outputs.data(), graphed.outputs.data(),
                    kGraphOutputBytes) != 0) {
      fail("graph-decode benchmark warmup", "output bytes differ");
    }
  }

  std::array<double, kGraphMeasuredRequests> prefill_gpu_seconds{};
  std::array<double, kGraphMeasuredRequests> prefill_wall_seconds{};
  std::array<double, kGraphMeasuredRequests> decode_gpu_seconds{};
  std::array<double, kGraphMeasuredRequests> decode_wall_seconds{};
  std::array<double, kGraphMeasuredRequests> pp_gpu_tokens_per_second{};
  std::array<double, kGraphMeasuredRequests> pp_wall_tokens_per_second{};
  std::array<double, kGraphMeasuredRequests> tg_gpu_tokens_per_second{};
  std::array<double, kGraphMeasuredRequests> tg_wall_tokens_per_second{};
  for (std::size_t sample = 0; sample < kGraphMeasuredRequests; ++sample) {
    const GraphRequestResult result = engine.run_graphed_request();
    validate_graph_request(result, "measured graph-decode request");
    if (std::memcmp(result.outputs.data(), graphed.outputs.data(),
                    kGraphOutputBytes) != 0) {
      fail("measured graph-decode request", "output bytes differ");
    }
    prefill_gpu_seconds[sample] = result.prefill_gpu_milliseconds / 1'000.0;
    prefill_wall_seconds[sample] = result.prefill_wall_seconds;
    decode_gpu_seconds[sample] = result.decode_gpu_milliseconds / 1'000.0;
    decode_wall_seconds[sample] = result.decode_wall_seconds;
    pp_gpu_tokens_per_second[sample] =
        kGraphPromptTokens / prefill_gpu_seconds[sample];
    pp_wall_tokens_per_second[sample] =
        kGraphPromptTokens / prefill_wall_seconds[sample];
    tg_gpu_tokens_per_second[sample] =
        kGraphDecodeReplays / decode_gpu_seconds[sample];
    tg_wall_tokens_per_second[sample] =
        kGraphDecodeReplays / decode_wall_seconds[sample];
  }

  console::section("Benchmark samples");
  console::field("benchmark_warmup_requests", kGraphWarmupRequests);
  console::field("benchmark_measured_requests", kGraphMeasuredRequests);
  console::field("benchmark_pp_tokens", kGraphPromptTokens);
  console::field("benchmark_tg_tokens", kGraphDecodeReplays);
  console::section("Benchmark medians");
  console::field("benchmark_prefill_gpu_seconds_median", median(prefill_gpu_seconds));
  console::field("benchmark_prefill_wall_seconds_median", median(prefill_wall_seconds));
  console::field("benchmark_decode_gpu_seconds_median", median(decode_gpu_seconds));
  console::field("benchmark_decode_wall_seconds_median", median(decode_wall_seconds));
  console::field("benchmark_pp_gpu_tokens_per_second_median", median(pp_gpu_tokens_per_second));
  console::field("benchmark_pp_wall_tokens_per_second_median", median(pp_wall_tokens_per_second));
  console::field("benchmark_tg_gpu_tokens_per_second_median", median(tg_gpu_tokens_per_second));
  console::field("benchmark_tg_wall_tokens_per_second_median", median(tg_wall_tokens_per_second));
  console::section("Individual samples");
  print_values("benchmark_prefill_gpu_seconds_raw", prefill_gpu_seconds);
  print_values("benchmark_prefill_wall_seconds_raw", prefill_wall_seconds);
  print_values("benchmark_decode_gpu_seconds_raw", decode_gpu_seconds);
  print_values("benchmark_decode_wall_seconds_raw", decode_wall_seconds);
  print_values("benchmark_pp_gpu_tokens_per_second_raw",
               pp_gpu_tokens_per_second);
  print_values("benchmark_pp_wall_tokens_per_second_raw",
               pp_wall_tokens_per_second);
  print_values("benchmark_tg_gpu_tokens_per_second_raw",
               tg_gpu_tokens_per_second);
  print_values("benchmark_tg_wall_tokens_per_second_raw",
               tg_wall_tokens_per_second);

  write_graph_outputs_exclusive(output_directory, graphed.outputs);
  if (output_directory != "-") {
    console::section("Output files");
    console::field("output_directory", output_directory);
    console::field("output_file", "outputs.u32");
  }
  return 0;
}


int run_profile_decode(const std::string& artifact_path) {
  console::section("Decode profiling");
  console::field("profile_decode_schedule", "m1024_prefill_then_1_graph_m1");
  console::field("profile_decode_fixture", "bos_plus_repeated_90_token_body");
  console::field("profile_decode_prompt_tokens", kGraphPromptTokens);
  console::field("profile_decode_replays", kProfileDecodeReplays);
  console::field("profile_decode_position", primitives::kGraphDecodeFirstPosition);
  console::field("profile_decode_expected_outputs", nlohmann::json::array({236764, 532}));
  console::field("profile_decode_profiler_range", "cudaProfilerApi");
  console::field("profile_decode_stream", "explicit_nonblocking");
  console::field("artifact", artifact_path);
  console::field("verification", "disabled (header+table only)");

  artifact::ArtifactFile file = artifact::ArtifactFile::Open(artifact_path);
  console::field("payload_sha256", artifact::digest_hex(file.header().payload_sha256));

  validate_cuda_device();
  std::size_t free_before = 0;
  std::size_t total = 0;
  check_cuda(cudaMemGetInfo(&free_before, &total),
             "cudaMemGetInfo before profile-decode load");

  const auto load_started = std::chrono::steady_clock::now();
  WeightArena weights(file);
  const double load_seconds = seconds_since(load_started);
  GraphDecodeEngine engine(weights);
  std::size_t free_after_initialization = 0;
  std::size_t total_after_initialization = 0;
  check_cuda(cudaMemGetInfo(&free_after_initialization,
                            &total_after_initialization),
             "cudaMemGetInfo after profile-decode initialization");
  if (total_after_initialization != total ||
      free_after_initialization > free_before) {
    fail("profile-decode CUDA memory", "inconsistent initialization memory");
  }

  const std::size_t cache_bytes =
      engine.local_cache_bytes() + engine.global_cache_bytes();
  const std::size_t planned_device_arena_bytes =
      weights.size() + engine.scratch_bytes() + cache_bytes +
      engine.persistent_state_bytes();
  console::section("Weight load and GPU memory");
  console::field("weight_copy_seconds", load_seconds);
  console::field("device_weight_arena_bytes", weights.size());
  console::field("device_scratch_arena_bytes", engine.scratch_bytes());
  console::field("local_kv_cache_bytes", engine.local_cache_bytes());
  console::field("global_kv_cache_bytes", engine.global_cache_bytes());
  console::field("total_kv_cache_bytes", cache_bytes);
  console::field("persistent_device_state_bytes", engine.persistent_state_bytes());
  console::field("planned_device_arena_bytes_before_graph", planned_device_arena_bytes);
  console::field("gpu_total_bytes", total);
  console::field("gpu_free_before_bytes", free_before);
  console::field("gpu_free_after_initialization_bytes", free_after_initialization);
  print_device_resident_delta("observed_gpu_initialization_delta_bytes",
                              free_before, free_after_initialization);

  const GraphBuildMetrics graph = engine.build_graph();
  console::section("CUDA graph build");
  console::field("graph_capture_wall_seconds", graph.capture_wall_seconds);
  console::field("graph_instantiate_wall_seconds", graph.instantiate_wall_seconds);
  console::field("graph_upload_wall_seconds", graph.upload_wall_seconds);
  console::field("graph_node_count", graph.node_count);
  console::field("gpu_free_before_graph_upload_bytes", graph.free_before_upload);
  console::field("gpu_free_after_graph_upload_bytes", graph.free_after_upload);
  print_device_resident_delta("graph_build_preupload_device_delta_bytes",
                              free_after_initialization,
                              graph.free_before_upload);
  print_device_resident_delta("graph_upload_device_delta_bytes",
                              graph.free_before_upload,
                              graph.free_after_upload);
  print_device_resident_delta("graph_total_resident_device_delta_bytes",
                              free_before, graph.free_after_upload);


  const GraphProfileResult result = engine.run_profiled_step();
  validate_graph_profile(result);
  console::section("Profile results");
  console::field("profile_prefill_gpu_milliseconds", result.prefill_gpu_milliseconds);
  console::field("profile_prefill_wall_seconds", result.prefill_wall_seconds);
  console::field("profile_single_step_cuda_event_milliseconds", result.decode_gpu_milliseconds);
  console::field("profile_single_step_wall_seconds", result.decode_wall_seconds);
  console::field("profile_single_step_timing_note", "nsys_kernel_span_is_authoritative");
  console::field("profile_initial_position", result.initial_position);
  console::field("profile_final_position", result.final_position);
  console::field("profile_outputs", result.outputs);
  console::field("profile_validation", "passed");
  return 0;
}


}  // namespace gewell::diagnostics
