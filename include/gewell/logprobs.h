#pragma once

#include <cstdint>

namespace gewell {

inline constexpr std::uint32_t kMaxTopLogprobs = 20;

struct AlternativeLogprob {
  std::uint32_t token = 0;
  float logprob = 0;
};

// Compact per-token score data transferred from the GPU. The selected token
// ID travels alongside this value in the existing output burst.
struct TokenLogprobs {
  float logprob = 0;
  std::uint32_t count = 0;
  AlternativeLogprob top[kMaxTopLogprobs]{};
};

}  // namespace gewell
