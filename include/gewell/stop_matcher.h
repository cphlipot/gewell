#pragma once

#include <cstddef>
#include <deque>
#include <string>
#include <string_view>
#include <vector>

namespace gewell::text {

// Earliest completed match wins; equal endpoints use request order. Only a
// suffix that can still start a stop is withheld. KMP keeps long repeated
// prefixes linear in input size rather than repeatedly scanning that suffix.
class StopMatcher {
 public:
  explicit StopMatcher(std::vector<std::string> stops);
  std::string push(std::string_view text);
  std::string finish();
  bool matched() const { return matched_; }
  std::size_t pending_bytes() const { return pending_.size(); }
  bool would_match(std::string_view text) const;

 private:
  struct Pattern {
    std::string text;
    std::vector<std::size_t> fallback;
  };
  static bool advance(const Pattern& pattern, std::size_t& state, char byte);
  std::vector<Pattern> patterns_;
  std::vector<std::size_t> states_;
  std::deque<char> pending_;
  bool matched_ = false;
};

}  // namespace gewell::text
