#include "gewell/stop_matcher.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace gewell::text {

StopMatcher::StopMatcher(std::vector<std::string> stops) {
  if (stops.size() > 4) throw std::invalid_argument("at most four stop strings are supported");
  for (auto& stop : stops) {
    if (stop.empty()) throw std::invalid_argument("stop strings must not be empty");
    Pattern pattern{std::move(stop), {}};
    pattern.fallback.resize(pattern.text.size());
    for (std::size_t i = 1, state = 0; i < pattern.text.size(); ++i) {
      while (state && pattern.text[i] != pattern.text[state]) state = pattern.fallback[state - 1];
      if (pattern.text[i] == pattern.text[state]) ++state;
      pattern.fallback[i] = state;
    }
    patterns_.push_back(std::move(pattern));
  }
  states_.resize(patterns_.size());
}

bool StopMatcher::advance(const Pattern& pattern, std::size_t& state, char byte) {
  while (state && byte != pattern.text[state]) state = pattern.fallback[state - 1];
  if (byte == pattern.text[state]) ++state;
  return state == pattern.text.size();
}

std::string StopMatcher::push(std::string_view text) {
  if (matched_) return {};
  if (patterns_.empty()) return std::string(text);
  std::string output;
  for (char byte : text) {
    pending_.push_back(byte);
    std::size_t retained = 0;
    for (std::size_t i = 0; i < patterns_.size(); ++i) {
      if (advance(patterns_[i], states_[i], byte)) {
        const auto visible = pending_.size() - patterns_[i].text.size();
        for (std::size_t j = 0; j < visible; ++j) {
          output.push_back(pending_.front());
          pending_.pop_front();
        }
        pending_.clear();
        matched_ = true;
        return output;
      }
      retained = std::max(retained, states_[i]);
    }
    while (pending_.size() > retained) {
      output.push_back(pending_.front());
      pending_.pop_front();
    }
  }
  return output;
}

std::string StopMatcher::finish() {
  std::string output(pending_.begin(), pending_.end());
  pending_.clear();
  return output;
}

bool StopMatcher::would_match(std::string_view text) const {
  if (matched_) return true;
  auto states = states_;
  for (char byte : text)
    for (std::size_t i = 0; i < patterns_.size(); ++i)
      if (advance(patterns_[i], states[i], byte)) return true;
  return false;
}

}  // namespace gewell::text
