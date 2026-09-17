#include "gewell/stop_matcher.h"

#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using gewell::text::StopMatcher;

namespace {

std::pair<std::string, bool> reference(const std::vector<std::string>& stops,
                                     const std::string& text) {
  for (std::size_t end = 1; end <= text.size(); ++end)
    for (const auto& stop : stops)
      if (stop.size() <= end && text.compare(end - stop.size(), stop.size(), stop) == 0)
        return {text.substr(0, end - stop.size()), true};
  return {text, false};
}

void check(const std::vector<std::string>& stops, const std::string& text) {
  const auto expected = reference(stops, text);
  for (std::size_t split = 0; split <= text.size(); ++split) {
    StopMatcher matcher(stops);
    auto actual = matcher.push(text.substr(0, split));
    if (matcher.would_match(text.substr(split)) != expected.second)
      throw std::runtime_error("non-consuming stop preview differs");
    actual += matcher.push(text.substr(split));
    actual += matcher.finish();
    if (std::pair{actual, matcher.matched()} != expected)
      throw std::runtime_error("split stop matching differs from first completed match");
  }
}

}  // namespace

int main() {
  try {
    check({}, "unchanged");
    check({"ab", "b"}, "zabc");
    check({"b", "ab"}, "zabc");
    check({"abc", "ab"}, "zabc");
    check({"aba", "bab"}, "aabababab");
    check({"longer", "q"}, "longqer");
    check({"END"}, "ENDmore");
    check({"END"}, "unmatched EN");
    check({"€", "é"}, "café costs €1");
    check({"👩‍💻"}, "hello 👩‍💻 and goodbye");
    std::mt19937 random(19);
    for (int trial = 0; trial < 1000; ++trial) {
      std::vector<std::string> stops(1 + random() % 4);
      for (auto& stop : stops)
        for (std::size_t n = 1 + random() % 8; n; --n) stop += char('a' + random() % 3);
      std::string text;
      for (int n = 0; n < 24; ++n) text += char('a' + random() % 3);
      check(stops, text);
    }
    // A long repeating prefix exercises bounded withholding and linear work.
    StopMatcher long_stop({std::string(100000, 'a') + 'b'});
    const auto prefix = long_stop.push(std::string(200000, 'a'));
    if (prefix != std::string(100000, 'a') || long_stop.pending_bytes() != 100000 ||
        long_stop.push("btail") != "" || !long_stop.matched() || !long_stop.finish().empty())
      throw std::runtime_error("long overlapping stop failed");
    std::cout << "stop matcher: boundary, Unicode, tie-order, preview and randomized checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "stop matcher test: " << error.what() << '\n';
    return 1;
  }
}
