#pragma once

#include <cstddef>

namespace gewell::runtime {

// Only the selected backend interprets these values. Runtime policy owns their
// lifetime and passes them back at execution and checkpoint boundaries.
struct TerminalState {
  void* value{};
  TerminalState() = default;
  TerminalState(std::nullptr_t) {}
  explicit TerminalState(void* value) : value(value) {}
  explicit operator bool() const { return value != nullptr; }
};

struct CompletionContext {
  void* value{};
  CompletionContext() = default;
  CompletionContext(std::nullptr_t) {}
  explicit CompletionContext(void* value) : value(value) {}
  explicit operator bool() const { return value != nullptr; }
};

}  // namespace gewell::runtime
