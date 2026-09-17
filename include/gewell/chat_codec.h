#pragma once

#include "gewell/tokenizer.h"
#include "json.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace gewell::text {

// The caller handles generation termination. In particular, these decoders do
// not suppress the contract's stop IDs. A full byte-fallback run must remain buffered:
// the supported byte-fallback pipeline replaces every byte if that run is invalid.
class IncrementalTextDecoder {
 public:
  explicit IncrementalTextDecoder(const Tokenizer& tokenizer,
                                  bool skip_special_tokens = false);
  [[nodiscard]] std::string push(std::uint32_t token);
  [[nodiscard]] std::string finish();
  // Preview a complete valid UTF-8 byte run for stop matching. Incomplete or
  // invalid runs remain ambiguous until finish() or a non-byte token.
  [[nodiscard]] std::string pending() const;

 private:
  const Tokenizer& tokenizer_;
  bool skip_special_tokens_;
  std::string bytes_;
};

struct ChatDelta {
  std::string reasoning;
  std::string content;
};

class ChatOutputDecoder {
 public:
  explicit ChatOutputDecoder(const Tokenizer& tokenizer);
  [[nodiscard]] ChatDelta push(std::uint32_t token);
  [[nodiscard]] ChatDelta finish();
  [[nodiscard]] ChatDelta pending() const;

 private:
  [[nodiscard]] ChatDelta route(std::string text);

  IncrementalTextDecoder decoder_;
  const TextContract& contract_;
  bool reasoning_ = false;
  std::optional<std::string> reasoning_prefix_;
};

}  // namespace gewell::text
