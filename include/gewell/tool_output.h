#pragma once

#include "gewell/chat_codec.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace gewell::text {

struct ToolCallDelta {
  std::size_t index = 0;
  // ID and name occur only on the first delta; later deltas append arguments.
  std::string id, name, arguments;
};

struct ToolCallOutput {
  std::string id, name, arguments;
  bool complete = false;
};

struct ToolDelta {
  std::string reasoning, content;
  std::vector<ToolCallDelta> tool_calls;
};

// Incrementally translates the pinned Gemma call grammar into OpenAI argument
// strings. It does not enforce parameter schemas or execute functions.
class ToolOutputDecoder {
 public:
  ToolOutputDecoder(const Tokenizer& tokenizer, std::vector<std::string> names,
                    std::string call_id_prefix, bool allow_calls,
                    bool initial_reasoning = false);
  ~ToolOutputDecoder();
  [[nodiscard]] ToolDelta push(std::uint32_t token);
  // EOS is a completed model output; an unfinished call there is malformed.
  // At a token-budget boundary, preserve partial argument serialization.
  [[nodiscard]] ToolDelta finish(bool eos = false);
  [[nodiscard]] ChatDelta pending() const;
  // True while ordinary answer tokens are routed to message.content rather
  // than hidden reasoning or function-call arguments.
  [[nodiscard]] bool in_content() const;
  [[nodiscard]] bool tool_handoff() const;
  [[nodiscard]] const std::vector<ToolCallOutput>& calls() const;
  [[nodiscard]] std::size_t retained_bytes() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gewell::text
