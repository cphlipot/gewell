#pragma once

#include "json.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gewell::text {

struct ChatToolCall {
  std::string id;
  std::string name;
  nlohmann::json arguments;
};

struct ChatMessage {
  std::string role;
  std::string content;
  // The first nonempty value of reasoning/reasoning_content, kept verbatim.
  std::string reasoning{};
  std::vector<ChatToolCall> tool_calls{};
  std::string tool_call_id{};
  std::string name{};
};

struct ChatTemplateOptions {
  bool enable_thinking = false;
  bool preserve_thinking = false;
};

// One immutable model contract is selected at initialization and outlives its
// tokenizer, decoders and compiled constraints. Reusable mechanics consume the
// actual supported pipeline; model-owned functions validate and render it.
struct TextContract {
  std::uint32_t vocabulary_size;
  std::uint32_t context_tokens;
  std::size_t merge_count;
  std::uint32_t bos;
  std::vector<std::uint32_t> stop_tokens;
  std::vector<std::uint32_t> constraint_stop_tokens;
  std::uint32_t channel_start, channel_end;
  std::uint32_t tool_call_start, tool_call_end, tool_handoff, string_delimiter;
  std::string_view thought_prefix;
  std::string_view reasoning_continuation;
  std::string_view space_marker;
  std::vector<std::uint32_t> special_tokens;
  std::vector<std::pair<std::uint32_t, std::string_view>> control_tokens;

  void (*validate_tokenizer_pipeline)(const nlohmann::json&);
  std::vector<ChatMessage> (*normalize_messages)(const nlohmann::json&);
  nlohmann::json (*normalize_tools)(const nlohmann::json&);
  ChatTemplateOptions (*template_options)(const nlohmann::json&);
  std::string (*render)(const std::vector<ChatMessage>&, ChatTemplateOptions,
                        const nlohmann::json&);

  [[nodiscard]] std::size_t mask_words() const { return (std::size_t(vocabulary_size) + 31) / 32; }
  [[nodiscard]] bool is_stop(std::uint32_t token) const {
    return std::find(stop_tokens.begin(), stop_tokens.end(), token) != stop_tokens.end();
  }
  [[nodiscard]] bool is_constraint_stop(std::uint32_t token) const {
    return std::find(constraint_stop_tokens.begin(), constraint_stop_tokens.end(), token) != constraint_stop_tokens.end();
  }
  [[nodiscard]] std::string render_chat(
      const std::vector<ChatMessage>& messages, ChatTemplateOptions options = {},
      const nlohmann::json& tools = nlohmann::json::array()) const {
    return render(messages, options, tools);
  }
};

}  // namespace gewell::text
