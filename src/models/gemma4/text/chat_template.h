#pragma once

#include "gewell/text_contract.h"

namespace gewell::gemma4 {
namespace text_detail {
std::vector<text::ChatMessage> normalize_chat_messages(const nlohmann::json&);
nlohmann::json normalize_chat_tools(const nlohmann::json&);
text::ChatTemplateOptions parse_chat_template_kwargs(const nlohmann::json&);
std::string render_chat(const std::vector<text::ChatMessage>&,
                        text::ChatTemplateOptions, const nlohmann::json&);
}  // namespace text_detail
}  // namespace gewell::gemma4
