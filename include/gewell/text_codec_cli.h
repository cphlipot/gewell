#pragma once

#include <string>

namespace gewell::text {
// CPU-only JSON-lines diagnostic for the explicitly selected local model.
int run_text_codec(const std::string& model_directory);
}  // namespace gewell::text
