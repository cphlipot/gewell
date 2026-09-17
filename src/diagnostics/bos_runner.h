#pragma once

#include <string>

namespace gewell::bos_runner {

// Runs the pinned Gemma 4 31B BF16 position-zero graph for BOS token 2.
// The artifact metadata is validated before any weights are copied to CUDA.
// Use `gewell verify PATH` for the opt-in full payload check.
// A capture directory of "-" disables captures; any other path must not exist.
int run(const std::string& artifact_path, const std::string& capture_directory);

}  // namespace gewell::bos_runner
