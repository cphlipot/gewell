#pragma once
#include <cstdint>
#include <string>

namespace gewell::app {
struct RuntimeSettings;
// Resident, bounded local JSONL jobs on stdin/stdout; raw token and BF16 logit
// files retain exact model output for evaluation and calibration.
int run_jobs(const std::string& artifact_path, std::uint32_t max_batch,
             RuntimeSettings settings, const std::string& qdq_mask_path = {});
}
