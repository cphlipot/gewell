#pragma once
#include "../weights.cuh"
#include "gewell/vision_engine.h"
#include "gewell/vision_executor.h"

namespace gewell::gemma4_31b::sm120 {
// Prepared features outlive the temporary encoder workspace and uploads.
class PreparedImage {
 public:
  PreparedImage(const WeightArena& weights, const std::vector<std::uint8_t>& pixels,
                const std::vector<std::uint8_t>& positions, std::uint32_t padded_patch_rows,
                std::uint32_t soft_token_count);
  const BFloat16* data() const { return static_cast<const BFloat16*>(features_.data()); }
  std::size_t size() const { return features_.size(); }
  std::size_t scratch_bytes() const { return scratch_bytes_; }
  float gpu_milliseconds() const { return gpu_milliseconds_; }
 private:
  DeviceAllocation features_;
  std::size_t scratch_bytes_{};
  float gpu_milliseconds_{};
};
}
