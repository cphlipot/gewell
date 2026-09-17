#pragma once

#include <string_view>

namespace gewell::attention {

// Prefill matmul operands only. Softmax/accumulation remain FP32; decode and
// image-containing chunks keep their existing attention implementation.
enum class Compute { bf16, fp8 };

constexpr std::string_view compute_name(Compute value) {
  return value == Compute::fp8 ? "fp8" : "bf16";
}

}  // namespace gewell::attention
