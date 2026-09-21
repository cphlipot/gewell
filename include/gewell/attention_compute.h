#pragma once

#include <string_view>

namespace gewell::attention {

// Text attention matmul operands, including decode and MTP. Softmax and
// accumulation remain FP32; image-containing prefill chunks stay BF16.
enum class Compute { bf16, fp8 };

constexpr std::string_view compute_name(Compute value) {
  return value == Compute::fp8 ? "fp8" : "bf16";
}

}  // namespace gewell::attention
