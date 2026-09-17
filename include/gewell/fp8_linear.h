#pragma once

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace gewell::fp8 {

inline constexpr std::size_t kMaxWorkspaceBytes = 32 * 1024 * 1024;

// E4M3FN weights in contiguous row-major [N,K], one byte per element.
// The positive FP32 globals are dequantization scales, not reciprocals.
struct Weight {
  const std::uint8_t* data{};
  float input_scale{};
  float weight_scale{};
  // Joined projections retain their own alpha (input*weight scale) per output
  // channel. Individual matrix views continue to use the scalar above.
  const float* channel_scales{};
};

enum class InputTransform { identity, gelu_tanh_multiply };

// Bounds every plan with 1..max_rows rows, including packed activations
// and the bounded cuBLASLt/CUTLASS workspace.
constexpr std::size_t scratch_upper_bound_bytes(std::size_t max_rows,
                                                std::size_t input_width) {
  return max_rows * input_width + kMaxWorkspaceBytes + 255;
}

// Native E4M3 x E4M3 Tensor Core GEMM, FP32 accumulation and BF16 output.
// Inputs/output are contiguous row-major, Y = X @ W^T. K and N must be
// divisible by16. Any positive M is supported without row padding.
// Construction selects the GEMM plan and must occur outside capture.
// The handle outlives the plan. A plan is not safe for simultaneous host calls.
class Plan {
 public:
  Plan(cublasLtHandle_t handle, std::uint32_t rows,
       std::uint32_t input_width, std::uint32_t output_width,
       bool channel_scales = false);
  ~Plan();
  Plan(Plan&&) noexcept;
  Plan& operator=(Plan&&) noexcept;
  Plan(const Plan&) = delete;
  Plan& operator=(const Plan&) = delete;

  std::size_t scratch_bytes() const;

  // Device buffers must not overlap; weights/output must be 16B aligned.
  // Scratch is 256B aligned and valid until stream completion. Sequential
  // calls on one stream may share it. No allocation or synchronization;
  // CUDA-graph capturable. Activations use round-to-nearest-even E4M3FN with
  // finite saturation after FP32 multiplication by 1 / input_scale.
  // Channel-scaled plans require a 16B-aligned device vector of output_width
  // FP32 alphas. GELU input is [rows,2*K] gate/up slices; both intermediate
  // BF16 rounding boundaries are preserved before FP8 packing.
  void run(const __nv_bfloat16* input, Weight weight, __nv_bfloat16* output,
           void* scratch, std::size_t scratch_capacity,
           cudaStream_t stream = nullptr,
           InputTransform transform = InputTransform::identity);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gewell::fp8
