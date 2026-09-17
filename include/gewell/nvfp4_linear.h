#pragma once

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace gewell::nvfp4 {

inline constexpr std::size_t kMaxWorkspaceBytes = 36 * 1024 * 1024;

// Checkpoint E2M1 weights: [output_width,input_width/2], with the even K
// element in the low nibble. Scales are positive E4M3 bytes in cuBLASLt's
// 128-row x 4-block tiled layout; each scale covers 16 consecutive K values.
// The two FP32 globals are dequantization scales, not their reciprocals.
struct Weight {
  const std::uint8_t* data{};
  const std::uint8_t* scales{};
  float input_scale{};
  float weight_scale{};
};

enum class InputTransform { identity, gelu_tanh_multiply };

// Restore stored E2M1 values and tiled E4M3 scales to row-major [N,K] BF16.
// Uses weight_scale only; input_scale belongs to activation quantization.
// No allocation/synchronization. K must be divisible by64, N by128.
void dequantize(Weight weight, std::uint32_t input_width,
                std::uint32_t output_width, __nv_bfloat16* output,
                cudaStream_t stream = nullptr);

constexpr std::size_t scale_storage_bytes(std::size_t rows,
                                          std::size_t input_width) {
  return ((rows + 127) / 128) * ((input_width + 63) / 64) * 512;
}

// Bounds every plan with 1..max_rows rows, including shapes that need a
// padded BF16 output and the largest workspace used by either GEMM backend.
constexpr std::size_t scratch_upper_bound_bytes(std::size_t max_rows,
                                                std::size_t input_width,
                                                std::size_t output_width) {
  const auto padded_rows = ((max_rows + 31) / 32) * 32;
  return padded_rows * (input_width / 2 + output_width * 2) +
         scale_storage_bytes(max_rows, input_width) + kMaxWorkspaceBytes +
         3 * 255;
}

// input_width must be divisible by 64. Padded scale entries must be zero.
constexpr std::size_t scale_offset(std::size_t row, std::size_t block,
                                    std::size_t input_width) {
  return ((row / 128) * (input_width / 64) + block / 4) * 512 +
         (row % 32) * 16 + ((row % 128) / 32) * 4 + block % 4;
}

// Native FP4 x FP4 Tensor Core GEMM, accumulating in FP32 and returning BF16.
// Inputs/output are contiguous row-major, Y = X @ W^T. K must be divisible
// by 64 and N by 128. Any positive row count is supported by internal padding.
// Construction selects a fixed SM120 MLP kernel or a cuBLASLt algorithm and
// must occur outside capture.
// The handle outlives the plan. A plan is not safe for simultaneous host calls.
class Plan {
 public:
  Plan(cublasLtHandle_t handle, std::uint32_t rows,
       std::uint32_t input_width, std::uint32_t output_width);
  ~Plan();
  Plan(Plan&&) noexcept;
  Plan& operator=(Plan&&) noexcept;
  Plan(const Plan&) = delete;
  Plan& operator=(const Plan&) = delete;

  std::size_t scratch_bytes() const;

  // All buffers are on device and must not overlap. Scratch is 256B aligned
  // and remains valid until stream completion. It may be shared by sequential
  // calls on one stream. run performs no allocation or synchronization and is
  // CUDA-graph capturable; activation quantization uses this weight's input
  // global scale plus dynamically computed K16 scales rounded to E4M3.
  // With gelu_tanh_multiply, input is [rows,2*K] containing gate/up slices;
  // BF16(GELU(gate)) and BF16(product) rounding precede FP4 quantization.
  void run(const __nv_bfloat16* input, Weight weight, __nv_bfloat16* output,
           void* scratch, std::size_t scratch_capacity,
           cudaStream_t stream = nullptr,
           InputTransform transform = InputTransform::identity);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gewell::nvfp4
