#include "gewell/fp8_linear.h"
#include "bf16_activation.cuh"
#include "fp8_scaled_gemm.cuh"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace gewell::fp8 {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("FP8 ") + operation + ": " +
                             cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("FP8 ") + operation +
                             ": cuBLAS status " + std::to_string(status));
  }
}

std::size_t align256(std::size_t value) { return (value + 255) & ~std::size_t(255); }

template<bool Gelu>
__global__ void quantize(const __nv_bfloat16* input, std::uint16_t* packed,
                         std::size_t pairs,
                         float inverse_scale, unsigned width) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < pairs; i += std::size_t(blockDim.x) * gridDim.x) {
    float2 values;
    if constexpr (Gelu) {
      const auto source = i / (width / 2) * (2 * width) + 2 * (i % (width / 2));
      values = make_float2(
          gewell::detail::gelu_tanh_multiply_bf16(float(input[source]), float(input[source + width])),
          gewell::detail::gelu_tanh_multiply_bf16(float(input[source + 1]), float(input[source + width + 1])));
    } else {
      values = make_float2(float(input[2 * i]), float(input[2 * i + 1]));
    }
    values.x = __fmul_rn(values.x, inverse_scale);
    values.y = __fmul_rn(values.y, inverse_scale);
    packed[i] = __nv_cvt_float2_to_fp8x2(values, __NV_SATFINITE, __NV_E4M3);
  }
}

}  // namespace

struct Plan::Impl {
  cublasLtHandle_t handle{};
  cublasLtMatmulDesc_t operation{};
  cublasLtMatrixLayout_t weight_layout{}, input_layout{}, output_layout{};
  cublasLtMatmulAlgo_t algorithm{};
  std::uint32_t rows{}, input_width{};
  std::size_t workspace_offset{}, workspace_bytes{}, total_bytes{};
  std::unique_ptr<detail::ScaledGemm<cutlass::bfloat16_t>> joined;
  std::unique_ptr<detail::ScaledGemm<cutlass::bfloat16_t, 128, 128, 64>> joined_medium;
  std::unique_ptr<detail::ScaledGemm<cutlass::bfloat16_t, 256, 128, 64>> joined_large;

  bool is_joined() const { return joined || joined_medium || joined_large; }

  ~Impl() {
    if (output_layout) cublasLtMatrixLayoutDestroy(output_layout);
    if (input_layout) cublasLtMatrixLayoutDestroy(input_layout);
    if (weight_layout) cublasLtMatrixLayoutDestroy(weight_layout);
    if (operation) cublasLtMatmulDescDestroy(operation);
  }
};

Plan::Plan(cublasLtHandle_t handle, std::uint32_t rows,
           std::uint32_t input_width, std::uint32_t output_width, bool channel_scales)
    : impl_(std::make_unique<Impl>()) {
  if (!handle || rows == 0 ||
      rows > std::uint32_t(std::numeric_limits<int>::max()) ||
      input_width == 0 || input_width % 16 ||
      input_width > std::uint32_t(std::numeric_limits<int>::max()) ||
      output_width == 0 || output_width % 16 ||
      output_width > std::uint32_t(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(
        "FP8 plan requires positive rows, K and N divisible by16, "
        "dimensions within int range, and a cuBLASLt handle");
  }
  auto& p = *impl_;
  p.handle = handle;
  p.rows = rows;
  p.input_width = input_width;
  if (channel_scales) {
    // The narrow tile serves decode/verification; wider N/K work improves
    // prefill. Selection is offline, with the same scaled FP32 epilogue.
    if (rows <= 64) {
      p.joined = std::make_unique<detail::ScaledGemm<cutlass::bfloat16_t>>(
          output_width, rows, input_width);
      p.workspace_bytes = p.joined->workspace_bytes();
    } else if (rows <= 256) {
      p.joined_medium = std::make_unique<detail::ScaledGemm<cutlass::bfloat16_t, 128, 128, 64>>(
          output_width, rows, input_width);
      p.workspace_bytes = p.joined_medium->workspace_bytes();
    } else {
      p.joined_large = std::make_unique<detail::ScaledGemm<cutlass::bfloat16_t, 256, 128, 64>>(
          output_width, rows, input_width);
      p.workspace_bytes = p.joined_large->workspace_bytes();
    }
    if (p.workspace_bytes > kMaxWorkspaceBytes)
      throw std::runtime_error("FP8 joined GEMM exceeds workspace bound");
    p.workspace_offset = align256(std::size_t(rows) * input_width);
    p.total_bytes = p.workspace_offset + p.workspace_bytes;
    return;
  }
  check_cublas(cublasLtMatmulDescCreate(&p.operation, CUBLAS_COMPUTE_32F,
                                       CUDA_R_32F),
               "create matmul descriptor");
  const cublasOperation_t transpose = CUBLAS_OP_T;
  check_cublas(cublasLtMatmulDescSetAttribute(
                   p.operation, CUBLASLT_MATMUL_DESC_TRANSA,
                   &transpose, sizeof(transpose)),
               "set weight transpose");
  // Scalar scale pointers default to unity; alpha applies the two globals.
  // Leave FAST_ACCUM disabled so FP8 partial sums are promoted to FP32.
  // Column-major (W^T)^T @ X^T produces row-major X @ W^T directly.
  check_cublas(cublasLtMatrixLayoutCreate(
                   &p.weight_layout, CUDA_R_8F_E4M3, input_width,
                   output_width, input_width),
               "create weight layout");
  check_cublas(cublasLtMatrixLayoutCreate(
                   &p.input_layout, CUDA_R_8F_E4M3, input_width,
                   rows, input_width),
               "create activation layout");
  check_cublas(cublasLtMatrixLayoutCreate(
                   &p.output_layout, CUDA_R_16BF, output_width,
                   rows, output_width),
               "create output layout");
  cublasLtMatmulPreference_t preference{};
  check_cublas(cublasLtMatmulPreferenceCreate(&preference), "create preference");
  const auto preference_status = cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &kMaxWorkspaceBytes, sizeof(kMaxWorkspaceBytes));
  if (preference_status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulPreferenceDestroy(preference);
    check_cublas(preference_status, "set workspace limit");
  }
  // The caller promises 16B weight/output alignment. B is our 256B-aligned
  // scratch; do not let the default 256B preference overstate A/C/D alignment.
  const std::uint32_t alignment = 16;
  for (const auto attribute : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                               CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                               CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES}) {
    const auto status = cublasLtMatmulPreferenceSetAttribute(
        preference, attribute, &alignment, sizeof(alignment));
    if (status != CUBLAS_STATUS_SUCCESS) {
      cublasLtMatmulPreferenceDestroy(preference);
      check_cublas(status, "set matrix alignment");
    }
  }
  cublasLtMatmulHeuristicResult_t candidates[8]{};
  int count = 0;
  const auto heuristic_status = cublasLtMatmulAlgoGetHeuristic(
      handle, p.operation, p.weight_layout, p.input_layout, p.output_layout,
      p.output_layout, preference, 8, candidates, &count);
  cublasLtMatmulPreferenceDestroy(preference);
  check_cublas(heuristic_status, "select native FP8 algorithm");
  if (count == 0 || candidates[0].state != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        "FP8: no native E4M3 cuBLASLt algorithm for rows=" +
        std::to_string(rows) + " K=" + std::to_string(input_width) +
        " N=" + std::to_string(output_width));
  }
  // Offline measurements on SM120 with rotating, real model weights. Select
  // among algorithms validated by cuBLASLt for these exact layouts/alignment;
  // other shapes keep its first choice. There is no serving-time autotuning.
  auto tile = CUBLASLT_MATMUL_TILE_UNDEFINED;
  int split_k = 1;
  if (rows == 1 && input_width == 5376 && output_width == 16384) {
    tile = CUBLASLT_MATMUL_TILE_128x64;
  } else if ((rows <= 32 &&
              ((input_width == 8192 && output_width == 5376) ||
               (input_width == 5376 && output_width == 21504))) ||
             (rows <= 16 && input_width == 5376 &&
              (output_width == 2048 || output_width == 4096))) {
    tile = CUBLASLT_MATMUL_TILE_64x32;
  } else if ((rows > 1 && rows <= 32 && input_width == 5376 && output_width == 16384) ||
             ((rows == 64 || rows == 128) &&
              ((input_width == 5376 && (output_width == 4096 || output_width == 8192)) ||
               (input_width == 8192 && output_width == 5376)))) {
    tile = CUBLASLT_MATMUL_TILE_64x64;
  } else if (rows <= 32 && input_width == 16384 && output_width == 5376) {
    tile = CUBLASLT_MATMUL_TILE_64x64;
    split_k = 4;
  } else if (rows == 1024 && input_width == 5376 && output_width == 8192) {
    tile = CUBLASLT_MATMUL_TILE_128x128;
  }
  auto result = candidates[0];
  for (int i = 0; tile != CUBLASLT_MATMUL_TILE_UNDEFINED && i < count; ++i) {
    if (candidates[i].state != CUBLAS_STATUS_SUCCESS) continue;
    int candidate_tile = 0, candidate_split = 0;
    std::size_t written = 0;
    check_cublas(cublasLtMatmulAlgoConfigGetAttribute(
        &candidates[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
        &candidate_tile, sizeof(candidate_tile), &written), "read algorithm tile");
    check_cublas(cublasLtMatmulAlgoConfigGetAttribute(
        &candidates[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
        &candidate_split, sizeof(candidate_split), &written), "read algorithm split K");
    if (candidate_tile == tile && candidate_split == split_k) {
      result = candidates[i];
      break;
    }
  }
  p.algorithm = result.algo;
  p.workspace_bytes = result.workspaceSize;
  p.workspace_offset = align256(std::size_t(rows) * input_width);
  p.total_bytes = p.workspace_offset + p.workspace_bytes;
}

Plan::~Plan() = default;
Plan::Plan(Plan&&) noexcept = default;
Plan& Plan::operator=(Plan&&) noexcept = default;

std::size_t Plan::scratch_bytes() const { return impl_->total_bytes; }

void Plan::run(const __nv_bfloat16* input, Weight weight, __nv_bfloat16* output,
                void* scratch, std::size_t scratch_capacity, cudaStream_t stream,
                InputTransform transform) {
  auto& p = *impl_;
  const float alpha = weight.input_scale * weight.weight_scale;
  if (!input || !weight.data || !output || !scratch ||
      std::uintptr_t(input) % alignof(__nv_bfloat16) ||
      std::uintptr_t(weight.data) % 16 || std::uintptr_t(output) % 16 ||
      std::uintptr_t(scratch) % 256 || scratch_capacity < p.total_bytes ||
      !(weight.input_scale > 0.0F) || !std::isfinite(weight.input_scale) ||
      !(weight.weight_scale > 0.0F) || !std::isfinite(weight.weight_scale) ||
      !(alpha > 0.0F) || !std::isfinite(alpha) ||
      !std::isfinite(1.0F / weight.input_scale) ||
      (p.is_joined() && (!weight.channel_scales || std::uintptr_t(weight.channel_scales) % 16)) ||
      (transform != InputTransform::identity && transform != InputTransform::gelu_tanh_multiply)) {
    throw std::invalid_argument(
        "FP8 run requires valid aligned device buffers, sufficient scratch, "
        "and finite positive global scales with a finite positive product "
        "and finite input-scale reciprocal");
  }
  auto* packed = static_cast<std::uint8_t*>(scratch);
  const std::size_t pairs = std::size_t(p.rows) * p.input_width / 2;
  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>((pairs + 255) / 256, 65535));
  if (transform == InputTransform::gelu_tanh_multiply)
    quantize<true><<<blocks, 256, 0, stream>>>(
        input, reinterpret_cast<std::uint16_t*>(packed), pairs, 1.0F / weight.input_scale, p.input_width);
  else
    quantize<false><<<blocks, 256, 0, stream>>>(
        input, reinterpret_cast<std::uint16_t*>(packed), pairs, 1.0F / weight.input_scale, p.input_width);
  check_cuda(cudaGetLastError(), "quantize activations");
  if (p.is_joined()) {
    auto* workspace = packed + p.workspace_offset;
    if (p.joined) p.joined->run(weight.data, packed, weight.channel_scales, output, false, workspace, stream);
    else if (p.joined_medium) p.joined_medium->run(weight.data, packed, weight.channel_scales, output, false, workspace, stream);
    else p.joined_large->run(weight.data, packed, weight.channel_scales, output, false, workspace, stream);
    return;
  }
  constexpr float beta = 0.0F;
  check_cublas(cublasLtMatmul(
                   p.handle, p.operation, &alpha, weight.data, p.weight_layout,
                   packed, p.input_layout, &beta, output, p.output_layout,
                   output, p.output_layout, &p.algorithm,
                   packed + p.workspace_offset, p.workspace_bytes, stream),
               "native FP8 matmul");
}

}  // namespace gewell::fp8
