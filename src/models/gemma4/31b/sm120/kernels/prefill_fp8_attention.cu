#include "gewell/fp8_attention.h"
#include "fp8_scaled_gemm.cuh"

#include <cublasLt.h>
#include <cuda_fp8.h>
#include <map>
#include <stdexcept>
#include <string>

namespace gewell::prefill_primitives {
namespace {
constexpr unsigned Tile = 1024, Heads = 32, MaxD = 512;
constexpr std::size_t Staged = 16 * Tile * 256;
constexpr std::size_t Workspace = 4 * 1024 * 1024;
void check(cudaError_t s) {
  if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
void check(cublasStatus_t s) {
  if (s != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("FP8 attention cuBLASLt status " + std::to_string(s));
}
std::size_t aligned(std::size_t n) { return (n + 255) / 256 * 256; }

// Each warp quantizes one Q/K row. All 1024 K rows are written, including
// zero padding, so partial key tiles use the same aligned matmul plans.
__global__ void pack_rows(const __nv_bfloat16* input, unsigned char* output,
                          float* scales, unsigned rows, unsigned valid_rows,
                          unsigned d) {
  const unsigned row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
  const bool valid = row % rows < valid_rows;
  float values[16];
  float maximum = 0;
  for (unsigned i = 0; i < d / 32; ++i) {
    values[i] = valid ? __bfloat162float(input[std::size_t(row) * d + lane + i * 32]) : 0;
    maximum = fmaxf(maximum, fabsf(values[i]));
  }
  for (unsigned delta = 16; delta; delta /= 2)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, delta));
  const float scale = maximum > 0 ? maximum / 448.0F : 1.0F;
  if (!lane) scales[row] = scale;
  for (unsigned i = 0; i < d / 32; ++i)
    output[std::size_t(row) * d + lane + i * 32] =
        __nv_cvt_float_to_fp8(values[i] / scale, __NV_SATFINITE, __NV_E4M3);
}

// Per-channel V scales are constant along the P*V contraction, allowing
// native FP8 multiplication followed by one FP32 scale per output channel.
// Storage scales are already applied by the existing fused-cache gather.
__global__ void pack_values(const __nv_bfloat16* input, unsigned char* output,
                            float* scales, unsigned count, unsigned d) {
  // Eight channels per CTA: read each value once, reduce its channel maximum,
  // then transpose the packed bytes through shared memory for coalesced stores.
  __shared__ float maxima[8][8];
  __shared__ unsigned char packed[8][Tile];
  const unsigned channel = threadIdx.x % 8, warp = threadIdx.x / 32;
  const unsigned dimension = blockIdx.x * 8 + channel, head = blockIdx.y;
  float values[32], maximum = 0;
  for (unsigned i = 0; i < 32; ++i) {
    const unsigned t = threadIdx.x / 8 + i * 32;
    values[i] = t < count ? float(input[(head * Tile + t) * d + dimension]) : 0;
    maximum = fmaxf(maximum, fabsf(values[i]));
  }
  maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, 8));
  maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, 16));
  if (threadIdx.x % 32 < 8) maxima[warp][channel] = maximum;
  __syncthreads();
  for (unsigned w = 0; w < 8; ++w) maximum = fmaxf(maximum, maxima[w][channel]);
  const float scale = maximum > 0 ? maximum / 448.0F : 1.0F;
  if (threadIdx.x < 8) scales[head * d + dimension] = scale / 256.0F;
  for (unsigned i = 0; i < 32; ++i)
    packed[channel][threadIdx.x / 8 + i * 32] =
        __nv_cvt_float_to_fp8(values[i] / scale, __NV_SATFINITE, __NV_E4M3);
  __syncthreads();
  for (unsigned i = threadIdx.x; i < 8 * Tile / 4; i += blockDim.x)
    reinterpret_cast<unsigned*>(output + (head * d + blockIdx.x * 8) * Tile)[i] =
        reinterpret_cast<const unsigned*>(packed)[i];
}

struct Matmul {
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t a{}, b{}, c{};
  cublasLtMatmulAlgo_t algorithm{};
  std::size_t workspace{};
  ~Matmul() {
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (op) cublasLtMatmulDescDestroy(op);
  }
  void initialize(cublasLtHandle_t handle, unsigned m, unsigned n, unsigned k,
                  unsigned ldc, unsigned batches) {
    check(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const auto trans = CUBLAS_OP_T;
    check(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    // Keep FAST_ACCUM disabled: FP8 operands, FP32 accumulation/output.
    check(cublasLtMatrixLayoutCreate(&a, CUDA_R_8F_E4M3, k, m, k));
    check(cublasLtMatrixLayoutCreate(&b, CUDA_R_8F_E4M3, k, n, k));
    check(cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, m, n, ldc));
    const std::int64_t strides[]{std::int64_t(k) * m, std::int64_t(k) * n, std::int64_t(ldc) * n};
    const cublasLtMatrixLayout_t layouts[]{a, b, c};
    for (unsigned i = 0; i < 3; ++i) {
      check(cublasLtMatrixLayoutSetAttribute(layouts[i], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batches, sizeof(batches)));
      check(cublasLtMatrixLayoutSetAttribute(layouts[i], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strides[i], sizeof(strides[i])));
    }
    cublasLtMatmulPreference_t preference{};
    check(cublasLtMatmulPreferenceCreate(&preference));
    auto status = cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &Workspace, sizeof(Workspace));
    cublasLtMatmulHeuristicResult_t result{};
    int count = 0;
    if (status == CUBLAS_STATUS_SUCCESS)
      status = cublasLtMatmulAlgoGetHeuristic(handle, op, a, b, c, c, preference, 1, &result, &count);
    cublasLtMatmulPreferenceDestroy(preference);
    check(status);
    if (!count || result.state != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("No native FP8 attention algorithm: M=" + std::to_string(m) + " N=" + std::to_string(n) + " K=" + std::to_string(k));
    algorithm = result.algo;
    workspace = result.workspaceSize;
  }
  void run(cublasLtHandle_t handle, const void* left, const void* right,
           float* output, void* scratch, cudaStream_t stream) {
    const float alpha = 1, beta = 0;
    check(cublasLtMatmul(handle, op, &alpha, left, a, right, b, &beta,
                          output, c, output, c, &algorithm, scratch, workspace, stream));
  }
};
struct Plans {
  Matmul qk;
  std::unique_ptr<fp8::detail::ScaledGemm<float>> pv;
  std::unique_ptr<fp8::detail::ScaledGemm<float, 128, 128, 64>> pv_wide;
};
}  // namespace

struct Fp8Attention::Impl {
  cublasLtHandle_t handle{};
  void* allocation{};
  unsigned char *q{}, *k{}, *v{};
  float *qs{}, *ks{}, *vs{};
  void* workspace{};
  std::size_t bytes{};
  unsigned max_rows{}, rows{}, d{}, heads{};
  std::map<std::pair<unsigned, unsigned>, std::unique_ptr<Plans>> plans;
  Plans* current{};
  ~Impl() {
    plans.clear();
    if (allocation) cudaFree(allocation);
    if (handle) cublasLtDestroy(handle);
  }
};

Fp8Attention::Fp8Attention(unsigned max_rows) : impl_(std::make_unique<Impl>()) {
  if (!max_rows || max_rows > 4096) throw std::invalid_argument("FP8 attention capacity must be 1..4096");
  auto& p = *impl_;
  p.max_rows = max_rows;
  check(cublasLtCreate(&p.handle));
  const auto q_bytes = aligned(std::size_t(Heads) * max_rows * MaxD);
  const auto qs_bytes = aligned(std::size_t(Heads) * max_rows * sizeof(float));
  const auto ks_bytes = 16 * Tile * sizeof(float), vs_bytes = 16 * MaxD * sizeof(float);
  p.bytes = q_bytes + 2 * Staged + qs_bytes + ks_bytes + vs_bytes + Workspace;
  check(cudaMalloc(&p.allocation, p.bytes));
  p.q = static_cast<unsigned char*>(p.allocation);
  p.k = p.q + q_bytes;
  p.v = p.k + Staged;
  p.qs = reinterpret_cast<float*>(p.v + Staged);
  p.ks = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(p.qs) + qs_bytes);
  p.vs = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(p.ks) + ks_bytes);
  p.workspace = reinterpret_cast<unsigned char*>(p.vs) + vs_bytes;
}
Fp8Attention::~Fp8Attention() = default;
const float* Fp8Attention::query_scales() const { return impl_->qs; }
const float* Fp8Attention::key_scales() const { return impl_->ks; }
std::size_t Fp8Attention::scratch_bytes() const { return impl_->bytes; }

void Fp8Attention::prepare(const __nv_bfloat16* query, unsigned rows,
                           unsigned d, unsigned heads, cudaStream_t stream) {
  auto& p = *impl_;
  if (!rows || rows > p.max_rows || !((d == 256 && heads == 16) || (d == 512 && heads == 4)))
    throw std::invalid_argument("Invalid FP8 attention shape");
  const auto key = std::make_pair(d, rows);
  auto found = p.plans.find(key);
  if (found == p.plans.end()) {
    auto plan = std::make_unique<Plans>();
    plan->qk.initialize(p.handle, Tile, rows * Heads / heads, d, Tile, heads);
    // The wider N tile amortizes PV traffic for global query batches. Keep
    // the narrow tile for short tails and local attention (offline sweep).
    if (d == MaxD && rows >= 128)
      plan->pv_wide = std::make_unique<fp8::detail::ScaledGemm<float, 128, 128, 64>>(
          d, rows * Heads / heads, Tile, heads, MaxD, Workspace);
    else
      plan->pv = std::make_unique<fp8::detail::ScaledGemm<float>>(
          d, rows * Heads / heads, Tile, heads, MaxD, Workspace);
    if ((plan->pv ? plan->pv->workspace_bytes() : plan->pv_wide->workspace_bytes()) > Workspace)
      throw std::runtime_error("FP8 attention PV exceeds workspace bound");
    found = p.plans.emplace(key, std::move(plan)).first;
  }
  p.current = found->second.get(); p.rows = rows; p.d = d; p.heads = heads;
  pack_rows<<<Heads * rows / 8, 256, 0, stream>>>(query, p.q, p.qs, rows, rows, d);
  check(cudaGetLastError());
}

void Fp8Attention::qk(const __nv_bfloat16* key, const __nv_bfloat16* value,
                      unsigned count, float* scores, cudaStream_t stream) {
  auto& p = *impl_;
  pack_rows<<<p.heads * Tile / 8, 256, 0, stream>>>(key, p.k, p.ks, Tile, count, p.d);
  pack_values<<<dim3(p.d / 8, p.heads), 256, 0, stream>>>(value, p.v, p.vs, count, p.d);
  check(cudaGetLastError());
  p.current->qk.run(p.handle, p.k, p.q, scores, p.workspace, stream);
}

void Fp8Attention::pv(const void* probabilities, float* numerator,
                      bool first, cudaStream_t stream) {
  auto& p = *impl_;
  if (p.current->pv_wide)
    p.current->pv_wide->run(p.v, probabilities, p.vs, numerator, !first, p.workspace, stream);
  else
    p.current->pv->run(p.v, probabilities, p.vs, numerator, !first, p.workspace, stream);
}

}  // namespace gewell::prefill_primitives
