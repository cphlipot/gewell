#include "bf16_common.cuh"
#include "rope_element.cuh"
#include <algorithm>
#include <cmath>

namespace gewell::bf16_primitives {
namespace {

using namespace detail;

constexpr unsigned kHiddenNormThreads = 1024;
constexpr unsigned kHiddenNormWarps =
    kHiddenNormThreads / kWarpThreads;
constexpr float kRmsNormEpsilon = 1.0e-6F;
static_assert(kHiddenNormThreads % kWarpThreads == 0);
static_assert(gemma4_31b::kHiddenSize > kHiddenNormThreads);

__device__ void norm_head(const BFloat16* input, const BFloat16* weight,
                          BFloat16* output, std::uint32_t width,
                          float epsilon, bool scaled) {
  __shared__ float sums[kThreads];
  float sum = 0.0F;
  for (std::uint32_t index = threadIdx.x; index < width;
       index += blockDim.x) {
    const float value = __bfloat162float(input[index]);
    sum += value * value;
  }
  sums[threadIdx.x] = sum;
  __syncthreads();

  for (unsigned offset = kThreads / 2; offset != 0; offset /= 2) {
    if (threadIdx.x < offset) {
      sums[threadIdx.x] += sums[threadIdx.x + offset];
    }
    __syncthreads();
  }

  const float inverse_rms = powf(sums[0] / static_cast<float>(width) + epsilon,
                                 -0.5F);
  for (std::uint32_t index = threadIdx.x; index < width;
       index += blockDim.x) {
    const float normalized =
        __bfloat162float(input[index]) * inverse_rms;
    // Gemma 4's generated Transformers class stores a direct multiplicative
    // scale. It does not use the historical Gemma (1 + weight) convention.
    output[index] =
        __float2bfloat16_rn(scaled ? normalized * __bfloat162float(weight[index])
                                  : normalized);
  }
}

__global__ void rms_norm_kernel(const BFloat16* input,
                                const BFloat16* weight, BFloat16* output,
                                std::uint32_t width, float epsilon, bool scaled) {
  const auto offset = std::size_t(blockIdx.x) * width;
  norm_head(input + offset, weight, output + offset, width, epsilon, scaled);
}

template<bool Global>
__global__ void qkv_rms_norm_kernel(const BFloat16* input,
                                    const BFloat16* q_weight,
                                    const BFloat16* k_weight, BFloat16* query,
                                    BFloat16* key, BFloat16* value) {
  constexpr unsigned d = Global ? 512 : 256, heads = Global ? 4 : 16;
  constexpr unsigned input_heads = 32 + (Global ? heads : 2 * heads);
  const auto row = blockIdx.x / (32 + 2 * heads);
  const auto head = blockIdx.x % (32 + 2 * heads);
  const bool q = head < 32, v = head >= 32 + heads;
  const auto input_head = Global && v ? head - heads : head;
  const auto output_head = q ? head : (head - 32) % heads;
  auto* output = q ? query : v ? value : key;
  norm_head(input + (std::size_t(row) * input_heads + input_head) * d,
            q ? q_weight : k_weight,
            output + (std::size_t(row) * (q ? 32 : heads) + output_head) * d,
            d, kRmsNormEpsilon, !v);
}

constexpr unsigned kQkvRopeBatchEntries = 16;
struct QkvRopeBatch {
  QkvRopeInput inputs[kQkvRopeBatchEntries];
  unsigned end_rows[kQkvRopeBatchEntries];
};

// One warp per head, shared by prefill and batched verification. Recreate
// norm_head's 256-thread reduction: first fold dimensions 256 apart, then
// offsets 128/64/32 and finally 16/8/4/2/1. Normalized RoPE pairs stay in the
// same lane's registers, preserving each BF16 rounding boundary.
template<bool Global>
__device__ __forceinline__ void qkv_rms_rope_head(const QkvRopeInput& input,
    const BFloat16* q_weight, const BFloat16* k_weight, unsigned head, unsigned row) {
  constexpr unsigned D = Global ? 512 : 256, heads = Global ? 4 : 16;
  constexpr unsigned input_heads = 32 + (Global ? heads : 2 * heads);
  constexpr unsigned elements = D / kWarpThreads;
  const unsigned lane = threadIdx.x % kWarpThreads;
  const bool q = head < 32, v = head >= 32 + heads;
  const unsigned source_head = Global && v ? head - heads : head;
  const unsigned output_head = q ? head : (head - 32) % heads;
  const auto* source = input.qkv + (std::size_t(row) * input_heads + source_head) * D;
  float values[elements];
  float sums[8];
#pragma unroll
  for (unsigned i = 0; i < elements; ++i)
    values[i] = __bfloat162float(source[i * kWarpThreads + lane]);
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) {
    sums[i] = values[i] * values[i];
    if constexpr (Global) sums[i] += values[i + 8] * values[i + 8];
  }
#pragma unroll
  for (unsigned offset = 4; offset; offset /= 2)
#pragma unroll
    for (unsigned i = 0; i < offset; ++i) sums[i] += sums[i + offset];
  const float sum = warp_sum(sums[0]);
  const float inverse_rms = powf(
      __shfl_sync(0xffffffffU, sum, 0) / static_cast<float>(D) + kRmsNormEpsilon, -0.5F);
#pragma unroll
  for (unsigned i = 0; i < elements; ++i) {
    const float normalized = values[i] * inverse_rms;
    values[i] = __bfloat162float(__float2bfloat16_rn(v ? normalized : normalized *
        __bfloat162float((q ? q_weight : k_weight)[i * kWarpThreads + lane])));
  }
  auto* output = q ? input.query : v ? input.value : input.key;
  const auto offset = v ? (std::size_t(row) * heads + output_head) * D
                        : (std::size_t(output_head) * input.rows + row) * D;
#pragma unroll
  for (unsigned i = 0; i < elements; ++i) {
    const unsigned d = i * kWarpThreads + lane;
    BFloat16 result = __float2bfloat16_rn(values[i]);
    if (!v) {
      constexpr unsigned half = elements / 2;
      const float rotated = i < half ? -values[i + half] : values[i - half];
      const auto factor = std::size_t(row) * D + d;
      const BFloat16 direct = __float2bfloat16_rn(values[i] * __bfloat162float(input.cosine[factor]));
      const BFloat16 cross = __float2bfloat16_rn(rotated * __bfloat162float(input.sine[factor]));
      result = __float2bfloat16_rn(__bfloat162float(direct) + __bfloat162float(cross));
    }
    output[offset + d] = result;
  }
}

template<bool Global>
__global__ void qkv_rms_rope_prefill_kernel(
    const __grid_constant__ QkvRopeInput input,
    const BFloat16* q_weight, const BFloat16* k_weight) {
  const unsigned head = blockIdx.x * (kThreads / kWarpThreads) + threadIdx.x / kWarpThreads;
  qkv_rms_rope_head<Global>(input, q_weight, k_weight, head, blockIdx.y);
}

template<bool Global, bool WarpHeads>
__global__ void qkv_rms_rope_batch_kernel(const __grid_constant__ QkvRopeBatch batch,
                                         const BFloat16* q_weight,
                                         const BFloat16* k_weight) {
  unsigned request = 0;
  while (blockIdx.y >= batch.end_rows[request]) ++request;
  const auto& input = batch.inputs[request];
  const unsigned row = blockIdx.y - (request ? batch.end_rows[request - 1] : 0);
  if constexpr (WarpHeads) {
    const unsigned head = blockIdx.x * (kThreads / kWarpThreads) + threadIdx.x / kWarpThreads;
    qkv_rms_rope_head<Global>(input, q_weight, k_weight, head, row);
  } else {
    // Tiny cohorts need a CTA per head to expose enough memory parallelism.
    constexpr unsigned D = Global ? 512 : 256, heads = Global ? 4 : 16;
    constexpr unsigned input_heads = 32 + (Global ? heads : 2 * heads);
    const unsigned head = blockIdx.x;
    const bool q = head < 32, v = head >= 32 + heads;
    const unsigned source_head = Global && v ? head - heads : head;
    const unsigned output_head = q ? head : (head - 32) % heads;
    __shared__ BFloat16 normalized[D];
    norm_head(input.qkv + (std::size_t(row) * input_heads + source_head) * D,
        q ? q_weight : k_weight, normalized, D, kRmsNormEpsilon, !v);
    __syncthreads();
    auto* output = q ? input.query : v ? input.value : input.key;
    const auto offset = v ? (std::size_t(row) * heads + output_head) * D
                          : (std::size_t(output_head) * input.rows + row) * D;
    for (unsigned d = threadIdx.x; d < D; d += blockDim.x)
      output[offset + d] = v ? normalized[d] : rope_element<D>(normalized,
          input.cosine + std::size_t(row) * D, input.sine + std::size_t(row) * D, d);
  }
}

__device__ __forceinline__ float hidden_block_sum(float value,
                                                   float* warp_sums) {
  const unsigned lane = threadIdx.x % kWarpThreads;
  const unsigned warp = threadIdx.x / kWarpThreads;
  value = warp_sum(value);
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  if (warp == 0) {
    value = lane < kHiddenNormWarps ? warp_sums[lane] : 0.0F;
    value = warp_sum(value);
    if (lane == 0) {
      warp_sums[0] = value;
    }
  }
  __syncthreads();
  return warp_sums[0];
}

__device__ __forceinline__ float hidden_inverse_rms(
    const BFloat16* values, float* warp_sums) {
  float sum = 0.0F;
  for (std::uint32_t index = threadIdx.x;
       index < gemma4_31b::kHiddenSize; index += kHiddenNormThreads) {
    const float value = __bfloat162float(values[index]);
    sum += value * value;
  }
  return rsqrtf(hidden_block_sum(sum, warp_sums) /
                    static_cast<float>(gemma4_31b::kHiddenSize) +
                kRmsNormEpsilon);
}

__global__ __launch_bounds__(kHiddenNormThreads, 1)
void rms_norm_hidden_m1_kernel(const BFloat16* input,
                               const BFloat16* weight,
                               BFloat16* output) {
  __shared__ float warp_sums[kHiddenNormWarps];
  const std::size_t row_offset =
      static_cast<std::size_t>(blockIdx.x) * gemma4_31b::kHiddenSize;
  input += row_offset;
  output += row_offset;
  const float inverse_rms = hidden_inverse_rms(input, warp_sums);
  for (std::uint32_t index = threadIdx.x;
       index < gemma4_31b::kHiddenSize; index += kHiddenNormThreads) {
    output[index] = __float2bfloat16_rn(
        __bfloat162float(input[index]) * inverse_rms *
        __bfloat162float(weight[index]));
  }
}

__global__ __launch_bounds__(kHiddenNormThreads, 1)
void post_attention_residual_pre_feedforward_norm_m1_kernel(
    BFloat16* branch_state, const BFloat16* post_attention_weight,
    const BFloat16* residual, const BFloat16* pre_feedforward_weight,
    BFloat16* normalized) {
  __shared__ float warp_sums[kHiddenNormWarps];
  const std::size_t row_offset =
      static_cast<std::size_t>(blockIdx.x) * gemma4_31b::kHiddenSize;
  branch_state += row_offset;
  residual += row_offset;
  normalized += row_offset;
  const float post_inverse_rms = hidden_inverse_rms(branch_state, warp_sums);
  for (std::uint32_t index = threadIdx.x;
       index < gemma4_31b::kHiddenSize; index += kHiddenNormThreads) {
    const BFloat16 post_normalized = __float2bfloat16_rn(
        __bfloat162float(branch_state[index]) * post_inverse_rms *
        __bfloat162float(post_attention_weight[index]));
    branch_state[index] = __float2bfloat16_rn(
        __bfloat162float(residual[index]) +
        __bfloat162float(post_normalized));
  }
  __syncthreads();

  const float pre_inverse_rms = hidden_inverse_rms(branch_state, warp_sums);
  for (std::uint32_t index = threadIdx.x;
       index < gemma4_31b::kHiddenSize; index += kHiddenNormThreads) {
    normalized[index] = __float2bfloat16_rn(
        __bfloat162float(branch_state[index]) * pre_inverse_rms *
        __bfloat162float(pre_feedforward_weight[index]));
  }
}

__global__ __launch_bounds__(kHiddenNormThreads, 1)
void post_feedforward_residual_scalar_next_norm_m1_kernel(
    BFloat16* branch_state, const BFloat16* post_feedforward_weight,
    const BFloat16* residual, const BFloat16* scalar,
    const BFloat16* next_norm_weight, BFloat16* next_normalized) {
  __shared__ float warp_sums[kHiddenNormWarps];
  const std::size_t row_offset =
      static_cast<std::size_t>(blockIdx.x) * gemma4_31b::kHiddenSize;
  branch_state += row_offset;
  residual += row_offset;
  next_normalized += row_offset;
  const float post_inverse_rms = hidden_inverse_rms(branch_state, warp_sums);
  const float layer_scale = __bfloat162float(scalar[0]);
  for (std::uint32_t index = threadIdx.x;
       index < gemma4_31b::kHiddenSize; index += kHiddenNormThreads) {
    const BFloat16 post_normalized = __float2bfloat16_rn(
        __bfloat162float(branch_state[index]) * post_inverse_rms *
        __bfloat162float(post_feedforward_weight[index]));
    const BFloat16 summed = __float2bfloat16_rn(
        __bfloat162float(residual[index]) +
        __bfloat162float(post_normalized));
    branch_state[index] = __float2bfloat16_rn(
        __bfloat162float(summed) * layer_scale);
  }
  __syncthreads();

  const float next_inverse_rms = hidden_inverse_rms(branch_state, warp_sums);
  for (std::uint32_t index = threadIdx.x;
       index < gemma4_31b::kHiddenSize; index += kHiddenNormThreads) {
    next_normalized[index] = __float2bfloat16_rn(
        __bfloat162float(branch_state[index]) * next_inverse_rms *
        __bfloat162float(next_norm_weight[index]));
  }
}

__global__ void residual_add_kernel(const BFloat16* residual,
                                    const BFloat16* branch,
                                    BFloat16* output,
                                    std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = __float2bfloat16_rn(__bfloat162float(residual[index]) +
                                        __bfloat162float(branch[index]));
  }
}

constexpr unsigned kPrefillNormThreads = 768;
constexpr unsigned kPrefillNormWarps = kPrefillNormThreads / kWarpThreads;
static_assert(gemma4_31b::kHiddenSize % kPrefillNormThreads == 0);

__device__ float prefill_norm_sum(float value, float* warp_sums) {
  const unsigned lane = threadIdx.x % kWarpThreads;
  const unsigned warp = threadIdx.x / kWarpThreads;
  value = warp_sum(value);
  if (lane == 0) warp_sums[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = warp_sum(lane < kPrefillNormWarps ? warp_sums[lane] : 0.0F);
    if (lane == 0) warp_sums[0] = value;
  }
  __syncthreads();
  return warp_sums[0];
}

template<bool Feedforward>
__global__ void residual_norm_prefill_kernel(
    BFloat16* branch, const BFloat16* post_weight, const BFloat16* residual,
    const BFloat16* scalar, const BFloat16* next_weight, BFloat16* normalized) {
  // Seven values per thread fit in registers, including both BF16 residual
  // boundaries. Separate reduction storage avoids a read/overwrite race
  // between the two norms without adding another block barrier.
  constexpr unsigned elements = gemma4_31b::kHiddenSize / kPrefillNormThreads;
  __shared__ float warp_sums[2][kPrefillNormWarps];
  const auto offset = std::size_t(blockIdx.x) * gemma4_31b::kHiddenSize;
  branch += offset;
  residual += offset;
  float values[elements], sum = 0.0F;
#pragma unroll
  for (unsigned i = 0; i < elements; ++i) {
    values[i] = __bfloat162float(branch[i * kPrefillNormThreads + threadIdx.x]);
    sum += values[i] * values[i];
  }
  const float post_inverse_rms = rsqrtf(prefill_norm_sum(sum, warp_sums[0]) /
      static_cast<float>(gemma4_31b::kHiddenSize) + kRmsNormEpsilon);
  sum = 0.0F;
#pragma unroll
  for (unsigned i = 0; i < elements; ++i) {
    const unsigned d = i * kPrefillNormThreads + threadIdx.x;
    const BFloat16 post = __float2bfloat16_rn(values[i] * post_inverse_rms * __bfloat162float(post_weight[d]));
    BFloat16 state = __float2bfloat16_rn(__bfloat162float(residual[d]) + __bfloat162float(post));
    if constexpr (Feedforward)
      state = __float2bfloat16_rn(__bfloat162float(state) * __bfloat162float(scalar[0]));
    values[i] = __bfloat162float(state);
    branch[d] = state;
    sum += values[i] * values[i];
  }
  if (next_weight) {
    const float next_inverse_rms = rsqrtf(prefill_norm_sum(sum, warp_sums[1]) /
        static_cast<float>(gemma4_31b::kHiddenSize) + kRmsNormEpsilon);
#pragma unroll
    for (unsigned i = 0; i < elements; ++i) {
      const unsigned d = i * kPrefillNormThreads + threadIdx.x;
      normalized[offset + d] = __float2bfloat16_rn(
          values[i] * next_inverse_rms * __bfloat162float(next_weight[d]));
    }
  }
}

__global__ void trained_scalar_kernel(BFloat16* values,
                                      const BFloat16* scalar,
                                      std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    values[index] = __float2bfloat16_rn(__bfloat162float(values[index]) *
                                        __bfloat162float(scalar[0]));
  }
}

}  // namespace

void post_attention_residual_pre_feedforward_norm_prefill(
    BFloat16* branch, const BFloat16* post_weight, const BFloat16* residual,
    const BFloat16* next_weight, BFloat16* normalized, std::uint32_t rows, cudaStream_t stream) {
  constexpr auto operation = "post_attention_residual_pre_feedforward_norm_prefill";
  for (const auto* pointer : {static_cast<const BFloat16*>(branch), post_weight,
                            residual, next_weight, static_cast<const BFloat16*>(normalized)})
    check_pointer(pointer, operation);
  check_decode_rows(rows, operation);
  if (branch == residual || branch == normalized || residual == normalized)
    fail(operation, "branch, residual, and output must be disjoint");
  residual_norm_prefill_kernel<false><<<rows, kPrefillNormThreads, 0, stream>>>(
      branch, post_weight, residual, nullptr, next_weight, normalized);
  check_cuda(cudaGetLastError(), operation);
}

void post_feedforward_residual_scalar_next_norm_prefill(
    BFloat16* branch, const BFloat16* post_weight, const BFloat16* residual,
    const BFloat16* scalar, const BFloat16* next_weight, BFloat16* normalized,
    std::uint32_t rows, cudaStream_t stream) {
  constexpr auto operation = "post_feedforward_residual_scalar_next_norm_prefill";
  for (const auto* pointer : {static_cast<const BFloat16*>(branch), post_weight, residual, scalar})
    check_pointer(pointer, operation);
  if (next_weight) check_pointer(normalized, operation);
  check_decode_rows(rows, operation);
  if (branch == residual || (next_weight && (branch == normalized || residual == normalized)))
    fail(operation, "branch, residual, and output must be disjoint");
  residual_norm_prefill_kernel<true><<<rows, kPrefillNormThreads, 0, stream>>>(
      branch, post_weight, residual, scalar, next_weight, normalized);
  check_cuda(cudaGetLastError(), operation);
}

void rms_norm(const BFloat16* input, const BFloat16* weight,
              BFloat16* output, std::uint32_t rows, std::uint32_t width,
              float epsilon, cudaStream_t stream) {
  check_pointer(input, "rms_norm input");
  check_pointer(weight, "rms_norm weight");
  check_pointer(output, "rms_norm output");
  if (rows == 0) {
    fail("rms_norm", "row count must be positive");
  }
  if (width != gemma4_31b::kVisionHeadSize &&
      width != gemma4_31b::kVisionHiddenSize &&
      width != gemma4_31b::kLocalHeadSize &&
      width != gemma4_31b::kGlobalHeadSize &&
      width != gemma4_31b::kHiddenSize) {
    fail("rms_norm", "width is not a Gemma 4 31B norm width");
  }
  if (!(epsilon > 0.0F) || !std::isfinite(epsilon)) {
    fail("rms_norm", "epsilon must be finite and positive");
  }
  rms_norm_kernel<<<rows, kThreads, 0, stream>>>(input, weight, output, width,
                                                 epsilon, true);
  check_cuda(cudaGetLastError(), "rms_norm kernel launch");
}

void rms_norm_unscaled(const BFloat16* input, BFloat16* output,
                       std::uint32_t rows, std::uint32_t width, float epsilon,
                       cudaStream_t stream) {
  check_pointer(input, "rms_norm_unscaled input");
  check_pointer(output, "rms_norm_unscaled output");
  if (rows == 0) {
    fail("rms_norm_unscaled", "row count must be positive");
  }
  if (width != gemma4_31b::kVisionHeadSize &&
      width != gemma4_31b::kVisionHiddenSize &&
      width != gemma4_31b::kLocalHeadSize &&
      width != gemma4_31b::kGlobalHeadSize &&
      width != gemma4_31b::kHiddenSize) {
    fail("rms_norm_unscaled", "width is not a Gemma 4 31B norm width");
  }
  if (!(epsilon > 0.0F) || !std::isfinite(epsilon)) {
    fail("rms_norm_unscaled", "epsilon must be finite and positive");
  }
  rms_norm_kernel<<<rows, kThreads, 0, stream>>>(
      input, nullptr, output, width, epsilon, false);
  check_cuda(cudaGetLastError(), "rms_norm_unscaled kernel launch");
}

void rms_norm_hidden_m1(const BFloat16* input, const BFloat16* weight,
                        BFloat16* output, cudaStream_t stream) {
  rms_norm_hidden_rows(input, weight, output, 1, stream);
}

void qkv_rms_norm(const BFloat16* qkv, const BFloat16* q_weight,
                  const BFloat16* k_weight, BFloat16* query, BFloat16* key,
                  BFloat16* value, std::uint32_t rows,
                  gemma4_31b::AttentionKind kind, cudaStream_t stream) {
  check_pointer(qkv, "qkv_rms_norm input");
  check_pointer(q_weight, "qkv_rms_norm Q weight");
  check_pointer(k_weight, "qkv_rms_norm K weight");
  check_pointer(query, "qkv_rms_norm Q");
  check_pointer(key, "qkv_rms_norm K");
  check_pointer(value, "qkv_rms_norm V");
  if (!rows || rows > INT_MAX / 64) fail("qkv_rms_norm", "invalid row count");
  if (kind == gemma4_31b::AttentionKind::global)
    qkv_rms_norm_kernel<true><<<rows * 40, kThreads, 0, stream>>>(
        qkv, q_weight, k_weight, query, key, value);
  else if (kind == gemma4_31b::AttentionKind::local)
    qkv_rms_norm_kernel<false><<<rows * 64, kThreads, 0, stream>>>(
        qkv, q_weight, k_weight, query, key, value);
  else fail("qkv_rms_norm", "invalid attention kind");
  check_cuda(cudaGetLastError(), "qkv_rms_norm kernel launch");
}

void qkv_rms_rope_batch(const std::vector<QkvRopeInput>& inputs,
    const BFloat16* q_weight, const BFloat16* k_weight,
    gemma4_31b::AttentionKind kind, cudaStream_t stream) {
  constexpr auto operation = "qkv_rms_rope_batch";
  const bool global = kind == gemma4_31b::AttentionKind::global;
  if (inputs.empty() || (!global && kind != gemma4_31b::AttentionKind::local))
    fail(operation, "invalid batch or attention kind");
  check_pointer(q_weight, operation);
  check_pointer(k_weight, operation);
  for (const auto& input : inputs) {
    if (!input.rows || input.rows > 65535) fail(operation, "invalid row count");
    for (const auto* pointer : {input.qkv, input.cosine, input.sine,
          static_cast<const BFloat16*>(input.query), static_cast<const BFloat16*>(input.key),
          static_cast<const BFloat16*>(input.value)}) check_pointer(pointer, operation);
  }
  QkvRopeBatch batch{};
  unsigned count = 0, rows = 0;
  const auto flush = [&] {
    if (!count) return;
    if (count == 1 && rows >= 32) {
      if (global) qkv_rms_rope_prefill_kernel<true><<<dim3(5, rows), kThreads, 0, stream>>>(batch.inputs[0], q_weight, k_weight);
      else qkv_rms_rope_prefill_kernel<false><<<dim3(8, rows), kThreads, 0, stream>>>(batch.inputs[0], q_weight, k_weight);
    } else if (rows >= 64) {
      if (global) qkv_rms_rope_batch_kernel<true, true><<<dim3(5, rows), kThreads, 0, stream>>>(batch, q_weight, k_weight);
      else qkv_rms_rope_batch_kernel<false, true><<<dim3(8, rows), kThreads, 0, stream>>>(batch, q_weight, k_weight);
    } else {
      if (global) qkv_rms_rope_batch_kernel<true, false><<<dim3(40, rows), kThreads, 0, stream>>>(batch, q_weight, k_weight);
      else qkv_rms_rope_batch_kernel<false, false><<<dim3(64, rows), kThreads, 0, stream>>>(batch, q_weight, k_weight);
    }
    check_cuda(cudaGetLastError(), operation);
    count = rows = 0;
  };
  for (const auto& input : inputs) {
    if (count == kQkvRopeBatchEntries || rows + input.rows > 65535) flush();
    batch.inputs[count] = input;
    batch.end_rows[count++] = rows += input.rows;
  }
  flush();
}

void rms_norm_hidden_rows(const BFloat16* input, const BFloat16* weight,
                          BFloat16* output, std::uint32_t rows,
                          cudaStream_t stream) {
  check_pointer(input, "rms_norm_hidden_rows input");
  check_pointer(weight, "rms_norm_hidden_rows weight");
  check_pointer(output, "rms_norm_hidden_rows output");
  check_decode_rows(rows, "rms_norm_hidden_rows");
  rms_norm_hidden_m1_kernel<<<rows, kHiddenNormThreads, 0, stream>>>(
      input, weight, output);
  check_cuda(cudaGetLastError(), "rms_norm_hidden_rows kernel launch");
}

void post_attention_residual_pre_feedforward_norm_m1(
    BFloat16* branch_state, const BFloat16* post_attention_weight,
    const BFloat16* residual, const BFloat16* pre_feedforward_weight,
    BFloat16* normalized, cudaStream_t stream) {
  post_attention_residual_pre_feedforward_norm_rows(
      branch_state, post_attention_weight, residual, pre_feedforward_weight,
      normalized, 1, stream);
}

void post_attention_residual_pre_feedforward_norm_rows(
    BFloat16* branch_state, const BFloat16* post_attention_weight,
    const BFloat16* residual, const BFloat16* pre_feedforward_weight,
    BFloat16* normalized, std::uint32_t rows, cudaStream_t stream) {
  check_pointer(branch_state,
                "post_attention_residual_pre_feedforward_norm_rows branch");
  check_pointer(post_attention_weight,
                "post_attention_residual_pre_feedforward_norm_rows post weight");
  check_pointer(residual,
                "post_attention_residual_pre_feedforward_norm_rows residual");
  check_pointer(pre_feedforward_weight,
                "post_attention_residual_pre_feedforward_norm_rows pre weight");
  check_pointer(normalized,
                "post_attention_residual_pre_feedforward_norm_rows output");
  check_decode_rows(rows,
                    "post_attention_residual_pre_feedforward_norm_rows");
  if (branch_state == residual || branch_state == normalized ||
      residual == normalized) {
    fail("post_attention_residual_pre_feedforward_norm_rows",
         "branch, residual, and normalized buffers must be distinct");
  }
  post_attention_residual_pre_feedforward_norm_m1_kernel<<<
      rows, kHiddenNormThreads, 0, stream>>>(
      branch_state, post_attention_weight, residual, pre_feedforward_weight,
      normalized);
  check_cuda(cudaGetLastError(),
             "post_attention_residual_pre_feedforward_norm_rows kernel launch");
}

void post_feedforward_residual_scalar_next_norm_m1(
    BFloat16* branch_state, const BFloat16* post_feedforward_weight,
    const BFloat16* residual, const BFloat16* scalar,
    const BFloat16* next_norm_weight, BFloat16* next_normalized,
    cudaStream_t stream) {
  post_feedforward_residual_scalar_next_norm_rows(
      branch_state, post_feedforward_weight, residual, scalar,
      next_norm_weight, next_normalized, 1, stream);
}

void post_feedforward_residual_scalar_next_norm_rows(
    BFloat16* branch_state, const BFloat16* post_feedforward_weight,
    const BFloat16* residual, const BFloat16* scalar,
    const BFloat16* next_norm_weight, BFloat16* next_normalized,
    std::uint32_t rows, cudaStream_t stream) {
  check_pointer(branch_state,
                "post_feedforward_residual_scalar_next_norm_rows branch");
  check_pointer(post_feedforward_weight,
                "post_feedforward_residual_scalar_next_norm_rows post weight");
  check_pointer(residual,
                "post_feedforward_residual_scalar_next_norm_rows residual");
  check_pointer(scalar,
                "post_feedforward_residual_scalar_next_norm_rows scalar");
  check_pointer(next_norm_weight,
                "post_feedforward_residual_scalar_next_norm_rows next weight");
  check_pointer(next_normalized,
                "post_feedforward_residual_scalar_next_norm_rows output");
  check_decode_rows(rows,
                    "post_feedforward_residual_scalar_next_norm_rows");
  if (branch_state == residual || branch_state == next_normalized ||
      residual == next_normalized) {
    fail("post_feedforward_residual_scalar_next_norm_rows",
         "branch, residual, and normalized buffers must be distinct");
  }
  post_feedforward_residual_scalar_next_norm_m1_kernel<<<
      rows, kHiddenNormThreads, 0, stream>>>(
      branch_state, post_feedforward_weight, residual, scalar,
      next_norm_weight, next_normalized);
  check_cuda(cudaGetLastError(),
             "post_feedforward_residual_scalar_next_norm_rows kernel launch");
}

void residual_add(const BFloat16* residual, const BFloat16* branch,
                  BFloat16* output, std::size_t elements,
                  cudaStream_t stream) {
  check_pointer(residual, "residual_add residual");
  check_pointer(branch, "residual_add branch");
  check_pointer(output, "residual_add output");
  if (elements == 0) {
    fail("residual_add", "element count must be positive");
  }
  residual_add_kernel<<<blocks_for(elements), kThreads, 0, stream>>>(
      residual, branch, output, elements);
  check_cuda(cudaGetLastError(), "residual_add kernel launch");
}

void trained_scalar(BFloat16* values, const BFloat16* scalar,
                    std::size_t elements, cudaStream_t stream) {
  check_pointer(values, "trained_scalar values");
  check_pointer(scalar, "trained_scalar scalar");
  if (elements == 0) {
    fail("trained_scalar", "element count must be positive");
  }
  trained_scalar_kernel<<<blocks_for(elements), kThreads, 0, stream>>>(
      values, scalar, elements);
  check_cuda(cudaGetLastError(), "trained_scalar kernel launch");
}

}  // namespace gewell::bf16_primitives
