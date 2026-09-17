#include "bf16_common.cuh"
#include <cub/cub.cuh>
#include <algorithm>
#include <cmath>

namespace gewell::bf16_primitives {
namespace {

using namespace detail;

constexpr std::size_t kSamplingAlignment = 256;

std::size_t align_up(std::size_t value, std::size_t alignment) {
  if (value > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
    fail("sampling scratch", "size overflows size_t");
  }
  return (value + alignment - 1) & ~(alignment - 1);
}

struct SamplingScratchPlan {
  std::size_t scores_in{};
  std::size_t scores_out{};
  std::size_t tokens_in{};
  std::size_t tokens_out{};
  std::size_t weights{};
  std::size_t cumulative{};
  std::size_t temporary{};
  std::size_t bytes{};
};

SamplingScratchPlan make_sampling_scratch_plan(std::uint32_t elements) {
  if (elements == 0 || elements > static_cast<std::uint32_t>(INT_MAX)) {
    fail("sampling scratch", "element count must fit a positive int");
  }
  const std::size_t count = elements;
  if (count > std::numeric_limits<std::size_t>::max() /
                  (4 * sizeof(float) + 2 * sizeof(std::uint32_t))) {
    fail("sampling scratch", "array size overflows size_t");
  }

  SamplingScratchPlan plan;
  std::size_t offset = 0;
  const auto reserve = [&offset](std::size_t bytes) {
    const std::size_t begin = align_up(offset, kSamplingAlignment);
    if (bytes > std::numeric_limits<std::size_t>::max() - begin) {
      fail("sampling scratch", "layout size overflows size_t");
    }
    offset = begin + bytes;
    return begin;
  };
  plan.scores_in = reserve(count * sizeof(float));
  plan.scores_out = reserve(count * sizeof(float));
  plan.tokens_in = reserve(count * sizeof(std::uint32_t));
  plan.tokens_out = reserve(count * sizeof(std::uint32_t));
  plan.weights = reserve(count * sizeof(float));
  plan.cumulative = reserve(count * sizeof(float));
  plan.temporary = align_up(offset, kSamplingAlignment);
  plan.bytes = plan.temporary;
  return plan;
}

std::size_t sampling_temporary_bytes(std::uint32_t elements) {
  std::size_t sort_bytes = 0;
  check_cuda(cub::DeviceRadixSort::SortPairsDescending(
                 nullptr, sort_bytes, static_cast<const float*>(nullptr),
                 static_cast<float*>(nullptr),
                 static_cast<const std::uint32_t*>(nullptr),
                 static_cast<std::uint32_t*>(nullptr), elements),
             "query sampling radix-sort scratch");
  std::size_t scan_bytes = 0;
  check_cuda(cub::DeviceScan::InclusiveSum(
                 nullptr, scan_bytes, static_cast<const float*>(nullptr),
                 static_cast<float*>(nullptr), elements),
             "query sampling scan scratch");
  return std::max(sort_bytes, scan_bytes);
}

__global__ void softcap_kernel(const BFloat16* logits, BFloat16* capped,
                               std::uint32_t elements, float cap) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    const BFloat16 divided =
        __float2bfloat16_rn(__bfloat162float(logits[index]) / cap);
    const BFloat16 squashed =
        __float2bfloat16_rn(tanhf(__bfloat162float(divided)));
    capped[index] =
        __float2bfloat16_rn(__bfloat162float(squashed) * cap);
  }
}

__device__ __forceinline__ void deterministic_argmax(const BFloat16* values,
                                            std::uint32_t elements,
                                            std::uint32_t* result) {
  __shared__ float best_values[kThreads];
  __shared__ std::uint32_t best_indices[kThreads];

  float best_value = -__int_as_float(0x7f800000);
  std::uint32_t best_index = 0xffffffffU;
  for (std::uint32_t index = threadIdx.x; index < elements;
       index += blockDim.x) {
    const float candidate = __bfloat162float(values[index]);
    if (candidate > best_value ||
        (candidate == best_value && index < best_index)) {
      best_value = candidate;
      best_index = index;
    }
  }
  best_values[threadIdx.x] = best_value;
  best_indices[threadIdx.x] = best_index;
  __syncthreads();

  for (unsigned offset = kThreads / 2; offset != 0; offset /= 2) {
    if (threadIdx.x < offset) {
      const float candidate_value = best_values[threadIdx.x + offset];
      const std::uint32_t candidate_index =
          best_indices[threadIdx.x + offset];
      if (candidate_value > best_values[threadIdx.x] ||
          (candidate_value == best_values[threadIdx.x] &&
           candidate_index < best_indices[threadIdx.x])) {
        best_values[threadIdx.x] = candidate_value;
        best_indices[threadIdx.x] = candidate_index;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    result[0] = best_indices[0];
  }
}

__global__ void deterministic_argmax_kernel(const BFloat16* values,
    std::uint32_t elements, std::uint32_t* result) {
  deterministic_argmax(values, elements, result);
}

__global__ void deterministic_argmax_rows_kernel(const BFloat16* values,
    std::uint32_t elements, std::uint32_t* result) {
  deterministic_argmax(values + std::size_t(blockIdx.x) * elements,
                        elements, result + blockIdx.x);
}

__global__ void make_sampling_inputs_kernel(const BFloat16* logits,
                                            std::uint32_t elements,
                                            float temperature, float* scores,
                                            std::uint32_t* tokens) {
  const std::uint32_t token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= elements) {
    return;
  }
  float score = __bfloat162float(logits[token]) / temperature;
  if (score == 0.0F) {
    score = 0.0F;
  }
  scores[token] = score;
  tokens[token] = token;
}

__global__ void make_sampling_weights_kernel(
    const float* sorted_scores, std::uint32_t retained, float* weights) {
  const std::uint32_t rank = blockIdx.x * blockDim.x + threadIdx.x;
  if (rank >= retained) {
    return;
  }
  weights[rank] = expf(sorted_scores[rank] - sorted_scores[0]);
}

__global__ void select_sampling_token_kernel(
    const std::uint32_t* sorted_tokens, const float* cumulative,
    std::uint32_t retained, float top_p, float uniform,
    std::uint32_t* selected) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const float total = cumulative[retained - 1];
  const float nucleus_target = top_p * total;
  std::uint32_t low = 0;
  std::uint32_t high = retained;
  while (low < high) {
    const std::uint32_t middle = low + (high - low) / 2;
    if (cumulative[middle] >= nucleus_target) {
      high = middle;
    } else {
      low = middle + 1;
    }
  }
  const std::uint32_t cutoff = low < retained ? low : retained - 1;
  const float sample_target = uniform * cumulative[cutoff];
  low = 0;
  high = cutoff + 1;
  while (low < high) {
    const std::uint32_t middle = low + (high - low) / 2;
    if (cumulative[middle] > sample_target) {
      high = middle;
    } else {
      low = middle + 1;
    }
  }
  const std::uint32_t rank = low <= cutoff ? low : cutoff;
  selected[0] = sorted_tokens[rank];
}

__global__ void seed_graph_decode_state_kernel(
    const std::uint32_t* first_token, std::uint32_t* current_token,
    std::uint32_t* absolute_position, std::uint32_t* outputs) {
  const std::uint32_t token = first_token[0];
  current_token[0] = token;
  absolute_position[0] = kGraphDecodeFirstPosition;
  outputs[0] = token;
}

__global__ void commit_graph_decode_state_kernel(
    const std::uint32_t* next_token, std::uint32_t* current_token,
    std::uint32_t* absolute_position, std::uint32_t* outputs) {
  const std::uint32_t position = absolute_position[0];
  if (position < kGraphDecodeFirstPosition ||
      position > kGraphDecodeLastPosition) {
    return;
  }
  const std::uint32_t token = next_token[0];
  outputs[position - (kGraphDecodeFirstPosition - 1)] = token;
  current_token[0] = token;
  absolute_position[0] = position + 1;
}

}  // namespace

void seed_graph_decode_state(const std::uint32_t* first_token,
                             std::uint32_t* current_token,
                             std::uint32_t* absolute_position,
                             std::uint32_t* outputs,
                             cudaStream_t stream) {
  check_pointer(first_token, "seed_graph_decode_state first token");
  check_pointer(current_token, "seed_graph_decode_state current token");
  check_pointer(absolute_position, "seed_graph_decode_state position");
  check_pointer(outputs, "seed_graph_decode_state outputs");
  seed_graph_decode_state_kernel<<<1, 1, 0, stream>>>(
      first_token, current_token, absolute_position, outputs);
  check_cuda(cudaGetLastError(), "seed_graph_decode_state kernel launch");
}

void commit_graph_decode_state(const std::uint32_t* next_token,
                               std::uint32_t* current_token,
                               std::uint32_t* absolute_position,
                               std::uint32_t* outputs,
                               cudaStream_t stream) {
  check_pointer(next_token, "commit_graph_decode_state next token");
  check_pointer(current_token, "commit_graph_decode_state current token");
  check_pointer(absolute_position, "commit_graph_decode_state position");
  check_pointer(outputs, "commit_graph_decode_state outputs");
  commit_graph_decode_state_kernel<<<1, 1, 0, stream>>>(
      next_token, current_token, absolute_position, outputs);
  check_cuda(cudaGetLastError(), "commit_graph_decode_state kernel launch");
}

void softcap_logits(const BFloat16* logits, BFloat16* capped,
                    std::uint32_t elements, float cap, cudaStream_t stream) {
  check_pointer(logits, "softcap_and_argmax logits");
  check_pointer(capped, "softcap_and_argmax capped");
  if (elements == 0) {
    fail("softcap_and_argmax", "element count must be positive");
  }
  if (!(cap > 0.0F) || !std::isfinite(cap)) {
    fail("softcap_and_argmax", "cap must be finite and positive");
  }
  softcap_kernel<<<blocks_for(elements), kThreads, 0, stream>>>(
      logits, capped, elements, cap);
  check_cuda(cudaGetLastError(), "softcap kernel launch");
}

void softcap_and_argmax(const BFloat16* logits, BFloat16* capped,
                        std::uint32_t* argmax, std::uint32_t elements,
                        float cap, cudaStream_t stream) {
  check_pointer(argmax, "softcap_and_argmax argmax");
  softcap_logits(logits, capped, elements, cap, stream);
  deterministic_argmax_kernel<<<1, kThreads, 0, stream>>>(capped, elements,
                                                          argmax);
  check_cuda(cudaGetLastError(), "deterministic_argmax kernel launch");
}

void softcap_and_argmax_rows(const BFloat16* logits, BFloat16* capped,
    std::uint32_t* argmax, std::uint32_t rows, std::uint32_t elements,
    float cap, cudaStream_t stream) {
  check_decode_rows(rows, "softcap_and_argmax_rows");
  if (!elements || rows > UINT_MAX / elements)
    fail("softcap_and_argmax_rows", "logit count exceeds capacity");
  check_pointer(argmax, "softcap_and_argmax_rows argmax");
  softcap_logits(logits, capped, rows * elements, cap, stream);
  deterministic_argmax_rows_kernel<<<rows, kThreads, 0, stream>>>(capped, elements, argmax);
  check_cuda(cudaGetLastError(), "batched deterministic_argmax kernel launch");
}

std::size_t sampling_scratch_bytes(std::uint32_t elements) {
  const SamplingScratchPlan plan = make_sampling_scratch_plan(elements);
  const std::size_t temporary_bytes = sampling_temporary_bytes(elements);
  if (temporary_bytes >
      std::numeric_limits<std::size_t>::max() - plan.temporary) {
    fail("sampling scratch", "temporary storage size overflows size_t");
  }
  return plan.temporary + temporary_bytes;
}

void sample_top_k_top_p(const BFloat16* logits, std::uint32_t elements,
                        float temperature, float top_p, std::uint32_t top_k,
                        float uniform, void* scratch,
                        std::size_t scratch_bytes, std::uint32_t* selected,
                        cudaStream_t stream) {
  check_pointer(logits, "sample_top_k_top_p logits");
  check_pointer(scratch, "sample_top_k_top_p scratch");
  check_pointer(selected, "sample_top_k_top_p selected token");
  if (!(temperature > 0.0F) || !std::isfinite(temperature)) {
    fail("sample_top_k_top_p", "temperature must be finite and positive");
  }
  if (top_p < 0.0F || top_p > 1.0F || !std::isfinite(top_p)) {
    fail("sample_top_k_top_p", "top_p must be finite and in [0, 1]");
  }
  if (uniform < 0.0 || uniform >= 1.0 || !std::isfinite(uniform)) {
    fail("sample_top_k_top_p", "uniform draw must be finite and in [0, 1)");
  }
  const SamplingScratchPlan plan = make_sampling_scratch_plan(elements);
  if (scratch_bytes <= plan.bytes) {
    fail("sample_top_k_top_p", "scratch allocation is too small");
  }
  auto* const bytes = static_cast<std::uint8_t*>(scratch);
  auto* const scores_in = reinterpret_cast<float*>(bytes + plan.scores_in);
  auto* const scores_out = reinterpret_cast<float*>(bytes + plan.scores_out);
  auto* const tokens_in =
      reinterpret_cast<std::uint32_t*>(bytes + plan.tokens_in);
  auto* const tokens_out =
      reinterpret_cast<std::uint32_t*>(bytes + plan.tokens_out);
  auto* const weights = reinterpret_cast<float*>(bytes + plan.weights);
  auto* const cumulative = reinterpret_cast<float*>(bytes + plan.cumulative);
  void* const temporary = bytes + plan.temporary;
  const std::uint32_t retained =
      top_k == 0 ? elements : std::min(top_k, elements);
  const std::size_t available_temporary_bytes =
      scratch_bytes - plan.temporary;

  make_sampling_inputs_kernel<<<blocks_for(elements), kThreads, 0, stream>>>(
      logits, elements, temperature, scores_in, tokens_in);
  check_cuda(cudaGetLastError(), "sampling input kernel launch");
  std::size_t temporary_bytes = available_temporary_bytes;
  check_cuda(cub::DeviceRadixSort::SortPairsDescending(
                 temporary, temporary_bytes, scores_in, scores_out, tokens_in,
                 tokens_out, elements, 0, 32, stream),
             "sampling radix sort");
  make_sampling_weights_kernel<<<blocks_for(retained), kThreads, 0, stream>>>(
      scores_out, retained, weights);
  check_cuda(cudaGetLastError(), "sampling weight kernel launch");
  temporary_bytes = available_temporary_bytes;
  check_cuda(cub::DeviceScan::InclusiveSum(
                 temporary, temporary_bytes, weights, cumulative, retained,
                 stream),
             "sampling probability scan");
  select_sampling_token_kernel<<<1, 1, 0, stream>>>(
      tokens_out, cumulative, retained, top_p, uniform, selected);
  check_cuda(cudaGetLastError(), "sampling selection kernel launch");
}

}  // namespace gewell::bf16_primitives
