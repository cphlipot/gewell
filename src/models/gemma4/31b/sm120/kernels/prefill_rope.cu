#include "prefill_common.cuh"
#include "rope_inverse_frequency.cuh"

namespace gewell::prefill_primitives {
namespace {

using namespace detail;

template <std::uint32_t HeadSize, std::uint32_t RotatedFrequencies,
          std::uint32_t Theta>
__global__ void generate_rope_factors_kernel(BFloat16* cosine,
                                             BFloat16* sine) {
  static_assert(HeadSize % 2 == 0);
  static_assert(RotatedFrequencies <= HeadSize / 2);
  static_assert((HeadSize == gemma4_31b::kLocalHeadSize &&
                 RotatedFrequencies == gemma4_31b::kLocalHeadSize / 2 &&
                 Theta == 10'000) ||
                (HeadSize == gemma4_31b::kGlobalHeadSize &&
                 RotatedFrequencies == 64 && Theta == 1'000'000));
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  constexpr std::size_t kElements =
      static_cast<std::size_t>(kTokenCount) * HeadSize;
  if (index >= kElements) {
    return;
  }

  const std::uint32_t position =
      static_cast<std::uint32_t>(index / HeadSize);
  const std::uint32_t dimension =
      static_cast<std::uint32_t>(index % HeadSize);
  const std::uint32_t frequency = dimension % (HeadSize / 2);
  if (position == 0 || frequency >= RotatedFrequencies) {
    cosine[index] = __float2bfloat16_rn(1.0F);
    sine[index] = __float2bfloat16_rn(0.0F);
    return;
  }

  const float inverse_frequency =
      HeadSize == gemma4_31b::kGlobalHeadSize
          ? rope_inverse_frequency::get<true>(frequency)
          : rope_inverse_frequency::get<false>(frequency);
  const float angle = static_cast<float>(position) * inverse_frequency;
  cosine[index] = __float2bfloat16_rn(cosf(angle));
  sine[index] = __float2bfloat16_rn(sinf(angle));
}

template <std::uint32_t HeadSize, std::uint32_t RotatedFrequencies>
__global__ void generate_rope_factors_chunk_kernel(
    BFloat16* cosine, BFloat16* sine, std::uint32_t base_position,
    std::uint32_t token_count, std::size_t elements) {
  static_assert(HeadSize % 2 == 0);
  static_assert(RotatedFrequencies <= HeadSize / 2);
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }

  const std::uint32_t chunk_position =
      static_cast<std::uint32_t>(index / HeadSize);
  const std::uint32_t position = base_position + chunk_position;
  const std::uint32_t dimension =
      static_cast<std::uint32_t>(index % HeadSize);
  const std::uint32_t frequency = dimension % (HeadSize / 2);
  if (position == 0 || frequency >= RotatedFrequencies) {
    cosine[index] = __float2bfloat16_rn(1.0F);
    sine[index] = __float2bfloat16_rn(0.0F);
    return;
  }

  const float inverse_frequency =
      HeadSize == gemma4_31b::kGlobalHeadSize
          ? rope_inverse_frequency::get<true>(frequency)
          : rope_inverse_frequency::get<false>(frequency);
  const float angle = static_cast<float>(position) * inverse_frequency;
  cosine[index] = __float2bfloat16_rn(cosf(angle));
  sine[index] = __float2bfloat16_rn(sinf(angle));
}

template <std::uint32_t HeadSize>
__global__ void apply_rope_transpose_kernel(
    const BFloat16* input_token_major, const BFloat16* cosine,
    const BFloat16* sine, BFloat16* output_head_major, std::uint32_t heads,
    std::size_t elements) {
  static_assert(HeadSize % 2 == 0);
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }

  const std::uint32_t dimension =
      static_cast<std::uint32_t>(index % HeadSize);
  const std::size_t row = index / HeadSize;
  const std::uint32_t head = static_cast<std::uint32_t>(row % heads);
  const std::uint32_t position = static_cast<std::uint32_t>(row / heads);
  const std::uint32_t paired_dimension =
      dimension < HeadSize / 2 ? dimension + HeadSize / 2
                              : dimension - HeadSize / 2;
  const std::size_t input_row =
      (static_cast<std::size_t>(position) * heads + head) * HeadSize;
  const float value =
      __bfloat162float(input_token_major[input_row + dimension]);
  float rotated =
      __bfloat162float(input_token_major[input_row + paired_dimension]);
  if (dimension < HeadSize / 2) {
    rotated = -rotated;
  }

  const std::size_t factor_index =
      static_cast<std::size_t>(position) * HeadSize + dimension;
  const BFloat16 direct_product = __float2bfloat16_rn(
      value * __bfloat162float(cosine[factor_index]));
  const BFloat16 rotated_product = __float2bfloat16_rn(
      rotated * __bfloat162float(sine[factor_index]));
  const BFloat16 result = __float2bfloat16_rn(
      __bfloat162float(direct_product) +
      __bfloat162float(rotated_product));
  const std::size_t output_index =
      (static_cast<std::size_t>(head) * kTokenCount + position) * HeadSize +
      dimension;
  output_head_major[output_index] = result;
}

template <std::uint32_t HeadSize>
__global__ void apply_rope_transpose_chunk_kernel(
    const BFloat16* input_token_major, const BFloat16* cosine,
    const BFloat16* sine, BFloat16* output_head_major, std::uint32_t heads,
    std::uint32_t token_count, std::size_t elements) {
  static_assert(HeadSize % 2 == 0);
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }

  const std::uint32_t dimension =
      static_cast<std::uint32_t>(index % HeadSize);
  const std::size_t row = index / HeadSize;
  const std::uint32_t head = static_cast<std::uint32_t>(row % heads);
  const std::uint32_t position = static_cast<std::uint32_t>(row / heads);
  const std::uint32_t paired_dimension =
      dimension < HeadSize / 2 ? dimension + HeadSize / 2
                              : dimension - HeadSize / 2;
  const std::size_t input_row =
      (static_cast<std::size_t>(position) * heads + head) * HeadSize;
  const float value =
      __bfloat162float(input_token_major[input_row + dimension]);
  float rotated =
      __bfloat162float(input_token_major[input_row + paired_dimension]);
  if (dimension < HeadSize / 2) {
    rotated = -rotated;
  }

  const std::size_t factor_index =
      static_cast<std::size_t>(position) * HeadSize + dimension;
  const BFloat16 direct_product = __float2bfloat16_rn(
      value * __bfloat162float(cosine[factor_index]));
  const BFloat16 rotated_product = __float2bfloat16_rn(
      rotated * __bfloat162float(sine[factor_index]));
  const BFloat16 result = __float2bfloat16_rn(
      __bfloat162float(direct_product) +
      __bfloat162float(rotated_product));
  const std::size_t output_index =
      (static_cast<std::size_t>(head) * token_count + position) * HeadSize +
      dimension;
  output_head_major[output_index] = result;
}

}  // namespace

void generate_rope_factors_m1024(BFloat16* local_cos, BFloat16* local_sin,
                                 BFloat16* global_cos, BFloat16* global_sin,
                                 cudaStream_t stream) {
  check_pointer(local_cos, "generate_rope_factors_m1024 local cosine");
  check_pointer(local_sin, "generate_rope_factors_m1024 local sine");
  check_pointer(global_cos, "generate_rope_factors_m1024 global cosine");
  check_pointer(global_sin, "generate_rope_factors_m1024 global sine");

  constexpr std::size_t kLocalElements =
      static_cast<std::size_t>(kTokenCount) * gemma4_31b::kLocalHeadSize;
  constexpr std::size_t kGlobalElements =
      static_cast<std::size_t>(kTokenCount) * gemma4_31b::kGlobalHeadSize;
  generate_rope_factors_kernel<gemma4_31b::kLocalHeadSize,
                               gemma4_31b::kLocalHeadSize / 2, 10'000>
      <<<blocks_for(kLocalElements), kThreads, 0, stream>>>(local_cos,
                                                            local_sin);
  check_cuda(cudaGetLastError(), "generate local M=1024 RoPE kernel launch");
  generate_rope_factors_kernel<gemma4_31b::kGlobalHeadSize, 64, 1'000'000>
      <<<blocks_for(kGlobalElements), kThreads, 0, stream>>>(global_cos,
                                                             global_sin);
  check_cuda(cudaGetLastError(), "generate global M=1024 RoPE kernel launch");
}

void apply_rope_transpose_m1024(
    const BFloat16* input_token_major, const BFloat16* cosine,
    const BFloat16* sine, BFloat16* output_head_major, std::uint32_t heads,
    gemma4_31b::AttentionKind kind, cudaStream_t stream) {
  check_pointer(input_token_major,
                "apply_rope_transpose_m1024 input token major");
  check_pointer(cosine, "apply_rope_transpose_m1024 cosine");
  check_pointer(sine, "apply_rope_transpose_m1024 sine");
  check_pointer(output_head_major,
                "apply_rope_transpose_m1024 output head major");
  check_kind(kind, "apply_rope_transpose_m1024");

  const bool global = kind == gemma4_31b::AttentionKind::global;
  const std::uint32_t kv_heads = global ? gemma4_31b::kGlobalKvHeadCount
                                        : gemma4_31b::kLocalKvHeadCount;
  if (heads != gemma4_31b::kQueryHeadCount && heads != kv_heads) {
    fail("apply_rope_transpose_m1024",
         "head count is neither Q nor the selected kind's KV count");
  }
  const std::uint32_t head_size = global ? gemma4_31b::kGlobalHeadSize
                                         : gemma4_31b::kLocalHeadSize;
  const std::size_t elements =
      static_cast<std::size_t>(kTokenCount) * heads * head_size;
  if (global) {
    apply_rope_transpose_kernel<gemma4_31b::kGlobalHeadSize>
        <<<blocks_for(elements), kThreads, 0, stream>>>(
            input_token_major, cosine, sine, output_head_major, heads,
            elements);
  } else {
    apply_rope_transpose_kernel<gemma4_31b::kLocalHeadSize>
        <<<blocks_for(elements), kThreads, 0, stream>>>(
            input_token_major, cosine, sine, output_head_major, heads,
            elements);
  }
  check_cuda(cudaGetLastError(),
             "apply M=1024 RoPE transpose kernel launch");
}

void generate_rope_factors_chunk(
    BFloat16* local_cos, BFloat16* local_sin, BFloat16* global_cos,
    BFloat16* global_sin, std::uint32_t base_position,
    std::uint32_t token_count, cudaStream_t stream) {
  check_pointer(local_cos, "generate_rope_factors_chunk local cosine");
  check_pointer(local_sin, "generate_rope_factors_chunk local sine");
  check_pointer(global_cos, "generate_rope_factors_chunk global cosine");
  check_pointer(global_sin, "generate_rope_factors_chunk global sine");
  check_chunk_range(base_position, token_count,
                    "generate_rope_factors_chunk");

  const std::size_t local_elements =
      static_cast<std::size_t>(token_count) *
      gemma4_31b::kLocalHeadSize;
  const std::size_t global_elements =
      static_cast<std::size_t>(token_count) *
      gemma4_31b::kGlobalHeadSize;
  generate_rope_factors_chunk_kernel<gemma4_31b::kLocalHeadSize,
                                     gemma4_31b::kLocalHeadSize / 2>
      <<<blocks_for(local_elements), kThreads, 0, stream>>>(
          local_cos, local_sin, base_position, token_count, local_elements);
  check_cuda(cudaGetLastError(), "generate local chunk RoPE kernel launch");
  generate_rope_factors_chunk_kernel<gemma4_31b::kGlobalHeadSize, 64>
      <<<blocks_for(global_elements), kThreads, 0, stream>>>(
          global_cos, global_sin, base_position, token_count, global_elements);
  check_cuda(cudaGetLastError(), "generate global chunk RoPE kernel launch");
}

void apply_rope_transpose_chunk(
    const BFloat16* input_token_major, const BFloat16* cosine,
    const BFloat16* sine, BFloat16* output_head_major, std::uint32_t heads,
    std::uint32_t token_count, gemma4_31b::AttentionKind kind,
    cudaStream_t stream) {
  check_pointer(input_token_major,
                "apply_rope_transpose_chunk input token major");
  check_pointer(cosine, "apply_rope_transpose_chunk cosine");
  check_pointer(sine, "apply_rope_transpose_chunk sine");
  check_pointer(output_head_major,
                "apply_rope_transpose_chunk output head major");
  check_kind(kind, "apply_rope_transpose_chunk");
  check_chunk_range(0, token_count, "apply_rope_transpose_chunk");

  const bool global = kind == gemma4_31b::AttentionKind::global;
  const std::uint32_t kv_heads = global ? gemma4_31b::kGlobalKvHeadCount
                                        : gemma4_31b::kLocalKvHeadCount;
  if (heads != gemma4_31b::kQueryHeadCount && heads != kv_heads) {
    fail("apply_rope_transpose_chunk",
         "head count is neither Q nor the selected kind's KV count");
  }
  const std::uint32_t head_size = global ? gemma4_31b::kGlobalHeadSize
                                         : gemma4_31b::kLocalHeadSize;
  const std::size_t elements =
      static_cast<std::size_t>(token_count) * heads * head_size;
  if (global) {
    apply_rope_transpose_chunk_kernel<gemma4_31b::kGlobalHeadSize>
        <<<blocks_for(elements), kThreads, 0, stream>>>(
            input_token_major, cosine, sine, output_head_major, heads,
            token_count, elements);
  } else {
    apply_rope_transpose_chunk_kernel<gemma4_31b::kLocalHeadSize>
        <<<blocks_for(elements), kThreads, 0, stream>>>(
            input_token_major, cosine, sine, output_head_major, heads,
            token_count, elements);
  }
  check_cuda(cudaGetLastError(), "apply chunk RoPE transpose kernel launch");
}

}  // namespace gewell::prefill_primitives
