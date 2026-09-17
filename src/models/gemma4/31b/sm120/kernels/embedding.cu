#include "bf16_common.cuh"
#include <algorithm>
#include <cmath>

namespace gewell::bf16_primitives {
namespace {

using namespace detail;

constexpr float kEmbeddingScaleBf16 = 73.5F;
constexpr unsigned kHostTokenEmbeddingRows = 1024;
struct HostTokenEmbeddingBatch {
  std::uint32_t tokens[kHostTokenEmbeddingRows];
};

__global__ void embedding_lookup_host_tokens_kernel(
    const BFloat16* table, BFloat16* output,
    const __grid_constant__ HostTokenEmbeddingBatch batch) {
  const auto row = blockIdx.y;
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < gemma4_31b::kHiddenSize) {
    output[std::size_t(row) * gemma4_31b::kHiddenSize + index] =
        __float2bfloat16_rn(__bfloat162float(
            table[std::size_t(batch.tokens[row]) * gemma4_31b::kHiddenSize + index]) *
            kEmbeddingScaleBf16);
  }
}

__global__ void embedding_lookup_kernel(const BFloat16* table,
                                        std::uint32_t token_id,
                                        BFloat16* output) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < gemma4_31b::kHiddenSize) {
    const std::size_t table_index =
        static_cast<std::size_t>(token_id) * gemma4_31b::kHiddenSize + index;
    const float value = __bfloat162float(table[table_index]);
    output[index] = __float2bfloat16_rn(value * kEmbeddingScaleBf16);
  }
}

__global__ void embedding_lookup_device_token_kernel(
    const BFloat16* table, const std::uint32_t* token_id,
    BFloat16* output) {
  const std::uint32_t token = token_id[0];
  if (token >= gemma4_31b::kVocabSize) {
    return;
  }
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < gemma4_31b::kHiddenSize) {
    const std::size_t table_index =
        static_cast<std::size_t>(token) * gemma4_31b::kHiddenSize + index;
    const float value = __bfloat162float(table[table_index]);
    output[index] = __float2bfloat16_rn(value * kEmbeddingScaleBf16);
  }
}

constexpr unsigned kDeviceTokenEmbeddingBatchEntries = 128;
struct DeviceTokenEmbeddingBatch {
  DeviceTokenEmbeddingInput inputs[kDeviceTokenEmbeddingBatchEntries];
};

__global__ void embedding_lookup_device_token_batch_kernel(
    const BFloat16* table,
    const __grid_constant__ DeviceTokenEmbeddingBatch batch) {
  const auto& input = batch.inputs[blockIdx.y];
  const std::uint32_t token = input.token_id[0];
  if (token >= gemma4_31b::kVocabSize) return;
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < gemma4_31b::kHiddenSize) {
    const auto table_index =
        static_cast<std::size_t>(token) * gemma4_31b::kHiddenSize + index;
    input.output[index] = __float2bfloat16_rn(
        __bfloat162float(table[table_index]) * kEmbeddingScaleBf16);
  }
}

__global__ void embedding_lookup_device_tokens_kernel(
    const BFloat16* table, const std::uint32_t* token_ids,
    BFloat16* output) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t token = token_ids[row];
  if (token >= gemma4_31b::kVocabSize) {
    return;
  }
  const std::uint32_t index = blockIdx.y * blockDim.x + threadIdx.x;
  if (index < gemma4_31b::kHiddenSize) {
    const std::size_t table_index =
        static_cast<std::size_t>(token) * gemma4_31b::kHiddenSize + index;
    const std::size_t output_index =
        static_cast<std::size_t>(row) * gemma4_31b::kHiddenSize + index;
    output[output_index] = __float2bfloat16_rn(
        __bfloat162float(table[table_index]) * kEmbeddingScaleBf16);
  }
}

}  // namespace

void embedding_lookup_host_tokens(const BFloat16* table,
                                  const std::uint32_t* token_ids,
                                  BFloat16* output, std::uint32_t rows,
                                  cudaStream_t stream) {
  check_pointer(table, "embedding_lookup_host_tokens table");
  check_pointer(token_ids, "embedding_lookup_host_tokens token ids");
  check_pointer(output, "embedding_lookup_host_tokens output");
  check_decode_rows(rows, "embedding_lookup_host_tokens");
  for (std::uint32_t row = 0; row < rows; ++row)
    if (token_ids[row] >= gemma4_31b::kVocabSize)
      fail("embedding_lookup_host_tokens", "token id is outside the Gemma 4 vocabulary");
  HostTokenEmbeddingBatch batch{};
  for (std::uint32_t first = 0; first < rows; first += kHostTokenEmbeddingRows) {
    const auto count = std::min(kHostTokenEmbeddingRows, rows - first);
    std::copy_n(token_ids + first, count, batch.tokens);
    embedding_lookup_host_tokens_kernel<<<
        dim3(blocks_for(gemma4_31b::kHiddenSize), count), kThreads, 0, stream>>>(
            table, output + std::size_t(first) * gemma4_31b::kHiddenSize, batch);
    check_cuda(cudaGetLastError(), "embedding_lookup_host_tokens kernel launch");
  }
}

void embedding_lookup(const BFloat16* table, std::uint32_t token_id,
                      BFloat16* output, cudaStream_t stream) {
  check_pointer(table, "embedding_lookup table");
  check_pointer(output, "embedding_lookup output");
  if (token_id >= gemma4_31b::kVocabSize) {
    fail("embedding_lookup", "token id is outside the Gemma 4 vocabulary");
  }
  embedding_lookup_kernel<<<
      blocks_for(gemma4_31b::kHiddenSize), kThreads, 0, stream>>>(
      table, token_id, output);
  check_cuda(cudaGetLastError(), "embedding_lookup kernel launch");
}

void embedding_lookup_device_token(const BFloat16* table,
                                   const std::uint32_t* token_id,
                                   BFloat16* output, cudaStream_t stream) {
  check_pointer(table, "embedding_lookup_device_token table");
  check_pointer(token_id, "embedding_lookup_device_token token id");
  check_pointer(output, "embedding_lookup_device_token output");
  embedding_lookup_device_token_kernel<<<
      blocks_for(gemma4_31b::kHiddenSize), kThreads, 0, stream>>>(
      table, token_id, output);
  check_cuda(cudaGetLastError(),
             "embedding_lookup_device_token kernel launch");
}

void embedding_lookup_device_token_batch(
    const BFloat16* table,
    const std::vector<DeviceTokenEmbeddingInput>& inputs,
    cudaStream_t stream) {
  check_pointer(table, "embedding_lookup_device_token_batch table");
  if (inputs.empty()) fail("embedding_lookup_device_token_batch", "empty batch");
  for (const auto& input : inputs) {
    check_pointer(input.token_id,
                  "embedding_lookup_device_token_batch token id");
    check_pointer(input.output,
                  "embedding_lookup_device_token_batch output");
  }
  DeviceTokenEmbeddingBatch batch{};
  for (std::size_t first = 0; first < inputs.size();
       first += kDeviceTokenEmbeddingBatchEntries) {
    const auto count = static_cast<unsigned>(std::min<std::size_t>(
        kDeviceTokenEmbeddingBatchEntries, inputs.size() - first));
    std::copy_n(inputs.begin() + first, count, batch.inputs);
    embedding_lookup_device_token_batch_kernel<<<
        dim3(blocks_for(gemma4_31b::kHiddenSize), count), kThreads, 0,
        stream>>>(table, batch);
    check_cuda(cudaGetLastError(),
               "embedding_lookup_device_token_batch kernel launch");
  }
}

void embedding_lookup_device_tokens(const BFloat16* table,
                                    const std::uint32_t* token_ids,
                                    BFloat16* output, std::uint32_t rows,
                                    cudaStream_t stream) {
  check_pointer(table, "embedding_lookup_device_tokens table");
  check_pointer(token_ids, "embedding_lookup_device_tokens token ids");
  check_pointer(output, "embedding_lookup_device_tokens output");
  check_decode_rows(rows, "embedding_lookup_device_tokens");
  const dim3 grid{rows, blocks_for(gemma4_31b::kHiddenSize)};
  embedding_lookup_device_tokens_kernel<<<grid, kThreads, 0, stream>>>(
      table, token_ids, output);
  check_cuda(cudaGetLastError(),
             "embedding_lookup_device_tokens kernel launch");
}

}  // namespace gewell::bf16_primitives
