#include "bf16_common.cuh"
#include "bf16_activation.cuh"
#include <algorithm>
#include <cmath>

namespace gewell::bf16_primitives {
namespace {

using namespace detail;

template<bool Interleaved>
__global__ void gelu_tanh_multiply_kernel(const BFloat16* gate,
                                          const BFloat16* up,
                                          BFloat16* output,
                                          std::size_t elements,
                                          std::uint32_t width) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const auto source = Interleaved ? (index / width) * (2 * width) + index % width : index;
    const float value = __bfloat162float(gate[source]);
    output[index] = __float2bfloat16_rn(gewell::detail::gelu_tanh_multiply_bf16(
        value, __bfloat162float(up[source])));
  }
}

}  // namespace

void gelu_tanh_multiply(const BFloat16* gate, const BFloat16* up,
                        BFloat16* output, std::size_t elements,
                        cudaStream_t stream) {
  check_pointer(gate, "gelu_tanh_multiply gate");
  check_pointer(up, "gelu_tanh_multiply up");
  check_pointer(output, "gelu_tanh_multiply output");
  if (elements == 0) {
    fail("gelu_tanh_multiply", "element count must be positive");
  }
  gelu_tanh_multiply_kernel<false><<<blocks_for(elements), kThreads, 0, stream>>>(
      gate, up, output, elements, 0);
  check_cuda(cudaGetLastError(), "gelu_tanh_multiply kernel launch");
}

void gelu_tanh_multiply_interleaved(const BFloat16* gate_up, BFloat16* output,
                                   std::uint32_t rows, std::uint32_t width,
                                   cudaStream_t stream) {
  check_pointer(gate_up, "gelu_tanh_multiply_interleaved input");
  check_pointer(output, "gelu_tanh_multiply_interleaved output");
  if (!rows || !width || width > UINT_MAX / 2)
    fail("gelu_tanh_multiply_interleaved", "invalid shape");
  const auto elements = std::size_t(rows) * width;
  gelu_tanh_multiply_kernel<true><<<blocks_for(elements), kThreads, 0, stream>>>(
      gate_up, gate_up + width, output, elements, width);
  check_cuda(cudaGetLastError(), "gelu_tanh_multiply_interleaved kernel launch");
}

}  // namespace gewell::bf16_primitives
