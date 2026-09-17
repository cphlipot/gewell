#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <memory>

namespace gewell::prefill_primitives {

// Native E4M3 prefill matmuls. One context per executor, reused serially on
// its stream. Owns bounded packing/workspace buffers and caches GEMM plans
// by query shape. Plan preparation must precede CUDA graph capture.
class Fp8Attention {
 public:
  explicit Fp8Attention(unsigned max_rows);
  ~Fp8Attention();
  Fp8Attention(const Fp8Attention&) = delete;
  Fp8Attention& operator=(const Fp8Attention&) = delete;

  void prepare(const __nv_bfloat16* query, unsigned rows, unsigned head_size,
               unsigned kv_heads, cudaStream_t stream);
  void qk(const __nv_bfloat16* key, const __nv_bfloat16* value,
          unsigned tile_count, float* scores, cudaStream_t stream);
  void pv(const void* probabilities, float* numerator,
          bool first_tile, cudaStream_t stream);
  const float* query_scales() const;
  const float* key_scales() const;
  std::size_t scratch_bytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gewell::prefill_primitives
