#pragma once

#include "gewell/compact_global_cache.h"
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
  void qk_compact(const __nv_bfloat16* current_key, const __nv_bfloat16* current_value,
          const compact_global_cache::PagedView<__nv_bfloat16>& cache,
          const __nv_bfloat16* norm, unsigned base_position, unsigned token_count,
          unsigned tile_start, unsigned tile_count, float* scores, cudaStream_t stream);
  void pv(const void* probabilities, float* numerator,
          bool first_tile, cudaStream_t stream);
  const float* query_scales() const;
  const float* key_scales() const;
  const float* value_token_scales() const;
  const float* value_output_scales() const;
  std::size_t scratch_bytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gewell::prefill_primitives
