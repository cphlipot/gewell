#pragma once

#include "gewell/mtp_target.h"

namespace gewell::mtp_attention {

inline constexpr std::uint32_t kQueryTileRows = 8;
inline constexpr std::uint32_t kKeysPerTile = 32;
inline constexpr std::uint32_t kMaxSplits = 256;

// Non-KV scratch is bounded independently of the draft depth and context.
// Consecutive groups of at most eight query rows reuse the same allocation.
std::size_t scratch_bytes(std::uint32_t capacity_rows,
                          std::uint32_t context_capacity = 262144);

// Read-only attention over committed KV followed by the causal staged prefix.
// Query and staged K are [heads, rows, D]; staged V and output are token-major.
// Global compact K reconstruction rounds V * k_norm to BF16 before the dot
// product, including staged keys. Scores and all partial reductions use FP32;
// P*V probabilities round once to BF16.
void run(const mtp_target::BFloat16* query,
         const mtp_target::BFloat16* staged_key,
         const mtp_target::BFloat16* staged_value,
         const mtp_target::CacheView& cache,
         const mtp_target::BFloat16* global_k_norm,
         std::uint32_t base_position, std::uint32_t rows,
         gemma4_31b::AttentionKind kind, mtp_target::BFloat16* context,
         void* scratch, std::size_t scratch_size, cudaStream_t stream);

struct BatchInput {
  const mtp_target::BFloat16* query{};
  const mtp_target::BFloat16* staged_key{};
  const mtp_target::BFloat16* staged_value{};
  mtp_target::CacheView cache{};
  std::uint32_t base_position{}, rows{};
  mtp_target::BFloat16* context{};
};

// Native E4M3 QK and PV with FP32 softmax/accumulation. Q/K/V scales are
// per vector; V scales are folded into P before per-query/tile P quantization.
// Uses the same bounded scratch as BF16 and never modifies cached/staged KV.
// frozen=true accepts one query per input over [0,base_position), for ordinary
// decode after commit and assistant attention; otherwise current rows are causal.
void run_fp8_batch(const std::vector<BatchInput>& inputs,
                   const mtp_target::BFloat16* global_k_norm,
                   gemma4_31b::AttentionKind kind, void* scratch,
                   std::size_t scratch_size, cudaStream_t stream,
                   bool frozen = false);

// Coalesce independent request tiles into common launches, with disjoint
// partial results in the caller's existing scratch. Splits balance visible KV
// work within each launch, subject to a block target and scratch capacity.
// Scores, softmax metadata and reductions stay FP32; P*V probabilities round
// once to BF16. Changing batch geometry can change reduction order.
// Single-request calls retain the serial split count.
// Large batches use bounded groups; no allocation, upload or synchronization.
void run_batch(const std::vector<BatchInput>& inputs,
               const mtp_target::BFloat16* global_k_norm,
               gemma4_31b::AttentionKind kind, void* scratch,
               std::size_t scratch_size, cudaStream_t stream);

// Coalesce independent assistant queries over committed global KV. Each input
// has rows=1, base_position=processed_tokens, and null staged K/V. Cache and
// probability arithmetic match run_frozen_global(); batching only repartitions
// the FP32 split reduction to expose enough concurrent work.
void run_frozen_global_batch(const std::vector<BatchInput>& inputs,
                             const mtp_target::BFloat16* global_k_norm,
                             void* scratch, std::size_t scratch_size,
                             cudaStream_t stream);

// Assistant global attention uses the same key reconstruction and arithmetic,
// over committed [0,processed_tokens) only. Query/output are [32,512]; scratch
// is scratch_bytes(1, processed_tokens). No cache or staged KV is written.
void run_frozen_global(const mtp_target::BFloat16* query,
                       const mtp_target::CacheView& cache,
                       const mtp_target::BFloat16* global_k_norm,
                       std::uint32_t processed_tokens,
                       mtp_target::BFloat16* context, void* scratch,
                       std::size_t scratch_size, cudaStream_t stream);

}  // namespace gewell::mtp_attention
