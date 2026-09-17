#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace gewell::replay_metrics {

enum Flag : std::uint32_t {
  nonfinite = 1U,
  outside_softcap = 2U,
  reference_argmax_mismatch = 4U,
  invalid_recorded_token = 8U,
  invalid_kl = 16U,
};

// The reference is the model that generated the forced token history. Python
// assigns BF16/candidate KL directions from that reference role, independently
// of which trajectory was replayed. Both margins are max(logits)-logits[token].
struct Row {
  double kl_ref_replay;
  double kl_replay_ref;
  double replay_margin;
  double reference_margin;
  std::uint32_t reference_argmax;
  std::uint32_t replay_argmax;
  std::uint32_t recorded_token;
  std::uint32_t flags;
};
static_assert(sizeof(Row) == 48);

// All pointers are device pointers; logits are row-major post-softcap BF16.
// The caller owns every allocation and synchronization. This call launches one
// block per row and returns immediately, throwing on contract/launch errors.
// Device validation sets Row::flags; metrics with a nonzero flag are unusable.
void compare_rows(const __nv_bfloat16* reference,
                  const __nv_bfloat16* replay,
                  const std::uint32_t* recorded_tokens,
                  std::uint32_t rows, std::uint32_t vocabulary_size,
                  Row* output, cudaStream_t stream = nullptr);

}  // namespace gewell::replay_metrics
