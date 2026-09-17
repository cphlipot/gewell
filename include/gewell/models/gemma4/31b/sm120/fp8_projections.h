#pragma once

#include "gewell/fp8_linear.h"
#include "gewell/models/gemma4/31b/model.h"
#include "gewell/models/gemma4/31b/sm120/projection_fusion.h"

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>

namespace gewell::fp8 {

using Weights = std::array<Weight, gemma4_31b::kLogicalTensorCount>;

inline bool adjacent(Weight first, Weight second, std::size_t elements) {
  return first.data && second.data && first.input_scale == second.input_scale &&
      reinterpret_cast<std::uintptr_t>(second.data) ==
          reinterpret_cast<std::uintptr_t>(first.data) + elements;
}

inline Weight qkv_weight(const Weights& weights, unsigned layer) {
  const auto q = 2 + 14 * layer - layer / 6;
  const bool global = gemma4_31b::is_global_layer(layer);
  if (!weights[q].channel_scales ||
      !adjacent(weights[q], weights[q + 1],
                std::size_t(global ? 16384 : 8192) * gemma4_31b::kHiddenSize) ||
      (!global && !adjacent(weights[q + 1], weights[q + 2],
                             std::size_t(4096) * gemma4_31b::kHiddenSize))) return {};
  return weights[q];
}

inline Weight gate_up_weight(Weight gate, Weight up) {
  return gate.channel_scales && adjacent(gate, up,
      std::size_t(gemma4_31b::kMlpSize) * gemma4_31b::kHiddenSize) ? gate : Weight{};
}

// Cache only selected projection shapes. All plans share stream-local scratch;
// prepare runs before capture and run only enqueues work.
class ProjectionPlans {
 public:
  ProjectionPlans(cublasLtHandle_t handle, const Weights& weights)
      : handle_(handle) {
    for (std::size_t id = 0; id < gemma4_31b::kPhysicalTensorCount; ++id) {
      const auto& spec = gemma4_31b::kPhysicalTensors[id];
      if (weights[id].data)
        shapes_.emplace(spec.shape.dimensions[1], spec.shape.dimensions[0], false);
      if (spec.role == gemma4_31b::TensorRole::q_proj && qkv_weight(weights, spec.layer).data)
        shapes_.emplace(gemma4_31b::kHiddenSize,
            gemma4_31b::sm120::fusion::qkv_width(gemma4_31b::is_global_layer(spec.layer)), true);
      if (spec.role == gemma4_31b::TensorRole::gate_proj &&
          gate_up_weight(weights[id], weights[id + 1]).data)
        shapes_.emplace(gemma4_31b::kHiddenSize, 2 * gemma4_31b::kMlpSize, true);
    }
  }
  ~ProjectionPlans() { if (scratch_) cudaFree(scratch_); }
  ProjectionPlans(const ProjectionPlans&) = delete;
  ProjectionPlans& operator=(const ProjectionPlans&) = delete;

  static constexpr std::size_t peak_scratch_bytes(std::uint32_t max_rows) {
    // The down-projection input bounds every supported projection width.
    return 2 * scratch_upper_bound_bytes(max_rows, gemma4_31b::kMlpSize);
  }

  void prepare(std::uint32_t rows) {
    for (const auto& [input_width, output_width, joined] : shapes_) {
      const auto key = std::make_tuple(rows, input_width, output_width, joined);
      if (plans_.count(key)) continue;
      auto plan = std::make_unique<Plan>(handle_, rows, input_width, output_width, joined);
      const auto needed = plan->scratch_bytes();
      if (needed > scratch_bytes_) {
        void* replacement = nullptr;
        const auto status = cudaMalloc(&replacement, needed);
        if (status != cudaSuccess)
          throw std::runtime_error(std::string("FP8 projection workspace: ") +
                                   cudaGetErrorString(status));
        if (scratch_) cudaFree(scratch_);
        scratch_ = replacement;
        scratch_bytes_ = needed;
      }
      plans_.emplace(key, std::move(plan));
    }
  }

  std::size_t scratch_bytes() const { return scratch_bytes_; }

  void run(std::uint32_t rows, std::uint32_t input_width,
           std::uint32_t output_width, const __nv_bfloat16* input,
           Weight weight, __nv_bfloat16* output, cudaStream_t stream,
           InputTransform transform = InputTransform::identity, bool joined = false) {
    plans_.at(std::make_tuple(rows, input_width, output_width, joined))->run(
        input, weight, output, scratch_, scratch_bytes_, stream, transform);
  }

  void run_joined(std::uint32_t rows, std::uint32_t output_width,
                  const __nv_bfloat16* input, Weight weight,
                  __nv_bfloat16* output, cudaStream_t stream) {
    run(rows, gemma4_31b::kHiddenSize, output_width, input, weight, output, stream,
        InputTransform::identity, true);
  }

 private:
  cublasLtHandle_t handle_;
  std::set<std::tuple<std::uint32_t, std::uint32_t, bool>> shapes_;
  std::map<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t, bool>,
           std::unique_ptr<Plan>> plans_;
  void* scratch_{};
  std::size_t scratch_bytes_{};
};

}  // namespace gewell::fp8
