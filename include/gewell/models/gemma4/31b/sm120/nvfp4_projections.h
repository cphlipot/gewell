#pragma once

#include "gewell/models/gemma4/31b/model.h"
#include "gewell/nvfp4_linear.h"
#include "gewell/nvfp4_policy.h"

#include <array>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>

namespace gewell::nvfp4 {

using Weights = std::array<Weight, gemma4_31b::kLogicalTensorCount>;

// The loader joins packed rows and their 128-row scale tiles in place. A
// shared input quantization and GEMM alpha require identical global scales.
inline Weight gate_up_weight(Weight gate, Weight up) {
  constexpr auto k = gemma4_31b::kHiddenSize, n = gemma4_31b::kMlpSize;
  static_assert(n % 128 == 0);
  if (!gate.data || !up.data || !gate.scales || !up.scales ||
      gate.input_scale != up.input_scale || gate.weight_scale != up.weight_scale ||
      reinterpret_cast<std::uintptr_t>(up.data) !=
          reinterpret_cast<std::uintptr_t>(gate.data) + std::size_t(n) * k / 2 ||
      reinterpret_cast<std::uintptr_t>(up.scales) !=
          reinterpret_cast<std::uintptr_t>(gate.scales) + scale_storage_bytes(n, k))
    return {};
  return gate;
}

// Shape plans share one workspace because a model executor uses one stream.
// prepare() runs before a layer traversal; run() only enqueues device work.
class ProjectionPlans {
 public:
  ProjectionPlans(cublasLtHandle_t handle, const Weights& weights,
                  ActivationPolicy policy = ActivationPolicy::always)
      : handle_(handle), policy_(policy) {
    for (std::size_t id = 0; id < gemma4_31b::kPhysicalTensorCount; ++id) {
      const auto& spec = gemma4_31b::kPhysicalTensors[id];
      if (weights[id].data)
        shapes_.emplace(spec.shape.dimensions[1], spec.shape.dimensions[0]);
      if (spec.role == gemma4_31b::TensorRole::gate_proj &&
          gate_up_weight(weights[id], weights[id + 1]).data)
        shapes_.emplace(gemma4_31b::kHiddenSize, 2 * gemma4_31b::kMlpSize);
    }
    if (policy == ActivationPolicy::prefill) {
      const auto status = cudaMalloc(&dequantized_, dequantized_weight_bytes());
      if (status != cudaSuccess)
        throw std::runtime_error(std::string("NVFP4 BF16 weight workspace: ") + cudaGetErrorString(status));
    }
  }
  ~ProjectionPlans() {
    if (scratch_) cudaFree(scratch_);
    if (dequantized_) cudaFree(dequantized_);
  }
  ProjectionPlans(const ProjectionPlans&) = delete;
  ProjectionPlans& operator=(const ProjectionPlans&) = delete;

  static constexpr std::size_t dequantized_weight_bytes() {
    // The MLP matrix bounds every supported attention projection too.
    return std::size_t(gemma4_31b::kHiddenSize) * gemma4_31b::kMlpSize * sizeof(__nv_bfloat16);
  }

  void prepare(std::uint32_t rows) {
    for (const auto& [input_width, output_width] : shapes_) {
      const auto key = std::make_tuple(rows, input_width, output_width);
      if (plans_.count(key)) continue;
      auto plan = std::make_unique<Plan>(handle_, rows, input_width, output_width);
      const auto needed = plan->scratch_bytes();
      if (needed > scratch_bytes_) {
        void* replacement = nullptr;
        const auto status = cudaMalloc(&replacement, needed);
        if (status != cudaSuccess)
          throw std::runtime_error(std::string("NVFP4 projection workspace: ") +
                                   cudaGetErrorString(status));
        if (scratch_) cudaFree(scratch_);
        scratch_ = replacement;
        scratch_bytes_ = needed;
      }
      plans_.emplace(key, std::move(plan));
    }
  }

  std::size_t scratch_bytes() const {
    // Each plan owns a 16-byte scale placeholder for heuristics.
    return scratch_bytes_ + plans_.size() * 16 +
        (dequantized_ ? dequantized_weight_bytes() : 0);
  }

  bool fp4_activations(Phase phase) const { return nvfp4::fp4_activations(policy_, phase); }

  const __nv_bfloat16* decode_weight(Weight weight, std::uint32_t input_width,
                                     std::uint32_t output_width, cudaStream_t stream) {
    if (!dequantized_) throw std::logic_error("NVFP4 BF16 decode workspace is disabled");
    dequantize(weight, input_width, output_width,
               static_cast<__nv_bfloat16*>(dequantized_), stream);
    return static_cast<const __nv_bfloat16*>(dequantized_);
  }

  void run(std::uint32_t rows, std::uint32_t input_width,
           std::uint32_t output_width, const __nv_bfloat16* input,
           Weight weight, __nv_bfloat16* output, cudaStream_t stream,
           InputTransform transform = InputTransform::identity) {
    plans_.at(std::make_tuple(rows, input_width, output_width))->run(
        input, weight, output, scratch_, scratch_bytes_, stream, transform);
  }

 private:
  cublasLtHandle_t handle_;
  ActivationPolicy policy_;
  std::set<std::pair<std::uint32_t, std::uint32_t>> shapes_;
  std::map<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>,
           std::unique_ptr<Plan>> plans_;
  void* scratch_{};
  std::size_t scratch_bytes_{};
  void* dequantized_{};
};

}  // namespace gewell::nvfp4
