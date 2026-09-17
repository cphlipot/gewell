#pragma once

#include "gewell/models/gemma4/31b/artifact.h"
#include "gewell/models/gemma4/31b/model.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace gewell::weight_qdq {

enum class Type : std::uint8_t {
  bf16,
  fp8,
  nvfp4,
  nvfp4_w4a4,
  fp8_w8a8,
};

struct TypeStats {
  std::uint32_t tensor_count{};
  std::uint64_t element_count{};
  std::uint64_t source_bf16_bytes{};
};

struct SelectionSummary {
  TypeStats bf16{};
  TypeStats fp8{};
  TypeStats nvfp4{};
  TypeStats nvfp4_w4a4{};
  TypeStats fp8_w8a8{};
};

class Mask {
 public:
  Mask() = default;

  // Each non-comment line is: LAYER PROJECTION TYPE. LAYER is '*' or 0..59,
  // PROJECTION is one of q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/
  // down_proj, and TYPE is bf16/fp8/nvfp4/nvfp4_w4a4/fp8_w8a8. Native
  // types require an already packed entry.
  // Rules are applied in order and
  // unspecified entries remain BF16.
  static Mask Parse(std::string_view source,
                    std::string source_name = "<memory>");
  static Mask Load(const std::string& path);

  [[nodiscard]] Type type_for(const gemma4_31b::TensorSpec& tensor) const;
  // Invalid layers and nonexistent global v_proj coordinates are errors.
  // Valid non-projection roles deliberately remain BF16.
  [[nodiscard]] Type type_for(std::uint32_t layer,
                              gemma4_31b::TensorRole role) const;
  [[nodiscard]] bool all_bf16() const;
  [[nodiscard]] bool has_qdq() const;
  [[nodiscard]] bool has_source() const { return has_source_; }
  [[nodiscard]] const std::string& source_name() const { return source_name_; }
  [[nodiscard]] const artifact::Digest& source_sha256() const {
    return source_sha256_;
  }

 private:
  static constexpr std::size_t kProjectionCount = 7;
  std::array<std::array<Type, kProjectionCount>,
             gemma4_31b::kLayerCount>
      types_{};
  bool has_source_{};
  std::string source_name_{};
  artifact::Digest source_sha256_{};
};

struct ApplySummary {
  SelectionSummary selection{};
  double seconds{};
};

[[nodiscard]] std::string_view type_name(Type type);
[[nodiscard]] SelectionSummary summarize(const Mask& mask);

// Reconstructs selected projection weights in place as BF16 after applying the
// declared logical quantizer. The artifact itself remains untouched. This call
// synchronizes before returning so later nonblocking inference streams see all
// reconstructed weights.
ApplySummary apply_in_place(const Mask& mask,
                            const artifact::ArtifactFile& file,
                            void* device_payload);

}  // namespace gewell::weight_qdq
