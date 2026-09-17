#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace gewell::gemma4_31b {

inline constexpr std::uint32_t kLayerCount = 60;
inline constexpr std::uint32_t kLocalLayerCount = 50;
inline constexpr std::uint32_t kGlobalLayerCount = 10;
inline constexpr std::uint32_t kHiddenSize = 5'376;
inline constexpr std::uint32_t kMlpSize = 21'504;
inline constexpr std::uint32_t kVocabSize = 262'144;
inline constexpr std::uint32_t kPadTokenId = 0;
inline constexpr std::uint32_t kBeginImageTokenId = 255'999;
inline constexpr std::uint32_t kImageTokenId = 258'880;
inline constexpr std::uint32_t kEndImageTokenId = 258'882;
inline constexpr std::uint32_t kQueryHeadCount = 32;
inline constexpr std::uint32_t kLocalKvHeadCount = 16;
inline constexpr std::uint32_t kGlobalKvHeadCount = 4;
inline constexpr std::uint32_t kLocalHeadSize = 256;
inline constexpr std::uint32_t kGlobalHeadSize = 512;
inline constexpr std::uint32_t kLocalWindowSize = 1'024;
inline constexpr std::uint32_t kVisionLayerCount = 27;
inline constexpr std::uint32_t kVisionHiddenSize = 1'152;
inline constexpr std::uint32_t kVisionMlpSize = 4'304;
inline constexpr std::uint32_t kVisionHeadCount = 16;
inline constexpr std::uint32_t kVisionHeadSize = 72;
inline constexpr std::uint32_t kVisionPatchSize = 16;
inline constexpr std::uint32_t kVisionPatchWidth = 768;
inline constexpr std::uint32_t kVisionPoolSize = 3;
inline constexpr std::uint32_t kVisionPositionCount = 10'240;
inline constexpr std::uint32_t kVisionMaxSoftTokenCount = 1'120;
inline constexpr std::uint64_t kBf16Bytes = 2;
inline constexpr std::uint64_t kStorageAlignment = 4'096;
inline constexpr std::size_t kTextPhysicalTensorCount = 832;
inline constexpr std::size_t kVisionPhysicalTensorCount = 356;
inline constexpr std::uint32_t kAssistantLayerCount = 4;
inline constexpr std::uint32_t kAssistantHiddenSize = 1'024;
inline constexpr std::uint32_t kAssistantMlpSize = 8'192;
inline constexpr std::size_t kAssistantPhysicalTensorCount = 48;
inline constexpr std::size_t kAssistantEmbeddingPhysicalId = 1'188;
inline constexpr std::size_t kAssistantLayerWeightsFirstPhysicalId = 1'189;
inline constexpr std::size_t kAssistantLayerTensorCount = 11;
inline constexpr std::size_t kAssistantFinalNormPhysicalId = 1'233;
inline constexpr std::size_t kAssistantPreProjectionPhysicalId = 1'234;
inline constexpr std::size_t kAssistantPostProjectionPhysicalId = 1'235;
inline constexpr std::size_t kAssistantLmHeadLogicalId = 1'237;
inline constexpr std::size_t kAssistantLmHeadPhysicalId = kAssistantEmbeddingPhysicalId;
inline constexpr std::size_t kPhysicalTensorCount = 1'236;
inline constexpr std::size_t kLogicalTensorCount = 1'238;
inline constexpr std::size_t kEmbeddingPhysicalId = 0;
inline constexpr std::size_t kFinalNormPhysicalId = 831;
inline constexpr std::size_t kVisionPatchProjectionPhysicalId = 832;
inline constexpr std::size_t kVisionPositionEmbeddingPhysicalId = 833;
inline constexpr std::size_t kVisionLayerWeightsFirstPhysicalId = 834;
inline constexpr std::size_t kVisionLayerTensorCount = 13;
inline constexpr std::size_t kVisionStdBiasPhysicalId = 1'185;
inline constexpr std::size_t kVisionStdScalePhysicalId = 1'186;
inline constexpr std::size_t kVisionProjectionPhysicalId = 1'187;
inline constexpr std::size_t kLmHeadLogicalId = 1'236;
inline constexpr std::int16_t kNoLayer = -1;

enum class DType : std::uint8_t {
  bf16 = 1,
};

enum class AttentionKind : std::uint8_t {
  local,
  global,
};

enum class TensorRole : std::uint8_t {
  embedding,
  input_norm,
  q_proj,
  k_proj,
  v_proj,
  q_norm,
  k_norm,
  o_proj,
  post_attention_norm,
  pre_feedforward_norm,
  gate_proj,
  up_proj,
  down_proj,
  post_feedforward_norm,
  layer_scalar,
  final_norm,
  vision_patch_proj,
  vision_position_embedding,
  vision_input_norm,
  vision_q_proj,
  vision_k_proj,
  vision_v_proj,
  vision_q_norm,
  vision_k_norm,
  vision_o_proj,
  vision_post_attention_norm,
  vision_pre_feedforward_norm,
  vision_gate_proj,
  vision_up_proj,
  vision_down_proj,
  vision_post_feedforward_norm,
  vision_std_bias,
  vision_std_scale,
  vision_projection,
  assistant_embedding,
  assistant_input_norm,
  assistant_q_proj,
  assistant_q_norm,
  assistant_o_proj,
  assistant_post_attention_norm,
  assistant_pre_feedforward_norm,
  assistant_gate_proj,
  assistant_up_proj,
  assistant_down_proj,
  assistant_post_feedforward_norm,
  assistant_layer_scalar,
  assistant_final_norm,
  assistant_pre_projection,
  assistant_post_projection,
};

enum class LogicalTensorRole : std::uint8_t {
  lm_head,
  assistant_lm_head,
};

struct TensorShape {
  std::array<std::uint32_t, 3> dimensions{};
  std::uint8_t rank{};

  [[nodiscard]] constexpr std::uint64_t element_count() const {
    if (rank == 1) {
      return dimensions[0];
    }
    if (rank == 2) {
      return static_cast<std::uint64_t>(dimensions[0]) * dimensions[1];
    }
    if (rank == 3) {
      return static_cast<std::uint64_t>(dimensions[0]) * dimensions[1] *
             dimensions[2];
    }
    return 0;
  }
};

struct TensorSpec {
  TensorRole role{};
  DType dtype{};
  std::int16_t layer{kNoLayer};
  TensorShape shape{};

  [[nodiscard]] constexpr std::uint64_t byte_count() const {
    return shape.element_count() * kBf16Bytes;
  }
};

struct LogicalAlias {
  LogicalTensorRole role{};
  std::size_t logical_id{};
  std::size_t physical_id{};
};

[[nodiscard]] constexpr TensorShape vector_shape(std::uint32_t size) {
  return {{{size, 0, 0}}, 1};
}

[[nodiscard]] constexpr TensorShape matrix_shape(std::uint32_t rows,
                                                 std::uint32_t columns) {
  return {{{rows, columns, 0}}, 2};
}

[[nodiscard]] constexpr TensorShape tensor_shape(std::uint32_t planes,
                                                 std::uint32_t rows,
                                                 std::uint32_t columns) {
  return {{{planes, rows, columns}}, 3};
}

[[nodiscard]] constexpr AttentionKind attention_kind(std::uint32_t layer) {
  return layer % 6 == 5 ? AttentionKind::global : AttentionKind::local;
}

[[nodiscard]] constexpr bool is_global_layer(std::uint32_t layer) {
  return attention_kind(layer) == AttentionKind::global;
}

[[nodiscard]] constexpr std::array<TensorSpec, kPhysicalTensorCount>
make_physical_tensors() {
  std::array<TensorSpec, kPhysicalTensorCount> tensors{};
  std::size_t id = 0;

  tensors[id++] = {TensorRole::embedding, DType::bf16, kNoLayer,
                   matrix_shape(kVocabSize, kHiddenSize)};

  for (std::uint32_t layer_index = 0; layer_index < kLayerCount;
       ++layer_index) {
    const auto layer = static_cast<std::int16_t>(layer_index);
    const bool global = is_global_layer(layer_index);
    const std::uint32_t head_size = global ? kGlobalHeadSize : kLocalHeadSize;
    const std::uint32_t q_width = kQueryHeadCount * head_size;
    const std::uint32_t kv_width =
        (global ? kGlobalKvHeadCount : kLocalKvHeadCount) * head_size;

    tensors[id++] = {TensorRole::input_norm, DType::bf16, layer,
                     vector_shape(kHiddenSize)};
    tensors[id++] = {TensorRole::q_proj, DType::bf16, layer,
                     matrix_shape(q_width, kHiddenSize)};
    tensors[id++] = {TensorRole::k_proj, DType::bf16, layer,
                     matrix_shape(kv_width, kHiddenSize)};
    if (!global) {
      tensors[id++] = {TensorRole::v_proj, DType::bf16, layer,
                       matrix_shape(kv_width, kHiddenSize)};
    }
    tensors[id++] = {TensorRole::q_norm, DType::bf16, layer,
                     vector_shape(head_size)};
    tensors[id++] = {TensorRole::k_norm, DType::bf16, layer,
                     vector_shape(head_size)};
    tensors[id++] = {TensorRole::o_proj, DType::bf16, layer,
                     matrix_shape(kHiddenSize, q_width)};
    tensors[id++] = {TensorRole::post_attention_norm, DType::bf16, layer,
                     vector_shape(kHiddenSize)};
    tensors[id++] = {TensorRole::pre_feedforward_norm, DType::bf16, layer,
                     vector_shape(kHiddenSize)};
    tensors[id++] = {TensorRole::gate_proj, DType::bf16, layer,
                     matrix_shape(kMlpSize, kHiddenSize)};
    tensors[id++] = {TensorRole::up_proj, DType::bf16, layer,
                     matrix_shape(kMlpSize, kHiddenSize)};
    tensors[id++] = {TensorRole::down_proj, DType::bf16, layer,
                     matrix_shape(kHiddenSize, kMlpSize)};
    tensors[id++] = {TensorRole::post_feedforward_norm, DType::bf16, layer,
                     vector_shape(kHiddenSize)};
    tensors[id++] = {TensorRole::layer_scalar, DType::bf16, layer,
                     vector_shape(1)};
  }

  tensors[id++] = {TensorRole::final_norm, DType::bf16, kNoLayer,
                   vector_shape(kHiddenSize)};

  tensors[id++] = {TensorRole::vision_patch_proj, DType::bf16, kNoLayer,
                   matrix_shape(kVisionHiddenSize, kVisionPatchWidth)};
  tensors[id++] = {
      TensorRole::vision_position_embedding, DType::bf16, kNoLayer,
      tensor_shape(2, kVisionPositionCount, kVisionHiddenSize)};

  for (std::uint32_t layer_index = 0; layer_index < kVisionLayerCount;
       ++layer_index) {
    const auto layer = static_cast<std::int16_t>(layer_index);
    tensors[id++] = {TensorRole::vision_input_norm, DType::bf16, layer,
                     vector_shape(kVisionHiddenSize)};
    tensors[id++] = {TensorRole::vision_q_proj, DType::bf16, layer,
                     matrix_shape(kVisionHiddenSize, kVisionHiddenSize)};
    tensors[id++] = {TensorRole::vision_k_proj, DType::bf16, layer,
                     matrix_shape(kVisionHiddenSize, kVisionHiddenSize)};
    tensors[id++] = {TensorRole::vision_v_proj, DType::bf16, layer,
                     matrix_shape(kVisionHiddenSize, kVisionHiddenSize)};
    tensors[id++] = {TensorRole::vision_q_norm, DType::bf16, layer,
                     vector_shape(kVisionHeadSize)};
    tensors[id++] = {TensorRole::vision_k_norm, DType::bf16, layer,
                     vector_shape(kVisionHeadSize)};
    tensors[id++] = {TensorRole::vision_o_proj, DType::bf16, layer,
                     matrix_shape(kVisionHiddenSize, kVisionHiddenSize)};
    tensors[id++] = {TensorRole::vision_post_attention_norm, DType::bf16,
                     layer, vector_shape(kVisionHiddenSize)};
    tensors[id++] = {TensorRole::vision_pre_feedforward_norm, DType::bf16,
                     layer, vector_shape(kVisionHiddenSize)};
    tensors[id++] = {TensorRole::vision_gate_proj, DType::bf16, layer,
                     matrix_shape(kVisionMlpSize, kVisionHiddenSize)};
    tensors[id++] = {TensorRole::vision_up_proj, DType::bf16, layer,
                     matrix_shape(kVisionMlpSize, kVisionHiddenSize)};
    tensors[id++] = {TensorRole::vision_down_proj, DType::bf16, layer,
                     matrix_shape(kVisionHiddenSize, kVisionMlpSize)};
    tensors[id++] = {TensorRole::vision_post_feedforward_norm, DType::bf16,
                     layer, vector_shape(kVisionHiddenSize)};
  }

  tensors[id++] = {TensorRole::vision_std_bias, DType::bf16, kNoLayer,
                   vector_shape(kVisionHiddenSize)};
  tensors[id++] = {TensorRole::vision_std_scale, DType::bf16, kNoLayer,
                   vector_shape(kVisionHiddenSize)};
  tensors[id++] = {TensorRole::vision_projection, DType::bf16, kNoLayer,
                   matrix_shape(kHiddenSize, kVisionHiddenSize)};
  tensors[id++] = {TensorRole::assistant_embedding, DType::bf16, kNoLayer,
                   matrix_shape(kVocabSize, kAssistantHiddenSize)};
  for (std::uint32_t layer_index = 0; layer_index < kAssistantLayerCount; ++layer_index) {
    const auto layer = static_cast<std::int16_t>(layer_index);
    const std::uint32_t head = layer_index == 3 ? kGlobalHeadSize : kLocalHeadSize;
    const std::uint32_t q_width = kQueryHeadCount * head;
    tensors[id++] = {TensorRole::assistant_input_norm, DType::bf16, layer,
                     vector_shape(kAssistantHiddenSize)};
    tensors[id++] = {TensorRole::assistant_q_proj, DType::bf16, layer,
                     matrix_shape(q_width, kAssistantHiddenSize)};
    tensors[id++] = {TensorRole::assistant_q_norm, DType::bf16, layer,
                     vector_shape(head)};
    tensors[id++] = {TensorRole::assistant_o_proj, DType::bf16, layer,
                     matrix_shape(kAssistantHiddenSize, q_width)};
    tensors[id++] = {TensorRole::assistant_post_attention_norm, DType::bf16, layer,
                     vector_shape(kAssistantHiddenSize)};
    tensors[id++] = {TensorRole::assistant_pre_feedforward_norm, DType::bf16, layer,
                     vector_shape(kAssistantHiddenSize)};
    tensors[id++] = {TensorRole::assistant_gate_proj, DType::bf16, layer,
                     matrix_shape(kAssistantMlpSize, kAssistantHiddenSize)};
    tensors[id++] = {TensorRole::assistant_up_proj, DType::bf16, layer,
                     matrix_shape(kAssistantMlpSize, kAssistantHiddenSize)};
    tensors[id++] = {TensorRole::assistant_down_proj, DType::bf16, layer,
                     matrix_shape(kAssistantHiddenSize, kAssistantMlpSize)};
    tensors[id++] = {TensorRole::assistant_post_feedforward_norm, DType::bf16, layer,
                     vector_shape(kAssistantHiddenSize)};
    tensors[id++] = {TensorRole::assistant_layer_scalar, DType::bf16, layer,
                     vector_shape(1)};
  }
  tensors[id++] = {TensorRole::assistant_final_norm, DType::bf16, kNoLayer,
                   vector_shape(kAssistantHiddenSize)};
  tensors[id++] = {TensorRole::assistant_pre_projection, DType::bf16, kNoLayer,
                   matrix_shape(kAssistantHiddenSize, 2 * kHiddenSize)};
  tensors[id++] = {TensorRole::assistant_post_projection, DType::bf16, kNoLayer,
                   matrix_shape(kHiddenSize, kAssistantHiddenSize)};
  return tensors;
}

inline constexpr auto kPhysicalTensors = make_physical_tensors();
inline constexpr std::array<LogicalAlias, 2> kLogicalAliases{{
    {LogicalTensorRole::lm_head, kLmHeadLogicalId, kEmbeddingPhysicalId},
    {LogicalTensorRole::assistant_lm_head, kAssistantLmHeadLogicalId, kAssistantEmbeddingPhysicalId},
}};
inline constexpr std::size_t kLmHeadPhysicalId =
    kLogicalAliases[0].physical_id;

[[nodiscard]] constexpr std::uint64_t align_up(std::uint64_t value,
                                               std::uint64_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

[[nodiscard]] constexpr std::uint64_t logical_weight_bytes() {
  std::uint64_t total = 0;
  for (const TensorSpec& tensor : kPhysicalTensors) {
    total += tensor.byte_count();
  }
  return total;
}

[[nodiscard]] constexpr std::uint64_t aligned_weight_bytes() {
  std::uint64_t total = 0;
  for (const TensorSpec& tensor : kPhysicalTensors) {
    total += align_up(tensor.byte_count(), kStorageAlignment);
  }
  return total;
}

[[nodiscard]] constexpr std::uint64_t logical_text_weight_bytes() {
  std::uint64_t total = 0;
  for (std::size_t i = 0; i < kTextPhysicalTensorCount; ++i)
    total += kPhysicalTensors[i].byte_count();
  return total;
}

[[nodiscard]] constexpr std::uint64_t aligned_text_weight_bytes() {
  std::uint64_t total = 0;
  for (std::size_t i = 0; i < kTextPhysicalTensorCount; ++i)
    total += align_up(kPhysicalTensors[i].byte_count(), kStorageAlignment);
  return total;
}

[[nodiscard]] constexpr std::string_view role_name(TensorRole role) {
  switch (role) {
    case TensorRole::embedding:
      return "embedding";
    case TensorRole::input_norm:
      return "input_norm";
    case TensorRole::q_proj:
      return "q_proj";
    case TensorRole::k_proj:
      return "k_proj";
    case TensorRole::v_proj:
      return "v_proj";
    case TensorRole::q_norm:
      return "q_norm";
    case TensorRole::k_norm:
      return "k_norm";
    case TensorRole::o_proj:
      return "o_proj";
    case TensorRole::post_attention_norm:
      return "post_attention_norm";
    case TensorRole::pre_feedforward_norm:
      return "pre_feedforward_norm";
    case TensorRole::gate_proj:
      return "gate_proj";
    case TensorRole::up_proj:
      return "up_proj";
    case TensorRole::down_proj:
      return "down_proj";
    case TensorRole::post_feedforward_norm:
      return "post_feedforward_norm";
    case TensorRole::layer_scalar:
      return "layer_scalar";
    case TensorRole::final_norm:
      return "final_norm";
    case TensorRole::vision_patch_proj:
      return "vision_patch_proj";
    case TensorRole::vision_position_embedding:
      return "vision_position_embedding";
    case TensorRole::vision_input_norm:
      return "vision_input_norm";
    case TensorRole::vision_q_proj:
      return "vision_q_proj";
    case TensorRole::vision_k_proj:
      return "vision_k_proj";
    case TensorRole::vision_v_proj:
      return "vision_v_proj";
    case TensorRole::vision_q_norm:
      return "vision_q_norm";
    case TensorRole::vision_k_norm:
      return "vision_k_norm";
    case TensorRole::vision_o_proj:
      return "vision_o_proj";
    case TensorRole::vision_post_attention_norm:
      return "vision_post_attention_norm";
    case TensorRole::vision_pre_feedforward_norm:
      return "vision_pre_feedforward_norm";
    case TensorRole::vision_gate_proj:
      return "vision_gate_proj";
    case TensorRole::vision_up_proj:
      return "vision_up_proj";
    case TensorRole::vision_down_proj:
      return "vision_down_proj";
    case TensorRole::vision_post_feedforward_norm:
      return "vision_post_feedforward_norm";
    case TensorRole::vision_std_bias:
      return "vision_std_bias";
    case TensorRole::vision_std_scale:
      return "vision_std_scale";
    case TensorRole::vision_projection:
      return "vision_projection";
    case TensorRole::assistant_embedding:
      return "assistant_embedding";
    case TensorRole::assistant_input_norm:
      return "assistant_input_norm";
    case TensorRole::assistant_q_proj:
      return "assistant_q_proj";
    case TensorRole::assistant_q_norm:
      return "assistant_q_norm";
    case TensorRole::assistant_o_proj:
      return "assistant_o_proj";
    case TensorRole::assistant_post_attention_norm:
      return "assistant_post_attention_norm";
    case TensorRole::assistant_pre_feedforward_norm:
      return "assistant_pre_feedforward_norm";
    case TensorRole::assistant_gate_proj:
      return "assistant_gate_proj";
    case TensorRole::assistant_up_proj:
      return "assistant_up_proj";
    case TensorRole::assistant_down_proj:
      return "assistant_down_proj";
    case TensorRole::assistant_post_feedforward_norm:
      return "assistant_post_feedforward_norm";
    case TensorRole::assistant_layer_scalar:
      return "assistant_layer_scalar";
    case TensorRole::assistant_final_norm:
      return "assistant_final_norm";
    case TensorRole::assistant_pre_projection:
      return "assistant_pre_projection";
    case TensorRole::assistant_post_projection:
      return "assistant_post_projection";

  }
  return "unknown";
}

static_assert(kPhysicalTensors.size() == kPhysicalTensorCount);
static_assert(kTextPhysicalTensorCount + kVisionPhysicalTensorCount + kAssistantPhysicalTensorCount ==
              kPhysicalTensorCount);
static_assert(kVisionPatchWidth ==
              3 * kVisionPatchSize * kVisionPatchSize);
static_assert(kVisionHeadCount * kVisionHeadSize == kVisionHiddenSize);
static_assert(kVisionLayerWeightsFirstPhysicalId +
                  kVisionLayerCount * kVisionLayerTensorCount ==
              kVisionStdBiasPhysicalId);
static_assert(kVisionProjectionPhysicalId + 1 == kAssistantEmbeddingPhysicalId);
static_assert(kPhysicalTensors[kFinalNormPhysicalId].role ==
              TensorRole::final_norm);
static_assert(kPhysicalTensors[kVisionPatchProjectionPhysicalId].role ==
              TensorRole::vision_patch_proj);
static_assert(kPhysicalTensors[kVisionPositionEmbeddingPhysicalId].role ==
              TensorRole::vision_position_embedding);
static_assert(kPhysicalTensors[kVisionStdBiasPhysicalId].role ==
              TensorRole::vision_std_bias);
static_assert(kPhysicalTensors[kVisionStdScalePhysicalId].role ==
              TensorRole::vision_std_scale);
static_assert(kPhysicalTensors[kVisionProjectionPhysicalId].role ==
              TensorRole::vision_projection);
static_assert(kLogicalAliases[0].logical_id == kLmHeadLogicalId);
static_assert(kLmHeadLogicalId >= kPhysicalTensorCount);
static_assert(kLmHeadLogicalId < kLogicalTensorCount);
static_assert(kLogicalTensorCount == kAssistantLmHeadLogicalId + 1);
static_assert(kLmHeadPhysicalId == kEmbeddingPhysicalId);
static_assert(kAssistantPostProjectionPhysicalId + 1 == kPhysicalTensorCount);
static_assert(kAssistantLayerWeightsFirstPhysicalId + kAssistantLayerCount * kAssistantLayerTensorCount == kAssistantFinalNormPhysicalId);
static_assert(kPhysicalTensors[kAssistantEmbeddingPhysicalId].role == TensorRole::assistant_embedding);
static_assert(kPhysicalTensors[kAssistantFinalNormPhysicalId].role == TensorRole::assistant_final_norm);
static_assert(kPhysicalTensors[kAssistantPreProjectionPhysicalId].role == TensorRole::assistant_pre_projection);
static_assert(kPhysicalTensors[kAssistantPostProjectionPhysicalId].role == TensorRole::assistant_post_projection);
static_assert(kLogicalAliases[1].physical_id == kAssistantLmHeadPhysicalId);
static_assert(logical_weight_bytes() == 63'485'214'944ULL);
static_assert(aligned_weight_bytes() == 63'486'726'144ULL);

}  // namespace gewell::gemma4_31b
