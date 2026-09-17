#include "gewell/console.h"
#include "gewell/models/gemma4/31b/artifact.h"
#include "gewell/models/gemma4/31b/model.h"
#include <algorithm>
#include <array>
#include <string>
namespace console = gewell::console;
namespace artifact = gewell::artifact;
namespace model = gewell::gemma4_31b;
namespace {
bool expect_tensor(std::size_t id, model::TensorRole role,
                   std::int16_t layer, std::uint32_t first,
                   std::uint32_t second = 0, std::uint32_t third = 0) {
  const auto& tensor = model::kPhysicalTensors[id];
  const std::uint8_t rank = third != 0 ? 3 : second != 0 ? 2 : 1;
  if (tensor.role == role && tensor.dtype == model::DType::bf16 &&
      tensor.layer == layer && tensor.shape.rank == rank &&
      tensor.shape.dimensions[0] == first &&
      tensor.shape.dimensions[1] == second &&
      tensor.shape.dimensions[2] == third) {
    return true;
  }

  console::message(std::string("tensor contract mismatch at physical id ") + std::to_string(id) +
                   " (expected role " + std::string(model::role_name(role)) + ")", true);
  return false;
}

bool validate_tensor_contract() {
  bool ok = true;
  std::size_t id = 0;
  std::uint32_t local_layers = 0;
  std::uint32_t global_layers = 0;

  ok &= expect_tensor(id++, model::TensorRole::embedding, model::kNoLayer,
                      model::kVocabSize, model::kHiddenSize);

  for (std::uint32_t layer_index = 0; layer_index < model::kLayerCount;
       ++layer_index) {
    const auto layer = static_cast<std::int16_t>(layer_index);
    const bool global = layer_index % 6 == 5;
    local_layers += global ? 0 : 1;
    global_layers += global ? 1 : 0;
    const std::uint32_t head_size =
        global ? model::kGlobalHeadSize : model::kLocalHeadSize;
    const std::uint32_t q_width = model::kQueryHeadCount * head_size;
    const std::uint32_t kv_width =
        (global ? model::kGlobalKvHeadCount : model::kLocalKvHeadCount) *
        head_size;

    ok &= expect_tensor(id++, model::TensorRole::input_norm, layer,
                        model::kHiddenSize);
    ok &= expect_tensor(id++, model::TensorRole::q_proj, layer, q_width,
                        model::kHiddenSize);
    ok &= expect_tensor(id++, model::TensorRole::k_proj, layer, kv_width,
                        model::kHiddenSize);
    if (!global) {
      ok &= expect_tensor(id++, model::TensorRole::v_proj, layer, kv_width,
                          model::kHiddenSize);
    }
    ok &= expect_tensor(id++, model::TensorRole::q_norm, layer, head_size);
    ok &= expect_tensor(id++, model::TensorRole::k_norm, layer, head_size);
    ok &= expect_tensor(id++, model::TensorRole::o_proj, layer,
                        model::kHiddenSize, q_width);
    ok &= expect_tensor(id++, model::TensorRole::post_attention_norm, layer,
                        model::kHiddenSize);
    ok &= expect_tensor(id++, model::TensorRole::pre_feedforward_norm, layer,
                        model::kHiddenSize);
    ok &= expect_tensor(id++, model::TensorRole::gate_proj, layer,
                        model::kMlpSize, model::kHiddenSize);
    ok &= expect_tensor(id++, model::TensorRole::up_proj, layer,
                        model::kMlpSize, model::kHiddenSize);
    ok &= expect_tensor(id++, model::TensorRole::down_proj, layer,
                        model::kHiddenSize, model::kMlpSize);
    ok &= expect_tensor(id++, model::TensorRole::post_feedforward_norm, layer,
                        model::kHiddenSize);
    ok &= expect_tensor(id++, model::TensorRole::layer_scalar, layer, 1);
  }

  ok &= expect_tensor(id++, model::TensorRole::final_norm, model::kNoLayer,
                      model::kHiddenSize);

  if (id != model::kVisionPatchProjectionPhysicalId) {
    console::message("text tensor count mismatch", true);
    ok = false;
  }
  ok &= expect_tensor(id++, model::TensorRole::vision_patch_proj,
                      model::kNoLayer, model::kVisionHiddenSize,
                      model::kVisionPatchWidth);
  ok &= expect_tensor(id++, model::TensorRole::vision_position_embedding,
                      model::kNoLayer, 2, model::kVisionPositionCount,
                      model::kVisionHiddenSize);

  if (id != model::kVisionLayerWeightsFirstPhysicalId) {
    console::message("vision pre-layer tensor count mismatch", true);
    ok = false;
  }
  for (std::uint32_t layer_index = 0;
       layer_index < model::kVisionLayerCount; ++layer_index) {
    const auto layer = static_cast<std::int16_t>(layer_index);
    ok &= expect_tensor(id++, model::TensorRole::vision_input_norm, layer,
                        model::kVisionHiddenSize);
    ok &= expect_tensor(id++, model::TensorRole::vision_q_proj, layer,
                        model::kVisionHiddenSize, model::kVisionHiddenSize);
    ok &= expect_tensor(id++, model::TensorRole::vision_k_proj, layer,
                        model::kVisionHiddenSize, model::kVisionHiddenSize);
    ok &= expect_tensor(id++, model::TensorRole::vision_v_proj, layer,
                        model::kVisionHiddenSize, model::kVisionHiddenSize);
    ok &= expect_tensor(id++, model::TensorRole::vision_q_norm, layer,
                        model::kVisionHeadSize);
    ok &= expect_tensor(id++, model::TensorRole::vision_k_norm, layer,
                        model::kVisionHeadSize);
    ok &= expect_tensor(id++, model::TensorRole::vision_o_proj, layer,
                        model::kVisionHiddenSize, model::kVisionHiddenSize);
    ok &= expect_tensor(id++,
                        model::TensorRole::vision_post_attention_norm, layer,
                        model::kVisionHiddenSize);
    ok &= expect_tensor(id++,
                        model::TensorRole::vision_pre_feedforward_norm, layer,
                        model::kVisionHiddenSize);
    ok &= expect_tensor(id++, model::TensorRole::vision_gate_proj, layer,
                        model::kVisionMlpSize, model::kVisionHiddenSize);
    ok &= expect_tensor(id++, model::TensorRole::vision_up_proj, layer,
                        model::kVisionMlpSize, model::kVisionHiddenSize);
    ok &= expect_tensor(id++, model::TensorRole::vision_down_proj, layer,
                        model::kVisionHiddenSize, model::kVisionMlpSize);
    ok &= expect_tensor(
        id++, model::TensorRole::vision_post_feedforward_norm, layer,
        model::kVisionHiddenSize);
  }
  ok &= expect_tensor(id++, model::TensorRole::vision_std_bias,
                      model::kNoLayer, model::kVisionHiddenSize);
  ok &= expect_tensor(id++, model::TensorRole::vision_std_scale,
                      model::kNoLayer, model::kVisionHiddenSize);
  ok &= expect_tensor(id++, model::TensorRole::vision_projection,
                      model::kNoLayer, model::kHiddenSize,
                      model::kVisionHiddenSize);
  if (id != model::kAssistantEmbeddingPhysicalId) {
    console::message("target tensor count mismatch", true);
    ok = false;
  }
  ok &= expect_tensor(id++, model::TensorRole::assistant_embedding,
                      model::kNoLayer, model::kVocabSize,
                      model::kAssistantHiddenSize);
  for (std::uint32_t layer_index = 0;
       layer_index < model::kAssistantLayerCount; ++layer_index) {
    const auto layer = static_cast<std::int16_t>(layer_index);
    const std::uint32_t head =
        layer_index == 3 ? model::kGlobalHeadSize : model::kLocalHeadSize;
    const std::uint32_t q_width = model::kQueryHeadCount * head;
    ok &= expect_tensor(id++, model::TensorRole::assistant_input_norm,
                        layer, model::kAssistantHiddenSize);
    ok &= expect_tensor(id++, model::TensorRole::assistant_q_proj, layer,
                        q_width, model::kAssistantHiddenSize);
    ok &= expect_tensor(id++, model::TensorRole::assistant_q_norm, layer, head);
    ok &= expect_tensor(id++, model::TensorRole::assistant_o_proj, layer,
                        model::kAssistantHiddenSize, q_width);
    ok &= expect_tensor(id++, model::TensorRole::assistant_post_attention_norm,
                        layer, model::kAssistantHiddenSize);
    ok &= expect_tensor(id++, model::TensorRole::assistant_pre_feedforward_norm,
                        layer, model::kAssistantHiddenSize);
    ok &= expect_tensor(id++, model::TensorRole::assistant_gate_proj, layer,
                        model::kAssistantMlpSize, model::kAssistantHiddenSize);
    ok &= expect_tensor(id++, model::TensorRole::assistant_up_proj, layer,
                        model::kAssistantMlpSize, model::kAssistantHiddenSize);
    ok &= expect_tensor(id++, model::TensorRole::assistant_down_proj, layer,
                        model::kAssistantHiddenSize, model::kAssistantMlpSize);
    ok &= expect_tensor(id++, model::TensorRole::assistant_post_feedforward_norm,
                        layer, model::kAssistantHiddenSize);
    ok &= expect_tensor(id++, model::TensorRole::assistant_layer_scalar, layer, 1);
  }
  ok &= expect_tensor(id++, model::TensorRole::assistant_final_norm,
                      model::kNoLayer, model::kAssistantHiddenSize);
  ok &= expect_tensor(id++, model::TensorRole::assistant_pre_projection,
                      model::kNoLayer, model::kAssistantHiddenSize,
                      2 * model::kHiddenSize);
  ok &= expect_tensor(id++, model::TensorRole::assistant_post_projection,
                      model::kNoLayer, model::kHiddenSize,
                      model::kAssistantHiddenSize);
  if (id != model::kPhysicalTensorCount) {
    console::message(std::string("physical tensor count mismatch: visited ") +
                     std::to_string(id) + ", expected " +
                     std::to_string(model::kPhysicalTensorCount), true);
    ok = false;
  }
  if (local_layers != model::kLocalLayerCount ||
      global_layers != model::kGlobalLayerCount) {
    console::message("attention schedule mismatch", true);
    ok = false;
  }
  if (model::kLogicalAliases.size() != 2 ||
      model::kLogicalAliases[0].role != model::LogicalTensorRole::lm_head ||
      model::kLogicalAliases[0].logical_id != model::kLmHeadLogicalId ||
      model::kLogicalAliases[0].physical_id != model::kEmbeddingPhysicalId) {
    console::message("logical lm_head must alias physical embedding id 0", true);
    ok = false;
  }
  if (model::kLogicalAliases[1].role !=
          model::LogicalTensorRole::assistant_lm_head ||
      model::kLogicalAliases[1].logical_id != model::kAssistantLmHeadLogicalId ||
      model::kLogicalAliases[1].physical_id !=
          model::kAssistantEmbeddingPhysicalId) {
    console::message("assistant lm_head must alias physical assistant embedding", true);
    ok = false;
  }
  if (model::logical_text_weight_bytes() != artifact::kLogicalDataBytes ||
      model::aligned_text_weight_bytes() != artifact::kPayloadBytes) {
    console::message("BF16 weight byte total mismatch", true);
    ok = false;
  }
  return ok;
}

}  // namespace
int main() { return validate_tensor_contract() ? 0 : 1; }
