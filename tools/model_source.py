"""Tensor inventory and local metadata for Gemma 4 31B conversion."""

from __future__ import annotations

import hashlib
from pathlib import Path


TEXT_PREFIX = "model.language_model."
VISION_PREFIX = "model.vision_tower."
TEXT_TENSOR_COUNT = 832
VISION_TENSOR_COUNT = 356
MODEL_TENSOR_COUNT = TEXT_TENSOR_COUNT + VISION_TENSOR_COUNT
VOCAB_SIZE = 262_144


def expected_text_tensor_names() -> frozenset[str]:
    """Return the exact stored BF16 tensor inventory for the text model."""

    names = {
        f"{TEXT_PREFIX}embed_tokens.weight",
        f"{TEXT_PREFIX}norm.weight",
    }
    for layer in range(60):
        root = f"{TEXT_PREFIX}layers.{layer}"
        names.update(
            {
                f"{root}.input_layernorm.weight",
                f"{root}.post_attention_layernorm.weight",
                f"{root}.pre_feedforward_layernorm.weight",
                f"{root}.post_feedforward_layernorm.weight",
                f"{root}.layer_scalar",
                f"{root}.mlp.gate_proj.weight",
                f"{root}.mlp.up_proj.weight",
                f"{root}.mlp.down_proj.weight",
                f"{root}.self_attn.q_proj.weight",
                f"{root}.self_attn.k_proj.weight",
                f"{root}.self_attn.o_proj.weight",
                f"{root}.self_attn.q_norm.weight",
                f"{root}.self_attn.k_norm.weight",
            }
        )
        if layer % 6 != 5:
            names.add(f"{root}.self_attn.v_proj.weight")
    return frozenset(names)


def expected_vision_tensor_names() -> frozenset[str]:
    """Return the exact stored BF16 tensor inventory for the vision model."""

    names = {
        "model.embed_vision.embedding_projection.weight",
        f"{VISION_PREFIX}patch_embedder.input_proj.weight",
        f"{VISION_PREFIX}patch_embedder.position_embedding_table",
        f"{VISION_PREFIX}std_bias",
        f"{VISION_PREFIX}std_scale",
    }
    for layer in range(27):
        root = f"{VISION_PREFIX}encoder.layers.{layer}"
        names.update(
            {
                f"{root}.input_layernorm.weight",
                f"{root}.post_attention_layernorm.weight",
                f"{root}.pre_feedforward_layernorm.weight",
                f"{root}.post_feedforward_layernorm.weight",
                f"{root}.mlp.gate_proj.linear.weight",
                f"{root}.mlp.up_proj.linear.weight",
                f"{root}.mlp.down_proj.linear.weight",
                f"{root}.self_attn.q_proj.linear.weight",
                f"{root}.self_attn.k_proj.linear.weight",
                f"{root}.self_attn.v_proj.linear.weight",
                f"{root}.self_attn.o_proj.linear.weight",
                f"{root}.self_attn.q_norm.weight",
                f"{root}.self_attn.k_norm.weight",
            }
        )
    return frozenset(names)


def expected_model_tensor_names() -> frozenset[str]:
    """Return every stored tensor in a multimodal checkpoint."""

    return expected_text_tensor_names() | expected_vision_tensor_names()


def load_model(snapshot: Path) -> dict[str, str]:
    """Record local provenance without restricting repository, revision or bytes."""
    snapshot = snapshot.resolve()
    def digest(name):
        path = snapshot / name
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "0" * 64
    return {"repository": "", "revision": "", "snapshot": str(snapshot),
            "config_sha256": digest("config.json"),
            "index_sha256": digest("model.safetensors.index.json")}
