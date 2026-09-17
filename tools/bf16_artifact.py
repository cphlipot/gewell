#!/usr/bin/env python3
"""Wire-format helpers for Gemma 4 31B BF16 weights.

This module is deliberately dependency-free.  The production tools always use
``expected_tensor_specs()``; accepting an explicit spec sequence in the lower
level helpers exists only so the format can be tested without a 63 GiB model.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Sequence

from tools.model_source import (
    TEXT_PREFIX,
    VISION_PREFIX,
    expected_model_tensor_names,
)


FORMAT_VERSION = 4
MAGIC = b"GEWBF16\0"
HEADER_BYTES = 4096
ENTRY_BYTES = 72
ALIGNMENT = 4096
SCALAR_TYPE_BF16_LE = 1
LAYOUT_C_ORDER = 1
TARGET_SM120A = 1
LM_HEAD_LOGICAL_ID = 1236
EMBED_PHYSICAL_ID = 0
TEXT_TENSOR_COUNT = 832
VISION_TENSOR_COUNT = 356
TARGET_TENSOR_COUNT = TEXT_TENSOR_COUNT + VISION_TENSOR_COUNT
ASSISTANT_TENSOR_COUNT = 48
ASSISTANT_EMBED_PHYSICAL_ID = 1188
ASSISTANT_LM_HEAD_LOGICAL_ID = 1237
PRODUCTION_TENSOR_COUNT = TEXT_TENSOR_COUNT
CHUNK_BYTES = 8 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024

HEADER_STRUCT = struct.Struct("<8s10I5Q2I32s40s32s32s32s32s32s")
ENTRY_STRUCT = struct.Struct("<HhHBBIIIQQ32s4x")
HEADER_HASH_OFFSET = 296
HEADER_USED_BYTES = HEADER_STRUCT.size


class ArtifactError(ValueError):
    """The source or converted artifact violates the fixed format."""


class Role(IntEnum):
    EMBED_TOKENS = 0
    INPUT_NORM = 1
    Q_PROJ = 2
    K_PROJ = 3
    V_PROJ = 4
    Q_NORM = 5
    K_NORM = 6
    O_PROJ = 7
    POST_ATTENTION_NORM = 8
    PRE_FEEDFORWARD_NORM = 9
    GATE_PROJ = 10
    UP_PROJ = 11
    DOWN_PROJ = 12
    POST_FEEDFORWARD_NORM = 13
    LAYER_SCALAR = 14
    FINAL_NORM = 15
    VISION_PATCH_PROJ = 16
    VISION_POSITION_EMBEDDING = 17
    VISION_INPUT_NORM = 18
    VISION_Q_PROJ = 19
    VISION_K_PROJ = 20
    VISION_V_PROJ = 21
    VISION_Q_NORM = 22
    VISION_K_NORM = 23
    VISION_O_PROJ = 24
    VISION_POST_ATTENTION_NORM = 25
    VISION_PRE_FEEDFORWARD_NORM = 26
    VISION_GATE_PROJ = 27
    VISION_UP_PROJ = 28
    VISION_DOWN_PROJ = 29
    VISION_POST_FEEDFORWARD_NORM = 30
    VISION_STD_BIAS = 31
    VISION_STD_SCALE = 32
    VISION_PROJECTION = 33
    ASSISTANT_EMBED_TOKENS = 34
    ASSISTANT_INPUT_NORM = 35
    ASSISTANT_Q_PROJ = 36
    ASSISTANT_Q_NORM = 37
    ASSISTANT_O_PROJ = 38
    ASSISTANT_POST_ATTENTION_NORM = 39
    ASSISTANT_PRE_FEEDFORWARD_NORM = 40
    ASSISTANT_GATE_PROJ = 41
    ASSISTANT_UP_PROJ = 42
    ASSISTANT_DOWN_PROJ = 43
    ASSISTANT_POST_FEEDFORWARD_NORM = 44
    ASSISTANT_LAYER_SCALAR = 45
    ASSISTANT_FINAL_NORM = 46
    ASSISTANT_PRE_PROJECTION = 47
    ASSISTANT_POST_PROJECTION = 48


ROLE_NAMES = {
    Role.EMBED_TOKENS: "embed_tokens",
    Role.INPUT_NORM: "input_norm",
    Role.Q_PROJ: "q_proj",
    Role.K_PROJ: "k_proj",
    Role.V_PROJ: "v_proj",
    Role.Q_NORM: "q_norm",
    Role.K_NORM: "k_norm",
    Role.O_PROJ: "o_proj",
    Role.POST_ATTENTION_NORM: "post_attention_norm",
    Role.PRE_FEEDFORWARD_NORM: "pre_feedforward_norm",
    Role.GATE_PROJ: "gate_proj",
    Role.UP_PROJ: "up_proj",
    Role.DOWN_PROJ: "down_proj",
    Role.POST_FEEDFORWARD_NORM: "post_feedforward_norm",
    Role.LAYER_SCALAR: "layer_scalar",
    Role.FINAL_NORM: "final_norm",
    Role.VISION_PATCH_PROJ: "vision_patch_proj",
    Role.VISION_POSITION_EMBEDDING: "vision_position_embedding",
    Role.VISION_INPUT_NORM: "vision_input_norm",
    Role.VISION_Q_PROJ: "vision_q_proj",
    Role.VISION_K_PROJ: "vision_k_proj",
    Role.VISION_V_PROJ: "vision_v_proj",
    Role.VISION_Q_NORM: "vision_q_norm",
    Role.VISION_K_NORM: "vision_k_norm",
    Role.VISION_O_PROJ: "vision_o_proj",
    Role.VISION_POST_ATTENTION_NORM: "vision_post_attention_norm",
    Role.VISION_PRE_FEEDFORWARD_NORM: "vision_pre_feedforward_norm",
    Role.VISION_GATE_PROJ: "vision_gate_proj",
    Role.VISION_UP_PROJ: "vision_up_proj",
    Role.VISION_DOWN_PROJ: "vision_down_proj",
    Role.VISION_POST_FEEDFORWARD_NORM: "vision_post_feedforward_norm",
    Role.VISION_STD_BIAS: "vision_std_bias",
    Role.VISION_STD_SCALE: "vision_std_scale",
    Role.VISION_PROJECTION: "vision_projection",
}
ROLE_NAMES.update({role: role.name.lower() for role in Role if role >= Role.ASSISTANT_EMBED_TOKENS})


@dataclass(frozen=True)
class TensorSpec:
    physical_id: int
    layer: int
    role: Role
    name: str
    source_name: str
    shape: tuple[int, ...]

    @property
    def source_component(self) -> str:
        return "assistant" if self.role >= Role.ASSISTANT_EMBED_TOKENS else "target"

    @property
    def byte_length(self) -> int:
        elements = 1
        for dimension in self.shape:
            elements *= dimension
        return elements * 2


@dataclass(frozen=True)
class SourceTensor:
    path: Path
    shard_name: str
    source_name: str
    offset: int
    byte_length: int


@dataclass(frozen=True)
class TensorEntry:
    physical_id: int
    layer: int
    role: Role
    rank: int
    dim0: int
    dim1: int
    dim2: int
    file_offset: int
    byte_length: int
    sha256: bytes

    @property
    def sha256_hex(self) -> str:
        return self.sha256.hex()


@dataclass(frozen=True)
class ArtifactHeader:
    physical_tensor_count: int
    logical_tensor_count: int
    data_offset: int
    logical_data_bytes: int
    payload_bytes: int
    file_bytes: int
    config_sha256: bytes
    source_index_sha256: bytes
    entry_table_sha256: bytes
    payload_sha256: bytes
    header_sha256: bytes


@dataclass(frozen=True)
class WrittenArtifact:
    path: Path
    header: ArtifactHeader
    entries: tuple[TensorEntry, ...]
    file_sha256: str


@dataclass(frozen=True)
class VerifiedArtifact:
    path: Path
    header: ArtifactHeader
    entries: tuple[TensorEntry, ...]
    file_sha256: str


@dataclass(frozen=True)
class SafeTensorRecord:
    dtype: str
    shape: tuple[int, ...]
    offset: int
    byte_length: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def _is_int(value: Any) -> bool:
    return type(value) is int


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    _require(_is_int(value) and value >= 0, "alignment input must be a non-negative integer")
    _require(_is_int(alignment) and alignment > 0, "alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(CHUNK_BYTES):
                digest.update(chunk)
    except OSError as error:
        raise ArtifactError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json_object(path: Path, *, canonical: bool = False) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ArtifactError(f"cannot read JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    if canonical:
        _require(raw == canonical_json_bytes(value), f"{path} is not canonical JSON")
    return value


def _text_layer_specs(layer: int, first_id: int) -> list[TensorSpec]:
    local = layer % 6 != 5
    root = f"layers.{layer}"
    attention = f"{root}.self_attn"
    mlp = f"{root}.mlp"
    head_dimension = 256 if local else 512
    q_rows = 8192 if local else 16384
    k_rows = 4096 if local else 2048
    o_columns = 8192 if local else 16384
    declarations: list[tuple[Role, str, tuple[int, ...]]] = [
        (Role.INPUT_NORM, f"{root}.input_layernorm.weight", (5376,)),
        (Role.Q_PROJ, f"{attention}.q_proj.weight", (q_rows, 5376)),
        (Role.K_PROJ, f"{attention}.k_proj.weight", (k_rows, 5376)),
    ]
    if local:
        declarations.append((Role.V_PROJ, f"{attention}.v_proj.weight", (4096, 5376)))
    declarations.extend(
        [
            (Role.Q_NORM, f"{attention}.q_norm.weight", (head_dimension,)),
            (Role.K_NORM, f"{attention}.k_norm.weight", (head_dimension,)),
            (Role.O_PROJ, f"{attention}.o_proj.weight", (5376, o_columns)),
            (Role.POST_ATTENTION_NORM, f"{root}.post_attention_layernorm.weight", (5376,)),
            (Role.PRE_FEEDFORWARD_NORM, f"{root}.pre_feedforward_layernorm.weight", (5376,)),
            (Role.GATE_PROJ, f"{mlp}.gate_proj.weight", (21504, 5376)),
            (Role.UP_PROJ, f"{mlp}.up_proj.weight", (21504, 5376)),
            (Role.DOWN_PROJ, f"{mlp}.down_proj.weight", (5376, 21504)),
            (Role.POST_FEEDFORWARD_NORM, f"{root}.post_feedforward_layernorm.weight", (5376,)),
            (Role.LAYER_SCALAR, f"{root}.layer_scalar", (1,)),
        ]
    )
    return [
        TensorSpec(first_id + index, layer, role, name, TEXT_PREFIX + name, shape)
        for index, (role, name, shape) in enumerate(declarations)
    ]


def _vision_layer_specs(layer: int, first_id: int) -> list[TensorSpec]:
    root = f"vision_tower.encoder.layers.{layer}"
    attention = f"{root}.self_attn"
    mlp = f"{root}.mlp"
    declarations: list[tuple[Role, str, tuple[int, ...]]] = [
        (Role.VISION_INPUT_NORM, f"{root}.input_layernorm.weight", (1152,)),
        (Role.VISION_Q_PROJ, f"{attention}.q_proj.linear.weight", (1152, 1152)),
        (Role.VISION_K_PROJ, f"{attention}.k_proj.linear.weight", (1152, 1152)),
        (Role.VISION_V_PROJ, f"{attention}.v_proj.linear.weight", (1152, 1152)),
        (Role.VISION_Q_NORM, f"{attention}.q_norm.weight", (72,)),
        (Role.VISION_K_NORM, f"{attention}.k_norm.weight", (72,)),
        (Role.VISION_O_PROJ, f"{attention}.o_proj.linear.weight", (1152, 1152)),
        (
            Role.VISION_POST_ATTENTION_NORM,
            f"{root}.post_attention_layernorm.weight",
            (1152,),
        ),
        (
            Role.VISION_PRE_FEEDFORWARD_NORM,
            f"{root}.pre_feedforward_layernorm.weight",
            (1152,),
        ),
        (Role.VISION_GATE_PROJ, f"{mlp}.gate_proj.linear.weight", (4304, 1152)),
        (Role.VISION_UP_PROJ, f"{mlp}.up_proj.linear.weight", (4304, 1152)),
        (Role.VISION_DOWN_PROJ, f"{mlp}.down_proj.linear.weight", (1152, 4304)),
        (
            Role.VISION_POST_FEEDFORWARD_NORM,
            f"{root}.post_feedforward_layernorm.weight",
            (1152,),
        ),
    ]
    return [
        TensorSpec(first_id + index, layer, role, name, f"model.{name}", shape)
        for index, (role, name, shape) in enumerate(declarations)
    ]


def assistant_tensor_specs() -> tuple[TensorSpec, ...]:
    declarations = [(Role.ASSISTANT_EMBED_TOKENS, -1, "model.embed_tokens.weight", (262144, 1024))]
    for layer in range(4):
        root = f"model.layers.{layer}"
        head = 512 if layer == 3 else 256
        q_width = 32 * head
        declarations.extend(
            (role, layer, f"{root}.{suffix}", shape)
            for role, suffix, shape in (
                (Role.ASSISTANT_INPUT_NORM, "input_layernorm.weight", (1024,)),
                (Role.ASSISTANT_Q_PROJ, "self_attn.q_proj.weight", (q_width, 1024)),
                (Role.ASSISTANT_Q_NORM, "self_attn.q_norm.weight", (head,)),
                (Role.ASSISTANT_O_PROJ, "self_attn.o_proj.weight", (1024, q_width)),
                (Role.ASSISTANT_POST_ATTENTION_NORM, "post_attention_layernorm.weight", (1024,)),
                (Role.ASSISTANT_PRE_FEEDFORWARD_NORM, "pre_feedforward_layernorm.weight", (1024,)),
                (Role.ASSISTANT_GATE_PROJ, "mlp.gate_proj.weight", (8192, 1024)),
                (Role.ASSISTANT_UP_PROJ, "mlp.up_proj.weight", (8192, 1024)),
                (Role.ASSISTANT_DOWN_PROJ, "mlp.down_proj.weight", (1024, 8192)),
                (Role.ASSISTANT_POST_FEEDFORWARD_NORM, "post_feedforward_layernorm.weight", (1024,)),
                (Role.ASSISTANT_LAYER_SCALAR, "layer_scalar", (1,)),
            )
        )
    declarations.extend((
        (Role.ASSISTANT_FINAL_NORM, -1, "model.norm.weight", (1024,)),
        (Role.ASSISTANT_PRE_PROJECTION, -1, "pre_projection.weight", (1024, 10752)),
        (Role.ASSISTANT_POST_PROJECTION, -1, "post_projection.weight", (5376, 1024)),
    ))
    return tuple(
        TensorSpec(ASSISTANT_EMBED_PHYSICAL_ID + index, layer, role,
                   f"assistant.{name.removeprefix('model.')}", name, shape)
        for index, (role, layer, name, shape) in enumerate(declarations)
    )


def all_tensor_specs() -> tuple[TensorSpec, ...]:
    specs = [
        TensorSpec(
            physical_id=0,
            layer=-1,
            role=Role.EMBED_TOKENS,
            name="embed_tokens.weight",
            source_name=f"{TEXT_PREFIX}embed_tokens.weight",
            shape=(262144, 5376),
        )
    ]
    for layer in range(60):
        specs.extend(_text_layer_specs(layer, len(specs)))
    specs.append(
        TensorSpec(
            physical_id=len(specs),
            layer=-1,
            role=Role.FINAL_NORM,
            name="norm.weight",
            source_name=f"{TEXT_PREFIX}norm.weight",
            shape=(5376,),
        )
    )
    _require(len(specs) == TEXT_TENSOR_COUNT, "internal text tensor count is not 832")
    _require(specs[-1].physical_id == 831, "final norm must be physical tensor 831")
    specs.extend(
        [
            TensorSpec(
                physical_id=len(specs),
                layer=-1,
                role=Role.VISION_PATCH_PROJ,
                name="vision_tower.patch_embedder.input_proj.weight",
                source_name=f"{VISION_PREFIX}patch_embedder.input_proj.weight",
                shape=(1152, 768),
            ),
            TensorSpec(
                physical_id=len(specs) + 1,
                layer=-1,
                role=Role.VISION_POSITION_EMBEDDING,
                name="vision_tower.patch_embedder.position_embedding_table",
                source_name=f"{VISION_PREFIX}patch_embedder.position_embedding_table",
                shape=(2, 10240, 1152),
            ),
        ]
    )
    for layer in range(27):
        specs.extend(_vision_layer_specs(layer, len(specs)))
    specs.extend(
        [
            TensorSpec(
                physical_id=len(specs),
                layer=-1,
                role=Role.VISION_STD_BIAS,
                name="vision_tower.std_bias",
                source_name=f"{VISION_PREFIX}std_bias",
                shape=(1152,),
            ),
            TensorSpec(
                physical_id=len(specs) + 1,
                layer=-1,
                role=Role.VISION_STD_SCALE,
                name="vision_tower.std_scale",
                source_name=f"{VISION_PREFIX}std_scale",
                shape=(1152,),
            ),
            TensorSpec(
                physical_id=len(specs) + 2,
                layer=-1,
                role=Role.VISION_PROJECTION,
                name="embed_vision.embedding_projection.weight",
                source_name="model.embed_vision.embedding_projection.weight",
                shape=(5376, 1152),
            ),
        ]
    )
    _require(len(specs) == TARGET_TENSOR_COUNT, "internal target tensor count is not 1188")
    _require(specs[-1].physical_id == 1187, "vision projection must be physical tensor 1187")
    _require(
        frozenset(spec.source_name for spec in specs) == expected_model_tensor_names(),
        "BF16 tensor table disagrees with the oracle source inventory",
    )
    specs.extend(assistant_tensor_specs())
    _require(len(specs) == TARGET_TENSOR_COUNT + ASSISTANT_TENSOR_COUNT, "internal BF16 tensor count is not 1236")
    return tuple(specs)


def expected_tensor_specs() -> tuple[TensorSpec, ...]:
    return all_tensor_specs()[:TEXT_TENSOR_COUNT]


def vision_tensor_specs() -> tuple[TensorSpec, ...]:
    return all_tensor_specs()[TEXT_TENSOR_COUNT:TARGET_TENSOR_COUNT]


def validate_specs(specs: Sequence[TensorSpec]) -> None:
    _require(bool(specs), "tensor spec table is empty")
    for expected_id, spec in enumerate(specs):
        _require(spec.physical_id == expected_id, f"tensor spec id {spec.physical_id} is out of order")
        _require(-1 <= spec.layer < 60, f"tensor {expected_id} has invalid layer")
        _require(1 <= len(spec.shape) <= 3, f"tensor {expected_id} has unsupported rank")
        for dimension in spec.shape:
            _require(
                _is_int(dimension) and 0 < dimension <= 0xFFFFFFFF,
                f"tensor {expected_id} has invalid dimension",
            )
        _require(spec.byte_length <= 0xFFFFFFFFFFFFFFFF, f"tensor {expected_id} is too large")


def data_offset_for_count(tensor_count: int) -> int:
    _require(_is_int(tensor_count) and tensor_count > 0, "tensor count must be positive")
    return align_up(HEADER_BYTES + tensor_count * ENTRY_BYTES)


def _fixed_ascii(value: str, width: int, label: str, *, pad: bool) -> bytes:
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ArtifactError(f"{label} is not ASCII") from error
    if pad:
        _require(len(encoded) < width, f"{label} does not fit its field")
        return encoded + b"\0" * (width - len(encoded))
    _require(len(encoded) == width, f"{label} must be exactly {width} bytes")
    return encoded


def _hash_bytes(value: str | bytes, label: str) -> bytes:
    if isinstance(value, bytes):
        _require(len(value) == 32, f"{label} must contain 32 bytes")
        return value
    _require(isinstance(value, str) and len(value) == 64, f"{label} must be lowercase SHA-256")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise ArtifactError(f"{label} must be lowercase SHA-256") from error
    _require(value == value.lower() and decoded.hex() == value, f"{label} must be lowercase SHA-256")
    return decoded


def encode_entry(entry: TensorEntry) -> bytes:
    return ENTRY_STRUCT.pack(
        entry.physical_id,
        entry.layer,
        int(entry.role),
        entry.rank,
        0,
        entry.dim0,
        entry.dim1,
        entry.dim2,
        entry.file_offset,
        entry.byte_length,
        entry.sha256,
    )


def decode_entry(raw: bytes, index: int) -> TensorEntry:
    _require(len(raw) == ENTRY_BYTES, f"entry {index} is truncated")
    try:
        physical_id, layer, role_value, rank, reserved, dim0, dim1, dim2, offset, length, digest = ENTRY_STRUCT.unpack(raw)
        role = Role(role_value)
    except (struct.error, ValueError) as error:
        raise ArtifactError(f"entry {index} is malformed") from error
    _require(reserved == 0, f"entry {index} reserved byte is nonzero")
    _require(raw[-4:] == b"\0" * 4, f"entry {index} reserved tail is nonzero")
    return TensorEntry(physical_id, layer, role, rank, dim0, dim1, dim2, offset, length, digest)


def _pack_header(
    *,
    tensor_count: int,
    data_offset: int,
    logical_bytes: int,
    payload_bytes: int,
    file_bytes: int,
    config_sha256: bytes,
    index_sha256: bytes,
    table_sha256: bytes,
    payload_sha256: bytes,
    header_sha256: bytes,
) -> bytes:
    prefix = HEADER_STRUCT.pack(
        MAGIC,
        FORMAT_VERSION,
        HEADER_BYTES,
        ENTRY_BYTES,
        tensor_count,
        tensor_count + 1,
        1,
        ALIGNMENT,
        SCALAR_TYPE_BF16_LE,
        LAYOUT_C_ORDER,
        TARGET_SM120A,
        HEADER_BYTES,
        data_offset,
        logical_bytes,
        payload_bytes,
        file_bytes,
        LM_HEAD_LOGICAL_ID,
        EMBED_PHYSICAL_ID,
        b"\0" * 32,
        b"\0" * 40,
        config_sha256,
        index_sha256,
        table_sha256,
        payload_sha256,
        header_sha256,
    )
    _require(len(prefix) == HEADER_USED_BYTES, "internal header layout is wrong")
    return prefix + b"\0" * (HEADER_BYTES - len(prefix))


def encode_header(
    *,
    tensor_count: int,
    data_offset: int,
    logical_bytes: int,
    payload_bytes: int,
    file_bytes: int,
    config_sha256: str | bytes,
    index_sha256: str | bytes,
    table_sha256: bytes,
    payload_sha256: bytes,
) -> bytes:
    config_hash = _hash_bytes(config_sha256, "config SHA-256")
    index_hash = _hash_bytes(index_sha256, "source index SHA-256")
    zero_hash = b"\0" * 32
    header = _pack_header(
        tensor_count=tensor_count,
        data_offset=data_offset,
        logical_bytes=logical_bytes,
        payload_bytes=payload_bytes,
        file_bytes=file_bytes,
        config_sha256=config_hash,
        index_sha256=index_hash,
        table_sha256=table_sha256,
        payload_sha256=payload_sha256,
        header_sha256=zero_hash,
    )
    header_hash = hashlib.sha256(header).digest()
    return _pack_header(
        tensor_count=tensor_count,
        data_offset=data_offset,
        logical_bytes=logical_bytes,
        payload_bytes=payload_bytes,
        file_bytes=file_bytes,
        config_sha256=config_hash,
        index_sha256=index_hash,
        table_sha256=table_sha256,
        payload_sha256=payload_sha256,
        header_sha256=header_hash,
    )


def decode_header(raw: bytes) -> tuple[ArtifactHeader, tuple[Any, ...]]:
    _require(len(raw) == HEADER_BYTES, "artifact header is truncated")
    try:
        fields = HEADER_STRUCT.unpack(raw[:HEADER_USED_BYTES])
    except struct.error as error:
        raise ArtifactError("artifact header is malformed") from error
    _require(raw[HEADER_USED_BYTES:] == b"\0" * (HEADER_BYTES - HEADER_USED_BYTES), "header reserved bytes are nonzero")
    return (
        ArtifactHeader(
            physical_tensor_count=fields[4],
            logical_tensor_count=fields[5],
            data_offset=fields[12],
            logical_data_bytes=fields[13],
            payload_bytes=fields[14],
            file_bytes=fields[15],
            config_sha256=fields[20],
            source_index_sha256=fields[21],
            entry_table_sha256=fields[22],
            payload_sha256=fields[23],
            header_sha256=fields[24],
        ),
        fields,
    )


def _read_exact(source: BinaryIO, length: int, label: str) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = source.read(min(remaining, CHUNK_BYTES))
        if not chunk:
            raise ArtifactError(f"unexpected EOF while reading {label}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _copy_source_range(
    source: BinaryIO,
    output: BinaryIO,
    offset: int,
    length: int,
    tensor_digest: Any,
    payload_digest: Any,
) -> None:
    source.seek(offset)
    remaining = length
    while remaining:
        chunk = source.read(min(remaining, CHUNK_BYTES))
        if not chunk:
            raise ArtifactError("source tensor ended before its declared byte length")
        output.write(chunk)
        tensor_digest.update(chunk)
        payload_digest.update(chunk)
        remaining -= len(chunk)


def write_artifact_partial(
    path: Path,
    specs: Sequence[TensorSpec],
    sources: Sequence[SourceTensor],
    *,
    config_sha256: str | bytes,
    index_sha256: str | bytes,
) -> WrittenArtifact:
    """Write a complete artifact to an exclusively-created partial path."""

    validate_specs(specs)
    _require(len(specs) == len(sources), "source tensor count does not match spec table")
    data_offset = data_offset_for_count(len(specs))
    entries: list[TensorEntry] = []
    payload_digest = hashlib.sha256()
    logical_bytes = 0
    payload_bytes = 0
    handles: dict[Path, BinaryIO] = {}
    try:
        with path.open("xb") as output:
            output.write(b"\0" * data_offset)
            cursor = data_offset
            for spec, declaration in zip(specs, sources, strict=True):
                _require(declaration.source_name == spec.source_name, f"source name mismatch for tensor {spec.physical_id}")
                _require(declaration.byte_length == spec.byte_length, f"source byte length mismatch for {spec.source_name}")
                _require(cursor % ALIGNMENT == 0, "internal tensor offset is not aligned")
                handle = handles.get(declaration.path)
                if handle is None:
                    try:
                        handle = declaration.path.open("rb")
                    except OSError as error:
                        raise ArtifactError(f"cannot open source shard {declaration.path}: {error}") from error
                    handles[declaration.path] = handle
                tensor_digest = hashlib.sha256()
                output.seek(cursor)
                _copy_source_range(handle, output, declaration.offset, declaration.byte_length, tensor_digest, payload_digest)
                slot_bytes = align_up(declaration.byte_length)
                padding = slot_bytes - declaration.byte_length
                if padding:
                    zeros = b"\0" * padding
                    output.write(zeros)
                    payload_digest.update(zeros)
                dim0 = spec.shape[0]
                dim1 = spec.shape[1] if len(spec.shape) >= 2 else 0
                dim2 = spec.shape[2] if len(spec.shape) == 3 else 0
                entries.append(
                    TensorEntry(
                        physical_id=spec.physical_id,
                        layer=spec.layer,
                        role=spec.role,
                        rank=len(spec.shape),
                        dim0=dim0,
                        dim1=dim1,
                        dim2=dim2,
                        file_offset=cursor,
                        byte_length=declaration.byte_length,
                        sha256=tensor_digest.digest(),
                    )
                )
                cursor += slot_bytes
                logical_bytes += declaration.byte_length
                payload_bytes += slot_bytes

            entry_bytes = b"".join(encode_entry(entry) for entry in entries)
            table_hash = hashlib.sha256(entry_bytes).digest()
            file_bytes = data_offset + payload_bytes
            _require(cursor == file_bytes, "internal artifact size mismatch")
            header_bytes = encode_header(
                tensor_count=len(specs),
                data_offset=data_offset,
                logical_bytes=logical_bytes,
                payload_bytes=payload_bytes,
                file_bytes=file_bytes,
                config_sha256=config_sha256,
                index_sha256=index_sha256,
                table_sha256=table_hash,
                payload_sha256=payload_digest.digest(),
            )
            output.seek(0)
            output.write(header_bytes)
            output.write(entry_bytes)
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError as error:
        raise ArtifactError(f"partial output already exists: {path}") from error
    except OSError as error:
        raise ArtifactError(f"cannot write artifact {path}: {error}") from error
    finally:
        for handle in handles.values():
            handle.close()

    decoded_header, _ = decode_header(header_bytes)
    return WrittenArtifact(path, decoded_header, tuple(entries), sha256_file(path))


def _validate_header_fields(
    raw: bytes,
    header: ArtifactHeader,
    fields: tuple[Any, ...],
    specs: Sequence[TensorSpec],
    expected_config_sha256: str | bytes | None = None,
    expected_index_sha256: str | bytes | None = None,
) -> None:
    tensor_count = len(specs)
    expected_data_offset = data_offset_for_count(tensor_count)
    fixed_expectations = {
        "magic": (fields[0], MAGIC),
        "format version": (fields[1], FORMAT_VERSION),
        "header bytes": (fields[2], HEADER_BYTES),
        "entry bytes": (fields[3], ENTRY_BYTES),
        "physical tensor count": (fields[4], tensor_count),
        "logical tensor count": (fields[5], tensor_count + 1),
        "alias count": (fields[6], 1),
        "alignment": (fields[7], ALIGNMENT),
        "scalar type": (fields[8], SCALAR_TYPE_BF16_LE),
        "layout": (fields[9], LAYOUT_C_ORDER),
        "target": (fields[10], TARGET_SM120A),
        "entries offset": (fields[11], HEADER_BYTES),
        "data offset": (fields[12], expected_data_offset),
        "lm_head logical id": (fields[16], LM_HEAD_LOGICAL_ID),
        "lm_head target id": (fields[17], EMBED_PHYSICAL_ID),
    }
    if expected_config_sha256 is not None:
        fixed_expectations["config SHA-256"] = (fields[20], _hash_bytes(expected_config_sha256, "config SHA-256"))
    if expected_index_sha256 is not None:
        fixed_expectations["source index SHA-256"] = (fields[21], _hash_bytes(expected_index_sha256, "source index SHA-256"))
    for label, (actual, expected) in fixed_expectations.items():
        _require(actual == expected, f"wrong {label}")

    zeroed = bytearray(raw)
    zeroed[HEADER_HASH_OFFSET : HEADER_HASH_OFFSET + 32] = b"\0" * 32
    _require(hashlib.sha256(zeroed).digest() == header.header_sha256, "header SHA-256 mismatch")


def verify_artifact_file(
    path: Path,
    specs: Sequence[TensorSpec],
    *,
    expected_config_sha256: str | bytes | None = None,
    expected_index_sha256: str | bytes | None = None,
) -> VerifiedArtifact:
    """Independently verify all metadata, tensor bytes, and zero padding."""

    validate_specs(specs)
    try:
        file_size = path.stat().st_size
        with path.open("rb") as source:
            header_raw = _read_exact(source, HEADER_BYTES, "artifact header")
            header, fields = decode_header(header_raw)
            _validate_header_fields(
                header_raw,
                header,
                fields,
                specs,
                expected_config_sha256,
                expected_index_sha256,
            )
            _require(header.file_bytes == file_size, "artifact file size mismatch")
            table_size = len(specs) * ENTRY_BYTES
            table_raw = _read_exact(source, table_size, "tensor table")
            _require(hashlib.sha256(table_raw).digest() == header.entry_table_sha256, "entry-table SHA-256 mismatch")
            index_padding = header.data_offset - HEADER_BYTES - table_size
            _require(index_padding >= 0, "data offset precedes tensor table")
            if index_padding:
                _require(
                    _read_exact(source, index_padding, "index padding") == b"\0" * index_padding,
                    "index padding is nonzero",
                )

            entries = tuple(
                decode_entry(table_raw[index * ENTRY_BYTES : (index + 1) * ENTRY_BYTES], index)
                for index in range(len(specs))
            )
            cursor = header.data_offset
            logical_bytes = 0
            payload_bytes = 0
            for spec, entry in zip(specs, entries, strict=True):
                expected_dim1 = spec.shape[1] if len(spec.shape) >= 2 else 0
                expected_dim2 = spec.shape[2] if len(spec.shape) == 3 else 0
                checks = {
                    "id": entry.physical_id == spec.physical_id,
                    "layer": entry.layer == spec.layer,
                    "role": entry.role == spec.role,
                    "rank": entry.rank == len(spec.shape),
                    "dim0": entry.dim0 == spec.shape[0],
                    "dim1": entry.dim1 == expected_dim1,
                    "dim2": entry.dim2 == expected_dim2,
                    "offset": entry.file_offset == cursor,
                    "byte length": entry.byte_length == spec.byte_length,
                }
                for label, valid in checks.items():
                    _require(valid, f"tensor {spec.physical_id} has wrong {label}")
                slot_bytes = align_up(entry.byte_length)
                cursor += slot_bytes
                logical_bytes += entry.byte_length
                payload_bytes += slot_bytes

            _require(header.logical_data_bytes == logical_bytes, "logical data byte count mismatch")
            _require(header.payload_bytes == payload_bytes, "payload byte count mismatch")
            _require(header.file_bytes == header.data_offset + payload_bytes, "declared file size is inconsistent")
            _require(cursor == header.file_bytes, "tensor table does not cover the payload")

            full_digest = hashlib.sha256()
            full_digest.update(header_raw)
            full_digest.update(table_raw)
            if index_padding:
                full_digest.update(b"\0" * index_padding)
            payload_digest = hashlib.sha256()
            for spec, entry in zip(specs, entries, strict=True):
                tensor_digest = hashlib.sha256()
                remaining = entry.byte_length
                while remaining:
                    chunk = source.read(min(remaining, CHUNK_BYTES))
                    if not chunk:
                        raise ArtifactError(f"unexpected EOF in tensor {spec.physical_id}")
                    tensor_digest.update(chunk)
                    payload_digest.update(chunk)
                    full_digest.update(chunk)
                    remaining -= len(chunk)
                _require(tensor_digest.digest() == entry.sha256, f"tensor {spec.physical_id} SHA-256 mismatch")
                padding = align_up(entry.byte_length) - entry.byte_length
                while padding:
                    chunk = source.read(min(padding, CHUNK_BYTES))
                    if not chunk:
                        raise ArtifactError(f"unexpected EOF in padding after tensor {spec.physical_id}")
                    _require(not any(chunk), f"padding after tensor {spec.physical_id} is nonzero")
                    payload_digest.update(chunk)
                    full_digest.update(chunk)
                    padding -= len(chunk)
            _require(source.read(1) == b"", "artifact has trailing bytes")
            _require(payload_digest.digest() == header.payload_sha256, "payload SHA-256 mismatch")
    except ArtifactError:
        raise
    except (OSError, OverflowError) as error:
        raise ArtifactError(f"cannot verify artifact {path}: {error}") from error

    return VerifiedArtifact(path, header, entries, full_digest.hexdigest())


def read_safetensors_header(path: Path) -> dict[str, SafeTensorRecord]:
    """Parse and validate a safetensors file without loading tensor data."""

    try:
        file_size = path.stat().st_size
        with path.open("rb") as source:
            length_raw = source.read(8)
            _require(len(length_raw) == 8, f"safetensors header is truncated: {path}")
            header_length = struct.unpack("<Q", length_raw)[0]
            _require(0 < header_length <= MAX_SAFETENSORS_HEADER_BYTES, f"invalid safetensors header length: {path}")
            _require(8 + header_length <= file_size, f"safetensors header exceeds file: {path}")
            header_raw = _read_exact(source, header_length, f"safetensors header {path}")
    except ArtifactError:
        raise
    except OSError as error:
        raise ArtifactError(f"cannot read safetensors file {path}: {error}") from error

    try:
        metadata = json.loads(
            header_raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ArtifactError(f"invalid safetensors JSON in {path}: {error}") from error
    _require(isinstance(metadata, dict), f"safetensors header is not an object: {path}")
    data_start = 8 + header_length
    records: dict[str, SafeTensorRecord] = {}
    spans: list[tuple[int, int, str]] = []
    for name, value in metadata.items():
        if name == "__metadata__":
            _require(isinstance(value, dict), f"invalid __metadata__ in {path}")
            continue
        _require(isinstance(name, str) and name, f"invalid tensor name in {path}")
        _require(isinstance(value, dict), f"invalid metadata for tensor {name}")
        _require(set(value) == {"dtype", "shape", "data_offsets"}, f"wrong metadata fields for tensor {name}")
        dtype = value.get("dtype")
        shape_value = value.get("shape")
        offsets = value.get("data_offsets")
        _require(isinstance(dtype, str), f"invalid dtype for tensor {name}")
        _require(isinstance(shape_value, list) and len(shape_value) <= 8, f"invalid rank for tensor {name}")
        _require(
            all(_is_int(dimension) and 0 <= dimension <= 0xFFFFFFFF for dimension in shape_value),
            f"invalid shape for tensor {name}",
        )
        _require(
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(_is_int(offset) and offset >= 0 for offset in offsets),
            f"invalid data offsets for tensor {name}",
        )
        start, end = offsets
        _require(start <= end, f"reversed data offsets for tensor {name}")
        element_bytes = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "F8_E4M3": 1, "U8": 1}.get(dtype)
        if element_bytes is not None:
            expected_length = element_bytes
            for dimension in shape_value:
                expected_length *= dimension
            _require(end - start == expected_length, f"byte length mismatch for tensor {name}")
        _require(data_start + end <= file_size, f"tensor {name} exceeds source shard")
        records[name] = SafeTensorRecord(dtype, tuple(shape_value), data_start + start, end - start)
        spans.append((start, end, name))

    spans.sort()
    previous_end = 0
    for start, end, name in spans:
        _require(start == previous_end, f"safetensors data has a gap or overlap before {name}")
        previous_end = end
    _require(data_start + previous_end == file_size, f"safetensors data does not cover {path}")
    return records


def safe_basename(value: Any, label: str) -> str:
    _require(isinstance(value, str) and value, f"{label} must be a non-empty filename")
    path = Path(value)
    _require(not path.is_absolute() and path.name == value and value not in {".", ".."}, f"unsafe {label}: {value!r}")
    return value


def iter_file_chunks(source: BinaryIO, length: int) -> Iterable[bytes]:
    remaining = length
    while remaining:
        chunk = source.read(min(remaining, CHUNK_BYTES))
        if not chunk:
            raise ArtifactError("unexpected EOF")
        yield chunk
        remaining -= len(chunk)
