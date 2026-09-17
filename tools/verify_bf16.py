#!/usr/bin/env python3
"""Independently verify a Gemma 4 31B text-only BF16 v4 artifact and its source checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Any, BinaryIO, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.model_source import expected_model_tensor_names, load_model
from tools.serving_assets import (
    MANIFEST_SCHEMA_VERSION,
    validate_serving_assets,
    validate_serving_metadata,
)
from tools.bf16_artifact import (
    ALIGNMENT,
    CHUNK_BYTES,
    LM_HEAD_LOGICAL_ID,
    ArtifactError,
    TensorSpec,
    VerifiedArtifact,
    align_up,
    expected_tensor_specs,
    load_json_object,
    read_safetensors_header,
    safe_basename,
    sha256_file,
    verify_artifact_file,
)


FORMAT_NAME = "gemma4-31b-bf16-v4"
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")

TOP_LEVEL_KEYS = {"aliases", "artifact", "format", "layout", "model", "schema_version", "serving", "source", "tensors"}
ALIAS_KEYS = {"logical_id", "name", "target", "target_physical_id"}
ARTIFACT_KEYS = {"entry_table_sha256", "file", "file_bytes", "file_sha256", "header_sha256", "payload_sha256"}
LAYOUT_KEYS = {
    "alignment",
    "logical_data_bytes",
    "logical_tensor_count",
    "payload_bytes",
    "physical_tensor_count",
    "target",
}
MODEL_KEYS = {
    "config_sha256",
    "index_sha256",
    "repository",
    "revision",
}
SOURCE_KEYS = {"index", "shards"}
SOURCE_FILE_KEYS = {"byte_length", "file", "sha256"}
TENSOR_KEYS = {
    "byte_length",
    "dtype",
    "file_offset",
    "layer",
    "layout",
    "name",
    "physical_id",
    "role",
    "sha256",
    "shape",
    "slot_length",
    "source_name",
    "source_component",
    "source_offset",
    "source_shard",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def _require_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    actual = set(value)
    _require(actual == keys, f"{label} keys differ: missing={sorted(keys - actual)} extra={sorted(actual - keys)}")
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and HEX_SHA256.fullmatch(value) is not None


def _load_source_index(model: dict[str, str]) -> tuple[Path, dict[str, str]]:
    snapshot = Path(model["snapshot"])
    index_path = snapshot / "model.safetensors.index.json"
    index = load_json_object(index_path)
    weight_map = index.get("weight_map")
    _require(isinstance(weight_map, dict), "source index has no weight_map object")
    _require(all(isinstance(name, str) and isinstance(value, str) for name, value in weight_map.items()), "invalid source weight_map")
    return index_path, weight_map


def validate_manifest(
    path: Path,
    artifact_path: Path,
    verified: VerifiedArtifact,
    specs: Sequence[TensorSpec],
    model: dict[str, str],
) -> dict[str, Any]:
    _require(not path.is_symlink(), f"refusing manifest symlink: {path}")
    manifest = load_json_object(path, canonical=True)
    validate_manifest_data(manifest, artifact_path, verified, specs, model)
    validate_serving_assets(artifact_path.parent, manifest["serving"])
    return manifest


def validate_manifest_data(
    manifest: dict[str, Any],
    artifact_path: Path,
    verified: VerifiedArtifact,
    specs: Sequence[TensorSpec],
    model: dict[str, str],
) -> dict[str, Any]:
    _require_keys(manifest, TOP_LEVEL_KEYS, "manifest")
    _require(type(manifest["schema_version"]) is int and manifest["schema_version"] == MANIFEST_SCHEMA_VERSION,
             "unsupported manifest schema")
    _require(manifest["format"] == FORMAT_NAME, "wrong manifest format")
    validate_serving_metadata(manifest["serving"])

    artifact = _require_keys(manifest["artifact"], ARTIFACT_KEYS, "manifest artifact")
    _require(safe_basename(artifact["file"], "artifact filename") == artifact_path.name, "manifest artifact filename mismatch")
    expected_artifact = {
        "entry_table_sha256": verified.header.entry_table_sha256.hex(),
        "file": artifact_path.name,
        "file_bytes": verified.header.file_bytes,
        "file_sha256": verified.file_sha256,
        "header_sha256": verified.header.header_sha256.hex(),
        "payload_sha256": verified.header.payload_sha256.hex(),
    }
    _require(artifact == expected_artifact, "manifest artifact metadata mismatch")

    layout = _require_keys(manifest["layout"], LAYOUT_KEYS, "manifest layout")
    _require(
        layout
        == {
            "alignment": ALIGNMENT,
            "logical_data_bytes": verified.header.logical_data_bytes,
            "logical_tensor_count": verified.header.logical_tensor_count,
            "payload_bytes": verified.header.payload_bytes,
            "physical_tensor_count": verified.header.physical_tensor_count,
            "target": "sm_120a",
        },
        "manifest layout mismatch",
    )

    manifest_model = _require_keys(manifest["model"], MODEL_KEYS, "manifest model")
    _require(
        manifest_model
        == {
            "config_sha256": model["config_sha256"],
            "index_sha256": model["index_sha256"],
            "repository": model["repository"],
            "revision": model["revision"],
        },
        "manifest model mismatch",
    )

    aliases = manifest["aliases"]
    _require(isinstance(aliases, list) and len(aliases) == 1, "manifest must declare exactly one alias")
    alias = _require_keys(aliases[0], ALIAS_KEYS, "manifest alias")
    _require(
        alias
        == {
            "logical_id": LM_HEAD_LOGICAL_ID,
            "name": "lm_head.weight",
            "target": "embed_tokens.weight",
            "target_physical_id": 0,
        },
        "manifest lm_head alias mismatch",
    )
    source = _require_keys(manifest["source"], SOURCE_KEYS, "manifest source")
    source_index = _require_keys(source["index"], SOURCE_FILE_KEYS, "manifest source index")
    index_path, weight_map = _load_source_index(model)
    _require(
        source_index
        == {
            "byte_length": index_path.stat().st_size,
            "file": "model.safetensors.index.json",
            "sha256": model["index_sha256"],
        },
        "manifest source index mismatch",
    )
    shard_values = source["shards"]
    _require(isinstance(shard_values, list) and len(shard_values) == len(set(weight_map.values())), "manifest source shard count mismatch")
    declared_shards: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(shard_values):
        declaration = _require_keys(value, SOURCE_FILE_KEYS, f"manifest source shard {index}")
        name = safe_basename(declaration["file"], "source shard filename")
        _require(name not in declared_shards, f"duplicate source shard {name}")
        _require(name in weight_map.values(), f"unknown source shard {name}")
        _require(type(declaration["byte_length"]) is int and declaration["byte_length"] > 0, f"invalid source shard size: {name}")
        _require(_is_sha256(declaration["sha256"]), f"invalid source shard hash: {name}")
        declared_shards[name] = declaration
    _require(list(declared_shards) == sorted(set(weight_map.values())), "source shards are not in canonical order")

    tensor_values = manifest["tensors"]
    _require(isinstance(tensor_values, list) and len(tensor_values) == len(specs), "manifest tensor count mismatch")
    for spec, entry, value in zip(specs, verified.entries, tensor_values, strict=True):
        tensor = _require_keys(value, TENSOR_KEYS, f"manifest tensor {spec.physical_id}")
        source_shard = weight_map.get(spec.source_name)
        _require(isinstance(source_shard, str), f"source index does not map {spec.source_name}")
        source_shard = safe_basename(source_shard, f"shard for {spec.source_name}")
        expected_values = {
            "byte_length": entry.byte_length,
            "dtype": "BF16",
            "file_offset": entry.file_offset,
            "layer": None if spec.layer == -1 else spec.layer,
            "layout": "C_ORDER",
            "name": spec.name,
            "physical_id": spec.physical_id,
            "role": spec.role.name.lower(),
            "sha256": entry.sha256_hex,
            "shape": list(spec.shape),
            "slot_length": align_up(entry.byte_length),
            "source_name": spec.source_name,
            "source_component": spec.source_component,
            "source_shard": source_shard,
        }
        for key, expected in expected_values.items():
            _require(tensor[key] == expected, f"manifest tensor {spec.physical_id} has wrong {key}")
        _require(type(tensor["source_offset"]) is int and tensor["source_offset"] >= 0, f"invalid source offset for tensor {spec.physical_id}")
        _require(_is_sha256(tensor["sha256"]), f"invalid tensor SHA-256 for tensor {spec.physical_id}")
    return manifest


def _hash_range(source: BinaryIO, offset: int, length: int, label: str) -> str:
    source.seek(offset)
    digest = hashlib.sha256()
    remaining = length
    while remaining:
        chunk = source.read(min(remaining, CHUNK_BYTES))
        if not chunk:
            raise ArtifactError(f"unexpected EOF while reading {label}")
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def verify_source(
    manifest: dict[str, Any],
    specs: Sequence[TensorSpec],
    model: dict[str, str],
) -> None:
    snapshot = Path(model["snapshot"])
    index_path, weight_map = _load_source_index(model)
    _require(sha256_file(index_path) == model["index_sha256"], "source index SHA-256 mismatch")
    _require(sha256_file(snapshot / "config.json") == model["config_sha256"], "target config SHA-256 mismatch")
    records_by_shard: dict[str, dict[str, Any]] = {}
    handles: dict[str, BinaryIO] = {}
    try:
        for shard in manifest["source"]["shards"]:
            name, expected_bytes, expected_sha256 = shard["file"], shard["byte_length"], shard["sha256"]
            path = snapshot / name
            _require(path.is_file(), f"source shard does not exist: {path}")
            _require(path.stat().st_size == expected_bytes, f"source shard size mismatch: {name}")
            _require(sha256_file(path) == expected_sha256, f"source shard SHA-256 mismatch: {name}")
            records_by_shard[name] = read_safetensors_header(path)
            handles[name] = path.open("rb")

        _require({s.source_name for s in specs} <= set(weight_map), "target source index tensor inventory mismatch")
        occurrences: dict[str, list[str]] = {}
        for name, records in records_by_shard.items():
            for tensor_name in records:
                occurrences.setdefault(tensor_name, []).append(name)
        _require({s.source_name for s in specs} <= set(occurrences), "source tensor inventory mismatch")

        tensor_values = manifest["tensors"]
        for spec, tensor in zip(specs, tensor_values, strict=True):
            shard_name = weight_map.get(spec.source_name)
            _require(isinstance(shard_name, str), f"source index does not map {spec.source_name}")
            _require(occurrences.get(spec.source_name) == [shard_name],
                     f"source tensor occurrence mismatch for {spec.source_name}")
            record = records_by_shard[shard_name].get(spec.source_name)
            _require(record is not None, f"source shard omits {spec.source_name}")
            _require(record.dtype == "BF16", f"source tensor is not BF16: {spec.source_name}")
            _require(record.shape == spec.shape, f"source tensor shape mismatch: {spec.source_name}")
            _require(record.offset == tensor["source_offset"], f"source tensor offset mismatch: {spec.source_name}")
            _require(record.byte_length == spec.byte_length, f"source tensor length mismatch: {spec.source_name}")
            _require(
                _hash_range(handles[shard_name], record.offset, record.byte_length, spec.source_name) == tensor["sha256"],
                f"source tensor bytes mismatch: {spec.source_name}",
            )
    except OSError as error:
        raise ArtifactError(f"cannot verify source checkpoint: {error}") from error
    finally:
        for handle in handles.values():
            handle.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--snapshot", type=Path, required=True, help="local source BF16 safetensors")
    parser.add_argument("--manifest", type=Path, help="canonical JSON mirror to cross-check")
    parser.add_argument("--check-source", action="store_true", help="also hash and compare the original safetensor shards")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _require(not args.artifact.is_symlink(), f"refusing artifact symlink: {args.artifact}")
        model = load_model(args.snapshot)
        specs = expected_tensor_specs()
        verified = verify_artifact_file(
            args.artifact,
            specs,
            expected_config_sha256=model["config_sha256"],
            expected_index_sha256=model["index_sha256"],
        )
        manifest_path = args.manifest
        if args.check_source and manifest_path is None:
            manifest_path = args.artifact.parent / "manifest.json"
        manifest = None
        if manifest_path is not None:
            manifest = validate_manifest(manifest_path, args.artifact, verified, specs, model)
        if args.check_source:
            _require(manifest is not None, "--check-source requires a manifest")
            verify_source(manifest, specs, model)
        print(f"verified: {args.artifact}")
        print(f"physical tensors: {len(verified.entries)}")
        print(f"logical bytes: {verified.header.logical_data_bytes}")
        print(f"payload bytes: {verified.header.payload_bytes}")
        print(f"file sha256: {verified.file_sha256}")
    except (ArtifactError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
