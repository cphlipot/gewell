from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

MODEL_REVISION = "fixture-revision"
from tools.bf16_artifact import (
    ALIGNMENT,
    ENTRY_BYTES,
    HEADER_BYTES,
    HEADER_HASH_OFFSET,
    HEADER_USED_BYTES,
    LM_HEAD_LOGICAL_ID,
    ArtifactError,
    Role,
    SourceTensor,
    TensorSpec,
    align_up,
    canonical_json_bytes,
    data_offset_for_count,
    expected_tensor_specs,
    all_tensor_specs,
    load_json_object,
    read_safetensors_header,
    safe_basename,
    sha256_file,
    verify_artifact_file,
    write_artifact_partial,
)
from tools.verify_bf16 import validate_manifest, verify_source
from tools.serving_assets import REQUIRED_ASSETS, serving_metadata
from tools.safetensors_source import read_snapshot


SERVING_ASSET_BYTES = {name: f"fixture: {name}\n".encode() for name in REQUIRED_ASSETS}
SERVING_ASSET_SHA256 = {name: hashlib.sha256(data).hexdigest() for name, data in SERVING_ASSET_BYTES.items()}
SOURCE_CONFIG_BYTES = b'{"source":"fixture"}'
CONFIG_SHA256 = hashlib.sha256(SOURCE_CONFIG_BYTES).hexdigest()
INDEX_SHA256 = "2" * 64
TABLE_HASH_OFFSET = 232


def _tiny_specs() -> tuple[TensorSpec, ...]:
    return (
        TensorSpec(
            0,
            -1,
            Role.EMBED_TOKENS,
            "embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
            (1, 2, 3),
        ),
        TensorSpec(
            1,
            -1,
            Role.FINAL_NORM,
            "norm.weight",
            "model.language_model.norm.weight",
            (3,),
        ),

    )


def _write_safetensors(path: Path, tensors: list[tuple[str, tuple[int, ...], bytes]]) -> None:
    metadata: dict[str, object] = {}
    payload = bytearray()
    for name, shape, raw in tensors:
        start = len(payload)
        payload.extend(raw)
        metadata[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [start, len(payload)]}
    encoded = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    header = encoded + b" " * (align_up(len(encoded), 8) - len(encoded))
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def _tiny_sources(directory: Path, specs: tuple[TensorSpec, ...]) -> tuple[tuple[SourceTensor, ...], tuple[bytes, ...]]:
    matrix = struct.pack("<6H", 0x0000, 0x8000, 0x0001, 0x3F80, 0xC000, 0x7F7F)
    norm = struct.pack("<3H", 0x3F00, 0x3F80, 0x4000)
    shard_a = directory / "model-00001-of-00002.safetensors"
    shard_b = directory / "model-00002-of-00002.safetensors"
    _write_safetensors(shard_a, [(specs[0].source_name, specs[0].shape, matrix)])
    _write_safetensors(
        shard_b,
        [
            (specs[1].source_name, specs[1].shape, norm),
        ],
    )
    a_records = read_safetensors_header(shard_a)
    b_records = read_safetensors_header(shard_b)
    declarations = []
    for spec, shard, records in ((specs[0], shard_a, a_records), (specs[1], shard_b, b_records)):
        record = records[spec.source_name]
        declarations.append(
            SourceTensor(shard, shard.name, spec.source_name, record.offset, record.byte_length)
        )
    return tuple(declarations), (matrix, norm)


def _write_tiny_artifact(directory: Path, name: str = "weights.partial"):
    specs = _tiny_specs()
    sources, raw_tensors = _tiny_sources(directory, specs)
    path = directory / name
    written = write_artifact_partial(
        path,
        specs,
        sources,
        config_sha256=CONFIG_SHA256,
        index_sha256=INDEX_SHA256,
    )
    return path, specs, sources, raw_tensors, written


def _verify(path: Path, specs: tuple[TensorSpec, ...]):
    return verify_artifact_file(
        path,
        specs,
        expected_config_sha256=CONFIG_SHA256,
        expected_index_sha256=INDEX_SHA256,
    )


def _write_mutation(path: Path, data: bytearray, suffix: str) -> Path:
    changed = path.with_name(path.name + suffix)
    changed.write_bytes(data)
    return changed


def _write_tiny_manifest_bundle(directory: Path):
    specs = _tiny_specs()
    sources, _ = _tiny_sources(directory, specs)
    (directory / "config.json").write_bytes(SOURCE_CONFIG_BYTES)
    index_path = directory / "model.safetensors.index.json"
    weight_map = {
        specs[0].source_name: "model-00001-of-00002.safetensors",
        specs[1].source_name: "model-00002-of-00002.safetensors",
    }
    index_path.write_bytes(canonical_json_bytes({"weight_map": weight_map}))
    contract = {
        "repository": "google/gemma-4-31B-it",
        "revision": MODEL_REVISION,
        "snapshot": str(directory),
        "config_sha256": CONFIG_SHA256,
        "index_sha256": INDEX_SHA256,
    }
    sources = tuple(
        SourceTensor(
            declaration.path,
            weight_map[spec.source_name],
            declaration.source_name,
            declaration.offset,
            declaration.byte_length,
        )
        for spec, declaration in zip(specs, sources, strict=True)
    )
    artifact_path = directory / "weights.gwt"
    written = write_artifact_partial(
        artifact_path,
        specs,
        sources,
        config_sha256=CONFIG_SHA256,
        index_sha256=INDEX_SHA256,
    )
    verified = _verify(artifact_path, specs)
    for name, data in SERVING_ASSET_BYTES.items():
        (directory / name).write_bytes(data)
    # Exercise the BF16 verifier with an independent fixture, without relying
    # on the retired BF16 converter to produce the manifest under test.
    manifest = {
        "schema_version": 6,
        "format": "gemma4-31b-bf16-v4",
        "model": {key: contract[key] for key in
                  ("repository", "revision", "config_sha256", "index_sha256")},
        "artifact": {
            "file": artifact_path.name,
            "file_bytes": written.header.file_bytes,
            "file_sha256": written.file_sha256,
            "header_sha256": written.header.header_sha256.hex(),
            "entry_table_sha256": written.header.entry_table_sha256.hex(),
            "payload_sha256": written.header.payload_sha256.hex(),
        },
        "layout": {
            "alignment": ALIGNMENT,
            "logical_data_bytes": written.header.logical_data_bytes,
            "logical_tensor_count": written.header.logical_tensor_count,
            "physical_tensor_count": written.header.physical_tensor_count,
            "payload_bytes": written.header.payload_bytes,
            "target": "sm_120a",
        },
        "aliases": [{"logical_id": LM_HEAD_LOGICAL_ID, "name": "lm_head.weight",
                     "target": "embed_tokens.weight", "target_physical_id": 0}],
        "serving": serving_metadata(SERVING_ASSET_BYTES),
        "source": {
            "index": {"file": index_path.name, "byte_length": index_path.stat().st_size,
                      "sha256": INDEX_SHA256},
            "shards": [{"file": name, "byte_length": (directory / name).stat().st_size,
                        "sha256": sha256_file(directory / name)}
                       for name in sorted(set(weight_map.values()))],
        },
        "tensors": [
            {"physical_id": spec.physical_id, "name": spec.name,
             "layer": None if spec.layer == -1 else spec.layer,
             "role": spec.role.name.lower(), "shape": list(spec.shape),
             "dtype": "BF16", "layout": "C_ORDER",
             "file_offset": entry.file_offset, "byte_length": entry.byte_length,
             "slot_length": align_up(entry.byte_length), "sha256": entry.sha256_hex,
             "source_name": spec.source_name, "source_component": spec.source_component,
             "source_shard": source.shard_name, "source_offset": source.offset}
            for spec, source, entry in zip(specs, sources, written.entries, strict=True)
        ],
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return artifact_path, manifest_path, manifest, verified, specs, contract


class Bf16ArtifactTests(unittest.TestCase):
    def test_production_inventory_and_sizes(self) -> None:
        text = expected_tensor_specs()
        self.assertEqual(len(text), 832)
        self.assertTrue(all(spec.role <= Role.FINAL_NORM for spec in text))
        self.assertEqual(sum(spec.byte_length for spec in text), 61_394_690_680)
        self.assertEqual(data_offset_for_count(len(text)), 65_536)
        specs = all_tensor_specs()
        self.assertEqual(len(specs), 1236)
        self.assertEqual(specs[0].name, "embed_tokens.weight")
        self.assertEqual(specs[831].name, "norm.weight")
        self.assertEqual(specs[832].name, "vision_tower.patch_embedder.input_proj.weight")
        self.assertEqual(specs[833].shape, (2, 10_240, 1_152))
        self.assertEqual(specs[1187].name, "embed_vision.embedding_projection.weight")
        self.assertEqual(specs[1188].name, "assistant.embed_tokens.weight")
        self.assertEqual(specs[1188].shape, (262144, 1024))
        self.assertEqual(specs[1233].name, "assistant.norm.weight")
        self.assertEqual(specs[1234].shape, (1024, 10752))
        self.assertEqual(specs[1235].shape, (5376, 1024))
        self.assertEqual(sum(spec.byte_length for spec in specs), 63_485_214_944)
        self.assertEqual(sum(align_up(spec.byte_length) for spec in specs), 63_486_726_144)
        self.assertEqual(data_offset_for_count(len(specs)), 94_208)
        self.assertEqual(HEADER_USED_BYTES, 328)
        for layer in range(60):
            roles = {
                spec.role
                for spec in specs
                if spec.layer == layer and spec.role <= Role.FINAL_NORM
            }
            self.assertEqual(Role.V_PROJ in roles, layer % 6 != 5)
        for layer in range(27):
            roles = {
                spec.role
                for spec in specs
                if spec.layer == layer and Role.VISION_PATCH_PROJ <= spec.role <= Role.VISION_PROJECTION
            }
            self.assertEqual(len(roles), 13)
            self.assertIn(Role.VISION_V_PROJ, roles)
        for layer in range(4):
            layer_specs = [spec for spec in specs if spec.layer == layer and spec.source_component == "assistant"]
            self.assertEqual(len(layer_specs), 11)
            self.assertEqual(layer_specs[1].shape, ((16384 if layer == 3 else 8192), 1024))
        self.assertEqual(sum(spec.byte_length for spec in specs[1188:]), 939_037_192)

    def test_tiny_artifact_preserves_bits_alignment_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            path, specs, sources, raw_tensors, first = _write_tiny_artifact(directory, "first.partial")
            verified = _verify(path, specs)
            self.assertEqual(first.file_sha256, verified.file_sha256)
            self.assertEqual(verified.header.logical_tensor_count, len(specs) + 1)
            artifact = path.read_bytes()
            for entry, raw in zip(verified.entries, raw_tensors, strict=True):
                self.assertEqual(entry.file_offset % ALIGNMENT, 0)
                self.assertEqual(artifact[entry.file_offset : entry.file_offset + entry.byte_length], raw)
                self.assertEqual(entry.sha256_hex, hashlib.sha256(raw).hexdigest())
                padding_start = entry.file_offset + entry.byte_length
                padding_end = entry.file_offset + align_up(entry.byte_length)
                self.assertEqual(artifact[padding_start:padding_end], b"\0" * (padding_end - padding_start))
            self.assertEqual(verified.entries[0].rank, 3)
            self.assertEqual(
                (verified.entries[0].dim0, verified.entries[0].dim1, verified.entries[0].dim2),
                (1, 2, 3),
            )

            second = write_artifact_partial(
                directory / "second.partial",
                specs,
                sources,
                config_sha256=CONFIG_SHA256,
                index_sha256=INDEX_SHA256,
            )
            self.assertEqual(path.read_bytes(), second.path.read_bytes())
            self.assertEqual(first.file_sha256, second.file_sha256)

    def test_header_table_tensor_and_padding_corruption_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            path, specs, _, _, written = _write_tiny_artifact(directory)
            original = path.read_bytes()
            mutations: dict[str, bytearray] = {}

            header = bytearray(original)
            header[0] ^= 1
            mutations["header"] = header

            table = bytearray(original)
            table[HEADER_BYTES + ENTRY_BYTES - 1] ^= 1
            mutations["table"] = table

            tensor = bytearray(original)
            tensor[written.entries[0].file_offset] ^= 1
            mutations["tensor"] = tensor

            padding = bytearray(original)
            padding[written.entries[0].file_offset + written.entries[0].byte_length] = 1
            mutations["padding"] = padding

            for label, data in mutations.items():
                with self.subTest(label=label):
                    changed = _write_mutation(path, data, f".{label}")
                    with self.assertRaises(ArtifactError):
                        _verify(changed, specs)

    def test_semantic_table_corruption_is_rejected_after_rehash(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            path, specs, _, _, _ = _write_tiny_artifact(directory)
            mutations = {
                "role": (4, int(Role.Q_PROJ), "wrong role"),
                "reserved_tail": (ENTRY_BYTES - 1, 1, "reserved tail is nonzero"),
            }
            for label, (entry_offset, value, error) in mutations.items():
                with self.subTest(label=label):
                    data = bytearray(path.read_bytes())
                    data[HEADER_BYTES + entry_offset] = value
                    table = data[HEADER_BYTES : HEADER_BYTES + len(specs) * ENTRY_BYTES]
                    data[TABLE_HASH_OFFSET : TABLE_HASH_OFFSET + 32] = hashlib.sha256(table).digest()
                    data[HEADER_HASH_OFFSET : HEADER_HASH_OFFSET + 32] = b"\0" * 32
                    data[HEADER_HASH_OFFSET : HEADER_HASH_OFFSET + 32] = hashlib.sha256(
                        data[:HEADER_BYTES]
                    ).digest()
                    changed = _write_mutation(path, data, f".{label}")
                    with self.assertRaisesRegex(ArtifactError, error):
                        _verify(changed, specs)

    def test_v4_header_authenticates_text_source_and_alias(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            path, specs, _, _, _ = _write_tiny_artifact(directory)
            for label, offset in {
                "format version": 8,
                "alias count": 28,
                "lm_head logical id": 88,
                "lm_head target id": 92,
                "header reserved bytes": 328,
            }.items():
                with self.subTest(label=label):
                    data = bytearray(path.read_bytes())
                    data[offset] ^= 1
                    data[HEADER_HASH_OFFSET:HEADER_HASH_OFFSET + 32] = b"\0" * 32
                    data[HEADER_HASH_OFFSET:HEADER_HASH_OFFSET + 32] = hashlib.sha256(data[:HEADER_BYTES]).digest()
                    changed = _write_mutation(path, data, f".header-{offset}")
                    with self.assertRaisesRegex(ArtifactError, label):
                        _verify(changed, specs)

    def test_truncation_and_trailing_data_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            path, specs, _, _, _ = _write_tiny_artifact(directory)
            original = path.read_bytes()
            variants = [b"", original[:7], original[: HEADER_BYTES - 1], original[:-1], original + b"x"]
            for index, raw in enumerate(variants):
                with self.subTest(index=index):
                    changed = directory / f"broken-{index}.gwt"
                    changed.write_bytes(raw)
                    with self.assertRaises(ArtifactError):
                        _verify(changed, specs)

    def test_existing_partial_is_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            path, specs, sources, _, _ = _write_tiny_artifact(directory)
            before = path.read_bytes()
            with self.assertRaisesRegex(ArtifactError, "partial output already exists"):
                write_artifact_partial(
                    path,
                    specs,
                    sources,
                    config_sha256=CONFIG_SHA256,
                    index_sha256=INDEX_SHA256,
                )
            self.assertEqual(path.read_bytes(), before)

    def test_safetensors_duplicate_key_and_truncation_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            duplicate = directory / "duplicate.safetensors"
            item = '{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}'
            header = (f'{{"x":{item},"x":{item}}}').encode()
            duplicate.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")
            with self.assertRaisesRegex(ArtifactError, "duplicate JSON key"):
                read_safetensors_header(duplicate)

            truncated = directory / "truncated.safetensors"
            truncated.write_bytes(b"short")
            with self.assertRaisesRegex(ArtifactError, "header is truncated"):
                read_safetensors_header(truncated)

    def test_canonical_json_and_safe_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            canonical = directory / "canonical.json"
            canonical.write_bytes(canonical_json_bytes({"b": 2, "a": 1}))
            self.assertEqual(load_json_object(canonical, canonical=True), {"a": 1, "b": 2})
            noncanonical = directory / "noncanonical.json"
            noncanonical.write_text('{"b":2,"a":1}')
            with self.assertRaisesRegex(ArtifactError, "not canonical JSON"):
                load_json_object(noncanonical, canonical=True)
            for unsafe in ("../weights.gwt", "/tmp/weights.gwt", "a/b"):
                with self.subTest(unsafe=unsafe), self.assertRaises(ArtifactError):
                    safe_basename(unsafe, "artifact filename")

    def test_manifest_is_cross_checked_against_binary(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            artifact_path, manifest_path, manifest, verified, specs, contract = _write_tiny_manifest_bundle(directory)
            validate_manifest(manifest_path, artifact_path, verified, specs, contract)

            manifest["artifact"]["file_sha256"] = "0" * 64
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            with self.assertRaisesRegex(ArtifactError, "artifact metadata mismatch"):
                validate_manifest(manifest_path, artifact_path, verified, specs, contract)

    def test_manifest_rejects_wrong_alias_or_source_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            artifact_path, manifest_path, manifest, verified, specs, contract = _write_tiny_manifest_bundle(directory)
            original = canonical_json_bytes(manifest)
            for label, mutate in (
                ("lm_head alias mismatch", lambda m: m["aliases"][0].update(target_physical_id=1)),
                ("wrong source_component", lambda m: m["tensors"][1].update(source_component="assistant")),
                ("wrong source_shard", lambda m: m["tensors"][1].update(source_shard="model-00001-of-00002.safetensors")),
            ):
                with self.subTest(label=label):
                    mutated = json.loads(original)
                    mutate(mutated)
                    manifest_path.write_bytes(canonical_json_bytes(mutated))
                    with self.assertRaisesRegex(ArtifactError, label):
                        validate_manifest(manifest_path, artifact_path, verified, specs, contract)

    def test_source_audit_does_not_need_assistant_and_rejects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            _, _, manifest, _, specs, old_contract = _write_tiny_manifest_bundle(directory)
            config_hash = hashlib.sha256((directory / "config.json").read_bytes()).hexdigest()
            index_hash = hashlib.sha256((directory / "model.safetensors.index.json").read_bytes()).hexdigest()
            contract = {**old_contract, "config_sha256": config_hash, "index_sha256": index_hash}
            prepared = read_snapshot(directory, specs)
            self.assertEqual(len({weight.tensor.path for weight in prepared.weights}), 2)
            self.assertFalse((directory / "model.safetensors").exists())
            verify_source(manifest, specs, contract)
            shard = directory / manifest["source"]["shards"][0]["file"]
            original = shard.read_bytes()
            shard.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            read_snapshot(directory, specs)  # Modified weights are a valid new source.
            with self.assertRaisesRegex(ArtifactError, "source shard SHA-256 mismatch"):
                verify_source(manifest, specs, contract)
            shard.write_bytes(original)
            (directory / "config.json").write_text('{"wrong":true}')
            with self.assertRaisesRegex(ArtifactError, "target config SHA-256 mismatch"):
                verify_source(manifest, specs, contract)

    def test_manifest_does_not_bind_oracle_contract_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            artifact_path, manifest_path, manifest, verified, specs, contract = _write_tiny_manifest_bundle(directory)
            self.assertEqual(manifest["schema_version"], 6)
            self.assertNotIn("contract_source_sha256", manifest["model"])
            self.assertNotIn("contract_resolved_sha256", manifest["model"])

            validate_manifest(manifest_path, artifact_path, verified, specs, dict(contract))

    def test_manifest_rejects_legacy_contract_hash_fields(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            artifact_path, manifest_path, manifest, verified, specs, contract = _write_tiny_manifest_bundle(directory)
            manifest["model"]["contract_source_sha256"] = "a" * 64
            manifest["model"]["contract_resolved_sha256"] = "b" * 64
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            with self.assertRaisesRegex(ArtifactError, "manifest model keys differ"):
                validate_manifest(manifest_path, artifact_path, verified, specs, contract)


if __name__ == "__main__":
    unittest.main()
