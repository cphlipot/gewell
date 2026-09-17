from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools import bf16_artifact as bf16
from tools import convert as converter
from tools import nvfp4_artifact as native
from tests.test_nvfp4_artifact import rewrite_hashes


def tiny_specs():
    return (
        bf16.TensorSpec(0, -1, bf16.Role.EMBED_TOKENS, "embed_tokens.weight", "embed.weight", (2, 3)),
        bf16.TensorSpec(1, 0, bf16.Role.Q_PROJ, "layers.0.self_attn.q_proj.weight", "q.weight", (2, 16)),
        bf16.TensorSpec(2, 0, bf16.Role.GATE_PROJ, "layers.0.mlp.gate_proj.weight", "gate.weight", (2, 16)),
        bf16.TensorSpec(3, 0, bf16.Role.UP_PROJ, "layers.0.mlp.up_proj.weight", "up.weight", (2, 16)),
    )


def fixture(directory: Path, *, mixed=True, ancestry="google"):
    sources = []
    for spec in tiny_specs():
        storage = (native.StorageType.FP8_W8A8 if mixed else native.StorageType.BF16) if spec.physical_id == 1 else (
            native.StorageType.NVFP4_W4A4 if spec.physical_id == 2 else native.StorageType.BF16)
        length = spec.byte_length if storage == native.StorageType.BF16 else (
            native.fp8_packed_bytes(*spec.shape) if storage == native.StorageType.FP8_W8A8 else native.packed_bytes(*spec.shape))
        weight = directory / f"weight{spec.physical_id}"
        weight.write_bytes(b"\x38" * length if storage != native.StorageType.BF16 else struct.pack("<H", 0x3f80) * (length // 2))
        source = bf16.SourceTensor(weight, weight.name, spec.source_name, 0, length)
        scales, globals_raw = None, b""
        if storage == native.StorageType.NVFP4_W4A4:
            path = directory / "scales"
            path.write_bytes(b"\x38" * (spec.shape[0] * spec.shape[1] // 16))
            scales = bf16.SourceTensor(path, path.name, "scales", 0, path.stat().st_size)
        if storage != native.StorageType.BF16:
            globals_raw = struct.pack("<2f", .25, .125)
        sources.append(native.TensorSource(source, storage, scales, globals_raw))
    return native.write_artifact_partial(directory / "source", tiny_specs(), sources)


class Fp8ArtifactTests(unittest.TestCase):
    def test_mixed_roundtrip_preserves_storage_payload_and_scales(self):
        for ancestry in ("google", "nvidia"):
            with self.subTest(ancestry=ancestry), tempfile.TemporaryDirectory() as temporary:
                written = fixture(Path(temporary), ancestry=ancestry)
                verified = native.verify_artifact_file(written.path, tiny_specs())
                self.assertEqual(verified.file_sha256, written.file_sha256)
                data = written.path.read_bytes()
                self.assertEqual(data[:8], native.MIXED_MAGIC)
                self.assertEqual(tuple(entry.storage_type for entry in verified.entries),
                                 (native.StorageType.BF16, native.StorageType.FP8_W8A8,
                                  native.StorageType.NVFP4_W4A4, native.StorageType.BF16))
                entry = verified.entries[1]
                self.assertEqual(entry.byte_length, 40)
                self.assertEqual(data[entry.file_offset:entry.file_offset + 40], b"\x38" * 32 + struct.pack("<2f", .25, .125))

    def test_nan_codes_rejected_with_valid_checksums(self):
        for code in (0x7f, 0xff):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temporary:
                artifact = fixture(Path(temporary))
                raw = bytearray(artifact.path.read_bytes())
                raw[artifact.entries[1].file_offset] = code
                artifact.path.write_bytes(raw)
                rewrite_hashes(artifact.path, entry_index=1)
                native.read_metadata(artifact.path, tiny_specs())
                with self.assertRaisesRegex(bf16.ArtifactError, "FP8 weights"):
                    native.verify_artifact_file(artifact.path, tiny_specs())

    def test_global_scales_checked_at_metadata_load(self):
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                artifact = fixture(Path(temporary))
                raw = bytearray(artifact.path.read_bytes())
                offset = artifact.entries[1].file_offset + 32
                raw[offset:offset + 4] = struct.pack("<f", value)
                artifact.path.write_bytes(raw)
                rewrite_hashes(artifact.path, entry_index=1)
                with self.assertRaisesRegex(bf16.ArtifactError, "finite and positive"):
                    native.read_metadata(artifact.path, tiny_specs())

    def test_mixed_rejects_unknown_tag(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = fixture(Path(temporary))
            raw = bytearray(artifact.path.read_bytes())
            raw[bf16.HEADER_BYTES + bf16.ENTRY_BYTES + 7] = 3
            artifact.path.write_bytes(raw)
            rewrite_hashes(artifact.path)
            with self.assertRaisesRegex(bf16.ArtifactError, "unsupported role or storage"):
                native.read_metadata(artifact.path, tiny_specs())

    def test_fp8_rejects_scales_outside_executable_range(self):
        for weight_scale, input_scale in ((1.0, 1e-40), (1e30, 1e30), (1e-30, 1e-30)):
            with self.subTest(weight_scale=weight_scale, input_scale=input_scale), tempfile.TemporaryDirectory() as temporary:
                globals_raw = struct.pack("<2f", weight_scale, input_scale)
                native.validate_globals(globals_raw)  # NVFP4 retains its existing scale contract.
                with self.assertRaisesRegex(bf16.ArtifactError, "executable FP32"):
                    native.validate_gemm_globals(globals_raw)
                artifact = fixture(Path(temporary))
                raw = bytearray(artifact.path.read_bytes())
                offset = artifact.entries[1].file_offset + 32
                raw[offset:offset + 8] = globals_raw
                artifact.path.write_bytes(raw)
                rewrite_hashes(artifact.path, entry_index=1)
                with self.assertRaisesRegex(bf16.ArtifactError, "executable FP32"):
                    native.read_metadata(artifact.path, tiny_specs())

    def test_mixed_rejects_fp8_embedding_tag(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = fixture(Path(temporary))
            raw = bytearray(artifact.path.read_bytes())
            raw[bf16.HEADER_BYTES + 7] = native.StorageType.FP8_W8A8
            artifact.path.write_bytes(raw)
            rewrite_hashes(artifact.path)
            with self.assertRaisesRegex(bf16.ArtifactError, "target text"):
                native.read_metadata(artifact.path, tiny_specs())

    def test_fp8_scope_and_dimensions(self):
        for role in native.TARGET_ROLES:
            spec = bf16.TensorSpec(0, 0, role, "projection", "weight", (3, 5))
            self.assertEqual(native.tensor_bytes(spec, native.StorageType.FP8_W8A8), 23)
        for role in (bf16.Role.EMBED_TOKENS, bf16.Role.Q_NORM, bf16.Role.VISION_Q_PROJ, bf16.Role.ASSISTANT_Q_PROJ):
            with self.subTest(role=role), self.assertRaisesRegex(bf16.ArtifactError, "target text"):
                native.tensor_bytes(bf16.TensorSpec(0, 0, role, "other", "weight", (2, 16)), native.StorageType.FP8_W8A8)
        with self.assertRaisesRegex(bf16.ArtifactError, "positive"):
            native.fp8_packed_bytes(0, 16)

    def test_all_projection_mask_and_source_defaults(self):
        specs = bf16.expected_tensor_specs()
        mask = "".join(f"* {bf16.ROLE_NAMES[role]} fp8_w8a8\n" for role in native.TARGET_ROLES)
        selected = native.parse_mask(mask, specs)
        self.assertTrue(all(storage == native.StorageType.FP8_W8A8 for spec, storage in zip(specs, selected) if spec.role in native.TARGET_ROLES))
        self.assertTrue(all(storage == native.StorageType.BF16 for spec, storage in zip(specs, selected) if spec.role not in native.TARGET_ROLES))
        initial = (native.StorageType.BF16, native.StorageType.BF16, native.StorageType.NVFP4_W4A4, native.StorageType.BF16)
        self.assertEqual(native.parse_mask("0 q_proj fp8_w8a8", tiny_specs(), initial=initial),
                         (initial[0], native.StorageType.FP8_W8A8, initial[2], initial[3]))


class Fp8ConversionTests(unittest.TestCase):
    def test_nearest_even_subnormals_saturation_and_signed_zero(self):
        values = np.array([0.0, -0.0, 1 / 1024, 3 / 1024, 1.0625, 1.1875,
                           448, 1000, -1.0625, -1000], dtype=np.float32)
        self.assertEqual(converter.quantize_e4m3(values, 1.0),
                         bytes((0, 0x80, 0, 2, 0x38, 0x3a, 0x7e, 0x7e, 0xb8, 0xfe)))
        all_values = converter.E4M3_VALUES
        self.assertEqual(converter.quantize_e4m3(all_values, 1.0), bytes(range(127)))
        mids = (all_values[:-1] + all_values[1:]) / 2
        self.assertEqual(converter.quantize_e4m3(mids, 1.0), bytes(i + (i % 2) for i in range(126)))
        with self.assertRaisesRegex(bf16.ArtifactError, "nonfinite"):
            converter.quantize_e4m3(np.array([float("nan")]), 1.0)

    def test_streamed_quantization_absmax_and_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for index, values in enumerate(([0.0, -0.0, 0.0, 0.0], [-448.0, 224.0, 1.0, 0.0])):
                path = directory / f"source{index}"
                path.write_bytes((np.array(values, dtype=np.float32).view(np.uint32) >> 16).astype("<u2").tobytes())
                source = bf16.SourceTensor(path, path.name, "weight", 0, 8)
                with patch.object(bf16, "CHUNK_BYTES", 4):
                    packed = converter.quantize_tensor(source, directory / f"packed{index}", .125)
                self.assertEqual(native.validate_globals(packed.globals), (1.0, .125))
                self.assertEqual(packed.weight.path.read_bytes(), bytes((0, 0x80, 0, 0) if index == 0 else (0xfe, 0x76, 0x38, 0)))

    def test_converter_retains_native_entries_and_ancestry(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = fixture(directory, mixed=False, ancestry="nvidia")
            scales = {tiny_specs()[1].name: .125}
            with patch.object(bf16, "expected_tensor_specs", return_value=tiny_specs()), \
                 patch.object(converter, "read_serving_assets", return_value={}), \
                 patch.object(converter, "publish_serving_assets", return_value={}), \
                 patch.object(bf16, "CHUNK_BYTES", 16):
                output, manifest_path = converter.convert(source.path, "0 q_proj fp8_w8a8", scales, directory / "output")
            verified = native.verify_artifact_file(output, tiny_specs())
            self.assertEqual(verified.entries[1].storage_type, native.StorageType.FP8_W8A8)
            for index in (0, 2, 3):
                self.assertEqual(verified.entries[index].sha256, source.entries[index].sha256)
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["format"], native.MIXED_FORMAT_NAME)
            self.assertEqual(manifest["tensors"][1]["input_scale"], .125)
            self.assertAlmostEqual(manifest["tensors"][1]["weight_scale"], 1 / 448)
            self.assertEqual(manifest["tensors"][2]["weight_scale_2"], .25)
            self.assertEqual(manifest["source"]["file_sha256"], source.file_sha256)
            self.assertFalse(any(output.parent.glob("*.partial")))
            self.assertFalse(any(output.parent.glob(".mixed-*")))

    def test_converter_accepts_verified_bf16_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            sources = []
            for spec in tiny_specs():
                path = directory / str(spec.physical_id)
                path.write_bytes(b"\0" * spec.byte_length)
                sources.append(bf16.SourceTensor(path, path.name, spec.source_name, 0, spec.byte_length))
            artifact = bf16.write_artifact_partial(directory / "baseline", tiny_specs(), sources,
                                                   config_sha256="a" * 64,
                                                   index_sha256="b" * 64)
            source, selected, _, _, _ = converter.prepare_source(artifact.path, tiny_specs(), "0 q_proj fp8_w8a8", {tiny_specs()[1].name: .125})
            self.assertEqual(source.file_sha256, artifact.file_sha256)
            self.assertEqual(selected[1], native.StorageType.FP8_W8A8)

    def test_invalid_calibration_and_packed_reinterpretation(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = fixture(Path(temporary), mixed=False)
            for value in (None, 0, -1, float("inf"), float("nan"), True, "0.1", 1e-100, 1e100):
                scales = {tiny_specs()[1].name: value}
                with self.subTest(value=value), self.assertRaises(bf16.ArtifactError):
                    converter.prepare_source(source.path, tiny_specs(), "0 q_proj fp8_w8a8", scales)
            with self.assertWarnsRegex(UserWarning, "lower precision"):
                converter.prepare_source(source.path, tiny_specs(), "0 gate_proj bf16", {})
            with self.assertWarnsRegex(UserWarning, "lower precision"):
                _, _, _, scales, _ = converter.prepare_source(source.path, tiny_specs(), "0 gate_proj fp8_w8a8", {})
            self.assertAlmostEqual(scales[tiny_specs()[2].name], 82 / 448)
            _, _, _, scales, _ = converter.prepare_source(source.path, tiny_specs(), "0 up_proj nvfp4_w4a4", {})
            self.assertAlmostEqual(scales[tiny_specs()[3].name], 82 / 2688)


if __name__ == "__main__":
    unittest.main()
