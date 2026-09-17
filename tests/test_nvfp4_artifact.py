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
from tools import nvfp4_artifact as native


def tiny_specs():
    return (
        bf16.TensorSpec(0, -1, bf16.Role.EMBED_TOKENS, "embed_tokens.weight", "embed.weight", (2, 3)),
        bf16.TensorSpec(1, 0, bf16.Role.GATE_PROJ, "layers.0.gate_proj.weight", "gate.weight", (129, 80)),
        bf16.TensorSpec(2, 0, bf16.Role.UP_PROJ, "layers.0.up_proj.weight", "up.weight", (129, 80)),
    )


def write_fixture(directory: Path):
    specs = tiny_specs()
    sources = []
    for spec, storage in zip(specs, (native.StorageType.BF16, native.StorageType.NVFP4_W4A4, native.StorageType.BF16), strict=True):
        data = bytes(i % 256 for i in range(spec.byte_length if storage == native.StorageType.BF16 else native.packed_bytes(*spec.shape)))
        weight = directory / f"weight{spec.physical_id}"
        weight.write_bytes(data)
        declaration = bf16.SourceTensor(weight, weight.name, spec.source_name, 0, len(data))
        if storage == native.StorageType.BF16:
            sources.append(native.TensorSource(declaration, storage, expected_sha256=hashlib.sha256(data).digest()))
        else:
            rows, columns = spec.shape
            raw = bytes(i % 127 for i in range(rows * columns // 16))
            scales = directory / "scales"
            scales.write_bytes(raw)
            source_scales = bf16.SourceTensor(scales, scales.name, "gate.weight_scale", 0, len(raw))
            sources.append(native.TensorSource(declaration, storage, source_scales, struct.pack("<2f", .25, .125)))
    artifact = native.write_artifact_partial(directory / "weights.gwt", specs, sources)
    return artifact, tuple(sources)


def rewrite_hashes(path: Path, *, entry_index: int | None = None) -> None:
    raw = bytearray(path.read_bytes())
    header, _ = bf16.decode_header(bytes(raw[:bf16.HEADER_BYTES]))
    if entry_index is not None:
        start = bf16.HEADER_BYTES + entry_index * bf16.ENTRY_BYTES
        fields = bf16.ENTRY_STRUCT.unpack(raw[start:start + bf16.ENTRY_BYTES])
        offset, length = fields[8:10]
        raw[start + 36:start + 68] = hashlib.sha256(raw[offset:offset + length]).digest()
    table_end = bf16.HEADER_BYTES + header.physical_tensor_count * bf16.ENTRY_BYTES
    raw[232:264] = hashlib.sha256(raw[bf16.HEADER_BYTES:table_end]).digest()
    raw[264:296] = hashlib.sha256(raw[header.data_offset:]).digest()
    raw[296:328] = b"\0" * 32
    raw[296:328] = hashlib.sha256(raw[:bf16.HEADER_BYTES]).digest()
    path.write_bytes(raw)


class NativeArtifactTests(unittest.TestCase):
    def test_arbitrary_provenance_is_informational(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact, _ = write_fixture(Path(temporary))
            raw = bytearray(artifact.path.read_bytes())
            raw[96:128] = b"community/fine-tune".ljust(32, b"\0")
            raw[128:168] = b"x" * 40
            raw[168:232] = b"a" * 64
            artifact.path.write_bytes(raw)
            rewrite_hashes(artifact.path)
            verified = native.verify_artifact_file(artifact.path, tiny_specs())
            self.assertEqual(verified.header.config_sha256, b"a" * 32)
            self.assertEqual(verified.header.source_index_sha256, b"a" * 32)

    def test_layout_round_trip_preserves_nibbles_scales_and_globals(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            artifact, sources = write_fixture(directory)
            verified = native.verify_artifact_file(artifact.path, tiny_specs())
            self.assertEqual(verified.file_sha256, artifact.file_sha256)
            data = artifact.path.read_bytes()
            entry = artifact.entries[1]
            rows, columns = tiny_specs()[1].shape
            start = entry.file_offset
            packed_length = native.packed_bytes(rows, columns)
            self.assertEqual(data[start:start + packed_length], sources[1].weight.path.read_bytes())
            swizzled = data[start + packed_length:start + entry.byte_length - 8]
            original = sources[1].block_scales.path.read_bytes()
            offsets = set()
            for row in range(rows):
                for scale in range(columns // 16):
                    offset = ((row // 128) * ((columns + 63) // 64) + scale // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + scale % 4
                    self.assertEqual(swizzled[offset], original[row * (columns // 16) + scale])
                    offsets.add(offset)
            self.assertTrue(all(value == 0 for offset, value in enumerate(swizzled) if offset not in offsets))
            self.assertEqual(data[start + entry.byte_length - 8:start + entry.byte_length], struct.pack("<2f", .25, .125))
            self.assertEqual(artifact.entries[2].storage_type, native.StorageType.BF16)

    def test_reject_payload_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact, _ = write_fixture(Path(temporary))
            raw = bytearray(artifact.path.read_bytes())
            raw[artifact.entries[1].file_offset] ^= 1
            artifact.path.write_bytes(raw)
            with self.assertRaisesRegex(bf16.ArtifactError, "tensor 1 SHA-256 mismatch"):
                native.verify_artifact_file(artifact.path, tiny_specs())

    def test_reject_bad_globals_even_with_correct_hashes(self):
        for value in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                artifact, _ = write_fixture(Path(temporary))
                raw = bytearray(artifact.path.read_bytes())
                entry = artifact.entries[1]
                offset = entry.file_offset + entry.byte_length - 8
                raw[offset:offset + 4] = struct.pack("<f", value)
                artifact.path.write_bytes(raw)
                rewrite_hashes(artifact.path, entry_index=1)
                with self.assertRaisesRegex(bf16.ArtifactError, "finite and positive"):
                    native.verify_artifact_file(artifact.path, tiny_specs())

    def test_reject_scale_padding_even_with_correct_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact, _ = write_fixture(Path(temporary))
            raw = bytearray(artifact.path.read_bytes())
            entry = artifact.entries[1]
            # First tile's scale column 5 is padding because K/16 == 5.
            raw[entry.file_offset + native.packed_bytes(129, 80) + 513] = 1
            artifact.path.write_bytes(raw)
            rewrite_hashes(artifact.path, entry_index=1)
            with self.assertRaisesRegex(bf16.ArtifactError, "scale padding"):
                native.verify_artifact_file(artifact.path, tiny_specs())

    def test_reject_unsupported_storage_even_with_correct_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact, _ = write_fixture(Path(temporary))
            raw = bytearray(artifact.path.read_bytes())
            raw[bf16.HEADER_BYTES + bf16.ENTRY_BYTES + 7] = 2
            artifact.path.write_bytes(raw)
            rewrite_hashes(artifact.path)
            with self.assertRaisesRegex(bf16.ArtifactError, "unsupported role or storage"):
                native.verify_artifact_file(artifact.path, tiny_specs())

    def test_reject_bf16_fallback_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, sources = write_fixture(directory)
            sources[2].weight.path.write_bytes(b"\0" * sources[2].weight.byte_length)
            with self.assertRaisesRegex(bf16.ArtifactError, "fallback SHA-256 mismatch"):
                native.write_artifact_partial(directory / "other", tiny_specs(), sources)

    def test_invalid_scales(self):
        for value in (0x7f, 0x80, 0xff):
            with self.assertRaisesRegex(bf16.ArtifactError, "finite and nonnegative"):
                native.swizzle_scale_chunk(bytes([value]), 1, 1)

    def test_sparse_mask_and_ordered_exemptions(self):
        specs = bf16.expected_tensor_specs()
        full = native.parse_mask(native.DEFAULT_MASK, specs)
        self.assertEqual(sum(full), 180)
        selected = native.parse_mask("* gate_proj nvfp4_w4a4\n0 gate_proj bf16 # exemption\n1 up_proj nvfp4_w4a4\n", specs)
        self.assertEqual(sum(selected), 60)
        self.assertEqual(selected[next(i for i, s in enumerate(specs) if s.layer == 0 and s.role == bf16.Role.GATE_PROJ)], native.StorageType.BF16)
        self.assertEqual(sum(native.parse_mask("", specs)), 0)
        all_projections = "".join(f"* {bf16.ROLE_NAMES[role]} nvfp4_w4a4\n" for role in native.TARGET_ROLES)
        self.assertEqual(sum(native.parse_mask(all_projections, specs)), 410)
        for text in ("* gate_proj nvfp4", "60 gate_proj bf16",
                     "5 v_proj bf16", "0 gate_proj nvfp4_w4a4 extra", "-1 gate_proj bf16", "* norm bf16"):
            with self.subTest(text=text), self.assertRaises(bf16.ArtifactError):
                native.parse_mask(text, specs)


if __name__ == "__main__":
    unittest.main()
