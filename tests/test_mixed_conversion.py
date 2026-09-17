"""Packed conversion across target projection roles, including NVFP4 attention."""

import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import bf16_artifact as bf16
from tools import convert as converter
from tools import nvfp4_artifact as native


def source_tensor(path, values, name="weight"):
    words = (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype("<u2")
    path.write_bytes(words.tobytes())
    return bf16.SourceTensor(path, path.name, name, 0, words.nbytes)


class MixedConversionTests(unittest.TestCase):
    def test_e2m1_ties_saturation_signed_zero_and_nibble_order(self):
        self.assertEqual(converter.quantize_e2m1(np.array(
            [0, -0., .25, .75, 1.25, 1.75, 2.5, 3.5, 5, -5, 6, -6, 8, -8, .5, -.5])),
            bytes.fromhex("80 20 42 64 e6 f7 f7 91"))

    def test_nvfp4_streaming_matches_frozen_group16_recipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = [2688., -2688.] + [0.] * 14
            values += [0, -0., .25, .75, 1.25, 1.75, 2.5, 3.5, 5, -5, 6, -6, 6, -6, .5, -.5]
            values += [0.] * 16 + [2 ** -16] + [0.] * 15
            source = source_tensor(root / "source", values)
            # Intentionally split the input chunk below a group boundary.
            with patch.object(bf16, "CHUNK_BYTES", 18):
                packed = converter.quantize_nvfp4_tensor(source, root / "packed", .125)
            self.assertEqual(native.validate_globals(packed.globals), (1., .125))
            self.assertEqual(packed.block_scales.path.read_bytes(), bytes((126, 56, 56, 1)))
            self.assertEqual(packed.weight.path.read_bytes(),
                             bytes.fromhex("f7") + bytes(7) +
                             bytes.fromhex("80 20 42 64 e6 f7 f7 91") + bytes(16))
            self.assertEqual(packed.weight.byte_length, 32)
            self.assertEqual(packed.block_scales.byte_length, 4)

    def test_zero_and_nonfinite_nvfp4_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = source_tensor(root / "source", [0., -0.] * 16)
            packed = converter.quantize_nvfp4_tensor(source, root / "zero", .125)
            self.assertEqual(native.validate_globals(packed.globals), (1., .125))
            self.assertEqual(packed.weight.path.read_bytes(), b"\x80" * 16)
            for value in (float("nan"), float("inf"), -float("inf")):
                source = source_tensor(root / "source", [value] + [0.] * 15)
                with self.assertRaisesRegex(bf16.ArtifactError, "nonfinite"):
                    converter.quantize_nvfp4_tensor(source, root / "invalid", .125)

    def test_every_projection_roundtrips_in_each_storage_and_mixed(self):
        specs = [bf16.TensorSpec(0, -1, bf16.Role.EMBED_TOKENS, "embed_tokens.weight", "embed", (4, 32))]
        for layer in (0, 1, 2, 5, 11, 17):
            for role in native.TARGET_ROLES:
                if layer in (5, 11, 17) and role == bf16.Role.V_PROJ:
                    continue
                name = f"layers.{layer}.{bf16.ROLE_NAMES[role]}.weight"
                specs.append(bf16.TensorSpec(len(specs), layer, role, name, name, (3, 32)))
        specs = tuple(specs)
        types = ("bf16", "fp8_w8a8", "nvfp4_w4a4")
        for mode in (*types, "mixed"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                sources = [source_tensor(root / str(spec.physical_id),
                            [(-1 if i % 2 else 1) * (i % 17) / 8 for i in range(spec.byte_length // 2)], spec.source_name)
                           for spec in specs]
                baseline = bf16.write_artifact_partial(root / "baseline", specs, sources,
                    config_sha256="a" * 64, index_sha256="b" * 64)
                mask = "".join(f"{spec.layer} {bf16.ROLE_NAMES[spec.role]} "
                               f"{types[spec.layer % 3] if spec.layer < 3 else types[(spec.layer - 5) // 6]}\n"
                               if mode == "mixed" else
                               f"{spec.layer} {bf16.ROLE_NAMES[spec.role]} {mode}\n"
                               for spec in specs[1:])
                selected = native.parse_mask(mask, specs)
                scales = {spec.name: .125 for spec in specs[1:]}
                with patch.object(bf16, "expected_tensor_specs", return_value=specs), \
                     patch.object(converter, "read_serving_assets", return_value={}), \
                     patch.object(converter, "publish_serving_assets", return_value={}), \
                     redirect_stdout(io.StringIO()):
                    artifact, manifest_path = converter.convert(baseline.path, mask, scales, root / "output")
                verified = native.verify_artifact_file(artifact, specs)
                self.assertEqual(tuple(entry.storage_type for entry in verified.entries), selected)
                with artifact.open("rb") as stream:
                    self.assertEqual(stream.read(8), native.MIXED_MAGIC)
                manifest = json.loads(manifest_path.read_text())
                self.assertEqual(manifest["format"], native.MIXED_FORMAT_NAME)
                for entry, original in zip(verified.entries, baseline.entries, strict=True):
                    if entry.storage_type == native.StorageType.BF16:
                        self.assertEqual(entry.sha256, original.sha256)
                    else:
                        self.assertEqual(manifest["tensors"][entry.physical_id]["input_scale"], .125)

    def test_nvfp4_rejects_missing_or_unexecutable_input_scales(self):
        from tests.test_fp8_artifact import fixture, tiny_specs
        with tempfile.TemporaryDirectory() as temporary:
            source = fixture(Path(temporary), mixed=False)
            for value in (None, 0, -1, True, "0.1", 1e-100, 1e100, float("nan")):
                scales = {} if value is None else {tiny_specs()[1].name: value}
                with self.subTest(value=value), self.assertRaises(bf16.ArtifactError):
                    converter.prepare_source(source.path, tiny_specs(), "0 q_proj nvfp4_w4a4", scales)


if __name__ == "__main__":
    unittest.main()
