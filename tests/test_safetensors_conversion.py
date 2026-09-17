"""Conversion works from structural inputs without vendor identity or a baseline."""
import hashlib
import json
from pathlib import Path
import struct
from unittest.mock import patch

import numpy as np
import pytest

from tools import bf16_artifact as bf16
from tools import convert
from tools import extract_vision
from tools import nvfp4_artifact as native
from tools.safetensors_source import read_snapshot
from tools.serving_assets import REQUIRED_ASSETS, validate_serving_assets


def specs():
    return (
        bf16.TensorSpec(0, -1, bf16.Role.EMBED_TOKENS, "embed_tokens.weight", "embed.weight", (2, 16)),
        bf16.TensorSpec(1, 0, bf16.Role.GATE_PROJ, "layers.0.mlp.gate_proj.weight", "gate.weight", (129, 80)),
        bf16.TensorSpec(2, 0, bf16.Role.UP_PROJ, "layers.0.mlp.up_proj.weight", "up.weight", (3, 32)),
    )


def write_safetensors(path, tensors):
    header, data = {}, bytearray()
    for name, dtype, shape, raw in tensors:
        start = len(data)
        data.extend(raw)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [start, len(data)]}
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    return header


def serving(directory):
    directory.mkdir(exist_ok=True)
    for name in REQUIRED_ASSETS:
        (directory / name).write_text('{"arbitrary":"metadata"}' if name.endswith(".json") else 'custom template metadata')


def run_convert(path, output, mask="", scales=None):
    with patch.object(bf16, "expected_tensor_specs", return_value=specs()):
        result, manifest = convert.convert(path, mask, scales or {}, output)
    assert result.name == "weights.gwt"
    metadata = json.loads(manifest.read_text())
    validate_serving_assets(output, metadata["serving"])
    assert {path.name for path in output.iterdir()} == {"weights.gwt", "manifest.json", "tokenizer.json"}
    verified = native.verify_artifact_file(result, specs())
    assert verified.file_sha256 == metadata["artifact"]["file_sha256"]
    return verified


def payload(artifact, index):
    entry = artifact.entries[index]
    with artifact.path.open("rb") as stream:
        stream.seek(entry.file_offset)
        return stream.read(entry.byte_length)


@pytest.mark.parametrize("indexed", [False, True])
def test_repacked_finetune_and_extra_tensors_are_accepted(tmp_path, indexed):
    source = tmp_path / "source"
    serving(source)
    (source / "config.json").write_text('{"source":"custom-finetune"}')
    weight_map = {}
    originals = []
    for i, (spec, dtype) in enumerate(zip(specs(), ("BF16", "F16", "F32"))):
        count = spec.byte_length // 2
        raw = b"\x80\x3f" * count if dtype == "BF16" else np.ones(count, dtype="<f2" if dtype == "F16" else "<f4").tobytes()
        tensors = [(spec.source_name, dtype, spec.shape, raw)]
        if i == 0:
            tensors.append(("unused.bias", "F32", (1,), struct.pack("<f", 2)))
        shard = source / f"custom-{i}.safetensors"
        weight_map.update({name: shard.name for name in write_safetensors(shard, tensors)})
        originals.append(raw)
    if indexed:
        (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    result = run_convert(source, tmp_path / "out")
    assert all(payload(result, i) == b"\x80\x3f" * (s.byte_length // 2) for i, s in enumerate(specs()))
    assert result.header.config_sha256 == hashlib.sha256((source / "config.json").read_bytes()).digest()
    # Different values are accepted without changing a model allowlist.
    shard = source / "custom-0.safetensors"
    data = bytearray(shard.read_bytes())
    record = bf16.read_safetensors_header(shard)[specs()[0].source_name]
    data[record.offset:record.offset + 2] = b"\x00\x40"
    shard.write_bytes(data)
    changed = run_convert(source, tmp_path / "changed")
    assert payload(changed, 0)[:2] == b"\x00\x40"
    with pytest.raises(bf16.ArtifactError, match="refusing to replace"):
        run_convert(source, tmp_path / "out")


def packed_source(directory):
    serving(directory)
    tensors = []
    for spec in specs():
        if spec.role == bf16.Role.GATE_PROJ:
            rows, columns = spec.shape
            prefix = spec.source_name.removesuffix("weight")
            tensors.extend([
                (spec.source_name, "U8", (rows, columns // 2), b"\x76" * (rows * columns // 2)),
                (prefix + "weight_scale", "F8_E4M3", (rows, columns // 16), b"\x38" * (rows * columns // 16)),
                (prefix + "weight_scale_2", "F32", (), struct.pack("<f", .25)),
                (prefix + "input_scale", "F32", (1,), struct.pack("<f", .125)),
            ])
        else:
            tensors.append((spec.source_name, "BF16", spec.shape, b"\x80\x3f" * (spec.byte_length // 2)))
    path = directory / "arbitrary.safetensors"
    write_safetensors(path, tensors)
    return path


def test_packed_import_retention_and_widening_use_source_values(tmp_path):
    source = packed_source(tmp_path / "source")
    packed = run_convert(source, tmp_path / "packed")
    assert packed.entries[1].storage_type == native.StorageType.NVFP4_W4A4
    with pytest.warns(UserWarning, match="lower precision"):
        widened = run_convert(source, tmp_path / "bf16", "0 gate_proj bf16")
    expected = b"\x80\x3f\xc0\x3f" * (specs()[1].byte_length // 4)  # E2M1 [4,6] * .25
    assert payload(widened, 1) == expected
    with pytest.warns(UserWarning, match="lower precision"):
        repacked = run_convert(packed.path, tmp_path / "native-bf16", "0 gate_proj bf16")
    assert payload(repacked, 1) == expected  # Includes a second 128-row scale tile.
    with pytest.warns(UserWarning, match="lower precision"):
        fp8 = run_convert(source, tmp_path / "fp8", "0 gate_proj fp8_w8a8", {specs()[1].name: .125})
    assert fp8.entries[1].storage_type == native.StorageType.FP8_W8A8
    with pytest.warns(UserWarning, match="lower precision"):
        decoded = run_convert(fp8.path, tmp_path / "decoded-fp8", "0 gate_proj bf16")
    values = (np.frombuffer(payload(decoded, 1), dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    np.testing.assert_allclose(values[::2], 1.0, atol=.04)
    np.testing.assert_allclose(values[1::2], 1.5, atol=.04)


def test_vision_only_source_uses_shared_decoder_and_preserves_bf16(tmp_path):
    vision_specs = (
        bf16.TensorSpec(832, -1, bf16.Role.VISION_PATCH_PROJ, "patch", "model.vision_tower.patch", (2, 3)),
        bf16.TensorSpec(833, -1, bf16.Role.VISION_POSITION_EMBEDDING, "position", "model.vision_tower.position", (2, 3)),
    )
    original = struct.pack("<6H", 0, 0x8000, 1, 0x3f80, 0x7f80, 0x7fc0)
    source = tmp_path / "vision-only.safetensors"
    write_safetensors(source, [(vision_specs[0].source_name, "BF16", (2, 3), original),
                               (vision_specs[1].source_name, "F8_E4M3", (2, 3), b"\x38" * 6)])
    output = tmp_path / "vision.safetensors"
    with patch.object(bf16, "vision_tensor_specs", return_value=vision_specs), pytest.warns(UserWarning, match="lower precision"):
        extract_vision.extract(source, output)
    records = bf16.read_safetensors_header(output)
    data = output.read_bytes()
    first, second = (records[s.source_name] for s in vision_specs)
    assert data[first.offset:first.offset + first.byte_length] == original
    assert data[second.offset:second.offset + second.byte_length] == b"\x80\x3f" * 6


@pytest.mark.parametrize("damage", ["missing", "shape", "scale", "duplicate", "index", "path"])
def test_invalid_structure_is_rejected(tmp_path, damage):
    path = packed_source(tmp_path)
    header_length, = struct.unpack("<Q", path.read_bytes()[:8])
    data = path.read_bytes()[8 + header_length:]
    header = json.loads(path.read_bytes()[8:8 + header_length])
    if damage == "missing":
        del header[specs()[0].source_name]
    elif damage == "shape":
        header[specs()[0].source_name]["shape"] = [16, 2]
    elif damage == "scale":
        header["gate.weight_scale_2"]["dtype"] = "U8"
    elif damage == "duplicate":
        (tmp_path / "duplicate.safetensors").write_bytes(path.read_bytes())
    elif damage in ("index", "path"):
        mapping = {name: path.name for name in header}
        mapping[specs()[0].source_name] = "../arbitrary.safetensors" if damage == "path" else "missing.safetensors"
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)
    with pytest.raises((bf16.ArtifactError, OSError)):
        read_snapshot(tmp_path, specs())


def test_plan_validates_without_writing_and_failed_conversion_is_not_published(tmp_path):
    source = packed_source(tmp_path / "source")
    output = tmp_path / "output"
    mask = tmp_path / "target.mask"
    mask.write_text("0 gate_proj bf16\n")
    with patch.object(bf16, "expected_tensor_specs", return_value=specs()), pytest.warns(UserWarning):
        assert convert.main(["--snapshot", str(source), "--mask", str(mask), "--plan", "--output", str(output)]) == 0
    assert not output.exists()
    with patch.object(bf16, "expected_tensor_specs", return_value=specs()), \
         patch.object(convert, "publish_serving_assets", side_effect=OSError("copy failed")):
        with pytest.raises(OSError, match="copy failed"):
            convert.convert(source, "", {}, output)
    assert not (output / "manifest.json").exists()
    assert not (output / "weights.gwt").exists()
    assert not list(output.glob("*.partial"))
