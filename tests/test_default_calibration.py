"""Reference calibration is usable without model execution or source pinning."""
import hashlib
import json
import math
import struct
from unittest.mock import patch

import pytest

from tools import bf16_artifact as bf16
from tools import convert
from tools import nvfp4_artifact as native
from tests.test_safetensors_conversion import payload, run_convert, serving, specs, write_safetensors


def fp32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def snapshot(directory, *, encoding="BF16", input_scale=None):
    serving(directory)
    # Conversion must not compare the profile's provenance to this source.
    (directory / "config.json").write_text('{"source":"custom-finetune"}')
    tensors = []
    for spec in specs():
        dtype = encoding if spec.role == bf16.Role.GATE_PROJ else "BF16"
        rows, columns = spec.shape
        if dtype == "BF16":
            tensors.append((spec.source_name, dtype, spec.shape, b"\x80\x3f" * (rows * columns)))
            continue
        if dtype == "U8":
            tensors.extend([
                (spec.source_name, dtype, (rows, columns // 2), b"\x76" * (rows * columns // 2)),
                ("gate.weight_scale", "F8_E4M3", (rows, columns // 16), b"\x38" * (rows * columns // 16)),
                ("gate.weight_scale_2", "F32", (), struct.pack("<f", .25)),
            ])
        else:
            tensors.extend([
                (spec.source_name, dtype, spec.shape, b"\x38" * (rows * columns)),
                ("gate.weight_scale", "F32", (), struct.pack("<f", .25)),
            ])
        if input_scale is not None:
            tensors.append(("gate.input_scale", "F32", (), struct.pack("<f", input_scale)))
    write_safetensors(directory / "model.safetensors", tensors)
    return directory


def manifest(artifact):
    return json.loads((artifact.path.parent / "manifest.json").read_text())


def test_reference_profile_covers_every_projection_in_both_formats(capsys):
    profile = bf16.load_json_object(convert.DEFAULT_CALIBRATION)
    production = bf16.expected_tensor_specs()
    expected = {s.name for s in production if s.role in native.TARGET_ROLES}
    assert len(expected) == 410
    assert profile["input_amax"].keys() == expected
    assert all(math.isfinite(value) and value > 0 for value in profile["input_amax"].values())
    assert profile["provenance"]["histories"] == 74
    assert profile["input_amax"]["layers.0.self_attn.q_proj.weight"] == 1264
    for storage, divisor in ((native.StorageType.FP8_W8A8, 448), (native.StorageType.NVFP4_W4A4, 2688)):
        selected = [storage if s.name in expected else native.StorageType.BF16 for s in production]
        scales, calibration = convert.resolve_input_scales(production, selected, {}, {})
        assert scales == {name: fp32(value / divisor) for name, value in profile["input_amax"].items()}
        assert set(calibration["origins"].values()) == {"default"}
    assert capsys.readouterr().err.count("Using bundled reference calibration") == 2


@pytest.mark.parametrize("native_source", [False, True])
def test_bf16_conversion_and_plan_use_defaults_without_gpu(tmp_path, native_source, capsys):
    source = snapshot(tmp_path / "source")
    if native_source:
        source = run_convert(source, tmp_path / "baseline").path
    mask = tmp_path / "target.mask"
    mask.write_text("0 gate_proj nvfp4_w4a4\n0 up_proj fp8_w8a8\n")
    output = tmp_path / "output"
    with patch.object(bf16, "expected_tensor_specs", return_value=specs()):
        assert convert.main(["--artifact" if native_source else "--snapshot", str(source),
                             "--mask", str(mask), "--output", str(output), "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["activation_scales"] == {"explicit": 0, "source": 0, "default": 2}
    assert not output.exists()
    artifact = run_convert(source, output, mask.read_text())
    metadata = manifest(artifact)
    assert artifact.path.stat().st_size == plan["file_bytes"]
    expected = {specs()[1].name: fp32(82 / 2688), specs()[2].name: fp32(82 / 448)}
    assert metadata["source"]["input_scales"] == expected
    for i in (1, 2):
        assert struct.unpack("<f", payload(artifact, i)[-4:])[0] == expected[specs()[i].name]
    calibration = metadata["source"]["calibration"]
    assert calibration["origins"] == dict.fromkeys(expected, "default")
    assert calibration["default_profile"] == plan["default_calibration"]
    assert calibration["default_profile"]["sha256"] == hashlib.sha256(convert.DEFAULT_CALIBRATION.read_bytes()).hexdigest()
    assert capsys.readouterr().err.count("Using bundled reference calibration") == 1


@pytest.mark.parametrize("encoding,divisor", [("F8_E4M3", 448), ("U8", 2688)])
def test_missing_source_activation_scale_uses_reference(tmp_path, encoding, divisor):
    source = snapshot(tmp_path / "source", encoding=encoding)
    artifact = run_convert(source, tmp_path / "output")
    assert struct.unpack("<2f", payload(artifact, 1)[-8:]) == (.25, fp32(82 / divisor))
    assert manifest(artifact)["source"]["calibration"]["origins"] == {specs()[1].name: "default"}


@pytest.mark.parametrize("native_source", [False, True])
@pytest.mark.parametrize("encoding", ["F8_E4M3", "U8"])
def test_explicit_over_source_and_byte_exact_weight_retention(tmp_path, native_source, encoding, capsys):
    source = snapshot(tmp_path / "source", encoding=encoding, input_scale=.125)
    original = run_convert(source, tmp_path / "original")
    if native_source:
        source = original.path
    # Defaults must not be read when source/explicit scales suffice.
    with patch.object(convert, "DEFAULT_CALIBRATION", tmp_path / "does-not-exist"):
        retained = run_convert(source, tmp_path / "retained")
        changed = run_convert(source, tmp_path / "changed", scales={specs()[1].name: .375})
    assert all(payload(retained, i) == payload(original, i) for i in range(len(specs())))
    assert payload(changed, 1)[:-4] == payload(original, 1)[:-4]
    assert payload(changed, 1)[-4:] == struct.pack("<f", .375)
    for artifact, origin in ((retained, "source"), (changed, "explicit")):
        assert manifest(artifact)["source"]["calibration"] == {"origins": {specs()[1].name: origin}}
    assert not capsys.readouterr().err


@pytest.mark.parametrize("native_source", [False, True])
def test_requantization_uses_target_default_and_partial_override(tmp_path, native_source):
    source = snapshot(tmp_path / "source", encoding="F8_E4M3", input_scale=.125)
    if native_source:
        source = run_convert(source, tmp_path / "original").path
    artifact = run_convert(source, tmp_path / "output", "0 gate_proj nvfp4_w4a4\n0 up_proj fp8_w8a8",
                           {specs()[2].name: .0625})
    assert struct.unpack("<f", payload(artifact, 1)[-4:])[0] == fp32(82 / 2688)
    assert struct.unpack("<f", payload(artifact, 2)[-4:])[0] == .0625
    assert manifest(artifact)["source"]["calibration"]["origins"] == {
        specs()[1].name: "default", specs()[2].name: "explicit"}


@pytest.mark.parametrize("value", [None, 0, -1, False, float("nan"), float("inf"), 1e-100, 1e100, "0.125"])
def test_invalid_explicit_scale_does_not_fall_back(tmp_path, value):
    source = snapshot(tmp_path / "source", encoding="F8_E4M3", input_scale=.125)
    with pytest.raises(bf16.ArtifactError):
        run_convert(source, tmp_path / "output", scales={specs()[1].name: value})
    assert not (tmp_path / "output").exists()


def test_unknown_override_does_not_silently_use_defaults(tmp_path):
    source = snapshot(tmp_path / "source")
    with pytest.raises(bf16.ArtifactError, match="unknown input scale names"):
        run_convert(source, tmp_path / "output", "0 gate_proj fp8_w8a8", {"layers.0.gate_proj.weight": .125})


def test_bf16_conversion_needs_no_calibration(tmp_path, capsys):
    source = snapshot(tmp_path / "source")
    with patch.object(convert, "DEFAULT_CALIBRATION", tmp_path / "does-not-exist"):
        artifact = run_convert(source, tmp_path / "output")
    assert manifest(artifact)["source"]["input_scales"] == {}
    assert manifest(artifact)["source"]["calibration"] == {"origins": {}}
    assert not capsys.readouterr().err
