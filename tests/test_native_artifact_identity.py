"""CPU-only checks of structural native loading and projection storage validation."""

import hashlib
from pathlib import Path
import shutil
import struct
import subprocess

import pytest

from tools import bf16_artifact as bf16
from tools import nvfp4_artifact as native


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def artifact_loader(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for the artifact fixture")
    directory = tmp_path_factory.mktemp("artifact-loader")
    source = directory / "main.cc"
    source.write_text(r'''
#include "gewell/models/gemma4/31b/artifact.h"
#include <iostream>
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  try {
    const auto file = gewell::artifact::ArtifactFile::Open(argv[1]);
    unsigned packed = 0;
    for (const auto& entry : file.entries())
      packed += entry.storage_type == gewell::artifact::StorageType::nvfp4_w4a4;
    std::cout << file.header().native_nvfp4 << " " << packed << "\n";
    return 0;
  } catch (const gewell::artifact::Error& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
''')
    binary = directory / "artifact-loader"
    subprocess.run([compiler, "-std=c++17", "-I", str(ROOT / "include"), str(source),
                    str(ROOT / "src/models/gemma4/31b/artifact.cc"), "-lcrypto", "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    return binary


def write_sparse_metadata(path, source_identity, *, packed=True, selection=None):
    """Allocate only metadata/global scales; tensor payload hashes are not scanned."""
    specs = bf16.expected_tensor_specs()
    entries = []
    cursor = bf16.data_offset_for_count(len(specs))
    for spec in specs:
        storage = (native.StorageType.NVFP4_W4A4 if packed and spec.layer == 0 and
                   spec.role == bf16.Role.GATE_PROJ else native.StorageType.BF16)
        if selection is not None:
            storage = selection[spec.physical_id]
        length = native.tensor_bytes(spec, storage)
        dims = (*spec.shape, *(0 for _ in range(3 - len(spec.shape))))
        entries.append(native.TensorEntry(spec.physical_id, spec.layer, spec.role,
                                         len(spec.shape), *dims, cursor, length, b"\0" * 32, storage))
        cursor += bf16.align_up(length)
    table = b"".join(native.encode_entry(entry) for entry in entries)
    header = native.encode_header(tensor_count=len(specs), data_offset=bf16.data_offset_for_count(len(specs)),
                                  logical_bytes=sum(e.byte_length for e in entries),
                                  payload_bytes=cursor - bf16.data_offset_for_count(len(specs)),
                                  file_bytes=cursor, table_sha256=hashlib.sha256(table).digest(),
                                  payload_sha256=b"\0" * 32, config_sha256=hashlib.sha256(source_identity.encode()).digest(),
                                  index_sha256=bytes(range(32)),
                                  mixed=any(e.storage_type == native.StorageType.FP8_W8A8 for e in entries))
    with path.open("xb") as output:
        output.write(header)
        output.write(table)
        output.truncate(cursor)
        for entry in entries:
            if entry.storage_type != native.StorageType.BF16:
                output.seek(entry.file_offset + entry.byte_length - 8)
                output.write(struct.pack("<2f", .25, .125))
    return header


def replace_header(path, raw):
    raw = bytearray(raw)
    raw[bf16.HEADER_HASH_OFFSET:bf16.HEADER_HASH_OFFSET + 32] = b"\0" * 32
    raw[bf16.HEADER_HASH_OFFSET:bf16.HEADER_HASH_OFFSET + 32] = hashlib.sha256(raw).digest()
    with path.open("r+b") as output:
        output.write(raw)


@pytest.mark.parametrize("identity", ["community/finetune", "repacked"])
@pytest.mark.parametrize("packed", [False, True])
def test_native_loader_accepts_arbitrary_sources(artifact_loader, tmp_path, identity, packed):
    path = tmp_path / "weights"
    write_sparse_metadata(path, identity, packed=packed)
    result = subprocess.run([str(artifact_loader), str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"1 {int(packed)}"


@pytest.mark.parametrize("storage", ["bf16", "fp8_w8a8", "nvfp4_w4a4"])
def test_native_loader_all_target_projections(artifact_loader, tmp_path, storage):
    specs = bf16.expected_tensor_specs()
    mask = "".join(f"* {bf16.ROLE_NAMES[role]} {storage}\n" for role in native.TARGET_ROLES)
    selection = native.parse_mask(mask, specs)
    path = tmp_path / "weights"
    write_sparse_metadata(path, "google", selection=selection)
    result = subprocess.run([str(artifact_loader), str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"{0 if storage == 'fp8_w8a8' else 1} {410 if storage == 'nvfp4_w4a4' else 0}"


def test_native_loader_accepts_changed_provenance_but_checks_integrity(artifact_loader, tmp_path):
    path = tmp_path / "weights.gwt"
    raw = bytearray(write_sparse_metadata(path, "custom"))
    raw[96:128] = b"community/fine-tune".ljust(32, b"\0")
    raw[128:168] = b"arbitrary revision".ljust(40, b"\0")
    raw[168:232] = bytes(range(64))
    replace_header(path, raw)
    result = subprocess.run([str(artifact_loader), str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with path.open("r+b") as stream:
        stream.seek(168)
        stream.write(b"x")
    result = subprocess.run([str(artifact_loader), str(path)], capture_output=True, text=True)
    assert result.returncode == 1 and "header SHA-256" in result.stderr


@pytest.mark.parametrize("identity", ["community/finetune", "repacked"])
def test_bf16_loader_accepts_arbitrary_source(artifact_loader, tmp_path, identity):
    path = tmp_path / "weights"
    raw = bytearray(write_sparse_metadata(path, identity, packed=False))
    raw[:8] = bf16.MAGIC
    struct.pack_into("<I", raw, 8, bf16.FORMAT_VERSION)
    replace_header(path, raw)
    result = subprocess.run([str(artifact_loader), str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0 0"


@pytest.fixture(scope="module")
def serving_loader(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required")
    directory = tmp_path_factory.mktemp("serving-loader")
    source = directory / "main.cc"
    source.write_text('''
#include "gewell/models/gemma4/31b/serving_assets.h"
#include <iostream>
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  try {
    auto bundle = gewell::gemma4_31b::ServingAssets::Open(argv[1]);
    std::cout << bundle.weights.header().physical_tensor_count << "\\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\\n";
    return 1;
  }
}
''')
    binary = directory / "serving-loader"
    files = ["src/models/gemma4/31b/artifact.cc", "src/models/gemma4/31b/serving_assets.cc",
             "src/tokenizer.cc", "src/models/gemma4/text/contract.cc", "src/models/gemma4/text/chat_template.cc"]
    subprocess.run([compiler, "-std=c++17", "-I", str(ROOT / "include"), "-I", str(ROOT / "vendor/nlohmann"),
                    str(source), *(str(ROOT / file) for file in files), "-lcrypto", "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    return binary


def test_serving_accepts_custom_provenance_and_checks_bundle_integrity(tmp_path, request):
    import os
    import json
    from tools.serving_assets import publish_serving_assets, read_serving_assets
    snapshot = os.environ.get("GEWELL_TEST_SERVING_SNAPSHOT")
    if not snapshot:
        pytest.skip("set GEWELL_TEST_SERVING_SNAPSHOT for the native serving asset integration check")
    loader = request.getfixturevalue("serving_loader")
    path = tmp_path / "weights.gwt"
    write_sparse_metadata(path, "community/custom", packed=False)
    header, entries = native.read_metadata(path, bf16.expected_tensor_specs())
    assets = read_serving_assets(Path(snapshot))
    # Even tokenizer JSON serialization may change; compatibility is structural.
    assets["tokenizer.json"] = json.dumps(json.loads(assets["tokenizer.json"]), separators=(",", ":")).encode()
    serving = publish_serving_assets(assets, tmp_path)
    serving.update(repository="community/tokenizer", revision="arbitrary-revision")
    manifest = {"aliases": [], "format": native.FORMAT_NAME, "layout": {}, "schema_version": 6,
                "source": {}, "tensors": [], "serving": serving,
                "model": {"repository": "community/finetune", "revision": "any-revision",
                          "config_sha256": header.config_sha256.hex(), "index_sha256": header.source_index_sha256.hex()},
                "artifact": {"file": path.name, "file_bytes": header.file_bytes,
                             "header_sha256": header.header_sha256.hex(), "entry_table_sha256": header.entry_table_sha256.hex(),
                             "payload_sha256": header.payload_sha256.hex()}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert {path.name for path in tmp_path.iterdir()} == {"weights.gwt", "manifest.json", "tokenizer.json"}
    result = subprocess.run([str(loader), str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "832"
    (tmp_path / "tokenizer.json").write_bytes(b" " + assets["tokenizer.json"][1:])
    result = subprocess.run([str(loader), str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 1 and "SHA-256 mismatch" in result.stderr
