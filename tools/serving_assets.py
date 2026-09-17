#!/usr/bin/env python3
"""Package the tokenizer vocabulary and merges with bundle integrity checks."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bf16_artifact import ArtifactError


MANIFEST_SCHEMA_VERSION = 6
REQUIRED_ASSETS = ("tokenizer.json",)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def read_serving_assets(snapshot: Path) -> dict[str, bytes]:
    """Read and hash every source before publishing any destination file."""

    assets = {}
    for name in REQUIRED_ASSETS:
        try:
            data = (snapshot / name).read_bytes()
        except OSError as error:
            raise ArtifactError(f"cannot read serving asset {snapshot / name}: {error}") from error
        _require(bool(data), f"empty serving asset: {name}")
        assets[name] = data
    return assets


def serving_metadata(assets: dict[str, bytes]) -> dict[str, Any]:
    _require(set(assets) == set(REQUIRED_ASSETS), "serving asset inventory mismatch")
    result = {"repository": "", "revision": "", "assets": [
        {"file": name, "byte_length": len(assets[name]),
         "sha256": hashlib.sha256(assets[name]).hexdigest()}
        for name in REQUIRED_ASSETS
    ]}
    validate_serving_metadata(result)
    return result


def validate_serving_metadata(serving: Any) -> None:
    _require(isinstance(serving, dict) and set(serving) == {"repository", "revision", "assets"},
             "manifest serving must contain only repository, revision, and assets")
    _require(isinstance(serving["repository"], str) and isinstance(serving["revision"], str),
             "manifest serving provenance must be strings")
    assets = serving["assets"]
    _require(isinstance(assets, list) and len(assets) == len(REQUIRED_ASSETS),
             "manifest serving asset inventory mismatch")
    for name, record in zip(REQUIRED_ASSETS, assets, strict=True):
        _require(isinstance(record, dict) and set(record) == {"file", "byte_length", "sha256"},
                 f"manifest serving asset keys differ: {name}")
        _require(record["file"] == name, "manifest serving asset filenames or order mismatch")
        _require(type(record["byte_length"]) is int and record["byte_length"] > 0,
                 f"invalid serving asset byte length: {name}")
        _require(isinstance(record["sha256"], str) and len(record["sha256"]) == 64 and
                 all(c in "0123456789abcdef" for c in record["sha256"]),
                 f"manifest serving asset SHA-256 mismatch: {name}")


def validate_serving_assets(directory: Path, serving: Any) -> None:
    validate_serving_metadata(serving)
    for record in serving["assets"]:
        path = directory / record["file"]
        _require(not path.is_symlink(), f"refusing serving asset symlink: {path}")
        try:
            _require(path.stat().st_size == record["byte_length"],
                     f"serving asset byte length mismatch: {path.name}")
            data = path.read_bytes()
        except OSError as error:
            raise ArtifactError(f"cannot read serving asset {path}: {error}") from error
        _require(hashlib.sha256(data).hexdigest() == record["sha256"],
                 f"serving asset SHA-256 mismatch: {path.name}")


def _sync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_temporary(directory: Path, name: str, data: bytes) -> Path:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".partial", dir=directory)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def publish_serving_assets(assets: dict[str, bytes], directory: Path) -> dict[str, Any]:
    """Publish real files; matching files from an interrupted package can be reused."""

    serving = serving_metadata(assets)
    directory.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_ASSETS:
        path = directory / name
        _require(not path.is_symlink(), f"refusing serving asset symlink: {path}")
        if path.exists():
            _require(path.is_file() and path.read_bytes() == assets[name],
                     f"refusing to replace existing serving asset: {path}")
            continue
        temporary = _write_temporary(directory, name, assets[name])
        try:
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    _sync_directory(directory)
    validate_serving_assets(directory, serving)
    return serving
