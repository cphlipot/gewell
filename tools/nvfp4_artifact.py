#!/usr/bin/env python3
"""Native NVFP4 W4A4 and mixed FP8 W8A8 artifact packing and verification.

Logical tensors and the table layout are shared with the BF16 profile. Native
NVFP4 entries contain packed E2M1, swizzled E4M3 block scales, and two FP32 globals.
Source metadata is recorded for provenance, never used as a model allowlist.
"""
from __future__ import annotations

import hashlib
import math
import os
import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from tools import bf16_artifact as bf16
from tools.bf16_artifact import ArtifactError, Role, SourceTensor, TensorSpec

MAGIC = b"GEWNVF4\0"
FORMAT_VERSION = 2
FORMAT_NAME = "gemma4-31b-nvfp4-w4a4-v2"
MIXED_MAGIC = b"GEWMIX1\0"
MIXED_FORMAT_NAME = "gemma4-31b-mixed-v2"
MLP_ROLES = (Role.GATE_PROJ, Role.UP_PROJ, Role.DOWN_PROJ)
TARGET_ROLES = (Role.Q_PROJ, Role.K_PROJ, Role.V_PROJ, Role.O_PROJ, *MLP_ROLES)
DEFAULT_MASK = "* gate_proj nvfp4_w4a4\n* up_proj nvfp4_w4a4\n* down_proj nvfp4_w4a4\n"


class StorageType(IntEnum):
    BF16 = 0
    NVFP4_W4A4 = 1
    FP8_W8A8 = 2


@dataclass(frozen=True)
class TensorEntry(bf16.TensorEntry):
    storage_type: StorageType = StorageType.BF16


@dataclass(frozen=True)
class TensorSource:
    weight: SourceTensor
    storage_type: StorageType
    block_scales: SourceTensor | None = None
    globals: bytes = b""
    expected_sha256: bytes | None = None  # BF16 fallback entry integrity.
    payload_passthrough: bool = False  # A complete, already packed native entry.


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def packed_bytes(rows: int, columns: int) -> int:
    require(rows > 0 and columns > 0 and columns % 16 == 0, "NVFP4 dimensions must be positive with K divisible by 16")
    return rows * columns // 2


def scale_bytes(rows: int, columns: int) -> int:
    packed_bytes(rows, columns)
    return bf16.align_up(rows, 128) * bf16.align_up(columns // 16, 4)


def tensor_bytes(spec: TensorSpec, storage: StorageType) -> int:
    if storage == StorageType.BF16:
        return spec.byte_length
    if storage == StorageType.FP8_W8A8:
        require(spec.role in TARGET_ROLES and 0 <= spec.layer < 60 and len(spec.shape) == 2,
                f"FP8 storage is only supported for target text projections: {spec.name}")
        return fp8_packed_bytes(*spec.shape) + 8
    require(storage == StorageType.NVFP4_W4A4 and spec.role in TARGET_ROLES and
            0 <= spec.layer < 60 and len(spec.shape) == 2,
            f"NVFP4 storage is only supported for target text projections: {spec.name}")
    return packed_bytes(*spec.shape) + scale_bytes(*spec.shape) + 8


def fp8_packed_bytes(rows: int, columns: int) -> int:
    require(rows > 0 and columns > 0, "FP8 dimensions must be positive")
    return rows * columns


def parse_mask(text: str, specs: Sequence[TensorSpec], *,
               initial: Sequence[StorageType] | None = None) -> tuple[StorageType, ...]:
    require(len(text.encode()) <= 1024 * 1024, "mask exceeds 1 MiB")
    selected = list(initial) if initial is not None else [StorageType.BF16] * len(specs)
    require(len(selected) == len(specs), "initial mask length mismatch")
    roles = {bf16.ROLE_NAMES[role]: role for role in TARGET_ROLES}
    for line_number, raw in enumerate(text.splitlines(), 1):
        tokens = raw.partition("#")[0].split()
        if not tokens:
            continue
        label = f"mask line {line_number}"
        require(len(tokens) == 3, f"{label}: expected LAYER PROJECTION TYPE")
        layer, projection, dtype = tokens
        require(projection in roles, f"{label}: unknown target projection {projection}")
        types = {"bf16": StorageType.BF16, "nvfp4_w4a4": StorageType.NVFP4_W4A4,
                 "fp8_w8a8": StorageType.FP8_W8A8}
        require(dtype in types, f"{label}: unsupported native artifact type {dtype}")
        storage = types[dtype]
        require(layer == "*" or (layer.isascii() and layer.isdecimal() and 0 <= int(layer) < 60),
                f"{label}: layer must be * or an integer in [0, 59]")
        matches = [i for i, spec in enumerate(specs) if spec.role == roles[projection] and 0 <= spec.layer < 60
                   and (layer == "*" or spec.layer == int(layer))]
        require(bool(matches), f"{label}: projection does not exist for selected layer")
        for i in matches:
            selected[i] = storage
    return tuple(selected)


def validate_globals(raw: bytes) -> tuple[float, float]:
    require(len(raw) == 8, "packed global scales must occupy eight bytes")
    values = struct.unpack("<2f", raw)
    require(all(math.isfinite(x) and x > 0 for x in values), "packed global scales must be finite and positive")
    return values


def validate_gemm_globals(raw: bytes) -> tuple[float, float]:
    values = validate_globals(raw)
    weight_scale, input_scale = map(np.float32, values)
    with np.errstate(over="ignore", under="ignore"):
        inverse = np.float32(1) / input_scale
        output_scale = weight_scale * input_scale
    require(bool(np.isfinite(inverse) and np.isfinite(output_scale) and output_scale > 0),
            "native global scales are outside the executable FP32 range")
    return values


def swizzle_scale_chunk(raw: bytes, rows: int, columns: int) -> bytes:
    """Swizzle up to 128 source rows; output includes zero-filled tile padding."""
    require(0 < rows <= 128 and columns > 0, "invalid scale tile dimensions")
    require(len(raw) == rows * columns, "block scale byte length mismatch")
    source = np.frombuffer(raw, dtype=np.uint8).reshape(rows, columns)
    require(bool(np.all(source < 0x7f)), "NVFP4 block scales must be finite and nonnegative")
    padded = np.zeros((128, bf16.align_up(columns, 4)), dtype=np.uint8)
    padded[:rows, :columns] = source
    # [row_group4, row32, k_tile, k4] -> [k_tile, row32, row_group4, k4]
    return padded.reshape(4, 32, -1, 4).transpose(2, 1, 0, 3).copy().tobytes()


def encode_entry(entry: TensorEntry) -> bytes:
    return bf16.ENTRY_STRUCT.pack(entry.physical_id, entry.layer, int(entry.role), entry.rank,
                                 int(entry.storage_type), entry.dim0, entry.dim1, entry.dim2,
                                 entry.file_offset, entry.byte_length, entry.sha256)


def encode_header(*, tensor_count: int, data_offset: int, logical_bytes: int,
                  payload_bytes: int, file_bytes: int, table_sha256: bytes,
                  payload_sha256: bytes, config_sha256: bytes = bytes(32),
                  index_sha256: bytes = bytes(32), mixed: bool = False) -> bytes:
    raw = bytearray(bf16.HEADER_STRUCT.pack(
        MIXED_MAGIC if mixed else MAGIC, FORMAT_VERSION, bf16.HEADER_BYTES, bf16.ENTRY_BYTES,
        tensor_count, tensor_count + 1, 1, bf16.ALIGNMENT,
        bf16.SCALAR_TYPE_BF16_LE, bf16.LAYOUT_C_ORDER, bf16.TARGET_SM120A,
        bf16.HEADER_BYTES, data_offset, logical_bytes, payload_bytes, file_bytes,
        bf16.LM_HEAD_LOGICAL_ID, bf16.EMBED_PHYSICAL_ID,
        bytes(32), bytes(40), config_sha256, index_sha256, table_sha256,
        payload_sha256, b"\0" * 32))
    raw.extend(b"\0" * (bf16.HEADER_BYTES - len(raw)))
    raw[bf16.HEADER_HASH_OFFSET:bf16.HEADER_HASH_OFFSET + 32] = hashlib.sha256(raw).digest()
    return bytes(raw)


def read_metadata(path: Path, specs: Sequence[TensorSpec], *, native: bool = True) -> tuple[bf16.ArtifactHeader, tuple[TensorEntry, ...]]:
    """Validate format, shapes, storage and integrity independently of source identity."""
    bf16.validate_specs(specs)
    with path.open("rb") as source:
        raw = bf16._read_exact(source, bf16.HEADER_BYTES, "artifact header")
        header, fields = bf16.decode_header(raw)
        mixed = raw[:8] == MIXED_MAGIC
        if native:
            expected = encode_header(tensor_count=len(specs), data_offset=bf16.data_offset_for_count(len(specs)),
                                     logical_bytes=header.logical_data_bytes, payload_bytes=header.payload_bytes,
                                     file_bytes=header.file_bytes, table_sha256=header.entry_table_sha256,
                                     payload_sha256=header.payload_sha256, config_sha256=header.config_sha256,
                                     index_sha256=header.source_index_sha256, mixed=mixed)
            # Repository/revision bytes are informational in existing weight files.
            expected = bytearray(expected)
            expected[96:168] = raw[96:168]
            expected[bf16.HEADER_HASH_OFFSET:bf16.HEADER_HASH_OFFSET + 32] = bytes(32)
            expected[bf16.HEADER_HASH_OFFSET:bf16.HEADER_HASH_OFFSET + 32] = hashlib.sha256(expected).digest()
            require(raw == expected, "native artifact header/checksum mismatch")
        else:
            bf16._validate_header_fields(raw, header, fields, specs)
        require(header.file_bytes == path.stat().st_size, "artifact file size mismatch")
        table = bf16._read_exact(source, len(specs) * bf16.ENTRY_BYTES, "tensor table")
        require(hashlib.sha256(table).digest() == header.entry_table_sha256, "entry-table SHA-256 mismatch")
        padding = header.data_offset - source.tell()
        require(padding >= 0 and not any(bf16._read_exact(source, padding, "index padding")), "index padding is nonzero")
        entries = []
        cursor = header.data_offset
        logical_bytes = 0
        for i, spec in enumerate(specs):
            encoded = table[i * bf16.ENTRY_BYTES:(i + 1) * bf16.ENTRY_BYTES]
            fields = bf16.ENTRY_STRUCT.unpack(encoded)
            try:
                storage = StorageType(fields[4])
                role = Role(fields[2])
            except ValueError as error:
                raise ArtifactError(f"tensor {i} has unsupported role or storage") from error
            require(mixed or storage != StorageType.FP8_W8A8, f"tensor {i} has unsupported role or storage")
            require(native or storage == StorageType.BF16, f"baseline tensor {i} is not BF16")
            require(not any(encoded[-4:]), f"tensor {i} reserved tail is nonzero")
            entry = TensorEntry(fields[0], fields[1], role, fields[3], *fields[5:], storage)
            dims = (*spec.shape, *(0 for _ in range(3 - len(spec.shape))))
            require((entry.physical_id, entry.layer, entry.role, entry.rank, entry.dim0, entry.dim1, entry.dim2) ==
                    (i, spec.layer, spec.role, len(spec.shape), *dims), f"tensor {i} specification mismatch")
            require(entry.file_offset == cursor, f"tensor {i} offset mismatch")
            require(entry.byte_length == tensor_bytes(spec, storage), f"tensor {i} byte length mismatch")
            cursor += bf16.align_up(entry.byte_length)
            logical_bytes += entry.byte_length
            require(cursor <= header.file_bytes, f"tensor {i} exceeds artifact")
            entries.append(entry)
        require(cursor == header.file_bytes and logical_bytes == header.logical_data_bytes and
                cursor - header.data_offset == header.payload_bytes, "artifact totals mismatch")
        for entry in entries:
            if entry.storage_type != StorageType.BF16:
                source.seek(entry.file_offset + entry.byte_length - 8)
                validator = validate_gemm_globals if entry.storage_type == StorageType.FP8_W8A8 else validate_globals
                validator(bf16._read_exact(source, 8, "global scales"))
    return header, tuple(entries)


def _source_chunks(declaration: SourceTensor) -> Iterator[bytes]:
    with declaration.path.open("rb") as source:
        source.seek(declaration.offset)
        yield from bf16.iter_file_chunks(source, declaration.byte_length)


def write_artifact_partial(path: Path, specs: Sequence[TensorSpec], sources: Iterable[TensorSource], *,
                           config_sha256: bytes = bytes(32), index_sha256: bytes = bytes(32), mixed: bool = False
                           ) -> bf16.WrittenArtifact:
    bf16.validate_specs(specs)
    data_offset = bf16.data_offset_for_count(len(specs))
    entries = []
    payload_digest = hashlib.sha256()
    with path.open("xb") as output:
        output.write(b"\0" * data_offset)
        cursor = data_offset
        for spec, declaration in zip(specs, sources, strict=True):
            digest = hashlib.sha256()
            length = tensor_bytes(spec, declaration.storage_type)
            copied = 0

            def emit(chunk: bytes) -> None:
                nonlocal copied
                output.write(chunk)
                digest.update(chunk)
                payload_digest.update(chunk)
                copied += len(chunk)

            require(declaration.weight.source_name == spec.source_name, f"source name mismatch: {spec.name}")
            expected_weight_bytes = (length if declaration.payload_passthrough else
                                     spec.byte_length if declaration.storage_type == StorageType.BF16 else
                                     fp8_packed_bytes(*spec.shape) if declaration.storage_type == StorageType.FP8_W8A8 else
                                     packed_bytes(*spec.shape))
            require(declaration.weight.byte_length == expected_weight_bytes, f"source weight length mismatch: {spec.name}")
            for chunk in _source_chunks(declaration.weight):
                emit(chunk)
            if declaration.payload_passthrough:
                require(declaration.expected_sha256 is not None, "passthrough entry requires expected SHA-256")
                require(declaration.block_scales is None and not declaration.globals,
                        "passthrough entry has separate quantization metadata")
            elif declaration.storage_type == StorageType.FP8_W8A8:
                require(declaration.block_scales is None, "FP8 entry has block scales")
                validate_gemm_globals(declaration.globals)
                emit(declaration.globals)
            elif declaration.storage_type == StorageType.NVFP4_W4A4:
                validate_globals(declaration.globals)
                require(declaration.block_scales is not None, f"missing block scales: {spec.name}")
                rows, columns = spec.shape
                scale_columns = columns // 16
                require(declaration.block_scales.byte_length == rows * scale_columns, "source block scale length mismatch")
                with declaration.block_scales.path.open("rb") as source:
                    source.seek(declaration.block_scales.offset)
                    for row in range(0, rows, 128):
                        count = min(128, rows - row)
                        raw = bf16._read_exact(source, count * scale_columns, "source block scales")
                        emit(swizzle_scale_chunk(raw, count, scale_columns))
                emit(declaration.globals)
            else:
                require(declaration.block_scales is None and not declaration.globals, "BF16 entry has quantization metadata")
            require(copied == length, f"written tensor length mismatch: {spec.name}")
            if declaration.expected_sha256 is not None:
                require(digest.digest() == declaration.expected_sha256, f"BF16 fallback SHA-256 mismatch: {spec.name}")
            dims = (*spec.shape, *(0 for _ in range(3 - len(spec.shape))))
            entries.append(TensorEntry(spec.physical_id, spec.layer, spec.role, len(spec.shape), *dims,
                                       cursor, length, digest.digest(), declaration.storage_type))
            padding = b"\0" * (bf16.align_up(length) - length)
            output.write(padding)
            payload_digest.update(padding)
            cursor += bf16.align_up(length)
        table = b"".join(map(encode_entry, entries))
        raw = encode_header(tensor_count=len(specs), data_offset=data_offset,
                            logical_bytes=sum(x.byte_length for x in entries), payload_bytes=cursor - data_offset,
                            file_bytes=cursor, table_sha256=hashlib.sha256(table).digest(),
                            payload_sha256=payload_digest.digest(), config_sha256=config_sha256, index_sha256=index_sha256,
                            mixed=mixed or any(x.storage_type == StorageType.FP8_W8A8 for x in entries))
        output.seek(0)
        output.write(raw)
        output.write(table)
        output.flush()
        os.fsync(output.fileno())
    header, _ = bf16.decode_header(raw)
    return bf16.WrittenArtifact(path, header, tuple(entries), bf16.sha256_file(path))


def verify_artifact_file(path: Path, specs: Sequence[TensorSpec], *,
                         native: bool = True) -> bf16.VerifiedArtifact:
    header, entries = read_metadata(path, specs, native=native)
    payload_digest = hashlib.sha256()
    file_digest = hashlib.sha256()
    with path.open("rb") as source:
        file_digest.update(bf16._read_exact(source, header.data_offset, "artifact index"))
        for spec, entry in zip(specs, entries, strict=True):
            digest = hashlib.sha256()
            weight_remaining = fp8_packed_bytes(*spec.shape) if entry.storage_type == StorageType.FP8_W8A8 else 0
            for chunk in bf16.iter_file_chunks(source, entry.byte_length):
                if weight_remaining:
                    count = min(weight_remaining, len(chunk))
                    codes = np.frombuffer(chunk, dtype=np.uint8, count=count)
                    require(not np.any((codes & 0x7f) == 0x7f), "invalid FP8 weights: E4M3 NaN")
                    weight_remaining -= count
                digest.update(chunk)
                payload_digest.update(chunk)
                file_digest.update(chunk)
            require(digest.digest() == entry.sha256, f"tensor {spec.physical_id} SHA-256 mismatch")
            padding = bf16._read_exact(source, bf16.align_up(entry.byte_length) - entry.byte_length, "tensor padding")
            require(not any(padding), f"padding after tensor {spec.physical_id} is nonzero")
            payload_digest.update(padding)
            file_digest.update(padding)
            if entry.storage_type == StorageType.NVFP4_W4A4:
                next_offset = source.tell()
                rows, columns = spec.shape
                source.seek(entry.file_offset + packed_bytes(rows, columns))
                scale_columns = bf16.align_up(columns // 16, 4)
                for row in range(0, rows, 128):
                    raw = bf16._read_exact(source, 128 * scale_columns, "block scales")
                    tiled = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 32, 4, 4)
                    linear = tiled.transpose(2, 1, 0, 3).reshape(128, scale_columns)
                    count = min(rows - row, 128)
                    require(bool(np.all(linear[:count, :columns // 16] < 0x7f)), "invalid NVFP4 block scales")
                    require(not np.any(linear[count:]) and not np.any(linear[:, columns // 16:]), "NVFP4 block scale padding is nonzero")
                validate_globals(bf16._read_exact(source, 8, "global scales"))
                source.seek(next_offset)
        require(source.read(1) == b"", "artifact has trailing bytes")
    require(payload_digest.digest() == header.payload_sha256, "payload SHA-256 mismatch")
    return bf16.VerifiedArtifact(path, header, entries, file_digest.hexdigest())
