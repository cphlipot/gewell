"""Write independently stored BF16 components using original safetensors names."""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Sequence

from tools import bf16_artifact as bf16


def write_component(path: Path, specs: Sequence[bf16.TensorSpec],
                    sources: Sequence[bf16.SourceTensor], *, metadata: dict[str, str],
                    expected_hashes: Sequence[bytes] | None = None) -> None:
    bf16._require(len(specs) == len(sources) and bool(specs), "component source count mismatch")
    bf16._require(not path.exists() and not path.is_symlink(), f"refusing to replace {path}")
    header = {"__metadata__": metadata}
    cursor = 0
    for spec, source in zip(specs, sources, strict=True):
        bf16._require(source.source_name == spec.source_name and source.byte_length == spec.byte_length,
                      f"component source mismatch: {spec.source_name}")
        bf16._require(spec.source_name not in header, "duplicate component tensor")
        header[spec.source_name] = {"dtype": "BF16", "shape": list(spec.shape),
                                    "data_offsets": [cursor, cursor + spec.byte_length]}
        cursor += spec.byte_length
    raw = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    raw += b" " * (bf16.align_up(len(raw), 8) - len(raw))
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    # Exclusive creation keeps a concurrent export's temporary file untouched.
    with partial.open("xb") as output:
        try:
            output.write(struct.pack("<Q", len(raw)))
            output.write(raw)
            for i, source in enumerate(sources):
                digest = hashlib.sha256()
                with source.path.open("rb") as stream:
                    stream.seek(source.offset)
                    for chunk in bf16.iter_file_chunks(stream, source.byte_length):
                        digest.update(chunk)
                        output.write(chunk)
                if expected_hashes is not None:
                    bf16._require(digest.digest() == expected_hashes[i],
                                  f"component source checksum mismatch: {source.source_name}")
            output.flush()
            os.fsync(output.fileno())
            bf16.read_safetensors_header(partial)
            os.link(partial, path)
        finally:
            partial.unlink(missing_ok=True)
