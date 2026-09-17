#!/usr/bin/env python3
"""Extract a compatible Gemma 4 vision tower and projector into BF16 safetensors."""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import bf16_artifact as bf16
from tools.component_weights import write_component
from tools.safetensors_source import read_snapshot, materialize_bf16, warn_precision
from tools.nvfp4_artifact import StorageType


def extract(snapshot: Path, output: Path) -> None:
    specs = bf16.vision_tensor_specs()
    source = read_snapshot(snapshot, specs)
    warn_precision(source.weights, [StorageType.BF16] * len(specs))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".vision-", dir=output.parent) as temporary:
        sources = [materialize_bf16(weight, Path(temporary) / str(i))
                   for i, weight in enumerate(source.weights)]
        source.assert_unchanged()
        write_component(output, specs, sources, metadata={"format": "pt", "component": "vision"})
        try:
            source.assert_unchanged()
        except BaseException:
            output.unlink()
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extract(args.snapshot, args.output)
    print(f"extracted vision + projector: {args.output}")


if __name__ == "__main__":
    main()
