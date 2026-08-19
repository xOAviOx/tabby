#!/usr/bin/env python3
"""
Emit the expected-weights fixture the browser test byte-matches against.

Reads tensors **directly from the source safetensors file**, applies the same dtype
conversion `convert.py` applies, and records a sha256 over the resulting bytes plus a
head/tail sample. The browser test reads the same tensors back off the GPU, hashes them
in the page, and compares -- so a passing test means the bytes survived conversion,
chunking, OPFS, sharding and GPU upload without a single byte moving.

The conversion routine is shared with convert.py on purpose. The independent
implementation being tested is the *browser loader*, not a second copy of the encoder;
duplicating the encoder here would only test that two Python functions agree.

Usage:
    python3 tools/dump_expected.py models/Qwen2.5-0.5B-Instruct \\
        --model-id qwen2.5-0.5b-instruct \\
        --out tests/fixtures/weights-qwen2.5-0.5b-instruct.json
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert import (  # noqa: E402
    ConversionError,
    encode_tensor,
    open_shards,
    target_dtype,
)

# Chosen to cover materially different cases rather than three of the same thing:
#   - the embedding matrix: largest tensor, spans many chunks, first in the stream
#   - a rank-1 bias: f32 path, tiny, unaligned length
#   - a last-layer matrix: deep in the stream, exercises offset arithmetic at the end
DEFAULT_TENSORS = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.bias",
    "model.norm.weight",
]

SAMPLE_BYTES = 256


def build_fixture(
    model_dir: Path,
    model_id: str,
    names: list[str],
    matrix_dtype: str,
) -> dict:
    shards = open_shards(model_dir)
    entries = []

    for name in names:
        if name not in shards:
            raise ConversionError(
                f"{name!r} is not in {model_dir}; available example: "
                f"{sorted(shards)[0]!r}"
            )
        source = shards[name].read(name)
        shape = tuple(int(d) for d in source.shape)
        dtype = target_dtype(name, shape, matrix_dtype)
        payload = encode_tensor(source, dtype).tobytes()

        entries.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "byteLength": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "head": base64.b64encode(payload[:SAMPLE_BYTES]).decode("ascii"),
                "tail": base64.b64encode(payload[-SAMPLE_BYTES:]).decode("ascii"),
            }
        )
        print(f"  {name}: {len(payload)} bytes, sha256 {entries[-1]['sha256'][:16]}...")

    return {
        "modelId": model_id,
        "source": str(model_dir),
        "sampleBytes": SAMPLE_BYTES,
        "tensors": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path, help="Hugging Face model directory")
    parser.add_argument("--model-id", required=True, help="id used by the browser loader")
    parser.add_argument("--out", type=Path, required=True, help="fixture JSON path")
    parser.add_argument("--tensors", nargs="*", default=None, help="tensor names to dump")
    parser.add_argument("--matrix-dtype", choices=["f16", "f32"], default="f16")
    args = parser.parse_args(argv)

    names = args.tensors if args.tensors else DEFAULT_TENSORS
    print(f"dumping {len(names)} expected tensors from {args.model_dir}")
    try:
        fixture = build_fixture(args.model_dir, args.model_id, names, args.matrix_dtype)
    except ConversionError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
