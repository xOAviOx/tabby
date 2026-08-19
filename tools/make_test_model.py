#!/usr/bin/env python3
"""
Generate a tiny synthetic model in Hugging Face layout, for tests.

The real Qwen conversion is ~1 GB, which is the right thing to gate on but the wrong
thing to run in every test. This builds a few-kilobyte model with the same *shape* of
problems -- tied embeddings, GQA, rank-1 f32 tensors, and (with a small --chunk-bytes)
tensors that span several chunk files -- so the loader's interesting paths are covered
on any machine in milliseconds.

Values are deterministic, derived from a fixed seed, so the fixture hashes are stable
across regenerations.

Usage:
    python3 tools/make_test_model.py --hf-out models/tiny-test \\
        --out public/models/tiny-test --chunk-bytes 4096
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent

# Deliberately small, deliberately awkward: vocab is not a multiple of anything, the
# head count divides hidden_size but num_key_value_heads < num_attention_heads.
CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "bos_token_id": 3,
    "eos_token_id": 5,
    "hidden_act": "silu",
    "hidden_size": 32,
    "intermediate_size": 61,
    "max_position_embeddings": 128,
    "model_type": "qwen2",
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "vocab_size": 203,
}


def f32_to_bf16_bytes(values: np.ndarray) -> bytes:
    """Truncate f32 to bf16 (keep the high 16 bits). Exactly invertible by convert.py."""
    u32 = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return (u32 >> np.uint32(16)).astype("<u2").tobytes()


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, dict] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, array in tensors.items():
        payload = f32_to_bf16_bytes(array)
        header[name] = {
            "dtype": "BF16",
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        blobs.append(payload)
        offset += len(payload)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # safetensors requires the data section to start 8-byte aligned.
    pad = (-len(header_bytes)) % 8
    header_bytes += b" " * pad

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        for blob in blobs:
            fh.write(blob)


def build_tensors(config: dict) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260819)
    hidden = config["hidden_size"]
    inter = config["intermediate_size"]
    vocab = config["vocab_size"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = hidden // heads
    kv_dim = kv_heads * head_dim

    def normal(*shape: int) -> np.ndarray:
        return rng.normal(0.0, 0.02, size=shape).astype(np.float32)

    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": normal(vocab, hidden),
    }
    for layer in range(config["num_hidden_layers"]):
        p = f"model.layers.{layer}."
        tensors[p + "input_layernorm.weight"] = np.ones(hidden, dtype=np.float32)
        tensors[p + "self_attn.q_proj.weight"] = normal(hidden, hidden)
        tensors[p + "self_attn.q_proj.bias"] = normal(hidden)
        tensors[p + "self_attn.k_proj.weight"] = normal(kv_dim, hidden)
        tensors[p + "self_attn.k_proj.bias"] = normal(kv_dim)
        tensors[p + "self_attn.v_proj.weight"] = normal(kv_dim, hidden)
        tensors[p + "self_attn.v_proj.bias"] = normal(kv_dim)
        tensors[p + "self_attn.o_proj.weight"] = normal(hidden, hidden)
        tensors[p + "post_attention_layernorm.weight"] = np.ones(hidden, dtype=np.float32)
        tensors[p + "mlp.gate_proj.weight"] = normal(inter, hidden)
        tensors[p + "mlp.up_proj.weight"] = normal(inter, hidden)
        tensors[p + "mlp.down_proj.weight"] = normal(hidden, inter)
    tensors["model.norm.weight"] = np.ones(hidden, dtype=np.float32)
    return tensors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-out", type=Path, default=Path("models/tiny-test"))
    parser.add_argument("--out", type=Path, default=Path("public/models/tiny-test"))
    parser.add_argument("--fixture", type=Path, default=Path("tests/fixtures/weights-tiny-test.json"))
    parser.add_argument("--chunk-bytes", type=int, default=4096)
    parser.add_argument("--skip-convert", action="store_true")
    args = parser.parse_args(argv)

    args.hf_out.mkdir(parents=True, exist_ok=True)
    (args.hf_out / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    tensors = build_tensors(CONFIG)
    write_safetensors(args.hf_out / "model.safetensors", tensors)
    total = sum(a.size for a in tensors.values())
    print(f"wrote {args.hf_out} ({len(tensors)} tensors, {total} params)")

    if args.skip_convert:
        return 0

    subprocess.run(
        [sys.executable, str(TOOLS / "convert.py"), str(args.hf_out),
         "--out", str(args.out), "--chunk-bytes", str(args.chunk_bytes)],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(TOOLS / "dump_expected.py"), str(args.hf_out),
         "--model-id", args.out.name, "--out", str(args.fixture),
         "--tensors",
         "model.embed_tokens.weight",
         "model.layers.1.self_attn.k_proj.bias",
         "model.layers.1.mlp.down_proj.weight"],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
