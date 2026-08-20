#!/usr/bin/env python3
"""
Convert a Hugging Face model directory into the engine's weight format.

Emits, into --out:

    model.json        header: full config + chunk directory + tensor directory
    model-000.bin ...  weight bytes, split into fixed-size chunks

The chunk files are a transport detail only. Logically they concatenate into one
contiguous byte stream, and every tensor records its offset into *that stream*, so a
tensor larger than one chunk simply spans several -- which the 272 MB embedding matrix
always will. The browser loader reassembles ranges across chunk boundaries.

Rank-2 weights are quantized block-wise (int4 by default); rank-1 tensors -- RMSNorm
gains and attention biases -- stay f32. They are a rounding error of the total size and
quantizing them costs accuracy for nothing.

Safetensors is parsed here directly rather than through the `safetensors` package.
That is not gratuitous: the package's numpy API cannot return BF16 tensors at all
(numpy has no bfloat16 dtype) and Qwen2.5 ships bf16, so the widening has to be done
by hand regardless. Doing the whole parse by hand costs ~40 lines and removes a
dependency. The format is a u64 header length, a JSON header, then raw tensor bytes.

Usage:
    python3 tools/convert.py models/Qwen2.5-0.5B-Instruct \\
        --out public/models/qwen2.5-0.5b-instruct
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

FORMAT_NAME = "browser-llm-weights"
FORMAT_VERSION = 1

DEFAULT_CHUNK_BYTES = 32 * 1024 * 1024

# Every tensor starts on this boundary within the logical stream. GPU buffer writes
# want 4-byte alignment at minimum; 16 keeps vec4 loads (M5) naturally aligned too.
TENSOR_ALIGNMENT = 16

# Config keys copied verbatim into the header. Nothing downstream may hardcode any of
# these -- the engine reads them back at load time. A missing required key is fatal
# rather than defaulted, so an unsupported architecture fails here and not at M2.
REQUIRED_CONFIG_KEYS = [
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "vocab_size",
    "rope_theta",
    "rms_norm_eps",
    "tie_word_embeddings",
]

OPTIONAL_CONFIG_KEYS = [
    "head_dim",
    "max_position_embeddings",
    "model_type",
    "architectures",
    "bos_token_id",
    "eos_token_id",
    "hidden_act",
    "attention_bias",
    "sliding_window",
    "torch_dtype",
]

# Quantization block, along the reduction (input) dimension. One scale per block.
DEFAULT_BLOCK_SIZE = 32

# safetensors dtype string -> (numpy dtype to read raw bytes as, element size)
SAFETENSORS_DTYPES: dict[str, np.dtype] = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("<i1"),
    "U8": np.dtype("<u1"),
    "BOOL": np.dtype("?"),
}


class ConversionError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------
# safetensors reading
# --------------------------------------------------------------------------------------


class SafetensorsFile:
    """Memory-mapped reader for a single .safetensors file."""

    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as fh:
            raw_len = fh.read(8)
            if len(raw_len) != 8:
                raise ConversionError(f"{path}: too short to be a safetensors file")
            (header_len,) = struct.unpack("<Q", raw_len)
            header_bytes = fh.read(header_len)
            if len(header_bytes) != header_len:
                raise ConversionError(f"{path}: truncated header")
        self.header: dict[str, Any] = json.loads(header_bytes)
        self.metadata = self.header.pop("__metadata__", {})
        self._data_start = 8 + header_len
        self._mmap = np.memmap(path, dtype=np.uint8, mode="r")

    def names(self) -> list[str]:
        return list(self.header.keys())

    def dtype_of(self, name: str) -> str:
        return self.header[name]["dtype"]

    def shape_of(self, name: str) -> list[int]:
        return list(self.header[name]["shape"])

    def read(self, name: str) -> np.ndarray:
        """Return the tensor as float32 (widening bf16 by hand) or its native dtype."""
        entry = self.header.get(name)
        if entry is None:
            raise ConversionError(f"{self.path}: no tensor named {name!r}")
        begin, end = entry["data_offsets"]
        raw = self._mmap[self._data_start + begin : self._data_start + end]
        shape = tuple(entry["shape"])
        dtype = entry["dtype"]

        if dtype == "BF16":
            # bf16 is the top 16 bits of an f32, so widening is exact: shift into place
            # and reinterpret. numpy cannot represent bf16 natively, hence the manual path.
            u16 = raw.view("<u2")
            u32 = u16.astype(np.uint32) << np.uint32(16)
            return u32.view(np.float32).reshape(shape)

        np_dtype = SAFETENSORS_DTYPES.get(dtype)
        if np_dtype is None:
            raise ConversionError(f"{name}: unsupported safetensors dtype {dtype!r}")
        return raw.view(np_dtype).reshape(shape)


def open_shards(model_dir: Path) -> dict[str, SafetensorsFile]:
    """Map tensor name -> the shard file holding it, honouring the HF index if present."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        files = {name: model_dir / f for name, f in index["weight_map"].items()}
        opened: dict[Path, SafetensorsFile] = {}
        result: dict[str, SafetensorsFile] = {}
        for name, path in files.items():
            if path not in opened:
                opened[path] = SafetensorsFile(path)
            result[name] = opened[path]
        return result

    single = model_dir / "model.safetensors"
    if not single.exists():
        raise ConversionError(f"{model_dir}: no model.safetensors or index file")
    handle = SafetensorsFile(single)
    return {name: handle for name in handle.names()}


# --------------------------------------------------------------------------------------
# chunked output stream
# --------------------------------------------------------------------------------------


class ChunkWriter:
    """
    Append-only writer over a logical byte stream that is materialised as fixed-size
    chunk files. Callers see one continuous offset space and never think about chunks.
    """

    def __init__(self, out_dir: Path, chunk_bytes: int, prefix: str = "model"):
        self.out_dir = out_dir
        self.chunk_bytes = chunk_bytes
        self.prefix = prefix
        self.offset = 0
        self._chunks: list[dict[str, Any]] = []
        self._fh = None
        self._hash = None
        self._written_in_chunk = 0
        out_dir.mkdir(parents=True, exist_ok=True)

    def _chunk_name(self, index: int) -> str:
        return f"{self.prefix}-{index:03d}.bin"

    def _open_next(self) -> None:
        name = self._chunk_name(len(self._chunks))
        self._fh = (self.out_dir / name).open("wb")
        self._hash = hashlib.sha256()
        self._written_in_chunk = 0

    def _close_current(self) -> None:
        if self._fh is None:
            return
        self._fh.close()
        self._chunks.append(
            {
                "name": self._chunk_name(len(self._chunks)),
                "bytes": self._written_in_chunk,
                "sha256": self._hash.hexdigest(),
            }
        )
        self._fh = None

    def write(self, data: bytes | memoryview) -> None:
        view = memoryview(data)
        while len(view) > 0:
            if self._fh is None:
                self._open_next()
            room = self.chunk_bytes - self._written_in_chunk
            piece = view[:room]
            self._fh.write(piece)
            self._hash.update(piece)
            self._written_in_chunk += len(piece)
            self.offset += len(piece)
            view = view[len(piece) :]
            if self._written_in_chunk >= self.chunk_bytes:
                self._close_current()

    def align(self, alignment: int) -> None:
        pad = (-self.offset) % alignment
        if pad:
            self.write(b"\0" * pad)

    def close(self) -> list[dict[str, Any]]:
        self._close_current()
        return self._chunks


# --------------------------------------------------------------------------------------
# conversion
# --------------------------------------------------------------------------------------


def quantize_blockwise(
    values: np.ndarray,
    bits: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Symmetric block-wise quantization of a [rows, cols] matrix along `cols`.

    Returns (packed u32 [rows, cols*bits/32], scales f16 [rows, cols/block_size]).

    The int4 path follows llama.cpp's Q4_0: the scale is derived from the *signed* element
    of largest magnitude so that element maps exactly onto the low end of the range, which
    uses all 16 levels instead of wasting one. Dequantization is `(nibble - 8) * scale`.
    """
    rows, cols = values.shape
    if cols % block_size != 0:
        raise ConversionError(f"cols {cols} is not a multiple of block size {block_size}")

    n_blocks = cols // block_size
    blocks = np.ascontiguousarray(values, dtype=np.float32).reshape(rows, n_blocks, block_size)

    if bits == 4:
        # The signed value with the largest absolute magnitude in each block.
        peak_index = np.argmax(np.abs(blocks), axis=2)
        peak = np.take_along_axis(blocks, peak_index[..., None], axis=2)[..., 0]
        scales = (peak / -8.0).astype(np.float32)
        inverse = np.where(scales != 0.0, 1.0 / np.where(scales == 0.0, 1.0, scales), 0.0)
        levels = np.clip(np.rint(blocks * inverse[..., None] + 8.0), 0, 15).astype(np.uint32)
        per_word = 8
    elif bits == 8:
        amax = np.max(np.abs(blocks), axis=2)
        scales = (amax / 127.0).astype(np.float32)
        inverse = np.where(scales != 0.0, 1.0 / np.where(scales == 0.0, 1.0, scales), 0.0)
        signed = np.clip(np.rint(blocks * inverse[..., None]), -128, 127).astype(np.int32)
        levels = (signed & 0xFF).astype(np.uint32)
        per_word = 4
    else:
        raise ConversionError(f"unsupported bit width {bits}; use 4 or 8")

    flat = levels.reshape(rows, cols)
    words = np.zeros((rows, cols // per_word), dtype=np.uint32)
    grouped = flat.reshape(rows, cols // per_word, per_word)
    shift = 32 // per_word
    for lane in range(per_word):
        words |= grouped[:, :, lane] << np.uint32(shift * lane)

    return words, scales.astype(np.float16)


def dequantize_blockwise(
    words: np.ndarray,
    scales: np.ndarray,
    cols: int,
    bits: int,
    block_size: int,
) -> np.ndarray:
    """Inverse of quantize_blockwise, used to report the error the conversion introduced."""
    rows = words.shape[0]
    per_word = 8 if bits == 4 else 4
    shift = 32 // per_word
    mask = (1 << shift) - 1

    lanes = []
    for lane in range(per_word):
        lanes.append((words >> np.uint32(shift * lane)) & np.uint32(mask))
    flat = np.stack(lanes, axis=2).reshape(rows, cols).astype(np.int32)

    if bits == 4:
        centred = flat - 8
    else:
        centred = np.where(flat > 127, flat - 256, flat)

    blocks = centred.reshape(rows, cols // block_size, block_size).astype(np.float32)
    return (blocks * scales.astype(np.float32)[..., None]).reshape(rows, cols)


def target_dtype(name: str, shape: tuple[int, ...], matrix_dtype: str) -> str:
    """
    Rank-2 weights carry essentially all the parameters and go to `matrix_dtype`.
    Rank-1 tensors -- RMSNorm gains and Qwen's q/k/v biases -- stay f32: they are a
    rounding error of the total size and are exactly what M5 says to keep in high
    precision, so there is nothing to gain by narrowing them.
    """
    return matrix_dtype if len(shape) >= 2 else "f32"


def encode_tensor(array: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "f16":
        out = array.astype(np.float16)
    elif dtype == "f32":
        out = array.astype(np.float32)
    else:
        raise ConversionError(f"unsupported target dtype {dtype!r}")
    return np.ascontiguousarray(out)


def check_overflow(name: str, source: np.ndarray, encoded: np.ndarray) -> int:
    """Count values that were finite before encoding and are not after."""
    if encoded.dtype != np.float16:
        return 0
    lost = int(np.count_nonzero(np.isfinite(source) & ~np.isfinite(encoded.astype(np.float32))))
    if lost:
        print(
            f"  WARNING {name}: {lost} value(s) overflowed f16 range (|x| > 65504)",
            file=sys.stderr,
        )
    return lost


def build_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise ConversionError(f"{model_dir}: missing config.json")
    raw = json.loads(config_path.read_text())

    out: dict[str, Any] = {}
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in raw]
    # num_key_value_heads is absent on non-GQA models; it equals the query head count.
    if "num_key_value_heads" in missing and "num_attention_heads" in raw:
        raw["num_key_value_heads"] = raw["num_attention_heads"]
        missing.remove("num_key_value_heads")
    if missing:
        raise ConversionError(f"config.json is missing required keys: {', '.join(missing)}")

    for key in REQUIRED_CONFIG_KEYS:
        out[key] = raw[key]
    for key in OPTIONAL_CONFIG_KEYS:
        if key in raw:
            out[key] = raw[key]

    # head_dim is derived when the config omits it, which most Llama-family configs do.
    if "head_dim" not in out:
        if out["hidden_size"] % out["num_attention_heads"] != 0:
            raise ConversionError(
                f"hidden_size {out['hidden_size']} not divisible by "
                f"num_attention_heads {out['num_attention_heads']}"
            )
        out["head_dim"] = out["hidden_size"] // out["num_attention_heads"]

    if out["num_attention_heads"] % out["num_key_value_heads"] != 0:
        raise ConversionError(
            f"num_attention_heads {out['num_attention_heads']} is not a multiple of "
            f"num_key_value_heads {out['num_key_value_heads']}"
        )

    gen_path = model_dir / "generation_config.json"
    if gen_path.exists():
        gen = json.loads(gen_path.read_text())
        for key in ("bos_token_id", "eos_token_id"):
            if key in gen:
                out[key] = gen[key]
    return out


def ordered_tensor_names(available: set[str], config: dict[str, Any]) -> list[str]:
    """
    Emit in forward-pass order so a future streaming loader can begin work before the
    download finishes. Any tensor present but not named by the template is appended
    rather than dropped -- silently losing a weight would be far worse than a large file.
    """
    names: list[str] = []

    def take(name: str) -> None:
        if name in available and name not in names:
            names.append(name)

    take("model.embed_tokens.weight")
    for layer in range(config["num_hidden_layers"]):
        p = f"model.layers.{layer}."
        take(p + "input_layernorm.weight")
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            take(p + f"self_attn.{proj}.weight")
            take(p + f"self_attn.{proj}.bias")
        take(p + "post_attention_layernorm.weight")
        for proj in ("gate_proj", "up_proj", "down_proj"):
            take(p + f"mlp.{proj}.weight")
            take(p + f"mlp.{proj}.bias")
    take("model.norm.weight")
    take("lm_head.weight")

    leftovers = sorted(available - set(names))
    if leftovers:
        print(f"  note: {len(leftovers)} unrecognised tensor(s) appended: {leftovers[:4]}...")
        names.extend(leftovers)
    return names


def convert(
    model_dir: Path,
    out_dir: Path,
    chunk_bytes: int,
    matrix_dtype: str,
    quant_bits: int = 0,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> dict[str, Any]:
    started = time.time()
    config = build_config(model_dir)
    shards = open_shards(model_dir)
    available = set(shards.keys())

    tied = bool(config.get("tie_word_embeddings", False))
    aliases: dict[str, str] = {}
    if tied:
        # The lm_head matrix is the embedding matrix. Storing it twice would add 272 MB
        # to the download for nothing; the registry resolves the alias at load time.
        if "lm_head.weight" in available:
            available.discard("lm_head.weight")
        aliases["lm_head.weight"] = "model.embed_tokens.weight"

    names = ordered_tensor_names(available, config)
    writer = ChunkWriter(out_dir, chunk_bytes)
    directory: list[dict[str, Any]] = []
    total_source_params = 0
    total_overflow = 0

    def emit(name: str, payload: bytes, dtype: str, shape: tuple[int, ...], quant: Any) -> None:
        writer.align(TENSOR_ALIGNMENT)
        offset = writer.offset
        writer.write(payload)
        directory.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "offset": offset,
                "byteLength": len(payload),
                "quant": quant,
                # Convenience for the loader's range planning; `offset` remains
                # authoritative because a tensor may span several chunks.
                "firstChunk": offset // chunk_bytes,
                "lastChunk": (offset + len(payload) - 1) // chunk_bytes
                if payload
                else offset // chunk_bytes,
            }
        )

    quantized_count = 0
    worst_quant_error = 0.0
    print(f"converting {len(names)} tensors from {model_dir}")
    for name in names:
        source = shards[name].read(name)
        shape = tuple(int(d) for d in source.shape)
        total_source_params += int(np.prod(shape)) if shape else 1

        # Only rank-2 weights are quantized, and only when the reduction dimension divides
        # evenly into blocks. A tensor that does not qualify falls back to f16 with a note
        # rather than being silently padded into a different shape.
        quantizable = quant_bits > 0 and len(shape) == 2 and shape[1] % block_size == 0
        if quant_bits > 0 and len(shape) == 2 and not quantizable:
            print(f"  note: {name} cols {shape[1]} not divisible by {block_size}; keeping f16")

        if quantizable:
            words, scales = quantize_blockwise(source, quant_bits, block_size)
            restored = dequantize_blockwise(words, scales, shape[1], quant_bits, block_size)
            denominator = float(np.abs(source).max()) or 1.0
            error = float(np.abs(restored - source).max()) / denominator
            worst_quant_error = max(worst_quant_error, error)
            quantized_count += 1

            emit(
                name,
                np.ascontiguousarray(words, dtype="<u4").tobytes(),
                f"q{quant_bits}",
                shape,
                {
                    "scheme": "symmetric",
                    "bits": quant_bits,
                    "blockSize": block_size,
                    "scales": f"{name}.scales",
                },
            )
            emit(
                f"{name}.scales",
                np.ascontiguousarray(scales, dtype="<f2").tobytes(),
                "f16",
                (shape[0], shape[1] // block_size),
                None,
            )
            continue

        dtype = target_dtype(name, shape, matrix_dtype)
        encoded = encode_tensor(source, dtype)
        total_overflow += check_overflow(name, source, encoded)
        emit(name, encoded.tobytes(), dtype, shape, None)

    chunks = writer.close()
    total_bytes = sum(c["bytes"] for c in chunks)

    header = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "source": {
            "modelDir": model_dir.name,
            "convertedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "converterVersion": FORMAT_VERSION,
        },
        "config": config,
        "quant": (
            {"bits": quant_bits, "blockSize": block_size, "scheme": "symmetric"}
            if quant_bits
            else None
        ),
        "chunkBytes": chunk_bytes,
        "totalBytes": total_bytes,
        "tensorAlignment": TENSOR_ALIGNMENT,
        "chunks": chunks,
        "tensors": directory,
        "aliases": aliases,
    }

    (out_dir / "model.json").write_text(json.dumps(header, indent=2) + "\n")

    # The tokenizer travels with the weights so the browser fetches one directory.
    copied = copy_tokenizer_files(model_dir, out_dir)
    if copied:
        print(f"  copied {', '.join(copied)}")

    elapsed = time.time() - started
    print(
        f"wrote {len(chunks)} chunk(s), {total_bytes / 1e6:.1f} MB total, "
        f"{total_source_params / 1e6:.1f}M params, in {elapsed:.1f}s"
    )
    if tied:
        print("  tie_word_embeddings: lm_head.weight aliased to model.embed_tokens.weight")
    if quantized_count:
        print(
            f"  quantized {quantized_count} tensors to int{quant_bits} "
            f"(block {block_size}); worst relative dequant error {worst_quant_error:.2%}"
        )
    if total_overflow:
        print(f"  WARNING: {total_overflow} value(s) overflowed f16 range", file=sys.stderr)
    return header


# Files the browser needs alongside the weights. Absent files are skipped: the tiny
# synthetic test model has no tokenizer, and that is not an error.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "generation_config.json")


def copy_tokenizer_files(model_dir: Path, out_dir: Path) -> list[str]:
    copied: list[str] = []
    for name in TOKENIZER_FILES:
        source = model_dir / name
        if not source.exists():
            continue
        shutil.copyfile(source, out_dir / name)
        copied.append(name)
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path, help="Hugging Face model directory")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--chunk-bytes",
        type=int,
        default=DEFAULT_CHUNK_BYTES,
        help=f"chunk size in bytes (default {DEFAULT_CHUNK_BYTES})",
    )
    parser.add_argument(
        "--matrix-dtype",
        choices=["f16", "f32"],
        default="f16",
        help="dtype for unquantized rank-2 weights (default f16)",
    )
    parser.add_argument(
        "--quant",
        type=int,
        choices=[0, 4, 8],
        default=0,
        help="block-wise quantization bit width for rank-2 weights; 0 disables (default 0)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=f"quantization block size along the reduction dim (default {DEFAULT_BLOCK_SIZE})",
    )
    args = parser.parse_args(argv)

    try:
        convert(
            args.model_dir,
            args.out,
            args.chunk_bytes,
            args.matrix_dtype,
            args.quant,
            args.block_size,
        )
    except ConversionError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
