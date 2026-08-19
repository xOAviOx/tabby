#!/usr/bin/env python3
"""
Dump PyTorch reference activations for the browser forward pass to be checked against.

For each fixed prompt this records, in float32 on CPU:

    token ids
    the embedding output
    the output of *every* decoder layer
    the output of the final RMSNorm
    the logits at the last position
    a 20-token greedy continuation

Per-layer outputs are captured with forward hooks rather than `output_hidden_states`,
because that flag returns the states *before* each layer plus the post-norm state, which
leaves the last layer's raw output unavailable -- exactly the one you want when drift
first appears at the end of the stack.

Output is one manifest plus flat little-endian f32 blobs, so the browser can fetch a
tensor by byte range without a numpy reader.

Usage:
    .venv/bin/python tools/golden.py models/Qwen2.5-0.5B-Instruct \\
        --out public/golden/qwen2.5-0.5b-instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Three prompts covering materially different inputs: a bare completion, the chat
# template the model was tuned for, and text with digits, punctuation runs, accented
# characters and CJK.
PROMPTS: list[tuple[str, str]] = [
    ("plain", "The capital of France is"),
    (
        "chat",
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n",
    ),
    (
        "mixed",
        "In 2024, the quick brown fox — naïve as ever — jumped over 3 lazy dogs. 你好!",
    ),
]

GREEDY_TOKENS = 20


class BlobWriter:
    """Appends f32 arrays to one flat file and hands back (offset, length) records."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self._fh = path.open("wb")

    def add(self, array: np.ndarray) -> dict:
        data = np.ascontiguousarray(array, dtype="<f4")
        payload = data.tobytes()
        record = {
            "offset": self.offset,
            "byteLength": len(payload),
            "shape": list(data.shape),
        }
        self._fh.write(payload)
        self.offset += len(payload)
        return record

    def close(self) -> int:
        self._fh.close()
        return self.offset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--greedy-tokens", type=int, default=GREEDY_TOKENS)
    args = parser.parse_args(argv)

    torch.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    # float32 on CPU: this is the reference the GPU path is judged against, so it should
    # carry no more rounding than necessary. The engine's f16 weights are a deliberate
    # difference and are what the 2e-2 tolerance is there to absorb.
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    decoder = model.model
    layers = decoder.layers
    print(f"model: {len(layers)} layers, hidden {model.config.hidden_size}")

    captured: dict[str, torch.Tensor] = {}
    handles = []

    def capture(name: str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor.detach().to(torch.float32).clone()

        return hook

    handles.append(decoder.embed_tokens.register_forward_hook(capture("embed")))
    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(capture(f"layer{index}")))
    handles.append(decoder.norm.register_forward_hook(capture("final_norm")))

    # The model ships a generation_config.json with do_sample, temperature, top_p and
    # repetition_penalty 1.1, and `generate` applies repetition_penalty even when
    # do_sample is False. That silently makes the "greedy" reference non-greedy: it
    # demoted a token that already appeared in the prompt and picked a different word.
    # Everything here is therefore pinned explicitly rather than inherited.
    greedy_config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.0,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        max_new_tokens=args.greedy_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=model.generation_config.eos_token_id,
    )

    blobs = BlobWriter(args.out / "golden.bin")
    entries = []

    for name, text in PROMPTS:
        captured.clear()
        ids = tokenizer.encode(text, add_special_tokens=False)
        input_ids = torch.tensor([ids], dtype=torch.long)

        with torch.no_grad():
            out = model(input_ids)
        logits = out.logits[0, -1].to(torch.float32)

        tensors = {"embed": blobs.add(captured["embed"][0].numpy())}
        for index in range(len(layers)):
            tensors[f"layer{index}"] = blobs.add(captured[f"layer{index}"][0].numpy())
        tensors["final_norm"] = blobs.add(captured["final_norm"][0].numpy())
        tensors["logits"] = blobs.add(logits.numpy())

        top5 = torch.topk(logits, 5)
        with torch.no_grad():
            generated = model.generate(input_ids, generation_config=greedy_config)
        continuation = generated[0, len(ids) :].tolist()

        entries.append(
            {
                "name": name,
                "text": text,
                "ids": ids,
                "numLayers": len(layers),
                "tensors": tensors,
                "top5": {
                    "ids": top5.indices.tolist(),
                    "values": [round(v, 6) for v in top5.values.tolist()],
                    "tokens": [tokenizer.decode([i]) for i in top5.indices.tolist()],
                },
                "greedy": {
                    "ids": continuation,
                    "text": tokenizer.decode(continuation, skip_special_tokens=False),
                },
            }
        )
        print(
            f"  {name}: {len(ids)} tokens, top1 {top5.indices[0].item()} "
            f"({tokenizer.decode([top5.indices[0].item()])!r})"
        )
        print(f"    greedy: {entries[-1]['greedy']['text']!r}")

    for handle in handles:
        handle.remove()
    total = blobs.close()

    manifest = {
        "modelDir": str(args.model_dir),
        "hiddenSize": model.config.hidden_size,
        "numLayers": len(layers),
        "vocabSize": model.config.vocab_size,
        "dtype": "f32",
        "blob": "golden.bin",
        "blobBytes": total,
        "prompts": entries,
    }
    (args.out / "golden.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}/golden.json and golden.bin ({total / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
