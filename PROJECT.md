# PROJECT.md — Browser-Native LLM Inference Engine (WebGPU + hand-written WGSL)

> Drop this file at the repo root. Start Claude Code with:
> `Read PROJECT.md. Confirm you understand the constraints and the current milestone, then implement M0 only. Stop at the M0 acceptance gate and report results.`
> Then, each session: `Read PROJECT.md and PROGRESS.md. Implement the next milestone. Stop at its gate.`

---

## 1. What we are building

A transformer LLM running **entirely in the browser** on WebGPU, with every compute kernel written by hand in WGSL. No server, no inference API, no backend of any kind. The user opens a static URL, the model downloads once into local storage, and all generation happens on their GPU.

The deliverable is a deployed static site plus a repo whose README contains a real benchmark table and an optimization writeup.

## 2. Hard constraints — do not violate these

**Absolutely forbidden dependencies.** The entire point of this project is hand-written kernels. Do NOT use, import, vendor, or "temporarily" fall back to:
- `@mlc-ai/web-llm`, `onnxruntime-web`, `@xenova/transformers` / `transformers.js`, `tensorflow.js`, `wonnx`, `webdnn`
- Any WASM inference runtime
- Any library that provides matmul, attention, or quantization kernels

If a milestone seems to require one of these, stop and report the blocker instead of importing it.

**Allowed dependencies (keep the list short):** TypeScript, Vite, and optionally a tiny UI library. For the offline conversion script: Python with `safetensors`, `numpy`, `torch` (CPU only). That is the whole list. Ask before adding anything else.

**No hardcoded model dimensions anywhere.** Every dimension (`hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `intermediate_size`, `vocab_size`, `rope_theta`, `rms_norm_eps`, `tie_word_embeddings`) is read from the model's `config.json` at conversion time, written into our binary header, and read back at load time. If you find yourself typing a number like `896` or `24` into a `.ts` or `.wgsl` file, you have made a mistake — pass it as a uniform or a shader override constant.

**Correctness before speed, always.** Never optimize a kernel that has not passed its numerical gate. Never move to the next milestone with a failing gate.

## 3. Target model

Default: **Qwen2.5-0.5B-Instruct**. Rationale: small download, standard Llama-style architecture (RMSNorm + RoPE + SwiGLU + GQA), good instruction following for its size.

The engine must be architecture-generic within the Llama/Qwen family so that swapping in `Llama-3.2-1B-Instruct` or `SmolLM2-360M-Instruct` requires only re-running the conversion script. Verify this claim by actually converting a second model at M6.

Note the large vocabulary (~152K on Qwen2.5). Its consequences are addressed in M1 (buffer splitting) and M4 (GPU-side top-k).

## 4. Repository layout

```
/tools/                     # Python, runs offline on the dev machine
  convert.py                # safetensors -> our .bin format, with quantization
  golden.py                 # dumps reference activations + logits as .npy/.bin
  tokenize_golden.py        # dumps (text -> token ids) pairs for tokenizer tests
/src/
  /engine/
    device.ts               # adapter/device init, feature + limit negotiation
    buffers.ts              # GPUBuffer allocation, splitting, staging uploads
    pipelines.ts            # shader module compilation + pipeline cache
    model.ts                # weight registry, header parsing, layer graph
    forward.ts              # prefill and decode passes, command encoding
    kvcache.ts              # KV cache allocation and ring/append logic
    sampler.ts              # greedy, temperature, top-k, top-p
    profiler.ts             # timestamp queries, per-kernel timing aggregation
  /shaders/                 # one .wgsl per kernel, no exceptions
  /tokenizer/
    bpe.ts                  # byte-level BPE encode/decode
  /reference/
    cpu.ts                  # scalar TypeScript reference implementation of every kernel
  /worker/
    inference.worker.ts     # owns the GPU device; message protocol to UI
  /ui/                      # chat interface + perf panel
/tests/
  kernels.test.ts           # GPU kernel vs CPU reference, per kernel
  golden.test.ts            # full forward pass vs PyTorch golden logits
  tokenizer.test.ts         # vs Python golden pairs
/PROGRESS.md                # you maintain this — see section 8
```

## 5. Milestones

Implement these **in order**. Each has a gate. Do not proceed past a failing gate; stop and report instead.

---

### M0 — Device, buffers, and one real kernel

Set up Vite + TypeScript. Initialize WebGPU: request adapter, inspect `adapter.limits` and `adapter.features`, request `shader-f16` and `timestamp-query` if available, and request raised `maxStorageBufferBindingSize` / `maxBufferSize` up to what the adapter reports. Log the negotiated limits to console and surface them in the UI — we will need them constantly.

Write a `pushErrorScope`/`popErrorScope` wrapper around every submit during development, and handle `device.lost`.

Implement one kernel end to end: **fp32 matvec** (`y = W @ x`, W row-major). Implement the same operation in `/src/reference/cpu.ts`. Write the test harness that runs both on random inputs and compares.

**Gate M0:** `npm test` runs the fp32 matvec against the CPU reference on at least 5 random shapes (including non-power-of-2 and shapes larger than one workgroup) and passes with max abs error < 1e-4. Negotiated device limits printed. Report the limits you actually got.

---

### M1 — Weight pipeline

Write `tools/convert.py`. It takes an HF model directory and emits:
- `model.json` — header: all config fields, plus a tensor directory (name, dtype, shape, byte offset, byte length, quantization block size, which chunk file it lives in)
- `model-000.bin`, `model-001.bin`, … — weight data split into ~32MB chunks

Start with **fp16 weights only** (quantization comes at M5 — do not do it now). Handle tied embeddings: if `tie_word_embeddings` is true, store the matrix once and reference it twice.

Browser side: fetch chunks with progress reporting, persist to **OPFS**, and on subsequent loads read from OPFS instead of the network. Implement resume-on-failure for interrupted downloads.

Upload weights into `GPUBuffer`s. **Any tensor exceeding the negotiated `maxStorageBufferBindingSize` must be split across multiple buffers**, with the split recorded in the tensor registry so kernels can iterate over shards. The embedding / lm_head matrix will almost certainly need this — design for it now rather than special-casing later.

**Gate M1:** Model loads from network on first run and from OPFS on second run (prove it with a timing log and by testing offline after first load). A test reads back three known weight tensors from GPU and byte-matches them against values read directly from the safetensors file in Python. Progress bar reflects real byte counts.

---

### M2 — Correct forward pass, unoptimized

Implement every remaining kernel in the simplest correct form. Optimization is explicitly forbidden in this milestone — naive one-thread-per-output is fine.

Kernels: embedding gather; RMSNorm; QKV projection; RoPE (read `rope_theta` from config); attention (scores, scaled softmax, weighted sum of V) with **grouped-query attention** — handle `num_key_value_heads < num_attention_heads` correctly by mapping query heads to KV heads; output projection; SwiGLU MLP (gate proj, up proj, SiLU, elementwise multiply, down proj); final RMSNorm; lm_head.

Also implement the byte-level BPE tokenizer, reading `tokenizer.json`. Byte-level BPE with the pretokenizer regex is finicky — implement it against golden pairs from `tools/tokenize_golden.py`, including whitespace-leading tokens, emoji, CJK, and the chat template's special tokens.

Use `tools/golden.py` to dump, from PyTorch on CPU: token ids, the hidden state after **every** layer, and final logits, for three fixed prompts.

**Gate M2:** (a) Tokenizer round-trips all golden pairs exactly. (b) Per-layer hidden states match PyTorch golden within 2e-2 max abs error, **checked layer by layer** — report the error for each layer so drift is visible. (c) Final logits' top-5 token ids match PyTorch exactly for all three prompts. (d) Greedy generation of 20 tokens produces coherent English. Speed is irrelevant here; report it anyway as the baseline.

---

### M3 — KV cache and prefill/decode split

Split the forward pass into a **prefill** path (batch of positions, tiled matmul, shared-memory staging) and a **decode** path (single position, matvec). Allocate the KV cache as a preallocated buffer sized by a configurable `maxSeqLen`; append per step; handle context overflow by refusing cleanly with a clear error rather than corrupting memory.

Move all GPU work into the Web Worker. Define an explicit message protocol (`load`, `progress`, `generate`, `token`, `stats`, `error`, `cancel`). Streaming tokens must reach the UI as they are produced, and `cancel` must actually stop generation mid-run.

**Gate M3:** Generating N tokens with the KV cache produces byte-identical output to the M2 no-cache path for the same prompt and greedy sampling. UI stays responsive (verify by animating something during generation). Cancel works. Report tok/s and TTFT.

---

### M4 — Sampling and chat

GPU-side sampling: temperature, top-k, top-p. Compute top-k on the GPU and read back only the k candidates — **never read the full logit vector to the CPU per token**, it is ~600KB per step on a 152K vocab.

Apply the model's chat template from `tokenizer_config.json`. Build the chat UI: message list, streaming output, stop button, model-loading progress, and a settings panel for temperature / top-k / top-p / max tokens / seed.

**Gate M4:** Multi-turn conversation works with correct template formatting. Seeded sampling is deterministic across runs. No per-token full-logit readback (prove it — log readback byte counts per token).

---

### M5 — Quantization (the performance milestone)

Add int4 block-wise quantization to `convert.py`: block size 32 along the input dimension, one fp16 scale per block, 8 nibbles packed per `u32`. Keep norms, biases, and RoPE tables in fp16/fp32. Make block size and per-tensor bit width configurable so int8 can be compared against int4.

Write the quantized matvec kernel: dequantize in-register while accumulating, never materializing fp16 weights in memory. This kernel is the single most important piece of code in the project — decode is memory-bandwidth bound, so its bandwidth efficiency determines nearly all of the final tok/s.

Then optimize, measuring after every change:
- one workgroup per output row (or small row group), strided reads, workgroup-shared reduction
- vectorized loads (`vec4<u32>`) to widen memory transactions
- `f16` arithmetic where `shader-f16` is available, with an f32 fallback path
- fuse where it pays: RMSNorm into the following projection, gate/up into one dispatch, RoPE into the QKV epilogue
- tune workgroup sizes empirically per kernel — do not guess, sweep and record
- online (streaming) softmax in attention to avoid a second pass over scores

Wire up `profiler.ts` with timestamp queries and build the perf panel: tok/s, TTFT, per-kernel milliseconds, achieved memory bandwidth (GB/s), model size in VRAM.

**Gate M5:** Quantized model's top-5 logits still match PyTorch for the golden prompts (looser tolerance is fine — report perplexity or top-1 agreement rate over ~200 tokens of held-out text vs the fp16 path). Decode throughput is at least 4× the M3 baseline. Perf panel shows a per-kernel breakdown. **Maintain a table in PROGRESS.md with one row per optimization: what changed, tok/s before, tok/s after.** That table is the writeup.

---

### M6 — Ship it

Second model converted and running (`Llama-3.2-1B-Instruct` or `SmolLM2-360M-Instruct`) with zero engine code changes — if changes are needed, that is a bug in M1's genericity, fix it there.

Capability detection with a graceful, informative failure page when WebGPU is missing (name the browser requirement, don't just show a blank screen). Deploy as a static site (Cloudflare Pages or Vercel). Verify the model caches correctly behind the CDN and that a cold visitor's download actually resumes if interrupted.

README containing: what this is and explicitly what it is not (one line acknowledging WebLLM/MLC exists and stating that this is a from-scratch kernel implementation); architecture diagram; the benchmark table (device, GPU, browser, model, quantization, prefill tok/s, decode tok/s, TTFT, VRAM) measured on at least three machines; the optimization table from M5; and a "what surprised me" section.

**Gate M6:** Live URL works from a clean browser profile. Second model runs. README benchmark table is filled with real measured numbers, not estimates.

---

## 6. WebGPU facts to design against

- Default limits you must assume unless the adapter grants more: `maxStorageBufferBindingSize` 128MiB, `maxBufferSize` 256MiB, `maxComputeInvocationsPerWorkgroup` 256, `maxComputeWorkgroupStorageSize` 16KB, `maxStorageBuffersPerShaderStage` 8. Always request raised limits, always handle being refused.
- `shader-f16` and `timestamp-query` are optional features. Feature-detect and provide fallbacks; do not assume.
- There is no `printf` in WGSL. Debug by writing intermediates to a storage buffer and reading back on the CPU. Build this readback helper early in M0 — you will use it constantly.
- Buffer mapping is async and a synchronization point. Keep readbacks out of the per-token hot path.
- Shader compilation is slow. Compile all pipelines once at load, cache them, never inside the generation loop.
- Uniform buffer offsets must respect `minUniformBufferOffsetAlignment` (typically 256).

## 7. Working style

- Small commits, one logical change each, conventional commit messages.
- Every kernel gets its CPU reference and its test **in the same commit as the kernel**.
- When a numerical gate fails, bisect layer by layer using the golden dumps rather than guessing. Report which layer diverges first.
- If you are blocked or a gate cannot be met, stop and write the blocker into PROGRESS.md with what you tried. Do not silently loosen a tolerance to make a test pass, and do not disable a test.
- Never fabricate a benchmark number. Every number in PROGRESS.md or the README must come from an actual run on a named device.

## 8. PROGRESS.md

Create and maintain this file. Structure:

```
## Current milestone: M<n>
## Device limits negotiated on <GPU / browser>
## Completed gates
  - M0: passed <date>, max abs err 3.1e-6
## Optimization log
  | change | kernel | tok/s before | tok/s after | notes |
## Blockers
## Open questions for Avi
```

Update it at the end of every session. It is the memory between sessions and the raw material for the README writeup.
