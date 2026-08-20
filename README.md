# Browser LLM

A transformer language model running entirely in the browser, on WebGPU, with every
compute kernel written by hand in WGSL. No server, no inference API, no backend. You open
a static page, the weights download once into local storage, and generation happens on
your GPU.

**What this is not.** [MLC's WebLLM](https://github.com/mlc-ai/web-llm) already does
browser inference, does it faster, and does it in production — this is a from-scratch
kernel implementation written to understand the problem end to end, not a replacement for
it. Nothing here imports an inference runtime, a matmul library, or a quantization kernel;
the interesting part *is* the part those libraries would have provided.

- **19 WGSL kernels**, 1,115 lines of shader code, ~7,400 lines of source total.
- **Zero runtime dependencies.** `package.json` has an empty `dependencies` block.
  TypeScript, Vite and Vitest are build and test tooling; nothing ships but our own code.
- **No hardcoded model dimensions.** Every shape is read from the model's `config.json` at
  conversion time and from our binary header at load time. Two models with different
  architectures run on the same engine, which the test suite checks rather than asserts.

## Architecture

```
  tools/  (Python, offline, never ships)
    convert.py ──── safetensors ──> model.json + model-NNN.bin chunks
                    (fp16, or int4/int8 block-quantized)
    golden.py ───── PyTorch CPU ──> per-layer activations + logits, the test oracle

  ─────────────────────────── browser ───────────────────────────

    store.ts        fetch chunks, resume on failure, persist to OPFS
        │           second load reads OPFS and touches the network zero times
        ▼
    model.ts        parse header, build the weight registry, upload to GPUBuffers
        │           tensors too large for one binding are sharded across buffers
        ▼
    forward.ts      one code path, two uses:
        │             prefill(n tokens)  ── tiled matmul, shared-memory staging
        │             decode(1 token)    ── quantized matvec, KV cache append
        │
        ├── kernels.ts    uniform packing, dispatch sizing, per-kernel launch
        ├── pipelines.ts  shader compilation + pipeline cache (never in the loop)
        ├── kvcache.ts    preallocated cache, refuses overflow rather than corrupting
        ├── sampler.ts    temperature / top-k / top-p, top-k computed on the GPU
        └── profiler.ts   timestamp queries, per-kernel milliseconds

    shaders/*.wgsl  embed_gather · rmsnorm · rope · attn_scores · softmax_rows
                    attn_output · kv_write · matmul_f16(_tiled) · matvec_f32
                    matvec_q4 · swiglu_q4 · embed_gather_q4 · residual_add
                    silu_mul · reduce_max · reduce_sumexp · topk_partial · topk_select

    worker/         owns the device; the UI thread only sends messages and draws
    tokenizer/      byte-level BPE + chat template, both driven by tokenizer.json
    reference/cpu.ts  scalar TypeScript twin of every kernel — the correctness oracle
```

The decode path is bandwidth-bound: every token streams the whole weight set through the
GPU and does almost no arithmetic per byte. That single fact determines most of the design
— int4 weights dequantized in-register, `vec4<u32>` loads, lanes partitioned across output
rows, and a fused SwiGLU so one activation read serves two projections.

## Benchmarks

Measured with `npm run bench -- tests/readme.bench.test.ts`, which prints these rows
directly. Prompt is 97 tokens; decode is best of 5 runs of 48 steps, because run-to-run
spread on this hardware is the same size as the effects being measured. Even so, repeating
the whole command moves these by a few percent — a second run gave 126.1 tok/s where the
table says 125.0. Treat the last digit as noise.

| Device | GPU | Browser | Model | Quant | Prefill tok/s | Decode tok/s | TTFT | VRAM |
|---|---|---|---|---|---|---|---|---|
| MacBook (Apple M4, 16 GB, macOS 15.5) | apple / metal-3 | Chromium 151.0.7922.34 headless | Qwen2.5-0.5B-Instruct | fp16 | 398 | 32.1 | 244 ms | 942 MiB |
| MacBook (Apple M4, 16 GB, macOS 15.5) | apple / metal-3 | Chromium 151.0.7922.34 headless | Qwen2.5-0.5B-Instruct | int4 (block 32) | 513 | 125.0 | 189 ms | 265 MiB |
| MacBook (Apple M4, 16 GB, macOS 15.5) | apple / metal-3 | Chromium 151.0.7922.34 headless | SmolLM2-360M-Instruct | fp16 | 450 | 34.0 | 213 ms | 690 MiB |
| MacBook (Apple M4, 16 GB, macOS 15.5) | apple / metal-3 | Chromium 151.0.7922.34 headless | SmolLM2-360M-Instruct | int4 (block 32) | 593 | 121.3 | 162 ms | 194 MiB |

> **This table is one machine short of honest.** The project's own gate asks for at least
> three, and no numbers for a second or third device exist yet, so none are printed here.
> Estimating them would defeat the point of measuring. Run the command above on another
> machine and paste its rows in.

Decode on a short prompt is faster than the table shows — the throughput gate measures
143.2 tok/s at int4 on a 5-token prompt, against 125.0 here at 97 tokens, because
attention grows with context. Both numbers are real; they answer different questions.

**Quality of the int4 path.** Top-1 agreement with the fp16 path is 185/200 = 92.5% over
held-out prose with both models fed the same tokens. Against the PyTorch goldens, top-1
matches on 2 of 3 prompts and the one disagreement is a genuine near-tie (a 0.96-logit gap
between two plausible openings). Worst relative dequantization error is 9.6%.

## The optimization log

One row per change, measured on the machine above, decode only, int4 Qwen2.5-0.5B.
**Three of these made things worse or nothing and are still here**, because a table that
lists only the wins is a sales pitch rather than a record.

| change | kernel | tok/s before | tok/s after |
|---|---|---|---|
| KV cache + prefill/decode split (M3) | whole pass | 7.46 | ~24–26 |
| tiled prefill, 4 positions/workgroup (M3) | `matmul_f16_tiled` | 121 prefill | 344 prefill |
| int4 block quantization, naive port of the f16 matvec | `matvec_q4` | 26.3 (fp16) | 51.5 |
| `vec4<u32>` loads + lanes partitioned across rows | `matvec_q4` | 51.5 | 57.9 |
| **RMSNorm as a workgroup reduction** | `rmsnorm` | 57.9 | **86.5** |
| stage activations in workgroup memory | `matvec_q4` | 86.5 | 84.5 — **reverted** |
| fuse gate_proj + up_proj + silu | `swiglu_q4` | 93.8 | 100.6 |
| rows-per-workgroup chosen per matmul | `matvec_q4` | 93.8 | 72.1 — **reverted** |
| sweep the workgroup shape | `matvec_q4`, `swiglu_q4` | — | 101.6 |
| **pool the per-dispatch uniform buffers** | whole pass, CPU side | 102.3 | **145.5** |
| `vec4<f32>` activation loads + `dot()` | `matvec_q4`, `swiglu_q4` | 102.3 | 102.3 — **reverted** |
| top-k readback instead of full logits | sampling | 101.1 | 99.3 — no speedup |

Net: **26.3 tok/s at fp16 → 145.5 tok/s at int4.** The bandwidth story is the more honest
one: fp16 was already moving 26.0 GB/s, and quantizing *dropped* that to 14.3 GB/s while
still winning on wall-clock because it moved 3.55× less data. Most of the work since has
been earning that efficiency back, to 39.8 GB/s.

## What surprised me

**The biggest win was in the kernel I never thought about.** A profiled decode step said
RMSNorm was ~56% of decode time. The M2 kernel used one *invocation* per row, which is
fine for prefill and pathological for decode: with a single token, one thread walked 896
elements twice while the rest of the GPU idled. Rewriting it as a workgroup reduction was
worth more than every matmul optimization to that point — and it was completely invisible
in the tok/s number, showing up only once each dispatch was timed separately.

**The second biggest win wasn't in a kernel at all, and the profiler is why it hid.**
After the kernel work, decode sat at ~100 tok/s against a 100 tok/s gate — close enough
that it kept looking like one more optimization away. It wasn't. The pass was allocating a
fresh `GPUBuffer` for each of ~411 dispatches' uniforms every token, then destroying all
411 after the readback. Encoding is serial with the GPU, so the device idled through all of
it. Timestamp queries cannot see CPU time, so the tool that found the RMSNorm win was
structurally incapable of finding this one — and it framed the problem as a kernel problem
for an entire milestone. Adding one number, `encodeMs`, made a 42% speedup obvious in a
single run.

**Two confident optimizations were worth exactly nothing.** Replacing eight scalar
activation loads with `vec4<f32>` loads and `dot()` measured 102.3 → 102.3: the Metal
backend was already coalescing them. And the full-logit readback — 607,744 bytes per token
against top-k's 72, an 8,400× difference that M4 was built to eliminate — is **not faster**
when you measure it, because both paths synchronize once per step regardless. Traffic and
latency are different problems, and the big number was the wrong one to chase.

**When a numerical gate fails, the reference is a suspect too.** Greedy output forked from
PyTorch at token 4, which reads exactly like f16 rounding flipping a near-tie. It wasn't:
the margin was 1.6 logits, and f16-rounding the weights moved the logits by less than 1e-5.
Qwen ships a `generation_config.json` with `repetition_penalty: 1.1`, and `model.generate`
applies it **even with `do_sample=False`**. The "greedy" reference was never greedy. The
fix was two lines in the golden dumper; finding it took bisecting against the sequence
instead of accepting the end-to-end diff.

**Run-to-run spread is the same size as the effects.** Three times a single-sample A/B
pointed the wrong way. The fused SwiGLU looked like a 4% *regression* on one sample and is
a 7% improvement over best-of-five. Two changes are marked reverted above only because a
best-of-N harness existed to catch them.

**Portability is a claim you have to test.** The engine ran a second architecture
(SmolLM2-360M: 32 layers, an odd 15 attention heads over 5 KV heads, a third the
vocabulary, no attention bias) with no changes to a single kernel — but the *tokenizer*
broke immediately. It looked for one `Split` node with a regex, which is Qwen's shape;
SmolLM2 has a `Sequence` of `Digits` and `ByteLevel` with no regex in the file at all,
because the `tokenizers` library hardcodes the GPT-2 regex and only serializes one when a
model overrides it. Everything I'd been careful about was fine. The thing I'd assumed was
generic wasn't.

**Small things that cost real time.** `active` is a reserved word in WGSL. Playwright's
default headless build ships no GPU stack — it exposes `navigator.gpu` but
`requestAdapter()` always resolves `null`, which looks identical to "your machine has no
WebGPU". And `about:blank` doesn't expose `navigator.gpu` at all, a confusingly different
symptom from the same-looking cause.

## Running it

```bash
npm install
npm run dev          # device panel, model loader, chat UI
npm test             # full suite in headless Chromium (needs a real GPU stack)
npm run bench        # decode sweep, optimization A/B, per-kernel profile
```

Tests run in a real browser because Node has no WebGPU — a kernel can only be validated
against an actual implementation. The suite skips any test whose model files are absent,
so a fresh clone is green without a gigabyte of weights.

### Converting a model

```bash
python3 tools/convert.py models/Qwen2.5-0.5B-Instruct \
    --out public/models/qwen2.5-0.5b-instruct-q4 --quant 4
```

`--quant 4` and `--quant 8` select block-wise quantization (block size configurable,
default 32); omit it for fp16. Norms, biases and RoPE tables always stay fp16/fp32.
Conversion and golden-dump commands for both models are in
[PROGRESS.md](PROGRESS.md).

## Deploying

The app is 132 KB built; the weights are 265 MiB and deliberately not part of it
(`copyPublicDir: false`). They are two different hosting problems, so they get two hosts.

Vercel's static upload cap is 100 MB on Hobby and 1 GB on Pro, so the weights cannot ship
in a Hobby deployment at all — and even where they fit, every cold visitor pulls the whole
model, which turns 100 GB of monthly bandwidth into about 370 visitors. The Hub is built
for exactly this traffic, sends CORS headers, and honours Range requests, which the
resume-on-failure path depends on.

```bash
# 1. Weights -> Hugging Face
hf auth login
hf upload <user>/browser-llm-weights \
    public/models/qwen2.5-0.5b-instruct-q4 qwen2.5-0.5b-instruct-q4

# 2. App -> Vercel, pointed at them
vercel --prod --build-env \
    VITE_MODEL_BASE=https://huggingface.co/<user>/browser-llm-weights/resolve/main/qwen2.5-0.5b-instruct-q4/
```

`VITE_MODEL_BASE` is read at build time. Unset, the app falls back to same-origin
`/models/<id>/`, which is what `npm run dev` serves — so development needs no configuration
and the deployed build needs no code change. Any host works if it sends CORS headers and
supports Range requests.

## Status

M0–M5 are complete and gated. M6 is in progress: the second model runs, and what remains
is the capability-detection page, deployment, and the two missing benchmark machines.
[PROGRESS.md](PROGRESS.md) is the working record — every gate, every measurement, the
blockers, and the things that went wrong.
