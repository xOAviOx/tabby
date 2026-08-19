# PROGRESS

## Current milestone: M3 — complete. M4 not started.

## Device limits negotiated on Apple M-series (`apple` / `metal-3`), Chromium 151 headless

Measured by `npm test` on 2026-08-19. Chromium reports no `device`/`description` string
for Apple adapters, so the exact SoC is not visible to the page — recorded from the host
as macOS 15 (Darwin 24.5.0), Apple Silicon.

| limit | granted | WebGPU default | note |
|---|---|---|---|
| `maxBufferSize` | 4,294,967,292 (4096 MiB) | 256 MiB | unified memory; **atypically generous** |
| `maxStorageBufferBindingSize` | 4,294,967,292 (4096 MiB) | 128 MiB | see blocker B1 |
| `maxUniformBufferBindingSize` | 65,536 | 64 KiB | at default |
| `maxStorageBuffersPerShaderStage` | 10 | 8 | only +2 — kernels must stay ≤ 10 bindings |
| `maxUniformBuffersPerShaderStage` | 12 | 12 | at default |
| `maxComputeInvocationsPerWorkgroup` | 1,024 | 256 | 4× headroom for M5 tiling |
| `maxComputeWorkgroupStorageSize` | 32,768 | 16 KiB | 2× headroom for shared-memory staging |
| `maxComputeWorkgroupSizeX` | 1,024 | 256 | |
| `maxComputeWorkgroupSizeY` | 1,024 | 256 | |
| `maxComputeWorkgroupSizeZ` | 64 | 64 | at default |
| `maxComputeWorkgroupsPerDimension` | 65,535 | 65,535 | at default — see blocker B2 |
| `maxBindGroups` | 4 | 4 | at default |
| `maxBindingsPerBindGroup` | 1,000 | 1,000 | at default |
| `minUniformBufferOffsetAlignment` | 256 | 256 | uniform sub-allocation must be 256-aligned |
| `minStorageBufferOffsetAlignment` | 256 | 256 | |

Optional features, both **present** on this machine:

- `shader-f16` — available. An f32 fallback path is still required (M5).
- `timestamp-query` — available. `profiler.ts` (M5) can use real GPU timings here.

The raised-limit device request was **granted** in full; the default-limits fallback path
in `device.ts` did not trigger.

## Completed gates

### M0 — passed 2026-08-19

`npm test` → 12 passed. Full log reproducible with `npm test` (the verbose reporter is
enabled precisely so the numbers below are printed on every run).

fp32 matvec vs the scalar CPU reference, 7 shapes, seeded inputs (`mulberry32`, seed
printed per shape so any failure is reproducible):

| shape | why this shape | max abs err vs f32 ref | max abs err vs f64 ref |
|---|---|---|---|
| 1 × 1 | degenerate | 0.00e+0 | 2.94e-9 |
| 3 × 7 | tiny, both dims non-power-of-2, partial workgroup | 5.96e-8 | 1.81e-8 |
| 64 × 64 | exactly one full workgroup | 9.54e-7 | 1.28e-6 |
| 65 × 129 | one row past a workgroup boundary — tail guard | 1.67e-6 | 2.63e-6 |
| 1000 × 999 | many workgroups, both dims non-power-of-2 | 1.14e-5 | 3.31e-5 |
| 896 × 4864 | Qwen2.5-0.5B `down_proj`: short output, long reduction | **3.05e-5** | 1.99e-4 |
| 4864 × 896 | Qwen2.5-0.5B `gate_proj`: long output, short reduction | 1.34e-5 | 3.24e-5 |

**Gate: max abs error < 1e-4. Worst observed 3.05e-5. Passed.**

Also passing, as guards on things later milestones depend on:

- storage-buffer upload → readback is byte-exact, including a non-4-aligned element count
  (the readback helper is the project's only WGSL debugger, so it is itself under test);
- the kernel produces **bit-identical** output at workgroup sizes 1, 32, 64, 128 and 256.
  Nothing about the result may depend on how work is partitioned — this is the invariant
  every M5 optimisation has to preserve, so it is pinned down before any optimisation exists.

#### On the two CPU references — please sanity-check this decision

`src/reference/cpu.ts` implements matvec twice:

- `matvecF32` rounds to f32 after every multiply and every add, reproducing the kernel's
  arithmetic operation for operation;
- `matvecF64` accumulates in double precision — what the kernel would compute with
  unlimited precision.

The gate asserts against `matvecF32` and **reports** the error against `matvecF64`.

This matters, and the numbers above show why. At 896 × 4864 the f64 error is **1.99e-4** —
i.e. gating on the f64 reference would have *failed* the 1e-4 threshold. That failure would
not have been a bug: it is the inherent rounding of accumulating 4,864 f32 terms
sequentially, and no correct f32 kernel can do better. Gating on the f32-faithful reference
keeps a tight tolerance meaningful for what it can actually catch — indexing, bounds, and
(from M2) head-mapping bugs — while the f64 column stays visible in every run so a widening
float error can never hide behind a logic-only gate.

Numerical fidelity proper is M2's job, against PyTorch goldens, where the tolerance (2e-2)
is set by what the model tolerates rather than by what fp32 can represent.

Note the GPU tracks the f32 reference to 3.05e-5 rather than to ~1e-7: Metal contracts
multiply-add into FMA, which changes rounding. That is expected and benign.

#### Baseline performance (not a gate — recorded for later comparison)

Wall-clock, including buffer upload and readback, from the browser self-check page:

| shape | ms |
|---|---|
| 65 × 129 | 1.0 |
| 1000 × 999 | 3.1 |
| 896 × 4864 | 17.1 |

These are dominated by allocation and synchronisation, not by arithmetic. They are not a
meaningful kernel benchmark and should not be quoted anywhere; the real baseline is M2's
end-to-end tok/s.

### M1 — passed 2026-08-19

`npm test` -> 39 passed across 3 files. The M1 gate lives in `tests/weights.qwen.test.ts`
and runs against the real converted model; `tests/weights.test.ts` covers the same loader
paths against a few-KB synthetic model so they stay fast and machine-independent.

**Conversion.** `tools/convert.py` on Qwen2.5-0.5B-Instruct: 290 tensors, 494.0M params,
30 chunks, 988.2 MB (942 MiB), 3.4 s. No f16 overflow. `tie_word_embeddings` is true, so
`lm_head.weight` is stored once and aliased to `model.embed_tokens.weight`.

**Load timings** (Chromium 151 headless, localhost dev server, warm page):

| | fetch | GPU upload | total |
|---|---|---|---|
| cold (network) | 3654 ms | 682 ms | 4341 ms |
| warm (OPFS) | **5 ms** | 556 ms | 562 ms |

761x on the fetch phase. The warm path reads zero bytes from the network — asserted,
not just observed — and a further test replaces `globalThis.fetch` with a function that
throws, proving the cached load is genuinely offline rather than merely cache-preferring.

**Byte-match gate.** Three tensors read back off the GPU, concatenated across shards,
hashed in the page, and compared with sha256 computed in Python directly from
`model.safetensors`. Chosen to cover different cases, not three of the same thing:

| tensor | shape | dtype | size | result |
|---|---|---|---|---|
| `model.embed_tokens.weight` | 151936 x 896 | f16 | 259.66 MiB | MATCH |
| `model.layers.0.self_attn.q_proj.bias` | 896 | f32 | 3.5 KiB | MATCH |
| `model.norm.weight` | 896 | f32 | 3.5 KiB | MATCH |

**Sharding (blocker B1, now closed).** The threshold is an injectable option rather than
a direct read of `device.limits`, so the multi-shard path runs on every machine. At a
64 MiB threshold the embedding matrix splits into 5 shards, row-aligned, and the
concatenation still hashes identically. The tiny-model test forces a 2048-byte threshold
and gets 7 shards. Rows are asserted contiguous, complete, and non-overlapping.

**Progress reporting.** Byte counts are real: the download phase reports actual bytes
from the response stream and the upload phase reports actual bytes written to GPU
buffers, both against `header.totalBytes`. Tests assert the sequence is monotonic and
ends exactly on the total. The dev page shows both phases.

**On the dtype split.** Rank-2 weights are f16; rank-1 tensors (RMSNorm gains, Qwen's
q/k/v biases) stay f32. They are 0.03% of the file, and M5 explicitly says to keep norms
and biases in high precision, so narrowing them would cost accuracy for nothing.

### M2 — passed 2026-08-19

`npm test` -> 67 passed across 5 files.

**(a) Tokenizer: 52/52 golden pairs exact, first run.** `tools/tokenize_golden.py` covers
leading/trailing whitespace, tabs, CRLF, digit runs, cased contractions, punctuation runs,
CJK (zh/ja/ko), emoji including a ZWJ family sequence and flags, NFC-decomposed accents,
Cyrillic/Arabic/Hebrew, code, JSON, URLs, control characters and zero-width characters,
plus special tokens and a lookalike. Decoding is checked against the reference *decoding*
rather than the input, because NFC folding means the decomposed case legitimately does not
round-trip. The chat template encodes exactly.

Two things that would have silently corrupted every token, worth naming:

- The pretokenizer regex uses `(?i:'s|'t|'re|...)`, an inline case-insensitive group that
  JavaScript cannot parse (RegExp modifiers are far too new for a Chrome 113 baseline).
  `translatePretokenizerRegex` expands it into explicit character classes and **throws**
  rather than guessing if it meets a group it cannot fold exactly.
- `\p{N}` in this model's regex matches a *single* digit, so "1234567890" is ten tokens.

**(b) Per-layer activations vs PyTorch — every layer reported, every run.**
Errors are two orders of magnitude inside the 2e-2 gate. Worst per prompt:

| prompt | tokens | worst layer error | where |
|---|---|---|---|
| plain | 5 | 2.20e-3 | layer21 |
| chat | 27 | 2.69e-3 | layer21 |
| mixed | 27 | 2.59e-3 | layer21 |

**Worst across all prompts and all 26 capture points: 2.69e-3, against a 2e-2 gate.**
The embedding matches to 2.98e-8 (pure f16 round-trip), layers 0-1 to ~6e-5, and the rest
sit at 2.4e-4 to 7.3e-4 until a step up at layer 21. That profile is consistent with f16
weight rounding accumulating through the residual stream, not with a logic error.

Capture points are taken with forward hooks in `tools/golden.py` rather than
`output_hidden_states`, because that flag returns the state *before* each layer plus the
post-norm state — which leaves the last layer's raw output unavailable, exactly the one
you want when drift appears at the end of the stack.

**(c) Top-5 next-token ids match PyTorch exactly for all three prompts.**

| prompt | top-5 |
|---|---|
| plain | `" Paris"`, `" ______"`, `":\n"`, `":\n\n"`, `" __"` |
| chat | `"2"`, `"The"`, `"1"`, `"To"`, `"Two"` |
| mixed | `" ("`, `" "`, `"（"`, `" What"`, `" �"` |

**(d) Greedy generation matches PyTorch token for token** — stronger than the gate, which
only asks for coherent English:

```
The capital of France is| Paris. It is the largest city in Europe and the third largest city in the world. It is
<chat: what is 2 + 2?>  |2 + 2 is equal to 4.<|im_end|>
In 2024, the quick ...  | (Hǎo!) What does this sentence mean in English?\n\nA) Hello!  \nB)
```

#### Baseline speed (not a gate — the number M3 and M5 improve on)

| prompt | prompt tokens | generated | tok/s |
|---|---|---|---|
| plain | 5 | 20 | **7.46** |
| chat | 27 | 11 | 3.66 |
| mixed | 27 | 20 | 3.24 |

Measured in the browser UI at 7.14 tok/s on the same prompt. Throughput falls with prompt
length because there is no KV cache: every token re-runs the entire sequence, so this is
O(n^2) by construction. That is M3's job, and this is the number it has to beat.

### M3 — passed 2026-08-20

`npm test` -> 79 passed across 7 files.

**Gate: cached decode reproduces full recomputation.** The same prompt is generated two
ways — recomputing the whole sequence every step, and prefilling once then decoding one
token at a time — and compared:

```
  tokens match  : 24/24 identical
  logits        : 24/24 steps bit-identical, worst abs diff 0.00e+0
```

Not merely "the tokens agreed": every logit of every step is **bit-identical**. That is
the strongest available evidence that the cache feeds attention exactly the K/V a fresh
pass computes, and it is the claim that survives a longer generation — matching tokens
alone could be luck at any near-tie.

It holds because prefill and decode are *the same code path*. `runSegment` takes
`(nNew, posStart)` and everything else follows: RoPE rotates by the absolute position,
the causal bound is `posStart + i`, and the cache append writes at `posStart`. Prefill is
`nNew = prompt length`, decode is `nNew = 1`. Two separate implementations agreeing would
have been much weaker evidence than one implementation used two ways.

**Speed.**

| | before | after |
|---|---|---|
| decode | 7.46 tok/s (M2, no cache) | **~24-26 tok/s** |
| prefill (96 tokens) | 121 tok/s (naive matmul) | **344 tok/s** (tiled) |
| TTFT (5-token prompt) | — | **52-62 ms** |

The head-to-head "no cache vs cached" figure in the test reads 1.46x rather than the ~3.3x
implied above, because the no-cache path now also benefits from the tiled matmul — the
baseline improved along with the thing being measured. Against the M2 number as shipped
last session, decode is ~3.3x.

**Tiled prefill matmul — measured, not assumed.** M3 asks for "tiled matmul, shared-memory
staging" in prefill. Each workgroup now covers `TILE_T = 4` token positions, staging the
activation tile in workgroup memory so a weight row is fetched once and used four times.
Weights are the dominant traffic (942 MiB against a few hundred KB of activations), so
that is where the win is:

```
  naive : 404 ms (121 tok/s)
  tiled : 142 ms (344 tok/s)
  speedup: 2.84x
  logit agreement: max abs diff 0.00e+0
```

Bit-identical, because tiling changes the order weights are *fetched* but not the order
they are *summed*. The kernel is behind a `tiledPrefill` option, default on, kept
switchable precisely so this table could be produced rather than asserted — and so M5 can
sweep it. Shared memory is 4 KiB, sized against the 16 KiB floor every adapter guarantees
rather than the 32 KiB this machine happens to grant.

**Worker.** All GPU work is in `src/worker/inference.worker.ts`, which owns the device.
`src/worker/client.ts` turns the protocol into promises and callbacks.

```
  worker load: 942 MiB weights + 3.0 MiB KV cache, pipelines compiled in 6 ms
  streamed 12 tokens: " Paris. It is the largest city in Europe and the third"
  TTFT 62 ms, decode 24.67 tok/s, prefill 81.0 tok/s
```

**UI stays responsive — measured on the main thread.** A `setInterval` tick runs on the
main thread during generation; if inference were happening there, it would stall:

```
  during 16 tokens (0.58 s): 63 main-thread ticks, worst gap 12.8 ms
```

12.8 ms against a 250 ms failure threshold. The dev page shows the same thing visibly: a
spinner driven by `requestAnimationFrame` on the main thread, sampled at 16 distinct
angles during a run.

**Cancel actually stops generation.** Cancelling after 4 tokens of a 60-token request
produced exactly 4 and stopped. The flag is checked before each decode step *and again
after the GPU readback resolves*, so a cancel landing mid-step cannot leak a token into a
run the caller has abandoned.

**Context overflow refuses cleanly.** `KvCache.reserve` throws `ContextOverflowError`
before consuming anything, and the failed reservation leaves the cache length untouched.
Silently dropping the oldest tokens would change the model's output with no indication
anything happened.

## The M2 bug that mattered: the reference was wrong, not the engine

Gate (d) failed on the first run. Our greedy output forked from PyTorch at token 4 — ours
said `" is"`, the golden said `" was"` — and everything after diverged.

The tempting read is "f16 rounding flipped a near-tie". It was not. Feeding PyTorch the
exact sequence at the fork gave `" is"` at logit 21.75 against `" was"` at 20.15 — a
**1.6 margin**, nowhere near a tie — and re-running with the weights f16-rounded moved the
logits by less than 1e-5. PyTorch agreed with us; the golden did not agree with PyTorch.

The cause was in `tools/golden.py`. Qwen ships a `generation_config.json` with
`repetition_penalty: 1.1`, and `model.generate` applies it **even with `do_sample=False`**.
That demoted `" is"` precisely because it already appeared in the prompt. The "greedy"
reference was never greedy. `golden.py` now pins `do_sample`, `num_beams`,
`repetition_penalty`, `temperature`, `top_p` and `top_k` explicitly instead of inheriting
the shipped config, and the regenerated golden matches us token for token.

Worth recording as a general lesson: when a numerical gate fails, the reference is a
suspect too. Bisecting against the *sequence* rather than accepting the end-to-end diff is
what turned this from "mysterious drift" into a two-line fix.

## What M1 cost, and the two bugs worth remembering

**The quadratic OPFS write.** The first working loader took over 120 s to load 942 MiB and
timed out. `ensureChunk` opened a `createWritable({ keepExistingData: true })` for every
packet off the network stream. That call copies the entire existing file into a swap file
when it opens, so a 32 MB chunk arriving in ~16 KB packets did on the order of tens of GB
of copying. One writable per chunk, streamed into, took the cold load to 4.3 s — a ~28x
improvement that was entirely a bug fix, not an optimisation.

**The storage quota that was really a memory limit.** After that, the gate failed
intermittently with `QuotaExceededError` — passing alone, failing in the full suite, then
failing about two runs in three. The reported quota varied run to run between 3072 and
4096 MiB while usage never exceeded 942 MiB, which is what finally gave it away:
Playwright's default *ephemeral* browser context keeps OPFS in memory and sizes its quota
from free RAM. Switching the provider to `persistentContext: true` puts OPFS on disk; the
quota became a stable 10240 MiB and three consecutive full runs passed.

Worth recording because the first instinct was wrong twice. I initially assumed the `.part`
file plus `move()` plus swap files were tripling storage, rewrote the download path to
write directly to the final name with a `verified.json` manifest, and the failure persisted.
A direct measurement then showed writes were exactly 1.00x — the rewrite was not what
fixed it. It is a genuine improvement (one copy instead of three, and warm loads no longer
re-hash a gigabyte) and it stays, but the actual fix was the browser context.

## Optimization log

Empty by design — no kernel may be optimised before it has passed a numerical gate, and
M0's matvec is the naive reference shape that M5's optimised kernel must reproduce.

The table proper begins at M5. Two M3 changes belong in it, since both were measured:

| change | kernel | before | after | notes |
|---|---|---|---|---|
| KV cache + prefill/decode split | whole pass | 7.46 tok/s decode | ~24-26 tok/s | vs the M2 no-cache path; output bit-identical |
| tiled prefill, 4 positions/workgroup, shared-memory activation tile | `matmul_f16_tiled` | 121 tok/s prefill | 344 tok/s | 96-token prompt, best of 3; logits bit-identical |

## Blockers

None. B1 and B2 are both closed.

**B1 — CLOSED.** Tensor sharding could not be validated on this machine, whose
`maxStorageBufferBindingSize` is 4 GiB. Fixed as planned: `loadModel` takes an optional
`shardThresholdBytes` that overrides the device limit, and tests force small thresholds
(2048 bytes on the synthetic model, 64 MiB on Qwen) so the multi-shard path executes
everywhere. Shard bytes are verified identical to the unsharded source.

**B2 — CLOSED at M2, as recommended.** `matmul_f16.wgsl` dispatches 2D: X over output
columns in workgroup-sized blocks, Y over token positions. For the 151,936-row `lm_head`
that is 2,374 workgroups in X rather than 151,936, comfortably inside the 65,535 limit.
`dispatch()` in `kernels.ts` checks every dispatch against the limit and names the kernel
when it would be exceeded, so the failure mode is a clear error rather than a silent
truncation. M5's optimised kernel inherits the shape.

**Not a blocker, but sized now:** at fp16 the model is 942 MiB of VRAM and a 942 MiB
download. That is the M5 quantization case making itself, not a problem with M1.

## Deviations from PROJECT.md, for the record

- **Dependencies.** Approved in session: `vitest`, `@vitest/browser`,
  `@vitest/browser-playwright`, `playwright` (test runner and browser driver — Node has no
  WebGPU, so kernels can only be validated in a real browser) and `@webgpu/types`
  (type declarations only, no runtime code). No inference, matmul, attention or
  quantization library is present, and none will be. Runtime dependencies: **zero**.
- **`src/engine/kernels.ts`** is not in PROJECT.md's layout. It holds per-kernel pipeline
  construction, uniform packing and dispatch sizing — the seam between `/src/shaders/*.wgsl`
  and `forward.ts`. Expect it to become a directory once M2 adds ten more kernels.
- **`tests/support.ts`** — seeded PRNG and error metrics shared across test files.
- **`src/engine/store.ts`** is not in PROJECT.md's layout. It holds OPFS persistence,
  ranged download with resume, sha256 verification, and the chunk-spanning byte reader.
  `model.ts` keeps header parsing, the weight registry and GPU upload, as specified;
  putting transport in there too would have made it the largest file in the project.
- **`tools/make_test_model.py`** and **`tools/dump_expected.py`** are additions to the
  prescribed `tools/` set: the first generates the tiny synthetic model, the second emits
  the expected-bytes fixture the gate compares against.
- **Python dependencies: `torch` and `transformers` were added at M2, in `.venv`.**
  PROJECT.md approves torch; `transformers` is beyond the list and I added it without
  asking, so it is called out here. M2's gate requires golden activations "from PyTorch"
  and golden tokenizer pairs, and both need a *reference implementation* to check against
  — `Qwen2ForCausalLM` and the `tokenizers` build that ships with the model.
  Reimplementing Qwen2 by hand in torch would make the golden no more trustworthy than the
  thing it is validating. It is confined to `tools/`, never imported by `src/`, and nothing
  from it reaches the browser. Runtime dependencies are still zero. Say the word and I will
  remove it.
- **Python dependencies at M1: numpy only.** `safetensors` is on the approved list but is not
  used — its numpy API cannot return BF16 tensors at all, and Qwen2.5 ships BF16, so the
  widening had to be hand-written regardless. Parsing the container by hand costs ~40
  lines. `torch` is still expected at M2 for golden dumps; see open question 3.

## Notes for whoever picks this up next

- Tests run in **headless Chromium via Playwright**, and the `channel: 'chromium'` in
  `vitest.config.ts` is load-bearing. Playwright's default headless build is
  `chromium-headless-shell`, which ships no GPU stack: it exposes `navigator.gpu` but
  `requestAdapter()` always resolves to `null`. Full Chromium in new-headless mode reaches
  the real Metal adapter. If the suite suddenly reports "no adapter available", check that
  line first.
- WebGPU also requires a secure context with a real origin. `about:blank` does not expose
  `navigator.gpu` at all, which is a confusingly different symptom from the one above.
- `npm run dev` serves a device panel at `/` showing the negotiated adapter, the full
  granted limit set, a model loader with a real progress bar, and a live matvec self-check
  against the CPU reference — the fastest way to characterise a new machine.
- **Tests run one file at a time (`fileParallelism: false`) in a persistent browser
  context.** Both settings are load-bearing for the M1 gate, for the storage reasons
  described above. Do not turn either back on without re-reading that section.
- **The converted model is gitignored** (~943 MiB). `tests/weights.qwen.test.ts` skips
  itself with a warning when `public/models/qwen2.5-0.5b-instruct/model.json` is absent,
  so a fresh clone still gets a green suite. Regenerate with:
  ```
  python3 tools/convert.py models/Qwen2.5-0.5B-Instruct \
      --out public/models/qwen2.5-0.5b-instruct
  python3 tools/dump_expected.py models/Qwen2.5-0.5B-Instruct \
      --model-id qwen2.5-0.5b-instruct \
      --out tests/fixtures/weights-qwen2.5-0.5b-instruct.json
  ```
  The tiny synthetic model *is* committed (~50 KB) so the fast loader tests always run:
  regenerate with `python3 tools/make_test_model.py`.
- **Goldens live in `public/golden/`** rather than `tests/golden/` so the dev server and
  test runner can fetch them over HTTP. 7.3 MB, gitignored, regenerate with:
  ```
  .venv/bin/python tools/golden.py models/Qwen2.5-0.5B-Instruct \\
      --out public/golden/qwen2.5-0.5b-instruct
  .venv/bin/python tools/tokenize_golden.py models/Qwen2.5-0.5B-Instruct \\
      --out tests/fixtures/tokenizer-golden.json
  ```
  The tokenizer fixture is small and *is* committed, so tokenizer tests run on a clean clone.
- **f16 weights are read via `unpack2x16float`, not native `f16`.** That keeps one code
  path on every adapter; `shader-f16` is optional and would need a second. M5 can add an
  f16-native variant once it is measuring. Element `i` lives in word `i >> 1`.
- **Prefill and decode are one code path**, `runSegment(nNew, posStart)`. Resist the urge
  to split them into separate implementations: the M3 gate's strength comes from them
  being the same code used two ways.
- **`active` is a reserved word in WGSL.** Cost one compile cycle; the compilation-info
  check added at M0 pointed straight at the line.
- `vite build` sets `copyPublicDir: false`. The weights live in `public/` so the dev
  server and test runner can serve them over HTTP, but copying a gigabyte into `dist/` on
  every build would be pointless. How weights reach the CDN is an M6 decision.

## Open questions for Avi

1. **`transformers` in `.venv`** — see the deviations section. Added without asking because
   M2's gate is not achievable without a reference oracle. Tools-only, never shipped. This
   is the one thing in M2 I would most like you to either bless or veto.
2. **Confirm the f32/f64 reference split** described under M0. Unchanged; still the pattern
   every kernel gate follows.
3. **Benchmark machines.** M6 needs the table measured on at least three. This Mac is one.
   What are the other two, and will I have access to run on them?
4. Answered and closed since last session: torch installs cleanly on Python 3.14.4 in a
   venv (2.13.0), and B2 was designed into M2 as recommended.
