# PROGRESS

## Current milestone: M1 — complete. M2 not started.

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

| change | kernel | tok/s before | tok/s after | notes |
|---|---|---|---|---|
| — | — | — | — | first entries land at M5 |

## Blockers

None. B1 is closed; B2 is still open and still shapes M2/M5.

**B1 — CLOSED.** Tensor sharding could not be validated on this machine, whose
`maxStorageBufferBindingSize` is 4 GiB. Fixed as planned: `loadModel` takes an optional
`shardThresholdBytes` that overrides the device limit, and tests force small thresholds
(2048 bytes on the synthetic model, 64 MiB on Qwen) so the multi-shard path executes
everywhere. Shard bytes are verified identical to the unsharded source.

**B2 — still open. `maxComputeWorkgroupsPerDimension` is 65,535, and the vocab is 151,936.**
M5 prescribes "one workgroup per output row" for the quantized matvec. For `lm_head` that
is 151,936 workgroups in X, 2.3x over the limit, and the limit is at the spec default so
no adapter will lift it. Every output-row-per-workgroup kernel needs a 2D dispatch or a
rows-per-workgroup factor. Cheap to design in at M2, expensive to retrofit at M5.
See open question 2 — this is the one decision I would like settled before M2 starts.

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
- **Python dependencies: numpy only.** `safetensors` is on the approved list but is not
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
- `vite build` sets `copyPublicDir: false`. The weights live in `public/` so the dev
  server and test runner can serve them over HTTP, but copying a gigabyte into `dist/` on
  every build would be pointless. How weights reach the CDN is an M6 decision.

## Open questions for Avi

1. **Confirm the f32/f64 reference split** described under M0. Unchanged from last
   session; it sets the pattern for every kernel gate from here on.
2. **B2 (workgroups-per-dimension vs vocab size) — I would like this settled before M2.**
   My recommendation is to design the rows-per-workgroup factor in from the start. The
   milestone text says naive, but 151,936 output rows against a 65,535 dispatch limit is
   a correctness requirement rather than an optimisation, and no adapter will lift it.
   Treating "naive" as "untiled, one thread per output" and still taking the 2D dispatch
   costs a few lines at M2 and saves rewriting every matvec at M5.
3. **`torch` on Python 3.14.** M2 needs `tools/golden.py` to dump PyTorch reference
   activations, and this machine runs Python 3.14.4, which is newer than most PyTorch
   wheels support. M1 did not need it (conversion is numpy-only). If torch will not
   install, the fallback is a 3.12 virtualenv used only by `tools/`. Flagging before M2
   rather than discovering it at the golden gate — want me to verify the install now?
4. **Benchmark machines.** M6 needs the table measured on at least three. This Mac is one.
   What are the other two, and will I have access to run on them?
5. **Should I push?** There are now local commits that have never been pushed. The
   session hook has been pushing its own commits to `xOAviOx/tabby` unasked, but I have
   not pushed mine.
