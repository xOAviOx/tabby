# PROGRESS

## Current milestone: M0 — complete. M1 not started.

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

## Optimization log

Empty by design — no kernel may be optimised before it has passed a numerical gate, and
M0's matvec is the naive reference shape that M5's optimised kernel must reproduce.

| change | kernel | tok/s before | tok/s after | notes |
|---|---|---|---|---|
| — | — | — | — | first entries land at M5 |

## Blockers

None blocking M1. Two findings that change the design of later milestones:

**B1 — this machine cannot validate tensor sharding.**
`maxStorageBufferBindingSize` here is 4 GiB, 32× the 128 MiB spec default. Qwen2.5-0.5B's
`lm_head` (151,936 × 896 fp16 ≈ 272 MB) therefore fits in a single buffer *on this machine
only*; on a default-limit device it does not. If M1's sharding path is only exercised
against real limits, it will never execute here and will break on the first reviewer's
laptop. Plan: make the shard threshold an injectable value rather than reading
`device.limits` directly, and have the M1 test force a small synthetic threshold (e.g.
1 MiB) so the multi-shard path is covered on every machine. `buffers.ts` already routes all
size checks through one `assertFitsLimits` function so there is a single place to inject.

**B2 — `maxComputeWorkgroupsPerDimension` is 65,535, and the vocab is 151,936.**
M5 prescribes "one workgroup per output row" for the quantized matvec. For `lm_head` that
is 151,936 workgroups in X — 2.3× over the limit, and this limit is *at the spec default*,
so no adapter will save us. Every output-row-per-workgroup kernel needs either a 2D dispatch
or a rows-per-workgroup factor from the start. Cheap if designed in at M2, expensive if
retrofitted at M5. Recorded now so it is not rediscovered later.

Neither of these is a reason to pause; both are noted so M1/M2 are built with them in mind.

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
  granted limit set, and a live matvec self-check against the CPU reference — the fastest
  way to characterise a new machine before benchmarking it.

## Open questions for Avi

1. **Confirm the f32/f64 reference split described above.** It is the one judgement call in
   M0 that a reasonable person could make differently, and it sets the pattern for every
   kernel gate from here on. The alternative — gate on f64 and widen the tolerance per
   shape — is defensible but would have meant loosening a stated tolerance at M0, which §7
   forbids.
2. **B2 (workgroups-per-dimension vs vocab size).** Want me to design the 2D-dispatch
   convention into M2's kernels from the start, or keep M2 strictly naive per the milestone
   text and absorb the change at M5? Milestone text says naive; the limit says decide early.
3. **Model source for M1.** Should `tools/convert.py` take a local HF snapshot directory you
   have already downloaded, or should it fetch `Qwen/Qwen2.5-0.5B-Instruct` from the Hub
   itself? The latter needs `huggingface_hub` in the Python toolchain — a dependency outside
   the approved list, hence the question.
4. **Benchmark machines.** M6 needs the table measured on at least three. This Mac is one.
   What are the other two, and will I have access to run on them?
