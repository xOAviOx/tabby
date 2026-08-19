import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { describeContext, requestGpuContext, type GpuContext } from '../src/engine/device.js';
import { PipelineCache } from '../src/engine/pipelines.js';
import { readF32, uploadStorageBuffer } from '../src/engine/buffers.js';
import {
  DEFAULT_MATVEC_WORKGROUP,
  DEFAULT_WORKGROUP,
  createKernels,
  dispatch,
  groupsFor,
  runMatvecF32,
  uniforms,
  type KernelSet,
} from '../src/engine/kernels.js';
import { BufferArena } from '../src/engine/buffers.js';
import { bindGroup } from '../src/engine/pipelines.js';
import { withErrorScopes } from '../src/engine/device.js';
import {
  attnOutput,
  attnScores,
  embedGather,
  matmul,
  matvecF32,
  matvecF64,
  residualAdd,
  rmsNorm,
  rope,
  siluMul,
  softmaxRows,
} from '../src/reference/cpu.js';
import { errorStats, fmt, randomF16, randomF32 } from './support.js';

/**
 * The M0 gate: max abs error against the f32-faithful CPU reference.
 * The f64 error is measured and printed alongside it but not asserted -- see the
 * header of src/reference/cpu.ts for why the two are separated.
 */
const GATE_TOLERANCE = 1e-4;

let ctx: GpuContext;
let cache: PipelineCache;

beforeAll(async () => {
  ctx = await requestGpuContext();
  cache = new PipelineCache(ctx.device);
});

afterAll(() => {
  ctx?.device.destroy();
});

describe('device', () => {
  it('negotiates and reports limits', () => {
    // The gate requires these to be printed; they are also the numbers M1 shards against.
    console.log(`\n=== negotiated WebGPU context ===\n${describeContext(ctx)}\n`);
    expect(ctx.device).toBeDefined();
    expect(ctx.limits.maxStorageBufferBindingSize).toBeGreaterThanOrEqual(128 * 1024 * 1024);
    expect(ctx.limits.maxComputeInvocationsPerWorkgroup).toBeGreaterThanOrEqual(256);
  });
});

describe('buffers', () => {
  it('round-trips f32 data through upload and readback byte-exactly', async () => {
    // The readback helper is the only way to inspect a WGSL intermediate, so it is
    // itself under test before anything relies on it.
    const data = randomF32(1024, 0xc0ffee);
    const buffer = uploadStorageBuffer(ctx.device, data, { label: 'roundtrip' });
    try {
      const out = await readF32(ctx.device, buffer, data.length);
      expect(Array.from(out)).toEqual(Array.from(data));
    } finally {
      buffer.destroy();
    }
  });

  it('reads back a non-4-aligned element count', async () => {
    const data = randomF32(7, 0x1234);
    const buffer = uploadStorageBuffer(ctx.device, data, { label: 'unaligned' });
    try {
      const out = await readF32(ctx.device, buffer, 5);
      expect(Array.from(out)).toEqual(Array.from(data.subarray(0, 5)));
    } finally {
      buffer.destroy();
    }
  });
});

interface Shape {
  m: number;
  n: number;
  why: string;
}

const SHAPES: Shape[] = [
  { m: 1, n: 1, why: 'degenerate 1x1' },
  { m: 3, n: 7, why: 'tiny, both dims non-power-of-2, single partial workgroup' },
  { m: 64, n: 64, why: 'exactly one full workgroup' },
  { m: 65, n: 129, why: 'one row past a workgroup boundary; exercises the tail guard' },
  { m: 1000, n: 999, why: 'many workgroups, both dims non-power-of-2' },
  { m: 896, n: 4864, why: 'Qwen2.5-0.5B down_proj shape: short output, long reduction' },
  { m: 4864, n: 896, why: 'Qwen2.5-0.5B gate_proj shape: long output, short reduction' },
];

describe('matvec_f32 vs CPU reference', () => {
  for (const { m, n, why } of SHAPES) {
    it(`[${m} x ${n}] ${why}`, async () => {
      const seed = m * 7919 + n;
      const w = randomF32(m * n, seed);
      const x = randomF32(n, seed ^ 0x5bf03635);

      const gpu = await runMatvecF32(ctx.device, cache, w, x, m, n);
      expect(gpu.length).toBe(m);

      const refF32 = matvecF32(w, x, m, n);
      const refF64 = matvecF64(w, x, m, n);
      const vsF32 = errorStats(gpu, refF32);
      const vsF64 = errorStats(gpu, refF64);

      console.log(
        `matvec[${m}x${n}] seed=${seed} ` +
          `maxAbsErr(vs f32 ref)=${fmt(vsF32.maxAbs)} rms=${fmt(vsF32.rms)} | ` +
          `maxAbsErr(vs f64 ref)=${fmt(vsF64.maxAbs)} rms=${fmt(vsF64.rms)}`,
      );

      if (vsF32.maxAbs >= GATE_TOLERANCE) {
        throw new Error(
          `matvec[${m}x${n}] seed=${seed}: max abs error ${fmt(vsF32.maxAbs)} at row ` +
            `${vsF32.argmax} (gpu=${gpu[vsF32.argmax]}, ref=${refF32[vsF32.argmax]}) ` +
            `exceeds gate ${fmt(GATE_TOLERANCE)}`,
        );
      }
      expect(vsF32.maxAbs).toBeLessThan(GATE_TOLERANCE);
    });
  }

  it('produces identical results across workgroup sizes', async () => {
    // Nothing about the result may depend on how work is partitioned. This is the
    // invariant every M5 optimisation has to preserve, so it is pinned down now.
    const m = 517;
    const n = 733;
    const w = randomF32(m * n, 0xa11ce);
    const x = randomF32(n, 0xb0b);

    const baseline = await runMatvecF32(ctx.device, cache, w, x, m, n, {
      workgroupSize: DEFAULT_MATVEC_WORKGROUP,
    });
    for (const workgroupSize of [1, 32, 128, 256]) {
      const out = await runMatvecF32(ctx.device, cache, w, x, m, n, { workgroupSize });
      expect(Array.from(out), `workgroupSize=${workgroupSize}`).toEqual(Array.from(baseline));
    }
  });

  it('rejects a shape whose weights do not match the declared dimensions', () => {
    expect(() => matvecF32(new Float32Array(10), new Float32Array(3), 4, 3)).toThrow();
  });
});

// =========================================================================================
// M2 kernels
//
// Each kernel is run against its CPU reference on deliberately awkward shapes: non-power-
// of-2 dimensions, partial workgroups, and grouped-query head counts. Weights are random
// f16 bit patterns widened exactly, so both sides see identical values and any difference
// is kernel logic rather than conversion.
// =========================================================================================

/**
 * Tolerance for kernels whose GPU form uses FMA contraction or a transcendental. Values
 * here are O(1), so this is ~1000x the f32 epsilon and still tight enough to catch any
 * indexing, masking or head-mapping mistake.
 */
const KERNEL_TOLERANCE = 1e-4;

interface KernelRun {
  pipeline: GPUComputePipeline;
  /** Bindings in order, starting at 0. Uniform first by convention. */
  buffers: Array<{ data: ArrayBufferView; readBack?: number }>;
  dispatch: [number, number?, number?];
  label: string;
}

/** Upload, dispatch once, and read back every buffer marked with an element count. */
async function runKernel(run: KernelRun): Promise<Float32Array[]> {
  const arena = new BufferArena(ctx.device);
  try {
    const gpuBuffers = run.buffers.map((entry, index) =>
      index === 0
        ? arena.uniform(entry.data, `${run.label}.dims`)
        : arena.upload(entry.data, { label: `${run.label}.b${index}` }),
    );
    const group = bindGroup(ctx.device, run.pipeline, gpuBuffers, `${run.label}.bind`);

    await withErrorScopes(ctx.device, run.label, () => {
      const encoder = ctx.device.createCommandEncoder({ label: run.label });
      const pass = encoder.beginComputePass({ label: run.label });
      dispatch(pass, run.pipeline, group, ctx.device.limits, run.dispatch, run.label);
      pass.end();
      ctx.device.queue.submit([encoder.finish()]);
    });

    const out: Float32Array[] = [];
    for (const [index, entry] of run.buffers.entries()) {
      if (entry.readBack !== undefined) {
        out.push(await readF32(ctx.device, gpuBuffers[index], entry.readBack));
      }
    }
    return out;
  } finally {
    arena.destroy();
  }
}

/** A tolerance of 0 means the kernel must be bit-exact, so the bound is inclusive. */
function expectClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  label: string,
  tolerance = KERNEL_TOLERANCE,
): void {
  const stats = errorStats(actual, expected);
  console.log(`${label}: maxAbsErr=${fmt(stats.maxAbs)} rms=${fmt(stats.rms)}`);
  if (stats.maxAbs > tolerance) {
    throw new Error(
      `${label}: max abs error ${fmt(stats.maxAbs)} at index ${stats.argmax} ` +
        `(gpu=${actual[stats.argmax]}, ref=${expected[stats.argmax]}) exceeds ${fmt(tolerance)}`,
    );
  }
  expect(stats.maxAbs).toBeLessThanOrEqual(tolerance);
}

describe('M2 kernels vs CPU reference', () => {
  let kernels: KernelSet;
  const WG = DEFAULT_WORKGROUP;

  beforeAll(async () => {
    kernels = await createKernels(cache);
  });

  it('embed_gather reads the right rows', async () => {
    const vocab = 97;
    const hidden = 37; // odd, so rows start on odd f16 element offsets
    const table = randomF16(vocab * hidden, 0x5eed1);
    const ids = new Uint32Array([0, 96, 5, 5, 42, 1]);

    const [gpu] = await runKernel({
      pipeline: kernels.embedGather,
      buffers: [
        { data: new Uint8Array(uniforms.embedGather(ids.length, hidden, 0, vocab)) },
        { data: table.words },
        { data: ids },
        { data: new Float32Array(ids.length * hidden), readBack: ids.length * hidden },
      ],
      dispatch: [groupsFor(hidden, WG), ids.length],
      label: 'embed_gather',
    });

    expectClose(gpu, embedGather(table.values, ids, hidden), 'embed_gather', 0);
  });

  it('embed_gather writes only rows inside its shard', async () => {
    // The sharded path: two shards covering a table together, dispatched separately.
    const vocab = 40;
    const hidden = 8;
    const table = randomF16(vocab * hidden, 0x5eed2);
    const ids = new Uint32Array([3, 25, 39, 0]);
    const split = 24;

    const arena = new BufferArena(ctx.device);
    try {
      const out = arena.storage(ids.length * hidden * 4, { label: 'shard.out' });
      const idBuf = arena.upload(ids, { label: 'shard.ids' });
      const halves = [
        { start: 0, count: split, words: table.words.slice(0, (split * hidden) / 2) },
        {
          start: split,
          count: vocab - split,
          words: table.words.slice((split * hidden) / 2),
        },
      ];

      await withErrorScopes(ctx.device, 'embed shards', () => {
        const encoder = ctx.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        for (const half of halves) {
          const dims = arena.uniform(
            new Uint8Array(uniforms.embedGather(ids.length, hidden, half.start, half.count)),
            'shard.dims',
          );
          const w = arena.upload(half.words, { label: 'shard.w' });
          const group = bindGroup(ctx.device, kernels.embedGather, [dims, w, idBuf, out], 'shard');
          dispatch(
            pass,
            kernels.embedGather,
            group,
            ctx.device.limits,
            [groupsFor(hidden, WG), ids.length],
            'embed shard',
          );
        }
        pass.end();
        ctx.device.queue.submit([encoder.finish()]);
      });

      const gpu = await readF32(ctx.device, out, ids.length * hidden);
      expectClose(gpu, embedGather(table.values, ids, hidden), 'embed_gather(sharded)', 0);
    } finally {
      arena.destroy();
    }
  });

  it('rmsnorm normalises each row independently', async () => {
    const nTokens = 5;
    const hidden = 131;
    const eps = 1e-6;
    const x = randomF32(nTokens * hidden, 0x1234);
    const gain = randomF32(hidden, 0x9a11);

    const [gpu] = await runKernel({
      pipeline: kernels.rmsNorm,
      buffers: [
        { data: new Uint8Array(uniforms.rmsNorm(nTokens, hidden, eps)) },
        { data: x },
        { data: gain },
        { data: new Float32Array(nTokens * hidden), readBack: nTokens * hidden },
      ],
      dispatch: [groupsFor(nTokens, WG)],
      label: 'rmsnorm',
    });

    expectClose(gpu, rmsNorm(x, gain, nTokens, hidden, eps), 'rmsnorm');
  });

  it('matmul_f16 matches with and without bias', async () => {
    for (const withBias of [false, true]) {
      const nTokens = 3;
      const outDim = 71;
      const inDim = 53;
      const w = randomF16(outDim * inDim, 0x9a7 + (withBias ? 1 : 0));
      const x = randomF32(nTokens * inDim, 0x11);
      const bias = randomF32(outDim, 0xb1a5);

      const [gpu] = await runKernel({
        pipeline: kernels.matmul,
        buffers: [
          { data: new Uint8Array(uniforms.matmul(nTokens, outDim, inDim, withBias, 0, outDim)) },
          { data: w.words },
          { data: x },
          { data: withBias ? bias : new Float32Array(4) },
          { data: new Float32Array(nTokens * outDim), readBack: nTokens * outDim },
        ],
        dispatch: [groupsFor(outDim, WG), nTokens],
        label: `matmul_f16(bias=${withBias})`,
      });

      expectClose(
        gpu,
        matmul(w.values, x, nTokens, outDim, inDim, withBias ? bias : null),
        `matmul_f16(bias=${withBias})`,
      );
    }
  });

  it('rope uses the rotate-half pairing', async () => {
    const nTokens = 7;
    const nHeads = 3;
    const headDim = 16;
    const theta = 1_000_000;
    const v = randomF32(nTokens * nHeads * headDim, 0x777);

    const [gpu] = await runKernel({
      pipeline: kernels.rope,
      buffers: [
        { data: new Uint8Array(uniforms.rope(nTokens, nHeads, headDim, 0, theta)) },
        { data: v, readBack: v.length },
      ],
      dispatch: [groupsFor((nHeads * headDim) / 2, WG), nTokens],
      label: 'rope',
    });

    expectClose(gpu, rope(v, nTokens, nHeads, headDim, theta), 'rope');
  });

  it('rope respects a non-zero starting position', async () => {
    const nTokens = 4;
    const nHeads = 2;
    const headDim = 8;
    const theta = 10_000;
    const posStart = 11;
    const v = randomF32(nTokens * nHeads * headDim, 0x5757);

    const [gpu] = await runKernel({
      pipeline: kernels.rope,
      buffers: [
        { data: new Uint8Array(uniforms.rope(nTokens, nHeads, headDim, posStart, theta)) },
        { data: v, readBack: v.length },
      ],
      dispatch: [groupsFor((nHeads * headDim) / 2, WG), nTokens],
      label: 'rope(posStart)',
    });

    expectClose(gpu, rope(v, nTokens, nHeads, headDim, theta, posStart), 'rope(posStart)');
  });

  it('attn_scores maps query heads to KV heads and masks causally', async () => {
    const nTokens = 6;
    const nHeads = 6;
    const nKvHeads = 2; // 3 query heads per KV head
    const headDim = 8;
    const q = randomF32(nTokens * nHeads * headDim, 0xa1);
    const k = randomF32(nTokens * nKvHeads * headDim, 0xb1);

    const [gpu] = await runKernel({
      pipeline: kernels.attnScores,
      buffers: [
        {
          data: new Uint8Array(
            uniforms.attnScores(
              nTokens,
              nTokens,
              nHeads,
              nKvHeads,
              headDim,
              0,
              0,
              1 / Math.sqrt(headDim),
            ),
          ),
        },
        { data: q },
        { data: k },
        {
          data: new Float32Array(nHeads * nTokens * nTokens),
          readBack: nHeads * nTokens * nTokens,
        },
      ],
      dispatch: [groupsFor(nTokens, WG), nTokens, nHeads],
      label: 'attn_scores',
    });

    const shape = { nNew: nTokens, totalLen: nTokens, nHeads, nKvHeads, headDim };
    const expected = attnScores(q, k, shape);
    expectClose(gpu, expected, 'attn_scores');

    // The mask must actually be applied, not merely produce small numbers.
    for (let h = 0; h < nHeads; h++) {
      for (let i = 0; i < nTokens; i++) {
        for (let j = i + 1; j < nTokens; j++) {
          expect(gpu[(h * nTokens + i) * nTokens + j]).toBeLessThan(-1e30);
        }
      }
    }
  });

  it('softmax_rows normalises only the causal prefix', async () => {
    const nHeads = 3;
    const nTokens = 5;
    const rows = nHeads * nTokens;
    const scores = randomF32(rows * nTokens, 0x50f7, 4);

    const [gpu] = await runKernel({
      pipeline: kernels.softmaxRows,
      buffers: [
        { data: new Uint8Array(uniforms.softmaxRows(rows, nTokens, nTokens, 0)) },
        { data: scores, readBack: rows * nTokens },
      ],
      dispatch: [groupsFor(rows, WG)],
      label: 'softmax_rows',
    });

    expectClose(gpu, softmaxRows(scores, rows, nTokens, nTokens), 'softmax_rows');

    for (let row = 0; row < rows; row++) {
      const valid = (row % nTokens) + 1;
      let sum = 0;
      for (let i = 0; i < nTokens; i++) {
        const v = gpu[row * nTokens + i];
        if (i >= valid) expect(v).toBe(0);
        sum += v;
      }
      expect(sum).toBeCloseTo(1, 5);
    }
  });

  it('attn_output weights V by the attention distribution', async () => {
    const nTokens = 6;
    const nHeads = 6;
    const nKvHeads = 3;
    const headDim = 8;
    const raw = randomF32(nHeads * nTokens * nTokens, 0xa77);
    const probs = softmaxRows(raw, nHeads * nTokens, nTokens, nTokens);
    const v = randomF32(nTokens * nKvHeads * headDim, 0xc1);

    const [gpu] = await runKernel({
      pipeline: kernels.attnOutput,
      buffers: [
        {
          data: new Uint8Array(
            uniforms.attnOutput(nTokens, nTokens, nHeads, nKvHeads, headDim, 0, 0),
          ),
        },
        { data: probs },
        { data: v },
        {
          data: new Float32Array(nTokens * nHeads * headDim),
          readBack: nTokens * nHeads * headDim,
        },
      ],
      dispatch: [groupsFor(nHeads * headDim, WG), nTokens],
      label: 'attn_output',
    });

    const shape = { nNew: nTokens, totalLen: nTokens, nHeads, nKvHeads, headDim };
    expectClose(gpu, attnOutput(probs, v, shape), 'attn_output');
  });

  it('residual_add accumulates in place', async () => {
    const n = 517;
    const x = randomF32(n, 0xadd1);
    const delta = randomF32(n, 0xadd2);

    const [gpu] = await runKernel({
      pipeline: kernels.residualAdd,
      buffers: [
        { data: new Uint8Array(uniforms.elementwise(n)) },
        { data: delta },
        { data: x, readBack: n },
      ],
      dispatch: [groupsFor(n, WG)],
      label: 'residual_add',
    });

    expectClose(gpu, residualAdd(x, delta), 'residual_add', 0);
  });

  it('silu_mul applies SwiGLU', async () => {
    const n = 333;
    const gate = randomF32(n, 0x5110, 6);
    const up = randomF32(n, 0x5111, 6);

    const [gpu] = await runKernel({
      pipeline: kernels.siluMul,
      buffers: [
        { data: new Uint8Array(uniforms.elementwise(n)) },
        { data: up },
        { data: gate, readBack: n },
      ],
      dispatch: [groupsFor(n, WG)],
      label: 'silu_mul',
    });

    expectClose(gpu, siluMul(gate, up), 'silu_mul');
  });
});
