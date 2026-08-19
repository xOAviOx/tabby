import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { describeContext, requestGpuContext, type GpuContext } from '../src/engine/device.js';
import { PipelineCache } from '../src/engine/pipelines.js';
import { readF32, uploadStorageBuffer } from '../src/engine/buffers.js';
import { DEFAULT_MATVEC_WORKGROUP, runMatvecF32 } from '../src/engine/kernels.js';
import { matvecF32, matvecF64 } from '../src/reference/cpu.js';
import { errorStats, fmt, randomF32 } from './support.js';

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
