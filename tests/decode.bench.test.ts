/**
 * M5 decode benchmark. Not a gate -- this is the harness the optimization table is
 * measured with. Run with `npm run bench`.
 *
 * Decode streams the entire weight set through the GPU once per token and does very
 * little arithmetic per byte, so the number that explains everything is achieved
 * bandwidth: weights x tok/s. Reporting tok/s alone hides whether a change helped because
 * it moved fewer bytes or because it moved the same bytes better.
 */

import { beforeAll, afterAll, describe, it } from 'vitest';

import { requestGpuContext, type GpuContext } from '../src/engine/device.js';
import { PipelineCache } from '../src/engine/pipelines.js';
import { loadModel, type LoadedModel } from '../src/engine/model.js';
import { ForwardPass, requireLogits } from '../src/engine/forward.js';
import { argmax } from '../src/engine/sampler.js';
import { BpeTokenizer } from '../src/tokenizer/bpe.js';

const Q4_ID = 'qwen2.5-0.5b-instruct-q4';
const Q4_BASE = new URL(`/models/${Q4_ID}/`, location.href).href;
const STEPS = 48;

let ctx: GpuContext;
let model: LoadedModel | null = null;
let tokenizer: BpeTokenizer | null = null;

beforeAll(async () => {
  ctx = await requestGpuContext();
  if (!(await fetch(new URL('model.json', Q4_BASE).href, { method: 'HEAD' })).ok) {
    console.warn('[skip] int4 model missing; run tools/convert.py --quant 4');
    return;
  }
  tokenizer = await BpeTokenizer.fromUrl(new URL('tokenizer.json', Q4_BASE).href);
  model = await loadModel(ctx.device, { baseUrl: Q4_BASE, modelId: `${Q4_ID}-bench` });
}, 900_000);

afterAll(() => {
  model?.weights.destroy();
  ctx?.device.destroy();
});

async function decodeRate(pass: ForwardPass, ids: number[]): Promise<number> {
  pass.reset();
  let next = argmax(requireLogits(await pass.prefill(ids)));
  for (let i = 0; i < 4; i++) next = argmax(requireLogits(await pass.decode(next)));

  const started = performance.now();
  for (let i = 0; i < STEPS; i++) next = argmax(requireLogits(await pass.decode(next)));
  return (STEPS / (performance.now() - started)) * 1000;
}

describe('quantized decode: workgroup sweep', () => {
  it('sweeps workgroup size against rows per workgroup', async () => {
    if (!model || !tokenizer) return;
    const ids = tokenizer.encode('The history of computing is a history of abstraction.');
    const weights = model.stats.vramBytes;
    const cache = new PipelineCache(ctx.device);

    const results: Array<{ wg: number; rows: number; rate: number; gbps: number }> = [];
    for (const wg of [16, 32, 64, 128]) {
      for (const rows of [2, 4, 8, 16]) {
        // The lm_head has 151,936 output rows; at 1 row per workgroup that is 2.3x over
        // maxComputeWorkgroupsPerDimension, so the low end is only reachable in 2D.
        const pass = await ForwardPass.create(ctx.device, model, cache, {
          maxSeqLen: 128,
          quantWorkgroupSize: wg,
          rowsPerWorkgroup: rows,
        });
        try {
          // Best of two: the first pass through a new pipeline pays for warm-up the
          // in-loop warm-up does not fully cover.
          const rate = Math.max(
            await decodeRate(pass, ids),
            await decodeRate(pass, ids),
            await decodeRate(pass, ids),
          );
          results.push({ wg, rows, rate, gbps: (weights * rate) / 1e9 });
        } catch (error) {
          console.log(`  wg=${wg} rows=${rows}: ${String(error).slice(0, 90)}`);
        } finally {
          pass.destroy();
        }
      }
    }

    results.sort((a, b) => b.rate - a.rate);
    console.log(
      `\n=== decode sweep (${(weights / 1048576).toFixed(0)} MiB int4 weights) ===\n` +
        results
          .map(
            (r) =>
              `  wg=${String(r.wg).padStart(3)} rows=${String(r.rows).padStart(2)}  ` +
              `${r.rate.toFixed(1).padStart(6)} tok/s  ${r.gbps.toFixed(1).padStart(5)} GB/s`,
          )
          .join('\n') +
        `\n  best: wg=${results[0].wg} rows=${results[0].rows} at ${results[0].rate.toFixed(1)} tok/s\n`,
    );
  }, 3_600_000);
});

describe('quantized decode: optimization A/B', () => {
  it('measures each optimization against the configuration without it', async () => {
    if (!model || !tokenizer) return;
    const ids = tokenizer.encode('The history of computing is a history of abstraction.');
    const cache = new PipelineCache(ctx.device);
    const weights = model.stats.vramBytes;

    const configs: Array<{ name: string; fuseSwiglu: boolean; adaptiveRows: boolean }> = [
      { name: 'neither', fuseSwiglu: false, adaptiveRows: false },
      { name: 'fused swiglu only', fuseSwiglu: true, adaptiveRows: false },
      { name: 'adaptive rows only', fuseSwiglu: false, adaptiveRows: true },
      { name: 'both', fuseSwiglu: true, adaptiveRows: true },
    ];

    for (const config of configs) {
      const pass = await ForwardPass.create(ctx.device, model, cache, {
        maxSeqLen: 128,
        fuseSwiglu: config.fuseSwiglu,
        adaptiveRows: config.adaptiveRows,
      });
      try {
        // Best of three: single samples move by several percent run to run, which is the
        // same size as the effect being measured.
        const runs = [
          await decodeRate(pass, ids),
          await decodeRate(pass, ids),
          await decodeRate(pass, ids),
          await decodeRate(pass, ids),
          await decodeRate(pass, ids),
        ];
        const best = Math.max(...runs);
        console.log(
          `  ${config.name.padEnd(18)}: ${best.toFixed(1).padStart(6)} tok/s  ` +
            `${((weights * best) / 1e9).toFixed(1).padStart(5)} GB/s  ` +
            `(runs ${runs.map((r) => r.toFixed(1)).join(', ')})`,
        );
      } finally {
        pass.destroy();
      }
    }
  }, 3_600_000);
});

/**
 * How much of the measured decode rate is the sampler's readback rather than the model?
 *
 * `decodeRate` above -- and therefore the M5 throughput gate -- calls `decode()` with no
 * `topK`, which reads the entire logit vector back to the CPU every step. On this
 * vocabulary that is ~608 KB per token and a full pipeline sync. The chat path never does
 * this; M4 added GPU-side top-k precisely to avoid it. So the gate is timing the engine
 * plus a readback the shipping path does not perform.
 */
describe('quantized decode: what the per-token readback costs', () => {
  it('measures the top-k path against the full-logit readback', async () => {
    if (!model || !tokenizer) return;
    const ids = tokenizer.encode('The history of computing is a history of abstraction.');
    const cache = new PipelineCache(ctx.device);
    const weights = model.stats.vramBytes;
    const pass = await ForwardPass.create(ctx.device, model, cache, { maxSeqLen: 128 });

    async function rate(topK: number): Promise<{ rate: number; bytes: number; encodeMs: number }> {
      pass.reset();
      let next = argmax(requireLogits(await pass.prefill(ids)));
      let bytes = 0;
      let encodeTotal = 0;
      const step = async (): Promise<void> => {
        const out = await pass.decode(next, topK > 0 ? { topK } : {});
        bytes = out.readbackBytes;
        encodeTotal += out.encodeMs;
        next = topK > 0 ? out.topK!.ids[0] : argmax(requireLogits(out));
      };
      for (let i = 0; i < 4; i++) await step();
      encodeTotal = 0;
      const started = performance.now();
      for (let i = 0; i < STEPS; i++) await step();
      return {
        rate: (STEPS / (performance.now() - started)) * 1000,
        bytes,
        encodeMs: encodeTotal / STEPS,
      };
    }

    try {
      for (const [name, topK] of [
        ['full logits (what the gate measures)', 0],
        ['top-k (what the chat path does)', 8],
      ] as Array<[string, number]>) {
        const runs = [await rate(topK), await rate(topK), await rate(topK), await rate(topK), await rate(topK)];
        const best = Math.max(...runs.map((r) => r.rate));
        console.log(
          `  ${name.padEnd(38)}: ${best.toFixed(1).padStart(6)} tok/s  ` +
            `${((weights * best) / 1e9).toFixed(1).padStart(5)} GB/s  ` +
            `${String(runs[0].bytes).padStart(7)} B/token  ` +
            `encode ${runs[0].encodeMs.toFixed(2)} ms of ${(1000 / best).toFixed(2)} ms/token  ` +
            `(runs ${runs.map((r) => r.rate.toFixed(1)).join(', ')})`,
        );
      }
    } finally {
      pass.destroy();
    }
  }, 3_600_000);
});

describe('quantized decode: where the time goes', () => {
  it('profiles one decode step per kernel', async () => {
    if (!model || !tokenizer) return;
    const cache = new PipelineCache(ctx.device);
    const pass = await ForwardPass.create(ctx.device, model, cache, { maxSeqLen: 128 });
    try {
      if (!pass.enableProfiling(true)) {
        console.log('  timestamp-query unavailable; no per-kernel timing on this adapter');
        return;
      }

      const ids = tokenizer.encode('The history of computing is a history of abstraction.');
      pass.reset();
      let next = argmax(requireLogits(await pass.prefill(ids)));
      for (let i = 0; i < 3; i++) next = argmax(requireLogits(await pass.decode(next)));

      // The reported step is a decode, which is what the throughput gate measures.
      await pass.decode(next);
      const report = await pass.profile();

      console.log(
        `\n=== decode step: ${report.passCount} dispatches, ` +
          `${report.totalMs.toFixed(2)} ms of GPU time ===\n` +
          report.kernels
            .slice(0, 14)
            .map(
              (k) =>
                `  ${k.label.padEnd(22)} ${String(k.calls).padStart(4)} calls  ` +
                `${k.totalMs.toFixed(3).padStart(8)} ms  ${(k.fraction * 100).toFixed(1).padStart(5)}%`,
            )
            .join('\n') +
          '\n\n  Note: profiling gives every dispatch its own pass, so absolute time here ' +
          'exceeds\n  the single-pass encoding used normally. The shares are what matter.\n',
      );
    } finally {
      pass.destroy();
    }
  }, 3_600_000);
});
