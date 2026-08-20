/**
 * The README benchmark table, measured rather than estimated.
 *
 * PROJECT.md requires the table to carry real numbers from named devices, on at least
 * three machines. This file exists so that "run it on another machine" is one command
 * rather than a research task: it prints finished Markdown rows, and whoever runs it
 * pastes them under the device they ran on.
 *
 *   npm run bench -- tests/readme.bench.test.ts
 *
 * Every model it can find is measured; missing ones are skipped, so a machine with only
 * the int4 models still produces its rows. Regeneration commands are in PROGRESS.md.
 *
 * Decode is best-of-N: run-to-run spread on this hardware is the same size as the
 * effects being measured, and two M5 changes were misread before that was understood.
 */

import { beforeAll, afterAll, describe, it } from 'vitest';

import { requestGpuContext, type GpuContext } from '../src/engine/device.js';
import { PipelineCache } from '../src/engine/pipelines.js';
import { loadModel, type LoadedModel } from '../src/engine/model.js';
import { ForwardPass, requireLogits } from '../src/engine/forward.js';
import { argmax } from '../src/engine/sampler.js';
import { BpeTokenizer } from '../src/tokenizer/bpe.js';

interface Candidate {
  id: string;
  model: string;
  quant: string;
}

const CANDIDATES: Candidate[] = [
  { id: 'qwen2.5-0.5b-instruct', model: 'Qwen2.5-0.5B-Instruct', quant: 'fp16' },
  { id: 'qwen2.5-0.5b-instruct-q4', model: 'Qwen2.5-0.5B-Instruct', quant: 'int4 (block 32)' },
  { id: 'smollm2-360m-instruct', model: 'SmolLM2-360M-Instruct', quant: 'fp16' },
  { id: 'smollm2-360m-instruct-q4', model: 'SmolLM2-360M-Instruct', quant: 'int4 (block 32)' },
];

/** Long enough that prefill is doing real tiled work rather than measuring overhead. */
const PROMPT = [
  'The history of computing is a history of abstraction. Each layer that programmers',
  'stopped having to think about became a layer that someone else had to implement well,',
  'and the machines underneath grew stranger even as the languages on top grew calmer.',
  'A graphics processor is the clearest example: thousands of lanes that are fast only',
  'when they all agree about what to do next, hidden behind an interface that pretends',
  'otherwise. Writing kernels by hand is a way of refusing that pretence for a while.',
].join(' ');

const DECODE_STEPS = 48;
const DECODE_RUNS = 5;

let ctx: GpuContext;

beforeAll(async () => {
  ctx = await requestGpuContext();
}, 900_000);

afterAll(() => {
  ctx?.device.destroy();
});

interface Row extends Candidate {
  vramMiB: number;
  prefillTokPerSec: number;
  ttftMs: number;
  decodeTokPerSec: number;
  promptTokens: number;
}

async function measure(candidate: Candidate): Promise<Row | null> {
  const base = new URL(`/models/${candidate.id}/`, location.href).href;
  if (!(await fetch(new URL('model.json', base).href, { method: 'HEAD' })).ok) {
    console.log(`  [skip] ${candidate.id} not converted on this machine`);
    return null;
  }

  const tokenizer = await BpeTokenizer.fromUrl(new URL('tokenizer.json', base).href);
  const model: LoadedModel = await loadModel(ctx.device, {
    baseUrl: base,
    modelId: `${candidate.id}-readme`,
  });
  const pass = await ForwardPass.create(ctx.device, model, new PipelineCache(ctx.device), {
    maxSeqLen: 512,
  });

  try {
    const ids = tokenizer.encode(PROMPT);

    // One untimed pass first: the first prefill through a new pipeline pays warm-up the
    // timed runs should not be charged for.
    pass.reset();
    await pass.prefill(ids);

    let bestPrefill = 0;
    let bestTtft = Infinity;
    for (let run = 0; run < 3; run++) {
      pass.reset();
      const started = performance.now();
      const prefilled = await pass.prefill(ids);
      const ttft = performance.now() - started;
      requireLogits(prefilled);
      bestTtft = Math.min(bestTtft, ttft);
      bestPrefill = Math.max(bestPrefill, (ids.length / ttft) * 1000);
    }

    let bestDecode = 0;
    for (let run = 0; run < DECODE_RUNS; run++) {
      pass.reset();
      let next = argmax(requireLogits(await pass.prefill(ids)));
      for (let i = 0; i < 4; i++) next = argmax(requireLogits(await pass.decode(next)));

      const started = performance.now();
      for (let i = 0; i < DECODE_STEPS; i++) {
        next = argmax(requireLogits(await pass.decode(next)));
      }
      bestDecode = Math.max(bestDecode, (DECODE_STEPS / (performance.now() - started)) * 1000);
    }

    return {
      ...candidate,
      vramMiB: model.stats.vramBytes / 1048576,
      prefillTokPerSec: bestPrefill,
      ttftMs: bestTtft,
      decodeTokPerSec: bestDecode,
      promptTokens: ids.length,
    };
  } finally {
    pass.destroy();
    model.weights.destroy();
  }
}

describe('README benchmark table', () => {
  it('measures every converted model and prints its Markdown rows', async () => {
    const rows: Row[] = [];
    for (const candidate of CANDIDATES) {
      const row = await measure(candidate);
      if (row) rows.push(row);
    }
    if (rows.length === 0) {
      console.log('  no converted models found; see PROGRESS.md for the conversion commands');
      return;
    }

    const adapter = ctx.info.vendor
      ? `${ctx.info.vendor}${ctx.info.architecture ? ` / ${ctx.info.architecture}` : ''}`
      : 'unknown adapter';

    console.log(
      `\n=== README rows (adapter: ${adapter}, prompt ${rows[0].promptTokens} tokens, ` +
        `decode best of ${DECODE_RUNS} x ${DECODE_STEPS} steps) ===\n\n` +
        '| Device | GPU | Browser | Model | Quant | Prefill tok/s | Decode tok/s | TTFT | VRAM |\n' +
        '|---|---|---|---|---|---|---|---|---|\n' +
        rows
          .map(
            (r) =>
              `| _fill in_ | ${adapter} | _fill in_ | ${r.model} | ${r.quant} | ` +
              `${r.prefillTokPerSec.toFixed(0)} | ${r.decodeTokPerSec.toFixed(1)} | ` +
              `${r.ttftMs.toFixed(0)} ms | ${r.vramMiB.toFixed(0)} MiB |`,
          )
          .join('\n') +
        '\n\n  Device and browser are not reliably readable from the page -- Chromium reports\n' +
        '  no device string for Apple adapters -- so they are filled in by hand from the host.\n',
    );
  }, 3_600_000);
});
