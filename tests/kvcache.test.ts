/**
 * Gate M3: incremental decoding must reproduce full recomputation exactly.
 *
 * The comparison runs the same prompt two ways -- recomputing the whole sequence for
 * every token (the M2 path), and prefilling once then decoding one token at a time -- and
 * requires the generated ids to be identical. Logit agreement is measured and reported
 * alongside, because "the tokens happened to match" and "the arithmetic is the same" are
 * different claims and only the second one survives a longer generation.
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { requestGpuContext, type GpuContext } from '../src/engine/device.js';
import { PipelineCache } from '../src/engine/pipelines.js';
import { loadModel, type LoadedModel } from '../src/engine/model.js';
import { ForwardPass } from '../src/engine/forward.js';
import { ContextOverflowError, KvCache } from '../src/engine/kvcache.js';
import { BpeTokenizer } from '../src/tokenizer/bpe.js';
import { argmax } from '../src/engine/sampler.js';
import { errorStats, fmt } from './support.js';

const MODEL_ID = 'qwen2.5-0.5b-instruct';
const MODEL_BASE = new URL(`/models/${MODEL_ID}/`, location.href).href;
const MAX_SEQ_LEN = 128;

let ctx: GpuContext;
let model: LoadedModel | null = null;
let forward: ForwardPass | null = null;
let tokenizer: BpeTokenizer | null = null;

beforeAll(async () => {
  ctx = await requestGpuContext();
  if (!(await fetch(new URL('model.json', MODEL_BASE).href, { method: 'HEAD' })).ok) {
    console.warn('[skip] converted model missing; run tools/convert.py');
    return;
  }
  tokenizer = await BpeTokenizer.fromUrl(new URL('tokenizer.json', MODEL_BASE).href);
  model = await loadModel(ctx.device, { baseUrl: MODEL_BASE, modelId: `${MODEL_ID}-kv` });
  forward = await ForwardPass.create(ctx.device, model, new PipelineCache(ctx.device), {
    maxSeqLen: MAX_SEQ_LEN,
  });
}, 300_000);

afterAll(() => {
  forward?.destroy();
  model?.weights.destroy();
  ctx?.device.destroy();
});

describe('KvCache bookkeeping', () => {
  const config = {
    numHiddenLayers: 3,
    numKeyValueHeads: 2,
    headDim: 4,
  } as ConstructorParameters<typeof KvCache>[1];

  it('lays layers out contiguously and reserves in order', () => {
    const cache = new KvCache(ctx.device, config, { maxSeqLen: 8 });
    try {
      expect(cache.kvDim).toBe(8);
      expect(cache.length).toBe(0);
      expect(cache.layerOffset(0)).toBe(0);
      expect(cache.layerOffset(1)).toBe(8 * 8);
      expect(cache.offsetOf(1, 3)).toBe((1 * 8 + 3) * 8);

      expect(cache.reserve(5)).toBe(0);
      expect(cache.length).toBe(5);
      expect(cache.reserve(2)).toBe(5);
      expect(cache.length).toBe(7);

      cache.reset();
      expect(cache.length).toBe(0);
      expect(cache.reserve(1)).toBe(0);
    } finally {
      cache.destroy();
    }
  });

  it('refuses to overflow rather than wrapping or truncating', () => {
    const cache = new KvCache(ctx.device, config, { maxSeqLen: 4 });
    try {
      cache.reserve(3);
      // Dropping the oldest tokens silently would change the model's output with no
      // indication that anything happened, so this has to be a hard error.
      expect(() => cache.reserve(2)).toThrow(ContextOverflowError);
      // The failed reservation must not have consumed anything.
      expect(cache.length).toBe(3);
      expect(cache.reserve(1)).toBe(3);
    } finally {
      cache.destroy();
    }
  });

  it('rejects a nonsensical capacity', () => {
    expect(() => new KvCache(ctx.device, config, { maxSeqLen: 0 })).toThrow(RangeError);
  });
});

describe('M3 gate: cached decode vs full recomputation', () => {
  it('produces identical tokens, and reports logit agreement', async () => {
    if (!forward || !tokenizer) return;

    const prompt = 'The capital of France is';
    const promptIds = tokenizer.encode(prompt);
    const newTokens = 24;

    // --- path A: recompute the whole sequence every step (the M2 behaviour) ----------
    const sequence = [...promptIds];
    const fullTokens: number[] = [];
    const fullLogits: Float32Array[] = [];
    for (let step = 0; step < newTokens; step++) {
      const { logits } = await forward.runFull(sequence);
      fullLogits.push(logits);
      const next = argmax(logits);
      fullTokens.push(next);
      sequence.push(next);
    }

    // --- path B: prefill once, then decode one token at a time -----------------------
    forward.reset();
    const cachedTokens: number[] = [];
    const cachedLogits: Float32Array[] = [];

    const prefillStart = performance.now();
    const prefilled = await forward.prefill(promptIds);
    const ttftMs = performance.now() - prefillStart;
    cachedLogits.push(prefilled.logits);
    let next = argmax(prefilled.logits);
    cachedTokens.push(next);

    const decodeStart = performance.now();
    for (let step = 1; step < newTokens; step++) {
      const logits = await forward.decode(next);
      cachedLogits.push(logits);
      next = argmax(logits);
      cachedTokens.push(next);
    }
    const decodeMs = performance.now() - decodeStart;

    expect(forward.position).toBe(promptIds.length + newTokens - 1);

    // The gate.
    expect(cachedTokens, 'cached tokens vs full recomputation').toEqual(fullTokens);

    let worstLogitError = 0;
    let identicalSteps = 0;
    for (let step = 0; step < newTokens; step++) {
      const stats = errorStats(cachedLogits[step], fullLogits[step]);
      worstLogitError = Math.max(worstLogitError, stats.maxAbs);
      if (stats.maxAbs === 0) identicalSteps += 1;
    }

    console.log(
      `\n=== M3 gate ===\n` +
        `  prompt        : ${JSON.stringify(prompt)} (${promptIds.length} tokens)\n` +
        `  generated     : ${JSON.stringify(tokenizer.decode(cachedTokens))}\n` +
        `  tokens match  : ${cachedTokens.length}/${fullTokens.length} identical\n` +
        `  logits        : ${identicalSteps}/${newTokens} steps bit-identical, ` +
        `worst abs diff ${fmt(worstLogitError)}\n` +
        `  TTFT          : ${ttftMs.toFixed(0)} ms (prefill of ${promptIds.length} tokens)\n` +
        `  decode        : ${decodeMs.toFixed(0)} ms for ${newTokens - 1} tokens ` +
        `(${(((newTokens - 1) / decodeMs) * 1000).toFixed(2)} tok/s)\n`,
    );

    // Logits are expected to agree to well within f32 noise; a real divergence would
    // mean the cache is feeding attention different K/V than a fresh pass computes.
    expect(worstLogitError).toBeLessThan(1e-3);
  }, 900_000);

  it('reports the speedup over the M2 no-cache path', async () => {
    if (!forward || !tokenizer) return;

    const promptIds = tokenizer.encode('The capital of France is');
    const steps = 12;

    const sequence = [...promptIds];
    const noCacheStart = performance.now();
    for (let step = 0; step < steps; step++) {
      const { logits } = await forward.runFull(sequence);
      sequence.push(argmax(logits));
    }
    const noCacheMs = performance.now() - noCacheStart;

    forward.reset();
    const cachedStart = performance.now();
    let next = argmax((await forward.prefill(promptIds)).logits);
    for (let step = 1; step < steps; step++) {
      next = argmax(await forward.decode(next));
    }
    const cachedMs = performance.now() - cachedStart;

    console.log(
      `  ${steps} tokens: no cache ${noCacheMs.toFixed(0)} ms ` +
        `(${((steps / noCacheMs) * 1000).toFixed(2)} tok/s) -> ` +
        `cached ${cachedMs.toFixed(0)} ms ` +
        `(${((steps / cachedMs) * 1000).toFixed(2)} tok/s), ` +
        `${(noCacheMs / cachedMs).toFixed(2)}x`,
    );
    expect(cachedMs).toBeLessThan(noCacheMs);
  }, 900_000);

  it('refuses cleanly when the context is exhausted', async () => {
    if (!forward || !tokenizer) return;
    forward.reset();

    // Fill the cache to one short of capacity, then ask for two more.
    const ids = new Array(MAX_SEQ_LEN - 1).fill(tokenizer.encode('a')[0]);
    await forward.prefill(ids);
    expect(forward.position).toBe(MAX_SEQ_LEN - 1);

    await forward.decode(ids[0]);
    expect(forward.position).toBe(MAX_SEQ_LEN);

    await expect(forward.decode(ids[0])).rejects.toThrow(ContextOverflowError);
    // The cache must be left intact, not half-advanced.
    expect(forward.position).toBe(MAX_SEQ_LEN);
    forward.reset();
  }, 900_000);
});

describe('tiled prefill matmul', () => {
  it('is measured against the naive kernel rather than assumed to help', async () => {
    if (!model || !tokenizer) return;

    // A long prompt is where prefill cost actually shows up; a 5-token prompt would say
    // nothing. 96 tokens fits the 128-token context these tests use.
    const promptIds = tokenizer.encode(
      'In a quiet town by the sea there lived a clockmaker who believed that every ' +
        'machine keeps a secret. Each morning he opened the shutters, wound the great ' +
        'brass movement in the window, and listened for the one tick that did not belong.',
    );
    const ids = promptIds.slice(0, 96);

    const cachePipelines = new PipelineCache(ctx.device);
    const variants: Array<{
      name: string;
      tiled: boolean;
      ms: number;
      logits: Float32Array<ArrayBufferLike>;
    }> = [];

    for (const tiled of [false, true]) {
      const pass = await ForwardPass.create(ctx.device, model, cachePipelines, {
        maxSeqLen: MAX_SEQ_LEN,
        tiledPrefill: tiled,
      });
      try {
        await pass.runFull(ids); // warm up: first run pays for lazy allocation
        const runs: number[] = [];
        let logits: Float32Array<ArrayBufferLike> = new Float32Array(0);
        for (let i = 0; i < 3; i++) {
          const started = performance.now();
          logits = (await pass.runFull(ids)).logits;
          runs.push(performance.now() - started);
        }
        variants.push({
          name: tiled ? 'tiled' : 'naive',
          tiled,
          ms: Math.min(...runs),
          logits,
        });
      } finally {
        pass.destroy();
      }
    }

    const naive = variants.find((v) => !v.tiled)!;
    const tiled = variants.find((v) => v.tiled)!;
    const stats = errorStats(tiled.logits, naive.logits);

    console.log(
      `\n=== prefill matmul, ${ids.length}-token prompt (best of 3) ===\n` +
        `  naive : ${naive.ms.toFixed(0)} ms (${((ids.length / naive.ms) * 1000).toFixed(0)} tok/s)\n` +
        `  tiled : ${tiled.ms.toFixed(0)} ms (${((ids.length / tiled.ms) * 1000).toFixed(0)} tok/s)\n` +
        `  speedup: ${(naive.ms / tiled.ms).toFixed(2)}x\n` +
        `  logit agreement: max abs diff ${fmt(stats.maxAbs)}\n`,
    );

    // Tiling changes the order weights are fetched but not the order they are summed, so
    // the two kernels should agree to f32 noise at worst.
    expect(stats.maxAbs).toBeLessThan(1e-2);
  }, 900_000);
});
