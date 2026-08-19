/**
 * The M1 gate, against the real converted Qwen2.5-0.5B-Instruct.
 *
 * Separate from weights.test.ts because it moves ~1 GB and needs the converted model
 * present on disk. The converted weights are gitignored, so on a machine that has not
 * run the conversion this file skips rather than fails -- regenerate with:
 *
 *   python3 tools/convert.py models/Qwen2.5-0.5B-Instruct \
 *       --out public/models/qwen2.5-0.5b-instruct
 *   python3 tools/dump_expected.py models/Qwen2.5-0.5B-Instruct \
 *       --model-id qwen2.5-0.5b-instruct \
 *       --out tests/fixtures/weights-qwen2.5-0.5b-instruct.json
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { requestGpuContext, type GpuContext } from '../src/engine/device.js';
import { readBuffer } from '../src/engine/buffers.js';
import { ModelStore, type LoadProgress } from '../src/engine/store.js';
import { loadModel, type GpuTensor, type LoadedModel } from '../src/engine/model.js';

import fixture from './fixtures/weights-qwen2.5-0.5b-instruct.json';

const MODEL_ID = 'qwen2.5-0.5b-instruct';
const BASE = new URL(`/models/${MODEL_ID}/`, location.href).href;
const OPFS_ID = `${MODEL_ID}-gate`;

let ctx: GpuContext;
let available = false;

beforeAll(async () => {
  ctx = await requestGpuContext();
  available = (await fetch(new URL('model.json', BASE).href, { method: 'HEAD' })).ok;
  if (!available) {
    console.warn(`[skip] ${BASE}model.json not found; run tools/convert.py to enable the M1 gate`);
  }
  // Storage quota is shared per origin. Playwright contexts are ephemeral, and an
  // ephemeral context gets a far smaller quota than a normal profile -- worth printing,
  // because "QuotaExceededError" is otherwise a very opaque failure for a ~1 GB model.
  const estimate = await navigator.storage.estimate();
  console.log(
    `OPFS quota: ${((estimate.quota ?? 0) / 1048576).toFixed(0)} MiB, ` +
      `already used ${((estimate.usage ?? 0) / 1048576).toFixed(1)} MiB`,
  );
});

afterAll(() => {
  ctx?.device.destroy();
});

async function sha256Hex(data: BufferSource): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', data);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

async function readTensorBytes(device: GPUDevice, tensor: GpuTensor): Promise<Uint8Array<ArrayBuffer>> {
  const out = new Uint8Array(tensor.byteLength);
  let written = 0;
  for (const shard of tensor.shards) {
    const bytes = new Uint8Array(await readBuffer(device, shard.buffer, shard.byteLength));
    out.set(bytes, written);
    written += shard.byteLength;
  }
  return out;
}

const MB = 1024 * 1024;

describe('M1 gate: Qwen2.5-0.5B-Instruct', () => {
  let cold: LoadedModel;
  let warm: LoadedModel;

  it('loads from the network on a cold cache, then from OPFS', async () => {
    if (!available) return;

    // Start from a genuinely empty OPFS namespace so "cold" means cold.
    await (await ModelStore.open(OPFS_ID)).clear();

    const progress: LoadProgress[] = [];
    cold = await loadModel(ctx.device, {
      baseUrl: BASE,
      modelId: OPFS_ID,
      onProgress: (p) => progress.push(p),
    });

    expect(cold.stats.servedFromCache).toBe(false);
    expect(cold.stats.networkBytes).toBe(cold.header.totalBytes);
    expect(cold.stats.cacheBytes).toBe(0);

    // Progress must be real byte counts over the real total, not a synthetic ramp.
    const download = progress.filter((p) => p.phase === 'download');
    expect(download.at(-1)!.loadedBytes).toBe(cold.header.totalBytes);
    expect(download.at(-1)!.totalBytes).toBe(cold.header.totalBytes);
    expect(download.length).toBeGreaterThan(cold.header.chunks.length);
    for (let i = 1; i < download.length; i++) {
      expect(download[i].loadedBytes).toBeGreaterThanOrEqual(download[i - 1].loadedBytes);
    }

    cold.weights.destroy();

    warm = await loadModel(ctx.device, { baseUrl: BASE, modelId: OPFS_ID });
    expect(warm.stats.servedFromCache).toBe(true);
    expect(warm.stats.networkBytes).toBe(0);
    expect(warm.stats.cacheBytes).toBe(warm.header.totalBytes);

    const speedup = cold.stats.downloadMs / Math.max(warm.stats.downloadMs, 0.001);
    console.log(
      `\n=== M1 load timings (${(cold.header.totalBytes / MB).toFixed(0)} MiB, ` +
        `${cold.header.chunks.length} chunks) ===\n` +
        `cold (network): fetch ${cold.stats.downloadMs.toFixed(0)}ms, ` +
        `upload ${cold.stats.uploadMs.toFixed(0)}ms, total ${cold.stats.totalMs.toFixed(0)}ms\n` +
        `warm (OPFS)   : fetch ${warm.stats.downloadMs.toFixed(0)}ms, ` +
        `upload ${warm.stats.uploadMs.toFixed(0)}ms, total ${warm.stats.totalMs.toFixed(0)}ms\n` +
        `cache speedup on the fetch phase: ${speedup.toFixed(1)}x\n` +
        `buffers ${warm.stats.bufferCount}, VRAM ${(warm.stats.vramBytes / MB).toFixed(0)} MiB\n`,
    );
  });

  it('loads with the network unavailable once cached', async () => {
    if (!available) return;

    const realFetch = globalThis.fetch;
    globalThis.fetch = (() => {
      throw new Error('network is offline');
    }) as typeof fetch;
    try {
      const offline = await loadModel(ctx.device, { baseUrl: BASE, modelId: OPFS_ID });
      expect(offline.stats.servedFromCache).toBe(true);
      expect(offline.stats.networkBytes).toBe(0);
      expect(offline.weights.count).toBe(290);
      offline.weights.destroy();
    } finally {
      globalThis.fetch = realFetch;
    }
  });

  it('reads Qwen2.5-0.5B dimensions from the header, none of them hardcoded', () => {
    if (!available) return;
    const c = warm.config;
    expect(c.hiddenSize).toBe(896);
    expect(c.numHiddenLayers).toBe(24);
    expect(c.numAttentionHeads).toBe(14);
    expect(c.numKeyValueHeads).toBe(2);
    expect(c.headDim).toBe(64);
    expect(c.queryHeadsPerKvHead).toBe(7);
    expect(c.intermediateSize).toBe(4864);
    expect(c.vocabSize).toBe(151936);
    expect(c.ropeTheta).toBe(1000000);
    expect(c.rmsNormEps).toBeCloseTo(1e-6, 12);
    expect(c.tieWordEmbeddings).toBe(true);
  });

  it('byte-matches three GPU tensors against the safetensors source', async () => {
    if (!available) return;

    console.log('\n=== M1 gate: GPU readback vs safetensors ===');
    for (const expected of fixture.tensors) {
      const tensor = warm.weights.get(expected.name);
      expect(tensor.dtype).toBe(expected.dtype);
      expect(tensor.shape).toEqual(expected.shape);
      expect(tensor.byteLength).toBe(expected.byteLength);

      const bytes = await readTensorBytes(ctx.device, tensor);
      const digest = await sha256Hex(bytes);
      const ok = digest === expected.sha256;
      console.log(
        `  ${ok ? 'MATCH' : 'DIFFER'}  ${expected.name}  ` +
          `${expected.shape.join('x')} ${expected.dtype}  ` +
          `${(expected.byteLength / MB).toFixed(2)} MiB  sha256 ${digest.slice(0, 16)}…`,
      );
      expect(digest, `${expected.name} sha256`).toBe(expected.sha256);
    }
  });

  it('aliases the tied lm_head to the embedding buffers', () => {
    if (!available) return;
    const embed = warm.weights.get('model.embed_tokens.weight');
    const head = warm.weights.get('lm_head.weight');
    expect(head.shards.map((s) => s.buffer)).toEqual(embed.shards.map((s) => s.buffer));
    // Storing it twice would add 260 MiB to a 943 MiB download for nothing.
    expect(warm.weights.byteLength).toBeLessThan(warm.header.totalBytes + embed.byteLength);
  });

  it('shards the 260 MiB embedding matrix when the binding limit is small', async () => {
    if (!available) return;

    // This machine grants a 4 GiB binding limit, so nothing would shard naturally.
    // A 64 MiB threshold is what a default-limit device effectively imposes.
    const sharded = await loadModel(ctx.device, {
      baseUrl: BASE,
      modelId: OPFS_ID,
      shardThresholdBytes: 64 * MB,
    });

    const embed = sharded.weights.get('model.embed_tokens.weight');
    expect(embed.shards.length).toBeGreaterThan(1);

    let expectedRow = 0;
    for (const shard of embed.shards) {
      expect(shard.rowStart).toBe(expectedRow);
      expect(shard.byteLength).toBeLessThanOrEqual(64 * MB);
      expectedRow += shard.rowCount;
    }
    expect(expectedRow).toBe(embed.rows);

    const expected = fixture.tensors.find((t) => t.name === 'model.embed_tokens.weight')!;
    expect(await sha256Hex(await readTensorBytes(ctx.device, embed))).toBe(expected.sha256);
    console.log(
      `  embedding ${embed.shape.join('x')} -> ${embed.shards.length} shards ` +
        `at a 64 MiB threshold, bytes still identical`,
    );

    sharded.weights.destroy();
  });

  it('cleans up', async () => {
    if (!available) return;
    warm.weights.destroy();
    await warm.store.clear();
  });
});
