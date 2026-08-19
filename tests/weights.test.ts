import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { requestGpuContext, type GpuContext } from '../src/engine/device.js';
import { readBuffer } from '../src/engine/buffers.js';
import {
  ModelStore,
  ChunkedByteReader,
  ensureChunk,
  type ChunkMeta,
  type LoadProgress,
} from '../src/engine/store.js';
import {
  loadModel,
  planShards,
  parseConfig,
  ModelLoadError,
  type GpuTensor,
  type LoadedModel,
} from '../src/engine/model.js';

import tinyFixture from './fixtures/weights-tiny-test.json';

const TINY_BASE = new URL('/models/tiny-test/', location.href).href;

let ctx: GpuContext;

beforeAll(async () => {
  ctx = await requestGpuContext();
});

afterAll(() => {
  ctx?.device.destroy();
});

// -------------------------------------------------------------------------------------
// helpers
// -------------------------------------------------------------------------------------

async function sha256Hex(data: BufferSource): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', data);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

function b64(bytes: Uint8Array): string {
  let s = '';
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s);
}

/** Pull a tensor's bytes back off the GPU, concatenating across shards. */
async function readTensorBytes(device: GPUDevice, tensor: GpuTensor): Promise<Uint8Array<ArrayBuffer>> {
  const out = new Uint8Array(tensor.byteLength);
  let written = 0;
  for (const shard of tensor.shards) {
    const bytes = new Uint8Array(await readBuffer(device, shard.buffer, shard.byteLength));
    out.set(bytes.subarray(0, Math.min(bytes.byteLength, out.byteLength - written)), written);
    written += shard.byteLength;
  }
  expect(written).toBe(tensor.byteLength);
  return out;
}

async function freshStore(modelId: string): Promise<ModelStore> {
  const store = await ModelStore.open(modelId);
  await store.clear();
  return store;
}

let uniqueCounter = 0;
function uniqueId(prefix: string): string {
  return `${prefix}-${Date.now().toString(36)}-${uniqueCounter++}`;
}

/**
 * The loader keys OPFS by modelId but always fetches from baseUrl, so a test can use a
 * throwaway modelId to guarantee a cold cache while still serving the same files.
 */
async function loadTiny(
  modelId: string,
  overrides: Partial<Parameters<typeof loadModel>[1]> = {},
): Promise<LoadedModel> {
  return loadModel(ctx.device, {
    baseUrl: TINY_BASE,
    modelId,
    ...overrides,
  });
}

// -------------------------------------------------------------------------------------

describe('config parsing', () => {
  it('derives head_dim and GQA grouping from the source config', () => {
    const config = parseConfig({
      hidden_size: 896,
      num_hidden_layers: 24,
      num_attention_heads: 14,
      num_key_value_heads: 2,
      intermediate_size: 4864,
      vocab_size: 151936,
      rope_theta: 1000000.0,
      rms_norm_eps: 1e-6,
      tie_word_embeddings: true,
      eos_token_id: [151645, 151643],
    });
    expect(config.headDim).toBe(64);
    expect(config.queryHeadsPerKvHead).toBe(7);
    expect(config.eosTokenIds).toEqual([151645, 151643]);
  });

  it('rejects a head configuration that GQA cannot express', () => {
    expect(() =>
      parseConfig({
        hidden_size: 96,
        num_hidden_layers: 2,
        num_attention_heads: 5,
        num_key_value_heads: 2,
        intermediate_size: 64,
        vocab_size: 100,
        rope_theta: 10000,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: false,
      }),
    ).toThrow(ModelLoadError);
  });

  it('rejects a missing required dimension rather than defaulting it', () => {
    expect(() => parseConfig({ hidden_size: 32 })).toThrow(/num_attention_heads/);
  });
});

describe('shard planning', () => {
  it('splits along whole rows and covers every row exactly once', () => {
    const plan = planShards(1000, 128, 10 * 128);
    expect(plan).toHaveLength(100);
    expect(plan[0]).toEqual({ rowStart: 0, rowCount: 10, byteLength: 1280 });
    const covered = plan.reduce((sum, p) => sum + p.rowCount, 0);
    expect(covered).toBe(1000);
    for (let i = 1; i < plan.length; i++) {
      expect(plan[i].rowStart).toBe(plan[i - 1].rowStart + plan[i - 1].rowCount);
    }
  });

  it('leaves a short final shard rather than over-allocating', () => {
    const plan = planShards(25, 100, 1000);
    expect(plan.map((p) => p.rowCount)).toEqual([10, 10, 5]);
  });

  it('produces one shard when the tensor already fits', () => {
    expect(planShards(64, 16, 1 << 20)).toHaveLength(1);
  });

  it('refuses a tensor whose single row exceeds the binding limit', () => {
    // Column sharding would be required; no model in scope needs it, so this must be a
    // loud failure rather than a silent truncation.
    expect(() => planShards(4, 5000, 4096)).toThrow(/column sharding/);
  });
});

describe('chunk store', () => {
  it('downloads, verifies and caches chunks, then serves them from OPFS', async () => {
    const modelId = uniqueId('store');
    const store = await freshStore(modelId);
    const header = await (await fetch(new URL('model.json', TINY_BASE).href)).json();
    const chunk: ChunkMeta = header.chunks[0];

    const first = await ensureChunk(store, chunk, { baseUrl: TINY_BASE });
    expect(first.fromCache).toBe(false);
    expect(first.networkBytes).toBe(chunk.bytes);

    const second = await ensureChunk(store, chunk, { baseUrl: TINY_BASE });
    expect(second.fromCache).toBe(true);
    expect(second.networkBytes).toBe(0);

    expect(await sha256Hex(await store.readAll(chunk.name))).toBe(chunk.sha256);
    await store.clear();
  });

  it('resumes an interrupted download from the byte it reached', async () => {
    const modelId = uniqueId('resume');
    const store = await freshStore(modelId);
    const header = await (await fetch(new URL('model.json', TINY_BASE).href)).json();
    const chunk: ChunkMeta = header.chunks[0];

    // Simulate an interruption by planting a truncated file, which is exactly the state
    // a killed download leaves behind now that chunks stream to their final name.
    const full = new Uint8Array(
      await (await fetch(new URL(chunk.name, TINY_BASE).href)).arrayBuffer(),
    );
    const cut = Math.floor(chunk.bytes / 3);
    await store.write(chunk.name, full.subarray(0, cut));

    const result = await ensureChunk(store, chunk, { baseUrl: TINY_BASE });
    expect(result.resumedFrom).toBe(cut);
    // Only the remainder crossed the network.
    expect(result.networkBytes).toBe(chunk.bytes - cut);
    expect(await sha256Hex(await store.readAll(chunk.name))).toBe(chunk.sha256);
    await store.clear();
  });

  it('rejects a chunk whose bytes do not match its recorded hash', async () => {
    const modelId = uniqueId('corrupt');
    const store = await freshStore(modelId);
    const header = await (await fetch(new URL('model.json', TINY_BASE).href)).json();
    const chunk: ChunkMeta = { ...header.chunks[0], sha256: 'de'.repeat(32) };

    await expect(ensureChunk(store, chunk, { baseUrl: TINY_BASE })).rejects.toThrow(/sha256/);
    // The bad data must not be left behind as a valid-looking cache entry.
    expect(await store.sizeOf(chunk.name)).toBe(-1);
    await store.clear();
  });

  it('detects a resumed download built on a corrupt prefix and drops it', async () => {
    const modelId = uniqueId('badprefix');
    const store = await freshStore(modelId);
    const header = await (await fetch(new URL('model.json', TINY_BASE).href)).json();
    const chunk: ChunkMeta = header.chunks[0];

    // A short file is indistinguishable from an interrupted download, so it is resumed.
    // If the existing prefix is garbage the result is full-size but wrong, and only the
    // hash catches it -- otherwise a bad prefix would be resumed onto forever.
    await store.write(chunk.name, new Uint8Array(chunk.bytes - 17).fill(0xab));
    await expect(ensureChunk(store, chunk, { baseUrl: TINY_BASE })).rejects.toThrow(/sha256/);
    expect(await store.sizeOf(chunk.name)).toBe(-1);

    // With the bad prefix gone, a plain retry succeeds.
    const retry = await ensureChunk(store, chunk, { baseUrl: TINY_BASE });
    expect(retry.resumedFrom).toBe(0);
    expect(await sha256Hex(await store.readAll(chunk.name))).toBe(chunk.sha256);
    await store.clear();
  });

  it('verifies an unrecorded but complete chunk instead of re-downloading it', async () => {
    const modelId = uniqueId('unrecorded');
    const store = await freshStore(modelId);
    const header = await (await fetch(new URL('model.json', TINY_BASE).href)).json();
    const chunk: ChunkMeta = header.chunks[0];

    // Correct bytes on disk but no manifest entry: the store must hash it once and
    // accept it, not pull it down again.
    const full = new Uint8Array(
      await (await fetch(new URL(chunk.name, TINY_BASE).href)).arrayBuffer(),
    );
    await store.write(chunk.name, full);

    const result = await ensureChunk(store, chunk, { baseUrl: TINY_BASE });
    expect(result.fromCache).toBe(true);
    expect(result.networkBytes).toBe(0);
    await store.clear();
  });

  it('reads ranges that span chunk boundaries', async () => {
    const modelId = uniqueId('reader');
    const store = await freshStore(modelId);
    const header = await (await fetch(new URL('model.json', TINY_BASE).href)).json();
    for (const chunk of header.chunks as ChunkMeta[]) {
      await ensureChunk(store, chunk, { baseUrl: TINY_BASE });
    }

    const reader = new ChunkedByteReader(store, header.chunks, header.chunkBytes);
    const whole = new Uint8Array(reader.totalBytes);
    let at = 0;
    for (const chunk of header.chunks as ChunkMeta[]) {
      whole.set(await store.readAll(chunk.name), at);
      at += chunk.bytes;
    }

    // A range deliberately straddling three chunk files.
    const start = header.chunkBytes - 7;
    const length = header.chunkBytes * 2 + 21;
    const got = await reader.read(start, length);
    expect(Array.from(got)).toEqual(Array.from(whole.subarray(start, start + length)));

    await expect(reader.read(reader.totalBytes - 4, 64)).rejects.toThrow(/outside/);
    await store.clear();
  });
});

describe('model loading (tiny synthetic model)', () => {
  it('loads from the network first, then entirely from OPFS', async () => {
    const modelId = uniqueId('cache');
    await freshStore(modelId);

    const firstProgress: LoadProgress[] = [];
    const first = await loadTiny(modelId, { onProgress: (p) => firstProgress.push(p) });
    expect(first.stats.servedFromCache).toBe(false);
    expect(first.stats.networkBytes).toBe(first.header.totalBytes);
    expect(first.stats.cacheBytes).toBe(0);
    first.weights.destroy();

    const second = await loadTiny(modelId);
    expect(second.stats.servedFromCache).toBe(true);
    expect(second.stats.networkBytes).toBe(0);
    expect(second.stats.cacheBytes).toBe(second.header.totalBytes);

    console.log(
      `tiny model: network load ${first.stats.downloadMs.toFixed(1)}ms ` +
        `(${first.stats.networkBytes} B) -> OPFS load ${second.stats.downloadMs.toFixed(1)}ms ` +
        `(${second.stats.cacheBytes} B cached)`,
    );

    // Progress must reflect real byte counts, not a synthetic ramp.
    const download = firstProgress.filter((p) => p.phase === 'download');
    expect(download.at(-1)!.loadedBytes).toBe(first.header.totalBytes);
    expect(download.at(-1)!.totalBytes).toBe(first.header.totalBytes);
    for (let i = 1; i < download.length; i++) {
      expect(download[i].loadedBytes).toBeGreaterThanOrEqual(download[i - 1].loadedBytes);
    }
    const upload = firstProgress.filter((p) => p.phase === 'upload');
    expect(upload.at(-1)!.loadedBytes).toBe(first.header.totalBytes);

    second.weights.destroy();
    await second.store.clear();
  });

  it('loads with the network unavailable once cached', async () => {
    const modelId = uniqueId('offline');
    await freshStore(modelId);

    const warm = await loadTiny(modelId);
    const warmTensorCount = warm.weights.count;
    warm.weights.destroy();

    // Hard proof of offline capability: any fetch at all now fails.
    const realFetch = globalThis.fetch;
    globalThis.fetch = (() => {
      throw new Error('network is offline');
    }) as typeof fetch;
    try {
      const offline = await loadTiny(modelId);
      expect(offline.stats.servedFromCache).toBe(true);
      expect(offline.stats.networkBytes).toBe(0);
      expect(offline.weights.count).toBe(warmTensorCount);
      offline.weights.destroy();
      await offline.store.clear();
    } finally {
      globalThis.fetch = realFetch;
    }
  });

  it('reads config values from model.json rather than any built-in default', async () => {
    const modelId = uniqueId('config');
    await freshStore(modelId);
    const model = await loadTiny(modelId);
    // These match tools/make_test_model.py, which is deliberately unlike Qwen.
    expect(model.config.hiddenSize).toBe(32);
    expect(model.config.numHiddenLayers).toBe(2);
    expect(model.config.numAttentionHeads).toBe(4);
    expect(model.config.numKeyValueHeads).toBe(2);
    expect(model.config.headDim).toBe(8);
    expect(model.config.intermediateSize).toBe(61);
    expect(model.config.vocabSize).toBe(203);
    expect(model.config.queryHeadsPerKvHead).toBe(2);
    model.weights.destroy();
    await model.store.clear();
  });

  it('resolves tied embeddings to the same buffers without a second copy', async () => {
    const modelId = uniqueId('tied');
    await freshStore(modelId);
    const model = await loadTiny(modelId);

    expect(model.header.aliases['lm_head.weight']).toBe('model.embed_tokens.weight');
    const embed = model.weights.get('model.embed_tokens.weight');
    const head = model.weights.get('lm_head.weight');
    expect(head.shards.map((s) => s.buffer)).toEqual(embed.shards.map((s) => s.buffer));
    // The registry holds one entry, not two.
    expect(model.weights.names()).not.toContain('lm_head.weight');

    model.weights.destroy();
    await model.store.clear();
  });

  it('byte-matches weights read back from the GPU against the safetensors source', async () => {
    const modelId = uniqueId('bytes');
    await freshStore(modelId);
    const model = await loadTiny(modelId);

    for (const expected of tinyFixture.tensors) {
      const tensor = model.weights.get(expected.name);
      expect(tensor.dtype).toBe(expected.dtype);
      expect(tensor.shape).toEqual(expected.shape);
      expect(tensor.byteLength).toBe(expected.byteLength);

      const bytes = await readTensorBytes(ctx.device, tensor);
      const digest = await sha256Hex(bytes);
      console.log(`tiny ${expected.name}: ${bytes.byteLength} B sha256 ${digest.slice(0, 16)}...`);
      expect(digest, `${expected.name} sha256`).toBe(expected.sha256);
      expect(b64(bytes.subarray(0, tinyFixture.sampleBytes))).toBe(expected.head);
      expect(b64(bytes.subarray(-tinyFixture.sampleBytes))).toBe(expected.tail);
    }

    model.weights.destroy();
    await model.store.clear();
  });

  it('shards oversized tensors across buffers and preserves the bytes exactly', async () => {
    const modelId = uniqueId('shard');
    await freshStore(modelId);

    // Forced small threshold. On this machine the real binding limit is 4 GiB, so
    // without injection no tensor would ever shard and this path would never run.
    const model = await loadTiny(modelId, { shardThresholdBytes: 2048 });

    const embed = model.weights.get('model.embed_tokens.weight');
    expect(embed.shards.length).toBeGreaterThan(1);
    console.log(
      `embed ${embed.shape.join('x')} split into ${embed.shards.length} shards ` +
        `at a 2048-byte threshold (${embed.bytesPerRow} B/row)`,
    );

    // Rows are partitioned, contiguous, and complete.
    let expectedRow = 0;
    for (const shard of embed.shards) {
      expect(shard.rowStart).toBe(expectedRow);
      expect(shard.byteLength).toBe(shard.rowCount * embed.bytesPerRow);
      expect(shard.byteLength).toBeLessThanOrEqual(2048);
      expectedRow += shard.rowCount;
    }
    expect(expectedRow).toBe(embed.rows);

    // And the concatenation is still byte-identical to the source.
    const expected = tinyFixture.tensors.find((t) => t.name === 'model.embed_tokens.weight')!;
    expect(await sha256Hex(await readTensorBytes(ctx.device, embed))).toBe(expected.sha256);

    model.weights.destroy();
    await model.store.clear();
  });

  it('reports VRAM and buffer counts that grow with sharding', async () => {
    const idA = uniqueId('vram-a');
    const idB = uniqueId('vram-b');
    await freshStore(idA);
    await freshStore(idB);

    const unsharded = await loadTiny(idA);
    const sharded = await loadTiny(idB, { shardThresholdBytes: 2048 });

    expect(sharded.stats.bufferCount).toBeGreaterThan(unsharded.stats.bufferCount);
    expect(unsharded.stats.tensorCount).toBe(sharded.stats.tensorCount);
    console.log(
      `buffers: ${unsharded.stats.bufferCount} unsharded vs ${sharded.stats.bufferCount} ` +
        `sharded; VRAM ${unsharded.stats.vramBytes} vs ${sharded.stats.vramBytes} B`,
    );

    unsharded.weights.destroy();
    sharded.weights.destroy();
    await unsharded.store.clear();
    await sharded.store.clear();
  });
});
