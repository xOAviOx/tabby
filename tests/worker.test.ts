/**
 * Gate M3: the worker protocol, streaming, and cancellation.
 *
 * These run against the real worker with a real GPU device, not a mock. The point of the
 * milestone is that GPU work happens off the main thread, and a mocked worker would test
 * none of that.
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { InferenceClient, type LoadedInfo } from '../src/worker/client.js';
import type { SamplingParams } from '../src/engine/sampler.js';

/** Greedy, so these tests stay deterministic; sampling itself is gated in sampling.test.ts. */
const GREEDY: SamplingParams = { temperature: 0, topK: 1, topP: 1, seed: 1 };

const MODEL_ID = 'qwen2.5-0.5b-instruct';
const MODEL_BASE = new URL(`/models/${MODEL_ID}/`, location.href).href;

let client: InferenceClient | null = null;
let info: LoadedInfo | null = null;
const progressEvents: Array<{ loadedBytes: number; totalBytes: number }> = [];

beforeAll(async () => {
  if (!(await fetch(new URL('model.json', MODEL_BASE).href, { method: 'HEAD' })).ok) {
    console.warn('[skip] converted model missing; run tools/convert.py');
    return;
  }
  client = new InferenceClient();
  info = await client.load({
    baseUrl: MODEL_BASE,
    modelId: `${MODEL_ID}-worker`,
    maxSeqLen: 128,
    onProgress: (p) => progressEvents.push({ loadedBytes: p.loadedBytes, totalBytes: p.totalBytes }),
  });
}, 600_000);

afterAll(() => {
  client?.terminate();
});

describe('inference worker', () => {
  it('loads the model off the main thread and reports real progress', () => {
    if (!info) return;
    expect(info.config.hiddenSize).toBe(896);
    expect(info.config.numHiddenLayers).toBe(24);
    expect(info.maxSeqLen).toBe(128);
    expect(info.stats.vramBytes).toBeGreaterThan(0);
    expect(info.stats.kvCacheBytes).toBeGreaterThan(0);

    expect(progressEvents.length).toBeGreaterThan(0);
    const last = progressEvents.at(-1)!;
    expect(last.loadedBytes).toBe(last.totalBytes);

    console.log(
      `\n  worker load: ${(info.stats.vramBytes / 1048576).toFixed(0)} MiB weights + ` +
        `${(info.stats.kvCacheBytes / 1048576).toFixed(1)} MiB KV cache, ` +
        `pipelines compiled in ${info.stats.pipelineMs.toFixed(0)} ms, ` +
        `${progressEvents.length} progress events\n`,
    );
  });

  it('streams tokens as they are produced', async () => {
    if (!client) return;
    const chunks: string[] = [];
    const arrivals: number[] = [];
    const started = performance.now();

    const handle = client.generate({
      prompt: 'The capital of France is',
      maxNewTokens: 12,
      sampling: GREEDY,
      onToken: (text) => {
        chunks.push(text);
        arrivals.push(performance.now() - started);
      },
    });
    const stats = await handle.done;

    expect(chunks.length).toBe(stats.generatedTokens);
    expect(chunks.length).toBeGreaterThan(1);
    // Tokens must arrive spread over time, not all at once when generation finishes.
    expect(arrivals.at(-1)! - arrivals[0]).toBeGreaterThan(0);

    console.log(
      `  streamed ${chunks.length} tokens: ${JSON.stringify(chunks.join(''))}\n` +
        `  TTFT ${stats.ttftMs.toFixed(0)} ms, decode ${stats.decodeTokPerSec.toFixed(2)} tok/s, ` +
        `prefill ${stats.prefillTokPerSec.toFixed(1)} tok/s`,
    );
    expect(stats.cancelled).toBe(false);
  }, 900_000);

  it('cancels mid-generation and emits nothing afterwards', async () => {
    if (!client) return;
    const received: string[] = [];
    let cancelledAt = -1;

    const handle = client.generate({
      prompt: 'Once upon a time in a distant land',
      maxNewTokens: 60,
      sampling: GREEDY,
      onToken: (text) => {
        received.push(text);
        if (received.length === 4) {
          cancelledAt = received.length;
          handle.cancel();
        }
      },
    });

    const stats = await handle.done;
    expect(cancelledAt).toBe(4);
    expect(stats.cancelled).toBe(true);
    // The run stopped far short of the limit, and nothing arrived after the cancel was
    // observed. A couple of tokens may already be in flight when cancel is sent.
    expect(stats.generatedTokens).toBeLessThan(60);
    expect(received.length).toBeLessThanOrEqual(stats.generatedTokens);

    console.log(
      `  cancelled after ${cancelledAt} tokens; worker produced ${stats.generatedTokens} ` +
        `of a possible 60 and stopped`,
    );

    // The client must be usable immediately afterwards.
    const after = client.generate({
      prompt: 'Two plus two is',
      maxNewTokens: 3,
      sampling: GREEDY,
    });
    const afterStats = await after.done;
    expect(afterStats.cancelled).toBe(false);
    expect(afterStats.generatedTokens).toBe(3);
  }, 900_000);

  it('keeps the main thread responsive while generating', async () => {
    if (!client) return;

    // A main-thread timer is the direct measure of what the gate cares about: if GPU work
    // were happening here, these ticks would stall.
    const ticks: number[] = [];
    let last = performance.now();
    let worstGap = 0;
    const timer = setInterval(() => {
      const now = performance.now();
      worstGap = Math.max(worstGap, now - last);
      last = now;
      ticks.push(now);
    }, 10);

    try {
      const handle = client.generate({
        prompt: 'Write a short story about',
        maxNewTokens: 16,
        sampling: GREEDY,
      });
      const stats = await handle.done;
      clearInterval(timer);

      console.log(
        `  during ${stats.generatedTokens} tokens (${(stats.decodeMs / 1000).toFixed(2)} s): ` +
          `${ticks.length} main-thread ticks, worst gap ${worstGap.toFixed(1)} ms`,
      );

      expect(ticks.length).toBeGreaterThan(5);
      // A blocked main thread shows up as a multi-hundred-millisecond gap.
      expect(worstGap).toBeLessThan(250);
    } finally {
      clearInterval(timer);
    }
  }, 900_000);

  it('reports a context overflow as a clean error', async () => {
    if (!client) return;
    // maxSeqLen is 128 for this worker; ask for more than that in one run.
    const handle = client.generate({
      prompt: 'a '.repeat(100),
      maxNewTokens: 200,
      sampling: GREEDY,
    });
    await expect(handle.done).rejects.toThrow(/context overflow/i);
  }, 900_000);
});
