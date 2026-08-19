/**
 * Gate M4: GPU sampling, seeded determinism, and the readback budget.
 *
 * The readback rule is the one with teeth. Reading the full logit vector back per token
 * is 151,936 * 4 = 607,744 bytes at this vocabulary; the top-k path has to move a few
 * hundred. Both numbers are measured and printed rather than asserted in the abstract.
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { requestGpuContext, type GpuContext } from '../src/engine/device.js';
import { PipelineCache } from '../src/engine/pipelines.js';
import { loadModel, type LoadedModel } from '../src/engine/model.js';
import { ForwardPass, requireLogits } from '../src/engine/forward.js';
import { MAX_TOP_K } from '../src/engine/kernels.js';
import {
  SeededRandom,
  sampleFromTopK,
  topK as cpuTopK,
  type SamplingParams,
} from '../src/engine/sampler.js';
import { BpeTokenizer } from '../src/tokenizer/bpe.js';
import { InferenceClient } from '../src/worker/client.js';
import { fmt } from './support.js';

const MODEL_ID = 'qwen2.5-0.5b-instruct';
const MODEL_BASE = new URL(`/models/${MODEL_ID}/`, location.href).href;
const VOCAB_READBACK_BYTES = 151936 * 4;

let ctx: GpuContext;
let model: LoadedModel | null = null;
let forward: ForwardPass | null = null;
let tokenizer: BpeTokenizer | null = null;
let client: InferenceClient | null = null;

beforeAll(async () => {
  ctx = await requestGpuContext();
  if (!(await fetch(new URL('model.json', MODEL_BASE).href, { method: 'HEAD' })).ok) {
    console.warn('[skip] converted model missing; run tools/convert.py');
    return;
  }
  tokenizer = await BpeTokenizer.fromUrl(new URL('tokenizer.json', MODEL_BASE).href);
  model = await loadModel(ctx.device, { baseUrl: MODEL_BASE, modelId: `${MODEL_ID}-sampling` });
  forward = await ForwardPass.create(ctx.device, model, new PipelineCache(ctx.device), {
    maxSeqLen: 256,
  });
  client = new InferenceClient();
  await client.load({ baseUrl: MODEL_BASE, modelId: `${MODEL_ID}-sampling-worker`, maxSeqLen: 256 });
}, 900_000);

afterAll(() => {
  client?.terminate();
  forward?.destroy();
  model?.weights.destroy();
  ctx?.device.destroy();
});

describe('GPU top-k', () => {
  it('matches an exact CPU top-k over the full logit vector', async () => {
    if (!forward || !tokenizer) return;
    const ids = tokenizer.encode('The capital of France is');

    // Ground truth: the whole vector on the CPU.
    const full = requireLogits(await forward.runFull(ids));
    const expected = cpuTopK(full, MAX_TOP_K);

    const gpu = (await forward.runFull(ids, { topK: MAX_TOP_K })).topK!;
    expect(gpu.ids).toEqual(expected.map((e) => e.id));
    for (let i = 0; i < MAX_TOP_K; i++) {
      expect(gpu.logits[i], `logit ${i}`).toBe(expected[i].value);
    }

    // The softmax denominator the GPU returns must agree with a CPU pass.
    let sum = 0;
    for (const value of full) sum += Math.exp(value - gpu.maxLogit);
    expect(gpu.maxLogit).toBe(expected[0].value);
    expect(Math.abs(gpu.sumExp - sum) / sum).toBeLessThan(1e-3);

    console.log(
      `  GPU top-${MAX_TOP_K} matches CPU exactly; sumExp gpu=${fmt(gpu.sumExp)} ` +
        `cpu=${fmt(sum)}`,
    );
  }, 900_000);

  it('is exact for every k the sampler can ask for', async () => {
    if (!forward || !tokenizer) return;
    const ids = tokenizer.encode('In the beginning');
    const expected = cpuTopK(requireLogits(await forward.runFull(ids)), MAX_TOP_K);

    for (const k of [1, 2, 8, 40, MAX_TOP_K]) {
      const gpu = (await forward.runFull(ids, { topK: k })).topK!;
      expect(gpu.ids, `k=${k}`).toEqual(expected.slice(0, k).map((e) => e.id));
    }
  }, 900_000);

  it('rejects a k the output block is not sized for', async () => {
    if (!forward || !tokenizer) return;
    const ids = tokenizer.encode('Hello');
    await expect(forward.runFull(ids, { topK: MAX_TOP_K + 1 })).rejects.toThrow(/topK/);
  }, 900_000);
});

describe('M4 gate: readback budget', () => {
  it('never reads the full logit vector back per token', async () => {
    if (!forward || !tokenizer) return;
    forward.reset();
    const ids = tokenizer.encode('The capital of France is');

    const prefill = await forward.prefill(ids, { topK: 40 });
    const perToken: number[] = [prefill.readbackBytes];
    let next = prefill.topK!.ids[0];
    for (let step = 0; step < 8; step++) {
      const result = await forward.decode(next, { topK: 40 });
      perToken.push(result.readbackBytes);
      next = result.topK!.ids[0];
    }

    const worst = Math.max(...perToken);
    const total = perToken.reduce((a, b) => a + b, 0);
    console.log(
      `\n=== M4 readback budget ===\n` +
        `  per token      : ${perToken.join(', ')} bytes\n` +
        `  worst          : ${worst} B\n` +
        `  total (9 steps): ${total} B\n` +
        `  full-vector    : ${VOCAB_READBACK_BYTES} B/token would be ` +
        `${(VOCAB_READBACK_BYTES / worst).toFixed(0)}x more\n`,
    );

    // 2 header floats + 2 per candidate.
    expect(worst).toBe((2 + 40 * 2) * 4);
    expect(worst).toBeLessThan(VOCAB_READBACK_BYTES / 1000);
  }, 900_000);

  it('reports the same budget through the worker', async () => {
    if (!client) return;
    const stats = await client.generate({
      prompt: 'The capital of France is',
      maxNewTokens: 10,
      sampling: { temperature: 0.7, topK: 40, topP: 0.9, seed: 7 },
    }).done;

    console.log(
      `  worker: ${stats.readbackBytesPerToken} B/token, ` +
        `${stats.totalReadbackBytes} B for ${stats.generatedTokens} tokens`,
    );
    expect(stats.readbackBytesPerToken).toBeLessThan(VOCAB_READBACK_BYTES / 1000);
    expect(stats.totalReadbackBytes).toBeLessThan(VOCAB_READBACK_BYTES);
  }, 900_000);
});

describe('M4 gate: seeded determinism', () => {
  const params = (seed: number): SamplingParams => ({
    temperature: 0.9,
    topK: 40,
    topP: 0.95,
    seed,
  });

  it('replays identically for the same seed and differs for another', async () => {
    if (!client) return;

    const run = async (seed: number): Promise<string> => {
      const chunks: string[] = [];
      await client!.generate({
        prompt: 'Once upon a time',
        maxNewTokens: 24,
        sampling: params(seed),
        onToken: (text) => chunks.push(text),
      }).done;
      return chunks.join('');
    };

    const first = await run(42);
    const again = await run(42);
    const other = await run(43);

    console.log(
      `\n=== M4 seeded determinism ===\n` +
        `  seed 42 : ${JSON.stringify(first)}\n` +
        `  seed 42 : ${JSON.stringify(again)}\n` +
        `  seed 43 : ${JSON.stringify(other)}\n`,
    );

    expect(again).toBe(first);
    // Temperature 0.9 over 24 tokens diverging for a different seed is what shows the
    // sampler is actually sampling and not quietly running greedy.
    expect(other).not.toBe(first);
  }, 900_000);

  it('is deterministic at temperature 0 regardless of seed', async () => {
    if (!client) return;
    const run = async (seed: number): Promise<string> => {
      const chunks: string[] = [];
      await client!.generate({
        prompt: 'The capital of France is',
        maxNewTokens: 10,
        sampling: { temperature: 0, topK: 1, topP: 1, seed },
        onToken: (text) => chunks.push(text),
      }).done;
      return chunks.join('');
    };
    expect(await run(1)).toBe(await run(999));
  }, 900_000);
});

describe('sampling maths', () => {
  const top = {
    maxLogit: 10,
    // exp(0)+exp(-1)+exp(-2)+exp(-3) over a denominator that leaves mass outside the pool
    sumExp: 2.5,
    ids: [1, 2, 3, 4],
    logits: [10, 9, 8, 7],
  };

  it('returns the argmax at temperature 0', () => {
    const outcome = sampleFromTopK(top, { temperature: 0, topK: 4, topP: 1, seed: 0 }, new SeededRandom(0));
    expect(outcome.id).toBe(1);
    expect(outcome.consideredCount).toBe(1);
  });

  it('narrows the candidate set as top-p tightens', () => {
    const wide = sampleFromTopK(top, { temperature: 1, topK: 4, topP: 1, seed: 0 }, new SeededRandom(0));
    const narrow = sampleFromTopK(top, { temperature: 1, topK: 4, topP: 0.5, seed: 0 }, new SeededRandom(0));
    expect(narrow.consideredCount).toBeLessThan(wide.consideredCount);
    expect(narrow.consideredCount).toBeGreaterThanOrEqual(1);
  });

  it('flags a nucleus clipped by k rather than by p', () => {
    // The pool holds less mass than top-p asked for, so p could not be honoured. Saying
    // so beats silently renormalising a truncated tail and calling it top-p.
    const outcome = sampleFromTopK(
      { ...top, sumExp: 100 },
      { temperature: 1, topK: 4, topP: 0.9, seed: 0 },
      new SeededRandom(0),
    );
    expect(outcome.poolMass).toBeLessThan(0.9);
    expect(outcome.poolExhausted).toBe(true);
  });

  it('spreads across candidates as temperature rises', () => {
    const random = new SeededRandom(12345);
    const chosen = new Set<number>();
    for (let i = 0; i < 200; i++) {
      chosen.add(
        sampleFromTopK(top, { temperature: 2, topK: 4, topP: 1, seed: 0 }, random).id,
      );
    }
    expect(chosen.size).toBeGreaterThan(1);
  });
});

describe('M4 gate: multi-turn chat', () => {
  const GREEDY: SamplingParams = { temperature: 0, topK: 1, topP: 1, seed: 0 };

  it('formats a conversation through the model template and answers in role', async () => {
    if (!client) return;
    expect(client).toBeDefined();

    const messages = [
      { role: 'system', content: 'You are a terse assistant. Answer in one short sentence.' },
      { role: 'user', content: 'What is the capital of France?' },
    ];

    const first: string[] = [];
    const firstStats = await client.generate({
      messages,
      maxNewTokens: 48,
      sampling: GREEDY,
      onToken: (text) => first.push(text),
    }).done;
    const firstReply = first.join('');

    // Turn two carries the assistant's own reply back in, which is the thing multi-turn
    // formatting actually has to get right.
    const second: string[] = [];
    await client.generate({
      messages: [
        ...messages,
        { role: 'assistant', content: firstReply },
        { role: 'user', content: 'And of Japan?' },
      ],
      maxNewTokens: 48,
      sampling: GREEDY,
      onToken: (text) => second.push(text),
    }).done;
    const secondReply = second.join('');

    console.log(
      `\n=== M4 multi-turn ===\n` +
        `  user      : What is the capital of France?\n` +
        `  assistant : ${JSON.stringify(firstReply)}\n` +
        `  user      : And of Japan?\n` +
        `  assistant : ${JSON.stringify(secondReply)}\n` +
        `  turn 1 stopped at EOS: ${firstStats.stopped}\n`,
    );

    expect(firstReply.toLowerCase()).toContain('paris');
    // Answering "Tokyo" to a bare "And of Japan?" is only possible if the earlier turns
    // were formatted into the prompt correctly.
    expect(secondReply.toLowerCase()).toContain('tokyo');
    // Chat turns should end on the template's stop token, not run to the length cap.
    expect(firstStats.stopped).toBe(true);
    // The special tokens themselves must not leak into user-visible text.
    expect(firstReply).not.toContain('<|im_end|>');
    expect(firstReply).not.toContain('<|im_start|>');
  }, 900_000);

  it('routes the system message into the prompt so it changes the answer', async () => {
    if (!client) return;

    // Asserting that a 0.5B model *obeys* a given instruction tests the model, not this
    // code. What is ours to guarantee is that the system turn reaches the prompt at all --
    // so the check is that changing it changes the output. Exact template rendering is
    // gated separately, against HF's own output, in tokenizer.test.ts.
    const ask = async (system: string): Promise<string> => {
      const chunks: string[] = [];
      await client!.generate({
        messages: [
          { role: 'system', content: system },
          { role: 'user', content: 'Describe the sea.' },
        ],
        maxNewTokens: 24,
        sampling: GREEDY,
        onToken: (text) => chunks.push(text),
      }).done;
      return chunks.join('');
    };

    const plain = await ask('You are a helpful assistant.');
    const pirate = await ask('You are a pirate. Always speak in pirate slang.');

    console.log(
      `  system "helpful": ${JSON.stringify(plain.slice(0, 80))}\n` +
        `  system "pirate" : ${JSON.stringify(pirate.slice(0, 80))}`,
    );
    // Greedy decoding, identical user turn: the only difference is the system message.
    expect(pirate).not.toBe(plain);
  }, 900_000);
});
