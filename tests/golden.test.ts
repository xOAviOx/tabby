/**
 * Gate M2(b), (c) and (d): the full forward pass against PyTorch.
 *
 * The comparison is deliberately layer by layer. A single end-to-end number tells you
 * nothing about *where* a forward pass went wrong, and with 24 layers the difference
 * between "RoPE convention is inverted" and "the GQA head mapping is off" is visible
 * only in which layer first drifts. Every layer's error is printed on every run.
 *
 * Requires the converted model and the goldens:
 *   python3 tools/convert.py models/Qwen2.5-0.5B-Instruct --out public/models/qwen2.5-0.5b-instruct
 *   .venv/bin/python tools/golden.py models/Qwen2.5-0.5B-Instruct --out public/golden/qwen2.5-0.5b-instruct
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { requestGpuContext, type GpuContext } from '../src/engine/device.js';
import { PipelineCache } from '../src/engine/pipelines.js';
import { loadModel, type LoadedModel } from '../src/engine/model.js';
import { ForwardPass, requireLogits } from '../src/engine/forward.js';
import { BpeTokenizer } from '../src/tokenizer/bpe.js';
import { generateGreedy, topK } from '../src/engine/sampler.js';
import { errorStats, fmt } from './support.js';

const MODEL_ID = 'qwen2.5-0.5b-instruct';
const MODEL_BASE = new URL(`/models/${MODEL_ID}/`, location.href).href;
const GOLDEN_BASE = new URL(`/golden/${MODEL_ID}/`, location.href).href;

/** Gate M2(b). Set by PROJECT.md; the f16 weights are the main source of drift. */
const LAYER_TOLERANCE = 2e-2;

interface TensorRecord {
  offset: number;
  byteLength: number;
  shape: number[];
}

interface GoldenPrompt {
  name: string;
  text: string;
  ids: number[];
  numLayers: number;
  tensors: Record<string, TensorRecord>;
  top5: { ids: number[]; values: number[]; tokens: string[] };
  greedy: { ids: number[]; text: string };
}

interface GoldenManifest {
  hiddenSize: number;
  numLayers: number;
  vocabSize: number;
  blobBytes: number;
  prompts: GoldenPrompt[];
}

let ctx: GpuContext;
let model: LoadedModel | null = null;
let forward: ForwardPass | null = null;
let tokenizer: BpeTokenizer | null = null;
let manifest: GoldenManifest | null = null;
let blob: ArrayBuffer | null = null;

function tensorOf(record: TensorRecord): Float32Array {
  return new Float32Array(blob!, record.offset, record.byteLength / 4);
}

beforeAll(async () => {
  ctx = await requestGpuContext();

  const [modelHead, goldenHead] = await Promise.all([
    fetch(new URL('model.json', MODEL_BASE).href, { method: 'HEAD' }),
    fetch(new URL('golden.json', GOLDEN_BASE).href, { method: 'HEAD' }),
  ]);
  if (!modelHead.ok || !goldenHead.ok) {
    console.warn('[skip] converted model or goldens missing; see this file’s header');
    return;
  }

  manifest = await (await fetch(new URL('golden.json', GOLDEN_BASE).href)).json();
  blob = await (await fetch(new URL('golden.bin', GOLDEN_BASE).href)).arrayBuffer();
  expect(blob.byteLength).toBe(manifest!.blobBytes);

  tokenizer = await BpeTokenizer.fromUrl(new URL('tokenizer.json', MODEL_BASE).href);
  model = await loadModel(ctx.device, { baseUrl: MODEL_BASE, modelId: `${MODEL_ID}-golden` });
  forward = await ForwardPass.create(ctx.device, model, new PipelineCache(ctx.device), {
    maxSeqLen: 128,
  });
}, 300_000);

afterAll(() => {
  forward?.destroy();
  model?.weights.destroy();
  ctx?.device.destroy();
});

describe('M2 gate: forward pass vs PyTorch goldens', () => {
  it('tokenizes each golden prompt to the reference ids', () => {
    if (!manifest || !tokenizer) return;
    for (const prompt of manifest.prompts) {
      expect(tokenizer.encode(prompt.text), prompt.name).toEqual(prompt.ids);
    }
  });

  it('matches PyTorch layer by layer, and reports every layer', async () => {
    if (!manifest || !forward) return;

    let worstOverall = 0;
    for (const prompt of manifest.prompts) {
      const result = await forward.runFull(prompt.ids, { captureActivations: true });
      expect(result.activations).not.toBeNull();
      const activations = result.activations!;

      // [embed, layer0..layerN-1, final_norm] -- the same points golden.py hooks.
      const names = [
        'embed',
        ...Array.from({ length: prompt.numLayers }, (_, i) => `layer${i}`),
        'final_norm',
      ];
      expect(activations.length).toBe(names.length);

      const lines: string[] = [];
      let firstDivergence: string | null = null;
      let worst = 0;

      for (const [index, name] of names.entries()) {
        const expected = tensorOf(prompt.tensors[name]);
        const actual = activations[index];
        expect(actual.length, `${prompt.name}/${name} length`).toBe(expected.length);

        const stats = errorStats(actual, expected);
        worst = Math.max(worst, stats.maxAbs);
        lines.push(
          `    ${name.padEnd(12)} maxAbs=${fmt(stats.maxAbs)} rms=${fmt(stats.rms)}` +
            (stats.maxAbs > LAYER_TOLERANCE ? '   <-- OVER TOLERANCE' : ''),
        );
        if (stats.maxAbs > LAYER_TOLERANCE && firstDivergence === null) {
          firstDivergence = name;
        }
      }

      console.log(
        `\n  ${prompt.name} (${prompt.ids.length} tokens), worst layer error ${fmt(worst)}:\n` +
          lines.join('\n'),
      );
      worstOverall = Math.max(worstOverall, worst);

      if (firstDivergence) {
        throw new Error(
          `${prompt.name}: activations first exceed ${fmt(LAYER_TOLERANCE)} at ` +
            `${firstDivergence}. Full per-layer errors above.`,
        );
      }
    }

    console.log(`\n  worst per-layer error across all prompts: ${fmt(worstOverall)}\n`);
    expect(worstOverall).toBeLessThanOrEqual(LAYER_TOLERANCE);
  }, 300_000);

  it('reproduces the top-5 next-token ids exactly', async () => {
    if (!manifest || !forward || !tokenizer) return;

    for (const prompt of manifest.prompts) {
      const logits = requireLogits(await forward.runFull(prompt.ids));
      const ours = topK(logits, 5);
      const ids = ours.map((entry) => entry.id);

      console.log(
        `  ${prompt.name}: top5 ${JSON.stringify(ids)} ` +
          `${JSON.stringify(ids.map((id) => tokenizer!.decode([id])))}`,
      );
      // Exact id match is the gate; the values themselves are allowed to drift.
      expect(ids, `${prompt.name} top-5 ids`).toEqual(prompt.top5.ids);
    }
  }, 300_000);

  it('greedily generates 20 tokens matching PyTorch', async () => {
    if (!manifest || !forward || !tokenizer) return;

    for (const prompt of manifest.prompts) {
      const result = await generateGreedy(
        prompt.ids,
        async (ids) => requireLogits(await forward!.runFull(ids)),
        { maxNewTokens: prompt.greedy.ids.length },
      );
      const text = tokenizer.decode(result.tokens);
      console.log(
        `\n  ${prompt.name}\n    prompt : ${JSON.stringify(prompt.text.slice(-60))}\n` +
          `    ours   : ${JSON.stringify(text)}\n` +
          `    torch  : ${JSON.stringify(prompt.greedy.text)}\n` +
          `    ${result.ms.toFixed(0)}ms for ${result.tokens.length} tokens ` +
          `(${((result.tokens.length / result.ms) * 1000).toFixed(2)} tok/s, no KV cache)`,
      );
      expect(result.tokens, `${prompt.name} greedy ids`).toEqual(prompt.greedy.ids);
    }
  }, 600_000);
});
