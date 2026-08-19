/**
 * Dev surface: negotiate a device, show what we were granted, load the converted model
 * with a real progress bar, and run the kernels that exist against their CPU references
 * so a machine can be characterised without running the test suite.
 *
 * This page is replaced by the chat UI at M4; the limits panel folds into the perf
 * panel at M5.
 */

import {
  describeContext,
  requestGpuContext,
  WebGpuUnavailableError,
  type GpuContext,
} from '../engine/device.js';
import { PipelineCache } from '../engine/pipelines.js';
import { runMatvecF32 } from '../engine/kernels.js';
import { matvecF32, matvecF64 } from '../reference/cpu.js';
import { ModelStore, type LoadProgress } from '../engine/store.js';
import { loadModel, type LoadedModel } from '../engine/model.js';
import { ForwardPass } from '../engine/forward.js';
import { generateGreedy } from '../engine/sampler.js';
import { BpeTokenizer } from '../tokenizer/bpe.js';

/** Which converted model the dev page loads. Produced by tools/convert.py. */
const MODEL_ID = 'qwen2.5-0.5b-instruct';
const MODEL_BASE = new URL(`/models/${MODEL_ID}/`, location.href).href;

const SELF_CHECK_SHAPES: Array<{ m: number; n: number }> = [
  { m: 3, n: 7 },
  { m: 65, n: 129 },
  { m: 1000, n: 999 },
  { m: 896, n: 4864 },
];

/** Same tolerance the test suite gates on. */
const TOLERANCE = 1e-4;

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing #${id}`);
  return node as T;
}

function show(id: string): HTMLElement {
  const node = el(id);
  node.hidden = false;
  return node;
}

function defineList(target: HTMLElement, rows: Array<[string, string]>): void {
  target.replaceChildren(
    ...rows.flatMap(([key, value]) => {
      const dt = document.createElement('dt');
      dt.textContent = key;
      const dd = document.createElement('dd');
      dd.textContent = value;
      return [dt, dd];
    }),
  );
}

function seededRandom(count: number, seed: number): Float32Array {
  let a = seed >>> 0;
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    out[i] = (((t ^ (t >>> 14)) >>> 0) / 4294967296) * 2 - 1;
  }
  return out;
}

function maxAbsError(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst;
}

const MIB = 1024 * 1024;

function formatLimit(key: string, value: number): string {
  return key.endsWith('Size') && value >= MIB
    ? `${value.toLocaleString()}  (${(value / MIB).toFixed(0)} MiB)`
    : value.toLocaleString();
}

function renderUnavailable(message: string, detail: string): void {
  const status = el('status');
  status.classList.add('fail');
  status.replaceChildren();
  const h = document.createElement('h2');
  h.textContent = message;
  const p = document.createElement('p');
  p.textContent = detail;
  const help = document.createElement('p');
  help.className = 'muted';
  help.textContent =
    'WebGPU needs Chrome or Edge 113+, or Safari 18+. On Linux it may require enabling ' +
    'the Vulkan backend. Firefox ships it in release from 141 on Windows only.';
  status.append(h, p, help);
}

async function main(): Promise<void> {
  let ctx;
  try {
    ctx = await requestGpuContext({
      onDeviceLost: (info) => renderUnavailable('GPU device lost', `${info.reason}: ${info.message}`),
    });
  } catch (err) {
    if (err instanceof WebGpuUnavailableError) {
      renderUnavailable('WebGPU is not available in this browser', err.message);
    } else {
      renderUnavailable('Could not initialise WebGPU', String(err));
    }
    return;
  }

  console.log(`=== negotiated WebGPU context ===\n${describeContext(ctx)}`);

  el('status').replaceChildren(
    Object.assign(document.createElement('p'), {
      className: 'ok',
      textContent: 'WebGPU device acquired.',
    }),
  );

  const { info } = ctx;
  defineList(show('adapter').querySelector('dl')!, [
    ['vendor', info.vendor || '(not reported)'],
    ['architecture', info.architecture || '(not reported)'],
    ['device', info.device || '(not reported)'],
    ['description', info.description || '(not reported)'],
    ['features', ctx.features.join(', ') || '(none)'],
    ['shader-f16', ctx.hasF16 ? 'yes' : 'no — f32 fallback path required'],
    ['timestamp-query', ctx.hasTimestampQuery ? 'yes' : 'no — wall-clock timing only'],
    ['raised limits', ctx.limitsWereRefused ? 'REFUSED — running on defaults' : 'granted'],
  ]);

  defineList(
    show('limits').querySelector('dl')!,
    Object.entries(ctx.limits)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, value]) => [key, formatLimit(key, value)] as [string, string]),
  );

  const table = show('selfcheck').querySelector('table')!;
  table.innerHTML =
    '<thead><tr><th>shape</th><th>max abs err (f32 ref)</th>' +
    '<th>max abs err (f64 ref)</th><th>ms</th><th></th></tr></thead><tbody></tbody>';
  const tbody = table.querySelector('tbody')!;

  const cache = new PipelineCache(ctx.device);
  for (const { m, n } of SELF_CHECK_SHAPES) {
    const w = seededRandom(m * n, m * 7919 + n);
    const x = seededRandom(n, (m * 7919 + n) ^ 0x5bf03635);

    const started = performance.now();
    const gpu = await runMatvecF32(ctx.device, cache, w, x, m, n);
    const elapsed = performance.now() - started;

    const errF32 = maxAbsError(gpu, matvecF32(w, x, m, n));
    const errF64 = maxAbsError(gpu, matvecF64(w, x, m, n));
    const pass = errF32 < TOLERANCE;

    const row = document.createElement('tr');
    for (const [text, cls] of [
      [`${m} x ${n}`, ''],
      [errF32.toExponential(2), ''],
      [errF64.toExponential(2), ''],
      [elapsed.toFixed(1), ''],
      [pass ? 'pass' : 'FAIL', pass ? 'ok' : 'bad'],
    ] as Array<[string, string]>) {
      const td = document.createElement('td');
      td.textContent = text;
      if (cls) td.className = cls;
      row.append(td);
    }
    tbody.append(row);
  }

  await setUpModelPanel(ctx);
}

// ---------------------------------------------------------------------------------------
// model loading panel
// ---------------------------------------------------------------------------------------

const MIB_BYTES = 1024 * 1024;

function formatBytes(bytes: number): string {
  if (bytes >= MIB_BYTES) return `${(bytes / MIB_BYTES).toFixed(1)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${bytes} B`;
}

function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(0)} ms`;
}

async function setUpModelPanel(ctx: GpuContext): Promise<void> {
  const section = show('model');
  const loadButton = el<HTMLButtonElement>('load-model');
  const clearButton = el<HTMLButtonElement>('clear-cache');
  const status = el('model-status');
  const progressWrap = el('progress-wrap');
  const fill = el('progress-fill');
  const label = el('progress-label');
  const body = el('model-body');

  let loaded: LoadedModel | null = null;

  const describeCache = async (): Promise<void> => {
    const store = await ModelStore.open(MODEL_ID);
    const bytes = await store.usageBytes();
    status.textContent = bytes > 0 ? `${formatBytes(bytes)} cached in OPFS` : 'not cached';
  };

  const onProgress = (p: LoadProgress): void => {
    progressWrap.hidden = false;
    const fraction = p.totalBytes > 0 ? p.loadedBytes / p.totalBytes : 0;
    fill.style.width = `${(fraction * 100).toFixed(2)}%`;
    // Real byte counts, not a synthetic ramp -- these are the numbers the loader moved.
    const source = p.phase === 'download' ? (p.fromCache ? 'cache' : 'network') : 'GPU';
    label.textContent =
      `${p.phase} (${source})  ${formatBytes(p.loadedBytes)} / ${formatBytes(p.totalBytes)}  ` +
      `${(fraction * 100).toFixed(1)}%  ${p.detail}`;
  };

  loadButton.addEventListener('click', async () => {
    loadButton.disabled = true;
    clearButton.disabled = true;
    body.hidden = true;
    status.textContent = 'loading…';
    try {
      loaded?.weights.destroy();
      loaded = await loadModel(ctx.device, {
        baseUrl: MODEL_BASE,
        modelId: MODEL_ID,
        onProgress,
      });
      const { stats, config } = loaded;
      fill.style.width = '100%';
      status.textContent = stats.servedFromCache ? 'loaded from OPFS' : 'downloaded and cached';

      defineList(body, [
        ['served from', stats.servedFromCache ? 'OPFS cache' : 'network'],
        ['fetch', `${formatMs(stats.downloadMs)}  (${formatBytes(stats.networkBytes)} network, ${formatBytes(stats.cacheBytes)} cache)`],
        ['GPU upload', formatMs(stats.uploadMs)],
        ['total', formatMs(stats.totalMs)],
        ['tensors / buffers', `${stats.tensorCount} / ${stats.bufferCount}`],
        ['VRAM', formatBytes(stats.vramBytes)],
        ['chunks', String(stats.chunkCount)],
        ['architecture', `${config.modelType}, ${config.numHiddenLayers} layers`],
        ['hidden / intermediate', `${config.hiddenSize} / ${config.intermediateSize}`],
        ['heads (Q / KV)', `${config.numAttentionHeads} / ${config.numKeyValueHeads} (${config.queryHeadsPerKvHead} per KV)`],
        ['head_dim', String(config.headDim)],
        ['vocab', config.vocabSize.toLocaleString()],
        ['rope_theta', String(config.ropeTheta)],
        ['rms_norm_eps', String(config.rmsNormEps)],
        ['tied embeddings', config.tieWordEmbeddings ? 'yes' : 'no'],
      ]);
      body.hidden = false;
      await setUpGeneration(ctx, loaded);
    } catch (err) {
      status.textContent = '';
      label.textContent = '';
      const p = document.createElement('p');
      p.className = 'bad';
      p.textContent = `Load failed: ${String(err)}`;
      section.append(p);
    } finally {
      loadButton.disabled = false;
      clearButton.disabled = false;
    }
  });

  clearButton.addEventListener('click', async () => {
    clearButton.disabled = true;
    loaded?.weights.destroy();
    loaded = null;
    body.hidden = true;
    progressWrap.hidden = true;
    fill.style.width = '0';
    const store = await ModelStore.open(MODEL_ID);
    await store.clear();
    await describeCache();
    clearButton.disabled = false;
  });

  await describeCache();
}

// ---------------------------------------------------------------------------------------
// generation panel
// ---------------------------------------------------------------------------------------

let generationReady = false;

async function setUpGeneration(ctx: GpuContext, model: LoadedModel): Promise<void> {
  const section = show('generate');
  if (generationReady) return;
  generationReady = true;

  const promptInput = el<HTMLInputElement>('prompt');
  const maxTokensInput = el<HTMLInputElement>('max-tokens');
  const button = el<HTMLButtonElement>('run-generate');
  const out = el('generate-out');
  const stats = el('generate-stats');

  stats.textContent = 'compiling kernels…';
  const tokenizer = await BpeTokenizer.fromUrl(new URL('tokenizer.json', MODEL_BASE).href);
  const forward = await ForwardPass.create(
    ctx.device,
    model,
    new PipelineCache(ctx.device),
    { maxTokens: 256 },
  );
  stats.textContent = 'ready';
  void section;

  button.addEventListener('click', async () => {
    button.disabled = true;
    const prompt = promptInput.value;
    const maxNewTokens = Math.max(1, Math.min(128, Number(maxTokensInput.value) || 20));

    const promptSpan = document.createElement('span');
    promptSpan.className = 'prompt';
    promptSpan.textContent = prompt;
    const outputSpan = document.createElement('span');
    out.replaceChildren(promptSpan, outputSpan);
    stats.textContent = 'generating…';

    try {
      const ids = tokenizer.encode(prompt);
      const produced: number[] = [];
      const result = await generateGreedy(
        ids,
        async (sequence) => (await forward.run(sequence)).logits,
        {
          maxNewTokens,
          stopTokens: model.config.eosTokenIds,
          onToken: (id) => {
            produced.push(id);
            // Decoded from scratch each step: a multi-byte character is often split
            // across tokens, so decoding incrementally would emit replacement chars.
            outputSpan.textContent = tokenizer.decode(produced, { skipSpecialTokens: true });
          },
        },
      );
      const tokPerSec = (result.tokens.length / result.ms) * 1000;
      stats.textContent =
        `${result.tokens.length} tokens in ${(result.ms / 1000).toFixed(2)} s ` +
        `(${tokPerSec.toFixed(2)} tok/s, greedy, no KV cache)` +
        (result.stopped ? ' — stopped at EOS' : '');
    } catch (err) {
      stats.textContent = `generation failed: ${String(err)}`;
    } finally {
      button.disabled = false;
    }
  });
}

void main();
