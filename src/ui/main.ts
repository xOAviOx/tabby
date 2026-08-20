/**
 * Chat UI.
 *
 * All GPU work lives in the inference worker; this file only renders. The spinner is
 * driven by requestAnimationFrame on the main thread, so it is the visible form of the
 * "UI stays responsive" property rather than decoration -- if inference were running
 * here, it would freeze.
 *
 * The diagnostics panel keeps the device/limits/self-check surface from M0-M1; it folds
 * into the perf panel at M5.
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
import { InferenceClient, type LoadedInfo } from '../worker/client.js';
import type { SamplingParams } from '../engine/sampler.js';
import type { ChatMessage } from '../tokenizer/chat_template.js';

// The int4 build: 265 MiB against 942 MiB for fp16, and ~3.4x the decode throughput.
const MODEL_ID = 'qwen2.5-0.5b-instruct-q4';
const MODEL_BASE = new URL(`/models/${MODEL_ID}/`, location.href).href;
const MAX_SEQ_LEN = 1024;

const SELF_CHECK_SHAPES = [
  { m: 3, n: 7 },
  { m: 65, n: 129 },
  { m: 1000, n: 999 },
  { m: 896, n: 4864 },
];

/** Same tolerance the test suite gates on. */
const TOLERANCE = 1e-4;
const MIB = 1024 * 1024;

// ---------------------------------------------------------------------------------------
// small DOM helpers
// ---------------------------------------------------------------------------------------

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

function formatBytes(bytes: number): string {
  if (bytes >= MIB) return `${(bytes / MIB).toFixed(1)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${bytes} B`;
}

function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(0)} ms`;
}

function formatLimit(key: string, value: number): string {
  return key.endsWith('Size') && value >= MIB
    ? `${value.toLocaleString()}  (${(value / MIB).toFixed(0)} MiB)`
    : value.toLocaleString();
}

function renderUnavailable(message: string, detail: string): void {
  const status = el('status');
  status.classList.add('fail');
  status.replaceChildren();
  const heading = document.createElement('h2');
  heading.textContent = message;
  const body = document.createElement('p');
  body.textContent = detail;
  const help = document.createElement('p');
  help.className = 'muted';
  help.textContent =
    'WebGPU needs Chrome or Edge 113+, or Safari 18+. On Linux it may require enabling ' +
    'the Vulkan backend. Firefox ships it in release from 141 on Windows only.';
  status.append(heading, body, help);
}

// ---------------------------------------------------------------------------------------
// chat
// ---------------------------------------------------------------------------------------

function readSampling(): SamplingParams {
  return {
    temperature: Number(el<HTMLInputElement>('temperature').value),
    topK: Number(el<HTMLInputElement>('top-k').value),
    topP: Number(el<HTMLInputElement>('top-p').value),
    seed: Number(el<HTMLInputElement>('seed').value) || 0,
  };
}

function bindSettingReadouts(): void {
  for (const [input, output] of [
    ['temperature', 'temperature-out'],
    ['top-k', 'top-k-out'],
    ['top-p', 'top-p-out'],
  ] as Array<[string, string]>) {
    const source = el<HTMLInputElement>(input);
    const target = el<HTMLOutputElement>(output);
    const sync = (): void => {
      target.textContent = source.value;
    };
    source.addEventListener('input', sync);
    sync();
  }
}

function setUpChat(client: InferenceClient, info: LoadedInfo): void {
  show('chat');
  bindSettingReadouts();

  const messagesEl = el('messages');
  const input = el<HTMLInputElement>('chat-input');
  const sendButton = el<HTMLButtonElement>('send');
  const stopButton = el<HTMLButtonElement>('stop');
  const resetButton = el<HTMLButtonElement>('reset-chat');
  const spinner = el('spinner');
  const statsEl = el('chat-stats');

  /** The conversation so far. Re-rendered through the model's template every turn. */
  const history: ChatMessage[] = [];
  let busy = false;

  const bubble = (role: string, text = ''): HTMLElement => {
    const node = document.createElement('div');
    node.className = `msg ${role}`;
    node.textContent = text;
    messagesEl.append(node);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return node;
  };

  let spinning = false;
  let angle = 0;
  const tick = (): void => {
    if (!spinning) return;
    angle = (angle + 6) % 360;
    spinner.style.transform = `rotate(${angle}deg)`;
    requestAnimationFrame(tick);
  };

  const setBusy = (value: boolean): void => {
    busy = value;
    sendButton.disabled = value;
    input.disabled = value;
    resetButton.disabled = value;
    stopButton.disabled = !value;
    spinning = value;
    spinner.classList.toggle('spinning', value);
    if (value) requestAnimationFrame(tick);
  };

  resetButton.addEventListener('click', () => {
    history.length = 0;
    messagesEl.replaceChildren();
    statsEl.textContent = `context ${info.maxSeqLen} tokens`;
  });

  el<HTMLFormElement>('composer').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (busy) return;
    const text = input.value.trim();
    if (!text) return;

    input.value = '';
    bubble('user', text);
    history.push({ role: 'user', content: text });

    const reply = bubble('assistant');
    const caret = document.createElement('span');
    caret.className = 'caret';
    reply.append(caret);

    const systemPrompt = el<HTMLInputElement>('system-prompt').value.trim();
    const messages: ChatMessage[] = systemPrompt
      ? [{ role: 'system', content: systemPrompt }, ...history]
      : [...history];

    setBusy(true);
    statsEl.textContent = 'thinking…';

    let answer = '';
    const handle = client.generate({
      messages,
      maxNewTokens: Number(el<HTMLInputElement>('max-tokens').value) || 256,
      sampling: readSampling(),
      onToken: (chunk) => {
        answer += chunk;
        caret.remove();
        reply.textContent = answer;
        reply.append(caret);
        messagesEl.scrollTop = messagesEl.scrollHeight;
      },
    });
    stopButton.onclick = () => handle.cancel();

    try {
      const stats = await handle.done;
      caret.remove();
      reply.textContent = answer;
      history.push({ role: 'assistant', content: answer });

      statsEl.textContent =
        `${stats.generatedTokens} tok · TTFT ${stats.ttftMs.toFixed(0)} ms · ` +
        `prefill ${stats.prefillTokPerSec.toFixed(0)} tok/s · ` +
        `decode ${stats.decodeTokPerSec.toFixed(1)} tok/s · ` +
        `${stats.readbackBytesPerToken} B/token readback` +
        (stats.cancelled ? ' · stopped' : '') +
        (stats.poolExhausted ? ' · top-p clipped by k' : '');
      lastTurnStats = {
        ttftMs: stats.ttftMs,
        decodeTokPerSec: stats.decodeTokPerSec,
        prefillTokPerSec: stats.prefillTokPerSec,
        readbackBytesPerToken: stats.readbackBytesPerToken,
      };
      perfRefresh?.();
    } catch (error) {
      caret.remove();
      reply.remove();
      // Drop the user turn too, so history stays a valid alternating conversation.
      history.pop();
      bubble('error', error instanceof Error ? error.message : String(error));
      statsEl.textContent = '';
    } finally {
      setBusy(false);
      input.focus();
    }
  });

  statsEl.textContent = `context ${info.maxSeqLen} tokens`;
  input.focus();
}

// ---------------------------------------------------------------------------------------
// model panel
// ---------------------------------------------------------------------------------------

async function setUpModelPanel(): Promise<void> {
  show('model');
  const loadButton = el<HTMLButtonElement>('load-model');
  const clearButton = el<HTMLButtonElement>('clear-cache');
  const status = el('model-status');
  const progressWrap = el('progress-wrap');
  const fill = el('progress-fill');
  const label = el('progress-label');
  const details = el('model-details');
  const body = el('model-body');

  let client: InferenceClient | null = null;

  const describeCache = async (): Promise<void> => {
    const store = await ModelStore.open(MODEL_ID);
    const bytes = await store.usageBytes();
    status.textContent = bytes > 0 ? `${formatBytes(bytes)} cached in OPFS` : 'not cached';
  };

  const onProgress = (progress: LoadProgress): void => {
    progressWrap.hidden = false;
    const fraction = progress.totalBytes > 0 ? progress.loadedBytes / progress.totalBytes : 0;
    fill.style.width = `${(fraction * 100).toFixed(2)}%`;
    // Real byte counts, not a synthetic ramp.
    const source =
      progress.phase === 'download' ? (progress.fromCache ? 'cache' : 'network') : 'GPU';
    label.textContent =
      `${progress.phase} (${source})  ${formatBytes(progress.loadedBytes)} / ` +
      `${formatBytes(progress.totalBytes)}  ${(fraction * 100).toFixed(1)}%  ${progress.detail}`;
  };

  loadButton.addEventListener('click', async () => {
    loadButton.disabled = true;
    clearButton.disabled = true;
    details.hidden = true;
    status.textContent = 'loading…';
    try {
      client?.terminate();
      client = new InferenceClient();
      const info = await client.load({
        baseUrl: MODEL_BASE,
        modelId: MODEL_ID,
        maxSeqLen: MAX_SEQ_LEN,
        onProgress,
      });

      const { stats, config } = info;
      fill.style.width = '100%';
      status.textContent = stats.servedFromCache ? 'loaded from OPFS' : 'downloaded and cached';

      defineList(body, [
        ['served from', stats.servedFromCache ? 'OPFS cache' : 'network'],
        [
          'fetch',
          `${formatMs(stats.downloadMs)}  (${formatBytes(stats.networkBytes)} network, ` +
            `${formatBytes(stats.cacheBytes)} cache)`,
        ],
        ['GPU upload', formatMs(stats.uploadMs)],
        ['pipelines', formatMs(stats.pipelineMs)],
        ['total', formatMs(stats.totalMs)],
        ['tensors / buffers', `${stats.tensorCount} / ${stats.bufferCount}`],
        ['weights VRAM', formatBytes(stats.vramBytes)],
        ['KV cache', `${formatBytes(stats.kvCacheBytes)} for ${info.maxSeqLen} tokens`],
        ['chat template', info.hasChatTemplate ? 'loaded' : 'none — completion only'],
        ['architecture', `${config.modelType}, ${config.numHiddenLayers} layers`],
        ['hidden / intermediate', `${config.hiddenSize} / ${config.intermediateSize}`],
        [
          'heads (Q / KV)',
          `${config.numAttentionHeads} / ${config.numKeyValueHeads} ` +
            `(${config.queryHeadsPerKvHead} per KV)`,
        ],
        ['head_dim', String(config.headDim)],
        ['vocab', config.vocabSize.toLocaleString()],
        ['rope_theta', String(config.ropeTheta)],
        ['tied embeddings', config.tieWordEmbeddings ? 'yes' : 'no'],
      ]);
      details.hidden = false;
      setUpChat(client, info);
      setUpPerfPanel(client, info);
    } catch (error) {
      status.textContent = '';
      label.textContent = '';
      const message = document.createElement('p');
      message.className = 'bad';
      message.textContent = `Load failed: ${String(error)}`;
      el('model').append(message);
    } finally {
      loadButton.disabled = false;
      clearButton.disabled = false;
    }
  });

  clearButton.addEventListener('click', async () => {
    clearButton.disabled = true;
    details.hidden = true;
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
// performance panel
// ---------------------------------------------------------------------------------------

/** Updated from each chat turn's stats so the panel reflects real generation, not a probe. */
let lastTurnStats: {
  ttftMs: number;
  decodeTokPerSec: number;
  prefillTokPerSec: number;
  readbackBytesPerToken: number;
} | null = null;

function setUpPerfPanel(client: InferenceClient, info: LoadedInfo): void {
  show('perf');
  const body = el('perf-body');
  const table = el<HTMLTableElement>('perf-kernels');
  const note = el('perf-note');
  const button = el<HTMLButtonElement>('run-profile');

  const refresh = (): void => {
    const rows: Array<[string, string]> = [
      ['weights in VRAM', formatBytes(info.stats.weightBytes)],
      ['KV cache', `${formatBytes(info.stats.kvCacheBytes)} for ${info.maxSeqLen} tokens`],
    ];
    if (lastTurnStats) {
      const gbps = (info.stats.weightBytes * lastTurnStats.decodeTokPerSec) / 1e9;
      rows.push(
        ['decode', `${lastTurnStats.decodeTokPerSec.toFixed(1)} tok/s`],
        ['prefill', `${lastTurnStats.prefillTokPerSec.toFixed(0)} tok/s`],
        ['TTFT', `${lastTurnStats.ttftMs.toFixed(0)} ms`],
        // Decode streams the whole weight set once per token, so this is the number that
        // says whether the kernels are using the memory system well.
        ['effective bandwidth', `${gbps.toFixed(1)} GB/s`],
        ['readback', `${lastTurnStats.readbackBytesPerToken} B / token`],
      );
    } else {
      rows.push(['decode', 'send a message to measure']);
    }
    defineList(body, rows);
  };
  refresh();
  perfRefresh = refresh;

  if (!info.stats.canProfile) {
    button.disabled = true;
    note.textContent = 'timestamp-query is unavailable on this adapter, so per-kernel timing is off.';
    return;
  }

  button.addEventListener('click', async () => {
    button.disabled = true;
    note.textContent = 'profiling…';
    try {
      const report = await client.profile();
      if (!report.supported) {
        note.textContent = 'timestamp-query is unavailable on this adapter.';
        return;
      }
      table.innerHTML =
        '<thead><tr><th>kernel</th><th>calls</th><th>ms</th><th>share</th></tr></thead><tbody></tbody>';
      const tbody = table.querySelector('tbody')!;
      for (const kernel of report.kernels.slice(0, 12)) {
        const row = document.createElement('tr');
        for (const text of [
          kernel.label,
          String(kernel.calls),
          kernel.totalMs.toFixed(3),
          `${(kernel.fraction * 100).toFixed(1)}%`,
        ]) {
          const cell = document.createElement('td');
          cell.textContent = text;
          row.append(cell);
        }
        tbody.append(row);
      }
      note.textContent =
        `${report.passCount} dispatches, ${report.totalMs.toFixed(2)} ms of GPU time. ` +
        'Profiling gives every dispatch its own compute pass, which is slower than normal ' +
        'encoding — the shares are meaningful, the absolute total is not.';
    } catch (error) {
      note.textContent = `profiling failed: ${String(error)}`;
    } finally {
      button.disabled = false;
    }
  });
}

let perfRefresh: (() => void) | null = null;

// ---------------------------------------------------------------------------------------
// diagnostics
// ---------------------------------------------------------------------------------------

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

async function setUpDiagnostics(ctx: GpuContext): Promise<void> {
  show('diagnostics');
  const { info } = ctx;

  defineList(el('adapter-body'), [
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
    el('limits-body'),
    Object.entries(ctx.limits)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, value]) => [key, formatLimit(key, value)] as [string, string]),
  );

  const table = el<HTMLTableElement>('selfcheck-body');
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
      const cell = document.createElement('td');
      cell.textContent = text;
      if (cls) cell.className = cls;
      row.append(cell);
    }
    tbody.append(row);
  }
}

// ---------------------------------------------------------------------------------------

async function main(): Promise<void> {
  let ctx: GpuContext;
  try {
    ctx = await requestGpuContext({
      onDeviceLost: (info) =>
        renderUnavailable('GPU device lost', `${info.reason}: ${info.message}`),
    });
  } catch (error) {
    if (error instanceof WebGpuUnavailableError) {
      renderUnavailable('WebGPU is not available in this browser', error.message);
    } else {
      renderUnavailable('Could not initialise WebGPU', String(error));
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

  await setUpModelPanel();
  await setUpDiagnostics(ctx);
}

void main();
