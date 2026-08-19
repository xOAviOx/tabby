/**
 * M0 dev surface: negotiate a device, show what we were granted, and run the one
 * kernel that exists against its CPU reference so a machine can be checked without
 * running the test suite.
 *
 * This page is replaced by the chat UI at M4; the limits panel folds into the perf
 * panel at M5.
 */

import { describeContext, requestGpuContext, WebGpuUnavailableError } from '../engine/device.js';
import { PipelineCache } from '../engine/pipelines.js';
import { runMatvecF32 } from '../engine/kernels.js';
import { matvecF32, matvecF64 } from '../reference/cpu.js';

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
}

void main();
