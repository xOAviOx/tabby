/// <reference lib="webworker" />

/**
 * The inference worker. Owns the GPUDevice, the weights, the KV cache and the tokenizer.
 *
 * Cancellation works because every decode step awaits GPU readback, which yields to the
 * event loop -- so a `cancel` message is delivered and the flag is observed before the
 * next token starts. No token is emitted after a cancel is seen.
 */

import { requestGpuContext, type GpuContext } from '../engine/device.js';
import { PipelineCache } from '../engine/pipelines.js';
import { loadModel, type LoadedModel } from '../engine/model.js';
import { ForwardPass } from '../engine/forward.js';
import { BpeTokenizer } from '../tokenizer/bpe.js';
import { loadChatTemplate, type LoadedChatTemplate } from '../tokenizer/chat_template.js';
import { SeededRandom, sampleFromTopK } from '../engine/sampler.js';
import { MAX_TOP_K } from '../engine/kernels.js';
import type { TopKResult } from '../engine/forward.js';
import type {
  GenerateRequest,
  LoadRequest,
  WorkerRequest,
  WorkerResponse,
} from './protocol.js';

const scope = self as unknown as DedicatedWorkerGlobalScope;

let ctx: GpuContext | null = null;
let model: LoadedModel | null = null;
let forward: ForwardPass | null = null;
let tokenizer: BpeTokenizer | null = null;
let chat: LoadedChatTemplate | null = null;
let maxSeqLen = 512;

/** Requests cancelled since they were issued. Checked between every decode step. */
const cancelled = new Set<number>();
let activeRequest: number | null = null;

function post(message: WorkerResponse): void {
  scope.postMessage(message);
}

function fail(error: unknown, requestId?: number): void {
  const err = error instanceof Error ? error : new Error(String(error));
  post({
    type: 'error',
    ...(requestId === undefined ? {} : { requestId }),
    name: err.name,
    message: err.message,
  });
}

async function handleLoad(request: LoadRequest): Promise<void> {
  ctx = await requestGpuContext({
    onDeviceLost: (info) => fail(new Error(`GPU device lost (${info.reason}): ${info.message}`)),
  });
  maxSeqLen = request.maxSeqLen ?? maxSeqLen;

  model = await loadModel(ctx.device, {
    baseUrl: request.baseUrl,
    modelId: request.modelId,
    onProgress: (progress) => post({ type: 'progress', ...progress }),
  });

  tokenizer = await BpeTokenizer.fromUrl(new URL('tokenizer.json', request.baseUrl).href);
  try {
    chat = await loadChatTemplate(new URL('tokenizer_config.json', request.baseUrl).href);
  } catch (err) {
    // A model without a usable template can still do raw completion, so this is a
    // downgrade rather than a failure -- but it is reported, not swallowed.
    console.warn('[worker] no usable chat template:', err);
    chat = null;
  }

  // Pipeline compilation is slow and must never happen inside the generation loop, so
  // it is finished before `ready` is announced.
  const pipelineStart = performance.now();
  forward = await ForwardPass.create(ctx.device, model, new PipelineCache(ctx.device), {
    maxSeqLen,
  });
  const pipelineMs = performance.now() - pipelineStart;

  post({
    type: 'ready',
    config: model.config,
    maxSeqLen,
    hasChatTemplate: chat !== null,
    stats: {
      downloadMs: model.stats.downloadMs,
      uploadMs: model.stats.uploadMs,
      totalMs: model.stats.totalMs,
      networkBytes: model.stats.networkBytes,
      cacheBytes: model.stats.cacheBytes,
      vramBytes: model.stats.vramBytes,
      kvCacheBytes: forward.cache.byteLength,
      tensorCount: model.stats.tensorCount,
      bufferCount: model.stats.bufferCount,
      servedFromCache: model.stats.servedFromCache,
      pipelineMs,
    },
  });
}

async function handleGenerate(request: GenerateRequest): Promise<void> {
  if (!forward || !tokenizer || !model) {
    throw new Error('generate received before the model finished loading');
  }

  const { requestId, sampling } = request;
  activeRequest = requestId;
  const stopTokens = new Set(model.config.eosTokenIds);

  // The candidate pool the GPU extracts. Temperature 0 needs only the argmax; otherwise
  // top-p may look past top-k, so the pool is the larger of the two intents.
  const poolSize =
    sampling.temperature <= 0
      ? 1
      : Math.min(MAX_TOP_K, Math.max(1, sampling.topK || MAX_TOP_K));

  let prompt: string;
  if (request.messages) {
    if (!chat) throw new Error('this model has no chat template; send a raw prompt instead');
    prompt = chat.template.render(request.messages, {
      addGenerationPrompt: true,
      bosToken: chat.bosToken,
      eosToken: chat.eosToken,
    });
  } else {
    prompt = request.prompt ?? '';
  }

  // Multi-turn re-renders and re-prefills the whole conversation. Correct and simple;
  // prefix reuse across turns is an optimisation, not an M4 requirement.
  forward.reset();
  const promptIds = tokenizer.encode(prompt);

  const random = new SeededRandom(sampling.seed);
  const started = performance.now();
  const produced: number[] = [];
  // Text is re-decoded from the full token list each step and the delta is sent: a
  // multi-byte character is frequently split across two tokens, so decoding a token in
  // isolation would emit replacement characters.
  let emitted = '';
  let stopped = false;
  let poolExhausted = false;
  let totalReadback = 0;
  let readbackPerToken = 0;

  const emit = (id: number): void => {
    produced.push(id);
    const full = tokenizer!.decode(produced, { skipSpecialTokens: true });
    const delta = full.slice(emitted.length);
    emitted = full;
    post({ type: 'token', requestId, id, text: delta, index: produced.length - 1 });
  };

  const choose = (result: { topK: TopKResult | null; readbackBytes: number }): number => {
    totalReadback += result.readbackBytes;
    readbackPerToken = result.readbackBytes;
    const outcome = sampleFromTopK(result.topK!, sampling, random);
    if (outcome.poolExhausted) poolExhausted = true;
    return outcome.id;
  };

  const prefilled = await forward.prefill(promptIds, { topK: poolSize });
  let next = choose(prefilled);
  const ttftMs = performance.now() - started;

  if (cancelled.has(requestId)) {
    finish(requestId, promptIds.length, 0, ttftMs, 0, false, true, readbackPerToken, totalReadback, poolExhausted);
    return;
  }

  emit(next);
  if (stopTokens.has(next)) stopped = true;

  const decodeStart = performance.now();
  while (!stopped && produced.length < request.maxNewTokens) {
    if (cancelled.has(requestId)) break;
    const result = await forward.decode(next, { topK: poolSize });
    // Re-check after the await: a cancel may have landed while the GPU was busy, and
    // emitting here would leak a token into a run the caller already abandoned.
    if (cancelled.has(requestId)) break;
    next = choose(result);
    emit(next);
    if (stopTokens.has(next)) stopped = true;
  }
  const decodeMs = performance.now() - decodeStart;

  finish(
    requestId,
    promptIds.length,
    produced.length,
    ttftMs,
    decodeMs,
    stopped,
    cancelled.has(requestId),
    readbackPerToken,
    totalReadback,
    poolExhausted,
  );
}

function finish(
  requestId: number,
  promptTokens: number,
  generatedTokens: number,
  ttftMs: number,
  decodeMs: number,
  stopped: boolean,
  wasCancelled: boolean,
  readbackBytesPerToken: number,
  totalReadbackBytes: number,
  poolExhausted: boolean,
): void {
  const decoded = Math.max(0, generatedTokens - 1);
  post({
    type: 'stats',
    requestId,
    stats: {
      ttftMs,
      decodeMs,
      promptTokens,
      generatedTokens,
      prefillTokPerSec: ttftMs > 0 ? (promptTokens / ttftMs) * 1000 : 0,
      decodeTokPerSec: decodeMs > 0 ? (decoded / decodeMs) * 1000 : 0,
      stopped,
      cancelled: wasCancelled,
      readbackBytesPerToken,
      totalReadbackBytes,
      poolExhausted,
    },
  });
  cancelled.delete(requestId);
  activeRequest = null;
}

scope.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  switch (request.type) {
    case 'load':
      handleLoad(request).catch((err) => fail(err));
      break;
    case 'generate':
      handleGenerate(request).catch((err) => fail(err, request.requestId));
      break;
    case 'cancel':
      // Recorded even if the request has not started yet, so a cancel that races the
      // generate message still takes effect.
      cancelled.add(request.requestId);
      break;
    default: {
      const exhaustive: never = request;
      fail(new Error(`unknown request ${JSON.stringify(exhaustive)}`));
    }
  }
};

scope.onerror = (event) => {
  fail(new Error(String(event)), activeRequest ?? undefined);
};
