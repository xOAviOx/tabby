/**
 * Message protocol between the UI and the inference worker.
 *
 * The worker owns the GPUDevice outright. Nothing on the main thread touches WebGPU, so
 * a long prefill or a slow decode step cannot block rendering -- which is the whole point
 * of moving it, and what the "UI stays responsive" gate checks.
 *
 * Every generate request carries a `requestId`. Responses echo it so a cancelled run's
 * late-arriving tokens can be discarded rather than appearing in the next conversation.
 */

import type { LoadPhase } from '../engine/store.js';
import type { ModelConfig } from '../engine/model.js';
import type { SamplingParams } from '../engine/sampler.js';
import type { ChatMessage } from '../tokenizer/chat_template.js';

export interface LoadRequest {
  type: 'load';
  baseUrl: string;
  modelId: string;
  maxSeqLen?: number;
}

export interface GenerateRequest {
  type: 'generate';
  requestId: number;
  /** Raw completion prompt. Mutually exclusive with `messages`. */
  prompt?: string;
  /** Chat turns, rendered through the model's own template from tokenizer_config.json. */
  messages?: ChatMessage[];
  maxNewTokens: number;
  sampling: SamplingParams;
}

/** Run one profiled decode step and return the per-kernel breakdown. */
export interface ProfileRequest {
  type: 'profile';
  requestId: number;
  prompt: string;
}

export interface CancelRequest {
  type: 'cancel';
  requestId: number;
}

export type WorkerRequest = LoadRequest | GenerateRequest | ProfileRequest | CancelRequest;

export interface ProgressResponse {
  type: 'progress';
  phase: LoadPhase;
  loadedBytes: number;
  totalBytes: number;
  detail: string;
  fromCache: boolean;
}

export interface LoadStats {
  downloadMs: number;
  uploadMs: number;
  totalMs: number;
  networkBytes: number;
  cacheBytes: number;
  vramBytes: number;
  kvCacheBytes: number;
  tensorCount: number;
  bufferCount: number;
  servedFromCache: boolean;
  pipelineMs: number;
  /** Weight bytes resident on the GPU; the denominator for effective bandwidth. */
  weightBytes: number;
  /** True when timestamp-query is available, so the perf panel can offer profiling. */
  canProfile: boolean;
}

export interface ReadyResponse {
  type: 'ready';
  config: ModelConfig;
  stats: LoadStats;
  maxSeqLen: number;
  /** True when the model shipped a chat template we could parse. */
  hasChatTemplate: boolean;
}

export interface TokenResponse {
  type: 'token';
  requestId: number;
  id: number;
  /** Text produced by this token, already correct across multi-byte boundaries. */
  text: string;
  index: number;
}

export interface GenerationStats {
  /** Time to first token, i.e. prefill plus one sample. */
  ttftMs: number;
  decodeMs: number;
  promptTokens: number;
  generatedTokens: number;
  prefillTokPerSec: number;
  decodeTokPerSec: number;
  stopped: boolean;
  cancelled: boolean;
  /**
   * Bytes copied GPU -> CPU per decode step. The M4 gate requires this to stay small:
   * reading the full logit vector would be ~608 KB at a 152k vocabulary.
   */
  readbackBytesPerToken: number;
  totalReadbackBytes: number;
  /** True if top-p ever wanted more probability mass than the top-k pool held. */
  poolExhausted: boolean;
}

export interface StatsResponse {
  type: 'stats';
  requestId: number;
  stats: GenerationStats;
}

export interface ProfileResponse {
  type: 'profile';
  requestId: number;
  supported: boolean;
  kernels: Array<{ label: string; calls: number; totalMs: number; fraction: number }>;
  totalMs: number;
  passCount: number;
}

export interface ErrorResponse {
  type: 'error';
  requestId?: number;
  name: string;
  message: string;
}

export type WorkerResponse =
  | ProgressResponse
  | ReadyResponse
  | TokenResponse
  | StatsResponse
  | ProfileResponse
  | ErrorResponse;
