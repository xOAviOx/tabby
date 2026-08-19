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

export interface LoadRequest {
  type: 'load';
  baseUrl: string;
  modelId: string;
  maxSeqLen?: number;
}

export interface GenerateRequest {
  type: 'generate';
  requestId: number;
  prompt: string;
  maxNewTokens: number;
  /** Continue from the existing KV cache instead of resetting it. */
  continueContext?: boolean;
}

export interface CancelRequest {
  type: 'cancel';
  requestId: number;
}

export type WorkerRequest = LoadRequest | GenerateRequest | CancelRequest;

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
}

export interface ReadyResponse {
  type: 'ready';
  config: ModelConfig;
  stats: LoadStats;
  maxSeqLen: number;
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
}

export interface StatsResponse {
  type: 'stats';
  requestId: number;
  stats: GenerationStats;
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
  | ErrorResponse;
