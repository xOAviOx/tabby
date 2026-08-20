/**
 * Main-thread wrapper around the inference worker.
 *
 * Turns the message protocol into promises and callbacks so callers never see a
 * `postMessage`. Requests are numbered here, and responses for a request that has been
 * cancelled or superseded are dropped rather than delivered late.
 */

import type { ModelConfig } from '../engine/model.js';
import type { LoadProgress } from '../engine/store.js';
import type { SamplingParams } from '../engine/sampler.js';
import type { ChatMessage } from '../tokenizer/chat_template.js';
import type {
  GenerationStats,
  LoadStats,
  WorkerRequest,
  WorkerResponse,
} from './protocol.js';

export interface LoadOptions {
  baseUrl: string;
  modelId: string;
  maxSeqLen?: number;
  onProgress?: (progress: LoadProgress) => void;
}

export interface LoadedInfo {
  config: ModelConfig;
  stats: LoadStats;
  maxSeqLen: number;
  hasChatTemplate: boolean;
}

export interface GenerateOptions {
  /** Raw completion prompt. Mutually exclusive with `messages`. */
  prompt?: string;
  /** Chat turns, rendered through the model's own template. */
  messages?: ChatMessage[];
  maxNewTokens: number;
  sampling: SamplingParams;
  onToken?: (text: string, id: number, index: number) => void;
}

export interface KernelTiming {
  label: string;
  calls: number;
  totalMs: number;
  fraction: number;
}

export interface ProfileResult {
  supported: boolean;
  kernels: KernelTiming[];
  totalMs: number;
  passCount: number;
}

export interface GenerateHandle {
  /** Resolves when generation ends, whether by stop token, limit, or cancel. */
  done: Promise<GenerationStats>;
  cancel: () => void;
}

export class InferenceClient {
  private readonly worker: Worker;
  private nextRequestId = 1;

  private loadResolve: ((info: LoadedInfo) => void) | null = null;
  private loadReject: ((error: Error) => void) | null = null;
  private onProgress: ((progress: LoadProgress) => void) | null = null;

  private profileResolve: ((result: ProfileResult) => void) | null = null;

  private active: {
    requestId: number;
    onToken?: (text: string, id: number, index: number) => void;
    resolve: (stats: GenerationStats) => void;
    reject: (error: Error) => void;
  } | null = null;

  constructor(worker?: Worker) {
    this.worker =
      worker ??
      new Worker(new URL('./inference.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => this.handle(event.data);
    this.worker.onerror = (event) => {
      const error = new Error(`worker error: ${event.message || String(event)}`);
      this.loadReject?.(error);
      this.active?.reject(error);
      this.loadReject = null;
      this.active = null;
    };
  }

  private send(request: WorkerRequest): void {
    this.worker.postMessage(request);
  }

  private handle(message: WorkerResponse): void {
    switch (message.type) {
      case 'progress':
        this.onProgress?.({
          phase: message.phase,
          loadedBytes: message.loadedBytes,
          totalBytes: message.totalBytes,
          detail: message.detail,
          fromCache: message.fromCache,
        });
        break;

      case 'ready': {
        const resolve = this.loadResolve;
        this.loadResolve = null;
        this.loadReject = null;
        resolve?.({
          config: message.config,
          stats: message.stats,
          maxSeqLen: message.maxSeqLen,
          hasChatTemplate: message.hasChatTemplate,
        });
        break;
      }

      case 'token':
        if (this.active?.requestId === message.requestId) {
          this.active.onToken?.(message.text, message.id, message.index);
        }
        break;

      case 'stats': {
        if (this.active?.requestId !== message.requestId) break;
        const { resolve } = this.active;
        this.active = null;
        resolve(message.stats);
        break;
      }

      case 'profile': {
        const resolve = this.profileResolve;
        this.profileResolve = null;
        resolve?.({
          supported: message.supported,
          kernels: message.kernels,
          totalMs: message.totalMs,
          passCount: message.passCount,
        });
        break;
      }

      case 'error': {
        const error = new Error(message.message);
        error.name = message.name;
        if (message.requestId !== undefined && this.active?.requestId === message.requestId) {
          const { reject } = this.active;
          this.active = null;
          reject(error);
        } else if (this.loadReject) {
          const reject = this.loadReject;
          this.loadResolve = null;
          this.loadReject = null;
          reject(error);
        } else {
          this.active?.reject(error);
          this.active = null;
        }
        break;
      }
    }
  }

  load(options: LoadOptions): Promise<LoadedInfo> {
    this.onProgress = options.onProgress ?? null;
    const promise = new Promise<LoadedInfo>((resolve, reject) => {
      this.loadResolve = resolve;
      this.loadReject = reject;
    });
    this.send({
      type: 'load',
      baseUrl: options.baseUrl,
      modelId: options.modelId,
      ...(options.maxSeqLen === undefined ? {} : { maxSeqLen: options.maxSeqLen }),
    });
    return promise;
  }

  generate(options: GenerateOptions): GenerateHandle {
    if (this.active) throw new Error('a generation is already running');
    const requestId = this.nextRequestId++;

    const done = new Promise<GenerationStats>((resolve, reject) => {
      this.active = {
        requestId,
        ...(options.onToken ? { onToken: options.onToken } : {}),
        resolve,
        reject,
      };
    });

    this.send({
      type: 'generate',
      requestId,
      maxNewTokens: options.maxNewTokens,
      sampling: options.sampling,
      ...(options.prompt === undefined ? {} : { prompt: options.prompt }),
      ...(options.messages === undefined ? {} : { messages: options.messages }),
    });

    return { done, cancel: () => this.send({ type: 'cancel', requestId }) };
  }

  profile(prompt = 'The history of computing is'): Promise<ProfileResult> {
    const requestId = this.nextRequestId++;
    const promise = new Promise<ProfileResult>((resolve) => {
      this.profileResolve = resolve;
    });
    this.send({ type: 'profile', requestId, prompt });
    return promise;
  }

  terminate(): void {
    this.worker.terminate();
  }
}
