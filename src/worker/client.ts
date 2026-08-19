/**
 * Main-thread wrapper around the inference worker.
 *
 * Turns the message protocol into promises and callbacks so callers never see a
 * `postMessage`. Requests are numbered here, and responses for a request that has been
 * cancelled or superseded are dropped rather than delivered late.
 */

import type { ModelConfig } from '../engine/model.js';
import type { LoadProgress } from '../engine/store.js';
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
}

export interface GenerateOptions {
  prompt: string;
  maxNewTokens: number;
  continueContext?: boolean;
  onToken?: (text: string, id: number, index: number) => void;
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
        resolve?.({ config: message.config, stats: message.stats, maxSeqLen: message.maxSeqLen });
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
      prompt: options.prompt,
      maxNewTokens: options.maxNewTokens,
      ...(options.continueContext === undefined
        ? {}
        : { continueContext: options.continueContext }),
    });

    return { done, cancel: () => this.send({ type: 'cancel', requestId }) };
  }

  terminate(): void {
    this.worker.terminate();
  }
}
