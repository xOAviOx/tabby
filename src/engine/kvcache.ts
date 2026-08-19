/**
 * KV cache: one preallocated pair of buffers for the whole model.
 *
 * Layout is [layer, position, kv_head, head_dim] for both K and V, which makes a layer's
 * slice contiguous and an append at a given position a plain offset write. Sizing is
 * fixed at construction from `maxSeqLen` -- growing a GPU buffer mid-generation would
 * mean reallocating and copying hundreds of megabytes at an unpredictable moment, so
 * running out of context is reported as a clean refusal instead.
 */

import { BufferArena } from './buffers.js';
import type { ModelConfig } from './model.js';

export class ContextOverflowError extends Error {
  readonly requested: number;
  readonly capacity: number;

  constructor(requested: number, capacity: number) {
    super(
      `context overflow: ${requested} tokens requested but the KV cache holds ${capacity}. ` +
        'Start a new conversation or rebuild the engine with a larger maxSeqLen.',
    );
    this.name = 'ContextOverflowError';
    this.requested = requested;
    this.capacity = capacity;
  }
}

export interface KvCacheOptions {
  maxSeqLen: number;
}

export class KvCache {
  readonly maxSeqLen: number;
  readonly numLayers: number;
  /** kv_heads * head_dim: elements written per position per layer. */
  readonly kvDim: number;

  readonly keys: GPUBuffer;
  readonly values: GPUBuffer;

  private readonly arena: BufferArena;
  private used = 0;

  constructor(device: GPUDevice, config: ModelConfig, options: KvCacheOptions) {
    if (options.maxSeqLen < 1) {
      throw new RangeError(`maxSeqLen must be at least 1, got ${options.maxSeqLen}`);
    }
    this.maxSeqLen = options.maxSeqLen;
    this.numLayers = config.numHiddenLayers;
    this.kvDim = config.numKeyValueHeads * config.headDim;

    this.arena = new BufferArena(device);
    const bytes = this.numLayers * this.maxSeqLen * this.kvDim * 4;
    this.keys = this.arena.storage(bytes, { label: 'kv.keys' });
    this.values = this.arena.storage(bytes, { label: 'kv.values' });
  }

  /** Positions currently held. */
  get length(): number {
    return this.used;
  }

  get capacity(): number {
    return this.maxSeqLen;
  }

  get byteLength(): number {
    return this.keys.size + this.values.size;
  }

  /** Element offset of (layer, position) within either cache buffer. */
  offsetOf(layer: number, position: number): number {
    return (layer * this.maxSeqLen + position) * this.kvDim;
  }

  /** Element offset of a layer's slice, for binding the cache to attention kernels. */
  layerOffset(layer: number): number {
    return layer * this.maxSeqLen * this.kvDim;
  }

  /**
   * Reserve room for `count` more positions and return the position they start at.
   * Throws rather than wrapping or truncating: silently dropping the oldest tokens
   * would change the model's output with no indication that anything happened.
   */
  reserve(count: number): number {
    if (this.used + count > this.maxSeqLen) {
      throw new ContextOverflowError(this.used + count, this.maxSeqLen);
    }
    const start = this.used;
    this.used += count;
    return start;
  }

  /** Forget everything. The buffers are kept; only the length resets. */
  reset(): void {
    this.used = 0;
  }

  destroy(): void {
    this.arena.destroy();
  }
}
