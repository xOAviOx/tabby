/**
 * Header parsing, the weight registry, and weight upload to the GPU.
 *
 * Two things here are load-bearing for later milestones:
 *
 * 1. **Nothing is hardcoded.** Every dimension comes from `model.json`, which came from
 *    the source `config.json`. Swapping models is a conversion, not a code change.
 *
 * 2. **Tensors shard along rows.** A tensor too large for one storage binding is split
 *    into row blocks, and the split is recorded so kernels can iterate. The threshold is
 *    injectable rather than read straight from `device.limits`, because on a machine
 *    that grants a 4 GiB binding size no real tensor would ever shard and the path would
 *    never be exercised. Tests force a small threshold; production passes none.
 */

import { createStorageBuffer } from './buffers.js';
import {
  ChunkedByteReader,
  ModelStore,
  ModelStoreError,
  ensureChunk,
  type ChunkMeta,
  type ProgressCallback,
} from './store.js';

export type TensorDType = 'f16' | 'f32';

const ELEMENT_BYTES: Record<TensorDType, number> = { f16: 2, f32: 4 };

/** Populated at M5. Declared now so the header schema does not change under us. */
export interface QuantMeta {
  scheme: string;
  bits: number;
  blockSize: number;
}

export interface TensorMeta {
  name: string;
  dtype: TensorDType;
  shape: number[];
  /** Offset into the logical concatenation of all chunk files. */
  offset: number;
  byteLength: number;
  quant: QuantMeta | null;
}

export interface ModelConfig {
  hiddenSize: number;
  numHiddenLayers: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  intermediateSize: number;
  vocabSize: number;
  ropeTheta: number;
  rmsNormEps: number;
  tieWordEmbeddings: boolean;
  maxPositionEmbeddings: number;
  modelType: string;
  hiddenAct: string;
  bosTokenId: number | null;
  eosTokenIds: number[];
  /** How many query heads share one KV head. 1 means plain multi-head attention. */
  queryHeadsPerKvHead: number;
}

export interface WeightHeader {
  format: string;
  version: number;
  config: ModelConfig;
  chunkBytes: number;
  totalBytes: number;
  chunks: ChunkMeta[];
  tensors: TensorMeta[];
  aliases: Record<string, string>;
  raw: unknown;
}

export class ModelLoadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ModelLoadError';
  }
}

const EXPECTED_FORMAT = 'browser-llm-weights';
const SUPPORTED_VERSION = 1;

function requireNumber(source: Record<string, unknown>, key: string): number {
  const value = source[key];
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ModelLoadError(`config.${key} is missing or not a finite number`);
  }
  return value;
}

export function parseConfig(raw: Record<string, unknown>): ModelConfig {
  const numAttentionHeads = requireNumber(raw, 'num_attention_heads');
  const numKeyValueHeads = requireNumber(raw, 'num_key_value_heads');
  const hiddenSize = requireNumber(raw, 'hidden_size');

  if (numKeyValueHeads <= 0 || numAttentionHeads % numKeyValueHeads !== 0) {
    throw new ModelLoadError(
      `num_attention_heads (${numAttentionHeads}) must be a positive multiple of ` +
        `num_key_value_heads (${numKeyValueHeads})`,
    );
  }

  const headDim =
    typeof raw.head_dim === 'number' ? raw.head_dim : hiddenSize / numAttentionHeads;
  if (!Number.isInteger(headDim)) {
    throw new ModelLoadError(`derived head_dim ${headDim} is not an integer`);
  }

  const eos = raw.eos_token_id;
  const eosTokenIds = Array.isArray(eos)
    ? eos.filter((v): v is number => typeof v === 'number')
    : typeof eos === 'number'
      ? [eos]
      : [];

  return {
    hiddenSize,
    numHiddenLayers: requireNumber(raw, 'num_hidden_layers'),
    numAttentionHeads,
    numKeyValueHeads,
    headDim,
    intermediateSize: requireNumber(raw, 'intermediate_size'),
    vocabSize: requireNumber(raw, 'vocab_size'),
    ropeTheta: requireNumber(raw, 'rope_theta'),
    rmsNormEps: requireNumber(raw, 'rms_norm_eps'),
    tieWordEmbeddings: Boolean(raw.tie_word_embeddings),
    maxPositionEmbeddings:
      typeof raw.max_position_embeddings === 'number' ? raw.max_position_embeddings : 0,
    modelType: typeof raw.model_type === 'string' ? raw.model_type : 'unknown',
    hiddenAct: typeof raw.hidden_act === 'string' ? raw.hidden_act : 'silu',
    bosTokenId: typeof raw.bos_token_id === 'number' ? raw.bos_token_id : null,
    eosTokenIds,
    queryHeadsPerKvHead: numAttentionHeads / numKeyValueHeads,
  };
}

export function parseHeader(raw: unknown): WeightHeader {
  if (typeof raw !== 'object' || raw === null) {
    throw new ModelLoadError('model.json is not an object');
  }
  const obj = raw as Record<string, unknown>;
  if (obj.format !== EXPECTED_FORMAT) {
    throw new ModelLoadError(`unexpected format ${String(obj.format)}, want ${EXPECTED_FORMAT}`);
  }
  if (obj.version !== SUPPORTED_VERSION) {
    throw new ModelLoadError(
      `model.json is version ${String(obj.version)}, this build supports ${SUPPORTED_VERSION}`,
    );
  }

  const tensors = (obj.tensors as TensorMeta[]).map((t) => {
    if (t.dtype !== 'f16' && t.dtype !== 'f32') {
      throw new ModelLoadError(`tensor ${t.name}: unsupported dtype ${t.dtype}`);
    }
    return t;
  });

  return {
    format: obj.format,
    version: obj.version,
    config: parseConfig(obj.config as Record<string, unknown>),
    chunkBytes: obj.chunkBytes as number,
    totalBytes: obj.totalBytes as number,
    chunks: obj.chunks as ChunkMeta[],
    tensors,
    aliases: (obj.aliases as Record<string, string>) ?? {},
    raw,
  };
}

/** One contiguous row block of a tensor, living in its own GPUBuffer. */
export interface TensorShard {
  buffer: GPUBuffer;
  /** Index of this shard's first row within the whole tensor. */
  rowStart: number;
  rowCount: number;
  byteLength: number;
}

export interface GpuTensor {
  name: string;
  dtype: TensorDType;
  shape: number[];
  /** shape[0]; rank-1 tensors are treated as `rows` scalars. */
  rows: number;
  /** Product of shape[1..], or 1 for rank-1. */
  cols: number;
  bytesPerRow: number;
  byteLength: number;
  shards: TensorShard[];
  quant: QuantMeta | null;
}

export function tensorGeometry(meta: TensorMeta): {
  rows: number;
  cols: number;
  bytesPerRow: number;
} {
  const elementBytes = ELEMENT_BYTES[meta.dtype];
  const rows = meta.shape.length > 0 ? meta.shape[0] : 1;
  const cols = meta.shape.slice(1).reduce((a, b) => a * b, 1);
  return { rows, cols, bytesPerRow: cols * elementBytes };
}

/**
 * Split a tensor into row blocks that each fit within `limitBytes`.
 *
 * Splitting along rows (not arbitrary byte boundaries) is what lets a kernel treat a
 * shard as a smaller matrix with a known row offset instead of reasoning about torn
 * rows. A single row wider than the limit cannot be handled this way and is a hard
 * error -- it would need column sharding, which no model in scope requires.
 */
export function planShards(
  rows: number,
  bytesPerRow: number,
  limitBytes: number,
): Array<{ rowStart: number; rowCount: number; byteLength: number }> {
  if (rows === 0) return [];
  if (bytesPerRow > limitBytes) {
    throw new ModelLoadError(
      `a single row is ${bytesPerRow} bytes, above the ${limitBytes}-byte binding limit; ` +
        'column sharding would be required',
    );
  }
  const rowsPerShard = Math.max(1, Math.floor(limitBytes / bytesPerRow));
  const plan: Array<{ rowStart: number; rowCount: number; byteLength: number }> = [];
  for (let rowStart = 0; rowStart < rows; rowStart += rowsPerShard) {
    const rowCount = Math.min(rowsPerShard, rows - rowStart);
    plan.push({ rowStart, rowCount, byteLength: rowCount * bytesPerRow });
  }
  return plan;
}

/**
 * Name -> GPU tensor, resolving the alias table so a tied `lm_head.weight` and
 * `model.embed_tokens.weight` return the same buffers without storing them twice.
 */
export class WeightRegistry {
  private readonly tensors = new Map<string, GpuTensor>();

  constructor(private readonly aliases: Record<string, string> = {}) {}

  add(tensor: GpuTensor): void {
    this.tensors.set(tensor.name, tensor);
  }

  resolve(name: string): string {
    return this.aliases[name] ?? name;
  }

  has(name: string): boolean {
    return this.tensors.has(this.resolve(name));
  }

  get(name: string): GpuTensor {
    const resolved = this.resolve(name);
    const tensor = this.tensors.get(resolved);
    if (!tensor) {
      const hint = resolved === name ? '' : ` (aliased from ${name})`;
      throw new ModelLoadError(`no weight named ${resolved}${hint}`);
    }
    return tensor;
  }

  names(): string[] {
    return [...this.tensors.keys()];
  }

  get count(): number {
    return this.tensors.size;
  }

  get byteLength(): number {
    let total = 0;
    for (const tensor of this.tensors.values()) total += tensor.byteLength;
    return total;
  }

  /** Largest shard count across all tensors -- how many dispatches a kernel may need. */
  get maxShards(): number {
    let most = 1;
    for (const tensor of this.tensors.values()) most = Math.max(most, tensor.shards.length);
    return most;
  }

  destroy(): void {
    for (const tensor of this.tensors.values()) {
      for (const shard of tensor.shards) shard.buffer.destroy();
    }
    this.tensors.clear();
  }
}

export interface LoadModelOptions {
  /** Directory URL containing model.json and the chunk files. */
  baseUrl: string;
  /** OPFS namespace for this model's cached chunks. */
  modelId: string;
  /**
   * Maximum bytes per storage buffer. Defaults to the device's binding limit.
   * Tests pass a small value to force the multi-shard path on any machine.
   */
  shardThresholdBytes?: number;
  onProgress?: ProgressCallback;
  signal?: AbortSignal;
  /** Verify each chunk's sha256 after download. Default true. */
  verifyChunks?: boolean;
  /** Bytes per writeBuffer call. Bounds peak host memory during upload. */
  uploadPieceBytes?: number;
}

export interface LoadStats {
  downloadMs: number;
  uploadMs: number;
  totalMs: number;
  /** Bytes actually pulled over the network. Zero on a fully cached load. */
  networkBytes: number;
  /** Bytes served from OPFS. */
  cacheBytes: number;
  chunkCount: number;
  tensorCount: number;
  bufferCount: number;
  vramBytes: number;
  servedFromCache: boolean;
}

export interface LoadedModel {
  header: WeightHeader;
  config: ModelConfig;
  weights: WeightRegistry;
  stats: LoadStats;
  store: ModelStore;
}

const DEFAULT_UPLOAD_PIECE = 8 * 1024 * 1024;
const HEADER_FILE = 'model.json';

/**
 * Fetch the header, preferring the OPFS copy so a cached model works fully offline.
 */
export async function loadHeader(
  store: ModelStore,
  baseUrl: string,
  signal?: AbortSignal,
): Promise<{ header: WeightHeader; fromCache: boolean }> {
  const cachedSize = await store.sizeOf(HEADER_FILE);
  if (cachedSize > 0) {
    const text = new TextDecoder().decode(await store.readAll(HEADER_FILE));
    return { header: parseHeader(JSON.parse(text)), fromCache: true };
  }

  const url = new URL(HEADER_FILE, baseUrl).href;
  const response = await fetch(url, { signal });
  if (!response.ok) {
    throw new ModelLoadError(`${HEADER_FILE}: HTTP ${response.status} ${response.statusText}`);
  }
  const text = await response.text();
  const header = parseHeader(JSON.parse(text));
  await store.write(HEADER_FILE, new TextEncoder().encode(text));
  return { header, fromCache: false };
}

export async function loadModel(
  device: GPUDevice,
  options: LoadModelOptions,
): Promise<LoadedModel> {
  const startedAt = performance.now();
  const store = await ModelStore.open(options.modelId);
  const { header } = await loadHeader(store, options.baseUrl, options.signal);

  // ---- download / cache -------------------------------------------------------------
  const downloadStart = performance.now();
  let networkBytes = 0;
  let cacheBytes = 0;
  let movedBytes = 0;

  const report = (phase: 'download' | 'upload', detail: string, fromCache: boolean): void => {
    options.onProgress?.({
      phase,
      loadedBytes: movedBytes,
      totalBytes: header.totalBytes,
      detail,
      fromCache,
    });
  };

  let allCached = true;
  for (const chunk of header.chunks) {
    options.signal?.throwIfAborted();
    const result = await ensureChunk(store, chunk, {
      baseUrl: options.baseUrl,
      signal: options.signal,
      verify: options.verifyChunks ?? true,
      onBytes: (delta, fromCache) => {
        movedBytes += delta;
        if (fromCache) cacheBytes += delta;
        else networkBytes += delta;
        report('download', chunk.name, fromCache);
      },
    });
    if (!result.fromCache) allCached = false;
  }
  const downloadMs = performance.now() - downloadStart;

  // ---- upload -----------------------------------------------------------------------
  const uploadStart = performance.now();
  const reader = new ChunkedByteReader(store, header.chunks, header.chunkBytes);

  const deviceLimit = Math.min(
    device.limits.maxStorageBufferBindingSize,
    device.limits.maxBufferSize,
  );
  const limitBytes = Math.min(options.shardThresholdBytes ?? deviceLimit, deviceLimit);
  const pieceBytes = options.uploadPieceBytes ?? DEFAULT_UPLOAD_PIECE;

  const registry = new WeightRegistry(header.aliases);
  let bufferCount = 0;
  let vramBytes = 0;
  movedBytes = 0;

  try {
    for (const meta of header.tensors) {
      options.signal?.throwIfAborted();
      const { rows, cols, bytesPerRow } = tensorGeometry(meta);
      const plan = planShards(rows, bytesPerRow, limitBytes);

      const shards: TensorShard[] = [];
      for (const [index, part] of plan.entries()) {
        const buffer = createStorageBuffer(device, part.byteLength, {
          label: plan.length === 1 ? meta.name : `${meta.name}#${index}`,
        });
        bufferCount += 1;
        vramBytes += buffer.size;

        // Stream the shard in bounded pieces so peak host memory stays at pieceBytes
        // rather than the size of the tensor (272 MB for the embedding matrix).
        const shardStart = meta.offset + part.rowStart * bytesPerRow;
        for (let done = 0; done < part.byteLength; ) {
          const take = Math.min(pieceBytes, part.byteLength - done);
          const bytes = await reader.read(shardStart + done, take);
          // writeBuffer requires a 4-byte multiple; f16 tensors can end on a 2-byte
          // boundary, and the buffer is already padded up by createStorageBuffer.
          const payload =
            bytes.byteLength % 4 === 0 ? bytes : padTo(bytes, bytes.byteLength + (4 - (bytes.byteLength % 4)));
          device.queue.writeBuffer(buffer, done, payload, 0, payload.byteLength);
          done += take;
          movedBytes += take;
        }

        shards.push({
          buffer,
          rowStart: part.rowStart,
          rowCount: part.rowCount,
          byteLength: part.byteLength,
        });
      }

      registry.add({
        name: meta.name,
        dtype: meta.dtype,
        shape: meta.shape,
        rows,
        cols,
        bytesPerRow,
        byteLength: meta.byteLength,
        shards,
        quant: meta.quant,
      });
      report('upload', meta.name, allCached);
    }

    await device.queue.onSubmittedWorkDone();
  } catch (err) {
    registry.destroy();
    throw err;
  }

  const uploadMs = performance.now() - uploadStart;

  return {
    header,
    config: header.config,
    weights: registry,
    store,
    stats: {
      downloadMs,
      uploadMs,
      totalMs: performance.now() - startedAt,
      networkBytes,
      cacheBytes,
      chunkCount: header.chunks.length,
      tensorCount: registry.count,
      bufferCount,
      vramBytes,
      servedFromCache: allCached,
    },
  };
}

function padTo(bytes: Uint8Array, length: number): Uint8Array {
  const out = new Uint8Array(length);
  out.set(bytes);
  return out;
}

export { ModelStore, ModelStoreError };
