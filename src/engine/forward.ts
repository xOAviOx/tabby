/**
 * Prefill and decode, both driven through one KV cache.
 *
 * The two paths differ only in how many positions they submit at once: prefill hands the
 * whole prompt to the kernels, decode hands them a single token at the cache's current
 * length. Everything else -- the projections, RoPE with its absolute position, the causal
 * bound, the cache append -- is the same code. That is deliberate: the M3 gate is that
 * incremental decoding reproduces full recomputation exactly, and two separate
 * implementations would make an agreement between them much weaker evidence.
 *
 * `runFull` is the M2 behaviour: reset the cache and prefill the entire sequence. The
 * golden tests use it, and the M3 gate compares it against prefill-then-decode.
 *
 * Everything is dispatched into a single command buffer per call and submitted once.
 * Captured activations are copied inside that same command buffer and read back once, so
 * bisecting 26 capture points against the goldens costs one synchronisation, not 26.
 *
 * Every matmul iterates the weight's shards, so the pass also works on a default-limit
 * device where the embedding and lm_head must split.
 */

import { BufferArena, readBuffer, readF32 } from './buffers.js';
import { withErrorScopes } from './device.js';
import { KvCache, type KvCacheOptions } from './kvcache.js';
import { Profiler, Recorder } from './profiler.js';
import {
  MATMUL_TILE_T,
  MAX_TOP_K,
  REDUCE_GROUPS,
  createKernels,
  groupsFor,
  pickRowsPerWorkgroup,
  uniforms,
  type KernelSet,
} from './kernels.js';
import { bindGroup, type PipelineCache } from './pipelines.js';
import { isQuantized } from './model.js';
import type { GpuTensor, LoadedModel, ModelConfig, WeightRegistry } from './model.js';

/** Narrow a result to its logits, failing loudly if the run sampled on the GPU instead. */
export function requireLogits(result: ForwardResult): Float32Array<ArrayBufferLike> {
  if (!result.logits) {
    throw new ForwardError(
      'this run requested topK, so the full logit vector was never read back',
    );
  }
  return result.logits;
}

export class ForwardError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ForwardError';
  }
}

export interface ForwardPassOptions {
  /** Context length. Sizes the KV cache and every scratch buffer. */
  maxSeqLen?: number;
  /** Workgroup size for the general kernels. */
  workgroupSize?: number;
  /** Workgroup size for the quantized kernels. Swept at M5. */
  quantWorkgroupSize?: number;
  /** Output rows per quantized-matvec workgroup. Swept at M5. */
  rowsPerWorkgroup?: number;
  /** Fuse gate_proj + up_proj + silu into one dispatch. Kept switchable to be measured. */
  fuseSwiglu?: boolean;
  /**
   * Choose rows-per-workgroup per matmul instead of one value everywhere. Measured at
   * 72 tok/s against 101 for the fixed value, so it is off by default; kept because the
   * negative result is worth being able to reproduce.
   */
  adaptiveRows?: boolean;
  /**
   * Use the tiled matmul for multi-position prefill. Default true. Exposed so the tiling
   * can be measured against the naive kernel rather than assumed to help, and so M5 can
   * sweep it.
   */
  tiledPrefill?: boolean;
}

export interface RunOptions {
  /** Capture the embedding output, every layer output and the final norm. */
  captureActivations?: boolean;
  /**
   * Compute the top-k candidates on the GPU and read back only those, instead of the
   * whole logit vector. At a 152k vocabulary the difference is ~330 bytes against
   * ~608 KB per token, which is the entire point of M4's readback rule.
   */
  topK?: number;
}

/** The k highest logits, plus what is needed to turn them into exact probabilities. */
export interface TopKResult {
  /** Largest logit over the full vocabulary. */
  maxLogit: number;
  /** sum(exp(x - maxLogit)) over the full vocabulary: the softmax denominator. */
  sumExp: number;
  /** Candidate ids, highest logit first. */
  ids: number[];
  /** Their raw logits, aligned with `ids`. */
  logits: number[];
}

export interface ForwardResult {
  /** Logits at the final position. Null when `topK` was requested instead. */
  logits: Float32Array | null;
  topK: TopKResult | null;
  /** Present when captureActivations was set: [embed, layer0..layerN-1, finalNorm]. */
  activations: Float32Array[] | null;
  nTokens: number;
  ms: number;
  /**
   * Wall time spent on the CPU building this step's commands, up to and including
   * `queue.submit`. Decode encodes ~400 dispatches per token and builds a uniform buffer
   * and a bind group for each, so this is a real share of the step and is invisible in
   * GPU timestamps. `ms - encodeMs` is roughly the wait on the GPU plus readback.
   */
  encodeMs: number;
  /** Bytes copied GPU -> CPU by this call. The M4 gate asserts on this. */
  readbackBytes: number;
}

const DEFAULT_MAX_SEQ_LEN = 512;

/** [maxLogit, sumExp, (value, index) * MAX_TOP_K] */
const SAMPLE_HEADER_FLOATS = 2;
const SAMPLE_OUT_FLOATS = SAMPLE_HEADER_FLOATS + MAX_TOP_K * 2;

type UniformFn = (data: ArrayBuffer, label: string) => GPUBuffer;

/** Where in the sequence a segment sits, and how much context it can see. */
interface Segment {
  /** Positions submitted now: the whole prompt in prefill, 1 in decode. */
  nNew: number;
  /** Absolute position of the first new token. */
  posStart: number;
  /** posStart + nNew. */
  totalLen: number;
}

/**
 * Unpack [maxLogit, sumExp, (value, index) * k]. The index is a u32 written through an
 * f32 array, so the same bytes are read twice with different views.
 */
function decodeSampleBlock(raw: ArrayBuffer, k: number): TopKResult {
  const asFloat = new Float32Array(raw);
  const asUint = new Uint32Array(raw);
  const ids: number[] = [];
  const logits: number[] = [];
  for (let i = 0; i < k; i++) {
    const slot = SAMPLE_HEADER_FLOATS + i * 2;
    ids.push(asUint[slot + 1]);
    logits.push(asFloat[slot]);
  }
  return { maxLogit: asFloat[0], sumExp: asFloat[1], ids, logits };
}

export class ForwardPass {
  private readonly device: GPUDevice;
  private readonly config: ModelConfig;
  private readonly weights: WeightRegistry;
  private readonly kernels: KernelSet;
  private readonly arena: BufferArena;
  private readonly maxSeqLen: number;
  private readonly tiledPrefill: boolean;
  private readonly fuseSwiglu: boolean;
  private readonly adaptiveRows: boolean;
  /** Per-dispatch-site uniform buffers, reused across steps. See pooledUniform. */
  private readonly uniformPool: GPUBuffer[] = [];
  private uniformCursor = 0;
  private profiler: Profiler | null = null;
  readonly cache: KvCache;

  private readonly ids: GPUBuffer;
  private readonly x: GPUBuffer;
  private readonly xNorm: GPUBuffer;
  private readonly q: GPUBuffer;
  private readonly k: GPUBuffer;
  private readonly v: GPUBuffer;
  private readonly scores: GPUBuffer;
  private readonly attnOut: GPUBuffer;
  private readonly proj: GPUBuffer;
  private readonly gate: GPUBuffer;
  private readonly up: GPUBuffer;
  private readonly lastHidden: GPUBuffer;
  private readonly logits: GPUBuffer;
  private readonly noBias: GPUBuffer;
  private readonly capture: GPUBuffer;
  private readonly logitsWork: GPUBuffer;
  private readonly partialsValue: GPUBuffer;
  private readonly partialsIndex: GPUBuffer;
  private readonly sampleOut: GPUBuffer;

  private constructor(
    device: GPUDevice,
    model: LoadedModel,
    kernels: KernelSet,
    options: KvCacheOptions & {
      tiledPrefill: boolean;
      fuseSwiglu: boolean;
      adaptiveRows: boolean;
    },
  ) {
    this.device = device;
    this.config = model.config;
    this.weights = model.weights;
    this.kernels = kernels;
    this.maxSeqLen = options.maxSeqLen;
    this.tiledPrefill = options.tiledPrefill;
    this.fuseSwiglu = options.fuseSwiglu;
    this.adaptiveRows = options.adaptiveRows;
    this.arena = new BufferArena(device);
    this.cache = new KvCache(device, model.config, options);

    const c = this.config;
    const n = this.maxSeqLen;
    const qDim = c.numAttentionHeads * c.headDim;
    const kvDim = c.numKeyValueHeads * c.headDim;
    const alloc = (elements: number, label: string): GPUBuffer =>
      this.arena.storage(elements * 4, { label });

    this.ids = alloc(n, 'fwd.ids');
    this.x = alloc(n * c.hiddenSize, 'fwd.x');
    this.xNorm = alloc(n * c.hiddenSize, 'fwd.xNorm');
    this.q = alloc(n * qDim, 'fwd.q');
    this.k = alloc(n * kvDim, 'fwd.k');
    this.v = alloc(n * kvDim, 'fwd.v');
    // [heads, nNew, totalLen], worst case nNew = totalLen = maxSeqLen.
    this.scores = alloc(c.numAttentionHeads * n * n, 'fwd.scores');
    this.attnOut = alloc(n * qDim, 'fwd.attnOut');
    this.proj = alloc(n * c.hiddenSize, 'fwd.proj');
    this.gate = alloc(n * c.intermediateSize, 'fwd.gate');
    this.up = alloc(n * c.intermediateSize, 'fwd.up');
    this.lastHidden = alloc(c.hiddenSize, 'fwd.lastHidden');
    this.logits = alloc(c.vocabSize, 'fwd.logits');

    // Bound wherever a matmul has no bias. WGSL requires the binding to exist even when
    // the shader never reads it.
    this.noBias = alloc(4, 'fwd.noBias');
    this.capture = alloc((c.numHiddenLayers + 2) * n * c.hiddenSize, 'fwd.capture');

    // Sampling scratch. `work` is a maskable copy of the logits, so extracting the top-k
    // never disturbs the originals the softmax statistics are computed from.
    this.logitsWork = alloc(c.vocabSize, 'fwd.logitsWork');
    this.partialsValue = alloc(REDUCE_GROUPS, 'fwd.partialsValue');
    this.partialsIndex = alloc(REDUCE_GROUPS, 'fwd.partialsIndex');
    this.sampleOut = alloc(SAMPLE_OUT_FLOATS, 'fwd.sampleOut');
  }

  static async create(
    device: GPUDevice,
    model: LoadedModel,
    cache: PipelineCache,
    options: ForwardPassOptions = {},
  ): Promise<ForwardPass> {
    const kernels = await createKernels(cache, {
      ...(options.workgroupSize === undefined ? {} : { workgroupSize: options.workgroupSize }),
      ...(options.quantWorkgroupSize === undefined
        ? {}
        : { quantWorkgroupSize: options.quantWorkgroupSize }),
      ...(options.rowsPerWorkgroup === undefined
        ? {}
        : { rowsPerWorkgroup: options.rowsPerWorkgroup }),
    });
    return new ForwardPass(device, model, kernels, {
      maxSeqLen: options.maxSeqLen ?? DEFAULT_MAX_SEQ_LEN,
      tiledPrefill: options.tiledPrefill ?? true,
      fuseSwiglu: options.fuseSwiglu ?? true,
      adaptiveRows: options.adaptiveRows ?? false,
    });
  }

  get maxSequenceLength(): number {
    return this.maxSeqLen;
  }

  /**
   * Turn per-kernel timing on. Profiling gives every dispatch its own compute pass so it
   * can carry a timestamp pair, which is measurably slower than the single-pass encoding
   * used normally -- so the numbers it reports are shares, not absolute throughput.
   */
  enableProfiling(enabled: boolean): boolean {
    if (!enabled) {
      this.profiler?.destroy();
      this.profiler = null;
      return false;
    }
    if (!Profiler.supported(this.device)) return false;
    this.profiler ??= new Profiler(this.device, true);
    return true;
  }

  /** Timings from the most recent run. Empty unless profiling is on. */
  async profile(): Promise<import('./profiler.js').ProfileReport> {
    if (!this.profiler) return { kernels: [], totalMs: 0, passCount: 0 };
    return this.profiler.collect();
  }

  /** Tokens currently in the KV cache. */
  get position(): number {
    return this.cache.length;
  }

  reset(): void {
    this.cache.reset();
  }

  destroy(): void {
    this.profiler?.destroy();
    this.cache.destroy();
    this.arena.destroy();
    for (const buffer of this.uniformPool) buffer.destroy();
    this.uniformPool.length = 0;
  }

  /** Process `ids` at the cache's current position, extending it. */
  async prefill(tokenIds: ArrayLike<number>, options: RunOptions = {}): Promise<ForwardResult> {
    const nNew = tokenIds.length;
    if (nNew === 0) throw new ForwardError('cannot run a forward pass on zero tokens');
    const posStart = this.cache.reserve(nNew);
    return this.runSegment(tokenIds, posStart, options);
  }

  /** Process a single token at the cache's current position. */
  async decode(tokenId: number, options: RunOptions = {}): Promise<ForwardResult> {
    const posStart = this.cache.reserve(1);
    return this.runSegment([tokenId], posStart, options);
  }

  /** Reset the cache and process the whole sequence. This is the M2 behaviour. */
  async runFull(tokenIds: ArrayLike<number>, options: RunOptions = {}): Promise<ForwardResult> {
    this.reset();
    return this.prefill(tokenIds, options);
  }

  private tensor(name: string): GpuTensor {
    return this.weights.get(name);
  }

  private async runSegment(
    tokenIds: ArrayLike<number>,
    posStart: number,
    options: RunOptions,
  ): Promise<ForwardResult> {
    const c = this.config;
    const capture = options.captureActivations ?? false;
    const topK = options.topK ?? 0;
    if (topK < 0 || topK > MAX_TOP_K) {
      throw new ForwardError(`topK must be between 0 and ${MAX_TOP_K}, got ${topK}`);
    }
    const nNew = tokenIds.length;
    const segment: Segment = { nNew, posStart, totalLen: posStart + nNew };
    const started = performance.now();
    let encodeMs = 0;

    const idArray = new Uint32Array(nNew);
    for (let i = 0; i < nNew; i++) {
      const id = tokenIds[i];
      if (!Number.isInteger(id) || id < 0 || id >= c.vocabSize) {
        throw new ForwardError(`token id ${id} at position ${i} is outside [0, ${c.vocabSize})`);
      }
      idArray[i] = id;
    }
    this.device.queue.writeBuffer(this.ids, 0, idArray);

    // Uniform buffers are allocated per call. Wasteful, but it keeps the dispatch code
    // readable and they are a few hundred bytes against ~1 GB of weights.
    this.uniformCursor = 0;
    const uniform: UniformFn = (data, label) => this.pooledUniform(data, label);

    try {
      await withErrorScopes(this.device, `forward(${nNew}@${posStart})`, () => {
        const encoder = this.device.createCommandEncoder({ label: 'forward' });
        const hiddenBytes = nNew * c.hiddenSize * 4;
        const slotBytes = this.maxSeqLen * c.hiddenSize * 4;

        this.profiler?.reset();
        const rec = new Recorder(encoder, this.device.limits, this.profiler);

        // A copy cannot be recorded inside a compute pass, so a capture interrupts the
        // recorder; it reopens lazily on the next dispatch. Still one command buffer.
        const captureInto = (slot: number, source: GPUBuffer): void => {
          if (!capture) return;
          rec.interrupt();
          encoder.copyBufferToBuffer(source, 0, this.capture, slot * slotBytes, hiddenBytes);
        };

        this.encodeEmbedding(rec, uniform, nNew);
        captureInto(0, this.x);

        for (let layer = 0; layer < c.numHiddenLayers; layer++) {
          this.encodeLayer(rec, uniform, segment, layer);
          captureInto(layer + 1, this.x);
        }

        this.encodeRmsNorm(
          rec,
          uniform,
          nNew,
          this.x,
          this.tensor('model.norm.weight'),
          this.xNorm,
          'final_norm',
        );
        captureInto(c.numHiddenLayers + 1, this.xNorm);

        // Only the last position needs logits, and at 152k vocab that is the difference
        // between one matmul row-block and nNew of them.
        rec.interrupt();
        encoder.copyBufferToBuffer(
          this.xNorm,
          (nNew - 1) * c.hiddenSize * 4,
          this.lastHidden,
          0,
          c.hiddenSize * 4,
        );
        this.encodeMatmul(
          rec,
          uniform,
          1,
          this.tensor('lm_head.weight'),
          null,
          this.lastHidden,
          this.logits,
          c.vocabSize,
          'lm_head',
        );

        if (topK > 0) this.encodeSampling(encoder, rec, uniform, topK);

        rec.finish();
        this.profiler?.resolve(encoder);
        this.device.queue.submit([encoder.finish()]);
        encodeMs = performance.now() - started;
      });

      let logits: Float32Array<ArrayBufferLike> | null = null;
      let topKResult: TopKResult | null = null;
      let readbackBytes = 0;

      if (topK > 0) {
        // The only thing crossing to the CPU is this block.
        const floats = SAMPLE_HEADER_FLOATS + topK * 2;
        const raw = await readBuffer(this.device, this.sampleOut, floats * 4);
        readbackBytes += floats * 4;
        topKResult = decodeSampleBlock(raw, topK);
      } else {
        logits = await readF32(this.device, this.logits, c.vocabSize);
        readbackBytes += c.vocabSize * 4;
      }

      let activations: Float32Array[] | null = null;
      if (capture) {
        const slots = c.numHiddenLayers + 2;
        const captureFloats = slots * this.maxSeqLen * c.hiddenSize;
        const all = await readF32(this.device, this.capture, captureFloats);
        readbackBytes += captureFloats * 4;
        activations = [];
        for (let slot = 0; slot < slots; slot++) {
          const base = slot * this.maxSeqLen * c.hiddenSize;
          activations.push(all.slice(base, base + nNew * c.hiddenSize));
        }
      }

      return {
        logits,
        topK: topKResult,
        activations,
        nTokens: nNew,
        ms: performance.now() - started,
        encodeMs,
        readbackBytes,
      };
    } finally {
      // Uniform buffers are pooled and outlive the call; see pooledUniform.
    }
  }

  /**
   * A uniform buffer for one dispatch site, reused across steps.
   *
   * Decode encodes ~400 dispatches per token and each needs its own small uniform block.
   * Allocating and destroying that many GPUBuffers every step was the largest single item
   * of CPU encode time, and encode is serial with the GPU -- the device is idle while it
   * happens. The dispatch sequence is deterministic for a given step shape, so site `i`
   * takes pool slot `i` on every step and is refilled with `writeBuffer`, whose writes are
   * ordered ahead of the submit that follows on the same queue.
   *
   * A slot is reallocated only when a step shape needs it larger than last time, which is
   * why the size check is `<` rather than `!==`.
   */
  private pooledUniform(data: ArrayBuffer, label: string): GPUBuffer {
    const size = Math.max(16, Math.ceil(data.byteLength / 16) * 16);
    const index = this.uniformCursor++;
    let buffer = this.uniformPool[index];
    if (!buffer || buffer.size < size) {
      buffer?.destroy();
      buffer = this.device.createBuffer({
        label: `uniform[${index}] ${label}`,
        size,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.uniformPool[index] = buffer;
    }
    this.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }

  // -------------------------------------------------------------------------------------
  // encoding helpers
  // -------------------------------------------------------------------------------------

  /**
   * Softmax statistics plus an exact top-k, entirely on the GPU.
   *
   * k rounds of (per-workgroup max, then reduce-and-mask) extract the k largest logits
   * in order. It is more dispatches than a histogram or partial sort, but each round is
   * a plain memory-bound pass over 608 KB that never leaves the device, and the result
   * is exact rather than approximate.
   */
  private encodeSampling(
    encoder: GPUCommandEncoder,
    rec: Recorder,
    uniform: UniformFn,
    topK: number,
  ): void {
    const c = this.config;
    const vocab = c.vocabSize;

    // Mask the copy, not the originals: the softmax denominator below needs them intact.
    rec.interrupt();
    encoder.copyBufferToBuffer(this.logits, 0, this.logitsWork, 0, vocab * 4);

    const reduce = (
      pipeline: GPUComputePipeline,
      dims: ArrayBuffer,
      buffers: GPUBuffer[],
      groups: number,
      label: string,
    ): void => {
      const group = bindGroup(
        this.device,
        pipeline,
        [uniform(dims, `${label}.dims`), ...buffers],
        `${label}.bind`,
      );
      rec.dispatch(pipeline, group, [groups], label);
    };

    // max over the full vocabulary, in two stages.
    reduce(
      this.kernels.reduceMax,
      uniforms.reduceMax(vocab, 0),
      [this.logits, this.partialsValue],
      REDUCE_GROUPS,
      'reduce_max.partial',
    );
    reduce(
      this.kernels.reduceMax,
      uniforms.reduceMax(REDUCE_GROUPS, 0),
      [this.partialsValue, this.sampleOut],
      1,
      'reduce_max.final',
    );

    // sum(exp(x - max)). Stage two binds `stats` to partialsValue purely to avoid
    // aliasing sampleOut as both readable and writable in one bind group; apply_exp is
    // off there, so the value is never read.
    reduce(
      this.kernels.reduceSumExp,
      uniforms.reduceSumExp(vocab, 0, true, 0),
      [this.logits, this.partialsValue, this.sampleOut],
      REDUCE_GROUPS,
      'reduce_sumexp.partial',
    );
    reduce(
      this.kernels.reduceSumExp,
      uniforms.reduceSumExp(REDUCE_GROUPS, 1, false, 0),
      [this.partialsValue, this.sampleOut, this.partialsValue],
      1,
      'reduce_sumexp.final',
    );

    for (let step = 0; step < topK; step++) {
      reduce(
        this.kernels.topkPartial,
        uniforms.topkPartial(vocab),
        [this.logitsWork, this.partialsValue, this.partialsIndex],
        REDUCE_GROUPS,
        'topk.partial',
      );
      reduce(
        this.kernels.topkSelect,
        uniforms.topkSelect(REDUCE_GROUPS, step, SAMPLE_HEADER_FLOATS),
        [this.partialsValue, this.partialsIndex, this.sampleOut, this.logitsWork],
        1,
        'topk.select',
      );
    }
  }

  private encodeEmbedding(rec: Recorder, uniform: UniformFn, nNew: number): void {
    const c = this.config;
    const table = this.tensor('model.embed_tokens.weight');
    const wg = this.kernels.workgroupSize;

    if (isQuantized(table)) {
      const quant = table.quant!;
      const scales = table.scales!;
      for (const shard of table.shards) {
        const dims = uniform(
          uniforms.embedGatherQuant(
            nNew,
            c.hiddenSize,
            quant.blockSize,
            shard.rowStart,
            shard.rowCount,
          ),
          'embed.dims',
        );
        const group = bindGroup(
          this.device,
          this.kernels.embedGatherQ4,
          [dims, shard.buffer, scales.shards[0].buffer, this.ids, this.x],
          'embed.bind',
        );
        rec.dispatch(this.kernels.embedGatherQ4, group, [groupsFor(c.hiddenSize, wg), nNew], 'embed_gather_q4');
      }
      return;
    }

    for (const shard of table.shards) {
      const dims = uniform(
        uniforms.embedGather(nNew, c.hiddenSize, shard.rowStart, shard.rowCount),
        'embed.dims',
      );
      const group = bindGroup(
        this.device,
        this.kernels.embedGather,
        [dims, shard.buffer, this.ids, this.x],
        'embed.bind',
      );
      rec.dispatch(this.kernels.embedGather, group, [groupsFor(c.hiddenSize, wg), nNew], 'embed_gather');
    }
  }

  private encodeRmsNorm(
    rec: Recorder,
    uniform: UniformFn,
    nNew: number,
    input: GPUBuffer,
    gain: GpuTensor,
    output: GPUBuffer,
    label: string,
  ): void {
    const c = this.config;
    if (gain.shards.length !== 1) {
      throw new ForwardError(`${label}: norm gain is sharded, which is not supported`);
    }
    const dims = uniform(uniforms.rmsNorm(nNew, c.hiddenSize, c.rmsNormEps), `${label}.dims`);
    const group = bindGroup(
      this.device,
      this.kernels.rmsNorm,
      [dims, input, gain.shards[0].buffer, output],
      `${label}.bind`,
    );
    // One workgroup per row; the kernel pins its own workgroup size.
    rec.dispatch(this.kernels.rmsNorm, group, [nNew], label);
  }

  private encodeMatmul(
    rec: Recorder,
    uniform: UniformFn,
    nNew: number,
    weight: GpuTensor,
    bias: GpuTensor | null,
    input: GPUBuffer,
    output: GPUBuffer,
    outStride: number,
    label: string,
  ): void {
    const inDim = weight.cols;
    const wg = this.kernels.workgroupSize;
    if (bias && bias.shards.length !== 1) {
      throw new ForwardError(`${label}: bias is sharded, which is not supported`);
    }
    const biasBuffer = bias ? bias.shards[0].buffer : this.noBias;

    if (isQuantized(weight)) {
      this.encodeQuantMatmul(rec, uniform, nNew, weight, biasBuffer, bias !== null, input, output, outStride, label);
      return;
    }

    // Prefill reuses each weight row across TILE_T positions; decode has only one
    // position, so the tiling would just add barriers.
    const tiled = this.tiledPrefill && nNew >= MATMUL_TILE_T;
    const pipeline = tiled ? this.kernels.matmulTiled : this.kernels.matmul;
    const rows = tiled ? groupsFor(nNew, MATMUL_TILE_T) : nNew;

    for (const shard of weight.shards) {
      const dims = uniform(
        uniforms.matmul(nNew, shard.rowCount, inDim, bias !== null, shard.rowStart, outStride),
        `${label}.dims`,
      );
      const group = bindGroup(
        this.device,
        pipeline,
        [dims, shard.buffer, input, biasBuffer, output],
        `${label}.bind`,
      );
      rec.dispatch(pipeline, group, [groupsFor(shard.rowCount, wg), rows], label);
    }
  }

  /**
   * Quantized matmul. One workgroup covers `rowsPerWorkgroup` output rows and reduces
   * across its lanes, which is the shape a bandwidth-bound decode wants: consecutive
   * lanes read consecutive packed words, so the loads coalesce.
   */
  private encodeQuantMatmul(
    rec: Recorder,
    uniform: UniformFn,
    nNew: number,
    weight: GpuTensor,
    biasBuffer: GPUBuffer,
    hasBias: boolean,
    input: GPUBuffer,
    output: GPUBuffer,
    outStride: number,
    label: string,
  ): void {
    const quant = weight.quant!;
    const scales = weight.scales;
    if (!scales) throw new ForwardError(`${label}: quantized tensor has no scales`);
    if (quant.bits !== 4) {
      throw new ForwardError(`${label}: only int4 has a kernel so far, got int${quant.bits}`);
    }

    for (const shard of weight.shards) {
      // Chosen per matmul: q/k/v/o and down_proj have ~900 output rows and would launch
      // only ~112 workgroups at the default, leaving most of the GPU idle.
      const rowsPerWg = this.adaptiveRows
        ? pickRowsPerWorkgroup(
            shard.rowCount,
            this.device.limits.maxComputeWorkgroupsPerDimension,
          )
        : this.kernels.rowsPerWorkgroup;
      const pipeline = this.kernels.matvecQ4Variants.get(rowsPerWg) ?? this.kernels.matvecQ4;
      const dims = uniform(
        uniforms.matvecQuant(
          nNew,
          shard.rowCount,
          weight.cols,
          quant.blockSize,
          hasBias,
          shard.rowStart,
          outStride,
        ),
        `${label}.dims`,
      );
      const group = bindGroup(
        this.device,
        pipeline,
        [dims, shard.buffer, scales.shards[0].buffer, input, biasBuffer, output],
        `${label}.bind`,
      );
      rec.dispatch(pipeline, group, [groupsFor(shard.rowCount, rowsPerWg), nNew], label);
    }
  }

  private encodeElementwise(
    rec: Recorder,
    uniform: UniformFn,
    pipeline: GPUComputePipeline,
    n: number,
    readOnly: GPUBuffer,
    readWrite: GPUBuffer,
    label: string,
  ): void {
    const dims = uniform(uniforms.elementwise(n), `${label}.dims`);
    const group = bindGroup(this.device, pipeline, [dims, readOnly, readWrite], `${label}.bind`);
    rec.dispatch(pipeline, group, [groupsFor(n, this.kernels.workgroupSize)], label);
  }

  private encodeKvWrite(
    rec: Recorder,
    uniform: UniformFn,
    source: GPUBuffer,
    destination: GPUBuffer,
    elements: number,
    dstStart: number,
    label: string,
  ): void {
    const dims = uniform(uniforms.kvWrite(elements, dstStart), `${label}.dims`);
    const group = bindGroup(
      this.device,
      this.kernels.kvWrite,
      [dims, source, destination],
      `${label}.bind`,
    );
    rec.dispatch(this.kernels.kvWrite, group, [groupsFor(elements, this.kernels.workgroupSize)], label);
  }

  private encodeLayer(
    rec: Recorder,
    uniform: UniformFn,
    segment: Segment,
    layer: number,
  ): void {
    const c = this.config;
    const wg = this.kernels.workgroupSize;
    const p = `model.layers.${layer}.`;
    const { nNew, posStart, totalLen } = segment;
    const qDim = c.numAttentionHeads * c.headDim;
    const kvDim = c.numKeyValueHeads * c.headDim;
    const kvOffset = this.cache.layerOffset(layer);

    // ---- attention block ----------------------------------------------------------
    this.encodeRmsNorm(
      rec,
      uniform,
      nNew,
      this.x,
      this.tensor(p + 'input_layernorm.weight'),
      this.xNorm,
      `L${layer}.input_norm`,
    );

    for (const [name, out, dim] of [
      ['q_proj', this.q, qDim],
      ['k_proj', this.k, kvDim],
      ['v_proj', this.v, kvDim],
    ] as Array<[string, GPUBuffer, number]>) {
      const biasName = p + `self_attn.${name}.bias`;
      this.encodeMatmul(
        rec,
        uniform,
        nNew,
        this.tensor(p + `self_attn.${name}.weight`),
        this.weights.has(biasName) ? this.tensor(biasName) : null,
        this.xNorm,
        out,
        dim,
        `L${layer}.${name}`,
      );
    }

    // RoPE is applied to Q and K only; V carries no position information. `posStart` is
    // what makes decode agree with full recomputation -- token n must be rotated by n
    // whether it arrived alone or as part of a batch.
    for (const [buffer, heads, label] of [
      [this.q, c.numAttentionHeads, 'q'],
      [this.k, c.numKeyValueHeads, 'k'],
    ] as Array<[GPUBuffer, number, string]>) {
      const dims = uniform(
        uniforms.rope(nNew, heads, c.headDim, posStart, c.ropeTheta),
        `L${layer}.rope_${label}.dims`,
      );
      const group = bindGroup(
        this.device,
        this.kernels.rope,
        [dims, buffer],
        `L${layer}.rope_${label}.bind`,
      );
      rec.dispatch(this.kernels.rope, group, [groupsFor((heads * c.headDim) / 2, wg), nNew], `L${layer}.rope_${label}`);
    }

    // Append this segment's K and V before attending, so the new positions are visible
    // to their own queries.
    this.encodeKvWrite(
      rec,
      uniform,
      this.k,
      this.cache.keys,
      nNew * kvDim,
      this.cache.offsetOf(layer, posStart),
      `L${layer}.k_cache`,
    );
    this.encodeKvWrite(
      rec,
      uniform,
      this.v,
      this.cache.values,
      nNew * kvDim,
      this.cache.offsetOf(layer, posStart),
      `L${layer}.v_cache`,
    );

    {
      const dims = uniform(
        uniforms.attnScores(
          nNew,
          totalLen,
          c.numAttentionHeads,
          c.numKeyValueHeads,
          c.headDim,
          posStart,
          kvOffset,
          1 / Math.sqrt(c.headDim),
        ),
        `L${layer}.scores.dims`,
      );
      const group = bindGroup(
        this.device,
        this.kernels.attnScores,
        [dims, this.q, this.cache.keys, this.scores],
        `L${layer}.scores.bind`,
      );
      rec.dispatch(this.kernels.attnScores, group, [groupsFor(totalLen, wg), nNew, c.numAttentionHeads], `L${layer}.attn_scores`);
    }

    {
      const rows = c.numAttentionHeads * nNew;
      const dims = uniform(
        uniforms.softmaxRows(rows, totalLen, nNew, posStart),
        `L${layer}.softmax.dims`,
      );
      const group = bindGroup(
        this.device,
        this.kernels.softmaxRows,
        [dims, this.scores],
        `L${layer}.softmax.bind`,
      );
      rec.dispatch(this.kernels.softmaxRows, group, [groupsFor(rows, wg)], `L${layer}.softmax`);
    }

    {
      const dims = uniform(
        uniforms.attnOutput(
          nNew,
          totalLen,
          c.numAttentionHeads,
          c.numKeyValueHeads,
          c.headDim,
          posStart,
          kvOffset,
        ),
        `L${layer}.attnout.dims`,
      );
      const group = bindGroup(
        this.device,
        this.kernels.attnOutput,
        [dims, this.scores, this.cache.values, this.attnOut],
        `L${layer}.attnout.bind`,
      );
      rec.dispatch(this.kernels.attnOutput, group, [groupsFor(qDim, wg), nNew], `L${layer}.attn_output`);
    }

    this.encodeMatmul(
      rec,
      uniform,
      nNew,
      this.tensor(p + 'self_attn.o_proj.weight'),
      null,
      this.attnOut,
      this.proj,
      c.hiddenSize,
      `L${layer}.o_proj`,
    );
    this.encodeElementwise(
      rec,
      uniform,
      this.kernels.residualAdd,
      nNew * c.hiddenSize,
      this.proj,
      this.x,
      `L${layer}.attn_residual`,
    );

    // ---- MLP block ----------------------------------------------------------------
    this.encodeRmsNorm(
      rec,
      uniform,
      nNew,
      this.x,
      this.tensor(p + 'post_attention_layernorm.weight'),
      this.xNorm,
      `L${layer}.post_norm`,
    );
    const gateWeight = this.tensor(p + 'mlp.gate_proj.weight');
    const upWeight = this.tensor(p + 'mlp.up_proj.weight');

    if (this.fuseSwiglu && isQuantized(gateWeight) && isQuantized(upWeight)) {
      // Fused: both projections read the same activations and feed the same SiLU, so
      // running them separately would move x twice and round-trip `up` through memory.
      const dims = uniform(
        uniforms.swigluQuant(nNew, c.intermediateSize, c.hiddenSize, gateWeight.quant!.blockSize),
        `L${layer}.swiglu.dims`,
      );
      const group = bindGroup(
        this.device,
        this.kernels.swigluQ4,
        [
          dims,
          gateWeight.shards[0].buffer,
          gateWeight.scales!.shards[0].buffer,
          upWeight.shards[0].buffer,
          upWeight.scales!.shards[0].buffer,
          this.xNorm,
          this.gate,
        ],
        `L${layer}.swiglu.bind`,
      );
      rec.dispatch(
        this.kernels.swigluQ4,
        group,
        [groupsFor(c.intermediateSize, this.kernels.rowsPerWorkgroup), nNew],
        `L${layer}.swiglu`,
      );
    } else {
      this.encodeMatmul(
        rec,
        uniform,
        nNew,
        gateWeight,
        null,
        this.xNorm,
        this.gate,
        c.intermediateSize,
        `L${layer}.gate_proj`,
      );
      this.encodeMatmul(
        rec,
        uniform,
        nNew,
        upWeight,
        null,
        this.xNorm,
        this.up,
        c.intermediateSize,
        `L${layer}.up_proj`,
      );
      this.encodeElementwise(
        rec,
        uniform,
        this.kernels.siluMul,
        nNew * c.intermediateSize,
        this.up,
        this.gate,
        `L${layer}.silu_mul`,
      );
    }

    this.encodeMatmul(
      rec,
      uniform,
      nNew,
      this.tensor(p + 'mlp.down_proj.weight'),
      null,
      this.gate,
      this.proj,
      c.hiddenSize,
      `L${layer}.down_proj`,
    );
    this.encodeElementwise(
      rec,
      uniform,
      this.kernels.residualAdd,
      nNew * c.hiddenSize,
      this.proj,
      this.x,
      `L${layer}.mlp_residual`,
    );
  }
}
