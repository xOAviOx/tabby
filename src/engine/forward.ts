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

import { BufferArena, readF32 } from './buffers.js';
import { withErrorScopes } from './device.js';
import { KvCache, type KvCacheOptions } from './kvcache.js';
import {
  MATMUL_TILE_T,
  createKernels,
  dispatch,
  groupsFor,
  uniforms,
  type KernelSet,
} from './kernels.js';
import { bindGroup, type PipelineCache } from './pipelines.js';
import type { GpuTensor, LoadedModel, ModelConfig, WeightRegistry } from './model.js';

export class ForwardError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ForwardError';
  }
}

export interface ForwardPassOptions {
  /** Context length. Sizes the KV cache and every scratch buffer. */
  maxSeqLen?: number;
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
}

export interface ForwardResult {
  /** Logits at the final position of this segment. */
  logits: Float32Array;
  /** Present when captureActivations was set: [embed, layer0..layerN-1, finalNorm]. */
  activations: Float32Array[] | null;
  nTokens: number;
  ms: number;
}

const DEFAULT_MAX_SEQ_LEN = 512;

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

export class ForwardPass {
  private readonly device: GPUDevice;
  private readonly config: ModelConfig;
  private readonly weights: WeightRegistry;
  private readonly kernels: KernelSet;
  private readonly arena: BufferArena;
  private readonly maxSeqLen: number;
  private readonly tiledPrefill: boolean;
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

  private constructor(
    device: GPUDevice,
    model: LoadedModel,
    kernels: KernelSet,
    options: KvCacheOptions & { tiledPrefill: boolean },
  ) {
    this.device = device;
    this.config = model.config;
    this.weights = model.weights;
    this.kernels = kernels;
    this.maxSeqLen = options.maxSeqLen;
    this.tiledPrefill = options.tiledPrefill;
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
  }

  static async create(
    device: GPUDevice,
    model: LoadedModel,
    cache: PipelineCache,
    options: ForwardPassOptions = {},
  ): Promise<ForwardPass> {
    const kernels = await createKernels(cache);
    return new ForwardPass(device, model, kernels, {
      maxSeqLen: options.maxSeqLen ?? DEFAULT_MAX_SEQ_LEN,
      tiledPrefill: options.tiledPrefill ?? true,
    });
  }

  get maxSequenceLength(): number {
    return this.maxSeqLen;
  }

  /** Tokens currently in the KV cache. */
  get position(): number {
    return this.cache.length;
  }

  reset(): void {
    this.cache.reset();
  }

  destroy(): void {
    this.cache.destroy();
    this.arena.destroy();
  }

  /** Process `ids` at the cache's current position, extending it. */
  async prefill(tokenIds: ArrayLike<number>, options: RunOptions = {}): Promise<ForwardResult> {
    const nNew = tokenIds.length;
    if (nNew === 0) throw new ForwardError('cannot run a forward pass on zero tokens');
    const posStart = this.cache.reserve(nNew);
    return this.runSegment(tokenIds, posStart, options.captureActivations ?? false);
  }

  /** Process a single token at the cache's current position. */
  async decode(tokenId: number): Promise<Float32Array> {
    const posStart = this.cache.reserve(1);
    const result = await this.runSegment([tokenId], posStart, false);
    return result.logits;
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
    capture: boolean,
  ): Promise<ForwardResult> {
    const c = this.config;
    const nNew = tokenIds.length;
    const segment: Segment = { nNew, posStart, totalLen: posStart + nNew };
    const started = performance.now();

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
    const perCall = new BufferArena(this.device);
    const uniform: UniformFn = (data, label) => perCall.uniform(new Uint8Array(data), label);

    try {
      await withErrorScopes(this.device, `forward(${nNew}@${posStart})`, () => {
        const encoder = this.device.createCommandEncoder({ label: 'forward' });
        const hiddenBytes = nNew * c.hiddenSize * 4;
        const slotBytes = this.maxSeqLen * c.hiddenSize * 4;

        // A copy cannot be recorded inside a compute pass, so each capture closes the
        // pass and opens a new one. It is still one command buffer and one submit.
        let pass = encoder.beginComputePass({ label: 'forward' });
        const captureInto = (slot: number, source: GPUBuffer): void => {
          if (!capture) return;
          pass.end();
          encoder.copyBufferToBuffer(source, 0, this.capture, slot * slotBytes, hiddenBytes);
          pass = encoder.beginComputePass({ label: 'forward' });
        };

        this.encodeEmbedding(pass, uniform, nNew);
        captureInto(0, this.x);

        for (let layer = 0; layer < c.numHiddenLayers; layer++) {
          this.encodeLayer(pass, uniform, segment, layer);
          captureInto(layer + 1, this.x);
        }

        this.encodeRmsNorm(
          pass,
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
        pass.end();
        encoder.copyBufferToBuffer(
          this.xNorm,
          (nNew - 1) * c.hiddenSize * 4,
          this.lastHidden,
          0,
          c.hiddenSize * 4,
        );
        const headPass = encoder.beginComputePass({ label: 'lm_head' });
        this.encodeMatmul(
          headPass,
          uniform,
          1,
          this.tensor('lm_head.weight'),
          null,
          this.lastHidden,
          this.logits,
          c.vocabSize,
          'lm_head',
        );
        headPass.end();

        this.device.queue.submit([encoder.finish()]);
      });

      const logits = await readF32(this.device, this.logits, c.vocabSize);

      let activations: Float32Array[] | null = null;
      if (capture) {
        const slots = c.numHiddenLayers + 2;
        const all = await readF32(this.device, this.capture, slots * this.maxSeqLen * c.hiddenSize);
        activations = [];
        for (let slot = 0; slot < slots; slot++) {
          const base = slot * this.maxSeqLen * c.hiddenSize;
          activations.push(all.slice(base, base + nNew * c.hiddenSize));
        }
      }

      return { logits, activations, nTokens: nNew, ms: performance.now() - started };
    } finally {
      perCall.destroy();
    }
  }

  // -------------------------------------------------------------------------------------
  // encoding helpers
  // -------------------------------------------------------------------------------------

  private encodeEmbedding(pass: GPUComputePassEncoder, uniform: UniformFn, nNew: number): void {
    const c = this.config;
    const table = this.tensor('model.embed_tokens.weight');
    const wg = this.kernels.workgroupSize;

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
      dispatch(
        pass,
        this.kernels.embedGather,
        group,
        this.device.limits,
        [groupsFor(c.hiddenSize, wg), nNew],
        'embed_gather',
      );
    }
  }

  private encodeRmsNorm(
    pass: GPUComputePassEncoder,
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
    dispatch(
      pass,
      this.kernels.rmsNorm,
      group,
      this.device.limits,
      [groupsFor(nNew, this.kernels.workgroupSize)],
      label,
    );
  }

  private encodeMatmul(
    pass: GPUComputePassEncoder,
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
      dispatch(
        pass,
        pipeline,
        group,
        this.device.limits,
        [groupsFor(shard.rowCount, wg), rows],
        label,
      );
    }
  }

  private encodeElementwise(
    pass: GPUComputePassEncoder,
    uniform: UniformFn,
    pipeline: GPUComputePipeline,
    n: number,
    readOnly: GPUBuffer,
    readWrite: GPUBuffer,
    label: string,
  ): void {
    const dims = uniform(uniforms.elementwise(n), `${label}.dims`);
    const group = bindGroup(this.device, pipeline, [dims, readOnly, readWrite], `${label}.bind`);
    dispatch(
      pass,
      pipeline,
      group,
      this.device.limits,
      [groupsFor(n, this.kernels.workgroupSize)],
      label,
    );
  }

  private encodeKvWrite(
    pass: GPUComputePassEncoder,
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
    dispatch(
      pass,
      this.kernels.kvWrite,
      group,
      this.device.limits,
      [groupsFor(elements, this.kernels.workgroupSize)],
      label,
    );
  }

  private encodeLayer(
    pass: GPUComputePassEncoder,
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
      pass,
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
        pass,
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
      dispatch(
        pass,
        this.kernels.rope,
        group,
        this.device.limits,
        [groupsFor((heads * c.headDim) / 2, wg), nNew],
        `L${layer}.rope_${label}`,
      );
    }

    // Append this segment's K and V before attending, so the new positions are visible
    // to their own queries.
    this.encodeKvWrite(
      pass,
      uniform,
      this.k,
      this.cache.keys,
      nNew * kvDim,
      this.cache.offsetOf(layer, posStart),
      `L${layer}.k_cache`,
    );
    this.encodeKvWrite(
      pass,
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
      dispatch(
        pass,
        this.kernels.attnScores,
        group,
        this.device.limits,
        [groupsFor(totalLen, wg), nNew, c.numAttentionHeads],
        `L${layer}.attn_scores`,
      );
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
      dispatch(
        pass,
        this.kernels.softmaxRows,
        group,
        this.device.limits,
        [groupsFor(rows, wg)],
        `L${layer}.softmax`,
      );
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
      dispatch(
        pass,
        this.kernels.attnOutput,
        group,
        this.device.limits,
        [groupsFor(qDim, wg), nNew],
        `L${layer}.attn_output`,
      );
    }

    this.encodeMatmul(
      pass,
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
      pass,
      uniform,
      this.kernels.residualAdd,
      nNew * c.hiddenSize,
      this.proj,
      this.x,
      `L${layer}.attn_residual`,
    );

    // ---- MLP block ----------------------------------------------------------------
    this.encodeRmsNorm(
      pass,
      uniform,
      nNew,
      this.x,
      this.tensor(p + 'post_attention_layernorm.weight'),
      this.xNorm,
      `L${layer}.post_norm`,
    );
    this.encodeMatmul(
      pass,
      uniform,
      nNew,
      this.tensor(p + 'mlp.gate_proj.weight'),
      null,
      this.xNorm,
      this.gate,
      c.intermediateSize,
      `L${layer}.gate_proj`,
    );
    this.encodeMatmul(
      pass,
      uniform,
      nNew,
      this.tensor(p + 'mlp.up_proj.weight'),
      null,
      this.xNorm,
      this.up,
      c.intermediateSize,
      `L${layer}.up_proj`,
    );
    this.encodeElementwise(
      pass,
      uniform,
      this.kernels.siluMul,
      nNew * c.intermediateSize,
      this.up,
      this.gate,
      `L${layer}.silu_mul`,
    );
    this.encodeMatmul(
      pass,
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
      pass,
      uniform,
      this.kernels.residualAdd,
      nNew * c.hiddenSize,
      this.proj,
      this.x,
      `L${layer}.mlp_residual`,
    );
  }
}
