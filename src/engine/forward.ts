/**
 * M2 forward pass: unoptimised, correct, no KV cache.
 *
 * The whole prompt is recomputed on every call. That is wasteful by design -- M3 adds
 * the cache and the prefill/decode split, and its gate is "byte-identical output to this
 * path", so this file is the thing that has to be obviously right rather than fast.
 *
 * Everything is dispatched into a single command buffer per call and submitted once.
 * Per-layer activations, when captured, are copied into a capture buffer inside that
 * same command buffer and read back once at the end, so bisecting against the PyTorch
 * goldens costs one synchronisation rather than one per layer.
 *
 * Every matmul iterates over the weight's shards. On this machine nothing shards, but
 * on a default-limit device the embedding and lm_head must, and a forward pass that
 * only works on generous adapters would be a bug waiting for M6.
 */

import { BufferArena, readF32 } from './buffers.js';
import { withErrorScopes } from './device.js';
import {
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

export interface ForwardOptions {
  /** Longest sequence this instance can process. Sizes every scratch buffer. */
  maxTokens?: number;
  /** Capture the embedding output, every layer output and the final norm. */
  captureActivations?: boolean;
}

export interface ForwardResult {
  /** Logits at the final position. */
  logits: Float32Array;
  /** Present when captureActivations was set: [embed, layer0..layerN-1, finalNorm]. */
  activations: Float32Array[] | null;
  nTokens: number;
  ms: number;
}

const DEFAULT_MAX_TOKENS = 512;

/** A scratch buffer plus the element count it is sized for. */
interface Scratch {
  buffer: GPUBuffer;
  elements: number;
}

export class ForwardPass {
  private readonly device: GPUDevice;
  private readonly config: ModelConfig;
  private readonly weights: WeightRegistry;
  private readonly kernels: KernelSet;
  private readonly arena: BufferArena;
  private readonly maxTokens: number;

  private readonly ids: GPUBuffer;
  private readonly x: Scratch;
  private readonly xNorm: Scratch;
  private readonly q: Scratch;
  private readonly k: Scratch;
  private readonly v: Scratch;
  private readonly scores: Scratch;
  private readonly attnOut: Scratch;
  private readonly proj: Scratch;
  private readonly gate: Scratch;
  private readonly up: Scratch;
  private readonly lastHidden: GPUBuffer;
  private readonly logits: GPUBuffer;
  private readonly noBias: GPUBuffer;
  private readonly capture: GPUBuffer;

  private constructor(
    device: GPUDevice,
    model: LoadedModel,
    kernels: KernelSet,
    maxTokens: number,
  ) {
    this.device = device;
    this.config = model.config;
    this.weights = model.weights;
    this.kernels = kernels;
    this.maxTokens = maxTokens;
    this.arena = new BufferArena(device);

    const c = this.config;
    const qDim = c.numAttentionHeads * c.headDim;
    const kvDim = c.numKeyValueHeads * c.headDim;

    const scratch = (elements: number, label: string): Scratch => ({
      buffer: this.arena.storage(elements * 4, { label }),
      elements,
    });

    this.ids = this.arena.storage(maxTokens * 4, { label: 'fwd.ids' });
    this.x = scratch(maxTokens * c.hiddenSize, 'fwd.x');
    this.xNorm = scratch(maxTokens * c.hiddenSize, 'fwd.xNorm');
    this.q = scratch(maxTokens * qDim, 'fwd.q');
    this.k = scratch(maxTokens * kvDim, 'fwd.k');
    this.v = scratch(maxTokens * kvDim, 'fwd.v');
    this.scores = scratch(c.numAttentionHeads * maxTokens * maxTokens, 'fwd.scores');
    this.attnOut = scratch(maxTokens * qDim, 'fwd.attnOut');
    this.proj = scratch(maxTokens * c.hiddenSize, 'fwd.proj');
    this.gate = scratch(maxTokens * c.intermediateSize, 'fwd.gate');
    this.up = scratch(maxTokens * c.intermediateSize, 'fwd.up');
    this.lastHidden = this.arena.storage(c.hiddenSize * 4, { label: 'fwd.lastHidden' });
    this.logits = this.arena.storage(c.vocabSize * 4, { label: 'fwd.logits' });

    // Bound wherever a matmul has no bias. WGSL requires the binding to exist even when
    // the shader never reads it.
    this.noBias = this.arena.storage(16, { label: 'fwd.noBias' });

    // embed + one per layer + final norm.
    const captureSlots = c.numHiddenLayers + 2;
    this.capture = this.arena.storage(captureSlots * maxTokens * c.hiddenSize * 4, {
      label: 'fwd.capture',
    });
  }

  static async create(
    device: GPUDevice,
    model: LoadedModel,
    cache: PipelineCache,
    options: ForwardOptions = {},
  ): Promise<ForwardPass> {
    const kernels = await createKernels(cache);
    const maxTokens = options.maxTokens ?? DEFAULT_MAX_TOKENS;
    return new ForwardPass(device, model, kernels, maxTokens);
  }

  get maxSequenceLength(): number {
    return this.maxTokens;
  }

  destroy(): void {
    this.arena.destroy();
  }

  private tensor(name: string): GpuTensor {
    return this.weights.get(name);
  }

  async run(tokenIds: ArrayLike<number>, options: ForwardOptions = {}): Promise<ForwardResult> {
    const nTokens = tokenIds.length;
    if (nTokens === 0) throw new ForwardError('cannot run a forward pass on zero tokens');
    if (nTokens > this.maxTokens) {
      throw new ForwardError(
        `sequence of ${nTokens} tokens exceeds maxTokens ${this.maxTokens}; ` +
          'construct ForwardPass with a larger maxTokens',
      );
    }

    const c = this.config;
    const capture = options.captureActivations ?? false;
    const started = performance.now();

    const idArray = new Uint32Array(nTokens);
    for (let i = 0; i < nTokens; i++) {
      const id = tokenIds[i];
      if (!Number.isInteger(id) || id < 0 || id >= c.vocabSize) {
        throw new ForwardError(`token id ${id} at position ${i} is outside [0, ${c.vocabSize})`);
      }
      idArray[i] = id;
    }
    this.device.queue.writeBuffer(this.ids, 0, idArray);

    // Uniform buffers are allocated per call. Wasteful, but M2 is correctness-only and
    // it keeps the dispatch code readable; M3 hoists them into a persistent block.
    const perCall = new BufferArena(this.device);
    const uniform = (data: ArrayBuffer, label: string): GPUBuffer =>
      perCall.uniform(new Uint8Array(data), label);

    try {
      await withErrorScopes(this.device, `forward(${nTokens} tokens)`, () => {
        const encoder = this.device.createCommandEncoder({ label: 'forward' });
        const hiddenBytes = nTokens * c.hiddenSize * 4;
        const slotBytes = this.maxTokens * c.hiddenSize * 4;

        // A copy cannot be recorded inside a compute pass, so each capture closes the
        // pass and opens a new one. It is still one command buffer and one submit.
        let pass = encoder.beginComputePass({ label: 'forward' });
        const captureInto = (slot: number, source: GPUBuffer): void => {
          if (!capture) return;
          pass.end();
          encoder.copyBufferToBuffer(source, 0, this.capture, slot * slotBytes, hiddenBytes);
          pass = encoder.beginComputePass({ label: 'forward' });
        };

        // Capture points mirror the forward hooks in tools/golden.py exactly:
        // embedding output, each layer output, then the final norm.
        this.encodeEmbedding(pass, uniform, nTokens);
        captureInto(0, this.x.buffer);

        for (let layer = 0; layer < c.numHiddenLayers; layer++) {
          this.encodeLayer(pass, uniform, nTokens, layer);
          captureInto(layer + 1, this.x.buffer);
        }

        this.encodeRmsNorm(
          pass,
          uniform,
          nTokens,
          this.x.buffer,
          this.tensor('model.norm.weight'),
          this.xNorm.buffer,
          'final_norm',
        );
        captureInto(c.numHiddenLayers + 1, this.xNorm.buffer);

        // Only the last position needs logits, and at 152k vocab that is the difference
        // between one matmul row-block and nTokens of them.
        pass.end();
        encoder.copyBufferToBuffer(
          this.xNorm.buffer,
          (nTokens - 1) * c.hiddenSize * 4,
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
        const all = await readF32(
          this.device,
          this.capture,
          slots * this.maxTokens * c.hiddenSize,
        );
        activations = [];
        for (let slot = 0; slot < slots; slot++) {
          const base = slot * this.maxTokens * c.hiddenSize;
          activations.push(all.slice(base, base + nTokens * c.hiddenSize));
        }
      }

      return { logits, activations, nTokens, ms: performance.now() - started };
    } finally {
      perCall.destroy();
    }
  }

  // -------------------------------------------------------------------------------------
  // encoding helpers
  // -------------------------------------------------------------------------------------

  private encodeEmbedding(
    pass: GPUComputePassEncoder,
    uniform: (data: ArrayBuffer, label: string) => GPUBuffer,
    nTokens: number,
  ): void {
    const c = this.config;
    const table = this.tensor('model.embed_tokens.weight');
    const wg = this.kernels.workgroupSize;

    for (const shard of table.shards) {
      const dims = uniform(
        uniforms.embedGather(nTokens, c.hiddenSize, shard.rowStart, shard.rowCount),
        'embed.dims',
      );
      const group = bindGroup(
        this.device,
        this.kernels.embedGather,
        [dims, shard.buffer, this.ids, this.x.buffer],
        'embed.bind',
      );
      dispatch(
        pass,
        this.kernels.embedGather,
        group,
        this.device.limits,
        [groupsFor(c.hiddenSize, wg), nTokens],
        'embed_gather',
      );
    }
  }

  private encodeRmsNorm(
    pass: GPUComputePassEncoder,
    uniform: (data: ArrayBuffer, label: string) => GPUBuffer,
    nTokens: number,
    input: GPUBuffer,
    gain: GpuTensor,
    output: GPUBuffer,
    label: string,
  ): void {
    const c = this.config;
    if (gain.shards.length !== 1) {
      throw new ForwardError(`${label}: norm gain is sharded, which is not supported`);
    }
    const dims = uniform(
      uniforms.rmsNorm(nTokens, c.hiddenSize, c.rmsNormEps),
      `${label}.dims`,
    );
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
      [groupsFor(nTokens, this.kernels.workgroupSize)],
      label,
    );
  }

  private encodeMatmul(
    pass: GPUComputePassEncoder,
    uniform: (data: ArrayBuffer, label: string) => GPUBuffer,
    nTokens: number,
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

    for (const shard of weight.shards) {
      const dims = uniform(
        uniforms.matmul(nTokens, shard.rowCount, inDim, bias !== null, shard.rowStart, outStride),
        `${label}.dims`,
      );
      const group = bindGroup(
        this.device,
        this.kernels.matmul,
        [dims, shard.buffer, input, biasBuffer, output],
        `${label}.bind`,
      );
      dispatch(
        pass,
        this.kernels.matmul,
        group,
        this.device.limits,
        [groupsFor(shard.rowCount, wg), nTokens],
        label,
      );
    }
  }

  private encodeElementwise(
    pass: GPUComputePassEncoder,
    uniform: (data: ArrayBuffer, label: string) => GPUBuffer,
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

  private encodeLayer(
    pass: GPUComputePassEncoder,
    uniform: (data: ArrayBuffer, label: string) => GPUBuffer,
    nTokens: number,
    layer: number,
  ): void {
    const c = this.config;
    const wg = this.kernels.workgroupSize;
    const p = `model.layers.${layer}.`;
    const qDim = c.numAttentionHeads * c.headDim;
    const kvDim = c.numKeyValueHeads * c.headDim;

    // ---- attention block ----------------------------------------------------------
    this.encodeRmsNorm(
      pass,
      uniform,
      nTokens,
      this.x.buffer,
      this.tensor(p + 'input_layernorm.weight'),
      this.xNorm.buffer,
      `L${layer}.input_norm`,
    );

    const projections: Array<[string, GPUBuffer, number]> = [
      ['q_proj', this.q.buffer, qDim],
      ['k_proj', this.k.buffer, kvDim],
      ['v_proj', this.v.buffer, kvDim],
    ];
    for (const [name, out, dim] of projections) {
      const biasName = p + `self_attn.${name}.bias`;
      this.encodeMatmul(
        pass,
        uniform,
        nTokens,
        this.tensor(p + `self_attn.${name}.weight`),
        this.weights.has(biasName) ? this.tensor(biasName) : null,
        this.xNorm.buffer,
        out,
        dim,
        `L${layer}.${name}`,
      );
    }

    // RoPE is applied to Q and K only; V carries no position information.
    for (const [buffer, heads, label] of [
      [this.q.buffer, c.numAttentionHeads, 'q'],
      [this.k.buffer, c.numKeyValueHeads, 'k'],
    ] as Array<[GPUBuffer, number, string]>) {
      const dims = uniform(
        uniforms.rope(nTokens, heads, c.headDim, 0, c.ropeTheta),
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
        [groupsFor((heads * c.headDim) / 2, wg), nTokens],
        `L${layer}.rope_${label}`,
      );
    }

    {
      const scale = 1 / Math.sqrt(c.headDim);
      const dims = uniform(
        uniforms.attnScores(
          nTokens,
          c.numAttentionHeads,
          c.numKeyValueHeads,
          c.headDim,
          scale,
        ),
        `L${layer}.scores.dims`,
      );
      const group = bindGroup(
        this.device,
        this.kernels.attnScores,
        [dims, this.q.buffer, this.k.buffer, this.scores.buffer],
        `L${layer}.scores.bind`,
      );
      dispatch(
        pass,
        this.kernels.attnScores,
        group,
        this.device.limits,
        [groupsFor(nTokens, wg), nTokens, c.numAttentionHeads],
        `L${layer}.attn_scores`,
      );
    }

    {
      const rows = c.numAttentionHeads * nTokens;
      const dims = uniform(
        uniforms.softmaxRows(rows, nTokens, nTokens),
        `L${layer}.softmax.dims`,
      );
      const group = bindGroup(
        this.device,
        this.kernels.softmaxRows,
        [dims, this.scores.buffer],
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
        uniforms.attnOutput(nTokens, c.numAttentionHeads, c.numKeyValueHeads, c.headDim),
        `L${layer}.attnout.dims`,
      );
      const group = bindGroup(
        this.device,
        this.kernels.attnOutput,
        [dims, this.scores.buffer, this.v.buffer, this.attnOut.buffer],
        `L${layer}.attnout.bind`,
      );
      dispatch(
        pass,
        this.kernels.attnOutput,
        group,
        this.device.limits,
        [groupsFor(qDim, wg), nTokens],
        `L${layer}.attn_output`,
      );
    }

    this.encodeMatmul(
      pass,
      uniform,
      nTokens,
      this.tensor(p + 'self_attn.o_proj.weight'),
      null,
      this.attnOut.buffer,
      this.proj.buffer,
      c.hiddenSize,
      `L${layer}.o_proj`,
    );
    this.encodeElementwise(
      pass,
      uniform,
      this.kernels.residualAdd,
      nTokens * c.hiddenSize,
      this.proj.buffer,
      this.x.buffer,
      `L${layer}.attn_residual`,
    );

    // ---- MLP block ----------------------------------------------------------------
    this.encodeRmsNorm(
      pass,
      uniform,
      nTokens,
      this.x.buffer,
      this.tensor(p + 'post_attention_layernorm.weight'),
      this.xNorm.buffer,
      `L${layer}.post_norm`,
    );
    this.encodeMatmul(
      pass,
      uniform,
      nTokens,
      this.tensor(p + 'mlp.gate_proj.weight'),
      null,
      this.xNorm.buffer,
      this.gate.buffer,
      c.intermediateSize,
      `L${layer}.gate_proj`,
    );
    this.encodeMatmul(
      pass,
      uniform,
      nTokens,
      this.tensor(p + 'mlp.up_proj.weight'),
      null,
      this.xNorm.buffer,
      this.up.buffer,
      c.intermediateSize,
      `L${layer}.up_proj`,
    );
    this.encodeElementwise(
      pass,
      uniform,
      this.kernels.siluMul,
      nTokens * c.intermediateSize,
      this.up.buffer,
      this.gate.buffer,
      `L${layer}.silu_mul`,
    );
    this.encodeMatmul(
      pass,
      uniform,
      nTokens,
      this.tensor(p + 'mlp.down_proj.weight'),
      null,
      this.gate.buffer,
      this.proj.buffer,
      c.hiddenSize,
      `L${layer}.down_proj`,
    );
    this.encodeElementwise(
      pass,
      uniform,
      this.kernels.residualAdd,
      nTokens * c.hiddenSize,
      this.proj.buffer,
      this.x.buffer,
      `L${layer}.mlp_residual`,
    );
  }
}
