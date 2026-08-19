/**
 * Kernel launch wrappers: pipeline construction, uniform packing and dispatch sizing
 * for each WGSL kernel in /src/shaders.
 *
 * Encoding is split from execution deliberately. `encode*` records into a caller-owned
 * compute pass so M3's forward pass can batch every layer into one command buffer;
 * the `run*` helpers are standalone one-shot versions for tests and smoke checks, and
 * they synchronise, so they must stay out of the per-token path.
 */

import { BufferArena, readF32 } from './buffers.js';
import { bindGroup, type PipelineCache } from './pipelines.js';
import { withErrorScopes } from './device.js';
import matvecF32Source from '../shaders/matvec_f32.wgsl?raw';
import embedGatherSource from '../shaders/embed_gather.wgsl?raw';
import rmsNormSource from '../shaders/rmsnorm.wgsl?raw';
import matmulF16Source from '../shaders/matmul_f16.wgsl?raw';
import ropeSource from '../shaders/rope.wgsl?raw';
import attnScoresSource from '../shaders/attn_scores.wgsl?raw';
import softmaxRowsSource from '../shaders/softmax_rows.wgsl?raw';
import attnOutputSource from '../shaders/attn_output.wgsl?raw';
import residualAddSource from '../shaders/residual_add.wgsl?raw';
import siluMulSource from '../shaders/silu_mul.wgsl?raw';
import kvWriteSource from '../shaders/kv_write.wgsl?raw';
import matmulF16TiledSource from '../shaders/matmul_f16_tiled.wgsl?raw';
import reduceMaxSource from '../shaders/reduce_max.wgsl?raw';
import reduceSumExpSource from '../shaders/reduce_sumexp.wgsl?raw';
import topkPartialSource from '../shaders/topk_partial.wgsl?raw';
import topkSelectSource from '../shaders/topk_select.wgsl?raw';

export const DEFAULT_MATVEC_WORKGROUP = 64;

export interface MatvecF32Buffers {
  /** Uniform holding {m, n}. */
  dims: GPUBuffer;
  /** Weight matrix [m, n] row-major, f32. */
  w: GPUBuffer;
  /** Input vector [n], f32. */
  x: GPUBuffer;
  /** Output vector [m], f32. */
  y: GPUBuffer;
}

export function matvecF32Pipeline(
  cache: PipelineCache,
  workgroupSize: number = DEFAULT_MATVEC_WORKGROUP,
): Promise<GPUComputePipeline> {
  return cache.compute({
    code: matvecF32Source,
    label: `matvec_f32(wg=${workgroupSize})`,
    constants: { wg_size: workgroupSize },
  });
}

/** Pack the {m, n} uniform this kernel expects. */
export function matvecF32Dims(m: number, n: number): Uint32Array {
  return new Uint32Array([m, n]);
}

export function encodeMatvecF32(
  pass: GPUComputePassEncoder,
  pipeline: GPUComputePipeline,
  group: GPUBindGroup,
  m: number,
  workgroupSize: number = DEFAULT_MATVEC_WORKGROUP,
): void {
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(m / workgroupSize));
}

export interface RunMatvecOptions {
  workgroupSize?: number;
}

/**
 * Upload W and x, run the kernel, read y back. Allocates and frees its own buffers and
 * blocks on readback -- for tests and diagnostics only.
 */
export async function runMatvecF32(
  device: GPUDevice,
  cache: PipelineCache,
  w: Float32Array,
  x: Float32Array,
  m: number,
  n: number,
  options: RunMatvecOptions = {},
): Promise<Float32Array> {
  const workgroupSize = options.workgroupSize ?? DEFAULT_MATVEC_WORKGROUP;
  const groups = Math.ceil(m / workgroupSize);
  const maxGroups = device.limits.maxComputeWorkgroupsPerDimension;
  if (groups > maxGroups) {
    throw new RangeError(
      `matvec needs ${groups} workgroups but maxComputeWorkgroupsPerDimension is ${maxGroups}`,
    );
  }

  const arena = new BufferArena(device);
  try {
    const buffers: MatvecF32Buffers = {
      dims: arena.uniform(matvecF32Dims(m, n), 'matvec.dims'),
      w: arena.upload(w, { label: 'matvec.w' }),
      x: arena.upload(x, { label: 'matvec.x' }),
      y: arena.storage(m * 4, { label: 'matvec.y' }),
    };

    const pipeline = await matvecF32Pipeline(cache, workgroupSize);
    const group = bindGroup(
      device,
      pipeline,
      [buffers.dims, buffers.w, buffers.x, buffers.y],
      'matvec.bindgroup',
    );

    await withErrorScopes(device, `matvec_f32 ${m}x${n}`, () => {
      const encoder = device.createCommandEncoder({ label: 'matvec' });
      const pass = encoder.beginComputePass({ label: 'matvec' });
      encodeMatvecF32(pass, pipeline, group, m, workgroupSize);
      pass.end();
      device.queue.submit([encoder.finish()]);
    });

    return await readF32(device, buffers.y, m);
  } finally {
    arena.destroy();
  }
}

// =======================================================================================
// M2 kernels
// =======================================================================================

export const DEFAULT_WORKGROUP = 64;

/**
 * Uniform buffers are padded to 16 bytes. WGSL's uniform address space has stricter
 * layout rules than storage and undersized bindings are a common backend-specific
 * validation failure, so every packer here rounds up rather than sizing exactly.
 */
function packUniform(words: Array<number | { f32: number }>): ArrayBuffer {
  const size = Math.max(16, Math.ceil((words.length * 4) / 16) * 16);
  const buffer = new ArrayBuffer(size);
  const view = new DataView(buffer);
  words.forEach((word, index) => {
    if (typeof word === 'number') view.setUint32(index * 4, word, true);
    else view.setFloat32(index * 4, word.f32, true);
  });
  return buffer;
}

export const uniforms = {
  embedGather: (nTokens: number, hidden: number, rowStart: number, rowCount: number) =>
    packUniform([nTokens, hidden, rowStart, rowCount]),
  rmsNorm: (nTokens: number, hidden: number, eps: number) =>
    packUniform([nTokens, hidden, { f32: eps }]),
  matmul: (
    nTokens: number,
    outDim: number,
    inDim: number,
    hasBias: boolean,
    rowStart = 0,
    outStride = outDim,
  ) => packUniform([nTokens, outDim, inDim, hasBias ? 1 : 0, rowStart, outStride]),
  rope: (nTokens: number, nHeads: number, headDim: number, posStart: number, theta: number) =>
    packUniform([nTokens, nHeads, headDim, posStart, { f32: theta }]),
  attnScores: (
    nNew: number,
    totalLen: number,
    nHeads: number,
    nKvHeads: number,
    headDim: number,
    posStart: number,
    kvOffset: number,
    scale: number,
  ) =>
    packUniform([nNew, totalLen, nHeads, nKvHeads, headDim, posStart, kvOffset, { f32: scale }]),
  softmaxRows: (nRows: number, rowLen: number, causalPeriod: number, causalOffset = 0) =>
    packUniform([nRows, rowLen, causalPeriod, causalOffset]),
  attnOutput: (
    nNew: number,
    totalLen: number,
    nHeads: number,
    nKvHeads: number,
    headDim: number,
    posStart: number,
    kvOffset: number,
  ) => packUniform([nNew, totalLen, nHeads, nKvHeads, headDim, posStart, kvOffset]),
  kvWrite: (n: number, dstStart: number) => packUniform([n, dstStart]),
  reduceMax: (n: number, outIndex: number) => packUniform([n, outIndex]),
  reduceSumExp: (n: number, outIndex: number, applyExp: boolean, maxIndex: number) =>
    packUniform([n, outIndex, applyExp ? 1 : 0, maxIndex]),
  topkPartial: (n: number) => packUniform([n]),
  topkSelect: (nPartials: number, step: number, outBase: number) =>
    packUniform([nPartials, step, outBase]),
  elementwise: (n: number) => packUniform([n]),
};

export interface KernelSet {
  embedGather: GPUComputePipeline;
  rmsNorm: GPUComputePipeline;
  matmul: GPUComputePipeline;
  /** Prefill variant: covers TILE_T positions per workgroup. See the shader header. */
  matmulTiled: GPUComputePipeline;
  rope: GPUComputePipeline;
  attnScores: GPUComputePipeline;
  softmaxRows: GPUComputePipeline;
  attnOutput: GPUComputePipeline;
  residualAdd: GPUComputePipeline;
  siluMul: GPUComputePipeline;
  kvWrite: GPUComputePipeline;
  reduceMax: GPUComputePipeline;
  reduceSumExp: GPUComputePipeline;
  topkPartial: GPUComputePipeline;
  topkSelect: GPUComputePipeline;
  workgroupSize: number;
}

/**
 * Build every pipeline once. Shader compilation is far too slow to sit anywhere near
 * the generation loop, so this runs at load and the result is held for the session.
 */
export async function createKernels(
  cache: PipelineCache,
  workgroupSize: number = DEFAULT_WORKGROUP,
): Promise<KernelSet> {
  const constants = { wg_size: workgroupSize };
  const build = (code: string, label: string): Promise<GPUComputePipeline> =>
    cache.compute({ code, label: `${label}(wg=${workgroupSize})`, constants });

  const [
    embedGather,
    rmsNorm,
    matmul,
    matmulTiled,
    rope,
    attnScores,
    softmaxRows,
    attnOutput,
    residualAdd,
    siluMul,
    kvWrite,
    reduceMax,
    reduceSumExp,
    topkPartial,
    topkSelect,
  ] = await Promise.all([
    build(embedGatherSource, 'embed_gather'),
    build(rmsNormSource, 'rmsnorm'),
    build(matmulF16Source, 'matmul_f16'),
    build(matmulF16TiledSource, 'matmul_f16_tiled'),
    build(ropeSource, 'rope'),
    build(attnScoresSource, 'attn_scores'),
    build(softmaxRowsSource, 'softmax_rows'),
    build(attnOutputSource, 'attn_output'),
    build(residualAddSource, 'residual_add'),
    build(siluMulSource, 'silu_mul'),
    build(kvWriteSource, 'kv_write'),
    // The reduction kernels pin their own workgroup size, so the override is irrelevant
    // to them; passing it anyway would only fragment the pipeline cache.
    cache.compute({ code: reduceMaxSource, label: 'reduce_max' }),
    cache.compute({ code: reduceSumExpSource, label: 'reduce_sumexp' }),
    cache.compute({ code: topkPartialSource, label: 'topk_partial' }),
    cache.compute({ code: topkSelectSource, label: 'topk_select' }),
  ]);

  return {
    embedGather,
    rmsNorm,
    matmul,
    matmulTiled,
    rope,
    attnScores,
    softmaxRows,
    attnOutput,
    residualAdd,
    siluMul,
    kvWrite,
    reduceMax,
    reduceSumExp,
    topkPartial,
    topkSelect,
    workgroupSize,
  };
}

/**
 * Dispatch, checking the per-dimension workgroup limit.
 *
 * This is not a theoretical guard: `maxComputeWorkgroupsPerDimension` is 65,535 on every
 * adapter (it sits at the spec default), and the lm_head has 151,936 output rows. Only
 * the 2D dispatch shape used by matmul_f16 keeps X under the limit -- one workgroup per
 * output row would exceed it by 2.3x on every machine.
 */
export function dispatch(
  pass: GPUComputePassEncoder,
  pipeline: GPUComputePipeline,
  group: GPUBindGroup,
  limits: GPUSupportedLimits,
  counts: [number, number?, number?],
  label: string,
): void {
  const [x, y = 1, z = 1] = counts;
  const max = limits.maxComputeWorkgroupsPerDimension;
  if (x > max || y > max || z > max) {
    throw new RangeError(
      `${label}: dispatch (${x}, ${y}, ${z}) exceeds maxComputeWorkgroupsPerDimension ${max}`,
    );
  }
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(x, y, z);
}

export function groupsFor(total: number, workgroupSize: number): number {
  return Math.ceil(total / workgroupSize);
}

/** Token positions each tiled-matmul workgroup covers. Must match TILE_T in the shader. */
export const MATMUL_TILE_T = 4;

/** Partial groups used by the reduction and top-k kernels. */
export const REDUCE_GROUPS = 256;

/** Largest k the sample-output block is sized for. */
export const MAX_TOP_K = 64;
