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
