/**
 * GPUBuffer allocation, uploads, and CPU readback.
 *
 * The readback path here is the project's only debugger: WGSL has no printf, so the
 * way to inspect an intermediate is to write it to a storage buffer and pull it back.
 * It is deliberately easy to call -- and deliberately never called per-token, because
 * buffer mapping is async and forces a pipeline flush.
 */

export function alignTo(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

export interface StorageBufferOptions {
  label: string;
  /** Extra usage flags on top of STORAGE | COPY_DST | COPY_SRC. */
  usage?: GPUBufferUsageFlags;
}

const STORAGE_USAGE =
  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

/**
 * Allocate a storage buffer of `byteLength` bytes.
 *
 * Throws rather than truncating when the request exceeds a negotiated limit: from M1
 * onward, oversized tensors are handled by sharding at the registry level, never by
 * silently clamping here.
 */
export function createStorageBuffer(
  device: GPUDevice,
  byteLength: number,
  options: StorageBufferOptions,
): GPUBuffer {
  const size = alignTo(byteLength, 4);
  assertFitsLimits(device, size, options.label);
  return device.createBuffer({
    label: options.label,
    size,
    usage: STORAGE_USAGE | (options.usage ?? 0),
  });
}

function assertFitsLimits(device: GPUDevice, size: number, label: string): void {
  const { maxBufferSize, maxStorageBufferBindingSize } = device.limits;
  if (size > maxBufferSize) {
    throw new RangeError(
      `buffer '${label}' needs ${size} bytes but maxBufferSize is ${maxBufferSize}; shard it.`,
    );
  }
  if (size > maxStorageBufferBindingSize) {
    throw new RangeError(
      `buffer '${label}' needs ${size} bytes but maxStorageBufferBindingSize is ` +
        `${maxStorageBufferBindingSize}; shard it.`,
    );
  }
}

/** Allocate a storage buffer and fill it from host memory in one shot. */
export function uploadStorageBuffer(
  device: GPUDevice,
  data: ArrayBufferView,
  options: StorageBufferOptions,
): GPUBuffer {
  const size = alignTo(data.byteLength, 4);
  assertFitsLimits(device, size, options.label);
  const buffer = device.createBuffer({
    label: options.label,
    size,
    usage: STORAGE_USAGE | (options.usage ?? 0),
    mappedAtCreation: true,
  });
  const src = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  new Uint8Array(buffer.getMappedRange()).set(src);
  buffer.unmap();
  return buffer;
}

/**
 * Uniform buffer sized up to 16 bytes minimum. WGSL's uniform address space has
 * stricter layout rules than storage, and undersized bindings are a common source of
 * validation errors on some backends.
 */
export function uploadUniformBuffer(
  device: GPUDevice,
  data: ArrayBufferView,
  label: string,
): GPUBuffer {
  const size = Math.max(16, alignTo(data.byteLength, 16));
  const buffer = device.createBuffer({
    label,
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  const src = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  new Uint8Array(buffer.getMappedRange()).set(src);
  buffer.unmap();
  return buffer;
}

/**
 * Copy `byteLength` bytes back to the CPU. Submits its own command buffer and waits
 * on buffer mapping, so this is a full synchronisation point -- debug and test only.
 */
export async function readBuffer(
  device: GPUDevice,
  src: GPUBuffer,
  byteLength: number,
  srcOffset = 0,
): Promise<ArrayBuffer> {
  if (srcOffset % 4 !== 0) throw new RangeError(`readBuffer srcOffset must be 4-aligned, got ${srcOffset}`);
  const copySize = alignTo(byteLength, 4);
  const staging = device.createBuffer({
    label: `readback(${src.label || 'unnamed'})`,
    size: copySize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder({ label: 'readback' });
  encoder.copyBufferToBuffer(src, srcOffset, staging, 0, copySize);
  device.queue.submit([encoder.finish()]);

  await staging.mapAsync(GPUMapMode.READ);
  const copy = staging.getMappedRange().slice(0, byteLength);
  staging.unmap();
  staging.destroy();
  return copy;
}

export async function readF32(
  device: GPUDevice,
  src: GPUBuffer,
  count: number,
  srcOffset = 0,
): Promise<Float32Array> {
  return new Float32Array(await readBuffer(device, src, count * 4, srcOffset));
}

export async function readU32(
  device: GPUDevice,
  src: GPUBuffer,
  count: number,
  srcOffset = 0,
): Promise<Uint32Array> {
  return new Uint32Array(await readBuffer(device, src, count * 4, srcOffset));
}

/** Tracks buffers so a test or a model reload can release VRAM deterministically. */
export class BufferArena {
  private readonly buffers: GPUBuffer[] = [];

  constructor(private readonly device: GPUDevice) {}

  storage(byteLength: number, options: StorageBufferOptions): GPUBuffer {
    return this.track(createStorageBuffer(this.device, byteLength, options));
  }

  upload(data: ArrayBufferView, options: StorageBufferOptions): GPUBuffer {
    return this.track(uploadStorageBuffer(this.device, data, options));
  }

  uniform(data: ArrayBufferView, label: string): GPUBuffer {
    return this.track(uploadUniformBuffer(this.device, data, label));
  }

  track<T extends GPUBuffer>(buffer: T): T {
    this.buffers.push(buffer);
    return buffer;
  }

  get byteLength(): number {
    return this.buffers.reduce((sum, b) => sum + b.size, 0);
  }

  destroy(): void {
    for (const buffer of this.buffers) buffer.destroy();
    this.buffers.length = 0;
  }
}
