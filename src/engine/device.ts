/**
 * WebGPU device acquisition and limit negotiation.
 *
 * Everything downstream is sized from what we negotiate here: how large a single
 * weight tensor may be before it must be sharded (M1), how many bindings a kernel
 * may take, how big a workgroup can get. Nothing in this file may assume a value --
 * we ask the adapter for its maximum and record what we were actually granted.
 */

/** Limits we always try to raise to the adapter's reported maximum. */
const NEGOTIATED_LIMITS = [
  'maxBufferSize',
  'maxStorageBufferBindingSize',
  'maxUniformBufferBindingSize',
  'maxStorageBuffersPerShaderStage',
  'maxUniformBuffersPerShaderStage',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupStorageSize',
  'maxComputeWorkgroupSizeX',
  'maxComputeWorkgroupSizeY',
  'maxComputeWorkgroupSizeZ',
  'maxComputeWorkgroupsPerDimension',
  'maxBindGroups',
  'maxBindingsPerBindGroup',
  'minUniformBufferOffsetAlignment',
  'minStorageBufferOffsetAlignment',
] as const;

/** Optional features we use when present, and fall back from when absent. */
const OPTIONAL_FEATURES: GPUFeatureName[] = ['shader-f16', 'timestamp-query'];

/**
 * `minUniformBufferOffsetAlignment` and `minStorageBufferOffsetAlignment` are
 * "minimum" limits: a *lower* value is the better one, so requesting the adapter's
 * reported value is correct but requesting a raised value is not.
 */
const MINIMUM_STYLE_LIMITS = new Set<string>([
  'minUniformBufferOffsetAlignment',
  'minStorageBufferOffsetAlignment',
]);

export interface GpuContext {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly info: GPUAdapterInfo;
  /** Every limit the device was actually granted, name -> value. */
  readonly limits: Readonly<Record<string, number>>;
  readonly features: readonly string[];
  readonly hasF16: boolean;
  readonly hasTimestampQuery: boolean;
  /** True when we had to fall back to default limits because the raised request was refused. */
  readonly limitsWereRefused: boolean;
}

/**
 * Why WebGPU could not be reached. These look alike from the page and have completely
 * different fixes, so the cause is carried as a value rather than left in prose:
 *
 *  - `insecure-context`: served over http:// somewhere other than localhost, or from
 *    about:blank. `navigator.gpu` is simply absent, which is indistinguishable from an
 *    old browser unless the context is checked.
 *  - `no-webgpu`: secure context, but the browser does not implement WebGPU.
 *  - `no-adapter`: the browser implements it and refused to give us one -- a blocklisted
 *    driver, a headless build with no GPU stack, or a machine with nothing suitable.
 */
export type WebGpuUnavailableReason = 'insecure-context' | 'no-webgpu' | 'no-adapter';

export class WebGpuUnavailableError extends Error {
  readonly reason: WebGpuUnavailableReason;

  constructor(message: string, reason: WebGpuUnavailableReason) {
    super(message);
    this.name = 'WebGpuUnavailableError';
    this.reason = reason;
  }
}

export interface RequestDeviceOptions {
  powerPreference?: GPUPowerPreference;
  /** Invoked when the device is lost. Generation must stop and the engine must re-init. */
  onDeviceLost?: (info: GPUDeviceLostInfo) => void;
}

function limitsFrom(limits: GPUSupportedLimits): Record<string, number> {
  const out: Record<string, number> = {};
  // GPUSupportedLimits is an IDL interface, not a plain object: walk the prototype.
  for (const key of Object.keys(Object.getPrototypeOf(limits))) {
    const value = (limits as unknown as Record<string, unknown>)[key];
    if (typeof value === 'number') out[key] = value;
  }
  // Belt and braces in case an implementation exposes them as own properties.
  for (const key of NEGOTIATED_LIMITS) {
    const value = (limits as unknown as Record<string, unknown>)[key];
    if (typeof value === 'number') out[key] = value;
  }
  return out;
}

function buildRequiredLimits(adapter: GPUAdapter): Record<string, number> {
  const required: Record<string, number> = {};
  const available = adapter.limits as unknown as Record<string, unknown>;
  for (const key of NEGOTIATED_LIMITS) {
    const value = available[key];
    if (typeof value !== 'number') continue;
    // Asking for exactly what the adapter reports is always grantable per spec;
    // asking for more is a hard OperationError.
    required[key] = value;
  }
  for (const key of MINIMUM_STYLE_LIMITS) {
    // Requesting the adapter's own alignment is a no-op but harmless; keep it so the
    // granted-limits log shows the value we designed against.
    if (typeof available[key] === 'number') required[key] = available[key] as number;
  }
  return required;
}

export async function requestGpuContext(options: RequestDeviceOptions = {}): Promise<GpuContext> {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    // A secure context is a precondition, not a detail: without one the API is not
    // exposed at all, and telling someone to upgrade a browser that is already new
    // enough sends them the wrong way entirely.
    if (typeof isSecureContext !== 'undefined' && !isSecureContext) {
      throw new WebGpuUnavailableError(
        'This page is not a secure context, so WebGPU is not exposed to it at all.',
        'insecure-context',
      );
    }
    throw new WebGpuUnavailableError(
      'navigator.gpu is undefined: this browser does not expose WebGPU.',
      'no-webgpu',
    );
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: options.powerPreference ?? 'high-performance',
  });
  if (!adapter) {
    throw new WebGpuUnavailableError(
      'requestAdapter() returned null: WebGPU is exposed but no adapter is available.',
      'no-adapter',
    );
  }

  const requiredFeatures = OPTIONAL_FEATURES.filter((f) => adapter.features.has(f));

  let device: GPUDevice;
  let limitsWereRefused = false;
  try {
    device = await adapter.requestDevice({
      label: 'llm-engine',
      requiredFeatures,
      requiredLimits: buildRequiredLimits(adapter),
    });
  } catch (err) {
    // Spec says adapter-reported limits are always grantable, but implementations
    // have shipped bugs here. Degrade to defaults rather than failing to start.
    console.warn('[gpu] raised-limit device request refused, retrying with defaults:', err);
    limitsWereRefused = true;
    device = await adapter.requestDevice({ label: 'llm-engine (default limits)', requiredFeatures });
  }

  device.lost.then((info) => {
    console.error(`[gpu] device lost (${info.reason}): ${info.message}`);
    options.onDeviceLost?.(info);
  });

  device.onuncapturederror = (event) => {
    console.error('[gpu] uncaptured error:', (event as GPUUncapturedErrorEvent).error);
  };

  const features = [...device.features].sort();
  return {
    adapter,
    device,
    // adapter.info is a recent addition; older implementations expose nothing here.
    info: adapter.info ?? ({} as GPUAdapterInfo),
    limits: Object.freeze(limitsFrom(device.limits)),
    features,
    hasF16: device.features.has('shader-f16'),
    hasTimestampQuery: device.features.has('timestamp-query'),
    limitsWereRefused,
  };
}

const MIB = 1024 * 1024;

/** Human-readable dump of the negotiated context. Printed at startup and in tests. */
export function describeContext(ctx: GpuContext): string {
  const { info } = ctx;
  const lines: string[] = [];
  const adapterDesc = [info.vendor, info.architecture, info.device, info.description]
    .filter(Boolean)
    .join(' / ');
  lines.push(`adapter        : ${adapterDesc || '(not reported)'}`);
  lines.push(`features       : ${ctx.features.length ? ctx.features.join(', ') : '(none)'}`);
  lines.push(`shader-f16     : ${ctx.hasF16 ? 'yes' : 'no (f32 fallback path required)'}`);
  lines.push(`timestamp-query: ${ctx.hasTimestampQuery ? 'yes' : 'no (wall-clock timing only)'}`);
  if (ctx.limitsWereRefused) lines.push(`limits         : RAISED REQUEST REFUSED -- running on defaults`);
  for (const key of NEGOTIATED_LIMITS) {
    const value = ctx.limits[key];
    if (value === undefined) continue;
    const suffix = key.endsWith('Size') && value >= MIB ? ` (${(value / MIB).toFixed(0)} MiB)` : '';
    lines.push(`${key.padEnd(15)}: ${value}${suffix}`);
  }
  return lines.join('\n');
}

/**
 * Error-scope wrapper. WGSL has no printf and validation failures are otherwise
 * asynchronous and easy to miss, so during development every submit goes through here.
 */
let errorScopesEnabled = true;
export function setErrorScopesEnabled(enabled: boolean): void {
  errorScopesEnabled = enabled;
}

const SCOPES: GPUErrorFilter[] = ['internal', 'out-of-memory', 'validation'];

export async function withErrorScopes<T>(
  device: GPUDevice,
  label: string,
  fn: () => T | Promise<T>,
): Promise<T> {
  if (!errorScopesEnabled) return await fn();

  for (const scope of SCOPES) device.pushErrorScope(scope);
  let result: T;
  try {
    result = await fn();
  } catch (err) {
    // Still drain the scopes so the stack stays balanced.
    await Promise.all(SCOPES.map(() => device.popErrorScope()));
    throw err;
  }
  // Scopes pop LIFO: validation, then out-of-memory, then internal.
  const errors = await Promise.all([...SCOPES].reverse().map(() => device.popErrorScope()));
  const failures = errors.filter((e): e is GPUError => e !== null);
  if (failures.length > 0) {
    throw new Error(
      `[gpu] ${label} produced ${failures.length} error(s):\n` +
        failures.map((e) => `  ${e.constructor.name}: ${e.message}`).join('\n'),
    );
  }
  return result;
}
