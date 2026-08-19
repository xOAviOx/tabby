/**
 * Shader module compilation and compute-pipeline caching.
 *
 * Shader compilation is slow enough that doing it inside the generation loop would
 * dominate the benchmark, so every pipeline this project uses is built once at load
 * and looked up by key afterwards. Compilation diagnostics are surfaced eagerly --
 * a WGSL warning is usually a bug we want to see now, not at M2's golden gate.
 */

export interface ComputePipelineSpec {
  /** WGSL source, imported with `?raw`. */
  code: string;
  label: string;
  entryPoint?: string;
  /**
   * Pipeline-overridable constants. This is how tuning parameters reach a shader
   * without ever being written into the WGSL as literals.
   */
  constants?: Record<string, number>;
}

function specKey(spec: ComputePipelineSpec): string {
  const constants = spec.constants
    ? Object.keys(spec.constants)
        .sort()
        .map((k) => `${k}=${spec.constants![k]}`)
        .join(',')
    : '';
  return `${spec.label} ${spec.entryPoint ?? 'main'} ${constants} ${spec.code}`;
}

export class PipelineCache {
  private readonly modules = new Map<string, GPUShaderModule>();
  private readonly pipelines = new Map<string, GPUComputePipeline>();

  constructor(private readonly device: GPUDevice) {}

  get size(): number {
    return this.pipelines.size;
  }

  /** Compile (or reuse) a shader module, throwing on any compilation error. */
  async module(code: string, label: string): Promise<GPUShaderModule> {
    const cached = this.modules.get(code);
    if (cached) return cached;

    const module = this.device.createShaderModule({ label, code });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    const warnings = info.messages.filter((m) => m.type === 'warning');

    for (const w of warnings) {
      console.warn(`[wgsl] ${label}:${w.lineNum}:${w.linePos} warning: ${w.message}`);
    }
    if (errors.length > 0) {
      const detail = errors
        .map((e) => `  ${label}:${e.lineNum}:${e.linePos} ${e.message}`)
        .join('\n');
      throw new Error(`WGSL compilation failed for '${label}':\n${detail}`);
    }

    this.modules.set(code, module);
    return module;
  }

  /** Build (or reuse) a compute pipeline with an auto-derived bind group layout. */
  async compute(spec: ComputePipelineSpec): Promise<GPUComputePipeline> {
    const key = specKey(spec);
    const cached = this.pipelines.get(key);
    if (cached) return cached;

    const module = await this.module(spec.code, spec.label);
    const pipeline = await this.device.createComputePipelineAsync({
      label: spec.label,
      layout: 'auto',
      compute: {
        module,
        entryPoint: spec.entryPoint ?? 'main',
        constants: spec.constants,
      },
    });

    this.pipelines.set(key, pipeline);
    return pipeline;
  }
}

/** Convenience for the common "bind N buffers at sequential indices" case. */
export function bindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  buffers: GPUBuffer[],
  label: string,
  group = 0,
): GPUBindGroup {
  return device.createBindGroup({
    label,
    layout: pipeline.getBindGroupLayout(group),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
}
