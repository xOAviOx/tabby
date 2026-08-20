/**
 * Per-kernel GPU timing via timestamp queries, and the dispatch recorder the forward pass
 * encodes through.
 *
 * WebGPU timestamps are written at pass boundaries, not around individual dispatches, so
 * timing a single kernel means giving it its own compute pass. That is why profiling is a
 * mode rather than something always on: in normal operation the whole forward pass is one
 * pass and one submit, and splitting it into ~460 passes to measure would change the thing
 * being measured. `Recorder` hides the difference -- callers just say "dispatch this".
 *
 * `timestamp-query` is an optional feature. Without it the profiler reports nothing rather
 * than inventing numbers, and the perf panel falls back to wall-clock totals.
 */

export interface KernelTiming {
  label: string;
  calls: number;
  totalMs: number;
  /** Share of all measured GPU time. */
  fraction: number;
}

export interface ProfileReport {
  kernels: KernelTiming[];
  totalMs: number;
  passCount: number;
}

/** Timestamp pairs a single profiled submit can hold. */
const DEFAULT_CAPACITY = 2048;

export class Profiler {
  private readonly querySet: GPUQuerySet | null;
  private readonly resolveBuffer: GPUBuffer | null;
  private readonly readBuffer: GPUBuffer | null;
  private readonly capacity: number;

  /** Label per timestamp pair, in the order the passes were recorded. */
  private labels: string[] = [];
  private used = 0;

  constructor(device: GPUDevice, readonly enabled: boolean, capacity: number = DEFAULT_CAPACITY) {
    this.capacity = capacity;
    if (!enabled) {
      this.querySet = null;
      this.resolveBuffer = null;
      this.readBuffer = null;
      return;
    }
    this.querySet = device.createQuerySet({
      label: 'profiler',
      type: 'timestamp',
      count: capacity * 2,
    });
    this.resolveBuffer = device.createBuffer({
      label: 'profiler.resolve',
      size: capacity * 2 * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this.readBuffer = device.createBuffer({
      label: 'profiler.read',
      size: capacity * 2 * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  static supported(device: GPUDevice): boolean {
    return device.features.has('timestamp-query');
  }

  reset(): void {
    this.labels = [];
    this.used = 0;
  }

  /** Timestamp descriptor for the next pass, or undefined when full or disabled. */
  claim(label: string): GPUComputePassTimestampWrites | undefined {
    if (!this.enabled || !this.querySet || this.used >= this.capacity) return undefined;
    const index = this.used++;
    this.labels.push(label);
    return {
      querySet: this.querySet,
      beginningOfPassWriteIndex: index * 2,
      endOfPassWriteIndex: index * 2 + 1,
    };
  }

  /** Must be called on the same encoder, after every profiled pass has ended. */
  resolve(encoder: GPUCommandEncoder): void {
    if (!this.enabled || !this.querySet || !this.resolveBuffer || !this.readBuffer) return;
    if (this.used === 0) return;
    encoder.resolveQuerySet(this.querySet, 0, this.used * 2, this.resolveBuffer, 0);
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, this.readBuffer, 0, this.used * 2 * 8);
  }

  /** Read the timings back and aggregate by label. Synchronises, so call once per run. */
  async collect(): Promise<ProfileReport> {
    if (!this.enabled || !this.readBuffer || this.used === 0) {
      return { kernels: [], totalMs: 0, passCount: 0 };
    }

    await this.readBuffer.mapAsync(GPUMapMode.READ, 0, this.used * 2 * 8);
    const stamps = new BigUint64Array(this.readBuffer.getMappedRange(0, this.used * 2 * 8).slice(0));
    this.readBuffer.unmap();

    const totals = new Map<string, { calls: number; totalMs: number }>();
    let totalMs = 0;
    for (let i = 0; i < this.used; i++) {
      const start = stamps[i * 2];
      const end = stamps[i * 2 + 1];
      // A query can come back as zero if the pass was elided or the timestamp was not
      // written; counting it would silently deflate every other kernel's share.
      if (end <= start) continue;
      const ms = Number(end - start) / 1e6;
      // Fold the per-layer prefix away: 24 rows of `L7.input_norm` say much less than
      // one row of `input_norm` with 24 calls.
      const label = this.labels[i].replace(/^L\d+\./, '');
      const entry = totals.get(label) ?? { calls: 0, totalMs: 0 };
      entry.calls += 1;
      entry.totalMs += ms;
      totals.set(label, entry);
      totalMs += ms;
    }

    const kernels: KernelTiming[] = [...totals.entries()]
      .map(([label, entry]) => ({
        label,
        calls: entry.calls,
        totalMs: entry.totalMs,
        fraction: totalMs > 0 ? entry.totalMs / totalMs : 0,
      }))
      .sort((a, b) => b.totalMs - a.totalMs);

    return { kernels, totalMs, passCount: this.used };
  }

  destroy(): void {
    this.querySet?.destroy();
    this.resolveBuffer?.destroy();
    this.readBuffer?.destroy();
  }
}

/**
 * Records dispatches into a command encoder.
 *
 * Normally it keeps one compute pass open for the whole forward pass. When profiling, it
 * opens a pass per dispatch so each kernel gets its own timestamp pair. Callers do not
 * need to know which mode is active.
 */
export class Recorder {
  private pass: GPUComputePassEncoder | null = null;

  constructor(
    private readonly encoder: GPUCommandEncoder,
    private readonly limits: GPUSupportedLimits,
    private readonly profiler: Profiler | null,
    private readonly label = 'forward',
  ) {}

  private get profiling(): boolean {
    return this.profiler?.enabled ?? false;
  }

  dispatch(
    pipeline: GPUComputePipeline,
    group: GPUBindGroup,
    counts: [number, number?, number?],
    label: string,
  ): void {
    const [x, y = 1, z = 1] = counts;
    const max = this.limits.maxComputeWorkgroupsPerDimension;
    if (x > max || y > max || z > max) {
      throw new RangeError(
        `${label}: dispatch (${x}, ${y}, ${z}) exceeds maxComputeWorkgroupsPerDimension ${max}`,
      );
    }
    // Zero workgroups is legal but pointless, and it would burn a timestamp slot.
    if (x === 0 || y === 0 || z === 0) return;

    if (this.profiling) {
      const timestampWrites = this.profiler!.claim(label);
      const pass = this.encoder.beginComputePass({
        label,
        ...(timestampWrites ? { timestampWrites } : {}),
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(x, y, z);
      pass.end();
      return;
    }

    if (!this.pass) this.pass = this.encoder.beginComputePass({ label: this.label });
    this.pass.setPipeline(pipeline);
    this.pass.setBindGroup(0, group);
    this.pass.dispatchWorkgroups(x, y, z);
  }

  /** Close the open pass so a copy can be recorded, then reopen lazily on the next dispatch. */
  interrupt(): void {
    this.pass?.end();
    this.pass = null;
  }

  finish(): void {
    this.interrupt();
  }
}
