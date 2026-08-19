/** Shared test utilities: deterministic randomness and error metrics. */

/**
 * mulberry32. Tests use a seeded PRNG rather than Math.random so that a numerical
 * failure is reproducible from the seed printed in the failure message.
 */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** `count` samples drawn uniformly from [-scale, scale]. */
export function randomF32(count: number, seed: number, scale = 1): Float32Array {
  const next = rng(seed);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = (next() * 2 - 1) * scale;
  return out;
}

export interface ErrorStats {
  maxAbs: number;
  /** Index where maxAbs occurred, for bisecting a failure. */
  argmax: number;
  rms: number;
}

export function errorStats(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
): ErrorStats {
  if (actual.length !== expected.length) {
    throw new Error(`length mismatch: ${actual.length} vs ${expected.length}`);
  }
  let maxAbs = 0;
  let argmax = -1;
  let sumSq = 0;
  for (let i = 0; i < actual.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    if (!Number.isFinite(diff)) {
      throw new Error(`non-finite value at ${i}: got ${actual[i]}, expected ${expected[i]}`);
    }
    if (diff > maxAbs) {
      maxAbs = diff;
      argmax = i;
    }
    sumSq += diff * diff;
  }
  return { maxAbs, argmax, rms: Math.sqrt(sumSq / actual.length) };
}

export function fmt(x: number): string {
  return x.toExponential(2);
}
