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

/**
 * Exact f16 -> f32 widening. Every f16 value is representable in f32, so this is lossless
 * and needs no rounding logic -- which is why tests generate f16 *bit patterns* and widen
 * them, rather than generating f32 and narrowing. It keeps the GPU and the CPU reference
 * looking at bit-identical values with no conversion code of its own under test.
 */
export function f16ToF32(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (mantissa / 1024);
  if (exponent === 31) return mantissa ? NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
}

export interface F16Data {
  /** Two f16 values packed per u32, the layout the shaders read. */
  words: Uint32Array;
  /** The same values widened to f32, for the CPU reference. */
  values: Float32Array;
}

/**
 * `count` random f16 values with exponents kept well inside the normal range, so no
 * test ever trips over infinities or denormals by accident.
 */
export function randomF16(count: number, seed: number): F16Data {
  const next = rng(seed);
  const values = new Float32Array(count);
  const words = new Uint32Array(Math.ceil(count / 2));
  for (let i = 0; i < count; i++) {
    const sign = next() < 0.5 ? 0 : 0x8000;
    const exponent = 10 + Math.floor(next() * 8); // 2^-5 .. 2^2
    const mantissa = Math.floor(next() * 1024);
    const bits = sign | (exponent << 10) | mantissa;
    values[i] = f16ToF32(bits);
    words[i >> 1] |= bits << (i % 2 === 0 ? 0 : 16);
  }
  return { words, values };
}
