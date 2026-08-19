/**
 * Token selection.
 *
 * M2 needs only greedy decoding, which is also what the golden comparison against
 * PyTorch requires: any sampling would make the two runs incomparable. Temperature,
 * top-k and top-p arrive at M4, and they move onto the GPU there -- reading a 152k-entry
 * logit vector back per token is ~600 KB of transfer per step and is exactly what M4's
 * gate forbids.
 */

import type { TopKResult } from './forward.js';

export function argmax(logits: Float32Array): number {
  let best = 0;
  let bestValue = logits[0];
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > bestValue) {
      bestValue = logits[i];
      best = i;
    }
  }
  return best;
}

/** The k highest-scoring ids, highest first. Used for the golden top-5 comparison. */
export function topK(logits: Float32Array, k: number): Array<{ id: number; value: number }> {
  const indices = Array.from(logits, (value, id) => ({ id, value }));
  indices.sort((a, b) => b.value - a.value || a.id - b.id);
  return indices.slice(0, k);
}

export interface GenerateOptions {
  maxNewTokens: number;
  /** Generation stops when one of these is produced. */
  stopTokens?: readonly number[];
  signal?: AbortSignal;
  /** Called with each new token as it is produced. */
  onToken?: (id: number, index: number) => void;
}

export interface GenerateResult {
  /** Newly generated ids, excluding the prompt. */
  tokens: number[];
  /** True when a stop token ended generation rather than the length limit. */
  stopped: boolean;
  ms: number;
}

/**
 * Greedy decoding against any function that maps a full token sequence to next-token
 * logits. M2 re-runs the whole prompt every step because there is no KV cache yet, so
 * this is O(n^2) by construction; M3 replaces the callback with the incremental path.
 */
export async function generateGreedy(
  promptIds: readonly number[],
  forward: (ids: readonly number[]) => Promise<Float32Array>,
  options: GenerateOptions,
): Promise<GenerateResult> {
  const stopTokens = new Set(options.stopTokens ?? []);
  const sequence = [...promptIds];
  const tokens: number[] = [];
  const started = performance.now();

  for (let step = 0; step < options.maxNewTokens; step++) {
    options.signal?.throwIfAborted();
    const next = argmax(await forward(sequence));
    tokens.push(next);
    sequence.push(next);
    options.onToken?.(next, step);
    if (stopTokens.has(next)) {
      return { tokens, stopped: true, ms: performance.now() - started };
    }
  }

  return { tokens, stopped: false, ms: performance.now() - started };
}

// =========================================================================================
// M4 sampling
// =========================================================================================


export interface SamplingParams {
  /** 0 selects the highest-probability token deterministically. */
  temperature: number;
  /** Candidate pool size. Applied on the GPU, before anything here runs. */
  topK: number;
  /** Nucleus threshold; 1 disables it. */
  topP: number;
  seed: number;
}

export const DEFAULT_SAMPLING: SamplingParams = {
  temperature: 0.7,
  topK: 40,
  topP: 0.9,
  seed: 0,
};

/**
 * mulberry32, the same generator the tests use. Seeded explicitly and advanced once per
 * token so a (seed, prompt, params) triple always replays identically -- which is what
 * the M4 determinism gate checks.
 */
export class SeededRandom {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
}

export interface SampleOutcome {
  id: number;
  /** Probability of the chosen token over the full vocabulary. */
  probability: number;
  /** How many candidates survived top-p. */
  consideredCount: number;
  /** Full-vocabulary probability mass held by the k candidates. */
  poolMass: number;
  /**
   * True when top-p asked for more mass than the candidate pool contains, i.e. the
   * nucleus was clipped by k rather than by p. Reported rather than hidden, because the
   * alternative is silently renormalising a truncated tail and calling it top-p.
   */
  poolExhausted: boolean;
}

/**
 * Turn a GPU top-k result into a sampled token.
 *
 * Because the GPU also returns the full-vocabulary softmax denominator, the candidate
 * probabilities here are exact rather than renormalised over the truncated set -- so
 * `poolMass` is a real number and top-p can tell when k cut it short.
 *
 * Temperature is applied after top-k, which is equivalent to applying it before: scaling
 * logits is monotonic, so it cannot change which tokens the top-k selects.
 */
export function sampleFromTopK(
  top: TopKResult,
  params: SamplingParams,
  random: SeededRandom,
): SampleOutcome {
  const { ids, logits, maxLogit, sumExp } = top;
  if (ids.length === 0) throw new Error('sampleFromTopK: empty candidate set');

  // Exact probabilities over the whole vocabulary.
  const exact = logits.map((logit) => Math.exp(logit - maxLogit) / sumExp);
  const poolMass = exact.reduce((a, b) => a + b, 0);

  if (params.temperature <= 0) {
    return {
      id: ids[0],
      probability: exact[0],
      consideredCount: 1,
      poolMass,
      poolExhausted: false,
    };
  }

  // Reweight within the pool at the requested temperature.
  const scaled = logits.map((logit) => Math.exp((logit - maxLogit) / params.temperature));
  const scaledTotal = scaled.reduce((a, b) => a + b, 0);
  const weights = scaled.map((w) => w / scaledTotal);

  // Nucleus: keep the smallest prefix whose mass reaches topP. Candidates arrive sorted.
  let cut = weights.length;
  let poolExhausted = false;
  if (params.topP < 1) {
    let cumulative = 0;
    cut = weights.length;
    for (let i = 0; i < weights.length; i++) {
      cumulative += weights[i];
      if (cumulative >= params.topP) {
        cut = i + 1;
        break;
      }
    }
    // The pool held less of the true distribution than top-p asked for.
    if (poolMass < params.topP) poolExhausted = true;
  }

  let total = 0;
  for (let i = 0; i < cut; i++) total += weights[i];

  const target = random.next() * total;
  let running = 0;
  let chosen = cut - 1;
  for (let i = 0; i < cut; i++) {
    running += weights[i];
    if (target < running) {
      chosen = i;
      break;
    }
  }

  return {
    id: ids[chosen],
    probability: exact[chosen],
    consideredCount: cut,
    poolMass,
    poolExhausted,
  };
}
