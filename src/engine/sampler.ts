/**
 * Token selection.
 *
 * M2 needs only greedy decoding, which is also what the golden comparison against
 * PyTorch requires: any sampling would make the two runs incomparable. Temperature,
 * top-k and top-p arrive at M4, and they move onto the GPU there -- reading a 152k-entry
 * logit vector back per token is ~600 KB of transfer per step and is exactly what M4's
 * gate forbids.
 */

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
