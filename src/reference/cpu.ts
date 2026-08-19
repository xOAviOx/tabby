/**
 * Scalar CPU reference implementations, one per GPU kernel.
 *
 * Two accumulation modes exist for every reduction, and the distinction matters:
 *
 *   - `*F32` reproduces the kernel's own arithmetic, rounding to f32 after each
 *     operation. Comparing the GPU against this isolates *logic* bugs -- indexing,
 *     bounds, head mapping -- from float noise, so a tight tolerance is meaningful.
 *   - `*F64` accumulates in double precision, which is what the kernel would compute
 *     with unlimited precision. The gap between the two is the kernel's inherent
 *     rounding error; tests report it rather than asserting on it, because the real
 *     numerical-fidelity check is M2's comparison against PyTorch goldens.
 *
 * Reporting both means a widening float error can never hide behind a logic-only gate.
 */

const f32 = Math.fround;

/**
 * y = W @ x, W row-major [M, N]. Rounds to f32 after every multiply and every add,
 * matching `matvec_f32.wgsl` operation for operation.
 */
export function matvecF32(
  w: Float32Array,
  x: Float32Array,
  m: number,
  n: number,
): Float32Array {
  assertShapes(w, x, m, n);
  const y = new Float32Array(m);
  for (let row = 0; row < m; row++) {
    const base = row * n;
    let acc = 0;
    for (let k = 0; k < n; k++) {
      acc = f32(acc + f32(w[base + k] * x[k]));
    }
    y[row] = acc;
  }
  return y;
}

/** y = W @ x accumulated in f64. Same traversal order, no intermediate rounding. */
export function matvecF64(
  w: Float32Array,
  x: Float32Array,
  m: number,
  n: number,
): Float64Array {
  assertShapes(w, x, m, n);
  const y = new Float64Array(m);
  for (let row = 0; row < m; row++) {
    const base = row * n;
    let acc = 0;
    for (let k = 0; k < n; k++) {
      acc += w[base + k] * x[k];
    }
    y[row] = acc;
  }
  return y;
}

function assertShapes(w: Float32Array, x: Float32Array, m: number, n: number): void {
  if (w.length !== m * n) {
    throw new Error(`matvec: W has ${w.length} elements, expected ${m}x${n}=${m * n}`);
  }
  if (x.length !== n) {
    throw new Error(`matvec: x has ${x.length} elements, expected ${n}`);
  }
}

// =======================================================================================
// M2 kernels
//
// Each function mirrors one .wgsl file operation for operation, accumulating in f32 so
// the comparison isolates kernel logic from float precision (see the header above).
// Weights arrive already dequantized: f16 unpacking is the GPU kernel's job, and the
// tests hand both sides the same f16-rounded values so the two agree exactly.
// =======================================================================================

/** out[t, d] = table[ids[t], d] */
export function embedGather(
  table: Float32Array,
  ids: Uint32Array,
  hidden: number,
): Float32Array {
  const out = new Float32Array(ids.length * hidden);
  for (let t = 0; t < ids.length; t++) {
    const src = ids[t] * hidden;
    if (src + hidden > table.length) {
      throw new Error(`embedGather: id ${ids[t]} is outside the embedding table`);
    }
    out.set(table.subarray(src, src + hidden), t * hidden);
  }
  return out;
}

/** out[t, :] = x[t, :] * rsqrt(mean(x^2) + eps) * gain */
export function rmsNorm(
  x: Float32Array,
  gain: Float32Array,
  nTokens: number,
  hidden: number,
  eps: number,
): Float32Array {
  const out = new Float32Array(nTokens * hidden);
  for (let t = 0; t < nTokens; t++) {
    const base = t * hidden;
    let sumSq = 0;
    for (let i = 0; i < hidden; i++) {
      const v = x[base + i];
      sumSq = f32(sumSq + f32(v * v));
    }
    const scale = f32(1 / Math.sqrt(f32(f32(sumSq / hidden) + eps)));
    for (let i = 0; i < hidden; i++) {
      out[base + i] = f32(f32(x[base + i] * scale) * gain[i]);
    }
  }
  return out;
}

/** y[t, m] = sum_k w[m, k] * x[t, k] + bias[m] */
export function matmul(
  w: Float32Array,
  x: Float32Array,
  nTokens: number,
  outDim: number,
  inDim: number,
  bias: Float32Array | null,
): Float32Array {
  const y = new Float32Array(nTokens * outDim);
  for (let t = 0; t < nTokens; t++) {
    const xBase = t * inDim;
    for (let m = 0; m < outDim; m++) {
      const wBase = m * inDim;
      let acc = 0;
      for (let k = 0; k < inDim; k++) {
        acc = f32(acc + f32(w[wBase + k] * x[xBase + k]));
      }
      if (bias) acc = f32(acc + bias[m]);
      y[t * outDim + m] = acc;
    }
  }
  return y;
}

/**
 * Rotary embedding, rotate-half convention: element i pairs with i + headDim/2.
 * Returns a new array rather than mutating, so callers can diff before and after.
 */
export function rope(
  v: Float32Array,
  nTokens: number,
  nHeads: number,
  headDim: number,
  theta: number,
  posStart = 0,
): Float32Array {
  const out = Float32Array.from(v);
  const half = headDim / 2;
  if (!Number.isInteger(half)) throw new Error(`rope: head_dim ${headDim} must be even`);

  for (let t = 0; t < nTokens; t++) {
    for (let h = 0; h < nHeads; h++) {
      const base = (t * nHeads + h) * headDim;
      for (let i = 0; i < half; i++) {
        const freq = f32(Math.pow(theta, (-2 * i) / headDim));
        const angle = f32((posStart + t) * freq);
        const cos = f32(Math.cos(angle));
        const sin = f32(Math.sin(angle));
        const lo = v[base + i];
        const hi = v[base + i + half];
        out[base + i] = f32(f32(lo * cos) - f32(hi * sin));
        out[base + i + half] = f32(f32(hi * cos) + f32(lo * sin));
      }
    }
  }
  return out;
}

export const ATTN_NEG_INF = -3.0e38;

export interface AttnShape {
  /** Query positions being computed now. */
  nNew: number;
  /** Key positions available, i.e. posStart + nNew. */
  totalLen: number;
  nHeads: number;
  nKvHeads: number;
  headDim: number;
  /** Absolute position of query 0. Non-zero only when decoding against a cache. */
  posStart?: number;
  /** Element offset of the layer slice within the K/V buffer. */
  kvOffset?: number;
}

/** scores[h, i, j] = dot(q[i,h], k[j, kvHead(h)]) * scale, causally masked. */
export function attnScores(
  q: Float32Array,
  k: Float32Array,
  shape: AttnShape,
): Float32Array {
  const { nNew, totalLen, nHeads, nKvHeads, headDim } = shape;
  const posStart = shape.posStart ?? 0;
  const kvOffset = shape.kvOffset ?? 0;
  const scale = f32(1 / Math.sqrt(headDim));
  const group = nHeads / nKvHeads;
  if (!Number.isInteger(group)) {
    throw new Error(`attnScores: ${nHeads} query heads is not a multiple of ${nKvHeads} KV heads`);
  }

  const out = new Float32Array(nHeads * nNew * totalLen);
  for (let h = 0; h < nHeads; h++) {
    const kvHead = Math.floor(h / group);
    for (let i = 0; i < nNew; i++) {
      for (let j = 0; j < totalLen; j++) {
        const index = (h * nNew + i) * totalLen + j;
        if (j > posStart + i) {
          out[index] = ATTN_NEG_INF;
          continue;
        }
        const qBase = (i * nHeads + h) * headDim;
        const kBase = kvOffset + (j * nKvHeads + kvHead) * headDim;
        let acc = 0;
        for (let d = 0; d < headDim; d++) {
          acc = f32(acc + f32(q[qBase + d] * k[kBase + d]));
        }
        out[index] = f32(acc * scale);
      }
    }
  }
  return out;
}

/**
 * Row-wise softmax. `causalPeriod` limits row r to its first
 * `causalOffset + (r % period) + 1` entries; 0 softmaxes the whole row.
 */
export function softmaxRows(
  x: Float32Array,
  nRows: number,
  rowLen: number,
  causalPeriod = 0,
  causalOffset = 0,
): Float32Array {
  const out = Float32Array.from(x);
  for (let row = 0; row < nRows; row++) {
    const valid = causalPeriod !== 0 ? causalOffset + (row % causalPeriod) + 1 : rowLen;
    const base = row * rowLen;

    let maxV = out[base];
    for (let i = 1; i < valid; i++) maxV = Math.max(maxV, out[base + i]);

    let sum = 0;
    for (let i = 0; i < valid; i++) {
      const e = f32(Math.exp(f32(out[base + i] - maxV)));
      out[base + i] = e;
      sum = f32(sum + e);
    }
    const inv = f32(1 / sum);
    for (let i = 0; i < valid; i++) out[base + i] = f32(out[base + i] * inv);
    for (let i = valid; i < rowLen; i++) out[base + i] = 0;
  }
  return out;
}

/** out[i, h, d] = sum_j scores[h, i, j] * v[j, kvHead(h), d] */
export function attnOutput(
  scores: Float32Array,
  v: Float32Array,
  shape: AttnShape,
): Float32Array {
  const { nNew, totalLen, nHeads, nKvHeads, headDim } = shape;
  const posStart = shape.posStart ?? 0;
  const kvOffset = shape.kvOffset ?? 0;
  const group = nHeads / nKvHeads;
  const out = new Float32Array(nNew * nHeads * headDim);
  for (let i = 0; i < nNew; i++) {
    for (let h = 0; h < nHeads; h++) {
      const kvHead = Math.floor(h / group);
      const scoreBase = (h * nNew + i) * totalLen;
      for (let d = 0; d < headDim; d++) {
        let acc = 0;
        for (let j = 0; j <= posStart + i; j++) {
          const vBase = kvOffset + (j * nKvHeads + kvHead) * headDim;
          acc = f32(acc + f32(scores[scoreBase + j] * v[vBase + d]));
        }
        out[(i * nHeads + h) * headDim + d] = acc;
      }
    }
  }
  return out;
}

export function residualAdd(x: Float32Array, delta: Float32Array): Float32Array {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = f32(x[i] + delta[i]);
  return out;
}

/** SwiGLU: silu(gate) * up, where silu(v) = v / (1 + exp(-v)). */
export function siluMul(gate: Float32Array, up: Float32Array): Float32Array {
  const out = new Float32Array(gate.length);
  for (let i = 0; i < gate.length; i++) {
    const g = gate[i];
    out[i] = f32(f32(g / f32(1 + f32(Math.exp(-g)))) * up[i]);
  }
  return out;
}
