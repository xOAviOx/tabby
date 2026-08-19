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
