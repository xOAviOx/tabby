// y = W @ x    with W [M, N] row-major, x [N], y [M].
//
// M0 baseline: one invocation per output row, one sequential pass over the row.
// This is the naive shape on purpose -- it is the correctness reference that the
// optimised decode kernel in M5 must reproduce. Do not tune this file.
//
// Every dimension arrives through `dims`; the only compile-time value is the
// workgroup size, which is a pipeline-overridable constant so M5 can sweep it.

struct Dims {
  // rows of W == length of y
  m : u32,
  // cols of W == length of x
  n : u32,
};

override wg_size : u32 = 64u;

@group(0) @binding(0) var<uniform>             dims : Dims;
@group(0) @binding(1) var<storage, read>       w    : array<f32>;
@group(0) @binding(2) var<storage, read>       x    : array<f32>;
@group(0) @binding(3) var<storage, read_write> y    : array<f32>;

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let row = gid.x;
  // M is rarely a multiple of the workgroup size; the tail invocations must not write.
  if (row >= dims.m) {
    return;
  }

  let base = row * dims.n;
  var acc : f32 = 0.0;
  for (var k : u32 = 0u; k < dims.n; k = k + 1u) {
    acc = acc + w[base + k] * x[k];
  }
  y[row] = acc;
}
