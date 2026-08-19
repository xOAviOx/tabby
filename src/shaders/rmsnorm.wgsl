// out[t, :] = x[t, :] * rsqrt(mean(x[t, :]^2) + eps) * gain
//
// One invocation per row. M2 is correctness-only, so the row reduction is a plain
// sequential loop rather than a workgroup reduction.

struct Dims {
  n_tokens : u32,
  hidden   : u32,
  eps      : f32,
};

override wg_size : u32 = 64u;

@group(0) @binding(0) var<uniform>             dims : Dims;
@group(0) @binding(1) var<storage, read>       x    : array<f32>;  // [n_tokens, hidden]
@group(0) @binding(2) var<storage, read>       gain : array<f32>;  // [hidden]
@group(0) @binding(3) var<storage, read_write> out  : array<f32>;  // [n_tokens, hidden]

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let t = gid.x;
  if (t >= dims.n_tokens) {
    return;
  }

  let base = t * dims.hidden;
  var sum_sq : f32 = 0.0;
  for (var i : u32 = 0u; i < dims.hidden; i = i + 1u) {
    let v = x[base + i];
    sum_sq = sum_sq + v * v;
  }

  // Qwen/Llama normalise by the mean square, not the sum, and add eps inside the sqrt.
  let scale = inverseSqrt(sum_sq / f32(dims.hidden) + dims.eps);
  for (var i : u32 = 0u; i < dims.hidden; i = i + 1u) {
    out[base + i] = x[base + i] * scale * gain[i];
  }
}
