// out[t, :] = x[t, :] * rsqrt(mean(x[t, :]^2) + eps) * gain
//
// One workgroup per row, with the sum of squares reduced across its lanes.
//
// The M2 version used one *invocation* per row, which is fine for prefill and pathological
// for decode: with a single token the whole 896-element reduction ran on one thread while
// the rest of the GPU idled. Profiling put 48 of those dispatches -- two per layer -- at
// roughly half of all decode time, which made this the largest single win in M5.

struct Dims {
  n_tokens : u32,
  hidden   : u32,
  eps      : f32,
};

/** Fixed rather than overridable: the workgroup array must be sized at compile time. */
const RWG : u32 = 256u;

@group(0) @binding(0) var<uniform>             dims : Dims;
@group(0) @binding(1) var<storage, read>       x    : array<f32>;
@group(0) @binding(2) var<storage, read>       gain : array<f32>;
@group(0) @binding(3) var<storage, read_write> out  : array<f32>;

var<workgroup> scratch : array<f32, RWG>;

@compute @workgroup_size(RWG)
fn main(
  @builtin(workgroup_id) wid : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
) {
  let t = wid.x;
  // Uniform across the workgroup, so returning here cannot desynchronise the barriers.
  if (t >= dims.n_tokens) {
    return;
  }

  let base = t * dims.hidden;
  var sum_sq : f32 = 0.0;
  for (var i = lid.x; i < dims.hidden; i = i + RWG) {
    let v = x[base + i];
    sum_sq = sum_sq + v * v;
  }
  scratch[lid.x] = sum_sq;
  workgroupBarrier();

  for (var stride = RWG / 2u; stride > 0u; stride = stride >> 1u) {
    if (lid.x < stride) {
      scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
    }
    workgroupBarrier();
  }

  // Qwen/Llama normalise by the mean square, not the sum, and add eps inside the sqrt.
  let scale = inverseSqrt(scratch[0] / f32(dims.hidden) + dims.eps);
  for (var i = lid.x; i < dims.hidden; i = i + RWG) {
    out[base + i] = x[base + i] * scale * gain[i];
  }
}
