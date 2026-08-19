// y[t, m] = sum_k w[m, k] * x[t, k] + bias[m]
//
// w is f16 row-major [out_dim, in_dim]; x and y are f32. Dispatch is 2D: x over output
// columns, y over token positions. That keeps the X dimension at ceil(out_dim/wg_size)
// rather than out_dim, which matters because the 151,936-row lm_head would otherwise
// need more workgroups than maxComputeWorkgroupsPerDimension (65,535) allows.

struct Dims {
  n_tokens : u32,
  // Rows held by THIS shard, not by the whole tensor.
  out_dim  : u32,
  in_dim   : u32,
  has_bias : u32,
  // Index of this shard's first row within the full tensor. A tensor too large for one
  // storage binding is split into row blocks, and each block is dispatched separately.
  row_start  : u32,
  // Full output dimension, i.e. the row stride of y.
  out_stride : u32,
};

override wg_size : u32 = 64u;

@group(0) @binding(0) var<uniform>             dims : Dims;
@group(0) @binding(1) var<storage, read>       w    : array<u32>;  // f16 [out_dim, in_dim]
@group(0) @binding(2) var<storage, read>       x    : array<f32>;  // [n_tokens, in_dim]
@group(0) @binding(3) var<storage, read>       bias : array<f32>;  // [out_dim] or unused
@group(0) @binding(4) var<storage, read_write> y    : array<f32>;  // [n_tokens, out_dim]

// f16 weights are bound as u32 and unpacked with unpack2x16float, which is core WGSL.
// Reading them as native f16 would require the optional shader-f16 feature and a second
// code path; this way one kernel runs everywhere. Element i lives in word i>>1, low half
// when i is even. M5 can add an f16-native variant once it is measuring.
fn load_f16(i : u32) -> f32 {
  let pair = unpack2x16float(w[i >> 1u]);
  return select(pair.x, pair.y, (i & 1u) == 1u);
}

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let m = gid.x;
  let t = gid.y;
  if (m >= dims.out_dim || t >= dims.n_tokens) {
    return;
  }

  let row = dims.row_start + m;
  let w_base = m * dims.in_dim;
  let x_base = t * dims.in_dim;
  var acc : f32 = 0.0;
  for (var k : u32 = 0u; k < dims.in_dim; k = k + 1u) {
    acc = acc + load_f16(w_base + k) * x[x_base + k];
  }
  if (dims.has_bias != 0u) {
    acc = acc + bias[row];
  }
  y[t * dims.out_stride + row] = acc;
}
