// In-place softmax over the last dimension of a [n_rows, row_len] tensor.
//
// Two passes over the row (max, then exp-sum), one invocation per row. M5 replaces this
// with an online/streaming softmax; here the plain form is the one that is obviously
// correct.

struct Dims {
  n_rows  : u32,
  row_len : u32,
  // Rows are laid out as (head, query). Row r covers query index r % causal_period, whose
  // absolute position is causal_offset + that index, so entries beyond it are masked.
  // Set causal_period to 0 to softmax the whole row.
  causal_period : u32,
  causal_offset : u32,
};

override wg_size : u32 = 64u;

@group(0) @binding(0) var<uniform>             dims : Dims;
@group(0) @binding(1) var<storage, read_write> x    : array<f32>;

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let row = gid.x;
  if (row >= dims.n_rows) {
    return;
  }

  var valid = dims.row_len;
  if (dims.causal_period != 0u) {
    valid = dims.causal_offset + (row % dims.causal_period) + 1u;
  }

  let base = row * dims.row_len;
  var max_v : f32 = x[base];
  for (var i : u32 = 1u; i < valid; i = i + 1u) {
    max_v = max(max_v, x[base + i]);
  }

  var sum : f32 = 0.0;
  for (var i : u32 = 0u; i < valid; i = i + 1u) {
    let e = exp(x[base + i] - max_v);
    x[base + i] = e;
    sum = sum + e;
  }

  let inv = 1.0 / sum;
  for (var i : u32 = 0u; i < valid; i = i + 1u) {
    x[base + i] = x[base + i] * inv;
  }
  // Masked tail is zeroed so nothing downstream reads stale scores.
  for (var i : u32 = valid; i < dims.row_len; i = i + 1u) {
    x[base + i] = 0.0;
  }
}
