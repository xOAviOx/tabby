// out[i, h, d] = sum_j scores[h, i, j] * v[j, kv_head(h), d]
//
// Same generalisation as attn_scores: `i` is a new query position, `j` runs over every
// cached key position. Masked entries are already zero after the softmax, but the loop
// stops at the causal bound anyway so decode does not read uninitialised cache.

struct Dims {
  n_new      : u32,
  total_len  : u32,
  n_heads    : u32,
  n_kv_heads : u32,
  head_dim   : u32,
  pos_start  : u32,
  kv_offset  : u32,
};

override wg_size : u32 = 64u;

@group(0) @binding(0) var<uniform>             dims   : Dims;
@group(0) @binding(1) var<storage, read>       scores : array<f32>;  // [n_heads, n_new, total_len]
@group(0) @binding(2) var<storage, read>       v      : array<f32>;  // cache
@group(0) @binding(3) var<storage, read_write> out    : array<f32>;  // [n_new, n_heads, head_dim]

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let lane = gid.x;   // head * head_dim + d
  let i = gid.y;      // query index within the new tokens
  if (i >= dims.n_new || lane >= dims.n_heads * dims.head_dim) {
    return;
  }

  let h = lane / dims.head_dim;
  let d = lane % dims.head_dim;
  let group = dims.n_heads / dims.n_kv_heads;
  let kv_head = h / group;

  let score_base = (h * dims.n_new + i) * dims.total_len;
  let last = dims.pos_start + i;
  var acc : f32 = 0.0;
  for (var j : u32 = 0u; j <= last; j = j + 1u) {
    let v_base = dims.kv_offset + (j * dims.n_kv_heads + kv_head) * dims.head_dim;
    acc = acc + scores[score_base + j] * v[v_base + d];
  }
  out[(i * dims.n_heads + h) * dims.head_dim + d] = acc;
}
