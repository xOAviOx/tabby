// out[i, h, d] = sum_j scores[h, i, j] * v[j, kv_head(h), d]
//
// Same grouped-query mapping as attn_scores. Masked entries are already zero, so the
// sum runs over the full row without a branch.

struct Dims {
  n_tokens   : u32,
  n_heads    : u32,
  n_kv_heads : u32,
  head_dim   : u32,
};

override wg_size : u32 = 64u;

@group(0) @binding(0) var<uniform>             dims   : Dims;
@group(0) @binding(1) var<storage, read>       scores : array<f32>;  // [n_heads, n_tokens, n_tokens]
@group(0) @binding(2) var<storage, read>       v      : array<f32>;  // [n_tokens, n_kv_heads, head_dim]
@group(0) @binding(3) var<storage, read_write> out    : array<f32>;  // [n_tokens, n_heads, head_dim]

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let lane = gid.x;   // head * head_dim + d
  let i = gid.y;      // query position
  if (i >= dims.n_tokens || lane >= dims.n_heads * dims.head_dim) {
    return;
  }

  let h = lane / dims.head_dim;
  let d = lane % dims.head_dim;
  let group = dims.n_heads / dims.n_kv_heads;
  let kv_head = h / group;

  let score_base = (h * dims.n_tokens + i) * dims.n_tokens;
  var acc : f32 = 0.0;
  for (var j : u32 = 0u; j <= i; j = j + 1u) {
    acc = acc + scores[score_base + j] * v[(j * dims.n_kv_heads + kv_head) * dims.head_dim + d];
  }
  out[(i * dims.n_heads + h) * dims.head_dim + d] = acc;
}
