// scores[h, i, j] = dot(q[i, h, :], k[j, kv_head(h), :]) / sqrt(head_dim), causally masked.
//
// Grouped-query attention: several query heads share one KV head. With 14 query heads
// and 2 KV heads, heads 0..6 read KV head 0 and heads 7..13 read KV head 1.
//
// Masked positions are written as a large negative value rather than skipped, so the
// softmax that follows can run over a fixed row length.

struct Dims {
  n_tokens    : u32,
  n_heads     : u32,
  n_kv_heads  : u32,
  head_dim    : u32,
  scale       : f32,
};

override wg_size : u32 = 64u;

const NEG_INF : f32 = -3.0e38;

@group(0) @binding(0) var<uniform>             dims   : Dims;
@group(0) @binding(1) var<storage, read>       q      : array<f32>;  // [n_tokens, n_heads, head_dim]
@group(0) @binding(2) var<storage, read>       k      : array<f32>;  // [n_tokens, n_kv_heads, head_dim]
@group(0) @binding(3) var<storage, read_write> scores : array<f32>;  // [n_heads, n_tokens, n_tokens]

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let j = gid.x;      // key position
  let i = gid.y;      // query position
  let h = gid.z;      // query head
  if (j >= dims.n_tokens || i >= dims.n_tokens || h >= dims.n_heads) {
    return;
  }

  let out_index = (h * dims.n_tokens + i) * dims.n_tokens + j;
  if (j > i) {
    scores[out_index] = NEG_INF;
    return;
  }

  let group = dims.n_heads / dims.n_kv_heads;
  let kv_head = h / group;

  let q_base = (i * dims.n_heads + h) * dims.head_dim;
  let k_base = (j * dims.n_kv_heads + kv_head) * dims.head_dim;
  var acc : f32 = 0.0;
  for (var d : u32 = 0u; d < dims.head_dim; d = d + 1u) {
    acc = acc + q[q_base + d] * k[k_base + d];
  }
  scores[out_index] = acc * dims.scale;
}
