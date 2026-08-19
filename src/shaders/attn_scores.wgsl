// scores[h, i, j] = dot(q[i, h, :], k[j, kv_head(h), :]) * scale, causally masked.
//
// Serves both paths. `i` indexes the *new* query positions being computed; `j` indexes
// every key position including those already in the cache. The whole-sequence M2 form is
// the degenerate case pos_start = 0, total_len = n_new, kv_offset = 0.
//
// Grouped-query attention: several query heads share one KV head. With 14 query heads
// and 2 KV heads, heads 0..6 read KV head 0 and heads 7..13 read KV head 1.
//
// Masked positions are written as a large negative value rather than skipped, so the
// softmax that follows can run over a fixed row length.

struct Dims {
  // Query positions being computed now: the whole prompt in prefill, 1 in decode.
  n_new      : u32,
  // Key positions available: pos_start + n_new.
  total_len  : u32,
  n_heads    : u32,
  n_kv_heads : u32,
  head_dim   : u32,
  // Absolute position of query 0, i.e. how many tokens are already cached.
  pos_start  : u32,
  // Element offset of this layer's slice within the KV cache buffer.
  kv_offset  : u32,
  scale      : f32,
};

override wg_size : u32 = 64u;

const NEG_INF : f32 = -3.0e38;

@group(0) @binding(0) var<uniform>             dims   : Dims;
@group(0) @binding(1) var<storage, read>       q      : array<f32>;  // [n_new, n_heads, head_dim]
@group(0) @binding(2) var<storage, read>       k      : array<f32>;  // cache: [.., n_kv_heads, head_dim]
@group(0) @binding(3) var<storage, read_write> scores : array<f32>;  // [n_heads, n_new, total_len]

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let j = gid.x;      // key position, absolute
  let i = gid.y;      // query index within the new tokens
  let h = gid.z;      // query head
  if (j >= dims.total_len || i >= dims.n_new || h >= dims.n_heads) {
    return;
  }

  let out_index = (h * dims.n_new + i) * dims.total_len + j;
  if (j > dims.pos_start + i) {
    scores[out_index] = NEG_INF;
    return;
  }

  let group = dims.n_heads / dims.n_kv_heads;
  let kv_head = h / group;

  let q_base = (i * dims.n_heads + h) * dims.head_dim;
  let k_base = dims.kv_offset + (j * dims.n_kv_heads + kv_head) * dims.head_dim;
  var acc : f32 = 0.0;
  for (var d : u32 = 0u; d < dims.head_dim; d = d + 1u) {
    acc = acc + q[q_base + d] * k[k_base + d];
  }
  scores[out_index] = acc * dims.scale;
}
