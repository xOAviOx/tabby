// out[t, d] = embedding[ids[t], d]
//
// The embedding table is f16; everything downstream of here is f32.

struct Dims {
  n_tokens : u32,
  hidden   : u32,
  // Rows of the embedding table held by this shard. The table is the one tensor that
  // routinely exceeds a storage binding, so each shard is dispatched separately and an
  // invocation writes only when its token falls inside this shard's row range.
  row_start : u32,
  row_count : u32,
};

override wg_size : u32 = 64u;

@group(0) @binding(0) var<uniform>             dims  : Dims;
@group(0) @binding(1) var<storage, read>       table : array<u32>;  // f16 [vocab, hidden]
@group(0) @binding(2) var<storage, read>       ids   : array<u32>;  // [n_tokens]
@group(0) @binding(3) var<storage, read_write> out   : array<f32>;  // [n_tokens, hidden]

fn load_f16(i : u32) -> f32 {
  let pair = unpack2x16float(table[i >> 1u]);
  return select(pair.x, pair.y, (i & 1u) == 1u);
}

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let d = gid.x;
  let t = gid.y;
  if (d >= dims.hidden || t >= dims.n_tokens) {
    return;
  }
  let id = ids[t];
  if (id < dims.row_start || id >= dims.row_start + dims.row_count) {
    return;
  }
  out[t * dims.hidden + d] = load_f16((id - dims.row_start) * dims.hidden + d);
}
