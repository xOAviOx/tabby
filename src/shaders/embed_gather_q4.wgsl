// out[t, d] = dequant(table[ids[t], d]) for an int4 block-quantized embedding table.
//
// Same packing as matvec_q4: eight weights per u32, one f16 scale per block of 32.
// A separate kernel from the f16 gather because branching on dtype inside the hot path
// of either would cost more than the duplication does.

struct Dims {
  n_tokens   : u32,
  hidden     : u32,
  block_size : u32,
  row_start  : u32,
  row_count  : u32,
};

override wg_size : u32 = 64u;

@group(0) @binding(0) var<uniform>             dims    : Dims;
@group(0) @binding(1) var<storage, read>       qweight : array<u32>;
@group(0) @binding(2) var<storage, read>       scales  : array<u32>;  // f16 pairs
@group(0) @binding(3) var<storage, read>       ids     : array<u32>;
@group(0) @binding(4) var<storage, read_write> out     : array<f32>;

fn load_scale(index : u32) -> f32 {
  let pair = unpack2x16float(scales[index >> 1u]);
  return select(pair.x, pair.y, (index & 1u) == 1u);
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

  let local_row = id - dims.row_start;
  let words_per_row = dims.hidden / 8u;
  let blocks_per_row = dims.hidden / dims.block_size;

  let packed = qweight[local_row * words_per_row + d / 8u];
  let level = (packed >> ((d % 8u) * 4u)) & 15u;
  let scale = load_scale(local_row * blocks_per_row + d / dims.block_size);
  out[t * dims.hidden + d] = (f32(level) - 8.0) * scale;
}
