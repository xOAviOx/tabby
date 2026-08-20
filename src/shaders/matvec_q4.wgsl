// y[t, m] = sum_k dequant(w[m, k]) * x[t, k] + bias[m], with int4 block-quantized w.
//
// This is the kernel that decides the project's tok/s. Decode is memory-bandwidth bound:
// every token streams the whole weight matrix through the GPU once and does almost no
// arithmetic per byte, so throughput is set by how efficiently these bytes are read.
//
// Weights are never materialised as f16 anywhere. A packed u32 carries eight int4 values
// and is dequantized in registers as it is consumed -- expanding to f16 first would put
// the 4x bandwidth saving straight back.
//
// Layout, matching tools/convert.py:
//   row r occupies cols/8 consecutive u32 words
//   word w holds elements [8w, 8w+8), nibble j at bits 4j
//   block_size weights share one f16 scale; dequant(q) = (q - 8) * scale
//
// Two structural choices, both measured rather than assumed (see PROGRESS.md):
//
//   * Loads are `vec4<u32>` -- 16 bytes, 32 weights, and with the default block size
//     exactly one quantization block, so one scale load serves one vector load.
//   * Lanes are *partitioned* across the workgroup's rows rather than every lane working
//     on every row. Each row is reduced within its own lane group, so the reductions run
//     concurrently and cost log2(lanes_per_row) barriers instead of one pass per row.
//
// Staging activations in workgroup memory was tried and *removed*: it cost 86.5 -> 84.5
// tok/s. The activation vector is small enough to stay cache-resident, so the barriers it
// adds cost more than the global reads it avoids. See the optimization table in
// PROGRESS.md.

struct Dims {
  // Token positions. 1 when decoding.
  n_tokens : u32,
  // Output rows held by this shard.
  n_rows   : u32,
  // Reduction dimension.
  n_cols   : u32,
  block_size : u32,
  has_bias   : u32,
  row_start  : u32,
  out_stride : u32,
};

override wg_size : u32 = 64u;
/** Output rows each workgroup covers. */
override rows_per_wg : u32 = 8u;

/** Weights per vec4<u32> load. */
const PER_QUAD : u32 = 32u;

@group(0) @binding(0) var<uniform>             dims    : Dims;
@group(0) @binding(1) var<storage, read>       qweight : array<vec4<u32>>;
@group(0) @binding(2) var<storage, read>       scales  : array<u32>;  // f16 pairs
@group(0) @binding(3) var<storage, read>       x       : array<f32>;
@group(0) @binding(4) var<storage, read>       bias    : array<f32>;
@group(0) @binding(5) var<storage, read_write> y       : array<f32>;

var<workgroup> partial : array<f32, wg_size>;

fn load_scale(index : u32) -> f32 {
  let pair = unpack2x16float(scales[index >> 1u]);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}

/** Eight int4 weights against eight activations. The block scale is applied by the caller. */
fn dot8(word : u32, base : u32) -> f32 {
  return (f32((word       ) & 15u) - 8.0) * x[base     ]
       + (f32((word >>  4u) & 15u) - 8.0) * x[base + 1u]
       + (f32((word >>  8u) & 15u) - 8.0) * x[base + 2u]
       + (f32((word >> 12u) & 15u) - 8.0) * x[base + 3u]
       + (f32((word >> 16u) & 15u) - 8.0) * x[base + 4u]
       + (f32((word >> 20u) & 15u) - 8.0) * x[base + 5u]
       + (f32((word >> 24u) & 15u) - 8.0) * x[base + 6u]
       + (f32((word >> 28u) & 15u) - 8.0) * x[base + 7u];
}

@compute @workgroup_size(wg_size)
fn main(
  @builtin(workgroup_id) wid : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
) {
  let lanes_per_row = wg_size / rows_per_wg;
  let local_row = lid.x / lanes_per_row;
  let lane = lid.x % lanes_per_row;

  let row = wid.x * rows_per_wg + local_row;
  let x_base = wid.y * dims.n_cols;

  let quads_per_row = dims.n_cols / PER_QUAD;
  let blocks_per_row = dims.n_cols / dims.block_size;
  let quads_per_block = dims.block_size / PER_QUAD;

  let q_base = row * quads_per_row;
  let s_base = row * blocks_per_row;

  var acc : f32 = 0.0;
  if (row < dims.n_rows) {
    // Adjacent lanes take adjacent vec4s, so a wavefront's loads stay contiguous.
    for (var q = lane; q < quads_per_row; q = q + lanes_per_row) {
      let packed = qweight[q_base + q];
      let scale = load_scale(s_base + q / quads_per_block);
      let k0 = x_base + q * PER_QUAD;
      let block_sum =
        dot8(packed.x, k0) +
        dot8(packed.y, k0 + 8u) +
        dot8(packed.z, k0 + 16u) +
        dot8(packed.w, k0 + 24u);
      acc = acc + block_sum * scale;
    }
  }

  partial[lid.x] = acc;
  workgroupBarrier();

  // Each row reduces inside its own lane group; the groups do not interfere, so all
  // rows_per_wg reductions proceed together.
  for (var stride = lanes_per_row / 2u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) {
      partial[lid.x] = partial[lid.x] + partial[lid.x + stride];
    }
    workgroupBarrier();
  }

  if (lane == 0u && row < dims.n_rows) {
    let out_row = dims.row_start + row;
    var value = partial[lid.x];
    if (dims.has_bias != 0u) {
      value = value + bias[out_row];
    }
    y[wid.y * dims.out_stride + out_row] = value;
  }
}
