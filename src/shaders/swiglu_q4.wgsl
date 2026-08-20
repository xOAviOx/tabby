// out[t, m] = silu(gate[m] . x[t]) * (up[m] . x[t]), with int4 block-quantized weights.
//
// Fuses three dispatches -- gate_proj, up_proj and silu_mul -- into one. The two
// projections consume the *same* activation vector and produce the same row index, so
// running them together halves the activation traffic, removes a full round trip of the
// intermediate `up` buffer through global memory, and cuts 72 dispatches per token to 24.
//
// Structure matches matvec_q4.wgsl: vec4 loads, lanes partitioned across rows, one
// reduction per row inside its own lane group. Only the epilogue differs.

struct Dims {
  n_tokens : u32,
  n_rows   : u32,
  n_cols   : u32,
  block_size : u32,
};

override wg_size : u32 = 64u;
override rows_per_wg : u32 = 8u;

/** Weights per vec4<u32> load. */
const PER_QUAD : u32 = 32u;

@group(0) @binding(0) var<uniform>             dims       : Dims;
@group(0) @binding(1) var<storage, read>       gate_q     : array<vec4<u32>>;
@group(0) @binding(2) var<storage, read>       gate_s     : array<u32>;  // f16 pairs
@group(0) @binding(3) var<storage, read>       up_q       : array<vec4<u32>>;
@group(0) @binding(4) var<storage, read>       up_s       : array<u32>;  // f16 pairs
@group(0) @binding(5) var<storage, read>       x          : array<f32>;
@group(0) @binding(6) var<storage, read_write> out        : array<f32>;

var<workgroup> partial_gate : array<f32, wg_size>;
var<workgroup> partial_up : array<f32, wg_size>;

fn scale_of(packed : u32, index : u32) -> f32 {
  let pair = unpack2x16float(packed);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}

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

fn quad_sum(packed : vec4<u32>, k0 : u32) -> f32 {
  return dot8(packed.x, k0) +
         dot8(packed.y, k0 + 8u) +
         dot8(packed.z, k0 + 16u) +
         dot8(packed.w, k0 + 24u);
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

  var acc_gate : f32 = 0.0;
  var acc_up : f32 = 0.0;
  if (row < dims.n_rows) {
    for (var q = lane; q < quads_per_row; q = q + lanes_per_row) {
      let k0 = x_base + q * PER_QUAD;
      let scale_index = s_base + q / quads_per_block;

      // One activation read serves both projections.
      acc_gate = acc_gate +
        quad_sum(gate_q[q_base + q], k0) * scale_of(gate_s[scale_index >> 1u], scale_index);
      acc_up = acc_up +
        quad_sum(up_q[q_base + q], k0) * scale_of(up_s[scale_index >> 1u], scale_index);
    }
  }
  partial_gate[lid.x] = acc_gate;
  partial_up[lid.x] = acc_up;
  workgroupBarrier();

  for (var stride = lanes_per_row / 2u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) {
      partial_gate[lid.x] = partial_gate[lid.x] + partial_gate[lid.x + stride];
      partial_up[lid.x] = partial_up[lid.x] + partial_up[lid.x + stride];
    }
    workgroupBarrier();
  }

  if (lane == 0u && row < dims.n_rows) {
    let g = partial_gate[lid.x];
    // silu(v) = v * sigmoid(v)
    out[wid.y * dims.n_rows + row] = (g / (1.0 + exp(-g))) * partial_up[lid.x];
  }
}
