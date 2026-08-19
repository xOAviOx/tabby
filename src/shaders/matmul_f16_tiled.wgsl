// Tiled prefill matmul: y[t, m] = sum_k w[m, k] * x[t, k] + bias[m]
//
// Same result as matmul_f16.wgsl, different memory behaviour. The naive kernel re-reads
// the whole weight row for every token position, and weights are by far the largest
// thing moving -- 942 MiB of them against a few hundred KB of activations. Here each
// workgroup covers TILE_T token positions at once, so a weight row is fetched once and
// used TILE_T times, cutting weight traffic by that factor.
//
// The activation tile is staged in workgroup memory: TILE_T * TILE_K f32, 4 KiB at the
// defaults, which fits the 16 KiB floor every adapter guarantees rather than the 32 KiB
// this machine happens to grant.
//
// Decode (one position) gains nothing from this and keeps using matmul_f16.wgsl.

struct Dims {
  n_tokens : u32,
  // Rows held by THIS shard, not by the whole tensor.
  out_dim  : u32,
  in_dim   : u32,
  has_bias : u32,
  // Index of this shard's first row within the full tensor.
  row_start  : u32,
  // Full output dimension, i.e. the row stride of y.
  out_stride : u32,
};

override wg_size : u32 = 64u;

// Token positions per workgroup, and the reduction chunk staged per iteration.
const TILE_T : u32 = 4u;
const TILE_K : u32 = 256u;

@group(0) @binding(0) var<uniform>             dims : Dims;
@group(0) @binding(1) var<storage, read>       w    : array<u32>;  // f16 [out_dim, in_dim]
@group(0) @binding(2) var<storage, read>       x    : array<f32>;  // [n_tokens, in_dim]
@group(0) @binding(3) var<storage, read>       bias : array<f32>;  // [out_stride] or unused
@group(0) @binding(4) var<storage, read_write> y    : array<f32>;  // [n_tokens, out_stride]

var<workgroup> x_tile : array<f32, TILE_T * TILE_K>;

fn load_f16(i : u32) -> f32 {
  let pair = unpack2x16float(w[i >> 1u]);
  return select(pair.x, pair.y, (i & 1u) == 1u);
}

@compute @workgroup_size(wg_size)
fn main(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
) {
  let m = gid.x;
  let t0 = gid.y * TILE_T;

  var acc : array<f32, TILE_T>;
  for (var s : u32 = 0u; s < TILE_T; s = s + 1u) {
    acc[s] = 0.0;
  }

  let in_range = m < dims.out_dim;
  let w_base = m * dims.in_dim;

  for (var k0 : u32 = 0u; k0 < dims.in_dim; k0 = k0 + TILE_K) {
    let span = min(TILE_K, dims.in_dim - k0);

    // Cooperative stage. Every invocation participates, including ones whose output row
    // is out of range -- they still have to reach the barriers.
    for (var s : u32 = 0u; s < TILE_T; s = s + 1u) {
      let t = t0 + s;
      for (var i : u32 = lid.x; i < span; i = i + wg_size) {
        var value : f32 = 0.0;
        if (t < dims.n_tokens) {
          value = x[t * dims.in_dim + k0 + i];
        }
        x_tile[s * TILE_K + i] = value;
      }
    }
    workgroupBarrier();

    if (in_range) {
      for (var i : u32 = 0u; i < span; i = i + 1u) {
        let weight = load_f16(w_base + k0 + i);
        for (var s : u32 = 0u; s < TILE_T; s = s + 1u) {
          acc[s] = acc[s] + weight * x_tile[s * TILE_K + i];
        }
      }
    }
    // Guards the next iteration's writes against threads still reading this one.
    workgroupBarrier();
  }

  if (!in_range) {
    return;
  }

  let row = dims.row_start + m;
  var bias_value : f32 = 0.0;
  if (dims.has_bias != 0u) {
    bias_value = bias[row];
  }
  for (var s : u32 = 0u; s < TILE_T; s = s + 1u) {
    let t = t0 + s;
    if (t < dims.n_tokens) {
      y[t * dims.out_stride + row] = acc[s] + bias_value;
    }
  }
}
