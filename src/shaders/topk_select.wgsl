// Reduce the per-workgroup maxima to one winner, record it, and mask it out.
//
// Runs as a single workgroup after each topk_partial pass. Writing NEG_INF back into the
// working copy is what makes the next round return the next-largest value; the original
// logits are untouched because the copy is what gets masked.
//
// The output block is laid out as [max, sumexp, (value, index) * k] so the whole sample
// result is one small readback -- ~330 bytes for k = 40 against 608 KB for the full
// logit vector.

struct Dims {
  /** Number of partials produced by topk_partial. */
  n_partials : u32,
  /** Which of the k slots this round fills. */
  step       : u32,
  /** Offset of the first (value, index) pair within the output block. */
  out_base   : u32,
};

const RWG : u32 = 256u;
const NEG_INF : f32 = -3.0e38;

@group(0) @binding(0) var<uniform>             dims    : Dims;
@group(0) @binding(1) var<storage, read>       values  : array<f32>;
@group(0) @binding(2) var<storage, read>       indices : array<u32>;
@group(0) @binding(3) var<storage, read_write> out     : array<f32>;
@group(0) @binding(4) var<storage, read_write> work    : array<f32>;

var<workgroup> best_value : array<f32, RWG>;
var<workgroup> best_index : array<u32, RWG>;

@compute @workgroup_size(RWG)
fn main(@builtin(local_invocation_id) lid : vec3<u32>) {
  var value : f32 = NEG_INF;
  var index : u32 = 0u;
  for (var i = lid.x; i < dims.n_partials; i = i + RWG) {
    if (values[i] > value || (values[i] == value && indices[i] < index)) {
      value = values[i];
      index = indices[i];
    }
  }
  best_value[lid.x] = value;
  best_index[lid.x] = index;
  workgroupBarrier();

  for (var stride = RWG / 2u; stride > 0u; stride = stride >> 1u) {
    if (lid.x < stride) {
      let other = lid.x + stride;
      let take = best_value[other] > best_value[lid.x] ||
        (best_value[other] == best_value[lid.x] && best_index[other] < best_index[lid.x]);
      if (take) {
        best_value[lid.x] = best_value[other];
        best_index[lid.x] = best_index[other];
      }
    }
    workgroupBarrier();
  }

  if (lid.x == 0u) {
    let slot = dims.out_base + dims.step * 2u;
    out[slot] = best_value[0];
    // The index is a u32 reinterpreted through an f32 array; the caller reads the same
    // buffer as a Uint32Array to recover it.
    out[slot + 1u] = bitcast<f32>(best_index[0]);
    work[best_index[0]] = NEG_INF;
  }
}
