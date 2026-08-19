// Tree reduction for sum(exp(x - max)), the softmax denominator.
//
// Knowing it means the k candidates read back can be turned into *exact* probabilities
// over the full vocabulary, rather than probabilities over the truncated set. That is
// what lets top-p report honestly whether the candidate pool actually covered the
// requested probability mass instead of silently renormalising a truncated tail.
//
// Stage one exponentiates; stage two just sums the partials, hence `apply_exp`.

struct Dims {
  n         : u32,
  out_index : u32,
  apply_exp : u32,
  /** Index of the precomputed max within `stats`. */
  max_index : u32,
};

const RWG : u32 = 256u;

@group(0) @binding(0) var<uniform>             dims   : Dims;
@group(0) @binding(1) var<storage, read>       input  : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
@group(0) @binding(3) var<storage, read>       stats  : array<f32>;

var<workgroup> scratch : array<f32, RWG>;

@compute @workgroup_size(RWG)
fn main(
  @builtin(workgroup_id) wid : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
  @builtin(num_workgroups) nwg : vec3<u32>,
) {
  // Only stage one needs the max. Stage two binds `stats` to an unrelated read-only
  // buffer, because aliasing the output buffer as both writable and readable in one
  // bind group is a validation error.
  var max_value : f32 = 0.0;
  if (dims.apply_exp != 0u) {
    max_value = stats[dims.max_index];
  }
  let chunk = (dims.n + nwg.x - 1u) / nwg.x;
  let start = wid.x * chunk;
  let end = min(start + chunk, dims.n);

  var total : f32 = 0.0;
  for (var i = start + lid.x; i < end; i = i + RWG) {
    if (dims.apply_exp != 0u) {
      total = total + exp(input[i] - max_value);
    } else {
      total = total + input[i];
    }
  }
  scratch[lid.x] = total;
  workgroupBarrier();

  for (var stride = RWG / 2u; stride > 0u; stride = stride >> 1u) {
    if (lid.x < stride) {
      scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
    }
    workgroupBarrier();
  }

  if (lid.x == 0u) {
    output[dims.out_index + wid.x] = scratch[0];
  }
}
