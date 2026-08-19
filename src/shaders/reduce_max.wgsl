// Tree reduction for the maximum of an array.
//
// Used in two stages with the same code: first over the 152k logits into one partial per
// workgroup, then over those partials into a single value. `out_index` places the result
// so the second stage can write into the shared sample-output block.

struct Dims {
  n         : u32,
  out_index : u32,
};

const RWG : u32 = 256u;
const NEG_INF : f32 = -3.0e38;

@group(0) @binding(0) var<uniform>             dims   : Dims;
@group(0) @binding(1) var<storage, read>       input  : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;

var<workgroup> scratch : array<f32, RWG>;

@compute @workgroup_size(RWG)
fn main(
  @builtin(workgroup_id) wid : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
  @builtin(num_workgroups) nwg : vec3<u32>,
) {
  let chunk = (dims.n + nwg.x - 1u) / nwg.x;
  let start = wid.x * chunk;
  let end = min(start + chunk, dims.n);

  var best : f32 = NEG_INF;
  for (var i = start + lid.x; i < end; i = i + RWG) {
    best = max(best, input[i]);
  }
  scratch[lid.x] = best;
  workgroupBarrier();

  for (var stride = RWG / 2u; stride > 0u; stride = stride >> 1u) {
    if (lid.x < stride) {
      scratch[lid.x] = max(scratch[lid.x], scratch[lid.x + stride]);
    }
    workgroupBarrier();
  }

  if (lid.x == 0u) {
    output[dims.out_index + wid.x] = scratch[0];
  }
}
