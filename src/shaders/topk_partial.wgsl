// One (value, index) maximum per workgroup, over a working copy of the logits.
//
// Paired with topk_select.wgsl and run k times: each round extracts the current global
// maximum and masks it out, so k rounds yield an exact top-k. That is far simpler than a
// histogram or a partial-sort, and the cost is bounded -- k passes over 152k floats is
// about 24 MB of reads for k = 40, well under a millisecond, and none of it crosses the
// PCIe/CPU boundary.

struct Dims {
  n : u32,
};

const RWG : u32 = 256u;
const NEG_INF : f32 = -3.0e38;

@group(0) @binding(0) var<uniform>             dims    : Dims;
@group(0) @binding(1) var<storage, read>       logits  : array<f32>;
@group(0) @binding(2) var<storage, read_write> values  : array<f32>;
@group(0) @binding(3) var<storage, read_write> indices : array<u32>;

var<workgroup> best_value : array<f32, RWG>;
var<workgroup> best_index : array<u32, RWG>;

@compute @workgroup_size(RWG)
fn main(
  @builtin(workgroup_id) wid : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
  @builtin(num_workgroups) nwg : vec3<u32>,
) {
  let chunk = (dims.n + nwg.x - 1u) / nwg.x;
  let start = wid.x * chunk;
  let end = min(start + chunk, dims.n);

  var value : f32 = NEG_INF;
  var index : u32 = 0u;
  for (var i = start + lid.x; i < end; i = i + RWG) {
    let candidate = logits[i];
    // Strictly greater keeps the lowest index on ties, which makes sampling
    // reproducible across runs and across workgroup counts.
    if (candidate > value) {
      value = candidate;
      index = i;
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
    values[wid.x] = best_value[0];
    indices[wid.x] = best_index[0];
  }
}
