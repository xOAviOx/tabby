// x += delta, elementwise.

struct Dims {
  n : u32,
};

override wg_size : u32 = 64u;

@group(0) @binding(0) var<uniform>             dims  : Dims;
@group(0) @binding(1) var<storage, read>       delta : array<f32>;
@group(0) @binding(2) var<storage, read_write> x     : array<f32>;

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= dims.n) {
    return;
  }
  x[i] = x[i] + delta[i];
}
