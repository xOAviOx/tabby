// Append freshly projected K or V into the KV cache.
//
// The cache is laid out [layer, position, kv_head, head_dim] and the source is
// [n_new, kv_head, head_dim], so the copy is contiguous and the kernel is a plain
// offset move. It exists as a kernel rather than a copyBufferToBuffer so it can sit
// inside the compute pass -- a copy would force the pass to close and reopen once per
// layer per tensor, 48 times a prefill.

struct Dims {
  // n_new * kv_heads * head_dim
  n        : u32,
  // Element offset of (this layer, pos_start) within the cache.
  dst_start : u32,
};

override wg_size : u32 = 64u;

@group(0) @binding(0) var<uniform>             dims : Dims;
@group(0) @binding(1) var<storage, read>       src  : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst  : array<f32>;

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= dims.n) {
    return;
  }
  dst[dims.dst_start + i] = src[i];
}
