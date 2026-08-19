// Rotary position embedding, applied in place to a [n_tokens, n_heads, head_dim] tensor.
//
// This is the "rotate half" convention Llama and Qwen use: the head vector is split down
// the middle and element i is paired with element i + head_dim/2. It is NOT the
// interleaved (i, i+1) pairing used by some other RoPE formulations -- getting this wrong
// still produces plausible-looking activations, so it is worth stating.
//
//   out[i]        = v[i] * cos - v[i + h/2] * sin
//   out[i + h/2]  = v[i + h/2] * cos + v[i] * sin
//   theta_i       = base ^ (-2i / head_dim)

struct Dims {
  n_tokens  : u32,
  n_heads   : u32,
  head_dim  : u32,
  // Index of the first token in the sequence. 0 for M2's full-sequence pass; the KV
  // cache in M3 needs a non-zero start.
  pos_start : u32,
  theta     : f32,
};

override wg_size : u32 = 64u;

@group(0) @binding(0) var<uniform>             dims : Dims;
@group(0) @binding(1) var<storage, read_write> v    : array<f32>;

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let half_dim = dims.head_dim / 2u;
  let lane = gid.x;                 // head * half_dim + pair index
  let t = gid.y;
  if (t >= dims.n_tokens || lane >= dims.n_heads * half_dim) {
    return;
  }

  let head = lane / half_dim;
  let i = lane % half_dim;

  let exponent = -2.0 * f32(i) / f32(dims.head_dim);
  let freq = pow(dims.theta, exponent);
  let angle = f32(dims.pos_start + t) * freq;
  let cos_a = cos(angle);
  let sin_a = sin(angle);

  let base = (t * dims.n_heads + head) * dims.head_dim;
  let lo = v[base + i];
  let hi = v[base + i + half_dim];
  v[base + i] = lo * cos_a - hi * sin_a;
  v[base + i + half_dim] = hi * cos_a + lo * sin_a;
}
