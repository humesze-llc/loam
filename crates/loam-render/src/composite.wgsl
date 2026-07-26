// Final composite pass for browser-WebGPU. Samples the offscreen scene texture,
// applies manual sRGB ENCODING, and writes the result to a linear-format
// swapchain texture. The scene half of that texture reads back linear either
// way, by sampler decode where the format is an sRGB one or by storage where
// it is not. The egui-painted half does not, on a canvas whose format has no
// sRGB sibling: egui-wgpu writes already-encoded values there and this pass
// encodes them again. See `RenderDevice`'s module doc for that arm.
//
// Why: Chrome's WebGPU canvas advertises only linear surface formats
// (Bgra8Unorm / Rgba8Unorm / Rgba16Float). Without this manual gamma encode the
// linear shader output displays uncorrected by the WebGPU canvas compositor
// (which expects sRGB-encoded bits in the swapchain), so everything reads ~2.2x
// darker than native. The accurate sRGB curve is piecewise (linear segment near
// 0, exponential elsewhere); we use the canonical form rather than the
// `pow(x, 1.0/2.2)` approximation so neutral grays land where the eye expects.

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) idx: u32) -> VsOut {
    // Fullscreen triangle covering [-1, 1] in NDC. One degenerate triangle is
    // cheaper than two; the bottom-right corner extends beyond the viewport and
    // gets clipped.
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var out: VsOut;
    out.clip_pos = vec4<f32>(positions[idx], 0.0, 1.0);
    out.uv = uvs[idx];
    return out;
}

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;

fn linear_to_srgb(c: f32) -> f32 {
    // IEC 61966-2-1 sRGB transfer function (linear -> non-linear). Piecewise
    // linear-near-zero, exponential elsewhere.
    if (c <= 0.0031308) {
        return 12.92 * c;
    }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
    // Scene samples arrive linear: `RenderDevice` gives scene_tex the canvas
    // format's sRGB sibling where it has one (`Bgra8UnormSrgb`, auto-decoded
    // here) and a linear-storage format otherwise (`Rgba16Float`, no decode
    // needed). Egui-painted texels on the no-sibling arm arrive encoded. We
    // re-encode to sRGB manually and write to a linear swapchain so the WebGPU
    // canvas compositor sees sRGB-encoded bits.
    let linear = textureSample(scene_tex, scene_sampler, in.uv);
    return vec4<f32>(
        linear_to_srgb(linear.r),
        linear_to_srgb(linear.g),
        linear_to_srgb(linear.b),
        linear.a,
    );
}
