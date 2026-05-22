// Point rasterizer: quad-expanded antialiased screen-space discs.
//
// Per-instance data is a point (position in R³, color, screen-space pixel radius). The vertex
// shader runs four times per instance (one per quad corner), expanding the point into a
// camera-facing square sized to fit the disc plus a 1-px AA falloff margin. The fragment shader
// computes distance from the disc center via the interpolated `[-1, 1]` quad-local UV and
// smoothsteps from solid interior to transparent edge.
//
// Same `(view_projection, viewport_size)` uniform shape as `line_raster.wgsl` because both
// pipelines need pixel-to-NDC conversion; structurally identical struct, kept separate so the
// two pipelines can evolve their uniforms independently if needed.

struct CameraUniform {
    view_projection: mat4x4<f32>,
    viewport_size:   vec2<f32>,
    _pad:            vec2<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VsOut {
    @builtin(position) clip:    vec4<f32>,
    // Quad-local coordinates in `[-1, 1]^2`. The disc occupies the inscribed circle.
    @location(0) uv:            vec2<f32>,
    @location(1) color:         vec4<f32>,
    // Pixel radius passes through so the fragment shader can scale its AA falloff width
    // (matches the line shader's `width_px` convention).
    @location(2) radius_px:     f32,
};

@vertex
fn vs_main(
    @location(0) corner:    u32,
    @location(1) pos:       vec3<f32>,
    @location(2) color:     vec4<f32>,
    @location(3) radius_px: f32,
) -> VsOut {
    let p_clip = camera.view_projection * vec4<f32>(pos, 1.0);
    let p_ndc  = p_clip.xyz / p_clip.w;

    // Quad corners map to {-1, +1}^2 in (dx, dy). Same corner-bit pattern as line_raster.
    let dx = select(-1.0, 1.0, corner == 1u || corner == 3u);
    let dy = select(-1.0, 1.0, corner >= 2u);

    // Half-extent in pixels, padded by 1 px for the AA falloff at the disc edge.
    let half_with_aa = radius_px + 1.0;

    // Convert the pixel offset to NDC via the half-viewport scale, same as line_raster.
    let half_vp  = camera.viewport_size * 0.5;
    let off_ndc  = vec2<f32>(dx, dy) * half_with_aa / half_vp;

    var out: VsOut;
    // Re-multiply by w so the hardware perspective divide recovers the intended NDC.
    out.clip = vec4<f32>(
        (p_ndc.xy + off_ndc) * p_clip.w,
        p_ndc.z * p_clip.w,
        p_clip.w,
    );
    out.uv        = vec2<f32>(dx, dy);
    out.color     = color;
    out.radius_px = radius_px;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Distance from disc center in quad-local coordinates. `uv` is in `[-1, 1]^2` and the disc
    // is the inscribed circle of radius `radius_px / (radius_px + 1)` (the +1 px AA margin
    // pushes the quad edge `1 px` outside the disc proper).
    let r = length(in.uv);
    let inner = max(0.0, in.radius_px / (in.radius_px + 1.0));
    let coverage = 1.0 - smoothstep(inner, 1.0, r);
    return vec4<f32>(in.color.rgb, in.color.a * coverage);
}
