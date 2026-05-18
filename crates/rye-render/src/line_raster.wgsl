// Line rasterizer: quad-expanded antialiased lines.
//
// Per-instance data is a line segment (start, end, colors, width). The vertex shader runs four
// times per instance (one per quad corner), expanding the line into a screen-space-oriented quad
// that's `width + 2 px` wide perpendicular to the line. The fragment shader smoothsteps coverage
// from `coverage_t == 0` (line center) to `coverage_t == ±1` (expanded edge), producing a
// 1-pixel AA falloff at the silhouette.

struct CameraUniform {
    view_projection: mat4x4<f32>,
    viewport_size:   vec2<f32>,
    _pad:            vec2<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VsOut {
    @builtin(position) clip:        vec4<f32>,
    @location(0)       coverage_t:  f32,
    @location(1)       width_px:    f32,
    @location(2)       color:       vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) corner:      u32,
    @location(1) start_pos:   vec3<f32>,
    @location(2) end_pos:     vec3<f32>,
    @location(3) start_color: vec4<f32>,
    @location(4) end_color:   vec4<f32>,
    @location(5) width_px:    f32,
) -> VsOut {
    let s_clip = camera.view_projection * vec4<f32>(start_pos, 1.0);
    let e_clip = camera.view_projection * vec4<f32>(end_pos,   1.0);
    let s_ndc  = s_clip.xyz / s_clip.w;
    let e_ndc  = e_clip.xyz / e_clip.w;

    // Corners 0, 2 belong to the start endpoint; 1, 3 to the end.
    let pick_start = (corner == 0u || corner == 2u);
    let base_ndc   = select(e_ndc, s_ndc, pick_start);
    let base_w     = select(e_clip.w, s_clip.w, pick_start);
    let color      = select(end_color, start_color, pick_start);

    // Direction along the segment in screen-pixel space (NDC * half-viewport).
    let half_vp     = camera.viewport_size * 0.5;
    let dir_pixels  = (e_ndc.xy - s_ndc.xy) * half_vp;
    let dir_pixels_safe = select(dir_pixels, vec2<f32>(1.0, 0.0), length(dir_pixels) < 1.0e-6);
    let dir2        = normalize(dir_pixels_safe);
    let perp2       = vec2<f32>(-dir2.y, dir2.x);

    // Perpendicular offset in pixels, converted back to NDC.
    let sign        = select(-1.0, 1.0, corner >= 2u);
    let half_with_aa = width_px * 0.5 + 1.0;  // +1 px AA falloff margin
    let perp_ndc    = perp2 / half_vp;
    let off_ndc     = perp_ndc * sign * half_with_aa;

    var out: VsOut;
    // Re-multiply by w so the hardware perspective divide recovers our NDC values exactly.
    // Depth (base_ndc.z) propagates from the projected endpoint so depth-tested compositing
    // (when the render pass attaches a depth buffer) sees correct line depth per fragment.
    out.clip = vec4<f32>(
        (base_ndc.xy + off_ndc) * base_w,
        base_ndc.z * base_w,
        base_w,
    );
    out.coverage_t = sign;
    out.width_px   = width_px;
    out.color      = color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // |coverage_t| is 0 at line center, 1 at the expanded edge. With +1 px AA margin baked
    // into the quad, the inner solid region runs from 0 to (width - 1) / (width + 1), and
    // the last pixel transitions to fully transparent.
    let inner   = max(0.0, (in.width_px - 1.0) / (in.width_px + 1.0));
    let coverage = 1.0 - smoothstep(inner, 1.0, abs(in.coverage_t));
    return vec4<f32>(in.color.rgb, in.color.a * coverage);
}
