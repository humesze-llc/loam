// Line rasterizer: static-mesh R⁴ variant. Mesh vertices live in R⁴, uploaded
// ONCE; per-frame the host writes one uniform (rotor + focal_distance +
// view_projection). vs_main applies rotor, Perspective4D projection, then
// view_projection per endpoint; quad-expansion + AA mirror `line_raster.wgsl`.
//
// Depth-cue coloring: every line gets ONE color from its post-rotor midpoint
// w-coordinate (cool back / warm front), so color tracks 4D depth across lines
// in a rotating polytope. `start_color.rgb` tints it (white = pure palette),
// `start_color.a` passes to the AA blend; `end_color` is ignored on this path.

struct TransformUniform {
    // Host-side `Rotor4::to_mat4()`, column-major.
    rotor_matrix: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    viewport_size: vec2<f32>,
    // Perspective4D focal distance along w; denominator clamped below.
    focal_distance: f32,
    _pad: f32,
};
@group(0) @binding(0) var<uniform> transform: TransformUniform;

struct VsOut {
    @builtin(position) clip:        vec4<f32>,
    @location(0)       coverage_t:  f32,
    @location(1)       width_px:    f32,
    @location(2)       color:       vec4<f32>,
};

// Perspective4D projection along w; mirrors `EuclideanR4::project_point`.
// Clamp the denominator so a vertex at w = focal does not produce infinities.
fn project_perspective_4d(p4: vec4<f32>, focal: f32) -> vec3<f32> {
    let denom = max(focal - p4.w, 1.0e-4);
    let scale = focal / denom;
    return p4.xyz * scale;
}

@vertex
fn vs_main(
    @location(0) corner:      u32,
    @location(1) start_pos:   vec4<f32>,  // R⁴ endpoint
    @location(2) end_pos:     vec4<f32>,  // R⁴ endpoint
    @location(3) start_color: vec4<f32>,
    @location(4) end_color:   vec4<f32>,
    @location(5) width_px:    f32,
) -> VsOut {
    // Rotor (R⁴) -> Perspective4D (R⁴ -> R³) -> view_projection (R³ -> clip).
    let s_4d = transform.rotor_matrix * start_pos;
    let e_4d = transform.rotor_matrix * end_pos;
    let s_3d = project_perspective_4d(s_4d, transform.focal_distance);
    let e_3d = project_perspective_4d(e_4d, transform.focal_distance);
    var a = transform.view_projection * vec4<f32>(s_3d, 1.0);
    var b = transform.view_projection * vec4<f32>(e_3d, 1.0);

    // Near-plane clip before the perspective divide; see `line_raster.wgsl`
    // for the rationale (behind-eye `w <= 0` would flip NDC across the screen).
    if (a.z < 0.0 && b.z < 0.0) {
        var culled: VsOut;
        culled.clip       = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        culled.coverage_t = 0.0;
        culled.width_px   = width_px;
        culled.color      = vec4<f32>(0.0);
        return culled;
    }
    if (a.z < 0.0) {
        a = mix(a, b, a.z / (a.z - b.z));
    } else if (b.z < 0.0) {
        b = mix(b, a, b.z / (b.z - a.z));
    }
    let s_ndc  = a.xyz / a.w;
    let e_ndc  = b.xyz / b.w;

    // Corners 0, 2 belong to the start endpoint; 1, 3 to the end.
    let pick_start = (corner == 0u || corner == 2u);
    let base_ndc   = select(e_ndc, s_ndc, pick_start);
    let base_w     = select(b.w, a.w, pick_start);

    // Depth-cue color from the segment's mean post-rotor w. The [-0.5, +0.5]
    // band matches a unit-circumradius tesseract (w on the unit 3-sphere);
    // the clamp keeps the lerp valid for other circumradii.
    let mid_w  = (s_4d.w + e_4d.w) * 0.5;
    let w_norm = clamp(mid_w + 0.5, 0.0, 1.0);
    // Cool-back / warm-front diverging pair.
    let back_tint  = vec3<f32>(0.30, 0.42, 0.58);
    let front_tint = vec3<f32>(1.00, 0.78, 0.45);
    let depth_rgb  = mix(back_tint, front_tint, w_norm);
    let color      = vec4<f32>(depth_rgb * start_color.rgb, start_color.a);

    let half_vp     = transform.viewport_size * 0.5;
    let dir_pixels  = (e_ndc.xy - s_ndc.xy) * half_vp;
    let dir_pixels_safe = select(dir_pixels, vec2<f32>(1.0, 0.0), length(dir_pixels) < 1.0e-6);
    let dir2        = normalize(dir_pixels_safe);
    let perp2       = vec2<f32>(-dir2.y, dir2.x);

    let sign         = select(-1.0, 1.0, corner >= 2u);
    let half_with_aa = width_px * 0.5 + 1.0;
    let perp_ndc     = perp2 / half_vp;
    let off_ndc      = perp_ndc * sign * half_with_aa;

    var out: VsOut;
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
    let inner    = max(0.0, (in.width_px - 1.0) / (in.width_px + 1.0));
    let coverage = 1.0 - smoothstep(inner, 1.0, abs(in.coverage_t));
    return vec4<f32>(in.color.rgb, in.color.a * coverage);
}
