// Line rasterizer: static-mesh R⁴ variant. Mesh vertices live in R⁴ and are
// uploaded ONCE; per-frame work is the host writing a single uniform with the
// current rotor + Perspective4D focal_distance + view_projection. The vertex
// shader applies rotor → Perspective4D projection → view_projection to each
// endpoint before the standard quad-expansion + AA logic.
//
// Quad-expansion + AA fragment shader are mirrored from `line_raster.wgsl`
// (the R³ dynamic variant); the only divergence is the per-endpoint transform
// chain in vs_main. Keeping the two shaders separate (vs trying to share via
// WGSL "imports") avoids the naga-side composer dependency the project has
// otherwise avoided so far.

struct TransformUniform {
    // 4×4 rotation matrix derived from the host-side `Rotor4` once per frame
    // via `Rotor4::to_mat4()`. Columns are the rotor applied to the canonical
    // basis vectors; column-major upload matches WGSL's `mat4x4<f32>` storage.
    rotor_matrix: mat4x4<f32>,
    // Standard R³→clip transform applied AFTER the 4D→3D projection.
    view_projection: mat4x4<f32>,
    // Render-target dimensions in pixels for pixel-to-NDC offset conversion
    // in the quad-expansion stage.
    viewport_size: vec2<f32>,
    // Perspective4D focal distance along the w-axis. Vertices at `w = focal`
    // are degenerate; the host clamps the denominator to avoid div-by-zero.
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

// Perspective4D projection: scale x/y/z by focal/(focal - w). Same formula as
// `EuclideanR4::project_point` on the host. Clamp denominator to keep
// numerically-degenerate w values (when a rotation places a vertex at
// w = focal) from producing infinities.
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
    // Stage 1: apply the per-frame rotor in R⁴ (host pre-converted Rotor4 to
    // a 4×4 matrix; here it's a single mat-vec multiply per endpoint).
    let s_4d = transform.rotor_matrix * start_pos;
    let e_4d = transform.rotor_matrix * end_pos;
    // Stage 2: project from R⁴ to R³ via the standard pinhole formula along w.
    let s_3d = project_perspective_4d(s_4d, transform.focal_distance);
    let e_3d = project_perspective_4d(e_4d, transform.focal_distance);
    // Stage 3: standard R³→clip via view*proj.
    let s_clip = transform.view_projection * vec4<f32>(s_3d, 1.0);
    let e_clip = transform.view_projection * vec4<f32>(e_3d, 1.0);
    let s_ndc  = s_clip.xyz / s_clip.w;
    let e_ndc  = e_clip.xyz / e_clip.w;

    // Corners 0, 2 belong to the start endpoint; 1, 3 to the end. (Mirrors
    // the same convention as the R³ variant; the quad-expansion + AA logic
    // below is bit-identical to `line_raster.wgsl`.)
    let pick_start = (corner == 0u || corner == 2u);
    let base_ndc   = select(e_ndc, s_ndc, pick_start);
    let base_w     = select(e_clip.w, s_clip.w, pick_start);
    let color      = select(end_color, start_color, pick_start);

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
