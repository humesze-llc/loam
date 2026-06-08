// Line rasterizer: quad-expanded antialiased lines. Each segment instance
// expands to a screen-oriented quad `width + 2 px` wide; the fragment shader
// smoothsteps coverage for a 1 px AA falloff at the silhouette.

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
    var a  = camera.view_projection * vec4<f32>(start_pos, 1.0);
    var b  = camera.view_projection * vec4<f32>(end_pos,   1.0);
    var ca = start_color;
    var cb = end_color;

    // Near-plane clip before the perspective divide. glam's `perspective_rh`
    // puts the near plane at `clip.z == 0`; `clip.z < 0` is nearer, including
    // behind the eye (`w <= 0`), where dividing by non-positive `w` flips NDC
    // across the screen and rubber-bands the segment.
    if (a.z < 0.0 && b.z < 0.0) {
        // Whole segment behind the near plane: emit outside the clip volume.
        var culled: VsOut;
        culled.clip       = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        culled.coverage_t = 0.0;
        culled.width_px   = width_px;
        culled.color      = vec4<f32>(0.0);
        return culled;
    }
    // Move the one behind-plane endpoint to `clip.z == 0`, carrying its color.
    if (a.z < 0.0) {
        let t = a.z / (a.z - b.z);
        a  = mix(a,  b,  t);
        ca = mix(ca, cb, t);
    } else if (b.z < 0.0) {
        let t = b.z / (b.z - a.z);
        b  = mix(b,  a,  t);
        cb = mix(cb, ca, t);
    }

    let s_ndc  = a.xyz / a.w;
    let e_ndc  = b.xyz / b.w;

    // Corners 0, 2 belong to the start endpoint; 1, 3 to the end.
    let pick_start = (corner == 0u || corner == 2u);
    let base_ndc   = select(e_ndc, s_ndc, pick_start);
    let base_w     = select(b.w, a.w, pick_start);
    let color      = select(cb, ca, pick_start);

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
    // Re-multiply by w so the hardware perspective divide recovers our NDC.
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
    // |coverage_t| is 0 at center, 1 at the expanded edge; the inner solid
    // region ends at (width - 1) / (width + 1) and the last px fades to clear.
    let inner   = max(0.0, (in.width_px - 1.0) / (in.width_px + 1.0));
    let coverage = 1.0 - smoothstep(inner, 1.0, abs(in.coverage_t));
    return vec4<f32>(in.color.rgb, in.color.a * coverage);
}
