// Geodesic lattice demo: user shading layer.
//
// The Space prelude, scene SDF, and geodesic march kernel are prepended by
// loam-shader (or the local assemble() call) before this file is compiled.
// Available functions:
//   loam_safe_normalize, loam_march_geodesic, loam_estimate_normal (kernel)
//   loam_distance, loam_exp, loam_parallel_transport (Space prelude)
//   loam_scene_sdf (scene module)
//
// Uniforms.params layout (see main.rs):
//   params[0] = panel x pixel offset  (e.g. 0, W/3, 2W/3)
//   params[1] = panel index            (0 = E³, 1 = H³, 2 = S³)
//   params[2] = panel pixel width      (e.g. W/3)
//   params[3] = fog scale              (space-tuned: 3.0 / 4.2 / 2.4)

struct Uniforms {
    camera_pos:     vec3<f32>,
    camera_forward: vec3<f32>,
    camera_right:   vec3<f32>,
    camera_up:      vec3<f32>,
    fov_y_tan:      f32,
    resolution:     vec2<f32>,
    time:           f32,
    tick:           f32,
    params:         vec4<f32>,
};

@group(0) @binding(0) var<uniform> ub: Uniforms;

// Scale from camera-space to Space coordinates.
// Must match BALL_SCALE in main.rs. camera distance * BALL_SCALE must be < 1.0.
const BALL_SCALE: f32 = 0.2;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

// Ray direction for this panel's pixel, accounting for panel x-offset and
// width so each third of the window gets its own full [-1,1] UV range.
fn panel_ray_dir(frag_pos: vec4<f32>) -> vec3<f32> {
    let x_offset = ub.params[0];
    let panel_w  = ub.params[2];
    let panel_h  = ub.resolution.y;
    let px = frag_pos.x - x_offset;
    let py = frag_pos.y;
    let uv = vec2<f32>(px / panel_w, py / panel_h) * 2.0 - 1.0;
    let aspect = panel_w / panel_h;
    return loam_safe_normalize(
        ub.camera_forward
            + ub.camera_right * ( uv.x * aspect * ub.fov_y_tan)
            + ub.camera_up    * (-uv.y          * ub.fov_y_tan),
        ub.camera_forward,
    );
}

// Per-space accent color for visual distinction between panels.
fn accent_color(panel_idx: f32) -> vec3<f32> {
    if panel_idx < 0.5 { return vec3<f32>(0.35, 0.60, 1.00); } // E³: blue
    if panel_idx < 1.5 { return vec3<f32>(1.00, 0.52, 0.18); } // H³: orange
    return vec3<f32>(0.22, 0.85, 0.58);                          // S³: teal
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let rd     = panel_ray_dir(frag_pos);
    let accent = accent_color(ub.params[1]);
    let fog_sc = ub.params[3];

    // Sky: vertical gradient tinted by space accent.
    let sky_t = 0.5 + 0.5 * rd.y;
    let sky   = mix(vec3<f32>(0.05, 0.04, 0.08), accent * 0.55, sky_t);

    let march = loam_march_geodesic(ub.camera_pos, rd, BALL_SCALE);
    var col: vec3<f32>;

    if march.w < 0.0 {
        col = sky;
    } else {
        let hit_p   = march.xyz;
        let n       = loam_estimate_normal(hit_p, BALL_SCALE);
        let sun_dir = loam_safe_normalize(vec3<f32>(0.5, 0.9, 0.3), vec3<f32>(0.0, 1.0, 0.0));
        let diff    = max(dot(n, sun_dir), 0.0);
        let rim     = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
        let base    = accent * 0.82;
        col = base * (0.15 + 0.75 * diff) + rim * 0.28 * accent;

        // Geodesic fog from camera to hit point.
        let fog_dist = loam_distance(ub.camera_pos * BALL_SCALE, hit_p);
        let fog      = exp(-fog_dist / max(fog_sc, 1e-4));
        col = mix(sky, col, fog);
    }

    // Gamma correction.
    col = pow(max(col, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
    return vec4<f32>(col, 1.0);
}
