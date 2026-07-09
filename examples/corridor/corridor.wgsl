// Corridor geodesic raymarch demo: user shading layer.
//
// The Space prelude, scene SDF, and geodesic march kernel are prepended by
// loam-shader before this file is compiled. Available functions:
//   loam_safe_normalize, loam_march_geodesic, loam_estimate_normal (kernel)
//   loam_distance, loam_exp, loam_parallel_transport (Space prelude)
//   loam_scene_sdf (scene module)
//
// The geodesic march makes the three spaces read differently: E³ walls are
// flat, H³ walls bow outward (parallel geodesics diverge), S³ walls converge.
//
// Edit while the example is running; ShaderDb hot-reloads this file.

struct Uniforms {
    camera_pos:     vec3<f32>,
    camera_forward: vec3<f32>,
    camera_right:   vec3<f32>,
    camera_up:      vec3<f32>,
    fov_y_tan:      f32,
    resolution:     vec2<f32>,
    time:           f32,
    tick:           f32,
    // params = [reserved, ball_scale, fog_scale, reserved]
    power_offset: f32,
    ball_scale:   f32,
    fog_scale:    f32,
    params_pad:   f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

// Color the surface by which part of the corridor was hit, using the
// normal to classify floor / ceiling / wall / pillar.
fn surface_base_color(n: vec3<f32>, hit: vec3<f32>) -> vec3<f32> {
    // Floor: normal points up strongly.
    if (n.y > 0.75) {
        // Checkerboard in Space coords; tile lines are geodesic in E³,
        // tanh-spaced in H³, sin-spaced in S³. The tiling is the grid.
        let cell = floor(hit.x * 10.0) + floor(hit.z * 10.0);
        let light_cell = 0.72;
        let dark_cell  = 0.48;
        let t = select(dark_cell, light_cell, (i32(cell) & 1) == 0);
        return vec3<f32>(t, t * 0.97, t * 0.90);
    }
    // Ceiling: normal points down.
    if (n.y < -0.75) {
        return vec3<f32>(0.22, 0.24, 0.30);
    }
    // Side walls: normal dominates in x.
    if (abs(n.x) > 0.75) {
        // Faint horizontal banding so perspective is obvious.
        let band = 0.85 + 0.15 * sin(hit.y * 30.0);
        return vec3<f32>(0.58, 0.55, 0.50) * band;
    }
    // Pillar (rounded surface): warm off-white.
    return vec3<f32>(0.92, 0.86, 0.72);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = (pos.xy / u.resolution) * 2.0 - 1.0;
    let aspect = u.resolution.x / u.resolution.y;
    let ndc = vec2<f32>(uv.x * aspect, -uv.y);

    let rd = loam_safe_normalize(
        u.camera_forward
            + u.camera_right * ndc.x * u.fov_y_tan
            + u.camera_up    * ndc.y * u.fov_y_tan,
        vec3<f32>(0.0, 0.0, -1.0),
    );

    let march_out = loam_march_geodesic(u.camera_pos, rd, u.ball_scale);
    let t = march_out.w;
    if (t < 0.0) {
        let horizon = clamp((rd.y + 1.0) * 0.5, 0.0, 1.0);
        let sky = mix(vec3<f32>(0.02, 0.03, 0.06), vec3<f32>(0.09, 0.11, 0.18), horizon);
        return vec4<f32>(sky, 1.0);
    }

    let hit_space = march_out.xyz;
    let n         = loam_estimate_normal(hit_space, u.ball_scale);
    let base      = surface_base_color(n, hit_space);

    let sun_dir = loam_safe_normalize(vec3<f32>(0.35, 0.75, -0.25), vec3<f32>(0.0, 1.0, 0.0));
    let diffuse = max(dot(n, sun_dir), 0.0);
    let ambient = 0.22;

    // Geodesic fog in the active Space.
    let cam_dist = loam_distance(u.camera_pos * u.ball_scale, hit_space);
    let fog      = exp(-cam_dist / max(u.fog_scale, 1e-4));

    let shaded = base * (diffuse + ambient) * fog;
    return vec4<f32>(pow(shaded, vec3<f32>(1.0 / 2.2)), 1.0);
}
