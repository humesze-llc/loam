// Geodesic spheres raymarch demo: user shading layer.
//
// The Space prelude, scene SDF, and geodesic march kernel are prepended by
// loam-shader before this file is compiled. Available functions:
//   loam_safe_normalize, loam_march_geodesic, loam_estimate_normal (kernel)
//   loam_distance, loam_exp, loam_parallel_transport (Space prelude)
//   loam_scene_sdf (scene module)
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
    // params layout maps to SceneParams in main.rs
    power_offset: f32, // reserved
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
        let horizon = (rd.y + 1.0) * 0.5;
        let sky = mix(vec3<f32>(0.03, 0.04, 0.10), vec3<f32>(0.08, 0.10, 0.18), horizon);
        return vec4<f32>(sky, 1.0);
    }

    let hit_space = march_out.xyz;
    let n         = loam_estimate_normal(hit_space, u.ball_scale);
    let sun_dir   = loam_safe_normalize(vec3<f32>(0.4, 0.7, 0.3), vec3<f32>(0.0, 1.0, 0.0));
    let diffuse   = max(dot(n, sun_dir), 0.0);
    let ambient   = 0.18;

    // Geodesic fog in the active Space.
    let cam_dist = loam_distance(u.camera_pos * u.ball_scale, hit_space);
    let fog      = exp(-cam_dist / max(u.fog_scale, 1e-4));

    let base   = vec3<f32>(0.35 + 0.3 * n.x, 0.55 + 0.2 * n.y, 0.9);
    let shaded = base * (diffuse + ambient) * fog;
    return vec4<f32>(pow(shaded, vec3<f32>(1.0 / 2.2)), 1.0);
}
