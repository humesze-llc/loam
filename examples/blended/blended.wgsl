// BlendedSpace demo: user shading layer.
//
// The Space prelude (BlendedSpace<E3, H3, LinearBlendX>), scene SDF, and
// geodesic march kernel are prepended by loam-shader before this file is
// compiled. Available functions:
//   loam_safe_normalize, loam_march_geodesic, loam_estimate_normal (kernel)
//   loam_distance, loam_exp, loam_parallel_transport (Space prelude)
//   loam_blended_alpha (BlendedSpace prelude, for tinting by zone)
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
    // params: [reserved, ball_scale, fog_scale, show_spheres]
    params0:     f32,
    ball_scale:  f32,
    fog_scale:   f32,
    show_spheres: f32,
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
    if t < 0.0 {
        // Sky: graded blue with a warm horizon hinting at the H³ side.
        let horizon = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
        let sky = mix(vec3<f32>(0.05, 0.08, 0.16), vec3<f32>(0.10, 0.14, 0.24), horizon);
        return vec4<f32>(sky, 1.0);
    }

    let hit_space = march_out.xyz;
    let n         = loam_estimate_normal(hit_space, u.ball_scale);
    let sun_dir   = loam_safe_normalize(vec3<f32>(0.4, 0.8, 0.3), vec3<f32>(0.0, 1.0, 0.0));
    let diffuse   = max(dot(n, sun_dir), 0.0);
    let ambient   = 0.22;

    // Tint by blending zone: E³ side cool blue, H³ side warm red.
    // The transition zone shows a smooth gradient; *that* is the
    // visible BlendedSpace seam.
    let alpha = loam_blended_alpha(hit_space);
    let e3_color = vec3<f32>(0.40, 0.62, 0.95);
    let h3_color = vec3<f32>(0.95, 0.45, 0.30);
    var base = mix(e3_color, h3_color, alpha);

    // Floor detection: hit normal points up and we're near y=0.
    // Apply a checkerboard so the floor reads obviously distinct
    // from the sky and from the spheres' tint.
    let is_floor = n.y > 0.85 && abs(hit_space.y) < 0.05;
    if is_floor {
        let cell = floor(hit_space.xz / 0.15);
        let parity = (cell.x + cell.y) - 2.0 * floor((cell.x + cell.y) * 0.5);
        let checker = mix(0.55, 0.95, parity);
        // Strong yellow-green checker on E³ side, magenta on H³,
        // so the *floor* shows the metric blend even more starkly
        // than the spheres.
        let floor_e3 = vec3<f32>(0.20, 0.85, 0.45);
        let floor_h3 = vec3<f32>(0.95, 0.30, 0.65);
        base = mix(floor_e3, floor_h3, alpha) * checker;
    }

    // Geodesic fog under the BlendedSpace metric.
    let cam_dist = loam_distance(u.camera_pos * u.ball_scale, hit_space);
    let fog      = exp(-cam_dist / max(u.fog_scale, 1e-4));

    let shaded = base * (diffuse + ambient) * fog;
    return vec4<f32>(pow(shaded, vec3<f32>(1.0 / 2.2)), 1.0);
}
