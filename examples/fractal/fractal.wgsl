// Mandelbulb raymarcher: the fractal demo for Loam.
//
// Edit this file while the example is running; the ShaderDb watcher
// recompiles and RayMarchNode rebuilds on the next frame.

struct Uniforms {
    camera_pos: vec3<f32>,
    camera_forward: vec3<f32>,
    camera_right: vec3<f32>,
    camera_up: vec3<f32>,
    fov_y_tan: f32,
    resolution: vec2<f32>,
    time: f32,
    tick: f32,
    // params layout maps to FractalParams in main.rs
    power_offset: f32,
    ball_scale: f32,
    fog_scale: f32,
    params_pad: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    // Big-triangle trick: one triangle covers the whole viewport.
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn sky_color(rd: vec3<f32>) -> vec3<f32> {
    let horizon = (rd.y + 1.0) * 0.5;
    return mix(vec3<f32>(0.03, 0.04, 0.10), vec3<f32>(0.08, 0.10, 0.18), horizon);
}

fn mandelbulb_de(p: vec3<f32>) -> f32 {
    var z = p;
    var dr = 1.0;
    var r = 0.0;
    let power = 8.0 + u.power_offset;
    let iterations = 12;
    for (var i = 0; i < iterations; i = i + 1) {
        r = length(z);
        if (r > 2.0) { break; }
        let theta = acos(clamp(z.z / r, -1.0, 1.0));
        let phi = atan2(z.y, z.x);
        dr = pow(r, power - 1.0) * power * dr + 1.0;
        let zr = pow(r, power);
        let nt = theta * power;
        let np = phi * power;
        z = zr * vec3<f32>(sin(nt) * cos(np), sin(nt) * sin(np), cos(nt)) + p;
    }
    return 0.5 * log(max(r, 1e-4)) * r / dr;
}

// Debug heat-map: set to 1 to visualise march iteration count instead of
// shading. Blue = miss/fast, red = slow convergence, white = hit surface.
// Useful for diagnosing space-specific march artifacts.
const DEBUG_ITER_HEAT: bool = false;

// Returns vec4(hit_pos_scene, t_scene) on hit, vec4(0,0,0,-1) on miss.
// w2 encodes iteration count for the debug heat-map (always filled).
struct MarchResult {
    pos: vec3<f32>,
    t: f32,
    iters: f32,
    t_arc: f32,
}

fn march_geodesic(ro_scene: vec3<f32>, rd_scene: vec3<f32>) -> MarchResult {
    // The Mandelbulb SDF is Euclidean: authored in flat R³ scene coordinates.
    // Combining a Euclidean SDF with geodesic ray steps requires a correct
    // chart inverse (p_scene = f(p_space)), which is non-trivial and space-
    // dependent. The inverse p_space/scale is only accurate at the origin.
    //
    // We therefore march Euclidean rays and apply Space geometry only where
    // it is well-defined: in the fog computation via loam_distance. This gives
    // correct curved-space attenuation (H³ fog grows faster; S³ fog saturates
    // at π) without phantom-surface artifacts from coordinate conversion error.
    //
    // Future: add loam_origin_distance(p) to the Space ABI and use
    //   p_scene = normalize(p_space) * loam_origin_distance(p_space) / scale
    // to enable geodesic ray bending with a geometrically correct SDF lookup.
    var p = ro_scene;
    var t = 0.0;
    var iters = 0.0;

    for (var i = 0; i < 128; i = i + 1) {
        iters = f32(i);
        let d = mandelbulb_de(p);
        if (d < 0.001) {
            return MarchResult(p, t, iters, t * u.ball_scale);
        }
        if (t > 20.0) {
            return MarchResult(vec3<f32>(0.0), -1.0, iters, t * u.ball_scale);
        }
        let step = max(d * 0.9, 0.001);
        p = p + rd_scene * step;
        t = t + step;
    }
    return MarchResult(vec3<f32>(0.0), -1.0, 128.0, t * u.ball_scale);
}

fn estimate_normal(p: vec3<f32>) -> vec3<f32> {
    let eps = 0.0015;
    let ex = vec3<f32>(eps, 0.0, 0.0);
    let ey = vec3<f32>(0.0, eps, 0.0);
    let ez = vec3<f32>(0.0, 0.0, eps);
    return normalize(vec3<f32>(
        mandelbulb_de(p + ex) - mandelbulb_de(p - ex),
        mandelbulb_de(p + ey) - mandelbulb_de(p - ey),
        mandelbulb_de(p + ez) - mandelbulb_de(p - ez),
    ));
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = (pos.xy / u.resolution) * 2.0 - 1.0;
    let aspect = u.resolution.x / u.resolution.y;
    // Flip Y: pos.y increases downward, we want view Y up.
    let ndc = vec2<f32>(uv.x * aspect, -uv.y);

    let rd = normalize(
        u.camera_forward
        + u.camera_right * ndc.x * u.fov_y_tan
        + u.camera_up * ndc.y * u.fov_y_tan
    );

    let sky = sky_color(rd);
    let sky_linear = pow(max(sky, vec3<f32>(0.0, 0.0, 0.0)), vec3<f32>(2.2));
    let mr = march_geodesic(u.camera_pos, rd);

    // Debug heat-map: blue = miss/fast, yellow = slow, red = max iters.
    if (DEBUG_ITER_HEAT) {
        let heat = mr.iters / 128.0;
        let miss = select(0.0, 1.0, mr.t < 0.0);
        // arc fill: green channel shows arc fraction consumed
        let arc_frac = clamp(mr.t_arc / max(LOAM_MAX_ARC, 1e-5), 0.0, 1.0);
        return vec4<f32>(heat, arc_frac * (1.0 - miss), miss * 0.5, 1.0);
    }

    if (mr.t < 0.0) {
        return vec4<f32>(sky, 1.0);
    }

    let hit = mr.pos;
    let n = estimate_normal(hit);
    let sun_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
    let diffuse = max(dot(n, sun_dir), 0.0);
    let ambient = 0.15;

    // Exercise the loam-math Space prelude: geodesic stepping is handled
    // by loam_exp / loam_parallel_transport, and camera-to-hit fog uses
    // loam_distance. Swapping the host Space changes both trajectory and
    // attenuation without rewriting the Mandelbulb SDF.
    //
    // `ball_scale` maps Euclidean scene coords into the unit Poincaré
    // ball when the host Space is hyperbolic. In Euclidean mode it is
    // 1.0, so this is a no-op and the output is byte-identical to the
    // pre-flag version. `fog_scale` is the distance at which fog goes
    // fully opaque.
    let scaled_pos = u.camera_pos * u.ball_scale;
    let scaled_hit = hit * u.ball_scale;
    let cam_dist = loam_distance(scaled_pos, scaled_hit);
    let fog = clamp(cam_dist / u.fog_scale, 0.0, 1.0);

    let base = vec3<f32>(0.35 + 0.3 * n.x, 0.55 + 0.2 * n.y, 0.9);
    let lit = base * (diffuse + ambient);
    let shaded = mix(lit, sky_linear, fog);
    return vec4<f32>(pow(max(shaded, vec3<f32>(0.0, 0.0, 0.0)), vec3<f32>(1.0 / 2.2)), 1.0);
}
