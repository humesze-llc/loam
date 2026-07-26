//! Smoke pins for the 3D raymarch chain: typed `loam_scene::Scene` -> WGSL ->
//! `loam_shader::ShaderDb` -> `RayMarchNode` / `GeodesicRayMarchNode`.
//!
//! No other code in the workspace instantiates any link in that chain, so it
//! compiles in isolation and rots silently. These tests exist to fail when a
//! signature or a WGSL-symbol contract drifts, nothing more: they assert the
//! chain still type-checks and the assembled shader still validates, never
//! what it draws. Keep them thin.
//!
//! Everything except the `gpu_probe` test is headless; naga does the same
//! validation `ShaderDb` runs before it reaches a device.

use glam::Vec3;
use loam_math::{EuclideanR3, WgslSpace};
use loam_render::device::RenderDevice;
use loam_render::{GeodesicRayMarchNode, RayMarchNode, RayMarchUniforms};
use loam_scene::{Scene, SceneNode};
use loam_shader::{validate_wgsl, ShaderDb, GEODESIC_MARCH_KERNEL};
use wgpu::{Device, TextureFormat};

/// Two leaves and one combinator: enough for the emitter to produce both a
/// helper function and a `let` binding, which is the whole shape of the emit.
fn probe_scene() -> Scene {
    Scene::new(SceneNode::sphere(Vec3::ZERO, 0.5).union(SceneNode::plane(Vec3::Y, -0.5)))
}

/// Mirror of `RayMarchUniforms` in WGSL. The Rust struct pads each `vec3` to
/// 16 bytes by hand; WGSL's `vec3<f32>` alignment produces the same offsets,
/// so the explicit `_padN` fields have no counterpart here.
const UNIFORMS_WGSL: &str = r#"
struct RayMarchUniforms {
    camera_pos: vec3<f32>,
    camera_forward: vec3<f32>,
    camera_right: vec3<f32>,
    camera_up: vec3<f32>,
    fov_y_tan: f32,
    resolution: vec2<f32>,
    time: f32,
    tick: f32,
    params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: RayMarchUniforms;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vid) / 2) * 4.0 - 1.0;
    let y = f32(i32(vid) & 1) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn ray_direction(pos: vec4<f32>) -> vec3<f32> {
    let ndc = pos.xy / u.resolution * 2.0 - vec2<f32>(1.0, 1.0);
    return u.camera_forward
        + u.camera_right * ndc.x * u.fov_y_tan
        + u.camera_up * ndc.y * u.fov_y_tan;
}
"#;

/// Calls every symbol `GEODESIC_MARCH_KERNEL` exports, so a kernel rename is a
/// validation failure rather than a black screen.
const GEODESIC_SHADING_WGSL: &str = r#"
@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let dir = loam_safe_normalize(ray_direction(pos), vec3<f32>(0.0, 0.0, -1.0));
    let hit = loam_march_geodesic(u.camera_pos, dir, 1.0);
    if hit.w < 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let n = loam_estimate_normal(hit.xyz, 1.0);
    return vec4<f32>(n * 0.5 + vec3<f32>(0.5), 1.0);
}
"#;

/// The kernel-free path: the user shader calls the scene emit directly, which
/// is the contract `ShaderDb::load_with_scene` exists to satisfy.
const SCENE_SHADING_WGSL: &str = r#"
@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = u.camera_pos + ray_direction(pos);
    let lit = step(loam_scene_sdf(p), 0.0);
    return vec4<f32>(vec3<f32>(lit), 1.0);
}
"#;

fn geodesic_user_shader() -> String {
    format!("{UNIFORMS_WGSL}{GEODESIC_SHADING_WGSL}")
}

fn scene_user_shader() -> String {
    format!("{UNIFORMS_WGSL}{SCENE_SHADING_WGSL}")
}

/// `ShaderDb`'s assembler is crate-private, so reproduce its ordering contract
/// (Space prelude, then scene module, then user source) here. The ordering
/// itself is pinned inside loam-shader by
/// `assemble_includes_scene_between_space_and_user`, and the `gpu_probe` test
/// below runs the real assembler, so a divergence cannot hide in both places.
fn assemble(space_wgsl: &str, scene_wgsl: &str, user_wgsl: &str) -> String {
    format!("{space_wgsl}\n{scene_wgsl}\n{user_wgsl}")
}

/// The uniform buffer is the 96 bytes `UNIFORMS_WGSL` lays out: four 16-byte
/// vec3 slots (the last packing `fov_y_tan`), one holding resolution/time/tick,
/// then `params`. The offsets are the mirror's, assigned by std140 in WGSL
/// declaration order, so reordering the Rust fields lands here too; size alone
/// would not, and a swapped `camera_right`/`camera_up` draws every frame with
/// the axes exchanged.
#[test]
fn uniforms_match_the_wgsl_mirror_layout() {
    use std::mem::offset_of;
    assert_eq!(std::mem::size_of::<RayMarchUniforms>(), 96);
    assert_eq!(offset_of!(RayMarchUniforms, camera_pos), 0);
    assert_eq!(offset_of!(RayMarchUniforms, camera_forward), 16);
    assert_eq!(offset_of!(RayMarchUniforms, camera_right), 32);
    assert_eq!(offset_of!(RayMarchUniforms, camera_up), 48);
    assert_eq!(offset_of!(RayMarchUniforms, fov_y_tan), 60);
    assert_eq!(offset_of!(RayMarchUniforms, resolution), 64);
    assert_eq!(offset_of!(RayMarchUniforms, time), 72);
    assert_eq!(offset_of!(RayMarchUniforms, tick), 76);
    assert_eq!(offset_of!(RayMarchUniforms, params), 80);
}

/// `execute_panel` is the multi-panel draw entry point on both nodes and has no
/// caller in the workspace. It needs a `RenderDevice`, which needs a real
/// surface, so the coercion pins the signature where a call cannot.
#[test]
fn panel_draw_entry_points_keep_their_signature() {
    type ExecutePanel<N> =
        fn(&mut N, &RenderDevice, &wgpu::TextureView, bool, [u32; 4]) -> anyhow::Result<()>;
    let _: ExecutePanel<RayMarchNode> = RayMarchNode::execute_panel;
    let _: ExecutePanel<GeodesicRayMarchNode> = GeodesicRayMarchNode::execute_panel;
}

/// Space prelude + `Scene::to_wgsl` + geodesic kernel + a shader carrying the
/// entry points `RayMarchNode::new` names is a valid WGSL module.
#[test]
fn geodesic_chain_assembles_into_valid_wgsl() {
    let scene_wgsl = probe_scene().to_wgsl(&EuclideanR3);
    let source = assemble(
        &EuclideanR3.wgsl_impl(),
        &format!("{scene_wgsl}{GEODESIC_MARCH_KERNEL}"),
        &geodesic_user_shader(),
    );
    validate_wgsl(&source).expect("geodesic raymarch chain should validate");
}

/// The same chain without the kernel: `Scene::to_wgsl` alone has to satisfy a
/// user shader that calls `loam_scene_sdf`.
#[test]
fn scene_chain_assembles_into_valid_wgsl() {
    let scene_wgsl = probe_scene().to_wgsl(&EuclideanR3);
    let source = assemble(&EuclideanR3.wgsl_impl(), &scene_wgsl, &scene_user_shader());
    validate_wgsl(&source).expect("scene raymarch chain should validate");
}

/// Runtime wiring for the geodesic path, as an app's `setup` would write it.
/// Compiled unconditionally; executed only by the `gpu_probe` test.
fn build_geodesic_node(
    device: &Device,
    surface_format: TextureFormat,
    shader_path: &std::path::Path,
    scene: &Scene,
) -> anyhow::Result<GeodesicRayMarchNode> {
    let mut db = ShaderDb::new(device.clone());
    let id = db.load_geodesic_scene(shader_path, &scene.to_wgsl(&EuclideanR3), &EuclideanR3)?;
    Ok(GeodesicRayMarchNode::from_module(
        device,
        surface_format,
        db.module(id),
        1,
    ))
}

/// Runtime wiring for the kernel-free path.
fn build_raymarch_node(
    device: &Device,
    surface_format: TextureFormat,
    shader_path: &std::path::Path,
    scene: &Scene,
) -> anyhow::Result<RayMarchNode> {
    let mut db = ShaderDb::new(device.clone());
    let id = db.load_with_scene(shader_path, &scene.to_wgsl(&EuclideanR3), &EuclideanR3)?;
    Ok(RayMarchNode::new(device, surface_format, db.module(id), 1))
}

fn frame_uniforms() -> RayMarchUniforms {
    RayMarchUniforms {
        resolution: [640.0, 360.0],
        ..Default::default()
    }
}

async fn request_device() -> Result<(wgpu::Device, wgpu::Queue), String> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| format!("request_adapter failed: {e}"))?;
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("raymarch-chain-smoke"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
            experimental_features: Default::default(),
        })
        .await
        .map_err(|e| format!("request_device failed: {e}"))
}

/// The real chain end to end: `ShaderDb` reads the shader off disk, assembles
/// it against the Space and the scene emit, and both nodes build a pipeline
/// from the resulting module. Ignored by default because it needs an adapter;
/// the `gpu_probe` suffix is what CI's software-adapter job selects on.
#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn both_nodes_build_from_a_real_shader_db_gpu_probe() {
    let (device, queue) = pollster::block_on(request_device()).expect("wgpu device");
    let surface_format = TextureFormat::Rgba8UnormSrgb;
    let scene = probe_scene();

    let dir = tempfile::tempdir().expect("tempdir");

    let geodesic_path = dir.path().join("geodesic.wgsl");
    std::fs::write(&geodesic_path, geodesic_user_shader()).expect("write geodesic shader");
    let mut geodesic = build_geodesic_node(&device, surface_format, &geodesic_path, &scene)
        .expect("geodesic chain should build a pipeline");
    geodesic.set_uniforms(&queue, frame_uniforms());

    let scene_path = dir.path().join("scene.wgsl");
    std::fs::write(&scene_path, scene_user_shader()).expect("write scene shader");
    let mut plain = build_raymarch_node(&device, surface_format, &scene_path, &scene)
        .expect("scene chain should build a pipeline");
    plain.set_uniforms(&queue, frame_uniforms());
}
