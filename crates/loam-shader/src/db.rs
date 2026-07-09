//! Shader database with hot-reload support.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use loam_asset::{AssetEvent, AssetEventKind};
use loam_math::WgslSpace;
use wgpu::{Device, ShaderModule, ShaderModuleDescriptor, ShaderSource};

/// WGSL parse or validation failure.
#[derive(Debug, thiserror::Error)]
pub enum WgslValidationError {
    #[error("WGSL parse error: {0}")]
    Parse(#[from] naga::front::wgsl::ParseError),
    #[error("WGSL validation error: {0}")]
    Validate(Box<naga::WithSpan<naga::valid::ValidationError>>),
}

/// Opaque handle to a shader in a [`ShaderDb`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShaderId(u32);

struct Entry {
    path: PathBuf,
    module: ShaderModule,
    scene_source: Option<String>,
    /// Bumped on every successful (re)compile; render code rebuilds its pipeline
    /// on a generation mismatch.
    generation: u64,
    /// Debug label for the module; reused on recompile.
    label: String,
}

/// Cache of compiled shaders, invalidated on asset events. A failed hot-reload
/// (or removed file) keeps the last good module rather than crashing; a later
/// create/modify restores it.
pub struct ShaderDb {
    device: Device,
    entries: HashMap<ShaderId, Entry>,
    path_index: HashMap<PathBuf, ShaderId>,
    next_id: u32,
}

impl ShaderDb {
    /// Construct. `device` is cloned on recompile (wgpu's Device is refcounted).
    pub fn new(device: Device) -> Self {
        Self {
            device,
            entries: HashMap::new(),
            path_index: HashMap::new(),
            next_id: 0,
        }
    }

    /// Load a shader from disk, prepending the Space's WGSL prelude. The returned
    /// [`ShaderId`] is stable across reloads; loading the same path twice yields
    /// the same ID and a fresh compilation.
    pub fn load<S: WgslSpace>(&mut self, path: impl AsRef<Path>, space: &S) -> Result<ShaderId> {
        self.load_inner(path, None, space)
    }

    /// Load a shader plus a scene module; the scene source is stored and reused on
    /// reloads of the shader file.
    pub fn load_with_scene<S: WgslSpace>(
        &mut self,
        path: impl AsRef<Path>,
        scene_source: &str,
        space: &S,
    ) -> Result<ShaderId> {
        self.load_inner(path, Some(scene_source), space)
    }

    /// Load a shader for geodesic ray marching: assembles Space prelude + scene
    /// SDF + geodesic march kernel ([`crate::GEODESIC_MARCH_KERNEL`], which defines
    /// `loam_march_geodesic` / `loam_estimate_normal` / `loam_safe_normalize`) + user
    /// shading. The scene + kernel is stored and reused on reloads.
    pub fn load_geodesic_scene<S: WgslSpace>(
        &mut self,
        path: impl AsRef<Path>,
        scene_source: &str,
        space: &S,
    ) -> Result<ShaderId> {
        let scene_with_kernel = format!(
            "{scene_source}// ---- loam geodesic march kernel ----\n{}",
            crate::GEODESIC_MARCH_KERNEL
        );
        self.load_inner(path, Some(&scene_with_kernel), space)
    }

    fn load_inner<S: WgslSpace>(
        &mut self,
        path: impl AsRef<Path>,
        scene_source: Option<&str>,
        space: &S,
    ) -> Result<ShaderId> {
        let path = canonicalize(path.as_ref())?;
        let source = std::fs::read_to_string(&path)
            .with_context(|| format!("reading shader {}", path.display()))?;
        let module = self.compile(&path, &source, scene_source, space)?;

        let label = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());

        if let Some(&id) = self.path_index.get(&path) {
            let entry = self.entries.get_mut(&id).expect("path_index out of sync");
            entry.module = module;
            entry.scene_source = scene_source.map(str::to_owned);
            entry.generation += 1;
            entry.label = label;
            Ok(id)
        } else {
            let id = ShaderId(self.next_id);
            self.next_id += 1;
            self.entries.insert(
                id,
                Entry {
                    path: path.clone(),
                    module,
                    scene_source: scene_source.map(str::to_owned),
                    generation: 1,
                    label,
                },
            );
            self.path_index.insert(path, id);
            Ok(id)
        }
    }

    /// Borrow the current compiled module for `id`.
    pub fn module(&self, id: ShaderId) -> &ShaderModule {
        &self
            .entries
            .get(&id)
            .expect("unknown ShaderId - was it loaded by this ShaderDb?")
            .module
    }

    /// Generation counter for `id`; bumps on every successful (re)compile so
    /// render code can rebuild its pipeline on mismatch.
    pub fn generation(&self, id: ShaderId) -> u64 {
        self.entries.get(&id).map(|e| e.generation).unwrap_or(0)
    }

    /// Apply filesystem events, recompiling affected shaders. Compile errors are
    /// logged but keep the last good module; rendering continues until fixed.
    pub fn apply_events<S: WgslSpace>(&mut self, events: &[AssetEvent], space: &S) {
        for event in events {
            let canonical = match canonicalize(&event.path) {
                Ok(p) => p,
                // Removed files can't be canonicalized; look up by raw path.
                Err(_) => event.path.clone(),
            };
            let Some(&id) = self.path_index.get(&canonical) else {
                continue;
            };
            match event.kind {
                AssetEventKind::Created | AssetEventKind::Modified => {
                    if let Err(e) = self.reload(id, space) {
                        tracing::warn!("shader reload failed for {}: {e:#}", canonical.display());
                    } else {
                        tracing::info!("reloaded shader {}", canonical.display());
                    }
                }
                AssetEventKind::Removed => {
                    tracing::warn!(
                        "shader file removed; keeping stale module: {}",
                        canonical.display()
                    );
                }
            }
        }
    }

    fn reload<S: WgslSpace>(&mut self, id: ShaderId, space: &S) -> Result<()> {
        let path = self.entries[&id].path.clone();
        let scene_source = self.entries[&id].scene_source.clone();
        let source = std::fs::read_to_string(&path)
            .with_context(|| format!("reading shader {}", path.display()))?;
        let module = self.compile(&path, &source, scene_source.as_deref(), space)?;
        let entry = self.entries.get_mut(&id).expect("id just looked up");
        entry.module = module;
        entry.generation += 1;
        Ok(())
    }

    fn compile<S: WgslSpace>(
        &self,
        path: &Path,
        user_source: &str,
        scene_source: Option<&str>,
        space: &S,
    ) -> Result<ShaderModule> {
        let full = assemble_source_with_scene(&space.wgsl_impl(), scene_source, user_source);
        validate_wgsl(&full).with_context(|| format!("validating shader {}", path.display()))?;
        let label = path.file_name().and_then(|n| n.to_str());
        Ok(self.device.create_shader_module(ShaderModuleDescriptor {
            label,
            source: ShaderSource::Wgsl(full.into()),
        }))
    }
}

fn canonicalize(path: &Path) -> Result<PathBuf> {
    path.canonicalize()
        .with_context(|| format!("canonicalizing {}", path.display()))
}

/// Concatenate the Space's WGSL prelude with the user shader source. Extracted for
/// testability; the Device-free part of the hot-reload path.
#[cfg(test)]
pub(crate) fn assemble_source(space_wgsl: &str, user_source: &str) -> String {
    assemble_source_with_scene(space_wgsl, None, user_source)
}

pub(crate) fn assemble_source_with_scene(
    space_wgsl: &str,
    scene_wgsl: Option<&str>,
    user_source: &str,
) -> String {
    let scene_len = scene_wgsl.map(str::len).unwrap_or(0);
    let mut out = String::with_capacity(space_wgsl.len() + scene_len + user_source.len() + 96);
    out.push_str("// ---- loam-math Space prelude ----\n");
    out.push_str(space_wgsl);
    if !space_wgsl.ends_with('\n') {
        out.push('\n');
    }
    if let Some(scene_wgsl) = scene_wgsl {
        out.push_str("// ---- loam-scene scene module ----\n");
        out.push_str(scene_wgsl);
        if !scene_wgsl.ends_with('\n') {
            out.push('\n');
        }
    }
    out.push_str("// ---- user shader ----\n");
    out.push_str(user_source);
    out
}

/// Parse and validate a complete WGSL module. Headless: rejects a broken
/// [`loam_math::WgslSpace`] prelude or user shader without a GPU adapter. `wgpu`
/// still does its own backend validation at module creation.
pub fn validate_wgsl(source: &str) -> std::result::Result<(), WgslValidationError> {
    let module = naga::front::wgsl::parse_str(source)?;
    let flags = naga::valid::ValidationFlags::all();
    let caps = naga::valid::Capabilities::empty();
    naga::valid::Validator::new(flags, caps)
        .validate(&module)
        .map_err(|e| WgslValidationError::Validate(Box::new(e)))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::{Pod, Zeroable};
    use glam::Vec3;
    use loam_math::{
        BlendedSpace, EuclideanR3, EuclideanR4, HyperbolicH3, LinearBlendX, Space, SphericalS3,
        WgslSpace,
    };

    const ABI_PROBE: &str = r#"
@compute @workgroup_size(1)
fn main() {
    let a = vec3<f32>(0.1, 0.2, 0.3);
    let b = vec3<f32>(0.2, -0.1, 0.05);
    let v = vec3<f32>(0.01, 0.02, -0.03);
    _ = loam_distance(a, b);
    _ = loam_origin_distance(a);
    _ = loam_exp(a, v);
    _ = loam_log(a, b);
    _ = loam_parallel_transport(a, b, v);
    _ = LOAM_MAX_ARC;
}
"#;

    // 4D variant of `ABI_PROBE`; `EuclideanR4` uses `vec4<f32>` throughout, v0 ABI
    // otherwise identical.
    const ABI_PROBE_VEC4: &str = r#"
@compute @workgroup_size(1)
fn main() {
    let a = vec4<f32>(0.1, 0.2, 0.3, 0.0);
    let b = vec4<f32>(0.2, -0.1, 0.05, 0.4);
    let v = vec4<f32>(0.01, 0.02, -0.03, 0.05);
    _ = loam_distance(a, b);
    _ = loam_origin_distance(a);
    _ = loam_exp(a, v);
    _ = loam_log(a, b);
    _ = loam_parallel_transport(a, b, v);
    _ = LOAM_MAX_ARC;
}
"#;

    #[test]
    fn assemble_includes_both_sources() {
        let s = assemble_source(
            "fn loam_distance(a: f32, b: f32) -> f32 { return 0.0; }",
            "@fragment fn main() {}",
        );
        assert!(s.contains("loam_distance"));
        assert!(s.contains("@fragment fn main"));
        assert!(s.find("loam_distance").unwrap() < s.find("@fragment fn main").unwrap());
    }

    #[test]
    fn assemble_adds_newline_between_sources() {
        let s = assemble_source("fn a() {}", "fn b() {}");
        // prelude line then newline then user section marker then user code.
        assert!(s.contains("fn a() {}\n// ---- user shader ----"));
    }

    #[test]
    fn assemble_handles_trailing_newline_in_prelude() {
        let s = assemble_source("fn a() {}\n", "fn b() {}");
        // No double newline from our side; the input one is sufficient.
        assert!(!s.contains("fn a() {}\n\n// ---- user shader ----"));
        assert!(s.contains("fn a() {}\n// ---- user shader ----"));
    }

    #[test]
    fn assemble_includes_scene_between_space_and_user() {
        let s = assemble_source_with_scene("fn space() {}", Some("fn scene() {}"), "fn user() {}");
        let i_space = s.find("fn space() {}").expect("space chunk present");
        let i_scene = s.find("fn scene() {}").expect("scene chunk present");
        let i_user = s.find("fn user() {}").expect("user chunk present");
        assert!(i_space < i_scene && i_scene < i_user);
    }

    #[test]
    fn euclidean_space_prelude_validates_against_abi_probe() {
        let src = assemble_source(&EuclideanR3.wgsl_impl(), ABI_PROBE);
        validate_wgsl(&src).expect("EuclideanR3 WGSL prelude should validate");
    }

    #[test]
    fn hyperbolic_space_prelude_validates_against_abi_probe() {
        let src = assemble_source(&HyperbolicH3.wgsl_impl(), ABI_PROBE);
        validate_wgsl(&src).expect("HyperbolicH3 WGSL prelude should validate");
    }

    #[test]
    fn spherical_space_prelude_validates_against_abi_probe() {
        let src = assemble_source(&SphericalS3.wgsl_impl(), ABI_PROBE);
        validate_wgsl(&src).expect("SphericalS3 WGSL prelude should validate");
    }

    /// `EuclideanR4`'s prelude is the v0 ABI in `vec4<f32>`. No render node
    /// consumes it yet (4D ships via hyperslice, not native geodesic march), but
    /// the flat-space ℝ⁴ math is honest, so naga validation pins the contract.
    #[test]
    fn euclidean_r4_space_prelude_validates_against_abi_probe() {
        let src = assemble_source(&EuclideanR4.wgsl_impl(), ABI_PROBE_VEC4);
        validate_wgsl(&src).expect("EuclideanR4 WGSL prelude should validate");
    }

    // Minimal stub of loam_scene_sdf so the kernel has something to call.
    const KERNEL_SCENE: &str = r#"
fn loam_scene_sdf(p: vec3<f32>) -> f32 {
    return loam_distance(p, vec3<f32>(0.0, 0.0, 0.0)) - 0.25;
}
"#;

    // Compute probe that exercises all three kernel entry points.
    const KERNEL_PROBE: &str = r#"
@compute @workgroup_size(1)
fn main() {
    let ro = vec3<f32>(0.0, 0.0, 2.0);
    let rd = vec3<f32>(0.0, 0.0, -1.0);
    _ = loam_march_geodesic(ro, rd, 0.2);
    _ = loam_estimate_normal(vec3<f32>(0.0, 0.0, 0.0), 0.2);
    _ = loam_safe_normalize(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0));
}
"#;

    fn assemble_geodesic_probe(space_wgsl: &str) -> String {
        assemble_source_with_scene(
            space_wgsl,
            Some(&format!("{KERNEL_SCENE}{}", crate::GEODESIC_MARCH_KERNEL)),
            KERNEL_PROBE,
        )
    }

    #[test]
    fn euclidean_geodesic_kernel_validates() {
        let src = assemble_geodesic_probe(&EuclideanR3.wgsl_impl());
        validate_wgsl(&src).expect("EuclideanR3 + geodesic kernel should validate");
    }

    #[test]
    fn hyperbolic_geodesic_kernel_validates() {
        let src = assemble_geodesic_probe(&HyperbolicH3.wgsl_impl());
        validate_wgsl(&src).expect("HyperbolicH3 + geodesic kernel should validate");
    }

    #[test]
    fn spherical_geodesic_kernel_validates() {
        let src = assemble_geodesic_probe(&SphericalS3.wgsl_impl());
        validate_wgsl(&src).expect("SphericalS3 + geodesic kernel should validate");
    }

    #[test]
    fn blended_e3_h3_prelude_validates_against_abi_probe() {
        let bs = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-2.0, 2.0));
        let src = assemble_source(&bs.wgsl_impl(), ABI_PROBE);
        validate_wgsl(&src).expect("BlendedSpace<E3,H3,LinearBlendX> WGSL prelude should validate");
    }

    #[test]
    fn blended_e3_h3_geodesic_kernel_validates() {
        let bs = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-2.0, 2.0));
        let src = assemble_geodesic_probe(&bs.wgsl_impl());
        validate_wgsl(&src)
            .expect("BlendedSpace<E3,H3,LinearBlendX> + geodesic kernel should validate");
    }

    // CPU port of `kernel.wgsl::loam_march_geodesic`, mirrored line-for-line, so
    // the `cpu_march_*` tests can check hit points against a known SDF without a
    // GPU adapter. `loam_max_arc` is a parameter here (the kernel reads it as a
    // prelude constant) so each test pins the value it exercises.
    fn march_geodesic_cpu<S: Space<Point = Vec3, Vector = Vec3>>(
        space: &S,
        sdf: impl Fn(Vec3) -> f32,
        ro: Vec3,
        rd: Vec3,
        ball_scale: f32,
        loam_max_arc: f32,
    ) -> Option<(Vec3, f32)> {
        let scale = ball_scale.max(1e-5);
        let mut p = ro * scale;

        let rd_unit = rd.try_normalize().unwrap_or(Vec3::new(0.0, 0.0, -1.0));
        let probe_eps = 1e-4_f32;
        let probed = space.exp(p, rd_unit * probe_eps);
        let riem_norm = space.distance(p, probed) / probe_eps;
        let mut v = rd_unit / riem_norm.max(1e-7);

        let mut t_scene = 0.0_f32;
        let mut t_arc = 0.0_f32;
        let hit_eps = 0.001 * scale;
        let min_step = 0.0001 * scale;

        for _ in 0..256 {
            // `loam_origin_distance(p)` in the kernel; Riemannian distance from
            // origin to p for every Space shipped today.
            if space.distance(Vec3::ZERO, p) > loam_max_arc * 0.92 {
                return None;
            }
            let d = sdf(p);
            if d < hit_eps {
                return Some((p, t_scene));
            }
            if t_scene > 40.0 || t_arc > loam_max_arc {
                return None;
            }
            let step = (d * 0.85).max(min_step);
            let next_p = space.exp(p, v * step);
            let next_v = space.parallel_transport(p, next_p, v);
            p = next_p;
            if next_v.length_squared() > 1e-12 {
                v = next_v;
            }
            t_scene += step / scale;
            t_arc += step;
        }
        None
    }

    /// Sphere-trace a unit ray against an origin-centered sphere in EuclideanR3;
    /// the hit should land on the surface within `hit_eps` and `t_scene` equal the
    /// camera-space distance to it.
    #[test]
    fn cpu_march_hits_centered_sphere_in_euclidean_r3() {
        let space = EuclideanR3;
        let sphere_radius = 0.5_f32;
        let sdf = |p: Vec3| p.length() - sphere_radius;
        let ro = Vec3::new(0.0, 0.0, 2.0);
        let rd = Vec3::new(0.0, 0.0, -1.0);
        let (hit, t) = march_geodesic_cpu(&space, sdf, ro, rd, 1.0, 1.0e9)
            .expect("ray should hit centered sphere");

        // Front of the sphere along -Z: (0, 0, 0.5).
        let expected = Vec3::new(0.0, 0.0, sphere_radius);
        let position_drift = (hit - expected).length();
        assert!(
            position_drift < 5e-3,
            "hit {hit:?} should be within hit_eps of expected {expected:?} (drift {position_drift})",
        );

        // t_scene = ro.z - radius = 1.5; tolerance covers last-step overshoot
        // (up to one hit_eps) plus float-stepping noise.
        let expected_t = 1.5_f32;
        assert!(
            (t - expected_t).abs() < 5e-3,
            "t_scene {t} should be ~{expected_t} (within last-step overshoot)",
        );
    }

    /// A ray pointing away from the only object must miss (exit on `t_scene > 40`
    /// or the 256-iteration cap).
    #[test]
    fn cpu_march_misses_when_ray_points_away_in_euclidean_r3() {
        let space = EuclideanR3;
        let sdf = |p: Vec3| p.length() - 0.5;
        let ro = Vec3::new(0.0, 0.0, 2.0);
        let rd = Vec3::new(0.0, 0.0, 1.0); // away from sphere
        let result = march_geodesic_cpu(&space, sdf, ro, rd, 1.0, 1.0e9);
        assert!(
            result.is_none(),
            "ray pointing away from sphere should miss; got {result:?}",
        );
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, Pod, Zeroable)]
    struct GpuCase {
        a: [f32; 4],
        b: [f32; 4],
        v: [f32; 4],
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, Pod, Zeroable)]
    struct GpuOut {
        distance: [f32; 4],
        exp_point: [f32; 4],
        log_vec: [f32; 4],
        transported: [f32; 4],
    }

    const GPU_PROBE: &str = r#"
struct Case {
    a: vec4<f32>,
    b: vec4<f32>,
    v: vec4<f32>,
};

struct ProbeOut {
    distance: vec4<f32>,
    exp_point: vec4<f32>,
    log_vec: vec4<f32>,
    transported: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> cases: array<Case>;
@group(0) @binding(1) var<storage, read_write> out: array<ProbeOut>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let c = cases[i];
    let a = c.a.xyz;
    let b = c.b.xyz;
    let v = c.v.xyz;
    out[i].distance = vec4<f32>(loam_distance(a, b), 0.0, 0.0, 0.0);
    out[i].exp_point = vec4<f32>(loam_exp(a, v), 0.0);
    out[i].log_vec = vec4<f32>(loam_log(a, b), 0.0);
    out[i].transported = vec4<f32>(loam_parallel_transport(a, b, v), 0.0);
}
"#;

    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing Space WGSL"]
    fn euclidean_space_gpu_probe_matches_cpu() {
        let space = EuclideanR3;
        let cases = [
            gpu_case(
                Vec3::new(0.1, 0.2, 0.3),
                Vec3::new(0.5, -0.1, 0.0),
                Vec3::new(0.01, 0.02, -0.03),
            ),
            gpu_case(
                Vec3::new(-2.0, 4.0, 0.5),
                Vec3::new(1.5, 0.25, -3.0),
                Vec3::new(0.5, -0.25, 0.125),
            ),
        ];
        let out = pollster::block_on(run_gpu_probe(&space, &cases)).expect("GPU probe");
        assert_probe_matches_cpu(&space, &cases, &out, 1e-5);
    }

    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing Space WGSL"]
    fn spherical_space_gpu_probe_matches_cpu() {
        let space = SphericalS3;
        // Points must be inside the upper hemisphere: |p|² < 1.
        let cases = [
            gpu_case(
                Vec3::new(0.1, 0.2, 0.3),
                Vec3::new(0.2, -0.1, 0.05),
                Vec3::new(0.01, 0.02, -0.03),
            ),
            gpu_case(
                Vec3::new(-0.3, 0.4, 0.1),
                Vec3::new(0.2, 0.3, -0.2),
                Vec3::new(0.02, -0.01, 0.015),
            ),
        ];
        let out = pollster::block_on(run_gpu_probe(&space, &cases)).expect("GPU probe");
        assert_probe_matches_cpu(&space, &cases, &out, 2e-4);
    }

    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing Space WGSL"]
    fn hyperbolic_space_gpu_probe_matches_cpu() {
        let space = HyperbolicH3;
        let cases = [
            gpu_case(
                Vec3::new(0.1, 0.2, 0.05),
                Vec3::new(0.2, -0.1, 0.08),
                Vec3::new(0.01, 0.02, -0.015),
            ),
            gpu_case(
                Vec3::new(-0.25, 0.1, 0.05),
                Vec3::new(0.15, 0.2, -0.1),
                Vec3::new(0.02, -0.015, 0.01),
            ),
        ];
        let out = pollster::block_on(run_gpu_probe(&space, &cases)).expect("GPU probe");
        assert_probe_matches_cpu(&space, &cases, &out, 2e-4);
    }

    fn gpu_case(a: Vec3, b: Vec3, v: Vec3) -> GpuCase {
        GpuCase {
            a: a.extend(0.0).to_array(),
            b: b.extend(0.0).to_array(),
            v: v.extend(0.0).to_array(),
        }
    }

    async fn run_gpu_probe<S: WgslSpace>(
        space: &S,
        cases: &[GpuCase],
    ) -> Result<Vec<GpuOut>, String> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("request_adapter failed: {e}"))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("loam-space-gpu-probe"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
                experimental_features: Default::default(),
            })
            .await
            .map_err(|e| format!("request_device failed: {e}"))?;

        let source = assemble_source(&space.wgsl_impl(), GPU_PROBE);
        validate_wgsl(&source).map_err(|e| e.to_string())?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("loam-space-gpu-probe"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let input_size = std::mem::size_of_val(cases) as u64;
        let output_size = (cases.len() * std::mem::size_of::<GpuOut>()) as u64;

        let input = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("loam-space-gpu-probe-input"),
            size: input_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        input
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(cases));
        input.unmap();

        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("loam-space-gpu-probe-output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("loam-space-gpu-probe-staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("loam-space-gpu-probe-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("loam-space-gpu-probe-bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("loam-space-gpu-probe-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("loam-space-gpu-probe-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("loam-space-gpu-probe-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("loam-space-gpu-probe-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(cases.len() as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output_size);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            tx.send(res).expect("map callback receiver should exist");
        });
        // `submission_index = None` waits on the most recent submission;
        // `timeout = None` waits indefinitely.
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| e.to_string())?;
        rx.recv()
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;

        let data = slice.get_mapped_range();
        let rows = bytemuck::cast_slice::<u8, GpuOut>(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(rows)
    }

    fn assert_probe_matches_cpu<S>(space: &S, cases: &[GpuCase], out: &[GpuOut], eps: f32)
    where
        S: Space<Point = Vec3, Vector = Vec3>,
    {
        assert_eq!(cases.len(), out.len());
        for (case, row) in cases.iter().zip(out) {
            let a = Vec3::from_array([case.a[0], case.a[1], case.a[2]]);
            let b = Vec3::from_array([case.b[0], case.b[1], case.b[2]]);
            let v = Vec3::from_array([case.v[0], case.v[1], case.v[2]]);

            assert_near(row.distance[0], space.distance(a, b), eps);
            assert_vec3_near(row.exp_point, space.exp(a, v), eps);
            assert_vec3_near(row.log_vec, space.log(a, b), eps);
            assert_vec3_near(row.transported, space.parallel_transport(a, b, v), eps);
        }
    }

    /// CPU/GPU parity for `BlendedSpace<EuclideanR3, HyperbolicH3, LinearBlendX>`,
    /// restricted to `loam_exp` (the highest-leverage method; transport has its own
    /// probe). `loam_log` and `loam_distance` are intentionally divergent (chart-diff
    /// vs Gauss-Newton; midpoint chord-metric vs full Riemannian) and the kernel
    /// doesn't call `loam_log`.
    ///
    /// Tolerance 5e-3: GPU runs 16 RK4 sub-steps, CPU 32; both 4th-order, so
    /// cumulative drift grows ~16x but stays small for the smooth conformal factor
    /// and the small `v` well inside the H3 ball.
    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing BlendedSpace WGSL"]
    fn blended_e3_h3_gpu_probe_exp_matches_cpu() {
        let space = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-0.5, 0.5));
        let cases = [
            // Pure E3 (alpha = 0): straight-line motion.
            gpu_case(
                Vec3::new(-1.0, 0.05, 0.0),
                Vec3::new(-0.8, 0.05, 0.0),
                Vec3::new(0.1, 0.0, 0.0),
            ),
            // Mid-zone (alpha ~ 0.5): exercises the conformal factor's gradient.
            gpu_case(
                Vec3::new(0.0, 0.05, 0.0),
                Vec3::new(0.1, 0.05, 0.0),
                Vec3::new(0.05, 0.0, 0.0),
            ),
            // Pure H3 (alpha = 1) at r=0.7: f(p) ~ 15.4x identity, non-linear.
            gpu_case(
                Vec3::new(0.7, 0.0, 0.0),
                Vec3::new(0.71, 0.05, 0.0),
                Vec3::new(0.02, 0.02, 0.0),
            ),
        ];
        let out =
            pollster::block_on(run_gpu_probe(&space, &cases)).expect("BlendedSpace GPU probe");

        for (case, row) in cases.iter().zip(&out) {
            let a = Vec3::from_array([case.a[0], case.a[1], case.a[2]]);
            let v = Vec3::from_array([case.v[0], case.v[1], case.v[2]]);
            let cpu = space.exp(a, v);
            let gpu = Vec3::new(row.exp_point[0], row.exp_point[1], row.exp_point[2]);
            let diff = (cpu - gpu).length();
            assert!(
                diff < 5e-3,
                "BlendedSpace exp parity failed at a={a:?} v={v:?}: cpu={cpu:?} gpu={gpu:?} diff={diff}",
            );
        }
    }

    /// CPU/GPU parity for
    /// `BlendedSpace<EuclideanR3, HyperbolicH3, LinearBlendX>::parallel_transport`.
    /// Both sides run 8 RK4 sub-steps along the chart-coordinate line `a` to `b`.
    /// Paths sample pure E3 (identity transport), the mid-zone, and pure H3 at
    /// moderate radius. Tolerance matches the exp probe's 5e-3.
    #[test]
    #[ignore = "requires a working wgpu adapter; run manually when changing BlendedSpace WGSL"]
    fn blended_e3_h3_gpu_probe_transport_matches_cpu() {
        let space = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-0.5, 0.5));
        let cases = [
            // Pure E3: transport is identity; drift is pure GPU-vs-CPU noise.
            gpu_case(
                Vec3::new(-1.0, 0.05, 0.0),
                Vec3::new(-0.8, 0.05, 0.0),
                Vec3::new(0.1, 0.0, 0.0),
            ),
            // Long traversal across the transition zone into H3 at r ~ 0.7; the
            // case that discriminates 8-step RK4 from single-step Euler.
            gpu_case(
                Vec3::new(-0.6, 0.0, 0.0),
                Vec3::new(0.7, 0.0, 0.0),
                Vec3::new(0.5, 0.5, 0.0),
            ),
            // Pure H3 at r ~ 0.7 where f(p) ~ 15.4x identity.
            gpu_case(
                Vec3::new(0.7, 0.0, 0.0),
                Vec3::new(0.72, 0.05, 0.0),
                Vec3::new(0.02, 0.02, 0.0),
            ),
        ];
        let out =
            pollster::block_on(run_gpu_probe(&space, &cases)).expect("BlendedSpace GPU probe");

        for (case, row) in cases.iter().zip(&out) {
            let a = Vec3::from_array([case.a[0], case.a[1], case.a[2]]);
            let b = Vec3::from_array([case.b[0], case.b[1], case.b[2]]);
            let v = Vec3::from_array([case.v[0], case.v[1], case.v[2]]);
            let cpu = space.parallel_transport(a, b, v);
            let gpu = Vec3::new(row.transported[0], row.transported[1], row.transported[2]);
            let diff = (cpu - gpu).length();
            assert!(
                diff < 5e-3,
                "BlendedSpace transport parity failed at a={a:?} b={b:?} v={v:?}: cpu={cpu:?} gpu={gpu:?} diff={diff}",
            );
        }
    }

    fn assert_vec3_near(actual: [f32; 4], expected: Vec3, eps: f32) {
        assert_near(actual[0], expected.x, eps);
        assert_near(actual[1], expected.y, eps);
        assert_near(actual[2], expected.z, eps);
    }

    fn assert_near(actual: f32, expected: f32, eps: f32) {
        assert!(
            (actual - expected).abs() <= eps,
            "actual {actual} differs from expected {expected} by more than {eps}",
        );
    }

    /// Hot-reload's text path: read, assemble against a Space prelude, validate via
    /// naga, mutate, repeat. Pins the Device-free I/O + assembly + validation that
    /// [`ShaderDb::reload`] depends on (the Device-bound layer is just
    /// `create_shader_module` over the same validated source).
    #[test]
    fn hot_reload_pipeline_reads_assembles_and_validates_mutated_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("probe.wgsl");

        let v1 = ABI_PROBE;
        std::fs::write(&path, v1).unwrap();

        let read1 = std::fs::read_to_string(&path).unwrap();
        let src1 = assemble_source(&EuclideanR3.wgsl_impl(), &read1);
        validate_wgsl(&src1).expect("v1 must validate");

        // Mutate in place (same path, changed bytes, as the watcher sees a save);
        // tweak a constant, not structure, so v2 still validates.
        let v2 = ABI_PROBE.replace("vec3<f32>(0.1, 0.2, 0.3)", "vec3<f32>(0.4, 0.5, 0.6)");
        assert_ne!(v1, v2, "test mutation should produce different source");
        std::fs::write(&path, &v2).unwrap();

        let read2 = std::fs::read_to_string(&path).unwrap();
        assert_ne!(read1, read2, "file mutation should change bytes");
        let src2 = assemble_source(&EuclideanR3.wgsl_impl(), &read2);
        validate_wgsl(&src2).expect("v2 must validate after mutation");
    }
}
