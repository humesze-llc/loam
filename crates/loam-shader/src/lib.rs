//! `loam-shader`: WGSL loading, Space intrinsic injection, hot reload.
//!
//! [`ShaderDb`] owns compiled `wgpu::ShaderModule`s and rebuilds them on filesystem
//! events from [`loam_asset::AssetWatcher`]. When a shader is loaded, the active
//! [`loam_math::WgslSpace`]'s `wgsl_impl` is prepended to the user source (and an optional
//! scene module can be inserted between them), so shader authors can call `loam_distance`,
//! `loam_exp`, etc. without manually importing anything.
//!
//! ## Scope note
//!
//! v0 is deliberately minimal: plain string concatenation, no preprocessor. When a second
//! shader in this workspace needs to share code with a first, we'll add `naga_oil` for real
//! `#import` resolution. The public API is designed to be stable across that change, only
//! the internal compile path swaps.

mod db;

pub use db::{validate_wgsl, ShaderDb, ShaderId, WgslValidationError};

/// Geodesic ray march kernel: `loam_safe_normalize`, `loam_march_geodesic`,
/// `loam_estimate_normal`. Assembled between the scene SDF and user shading
/// WGSL by [`ShaderDb::load_geodesic_scene`].
pub const GEODESIC_MARCH_KERNEL: &str = include_str!("kernel.wgsl");
