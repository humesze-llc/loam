//! Camera abstractions for Loam.
//!
//! Two layers, used together:
//!
//! 1. [`Camera<S>`]: Space-generic position + tangent frame. Pure data; storage
//!    agnostic to which controller is driving it. Works for any `Space` whose `Point`
//!    and `Vector` are `glam::Vec3` (i.e. all the closed-form 3D Spaces today).
//! 2. [`CameraController`]: input-driven logic that mutates a `Camera<S>` each frame.
//!    Concrete impls: [`OrbitController`], [`FirstPersonController`].

mod camera;
mod controller;

pub use camera::{Camera, Ray};
pub use controller::{CameraController, FirstPersonController, OrbitController};

use glam::Vec3;

/// Camera basis vectors produced each frame; feed directly into shader uniforms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraView {
    pub position: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
}
