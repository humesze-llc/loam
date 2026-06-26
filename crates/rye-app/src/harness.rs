//! Offline / headless render harness: drive a [`Scene`] through a surfaceless
//! [`RenderDevice`] and read the result back to disk, with no window, swapchain,
//! or live input. The deterministic-iteration counterpart to the live
//! [`crate::capture`] module: identical GPU readback, but frame timing is
//! caller-chosen rather than wall-clock, so a given input reproduces a given
//! frame. Native + `harness` feature only.
//!
//! M0 is one scene-only frame at t=0 ([`render_scene_frame`]); the fixed-step
//! update loop, synthetic egui input, and montage land in later milestones.

use std::path::Path;

use anyhow::{Context, Result};
use rye_render::device::RenderDevice;
use rye_render::Viewport;
use rye_shader::ShaderDb;

use crate::capture;
use crate::scene::Scene;
use crate::SetupCtx;

/// Render a single frame of `S` at its just-constructed (t=0) state into a PNG
/// at `out`. Builds a headless device, runs `Scene::new`, renders once, reads
/// the color target back, and writes it. No `update` is called: the frame is
/// whatever the scene's constructor establishes.
pub fn render_scene_frame<S: Scene>(width: u32, height: u32, out: &Path) -> Result<()> {
    let rd =
        pollster::block_on(RenderDevice::new_headless(width, height)).context("new_headless")?;

    let mut shader_db = ShaderDb::new(rd.device.clone());
    let mut scene = {
        let mut ctx = SetupCtx {
            rd: &rd,
            shader_db: &mut shader_db,
            // No hot-reload offline: the harness renders a fixed input, not a
            // live editing session.
            watcher: None,
            time: 0.0,
        };
        S::new(&mut ctx).map_err(|e| e.context("Scene::new"))?
    };

    let view = rd
        .headless_view()
        .expect("new_headless always allocates a headless color target");
    scene
        .render(&rd, view, Viewport::full([width, height]))
        .map_err(|e| e.context("Scene::render"))?;

    let texture = rd
        .headless_texture()
        .expect("new_headless always allocates a headless color target");
    let img = capture::read_texture_rgba(
        &rd.device,
        &rd.queue,
        texture,
        width,
        height,
        rd.target_format(),
    )
    .context("headless readback")?;

    write_png(out, &img.rgba, img.width, img.height)
}

/// Write tightly-packed RGBA8 to a PNG, creating the parent directory. Mirrors
/// `capture::write_png_bytes`, which is private to that module.
fn write_png(path: &Path, rgba: &[u8], width: u32, height: u32) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create output dir {}", parent.display()))?;
    }
    let img: image::RgbaImage = image::ImageBuffer::from_raw(width, height, rgba.to_vec())
        .with_context(|| format!("RGBA buffer doesn't match {width}x{height}"))?;
    img.save_with_format(path, image::ImageFormat::Png)
        .with_context(|| format!("write png {}", path.display()))
}
