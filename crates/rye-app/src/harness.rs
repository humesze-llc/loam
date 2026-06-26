//! Offline / headless render harness: drive a [`Scene`] through a surfaceless
//! [`RenderDevice`] and read the result back to disk, with no window, swapchain,
//! or live input. The deterministic-iteration counterpart to the live
//! [`crate::capture`] module: identical GPU readback, but the timeline is
//! caller-driven by a fixed dt rather than wall-clock, so a given input
//! reproduces a given frame bit-for-bit on the same adapter. Native + `harness`
//! feature only.
//!
//! Determinism scope: the *driven state* (sim + fixed-dt animation) is
//! reproducible by construction; the *pixels* are reproducible only on the same
//! GPU/driver, since wgpu rasterization is not bit-portable across adapters.

use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rye_input::FrameInput;
use rye_render::device::RenderDevice;
use rye_render::Viewport;
use rye_shader::ShaderDb;

use crate::capture;
use crate::scene::Scene;
use crate::{FrameCtx, SetupCtx};

/// A headless render request: dimensions, the timeline window `[from, to]`
/// seconds sampled at `fps`, and the output path.
pub struct OfflineRender<'a> {
    pub width: u32,
    pub height: u32,
    pub from: f32,
    pub to: f32,
    pub fps: u32,
    /// Single frame (one sample) writes here when it ends in `.png`; a sequence
    /// treats this as a directory and writes `frame_NNNN.png` into it.
    pub out: &'a Path,
}

/// Render `S` over the configured timeline and write the frames. Builds a
/// headless device, runs `Scene::new`, fast-forwards to `from` with fixed-dt
/// `update` calls, then renders each sample at `dt = 1/fps`, advancing the scene
/// by one `update(dt)` between samples. Scene-only: `ui` (the egui overlay) is
/// not driven here; cursor injection via synthetic egui input lands in M2.
/// Returns the written paths in order.
pub fn render_scene<S: Scene>(cfg: &OfflineRender) -> Result<Vec<PathBuf>> {
    let rd = pollster::block_on(RenderDevice::new_headless(cfg.width, cfg.height))
        .context("new_headless")?;

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

    let fps = cfg.fps.max(1);
    let dt = 1.0 / fps as f32;
    // Inclusive endpoints: `from==to` yields a single frame; 0..2.4 @ 12 yields
    // ceil-of-rounded intervals plus one.
    let span = (cfg.to - cfg.from).max(0.0);
    let intervals = (span * fps as f32).round() as usize;
    let total = intervals + 1;

    // Fast-forward to `from` so a window not starting at 0 still replays the
    // exact fixed-dt state history that led there.
    let pre = (cfg.from * fps as f32).round() as usize;
    for _ in 0..pre {
        let mut fctx = frame_ctx(&rd, dt, dt);
        scene.update(&mut fctx);
    }

    let view = rd
        .headless_view()
        .expect("new_headless always allocates a headless color target");
    let texture = rd
        .headless_texture()
        .expect("new_headless always allocates a headless color target");

    let mut written = Vec::with_capacity(total);
    for i in 0..total {
        scene
            .render(&rd, view, Viewport::full([cfg.width, cfg.height]))
            .map_err(|e| e.context("Scene::render"))?;
        let img =
            capture::read_texture_rgba(&rd.device, &rd.queue, texture, cfg.width, cfg.height, rd.target_format())
                .context("headless readback")?;
        let path = frame_path(cfg.out, i, total);
        write_png(&path, &img.rgba, img.width, img.height)?;
        written.push(path);

        if i + 1 < total {
            let t = cfg.from + (i + 1) as f32 * dt;
            let mut fctx = frame_ctx(&rd, dt, t);
            scene.update(&mut fctx);
        }
    }

    Ok(written)
}

/// Construct a synthetic per-frame context: fixed `dt`, empty input, no UI
/// focus. `time` is informational (flatland's animation advances on `dt`).
fn frame_ctx<'a>(rd: &'a RenderDevice, dt: f32, time: f32) -> FrameCtx<'a> {
    FrameCtx {
        rd,
        input: FrameInput::default(),
        time,
        fps: 1.0 / dt,
        n_ticks: 0,
        tick: 0,
        dt,
        ui_has_focus: false,
        _non_exhaustive: PhantomData,
    }
}

/// Output path for frame `i`. A lone frame to a `.png` target writes that file;
/// otherwise frames are `frame_NNNN.png` under the `out` directory.
fn frame_path(out: &Path, i: usize, total: usize) -> PathBuf {
    if total == 1 && out.extension().is_some_and(|e| e.eq_ignore_ascii_case("png")) {
        return out.to_path_buf();
    }
    out.join(format!("frame_{i:04}.png"))
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
