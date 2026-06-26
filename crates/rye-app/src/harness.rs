//! Offline / headless render harness: drive a [`Scene`] through a surfaceless
//! [`RenderDevice`] and read the result back to disk, with no window, swapchain,
//! or live input. The deterministic-iteration counterpart to the live
//! [`crate::capture`] module: identical GPU readback, but the timeline is
//! caller-driven by a fixed dt rather than wall-clock, so a given input
//! reproduces a given frame bit-for-bit on the same adapter. Native + `harness`
//! feature only.
//!
//! Cursor-driven scenes are fed a scripted [`CursorTrack`] through a headless
//! [`egui::Context`]: the same channel the live app reads pointer hover from, so
//! gaze / wake behavior reproduces without a mouse. The egui pass runs every
//! frame for its `ui()` side effects; its overlay is not painted into the frame
//! (scene-only). Painting the composed overlay is a later milestone.
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

use crate::egui;
use crate::capture;
use crate::scene::Scene;
use crate::{FrameCtx, SetupCtx};

/// A scripted cursor: hold-keyframed pointer state in egui screen points
/// (pixels at 1x scale). `None` at a key means the cursor is absent (off
/// surface) from that time. Between keys the previous key's state holds.
pub struct CursorTrack {
    /// `(time_seconds, position)`, sorted by time; `None` position is absent.
    keys: Vec<(f32, Option<(f32, f32)>)>,
}

impl CursorTrack {
    /// Build from keyframes; sorted by time so out-of-order input is fine.
    pub fn new(mut keys: Vec<(f32, Option<(f32, f32)>)>) -> Self {
        keys.sort_by(|a, b| a.0.total_cmp(&b.0));
        Self { keys }
    }

    /// Cursor position at `t` (the latest key at or before `t`), or `None` when
    /// no key has fired yet or the active key is absent.
    pub fn sample(&self, t: f32) -> Option<(f32, f32)> {
        let mut cur = None;
        for &(kt, pos) in &self.keys {
            if kt <= t {
                cur = pos;
            } else {
                break;
            }
        }
        cur
    }
}

/// A headless render request: dimensions, the timeline window `[from, to]`
/// seconds sampled at `fps`, an optional scripted cursor, and the output path.
pub struct OfflineRender<'a> {
    pub width: u32,
    pub height: u32,
    pub from: f32,
    pub to: f32,
    pub fps: u32,
    /// Scripted pointer fed through headless egui; `None` = no cursor ever.
    pub cursor: Option<CursorTrack>,
    /// Single frame (one sample) writes here when it ends in `.png`; a sequence
    /// treats this as a directory and writes `frame_NNNN.png` into it.
    pub out: &'a Path,
}

/// Render `S` over the configured timeline and write the frames. Builds a
/// headless device, runs `Scene::new`, fast-forwards to `from`, then renders
/// each sample at `dt = 1/fps`, advancing the scene by one `update(dt)` between
/// samples. Each frame drives `ui()` through a headless egui context with the
/// scripted cursor (so cursor-gated behavior fires), then renders scene-only.
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

    let egui_ctx = egui::Context::default();

    let fps = cfg.fps.max(1);
    let dt = 1.0 / fps as f32;
    // Inclusive endpoints: `from==to` yields a single frame.
    let span = (cfg.to - cfg.from).max(0.0);
    let total = (span * fps as f32).round() as usize + 1;
    let pre = (cfg.from * fps as f32).round() as usize;

    let view = rd
        .headless_view()
        .expect("new_headless always allocates a headless color target");
    let texture = rd
        .headless_texture()
        .expect("new_headless always allocates a headless color target");

    // Sim time advances from 0 so a window that starts after 0 still replays the
    // exact fixed-dt history (cursor included) that led there.
    let mut sim_t = 0.0;
    for _ in 0..pre {
        drive_ui(&egui_ctx, &mut scene, &rd, cfg, sim_t, dt);
        advance(&mut scene, &rd, dt, sim_t);
        sim_t += dt;
    }

    let mut written = Vec::with_capacity(total);
    for i in 0..total {
        drive_ui(&egui_ctx, &mut scene, &rd, cfg, sim_t, dt);
        scene
            .render(&rd, view, Viewport::full([cfg.width, cfg.height]))
            .map_err(|e| e.context("Scene::render"))?;
        let img = capture::read_texture_rgba(
            &rd.device,
            &rd.queue,
            texture,
            cfg.width,
            cfg.height,
            rd.target_format(),
        )
        .context("headless readback")?;
        let path = frame_path(cfg.out, i, total);
        write_png(&path, &img.rgba, img.width, img.height)?;
        written.push(path);

        if i + 1 < total {
            advance(&mut scene, &rd, dt, sim_t);
            sim_t += dt;
        }
    }

    Ok(written)
}

/// Run one fixed-step `update(dt)` on the scene.
fn advance<S: Scene>(scene: &mut S, rd: &RenderDevice, dt: f32, t: f32) {
    let mut fctx = frame_ctx(rd, dt, t);
    scene.update(&mut fctx);
}

/// Drive the scene's egui `ui()` once with the scripted cursor at `t`, through a
/// headless egui context. The overlay output is discarded (scene-only); only the
/// `ui()` side effects (cursor hover -> gaze / wake) are kept.
fn drive_ui<S: Scene>(
    egui_ctx: &egui::Context,
    scene: &mut S,
    rd: &RenderDevice,
    cfg: &OfflineRender,
    t: f32,
    dt: f32,
) {
    // A present cursor is a fresh pointer position; an absent one clears hover.
    let event = match cfg.cursor.as_ref().and_then(|c| c.sample(t)) {
        Some((x, y)) => egui::Event::PointerMoved(egui::pos2(x, y)),
        None => egui::Event::PointerGone,
    };
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::pos2(0.0, 0.0),
            egui::vec2(cfg.width as f32, cfg.height as f32),
        )),
        time: Some(t as f64),
        events: vec![event],
        ..Default::default()
    };
    let mut fctx = frame_ctx(rd, dt, t);
    let _ = egui_ctx.run(raw, |ctx| {
        scene.ui(ctx, &mut fctx);
    });
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
