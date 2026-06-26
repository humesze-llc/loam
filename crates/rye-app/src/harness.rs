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

    // A `.png` target over multiple frames is a contact sheet; a directory is a
    // sequence; a lone `.png` is that frame. The CSV curve dump rides alongside.
    let montage = total > 1 && is_png(cfg.out);

    let mut written = Vec::new();
    let mut frames: Vec<image::RgbaImage> = if montage {
        Vec::with_capacity(total)
    } else {
        Vec::new()
    };
    let mut names: Vec<&'static str> = Vec::new();
    let mut rows: Vec<(f32, Vec<f32>)> = Vec::with_capacity(total);

    for i in 0..total {
        drive_ui(&egui_ctx, &mut scene, &rd, cfg, sim_t, dt);

        // Sample the curve values for the state about to be rendered.
        let scalars = scene.debug_scalars();
        if i == 0 {
            names = scalars.iter().map(|(n, _)| *n).collect();
        }
        rows.push((sim_t, scalars.iter().map(|(_, v)| *v).collect()));

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

        if montage {
            frames.push(
                image::RgbaImage::from_raw(img.width, img.height, img.rgba)
                    .context("frame buffer size mismatch")?,
            );
        } else {
            let path = frame_path(cfg.out, i, total);
            write_png(&path, &img.rgba, img.width, img.height)?;
            written.push(path);
        }

        if i + 1 < total {
            advance(&mut scene, &rd, dt, sim_t);
            sim_t += dt;
        }
    }

    if montage {
        written.push(write_montage(cfg.out, &frames)?);
    }
    if !names.is_empty() {
        let csv = csv_path(cfg.out);
        write_csv(&csv, &rd, cfg, &names, &rows)?;
        written.push(csv);
    }

    Ok(written)
}

/// Composite frames into a reading-order grid contact sheet (roughly square,
/// one-third scale, thin gaps). No labels burned in: the time axis lives in the
/// CSV sidecar, which keeps font deps out.
fn write_montage(out: &Path, frames: &[image::RgbaImage]) -> Result<PathBuf> {
    const SCALE: u32 = 3;
    const GAP: u32 = 4;
    let bg = image::Rgba([20, 24, 30, 255]);

    let n = frames.len();
    let cols = (n as f32).sqrt().ceil() as u32;
    let rows = (n as u32).div_ceil(cols);
    let (cw, ch) = (frames[0].width() / SCALE, frames[0].height() / SCALE);
    let width = cols * cw + (cols + 1) * GAP;
    let height = rows * ch + (rows + 1) * GAP;

    let mut canvas = image::RgbaImage::from_pixel(width, height, bg);
    for (i, f) in frames.iter().enumerate() {
        let cell = image::imageops::resize(f, cw, ch, image::imageops::FilterType::Triangle);
        let col = i as u32 % cols;
        let row = i as u32 / cols;
        let x = (GAP + col * (cw + GAP)) as i64;
        let y = (GAP + row * (ch + GAP)) as i64;
        image::imageops::overlay(&mut canvas, &cell, x, y);
    }

    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create output dir {}", parent.display()))?;
    }
    canvas
        .save_with_format(out, image::ImageFormat::Png)
        .with_context(|| format!("write montage {}", out.display()))?;
    Ok(out.to_path_buf())
}

/// CSV curve dump: a determinism-context header comment, then `frame,time,<scalar
/// columns>`. The header pins what the pixels depend on (adapter/backend/format/
/// size) so a cross-machine mismatch is explained, not mysterious.
fn write_csv(
    path: &Path,
    rd: &RenderDevice,
    cfg: &OfflineRender,
    names: &[&str],
    rows: &[(f32, Vec<f32>)],
) -> Result<()> {
    // Built with `push_str`/`format!`, infallible to a String, so the cold path
    // carries no fallible formatting calls to handle.
    let info = rd.adapter.get_info();
    let mut s = format!(
        "# adapter={} backend={:?} format={:?} size={}x{} from={} to={} fps={}\n",
        info.name,
        info.backend,
        rd.target_format(),
        cfg.width,
        cfg.height,
        cfg.from,
        cfg.to,
        cfg.fps,
    );
    s.push_str("frame,time");
    for n in names {
        s.push(',');
        s.push_str(n);
    }
    s.push('\n');
    for (i, (t, vals)) in rows.iter().enumerate() {
        s.push_str(&format!("{i},{t}"));
        for v in vals {
            s.push_str(&format!(",{v}"));
        }
        s.push('\n');
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create output dir {}", parent.display()))?;
    }
    std::fs::write(path, s).with_context(|| format!("write csv {}", path.display()))
}

/// `true` if `p` ends in a `.png` extension (case-insensitive).
fn is_png(p: &Path) -> bool {
    p.extension().is_some_and(|e| e.eq_ignore_ascii_case("png"))
}

/// CSV sidecar path: `foo.png` -> `foo.csv`; a directory -> `dir/curves.csv`.
fn csv_path(out: &Path) -> PathBuf {
    if is_png(out) {
        out.with_extension("csv")
    } else {
        out.join("curves.csv")
    }
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
