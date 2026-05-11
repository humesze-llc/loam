//! Frame capture: PNG single-shot snapshots and PNG sequences, with two taps
//! (`pre`-egui = pure 3D scene, `post`-egui = final composite as DWM receives it).
//!
//! The diagnostic priority. PNG sequences write one independent file per frame so an
//! aliasing or compositor bug can be inspected pixel-by-pixel without inter-frame
//! compression artifacts obscuring the signal. The `pre`/`post` split lets the caller
//! attribute the artifact to the raymarcher, the egui paint stage, or DWM.
//!
//! ## How requests flow
//!
//! Console commands push [`CaptureRequest`]s onto a global queue via [`enqueue`]. The
//! [`Runner`](crate::Runner) drains the queue once per frame, mutates the [`Capture`]
//! state machine, and issues GPU copies at the two tap points in the render loop. PNG
//! writes happen synchronously on the main thread (poll-wait on the buffer map); a
//! capture frame is allowed to stutter render rate.
//!
//! ## Pre-egui tap and MSAA
//!
//! With MSAA off, the 3D pass writes directly to the swapchain view, so the pre-egui
//! tap can copy it as-is. With MSAA on, the 3D content sits in the multisampled
//! attachment and is only resolved into the swapchain at the end of the egui pass; a
//! direct copy of multisamples isn't supported. Phase 1 skips pre-egui captures when
//! MSAA is on; disable MSAA via [`RunConfig::msaa_samples`](crate::RunConfig) for
//! diagnostic capture sessions.
//!
//! ## Output layout
//!
//! - One-shot: `{dir}/{name}_post.png` (or `_pre.png` / both)
//! - Sequence: `{dir}/{name}/{stage}_{frame:06}.png`
//!
//! `dir` defaults to `./captures/`; `name` defaults to `{example}_{unix_secs}`.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context as _, Result};
use wgpu::{
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Device, Extent3d, MapMode, Origin3d,
    PollType, Queue, TexelCopyBufferInfo, TexelCopyBufferLayout, TexelCopyTextureInfo, Texture,
    TextureAspect, TextureFormat,
};

use rye_egui::{cmd, Console, ConsoleWriter};

/// Where in the frame pipeline the capture is taken from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CaptureStage {
    /// Pure 3D scene before egui paints. Requires MSAA off.
    Pre,
    /// Final composite after egui paints. What DWM receives.
    Post,
    /// Both, written to two separate files per frame.
    Both,
}

impl CaptureStage {
    fn wants_pre(self) -> bool {
        matches!(self, CaptureStage::Pre | CaptureStage::Both)
    }
    fn wants_post(self) -> bool {
        matches!(self, CaptureStage::Post | CaptureStage::Both)
    }
}

/// A capture command queued for the runner. Pushed by console commands and hotkey
/// binds via [`enqueue`]; drained once per frame.
#[derive(Debug)]
pub enum CaptureRequest {
    /// Capture exactly one frame and stop.
    OneShot {
        stage: CaptureStage,
        dir: Option<PathBuf>,
        name: Option<String>,
    },
    /// Start a PNG sequence. Continues until [`CaptureRequest::Stop`].
    StartSequence {
        stage: CaptureStage,
        dir: Option<PathBuf>,
        name: Option<String>,
    },
    /// Stop the current sequence, if any. No-op when idle.
    Stop,
    /// If a sequence is running, stop it; otherwise start a new sequence with the given
    /// stage and dir. The handy F9-bound shape: press to start, press again to stop.
    Toggle {
        stage: CaptureStage,
        dir: Option<PathBuf>,
        name: Option<String>,
    },
}

static QUEUE: Mutex<Vec<CaptureRequest>> = Mutex::new(Vec::new());

/// Push a capture request onto the queue. Drained by the runner on the next frame.
pub fn enqueue(req: CaptureRequest) {
    QUEUE.lock().expect("capture queue poisoned").push(req);
}

pub(crate) fn drain_requests() -> Vec<CaptureRequest> {
    std::mem::take(&mut *QUEUE.lock().expect("capture queue poisoned"))
}

/// Runner-owned state machine. Drives the per-frame copy + write.
pub(crate) struct Capture {
    default_dir: PathBuf,
    state: CaptureState,
}

enum CaptureState {
    Idle,
    OneShot {
        stage: CaptureStage,
        path_pre: Option<PathBuf>,
        path_post: Option<PathBuf>,
        /// True once we've issued the copies; used to transition to Idle after writes.
        consumed: bool,
    },
    Sequence {
        stage: CaptureStage,
        dir: PathBuf,
        frame_idx: u32,
    },
}

impl Capture {
    pub(crate) fn new() -> Self {
        Self {
            default_dir: PathBuf::from("captures"),
            state: CaptureState::Idle,
        }
    }

    pub(crate) fn apply_requests(&mut self, requests: Vec<CaptureRequest>) -> Vec<String> {
        let mut log = Vec::new();
        for req in requests {
            match req {
                CaptureRequest::OneShot { stage, dir, name } => {
                    let dir = dir.unwrap_or_else(|| self.default_dir.clone());
                    let name = name.unwrap_or_else(default_name);
                    let path_pre = stage
                        .wants_pre()
                        .then(|| dir.join(format!("{name}_pre.png")));
                    let path_post = stage
                        .wants_post()
                        .then(|| dir.join(format!("{name}_post.png")));
                    self.state = CaptureState::OneShot {
                        stage,
                        path_pre,
                        path_post,
                        consumed: false,
                    };
                    log.push(format!("capture: one-shot queued ({stage:?})"));
                }
                CaptureRequest::StartSequence { stage, dir, name } => {
                    let dir = dir
                        .unwrap_or_else(|| self.default_dir.clone())
                        .join(name.unwrap_or_else(default_name));
                    self.state = CaptureState::Sequence {
                        stage,
                        dir,
                        frame_idx: 0,
                    };
                    log.push(format!("capture: sequence started ({stage:?})"));
                }
                CaptureRequest::Stop => self.stop(&mut log),
                CaptureRequest::Toggle { stage, dir, name } => {
                    if matches!(self.state, CaptureState::Sequence { .. }) {
                        self.stop(&mut log);
                    } else {
                        let dir = dir
                            .unwrap_or_else(|| self.default_dir.clone())
                            .join(name.unwrap_or_else(default_name));
                        self.state = CaptureState::Sequence {
                            stage,
                            dir,
                            frame_idx: 0,
                        };
                        log.push(format!("capture: sequence started ({stage:?})"));
                    }
                }
            }
        }
        log
    }

    fn stop(&mut self, log: &mut Vec<String>) {
        match &self.state {
            CaptureState::Sequence { frame_idx, dir, .. } => {
                log.push(format!(
                    "capture: sequence stopped, {frame_idx} frame(s) at {}",
                    dir.display()
                ));
                self.state = CaptureState::Idle;
            }
            CaptureState::OneShot { .. } => {
                self.state = CaptureState::Idle;
                log.push("capture: pending one-shot cancelled".into());
            }
            CaptureState::Idle => {
                log.push("capture: stop with no active session (no-op)".into());
            }
        }
    }

    pub(crate) fn wants_pre(&self) -> bool {
        match &self.state {
            CaptureState::Idle => false,
            CaptureState::OneShot {
                stage,
                consumed: false,
                ..
            } => stage.wants_pre(),
            CaptureState::OneShot { consumed: true, .. } => false,
            CaptureState::Sequence { stage, .. } => stage.wants_pre(),
        }
    }

    pub(crate) fn wants_post(&self) -> bool {
        match &self.state {
            CaptureState::Idle => false,
            CaptureState::OneShot {
                stage,
                consumed: false,
                ..
            } => stage.wants_post(),
            CaptureState::OneShot { consumed: true, .. } => false,
            CaptureState::Sequence { stage, .. } => stage.wants_post(),
        }
    }

    /// Resolve `(path_pre, path_post)` for the current frame, given a `stage` mask.
    pub(crate) fn frame_paths(&self) -> (Option<PathBuf>, Option<PathBuf>) {
        match &self.state {
            CaptureState::Idle => (None, None),
            CaptureState::OneShot {
                path_pre,
                path_post,
                consumed: false,
                ..
            } => (path_pre.clone(), path_post.clone()),
            CaptureState::OneShot { consumed: true, .. } => (None, None),
            CaptureState::Sequence {
                stage,
                dir,
                frame_idx,
            } => {
                let pre = stage
                    .wants_pre()
                    .then(|| dir.join(format!("pre_{frame_idx:06}.png")));
                let post = stage
                    .wants_post()
                    .then(|| dir.join(format!("post_{frame_idx:06}.png")));
                (pre, post)
            }
        }
    }

    /// Advance the state machine after a successfully written frame. One-shot becomes
    /// Idle; sequence increments the frame counter.
    pub(crate) fn advance_frame(&mut self) {
        match &mut self.state {
            CaptureState::OneShot { consumed, .. } => {
                *consumed = true;
                self.state = CaptureState::Idle;
            }
            CaptureState::Sequence { frame_idx, .. } => {
                *frame_idx = frame_idx.saturating_add(1);
            }
            CaptureState::Idle => {}
        }
    }
}

fn default_name() -> String {
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("capture_{unix}")
}

// ---------------------------------------------------------------------------
// GPU readback
// ---------------------------------------------------------------------------

/// Raw RGBA8 pixels in row-major order, already R/B-swapped if the source format was
/// BGRA. Ready to hand to an image encoder.
pub(crate) struct RawImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Synchronous swapchain-texture readback. Copies the texture into a tightly-packed
/// RGBA8 byte buffer; the caller writes it out as PNG. Phase 1 is sync (poll-wait);
/// a capture frame is allowed to stutter.
pub(crate) fn read_texture_rgba(
    device: &Device,
    queue: &Queue,
    texture: &Texture,
    width: u32,
    height: u32,
    format: TextureFormat,
) -> Result<RawImage> {
    // wgpu requires `bytes_per_row` to be a multiple of 256. We allocate a padded
    // staging buffer, then strip the padding when copying out.
    let unpadded_bpr = width.checked_mul(4).context("width * 4 overflows u32")?;
    let padded_bpr = unpadded_bpr.next_multiple_of(256);
    let buffer_size = (padded_bpr as u64) * (height as u64);

    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("rye-app::capture-staging"),
        size: buffer_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("rye-app::capture-copy"),
    });
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: None,
            },
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    // Map + poll-wait. The closure runs on whichever thread the polling happens on; we
    // only care that the wait blocks until it does.
    let slice = buffer.slice(..);
    slice.map_async(MapMode::Read, |_| {});
    // wgpu v27 made `PollType::Wait` a struct variant. `submission_index = None` waits
    // on the most recent submission; `timeout = None` waits indefinitely (same effective
    // semantics as the v26 unit variant).
    device
        .poll(PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .context("device.poll on capture readback failed")?;

    let data = slice.get_mapped_range();
    let mut rgba = Vec::with_capacity((unpadded_bpr * height) as usize);
    for row in 0..height as usize {
        let start = row * padded_bpr as usize;
        let end = start + unpadded_bpr as usize;
        rgba.extend_from_slice(&data[start..end]);
    }
    drop(data);
    buffer.unmap();

    // sRGB swapchain formats arrive as BGRA on most Windows surfaces; PNG wants RGBA.
    if format_is_bgra(format) {
        for px in rgba.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
    }

    Ok(RawImage {
        width,
        height,
        rgba,
    })
}

fn format_is_bgra(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb
    )
}

/// Write a [`RawImage`] to `path` as PNG. Creates parent directories as needed.
pub(crate) fn write_png(path: &Path, image: &RawImage) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create capture dir {}", parent.display()))?;
    }
    let img: ::image::RgbaImage =
        ::image::ImageBuffer::from_raw(image.width, image.height, image.rgba.clone()).ok_or_else(
            || {
                anyhow!(
                    "RGBA buffer size doesn't match {}x{}",
                    image.width,
                    image.height
                )
            },
        )?;
    img.save_with_format(path, ::image::ImageFormat::Png)
        .with_context(|| format!("write png {}", path.display()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Console command registration
// ---------------------------------------------------------------------------

/// Register `capture` console commands on the given console. Commands all push to the
/// global queue via [`enqueue`]; no context coupling required.
///
/// Commands registered:
/// - `capture png    [pre|post|both] [dir]`: one-shot single PNG
/// - `capture frames [pre|post|both] [dir]`: start PNG sequence
/// - `capture toggle [pre|post|both] [dir]`: start if idle, stop if recording
/// - `capture stop`: stop sequence (or cancel a pending one-shot)
pub fn register_commands<Ctx: 'static>(console: &mut Console<Ctx>) {
    console.register(cmd(
        "capture",
        capture_help(),
        |args, _ctx: &mut Ctx, out| run_capture(args, out),
    ));
}

fn capture_help() -> &'static str {
    "capture <png|frames|toggle|stop> [pre|post|both] [dir]"
}

fn run_capture(args: &[&str], out: &mut ConsoleWriter) -> Result<()> {
    let Some((sub, rest)) = args.split_first() else {
        out.error("usage: capture <png|frames|stop|dir> [pre|post|both] [dir]");
        return Ok(());
    };
    match *sub {
        "png" => {
            let (stage, dir) = parse_stage_and_dir(rest)?;
            enqueue(CaptureRequest::OneShot {
                stage,
                dir,
                name: None,
            });
            out.line(format!("queued one-shot ({stage:?})"));
        }
        "frames" => {
            let (stage, dir) = parse_stage_and_dir(rest)?;
            enqueue(CaptureRequest::StartSequence {
                stage,
                dir,
                name: None,
            });
            out.line(format!("started sequence ({stage:?})"));
        }
        "stop" => {
            enqueue(CaptureRequest::Stop);
            out.line("stop queued");
        }
        "toggle" => {
            let (stage, dir) = parse_stage_and_dir(rest)?;
            enqueue(CaptureRequest::Toggle {
                stage,
                dir,
                name: None,
            });
            out.line(format!("toggle queued ({stage:?})"));
        }
        "dir" => {
            // For Phase 1 the per-command positional arg is the only knob; a persistent
            // default-dir setter is a Phase 2 nicety once we know how often users want it.
            out.error("`capture dir` is reserved; pass the dir as a positional arg to png/frames");
        }
        other => {
            out.error(format!("unknown sub-command `{other}`. {}", capture_help()));
        }
    }
    Ok(())
}

fn parse_stage_and_dir(args: &[&str]) -> Result<(CaptureStage, Option<PathBuf>)> {
    let mut stage = CaptureStage::Post;
    let mut dir: Option<PathBuf> = None;
    for arg in args {
        match *arg {
            "pre" => stage = CaptureStage::Pre,
            "post" => stage = CaptureStage::Post,
            "both" => stage = CaptureStage::Both,
            other => dir = Some(PathBuf::from(other)),
        }
    }
    Ok((stage, dir))
}

/// Bind the default capture hotkeys on the given console:
/// - `F12`: `capture png post` (one-shot screenshot)
/// - `F9`:  `capture toggle post` (press to start a sequence, again to stop)
///
/// Shift-modified binds aren't currently supported by the console's bind table; users
/// can run `capture png both` (or any other variant) from the prompt directly.
pub fn bind_default_hotkeys<Ctx: 'static>(console: &mut Console<Ctx>) {
    console.bind(rye_egui::egui::Key::F12, "capture png post");
    console.bind(rye_egui::egui::Key::F9, "capture toggle post");
}
