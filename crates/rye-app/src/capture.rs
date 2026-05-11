//! Frame capture: PNG single-shot snapshots, PNG sequences, and animated GIF streams,
//! with two taps (`pre`-egui = pure 3D scene, `post`-egui = final composite as DWM
//! receives it).
//!
//! The diagnostic priority. PNG sequences write one independent file per frame so an
//! aliasing or compositor bug can be inspected pixel-by-pixel without inter-frame
//! compression artifacts obscuring the signal. The `pre`/`post` split lets the caller
//! attribute the artifact to the raymarcher, the egui paint stage, or DWM.
//!
//! The sharing priority. GIF streams encode incrementally to a single file (palette-
//! quantized via NeuQuant, looped infinitely) for social-media or Discord posts; the
//! quality tradeoff vs PNG is acceptable for short demo clips.
//!
//! ## How requests flow
//!
//! Console commands push [`CaptureRequest`]s onto a global queue via [`enqueue`]. The
//! [`Runner`](crate::Runner) drains the queue once per frame, mutates the [`Capture`]
//! state machine, and issues GPU copies at the two tap points in the render loop.
//! Encoding (PNG write, GIF palette + write) happens synchronously on the main thread.
//! A capture frame is allowed to stutter render rate; async encoding is a Phase 3
//! target if real-time recording becomes important.
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
//! - One-shot PNG: `{dir}/{name}_post.png` (or `_pre.png` / both)
//! - PNG sequence: `{dir}/{name}/{stage}_{frame:06}.png`
//! - GIF stream:   `{dir}/{name}.gif`
//!
//! `dir` defaults to `./captures/`; `name` defaults to `{example}_{unix_secs}`.
//!
//! ## Converting a captured GIF to WebP (for README embeds)
//!
//! GIF is the streaming output we ship. For smaller / higher-quality WebP, post-process
//! with ffmpeg (one pass handles format conversion + downscale + fps cap):
//!
//! ```text
//! ffmpeg -i in.gif \
//!   -vf "fps=30,scale=720:-1:flags=lanczos" \
//!   -loop 0 -lossless 0 -q:v 75 \
//!   out.webp
//! ```

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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
    /// Both, written to two separate files per frame. PNG-only; GIF can't multiplex.
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

/// Output format for streaming captures.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CaptureFormat {
    /// One PNG file per frame; both pre and post stages can be written in parallel.
    /// FPS unlimited by default (every render frame); pass an explicit fps to throttle.
    Png,
    /// Single animated GIF, palette-quantized via NeuQuant, infinite loop. Single
    /// stage only (pre OR post, not both). Default fps 30.
    Gif,
}

impl CaptureFormat {
    fn default_fps(self) -> Option<u16> {
        match self {
            CaptureFormat::Png => None, // unlimited; every frame
            CaptureFormat::Gif => Some(30),
        }
    }

    fn supports_both_stages(self) -> bool {
        matches!(self, CaptureFormat::Png)
    }
}

/// A capture command queued for the runner. Pushed by console commands and hotkey
/// binds via [`enqueue`]; drained once per frame.
#[derive(Debug)]
pub enum CaptureRequest {
    /// Capture exactly one frame as PNG and stop. Stage `Both` writes two files.
    OneShot {
        stage: CaptureStage,
        dir: Option<PathBuf>,
        name: Option<String>,
    },
    /// Start a streaming sequence. Continues until [`CaptureRequest::Stop`] or until
    /// the next [`CaptureRequest::Toggle`].
    StartSequence {
        format: CaptureFormat,
        stage: CaptureStage,
        dir: Option<PathBuf>,
        name: Option<String>,
        /// Capture rate cap in frames per second. `None` means "every render frame";
        /// for GIF the runner uses [`CaptureFormat::default_fps`] when `None`.
        fps: Option<u16>,
        /// Output width in pixels for downscaled streams. Height computed to preserve
        /// aspect ratio. `None` captures at native swapchain resolution. GIF-only;
        /// PNG sequences ignore this for diagnostic fidelity.
        scale: Option<u32>,
    },
    /// Stop the current sequence, if any. No-op when idle.
    Stop,
    /// Toggle a streaming sequence with the given parameters: stop if any sequence is
    /// running, start with these params if idle. The F9-bound default shape.
    Toggle {
        format: CaptureFormat,
        stage: CaptureStage,
        dir: Option<PathBuf>,
        name: Option<String>,
        fps: Option<u16>,
        scale: Option<u32>,
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

// ---------------------------------------------------------------------------
// Capture state machine
// ---------------------------------------------------------------------------

/// Runner-owned state machine. Drives the per-frame copy + write.
pub(crate) struct Capture {
    default_dir: PathBuf,
    state: CaptureState,
}

enum CaptureState {
    Idle,
    OneShot {
        path_pre: Option<PathBuf>,
        path_post: Option<PathBuf>,
    },
    Sequence {
        stage: CaptureStage,
        writer: SequenceWriter,
        /// Minimum interval between captured frames. `None` = unlimited.
        fps_interval: Option<Duration>,
        /// Wall-clock time of the last captured frame; `None` until the first capture.
        last_capture_time: Option<Instant>,
        frame_count: u32,
    },
}

/// Per-sequence-format incremental writer.
enum SequenceWriter {
    /// PNG sequence: one independent file per frame, stage-labelled.
    Png { dir: PathBuf },
    /// Animated GIF: a streaming `gif::Encoder` writing frames to a single file as they
    /// arrive. The encoder requires width + height up front (the LSD is fixed for the
    /// whole stream), so it stays `None` until the first frame establishes the
    /// dimensions. Dropping the encoder flushes the GIF trailer.
    Gif {
        encoder: Option<gif::Encoder<BufWriter<File>>>,
        path: PathBuf,
        /// Pre-computed delay in centiseconds, from the sequence's target fps.
        delay_cs: u16,
        /// Optional output width in pixels. Frames are Lanczos3-resampled before
        /// palette quantization; aspect ratio is preserved.
        scale: Option<u32>,
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
                        path_pre,
                        path_post,
                    };
                    log.push(format!("capture: one-shot queued ({stage:?})"));
                }
                CaptureRequest::StartSequence {
                    format,
                    stage,
                    dir,
                    name,
                    fps,
                    scale,
                } => match self.start_sequence(format, stage, dir, name, fps, scale) {
                    Ok(msg) => log.push(msg),
                    Err(e) => log.push(format!("capture: failed to start sequence: {e:#}")),
                },
                CaptureRequest::Stop => self.stop(&mut log),
                CaptureRequest::Toggle {
                    format,
                    stage,
                    dir,
                    name,
                    fps,
                    scale,
                } => {
                    if matches!(self.state, CaptureState::Sequence { .. }) {
                        self.stop(&mut log);
                    } else {
                        match self.start_sequence(format, stage, dir, name, fps, scale) {
                            Ok(msg) => log.push(msg),
                            Err(e) => log.push(format!("capture: failed to start sequence: {e:#}")),
                        }
                    }
                }
            }
        }
        log
    }

    fn start_sequence(
        &mut self,
        format: CaptureFormat,
        mut stage: CaptureStage,
        dir: Option<PathBuf>,
        name: Option<String>,
        fps: Option<u16>,
        scale: Option<u32>,
    ) -> Result<String> {
        let dir = dir.unwrap_or_else(|| self.default_dir.clone());
        let name = name.unwrap_or_else(default_name);
        let fps = fps.or_else(|| format.default_fps());

        // GIF (and any future single-file format) can't multiplex two stages into one
        // stream. Silently downgrade `Both` to `Post`; the user almost certainly meant
        // the final composite when picking a sharing format.
        if !format.supports_both_stages() && stage == CaptureStage::Both {
            stage = CaptureStage::Post;
        }

        let writer = match format {
            CaptureFormat::Png => {
                let dir = dir.join(&name);
                std::fs::create_dir_all(&dir)
                    .with_context(|| format!("create png sequence dir {}", dir.display()))?;
                SequenceWriter::Png { dir }
            }
            CaptureFormat::Gif => {
                let path = dir.join(format!("{name}.gif"));
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)
                        .with_context(|| format!("create gif parent dir {}", parent.display()))?;
                }
                let delay_cs = fps_to_centiseconds(fps.unwrap_or(30));
                SequenceWriter::Gif {
                    encoder: None,
                    path,
                    delay_cs,
                    scale,
                }
            }
        };

        let fps_interval = fps.map(|f| Duration::from_secs_f64(1.0 / f.max(1) as f64));
        self.state = CaptureState::Sequence {
            stage,
            writer,
            fps_interval,
            last_capture_time: None,
            frame_count: 0,
        };
        Ok(format!(
            "capture: sequence started ({format:?}, {stage:?}, fps={})",
            fps.map(|f| f.to_string())
                .unwrap_or_else(|| "unlimited".into())
        ))
    }

    fn stop(&mut self, log: &mut Vec<String>) {
        let state = std::mem::replace(&mut self.state, CaptureState::Idle);
        match state {
            CaptureState::Sequence {
                writer,
                frame_count,
                ..
            } => {
                let path = match &writer {
                    SequenceWriter::Png { dir } => dir.clone(),
                    SequenceWriter::Gif { path, .. } => path.clone(),
                };
                // Dropping `writer` (and its encoder) flushes any pending GIF trailer.
                drop(writer);
                log.push(format!(
                    "capture: sequence stopped, {frame_count} frame(s) at {}",
                    path.display()
                ));
            }
            CaptureState::OneShot { .. } => {
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
            CaptureState::OneShot { path_pre, .. } => path_pre.is_some(),
            CaptureState::Sequence { stage, .. } => stage.wants_pre(),
        }
    }

    pub(crate) fn wants_post(&self) -> bool {
        match &self.state {
            CaptureState::Idle => false,
            CaptureState::OneShot { path_post, .. } => path_post.is_some(),
            CaptureState::Sequence { stage, .. } => stage.wants_post(),
        }
    }

    /// Should we capture this frame? FPS-gated for streaming sequences; always true for
    /// one-shots and unlimited PNG sequences.
    pub(crate) fn should_capture(&self, now: Instant) -> bool {
        match &self.state {
            CaptureState::Idle => false,
            CaptureState::OneShot { .. } => true,
            CaptureState::Sequence {
                fps_interval,
                last_capture_time,
                ..
            } => match (fps_interval, last_capture_time) {
                (None, _) => true,
                (Some(_), None) => true,
                (Some(interval), Some(last)) => now.duration_since(*last) >= *interval,
            },
        }
    }

    /// Hand one stage's pixels to the active writer. Logs and swallows encode errors so
    /// a transient failure doesn't tear down the render loop.
    pub(crate) fn consume_frame(
        &mut self,
        is_pre: bool,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
    ) -> Result<()> {
        match &mut self.state {
            CaptureState::Idle => Ok(()),
            CaptureState::OneShot {
                path_pre,
                path_post,
            } => {
                let path = if is_pre {
                    path_pre.take()
                } else {
                    path_post.take()
                };
                if let Some(path) = path {
                    write_png_bytes(&path, &rgba, width, height)?;
                    tracing::info!("capture: wrote {}", path.display());
                }
                Ok(())
            }
            CaptureState::Sequence {
                writer,
                frame_count,
                ..
            } => writer.write_frame(is_pre, *frame_count, &rgba, width, height),
        }
    }

    /// Mark the current frame as written. One-shot transitions to Idle when both
    /// requested stages have been consumed; sequence increments the frame counter and
    /// updates the FPS clock.
    pub(crate) fn advance_frame(&mut self, now: Instant) {
        match &mut self.state {
            CaptureState::OneShot {
                path_pre,
                path_post,
            } => {
                if path_pre.is_none() && path_post.is_none() {
                    self.state = CaptureState::Idle;
                }
            }
            CaptureState::Sequence {
                last_capture_time,
                frame_count,
                ..
            } => {
                *last_capture_time = Some(now);
                *frame_count = frame_count.saturating_add(1);
            }
            CaptureState::Idle => {}
        }
    }
}

impl SequenceWriter {
    fn write_frame(
        &mut self,
        is_pre: bool,
        frame_idx: u32,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Result<()> {
        match self {
            SequenceWriter::Png { dir } => {
                let label = if is_pre { "pre" } else { "post" };
                let path = dir.join(format!("{label}_{frame_idx:06}.png"));
                write_png_bytes(&path, rgba, width, height)?;
                Ok(())
            }
            SequenceWriter::Gif {
                encoder,
                path,
                delay_cs,
                scale,
            } => {
                let (out_w, out_h) = scaled_dims(width, height, *scale)?;
                let w_u16: u16 = out_w.try_into().context("gif width > 65535")?;
                let h_u16: u16 = out_h.try_into().context("gif height > 65535")?;

                // Lazily build the encoder against the first frame's output dimensions.
                // The LSD (Logical Screen Descriptor) is fixed for the whole stream, so
                // we can't accept a resize mid-capture; it's pinned to the first frame.
                let enc = match encoder {
                    Some(e) => e,
                    None => {
                        let file = File::create(&*path)
                            .with_context(|| format!("create gif output {}", path.display()))?;
                        let mut e = gif::Encoder::new(BufWriter::new(file), w_u16, h_u16, &[])
                            .context("init gif encoder")?;
                        e.set_repeat(gif::Repeat::Infinite).context("gif repeat")?;
                        *encoder = Some(e);
                        encoder.as_mut().unwrap()
                    }
                };

                // Downscale before NeuQuant if requested; smaller buffers also encode
                // faster per frame. Lanczos3 picks up edges and gradients better than
                // Triangle / Nearest, at moderate extra cost.
                let mut buf: Vec<u8> = if scale.is_some() {
                    let src: ::image::RgbaImage =
                        ::image::ImageBuffer::from_raw(width, height, rgba.to_vec())
                            .ok_or_else(|| anyhow!("RGBA size mismatch at {width}x{height}"))?;
                    let dst = ::image::imageops::resize(
                        &src,
                        out_w,
                        out_h,
                        ::image::imageops::FilterType::Lanczos3,
                    );
                    dst.into_raw()
                } else {
                    rgba.to_vec()
                };

                // `gif::Frame::from_rgba_speed` mutates the pixel buffer in place
                // (NeuQuant runs over it). Speed 10 is the gif crate's default: a
                // reasonable quality / latency balance. Lower (1-5) is better quality
                // but encode time can dominate; higher (20-30) is visibly worse on
                // gradients.
                let mut frame = gif::Frame::from_rgba_speed(w_u16, h_u16, &mut buf, 10);
                frame.delay = *delay_cs;
                enc.write_frame(&frame).context("gif encode")?;
                Ok(())
            }
        }
    }
}

/// Compute the output dimensions for a captured frame given an optional target width.
/// Aspect ratio preserved; height rounds to the nearest pixel and is clamped to >= 1
/// so degenerate 1xN scenes don't crash the encoder.
fn scaled_dims(width: u32, height: u32, scale: Option<u32>) -> Result<(u32, u32)> {
    let Some(target_w) = scale else {
        return Ok((width, height));
    };
    if target_w == 0 || width == 0 || height == 0 {
        return Err(anyhow!(
            "invalid scale: target_w={target_w}, src={width}x{height}"
        ));
    }
    let h = ((target_w as u64 * height as u64 + (width as u64) / 2) / width as u64) as u32;
    Ok((target_w, h.max(1)))
}

fn fps_to_centiseconds(fps: u16) -> u16 {
    // GIF delay is in centiseconds (1/100 s). Round to nearest, clamp to >= 1 so the
    // encoder doesn't reject a 0-delay frame.
    let cs = (100.0_f32 / fps.max(1) as f32).round() as u16;
    cs.max(1)
}

fn default_name() -> String {
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("capture_{unix}")
}

// ---------------------------------------------------------------------------
// GPU readback + PNG writer
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

    let slice = buffer.slice(..);
    slice.map_async(MapMode::Read, |_| {});
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

fn write_png_bytes(path: &Path, rgba: &[u8], width: u32, height: u32) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create capture dir {}", parent.display()))?;
    }
    let img: ::image::RgbaImage = ::image::ImageBuffer::from_raw(width, height, rgba.to_vec())
        .ok_or_else(|| anyhow!("RGBA buffer size doesn't match {width}x{height}"))?;
    img.save_with_format(path, ::image::ImageFormat::Png)
        .with_context(|| format!("write png {}", path.display()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Console command registration
// ---------------------------------------------------------------------------

/// Register `capture` console commands on the given console.
///
/// Commands registered:
/// - `capture png     [pre|post|both] [dir]`: one-shot PNG screenshot
/// - `capture frames  [pre|post|both] [dir] [fps=N]`: start PNG sequence
/// - `capture gif     [post|pre]      [dir] [fps=N] [scale=W]`: start GIF stream
/// - `capture toggle  [png|gif] [pre|post|both] [dir] [fps=N] [scale=W]`: toggle a sequence
/// - `capture stop`: stop sequence (or cancel a pending one-shot)
///
/// `fps=N` caps the capture rate (default 30 for GIF, unlimited for PNG sequences).
/// `scale=W` downscales each frame to width W in pixels before encoding (Lanczos3,
/// aspect preserved); GIF-only. Args are key-value, not in arg_choices, so tab
/// completion doesn't surface them (free-form values aren't enumerable).
pub fn register_commands<Ctx: 'static>(console: &mut Console<Ctx>) {
    console.register(
        cmd("capture", capture_help(), |args, _ctx: &mut Ctx, out| {
            run_capture(args, out)
        })
        // Positional arg-choice grammar drives tab-completion + ghost preview.
        // Subcommand at index 0, stage at index 1; output dir is intentionally
        // undeclared (no filesystem completion).
        .with_args(&[
            &["png", "frames", "gif", "toggle", "stop"],
            &["pre", "post", "both"],
        ]),
    );
}

fn capture_help() -> &'static str {
    "capture <png|frames|gif|toggle|stop> [pre|post|both] [dir] [fps=N] [scale=W]"
}

fn run_capture(args: &[&str], out: &mut ConsoleWriter) -> Result<()> {
    let Some((sub, rest)) = args.split_first() else {
        out.error(format!("usage: {}", capture_help()));
        return Ok(());
    };
    match *sub {
        "png" => {
            let p = parse_capture_args(rest);
            enqueue(CaptureRequest::OneShot {
                stage: p.stage,
                dir: p.dir,
                name: None,
            });
            out.line(format!("queued one-shot ({:?})", p.stage));
        }
        "frames" => {
            let p = parse_capture_args(rest);
            enqueue(CaptureRequest::StartSequence {
                format: CaptureFormat::Png,
                stage: p.stage,
                dir: p.dir,
                name: None,
                fps: p.fps,
                scale: None,
            });
            out.line(format!("started PNG sequence ({:?})", p.stage));
        }
        "gif" => {
            let p = parse_capture_args(rest);
            enqueue(CaptureRequest::StartSequence {
                format: CaptureFormat::Gif,
                stage: p.stage,
                dir: p.dir,
                name: None,
                fps: p.fps,
                scale: p.scale,
            });
            out.line(format!(
                "started GIF stream ({:?}, fps={}, scale={})",
                p.stage,
                p.fps.map_or("default".into(), |f| f.to_string()),
                p.scale.map_or("native".into(), |s| s.to_string()),
            ));
        }
        "stop" => {
            enqueue(CaptureRequest::Stop);
            out.line("stop queued");
        }
        "toggle" => {
            let (format, after_format) = parse_format(rest);
            let p = parse_capture_args(after_format);
            enqueue(CaptureRequest::Toggle {
                format,
                stage: p.stage,
                dir: p.dir,
                name: None,
                fps: p.fps,
                scale: p.scale,
            });
            out.line(format!("toggle queued ({format:?}, {:?})", p.stage));
        }
        other => {
            out.error(format!("unknown sub-command `{other}`. {}", capture_help()));
        }
    }
    Ok(())
}

struct ParsedCaptureArgs {
    stage: CaptureStage,
    dir: Option<PathBuf>,
    fps: Option<u16>,
    scale: Option<u32>,
}

impl Default for ParsedCaptureArgs {
    fn default() -> Self {
        Self {
            stage: CaptureStage::Post,
            dir: None,
            fps: None,
            scale: None,
        }
    }
}

/// Tokenise post-format positional args:
/// - `pre|post|both` => stage
/// - `fps=N`         => target frame rate
/// - `scale=N`       => output width in pixels (GIF only)
/// - anything else   => treat as output directory
fn parse_capture_args(args: &[&str]) -> ParsedCaptureArgs {
    let mut p = ParsedCaptureArgs::default();
    for arg in args {
        if let Some(v) = arg.strip_prefix("fps=") {
            if let Ok(n) = v.parse::<u16>() {
                p.fps = Some(n);
            }
        } else if let Some(v) = arg.strip_prefix("scale=") {
            if let Ok(n) = v.parse::<u32>() {
                p.scale = Some(n);
            }
        } else {
            match *arg {
                "pre" => p.stage = CaptureStage::Pre,
                "post" => p.stage = CaptureStage::Post,
                "both" => p.stage = CaptureStage::Both,
                other => p.dir = Some(PathBuf::from(other)),
            }
        }
    }
    p
}

fn parse_format<'a>(args: &'a [&'a str]) -> (CaptureFormat, &'a [&'a str]) {
    match args.split_first() {
        Some((&"png", rest)) | Some((&"frames", rest)) => (CaptureFormat::Png, rest),
        Some((&"gif", rest)) => (CaptureFormat::Gif, rest),
        // No format keyword recognised; treat the whole list as stage/dir args and use
        // the default streaming format (gif, the "share this clip" shape).
        _ => (CaptureFormat::Gif, args),
    }
}

/// Bind the default capture hotkeys on the given console:
/// - `F12`: `capture png post` (one-shot screenshot)
/// - `F9`:  `capture toggle gif post` (press to start a GIF, again to stop)
pub fn bind_default_hotkeys<Ctx: 'static>(console: &mut Console<Ctx>) {
    console.bind(rye_egui::egui::Key::F12, "capture png post");
    console.bind(rye_egui::egui::Key::F9, "capture toggle gif post");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_dims_preserves_aspect_ratio() {
        // 1920x1080 -> 720 width should give 405 height (16:9).
        assert_eq!(scaled_dims(1920, 1080, Some(720)).unwrap(), (720, 405));
        // Square downsample.
        assert_eq!(scaled_dims(1024, 1024, Some(512)).unwrap(), (512, 512));
        // Tall portrait.
        assert_eq!(scaled_dims(1080, 1920, Some(360)).unwrap(), (360, 640));
        // None -> identity.
        assert_eq!(scaled_dims(800, 600, None).unwrap(), (800, 600));
        // Height clamped to >= 1 for degenerate aspect ratios.
        assert_eq!(scaled_dims(10000, 1, Some(100)).unwrap(), (100, 1));
    }

    #[test]
    fn scaled_dims_rejects_zero_target() {
        assert!(scaled_dims(1920, 1080, Some(0)).is_err());
        assert!(scaled_dims(0, 1080, Some(720)).is_err());
        assert!(scaled_dims(1920, 0, Some(720)).is_err());
    }

    #[test]
    fn parse_capture_args_extracts_kv_pairs() {
        let p = parse_capture_args(&["pre", "./shots", "fps=24", "scale=480"]);
        assert_eq!(p.stage, CaptureStage::Pre);
        assert_eq!(p.dir.as_deref(), Some(std::path::Path::new("./shots")));
        assert_eq!(p.fps, Some(24));
        assert_eq!(p.scale, Some(480));
    }

    #[test]
    fn parse_capture_args_ignores_malformed_kv() {
        // `fps=` with non-numeric value silently drops; user gets default.
        let p = parse_capture_args(&["fps=abc", "scale=xyz"]);
        assert!(p.fps.is_none());
        assert!(p.scale.is_none());
    }
}
