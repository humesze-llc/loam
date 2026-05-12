//! Frame capture: PNG single-shot snapshots, PNG sequences, animated GIF streams,
//! and animated PNG (APNG) streams, with two taps (`pre`-egui = pure 3D scene,
//! `post`-egui = final composite as DWM receives it).
//!
//! ## Picking a format
//!
//! - **One-shot PNG** (`capture png`): lossless screenshot. Diagnostic / share /
//!   anything.
//! - **PNG sequence** (`capture frames`): one independent file per captured frame.
//!   Lossless, the diagnostic priority; also the master format if you intend to
//!   post-process with ffmpeg (palettegen GIF, libwebp WebP, libx264 MP4).
//! - **APNG** (`capture apng`): animated PNG, lossless, 24-bit color per frame,
//!   one file per recording. Renders inline on GitHub markdown. **Recommended for
//!   shareable clips of raymarched / continuous-tone content.** Larger files than
//!   GIF; memory cost during recording is roughly `frames * width * height * 4`
//!   bytes (frames are buffered until stop). Practical limit ~5 s at 1080p.
//! - **GIF** (`capture gif`): animated GIF, palette-quantized via NeuQuant, one
//!   file per recording. Convenient (Discord pastes the file inline) but has a
//!   **quality ceiling**: 256-color palette, per-frame palette regeneration
//!   produces visible flicker on raymarched / continuous-tone content (the
//!   palette wobbles between frames as NeuQuant picks slightly different 256-color
//!   subsets, and per-pixel indices oscillate at palette-region boundaries as
//!   rendering noise jitters). Acceptable for cartoon / UI / cell-shaded content;
//!   for raymarched content, prefer APNG or `capture frames` + ffmpeg. The
//!   `palette=global` flag mitigates but doesn't eliminate the flicker.
//!
//! ## How requests flow
//!
//! Console commands push [`CaptureRequest`]s onto a global queue via [`enqueue`]. The
//! runner drains the queue once per frame, mutates the internal `Capture` state
//! machine, and issues GPU copies at the two tap points in the render loop.
//!
//! Encoding splits by format:
//!
//! - **PNG (one-shot + sequence)**: synchronous on the main thread. PNG compression is
//!   fast enough that the per-frame cost is dominated by GPU readback. Diagnostic
//!   capture wants every frame written; no drops.
//! - **GIF stream**: encoded on a background worker thread. The main
//!   thread sends raw RGBA over a bounded channel; the worker runs NeuQuant + Lanczos
//!   resample + LZW + disk write. If the worker can't keep up, the main thread drops
//!   the frame rather than stalling the renderer; the dropped count is surfaced when
//!   the sequence stops.
//!
//! ## Pre-egui tap and MSAA
//!
//! With MSAA off, the 3D pass writes directly to the swapchain view, so the pre-egui
//! tap can copy it as-is. With MSAA on, the 3D content sits in the multisampled
//! attachment and is only resolved into the swapchain at the end of the egui pass; a
//! direct copy of multisamples isn't supported. The runner skips pre-egui captures
//! when MSAA is on (logs a warning); disable MSAA via
//! [`RunConfig::msaa_samples`](crate::RunConfig) for diagnostic capture sessions.
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
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
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
    /// stage only (pre OR post, not both). Default fps 30. Has a quality ceiling for
    /// continuous-tone content; see module docs.
    Gif,
    /// Single animated PNG (APNG): lossless 24-bit-per-frame, infinite loop. Single
    /// stage only. Default fps 30. Recommended for shareable clips of raymarched
    /// content. Worker buffers all frames in memory until stop, so practical
    /// recording length is bounded by available RAM (~5 s at 1080p).
    Apng,
}

/// How GIF frames are palette-quantized.
///
/// NeuQuant (Anthony Dekker, 1994) is the quantization algorithm: a self-organizing
/// map that picks 256 representative colors (palette "neurons") from the input
/// pixels. The mode controls *what* it trains on.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum PaletteMode {
    /// Per-frame NeuQuant. Each frame gets its own optimal 256-color palette.
    /// Fast and high quality per-frame, but consecutive frames pick slightly
    /// different palettes so smooth gradients shimmer (the classic GIF flicker).
    #[default]
    Local,
    /// One global NeuQuant palette trained on a warmup window of the first
    /// ~`GIF_WARMUP_FRAMES` captures, then reused for every frame via `index_of`.
    /// Eliminates the per-frame palette wobble at the cost of slightly worse
    /// fidelity (the palette has to cover all colors in the recording, not just
    /// the current frame). The warmup buffer also avoids training on transient
    /// overlays (e.g., the console showing when you typed the start command).
    Global,
}

impl CaptureFormat {
    fn default_fps(self) -> Option<u16> {
        match self {
            CaptureFormat::Png => None, // unlimited; every frame
            CaptureFormat::Gif | CaptureFormat::Apng => Some(30),
        }
    }

    fn supports_both_stages(self) -> bool {
        // Only PNG sequences write two parallel files; the single-file animated
        // formats can only carry one stream.
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
        /// for GIF and APNG the runner falls back to a 30 fps default when `None`.
        fps: Option<u16>,
        /// Output width in pixels for downscaled streams. Height computed to preserve
        /// aspect ratio. `None` captures at native swapchain resolution. GIF-only;
        /// PNG sequences ignore this for diagnostic fidelity.
        scale: Option<u32>,
        /// GIF palette strategy. `Local` (default) does per-frame NeuQuant; `Global`
        /// trains one NeuQuant during a warmup buffer and reuses it. Ignored for
        /// PNG sequences.
        palette: PaletteMode,
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
        palette: PaletteMode,
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
// Status broadcast
// ---------------------------------------------------------------------------

/// The runner publishes the current [`Capture::status`] here once per frame so UI code
/// (the panel, optional console title text, etc.) can read it without owning a
/// reference to the runner.
static STATUS: Mutex<Option<String>> = Mutex::new(None);

/// Compact one-line status string set by the runner each frame. `None` when idle.
/// Currently surfaced in the window title; the panel UI reads it via this function.
pub fn current_status() -> Option<String> {
    STATUS.lock().ok().and_then(|g| g.clone())
}

pub(crate) fn publish_status(status: Option<String>) {
    if let Ok(mut g) = STATUS.lock() {
        *g = status;
    }
}

/// Console-driven toggle for the capture panel. The `capture panel` subcommand flips
/// this; [`CapturePanel::show`] mirrors it so the panel opens/closes without per-demo
/// plumbing.
static PANEL_OPEN: AtomicBool = AtomicBool::new(false);

fn toggle_panel_global() -> bool {
    let now_open = !PANEL_OPEN.load(Ordering::Relaxed);
    PANEL_OPEN.store(now_open, Ordering::Relaxed);
    now_open
}

// ---------------------------------------------------------------------------
// Capture state machine
// ---------------------------------------------------------------------------

/// Runner-owned state machine. Drives the per-frame copy + write.
pub(crate) struct Capture {
    default_dir: PathBuf,
    state: CaptureState,
    /// Detached GIF-encoder threads that are still flushing buffered frames after
    /// the user `stop`ed. Joined at runner shutdown so trailers finish even when
    /// the encode outlives the recording session. Finished handles are reaped
    /// opportunistically on each new stop to keep the vec from growing unbounded.
    pending: Vec<JoinHandle<()>>,
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
    /// PNG sequence: one independent file per frame, stage-labelled. Encoded on the
    /// main thread; PNG compression is fast enough that the per-frame cost is dominated
    /// by GPU readback, not encoding.
    Png { dir: PathBuf },
    /// Animated GIF: encoded on a background worker thread to keep NeuQuant +
    /// Lanczos resample + LZW off the render loop. Frames cross a bounded channel;
    /// when the worker can't keep up, the main thread drops the incoming frame rather
    /// than stalling the renderer. The user sees the dropped count on stop.
    Gif {
        worker: GifWorker,
        path: PathBuf,
        /// Delay for the first frame, in centiseconds. Subsequent frames use
        /// wall-clock-derived delays (see `GifFrame::captured_at`).
        default_delay_cs: u16,
        /// Optional output width in pixels. Lanczos3 resample on the worker thread
        /// before NeuQuant; aspect ratio preserved.
        scale: Option<u32>,
        /// Per-frame vs shared-palette quantization.
        palette_mode: PaletteMode,
        /// Some during the warmup buffer for `Global` mode; `None` for `Local`
        /// mode and after the warmup buffer has been drained.
        warming: Option<WarmingState>,
        /// Trained global palette (Some after warmup completes in `Global` mode).
        /// Cloned into every emitted GifFrame so the worker uses the same palette
        /// for indexing each frame.
        global_palette: Option<Arc<color_quant::NeuQuant>>,
    },
    /// Animated PNG: lossless 24-bit-per-frame, infinite loop. The worker buffers
    /// every captured frame in memory until stop, then writes the assembled APNG
    /// in one pass (APNG's `acTL` chunk needs the frame count up front). Memory
    /// cost = `frames * width * height * 4` bytes; cap recordings to a few seconds
    /// at 1080p to avoid pressure.
    Apng {
        worker: ApngWorker,
        path: PathBuf,
        /// Optional output width in pixels. Lanczos3 resample on the worker thread;
        /// aspect ratio preserved.
        scale: Option<u32>,
    },
}

/// Warmup buffer for global-palette mode. Caches the first
/// [`GIF_WARMUP_FRAMES`] captures so the palette is trained on representative
/// content rather than whatever happens to be on screen at the moment recording
/// starts (which often includes the console overlay used to start it).
struct WarmingState {
    buffer: Vec<WarmupFrame>,
    target_frames: u32,
}

struct WarmupFrame {
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    captured_at: Instant,
}

/// Number of captures buffered before training the global palette. At ~30 fps
/// capture this is ~1 second of warmup, enough for the console to close and the
/// scene to stabilise. Memory: `frames × width × height × 4` bytes; at 800x600
/// this is ~57 MB during warmup, released after training.
const GIF_WARMUP_FRAMES: u32 = 30;

/// Owns the GIF encoder thread and the SPSC channel feeding it. The main thread calls
/// [`GifWorker::try_send`] with each captured frame; the worker drains the channel,
/// runs NeuQuant + Lanczos resample + LZW, and writes to disk. Dropping the worker
/// closes the channel and joins the thread, guaranteeing the GIF trailer is flushed.
pub(crate) struct GifWorker {
    tx: Option<SyncSender<GifFrame>>,
    handle: Option<JoinHandle<()>>,
    dropped: Arc<AtomicU32>,
}

/// One frame's worth of work for the GIF encoder thread.
struct GifFrame {
    rgba: Vec<u8>,
    src_width: u32,
    src_height: u32,
    /// Wall-clock time when this frame was captured. The worker computes each
    /// frame's GIF delay as `captured_at - last_encoded_captured_at`, so when
    /// frames are dropped under backpressure the next encoded frame gets a
    /// proportionally longer delay and total playback duration matches recording
    /// duration. Constant-delay encoding (the previous behavior) caused dropped
    /// frames to compress wall time, producing visibly fast playback.
    captured_at: Instant,
    /// Fallback delay for the very first frame (no previous timestamp to diff
    /// against). Derived from the user's target fps at sequence start.
    default_delay_cs: u16,
    scale: Option<u32>,
    /// When `Some`, the worker indexes pixels against this shared NeuQuant and
    /// writes each frame with `palette: None` so they all reference the global
    /// color table opened on the first frame. When `None`, the worker runs
    /// per-frame NeuQuant via `Frame::from_rgba_speed` (local palette per frame).
    global_palette: Option<Arc<color_quant::NeuQuant>>,
}

/// Channel capacity. Sized at ~8 frames so a small encode-time spike doesn't
/// immediately drop frames, but the worker can't drift more than ~270 ms behind the
/// renderer at 30 fps. Higher capacities trade smoothness for latency.
const GIF_CHANNEL_CAPACITY: usize = 8;

impl GifWorker {
    fn spawn(path: PathBuf) -> Self {
        let (tx, rx) = sync_channel::<GifFrame>(GIF_CHANNEL_CAPACITY);
        let dropped = Arc::new(AtomicU32::new(0));
        let handle = thread::Builder::new()
            .name("rye-app::gif-encoder".into())
            .spawn(move || gif_encoder_loop(path, rx))
            .expect("spawn gif encoder thread");
        Self {
            tx: Some(tx),
            handle: Some(handle),
            dropped,
        }
    }

    /// Close the input channel and hand back the worker's `JoinHandle` without
    /// waiting. The worker keeps draining the buffered frames in the background and
    /// exits naturally when the channel is empty; the caller parks the handle so
    /// it can be joined at app shutdown (the GIF trailer is flushed when the worker
    /// thread's encoder drops).
    fn detach(mut self) -> JoinHandle<()> {
        self.tx.take();
        self.handle.take().expect("worker handle present")
    }

    /// Non-blocking enqueue. If the channel is full, increments the dropped counter
    /// and returns; the renderer never stalls on a slow encoder.
    fn try_send(&self, frame: GifFrame) {
        let Some(tx) = self.tx.as_ref() else { return };
        match tx.try_send(frame) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                let count = self.dropped.fetch_add(1, Ordering::Relaxed) + 1;
                if count.is_power_of_two() {
                    tracing::warn!(
                        "capture: GIF encoder queue full; dropped {count} frame(s) so far"
                    );
                }
            }
            Err(TrySendError::Disconnected(_)) => {
                tracing::error!("capture: GIF encoder thread exited unexpectedly");
            }
        }
    }

    fn dropped(&self) -> u32 {
        self.dropped.load(Ordering::Relaxed)
    }
}

impl Drop for GifWorker {
    fn drop(&mut self) {
        // Fallback path for the case where the worker was never detached (e.g., a
        // panic unwinds through the Capture). Closing the channel and joining keeps
        // the GIF trailer correct. The `Capture::stop` path always calls `detach`
        // before this, so the normal flow leaves both `tx` and `handle` already
        // `None`-d here and this is a no-op.
        self.tx.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn gif_encoder_loop(path: PathBuf, rx: Receiver<GifFrame>) {
    let mut encoder: Option<gif::Encoder<BufWriter<File>>> = None;
    // Wall-clock timestamp of the previous successfully-encoded frame. Each new
    // frame's GIF delay is computed as the elapsed centiseconds between this and
    // the current frame's `captured_at`. First frame falls back to the configured
    // default delay (derived from the user's target fps).
    let mut last_captured_at: Option<Instant> = None;
    for frame in rx {
        if let Err(e) = encode_one_frame(&path, &mut encoder, &mut last_captured_at, frame) {
            tracing::error!("capture: gif encode error: {e:#}");
            return;
        }
    }
    // Encoder drops here, flushing the GIF trailer; log so the user knows the
    // background encode completed.
    drop(encoder);
    tracing::info!("capture: gif file finalised at {}", path.display());
}

/// Per-frame palette encode. NeuQuant runs over each frame independently, picks
/// the best 256-color palette for that frame, and writes a local-palette frame.
/// This produces visible flicker on raymarched content because consecutive frames
/// pick slightly different palettes; a global-palette attempt (built once from the
/// first frame's pixels and reused via `index_of` for all subsequent frames)
/// rendered colors incorrectly in practice and was reverted on 2026-05-13. See
/// issue tracker (GIF flicker is a known limitation; use PNG sequence + ffmpeg
/// for sharable high-quality clips). The diagnostic case still uses PNG.
fn encode_one_frame(
    path: &Path,
    encoder: &mut Option<gif::Encoder<BufWriter<File>>>,
    last_captured_at: &mut Option<Instant>,
    frame: GifFrame,
) -> Result<()> {
    let (out_w, out_h) = scaled_dims(frame.src_width, frame.src_height, frame.scale)?;
    let w_u16: u16 = out_w.try_into().context("gif width > 65535")?;
    let h_u16: u16 = out_h.try_into().context("gif height > 65535")?;

    // Open the encoder on first frame. In global-palette mode the LSD gets the
    // shared palette so subsequent frames can write `palette: None`; in local mode
    // we pass an empty global palette and every frame writes its own local table.
    let enc = match encoder {
        Some(e) => e,
        None => {
            let file = File::create(path)
                .with_context(|| format!("create gif output {}", path.display()))?;
            let global_palette_bytes: Vec<u8> = frame
                .global_palette
                .as_ref()
                .map(|nq| nq.color_map_rgb())
                .unwrap_or_default();
            let mut e =
                gif::Encoder::new(BufWriter::new(file), w_u16, h_u16, &global_palette_bytes)
                    .context("init gif encoder")?;
            e.set_repeat(gif::Repeat::Infinite).context("gif repeat")?;
            tracing::info!(
                "capture: gif encoder opened {}x{} target {}cs/frame ({} fps); \
                 palette={}; actual delays computed from wall-clock per-frame",
                w_u16,
                h_u16,
                frame.default_delay_cs,
                if frame.default_delay_cs == 0 {
                    0
                } else {
                    100 / frame.default_delay_cs as u32
                },
                if frame.global_palette.is_some() {
                    "global (shared NeuQuant)"
                } else {
                    "per-frame"
                }
            );
            *encoder = Some(e);
            encoder.as_mut().unwrap()
        }
    };

    // Per-frame delay = elapsed wall-clock time since the previous successfully
    // encoded frame. When backpressure drops frames, the next surviving frame
    // gets a longer delay so total playback duration matches recording duration.
    // First frame has no previous timestamp; use the configured target delay.
    let delay_cs = match *last_captured_at {
        None => frame.default_delay_cs,
        Some(prev) => {
            let ms = frame.captured_at.duration_since(prev).as_millis() as u64;
            // Round to nearest centisecond, clamp to >= 1 (gif minimum) and <= u16::MAX.
            let cs = (ms + 5) / 10;
            cs.clamp(1, u16::MAX as u64) as u16
        }
    };
    *last_captured_at = Some(frame.captured_at);

    // Lanczos3 downscale (when requested) before quantization. Per-frame NeuQuant
    // then runs on the resampled pixels.
    let mut buf: Vec<u8> = if frame.scale.is_some() {
        let src: ::image::RgbaImage =
            ::image::ImageBuffer::from_raw(frame.src_width, frame.src_height, frame.rgba)
                .ok_or_else(|| {
                    anyhow!(
                        "RGBA size mismatch at {}x{}",
                        frame.src_width,
                        frame.src_height
                    )
                })?;
        let dst =
            ::image::imageops::resize(&src, out_w, out_h, ::image::imageops::FilterType::Lanczos3);
        dst.into_raw()
    } else {
        frame.rgba
    };

    // Build the frame. Global mode indexes against the shared NeuQuant and emits
    // `palette: None` to use the global color table; local mode runs `from_rgba_speed`
    // which trains a NeuQuant per frame and attaches it as a local table.
    let mut gif_frame = if let Some(nq) = &frame.global_palette {
        // Alpha normalization for index_of consistency with how the palette was
        // trained (`train_global_palette` normalizes too).
        for px in buf.chunks_exact_mut(4) {
            if px[3] != 0 {
                px[3] = 0xFF;
            }
        }
        let mut indices = Vec::with_capacity((out_w as usize) * (out_h as usize));
        for px in buf.chunks_exact(4) {
            indices.push(nq.index_of(px) as u8);
        }
        gif::Frame {
            width: w_u16,
            height: h_u16,
            buffer: std::borrow::Cow::Owned(indices),
            // `palette: None` -> use the global color table the encoder was opened with.
            palette: None,
            ..gif::Frame::default()
        }
    } else {
        // Per-frame NeuQuant. Speed 10 is the gif crate's default and matches the
        // rate the encoder thread can actually sustain (~30 fps at 800x600);
        // samplefac=1 trades palette stability for sluggish playback because the
        // backpressure drops dominate.
        gif::Frame::from_rgba_speed(w_u16, h_u16, &mut buf, 10)
    };
    gif_frame.delay = delay_cs;
    // `DisposalMethod::Any` (the gif crate default) leaves disposal unspecified so
    // decoders pick whatever they normally do for full-frame opaque content (~Keep).
    // An earlier attempt to force `Background` caused a perceptible inter-frame
    // flash on full-frame content, which read as worse flicker.
    gif_frame.dispose = gif::DisposalMethod::Any;
    enc.write_frame(&gif_frame).context("gif encode")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// APNG worker
// ---------------------------------------------------------------------------

/// Owns the APNG encoder thread. Unlike the GIF worker, APNG can't stream a
/// frame at a time because the `acTL` chunk (which carries the total frame
/// count) must be written before the first frame data. So the worker buffers
/// every received frame in memory; on stop, the channel closes and the loop
/// assembles + writes the full APNG file.
pub(crate) struct ApngWorker {
    tx: Option<SyncSender<ApngFrame>>,
    handle: Option<JoinHandle<()>>,
    /// Updated by the worker as it receives frames. Surfaced to the user as
    /// part of the recording status so they can see the buffer growing.
    frame_count: Arc<AtomicU32>,
}

struct ApngFrame {
    rgba: Vec<u8>,
    src_width: u32,
    src_height: u32,
    captured_at: Instant,
    scale: Option<u32>,
}

/// Capacity of the main-thread -> worker channel. The worker accumulates
/// without bound once it pulls a frame off the channel, so this is just the
/// in-flight buffer; the actual memory ceiling is the worker's internal Vec.
const APNG_CHANNEL_CAPACITY: usize = 16;

impl ApngWorker {
    fn spawn(path: PathBuf) -> Self {
        let (tx, rx) = sync_channel::<ApngFrame>(APNG_CHANNEL_CAPACITY);
        let frame_count = Arc::new(AtomicU32::new(0));
        let frame_count_for_worker = frame_count.clone();
        let handle = thread::Builder::new()
            .name("rye-app::apng-encoder".into())
            .spawn(move || apng_encoder_loop(path, rx, frame_count_for_worker))
            .expect("spawn apng encoder thread");
        Self {
            tx: Some(tx),
            handle: Some(handle),
            frame_count,
        }
    }

    /// Non-blocking enqueue. If the channel is full the frame is dropped (the
    /// worker is presumably moving them into its internal buffer just fine; a
    /// full channel means the worker is unusually slow).
    fn try_send(&self, frame: ApngFrame) {
        let Some(tx) = self.tx.as_ref() else { return };
        if let Err(TrySendError::Disconnected(_)) = tx.try_send(frame) {
            tracing::error!("capture: apng encoder thread exited unexpectedly");
        }
    }

    fn frame_count(&self) -> u32 {
        self.frame_count.load(Ordering::Relaxed)
    }

    /// Close the input channel and hand back the worker's `JoinHandle`. The
    /// worker drains the channel into its in-memory buffer, then assembles
    /// and writes the APNG when the channel closes. Mirrors `GifWorker::detach`.
    fn detach(mut self) -> JoinHandle<()> {
        self.tx.take();
        self.handle.take().expect("worker handle present")
    }
}

impl Drop for ApngWorker {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn apng_encoder_loop(path: PathBuf, rx: Receiver<ApngFrame>, frame_count: Arc<AtomicU32>) {
    let mut frames: Vec<ApngFrame> = Vec::new();
    for frame in rx {
        frames.push(frame);
        frame_count.store(frames.len() as u32, Ordering::Relaxed);
    }
    if frames.is_empty() {
        tracing::info!("capture: apng stopped before any frames captured; no file written");
        return;
    }
    if let Err(e) = write_apng(&path, frames) {
        tracing::error!("capture: apng write failed: {e:#}");
        return;
    }
    tracing::info!("capture: apng file finalised at {}", path.display());
}

fn write_apng(path: &Path, frames: Vec<ApngFrame>) -> Result<()> {
    // First frame's dimensions become the APNG's fixed canvas size.
    let (out_w, out_h) = scaled_dims(frames[0].src_width, frames[0].src_height, frames[0].scale)?;
    let file =
        File::create(path).with_context(|| format!("create apng output {}", path.display()))?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), out_w, out_h);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    // `set_animated(num_frames, num_plays)`; num_plays=0 loops infinitely.
    encoder
        .set_animated(frames.len() as u32, 0)
        .context("apng set_animated")?;
    let mut writer = encoder.write_header().context("apng write_header")?;

    let mut last_captured_at: Option<Instant> = None;
    for frame in frames {
        let (fw, fh) = scaled_dims(frame.src_width, frame.src_height, frame.scale)?;
        if fw != out_w || fh != out_h {
            return Err(anyhow!(
                "apng frame dims {fw}x{fh} != first frame {out_w}x{out_h}"
            ));
        }
        let rgba = if frame.scale.is_some() {
            let src: ::image::RgbaImage =
                ::image::ImageBuffer::from_raw(frame.src_width, frame.src_height, frame.rgba)
                    .ok_or_else(|| {
                        anyhow!(
                            "RGBA size mismatch at {}x{}",
                            frame.src_width,
                            frame.src_height
                        )
                    })?;
            ::image::imageops::resize(&src, out_w, out_h, ::image::imageops::FilterType::Lanczos3)
                .into_raw()
        } else {
            frame.rgba
        };

        // Per-frame delay: wall-clock elapsed since previous frame, as `ms / 1000`.
        // Clamped to >= 1 ms so the encoder doesn't reject zero. First frame uses
        // a default of 33 ms (~30 fps) since no previous timestamp exists.
        let delay_ms = match last_captured_at {
            None => 33u16,
            Some(prev) => {
                let ms = frame.captured_at.duration_since(prev).as_millis();
                ms.min(u16::MAX as u128).max(1) as u16
            }
        };
        last_captured_at = Some(frame.captured_at);

        writer
            .set_frame_delay(delay_ms, 1000)
            .context("apng set_frame_delay")?;
        writer
            .write_image_data(&rgba)
            .context("apng write_image_data")?;
    }
    writer.finish().context("apng finish")?;
    Ok(())
}

impl Capture {
    pub(crate) fn new() -> Self {
        Self {
            default_dir: PathBuf::from("captures"),
            state: CaptureState::Idle,
            pending: Vec::new(),
        }
    }

    /// Reap any background GIF workers that have already finished, so the pool
    /// doesn't grow unbounded across many stop/start cycles.
    fn reap_finished(&mut self) {
        let mut still_running = Vec::with_capacity(self.pending.len());
        for h in self.pending.drain(..) {
            if h.is_finished() {
                let _ = h.join();
            } else {
                still_running.push(h);
            }
        }
        self.pending = still_running;
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
                    palette,
                } => match self.start_sequence(format, stage, dir, name, fps, scale, palette) {
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
                    palette,
                } => {
                    if matches!(self.state, CaptureState::Sequence { .. }) {
                        self.stop(&mut log);
                    } else {
                        match self.start_sequence(format, stage, dir, name, fps, scale, palette) {
                            Ok(msg) => log.push(msg),
                            Err(e) => log.push(format!("capture: failed to start sequence: {e:#}")),
                        }
                    }
                }
            }
        }
        log
    }

    // Each arg corresponds to a `CaptureRequest::StartSequence` field; bundling them
    // into a struct would just rename the same set of values without buying clarity.
    #[allow(clippy::too_many_arguments)]
    fn start_sequence(
        &mut self,
        format: CaptureFormat,
        mut stage: CaptureStage,
        dir: Option<PathBuf>,
        name: Option<String>,
        fps: Option<u16>,
        scale: Option<u32>,
        palette: PaletteMode,
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
                let default_delay_cs = fps_to_centiseconds(fps.unwrap_or(30));
                let worker = GifWorker::spawn(path.clone());
                let warming = match palette {
                    PaletteMode::Local => None,
                    PaletteMode::Global => Some(WarmingState {
                        buffer: Vec::with_capacity(GIF_WARMUP_FRAMES as usize),
                        target_frames: GIF_WARMUP_FRAMES,
                    }),
                };
                SequenceWriter::Gif {
                    worker,
                    path,
                    default_delay_cs,
                    scale,
                    palette_mode: palette,
                    warming,
                    global_palette: None,
                }
            }
            CaptureFormat::Apng => {
                let path = dir.join(format!("{name}.apng"));
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)
                        .with_context(|| format!("create apng parent dir {}", parent.display()))?;
                }
                let worker = ApngWorker::spawn(path.clone());
                SequenceWriter::Apng {
                    worker,
                    path,
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
            "capture: sequence started ({format:?}, {stage:?}, fps={}, palette={palette:?})",
            fps.map(|f| f.to_string())
                .unwrap_or_else(|| "unlimited".into())
        ))
    }

    fn stop(&mut self, log: &mut Vec<String>) {
        self.reap_finished();
        let state = std::mem::replace(&mut self.state, CaptureState::Idle);
        match state {
            CaptureState::Sequence {
                writer,
                frame_count,
                ..
            } => match writer {
                SequenceWriter::Png { dir } => {
                    log.push(format!(
                        "capture: PNG sequence stopped, {frame_count} frame(s) at {}",
                        dir.display()
                    ));
                }
                SequenceWriter::Gif {
                    worker,
                    path,
                    default_delay_cs,
                    scale,
                    warming,
                    ..
                } => {
                    // If we stopped during warmup, train a palette on whatever
                    // captures we have and flush them to the worker so the user
                    // still gets a (short) GIF instead of an empty file.
                    if let Some(mut w) = warming {
                        if !w.buffer.is_empty() {
                            tracing::info!(
                                "capture: stopped during warmup ({} frame(s)); \
                                 training palette on partial buffer",
                                w.buffer.len()
                            );
                            let nq = Arc::new(train_global_palette(&w.buffer));
                            for f in w.buffer.drain(..) {
                                worker.try_send(GifFrame {
                                    rgba: f.rgba,
                                    src_width: f.width,
                                    src_height: f.height,
                                    captured_at: f.captured_at,
                                    default_delay_cs,
                                    scale,
                                    global_palette: Some(nq.clone()),
                                });
                            }
                        }
                    }
                    let dropped = worker.dropped();
                    // Detach: close the input channel and park the worker's handle in
                    // the pending pool. The worker keeps encoding its buffered frames
                    // in the background while the main thread returns immediately.
                    let handle = worker.detach();
                    self.pending.push(handle);
                    let drop_note = if dropped > 0 {
                        format!(", {dropped} dropped under backpressure")
                    } else {
                        String::new()
                    };
                    log.push(format!(
                        "capture: GIF stream stopped, {frame_count} frame(s){drop_note} \
                         encoding in background -> {}",
                        path.display()
                    ));
                }
                SequenceWriter::Apng { worker, path, .. } => {
                    let buffered = worker.frame_count();
                    let handle = worker.detach();
                    self.pending.push(handle);
                    log.push(format!(
                        "capture: APNG stream stopped, {buffered} frame(s) buffered; \
                         assembling and writing in background -> {}",
                        path.display()
                    ));
                }
            },
            CaptureState::OneShot { .. } => {
                log.push("capture: pending one-shot cancelled".into());
            }
            CaptureState::Idle => {
                log.push("capture: stop with no active session (no-op)".into());
            }
        }
    }

    /// Compact status string for display in the window title or a UI widget. `None`
    /// when idle; otherwise a terse line like `REC 42` or `REC 42 (3 dropped)`.
    pub(crate) fn status(&self) -> Option<String> {
        match &self.state {
            CaptureState::Idle => None,
            CaptureState::OneShot { .. } => Some("snap".into()),
            CaptureState::Sequence {
                writer,
                frame_count,
                ..
            } => {
                // During GIF global-palette warmup, surface the progress so the
                // user knows recording hasn't started writing yet.
                if let SequenceWriter::Gif {
                    warming: Some(w), ..
                } = writer
                {
                    return Some(format!("WARMING {}/{}", w.buffer.len(), w.target_frames));
                }
                let dropped = match writer {
                    SequenceWriter::Png { .. } => 0,
                    SequenceWriter::Gif { worker, .. } => worker.dropped(),
                    SequenceWriter::Apng { .. } => 0,
                };
                if dropped > 0 {
                    Some(format!("REC {frame_count} ({dropped} dropped)"))
                } else {
                    Some(format!("REC {frame_count}"))
                }
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
    /// a transient failure doesn't tear down the render loop. `captured_at` is the
    /// wall-clock instant the frame was sampled; the GIF worker uses it to compute
    /// per-frame delays so playback duration matches recording duration even when
    /// frames are dropped under backpressure.
    pub(crate) fn consume_frame(
        &mut self,
        is_pre: bool,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
        captured_at: Instant,
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
            } => writer.write_frame(is_pre, *frame_count, &rgba, width, height, captured_at),
        }
    }

    /// Block until all detached GIF workers finish flushing. Called from `Capture`'s
    /// Drop impl so any background encodes that outlived the recording session still
    /// produce a complete file when the runner exits.
    fn join_pending(&mut self) {
        for h in self.pending.drain(..) {
            let _ = h.join();
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

impl Drop for Capture {
    fn drop(&mut self) {
        // Wait on any still-encoding background GIF workers so their files have
        // valid trailers before the process exits. Worst case at app close: a brief
        // pause while the last few buffered frames finish encoding.
        self.join_pending();
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
        captured_at: Instant,
    ) -> Result<()> {
        match self {
            SequenceWriter::Png { dir } => {
                let label = if is_pre { "pre" } else { "post" };
                let path = dir.join(format!("{label}_{frame_idx:06}.png"));
                write_png_bytes(&path, rgba, width, height)?;
                Ok(())
            }
            SequenceWriter::Gif {
                worker,
                default_delay_cs,
                scale,
                palette_mode,
                warming,
                global_palette,
                ..
            } => {
                match *palette_mode {
                    // Local-palette mode: pass RGBA straight to the worker, which
                    // runs per-frame NeuQuant via `Frame::from_rgba_speed`.
                    PaletteMode::Local => {
                        worker.try_send(GifFrame {
                            rgba: rgba.to_vec(),
                            src_width: width,
                            src_height: height,
                            captured_at,
                            default_delay_cs: *default_delay_cs,
                            scale: *scale,
                            global_palette: None,
                        });
                    }
                    PaletteMode::Global => {
                        if let Some(w) = warming.as_mut() {
                            // Warming: buffer the frame.
                            w.buffer.push(WarmupFrame {
                                rgba: rgba.to_vec(),
                                width,
                                height,
                                captured_at,
                            });
                            if w.buffer.len() as u32 >= w.target_frames {
                                // Train the global palette on the concatenated
                                // buffer and start emitting. One-time main-thread
                                // pause (~50-100 ms at 800x600).
                                let nq = Arc::new(train_global_palette(&w.buffer));
                                tracing::info!(
                                    "capture: gif global palette trained from {} frames",
                                    w.buffer.len()
                                );
                                // Drain the buffer through the worker first so the
                                // first ~1 s of recording isn't lost.
                                for f in w.buffer.drain(..) {
                                    worker.try_send(GifFrame {
                                        rgba: f.rgba,
                                        src_width: f.width,
                                        src_height: f.height,
                                        captured_at: f.captured_at,
                                        default_delay_cs: *default_delay_cs,
                                        scale: *scale,
                                        global_palette: Some(nq.clone()),
                                    });
                                }
                                *global_palette = Some(nq);
                                *warming = None;
                            }
                        } else if let Some(nq) = global_palette.as_ref() {
                            // Post-warmup: send with the shared palette.
                            worker.try_send(GifFrame {
                                rgba: rgba.to_vec(),
                                src_width: width,
                                src_height: height,
                                captured_at,
                                default_delay_cs: *default_delay_cs,
                                scale: *scale,
                                global_palette: Some(nq.clone()),
                            });
                        } else {
                            tracing::error!(
                                "capture: gif global mode lost palette state (frame dropped)"
                            );
                        }
                    }
                }
                Ok(())
            }
            SequenceWriter::Apng { worker, scale, .. } => {
                // APNG worker buffers in memory; just hand the raw RGBA over.
                worker.try_send(ApngFrame {
                    rgba: rgba.to_vec(),
                    src_width: width,
                    src_height: height,
                    captured_at,
                    scale: *scale,
                });
                Ok(())
            }
        }
    }
}

/// Train a NeuQuant on a sparse sample of pixels drawn from every warmup frame, so
/// the palette captures color variation across the recording rather than just one
/// instant. Sparse sampling (every Nth pixel) keeps the training buffer bounded:
/// at 800x600 with 30 warmup frames and stride 16, we feed NeuQuant ~110 K
/// samples, which is plenty for a 256-color palette.
fn train_global_palette(buffer: &[WarmupFrame]) -> color_quant::NeuQuant {
    const STRIDE: usize = 16;
    let mut samples: Vec<u8> = Vec::new();
    for frame in buffer {
        for px in frame.rgba.chunks_exact(4).step_by(STRIDE) {
            // Match `Frame::from_rgba_speed`: normalize alpha for opaque pixels so
            // NeuQuant's 4D distance metric isn't biased by varying alpha.
            samples.push(px[0]);
            samples.push(px[1]);
            samples.push(px[2]);
            samples.push(if px[3] != 0 { 0xFF } else { 0 });
        }
    }
    color_quant::NeuQuant::new(10, 256, &samples)
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
/// RGBA8 byte buffer; the caller writes it out as PNG. The readback is sync
/// (poll-wait on the buffer map), so a capture frame is allowed to stutter.
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
        // All kv keys appear at the arg level (`fps=`, `palette=`, `scale=`); the
        // *values* for `palette=` are declared separately via
        // `with_value_choices` so the console can do two-step completion: first
        // Tab lands on `palette=`, then ghost/Tab cycles `global`/`local`.
        // `fps=` and `scale=` have no value choices (free-form numeric input)
        // so completion stops at the bare key.
        .with_args(&[
            &["png", "frames", "gif", "apng", "toggle", "stop", "panel"],
            &["both", "fps=", "palette=", "post", "pre", "scale="],
            &["fps=", "palette=", "scale="],
            &["fps=", "palette=", "scale="],
        ])
        .with_value_choices("palette", &["local", "global"]),
    );
}

fn capture_help() -> &'static str {
    "capture <png|frames|gif|apng|toggle|stop|panel> [pre|post|both] [dir] [fps=N] \
     [scale=W] [palette=local|global]"
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
                palette: PaletteMode::default(),
            });
            out.line(format!("started PNG sequence ({:?})", p.stage));
        }
        "gif" => {
            let p = parse_capture_args(rest);
            // Always surface the GIF quality caveat. Raymarched continuous-tone
            // content fights the 256-color palette and produces visible flicker.
            out.error(
                "GIF: per-frame NeuQuant flickers on raymarched content. Prefer \
                 `capture apng` for shareable clips, or `capture frames` + ffmpeg \
                 palettegen for high-quality post-processed GIFs.",
            );
            tracing::warn!(
                "capture: GIF quality is limited for raymarched content (per-frame \
                 palette regeneration causes flicker); prefer apng or PNG sequence"
            );
            // `palette=global` is the more experimental path (warmup buffer, shared
            // NeuQuant, training-data bias if anything is visible during warmup).
            // Stack a second warning so the extra caveats are surfaced.
            if p.palette == PaletteMode::Global {
                out.error(
                    "GIF palette=global: palette is trained from the first ~1s of \
                     captures; anything on screen during that window (the console, \
                     transient overlays) biases the palette toward those colors and \
                     the rest of the recording looks desaturated. Capture pre-egui \
                     (`capture gif pre palette=global`) to avoid.",
                );
                tracing::warn!(
                    "capture: GIF palette=global trains on the warmup buffer; ensure \
                     no transient UI is visible during the first ~1s of capture"
                );
            }
            enqueue(CaptureRequest::StartSequence {
                format: CaptureFormat::Gif,
                stage: p.stage,
                dir: p.dir,
                name: None,
                fps: p.fps,
                scale: p.scale,
                palette: p.palette,
            });
            out.line(format!(
                "started GIF stream ({:?}, fps={}, scale={}, palette={:?})",
                p.stage,
                p.fps.map_or("default".into(), |f| f.to_string()),
                p.scale.map_or("native".into(), |s| s.to_string()),
                p.palette,
            ));
        }
        "apng" => {
            let p = parse_capture_args(rest);
            enqueue(CaptureRequest::StartSequence {
                format: CaptureFormat::Apng,
                stage: p.stage,
                dir: p.dir,
                name: None,
                fps: p.fps,
                scale: p.scale,
                palette: PaletteMode::default(),
            });
            out.line(format!(
                "started APNG stream ({:?}, fps={}, scale={})",
                p.stage,
                p.fps.map_or("default".into(), |f| f.to_string()),
                p.scale.map_or("native".into(), |s| s.to_string()),
            ));
        }
        "stop" => {
            enqueue(CaptureRequest::Stop);
            out.line("stop queued");
        }
        "panel" => {
            let now_open = toggle_panel_global();
            out.line(if now_open {
                "panel opened"
            } else {
                "panel closed"
            });
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
                palette: p.palette,
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
    palette: PaletteMode,
}

impl Default for ParsedCaptureArgs {
    fn default() -> Self {
        Self {
            stage: CaptureStage::Post,
            dir: None,
            fps: None,
            scale: None,
            palette: PaletteMode::default(),
        }
    }
}

/// Tokenise post-format positional args:
/// - `pre|post|both`     => stage
/// - `fps=N`             => target frame rate
/// - `scale=N`           => output width in pixels (GIF only)
/// - `palette=local`     => per-frame NeuQuant (GIF only)
/// - `palette=global`    => warmup-trained global NeuQuant (GIF only)
/// - anything else       => treat as output directory
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
        } else if let Some(v) = arg.strip_prefix("palette=") {
            match v {
                "local" => p.palette = PaletteMode::Local,
                "global" => p.palette = PaletteMode::Global,
                _ => {}
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
        Some((&"apng", rest)) => (CaptureFormat::Apng, rest),
        // No format keyword recognised; treat the whole list as stage/dir args and use
        // the default streaming format (gif, the "share this clip" shape).
        _ => (CaptureFormat::Gif, args),
    }
}

/// Bind the default capture hotkeys on the given console:
/// - `F12`: `capture png post` (one-shot screenshot)
/// - `F9`:  `capture toggle gif post` (press to start a GIF, again to stop)
/// - `F11`: `capture panel` (toggle the parameters UI)
pub fn bind_default_hotkeys<Ctx: 'static>(console: &mut Console<Ctx>) {
    console.bind(rye_egui::egui::Key::F12, "capture png post");
    console.bind(rye_egui::egui::Key::F9, "capture toggle gif post");
    console.bind(rye_egui::egui::Key::F11, "capture panel");
}

// ---------------------------------------------------------------------------
// Panel UI
// ---------------------------------------------------------------------------

/// Egui widget for setting capture parameters (format, output dir, fps, scale) and
/// driving start / stop / one-shot via buttons.
///
/// The panel owns its own widget state (text edits, slider values, etc.); pushing the
/// `Start`/`Screenshot` buttons synthesises a [`CaptureRequest`] and pushes it onto the
/// global queue. Visibility is driven by [`CapturePanel::open`] *or* by the global
/// toggle the `capture panel` console subcommand flips, whichever changed last.
///
/// Wire it in your demo with two calls (state + per-frame show):
///
/// ```ignore
/// // setup:
/// let capture_panel = rye_app::capture::CapturePanel::new();
///
/// // each frame in App::ui:
/// self.capture_panel.show(egui_ctx);
/// ```
pub struct CapturePanel {
    /// Visible? Toggle via [`CapturePanel::toggle`] or the console `capture panel`
    /// subcommand (which flips a global the panel mirrors each frame).
    pub open: bool,
    output_dir: String,
    name: String,
    format: CaptureFormat,
    stage: CaptureStage,
    fps: u16,
    scale_enabled: bool,
    scale_width: u32,
    palette_mode: PaletteMode,
}

impl Default for CapturePanel {
    fn default() -> Self {
        Self {
            open: false,
            output_dir: "captures".into(),
            name: String::new(),
            format: CaptureFormat::Gif,
            stage: CaptureStage::Post,
            fps: 30,
            scale_enabled: false,
            scale_width: 720,
            palette_mode: PaletteMode::default(),
        }
    }
}

impl CapturePanel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn toggle(&mut self) {
        self.open = !self.open;
        PANEL_OPEN.store(self.open, Ordering::Relaxed);
    }

    /// Per-frame entry point. Mirrors the global console-driven toggle into
    /// `self.open`, then renders the egui window when open.
    pub fn show(&mut self, ctx: &rye_egui::egui::Context) {
        let global = PANEL_OPEN.load(Ordering::Relaxed);
        if global != self.open {
            self.open = global;
        }
        if !self.open {
            return;
        }
        let mut open_flag = self.open;
        rye_egui::egui::Window::new("capture")
            .open(&mut open_flag)
            .resizable(true)
            .default_width(280.0)
            .show(ctx, |ui| self.body(ui));
        if open_flag != self.open {
            self.open = open_flag;
            PANEL_OPEN.store(self.open, Ordering::Relaxed);
        }
    }

    fn body(&mut self, ui: &mut rye_egui::egui::Ui) {
        let recording_status = current_status();
        let recording = recording_status.is_some();

        ui.label(format!(
            "Status: {}",
            recording_status.as_deref().unwrap_or("Idle")
        ));
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Dir:");
            ui.add(rye_egui::egui::TextEdit::singleline(&mut self.output_dir).desired_width(180.0));
        });
        ui.horizontal(|ui| {
            ui.label("Name:");
            ui.add(rye_egui::egui::TextEdit::singleline(&mut self.name).desired_width(160.0));
            if self.name.is_empty() {
                ui.weak("(auto)");
            }
        });

        ui.horizontal(|ui| {
            ui.label("Format:");
            ui.radio_value(&mut self.format, CaptureFormat::Png, "PNG");
            ui.radio_value(&mut self.format, CaptureFormat::Gif, "GIF");
            ui.radio_value(&mut self.format, CaptureFormat::Apng, "APNG");
        });

        // Stage radio: only PNG sequences support pre/both; GIF is forced to Post on
        // request so the radio is read-only there.
        let stage_enabled = self.format == CaptureFormat::Png;
        ui.add_enabled_ui(stage_enabled, |ui| {
            ui.horizontal(|ui| {
                ui.label("Stage:");
                ui.radio_value(&mut self.stage, CaptureStage::Pre, "pre");
                ui.radio_value(&mut self.stage, CaptureStage::Post, "post");
                ui.radio_value(&mut self.stage, CaptureStage::Both, "both");
            });
        });

        ui.horizontal(|ui| {
            ui.label("FPS:");
            ui.add(rye_egui::egui::Slider::new(&mut self.fps, 1..=60));
        });

        // Scale is GIF-only; the LSD is locked at the first frame so we don't expose
        // mid-stream resize. For PNG, every frame is a separate file and downscale
        // would defeat the diagnostic goal.
        let scale_supported = matches!(self.format, CaptureFormat::Gif | CaptureFormat::Apng);
        ui.add_enabled_ui(scale_supported, |ui| {
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.scale_enabled, "Scale:");
                ui.add_enabled(
                    self.scale_enabled,
                    rye_egui::egui::Slider::new(&mut self.scale_width, 240..=2160).suffix(" px"),
                );
            });
        });

        // Palette mode: GIF-only. Local = per-frame NeuQuant (current default, may
        // flicker); Global = warmup-trained shared palette (no flicker after the
        // first ~1 s warmup). PNG and APNG don't use a palette (24-bit per frame).
        ui.add_enabled_ui(self.format == CaptureFormat::Gif, |ui| {
            ui.horizontal(|ui| {
                ui.label("Palette:");
                ui.radio_value(&mut self.palette_mode, PaletteMode::Local, "local");
                ui.radio_value(&mut self.palette_mode, PaletteMode::Global, "global");
            });
        });

        ui.separator();

        ui.horizontal(|ui| {
            if ui.button("Screenshot").clicked() {
                enqueue(CaptureRequest::OneShot {
                    stage: CaptureStage::Both,
                    dir: Some(PathBuf::from(&self.output_dir)),
                    name: (!self.name.is_empty()).then(|| self.name.clone()),
                });
            }
            let label = if recording { "Stop" } else { "Start" };
            if ui.button(label).clicked() {
                if recording {
                    enqueue(CaptureRequest::Stop);
                } else {
                    enqueue(CaptureRequest::StartSequence {
                        format: self.format,
                        stage: self.stage,
                        dir: Some(PathBuf::from(&self.output_dir)),
                        name: (!self.name.is_empty()).then(|| self.name.clone()),
                        fps: Some(self.fps),
                        scale: self.scale_enabled.then_some(self.scale_width),
                        palette: self.palette_mode,
                    });
                }
            }
        });
    }
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
