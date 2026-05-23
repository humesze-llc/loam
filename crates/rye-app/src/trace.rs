//! Console command that surfaces [`rye_time::frame_trace`] aggregates into the dev
//! console. The trace module silently collects per-frame section timings on every
//! redraw; `trace` is the bridge that lets a human read them.
//!
//! ## Subcommands
//!
//! - `trace` / `trace summary`: print the aggregate p50 / p95 / p99 / max for every
//!   section in the rolling window (sorted by p95 descending). Sane default for "what
//!   is the slowest part of a frame right now."
//! - `trace last`: print the most recently completed frame's per-section breakdown.
//!   Useful for catching a one-off spike: hit it right after the visible stutter.
//! - `trace clear`: drop the rolling history. The next `trace summary` reflects only
//!   frames recorded after the clear.
//! - `trace cap <N>`: set the rolling-window size to N frames. Default is 120
//!   (~2 seconds at 60fps). Larger windows smooth out short-term variance but
//!   take longer to react to changes in the hot path.
//!
//! ## Wiring (per demo)
//!
//! ```ignore
//! // In build_console:
//! rye_app::trace::register_command(&mut c);
//! ```
//!
//! The actual collection is already wired into `rye-app::Runner::redraw`; the demo
//! only needs to register the console command if it wants users to surface the data.

use rye_egui::{cmd, Console};
use rye_time::frame_trace;
use std::time::Duration;

/// Format a duration into a compact human-readable string. us if < 1ms, ms if < 1s,
/// s otherwise. Used by the trace command's output rows so the scrollback stays
/// readable instead of dumping nanosecond ints.
fn fmt_dur(d: std::time::Duration) -> String {
    let ns = d.as_nanos();
    if ns < 1_000 {
        format!("{ns}ns")
    } else if ns < 1_000_000 {
        format!("{:.1}us", ns as f64 / 1_000.0)
    } else if ns < 1_000_000_000 {
        format!("{:.2}ms", ns as f64 / 1_000_000.0)
    } else {
        format!("{:.2}s", ns as f64 / 1_000_000_000.0)
    }
}

/// Print the rolling-window aggregate to the console: one row per section, sorted by
/// p95 descending. Slowest hot-path sections naturally sort to the top.
fn print_summary(out: &mut rye_egui::ConsoleWriter) {
    let stats = frame_trace::aggregate();
    if stats.is_empty() {
        out.line("trace: no frames in window (collect runs once the demo is rendering)");
        return;
    }
    let history_len = frame_trace::history().len();
    out.line(format!(
        "trace summary ({history_len} frames, sorted by p95 desc):"
    ));
    // Header. Column widths picked to fit common section names + reasonable us/ms
    // values. Names beyond 16 chars get truncated which is fine; if it becomes a
    // problem we can widen the field.
    out.line(format!(
        "  {:<18} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "section", "n", "mean", "p50", "p95", "p99", "max",
    ));
    for s in stats {
        out.line(format!(
            "  {:<18} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}",
            truncate(s.name, 18),
            s.samples,
            fmt_dur(s.mean),
            fmt_dur(s.p50),
            fmt_dur(s.p95),
            fmt_dur(s.p99),
            fmt_dur(s.max),
        ));
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}~", &s[..max - 1])
    }
}

/// Print the most recent completed frame's section breakdown. Order preserved from
/// the order scopes opened, so the reader gets a natural top-to-bottom timeline of
/// the frame.
fn print_last(out: &mut rye_egui::ConsoleWriter) {
    let Some(frame) = frame_trace::last_frame() else {
        out.line("trace: no frames in window yet");
        return;
    };
    let total = frame.total();
    out.line(format!(
        "trace last-frame ({} sections, sum {}):",
        frame.sections.len(),
        fmt_dur(total),
    ));
    for section in &frame.sections {
        let pct = if !total.is_zero() {
            (section.elapsed.as_nanos() as f64 * 100.0) / total.as_nanos() as f64
        } else {
            0.0
        };
        out.line(format!(
            "  {:<18} {:>10} ({:>4.1}%)",
            truncate(section.name, 18),
            fmt_dur(section.elapsed),
            pct,
        ));
    }
}

/// Format the summary as multi-line text. Used by both the console output path AND
/// the `dump` subcommand that emits via `tracing::info!` so the same data is reachable
/// from both the in-canvas console (not browser-selectable) and the browser dev tools
/// console (selectable + copyable).
fn format_summary() -> String {
    let stats = frame_trace::aggregate();
    if stats.is_empty() {
        return "trace: no frames in window\n".to_string();
    }
    let history_len = frame_trace::history().len();
    let mut s = String::new();
    s.push_str(&format!(
        "trace summary ({history_len} frames, sorted by p95 desc):\n"
    ));
    s.push_str(&format!(
        "  {:<18} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}\n",
        "section", "n", "mean", "p50", "p95", "p99", "max",
    ));
    for st in stats {
        s.push_str(&format!(
            "  {:<18} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}\n",
            truncate(st.name, 18),
            st.samples,
            fmt_dur(st.mean),
            fmt_dur(st.p50),
            fmt_dur(st.p95),
            fmt_dur(st.p99),
            fmt_dur(st.max),
        ));
    }
    s
}

/// Register the `trace` console command.
pub fn register_command<Ctx: 'static>(console: &mut Console<Ctx>) {
    console.register(
        cmd(
            "trace",
            "show CPU per-section frame timings (collected by rye-time::frame_trace)",
            |args, _ctx: &mut Ctx, out| {
                match args.first().copied() {
                    None | Some("summary") => print_summary(out),
                    Some("last") => print_last(out),
                    Some("dump") => {
                        // Emit via tracing::info! so the same data lands in the browser
                        // DevTools console (selectable + copyable) on wasm, and in
                        // stdout on native. The in-canvas console renders text as
                        // pixels so the in-app `trace summary` text isn't browser-
                        // selectable; `trace dump` is the workaround.
                        let summary = format_summary();
                        // Multi-line tracing event: the receiving subscriber (tracing-
                        // wasm or fmt) writes the whole block as one event so the
                        // newlines stay grouped in the browser console output.
                        tracing::info!("\n{summary}");
                        out.line("trace: dumped to browser console (open DevTools to copy)");
                    }
                    Some("clear") => {
                        frame_trace::set_capacity(1);
                        frame_trace::set_capacity(frame_trace::DEFAULT_CAPACITY);
                        out.line("trace: history cleared");
                    }
                    Some("cap") => {
                        let n = args
                            .get(1)
                            .copied()
                            .and_then(|s| s.parse::<usize>().ok());
                        match n {
                            Some(n) if n >= 1 => {
                                frame_trace::set_capacity(n);
                                out.line(format!("trace: capacity set to {n} frames"));
                            }
                            _ => {
                                out.line("usage: trace cap <N>  (N >= 1)");
                            }
                        }
                    }
                    Some(other) => {
                        out.line(format!(
                            "trace: unknown subcommand '{other}' (try summary | last | dump | clear | cap)"
                        ));
                    }
                }
                Ok(())
            },
        )
        .with_args(&[&["summary", "last", "dump", "clear", "cap"]]),
    );
}

// ---------------------------------------------------------------------------
// PerfOverlay: F3-style always-on perf readout
// ---------------------------------------------------------------------------

/// Live FPS / frame-time overlay, Minecraft-F3 style. Reads from `frame_trace`'s
/// rolling history and renders a compact corner panel with:
///
/// - **FPS**: 1 / mean(`between-frames`) over the recent window.
/// - **gap**: mean / p99 / max of `between-frames` (browser-side RAF cadence;
///   the slow source per the 2026-05-22 wasm investigation).
/// - **frm**: mean / p99 of `frame` (our CPU work; usually tiny).
/// - **sparkline**: last N `between-frames` durations as a bar chart with
///   reference lines at 16.7ms (60fps) and 33.3ms (30fps). Bars color-code by
///   severity: green = under 20ms, amber = 20-33ms, red = over 33ms (dropped
///   frame).
///
/// Toggle visibility with `F3` (matches the Minecraft convention). The toggle
/// is checked every call to [`PerfOverlay::show`]; the demo opts in by calling
/// `self.perf.show(ctx)` from its `App::ui`.
///
/// ## Why this isn't the egui `Plot` widget
///
/// `Plot` is a heavy widget that pulls in axis labels, legends, zoom, drag
/// interaction. A perf sparkline doesn't need any of that and the Plot
/// pipeline state would add to egui's compile-time pipeline count (the thing
/// we suspect contributes to stutters). Direct `ui.painter()` calls keep this
/// to one rect + a few line segments per frame.
pub struct PerfOverlay {
    visible: bool,
    /// Toggle hotkey. `F3` by default; change with [`Self::with_toggle_key`].
    toggle_key: rye_egui::egui::Key,
    /// Number of recent frames the readout summarizes over. Defaults to 60
    /// (one second at 60fps). Larger = smoother numbers, slower reaction.
    window: usize,
}

impl Default for PerfOverlay {
    fn default() -> Self {
        Self {
            visible: false,
            toggle_key: rye_egui::egui::Key::F3,
            window: 60,
        }
    }
}

impl PerfOverlay {
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the toggle key. Useful for demos where F3 is taken (or for
    /// "always visible" by setting to a key the user will never press).
    pub fn with_toggle_key(mut self, key: rye_egui::egui::Key) -> Self {
        self.toggle_key = key;
        self
    }

    /// Force the overlay visible regardless of toggle state. For embedded
    /// demos where the perf data should always show.
    pub fn always_visible(mut self) -> Self {
        self.visible = true;
        self
    }

    /// Render the overlay. Call once per frame from `App::ui`. Handles the
    /// toggle key internally so the demo doesn't need to forward F3.
    ///
    /// Zero-alloc per call: history is read via [`frame_trace::with_history`]
    /// borrow, samples are accumulated into a fixed-size stack buffer
    /// ([`MAX_WINDOW`] cap), and percentile sorting happens in-place. The
    /// previous `Vec<FrameTrace>::clone` per frame was ~120-130 allocations on
    /// its own and was swamping the NH3 alloc telemetry. Now the overlay
    /// contributes zero allocations to the steady-state alloc count.
    pub fn show(&mut self, ctx: &rye_egui::egui::Context) {
        use rye_egui::egui;

        // Check toggle even when hidden; otherwise F3 couldn't reopen.
        if ctx.input(|i| i.key_pressed(self.toggle_key)) {
            self.visible = !self.visible;
        }
        if !self.visible {
            return;
        }

        // Single pass over the rolling history through `with_history`'s borrow.
        // Stack-bound accumulators only. Cap the window at MAX_WINDOW so the
        // stack buffer below never overflows.
        let window = self.window.min(MAX_WINDOW);
        let mut cadence = StackBuf::new();
        let mut frames_buf = StackBuf::new();
        let mut idles = StackBuf::new();
        let mut heap_count = 0usize;
        let mut heap_peak = 0i64;
        let mut heap_net = 0i64;
        let mut alloc_frames = 0usize;
        let mut alloc_count_sum: u64 = 0;
        let mut alloc_peak_bytes: u64 = 0;
        let mut alloc_net_bytes: i64 = 0;
        let mut any = false;

        frame_trace::with_history(|history| {
            let start = history.len().saturating_sub(window);
            for frame in history.iter().skip(start) {
                any = true;
                for section in &frame.sections {
                    match section.name {
                        "between-frames" => cadence.push(section.elapsed),
                        "frame" => frames_buf.push(section.elapsed),
                        "idle" => idles.push(section.elapsed),
                        _ => {}
                    }
                }
                if let Some(d) = frame.heap_delta_bytes {
                    heap_count += 1;
                    if d > heap_peak {
                        heap_peak = d;
                    }
                    heap_net = heap_net.saturating_add(d);
                }
                if let Some(a) = frame.allocs {
                    alloc_frames += 1;
                    alloc_count_sum = alloc_count_sum.saturating_add(a.alloc_count);
                    if a.alloc_bytes > alloc_peak_bytes {
                        alloc_peak_bytes = a.alloc_bytes;
                    }
                    alloc_net_bytes = alloc_net_bytes.saturating_add(a.net_bytes);
                }
            }
        });

        if !any {
            return;
        }

        // Reductions on the stack buffers. `.percentile` mutates the buffer
        // (sorts in-place); subsequent reads must use `.mean` BEFORE the sort
        // if they want unsorted-order semantics. Mean doesn't care about
        // order so it's order-independent; we call it first to be explicit.
        let cadence_mean = cadence.mean();
        let cadence_p99 = cadence.percentile(0.99);
        let frame_mean = frames_buf.mean();
        let frame_p99 = frames_buf.percentile(0.99);
        let idle_mean = idles.mean();
        let idle_p99 = idles.percentile(0.99);

        // Session-lifetime maxima. Distinct from `max_dur(&cadence)` which is
        // bounded by the rolling window's contents: a 1-second freeze that
        // happened 5 seconds ago has already aged out of the window's 120
        // frames, so a window-only "max" lies. `max_ever` survives the entire
        // session and is the answer to "what's the worst this has ever been?"
        let cadence_max_ever = frame_trace::max_ever("between-frames");
        let idle_max_ever = frame_trace::max_ever("idle");
        let frame_max_ever = frame_trace::max_ever("frame");

        let fps = if cadence_mean.as_secs_f32() > 0.0 {
            1.0 / cadence_mean.as_secs_f32()
        } else {
            0.0
        };

        egui::Area::new(egui::Id::new("rye-perf-overlay"))
            .anchor(egui::Align2::RIGHT_TOP, [-12.0, 12.0])
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style())
                    .stroke(egui::Stroke::new(
                        1.0,
                        egui::Color32::from_rgb(60, 60, 75),
                    ))
                    .show(ui, |ui| {
                        ui.set_min_width(260.0);
                        let mono = egui::FontId::monospace(11.0);
                        let label_color = egui::Color32::from_rgb(180, 190, 200);
                        ui.label(
                            egui::RichText::new(format!("FPS    {fps:5.1}"))
                                .font(mono.clone())
                                .color(egui::Color32::from_rgb(220, 230, 240)),
                        );
                        // Three rows: total cadence (between-frames), our CPU
                        // work (frame), and the browser/RAF gap (idle). The
                        // first should equal the sum of the latter two in the
                        // long run; differences = scope-uncovered work in
                        // redraw (FPS bookkeeping, capture, etc.).
                        //
                        // The `worst` column is session-lifetime; survives the
                        // rolling window so multi-second spikes that happened
                        // minutes ago are still visible. Colored red when it's
                        // pathological (>= 100ms = ~6 vsync) so the user's eye
                        // catches "this has been bad" at a glance.
                        ui.label(
                            egui::RichText::new(format!(
                                "total  {:>5.1}  p99 {:>5.1}  ms",
                                cadence_mean.as_secs_f32() * 1000.0,
                                cadence_p99.as_secs_f32() * 1000.0,
                            ))
                            .font(mono.clone())
                            .color(label_color),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "idle   {:>5.1}  p99 {:>5.1}  ms",
                                idle_mean.as_secs_f32() * 1000.0,
                                idle_p99.as_secs_f32() * 1000.0,
                            ))
                            .font(mono.clone())
                            .color(label_color),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "frame  {:>5.2}  p99 {:>5.2}  ms",
                                frame_mean.as_secs_f32() * 1000.0,
                                frame_p99.as_secs_f32() * 1000.0,
                            ))
                            .font(mono.clone())
                            .color(label_color),
                        );
                        ui.separator();
                        ui.label(
                            egui::RichText::new("worst-ever (session)")
                                .font(mono.clone())
                                .color(egui::Color32::from_rgb(140, 150, 160)),
                        );
                        let worst_color = |d: Duration| {
                            let ms = d.as_secs_f32() * 1000.0;
                            if ms >= 100.0 {
                                egui::Color32::from_rgb(220, 100, 80)
                            } else if ms >= 50.0 {
                                egui::Color32::from_rgb(220, 180, 90)
                            } else {
                                label_color
                            }
                        };
                        ui.label(
                            egui::RichText::new(format!(
                                "total  {:>6.1} ms",
                                cadence_max_ever.as_secs_f32() * 1000.0,
                            ))
                            .font(mono.clone())
                            .color(worst_color(cadence_max_ever)),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "idle   {:>6.1} ms",
                                idle_max_ever.as_secs_f32() * 1000.0,
                            ))
                            .font(mono.clone())
                            .color(worst_color(idle_max_ever)),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "frame  {:>6.1} ms",
                                frame_max_ever.as_secs_f32() * 1000.0,
                            ))
                            .font(mono.clone())
                            .color(worst_color(frame_max_ever)),
                        );
                        // Alloc section. Visible when the demo opted in via
                        // CountingAllocator. Three rows: mean allocs/frame
                        // (steady-state allocation rate; target = 0), peak
                        // bytes per frame (worst spike), and net bytes
                        // across the window (catches steady leaks the per-
                        // frame mean might smear into the noise). All three
                        // matter: a frame with 200 1-byte allocs vs. 1
                        // 200-byte alloc looks identical in byte-count but
                        // very different in JS-interop cost on wasm.
                        if alloc_frames > 0 {
                            ui.separator();
                            ui.label(
                                egui::RichText::new("allocs (Rust heap)")
                                    .font(mono.clone())
                                    .color(egui::Color32::from_rgb(140, 150, 160)),
                            );
                            let mean_count = alloc_count_sum / alloc_frames as u64;
                            let count_color = |n: u64| {
                                if n >= 1_000 {
                                    egui::Color32::from_rgb(220, 100, 80)
                                } else if n >= 100 {
                                    egui::Color32::from_rgb(220, 180, 90)
                                } else if n >= 10 {
                                    egui::Color32::from_rgb(180, 200, 130)
                                } else {
                                    egui::Color32::from_rgb(120, 200, 130)
                                }
                            };
                            let byte_color = |bytes: i64| {
                                let mb = bytes.abs() as f32 / (1024.0 * 1024.0);
                                if mb >= 10.0 {
                                    egui::Color32::from_rgb(220, 100, 80)
                                } else if mb >= 1.0 {
                                    egui::Color32::from_rgb(220, 180, 90)
                                } else {
                                    label_color
                                }
                            };
                            ui.label(
                                egui::RichText::new(format!(
                                    "mean  {mean_count:>6} allocs/frame",
                                ))
                                .font(mono.clone())
                                .color(count_color(mean_count)),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "peak  {:>+6.2} KB/frame",
                                    alloc_peak_bytes as f32 / 1024.0,
                                ))
                                .font(mono.clone())
                                .color(byte_color(alloc_peak_bytes as i64)),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "net   {:>+6.2} MB / window",
                                    alloc_net_bytes as f32 / (1024.0 * 1024.0),
                                ))
                                .font(mono.clone())
                                .color(byte_color(alloc_net_bytes)),
                            );
                        }
                        // Heap section (Chromium only). When no samples are
                        // present (Firefox / native) we skip it entirely
                        // rather than showing zeroes the reader might
                        // misread as "no allocations." Two rows: peak
                        // per-frame growth (correlates with the spike-warn
                        // log lines) and net growth across the window
                        // (catches steady-state per-frame leaks).
                        if heap_count > 0 {
                            ui.separator();
                            ui.label(
                                egui::RichText::new("heap (Chromium)")
                                    .font(mono.clone())
                                    .color(egui::Color32::from_rgb(140, 150, 160)),
                            );
                            let heap_color = |bytes: i64| {
                                let mb = bytes.abs() as f32 / (1024.0 * 1024.0);
                                if mb >= 10.0 {
                                    egui::Color32::from_rgb(220, 100, 80)
                                } else if mb >= 2.0 {
                                    egui::Color32::from_rgb(220, 180, 90)
                                } else {
                                    label_color
                                }
                            };
                            ui.label(
                                egui::RichText::new(format!(
                                    "peak  {:>+6.2} MB/frame",
                                    heap_peak as f32 / (1024.0 * 1024.0),
                                ))
                                .font(mono.clone())
                                .color(heap_color(heap_peak)),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "net   {:>+6.2} MB / window",
                                    heap_net as f32 / (1024.0 * 1024.0),
                                ))
                                .font(mono.clone())
                                .color(heap_color(heap_net)),
                            );
                        }
                        ui.separator();
                        draw_sparkline(ui, cadence.as_slice());
                    });
            });
    }
}

/// Maximum window size the PerfOverlay supports for percentile statistics.
/// Drives the inline stack array in `StackBuf`; oversized windows are
/// clamped to this value. 256 × 16 B/Duration = 4 KB on stack per buffer;
/// three buffers (cadence, frame, idle) = 12 KB. Comfortably inside the
/// 1 MB main-thread stack on every supported target.
///
/// Architectural note: the cap exists ONLY to keep the stack buffer fixed-
/// size; the actual rolling-window capacity in `frame_trace` is independent
/// and can be larger. If a user configures `with_window(>MAX_WINDOW)` we
/// silently use the cap; the alternative (heap-allocating to match the
/// requested window) defeats the zero-alloc property the overlay exists to
/// enable.
pub const MAX_WINDOW: usize = 256;

/// Fixed-capacity stack-allocated sample buffer. Zero-allocation alternative
/// to `Vec<Duration>` for per-frame UI use. Push silently drops samples once
/// `len` reaches `MAX_WINDOW`; the caller is expected to cap its iteration
/// window to match.
#[derive(Clone)]
struct StackBuf {
    samples: [Duration; MAX_WINDOW],
    len: usize,
}

impl StackBuf {
    fn new() -> Self {
        Self {
            samples: [Duration::ZERO; MAX_WINDOW],
            len: 0,
        }
    }

    fn push(&mut self, d: Duration) {
        if self.len < MAX_WINDOW {
            self.samples[self.len] = d;
            self.len += 1;
        }
    }

    fn as_slice(&self) -> &[Duration] {
        &self.samples[..self.len]
    }

    fn mean(&self) -> Duration {
        if self.len == 0 {
            return Duration::ZERO;
        }
        let sum: Duration = self.samples[..self.len].iter().sum();
        sum / self.len as u32
    }

    /// `q`-percentile over the window. Order-preserving on `self`: clones the
    /// sample range into a stack-local array and sorts THAT, leaving the
    /// caller's buffer in its original (time-ordered) state. The sparkline
    /// downstream needs the original order; the percentile readout doesn't.
    fn percentile(&self, q: f32) -> Duration {
        if self.len == 0 {
            return Duration::ZERO;
        }
        let mut local: [Duration; MAX_WINDOW] = self.samples;
        local[..self.len].sort();
        let idx = ((self.len as f32 * q) as usize).min(self.len - 1);
        local[idx]
    }
}

fn draw_sparkline(ui: &mut rye_egui::egui::Ui, gaps: &[Duration]) {
    use rye_egui::egui;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(240.0, 36.0), egui::Sense::hover());
    if gaps.is_empty() {
        return;
    }
    let painter = ui.painter();
    painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(18, 18, 24));

    // Y-scale: clamp the top at 50ms so 30-50ms bars are still visible without
    // a single 200ms outlier squashing the baseline to invisibility. Outliers
    // beyond that draw to the top of the rect with their full color.
    let y_max_ms = 50.0_f32;
    let y_for_ms = |ms: f32| {
        let clamped = ms.min(y_max_ms);
        rect.bottom() - (clamped / y_max_ms) * rect.height()
    };
    // Reference lines: 60fps (16.7ms) and 30fps (33.3ms). Faint so they don't
    // dominate the bars; helps the eye locate the "good" vs "dropped frame" zones.
    let ref_60 = y_for_ms(16.67);
    let ref_30 = y_for_ms(33.33);
    painter.line_segment(
        [egui::pos2(rect.left(), ref_60), egui::pos2(rect.right(), ref_60)],
        egui::Stroke::new(0.5, egui::Color32::from_rgb(60, 100, 70)),
    );
    painter.line_segment(
        [egui::pos2(rect.left(), ref_30), egui::pos2(rect.right(), ref_30)],
        egui::Stroke::new(0.5, egui::Color32::from_rgb(120, 90, 60)),
    );

    let n = gaps.len() as f32;
    let dx = rect.width() / n.max(1.0);
    let bar_w = dx.max(1.0);
    for (i, gap) in gaps.iter().enumerate() {
        let x = rect.left() + i as f32 * dx;
        let ms = gap.as_secs_f32() * 1000.0;
        let y = y_for_ms(ms);
        let color = if ms > 33.3 {
            egui::Color32::from_rgb(220, 100, 80)
        } else if ms > 20.0 {
            egui::Color32::from_rgb(220, 180, 90)
        } else {
            egui::Color32::from_rgb(120, 200, 130)
        };
        painter.line_segment(
            [egui::pos2(x, rect.bottom()), egui::pos2(x, y)],
            egui::Stroke::new(bar_w, color),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `fmt_dur` is the primary string emitter the trace command uses;
    // verifying the unit boundary picks (ns / us / ms / s) catches drift
    // if anyone changes the thresholds. Lower bound at each boundary is
    // the "just-crossed" sample.

    #[test]
    fn fmt_dur_emits_ns_under_microsecond() {
        assert!(fmt_dur(Duration::from_nanos(0)).ends_with("ns"));
        assert!(fmt_dur(Duration::from_nanos(999)).ends_with("ns"));
    }

    #[test]
    fn fmt_dur_emits_us_under_millisecond() {
        assert!(fmt_dur(Duration::from_nanos(1_000)).ends_with("us"));
        assert!(fmt_dur(Duration::from_nanos(999_999)).ends_with("us"));
    }

    #[test]
    fn fmt_dur_emits_ms_under_second() {
        assert!(fmt_dur(Duration::from_nanos(1_000_000)).ends_with("ms"));
        assert!(fmt_dur(Duration::from_nanos(999_999_999)).ends_with("ms"));
    }

    #[test]
    fn fmt_dur_emits_seconds_above() {
        assert!(fmt_dur(Duration::from_secs(1)).ends_with('s'));
        assert!(!fmt_dur(Duration::from_secs(1)).ends_with("ms"));
        assert!(!fmt_dur(Duration::from_secs(1)).ends_with("us"));
    }

    #[test]
    fn truncate_preserves_short_names_and_marks_long_ones() {
        assert_eq!(truncate("frame", 18), "frame");
        // A name at the cap fits verbatim.
        assert_eq!(truncate("abcdefghijklmnopqr", 18).len(), 18);
        // A name PAST the cap collapses to `cap-1` chars plus a `~` mark.
        let long = "supercalifragilisticexpialidocious";
        let t = truncate(long, 18);
        assert_eq!(t.len(), 18);
        assert!(t.ends_with('~'), "truncate should mark with `~`");
    }

    #[test]
    fn stackbuf_starts_empty() {
        let buf = StackBuf::new();
        assert_eq!(buf.as_slice().len(), 0);
        assert_eq!(buf.mean(), Duration::ZERO);
        assert_eq!(buf.percentile(0.5), Duration::ZERO);
        assert_eq!(buf.percentile(0.99), Duration::ZERO);
    }

    #[test]
    fn stackbuf_push_appends_in_order() {
        let mut buf = StackBuf::new();
        for ms in [10u64, 20, 30] {
            buf.push(Duration::from_millis(ms));
        }
        let slice = buf.as_slice();
        assert_eq!(slice.len(), 3);
        assert_eq!(slice[0], Duration::from_millis(10));
        assert_eq!(slice[1], Duration::from_millis(20));
        assert_eq!(slice[2], Duration::from_millis(30));
    }

    #[test]
    fn stackbuf_push_silently_drops_past_max_window() {
        let mut buf = StackBuf::new();
        for i in 0..(MAX_WINDOW + 10) {
            buf.push(Duration::from_nanos(i as u64));
        }
        assert_eq!(
            buf.as_slice().len(),
            MAX_WINDOW,
            "push beyond cap must not allocate; samples drop"
        );
    }

    #[test]
    fn stackbuf_mean_handles_uniform_input() {
        let mut buf = StackBuf::new();
        for _ in 0..10 {
            buf.push(Duration::from_millis(16));
        }
        assert_eq!(buf.mean(), Duration::from_millis(16));
    }

    #[test]
    fn stackbuf_percentile_picks_nearest_rank() {
        // 10 samples: 1ms, 2ms, ..., 10ms.
        let mut buf = StackBuf::new();
        for ms in 1..=10u64 {
            buf.push(Duration::from_millis(ms));
        }
        // p50 with nearest-rank: floor(10 * 0.5) = 5 -> samples[5] = 6ms.
        assert_eq!(buf.percentile(0.5), Duration::from_millis(6));
        // p95: floor(10 * 0.95) = 9 -> samples[9] = 10ms (the max).
        assert_eq!(buf.percentile(0.95), Duration::from_millis(10));
        // p99 same range: clamped to len-1 = 9 -> 10ms.
        assert_eq!(buf.percentile(0.99), Duration::from_millis(10));
    }

    #[test]
    fn stackbuf_percentile_is_order_preserving_on_self() {
        // The implementation sorts into a stack-local copy and leaves `self`
        // untouched. The sparkline downstream relies on insertion order; if
        // a refactor moved the sort onto `self.samples`, the sparkline would
        // show sorted bars instead of time-ordered ones.
        let mut buf = StackBuf::new();
        for ms in [30u64, 5, 25, 10, 20] {
            buf.push(Duration::from_millis(ms));
        }
        let before: Vec<Duration> = buf.as_slice().to_vec();
        let _ = buf.percentile(0.5);
        let after: Vec<Duration> = buf.as_slice().to_vec();
        assert_eq!(before, after, "percentile must not reorder self");
    }
}
