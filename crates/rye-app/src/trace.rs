//! Console command that surfaces [`rye_time::frame_trace`] aggregates into the dev
//! console. The trace module silently collects per-frame section timings on every
//! redraw; `trace` is the bridge that lets a human read them.
//!
//! ## Subcommands
//!
//! - `trace` / `trace summary` — print the aggregate p50 / p95 / p99 / max for every
//!   section in the rolling window (sorted by p95 descending). Sane default for "what
//!   is the slowest part of a frame right now."
//! - `trace last` — print the most recently completed frame's per-section breakdown.
//!   Useful for catching a one-off spike: hit it right after the visible stutter.
//! - `trace clear` — drop the rolling history. The next `trace summary` reflects only
//!   frames recorded after the clear.
//! - `trace cap <N>` — set the rolling-window size to N frames. Default is 120
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
    out.line(&format!(
        "trace summary ({history_len} frames, sorted by p95 desc):"
    ));
    // Header. Column widths picked to fit common section names + reasonable us/ms
    // values. Names beyond 16 chars get truncated which is fine; if it becomes a
    // problem we can widen the field.
    out.line(&format!(
        "  {:<18} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "section", "n", "mean", "p50", "p95", "p99", "max",
    ));
    for s in stats {
        out.line(&format!(
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
    out.line(&format!(
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
        out.line(&format!(
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
    /// demos on blog posts where the perf data should always show.
    pub fn always_visible(mut self) -> Self {
        self.visible = true;
        self
    }

    /// Render the overlay. Call once per frame from `App::ui`. Handles the
    /// toggle key internally so the demo doesn't need to forward F3.
    pub fn show(&mut self, ctx: &rye_egui::egui::Context) {
        use rye_egui::egui;

        // Check toggle even when hidden; otherwise F3 couldn't reopen.
        if ctx.input(|i| i.key_pressed(self.toggle_key)) {
            self.visible = !self.visible;
        }
        if !self.visible {
            return;
        }

        let history = frame_trace::history();
        if history.is_empty() {
            return;
        }

        let start = history.len().saturating_sub(self.window);
        let recent = &history[start..];

        let cadence: Vec<Duration> = collect_section(recent, "between-frames");
        let frames: Vec<Duration> = collect_section(recent, "frame");
        let idles: Vec<Duration> = collect_section(recent, "idle");

        let cadence_mean = mean(&cadence);
        let cadence_p99 = percentile(&cadence, 0.99);
        let cadence_max = max_dur(&cadence);
        let frame_mean = mean(&frames);
        let frame_p99 = percentile(&frames, 0.99);
        let idle_mean = mean(&idles);
        let idle_p99 = percentile(&idles, 0.99);
        let idle_max = max_dur(&idles);

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
                        ui.label(
                            egui::RichText::new(format!(
                                "total  {:>5.1}  p99 {:>5.1}  max {:>5.1}  ms",
                                cadence_mean.as_secs_f32() * 1000.0,
                                cadence_p99.as_secs_f32() * 1000.0,
                                cadence_max.as_secs_f32() * 1000.0,
                            ))
                            .font(mono.clone())
                            .color(label_color),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "idle   {:>5.1}  p99 {:>5.1}  max {:>5.1}  ms",
                                idle_mean.as_secs_f32() * 1000.0,
                                idle_p99.as_secs_f32() * 1000.0,
                                idle_max.as_secs_f32() * 1000.0,
                            ))
                            .font(mono.clone())
                            .color(label_color),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "frame  {:>5.2}  p99 {:>5.2}                ms",
                                frame_mean.as_secs_f32() * 1000.0,
                                frame_p99.as_secs_f32() * 1000.0,
                            ))
                            .font(mono)
                            .color(label_color),
                        );
                        draw_sparkline(ui, &cadence);
                    });
            });
    }
}

fn collect_section(frames: &[frame_trace::FrameTrace], name: &str) -> Vec<Duration> {
    frames
        .iter()
        .flat_map(|f| f.sections.iter())
        .filter(|s| s.name == name)
        .map(|s| s.elapsed)
        .collect()
}

fn mean(samples: &[Duration]) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    samples.iter().sum::<Duration>() / samples.len() as u32
}

fn max_dur(samples: &[Duration]) -> Duration {
    samples.iter().copied().max().unwrap_or(Duration::ZERO)
}

fn percentile(samples: &[Duration], q: f32) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    let mut sorted: Vec<Duration> = samples.to_vec();
    sorted.sort();
    let idx = ((sorted.len() as f32 * q) as usize).min(sorted.len() - 1);
    sorted[idx]
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
