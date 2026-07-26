//! Console command that surfaces [`loam_time::frame_trace`] aggregates into
//! the dev console. Collection is already wired into `Runner::redraw`; this
//! is the read side.
//!
//! Subcommands: `trace`/`trace summary` (aggregate p50/p95/p99/max, p95
//! desc), `trace last` (most recent frame breakdown), `trace clear`,
//! `trace cap <N>` (rolling-window size).
//!
//! ```ignore
//! loam_app::trace::register_command(&mut c);
//! ```

use loam_egui::{cmd, Console};
use loam_time::frame_trace;
use std::time::Duration;

/// Compact duration string: ns < 1us, us < 1ms, ms < 1s, s otherwise.
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

/// Print the rolling-window aggregate: one row per section, p95 descending.
fn print_summary(out: &mut loam_egui::ConsoleWriter) {
    let stats = frame_trace::aggregate();
    if stats.is_empty() {
        out.line("trace: no frames in window (collect runs once the demo is rendering)");
        return;
    }
    let history_len = frame_trace::history().len();
    out.line(format!(
        "trace summary ({history_len} frames, sorted by p95 desc):"
    ));
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

/// Print the most recent frame's section breakdown, in scope-open order.
fn print_last(out: &mut loam_egui::ConsoleWriter) {
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

/// Multi-line summary text. Used by the `dump` subcommand so the same data
/// reaches the browser DevTools console (selectable/copyable), which the
/// pixel-rendered in-canvas console is not.
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
            "show CPU per-section frame timings (collected by loam-time::frame_trace)",
            |args, _ctx: &mut Ctx, out| {
                match args.first().copied() {
                    None | Some("summary") => print_summary(out),
                    Some("last") => print_last(out),
                    Some("dump") => {
                        // tracing::info! so the block lands in the browser
                        // DevTools console (copyable) on wasm and stdout on
                        // native; one event keeps the newlines grouped.
                        let summary = format_summary();
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

/// Live FPS / frame-time overlay, F3-toggled. Reads `frame_trace`'s rolling
/// history and paints FPS, between-frames cadence, frame CPU work, and a
/// sparkline. Opt in by calling [`PerfOverlay::show`] from `App::ui`.
///
/// Hand-painted rather than the egui `Plot` widget: `Plot` adds pipeline
/// state to egui's compile-time pipeline count (a suspected stutter source);
/// direct `ui.painter()` calls cost one rect plus a few segments per frame.
pub struct PerfOverlay {
    visible: bool,
    /// Toggle hotkey. `F3` by default; change with [`Self::with_toggle_key`].
    toggle_key: loam_egui::egui::Key,
    /// Recent frames the readout summarizes over. Larger = smoother numbers,
    /// slower reaction.
    window: usize,
}

impl Default for PerfOverlay {
    fn default() -> Self {
        Self {
            visible: false,
            toggle_key: loam_egui::egui::Key::F3,
            window: 60,
        }
    }
}

/// Body width of the readout; also the right-inset basis for its boot seat.
const OVERLAY_WIDTH: f32 = 260.0;

/// Gap between the overlay and the edges of the space panels leave.
const OVERLAY_MARGIN: f32 = 12.0;

/// Boot position of the draggable overlay: inset from the top-right of the
/// space the host's panels leave. `available_rect` and not `content_rect`,
/// which is the viewport minus OS safe-area insets and is panel-unaware, so
/// seating against it puts the readout behind the shell's menu bar.
fn perf_overlay_seat(ctx: &loam_egui::egui::Context) -> loam_egui::egui::Pos2 {
    let area = ctx.available_rect();
    loam_egui::egui::pos2(
        (area.right() - OVERLAY_MARGIN - OVERLAY_WIDTH).max(area.left()),
        area.top() + OVERLAY_MARGIN,
    )
}

impl PerfOverlay {
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the toggle key (for demos where F3 is taken).
    pub fn with_toggle_key(mut self, key: loam_egui::egui::Key) -> Self {
        self.toggle_key = key;
        self
    }

    /// Force the overlay visible regardless of toggle state.
    pub fn always_visible(mut self) -> Self {
        self.visible = true;
        self
    }

    /// Render the overlay once per frame from `App::ui`; handles the toggle
    /// key internally.
    ///
    /// Zero-alloc per call: history read via [`frame_trace::with_history`],
    /// samples into a fixed-size stack buffer ([`MAX_WINDOW`] cap),
    /// percentile sort in place. The prior per-frame `Vec` clone was ~120
    /// allocations and dominated the alloc telemetry.
    pub fn show(&mut self, ctx: &loam_egui::egui::Context) {
        use loam_egui::egui;

        // Check toggle even when hidden; otherwise F3 couldn't reopen.
        if ctx.input(|i| i.key_pressed(self.toggle_key)) {
            self.visible = !self.visible;
        }
        if !self.visible {
            return;
        }

        // Cap the window at MAX_WINDOW so the stack buffers can't overflow.
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

        let cadence_mean = cadence.mean();
        let cadence_p99 = cadence.percentile(0.99);
        let frame_mean = frames_buf.mean();
        let frame_p99 = frames_buf.percentile(0.99);
        let idle_mean = idles.mean();
        let idle_p99 = idles.percentile(0.99);

        // Session-lifetime maxima: a freeze that aged out of the rolling
        // window still shows here, where a window-bounded max would lie.
        let cadence_max_ever = frame_trace::max_ever("between-frames");
        let idle_max_ever = frame_trace::max_ever("idle");
        let frame_max_ever = frame_trace::max_ever("frame");

        let fps = if cadence_mean.as_secs_f32() > 0.0 {
            1.0 / cadence_mean.as_secs_f32()
        } else {
            0.0
        };

        // `default_pos` (not `anchor`, which ignores drag) so `.movable`
        // works and egui persists the dragged offset.
        egui::Area::new(egui::Id::new("loam-perf-overlay"))
            .default_pos(perf_overlay_seat(ctx))
            .movable(true)
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style())
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(60, 60, 75)))
                    .show(ui, |ui| {
                        ui.set_min_width(OVERLAY_WIDTH);
                        let mono = egui::FontId::monospace(11.0);
                        let label_color = egui::Color32::from_rgb(180, 190, 200);
                        ui.label(
                            egui::RichText::new(format!("FPS    {fps:5.1}"))
                                .font(mono.clone())
                                .color(egui::Color32::from_rgb(220, 230, 240)),
                        );
                        // total (between-frames) should equal frame + idle in
                        // the long run; the gap is scope-uncovered redraw work.
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
                        // Alloc section (demo opted into CountingAllocator).
                        // alloc count is tracked separately from bytes because
                        // many tiny allocs cost more in wasm JS-interop than
                        // one large alloc of the same total size.
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
                                egui::RichText::new(format!("mean  {mean_count:>6} allocs/frame",))
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
                        // Heap section (Chromium only). Skipped when no
                        // samples (Firefox/native) rather than showing zeroes
                        // a reader might misread as "no allocations."
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

/// Max window size for the overlay's percentile stats; sizes the inline
/// stack array in `StackBuf`. 256 x 16 B/Duration = 4 KB per buffer, three
/// buffers = 12 KB, well inside the 1 MB main-thread stack. The cap keeps
/// the buffer fixed-size; `frame_trace`'s own capacity is independent and
/// may be larger, in which case the overlay silently uses this cap to
/// preserve its zero-alloc property.
pub const MAX_WINDOW: usize = 256;

/// Fixed-capacity stack sample buffer; zero-alloc replacement for a
/// per-frame `Vec<Duration>`. Push drops silently at [`MAX_WINDOW`].
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

    /// `q`-percentile (nearest-rank). Order-preserving on `self`: sorts a
    /// stack-local copy so the sparkline downstream keeps its time order.
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

fn draw_sparkline(ui: &mut loam_egui::egui::Ui, gaps: &[Duration]) {
    use loam_egui::egui;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(240.0, 36.0), egui::Sense::hover());
    if gaps.is_empty() {
        return;
    }
    let painter = ui.painter();
    painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(18, 18, 24));

    // Clamp the top at 50ms so a single 200ms outlier doesn't squash the
    // baseline; outliers draw to the top of the rect at full color.
    let y_max_ms = 50.0_f32;
    let y_for_ms = |ms: f32| {
        let clamped = ms.min(y_max_ms);
        rect.bottom() - (clamped / y_max_ms) * rect.height()
    };
    // Reference lines: 60fps (16.7ms) and 30fps (33.3ms).
    let ref_60 = y_for_ms(16.67);
    let ref_30 = y_for_ms(33.33);
    painter.line_segment(
        [
            egui::pos2(rect.left(), ref_60),
            egui::pos2(rect.right(), ref_60),
        ],
        egui::Stroke::new(0.5, egui::Color32::from_rgb(60, 100, 70)),
    );
    painter.line_segment(
        [
            egui::pos2(rect.left(), ref_30),
            egui::pos2(rect.right(), ref_30),
        ],
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

    // Pins the ns/us/ms/s unit boundaries against threshold drift.

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
        assert_eq!(truncate("abcdefghijklmnopqr", 18).len(), 18);
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
        let mut buf = StackBuf::new();
        for ms in 1..=10u64 {
            buf.push(Duration::from_millis(ms));
        }
        // floor(10 * 0.5) = 5 -> samples[5] = 6ms.
        assert_eq!(buf.percentile(0.5), Duration::from_millis(6));
        assert_eq!(buf.percentile(0.95), Duration::from_millis(10));
        // p99 clamps to len-1.
        assert_eq!(buf.percentile(0.99), Duration::from_millis(10));
    }

    #[test]
    fn stackbuf_percentile_is_order_preserving_on_self() {
        // Guards the sparkline's reliance on insertion order: a refactor that
        // sorted `self.samples` would show sorted bars, not time-ordered ones.
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

#[cfg(test)]
mod seat_tests {
    use super::*;
    use loam_egui::egui;

    /// Any bar taller than `OVERLAY_MARGIN` separates the two rects; an exact
    /// height keeps the assertion independent of font metrics.
    const BAR_HEIGHT: f32 = 64.0;

    const PANEL_WIDTH: f32 = 120.0;

    /// Wide enough that the seat's left clamp stays untaken.
    const VIEWPORT: egui::Vec2 = egui::vec2(1280.0, 800.0);

    /// Narrower than `PANEL_WIDTH + OVERLAY_MARGIN + OVERLAY_WIDTH`, so the
    /// band a side panel leaves cannot hold the readout and the clamp fires.
    const NARROW_VIEWPORT: egui::Vec2 = egui::vec2(200.0, 400.0);

    /// egui's fallback viewport is ~10000px wide, so a default `RawInput`
    /// leaves the clamp unreachable no matter what the panels do.
    fn viewport(size: egui::Vec2) -> egui::RawInput {
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
            ..Default::default()
        }
    }

    #[test]
    fn perf_overlay_boots_into_the_top_right_of_the_panel_band() {
        // One frame of history, so `show` gets past its "no samples" early
        // return. The store is thread-local and each test owns its thread.
        frame_trace::begin_frame();
        frame_trace::end_frame();

        let ctx = egui::Context::default();
        let mut overlay = PerfOverlay::new().always_visible();
        let mut bar_bottom = 0.0;
        let mut band = egui::Rect::NOTHING;
        let _ = ctx.run(viewport(VIEWPORT), |ctx| {
            bar_bottom = egui::TopBottomPanel::top("shell-menu-bar")
                .exact_height(BAR_HEIGHT)
                .show(ctx, |ui| {
                    ui.label("bar");
                })
                .response
                .rect
                .bottom();
            band = ctx.available_rect();
            overlay.show(ctx);
        });
        let rect = ctx
            .memory(|m| m.area_rect(egui::Id::new("loam-perf-overlay")))
            .expect("overlay area is registered once shown");
        assert!(
            rect.top() >= bar_bottom,
            "overlay top {} must clear the menu bar bottom {bar_bottom}",
            rect.top(),
        );
        // Halves rather than exact insets: egui constrains a placed area to
        // the viewport, so frame padding can shift the right edge inward.
        assert!(
            rect.center().x > band.center().x,
            "overlay center x {} must sit in the band's right half of {band:?}",
            rect.center().x,
        );
        assert!(
            rect.center().y < band.center().y,
            "overlay center y {} must sit in the band's top half of {band:?}",
            rect.center().y,
        );
    }

    #[test]
    fn perf_overlay_seat_insets_from_the_top_right_of_the_panel_band() {
        let ctx = egui::Context::default();
        let mut seat = None;
        let mut band = egui::Rect::NOTHING;
        let _ = ctx.run(viewport(VIEWPORT), |ctx| {
            egui::SidePanel::left("left")
                .exact_width(PANEL_WIDTH)
                .show(ctx, |ui| {
                    ui.label("left");
                });
            egui::SidePanel::right("right")
                .exact_width(PANEL_WIDTH)
                .show(ctx, |ui| {
                    ui.label("right");
                });
            band = ctx.available_rect();
            seat = Some(perf_overlay_seat(ctx));
        });
        let seat = seat.expect("run closure sets the seat");
        assert_eq!(
            seat.x,
            band.right() - OVERLAY_MARGIN - OVERLAY_WIDTH,
            "seat must inset from the band's right edge, not its left, in {band:?}",
        );
        assert_eq!(
            seat.y,
            band.top() + OVERLAY_MARGIN,
            "seat must inset from the band's top edge, not its bottom, in {band:?}",
        );
    }

    #[test]
    fn perf_overlay_seat_clamps_to_the_band_left_when_the_band_is_narrower_than_the_readout() {
        let ctx = egui::Context::default();
        let mut seat = None;
        let mut band = egui::Rect::NOTHING;
        let _ = ctx.run(viewport(NARROW_VIEWPORT), |ctx| {
            egui::SidePanel::left("left")
                .exact_width(PANEL_WIDTH)
                .show(ctx, |ui| {
                    ui.label("left");
                });
            band = ctx.available_rect();
            seat = Some(perf_overlay_seat(ctx));
        });
        let seat = seat.expect("run closure sets the seat");
        assert!(
            band.width() < OVERLAY_MARGIN + OVERLAY_WIDTH && band.left() > 0.0,
            "fixture must leave a band {band:?} too narrow for the readout and \
             offset from the viewport origin, or the clamp goes untested",
        );
        assert_eq!(
            seat.x,
            band.left(),
            "a band too narrow for the readout must seat it at the band's left \
             edge, not at the viewport origin or off-screen",
        );
    }
}
