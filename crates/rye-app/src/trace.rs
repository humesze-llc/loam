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
