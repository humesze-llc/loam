//! Tracing-to-console bridge: each `tracing::info!`/`warn!`/`error!` event is
//! formatted into a bounded ring buffer. `log on|off|toggle` controls whether
//! the buffer mirrors into scrollback; the buffer always fills, so toggling on
//! shows history since startup (capped at `BUFFER_CAP`).
//!
//! Does not capture raw `println!`/`eprintln!` (those need an fd redirect) or
//! events dropped by `EnvFilter` /
//! [`RunConfig::log_filter`](crate::RunConfig).
//!
//! ```ignore
//! rye_app::log::register_command(&mut c);     // build_console
//! rye_app::log::pump_into(&mut self.console); // App::ui, before console.ui
//! ```
//!
//! Subscriber init is automatic in
//! [`run_with_config`](crate::run_with_config).

use std::collections::VecDeque;
#[cfg(not(target_arch = "wasm32"))]
use std::fmt::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

#[cfg(not(target_arch = "wasm32"))]
use rye_egui::LineKind;
use rye_egui::{cmd, Console, HistoryLine};
#[cfg(not(target_arch = "wasm32"))]
use tracing::field::{Field, Visit};
#[cfg(not(target_arch = "wasm32"))]
use tracing::Event;
#[cfg(not(target_arch = "wasm32"))]
use tracing_subscriber::layer::{Context, Layer};

/// Ring buffer cap (oldest lines drop). ~100 chars/entry, so ~200 KB full.
#[cfg(not(target_arch = "wasm32"))]
const BUFFER_CAP: usize = 2000;

static ENABLED: AtomicBool = AtomicBool::new(false);
static BUFFER: Mutex<VecDeque<HistoryLine>> = Mutex::new(VecDeque::new());

/// Per-WindowEvent log toggle: when on, the runner emits a `tracing::info!`
/// per dispatched WindowEvent for spike-correlation (cursor-moves filtered
/// as noise). Global static, not a `Runner` field, because the console
/// closure can't reach the runner.
static EVENTS_ENABLED: AtomicBool = AtomicBool::new(false);

/// Read by the runner's `window_event` handler; toggle via
/// `log events on|off|toggle`.
pub fn events_enabled() -> bool {
    EVENTS_ENABLED.load(Ordering::Relaxed)
}

/// Set the per-event log state explicitly.
pub fn set_events_enabled(b: bool) {
    EVENTS_ENABLED.store(b, Ordering::Relaxed);
}

/// Toggle and return the new state.
pub fn toggle_events() -> bool {
    let new = !events_enabled();
    set_events_enabled(new);
    new
}

/// True when log events should mirror into the console scrollback this frame.
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub fn set_enabled(b: bool) {
    ENABLED.store(b, Ordering::Relaxed);
}

/// Flip the enabled flag. Returns the new state.
pub fn toggle() -> bool {
    let new = !enabled();
    set_enabled(new);
    new
}

/// Drain pending log lines. Disabled returns empty without draining, so
/// newly-enabled mirroring still shows recent history.
pub fn drain() -> Vec<HistoryLine> {
    if !enabled() {
        return Vec::new();
    }
    let Ok(mut buf) = BUFFER.lock() else {
        return Vec::new();
    };
    buf.drain(..).collect()
}

/// Convenience: pump pending log lines into a console's scrollback. Call once per
/// frame, before `Console::ui`.
pub fn pump_into<Ctx: 'static>(console: &mut Console<Ctx>) {
    for line in drain() {
        console.write(line);
    }
}

/// Register the `log` console command. Independent toggles:
///
/// - **`log [on|off|toggle]`**: tracing -> scrollback.
/// - **`log echo [on|off|toggle]`**: scrollback -> browser DevTools console
///   (wasm32 only; native already has stderr/stdout).
/// - **`log events [on|off|toggle]`**: per-WindowEvent tracing.
///
/// The two directions use different transports (tracing in, raw
/// `console.log` out) to avoid a scrollback -> tracing -> scrollback
/// feedback loop.
pub fn register_command<Ctx: 'static>(console: &mut Console<Ctx>) {
    console.register(
        cmd(
            "log",
            "mirror tracing events into the scrollback (`log [on|off|toggle]`) \
             or echo scrollback to the browser console (`log echo [on|off|toggle]`)",
            |args, _ctx: &mut Ctx, out| {
                if args.first().copied() == Some("echo") {
                    let new = match args.get(1).copied() {
                        Some("on") => {
                            rye_egui::set_console_echo(true);
                            true
                        }
                        Some("off") => {
                            rye_egui::set_console_echo(false);
                            false
                        }
                        _ => {
                            let next = !rye_egui::console_echo_enabled();
                            rye_egui::set_console_echo(next);
                            next
                        }
                    };
                    out.line(if new {
                        "log echo (scrollback -> browser console): on"
                    } else {
                        "log echo (scrollback -> browser console): off"
                    });
                    #[cfg(not(target_arch = "wasm32"))]
                    out.line(
                        "  (note: `log echo` is a no-op on native; the browser-console \
                         path is wasm32-only)",
                    );
                    return Ok(());
                }
                if args.first().copied() == Some("events") {
                    let new = match args.get(1).copied() {
                        Some("on") => {
                            set_events_enabled(true);
                            true
                        }
                        Some("off") => {
                            set_events_enabled(false);
                            false
                        }
                        _ => toggle_events(),
                    };
                    out.line(if new {
                        "log events (per-WindowEvent tracing): on"
                    } else {
                        "log events (per-WindowEvent tracing): off"
                    });
                    if new {
                        out.line(
                            "  (cursor-move events are suppressed; non-cursor events \
                             emit one tracing::info! each as the runner dispatches them)",
                        );
                    }
                    return Ok(());
                }
                let new = match args.first().copied() {
                    Some("on") => {
                        set_enabled(true);
                        true
                    }
                    Some("off") => {
                        set_enabled(false);
                        false
                    }
                    _ => toggle(),
                };
                out.line(if new {
                    "log mirror (tracing -> scrollback): on"
                } else {
                    "log mirror (tracing -> scrollback): off"
                });
                Ok(())
            },
        )
        .with_args(&[
            &["on", "off", "toggle", "echo", "events"],
            &["on", "off", "toggle"],
        ]),
    );
}

// ---------------------------------------------------------------------------
// Tracing Layer
// ---------------------------------------------------------------------------
//
// Native-only: wasm32 routes events through `tracing-wasm` and never installs
// this layer, so gating the items keeps the wasm build warning-free.

/// Tracing layer that pushes formatted events into [`BUFFER`]. Installed by
/// [`crate::run_with_config`]; always captures, mirror gated by [`ENABLED`].
#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct ConsoleLayer;

#[cfg(not(target_arch = "wasm32"))]
impl<S> Layer<S> for ConsoleLayer
where
    S: tracing::Subscriber,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let meta = event.metadata();
        let level = *meta.level();

        let mut visitor = FieldVisitor::default();
        event.record(&mut visitor);

        let text = if visitor.message.is_empty() {
            format!("[{level}] {}", meta.target())
        } else if visitor.fields.is_empty() {
            format!("[{level}] {}", visitor.message)
        } else {
            format!("[{level}] {} ({})", visitor.message, visitor.fields)
        };
        let kind = match level {
            tracing::Level::ERROR => LineKind::Error,
            tracing::Level::WARN => LineKind::Error,
            tracing::Level::INFO => LineKind::System,
            tracing::Level::DEBUG | tracing::Level::TRACE => LineKind::Output,
        };

        push(HistoryLine { kind, text });
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn push(line: HistoryLine) {
    let Ok(mut buf) = BUFFER.lock() else { return };
    buf.push_back(line);
    while buf.len() > BUFFER_CAP {
        buf.pop_front();
    }
}

/// Splits an event's `message` field from its other key=value fields into a
/// one-line scrollback row.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct FieldVisitor {
    message: String,
    fields: String,
}

#[cfg(not(target_arch = "wasm32"))]
impl Visit for FieldVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            let _ = write!(self.message, "{value:?}");
        } else {
            if !self.fields.is_empty() {
                self.fields.push_str(", ");
            }
            let _ = write!(self.fields, "{}={value:?}", field.name());
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message.push_str(value);
        } else {
            if !self.fields.is_empty() {
                self.fields.push_str(", ");
            }
            let _ = write!(self.fields, "{}={value:?}", field.name());
        }
    }
}
