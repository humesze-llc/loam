//! Tracing-to-console bridge: every `tracing::info!` / `warn!` / `error!` event the
//! app emits gets formatted and pushed into a bounded ring buffer. A console command
//! (`log on|off|toggle`) controls whether the buffer mirrors into scrollback each
//! frame. The buffer is always filling regardless of mirror state, so toggling on
//! shows whatever events were emitted since startup (capped at `BUFFER_CAP`).
//!
//! ## What this does NOT capture
//!
//! - Raw `println!` / `eprintln!`. Those write to `fd 1` / `fd 2` directly; capturing
//!   them needs an OS-level pipe redirect. Use `tracing::info!(...)` instead.
//! - Events filtered out by `EnvFilter` (`RUST_LOG=...` or the runtime filter passed
//!   to [`RunConfig::log_filter`](crate::RunConfig)).
//!
//! ## Wiring (per demo)
//!
//! ```ignore
//! // In build_console:
//! rye_app::log::register_command(&mut c);
//!
//! // In App::ui, before console.ui:
//! rye_app::log::pump_into(&mut self.console);
//! self.console.ui(ctx, &mut self.demo);
//! ```
//!
//! The subscriber init happens automatically inside [`run_with_config`](crate::run_with_config).

use std::collections::VecDeque;
#[cfg(not(target_arch = "wasm32"))]
use std::fmt::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use rye_egui::{cmd, Console, HistoryLine};
#[cfg(not(target_arch = "wasm32"))]
use rye_egui::LineKind;
#[cfg(not(target_arch = "wasm32"))]
use tracing::field::{Field, Visit};
#[cfg(not(target_arch = "wasm32"))]
use tracing::Event;
#[cfg(not(target_arch = "wasm32"))]
use tracing_subscriber::layer::{Context, Layer};

/// Ring buffer cap. Sized so a busy capture session doesn't blow memory; older lines
/// drop off the front. Each entry is a short formatted string (~100 chars), so the
/// total footprint is ~200 KB at full capacity.
#[cfg(not(target_arch = "wasm32"))]
const BUFFER_CAP: usize = 2000;

static ENABLED: AtomicBool = AtomicBool::new(false);
static BUFFER: Mutex<VecDeque<HistoryLine>> = Mutex::new(VecDeque::new());

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

/// Drain pending log lines. When disabled, returns empty without touching the buffer
/// (so newly-enabled mirroring still shows recent history). When enabled, takes
/// everything queued and hands it to the caller.
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

/// Register the `log` console command. Two independent toggles:
///
/// - **`log [on|off|toggle]`** controls **tracing -> scrollback**: when on, any
///   `tracing::info!` / `warn!` / `error!` event the app emits also appears in
///   the in-canvas console scrollback. Useful for surfacing background events
///   (capture status, hot-reload notifications, framework warnings) where the
///   user has the console open.
/// - **`log echo [on|off|toggle]`** controls **scrollback -> browser**: when
///   on, every line added to the in-canvas scrollback (command output, user
///   prompts, error lines, even the `log on`-mirrored tracing events) also
///   echoes to the browser DevTools console via `web_sys::console::log_1`.
///   wasm32 only; native does nothing because stderr / stdout already covers
///   the same need.
///
/// Architectural note: the two directions deliberately use different
/// transports (tracing for in, raw `console.log` for out). Routing both
/// through tracing would create a feedback loop: a tracing event would land
/// in the scrollback, get re-emitted to tracing, land again, ad infinitum.
/// The asymmetric transport is the simplest invariant that breaks the loop
/// without per-line origin tagging.
pub fn register_command<Ctx: 'static>(console: &mut Console<Ctx>) {
    console.register(
        cmd(
            "log",
            "mirror tracing events into the scrollback (`log [on|off|toggle]`) \
             or echo scrollback to the browser console (`log echo [on|off|toggle]`)",
            |args, _ctx: &mut Ctx, out| {
                // Distinguish the `echo` subcommand from the legacy on/off/toggle
                // args. Order matters: `log echo` matches the subcommand path
                // and the second arg disambiguates within it.
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
                        // Bare `log echo` or `log echo toggle` flips the flag.
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
                    // On native the echo state has no surface to mirror to;
                    // flag the no-op so the user knows.
                    #[cfg(not(target_arch = "wasm32"))]
                    out.line(
                        "  (note: `log echo` is a no-op on native; the browser-console \
                         path is wasm32-only)",
                    );
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
        .with_args(&[&["on", "off", "toggle", "echo"], &["on", "off", "toggle"]]),
    );
}

// ---------------------------------------------------------------------------
// Tracing Layer
// ---------------------------------------------------------------------------
//
// The layer + its supporting visitors are native-only: on wasm32 we route tracing
// events through `tracing-wasm` (which writes directly to the browser console with
// matching severity levels) and never install this layer, so gating the items keeps
// the wasm build warning-free.

/// Tracing subscriber layer that pushes formatted events into [`BUFFER`].
///
/// Installed automatically by [`crate::run_with_config`]. Always captures; the mirror
/// to console scrollback is gated by [`ENABLED`].
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

/// Extracts the event's `message` field (the printf-style payload of
/// `tracing::info!("text {x}")`) and any other key=value fields into a one-liner
/// suitable for a scrollback row.
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
