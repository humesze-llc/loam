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

/// Per-WindowEvent log toggle. When on, the runner emits a `tracing::info!`
/// for every meaningful WindowEvent it dispatches to `on_event`. Used for
/// spike-correlation: if the perf overlay reports a 500ms spike, the log
/// will have any input events that preceded it within the same frame
/// window, which narrows the cause (resize event? focus change? specific
/// key press?). Cursor-moves are filtered because they fire at 60Hz+ and
/// would drown out the signal.
///
/// Architectural note: this is a process-global static rather than a field
/// on `Runner` because the toggle is set from inside a `Console` command
/// closure that doesn't have access to the runner. The trade-off is that
/// multi-App tests with their own runners would share the flag, which
/// doesn't matter for the actual use case (one demo per process).
static EVENTS_ENABLED: AtomicBool = AtomicBool::new(false);

/// Read by the runner's `window_event` handler. Demos shouldn't need this
/// directly; toggle via the `log events on|off|toggle` console command.
pub fn events_enabled() -> bool {
    EVENTS_ENABLED.load(Ordering::Relaxed)
}

/// Set the per-event log state explicitly. Same caveat as `set_enabled`:
/// race-prone if toggled from multiple threads, fine for a console
/// command's single-threaded toggle.
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
                // Three subcommand families:
                //
                //   log [on|off|toggle]            -> tracing  -> scrollback
                //   log echo  [on|off|toggle]      -> scrollback -> browser console
                //   log events [on|off|toggle]     -> per-WindowEvent tracing::info!
                //
                // Architectural note: the three are independent toggles so
                // diagnostic combinations don't fight each other. e.g. during
                // a spike investigation you'd typically run `log events on +
                // log echo on` to capture event timestamps + scrollback in the
                // browser DevTools console; for steady-state debug `log on`
                // alone mirrors tracing to scrollback without echo-spam.
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
