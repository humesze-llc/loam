//! Console command that toggles the surface present mode, the upstream side of
//! the [`crate::frame_pacing`] fps cap.
//!
//! ## Why this matters for fps
//!
//! The [`fps`] command alone cannot exceed the display refresh rate when the
//! surface uses `PresentMode::Fifo` (the default), because `present` blocks at
//! vsync. `vsync off` swaps the surface to `Mailbox` (or `Immediate` as
//! fallback) so the cap can drive cadence above the display rate; useful for
//! benchmarking, perf profiling, or chasing input latency.
//!
//! ## Subcommands
//!
//! - `vsync`: print the current present mode.
//! - `vsync on`: request `PresentMode::Fifo` on the runner's next redraw.
//! - `vsync off`: request the best non-Fifo mode the adapter advertised; the
//!   runner picks `Mailbox` first, falling back to `Immediate`, falling back
//!   to leaving the mode alone (typical browser case; surfaces there
//!   advertise only Fifo, so this command is effectively a no-op).
//!
//! ## Wiring (per demo)
//!
//! ```ignore
//! rye_app::vsync::register_command(&mut console);
//! ```
//!
//! [`fps`]: crate::fps

use rye_egui::{cmd, Console};

use crate::frame_pacing;

/// Register the `vsync` console command.
pub fn register_command<Ctx: 'static>(console: &mut Console<Ctx>) {
    console.register(
        cmd(
            "vsync",
            "show or set the surface present mode (on = Fifo, off = Mailbox/Immediate)",
            |args, _ctx: &mut Ctx, out| {
                match args.first().copied() {
                    None => {
                        // Without access to RenderDevice from the handler we
                        // can only report what was last requested. That's
                        // sufficient for the common workflow; the user
                        // typed it, they know what they asked for. A future
                        // refactor that plumbs rd through Ctx could read the
                        // applied mode here instead.
                        out.line("vsync: use 'vsync on' (Fifo) or 'vsync off' (Mailbox/Immediate)");
                    }
                    Some("on") => {
                        frame_pacing::request_vsync_on();
                        out.line("vsync: requested ON (Fifo); applies on next frame");
                    }
                    Some("off") => {
                        frame_pacing::request_vsync_off();
                        out.line(
                            "vsync: requested OFF (Mailbox preferred, Immediate fallback) \
                            ; applies on next frame",
                        );
                    }
                    Some(other) => {
                        out.line(format!(
                            "vsync: unknown subcommand '{other}' (try 'on' or 'off')"
                        ));
                    }
                }
                Ok(())
            },
        )
        .with_args(&[&["on", "off"]]),
    );
}

#[cfg(test)]
mod tests {
    use crate::frame_pacing;
    use crate::frame_pacing::TEST_LOCK;

    fn build_console() -> rye_egui::Console<()> {
        let mut c = rye_egui::Console::<()>::new();
        super::register_command(&mut c);
        c
    }

    fn clear_pending() {
        let _ = frame_pacing::take_pending_vsync();
    }

    #[test]
    fn vsync_on_records_pending_request() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_pending();
        let mut c = build_console();
        let mut ctx = ();
        c.execute("vsync on", &mut ctx);
        assert_eq!(
            frame_pacing::take_pending_vsync(),
            Some(true),
            "vsync on should queue a vsync-on request"
        );
    }

    #[test]
    fn vsync_off_records_pending_request() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_pending();
        let mut c = build_console();
        let mut ctx = ();
        c.execute("vsync off", &mut ctx);
        assert_eq!(
            frame_pacing::take_pending_vsync(),
            Some(false),
            "vsync off should queue a vsync-off request"
        );
    }

    #[test]
    fn vsync_bare_invocation_does_not_change_pending() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_pending();
        let mut c = build_console();
        let mut ctx = ();
        c.execute("vsync", &mut ctx);
        assert_eq!(
            frame_pacing::take_pending_vsync(),
            None,
            "bare `vsync` should print help, not request a transition"
        );
    }

    #[test]
    fn vsync_unknown_subcommand_does_not_change_pending() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_pending();
        let mut c = build_console();
        let mut ctx = ();
        c.execute("vsync foo", &mut ctx);
        assert_eq!(
            frame_pacing::take_pending_vsync(),
            None,
            "unknown subcommand should print error, not queue a transition"
        );
    }
}
