//! Console command that reads / writes the target framerate via
//! [`crate::frame_pacing`]. Sibling to [`crate::trace`] in structure.
//!
//! ## Subcommands
//!
//! - `fps`: print the current target.
//! - `fps <n>`: set the target to `n` frames per second. Accepts integers and
//!   floats; `n` must be in `(0, 1000]`.
//! - `fps unlimited` (alias: `off`, `0`): remove the cap entirely. On native
//!   the surface's `PresentMode` (vsync) is the upper bound; on wasm the
//!   browser's `requestAnimationFrame` cadence remains the upper bound.
//!
//! ## Wiring (per demo)
//!
//! ```ignore
//! loam_app::fps::register_command(&mut console);
//! ```

use loam_egui::{cmd, Console};

use crate::frame_pacing;

/// Maximum accepted fps. Above this we reject the input so a stray `fps 999999`
/// doesn't silently make the cap a no-op. 1000 fps (1 ms period) is well past
/// any practical display refresh rate.
const MAX_ACCEPTED_FPS: f32 = 1000.0;

fn print_current(out: &mut loam_egui::ConsoleWriter) {
    let f = frame_pacing::target_fps();
    if f <= 0.0 {
        out.line("fps: unlimited (uncapped; surface/vsync or browser RAF is the upper bound)");
    } else {
        out.line(format!("fps: target {f:.1}"));
    }
}

/// Register the `fps` console command.
pub fn register_command<Ctx: 'static>(console: &mut Console<Ctx>) {
    console.register(
        cmd(
            "fps",
            "show or set the target framerate (default 60; use 'unlimited' to remove the cap)",
            |args, _ctx: &mut Ctx, out| {
                match args.first().copied() {
                    None => print_current(out),
                    Some("unlimited") | Some("off") | Some("0") => {
                        frame_pacing::set_target_fps(0.0);
                        out.line("fps: unlimited (cap removed)");
                    }
                    Some(s) => match s.parse::<f32>() {
                        Ok(f) if f > 0.0 && f <= MAX_ACCEPTED_FPS => {
                            frame_pacing::set_target_fps(f);
                            out.line(format!("fps: target set to {f:.1}"));
                        }
                        _ => {
                            out.line(format!(
                                "usage: fps [<n> | unlimited]  (n in (0, {MAX_ACCEPTED_FPS:.0}])"
                            ));
                        }
                    },
                }
                Ok(())
            },
        )
        .with_args(&[&["unlimited", "off", "30", "60", "120", "144", "240"]]),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_pacing;
    // Tests touch the process-global `frame_pacing` atomic. Default cargo
    // test parallelism would race the writes against each others' reads;
    // every test in this crate that touches the pacing atomics shares
    // `frame_pacing::TEST_LOCK`.
    use crate::frame_pacing::TEST_LOCK;

    fn build_console() -> loam_egui::Console<()> {
        let mut c = loam_egui::Console::<()>::new();
        register_command(&mut c);
        c
    }

    #[test]
    fn fps_numeric_arg_updates_target() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut c = build_console();
        let mut ctx = ();
        c.execute("fps 30", &mut ctx);
        let target = frame_pacing::target_fps();
        assert!(
            (target - 30.0).abs() < 0.1,
            "fps 30 should set target near 30, got {target}"
        );
        frame_pacing::set_target_fps(60.0);
    }

    #[test]
    fn fps_unlimited_removes_cap() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut c = build_console();
        let mut ctx = ();
        c.execute("fps unlimited", &mut ctx);
        assert_eq!(frame_pacing::target_fps(), 0.0);
        assert!(frame_pacing::target_period().is_none());
        frame_pacing::set_target_fps(60.0);
    }

    #[test]
    fn fps_off_alias_matches_unlimited() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut c = build_console();
        let mut ctx = ();
        c.execute("fps off", &mut ctx);
        assert_eq!(frame_pacing::target_fps(), 0.0);
        frame_pacing::set_target_fps(60.0);
    }

    #[test]
    fn fps_zero_alias_matches_unlimited() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut c = build_console();
        let mut ctx = ();
        c.execute("fps 0", &mut ctx);
        assert_eq!(frame_pacing::target_fps(), 0.0);
        frame_pacing::set_target_fps(60.0);
    }

    #[test]
    fn fps_out_of_range_does_not_change_target() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut c = build_console();
        let mut ctx = ();
        let before = frame_pacing::target_fps();
        // Past the MAX_ACCEPTED_FPS guard; the handler prints a usage line
        // but does not poke the atomic.
        c.execute("fps 100000", &mut ctx);
        let after = frame_pacing::target_fps();
        assert_eq!(before, after, "out-of-range input should be a no-op");
        frame_pacing::set_target_fps(60.0);
    }
}
