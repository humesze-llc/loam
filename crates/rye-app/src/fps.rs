//! Console command that reads / writes the target framerate via
//! [`crate::frame_pacing`]. Sibling to [`crate::trace`] in structure.
//!
//! ## Subcommands
//!
//! - `fps` — print the current target.
//! - `fps <n>` — set the target to `n` frames per second. Accepts integers and
//!   floats; `n` must be in `(0, 1000]`.
//! - `fps unlimited` (alias: `off`, `0`) — remove the cap entirely. On native
//!   the surface's `PresentMode` (vsync) is the upper bound; on wasm the
//!   browser's `requestAnimationFrame` cadence remains the upper bound.
//!
//! ## Wiring (per demo)
//!
//! ```ignore
//! rye_app::fps::register_command(&mut console);
//! ```

use rye_egui::{cmd, Console};

use crate::frame_pacing;

/// Maximum accepted fps. Above this we reject the input so a stray `fps 999999`
/// doesn't silently make the cap a no-op. 1000 fps (1 ms period) is well past
/// any practical display refresh rate.
const MAX_ACCEPTED_FPS: f32 = 1000.0;

fn print_current(out: &mut rye_egui::ConsoleWriter) {
    let f = frame_pacing::target_fps();
    if f <= 0.0 {
        out.line("fps: unlimited (uncapped — surface/vsync or browser RAF is the upper bound)");
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
