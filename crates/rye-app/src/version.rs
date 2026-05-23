//! Console command that prints the demo's build identity. Sibling to
//! [`crate::trace`], [`crate::fps`], [`crate::vsync`] in structure.
//!
//! ## What it shows
//!
//! - Crate name (passed by the caller; the demo's binary name).
//! - Crate version (from `CARGO_PKG_VERSION` at compile time).
//! - Git short hash + dirty marker (from the demo's `build.rs` baking
//!   `BUILD_HASH` and `BUILD_DIRTY` env vars; if a demo's `build.rs`
//!   doesn't bake these, the demo passes empty strings and the output
//!   degrades gracefully).
//!
//! Example output: `polytope_playground v0.1.0 (a3f9c1d2+dirty)`.
//!
//! ## Wiring (per demo)
//!
//! ```ignore
//! rye_app::version::register_command(
//!     &mut console,
//!     env!("CARGO_PKG_NAME"),
//!     env!("CARGO_PKG_VERSION"),
//!     env!("BUILD_HASH"),
//!     env!("BUILD_DIRTY"),
//! );
//! ```

use rye_egui::{cmd, Console};

/// Register the `version` console command for a demo.
///
/// Pass `env!()` strings from the demo's own crate so each demo reports
/// its own name + version + build hash. The hash and dirty fields can be
/// empty if the demo doesn't have a `build.rs` baking those env vars; the
/// output collapses to just the crate name + version.
pub fn register_command<Ctx: 'static>(
    console: &mut Console<Ctx>,
    crate_name: &'static str,
    crate_version: &'static str,
    build_hash: &'static str,
    build_dirty: &'static str,
) {
    console.register(
        cmd(
            "version",
            "show the demo's crate version + git build hash",
            move |_args, _ctx: &mut Ctx, out| {
                let line = if build_hash.is_empty() {
                    format!("{crate_name} v{crate_version}")
                } else {
                    format!(
                        "{crate_name} v{crate_version} ({build_hash}{build_dirty})"
                    )
                };
                out.line(line);
                Ok(())
            },
        ),
    );
}
