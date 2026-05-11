//! `rye-egui`: integration glue between `rye-app` and the [egui] immediate-mode UI
//! library.
//!
//! Named for what it is, not what it abstracts. The crate wraps egui and provides a
//! wgpu paint pass + world-anchored label helper; if a future migration to a different
//! UI framework happens, this crate gets replaced rather than retargeted.
//!
//! ## Surface
//!
//! - [`UiIntegration`]: per-app egui state (Context, winit translator, wgpu renderer).
//!   Owned by `rye_app::Runner`; apps don't construct it directly.
//! - [`world_to_screen`]: project a world-space point to screen pixel coordinates via
//!   a camera + viewport. The cheap pattern for "egui label that follows a 3D object."
//! - [`BottomOverlay`]: floating bottom-anchored overlay panel with flicker-free
//!   animated size transitions. Solves the single-frame jump that egui's
//!   [`Area`](egui::Area) produces when content size changes drastically. The widget
//!   you reach for when building a game HUD that grows/shrinks with state.
//! - [`LinearIndicator`]: read-only horizontal scrub bar showing where a value sits in
//!   a 1D parameter range. Useful for "where am I in this parameter" debug HUDs (the
//!   `w` slice plane in a 4D viewer, current frame in a recorded sequence, etc.).
//! - [`Console`]: Quake-style developer console (drop-down overlay, command registry,
//!   scrollback, hotkey binds, tab autocomplete). Generic over a `Ctx` type so
//!   consuming crates choose what state commands operate on.
//!
//! [egui]: https://github.com/emilk/egui
//!
//! Apps interact with the UI by overriding `App::ui(&mut self, ctx, frame)` and writing
//! immediate-mode egui code:
//!
//! ```ignore
//! fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
//!     egui::Window::new("Settings").show(ctx, |ui| {
//!         ui.add(egui::Slider::new(&mut self.fov, 30.0..=120.0).text("FOV"));
//!         if ui.button("Reset").clicked() {
//!             self.reset();
//!         }
//!     });
//! }
//! ```
//!
//! ## Input gating
//!
//! egui consumes input it cares about (clicks on widgets, typing into a focused text
//! input). Gameplay code that reads the same WASD keys or mouse delta should gate on
//! `frame.ui_has_focus()` so a player typing into a settings field doesn't also fire
//! movement.
//!
//! ## Why egui (not iced or a from-scratch UI)
//!
//! Immediate mode matches `rye-app`'s "library-style composition, no ECS" pattern:
//! apps construct and call UI inside `App::ui`. egui integrates with wgpu directly via
//! `egui-wgpu`; no rendering glue beyond what this crate provides. Pure-Rust
//! dependency tree.
//!
//! ## What's deliberately out of scope
//!
//! - **3D-billboard egui** (egui rendered to texture, sampled in the 3D scene with
//!   ray-cast interaction). Possible but unnecessary for current use cases; the
//!   screen-space-with-world-anchoring pattern via [`world_to_screen`] covers labels
//!   and HUDs that follow 3D objects.
//! - **A full custom widget set on top of egui.** egui's defaults cover most needs;
//!   this crate only adds widgets that work around concrete egui limitations
//!   (currently: [`BottomOverlay`] for flicker-free anchored HUDs).

mod bivector_matrix;
pub mod console;
pub mod dnd;
mod integration;
mod linear_indicator;
pub mod media;
mod overlay;
mod slider_edit;
mod world;

pub use bivector_matrix::{bivector_matrix, cell_text as bivector_matrix_cell_text};
pub use console::{cmd, Command, Console, ConsoleWriter, HistoryLine, LineKind};
pub use integration::UiIntegration;
pub use linear_indicator::LinearIndicator;
pub use overlay::BottomOverlay;
pub use slider_edit::{slider_with_edit, SliderInteraction};
pub use world::world_to_screen;

// Re-export egui so apps depend on `rye-egui` only and the version pin lives in one
// place.
pub use egui;
