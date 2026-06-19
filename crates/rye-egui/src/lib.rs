//! `rye-egui`: integration glue between `rye-app` and the [egui] immediate-mode UI
//! library.
//!
//! Wraps egui with a wgpu paint pass plus a few widgets; a future UI-framework
//! migration replaces this crate rather than retargeting it.
//!
//! ## Surface
//!
//! - [`UiIntegration`]: per-app egui state, owned by `rye_app::Runner`.
//! - [`world_to_screen`]: project a world-space point to screen pixels via a
//!   camera + viewport (an egui label that follows a 3D object).
//! - [`BottomOverlay`]: bottom-anchored HUD panel with flicker-free animated
//!   resize, avoiding the single-frame jump egui's [`Area`](egui::Area)
//!   produces on a large content-size change.
//! - [`LinearIndicator`]: read-only scrub bar showing where a value sits in a
//!   1D range.
//! - [`Console`]: Quake-style developer console (command registry, scrollback,
//!   hotkey binds, tab autocomplete), generic over a `Ctx` type.
//!
//! [egui]: https://github.com/emilk/egui
//!
//! ## Input gating
//!
//! Gameplay code reading WASD or mouse delta should gate on
//! `frame.ui_has_focus()` so typing into a settings field doesn't also fire
//! movement.

mod bivector_matrix;
pub mod console;
pub mod dnd;
mod floating;
mod integration;
mod linear_indicator;
pub mod media;
mod overlay;
mod slider_edit;
pub mod tween;
mod world;

pub use bivector_matrix::{bivector_matrix, cell_text as bivector_matrix_cell_text};
pub use console::{
    cmd, console_echo_enabled, set_console_echo, subcommands, Command, Console, ConsoleWriter,
    HistoryLine, LineKind, SubcommandSet,
};
pub use floating::{
    callout, floating_panel, floating_panel_builder, sticky_menu, CalloutState,
    FloatingPanelBuilder,
};
pub use integration::UiIntegration;
pub use linear_indicator::LinearIndicator;
pub use overlay::BottomOverlay;
pub use slider_edit::{slider_with_edit, SliderInteraction};
pub use tween::{ease_in_out_cubic, ease_out_cubic, Animated};
pub use world::world_to_screen;

// Re-export egui so the version pin lives in one place.
pub use egui;
