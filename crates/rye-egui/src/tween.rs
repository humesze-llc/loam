//! Re-export of [`rye_anim`]. Animation moved to its own presentation-tier crate
//! (usable by camera/render, not just UI); this alias keeps existing
//! `rye_egui::tween` / `rye_egui::{Animated, ease_*}` paths working.

pub use rye_anim::*;
