//! Active-set rotation mode: six basis-plane checkboxes drive the
//! angular velocity. Sum-of-bivectors is commutative, so toggle order
//! doesn't matter; only the active set does.
//!
//! This module owns:
//!
//! - [`combo_name`]: pretty-name for recognizable active-set
//!   combinations (single planes, isoclinics, all-w, etc.). Used by
//!   the formula popup in `ui.rs` to give a readable name to the
//!   current rotation when in Active mode.
//! - The active-mode rendering methods on [`Demo`]:
//!   `render_active_mode` and `render_plane_slider_cell`.

use rye_app::egui;
use rye_math::Plane4;

use crate::consts::CONTROL_H;

/// Wrap a degree value into `(-720, 720]` so the slider handle stays
/// inside the widget's clamping range while continuous spin advances
/// the raw angle past one period. `1440` is the full two-cycle range
/// (`+720` minus `-720`); `rem_euclid` keeps the result non-negative,
/// then we re-center.
fn wrap_slider_deg(d: f32) -> f32 {
    let m = d.rem_euclid(1440.0);
    if m > 720.0 {
        m - 1440.0
    } else {
        m
    }
}
use crate::state::Demo;

/// Name a recognizable combination of active planes. Indices match
/// `Plane4::ALL`: `0=xy 1=xz 2=xw 3=yz 4=yw 5=zw`. Order-independent,
/// only the active *set* matters.
///
/// Curated entries cover common 4D-geometry classics: single
/// stretches, the three perpendicular-pair isoclinics (the only
/// commuting bivector pairs in 4D, related to left/right Hopf
/// maps), pure-3D rotations, and the famous "all w-planes"
/// composition that drives the cross-section through its main-
/// diagonal extreme.
pub(crate) fn combo_name(active: &[bool; 6]) -> Option<&'static str> {
    let mut mask = 0u8;
    for (i, &on) in active.iter().enumerate() {
        if on {
            mask |= 1 << i;
        }
    }
    let xy = 1 << 0;
    let xz = 1 << 1;
    let xw = 1 << 2;
    let yz = 1 << 3;
    let yw = 1 << 4;
    let zw = 1 << 5;
    let m = mask;
    Some(match m {
        0 => return None,
        x if x == xw => "x-into-w stretch",
        x if x == yw => "y-into-w stretch",
        x if x == zw => "z-into-w stretch",
        x if x == xy => "xy spin (3D only)",
        x if x == xz => "xz spin (3D only)",
        x if x == yz => "yz spin (3D only)",
        x if x == xw | yz => "isoclinic xw+yz",
        x if x == xz | yw => "isoclinic xz+yw",
        x if x == xy | zw => "isoclinic xy+zw",
        x if x == xy | xz | yz => "full 3D spin",
        x if x == xw | yw | zw => "main-diagonal spin (all-w)",
        x if x == xy | xz | xw | yz | yw | zw => "chaotic SO(4) drift",
        _ => "compound",
    })
}

impl Demo {
    /// Active body: 3-per-row 2-row grid of
    /// `[checkbox][label][slider][value]`. Pinned widths so
    /// columns align across rows. Each value cell is right-click
    /// editable via the shared `slider_with_edit` helper.
    pub(crate) fn render_active_mode(&mut self, ui: &mut egui::Ui) {
        const TOP_ROW: [usize; 3] = [0, 1, 3]; // xy, xz, yz
        const BOTTOM_ROW: [usize; 3] = [2, 4, 5]; // xw, yw, zw

        const CELL_INNER_SPACING: f32 = 4.0;
        const CHECKBOX_W: f32 = 18.0;
        const LABEL_W: f32 = 22.0;
        const VALUE_W: f32 = 56.0;
        const ROW_GAP: f32 = 6.0;

        let total_w = ui.available_width();
        let cell_w = ((total_w - 2.0 * ROW_GAP) / 3.0).floor();
        let slider_w =
            (cell_w - CHECKBOX_W - LABEL_W - VALUE_W - 3.0 * CELL_INNER_SPACING).max(40.0);

        for plane_indices in [TOP_ROW, BOTTOM_ROW] {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = ROW_GAP;
                for &i in &plane_indices {
                    ui.allocate_ui_with_layout(
                        egui::vec2(cell_w, CONTROL_H),
                        egui::Layout::left_to_right(egui::Align::Center),
                        |ui| {
                            ui.spacing_mut().item_spacing.x = CELL_INNER_SPACING;
                            ui.spacing_mut().slider_width = slider_w;
                            self.render_plane_slider_cell(
                                ui, i, CHECKBOX_W, LABEL_W, slider_w, VALUE_W,
                            );
                        },
                    );
                }
            });
        }
    }

    /// One plane cell. All component widths pinned by the caller
    /// so the cell renders identically regardless of which row
    /// or column it's in.
    ///
    /// The slider reads/writes `base_angles[plane_idx]` via the
    /// displayed-angle formula (`base + spin_contribution`), so each
    /// slider only mutates its own plane's factor in the active-mode
    /// rotor product. Independent sliders, no log/exp round-trips,
    /// no cross-slider redistribution.
    pub(crate) fn render_plane_slider_cell(
        &mut self,
        ui: &mut egui::Ui,
        plane_idx: usize,
        checkbox_w: f32,
        label_w: f32,
        slider_w: f32,
        value_w: f32,
    ) {
        let plane = Plane4::ALL[plane_idx];
        // Slider value is the *displayed* angle (base + spin contribution),
        // wrapped into (-720°, 720°]. The wrap is purely for the slider
        // handle: during continuous spin the raw displayed angle grows
        // unboundedly, which would pin the slider against its 720° clamp
        // and stop showing motion. Wrapping into the period keeps the
        // handle moving cyclically. `active_rotor()` uses the RAW
        // displayed angle (no wrap), so the rotor itself is identical
        // regardless of how this value is normalized for display --
        // `exp(plane * (x + 2π·k))` is the same rotor for any integer k
        // when the plane is a simple unit bivector.
        let raw_deg = self.active_displayed_angle(plane_idx).to_degrees();
        let mut deg = wrap_slider_deg(raw_deg);
        ui.add_sized(
            [checkbox_w, 18.0],
            egui::Checkbox::new(&mut self.active[plane_idx], ""),
        );
        ui.add_sized(
            [label_w, 18.0],
            egui::Label::new(egui::RichText::new(plane.label()).monospace()),
        );
        let slider = egui::Slider::new(&mut deg, -720.0..=720.0)
            .show_value(false)
            .smart_aim(false)
            .clamping(egui::SliderClamping::Always);
        let slider_resp = ui.add_sized([slider_w, 18.0], slider);
        let formatted = format!("{deg:>+6.1}°");
        let mut popup_changed = false;
        ui.allocate_ui_with_layout(
            egui::vec2(value_w, 18.0),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                let label_resp = ui.add(
                    egui::Button::new(egui::RichText::new(formatted).monospace())
                        .frame(false)
                        .small(),
                );
                label_resp
                    .on_hover_cursor(egui::CursorIcon::ContextMenu)
                    .on_hover_text("Right-click to edit value")
                    .context_menu(|ui| {
                        let drag_resp = ui.add(
                            egui::DragValue::new(&mut deg)
                                .range(-720.0..=720.0)
                                .suffix("°")
                                .fixed_decimals(1),
                        );
                        if drag_resp.changed() {
                            popup_changed = true;
                        }
                    });
            },
        );
        if slider_resp.changed() || popup_changed {
            // Solve for base_angles so the displayed angle matches the
            // slider's new position:
            //     displayed = base + spin_contribution
            //  => base = displayed - spin_contribution
            let target_rad = deg.to_radians();
            let spin_contribution = if self.active[plane_idx] {
                self.rot_time * crate::consts::BASE_ROTATION_RATE
            } else {
                0.0
            };
            self.base_angles[plane_idx] = target_rad - spin_contribution;
            self.rot_state = self.active_rotor();
            self.write_all(self.rot_state);
        }
    }
}
