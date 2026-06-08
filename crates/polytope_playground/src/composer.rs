//! Composer mode: typed-formula parser + sequence-of-terms UI.
//!
//! The user builds a `Vec<RotorTerm>` by typing a single-term
//! expression (`90° (xy + zw)`) or appending basis-plane chips into a
//! draft. Terms render as draggable cards that reorder via DnD, with
//! cross-term plane migration via plane pills. The scrub slider
//! projects `log(rot_state)` onto the seq's net bivector direction so
//! scrubbing preserves any perpendicular rotation already applied.

use rye_app::egui;
use rye_egui::{
    dnd::{
        apply_drop_pre_pass as dnd_apply_drop_pre_pass,
        drag_source_collapsing as dnd_drag_source_collapsing, drop_target_idx, force_opaque_active,
        make_room_gap, pickup_t as drag_pickup_t,
    },
    slider_with_edit,
};
use rye_math::{Bivector, Plane4, Rotor};

use crate::consts::{CARD_ITEM_SPACING_X, CONTROL_H, MINI_BUTTON_W};
use crate::state::{render_plane_sum, DeferredAction, Demo, DragPayload, RotorTerm};

// ---------------------------------------------------------------------------
// Formula parser
// ---------------------------------------------------------------------------

/// Parse one term like `90° (xy + zw)`, `90 xy`, `0.5 rad xy` into a
/// [`RotorTerm`]. Degrees default; `rad` overrides. The `*` / `·`
/// separator and outer parens are optional. One expression per call;
/// rotor multiplication across terms lives in the seq.
pub(crate) fn parse_formula_term(input: &str) -> Result<RotorTerm, String> {
    let normalized = input.trim().replace('·', "*").replace('°', "deg ");
    let s = normalized.trim();
    if s.is_empty() {
        return Err("empty input".into());
    }
    let (scalar, rest) = peel_scalar(s)?;
    let bivec_str = rest.trim();
    let inner = if bivec_str.starts_with('(') && bivec_str.ends_with(')') {
        bivec_str[1..bivec_str.len() - 1].trim()
    } else {
        bivec_str
    };
    if inner.is_empty() {
        return Err("missing bivector after scalar".into());
    }
    let mut planes = Vec::new();
    for part in inner.split('+') {
        let p = part.trim();
        if p.is_empty() {
            return Err("empty plane between '+'".into());
        }
        planes.push(parse_plane(p)?);
    }
    Ok(RotorTerm { planes, scalar })
}

fn peel_scalar(s: &str) -> Result<(Option<f32>, &str), String> {
    let bytes = s.as_bytes();
    let mut i = 0;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        i += 1;
    }
    let digits_start = i;
    while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
        i += 1;
    }
    if i == digits_start {
        return Ok((None, s));
    }
    let num_str = &s[..i];
    let value: f32 = num_str
        .parse()
        .map_err(|_| format!("not a number: `{num_str}`"))?;
    let mut tail = s[i..].trim_start();
    let radians = if let Some(rest) = tail.strip_prefix("rad") {
        tail = rest.trim_start();
        value
    } else if let Some(rest) = tail.strip_prefix("deg") {
        tail = rest.trim_start();
        value.to_radians()
    } else {
        value.to_radians()
    };
    if let Some(rest) = tail.strip_prefix('*') {
        tail = rest.trim_start();
    }
    Ok((Some(radians), tail))
}

fn parse_plane(s: &str) -> Result<Plane4, String> {
    match s {
        "xy" => Ok(Plane4::Xy),
        "xz" => Ok(Plane4::Xz),
        "xw" => Ok(Plane4::Xw),
        "yz" => Ok(Plane4::Yz),
        "yw" => Ok(Plane4::Yw),
        "zw" => Ok(Plane4::Zw),
        _ => Err(format!("unknown plane `{s}` (expected xy/xz/xw/yz/yw/zw)")),
    }
}

// ---------------------------------------------------------------------------
// Composer-mode rendering
// ---------------------------------------------------------------------------

impl Demo {
    pub(crate) fn render_composer_mode(&mut self, ui: &mut egui::Ui) {
        ui.separator();

        // `f: [text input] [Add] [+xy] ... [+zw]`. Chips append to the
        // draft; the text input takes a full expression.
        ui.horizontal_wrapped(|ui| {
            ui.label("f:");
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.formula_input)
                    .hint_text("e.g. 90° (xy + zw)")
                    .desired_width(180.0),
            );
            let submitted = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            let add_clicked = ui.small_button("Add").clicked();
            if submitted || add_clicked {
                match parse_formula_term(&self.formula_input) {
                    Ok(term) => {
                        self.pending_actions.push(DeferredAction::SeqPushTerm(term));
                        self.formula_input.clear();
                        self.formula_error = None;
                        if submitted {
                            resp.request_focus();
                        }
                    }
                    Err(e) => self.formula_error = Some(e),
                }
            } else if self.formula_input.is_empty() {
                self.formula_error = None;
            }
            ui.separator();
            for plane in Plane4::ALL.iter() {
                if ui
                    .small_button(format!("+{}", plane.label()))
                    .on_hover_text("Add to the current draft term")
                    .clicked()
                {
                    self.pending_actions.push(DeferredAction::DraftPush(*plane));
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add_enabled(!self.seq.is_empty(), egui::Button::new("Clear"))
                    .on_hover_text("Remove all terms from the sequence")
                    .clicked()
                {
                    self.seq.clear();
                }
            });
        });
        if let Some(err) = &self.formula_error {
            ui.colored_label(
                egui::Color32::from_rgb(255, 120, 120),
                format!("parse error: {err}"),
            );
        }

        // Draft preview as a card; Add commits to seq, × discards.
        if !self.draft.is_empty() {
            egui::Frame::group(ui.style())
                .inner_margin(4.0)
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(egui::RichText::new("draft").small().weak());
                        render_plane_sum(ui, &self.draft, |ui, _, plane| {
                            ui.monospace(plane.label());
                        });
                        ui.add_space(8.0);
                        if ui
                            .small_button("Add")
                            .on_hover_text("Commit as one-shot term in sequence")
                            .clicked()
                        {
                            self.pending_actions.push(DeferredAction::SeqCommitDraft);
                        }
                        if ui
                            .add(
                                egui::Button::new(egui::RichText::new("×").size(14.0))
                                    .min_size(egui::vec2(MINI_BUTTON_W, MINI_BUTTON_W)),
                            )
                            .on_hover_text("Discard draft")
                            .clicked()
                        {
                            self.pending_actions.push(DeferredAction::DraftClear);
                        }
                    });
                });
        }

        self.render_composer_seq_cards(ui);
        self.render_composer_scrub_slider(ui);
    }

    /// Slide-to-rotate slider along the seq's net bivector direction.
    /// Slider value is the projection of `log(rot_state)` onto unit
    /// `D = compose_omega()/|compose_omega()|`, in degrees; the
    /// perpendicular component is preserved on drag so other rotations
    /// stay put. Hidden when the seq's net bivector is zero.
    pub(crate) fn render_composer_scrub_slider(&mut self, ui: &mut egui::Ui) {
        let omega = self.compose_omega();
        let mag_sq = omega.magnitude_squared();
        if mag_sq < 1e-12 {
            return;
        }
        let unit = omega * (1.0 / mag_sq.sqrt());
        let bivec = self.rot_state.log();
        let proj_rad = bivec.dot(unit);
        let mut proj_deg = proj_rad.to_degrees();

        const VALUE_CELL_W: f32 = 86.0;
        let avail = ui.available_width();
        let spacing = ui.spacing().item_spacing.x;
        let slider_w = (avail - VALUE_CELL_W - spacing).max(140.0);
        ui.spacing_mut().slider_width = slider_w;
        let row_size = egui::vec2(avail, CONTROL_H);
        let row_layout = egui::Layout::left_to_right(egui::Align::Center);

        // -360..360 for the Spin(4) double cover: 360° lands at the
        // negative rotor -1, 720° returns to identity.
        let formatted = format!("f {proj_deg:>+6.1}°");
        ui.allocate_ui_with_layout(row_size, row_layout, |ui| {
            let interaction = slider_with_edit(
                ui,
                &mut proj_deg,
                -360.0..=360.0,
                &formatted,
                "°",
                1,
                VALUE_CELL_W,
            );
            if interaction.changed {
                let new_proj = proj_deg.to_radians();
                let old_proj = bivec.dot(unit);
                let new_b = bivec + unit * (new_proj - old_proj);
                self.rot_state = new_b.exp();
                self.write_all(self.rot_state);
            }
        });
    }

    /// Composer's seq-card row. Each [`RotorTerm`] card is both a drag
    /// source (Term payload, reorders the seq) and a drop zone (Entry
    /// payload, cross-term plane migration). Mutations are gathered
    /// during rendering and applied at the end so the loop can borrow
    /// `self.seq` immutably in flight.
    pub(crate) fn render_composer_seq_cards(&mut self, ui: &mut egui::Ui) {
        let mut entry_moves: Vec<(usize, usize, usize)> = Vec::new();
        let mut remove_term: Option<usize> = None;
        let mut remove_scalar: Option<usize> = None;
        let mut add_scalar: Option<usize> = None;
        let mut edit_scalar: Option<(usize, f32)> = None;

        if !self.seq.is_empty() {
            ui.label("Sequence:");
            let term_h = CONTROL_H;
            let dragging_term = matches!(
                egui::DragAndDrop::payload::<DragPayload>(ui.ctx()).as_deref(),
                Some(DragPayload::Term(_))
            );
            let term_row_rect_id = ui.make_persistent_id("term-row-rect");
            let last_term_row_rect: Option<egui::Rect> =
                ui.ctx().memory(|m| m.data.get_temp(term_row_rect_id));
            let term_drop_idx = last_term_row_rect
                .and_then(|rect| drop_target_idx(ui.ctx(), dragging_term, rect, self.seq.len()));
            let dragged_term_idx =
                match egui::DragAndDrop::payload::<DragPayload>(ui.ctx()).as_deref() {
                    Some(DragPayload::Term(i)) => Some(*i),
                    _ => None,
                };
            let dragged_term_width = dragged_term_idx
                .map(|i| ui.make_persistent_id(("term-card", i)).with("width"))
                .and_then(|key| ui.ctx().memory(|m| m.data.get_temp::<f32>(key)))
                .unwrap_or(72.0);
            let term_row_resp = ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = CARD_ITEM_SPACING_X;
                let _ = dnd_apply_drop_pre_pass::<RotorTerm, DragPayload>(
                    ui,
                    &mut self.seq,
                    term_drop_idx,
                    |p| match p {
                        DragPayload::Term(i) => Some(*i),
                        _ => None,
                    },
                    "term-gap",
                    "term-card",
                    32,
                );
                let still_dragging_term = matches!(
                    egui::DragAndDrop::payload::<DragPayload>(ui.ctx()).as_deref(),
                    Some(DragPayload::Term(_))
                );
                let render_term_drop_idx = if still_dragging_term {
                    term_drop_idx
                } else {
                    None
                };
                for term_idx in 0..self.seq.len() {
                    let gap_id = ui.make_persistent_id(("term-gap", term_idx));
                    let _ = make_room_gap(
                        ui,
                        render_term_drop_idx == Some(term_idx),
                        gap_id,
                        term_h,
                        dragged_term_width,
                    );
                    if term_idx > 0 {
                        ui.label(egui::RichText::new("·").size(16.0).strong());
                    }
                    let card_id = ui.make_persistent_id(("term-card", term_idx));
                    let pickup_t = drag_pickup_t(ui.ctx(), card_id);
                    let stroke = if pickup_t > 0.0 {
                        egui::Stroke::new(
                            1.0 + pickup_t * 1.5,
                            egui::Color32::from_rgb(255, 200, 60),
                        )
                    } else {
                        egui::Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color)
                    };
                    let card_resp = dnd_drag_source_collapsing(
                        ui,
                        card_id,
                        DragPayload::Term(term_idx),
                        |ui| {
                            if ui.ctx().is_being_dragged(card_id) {
                                force_opaque_active(ui);
                            }
                            egui::Frame::default()
                                .fill(ui.visuals().widgets.noninteractive.bg_fill)
                                .stroke(stroke)
                                .inner_margin(3.0)
                                .corner_radius(egui::CornerRadius::same(3))
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        let term = &self.seq[term_idx];
                                        if let Some(phi) = term.scalar {
                                            // Inline DragValue: drag to adjust,
                                            // click to type.
                                            let phi_color = egui::Color32::from_rgb(255, 150, 150);
                                            let mut deg = phi.to_degrees();
                                            let drag_resp = ui
                                                .scope(|ui| {
                                                    ui.visuals_mut().override_text_color =
                                                        Some(phi_color);
                                                    ui.add(
                                                        egui::DragValue::new(&mut deg)
                                                            .suffix("°")
                                                            .speed(1.0)
                                                            .fixed_decimals(2)
                                                            .range(-720.0..=720.0),
                                                    )
                                                })
                                                .inner;
                                            if drag_resp.changed() {
                                                edit_scalar = Some((term_idx, deg.to_radians()));
                                            }
                                            drag_resp.on_hover_text(
                                                "Drag to adjust; click to type. \
                                                     Right-click the term to remove the scalar.",
                                            );
                                            ui.monospace("·");
                                        }
                                        let planes = self.seq[term_idx].planes.clone();
                                        render_plane_sum(ui, &planes, |ui, plane_idx, plane| {
                                            let pill_id = ui.make_persistent_id((
                                                "plane-pill",
                                                term_idx,
                                                plane_idx,
                                            ));
                                            ui.dnd_drag_source(
                                                pill_id,
                                                DragPayload::Entry(term_idx, plane_idx),
                                                |ui| {
                                                    ui.monospace(plane.label());
                                                },
                                            )
                                            .response
                                            .on_hover_cursor(egui::CursorIcon::Grab);
                                        });
                                    });
                                });
                        },
                    );
                    let is_self_dragged = ui.ctx().is_being_dragged(card_id);
                    if !is_self_dragged {
                        let card_rect = card_resp.rect;
                        let dragging_entry = matches!(
                            egui::DragAndDrop::payload::<DragPayload>(ui.ctx()).as_deref(),
                            Some(DragPayload::Entry(_, _))
                        );
                        let cursor = ui.ctx().input(|i| i.pointer.hover_pos());
                        let hovered =
                            dragging_entry && cursor.is_some_and(|p| card_rect.contains(p));
                        if hovered && ui.ctx().input(|i| i.pointer.any_released()) {
                            if let Some(arc) =
                                egui::DragAndDrop::take_payload::<DragPayload>(ui.ctx())
                            {
                                if let DragPayload::Entry(from_t, idx) = *arc {
                                    if from_t != term_idx {
                                        entry_moves.push((from_t, idx, term_idx));
                                    }
                                }
                            }
                        }
                    }
                    if !ui.ctx().is_being_dragged(card_id) {
                        let width_key = card_id.with("width");
                        let w = card_resp.rect.width();
                        ui.ctx().memory_mut(|m| m.data.insert_temp(width_key, w));
                    }
                    let scalar_now = self.seq[term_idx].scalar;
                    let menu_resp = card_resp.interact(egui::Sense::click());
                    menu_resp.context_menu(|ui| {
                        // Scalar editing is inline above; the menu owns only
                        // add/remove-scalar and delete-term.
                        if scalar_now.is_some() {
                            if ui.button("Remove scalar (φ)").clicked() {
                                remove_scalar = Some(term_idx);
                                ui.close_kind(egui::UiKind::Menu);
                            }
                        } else if ui.button("Add scalar (φ = 90°)").clicked() {
                            add_scalar = Some(term_idx);
                            ui.close_kind(egui::UiKind::Menu);
                        }
                        ui.separator();
                        if ui.button("Delete term").clicked() {
                            remove_term = Some(term_idx);
                            ui.close_kind(egui::UiKind::Menu);
                        }
                    });
                }
                let trailing_id = ui.make_persistent_id(("term-gap", self.seq.len()));
                let _ = make_room_gap(
                    ui,
                    render_term_drop_idx == Some(self.seq.len()),
                    trailing_id,
                    term_h,
                    dragged_term_width,
                );
                if !entry_moves.is_empty() || remove_term.is_some() {
                    let ctx = ui.ctx();
                    for i in 0..32 {
                        let card_id = ui.make_persistent_id(("term-card", i));
                        let _ = ctx.animate_value_with_time(card_id.with("pickup"), 0.0, 0.0);
                    }
                }
            });
            ui.ctx().memory_mut(|m| {
                m.data
                    .insert_temp(term_row_rect_id, term_row_resp.response.rect)
            });
        }

        // Apply deferred mutations in an order that keeps indices valid.
        if let Some(i) = add_scalar {
            if let Some(t) = self.seq.get_mut(i) {
                t.scalar = Some(std::f32::consts::FRAC_PI_2);
            }
        }
        if let Some(i) = remove_scalar {
            if let Some(t) = self.seq.get_mut(i) {
                t.scalar = None;
            }
        }
        if let Some((i, new_phi)) = edit_scalar {
            if let Some(t) = self.seq.get_mut(i) {
                if t.scalar.is_some() {
                    t.scalar = Some(new_phi);
                }
            }
        }
        // Sort by (source term, plane idx descending) so removals don't
        // shift earlier indices.
        entry_moves.sort_by_key(|(from, idx, _)| (*from, std::cmp::Reverse(*idx)));
        for (from_t, idx, to_t) in entry_moves {
            if let Some(src) = self.seq.get_mut(from_t) {
                if idx < src.planes.len() {
                    let plane = src.planes.remove(idx);
                    if let Some(dest) = self.seq.get_mut(to_t) {
                        dest.planes.push(plane);
                    }
                }
            }
        }
        // Drop emptied terms (after entry moves).
        self.seq.retain(|t| !t.planes.is_empty());
        if let Some(i) = remove_term {
            if i < self.seq.len() {
                self.seq.remove(i);
            }
        }
    }
}
