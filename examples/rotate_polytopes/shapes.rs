//! Shape-row UI: a horizontal row of shape cards with `+`-button
//! menu, drag-and-drop reorder, and per-card "Remove from row"
//! context menu. Drives `self.row` (the visible polytopes in
//! `Shapes` view mode); mutations here trigger `rebuild_bodies()`.

use rye_app::egui;
use rye_egui::{
    dnd::{
        apply_drop_pre_pass as dnd_apply_drop_pre_pass,
        drag_source_collapsing as dnd_drag_source_collapsing, drop_target_idx, force_opaque_active,
        make_room_gap, pickup_t as drag_pickup_t,
    },
    media::add_button,
};
use rye_physics::polytope::Polytope4;
use rye_render::raymarch::RaymarchShape;

use crate::catalog::{render_shape_catalog_menu, ShapeEntry};
use crate::consts::{CARD_ITEM_SPACING_X, CONTROL_H, CONTROL_W, MAX_ROW_LEN, SHAPE_CARD_WIDTH};
use crate::state::Demo;

impl Demo {
    /// Shape row + add-menu + drag-and-drop reorder. Extracted as a
    /// method so it can be called from both rotation modes (after
    /// the active-set checkboxes in `Active`, after the seq +
    /// Apply/Clear in `Composer`).
    pub(crate) fn render_shapes_section(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        // Only the SDF raymarch path is heavy on the 120/600-cell (~24 KB of const face-
        // normal data + per-pixel Wolfe-greedy projection); the rasterized section-faces
        // path keeps both polychora at vsync regardless of how many are in the row, so
        // the warning is irrelevant when `surface_raster_enabled`.
        let has_heavy_sdf = !self.surface_raster_enabled
            && self.row.iter().any(|e| {
                matches!(
                    e.shape,
                    RaymarchShape::Polytope(Polytope4::Cell120 | Polytope4::Cell600)
                )
            });
        if has_heavy_sdf {
            ui.colored_label(
                egui::Color32::from_rgb(242, 130, 70),
                "120/600-cell SDFs are heavy; expect <60 fps. Try `surface raster`.",
            );
        }
        let mut row_changed = false;

        let mut remove_idx: Option<usize> = None;
        let row_len = self.row.len();
        let row_h = CONTROL_H;
        // Slot index where the drop should land. Computed once
        // from cursor position and last-frame's row geometry, so
        // every slot agrees on which one is "the target."
        let row_rect_id = ui.make_persistent_id("shape-row-rect");
        let last_row_rect: Option<egui::Rect> = ui.ctx().memory(|m| m.data.get_temp(row_rect_id));
        let dragging_shape = egui::DragAndDrop::payload::<usize>(ui.ctx()).is_some();
        let drop_idx =
            last_row_rect.and_then(|rect| drop_target_idx(ui.ctx(), dragging_shape, rect, row_len));
        let row_rect = egui::ScrollArea::horizontal()
            .auto_shrink([false, true])
            .id_salt("rotate-polytopes-shapes-scroll")
            .show(ui, |ui| {
                let row_response =
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Min), |ui| {
                        ui.spacing_mut().item_spacing.x = CARD_ITEM_SPACING_X;

                        // Drop pre-pass: apply the reorder NOW so the
                        // render loop sees the new order and gaps
                        // are closed. See `dnd_apply_drop_pre_pass`
                        // for the issue-#54 rationale.
                        if dnd_apply_drop_pre_pass::<ShapeEntry, usize>(
                            ui,
                            &mut self.row,
                            drop_idx,
                            |p| Some(*p),
                            "shape-gap",
                            "shape-card",
                            MAX_ROW_LEN,
                        ) {
                            row_changed = true;
                        }
                        let still_dragging =
                            egui::DragAndDrop::payload::<usize>(ui.ctx()).is_some();
                        let render_drop_idx = if still_dragging { drop_idx } else { None };
                        let row_len = self.row.len();
                        for (i, entry) in self.row.iter().enumerate() {
                            let gap_id = ui.make_persistent_id(("shape-gap", i));
                            let _ = make_room_gap(
                                ui,
                                render_drop_idx == Some(i),
                                gap_id,
                                row_h,
                                SHAPE_CARD_WIDTH + 8.0,
                            );
                            if Self::render_shape_card(ui, i, entry, row_len) {
                                remove_idx = Some(i);
                            }
                        }
                        let trailing_id = ui.make_persistent_id(("shape-gap", row_len));
                        let _ = make_room_gap(
                            ui,
                            render_drop_idx == Some(row_len),
                            trailing_id,
                            row_h,
                            SHAPE_CARD_WIDTH + 16.0,
                        );
                        if self.row.len() < MAX_ROW_LEN {
                            let plus_resp = add_button(ui, egui::vec2(CONTROL_W, CONTROL_H - 2.0))
                                .on_hover_text("Add a shape to the row");
                            egui::Popup::menu(&plus_resp).show(|ui| {
                                ui.set_min_width(140.0);
                                render_shape_catalog_menu(ui, |entry| {
                                    self.row.push(entry);
                                    row_changed = true;
                                });
                            });
                        }
                        // Per-index animation state is keyed by ids
                        // resolved against THIS ui's scope. After a
                        // reorder, the cards now sitting at the old
                        // indices would otherwise inherit the
                        // previous occupants' `pickup_t = 1.0` and
                        // ghost-fade. Snap defaults here, while the
                        // ui scope still resolves to the same ids
                        // we used during rendering.
                        if remove_idx.is_some() {
                            let ctx = ui.ctx();
                            for i in 0..=MAX_ROW_LEN {
                                let card_id = ui.make_persistent_id(("shape-card", i));
                                let _ =
                                    ctx.animate_value_with_time(card_id.with("pickup"), 0.0, 0.0);
                            }
                        }
                    });
                row_response.response.rect
            })
            .inner;
        ui.ctx()
            .memory_mut(|m| m.data.insert_temp(row_rect_id, row_rect));
        if let Some(i) = remove_idx {
            self.row.remove(i);
            row_changed = true;
        }
        if row_changed {
            self.rebuild_bodies();
        }
    }

    /// One shape card: drag source for reorder, hover-name
    /// tooltip, right-click "Remove from row" context menu.
    /// Returns `true` when the user clicked Remove this frame so
    /// the caller can record the index for end-of-frame removal
    /// (in-flight removal would invalidate the row's iteration).
    fn render_shape_card(ui: &mut egui::Ui, i: usize, entry: &ShapeEntry, row_len: usize) -> bool {
        let card_id = ui.make_persistent_id(("shape-card", i));
        let pickup_t = drag_pickup_t(ui.ctx(), card_id);
        let card_fill = ui.visuals().widgets.noninteractive.bg_fill;
        let stroke_color = if pickup_t > 0.0 {
            egui::Color32::from_rgb(255, 200, 60)
        } else {
            ui.visuals().widgets.noninteractive.bg_stroke.color
        };
        let stroke = egui::Stroke::new(1.0 + pickup_t * 1.5, stroke_color);
        let drag_resp = dnd_drag_source_collapsing(ui, card_id, i, |ui| {
            if ui.ctx().is_being_dragged(card_id) {
                force_opaque_active(ui);
            }
            egui::Frame::default()
                .fill(card_fill)
                .stroke(stroke)
                .inner_margin(egui::Margin::symmetric(4, 6))
                .corner_radius(egui::CornerRadius::same(3))
                .show(ui, |ui| {
                    ui.allocate_ui_with_layout(
                        egui::vec2(SHAPE_CARD_WIDTH, 0.0),
                        egui::Layout::top_down(egui::Align::Center),
                        |ui| {
                            ui.add(
                                egui::Label::new(egui::RichText::new(entry.label).strong())
                                    .selectable(false)
                                    .wrap_mode(egui::TextWrapMode::Extend),
                            );
                        },
                    );
                });
        });
        let mut removed = false;
        drag_resp
            .on_hover_cursor(egui::CursorIcon::Grab)
            .on_hover_text(entry.long_name)
            .interact(egui::Sense::click())
            .context_menu(|ui| {
                if row_len > 1 && ui.button("Remove from row").clicked() {
                    removed = true;
                    ui.close_kind(egui::UiKind::Menu);
                }
            });
        removed
    }
}
