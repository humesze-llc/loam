//! Drag-and-drop card-row primitives.
//!
//! egui's [`Ui::dnd_drag_source`] / [`Ui::dnd_drop_zone`] don't compose into a
//! horizontal reorderable card row: the dragged source keeps occupying its slot, and
//! there is no "make room" gap at the drop target. This module supplies the pieces:
//!
//! - [`drag_source_collapsing`]: drag source whose copy floats on the Tooltip layer
//!   while its original slot collapses to zero width.
//! - [`make_room_gap`]: animated insertion gap that opens at the target and snaps
//!   closed on drop.
//! - [`drop_target_idx`]: cursor position to insertion-slot index over a row rect.
//! - [`apply_drop_pre_pass`]: applies the reorder before the render loop, killing the
//!   one-frame "settles into place" lag.
//! - [`pickup_t`]: pickup-glow animation value for a drag source.
//! - [`force_opaque_active`]: lift the dragged body's visuals to active (the Tooltip
//!   layer otherwise dims them).
//!
//! See the `polytope_playground` example for an integrated use.

use egui::{
    self, emath::TSTransform, vec2, DragAndDrop, Id, LayerId, Order, Rect, Response, Sense, Ui,
    UiBuilder,
};

/// Drag source whose dragged copy floats on the Tooltip layer while its original slot
/// collapses to zero width.
///
/// Stock [`Ui::dnd_drag_source`] hosts the body via `scope_builder`, which advances the
/// parent cursor and leaves a phantom slot. This bypasses it with [`Ui::new_child`]
/// (no cursor advance) and registers a hit-rect at the body's natural position so
/// callers still get a usable response. Composes with [`make_room_gap`] for a clean
/// "card lifts away, gap opens" effect with no horizontal layout shift.
pub fn drag_source_collapsing<P>(
    ui: &mut Ui,
    id: Id,
    payload: P,
    body: impl FnOnce(&mut Ui),
) -> Response
where
    P: 'static + Send + Sync,
{
    let ctx = ui.ctx().clone();
    let is_dragged = ctx.is_being_dragged(id);
    if !is_dragged {
        return ui.dnd_drag_source(id, payload, body).response;
    }
    DragAndDrop::set_payload(&ctx, payload);
    let layer_id = LayerId::new(Order::Tooltip, id);
    let mut child = ui.new_child(UiBuilder::new().layer_id(layer_id));
    body(&mut child);
    let body_rect = child.min_rect();
    if let Some(pos) = ctx.pointer_interact_pos() {
        let delta = pos - body_rect.center();
        ctx.transform_layer_shapes(layer_id, TSTransform::from_translation(delta));
    }
    ui.interact(body_rect, id, Sense::hover())
}

/// Animated insertion gap at one slot of a horizontal row: the `is_target` slot
/// expands to `open_width` over ~120 ms, others stay at zero width.
///
/// Returns `true` if a pointer release occurred on the targeted gap this frame. Most
/// callers should use [`apply_drop_pre_pass`], which handles take-payload + reorder
/// atomically before the render loop.
pub fn make_room_gap(
    ui: &mut Ui,
    is_target: bool,
    slot_id: Id,
    height: f32,
    open_width: f32,
) -> bool {
    let target_w = if is_target { open_width } else { 0.0 };
    let smooth_w = ui.ctx().animate_value_with_time(slot_id, target_w, 0.12);
    if smooth_w >= 0.5 {
        let _ = ui.allocate_exact_size(vec2(smooth_w, height), Sense::hover());
    }
    let dropped = is_target && ui.ctx().input(|i| i.pointer.any_released());
    if dropped {
        // Snap the gap closed instantly; otherwise it animates shut over ~120 ms and
        // the row's right side rubberbands leftward, reading as jank.
        let _ = ui.ctx().animate_value_with_time(slot_id, 0.0, 0.0);
    }
    dropped
}

/// Map cursor x over `row_rect` to a 0-based insertion slot in `0..=item_count`.
/// `None` when not dragging or outside the row band; the band extends ±40 pt
/// vertically so a card dragged slightly off the row still snaps.
pub fn drop_target_idx(
    ctx: &egui::Context,
    is_dragging: bool,
    row_rect: Rect,
    item_count: usize,
) -> Option<usize> {
    if !is_dragging {
        return None;
    }
    let cursor = ctx.input(|i| i.pointer.hover_pos())?;
    let band = row_rect.expand2(vec2(0.0, 40.0));
    if !band.x_range().contains(cursor.x) || !band.y_range().contains(cursor.y) {
        return None;
    }
    let n_slots = item_count + 1;
    let slot_w = (row_rect.width() / n_slots as f32).max(1.0);
    let rel = (cursor.x - row_rect.left()).max(0.0);
    Some(((rel / slot_w) as usize).min(item_count))
}

/// Inside a [`drag_source_collapsing`] body, force opaque widget visuals. The dragged
/// body paints to a Tooltip layer where widgets never register hover and default to
/// the dimmed `inactive` style; this lifts inactive + noninteractive fills/strokes to
/// `active` so the ghost reads as a solid card.
pub fn force_opaque_active(ui: &mut Ui) {
    let active = ui.visuals().widgets.active;
    let v = ui.visuals_mut();
    v.widgets.inactive.bg_fill = active.bg_fill;
    v.widgets.inactive.weak_bg_fill = active.weak_bg_fill;
    v.widgets.inactive.fg_stroke = active.fg_stroke;
    v.widgets.inactive.bg_stroke = active.bg_stroke;
    v.widgets.noninteractive.bg_fill = active.bg_fill;
    v.widgets.noninteractive.weak_bg_fill = active.weak_bg_fill;
}

/// "Pickup" pulse intensity in `[0.0, 1.0]` for `drag_id`, animating 0->1 over 120 ms
/// on drag start and back on drag end. Interpolate stroke width / color with it so the
/// card visibly lifts on pickup.
pub fn pickup_t(ctx: &egui::Context, drag_id: Id) -> f32 {
    let target = if ctx.is_being_dragged(drag_id) {
        1.0
    } else {
        0.0
    };
    ctx.animate_value_with_time(drag_id.with("pickup"), target, 0.12)
}

/// Detect a "pointer released over a drop slot" event this frame and apply the reorder
/// to `vec` before the row's render loop runs; returns `true` when a move fired. Running
/// before the loop avoids the one-frame "settles into place" lag an end-of-frame reorder
/// would leave.
///
/// `filter` selects which payloads count and extracts the source index (homogeneous
/// rows pass `|p| Some(*p)`; enum-payload rows match their variant).
///
/// `gap_id_prefix` / `card_id_prefix` drive the snap loop that closes gaps and resets
/// pickup glow on success; pass the same prefixes the row uses for its
/// `make_persistent_id`s.
pub fn apply_drop_pre_pass<T, P>(
    ui: &mut Ui,
    vec: &mut Vec<T>,
    drop_idx: Option<usize>,
    filter: impl FnOnce(&P) -> Option<usize>,
    gap_id_prefix: &'static str,
    card_id_prefix: &'static str,
    max_count: usize,
) -> bool
where
    P: 'static + Send + Sync,
{
    if !ui.ctx().input(|i| i.pointer.any_released()) {
        return false;
    }
    let Some(to) = drop_idx else {
        return false;
    };
    let Some(arc) = DragAndDrop::payload::<P>(ui.ctx()) else {
        return false;
    };
    let Some(from) = filter(&arc) else {
        return false;
    };
    let _ = DragAndDrop::take_payload::<P>(ui.ctx());
    if from == to || from >= vec.len() {
        return false;
    }
    let item = vec.remove(from);
    let dest = if to > from { to - 1 } else { to };
    vec.insert(dest.min(vec.len()), item);
    let ctx = ui.ctx();
    for i in 0..=max_count {
        let gap_id = ui.make_persistent_id((gap_id_prefix, i));
        let _ = ctx.animate_value_with_time(gap_id, 0.0, 0.0);
        let card_id = ui.make_persistent_id((card_id_prefix, i));
        let _ = ctx.animate_value_with_time(card_id.with("pickup"), 0.0, 0.0);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn screen() -> Rect {
        Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(800.0, 600.0))
    }

    /// egui's drag detection needs `time` to advance past `max_click_duration`, so
    /// these input helpers thread a monotonic clock to clear the click threshold.
    fn warmup_input(time: f64) -> egui::RawInput {
        egui::RawInput {
            screen_rect: Some(screen()),
            time: Some(time),
            ..Default::default()
        }
    }

    fn pointer_press(time: f64, pos: egui::Pos2) -> egui::RawInput {
        let mut input = warmup_input(time);
        input.events.push(egui::Event::PointerMoved(pos));
        input.events.push(egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: Default::default(),
        });
        input
    }

    fn pointer_move(time: f64, pos: egui::Pos2) -> egui::RawInput {
        let mut input = warmup_input(time);
        input.events.push(egui::Event::PointerMoved(pos));
        input
    }

    fn pointer_release(time: f64, pos: egui::Pos2) -> egui::RawInput {
        let mut input = warmup_input(time);
        input.events.push(egui::Event::PointerMoved(pos));
        input.events.push(egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: Default::default(),
        });
        input
    }

    #[test]
    fn drop_target_idx_returns_none_when_not_dragging() {
        let ctx = egui::Context::default();
        let row = Rect::from_min_size(egui::pos2(0.0, 0.0), vec2(100.0, 30.0));
        assert_eq!(drop_target_idx(&ctx, false, row, 4), None);
    }

    /// Press + drag past the start-drag threshold flips `is_being_dragged(id)` true
    /// and makes the payload available to drop targets.
    #[test]
    fn drag_source_collapsing_starts_drag() {
        let ctx = egui::Context::default();
        let id = Id::new("dnd-test-card");
        let card_pos = egui::pos2(60.0, 30.0);
        let render = |ctx: &egui::Context| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let _ = drag_source_collapsing(ui, id, 42_usize, |ui| {
                    ui.allocate_exact_size(vec2(80.0, 18.0), Sense::hover());
                });
            });
        };
        let _ = ctx.run(warmup_input(0.0), render);
        let _ = ctx.run(pointer_press(0.05, card_pos), render);
        let _ = ctx.run(pointer_move(0.10, card_pos + vec2(20.0, 0.0)), render);
        let _ = ctx.run(pointer_move(0.15, card_pos + vec2(40.0, 0.0)), render);
        assert!(
            ctx.is_being_dragged(id),
            "drag should be active after press + move past threshold"
        );
        assert!(
            DragAndDrop::has_payload_of_type::<usize>(&ctx),
            "drag payload should be set after drag starts"
        );
    }

    /// The same source rendered into two Areas must not trip egui's "same id in two
    /// layers" debug_assert; `make_persistent_id` resolves per-Area to distinct ids.
    /// Guards against a regression to globally-stable ids.
    #[test]
    fn drag_source_collapsing_two_pass_no_layer_collision() {
        let ctx = egui::Context::default();
        let render = |ctx: &egui::Context| {
            let _ = egui::Area::new(Id::new("measure"))
                .order(Order::Background)
                .interactable(false)
                .fixed_pos(egui::pos2(-99_999.0, -99_999.0))
                .show(ctx, |ui| {
                    ui.set_invisible();
                    let id = ui.make_persistent_id("test-card");
                    let _ = drag_source_collapsing(ui, id, 7_usize, |ui| {
                        ui.allocate_exact_size(vec2(80.0, 18.0), Sense::hover());
                    });
                });
            let _ = egui::Area::new(Id::new("visible"))
                .fixed_pos(egui::pos2(0.0, 0.0))
                .movable(false)
                .show(ctx, |ui| {
                    let id = ui.make_persistent_id("test-card");
                    let _ = drag_source_collapsing(ui, id, 7_usize, |ui| {
                        ui.allocate_exact_size(vec2(80.0, 18.0), Sense::hover());
                    });
                });
        };
        let _ = ctx.run(warmup_input(0.0), render);
        let _ = ctx.run(warmup_input(0.05), render);
    }

    /// On release with a valid drop slot and source-index payload,
    /// `apply_drop_pre_pass` reorders the vec in place (`to - 1` rightward, `to`
    /// leftward).
    #[test]
    fn apply_drop_pre_pass_reorders_vec_on_release() {
        let ctx = egui::Context::default();
        let mut vec = vec!['a', 'b', 'c', 'd'];
        DragAndDrop::set_payload(&ctx, 0_usize);
        let pos = egui::pos2(50.0, 30.0);
        let _ = ctx.run(pointer_release(0.05, pos), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let moved = apply_drop_pre_pass::<char, usize>(
                    ui,
                    &mut vec,
                    Some(3),
                    |p| Some(*p),
                    "test-gap",
                    "test-card",
                    8,
                );
                assert!(
                    moved,
                    "release with valid payload + drop_idx should reorder"
                );
            });
        });
        assert_eq!(vec, vec!['b', 'c', 'a', 'd']);
    }

    /// No release event in the frame -> no reorder, vec untouched.
    #[test]
    fn apply_drop_pre_pass_noop_without_release() {
        let ctx = egui::Context::default();
        let mut vec = vec!['a', 'b', 'c'];
        DragAndDrop::set_payload(&ctx, 0_usize);
        let _ = ctx.run(warmup_input(0.05), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let moved = apply_drop_pre_pass::<char, usize>(
                    ui,
                    &mut vec,
                    Some(2),
                    |p| Some(*p),
                    "test-gap",
                    "test-card",
                    8,
                );
                assert!(!moved);
            });
        });
        assert_eq!(vec, vec!['a', 'b', 'c']);
    }

    /// `pickup_t` stays at 0.0 when no drag is active.
    #[test]
    fn pickup_t_zero_when_not_dragging() {
        let ctx = egui::Context::default();
        let id = Id::new("pickup-test");
        let _ = ctx.run(warmup_input(0.0), |_| {});
        let t = pickup_t(&ctx, id);
        assert_eq!(t, 0.0);
    }
}
