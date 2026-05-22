//! Drag-and-drop card-row primitives.
//!
//! egui ships [`Ui::dnd_drag_source`] and [`Ui::dnd_drop_zone`] but they don't compose
//! into a horizontal reorderable card row out of the box. The dragged source still
//! occupies its slot in the parent layout (egui's `scope_builder` advances the cursor
//! by the body's natural width), and there's no built-in "make room" gap that opens at
//! the cursor's drop target during a drag. This module provides the missing pieces:
//!
//! - [`drag_source_collapsing`]: drag source whose dragged copy floats on the Tooltip
//!   layer AND whose original slot collapses to zero width while in flight.
//! - [`make_room_gap`]: animated insertion gap that opens at the targeted slot and snaps
//!   closed on drop.
//! - [`drop_target_idx`]: cursor-position to insertion-slot index over a row's bounding
//!   rect.
//! - [`apply_drop_pre_pass`]: applies a reorder to a `Vec<T>` BEFORE the row's render
//!   loop runs, eliminating the one-frame "settles into place" lag where the drop frame
//!   paints with the old order plus the still-open gap.
//! - [`pickup_t`]: pickup-glow animation value for any drag source.
//! - [`force_opaque_active`]: lift the dragged body's visuals to active (egui's Tooltip
//!   layer otherwise dims them).
//!
//! Together these make a draggable row whose visual feedback is solid through pickup,
//! mid-drag, and drop. See the `polytope_playground` example for an integrated use.

use egui::{
    self, emath::TSTransform, vec2, DragAndDrop, Id, LayerId, Order, Rect, Response, Sense, Ui,
    UiBuilder,
};

/// Drag source whose dragged copy floats on the Tooltip layer AND whose original slot
/// collapses to zero width while in flight.
///
/// egui's stock [`Ui::dnd_drag_source`] uses `scope_builder` to host the dragged body,
/// which advances the parent cursor by the body's natural width. That leaves a phantom
/// card-shaped slot alongside the floating tooltip preview. We bypass `scope_builder`
/// via [`Ui::new_child`] (which does NOT advance the parent cursor) and register a
/// hit-rect at the body's natural position so callers (context menus, hover text) still
/// get a usable response.
///
/// The dragged card therefore occupies zero width in the row while the drag is in
/// flight, which composes with [`make_room_gap`] to produce a clean "card lifts away,
/// gap opens at drop target" effect with no horizontal layout shift.
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

/// Animated "make room" insertion gap at one slot of a horizontal row. The slot whose
/// `is_target` is `true` expands to `open_width` over ~120 ms; others stay at zero
/// width. Cards on either side slide outward as the gap opens, giving a clear drop
/// preview without a separate marker line.
///
/// Returns `true` if a pointer release occurred on the targeted gap this frame; the
/// caller takes whatever payload it expects from [`DragAndDrop`] and applies the move.
/// Most callers should use [`apply_drop_pre_pass`] instead, which handles take-payload
/// and reorder atomically before the render loop runs.
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
        // Snap the gap closed instantly on drop. Without this the gap animates from
        // `open_width` -> 0 over the next ~120 ms while the row's right side
        // rubberbands leftward as the gap closes; a visible "settle" the user reads
        // as jank.
        let _ = ui.ctx().animate_value_with_time(slot_id, 0.0, 0.0);
    }
    dropped
}

/// Map cursor x-position over a row's bounding `row_rect` to a 0-based insertion slot
/// index in `0..=item_count`. Returns `None` when no drag is active (`is_dragging` is
/// `false`) or the cursor isn't over the row band. Hit band extends ±40 pt vertically
/// so a card dragged a bit above or below the row still snaps to a slot.
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

/// Inside a [`drag_source_collapsing`] body, force fully-opaque widget visuals on the
/// current ui when the source is being dragged. egui paints the body to a Tooltip layer
/// when dragged, where widgets never register hover and therefore default to the dimmed
/// `inactive` style; this override lifts inactive and noninteractive fills/strokes to
/// match `active` so the floating ghost reads as a solid card.
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

/// "Pickup" pulse intensity in `[0.0, 1.0]` for the card identified by `drag_id`.
/// Animates from 0 to 1 in 120 ms when the source starts being dragged, and back to 0
/// over the same time when the drag ends. Use it to interpolate stroke width / color on
/// the dragged frame so the card visibly "lifts" on pickup.
pub fn pickup_t(ctx: &egui::Context, drag_id: Id) -> f32 {
    let target = if ctx.is_being_dragged(drag_id) {
        1.0
    } else {
        0.0
    };
    ctx.animate_value_with_time(drag_id.with("pickup"), target, 0.12)
}

/// Pre-pass that detects a "pointer released over a drop slot" event in THIS frame and
/// applies the reorder to `vec` immediately, before the row's render loop runs. Returns
/// `true` when a move actually fired.
///
/// Without this pre-pass, an end-of-frame `apply_reorders` runs too late: the row's
/// render loop iterates the OLD vec ordering, and [`make_room_gap`] allocates the
/// still-open gap at `drop_idx`, so the drop frame paints `[old layout][open gap][rest]`
/// for one frame before the next frame's render catches up. That's the visible "settles
/// into place" lag.
///
/// `filter` decides which payloads count as a reorder and extracts the source index. A
/// homogeneous-payload row passes `|p| Some(*p)`; an enum-payload row matches the
/// variant it cares about and returns `None` for the others.
///
/// `gap_id_prefix` and `card_id_prefix` drive the snap loop that closes gaps and resets
/// pickup animations on success, so the post-reorder render shows no leftover open slot
/// or stale pickup glow. Pass the same prefixes the row uses to build its
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

    /// egui's drag detection compares `time - press_start_time` against
    /// `Options::max_click_duration`. Without advancing `time` between frames, every
    /// press is "still within the click window" and `is_decidedly_dragging` stays
    /// false. Each helper here threads a monotonic clock so the test driver clears the
    /// click threshold.
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

    /// Press + drag past egui's start-drag threshold (~6 px) on a
    /// `drag_source_collapsing` body. After the third frame `is_being_dragged(id)`
    /// flips true and the payload is available to drop targets.
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

    /// Same source rendered into TWO Areas with different layers must NOT fire egui's
    /// "same id in two layers" debug_assert. `make_persistent_id` resolves through the
    /// per-Area scope, so the two passes see distinct ids and the check is satisfied.
    /// This guards against a regression to globally-stable ids.
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

    /// On pointer release with a valid drop slot and a payload that carries the source
    /// index, `apply_drop_pre_pass` mutates the vec in place: the dragged item is
    /// removed from `from` and reinserted at the correct destination (`to - 1` when
    /// moving rightward, `to` when moving leftward).
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

    /// `pickup_t` rises toward 1.0 while a drag is active. We can't drive a real drag
    /// in this scope cheaply, so we verify the non-dragged case stays at 0.0 (the rest
    /// is animation, owned by egui).
    #[test]
    fn pickup_t_zero_when_not_dragging() {
        let ctx = egui::Context::default();
        let id = Id::new("pickup-test");
        let _ = ctx.run(warmup_input(0.0), |_| {});
        let t = pickup_t(&ctx, id);
        assert_eq!(t, 0.0);
    }
}
