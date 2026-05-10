//! Slider with a side-cell text label that opens a precise edit
//! popup on right-click.
//!
//! egui's stock [`Slider`](egui::Slider) renders its value as a
//! variable-width text inside the slider, which means the slider's
//! visual right edge shifts as the value's character count changes
//! (e.g., `0.5` -> `12.34`). When several sliders sit in a vertical
//! stack, that shift makes the whole column jitter frame-to-frame.
//!
//! This widget separates the value display into a fixed-width side
//! cell and disables the slider's own value rendering. The side cell
//! exposes a context menu with a [`DragValue`](egui::DragValue) for
//! precise numeric entry, useful when the slider's drag granularity
//! is too coarse for the desired value.

use egui::{
    vec2, Align, Button, CursorIcon, DragValue, Layout, RichText, Slider, SliderClamping, Ui,
};

/// What happened to a [`slider_with_edit`] this frame. `changed`
/// fires from either source (slider drag, popup edit, or any
/// external mutation that the slider observes); `dragged` is
/// strictly user-on-the-slider this frame. Callers that recompute
/// expensive state ONLY when the user is actively scrubbing should
/// gate on `dragged` so they don't refire when something else
/// (e.g., a per-frame integrator) advances the value.
#[derive(Copy, Clone, Debug, Default)]
pub struct SliderInteraction {
    pub changed: bool,
    pub dragged: bool,
}

/// Render a slider with a fixed-width side cell that displays
/// `formatted` and opens a precise-edit [`DragValue`](egui::DragValue)
/// popup on right-click. Returns a [`SliderInteraction`] reporting
/// whether the value changed and whether the user was actively
/// dragging this frame.
///
/// `value_cell_w` is the fixed width allocated to the side label
/// cell. Without a fixed width the cell would resize as the value's
/// character count varies, shifting the slider's right edge frame-
/// to-frame.
pub fn slider_with_edit(
    ui: &mut Ui,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    formatted: &str,
    edit_suffix: &str,
    edit_decimals: usize,
    value_cell_w: f32,
) -> SliderInteraction {
    let slider_resp = ui.add(
        Slider::new(value, range.clone())
            .show_value(false)
            .smart_aim(false)
            .clamping(SliderClamping::Always),
    );
    let mut popup_changed = false;
    ui.allocate_ui_with_layout(
        vec2(value_cell_w, 14.0),
        Layout::left_to_right(Align::Center),
        |ui| {
            let label_resp = ui.add(
                Button::new(RichText::new(formatted).monospace())
                    .frame(false)
                    .small(),
            );
            label_resp
                .on_hover_cursor(CursorIcon::ContextMenu)
                .on_hover_text("Right-click to edit value")
                .context_menu(|ui| {
                    let drag_resp = ui.add(
                        DragValue::new(value)
                            .range(range)
                            .suffix(edit_suffix)
                            .fixed_decimals(edit_decimals),
                    );
                    if drag_resp.changed() {
                        popup_changed = true;
                    }
                });
        },
    );
    SliderInteraction {
        changed: slider_resp.changed() || popup_changed,
        dragged: slider_resp.dragged(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn screen() -> egui::Rect {
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(800.0, 600.0))
    }

    /// Rendering with no input returns `false` and leaves `value`
    /// untouched. Sanity check that the widget is invokable and
    /// doesn't fire a phantom "changed" on the first frame.
    #[test]
    fn no_input_returns_false_and_preserves_value() {
        let ctx = egui::Context::default();
        let mut value = 0.5_f32;
        let input = egui::RawInput {
            screen_rect: Some(screen()),
            time: Some(0.0),
            ..Default::default()
        };
        let mut interaction = SliderInteraction::default();
        let _ = ctx.run(input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                interaction = slider_with_edit(ui, &mut value, 0.0..=1.0, "0.50", "", 2, 60.0);
            });
        });
        assert!(
            !interaction.changed,
            "no input should not fire a changed event"
        );
        assert!(!interaction.dragged, "no input should not report dragged");
        assert_eq!(value, 0.5);
    }

    /// The side cell is allocated at exactly `value_cell_w` regardless
    /// of how many characters `formatted` has. This is the contract
    /// callers depend on: the slider's right edge stays put as the
    /// formatted value's character count changes between frames.
    #[test]
    fn side_cell_width_is_fixed() {
        let ctx = egui::Context::default();
        let mut total_widths: Vec<f32> = Vec::new();
        let layout_input = egui::RawInput {
            screen_rect: Some(screen()),
            time: Some(0.0),
            ..Default::default()
        };
        // Render the same slider with two formatted strings of
        // wildly different widths. The total widget rect width
        // should be the same.
        for formatted in ["0", "+999.99"] {
            let mut value = 0.5_f32;
            let mut total = 0.0_f32;
            let _ = ctx.run(layout_input.clone(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let before = ui.next_widget_position().x;
                    let _ = slider_with_edit(ui, &mut value, 0.0..=1.0, formatted, "", 2, 60.0);
                    let after = ui.next_widget_position().x;
                    total = after - before;
                });
            });
            total_widths.push(total);
        }
        let drift = (total_widths[0] - total_widths[1]).abs();
        assert!(
            drift < 1.0,
            "slider total width must be invariant across formatted-string lengths; \
             got {} vs {} (drift {:.2})",
            total_widths[0],
            total_widths[1],
            drift,
        );
    }
}
