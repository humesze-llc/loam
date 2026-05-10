//! Egui rendering for the console in two modes: docked half-screen drop-down
//! (default) and detached draggable [`egui::Window`].
//!
//! Both modes share the same inner content (title row, scrollback, input
//! row) via [`draw_content`]; only the outer container differs. Switching
//! modes is a state flip on [`Console`] and toggles which container the
//! renderer chooses on the next frame.
//!
//! ## Docked
//!
//! An [`egui::Area`] anchored at the viewport top, sized to the full width
//! and `PANEL_HEIGHT_FRACTION` of viewport height. Vertical translation
//! `-panel_height * (1.0 - progress)` slides the panel down from above as
//! `progress` interpolates 0 -> 1 in [`super::ANIM_DURATION_SECS`]. The whole
//! panel is always laid out at full height; only the position changes, so
//! the input row stays pinned to the bottom of the panel during the slide.
//!
//! ## Detached
//!
//! An [`egui::Window`] with `title_bar(false)` (we render our own title row
//! inside) plus `resizable(true)` and `movable(true)`. egui persists position
//! and size across frames via the window id.
//!
//! ## Focus
//!
//! In docked mode the input row's TextEdit re-requests focus every frame so
//! typing always lands there; the docked panel is modal-by-design (no other
//! egui widgets sit above it). In detached mode focus is only requested on
//! `pending_focus` (the open frame), so the user can click outside the
//! window to give keyboard back to the app.

use egui::{
    Color32, FontId, Frame, Layout, Margin, Order, RichText, ScrollArea, Sense, Stroke, TextEdit,
};

use super::{Console, HistoryLine, LineKind, PANEL_HEIGHT_FRACTION};
use crate::media::dock_chevrons;

const COLOR_BG: Color32 = Color32::from_rgba_premultiplied(12, 12, 16, 230);
const COLOR_INPUT_ECHO: Color32 = Color32::from_rgb(230, 230, 235);
const COLOR_OUTPUT: Color32 = Color32::from_rgb(180, 180, 188);
const COLOR_ERROR: Color32 = Color32::from_rgb(245, 130, 130);
const COLOR_SYSTEM: Color32 = Color32::from_rgb(140, 200, 220);
const COLOR_PROMPT: Color32 = Color32::from_rgb(160, 200, 140);
const COLOR_TITLE: Color32 = Color32::from_rgb(200, 200, 210);
const COLOR_SEPARATOR: Color32 = Color32::from_rgb(60, 60, 70);
const FONT_SIZE: f32 = 13.0;
const ROW_TITLE_HEIGHT: f32 = 22.0;
const ROW_INPUT_HEIGHT: f32 = 24.0;

/// Default detached-window dimensions, used the first frame the user switches
/// to detached mode. Subsequent frames respect whatever position/size egui's
/// window memory has remembered.
const DETACHED_DEFAULT_W: f32 = 520.0;
const DETACHED_DEFAULT_H: f32 = 320.0;
const DETACHED_MIN_W: f32 = 280.0;
const DETACHED_MIN_H: f32 = 120.0;

pub(super) fn draw<Ctx: 'static>(
    console: &mut Console<Ctx>,
    ctx: &egui::Context,
    app_ctx: &mut Ctx,
    progress: f32,
) {
    if console.detached {
        draw_detached(console, ctx, app_ctx);
    } else {
        draw_docked(console, ctx, app_ctx, progress);
    }
}

fn draw_docked<Ctx: 'static>(
    console: &mut Console<Ctx>,
    ctx: &egui::Context,
    app_ctx: &mut Ctx,
    progress: f32,
) {
    let viewport = ctx.content_rect();
    let panel_height = (viewport.height() * PANEL_HEIGHT_FRACTION).round();
    let y_offset = -panel_height * (1.0 - progress);
    let width = viewport.width();

    // Click-outside-defocus: pointer presses below the panel rect release
    // input focus so mouse + keyboard go back to the app while the console
    // stays open. Pressing back inside the panel re-enables the per-frame
    // focus re-request the input row uses to keep typing anchored.
    //
    // Computed against the FULL panel rect (no animation offset), so a click
    // during the slide-in/out doesn't toggle the wrong way as the panel
    // passes under the cursor.
    let panel_rect = egui::Rect::from_min_size(
        egui::pos2(viewport.min.x, viewport.min.y),
        egui::vec2(width, panel_height),
    );
    let pointer_pressed = ctx.input(|i| i.pointer.any_pressed());
    if pointer_pressed {
        if let Some(pos) = ctx.input(|i| i.pointer.interact_pos()) {
            console.user_defocused = !panel_rect.contains(pos);
        }
    }

    egui::Area::new(egui::Id::new("rye_console_area"))
        .order(Order::Foreground)
        .fixed_pos(egui::pos2(viewport.min.x, viewport.min.y + y_offset))
        .show(ctx, |ui| {
            let frame = Frame::default()
                .fill(COLOR_BG)
                .inner_margin(Margin::same(0));
            frame.show(ui, |ui| {
                ui.set_min_size(egui::vec2(width, panel_height));
                ui.set_max_size(egui::vec2(width, panel_height));
                ui.allocate_ui_with_layout(
                    egui::vec2(width, panel_height),
                    Layout::top_down(egui::Align::Min),
                    |ui| {
                        let scroll_h = panel_height - ROW_TITLE_HEIGHT - ROW_INPUT_HEIGHT - 2.0;
                        draw_content(ui, console, app_ctx, width, scroll_h);
                    },
                );
            });
        });
}

fn draw_detached<Ctx: 'static>(console: &mut Console<Ctx>, ctx: &egui::Context, app_ctx: &mut Ctx) {
    let viewport = ctx.content_rect();
    let default_pos = egui::pos2(
        (viewport.right() - DETACHED_DEFAULT_W - 16.0).max(viewport.left() + 16.0),
        viewport.top() + 80.0,
    );
    let frame = Frame::default()
        .fill(COLOR_BG)
        .stroke(Stroke::new(1.0, COLOR_SEPARATOR))
        .inner_margin(Margin::same(0))
        .corner_radius(egui::CornerRadius::same(4));

    egui::Window::new("rye_console_window")
        .id(egui::Id::new("rye_console_window"))
        .title_bar(false)
        .resizable(true)
        .collapsible(false)
        .movable(true)
        .default_pos(default_pos)
        .default_size(egui::vec2(DETACHED_DEFAULT_W, DETACHED_DEFAULT_H))
        .min_width(DETACHED_MIN_W)
        .min_height(DETACHED_MIN_H)
        .frame(frame)
        .show(ctx, |ui| {
            // Tight vertical layout: drop inter-item spacing between title,
            // separators, scrollback, and input row so they sum to exactly
            // `available_height`. Without this, the default `item_spacing.y`
            // (~3 px) accumulates across the four gaps and pushes content
            // past the Window's interior, which auto-sizes the Window larger
            // each frame in a positive-feedback loop (the previous
            // "stretches vertically" bug).
            //
            // Likewise, `ui.available_*` is the Window's INNER content area;
            // the Window's outer rect (e.g., `Memory::area_rect`) includes
            // egui's resize-handle chrome on each side, and over-allocating
            // by that chrome stretches the Window horizontally one frame at
            // a time (the "stretches horizontally" follow-up bug from the
            // cached-rect attempt at the same fix).
            ui.spacing_mut().item_spacing.y = 0.0;
            let width = ui.available_width();
            let scroll_h =
                (ui.available_height() - ROW_TITLE_HEIGHT - ROW_INPUT_HEIGHT - 2.0).max(60.0);
            draw_content(ui, console, app_ctx, width, scroll_h);
        });
}

/// Inner content shared by both modes. `scroll_height` is the pinned height
/// of the scrollback area; both modes pre-compute it from their container's
/// outer size minus the title and input rows.
fn draw_content<Ctx: 'static>(
    ui: &mut egui::Ui,
    console: &mut Console<Ctx>,
    app_ctx: &mut Ctx,
    width: f32,
    scroll_height: f32,
) {
    draw_title_row(ui, console, width);
    draw_separator(ui, width);
    draw_scrollback(ui, console, scroll_height, width);
    draw_separator(ui, width);
    draw_input_row(ui, console, app_ctx, width);
}

fn draw_title_row<Ctx>(ui: &mut egui::Ui, console: &mut Console<Ctx>, width: f32) {
    ui.allocate_ui_with_layout(
        egui::vec2(width, ROW_TITLE_HEIGHT),
        Layout::left_to_right(egui::Align::Center),
        |ui| {
            ui.add_space(8.0);
            ui.label(
                RichText::new("console")
                    .color(COLOR_TITLE)
                    .font(FontId::monospace(FONT_SIZE))
                    .strong(),
            );
            ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(8.0);
                let tip = if console.detached {
                    "Re-attach as the half-screen drop-down"
                } else {
                    "Detach as a draggable window"
                };
                if dock_chevrons(ui, egui::vec2(12.0, 16.0), console.detached, tip).clicked() {
                    console.detached = !console.detached;
                }
                ui.add_space(8.0);
                if !console.status.is_empty() {
                    ui.label(
                        RichText::new(&console.status)
                            .color(COLOR_TITLE)
                            .font(FontId::monospace(FONT_SIZE)),
                    );
                }
            });
        },
    );
}

fn draw_separator(ui: &mut egui::Ui, width: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 1.0), Sense::hover());
    ui.painter().hline(
        rect.x_range(),
        rect.center().y,
        Stroke::new(1.0, COLOR_SEPARATOR),
    );
}

fn draw_scrollback<Ctx>(ui: &mut egui::Ui, console: &Console<Ctx>, height: f32, width: f32) {
    ui.allocate_ui_with_layout(
        egui::vec2(width, height),
        Layout::top_down(egui::Align::Min),
        |ui| {
            ScrollArea::vertical()
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .max_height(height)
                .show(ui, |ui| {
                    ui.add_space(4.0);
                    for line in &console.history {
                        ui.horizontal(|ui| {
                            ui.add_space(8.0);
                            ui.label(line_text(line));
                        });
                    }
                    ui.add_space(4.0);
                });
        },
    );
}

fn line_text(line: &HistoryLine) -> RichText {
    let color = match line.kind {
        LineKind::Input => COLOR_INPUT_ECHO,
        LineKind::Output => COLOR_OUTPUT,
        LineKind::Error => COLOR_ERROR,
        LineKind::System => COLOR_SYSTEM,
    };
    RichText::new(&line.text)
        .color(color)
        .font(FontId::monospace(FONT_SIZE))
}

fn draw_input_row<Ctx: 'static>(
    ui: &mut egui::Ui,
    console: &mut Console<Ctx>,
    app_ctx: &mut Ctx,
    width: f32,
) {
    ui.allocate_ui_with_layout(
        egui::vec2(width, ROW_INPUT_HEIGHT),
        Layout::left_to_right(egui::Align::Center),
        |ui| {
            ui.add_space(8.0);
            ui.label(
                RichText::new(">")
                    .color(COLOR_PROMPT)
                    .font(FontId::monospace(FONT_SIZE))
                    .strong(),
            );

            // Submit on Enter detected BEFORE rendering the TextEdit, so we
            // can consume the key event and prevent it from being
            // interpreted as TextEdit input. Using `Response::lost_focus +
            // Enter` is the egui-idiomatic pattern but conflicts with the
            // unconditional `request_focus()` we issue every frame in docked
            // mode to keep typing anchored on the input box; the focus never
            // gets a chance to be "lost" between frames, so `lost_focus()`
            // never fires. `consume_key` sidesteps the focus-state dance
            // entirely.
            let enter_pressed =
                ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Enter));

            let prev_input = console.input.clone();
            let response = ui.add(
                TextEdit::singleline(&mut console.input)
                    .font(FontId::monospace(FONT_SIZE))
                    .frame(false)
                    .desired_width(width - 32.0)
                    .text_color(COLOR_INPUT_ECHO),
            );

            // Any input change outside of tab-cycling invalidates the
            // tab-completion state.
            if console.input != prev_input {
                console.tab = None;
            }

            // Focus policy:
            //   - `pending_focus` is set on open; one-shot focus request
            //     applies in both modes.
            //   - Docked: re-request focus every frame so the panel feels
            //     modal (no other egui widget should steal typing) UNLESS
            //     `user_defocused` (the user clicked outside the panel area
            //     to talk to the app); then leave focus alone until they
            //     click back inside the panel.
            //   - Detached: leave focus alone after the initial request so
            //     the user can click outside the window to give keyboard
            //     back to the app.
            if console.pending_focus {
                response.request_focus();
                console.pending_focus = false;
            } else if !console.detached && !console.user_defocused && !response.has_focus() {
                response.request_focus();
            }

            if enter_pressed {
                let line = std::mem::take(&mut console.input);
                console.execute(&line, app_ctx);
                response.request_focus();
            }
        },
    );
}
