//! Egui rendering for the half-screen drop-down console.
//!
//! Layout (top to bottom):
//!
//! 1. Title row: `rye console` on the left, optional status string on
//!    the right, separator beneath.
//! 2. Scrollback: monospace, color-coded by [`LineKind`], wrapped in a
//!    `ScrollArea` that sticks to the bottom.
//! 3. Input row: `> ` prompt, monospace `TextEdit::singleline`. Enter
//!    fires execution; focus is re-requested every frame so typing
//!    always lands here while the panel is open.
//!
//! The panel is rendered with a vertical translation of
//! `-panel_height * (1.0 - progress)`, sliding down from above as
//! `progress` interpolates 0 -> 1 in [`super::ANIM_DURATION_SECS`].
//! The whole panel is always laid out at full height; only the
//! position changes. That keeps the input row pinned to the bottom of
//! the panel during the slide rather than rubber-banding into place
//! when the slide completes.

use egui::{
    Color32, FontId, Frame, Layout, Margin, Order, RichText, ScrollArea, Sense, Stroke, TextEdit,
};

use super::{Console, HistoryLine, LineKind, PANEL_HEIGHT_FRACTION};

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

pub(super) fn draw<Ctx: 'static>(
    console: &mut Console<Ctx>,
    ctx: &egui::Context,
    app_ctx: &mut Ctx,
    progress: f32,
) {
    let viewport = ctx.content_rect();
    let panel_height = (viewport.height() * PANEL_HEIGHT_FRACTION).round();
    let y_offset = -panel_height * (1.0 - progress);

    egui::Area::new(egui::Id::new("rye_console_area"))
        .order(Order::Foreground)
        .fixed_pos(egui::pos2(viewport.min.x, viewport.min.y + y_offset))
        .show(ctx, |ui| {
            let frame = Frame::default()
                .fill(COLOR_BG)
                .inner_margin(Margin::same(0));
            frame.show(ui, |ui| {
                ui.set_min_size(egui::vec2(viewport.width(), panel_height));
                ui.set_max_size(egui::vec2(viewport.width(), panel_height));
                ui.allocate_ui_with_layout(
                    egui::vec2(viewport.width(), panel_height),
                    Layout::top_down(egui::Align::Min),
                    |ui| {
                        draw_title_row(ui, console, viewport.width());
                        draw_separator(ui, viewport.width());
                        let scroll_h = panel_height - ROW_TITLE_HEIGHT - ROW_INPUT_HEIGHT - 2.0;
                        draw_scrollback(ui, console, scroll_h, viewport.width());
                        draw_separator(ui, viewport.width());
                        draw_input_row(ui, console, app_ctx, viewport.width());
                    },
                );
            });
        });
}

fn draw_title_row<Ctx>(ui: &mut egui::Ui, console: &Console<Ctx>, width: f32) {
    ui.allocate_ui_with_layout(
        egui::vec2(width, ROW_TITLE_HEIGHT),
        Layout::left_to_right(egui::Align::Center),
        |ui| {
            ui.add_space(8.0);
            ui.label(
                RichText::new("rye console")
                    .color(COLOR_TITLE)
                    .font(FontId::monospace(FONT_SIZE))
                    .strong(),
            );
            ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
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

            if console.pending_focus {
                response.request_focus();
                console.pending_focus = false;
            } else if !response.has_focus() {
                response.request_focus();
            }

            // Submit on Enter (TextEdit::singleline reports lost_focus on Enter).
            let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
            if response.lost_focus() && enter {
                let line = std::mem::take(&mut console.input);
                console.execute(&line, app_ctx);
                response.request_focus();
            }
        },
    );
}
