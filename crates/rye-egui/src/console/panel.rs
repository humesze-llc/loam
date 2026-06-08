//! Egui rendering for the console in two modes: docked half-screen drop-down
//! (default) and detached draggable [`egui::Window`].
//!
//! Both modes share the same inner content via [`draw_content`]; only the outer
//! container differs. Docked is an [`egui::Area`] sized to full width and
//! `PANEL_HEIGHT_FRACTION` of viewport height, slid down by
//! `-panel_height * (1.0 - progress)` over [`super::ANIM_DURATION_SECS`]; it is laid
//! out at full height always, so the input row stays pinned during the slide.
//! Detached is an [`egui::Window`] with a custom title row (`title_bar(false)`).
//!
//! Focus: docked re-requests focus every frame (modal-by-design); detached requests
//! only on `pending_focus`, so the user can click out to give keyboard to the app.

use egui::{
    text::{LayoutJob, TextFormat},
    Color32, FontId, Frame, Label, Layout, Margin, Order, RichText, ScrollArea, Sense, Stroke,
    TextEdit, TextWrapMode,
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
/// Tab-completion preview painted after the live input; dim enough to read as a hint.
const COLOR_GHOST: Color32 = Color32::from_rgb(110, 110, 120);
const FONT_SIZE: f32 = 13.0;
const ROW_TITLE_HEIGHT: f32 = 22.0;
const ROW_INPUT_HEIGHT: f32 = 24.0;

/// Default detached-window dimensions for the first detach frame; later frames
/// respect egui's remembered window position/size.
const DETACHED_DEFAULT_W: f32 = 520.0;
const DETACHED_DEFAULT_H: f32 = 320.0;

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

    // Click-outside releases input focus to the app; clicking back inside restores
    // the input row's per-frame focus re-request. Tested against the full panel rect
    // (no animation offset) so a click during the slide doesn't toggle wrongly.
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

    // No explicit min_width / min_height: an enforced minimum plus egui's resize
    // logic drifts the window sideways when the left edge is pulled below it.
    // Letting it go arbitrarily small avoids the bug; the user can resize back up.
    egui::Window::new("rye_console_window")
        .id(egui::Id::new("rye_console_window"))
        .title_bar(false)
        .resizable(true)
        .collapsible(false)
        .movable(true)
        .default_pos(default_pos)
        .default_size(egui::vec2(DETACHED_DEFAULT_W, DETACHED_DEFAULT_H))
        .frame(frame)
        .show(ctx, |ui| {
            // Zero item spacing so the rows sum to exactly `available_height`;
            // otherwise the accumulated gaps push content past the interior and the
            // Window auto-grows each frame. `available_*` is the INNER content area,
            // so sizing off it (not the chrome-inclusive outer rect) avoids the
            // matching horizontal-stretch feedback loop.
            ui.spacing_mut().item_spacing.y = 0.0;
            let width = ui.available_width();
            let scroll_h =
                (ui.available_height() - ROW_TITLE_HEIGHT - ROW_INPUT_HEIGHT - 2.0).max(60.0);
            draw_content(ui, console, app_ctx, width, scroll_h);
        });
}

/// Inner content shared by both modes. `scroll_height` is the pinned scrollback
/// height, pre-computed by each mode from its outer size minus title + input rows.
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
    // Word-wrap relies on the label inheriting the vertical ScrollArea's content
    // width (no `ui.horizontal()`, which leaves the area unbounded and defeats the
    // wrap heuristic) plus an explicit `TextWrapMode::Wrap` (default elides). Side
    // padding comes from a `Frame::inner_margin` so it sits inside the wrap-width
    // constraint. No hanging indent on continuation lines: it breaks copy-paste of
    // code blocks; CPU-side soft-wrap into `LayoutSection`s is the right fix if needed.
    ui.allocate_ui_with_layout(
        egui::vec2(width, height),
        Layout::top_down(egui::Align::Min),
        |ui| {
            ScrollArea::vertical()
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .max_height(height)
                .show(ui, |ui| {
                    Frame::new()
                        .inner_margin(Margin {
                            left: 8,
                            right: 8,
                            top: 4,
                            bottom: 4,
                        })
                        .show(ui, |ui| {
                            // One selectable Label over a multi-color LayoutJob:
                            // drag-select crosses line boundaries and Ctrl-C copies
                            // real text. Per-line Labels broke browser copy (selection
                            // highlighted rasterized pixels with no text behind them).
                            ui.add(
                                Label::new(scrollback_layout_job(&console.history))
                                    .wrap_mode(TextWrapMode::Wrap)
                                    .selectable(true),
                            );
                        });
                });
        },
    );
}

/// Color for a given line kind.
fn line_color(kind: LineKind) -> Color32 {
    match kind {
        LineKind::Input => COLOR_INPUT_ECHO,
        LineKind::Output => COLOR_OUTPUT,
        LineKind::Error => COLOR_ERROR,
        LineKind::System => COLOR_SYSTEM,
    }
}

/// Build one `LayoutJob` for the scrollback: per-line color via per-section
/// `TextFormat`s, default-format `\n` separators between lines. One Label over this
/// job is cross-line selectable and copyable; see `draw_scrollback` for why.
fn scrollback_layout_job(history: &std::collections::VecDeque<HistoryLine>) -> LayoutJob {
    let mut job = LayoutJob::default();
    let font = FontId::monospace(FONT_SIZE);
    let last = history.len().saturating_sub(1);
    for (i, line) in history.iter().enumerate() {
        let fmt = TextFormat {
            font_id: font.clone(),
            color: line_color(line.kind),
            ..Default::default()
        };
        job.append(&line.text, 0.0, fmt);
        if i < last {
            // Default format so the separator never contributes color.
            job.append("\n", 0.0, TextFormat::default());
        }
    }
    job
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

            // Consume Enter before the TextEdit renders so it isn't typed as input.
            // The idiomatic `lost_focus + Enter` never fires here because docked mode
            // re-requests focus every frame, so focus is never "lost"; consume_key
            // sidesteps the focus-state dance.
            let enter_pressed =
                ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Enter));

            let prev_input = console.input.clone();
            let ghost = console.tab_preview();
            // `TextEdit::show` returns the real galley origin so the ghost lands
            // exactly where input ends; a separate layout mis-measured the TextEdit's
            // internal padding by a few pixels.
            let output = TextEdit::singleline(&mut console.input)
                .font(FontId::monospace(FONT_SIZE))
                .frame(false)
                .desired_width(width - 32.0)
                .text_color(COLOR_INPUT_ECHO)
                .show(ui);
            let response = output.response;

            if let Some(ghost) = ghost {
                let ghost_pos = output.galley_pos + egui::vec2(output.galley.size().x, 0.0);
                ui.painter().text(
                    ghost_pos,
                    egui::Align2::LEFT_TOP,
                    ghost,
                    FontId::monospace(FONT_SIZE),
                    COLOR_GHOST,
                );
            }

            // After tab-complete / history nav replaces the buffer, snap the cursor
            // to the end; otherwise it stays at byte 0.
            if console.pending_cursor_to_end {
                let mut state = output.state;
                let end = egui::text::CCursor::new(console.input.chars().count());
                state
                    .cursor
                    .set_char_range(Some(egui::text::CCursorRange::one(end)));
                state.store(ui.ctx(), response.id);
                console.pending_cursor_to_end = false;
            }

            // Any input change outside tab-cycling invalidates tab-completion state.
            if console.input != prev_input {
                console.tab = None;
            }

            // Focus: `pending_focus` is a one-shot request on open (both modes).
            // Docked re-requests every frame to stay modal, unless `user_defocused`
            // (clicked out to the app). Detached leaves focus alone after the initial
            // request so the user can click out.
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

#[cfg(test)]
mod tests {
    //! Structural invariants of `scrollback_layout_job` read off the `sections`
    //! vector: one section per line, per-kind coloring, newline separators between
    //! lines. UI rendering is not exercised.
    use super::*;
    use std::collections::VecDeque;

    fn history(lines: &[(LineKind, &str)]) -> VecDeque<HistoryLine> {
        lines
            .iter()
            .map(|(kind, text)| HistoryLine {
                kind: *kind,
                text: (*text).to_string(),
            })
            .collect()
    }

    #[test]
    fn empty_history_yields_empty_job() {
        let job = scrollback_layout_job(&VecDeque::new());
        assert!(job.text.is_empty());
        assert!(job.sections.is_empty());
    }

    /// One line yields one colored section with no trailing newline.
    #[test]
    fn single_line_emits_one_colored_section() {
        let h = history(&[(LineKind::Output, "hello")]);
        let job = scrollback_layout_job(&h);
        assert_eq!(job.text, "hello");
        assert_eq!(job.sections.len(), 1);
        assert_eq!(job.sections[0].format.color, COLOR_OUTPUT);
    }

    /// N lines yield 2N-1 sections (N lines + N-1 newline separators).
    #[test]
    fn multiple_lines_alternate_with_newline_separators() {
        let h = history(&[
            (LineKind::Input, "> reset"),
            (LineKind::Output, "ok"),
            (LineKind::Error, "boom"),
        ]);
        let job = scrollback_layout_job(&h);
        assert_eq!(job.text, "> reset\nok\nboom");
        assert_eq!(job.sections.len(), 5);
    }

    /// Each LineKind maps to its kind-appropriate section color.
    #[test]
    fn per_line_color_matches_line_kind() {
        let h = history(&[
            (LineKind::Input, "in"),
            (LineKind::Output, "out"),
            (LineKind::Error, "err"),
            (LineKind::System, "sys"),
        ]);
        let job = scrollback_layout_job(&h);
        // Order: input, "\n", output, "\n", error, "\n", system.
        let input_section = &job.sections[0];
        let output_section = &job.sections[2];
        let error_section = &job.sections[4];
        let system_section = &job.sections[6];
        assert_eq!(input_section.format.color, COLOR_INPUT_ECHO);
        assert_eq!(output_section.format.color, COLOR_OUTPUT);
        assert_eq!(error_section.format.color, COLOR_ERROR);
        assert_eq!(system_section.format.color, COLOR_SYSTEM);
    }

    /// All line sections share the monospace font; mixed fonts would visually
    /// split cross-line selection.
    #[test]
    fn all_line_sections_share_monospace_font() {
        let h = history(&[
            (LineKind::Input, "a"),
            (LineKind::Output, "b"),
            (LineKind::Error, "c"),
        ]);
        let job = scrollback_layout_job(&h);
        let expected_font = FontId::monospace(FONT_SIZE);
        // Sections 0, 2, 4 are line content; 1, 3 are newlines.
        for idx in [0, 2, 4] {
            assert_eq!(
                job.sections[idx].format.font_id, expected_font,
                "section {idx} (line content) should be monospace",
            );
        }
    }
}
