//! Custom-painted media-player vocabulary buttons: play/pause,
//! skip (×N rate toggle), refresh, plus, chevron.
//!
//! Each is drawn from primitive shapes (triangles, rects, line
//! segments, arcs) rather than font glyphs. egui's default font has
//! patchy coverage of the Mathematical Operators block (circular-
//! arrow code points and several arrow glyphs are missing on most
//! platforms), so a row built from font characters renders
//! inconsistently. Drawing the icons from primitives makes the
//! controls look identical on every platform and avoids depending
//! on a font asset.
//!
//! All sizes are passed in by the caller; this module does not
//! impose a layout. A typical caller pins a row of these to one
//! `(width, height)` so the visual cadence is consistent.

use egui::{
    pos2, vec2, CornerRadius, Pos2, Rect, Response, Sense, Shape, Stroke, StrokeKind, Ui, Vec2,
};

/// Single button that toggles between a play triangle (when
/// `playing == false`) and a pause symbol (two bars, when
/// `playing == true`). Caller reads `.clicked()` on the returned
/// response.
pub fn play_pause_button(ui: &mut Ui, size: Vec2, playing: bool) -> Response {
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    let style = ui.style().interact(&response);
    ui.painter().rect(
        rect,
        CornerRadius::same(3),
        style.bg_fill,
        style.bg_stroke,
        StrokeKind::Inside,
    );
    let color = style.fg_stroke.color;
    let cx = rect.center().x;
    let cy = rect.center().y;
    if playing {
        let bar_w = 4.0;
        let bar_h = 12.0;
        let gap = 3.0;
        let half_gap = gap / 2.0;
        ui.painter().rect_filled(
            Rect::from_min_size(
                pos2(cx - half_gap - bar_w, cy - bar_h / 2.0),
                vec2(bar_w, bar_h),
            ),
            CornerRadius::ZERO,
            color,
        );
        ui.painter().rect_filled(
            Rect::from_min_size(pos2(cx + half_gap, cy - bar_h / 2.0), vec2(bar_w, bar_h)),
            CornerRadius::ZERO,
            color,
        );
    } else {
        let r_h = 7.0;
        let r_w = 8.0;
        let p1 = pos2(cx - r_w * 0.4, cy - r_h);
        let p2 = pos2(cx - r_w * 0.4, cy + r_h);
        let p3 = pos2(cx + r_w * 0.7, cy);
        ui.painter()
            .add(Shape::convex_polygon(vec![p1, p2, p3], color, Stroke::NONE));
    }
    response
}

/// Rate "skip" button drawn as one or two solid triangles.
/// Highlights when `*rate == value`; clicking when already selected
/// resets `rate = 1.0` (lets the user step out of a non-default rate
/// without a global Reset).
///
/// `double = true` paints two adjacent triangles (`<<` / `>>`),
/// `false` paints one (`<` / `>`). `forward = true` points right.
pub fn rate_toggle(
    ui: &mut Ui,
    size: Vec2,
    rate: &mut f32,
    value: f32,
    double: bool,
    forward: bool,
) -> Response {
    let selected = (*rate - value).abs() < 1e-6;
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    let style = ui.style().interact_selectable(&response, selected);
    ui.painter().rect(
        rect,
        CornerRadius::same(2),
        style.bg_fill,
        style.bg_stroke,
        StrokeKind::Inside,
    );
    let color = style.fg_stroke.color;
    let cx = rect.center().x;
    let cy = rect.center().y;
    let r_w = 4.5_f32;
    let r_h = 5.5_f32;
    let triangle_at = |tip_x: f32| -> Vec<Pos2> {
        if forward {
            vec![
                pos2(tip_x - r_w * 0.5, cy - r_h),
                pos2(tip_x - r_w * 0.5, cy + r_h),
                pos2(tip_x + r_w * 0.7, cy),
            ]
        } else {
            vec![
                pos2(tip_x + r_w * 0.5, cy - r_h),
                pos2(tip_x + r_w * 0.5, cy + r_h),
                pos2(tip_x - r_w * 0.7, cy),
            ]
        }
    };
    if double {
        let offset = 4.0;
        ui.painter().add(Shape::convex_polygon(
            triangle_at(cx - offset),
            color,
            Stroke::NONE,
        ));
        ui.painter().add(Shape::convex_polygon(
            triangle_at(cx + offset),
            color,
            Stroke::NONE,
        ));
    } else {
        ui.painter()
            .add(Shape::convex_polygon(triangle_at(cx), color, Stroke::NONE));
    }
    if response.clicked() {
        *rate = if selected { 1.0 } else { value };
    }
    response.on_hover_text(format!("Set rate to ×{value} (click again to reset to ×1)"))
}

/// `+` button painted as two crossed bars on a button-styled rect.
/// Same primitive-shape vocabulary as the play / rate / chevron
/// buttons so a row of them reads as one consistent set of custom-
/// painted controls.
pub fn add_button(ui: &mut Ui, size: Vec2) -> Response {
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    let style = ui.style().interact(&response);
    ui.painter().rect(
        rect,
        CornerRadius::same(2),
        style.bg_fill,
        style.bg_stroke,
        StrokeKind::Inside,
    );
    let cx = rect.center().x;
    let cy = rect.center().y;
    let arm = 5.5_f32;
    let thick = 2.0_f32;
    let color = style.fg_stroke.color;
    ui.painter().rect_filled(
        Rect::from_center_size(pos2(cx, cy), vec2(arm * 2.0, thick)),
        CornerRadius::ZERO,
        color,
    );
    ui.painter().rect_filled(
        Rect::from_center_size(pos2(cx, cy), vec2(thick, arm * 2.0)),
        CornerRadius::ZERO,
        color,
    );
    response
}

/// `R` retry button: a clockwise arc with an arrowhead, painted
/// from primitives. Replaces a font glyph (egui's default font has
/// patchy coverage of the Mathematical Operators block where
/// circular-arrow code points live).
pub fn refresh_button(ui: &mut Ui, size: Vec2) -> Response {
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    let style = ui.style().interact(&response);
    ui.painter().rect(
        rect,
        CornerRadius::same(2),
        style.bg_fill,
        style.bg_stroke,
        StrokeKind::Inside,
    );
    let cx = rect.center().x;
    let cy = rect.center().y;
    let radius = 6.5_f32;
    let stroke = Stroke::new(1.6, style.fg_stroke.color);
    use std::f32::consts::PI;
    let start_angle: f32 = -PI / 2.0 + 0.45;
    let sweep: f32 = PI * 1.55;
    let n = 16;
    let points: Vec<Pos2> = (0..=n)
        .map(|i| {
            let t = i as f32 / n as f32;
            let angle = start_angle + t * sweep;
            pos2(cx + radius * angle.cos(), cy + radius * angle.sin())
        })
        .collect();
    ui.painter().add(Shape::line(points, stroke));
    let arrow_size = 3.5_f32;
    let anchor = pos2(
        cx + radius * start_angle.cos(),
        cy + radius * start_angle.sin(),
    );
    let tan = start_angle - PI / 2.0;
    let perp = tan + PI / 2.0;
    let tip = pos2(
        anchor.x + arrow_size * tan.cos(),
        anchor.y + arrow_size * tan.sin(),
    );
    let base_l = pos2(
        anchor.x + arrow_size * 0.8 * perp.cos(),
        anchor.y + arrow_size * 0.8 * perp.sin(),
    );
    let base_r = pos2(
        anchor.x - arrow_size * 0.8 * perp.cos(),
        anchor.y - arrow_size * 0.8 * perp.sin(),
    );
    ui.painter().add(Shape::convex_polygon(
        vec![tip, base_l, base_r],
        style.fg_stroke.color,
        Stroke::NONE,
    ));
    response
}

/// Allocate a clickable button with a custom-painted up- or down-
/// chevron (`^` / `v`, drawn as two stroked line segments). Used
/// instead of a font glyph so it doesn't depend on the egui font
/// having Mathematical Operators (∧/∨) coverage. Returns the
/// response so the caller can read `.clicked()`.
pub fn chevron_button(ui: &mut Ui, size: Vec2, up: bool, hover: &str) -> Response {
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    let style = ui.style().interact(&response);
    ui.painter().rect(
        rect,
        CornerRadius::same(2),
        style.bg_fill,
        style.bg_stroke,
        StrokeKind::Inside,
    );
    let cx = rect.center().x;
    let cy = rect.center().y;
    let dx = 6.0;
    let dy = 4.0;
    let stroke = Stroke::new(2.0, style.fg_stroke.color);
    if up {
        ui.painter()
            .line_segment([pos2(cx - dx, cy + dy), pos2(cx, cy - dy)], stroke);
        ui.painter()
            .line_segment([pos2(cx + dx, cy + dy), pos2(cx, cy - dy)], stroke);
    } else {
        ui.painter()
            .line_segment([pos2(cx - dx, cy - dy), pos2(cx, cy + dy)], stroke);
        ui.painter()
            .line_segment([pos2(cx + dx, cy - dy), pos2(cx, cy + dy)], stroke);
    }
    response.on_hover_text(hover)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn screen() -> Rect {
        Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(800.0, 600.0))
    }

    /// Centralised click driver: lays out `widget` once to capture
    /// its rect, then drives a press + release in a SECOND frame at
    /// the rect centre and re-runs `widget` so its `Response` sees
    /// the released click. Returns the second-frame response.
    fn click_at_centre<R>(
        mut widget: impl FnMut(&mut Ui) -> Response,
        result: impl Fn(&Response) -> R,
    ) -> R {
        let ctx = egui::Context::default();
        let mut rect = Rect::ZERO;
        let layout_input = egui::RawInput {
            screen_rect: Some(screen()),
            time: Some(0.0),
            ..Default::default()
        };
        let _ = ctx.run(layout_input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                rect = widget(ui).rect;
            });
        });
        let centre = rect.center();
        let mut click_input = egui::RawInput {
            screen_rect: Some(screen()),
            time: Some(0.05),
            ..Default::default()
        };
        click_input.events.push(egui::Event::PointerMoved(centre));
        click_input.events.push(egui::Event::PointerButton {
            pos: centre,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: Default::default(),
        });
        click_input.events.push(egui::Event::PointerButton {
            pos: centre,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: Default::default(),
        });
        let mut out: Option<R> = None;
        let _ = ctx.run(click_input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let resp = widget(ui);
                out = Some(result(&resp));
            });
        });
        out.expect("widget closure ran in click frame")
    }

    #[test]
    fn play_pause_button_click_fires() {
        let clicked = click_at_centre(
            |ui| play_pause_button(ui, vec2(36.0, 29.0), false),
            |r| r.clicked(),
        );
        assert!(
            clicked,
            "play_pause_button should report clicked() after a press+release"
        );
    }

    /// Clicking a `rate_toggle` whose value differs from `rate`
    /// sets `rate = value`. The widget's "selectable" semantics
    /// branch on `(*rate - value).abs() < 1e-6`.
    #[test]
    fn rate_toggle_click_selects_value() {
        let mut rate = 1.0_f32;
        click_at_centre(
            |ui| rate_toggle(ui, vec2(28.0, 29.0), &mut rate, 2.0, false, true),
            |_| (),
        );
        assert_eq!(
            rate, 2.0,
            "click on unselected rate_toggle should set rate to its value"
        );
    }

    /// Clicking an already-selected `rate_toggle` resets `rate` to
    /// 1.0. This is the "step out of a non-default rate without a
    /// global Reset" affordance.
    #[test]
    fn rate_toggle_click_when_selected_resets_to_one() {
        let mut rate = 2.0_f32;
        click_at_centre(
            |ui| rate_toggle(ui, vec2(28.0, 29.0), &mut rate, 2.0, false, true),
            |_| (),
        );
        assert_eq!(
            rate, 1.0,
            "click on selected rate_toggle should reset rate to 1.0"
        );
    }

    #[test]
    fn add_button_click_fires() {
        let clicked = click_at_centre(|ui| add_button(ui, vec2(28.0, 27.0)), |r| r.clicked());
        assert!(clicked);
    }

    #[test]
    fn refresh_button_click_fires() {
        let clicked = click_at_centre(|ui| refresh_button(ui, vec2(28.0, 29.0)), |r| r.clicked());
        assert!(clicked);
    }

    #[test]
    fn chevron_button_click_fires() {
        let clicked = click_at_centre(
            |ui| chevron_button(ui, vec2(28.0, 29.0), true, "tooltip"),
            |r| r.clicked(),
        );
        assert!(clicked);
    }

    /// Allocated rect width / height match the `size` argument
    /// exactly. This is the contract callers depend on for row
    /// layout: a 5-button row with `(28, 29)` per button produces a
    /// predictable total width.
    #[test]
    fn allocated_size_matches_size_argument() {
        let ctx = egui::Context::default();
        let layout_input = egui::RawInput {
            screen_rect: Some(screen()),
            time: Some(0.0),
            ..Default::default()
        };
        let mut sizes: Vec<Vec2> = Vec::new();
        let _ = ctx.run(layout_input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                sizes.push(play_pause_button(ui, vec2(36.0, 29.0), false).rect.size());
                let mut rate = 1.0_f32;
                sizes.push(
                    rate_toggle(ui, vec2(28.0, 29.0), &mut rate, 2.0, false, true)
                        .rect
                        .size(),
                );
                sizes.push(add_button(ui, vec2(28.0, 27.0)).rect.size());
                sizes.push(refresh_button(ui, vec2(28.0, 29.0)).rect.size());
                sizes.push(chevron_button(ui, vec2(28.0, 29.0), true, "").rect.size());
            });
        });
        assert_eq!(sizes[0], vec2(36.0, 29.0));
        assert_eq!(sizes[1], vec2(28.0, 29.0));
        assert_eq!(sizes[2], vec2(28.0, 27.0));
        assert_eq!(sizes[3], vec2(28.0, 29.0));
        assert_eq!(sizes[4], vec2(28.0, 29.0));
    }
}
