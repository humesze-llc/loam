//! Horizontal scrub-bar indicator: a 1D track with a marker at the
//! current value and a label.
//!
//! Designed for "where am I in this 1D parameter range" debug HUDs:
//! the `w` slice plane in a 4D viewer, the current frame in a
//! recorded sequence, the player's depth in a Busemann-coordinate
//! tube, etc. Read-only; for editable scrub use `egui::Slider`.

use egui::{Color32, Pos2, Rect, Response, Sense, Stroke, StrokeKind, Ui, Vec2};

/// Read-only horizontal indicator showing where `value` sits in
/// `range` as a small marker on a track. Optional label drawn at
/// the right edge.
///
/// ```ignore
/// rye_egui::LinearIndicator::new("w_slice", w, -1.5..=1.5)
///     .desired_width(220.0)
///     .show(ui);
/// ```
pub struct LinearIndicator<'a> {
    label: &'a str,
    value: f32,
    range: std::ops::RangeInclusive<f32>,
    desired_width: f32,
    height: f32,
}

impl<'a> LinearIndicator<'a> {
    pub fn new(label: &'a str, value: f32, range: std::ops::RangeInclusive<f32>) -> Self {
        Self {
            label,
            value,
            range,
            desired_width: 220.0,
            height: 14.0,
        }
    }

    pub fn desired_width(mut self, w: f32) -> Self {
        self.desired_width = w;
        self
    }

    pub fn height(mut self, h: f32) -> Self {
        self.height = h;
        self
    }

    pub fn show(self, ui: &mut Ui) -> Response {
        // Allocate enough space for the track plus a small label
        // cell at the right edge so the indicator reads as a single
        // labeled row.
        const LABEL_W: f32 = 78.0;
        let total = Vec2::new(self.desired_width + LABEL_W, self.height.max(14.0));
        let (rect, response) = ui.allocate_exact_size(total, Sense::hover());

        let track = Rect::from_min_size(rect.min, Vec2::new(self.desired_width, self.height));
        let painter = ui.painter();
        let visuals = &ui.style().visuals;

        painter.rect(
            track,
            egui::CornerRadius::same(2),
            visuals.faint_bg_color,
            Stroke::new(1.0, visuals.widgets.noninteractive.bg_stroke.color),
            StrokeKind::Inside,
        );

        // Tick at zero if it's inside the range. Visual cue that
        // the parameter is signed and zero is meaningful.
        let lo = *self.range.start();
        let hi = *self.range.end();
        if lo < 0.0 && hi > 0.0 {
            let t = (-lo) / (hi - lo);
            let x = track.left() + t * track.width();
            let mid_y = track.center().y;
            painter.line_segment(
                [Pos2::new(x, mid_y - 4.0), Pos2::new(x, mid_y + 4.0)],
                Stroke::new(1.0, visuals.weak_text_color()),
            );
        }

        // Marker at value.
        let value_clamped = self.value.clamp(lo, hi);
        let t = if hi > lo {
            (value_clamped - lo) / (hi - lo)
        } else {
            0.5
        };
        let mx = track.left() + t * track.width();
        let my = track.center().y;
        painter.circle_filled(Pos2::new(mx, my), 4.0, Color32::from_rgb(255, 200, 60));
        painter.circle_stroke(
            Pos2::new(mx, my),
            4.0,
            Stroke::new(1.0, visuals.strong_text_color()),
        );

        // Label cell to the right of the track.
        let label_text = format!("{} {:>+.3}", self.label, self.value);
        painter.text(
            Pos2::new(track.right() + 8.0, my),
            egui::Align2::LEFT_CENTER,
            label_text,
            egui::FontId::monospace(11.0),
            visuals.text_color(),
        );

        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Renders without panicking and produces a non-empty response
    /// rect. Sanity check the headless allocation path.
    #[test]
    fn renders_in_central_panel() {
        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0))),
            ..Default::default()
        };
        let mut out_rect = Rect::NOTHING;
        let _ = ctx.run(input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let resp = LinearIndicator::new("w", 0.5, -1.5..=1.5).show(ui);
                out_rect = resp.rect;
            });
        });
        assert!(out_rect.width() > 0.0 && out_rect.height() > 0.0);
    }

    /// Out-of-range values clamp to the track without panicking.
    #[test]
    fn out_of_range_clamps() {
        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0))),
            ..Default::default()
        };
        let _ = ctx.run(input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                LinearIndicator::new("over", 99.0, 0.0..=1.0).show(ui);
                LinearIndicator::new("under", -99.0, 0.0..=1.0).show(ui);
            });
        });
    }
}
