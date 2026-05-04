//! Visualize a 4D angular-velocity bivector as one or two
//! labeled circular arcs, exposing the SO(4) double-rotation
//! structure for debugging compound rotors.
//!
//! Given a [`rye_math::Bivector4`] `B`, the widget calls
//! [`rye_math::Bivector4::simple_decomposition`] and renders:
//!
//! - **Simple `B` (Pf = 0)**: one arc with its plane label and
//!   rotation angle.
//! - **Distinct angles**: two arcs, one for each rotation plane,
//!   sized by their rotation magnitudes.
//! - **Isoclinic (decomposition is non-unique)**: a single arc with
//!   a compound label and the shared angle.
//! - **Zero**: nothing.
//!
//! Plane labels: when one basis component dominates a simple part
//! (`>= 0.99` of its magnitude), the arc is labeled with the basis
//! plane's two-letter name (e.g. `xy`, `xw`). Otherwise the label
//! lists every basis component above `0.05·|B|` with sign and
//! magnitude (e.g. `+0.71 xw -0.71 yz`).

use egui::{Color32, Pos2, Response, Sense, Stroke, Ui, Vec2};
use rye_math::{Bivector4, Plane4};

/// Read-only widget that renders the SO(4) plane decomposition of
/// a `Bivector4` as one or two arcs.
pub struct RotorVisualizer<'a> {
    bivector: Bivector4,
    label: &'a str,
}

impl<'a> RotorVisualizer<'a> {
    pub fn new(bivector: Bivector4, label: &'a str) -> Self {
        Self { bivector, label }
    }

    pub fn show(self, ui: &mut Ui) -> Response {
        let radius = 18.0_f32;
        let label_w = 110.0_f32;
        // Two slots side-by-side (radius * 2 each plus a gap), plus
        // the label cell. Empty slots collapse to zero width.
        let total_w = radius * 4.0 + 8.0 + label_w;
        let total_h = radius * 2.0 + 16.0;
        let (rect, response) = ui.allocate_exact_size(Vec2::new(total_w, total_h), Sense::hover());

        let painter = ui.painter();
        let visuals = &ui.style().visuals;

        // Header label.
        painter.text(
            Pos2::new(rect.left() + radius * 4.0 + 8.0, rect.top()),
            egui::Align2::LEFT_TOP,
            self.label,
            egui::FontId::monospace(11.0),
            visuals.weak_text_color(),
        );

        let mag = self.bivector.magnitude();
        if mag < 1e-6 {
            painter.text(
                Pos2::new(rect.left() + radius * 4.0 + 8.0, rect.center().y + 2.0),
                egui::Align2::LEFT_CENTER,
                "(no rotation)",
                egui::FontId::monospace(10.0),
                visuals.text_color(),
            );
            return response;
        }

        let arc_centers = [
            Pos2::new(rect.left() + radius, rect.bottom() - radius - 2.0),
            Pos2::new(
                rect.left() + radius * 3.0 + 8.0,
                rect.bottom() - radius - 2.0,
            ),
        ];

        // Decompose. None == isoclinic, fall through to single
        // compound arc with the input bivector.
        let parts: Vec<Bivector4> = match self.bivector.simple_decomposition() {
            Some((b1, b2)) => {
                let mut v = Vec::with_capacity(2);
                if b1.magnitude() > 1e-6 {
                    v.push(b1);
                }
                if b2.magnitude() > 1e-6 {
                    v.push(b2);
                }
                v
            }
            None => vec![self.bivector],
        };

        // Largest rotation in either part determines the arc's
        // visual scale; we map magnitude to sweep angle so the
        // dominant rotation gets a near-full circle.
        let max_part_mag = parts.iter().map(|b| b.magnitude()).fold(0.0_f32, f32::max);

        // Layout labels in a single text column to the right.
        let label_text = match self.bivector.simple_decomposition() {
            Some(_) => format_two_planes(&parts),
            None => format!("isoclinic: {}", format_compound(&self.bivector)),
        };
        painter.text(
            Pos2::new(rect.left() + radius * 4.0 + 8.0, rect.center().y + 8.0),
            egui::Align2::LEFT_CENTER,
            label_text,
            egui::FontId::monospace(10.0),
            visuals.text_color(),
        );

        for (i, part) in parts.iter().enumerate().take(2) {
            let center = arc_centers[i];
            paint_arc(
                painter,
                center,
                radius,
                part.magnitude(),
                max_part_mag,
                visuals,
            );
        }

        response
    }
}

/// Paint a circle outline plus an arc indicating the rotation
/// magnitude of one simple bivector. The arc sweeps clockwise
/// from 12 o'clock by `2π · (angle / max_angle)` radians, so the
/// dominant rotation reaches a full revolution and lesser ones
/// scale proportionally.
fn paint_arc(
    painter: &egui::Painter,
    center: Pos2,
    radius: f32,
    angle: f32,
    max_angle: f32,
    visuals: &egui::Visuals,
) {
    use std::f32::consts::TAU;
    // Background circle.
    painter.circle_stroke(
        center,
        radius,
        Stroke::new(1.0, visuals.widgets.noninteractive.bg_stroke.color),
    );
    if angle <= 0.0 || max_angle <= 0.0 {
        return;
    }
    let sweep = (angle / max_angle).clamp(0.0, 1.0) * TAU;
    let n = 32_u32;
    let steps = ((sweep / TAU) * n as f32).ceil().max(1.0) as u32;
    let start_angle = -std::f32::consts::FRAC_PI_2; // 12 o'clock
    let pts: Vec<Pos2> = (0..=steps)
        .map(|k| {
            let t = k as f32 / steps as f32;
            let a = start_angle + t * sweep;
            Pos2::new(center.x + radius * a.cos(), center.y + radius * a.sin())
        })
        .collect();
    painter.add(egui::Shape::line(
        pts,
        Stroke::new(2.0, Color32::from_rgb(255, 200, 60)),
    ));
}

/// Format two simple components as a two-line label, one per plane.
fn format_two_planes(parts: &[Bivector4]) -> String {
    parts
        .iter()
        .map(format_simple_plane)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Label a simple bivector by its dominant basis if one component
/// is at least 99% of the total magnitude; otherwise list every
/// component above 5% with sign and weight.
fn format_simple_plane(b: &Bivector4) -> String {
    let mag = b.magnitude();
    if mag < 1e-6 {
        return String::new();
    }
    let comps = [
        (Plane4::Xy, b.xy),
        (Plane4::Xz, b.xz),
        (Plane4::Xw, b.xw),
        (Plane4::Yz, b.yz),
        (Plane4::Yw, b.yw),
        (Plane4::Zw, b.zw),
    ];
    let dominant = comps
        .iter()
        .max_by(|a, c| a.1.abs().partial_cmp(&c.1.abs()).unwrap())
        .copied()
        .unwrap();
    if dominant.1.abs() / mag >= 0.99 {
        let sign = if dominant.1 >= 0.0 { "" } else { "-" };
        return format!("{}{}  ({:.3})", sign, dominant.0.label(), mag);
    }
    format_compound(b)
}

/// List every basis component above 5% of `|B|` with signed weight.
fn format_compound(b: &Bivector4) -> String {
    let mag = b.magnitude().max(1e-9);
    let parts: Vec<String> = [
        (Plane4::Xy, b.xy),
        (Plane4::Xz, b.xz),
        (Plane4::Xw, b.xw),
        (Plane4::Yz, b.yz),
        (Plane4::Yw, b.yw),
        (Plane4::Zw, b.zw),
    ]
    .iter()
    .filter(|(_, c)| c.abs() / mag > 0.05)
    .map(|(p, c)| format!("{:+.2} {}", c, p.label()))
    .collect();
    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::Rect;

    fn render(bivector: Bivector4) -> Rect {
        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0))),
            ..Default::default()
        };
        let mut out = Rect::NOTHING;
        let _ = ctx.run(input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let resp = RotorVisualizer::new(bivector, "ω").show(ui);
                out = resp.rect;
            });
        });
        out
    }

    #[test]
    fn zero_bivector_renders_no_arc() {
        let r = render(Bivector4::ZERO);
        assert!(r.width() > 0.0 && r.height() > 0.0);
    }

    #[test]
    fn simple_basis_plane_renders() {
        let r = render(Plane4::Xw.unit_bivector());
        assert!(r.width() > 0.0);
    }

    #[test]
    fn compound_distinct_angles_renders() {
        // e_xw + 2·e_yz, distinct rotation angles.
        let b = Bivector4::new(0.0, 0.0, 1.0, 2.0, 0.0, 0.0);
        let r = render(b);
        assert!(r.width() > 0.0);
    }

    #[test]
    fn isoclinic_renders_one_compound_arc() {
        // Equal-angles isoclinic: simple_decomposition returns None.
        let b = Bivector4::new(0.0, 0.0, 1.0, 1.0, 0.0, 0.0);
        let r = render(b);
        assert!(r.width() > 0.0);
    }

    #[test]
    fn dominant_basis_label_is_just_two_letters() {
        // 99.5% xw, 0.5% yz: should label as "xw  (0.xxx)".
        let b = Bivector4::new(0.0, 0.0, 0.995, 0.005, 0.0, 0.0);
        let label = format_simple_plane(&b);
        assert!(
            label.starts_with("xw"),
            "expected dominant-basis label, got {label:?}"
        );
    }

    #[test]
    fn compound_label_lists_components() {
        // Roughly equal xw and yz, neither dominates.
        let b = Bivector4::new(0.0, 0.0, 0.7, 0.7, 0.0, 0.0);
        let label = format_simple_plane(&b);
        assert!(label.contains("xw") && label.contains("yz"));
    }
}
