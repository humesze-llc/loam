//! Antisymmetric-matrix display of a [`Bivector4`].
//!
//! A 4D bivector maps to the antisymmetric 4×4 `M_ij = B_ij = -M_ji`, with
//! upper-triangle entries the six components in the `e_i ∧ e_j` basis:
//!
//! - `M_01 = xy`, `M_02 = xz`, `M_03 = xw`
//! - `M_12 = yz`, `M_13 = yw`, `M_23 = zw`
//!
//! For an angular-velocity bivector `ω`, this is the operator `dx/dt = ω x`,
//! so the display reads "rate of change of axis-`i` due to motion in plane
//! `(i, j)`," in degrees per unit time.
//!
//! Reference: Hestenes & Sobczyk, *Clifford Algebra to Geometric Calculus*,
//! ch. 1 (2-blade / antisymmetric-tensor correspondence).

use egui::{Grid, Label, Response, RichText, Ui};
use loam_math::{Bivector4, Plane4};

/// Render `b` as a labeled antisymmetric 4×4 matrix, cells in
/// degrees-per-unit-time with a `+5.1` format.
pub fn bivector_matrix(ui: &mut Ui, b: &Bivector4) -> Response {
    Grid::new("loam_egui_bivector_matrix")
        .num_columns(5)
        .spacing([8.0, 2.0])
        .show(ui, |ui| {
            ui.label("");
            for axis in AXIS {
                ui.add(Label::new(RichText::new(axis).monospace().weak()));
            }
            ui.end_row();
            for (row, row_axis) in AXIS.iter().enumerate() {
                ui.add(Label::new(RichText::new(*row_axis).monospace().weak()));
                for col in 0..4 {
                    let text = cell_text(b, row, col);
                    ui.add(Label::new(RichText::new(text).monospace()));
                }
                ui.end_row();
            }
        })
        .response
}

/// Axis labels for the header row and each row's leading cell.
const AXIS: [&str; 4] = ["x", "y", "z", "w"];

/// Text for matrix cell `(row, col)`. Diagonal is `"0"`; off-diagonal is the
/// signed degrees value with the lower triangle negated `M_ji = -M_ij`.
pub fn cell_text(b: &Bivector4, row: usize, col: usize) -> String {
    if row == col {
        "0".to_string()
    } else if row < col {
        format!("{:>+5.1}", upper_pair(b, row, col).to_degrees())
    } else {
        format!("{:>+5.1}", -upper_pair(b, col, row).to_degrees())
    }
}

/// Maps `(row, col)` with `row < col` to its bivector basis component.
fn upper_pair(b: &Bivector4, row: usize, col: usize) -> f32 {
    let plane = match (row, col) {
        (0, 1) => Plane4::Xy,
        (0, 2) => Plane4::Xz,
        (0, 3) => Plane4::Xw,
        (1, 2) => Plane4::Yz,
        (1, 3) => Plane4::Yw,
        (2, 3) => Plane4::Zw,
        _ => unreachable!("upper_pair expects row < col, got ({row}, {col})"),
    };
    b.component(plane)
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Pos2, Rect, Vec2};

    /// `Bivector4` with a single plane set to 1 radian.
    fn pure(plane: Plane4) -> Bivector4 {
        let mut b = Bivector4::ZERO;
        b.set_component(plane, 1.0);
        b
    }

    #[test]
    fn cell_text_diagonal_is_zero() {
        let b = Bivector4::ZERO;
        for i in 0..4 {
            assert_eq!(cell_text(&b, i, i), "0");
        }
    }

    #[test]
    fn cell_text_zero_bivector_off_diagonal_is_signed_zero() {
        let b = Bivector4::ZERO;
        // `{:>+5.1}` shows zero as ` +0.0`; pinned to lock the column width.
        assert_eq!(cell_text(&b, 0, 1), " +0.0");
        assert_eq!(cell_text(&b, 1, 0), " -0.0");
    }

    #[test]
    fn cell_text_pure_xy_plane() {
        let b = pure(Plane4::Xy);
        // 1 rad = 57.295... deg.
        assert_eq!(cell_text(&b, 0, 1), "+57.3");
        assert_eq!(cell_text(&b, 1, 0), "-57.3");
        assert_eq!(cell_text(&b, 0, 2), " +0.0");
        assert_eq!(cell_text(&b, 2, 3), " +0.0");
    }

    #[test]
    fn cell_text_each_basis_plane_lights_correct_cell() {
        let cases = [
            (Plane4::Xy, (0, 1)),
            (Plane4::Xz, (0, 2)),
            (Plane4::Xw, (0, 3)),
            (Plane4::Yz, (1, 2)),
            (Plane4::Yw, (1, 3)),
            (Plane4::Zw, (2, 3)),
        ];
        for (plane, (row, col)) in cases {
            let b = pure(plane);
            assert_eq!(
                cell_text(&b, row, col),
                "+57.3",
                "plane {plane:?} should populate ({row}, {col})",
            );
            assert_eq!(
                cell_text(&b, col, row),
                "-57.3",
                "plane {plane:?} mirror ({col}, {row}) should be negated",
            );
        }
    }

    #[test]
    fn cell_text_small_angle_rounds_to_zero_at_one_decimal() {
        // 0.001 rad = 0.057 deg, rounds to +0.1 at one decimal.
        let mut b = Bivector4::ZERO;
        b.set_component(Plane4::Xy, 0.001);
        assert_eq!(cell_text(&b, 0, 1), " +0.1");
    }

    /// Headless render check: non-empty rect, no panic.
    #[test]
    fn renders_in_central_panel() {
        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0))),
            ..Default::default()
        };
        let mut b = Bivector4::ZERO;
        b.set_component(Plane4::Xy, 0.5);
        b.set_component(Plane4::Zw, 0.3);

        let mut resp_size = Vec2::ZERO;
        let _ = ctx.run(input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let resp = bivector_matrix(ui, &b);
                resp_size = resp.rect.size();
            });
        });
        assert!(resp_size.x > 0.0 && resp_size.y > 0.0);
    }
}
