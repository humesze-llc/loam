//! Antisymmetric-matrix display of a [`Bivector4`].
//!
//! For a bivector `B` in 4D, the natural matrix representation is the
//! antisymmetric 4×4 with `M_ij = B_ij = -M_ji`, where the upper-
//! triangle entries are the six bivector components in the
//! `e_i ∧ e_j` basis convention:
//!
//! - `M_01 = xy`, `M_02 = xz`, `M_03 = xw`
//! - `M_12 = yz`, `M_13 = yw`, `M_23 = zw`
//! - Diagonal is zero, lower triangle is `-M_ji`.
//!
//! For an angular-velocity bivector `ω`, this matrix is the operator
//! such that `dx/dt = ω x` (treating `x` as a 4-vector), so the
//! display reads "rate of change of axis-`i` due to motion in plane
//! `(i, j)`." Values render in degrees per unit time.
//!
//! Reference: any geometric-algebra textbook on bivectors as
//! infinitesimal rotors. Hestenes & Sobczyk, *Clifford Algebra to
//! Geometric Calculus*, ch. 1, treats the 2-blade / antisymmetric-
//! tensor correspondence directly.

use egui::{Grid, Label, Response, RichText, Ui};
use rye_math::{Bivector4, Plane4};

/// Render `b` as a labeled antisymmetric 4×4 matrix. Cells display
/// degrees-per-unit-time with a `+5.1` format (sign, four digits,
/// one decimal).
pub fn bivector_matrix(ui: &mut Ui, b: &Bivector4) -> Response {
    Grid::new("rye_egui_bivector_matrix")
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

/// Axis labels used for both the column header row and the leading
/// label cell of each row.
const AXIS: [&str; 4] = ["x", "y", "z", "w"];

/// Text rendered in matrix cell `(row, col)`. Public for testing the
/// formatting contract independently of egui's render path. Diagonal
/// is `"0"`; off-diagonal is `"+%.1"` of the signed degrees value;
/// lower triangle inherits the negation `M_ji = -M_ij`.
pub fn cell_text(b: &Bivector4, row: usize, col: usize) -> String {
    if row == col {
        "0".to_string()
    } else if row < col {
        format!("{:>+5.1}", upper_pair(b, row, col).to_degrees())
    } else {
        format!("{:>+5.1}", -upper_pair(b, col, row).to_degrees())
    }
}

/// Upper-triangle accessor: maps `(row, col)` with `row < col` to the
/// corresponding bivector basis component. `(0, 1) -> xy` and so on.
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

    /// Construct a `Bivector4` with a single non-zero plane set to
    /// 1 radian. The matrix should show `+57.3` at the corresponding
    /// upper-triangle cell, `-57.3` at the mirror lower-triangle
    /// cell, and `0` everywhere else.
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
        // The `{:>+5.1}` format shows zero as ` +0.0` (leading space,
        // sign, 0.0). We pin the format to lock the column-width
        // contract that drove the original cell_text design.
        assert_eq!(cell_text(&b, 0, 1), " +0.0");
        assert_eq!(cell_text(&b, 1, 0), " -0.0");
    }

    #[test]
    fn cell_text_pure_xy_plane() {
        let b = pure(Plane4::Xy);
        // 1 rad = 57.295... deg, formatted with one decimal = +57.3.
        assert_eq!(cell_text(&b, 0, 1), "+57.3");
        assert_eq!(cell_text(&b, 1, 0), "-57.3");
        // Other off-diagonal cells stay zero.
        assert_eq!(cell_text(&b, 0, 2), " +0.0");
        assert_eq!(cell_text(&b, 2, 3), " +0.0");
    }

    #[test]
    fn cell_text_each_basis_plane_lights_correct_cell() {
        // Pair-by-pair audit: which (row, col) pair corresponds to
        // each basis plane.
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
        // 0.001 rad = 0.057 deg, rounds to +0.1 at one-decimal precision.
        let mut b = Bivector4::ZERO;
        b.set_component(Plane4::Xy, 0.001);
        // Format produces " +0.1" (space, sign, value).
        assert_eq!(cell_text(&b, 0, 1), " +0.1");
    }

    /// Headless render check: the widget allocates a non-empty rect
    /// and doesn't panic with a representative bivector.
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
