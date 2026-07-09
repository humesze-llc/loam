//! Filmstrip view: one polytope sampled across `w`, `t`, or both (a
//! 2D grid). The grid split lives in
//! `loam_render::Viewport::split_vertical`; this module owns the
//! per-cell controls and the axis-label overlay.

use loam_app::egui;
use loam_physics::polytope::Polytope4;
use loam_render::raymarch::RaymarchShape;

use crate::catalog::render_shape_catalog_menu;
use crate::consts::BODY_SIZE;
use crate::state::Demo;

impl Demo {
    /// Axis labels around the filmstrip grid: w tags on the w-carrying
    /// edge, t tags on the t-carrying edge. The cell nearest offset
    /// zero is highlighted in gold.
    pub(crate) fn render_filmstrip_cell_labels(&mut self, ctx: &egui::Context) {
        let (cols, rows, w_on_cols) = match (self.strip_w, self.strip_t) {
            (true, true) => {
                if self.strip_swap_axes {
                    (self.strip_count_t, self.strip_count_w, false)
                } else {
                    (self.strip_count_w, self.strip_count_t, true)
                }
            }
            (true, false) => (self.strip_count_w, 1, true),
            (false, true) => (1, self.strip_count_t, false),
            (false, false) => return,
        };
        if cols == 0 || rows == 0 {
            return;
        }
        let screen = ctx.content_rect();
        let cell_w_px = screen.width() / cols as f32;
        let cell_h_px = screen.height() / rows as f32;
        let strip_w_extent = BODY_SIZE;

        let label_color = |is_center: bool| {
            if is_center {
                egui::Color32::from_rgb(255, 217, 140)
            } else {
                egui::Color32::from_gray(220)
            }
        };
        let label_frame = egui::Frame::default()
            .fill(egui::Color32::from_black_alpha(160))
            .inner_margin(egui::Margin::symmetric(6, 2))
            .corner_radius(3);

        // w fans symmetrically (center index is current); t fans
        // forward from `rot_time` (index 0 is current).
        let w_axis_label = |i: usize, n: usize| -> (String, bool) {
            let off = if n <= 1 {
                0.0
            } else {
                let t = i as f32 / (n - 1) as f32;
                -strip_w_extent + t * (2.0 * strip_w_extent)
            };
            let mid = if n == 0 { 0 } else { n / 2 };
            (format!("w={:>+.3}", self.w_slice + off), i == mid)
        };
        let t_axis_label = |i: usize, n: usize| -> (String, bool) {
            let off = if n <= 1 {
                0.0
            } else {
                let t = i as f32 / (n - 1) as f32;
                t * self.strip_t_extent
            };
            (format!("t={:.2}s", self.rot_time + off), i == 0)
        };

        // Top edge: column labels.
        for i in 0..cols {
            let center_x = screen.left() + (i as f32 + 0.5) * cell_w_px;
            let (text, is_center) = if w_on_cols {
                w_axis_label(i, cols)
            } else {
                t_axis_label(i, cols)
            };
            let pos = egui::pos2(center_x, screen.top() + 96.0);
            egui::Area::new(egui::Id::new(("strip-col-label", i)))
                .fixed_pos(pos)
                .pivot(egui::Align2::CENTER_TOP)
                .order(egui::Order::Foreground)
                .show(ctx, |ui| {
                    label_frame.show(ui, |ui| {
                        ui.add(egui::Label::new(
                            egui::RichText::new(text)
                                .color(label_color(is_center))
                                .monospace()
                                .size(12.0),
                        ));
                    });
                });
        }
        // Left edge: row labels (only when > 1 row).
        if rows > 1 {
            for j in 0..rows {
                let center_y = screen.top() + (j as f32 + 0.5) * cell_h_px;
                let (text, is_center) = if w_on_cols {
                    t_axis_label(j, rows)
                } else {
                    w_axis_label(j, rows)
                };
                let pos = egui::pos2(screen.left() + 16.0, center_y);
                egui::Area::new(egui::Id::new(("strip-row-label", j)))
                    .fixed_pos(pos)
                    .pivot(egui::Align2::LEFT_CENTER)
                    .order(egui::Order::Foreground)
                    .show(ctx, |ui| {
                        label_frame.show(ui, |ui| {
                            ui.add(egui::Label::new(
                                egui::RichText::new(text)
                                    .color(label_color(is_center))
                                    .monospace()
                                    .size(12.0),
                            ));
                        });
                    });
            }
        }
    }

    /// Single-view body: just the subject picker. Single mode renders
    /// one `strip_subject` with no w/t fan, which is why it exists: the
    /// Schlegel boundary-cell stepper's `cell_count()` bound is well-
    /// defined only against one unambiguous polytope.
    pub(crate) fn render_single_body(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        let heavy = matches!(
            self.strip_subject.shape,
            RaymarchShape::Polytope(Polytope4::Cell120 | Polytope4::Cell600)
        );
        if heavy && self.surface_mode.uses_sdf_for_polychora() {
            ui.colored_label(
                egui::Color32::from_rgb(242, 130, 70),
                "120/600-cell SDFs are heavy; expect <60 fps. Try `surface raster`.",
            );
        }
        ui.horizontal(|ui| {
            let subject_button = ui
                .button(format!("subject: {}", self.strip_subject.label))
                .on_hover_text("Pick the single polytope to inspect");
            egui::Popup::menu(&subject_button).show(|ui| {
                ui.set_min_width(140.0);
                render_shape_catalog_menu(ui, |entry| {
                    self.strip_subject = entry;
                });
            });
            ui.label("Projection + boundary cell live in the Render settings (gear).");
        });
    }

    /// Filmstrip body: catalog subject combo plus per-axis count
    /// DragValues. Carries its own heavy-shape warning since
    /// `render_shapes_section`, which usually shows it, is hidden here.
    pub(crate) fn render_filmstrip_body(&mut self, ui: &mut egui::Ui) {
        let heavy = matches!(
            self.strip_subject.shape,
            RaymarchShape::Polytope(Polytope4::Cell120 | Polytope4::Cell600)
        );
        if heavy {
            ui.colored_label(
                egui::Color32::from_rgb(242, 130, 70),
                "120/600-cell SDFs are heavy; expect <60 fps.",
            );
        }
        // Invariant: at least one of `strip_w`/`strip_t` stays on, so
        // toggling off the last active axis is a no-op.
        ui.horizontal(|ui| {
            let mut w_on = self.strip_w;
            let mut t_on = self.strip_t;
            if ui
                .checkbox(&mut w_on, "w cells")
                .on_hover_text("Sample across w around the slider's value")
                .changed()
                && (w_on || self.strip_t)
            {
                self.strip_w = w_on;
            }
            if ui
                .checkbox(&mut t_on, "t cells")
                .on_hover_text(
                    "Sample across animation time around the t slider; \
                     fans by ±strip_t_extent seconds",
                )
                .changed()
                && (t_on || self.strip_w)
            {
                self.strip_t = t_on;
            }
            if self.strip_w && self.strip_t {
                ui.checkbox(&mut self.strip_swap_axes, "swap axes")
                    .on_hover_text(
                        "Default puts w on columns, t on rows. \
                         Swap to put t on columns, w on rows.",
                    );
            }
        });
        ui.horizontal(|ui| {
            if self.strip_w {
                ui.add(
                    egui::DragValue::new(&mut self.strip_count_w)
                        .range(3..=21)
                        .speed(0.2)
                        .prefix("w: "),
                );
            }
            if self.strip_t {
                ui.add(
                    egui::DragValue::new(&mut self.strip_count_t)
                        .range(3..=21)
                        .speed(0.2)
                        .prefix("t: "),
                );
                ui.add(
                    egui::DragValue::new(&mut self.strip_t_extent)
                        .range(0.1..=10.0)
                        .speed(0.02)
                        .fixed_decimals(2)
                        .suffix("s")
                        .prefix("Δt: "),
                )
                .on_hover_text(
                    "Forward extent of the t fan; cells span \
                     [t, t+Δt] seconds of animation time",
                );
            }
            let subject_button = ui
                .button(format!("subject: {}", self.strip_subject.label))
                .on_hover_text("Pick the polytope rendered in each filmstrip cell");
            egui::Popup::menu(&subject_button).show(|ui| {
                ui.set_min_width(140.0);
                render_shape_catalog_menu(ui, |entry| {
                    self.strip_subject = entry;
                });
            });
        });
    }
}
