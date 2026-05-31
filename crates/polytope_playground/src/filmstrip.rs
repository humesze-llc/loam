//! Filmstrip view: one polytope sampled across an axis of `w`, an
//! axis of `t`, or both at once (a 2D grid). The grid math itself
//! lives in `rye_render::Viewport::split_vertical`; this module
//! handles the demo's per-cell parameter controls (axis toggles,
//! counts, t-extent, subject picker) and the per-cell axis-label
//! overlay drawn on top of the rendered scene.

use rye_app::egui;
use rye_physics::polytope::Polytope4;
use rye_render::raymarch::RaymarchShape;

use crate::catalog::render_shape_catalog_menu;
use crate::consts::BODY_SIZE;
use crate::state::Demo;

impl Demo {
    /// Render axis labels around the filmstrip grid. Top edge
    /// gets w-value tags above each column (whichever axis
    /// carries w); left edge gets t-offset tags beside each row
    /// (whichever axis carries t). The cell whose offset along
    /// each axis is closest to zero is highlighted in active-set
    /// warning gold. For 1D cases the orthogonal-axis labels
    /// are omitted (just one row or one column).
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

        // Per-axis cell label + center-cell flag. `axis_label`
        // computes the (text, is_current) pair: w cells fan
        // symmetrically around the slider so the center index
        // is "current"; t cells fan FORWARD from the current
        // `rot_time`, so index 0 is "current" and the rest are
        // future predictions.
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

    /// Single-view body: just the subject picker. Single mode renders exactly
    /// one shape (`strip_subject`, shared with the filmstrip) with the full
    /// surface / wireframe / projection / points stack, so this body needs no
    /// w/t-axis fan controls. The subject is chosen from the same catalog menu
    /// the filmstrip and shape-row `+` button use, so a user who picks a
    /// 120/600-cell here gets the same heavy-SDF hint they would in those views.
    ///
    /// The Schlegel boundary-cell stepper (in the Render modal) reads this
    /// subject's `cell_count()` for its upper bound, which is the whole reason
    /// Single mode exists: the cell index is well-defined only against one
    /// unambiguous polytope.
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
            // Same catalog menu as the filmstrip subject picker + the shape-row
            // `+` button, so the visuals (nested category submenus, hover names)
            // stay identical across every shape-selection surface.
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

    /// Filmstrip body: subject combo (over the catalog, so the
    /// user can pick any of the shipped polytopes independent of
    /// `self.row`) plus per-axis count DragValues. Heavy-shape
    /// warning surfaces here when the subject is 120/600-cell
    /// since `render_shapes_section` (where the warning otherwise
    /// lives) is hidden in this view.
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
        // Row 1: axis toggles + (when both are on) the swap.
        // Invariant: at least one of `strip_w` / `strip_t` must
        // be on. Clicking the on-toggle while the other is off
        // is a no-op (visual checkbox stays checked).
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
        // Row 2: counts + t-extent + subject combo.
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
            // Same Popup::menu pattern as the `+` shape menu in
            // the shape row, so the subject picker has identical
            // visuals (nested category submenus).
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
