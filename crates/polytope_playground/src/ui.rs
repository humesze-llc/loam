//! Cross-cutting overlay UI: top menu bar, help window, the
//! `BottomOverlay` (rotation tabs, mode-specific body dispatcher,
//! always-visible w/t sliders, rate row), and the deferred-mutation
//! drain that fires after the overlay's two-pass measure-then-render
//! finishes.
//!
//! The mode-specific bodies (active / composer / filmstrip / shapes)
//! and the formula popup live in their own modules; this file owns the
//! chrome that wraps them.

use rye_app::egui;
use rye_egui::{
    media::{chevron_button, play_pause_button, rate_toggle, refresh_button},
    slider_with_edit,
};
use rye_math::Rotor4;

use crate::consts::{CONTROL_H, CONTROL_W, PLAY_PAUSE_W};
use crate::state::{
    hyperslice_cull_active, DeferredAction, Demo, RotationMode, RotorTerm, SectionLayer,
    SurfaceMode, ViewMode, WireframeColorMode, WireframeProjection,
};

/// Render one [`SectionLayer`]'s controls: a perimeter-outline checkbox and a
/// fill-alpha slider whose `0` end is the layer's off state. `title` names the
/// layer; `tooltip` explains what it shows. The perimeter outline only draws in
/// the wireframe overlay, so its checkbox tooltip says so and notes that the
/// overlay must be on (the fill draws in Raster mode regardless).
///
/// Free function (not an `impl Demo` method) so it can take a `&mut SectionLayer`
/// destructured out of `Demo` in the render-panel closure without re-borrowing
/// the whole `Demo`.
fn section_layer_controls(
    ui: &mut egui::Ui,
    title: &str,
    tooltip: &str,
    layer: &mut SectionLayer,
    wireframe_enabled: bool,
) {
    ui.label(egui::RichText::new(title).italics())
        .on_hover_text(tooltip);
    let perimeter_resp = ui
        .checkbox(&mut layer.perimeter, "Perimeter outline")
        .on_hover_text(
            "Cap-boundary outline. Draws in the wireframe overlay, so enable Wireframe to see it.",
        );
    if layer.perimeter && !wireframe_enabled {
        perimeter_resp
            .on_hover_text("Wireframe overlay is off, so the outline is not currently drawn.");
    }
    ui.horizontal(|ui| {
        ui.label("Fill alpha");
        // 0 is the off state; the slider spans the full [0, 1] so the user can
        // drag a layer off or to any visible opacity. The fill draws in Raster
        // surface mode; below 1.0 it composites through the no-depth-write
        // pipeline so layers behind show through.
        ui.add(egui::Slider::new(&mut layer.surface_alpha, 0.0..=1.0).fixed_decimals(2))
            .on_hover_text("0 = off; below 1.0 composites translucently over the layers behind");
    });
}

impl Demo {
    /// Expanded section of the bottom overlay. Two tab rows
    /// stacked vertically:
    ///
    /// 1. **View tabs** (Shapes / Single / Filmstrip): top-level
    ///    visual demo. Shapes shows the multi-shape row; Single
    ///    shows one subject with the full projection stack (the
    ///    mode Schlegel's cell-index needs); Filmstrip shows one
    ///    shape across N w-slices.
    /// 2. **Rotation tabs** (Active set / Composer): how the
    ///    rotor evolves. Independent of view mode.
    ///
    /// Always-visible controls (Spin/Pause, rate buttons,
    /// sliders) live below this in `render_overlay`.
    pub(crate) fn render_expanded_body(&mut self, ui: &mut egui::Ui) {
        self.render_view_tab_row(ui);
        match self.view_mode {
            ViewMode::Shapes => self.render_shapes_section(ui),
            ViewMode::Single => self.render_single_body(ui),
            ViewMode::Filmstrip => self.render_filmstrip_body(ui),
        }
        ui.separator();
        self.render_rotation_tab_row(ui);
        if self.rotation_mode == RotationMode::Active {
            self.render_active_mode(ui);
        } else {
            self.render_composer_mode(ui);
        }
    }

    /// Top tab row of the expanded body: visual demo selector.
    /// Shapes (multi-shape side-by-side row), Single (one subject
    /// with the full projection stack), Filmstrip (one shape
    /// across multiple w-slices). Tab change is staged into
    /// `pending_view_mode` for the same `BottomOverlay` two-pass
    /// reason as `pending_mode`.
    pub(crate) fn render_view_tab_row(&mut self, ui: &mut egui::Ui) {
        let mut staged = self.view_mode;
        ui.horizontal(|ui| {
            ui.selectable_value(&mut staged, ViewMode::Shapes, "Shapes")
                .on_hover_text("Side-by-side row of shapes at one w-slice");
            ui.selectable_value(&mut staged, ViewMode::Single, "Single")
                .on_hover_text(
                    "One subject (the picker below) at one w-slice with the full \
                     surface / wireframe / projection stack. Schlegel's boundary-cell \
                     selection needs this unambiguous single shape.",
                );
            ui.selectable_value(&mut staged, ViewMode::Filmstrip, "Filmstrip")
                .on_hover_text(
                    "One shape rendered N times across w-slices fanning out by \
                     ±BODY_SIZE around the w slider's value",
                );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.checkbox(&mut self.show_formula, "Show formula")
                    .on_hover_text("Top-right popup with the live exp(...) form of the rotor");
            });
        });
        if staged != self.view_mode {
            self.pending_view_mode = Some(staged);
        }
    }

    /// Rotation-mode tabs: which source drives `omega`. The tab
    /// change is staged into `self.pending_mode` rather than
    /// applied directly so `BottomOverlay`'s two-pass measure-
    /// then-render captures the same body height in both passes.
    pub(crate) fn render_rotation_tab_row(&mut self, ui: &mut egui::Ui) {
        let mut staged = self.rotation_mode;
        ui.horizontal(|ui| {
            ui.selectable_value(&mut staged, RotationMode::Active, "Active set")
                .on_hover_text("Six checkbox-toggled bivectors (xy, xz, ...)");
            ui.selectable_value(&mut staged, RotationMode::Composer, "Composer")
                .on_hover_text("Sum of bivectors from the composed sequence");
        });
        if staged != self.rotation_mode {
            self.pending_mode = Some(staged);
        }
    }

    /// Top menu bar: Edit / View. Always visible.
    ///
    /// The File menu is intentionally absent until persistence
    /// and `Quit` via `ViewportCommand::Close` are wired.
    pub(crate) fn render_menu_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("polytope-playground-menu-bar").show(ctx, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("Edit", |ui| {
                    if ui.button("Reset orientation").clicked() {
                        self.rot_state = Rotor4::IDENTITY;
                        self.write_all(self.rot_state);
                        ui.close_kind(egui::UiKind::Menu);
                    }
                    if ui
                        .add(egui::Button::new("Reset all").shortcut_text("R"))
                        .clicked()
                    {
                        self.reset();
                        ui.close_kind(egui::UiKind::Menu);
                    }
                });
                rye_egui::sticky_menu(ui, "View", |ui| {
                    // Sticky toggles: clicking each checkbox does NOT close the dropdown
                    // (per `sticky_menu`'s `CloseOnClickOutside` semantics), so the user
                    // can flip multiple visibility flags without reopening.
                    ui.checkbox(&mut self.show_controls, "Rotation controls (H)");
                    ui.checkbox(&mut self.show_formula, "Formula popup");
                    ui.checkbox(&mut self.example_callout.open, "Example callout");
                    ui.checkbox(&mut self.mode_annotation_open.open, "Mode annotation")
                        .on_hover_text(
                            "Floating note explaining the active projection / space \
                             mode. Appears only for a non-default projection or with \
                             Curvature turned up.",
                        );
                    ui.separator();
                    // One-shot action: opens the About window and the menu should fold
                    // away. `Popup::close_all(ctx)` cooperates with the sticky-popup
                    // default to give the user an "explicit close" path for entries that
                    // aren't sticky toggles.
                    if ui.button("About this program").clicked() {
                        self.show_help = true;
                        egui::Popup::close_all(ui.ctx());
                    }
                });
            });
        });
    }

    /// Floating `Render` settings modal. Surfaces the same toggles the console exposes
    /// (`surface`, `wireframe`, `wireframe points`) so new readers can discover the
    /// rendering modes without typing commands. Each control writes through the same Demo
    /// fields the console handlers do; the two interfaces stay in lockstep automatically.
    ///
    /// Off by default; opened via the gear button in the bottom overlay. Hosted in
    /// [`rye_egui::floating_panel`] for consistency with the engine's other floating
    /// surfaces (the help modal migrates to the same primitive in the same sprint).
    pub(crate) fn render_render_panel(&mut self, ctx: &egui::Context) {
        // Snapshot fields the panel can mutate so we can detect a surface-mode change and
        // call `rebuild_bodies()` AFTER the panel closes its borrow. Doing the rebuild
        // inside the closure would need `&mut self` while `&mut self.show_render_panel`
        // is still active.
        let prev_surface = self.surface_mode;
        // Snapshot the projection so we can detect a Schlegel select / cell-index
        // change after the panel's borrow ends and re-resolve the cache (the same
        // deferred-mutation shape `prev_surface` uses for `rebuild_bodies`).
        let prev_projection = self.wireframe_projection;
        // Computed BEFORE the destructure so the closure can read it as a captured value
        // (the destructure exclusively borrows `self.row`).
        let sdf_disabled = self.sdf_blocked_by_heavy_polychora();
        // Cell count of the polytope a Schlegel cell index refers to (the leading
        // polychoron). Captured for the cell-index DragValue's clamp. `None` when
        // the row has no polychoron, in which case the stepper is disabled.
        let schlegel_cell_count = self.schlegel_subject().map(|p| p.cell_count() as u32);
        // Destructure-borrow the fields the panel writes so the closure doesn't capture
        // a whole `&mut self`. This sidesteps the borrow conflict with `show_render_panel`
        // and lets the closure remain a plain `FnOnce(&mut Ui)`.
        let Self {
            show_render_panel,
            surface_mode,
            cross_section,
            projected_cap,
            wireframe_enabled,
            wireframe_nearest_active,
            wireframe_color_mode,
            wireframe_projection,
            wireframe_hyperslice,
            wireframe_hyperslice_thickness,
            space_blend,
            points_enabled,
            points_show_vertices,
            points_show_cell_centers,
            points_size_px,
            ..
        } = self;
        rye_egui::floating_panel(
            ctx,
            "polytope-playground-render",
            "Render",
            show_render_panel,
            |ui| {
                ui.label(egui::RichText::new("Surface").strong());
                ui.radio_value(surface_mode, SurfaceMode::Raster, "Raster (default)");
                // SDF disabled when the row contains a 120-cell or 600-cell. Those
                // SDF kernels overrun the browser's WebGPU shader budget and crash
                // the tab; the user has to remove the heavy polychora first. The
                // disabled radio surfaces the reason via tooltip so they're not
                // wondering why the option grayed out.
                ui.add_enabled_ui(!sdf_disabled, |ui| {
                    let resp = ui.radio_value(surface_mode, SurfaceMode::Sdf, "SDF raymarch");
                    if sdf_disabled {
                        resp.on_disabled_hover_text(
                            "Disabled: 120-cell/600-cell SDFs crash the browser tab. \
                             Remove the heavy polychora to re-enable.",
                        );
                    }
                });
                ui.radio_value(surface_mode, SurfaceMode::Off, "Off");
                ui.separator();

                // Cross-section layers. The slice renders as two overlaid layers
                // in one viewport: the HONEST cross-section (drop-w, never
                // reprojected; what the SDF shows) and the PROJECTED cap (the
                // slice reprojected through the active wireframe projection). Each
                // has its own perimeter outline + fill alpha. Fills draw in Raster
                // mode; perimeter outlines draw in the wireframe overlay, so they
                // gate on `wireframe_enabled` and the tooltips say so.
                ui.label(egui::RichText::new("Cross-section").strong());
                section_layer_controls(
                    ui,
                    "Honest (drop-w)",
                    "The drop-w slice, never reprojected: the geometry the SDF \
                     shows. On by default so a projection change never distorts \
                     the slice.",
                    cross_section,
                    *wireframe_enabled,
                );
                section_layer_controls(
                    ui,
                    "Projected cap",
                    "The same slice reprojected through the active wireframe \
                     projection, so it can sit on a Schlegel / stereographic \
                     wireframe. Off by default.",
                    projected_cap,
                    *wireframe_enabled,
                );
                ui.separator();

                ui.label(egui::RichText::new("Wireframe").strong());
                ui.checkbox(wireframe_enabled, "Enabled");
                ui.add_enabled_ui(*wireframe_enabled, |ui| {
                    ui.checkbox(wireframe_nearest_active, "Nearest-active gradient");
                    ui.horizontal(|ui| {
                        ui.label("Color");
                        for mode in WireframeColorMode::ALL {
                            ui.radio_value(wireframe_color_mode, mode, mode.label());
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Projection");
                        for mode in WireframeProjection::ALL {
                            // Compare by VARIANT, not full `PartialEq`: a Schlegel
                            // cell-index change must keep the Schlegel button
                            // selected. Selecting Schlegel preserves the current
                            // cell index (the contextual stepper below owns it).
                            let selected = wireframe_projection.same_variant(mode);
                            if ui.radio(selected, mode.label()).clicked() && !selected {
                                *wireframe_projection = match (mode, *wireframe_projection) {
                                    (
                                        WireframeProjection::Schlegel { .. },
                                        WireframeProjection::Schlegel { cell_index },
                                    ) => WireframeProjection::Schlegel { cell_index },
                                    (m, _) => m,
                                };
                            }
                        }
                    });
                    // Contextual parameter row: only the active mode's knob shows.
                    // Schlegel -> cell-index stepper (clamped to the leading
                    // polytope's cell count). Other modes need no contextual
                    // param; Hyperslice's slab width lives with the cull toggle
                    // below, so it is not duplicated here.
                    if let WireframeProjection::Schlegel { cell_index } = wireframe_projection {
                        ui.horizontal(|ui| {
                            ui.label("Boundary cell");
                            match schlegel_cell_count {
                                Some(count) => {
                                    ui.add(
                                        egui::DragValue::new(cell_index)
                                            .range(0..=count - 1)
                                            .speed(0.25),
                                    );
                                    ui.label(format!("of {count}"));
                                }
                                None => {
                                    ui.add_enabled(false, egui::DragValue::new(cell_index))
                                        .on_disabled_hover_text(
                                            "No polychoron in the row to project",
                                        );
                                }
                            }
                        });
                    }
                    // Hyperslice cull: thin the edge graph to a w-slab around the
                    // current slice. The thickness stepper is enabled whenever the
                    // cull is running, which includes the Hyperslice PROJECTION
                    // mode (it resolves to drop-w and turns the cull on without the
                    // standalone toggle). Gating on the toggle alone left the slab
                    // width greyed out under that projection even though the cull
                    // was active.
                    ui.checkbox(wireframe_hyperslice, "Hyperslice cull (w-slab)");
                    let cull_active =
                        hyperslice_cull_active(*wireframe_hyperslice, *wireframe_projection);
                    ui.add_enabled_ui(cull_active, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Slab width");
                            ui.add(
                                egui::DragValue::new(wireframe_hyperslice_thickness)
                                    .range(
                                        crate::consts::HYPERSLICE_MIN_THICKNESS
                                            ..=2.0 * crate::consts::W_RANGE,
                                    )
                                    .speed(0.01),
                            );
                        });
                    });
                });
                // Curvature is a DIFFERENT axis from Projection above. The
                // Projection radio picks the 4D->R³ map (where a vertex lands
                // on screen); Curvature morphs the EDGE GEOMETRY between a flat
                // R⁴ chord and an S³ great-circle arc (whether the line between
                // two landed vertices is straight or bowed). They compose, but
                // conflating them confuses, so the slider sits in its own
                // separator-delimited group below the projection knob. Writes
                // the same `space_blend` field the `wireframe space` /
                // top-level `space` console command drives, clamped to [0, 1].
                //
                // Deliberately OUTSIDE the `wireframe_enabled` enable-gate the
                // other wireframe controls sit in: the morph is only visible in
                // the wireframe layer, but gating the slider behind that toggle
                // is the same dead-end the bare `space spherical` command had
                // (mutate a field nothing draws). A user reaching for the
                // Curvature slider IS asking to see curved edges, so dragging it
                // above zero auto-enables the wireframe, mirroring the console
                // handler. Holding it at zero never flips the overlay on.
                ui.separator();
                ui.label(egui::RichText::new("Curvature (flat <-> spherical)").strong());
                ui.horizontal(|ui| {
                    ui.label("Blend");
                    if ui
                        .add(egui::Slider::new(space_blend, 0.0..=1.0).fixed_decimals(2))
                        .changed()
                        && *space_blend > 0.0
                    {
                        *wireframe_enabled = true;
                    }
                });
                ui.separator();

                ui.label(egui::RichText::new("Points").strong());
                ui.checkbox(points_enabled, "Enabled");
                ui.add_enabled_ui(*points_enabled, |ui| {
                    ui.checkbox(points_show_vertices, "Vertex markers");
                    ui.checkbox(points_show_cell_centers, "Cell centers");
                    ui.horizontal(|ui| {
                        ui.label("Size (px)");
                        ui.add(
                            egui::DragValue::new(points_size_px)
                                .range(1.0..=32.0)
                                .speed(0.25),
                        );
                    });
                });
            },
        );
        // The destructure-borrow's lifetime ended above; safe to call `&mut self`
        // methods again. Replay the console handler's `rebuild_bodies()` whenever the
        // user flipped surface mode through the panel, so the SDF kernel's body list
        // stays in sync with the new mode (`BodyKind::Invalid` for inert polychora).
        if self.surface_mode != prev_surface {
            self.rebuild_bodies();
        }
        // Re-resolve the cached Schlegel face-plane params if the user changed the
        // projection mode or dragged the cell-index stepper through the panel.
        // Deferred to here (same shape as the surface-mode rebuild) because the
        // resolve runs the LazyLock cell-table fit on first access and must not be
        // called inside the destructure-borrowed closure (it needs `&mut self`).
        if self.wireframe_projection != prev_projection {
            self.resolve_schlegel_cache();
        }
    }

    pub(crate) fn render_help_window(&mut self, ctx: &egui::Context) {
        rye_egui::floating_panel_builder(
            ctx,
            "polytope-playground-about",
            "About Polytope Playground",
            &mut self.show_help,
        )
        .resizable(true)
        .collapsible(false)
        .default_size(560.0, 460.0)
        .default_pos(egui::pos2(80.0, 80.0))
        .show(|ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("What this program shows");
                ui.label(
                    "You're looking at 3D cross-sections of four-dimensional \
                         polytopes. As they rotate through 4D space their cross-\
                         sections morph in characteristic ways; the point of the \
                         demo is to make 4D shape intuition reachable from 3D.",
                );
                ui.add_space(8.0);

                ui.heading("3D cross-sections, briefly");
                ui.label(
                    "A cross-section is what you get when a higher-\
                         dimensional object passes through a lower-dimensional \
                         space. A 3D apple intersecting a 2D table gives a 2D \
                         shape (a circle, an oval) that changes as the apple \
                         moves. One dimension up: a 4D polytope passing through \
                         3D gives a 3D shape that changes with the slicing w. \
                         That's what the w slider scrubs.",
                );
                ui.add_space(8.0);

                ui.heading("The shapes");
                ui.label("All six convex regular 4-polytopes (\"polychora\") ship:");
                ui.label("• 5-cell (pentachoron); 5 tetrahedra; the 4D simplex.");
                ui.label("• 8-cell (tesseract); 8 cubes; the 4D cube.");
                ui.label(
                    "• 16-cell (hexadecachoron); 16 tetrahedra; the 4D analog \
                         of the octahedron.",
                );
                ui.label(
                    "• 24-cell (icositetrachoron); 24 octahedra; uniquely 4D, \
                         no 3D analog.",
                );
                ui.label("• 120-cell (hecatonicosachoron); 120 dodecahedra.");
                ui.label(
                    "• 600-cell (hexacosichoron); 600 tetrahedra; the 4D \
                         analog of the icosahedron.",
                );
                ui.add_space(8.0);

                ui.heading("Rotation");
                ui.label(
                    "4D rotations are generated by bivectors (2-planes), not \
                         axes. There are six independent planes: xy, xz, xw, yz, \
                         yw, zw. The three w-involving planes pull a visible \
                         axis through the hidden 4th dimension and produce the \
                         interesting cross-section morphs; the three pure-3D \
                         planes rotate the cross-section as a rigid 3D shape.",
                );
                ui.label(
                    "Active-set mode: each plane has a checkbox (include in \
                         spin) and a -180..=180° slider (the rotor's component \
                         in that plane). Composer mode: build a sequence of \
                         exp(scalar · planes) terms via chips or the typed \
                         formula bar.",
                );
                ui.add_space(8.0);

                ui.heading("Views");
                ui.label(
                    "Shapes view: a row of polytopes side-by-side at one \
                         w-slice. Drag-and-drop to reorder. Filmstrip view: one \
                         polytope rendered N times across w-slices fanning out \
                         by ±BODY_SIZE around the slider's value, so the centre \
                         cell tracks w.",
                );
                ui.add_space(8.0);

                ui.heading("Keyboard");
                ui.label("• Space / T: toggle continuous spin.");
                ui.label("• Up / Down arrows: scrub w with the keyboard.");
                ui.label("• 1..6: toggle a plane in the Active set.");
                ui.label("• H: expand / collapse the controls panel.");
                ui.label("• R: full reset.");
                ui.label("• Esc: exit.");
                ui.add_space(8.0);

                ui.heading("Mouse");
                ui.label("• Drag in the viewport: orbit camera.");
                ui.label(
                    "• Right-click on any value label (w, t, plane angle, \
                         scalar): typed-edit popup.",
                );
                ui.label(
                    "• Drag the controls panel by its frame to move it; \
                         drag the formula popup the same way.",
                );
            });
        });
    }

    /// Unified controls overlay. `egui::Window` with
    /// `pivot(CENTER_BOTTOM)` so the bottom edge is the anchor
    /// and the panel grows upward when the expanded body is
    /// shown. Always draggable.
    pub(crate) fn render_overlay(&mut self, ctx: &egui::Context) {
        let screen = ctx.content_rect();
        let pad = 16.0;
        // Cap the overlay width to roughly the 800x600 layout (the
        // shape the demo was designed against; full-screen widths
        // stretched the slider strip into a usability problem).
        // Falls back to the window width if the window is narrower.
        const OVERLAY_MAX_WIDTH: f32 = 768.0;
        const OVERLAY_MIN_WIDTH: f32 = 280.0;
        let natural_w = screen.width() - 2.0 * pad;
        let area_w = natural_w.clamp(OVERLAY_MIN_WIDTH, OVERLAY_MAX_WIDTH);

        let visuals = &ctx.style().visuals;
        let frame = egui::Frame::default()
            .fill(visuals.window_fill)
            .stroke(visuals.window_stroke)
            .corner_radius(visuals.window_corner_radius)
            .inner_margin(10.0);

        let default_bottom_centre = egui::pos2(screen.center().x, screen.bottom() - pad);

        // Snapshot the single-view subject so a change made through the Single
        // body's picker this frame can trigger the same body-rebuild + Schlegel
        // re-resolve a row edit does (the subject IS the rendered row in Single
        // mode, so changing it changes which polychoron the diagram projects
        // through and its cell count). Compared after the deferred-apply block.
        let prev_strip_subject = self.strip_subject;

        egui::Window::new("polytope-playground-overlay")
            .id(egui::Id::new("polytope-playground-overlay"))
            .title_bar(false)
            .resizable(false)
            .collapsible(false)
            .movable(true)
            .auto_sized()
            .pivot(egui::Align2::CENTER_BOTTOM)
            .default_pos(default_bottom_centre)
            .default_width(area_w)
            .frame(frame)
            .show(ctx, |ui| {
                ui.set_width(area_w);
                if self.expanded {
                    self.render_expanded_body(ui);
                    ui.separator();
                }
                self.render_slider_strip(ui, area_w);
                self.render_rate_row(ui);
            });

        // Apply any deferred state changes AFTER the overlay
        // finishes rendering, so both BottomOverlay passes saw
        // the same content this frame.
        if let Some(new_mode) = self.pending_mode.take() {
            self.rotation_mode = new_mode;
        }
        if let Some(new_view) = self.pending_view_mode.take() {
            self.view_mode = new_view;
            // The rendered row changes with the view mode (full row in Shapes,
            // the lone `strip_subject` in Single), so re-emit the SDF body slots
            // and re-resolve the Schlegel cache: the leading polychoron (and its
            // cell count) the diagram projects through can differ between the row
            // and the single subject. Filmstrip re-uploads its own per-cell
            // bodies during render and restores the row after, so this rebuild is
            // harmless there too.
            self.rebuild_bodies();
            self.resolve_schlegel_cache();
        }
        for action in std::mem::take(&mut self.pending_actions) {
            match action {
                DeferredAction::DraftPush(plane) => self.draft.push(plane),
                DeferredAction::SeqCommitDraft => {
                    if !self.draft.is_empty() {
                        self.seq.push(RotorTerm {
                            planes: self.draft.clone(),
                            scalar: None,
                        });
                        self.draft.clear();
                    }
                }
                DeferredAction::DraftClear => self.draft.clear(),
                DeferredAction::SeqPushTerm(term) => self.seq.push(term),
            }
        }
        // A Single-view subject change is a render-row change: re-emit the SDF
        // body slots and re-resolve the Schlegel cache against the new subject's
        // topology. Gated on `Single` so picking a filmstrip subject (which does
        // not change the Schlegel-relevant row) skips the work.
        if self.view_mode == ViewMode::Single && self.strip_subject != prev_strip_subject {
            self.rebuild_bodies();
            self.resolve_schlegel_cache();
        }
    }

    /// Two big sliders (w, t) with fixed-width monospace value
    /// labels.
    pub(crate) fn render_slider_strip(&mut self, ui: &mut egui::Ui, _area_w: f32) {
        // Sized so "w +0.000" / "t  7.12s" (8 monospace chars at
        // FONT_SIZE 13) fit with a few px of breathing room. Larger
        // t values (10+ chars at huge `rot_time`) would clip the
        // tail; that's an acceptable trade for killing the visible
        // deadspace at typical magnitudes.
        const VALUE_CELL_W: f32 = 72.0;
        let avail = ui.available_width();
        let spacing = ui.spacing().item_spacing.x;
        let slider_w = (avail - VALUE_CELL_W - spacing).max(140.0);
        ui.spacing_mut().slider_width = slider_w;

        let row_size = egui::vec2(avail, CONTROL_H);
        let row_layout = egui::Layout::left_to_right(egui::Align::Center);
        // Surface-scaled W range so a `surface scale 4.0` body has a slider
        // wide enough for the slice plane to leave it.
        let w_range = self.effective_w_range();
        ui.allocate_ui_with_layout(row_size, row_layout, |ui| {
            let formatted = format!("w {:>+.3}", self.w_slice);
            slider_with_edit(
                ui,
                &mut self.w_slice,
                -w_range..=w_range,
                &formatted,
                "",
                3,
                VALUE_CELL_W,
            );
        });
        let t_max = self.t_slider_max;
        let mut t_dragged = false;
        ui.allocate_ui_with_layout(row_size, row_layout, |ui| {
            let formatted = format!("t {:>5.2}s", self.rot_time);
            // Same `slider_with_edit` widget as the w slider so
            // click-drag and right-click-edit behave identically
            // across the two rows. Gate the scrub recomputation on
            // `dragged` (not `changed`) because the spin's per-frame
            // `rot_time += dt` would otherwise re-fire the
            // `(omega * t).exp()` rebuild every frame, snapping the
            // rotor when omega shifts (e.g., toggling active planes
            // while spinning).
            let interaction = slider_with_edit(
                ui,
                &mut self.rot_time,
                0.0..=t_max,
                &formatted,
                "s",
                2,
                VALUE_CELL_W,
            );
            t_dragged = interaction.dragged;
        });
        if t_dragged {
            // `rotor_at_time` dispatches Active (product-of-exp) vs
            // Composer (sum-of-bivectors), so scrubbing the t slider
            // reproduces exactly what the continuous-spin path would
            // have integrated to at this `rot_time` in either mode.
            self.rot_state = self.rotor_at_time(self.rot_time);
            self.write_all(self.rot_state);
        }
    }

    /// Always-visible single row directly under the sliders.
    /// Center-justified play / rate / refresh cluster with the
    /// right-aligned utility cluster on the same line:
    ///
    /// ```text
    ///                  [<<] [<] [play/pause] [>] [>>] [refresh]    [?] [^]
    /// ```
    pub(crate) fn render_rate_row(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            const PLAY_GROUP_W: f32 = 215.0;
            let total_w = ui.available_width();
            let leading = ((total_w - PLAY_GROUP_W) / 2.0).max(8.0);

            ui.add_space(leading);
            let ctrl_size = egui::vec2(CONTROL_W, CONTROL_H);
            let play_size = egui::vec2(PLAY_PAUSE_W, CONTROL_H);
            rate_toggle(ui, ctrl_size, &mut self.rate_scale, 0.25, true, false);
            rate_toggle(ui, ctrl_size, &mut self.rate_scale, 0.5, false, false);
            if play_pause_button(ui, play_size, self.rotate)
                .on_hover_text("Toggle continuous rotation (Space)")
                .clicked()
            {
                self.rotate = !self.rotate;
            }
            rate_toggle(ui, ctrl_size, &mut self.rate_scale, 2.0, false, true);
            rate_toggle(ui, ctrl_size, &mut self.rate_scale, 4.0, true, true);
            if refresh_button(ui, ctrl_size)
                .on_hover_text("Reset slice, rate, active set, orientation, time (R)")
                .clicked()
            {
                self.reset();
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if chevron_button(
                    ui,
                    egui::vec2(CONTROL_W, CONTROL_H),
                    !self.expanded,
                    if self.expanded {
                        "Collapse (H)"
                    } else {
                        "Expand controls (H)"
                    },
                )
                .clicked()
                {
                    self.expanded = !self.expanded;
                }
                // Gear + `?`: matching `CONTROL_W × CONTROL_H` so the utility buttons in
                // the row read as a single coherent set with the chevron + play / step
                // buttons. (Pre-sprint `?` was sized `MINI_BUTTON_W` square, odd one
                // out; bumped along with the new gear for visual consistency.)
                let util_size = egui::vec2(CONTROL_W, CONTROL_H);
                if ui
                    .add(egui::Button::new(egui::RichText::new("⚙").strong()).min_size(util_size))
                    .on_hover_text("Render settings")
                    .clicked()
                {
                    self.show_render_panel = !self.show_render_panel;
                }
                if ui
                    .add(egui::Button::new(egui::RichText::new("?").strong()).min_size(util_size))
                    .on_hover_text("About this program")
                    .clicked()
                {
                    self.show_help = true;
                }
            });
        });
    }
}
