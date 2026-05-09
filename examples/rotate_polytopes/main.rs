//! Interactive demo of 4D rotation over `Hyperslice4DNode`. Renders
//! a row of convex regular polychora (5-cell, tesseract, 16-cell,
//! 24-cell by default; 120-cell and 600-cell selectable via
//! `--shapes` or the in-app `+` button) on a 4D `y = 0` floor,
//! with `w`-slice scrubbing and two UIs for composing arbitrary
//! 4D rotations.
//!
//! In **Active set** mode the user toggles individual rotation
//! planes (1..6 -> xy, xz, xw, yz, yw, zw); active planes'
//! bivectors sum into the per-frame angular velocity, which
//! integrates into a rotor via `(ω · dt).exp()`. Sum-of-bivectors
//! composition is commutative, so toggle order doesn't matter and
//! the result is always predictable from the visible active set.
//!
//! In **Composer** mode the user builds a sequence of `RotorTerm`s
//! (each a sum of planes with an optional scalar magnitude),
//! reorders them with drag-and-drop, and either applies them as a
//! one-shot rotor multiplication or feeds the seq into the
//! continuous-spin angular velocity.
//!
//! All six convex regular 4-polytopes ship; the 120-cell and
//! 600-cell use a Rust-side face-hyperplane generator (their orbit
//! sets are too large to inline as WGSL literals). Their SDFs run
//! a true-Euclidean Wolfe greedy hyperplane projection, not a
//! max-plane lower bound.
//!
//! All live state and controls help are drawn as a `rye-egui`
//! overlay via the `App::ui` hook.
//!
//! ## Controls
//!
//! - **Mouse left-drag**: orbit camera.
//! - **Up / Down arrows**: scrub `w`-slice (0.5 u/s).
//! - **Space / T**: toggle 4D rotation (pause/resume freezes
//!   orientation in place, does NOT snap back to identity).
//! - **1..6**: toggle the corresponding rotation plane on/off.
//!   The mapping is `1=xy, 2=xz, 3=xw, 4=yz, 5=yw, 6=zw`. Active
//!   planes' bivectors sum into the angular velocity. Famous
//!   compositions: `3` alone = single xw stretch; `3+4` =
//!   isoclinic xw+yz; `3+5+6` = three w-planes drift through
//!   SO(4). Pure-3D combinations (`1+2+4`) just rotate the
//!   cross-section as a rigid 3D shape.
//! - **R**: full reset, slice, rate, all toggles off, AND
//!   orientation back to canonical pose.
//! - **H**: toggle the bottom-overlay expanded section.
//! - **Esc**: exit.
//!
//! ## CLI
//!
//! - `--shapes name1 name2 ...`: choose the polytopes to render
//!   in left-to-right order. Names accepted include the math form
//!   (`5-cell`, `tesseract`, `16-cell`, `24-cell`, `120-cell`,
//!   `600-cell`) and Platonic-slice aliases (`tetrahedron`, `cube`,
//!   `octahedron`, `cuboctahedron`, `dodecahedron`, `icosahedron`).

use anyhow::{anyhow, Result};
use glam::{Vec3, Vec4};
use rye_app::{egui, run_with_config, App, Camera, FrameCtx, OrbitController, RunConfig, SetupCtx};
use rye_math::{Bivector, EuclideanR3, Rotor, Rotor4};
use rye_render::{
    device::RenderDevice,
    raymarch::{
        polytope_extended_sdfs_wgsl, BodyUniform, Hyperslice4DNode, HYPERSLICE_KERNEL_WGSL,
    },
    Viewport,
};
use rye_sdf::{Scene4, SceneNode4};
use winit::window::WindowAttributes;

mod active;
mod catalog;
mod composer;
mod consts;
mod filmstrip;
mod shapes;
mod state;
mod ui;

use active::combo_name;
use catalog::{parse_row_from_args, SHAPE_CATALOG};
use consts::{BODY_SIZE, BODY_Y, T_SLIDER_INITIAL, W_RANGE, W_SCRUB_RATE};
use state::{body_position, RotatePolytopesApp, RotationMode, ViewMode};

impl App for RotatePolytopesApp {
    type Space = EuclideanR3;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let row = parse_row_from_args()?;
        if row.is_empty() {
            return Err(anyhow!("--shapes produced an empty row"));
        }

        let scene = Scene4::new(SceneNode4::halfspace(Vec4::Y, 0.0));
        // Always include the extended polytope WGSL so any of the six
        // shapes can be added to the row at runtime via the panel.
        // The ~24 KB const-array cost is fixed per app and acceptable
        // for a viz/demo target.
        let shader_source = format!(
            "{kernel}\n{polytope}\n{scene}\n",
            kernel = HYPERSLICE_KERNEL_WGSL,
            polytope = polytope_extended_sdfs_wgsl(),
            scene = scene.to_hyperslice_wgsl("u.w_slice"),
        );
        let module = ctx
            .rd
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rotate_polytopes shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });
        let mut node = Hyperslice4DNode::new(
            &ctx.rd.device,
            ctx.rd.surface_bundle.config.format,
            &module,
            ctx.rd.sample_count(),
        );

        let n = row.len();
        let bodies: Vec<BodyUniform> = row
            .iter()
            .enumerate()
            .map(|(slot, entry)| {
                BodyUniform::polytope_with_rotor(
                    body_position(slot, n),
                    entry.shape,
                    BODY_SIZE,
                    Rotor4::IDENTITY,
                    entry.body_color,
                )
            })
            .collect();
        node.set_bodies(&bodies);

        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 3.0, 9.0);
        let mut orbit: OrbitController<EuclideanR3> = OrbitController::default();
        // Wider orbit so all four bodies in the row are visible at
        // default zoom; user can scroll-zoom in.
        orbit.set_orbit(9.5, -0.25);

        // Always start at w=0 regardless of row contents. Auto-shifting
        // to the 120/600-cell's "Platonic-named" cross-section was
        // confusing in mixed rows: the other shapes' slices got pulled
        // off-centre. Users who want the dodecahedral / icosahedral
        // view scrub there with the slider.
        let initial_w = 0.0;

        Ok(Self {
            space: EuclideanR3,
            camera,
            orbit,
            node,
            row,
            w_slice: initial_w,
            slider_up_held: false,
            slider_down_held: false,
            rotate: false,
            rot_state: Rotor4::IDENTITY,
            // Default: xw spin enabled (active[2] = Plane4::Xw). A
            // first-time user who hits "Spin" before toggling any
            // checkbox now sees motion immediately; the most
            // characteristic 4D rotation, pulling the visible x-axis
            // through the hidden w-axis.
            active: [false, false, true, false, false, false],
            rate_scale: 1.0,
            rot_time: 0.0,
            t_slider_max: T_SLIDER_INITIAL,
            expanded: false,
            show_help: false,
            overlay_pinned_width: None,
            show_formula: false,
            show_controls: true,
            view_mode: ViewMode::Shapes,
            strip_w: true,
            strip_t: false,
            strip_swap_axes: false,
            strip_count_w: 11,
            strip_count_t: 5,
            // Match the t slider's initial range
            // (`T_SLIDER_INITIAL`) so a row of t cells covers the
            // same animation interval the t slider can scrub
            // through at high precision.
            strip_t_extent: T_SLIDER_INITIAL,
            strip_subject: SHAPE_CATALOG[3],
            rotation_mode: RotationMode::Active,
            pending_mode: None,
            pending_view_mode: None,
            pending_actions: Vec::new(),
            seq: Vec::new(),
            draft: Vec::new(),
            formula_input: String::new(),
            formula_error: None,
        })
    }

    fn space(&self) -> &EuclideanR3 {
        &self.space
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        let dt_secs = ctx.n_ticks as f32 / 60.0;

        // Slice scrub.
        let dir = (self.slider_up_held as i32 - self.slider_down_held as i32) as f32;
        if dir != 0.0 {
            self.w_slice = (self.w_slice + dir * W_SCRUB_RATE * dt_secs).clamp(-W_RANGE, W_RANGE);
        }

        // 4D rotation animation. Both bodies share the same rotor
        // so the user can directly compare their slice signatures
        // under identical 4D motion. `rot_state` is the spin
        // baseline; the manual-rotation window's sliders ride on
        // top as a transient display offset (composed at write_all
        // time), so the user can scrub orientation while the spin
        // is running without disturbing the spin itself.
        if self.rotate {
            // Animation time advances by `dt_real * rate_scale`
            // so the rate buttons make `t` count faster/slower
            // (per-real-second). The integrated rotation is
            // `exp(omega_animation * dt_animation)` per frame,
            // which = `exp(omega_animation * rate_scale * dt_real)`.
            // This way `rot_state` and `rot_time` stay in sync:
            // dragging `t` to N reproduces what the spin would
            // have integrated to at animation time N, regardless
            // of how the rate varied along the way.
            let dt_animation = dt_secs * self.rate_scale;
            self.rot_time += dt_animation;
            // Grow the t-slider's max range when the spin has
            // pushed `rot_time` past it, capped so the value
            // can't run away if (e.g.) `rate_scale` is huge or
            // the demo is left running for days. 1e6 seconds
            // (~12 days at ×1) is past any realistic use; if
            // we hit it, `rot_time` clamps to the cap.
            const T_SLIDER_CAP: f32 = 1.0e6;
            if self.rot_time > self.t_slider_max {
                let new_max = (self.rot_time * 2.0).min(T_SLIDER_CAP);
                self.t_slider_max = new_max;
                if self.rot_time > T_SLIDER_CAP {
                    self.rot_time = T_SLIDER_CAP;
                }
            }
            let omega = self.omega_animation() * dt_animation;
            if omega.magnitude_squared() > 0.0 {
                let delta = omega.exp();
                self.rot_state = (delta * self.rot_state).normalize();
            }
        }
        self.write_all(self.rot_state);

        // Camera. Gate the orbit on `!ui_has_focus` so dragging the
        // egui w-slice slider doesn't also rotate the camera.
        //
        // In 2D grid filmstrip mode the body sits low in each
        // cell because the orbit target is at y = 0 (origin)
        // while the body is at y = BODY_Y; that puts the body
        // near the horizon and crowds the polytope at the
        // bottom of every cell. Lifting the orbit target up to
        // body height re-centres the polytope vertically in
        // each cell so the grid reads as a tidy matrix instead
        // of a row of horizon shots.
        let lift_orbit = self.view_mode == ViewMode::Filmstrip && self.strip_w && self.strip_t;
        self.orbit.target.y = if lift_orbit { BODY_Y } else { 0.0 };
        use rye_camera::CameraController;
        if !ctx.ui_has_focus {
            self.orbit
                .advance(ctx.input, &mut self.camera, &EuclideanR3, 0.0);
        }
        let view = self.camera.view();

        // Hyperslice uniforms.
        let cfg = &ctx.rd.surface_bundle.config;
        {
            let u = self.node.uniforms_mut();
            u.camera_pos = view.position.to_array();
            u.camera_forward = view.forward.to_array();
            u.camera_right = view.right.to_array();
            u.camera_up = view.up.to_array();
            u.fov_y_tan = (60.0_f32.to_radians() * 0.5).tan();
            u.resolution = [cfg.width as f32, cfg.height as f32];
            u.time = ctx.time;
            u.tick = ctx.tick as f32;
            u.w_slice = self.w_slice;
        }
        self.node.flush_uniforms(&ctx.rd.queue);
    }

    fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
        // Disable Ctrl+/Ctrl- keyboard-zoom. egui's built-in zoom
        // changes pixels_per_point but the wgpu surface stays at the
        // native resolution, so the scene ends up letter-boxed
        // (black bars) and the tessellator complains about clipped
        // geometry. UI scale stays at native PPP; the scene already
        // supports mouse-wheel orbit-zoom.
        ctx.options_mut(|o| o.zoom_with_keyboard = false);

        // Menu bar always visible at the top. Renders before
        // every other UI so its docked space is reserved (and
        // `ctx.content_rect()` reflects the area below it for
        // subsequent positioning calculations).
        self.render_menu_bar(ctx);

        // Top-left: title + fps + framebuffer size. Replaces the old
        // panel header now that the side panel is gone. Larger
        // typography so the title reads as the program's nameplate
        // rather than just another label.
        let cfg = &frame.rd.surface_bundle.config;
        let (fb_w, fb_h) = (cfg.width, cfg.height);
        // y offset clears the menu bar (~24-28px depending on
        // font) plus a small visual margin. egui::Area's anchor
        // is screen-relative, not content-rect-relative, so the
        // offset must include the menu bar height manually.
        egui::Area::new(egui::Id::new("rotate-polytopes-title"))
            .anchor(egui::Align2::LEFT_TOP, [20.0, 50.0])
            .show(ctx, |ui| {
                ui.add(egui::Label::new(
                    egui::RichText::new("4D Polytope Rotation")
                        .size(22.0)
                        .strong()
                        .color(egui::Color32::WHITE),
                ));
                ui.add(egui::Label::new(
                    egui::RichText::new(format!("{:.0} fps   {}×{}", frame.fps, fb_w, fb_h))
                        .size(13.0)
                        .color(egui::Color32::from_gray(190)),
                ));
            });

        // Live rotation formula popup, plus combo name (Active
        // mode) and the rotor's bivector decomposition matrix.
        // Defaults to top-right; freely draggable. Off by
        // default; toggled by the "Show formula" checkbox.
        if self.show_formula {
            let formula = self.formula_string();
            let name = if self.rotation_mode == RotationMode::Active {
                combo_name(&self.active)
            } else {
                None
            };
            let bivec = self.rot_state.log();
            let screen = ctx.content_rect();
            let default_pos = egui::pos2(screen.right() - 280.0, screen.top() + 16.0);
            let popup_frame = egui::Frame::popup(&ctx.style()).inner_margin(8.0);
            // Cap width so a long formula or term sum doesn't
            // make the popup expand off-screen. The matrix's
            // intrinsic width sets the lower bound (~280 px);
            // formula and combo-name labels wrap inside.
            const FORMULA_POPUP_W: f32 = 320.0;
            egui::Window::new("formula")
                .id(egui::Id::new("rotate-polytopes-formula"))
                .title_bar(false)
                .resizable(false)
                .collapsible(false)
                .movable(true)
                .default_pos(default_pos)
                .default_width(FORMULA_POPUP_W)
                .max_width(FORMULA_POPUP_W)
                .frame(popup_frame)
                .show(ctx, |ui| {
                    ui.set_max_width(FORMULA_POPUP_W);
                    if !formula.is_empty() {
                        ui.add(egui::Label::new(egui::RichText::new(&formula).monospace()).wrap());
                    }
                    if let Some(n) = name {
                        ui.add(egui::Label::new(
                            egui::RichText::new(n).color(egui::Color32::from_rgb(255, 217, 140)),
                        ));
                    }
                    ui.separator();
                    ui.label(egui::RichText::new("log(R) bivector").small().weak());
                    rye_egui::bivector_matrix(ui, &bivec);
                });
        }

        // Filmstrip cell labels: per-cell `w` annotation overlaid
        // on top of the rendered scene so users can see which cell
        // tracks the slider and read the cell-by-cell w sweep.
        if self.view_mode == ViewMode::Filmstrip {
            self.render_filmstrip_cell_labels(ctx);
        }

        // Bottom-anchored unified controls overlay. Hidden by
        // default; toggle via `View > Rotation controls` or `H`.
        if self.show_controls {
            self.render_overlay(ctx);
        }

        // Modal help window (opened by the `?` button).
        self.render_help_window(ctx);
    }

    fn on_event(&mut self, ev: &winit::event::WindowEvent, _ctx: &mut FrameCtx<'_>) {
        use winit::event::{ElementState, WindowEvent};
        use winit::keyboard::{KeyCode, PhysicalKey};
        let WindowEvent::KeyboardInput { event, .. } = ev else {
            return;
        };
        let PhysicalKey::Code(kc) = event.physical_key else {
            return;
        };
        let pressed = event.state == ElementState::Pressed;
        match kc {
            KeyCode::ArrowUp => self.slider_up_held = pressed,
            KeyCode::ArrowDown => self.slider_down_held = pressed,
            KeyCode::KeyR if pressed => self.reset(),
            KeyCode::KeyH if pressed => self.show_controls = !self.show_controls,
            KeyCode::KeyT | KeyCode::Space if pressed => {
                // Pause / resume only, DO NOT touch rot_state. The
                // bodies keep their current orientation when paused
                // and resume from there when toggled back on. Both
                // T (legacy) and Space (media-player convention)
                // bind to the same toggle.
                self.rotate = !self.rotate;
            }
            // Plane toggles. Sum-of-bivectors composition is
            // commutative, so the order in which planes are toggled
            // doesn't affect the resulting motion, only the active
            // set matters.
            KeyCode::Digit1 | KeyCode::Numpad1 if pressed => self.active[0] = !self.active[0],
            KeyCode::Digit2 | KeyCode::Numpad2 if pressed => self.active[1] = !self.active[1],
            KeyCode::Digit3 | KeyCode::Numpad3 if pressed => self.active[2] = !self.active[2],
            KeyCode::Digit4 | KeyCode::Numpad4 if pressed => self.active[3] = !self.active[3],
            KeyCode::Digit5 | KeyCode::Numpad5 if pressed => self.active[4] = !self.active[4],
            KeyCode::Digit6 | KeyCode::Numpad6 if pressed => self.active[5] = !self.active[5],
            _ => {}
        }
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        // Scene renders to the full window. The bottom controls
        // overlay floats on top; `BottomOverlay` is an Area, not
        // a docked panel, so the scene viewport doesn't need to
        // skip a bottom strip.
        let cfg = &rd.surface_bundle.config;
        let viewport = Viewport::full([cfg.width, cfg.height]);
        if self.view_mode == ViewMode::Filmstrip {
            // Filmstrip: each cell shows the `strip_subject`
            // polytope (independent of `self.row`) at a different
            // `w_slice`. We swap the GPU body list to just the
            // subject for the duration of this render, then
            // restore via `rebuild_bodies` so the Shapes view and
            // any subsequent state read sees the full row.
            let entry = self.strip_subject;
            // 2D filmstrip rendering. cols is the column count
            // (horizontal axis), rows is the row count (vertical).
            // Default axis assignment: w on columns, t on rows;
            // `strip_swap_axes` flips it.
            //
            // Per-cell rendering: viewport (cell rect), w_slice
            // (cell's w), and body (cell's rotor for that t).
            // The base rotor `self.rot_state` is offset along
            // omega_animation by `(t_offset)` to give the cell's
            // rotor: `exp(omega_animation * t_offset) * rot_state`.
            // For the w-only and t-only 1D cases, the second
            // axis collapses to a single cell with offset=0.
            let strip_w_extent = BODY_SIZE;
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
                // UI invariant prevents both being off; defensive.
                (false, false) => (1, 1, true),
            };
            let col_vps = viewport.split_horizontal(cols as u32);
            let omega = self.omega_animation();
            let mut grid_cells: Vec<(Viewport, f32, BodyUniform)> = Vec::with_capacity(cols * rows);
            for (col_idx, col_vp) in col_vps.into_iter().enumerate() {
                let row_vps = col_vp.split_vertical(rows as u32);
                for (row_idx, cell_vp) in row_vps.into_iter().enumerate() {
                    // Decide what (w_offset, t_offset) this cell
                    // corresponds to based on which axis carries
                    // which dimension.
                    let (w_idx, w_n, t_idx, t_n) = if w_on_cols {
                        (col_idx, cols, row_idx, rows)
                    } else {
                        (row_idx, rows, col_idx, cols)
                    };
                    let w_t = if w_n <= 1 {
                        0.5
                    } else {
                        w_idx as f32 / (w_n - 1) as f32
                    };
                    let w_offset = -strip_w_extent + w_t * (2.0 * strip_w_extent);
                    let cell_w_slice = self.w_slice + w_offset;
                    let t_offset = if !self.strip_t || t_n <= 1 {
                        0.0
                    } else {
                        // Fan FORWARD only: cell 0 = now, cell
                        // last = rot_time + strip_t_extent. Reads
                        // as "what the rotor will look like at
                        // this future time."
                        let t_norm = t_idx as f32 / (t_n - 1) as f32;
                        t_norm * self.strip_t_extent
                    };
                    // Cell's rotor: spin from `rot_state` by
                    // `omega * t_offset` (animation-time offset).
                    let cell_rotor = if t_offset == 0.0 {
                        self.rot_state
                    } else {
                        ((omega * t_offset).exp() * self.rot_state).normalize()
                    };
                    let body = BodyUniform::polytope_with_rotor(
                        [0.0, BODY_Y, 0.0, 0.0],
                        entry.shape,
                        BODY_SIZE,
                        cell_rotor,
                        entry.body_color,
                    );
                    grid_cells.push((cell_vp, cell_w_slice, body));
                }
            }
            let result = self.node.execute_strip(rd, view, &grid_cells);
            // Restore the full row of bodies for any non-strip
            // consumer (state save, mode switch, etc.).
            self.rebuild_bodies();
            result
        } else {
            {
                let u = self.node.uniforms_mut();
                u.resolution = viewport.resolution_f32();
                u.viewport_origin = [viewport.x as f32, viewport.y as f32];
            }
            self.node.flush_uniforms(&rd.queue);
            self.node.execute_in_viewport(rd, view, viewport)
        }
    }

    fn title(&self, _fps: f32) -> std::borrow::Cow<'static, str> {
        // Window title is now decorative, all live state is in the
        // overlay. Keep the title static so OS task switchers show
        // a stable label.
        std::borrow::Cow::Borrowed("rotate polytopes")
    }
}

fn main() -> Result<()> {
    let config = RunConfig {
        window: WindowAttributes::default()
            .with_title("rotate polytopes")
            .with_visible(false),
        ..RunConfig::default()
    };
    run_with_config::<RotatePolytopesApp>(config)
}

// ---------------------------------------------------------------------------
// Layout regression tests
// ---------------------------------------------------------------------------
//
// `cargo test --example rotate_polytopes` to run.
//
// These tests headless-render the shape row through `egui::Context::run`
// and inspect the actual placed-rect positions of every card and the
// trailing `+` button. They guard against the "descending staircase"
// regression where adding a long-label shape (120/600-cell) caused
// label-wrapping to grow that card's frame, which in turn pushed
// egui's horizontal Center cross-alignment to recompute against a
// new max-height; leaving earlier cards aligned to the old (lower)
// center while the new card centered higher.
//
// `egui::Context` works fine without a renderer for layout-only
// tests; nothing here touches the GPU.

#[cfg(test)]
mod alignment_tests {
    use super::*;

    /// Headless-render the same widget layout as `render_shapes_section`
    /// (minus the surrounding ScrollArea + Frame::popup, which don't
    /// affect intra-row alignment) and capture each card's response
    /// rect plus the trailing `+` button's rect.
    fn capture_row_rects(row: &[ShapeEntry]) -> Vec<egui::Rect> {
        let ctx = egui::Context::default();
        let mut rects = Vec::new();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                // Top-align cross-axis: with `Align::Min` egui places
                // each widget at the row's top edge, skipping the
                // `frame_size.y = max(child, avail)` recursion that
                // Center alignment uses (and that recursion is what
                // produced the converging staircase tops 14 -> 18.5
                // -> 20.75 -> 21.88; each card pulled halfway toward
                // the avail.center as `avail` grew with placed
                // widgets).
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Min), |ui| {
                    for (i, entry) in row.iter().enumerate() {
                        let drag_id = ui.make_persistent_id(("shape-card", i));
                        let frame = egui::Frame::default()
                            .fill(egui::Color32::from_rgb(80, 80, 80))
                            .stroke(egui::Stroke::new(1.0, egui::Color32::GRAY))
                            .inner_margin(egui::Margin::symmetric(4, 6))
                            .corner_radius(egui::CornerRadius::same(3));
                        let (inner_resp, _) = ui.dnd_drop_zone::<usize, _>(frame, |ui| {
                            let _ = ui.dnd_drag_source(drag_id, i, |ui| {
                                ui.allocate_ui_with_layout(
                                    egui::vec2(SHAPE_CARD_WIDTH, 0.0),
                                    egui::Layout::top_down(egui::Align::Center),
                                    |ui| {
                                        ui.add(
                                            egui::Label::new(
                                                egui::RichText::new(entry.label)
                                                    .strong()
                                                    .color(egui::Color32::WHITE),
                                            )
                                            .selectable(false)
                                            .wrap_mode(egui::TextWrapMode::Extend),
                                        );
                                    },
                                );
                            });
                        });
                        rects.push(inner_resp.response.rect);
                    }
                    let plus = add_button(ui, egui::vec2(CONTROL_W, CONTROL_H - 2.0));
                    rects.push(plus.rect);
                });
            });
        });
        rects
    }

    fn rect_table(rects: &[egui::Rect]) -> String {
        rects
            .iter()
            .enumerate()
            .map(|(i, r)| {
                format!(
                    "[{i}] top={:.2} bottom={:.2} center.y={:.2} h={:.2}",
                    r.top(),
                    r.bottom(),
                    r.center().y,
                    r.height()
                )
            })
            .collect::<Vec<_>>()
            .join("\n        ")
    }

    /// All widgets must share a top y. With Top-align cross-axis,
    /// this is the meaningful invariant; heights may vary (the +
    /// button is intentionally 2pt shorter than the cards) but
    /// tops align.
    fn assert_top_aligned(rects: &[egui::Rect], context: &str) {
        if rects.is_empty() {
            return;
        }
        let first_top = rects[0].top();
        for (i, rect) in rects.iter().enumerate() {
            let top = rect.top();
            assert!(
                (top - first_top).abs() < 0.5,
                "{context}: widget {i} top={top:.2} differs from widget 0's \
                 top={first_top:.2}\n        {table}",
                table = rect_table(rects),
            );
        }
    }

    /// Cards (everything except the trailing + button) must have
    /// uniform height. The + is excluded because it's intentionally
    /// 2pt shorter for visual balance.
    fn assert_cards_h_uniform(rects: &[egui::Rect], context: &str) {
        if rects.len() < 2 {
            return;
        }
        let cards = &rects[..rects.len() - 1];
        let first_h = cards[0].height();
        for (i, rect) in cards.iter().enumerate() {
            let h = rect.height();
            assert!(
                (h - first_h).abs() < 0.5,
                "{context}: card {i} height={h:.2} differs from card 0's \
                 height={first_h:.2}\n        {table}",
                table = rect_table(rects),
            );
        }
    }

    #[test]
    fn default_row_4_shapes_aligned() {
        let row = DEFAULT_ROW.to_vec();
        let rects = capture_row_rects(&row);
        assert_cards_h_uniform(&rects, "default 4-shape row");
        assert_top_aligned(&rects, "default 4-shape row");
    }

    #[test]
    fn row_with_120cell_aligned() {
        let mut row = DEFAULT_ROW.to_vec();
        row.push(parse_shape_name("120-cell").unwrap());
        let rects = capture_row_rects(&row);
        assert_cards_h_uniform(&rects, "default + 120-cell");
        assert_top_aligned(&rects, "default + 120-cell");
    }

    #[test]
    fn row_with_120cell_and_600cell_aligned() {
        let mut row = DEFAULT_ROW.to_vec();
        row.push(parse_shape_name("120-cell").unwrap());
        row.push(parse_shape_name("600-cell").unwrap());
        let rects = capture_row_rects(&row);
        assert_cards_h_uniform(&rects, "default + 120-cell + 600-cell");
        assert_top_aligned(&rects, "default + 120-cell + 600-cell");
    }
}

/// Drag-and-drop regression tests for `dnd_drag_source_collapsing`.
/// The headless `egui::Context::run` driver lets us simulate a
/// pointer press + drag-past-threshold and assert the helper's
/// drag detection still wakes up. Two prior regressions this guards
/// against:
///   1. Switching the drag id from `ui.make_persistent_id` to
///      `egui::Id::new` accidentally broke detection (this exists
///      to verify the helper works with both kinds of id).
///   2. Wrapping the body in a `Frame` (so the whole card follows
///      the cursor) must not eat the drag's hit-test rect; the
///      drag rect is the body's rect, which equals the Frame's
///      outer rect after `Frame::show`.
#[cfg(test)]
mod drag_tests {
    use super::*;

    fn screen() -> egui::Rect {
        egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 600.0))
    }

    /// Egui's drag detection uses `time - press_start_time` against
    /// `Options::max_click_duration`. Without advancing `time`
    /// between frames, every press is "still within click window"
    /// and `is_decidedly_dragging` returns false, even with
    /// movement. We thread a monotonic clock so each frame's input
    /// has `time = N * 50ms`; well past the default click duration.
    fn pointer_press(time: f64, pos: egui::Pos2) -> egui::RawInput {
        let mut input = egui::RawInput::default();
        input.screen_rect = Some(screen());
        input.time = Some(time);
        input.events.push(egui::Event::PointerMoved(pos));
        input.events.push(egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: Default::default(),
        });
        input
    }

    fn pointer_move(time: f64, pos: egui::Pos2) -> egui::RawInput {
        let mut input = egui::RawInput::default();
        input.screen_rect = Some(screen());
        input.time = Some(time);
        input.events.push(egui::Event::PointerMoved(pos));
        input
    }

    fn warmup_input(time: f64) -> egui::RawInput {
        let mut input = egui::RawInput::default();
        input.screen_rect = Some(screen());
        input.time = Some(time);
        input
    }

    /// Simulate "click on card, then drag past the drag threshold"
    /// against `dnd_drag_source_collapsing` and assert that
    /// `ctx.is_being_dragged(id)` becomes true. Press alone is not
    /// enough; egui requires movement past `start_drag_threshold`
    /// (~6 px) before flipping the drag flag.
    fn drive_drag(id: egui::Id) -> egui::Context {
        let ctx = egui::Context::default();
        let card_pos = egui::pos2(60.0, 30.0);
        let render = |ctx: &egui::Context| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let _ = dnd_drag_source_collapsing(ui, id, 42_usize, |ui| {
                    egui::Frame::default()
                        .fill(egui::Color32::DARK_GRAY)
                        .inner_margin(egui::Margin::symmetric(4, 6))
                        .show(ui, |ui| {
                            ui.allocate_exact_size(egui::vec2(80.0, 18.0), egui::Sense::hover());
                        });
                });
            });
        };
        let _ = ctx.run(warmup_input(0.0), render);
        let _ = ctx.run(pointer_press(0.05, card_pos), render);
        let _ = ctx.run(pointer_move(0.10, card_pos + egui::vec2(20.0, 0.0)), render);
        let _ = ctx.run(pointer_move(0.15, card_pos + egui::vec2(40.0, 0.0)), render);
        ctx
    }

    /// Baseline: stock `Ui::dnd_drag_source` must start a drag with
    /// our test driver. If THIS fails, the test driver is wrong (not
    /// the helper); the helper-specific tests below are then
    /// meaningless until the driver is fixed.
    #[test]
    fn baseline_stock_dnd_drag_source_starts_drag() {
        let ctx = egui::Context::default();
        let id = egui::Id::new("baseline-test");
        let mut last_rect = egui::Rect::NOTHING;
        let render = |ctx: &egui::Context, last_rect: &mut egui::Rect| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let resp = ui.dnd_drag_source(id, 1_usize, |ui| {
                    egui::Frame::default()
                        .fill(egui::Color32::DARK_GRAY)
                        .inner_margin(egui::Margin::symmetric(4, 6))
                        .show(ui, |ui| {
                            ui.allocate_exact_size(egui::vec2(80.0, 18.0), egui::Sense::hover());
                        });
                });
                *last_rect = resp.response.rect;
            });
        };
        let card_pos = egui::pos2(60.0, 30.0);
        let _ = ctx.run(warmup_input(0.0), |c| render(c, &mut last_rect));
        let _ = ctx.run(pointer_press(0.05, card_pos), |c| render(c, &mut last_rect));
        let _ = ctx.run(pointer_move(0.10, card_pos + egui::vec2(20.0, 0.0)), |c| {
            render(c, &mut last_rect)
        });
        let _ = ctx.run(pointer_move(0.15, card_pos + egui::vec2(40.0, 0.0)), |c| {
            render(c, &mut last_rect)
        });
        assert!(
            ctx.is_being_dragged(id),
            "stock dnd_drag_source should detect drag with this driver"
        );
    }

    /// `egui::Id::new(...)` keys must drive `dnd_drag_source_collapsing`
    /// just as well as `ui.make_persistent_id`. The regression that
    /// motivated this test: shape and term cards stopped responding
    /// to drags after a refactor that switched their drag ids to
    /// `Id::new` for stable per-row-index keys.
    #[test]
    fn id_new_starts_drag() {
        let id = egui::Id::new(("rotate-polytopes-shape-card-test", 0_usize));
        let ctx = drive_drag(id);
        assert!(
            ctx.is_being_dragged(id),
            "drag should be active after press + move past threshold; \
             dnd_drag_source_collapsing failed to wire up the drag rect"
        );
        assert!(
            egui::DragAndDrop::has_payload_of_type::<usize>(&ctx),
            "drag payload should be set after drag starts"
        );
    }

    /// Regression test for the bug the user hit: drag-source ids
    /// keyed by `egui::Id::new(...)` (i.e., NOT scoped to the
    /// rendering ui) collide across `BottomOverlay`'s two passes
    /// and silently break drag detection in release / panic the
    /// `debug_assert!` in debug. The production fix is to derive
    /// the drag id from the per-pass ui scope via
    /// `ui.make_persistent_id(...)` so the two passes see
    /// distinct ids.
    ///
    /// We can't directly test drag detection inside Areas in
    /// headless `Context::run` (Area-routed input doesn't seem to
    /// reach the interaction step the same way it does in a real
    /// winit-driven loop). Instead we verify that:
    /// 1. Rendering the same source closure in two `Area`s with
    ///    different layers does NOT trigger the debug-assert when
    ///    ids are scoped per-ui (`make_persistent_id`).
    /// 2. The IDs actually ARE distinct between the two passes.
    /// The first part; running this test without panic in debug
    ///; is what catches a regression to globally-stable ids.
    #[test]
    fn make_persistent_id_per_pass_avoids_layer_collision() {
        let ctx = egui::Context::default();
        let mut measure_id: Option<egui::Id> = None;
        let mut visible_id: Option<egui::Id> = None;
        let render = |ctx: &egui::Context,
                      measure_id: &mut Option<egui::Id>,
                      visible_id: &mut Option<egui::Id>| {
            let _ = egui::Area::new(egui::Id::new("measure"))
                .order(egui::Order::Background)
                .interactable(false)
                .fixed_pos(egui::pos2(-99_999.0, -99_999.0))
                .show(ctx, |ui| {
                    ui.set_invisible();
                    let id = ui.make_persistent_id("test-card");
                    *measure_id = Some(id);
                    let _ = ui.dnd_drag_source(id, 7_usize, |ui| {
                        ui.allocate_exact_size(egui::vec2(80.0, 18.0), egui::Sense::hover());
                    });
                });
            let _ = egui::Area::new(egui::Id::new("visible"))
                .fixed_pos(egui::pos2(0.0, 0.0))
                .movable(false)
                .show(ctx, |ui| {
                    let id = ui.make_persistent_id("test-card");
                    *visible_id = Some(id);
                    let _ = ui.dnd_drag_source(id, 7_usize, |ui| {
                        ui.allocate_exact_size(egui::vec2(80.0, 18.0), egui::Sense::hover());
                    });
                });
        };
        // Render without panicking. If a future change reverts to
        // `egui::Id::new(...)` for the drag id, both passes resolve
        // to the same id, the same id ends up in two layers, and
        // egui's `debug_assert!` panics here.
        let _ = ctx.run(warmup_input(0.0), |c| {
            render(c, &mut measure_id, &mut visible_id)
        });
        let _ = ctx.run(warmup_input(0.05), |c| {
            render(c, &mut measure_id, &mut visible_id)
        });
        let measure_id = measure_id.expect("measure ran");
        let visible_id = visible_id.expect("visible ran");
        assert_ne!(
            measure_id, visible_id,
            "ui.make_persistent_id resolves through per-ui scope, so the same \
             source must produce different ids in measure vs visible passes; \
             if these ids ever match, the next regression is the debug_assert \
             in egui's WidgetRects::insert"
        );
    }

    /// Regression test for the "card snaps to the right for a frame"
    /// bug: the make-room gap's `open_width` must match the rendered
    /// card slot's outer width (the Frame's outer rect, not the
    /// inner content), otherwise dropping a card causes a one-frame
    /// horizontal layout shift as the gap closes and the card
    /// occupies a slightly-different-sized slot.
    #[test]
    fn shape_gap_open_width_matches_card_slot_width() {
        let ctx = egui::Context::default();
        let mut card_outer_w = 0.0_f32;
        let _ = ctx.run(warmup_input(0.0), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let resp = egui::Frame::default()
                    .fill(egui::Color32::DARK_GRAY)
                    .inner_margin(egui::Margin::symmetric(4, 6))
                    .corner_radius(egui::CornerRadius::same(3))
                    .show(ui, |ui| {
                        ui.allocate_ui_with_layout(
                            egui::vec2(SHAPE_CARD_WIDTH, 0.0),
                            egui::Layout::top_down(egui::Align::Center),
                            |ui| {
                                ui.add(
                                    egui::Label::new(egui::RichText::new("test").strong())
                                        .selectable(false)
                                        .wrap_mode(egui::TextWrapMode::Extend),
                                );
                            },
                        );
                    });
                card_outer_w = resp.response.rect.width();
            });
        });
        let gap_open_width = SHAPE_CARD_WIDTH + 8.0;
        let drift = (gap_open_width - card_outer_w).abs();
        assert!(
            drift < 1.0,
            "make-room gap open width ({gap_open_width:.1}) must match the \
             rendered shape card outer width ({card_outer_w:.1}); a mismatch \
             produces a one-frame horizontal rubberband when the gap closes \
             and the card takes its slot. drift = {drift:.1} pt"
        );
    }

    /// Simulates dragging a card from one slot to another and
    /// verifies that the row's total width is INVARIANT through
    /// the drag -> drop transition. If the dragged card takes some
    /// space during drag and a different amount after drop, OR if
    /// the make-room gap's width doesn't match the dropped card's
    /// slot width, the OTHER cards shift horizontally on drop.
    /// That's the rubberband the user sees.
    ///
    /// Render N "cards" with stable widths via the same helper
    /// (`dnd_drag_source_collapsing` + `make_room_gap`) the live
    /// shape row uses, simulate a press + drag-past-threshold +
    /// hover-over-target + release, and capture neighbouring card
    /// positions on the last drag frame and on the post-drop
    /// frame.
    #[test]
    fn shape_row_total_width_invariant_through_drop() {
        const N: usize = 4;
        const CARD_W: f32 = SHAPE_CARD_WIDTH + 8.0;
        const SPACING: f32 = 4.0;
        let ctx = egui::Context::default();
        // We measure widths under a "card 0 is being dragged"
        // scenario (drop at trailing slot N) and compare with the
        // post-drop scenario (no drag in flight, all cards rendered
        // normally).
        let target_slot = N;

        let mut total_during_drag = 0.0_f32;
        let render_during_drag = |ctx: &egui::Context, total: &mut f32| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Min), |ui| {
                    ui.spacing_mut().item_spacing.x = SPACING;
                    let drop_idx = Some(target_slot);
                    for i in 0..N {
                        // Make-room gap before card i.
                        let gap_id = ui.make_persistent_id(("gap", i));
                        let _ = make_room_gap(ui, drop_idx == Some(i), gap_id, 18.0, CARD_W);
                        let card_id = ui.make_persistent_id(("card", i));
                        let _ = dnd_drag_source_collapsing(ui, card_id, i, |ui| {
                            egui::Frame::default()
                                .inner_margin(egui::Margin::symmetric(4, 6))
                                .show(ui, |ui| {
                                    ui.allocate_exact_size(
                                        egui::vec2(SHAPE_CARD_WIDTH, 0.0),
                                        egui::Sense::hover(),
                                    );
                                });
                        });
                    }
                    // Trailing gap.
                    let trail_id = ui.make_persistent_id(("gap", N));
                    let _ = make_room_gap(ui, drop_idx == Some(N), trail_id, 18.0, CARD_W);
                    *total = ui.min_rect().width();
                });
            });
        };

        // Drive a real drag on card `dragged_idx`. Card centers
        // are predictable: card 0 center = CARD_W/2 = 36.
        let card0_center = egui::pos2(CARD_W / 2.0, 9.0);
        let _ = ctx.run(warmup_input(0.0), |c| {
            render_during_drag(c, &mut total_during_drag)
        });
        let _ = ctx.run(pointer_press(0.05, card0_center), |c| {
            render_during_drag(c, &mut total_during_drag)
        });
        // Move past drag threshold AND past the row to land at
        // the trailing slot. card0_center is at x=36, drag to x=400.
        let target_pos = egui::pos2(400.0, 9.0);
        let _ = ctx.run(pointer_move(0.10, target_pos), |c| {
            render_during_drag(c, &mut total_during_drag)
        });
        // Several frames at the same target so the gap can settle
        // open at full width.
        for k in 0..15 {
            let t = 0.15 + (k as f64) * 0.02;
            let _ = ctx.run(pointer_move(t, target_pos), |c| {
                render_during_drag(c, &mut total_during_drag)
            });
        }
        let drag_total = total_during_drag;
        let dragged_id = ctx.dragged_id();
        // Release.
        let mut release_input = egui::RawInput::default();
        release_input.screen_rect = Some(screen());
        release_input.time = Some(0.6);
        release_input
            .events
            .push(egui::Event::PointerMoved(target_pos));
        release_input.events.push(egui::Event::PointerButton {
            pos: target_pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: Default::default(),
        });
        let _ = ctx.run(release_input, |c| {
            render_during_drag(c, &mut total_during_drag)
        });
        // Frame after release: drag is over, no make-room gap, no
        // dragged card collapse. Re-render to measure post-drop.
        let _ = ctx.run(warmup_input(0.65), |c| {
            render_during_drag(c, &mut total_during_drag)
        });
        let post_drop_total = total_during_drag;

        eprintln!(
            "drag_total = {drag_total:.1}, post_drop_total = {post_drop_total:.1}, \
             dragged_id = {dragged_id:?}"
        );
        let drift = (drag_total - post_drop_total).abs();
        assert!(
            drift < 1.0,
            "row total width must stay constant from drag -> drop, otherwise \
             cards rubberband horizontally on release. drag={drag_total:.1}, \
             post_drop={post_drop_total:.1}, drift={drift:.1}"
        );
    }

    /// Same regression check applied to `dnd_drag_source_collapsing`:
    /// the helper must round-trip through a content closure that
    /// runs in two egui layers without producing a same-id-in-two-
    /// layers panic.
    #[test]
    fn collapsing_helper_in_two_pass_no_layer_collision() {
        let ctx = egui::Context::default();
        let render = |ctx: &egui::Context| {
            let _ = egui::Area::new(egui::Id::new("measure"))
                .order(egui::Order::Background)
                .interactable(false)
                .fixed_pos(egui::pos2(-99_999.0, -99_999.0))
                .show(ctx, |ui| {
                    ui.set_invisible();
                    let id = ui.make_persistent_id("test-card");
                    let _ = dnd_drag_source_collapsing(ui, id, 7_usize, |ui| {
                        ui.allocate_exact_size(egui::vec2(80.0, 18.0), egui::Sense::hover());
                    });
                });
            let _ = egui::Area::new(egui::Id::new("visible"))
                .fixed_pos(egui::pos2(0.0, 0.0))
                .movable(false)
                .show(ctx, |ui| {
                    let id = ui.make_persistent_id("test-card");
                    let _ = dnd_drag_source_collapsing(ui, id, 7_usize, |ui| {
                        ui.allocate_exact_size(egui::vec2(80.0, 18.0), egui::Sense::hover());
                    });
                });
        };
        let _ = ctx.run(warmup_input(0.0), render);
        let _ = ctx.run(warmup_input(0.05), render);
    }

    /// `ui.make_persistent_id(...)` keys must also work; protect
    /// against a future regression that hard-codes one id flavour.
    #[test]
    fn make_persistent_id_starts_drag() {
        let ctx = egui::Context::default();
        let render = |ctx: &egui::Context, captured_id: &mut Option<egui::Id>| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let id = ui.make_persistent_id(("test-card", 0_usize));
                *captured_id = Some(id);
                let _ = dnd_drag_source_collapsing(ui, id, 99_usize, |ui| {
                    egui::Frame::default()
                        .fill(egui::Color32::DARK_GRAY)
                        .inner_margin(egui::Margin::symmetric(4, 6))
                        .show(ui, |ui| {
                            ui.allocate_exact_size(egui::vec2(80.0, 18.0), egui::Sense::hover());
                        });
                });
            });
        };
        let card_pos = egui::pos2(60.0, 30.0);
        let mut id = None;
        let _ = ctx.run(warmup_input(0.0), |ctx| render(ctx, &mut id));
        let _ = ctx.run(pointer_press(0.05, card_pos), |ctx| render(ctx, &mut id));
        let _ = ctx.run(
            pointer_move(0.10, card_pos + egui::vec2(20.0, 0.0)),
            |ctx| render(ctx, &mut id),
        );
        let _ = ctx.run(
            pointer_move(0.15, card_pos + egui::vec2(40.0, 0.0)),
            |ctx| render(ctx, &mut id),
        );
        let id = id.expect("captured id");
        assert!(
            ctx.is_being_dragged(id),
            "drag should be active for make_persistent_id keys too"
        );
    }
}
