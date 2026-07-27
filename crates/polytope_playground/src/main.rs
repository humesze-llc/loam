//! Interactive 4D-rotation demo over `Hyperslice4DNode`: a row of convex
//! regular polychora (5-cell, tesseract, 16-cell, 24-cell by default;
//! 120-cell and 600-cell via `--shapes` or the in-app `+`) on a 4D `y = 0`
//! floor, with `w`-slice scrubbing and two rotation-composition UIs.
//!
//! Active-set mode toggles individual rotation planes whose bivectors sum
//! into the per-frame angular velocity (integrated via `(ω · dt).exp()`);
//! the sum is commutative, so toggle order is irrelevant. Composer mode
//! builds a reorderable sequence of `RotorTerm`s applied as a one-shot
//! product or fed into the continuous spin.
//!
//! The 120-cell and 600-cell use a Rust-side face-hyperplane generator
//! (too large to inline as WGSL); their SDFs run a true-Euclidean Wolfe
//! greedy hyperplane projection, not a max-plane lower bound. Live state
//! and controls draw as a `loam-egui` overlay via `App::ui`.
//!
//! ## Controls
//!
//! - **Mouse left-drag**: orbit camera.
//! - **Up / Down arrows**: scrub `w`-slice (0.5 u/s).
//! - **Space / T**: toggle 4D rotation (freezes in place; no snap-back).
//! - **1..6**: toggle plane `1=xy, 2=xz, 3=xw, 4=yz, 5=yw, 6=zw`. `3+4`
//!   is the isoclinic xw+yz; pure-3D combos (`1+2+4`) just rotate the
//!   cross-section as a rigid 3D shape.
//! - **R**: full reset (slice, rate, toggles, AND orientation).
//! - **H**: toggle the bottom-overlay expanded section.
//! - **Esc**: exit.
//!
//! ## Arguments
//!
//! Native form `--key=value`, browser form `?key=value`.
//!
//! - `shapes=name1,name2`: polytopes left-to-right. Accepts the math form
//!   (`5-cell`, `tesseract`, `16-cell`, `24-cell`, `120-cell`, `600-cell`)
//!   and Platonic-slice aliases (`tetrahedron`, `cube`, `octahedron`,
//!   `cuboctahedron`, `dodecahedron`, `icosahedron`).
//! - `scene=slug`: which scene the shell boots. Unknown slugs warn and fall
//!   back to the first. The `scene` console command lists the registry and
//!   switches at runtime.
//! - `embed=1`: hide the shell menu bar for page embeds (any value but `0`
//!   and `false` enables it). The `scene` command is then the only in-app
//!   switcher.

use anyhow::{anyhow, Result};
use glam::{Mat4, Vec2, Vec3, Vec4};
use loam_app::{
    args::Args,
    egui,
    freecam::{CursorMode, Freecam},
    Camera, CameraController, FrameCtx, OrbitController, RunConfig, SetupCtx,
};
use loam_egui::{Console, ConsoleUi};
use loam_math::WPlane;
use loam_math::{Bivector, EuclideanR3, Rotor, Rotor4};
use loam_physics::polytope::{
    polytope_section_faces_append, polytope_section_overlay_with_vertices, vertex_color_by_position,
};
use loam_render::{
    device::RenderDevice,
    raymarch::{
        polytope_extended_sdfs_wgsl, BodyUniform, Hyperslice4DNode, HYPERSLICE_KERNEL_WGSL,
    },
    DepthBuffer, DepthMode, LineRasterNode, PointRasterNode, TriangleRasterNode, Viewport,
};

/// Depth-attachment format for the rasterized section-faces pass. 32-bit
/// float depth-sorts the 600-cell's densely-packed caps without artifacting.
const SECTION_FACES_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

use loam_scene::{Scene4, SceneNode4};
use loam_shape::LineMesh;
use winit::window::WindowAttributes;

mod active;
mod catalog;
mod color;
mod composer;
mod console;
mod consts;
mod filmstrip;
mod physics;
mod projections;
mod render;
mod sections;
mod shapes;
mod shell;
mod state;
mod ui;
mod wireframe_geom;

use active::combo_name;
use catalog::{parse_row, SHAPE_CATALOG};
use color::{unique_edge_palette, w_depth_color};
#[cfg(test)]
use consts::SPACE_TESSELLATION_SAMPLES;
use consts::{
    BODY_SIZE, BODY_Y, HYPERSLICE_MIN_THICKNESS, T_SCRUB_RATE, T_SLIDER_INITIAL, W_SCRUB_RATE,
};
#[cfg(test)]
use loam_physics::polytope::Polytope4;
use physics::PlaygroundPhysics;
use state::{
    CameraMode, Demo, RotationMode, RowFrame, SurfaceMode, ViewMode, WireframeColorMode,
    WireframeProjection,
};
use wireframe_geom::*;

/// Per-cell "crossing strength" in `[0, 1]`: 1 at the cell's w-midpoint
/// (widest cap), 0 outside its w-range, linear in `|w_slice - midpoint|`
/// over the half-extent. A cheap proxy for cap area, shared by
/// `render_wireframe_overlay` and `render_points`. `out` is cleared on entry
/// and reuses the caller's allocation.
fn compute_cell_strengths(
    cells: &[&[u32]],
    local_vertices: &[Vec4],
    w_slice: f32,
    out: &mut Vec<f32>,
) {
    out.clear();
    out.extend(cells.iter().map(|cell| {
        let (w_min, w_max) = cell_w_range(cell, local_vertices);
        let half_extent = (w_max - w_min) * 0.5;
        if half_extent <= 0.0 {
            return 0.0;
        }
        let mid = (w_min + w_max) * 0.5;
        let dist = (w_slice - mid).abs();
        (1.0 - dist / half_extent).clamp(0.0, 1.0)
    }));
}

#[cfg(test)]
use catalog::{parse_shape_name, ShapeEntry, DEFAULT_ROW};
#[cfg(test)]
use consts::{CONTROL_H, CONTROL_W, SHAPE_CARD_WIDTH};
#[cfg(test)]
use loam_egui::dnd::{drag_source_collapsing as dnd_drag_source_collapsing, make_room_gap};
#[cfg(test)]
use loam_egui::media::add_button;

/// Boot position of the draggable formula popup: inset from the top-right of
/// the space the shell's menu-bar panel leaves. `available_rect` and not
/// `content_rect`, which is the viewport minus OS safe-area insets and does
/// not shrink for a panel, so seating against it hides the popup's first rows
/// behind the bar.
fn formula_popup_seat(ctx: &egui::Context) -> egui::Pos2 {
    const RIGHT_INSET: f32 = 280.0;
    const TOP_INSET: f32 = 16.0;
    let area = ctx.available_rect();
    egui::pos2(area.right() - RIGHT_INSET, area.top() + TOP_INSET)
}

impl Demo {
    pub(crate) fn new(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let row = parse_row(&Args::current())?;

        let scene = Scene4::new(SceneNode4::halfspace(Vec4::Y, 0.0));
        // Always include every shape's WGSL so any can be added at runtime.
        // Floor visibility is gated at runtime via `u.params.x` (set each
        // frame in `update`): 0.0 makes the halfspace SDF return 1e9 so the
        // marcher never paints the checkerboard. See
        // [`Scene4::to_hyperslice_wgsl_gated`].
        let shader_source = format!(
            "{kernel}\n{polytope}\n{scene}\n",
            kernel = HYPERSLICE_KERNEL_WGSL,
            polytope = polytope_extended_sdfs_wgsl(),
            scene = scene.to_hyperslice_wgsl_gated("u.w_slice", "u.params.x"),
        );
        let module = ctx
            .rd
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("polytope_playground shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });
        let mut node = Hyperslice4DNode::new(
            &ctx.rd.device,
            ctx.rd.target_format(),
            &module,
            ctx.rd.sample_count(),
        );

        // Initial SDF body uniforms, through the same builder every later
        // re-upload uses: with the raster default, polychoral entries emit
        // `BodyUniform::default()` (kind = Invalid) so the kernel skips them
        // and the rasterizer draws them instead.
        let surface_mode = SurfaceMode::default();
        let physics = PlaygroundPhysics::new(row.len(), BODY_SIZE);
        let bodies: Vec<BodyUniform> = row
            .iter()
            .enumerate()
            .map(|(slot, entry)| {
                state::sdf_body_uniform(
                    &physics,
                    entry,
                    slot,
                    row.len(),
                    Rotor4::IDENTITY,
                    BODY_SIZE,
                    surface_mode,
                )
            })
            .collect();
        node.set_bodies(&bodies);

        // Section perimeter (cyan outlines): ReadOnly depth-test against the
        // shared section-faces attachment, so polytope A's perimeter is
        // occluded by B's caps when A is behind B. `LessEqual` lets a line at
        // exactly its own cap's depth still draw on top of that cap.
        let section_edges = LineRasterNode::new(
            &ctx.rd.device,
            ctx.rd.target_format(),
            DepthMode::ReadOnly {
                format: SECTION_FACES_DEPTH_FORMAT,
            },
            ctx.rd.sample_count(),
        );
        // Parent wireframe: ReadOnly depth-test against the section-faces
        // attachment, so lines behind a cap are occluded and lines in front
        // draw over. No depth-write. In SDF mode no pass writes depth, so the
        // test passes everywhere and the SDF visual is unchanged.
        let parent_wireframe = LineRasterNode::new(
            &ctx.rd.device,
            ctx.rd.target_format(),
            DepthMode::ReadOnly {
                format: SECTION_FACES_DEPTH_FORMAT,
            },
            ctx.rd.sample_count(),
        );

        // Point-disc rasterizer for the optional vertex + cell-center sprite
        // overlay. No depth attachment: these are always-visible debug
        // markers. A ReadOnly test hid a vertex behind its own cap, since
        // drop-w projects it to the cap's (x, y, z) at slightly farther depth.
        let points_node = PointRasterNode::new(
            &ctx.rd.device,
            ctx.rd.target_format(),
            DepthMode::Off,
            ctx.rd.sample_count(),
        );

        // Rasterized cross-section faces: filled cell-caps, face-normal
        // Lambert. ReadWrite depth so caps of the same polychoron occlude each
        // other; the buffer is sized + cleared per-frame in the render path
        // when surface mode is `Raster` (see `Demo::render_section_faces`).
        let section_faces = TriangleRasterNode::new(
            &ctx.rd.device,
            ctx.rd.target_format(),
            DepthMode::ReadWrite {
                format: SECTION_FACES_DEPTH_FORMAT,
            },
            loam_render::FragmentShading::FaceNormalLambert,
            ctx.rd.sample_count(),
        );
        // Translucent variant: same pipeline without depth-write, used when
        // `surface_alpha < 1.0` so the wireframe (drawn after, `LessEqual`)
        // shows through the cap. ReadOnly depth lets caps within one polytope
        // overpaint in submission order where their R³ projections overlap,
        // but caps tile the section disjointly so overdraw is rare; a
        // depth-prepass fix is punted until the artifact is observed.
        let section_faces_translucent = TriangleRasterNode::new(
            &ctx.rd.device,
            ctx.rd.target_format(),
            DepthMode::ReadOnly {
                format: SECTION_FACES_DEPTH_FORMAT,
            },
            loam_render::FragmentShading::FaceNormalLambert,
            ctx.rd.sample_count(),
        );

        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 3.0, 9.0);
        let mut orbit: OrbitController<EuclideanR3> = OrbitController::default();
        // Startup framing that fits the whole row; MAX_DISTANCE only widens
        // the scroll-out range, not this initial distance.
        orbit.set_orbit(8.0, -0.25);

        // Inactive at startup; `camera freecam` calls `set_active` to grab the
        // cursor and seed the freecam position from the camera.
        let freecam = Freecam::new();

        // Always start at w=0; auto-shifting to a shape's Platonic-named slice
        // pulled the other shapes' slices off-centre in mixed rows.
        let initial_w = 0.0;

        Ok(Self {
            space: EuclideanR3,
            physics,
            camera,
            orbit,
            freecam,
            camera_mode: CameraMode::default(),
            node,
            section_edges,
            parent_wireframe,
            wireframe_enabled: false,
            wireframe_nearest_active: true,
            cross_section: state::SectionLayer::CROSS_SECTION_DEFAULT,
            projected_cap: state::SectionLayer::PROJECTED_CAP_DEFAULT,
            wireframe_color_mode: WireframeColorMode::default(),
            wireframe_projection: WireframeProjection::default(),
            // Drop-w default needs no Schlegel cache until Schlegel is picked.
            schlegel_params: None,
            stereographic_pole: state::STEREOGRAPHIC_DEFAULT_POLE,
            wireframe_hyperslice: false,
            wireframe_hyperslice_thickness: consts::HYPERSLICE_DEFAULT_THICKNESS,
            wireframe_width_px: 1.8,
            wireframe_alpha: 1.0,
            unique_edge_palette_cache: std::collections::HashMap::new(),
            cell_centers_cache: std::collections::HashMap::new(),
            surface_scale: 1.0,
            floor_enabled: true,
            section_faces,
            section_faces_translucent,
            section_faces_projected_scratch: loam_shape::TriangleMesh::<3>::default(),
            section_clip_projected_scratch: Vec::new(),
            points_node,
            points_enabled: false,
            points_show_vertices: true,
            points_show_cell_centers: true,
            points_size_px: 4.0,
            points_mesh_scratch: loam_shape::PointMesh::<3>::default(),
            section_faces_depth: None,
            section_world_vertices_scratch: Vec::new(),
            section_faces_mesh_scratch: loam_shape::TriangleMesh::<3>::default(),
            body_uniform_scratch: Vec::new(),
            slerp_scratch: Vec::new(),
            wireframe_section_edges_scratch: LineMesh::<3>::default(),
            wireframe_parent_lines_scratch: LineMesh::<3>::default(),
            overlay_local_vertices_scratch: Vec::new(),
            overlay_center_locals_scratch: Vec::new(),
            overlay_cell_strengths_scratch: Vec::new(),
            surface_mode,
            row,
            w_slice: initial_w,
            slider_up_held: false,
            slider_down_held: false,
            slider_left_held: false,
            slider_right_held: false,
            rotate: false,
            rot_state: Rotor4::IDENTITY,
            // xw spin on by default (active[2]): the most characteristic 4D
            // rotation, so "Spin" shows motion before any checkbox is toggled.
            active: [false, false, true, false, false, false],
            base_angles: [0.0; 6],
            rate_scale: 1.0,
            rot_time: 0.0,
            t_slider_max: T_SLIDER_INITIAL,
            expanded: false,
            show_help: false,
            show_render_panel: false,
            example_callout: loam_egui::CalloutState {
                window_pos: egui::Pos2::new(220.0, 120.0),
                open: false,
            },
            // Unwired: the per-projection annotation callout stays closed and
            // its render call early-returns; kept for a future re-wire.
            mode_annotation_open: loam_egui::CalloutState {
                window_pos: egui::Pos2::new(220.0, 300.0),
                open: false,
            },
            show_formula: false,
            show_controls: true,
            view_mode: ViewMode::Shapes,
            strip_w: true,
            strip_t: false,
            strip_swap_axes: false,
            strip_count_w: 11,
            strip_count_t: 5,
            // Match `T_SLIDER_INITIAL` so the t-cell row covers the same
            // animation interval as the t slider.
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

    pub(crate) fn space(&self) -> &EuclideanR3 {
        &self.space
    }

    pub(crate) fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        let dt_secs = ctx.n_ticks as f32 / 60.0;

        // Slice scrub (w axis). Clamp to the surface-scaled range so the
        // keyboard scrub matches the slider bounds after `surface scale`.
        let dir = (self.slider_up_held as i32 - self.slider_down_held as i32) as f32;
        if dir != 0.0 {
            let w_range = self.effective_w_range();
            self.w_slice = (self.w_slice + dir * W_SCRUB_RATE * dt_secs).clamp(-w_range, w_range);
        }

        // Time scrub (t axis). Mirrors the t-slider drag: rebuild `rot_state`
        // from the new `rot_time` via `rotor_at_time`, floored at zero.
        let t_dir = (self.slider_right_held as i32 - self.slider_left_held as i32) as f32;
        if t_dir != 0.0 {
            self.rot_time = (self.rot_time + t_dir * T_SCRUB_RATE * dt_secs).max(0.0);
            // Same runaway guard as the spin path.
            const T_SLIDER_CAP: f32 = 1.0e6;
            if self.rot_time > self.t_slider_max {
                let new_max = (self.rot_time * 2.0).min(T_SLIDER_CAP);
                self.t_slider_max = new_max;
                if self.rot_time > T_SLIDER_CAP {
                    self.rot_time = T_SLIDER_CAP;
                }
            }
            self.rot_state = self.rotor_at_time(self.rot_time);
        }

        // 4D rotation animation. All bodies share one rotor so their slice
        // signatures are directly comparable. `rot_state` is the spin
        // baseline; manual-rotation sliders ride on top as a transient offset
        // composed at write_all time, leaving the spin undisturbed.
        if self.rotate {
            let dt_animation = dt_secs * self.rate_scale;
            self.rot_time += dt_animation;
            // Grow the t-slider max past `rot_time`, capped at 1e6 s (~12 days
            // at ×1) so a huge `rate_scale` or long run can't run it away.
            const T_SLIDER_CAP: f32 = 1.0e6;
            if self.rot_time > self.t_slider_max {
                let new_max = (self.rot_time * 2.0).min(T_SLIDER_CAP);
                self.t_slider_max = new_max;
                if self.rot_time > T_SLIDER_CAP {
                    self.rot_time = T_SLIDER_CAP;
                }
            }
        }
        // Recompose `rot_state` each frame: Active rebuilds from `rot_time`,
        // Composer integrates the omega-bivector into `rot_state` directly.
        match self.rotation_mode {
            RotationMode::Active => {
                self.rot_state = self.active_rotor();
            }
            RotationMode::Composer => {
                if self.rotate {
                    let dt_animation = dt_secs * self.rate_scale;
                    let omega = self.omega_animation() * dt_animation;
                    if omega.magnitude_squared() > 0.0 {
                        let delta = omega.exp();
                        self.rot_state = (delta * self.rot_state).normalize();
                    }
                }
            }
        }
        // Rigid bodies advance on the tick count, not on `dt_secs`, so a
        // trajectory is frame-rate independent. `write_all` then reconciles
        // the world with the rendered row and uploads the resulting poses.
        self.physics.step(ctx.n_ticks);
        self.write_all(self.rot_state);

        // Gate the orbit on `!ui_has_focus` so dragging the egui slider
        // doesn't also rotate the camera. In the 2D grid filmstrip, lift the
        // orbit target to body height so the polytope re-centres in each cell
        // (at y = 0 it sits near the horizon, crowding the cell bottom).
        let lift_orbit = self.view_mode == ViewMode::Filmstrip && self.strip_w && self.strip_t;
        self.orbit.target.y = if lift_orbit { BODY_Y } else { 0.0 };
        if !ctx.ui_has_focus {
            match self.camera_mode {
                CameraMode::Orbit => {
                    self.orbit
                        .advance(ctx.input, &mut self.camera, &EuclideanR3, dt_secs);
                }
                CameraMode::FreeRoam => {
                    // Handles look + WASD + cursor-grab gating internally;
                    // no-ops while the cursor is released (Alt-toggled).
                    self.freecam.advance(ctx.input, &mut self.camera, dt_secs);
                }
            }
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
            // Floor gate read by the injected wrapper around `loam_scene_sdf`:
            // 1.0 = floor on, 0.0 = wrapper short-circuits to 1e9.
            u.params[0] = if self.floor_enabled { 1.0 } else { 0.0 };
        }
        self.node.flush_uniforms(&ctx.rd.queue);
    }

    pub(crate) fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
        // Disable keyboard zoom: it changes PPP but the wgpu surface stays at
        // native resolution, so the scene letter-boxes and the tessellator
        // complains. Mouse-wheel orbit-zoom is the scene's zoom.
        ctx.options_mut(|o| o.zoom_with_keyboard = false);

        // Build label (short git hash + dirty marker), top-right with
        // symmetric padding: `MENU_BAR_PAD + LABEL_INSET` below the menu bar,
        // `-LABEL_INSET` in from the right edge.
        const MENU_BAR_PAD: f32 = 24.0;
        const LABEL_INSET: f32 = 14.0;
        let build_label = format!("build: {}{}", env!("BUILD_HASH"), env!("BUILD_DIRTY"),);
        egui::Area::new(egui::Id::new("polytope-playground-build"))
            .anchor(
                egui::Align2::RIGHT_TOP,
                [-LABEL_INSET, MENU_BAR_PAD + LABEL_INSET],
            )
            .show(ctx, |ui| {
                ui.add(egui::Label::new(
                    egui::RichText::new(build_label)
                        .monospace()
                        .size(11.0)
                        .color(egui::Color32::from_gray(140)),
                ));
            });

        // Live rotation formula popup: formula, combo name (Active mode), and
        // the rotor's log(R) bivector matrix. Off by default; draggable.
        if self.show_formula {
            let formula = self.formula_string();
            let name = if self.rotation_mode == RotationMode::Active {
                combo_name(&self.active)
            } else {
                None
            };
            let bivec = self.rot_state.log();
            let default_pos = formula_popup_seat(ctx);
            let popup_frame = egui::Frame::popup(&ctx.style()).inner_margin(8.0);
            // Cap width so a long formula doesn't expand the popup off-screen;
            // the matrix's intrinsic ~280 px is the lower bound.
            const FORMULA_POPUP_W: f32 = 320.0;
            egui::Window::new("formula")
                .id(egui::Id::new("polytope-playground-formula"))
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
                    loam_egui::bivector_matrix(ui, &bivec);
                });
        }

        // Per-cell `w` annotation overlaid on the scene.
        if self.view_mode == ViewMode::Filmstrip {
            self.render_filmstrip_cell_labels(ctx);
        }

        // Bottom controls overlay. Toggle via `View > Rotation controls` / `H`.
        if self.show_controls {
            self.render_overlay(ctx);
        }

        self.render_help_window(ctx);
        // Off by default so the scene fills the window on first launch.
        self.render_render_panel(ctx);
        // Demonstrates `loam_egui::callout` against the first polychoron.
        self.render_example_callout(ctx, frame);
        // No-ops in the default drop-w scene; else explains the projection.
        self.render_mode_annotation(ctx, frame);
    }

    /// Per-projection educational annotation via [`state::mode_annotation`],
    /// anchored to the leading polychoron's body center (the whole-shape anchor,
    /// vs the vertex anchor in [`Self::render_example_callout`]) and reprojected
    /// per frame. No-op when off, the row has no polychoron, or the projection
    /// is drop-w (the mapping returns `None`).
    fn render_mode_annotation(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
        if !self.mode_annotation_open.open {
            return;
        }
        let Some(annotation) = state::mode_annotation(self.wireframe_projection) else {
            return;
        };

        // Anchor: the leading polychoron's body center in world R³.
        let row_frame = self.row_frame();
        let Some((slot, _entry)) = row_frame
            .row
            .iter()
            .enumerate()
            .find(|(_, e)| e.shape.polytope4().is_some())
        else {
            return;
        };
        let world_pos = row_frame.pose(slot).position_r3();

        let view_dir = self.camera.view();
        let cfg = &frame.rd.surface_bundle.config;
        let ppp = ctx.pixels_per_point();
        let vp_w = (cfg.width as f32 / ppp).round() as u32;
        let vp_h = (cfg.height as f32 / ppp).round() as u32;
        let Some(screen_pos) = loam_egui::world_to_screen(
            world_pos,
            &view_dir,
            60.0_f32.to_radians(),
            (vp_w, vp_h),
            0.1,
            100.0,
        ) else {
            return;
        };

        loam_egui::callout(
            ctx,
            "polytope-playground-mode-annotation",
            screen_pos,
            &mut self.mode_annotation_open,
            annotation.title,
            |ui| {
                ui.label(&annotation.body);
            },
        );
    }

    /// Demonstrate the `loam_egui::callout` primitive against the first polychoron:
    /// the anchor
    /// follows vertex 0 through the rotor + body-position + projection chain,
    /// reprojected per frame. No-op when the row has no polychora.
    fn render_example_callout(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
        if !self.example_callout.open {
            return;
        }
        // First polychoron in the rendered row; its vertex 0 is the anchor.
        let row_frame = self.row_frame();
        let Some((slot, entry)) = row_frame
            .row
            .iter()
            .enumerate()
            .find(|(_, e)| e.shape.polytope4().is_some())
        else {
            return;
        };
        let polytope = entry.shape.polytope4().expect("filter guarantees Some");
        let label = entry.label;
        // Through the same seam the raster passes use, so the leader line lands
        // on the vertex the wireframe drew rather than on the layout.
        let world_pos = row_frame.anchor_r3(slot, polytope.topology().vertices[0]);

        // Reproject world R³ -> screen pixels via the rasterizer's camera;
        // `None` (anchor offscreen) draws nothing.
        let view_dir = self.camera.view();
        let cfg = &frame.rd.surface_bundle.config;
        let ppp = ctx.pixels_per_point();
        let vp_w = (cfg.width as f32 / ppp).round() as u32;
        let vp_h = (cfg.height as f32 / ppp).round() as u32;
        let Some(screen_pos) = loam_egui::world_to_screen(
            world_pos,
            &view_dir,
            60.0_f32.to_radians(),
            (vp_w, vp_h),
            0.1,
            100.0,
        ) else {
            return;
        };

        let title = format!("{label} vertex 0");
        loam_egui::callout(
            ctx,
            "polytope-playground-example-callout",
            screen_pos,
            &mut self.example_callout,
            &title,
            |ui| {
                ui.label(
                    "Example callout: this leader line tracks vertex 0 of the first \
                     polychoron in the row as it rotates through 4D. Drag the panel \
                     anywhere; the line keeps the anchor live. Same primitive \
                     (loam_egui::callout) is the foundation for future tutorial \
                     overlays in Polytope Playground.",
                );
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Anchor coordinates").strong());
                ui.label(format!(
                    "world R³: ({:.2}, {:.2}, {:.2})",
                    world_pos.x, world_pos.y, world_pos.z
                ));
            },
        );
    }

    pub(crate) fn on_key(
        &mut self,
        kc: winit::keyboard::KeyCode,
        state: winit::event::ElementState,
        _ctx: &mut FrameCtx<'_>,
    ) {
        use winit::event::ElementState;
        use winit::keyboard::KeyCode;
        let pressed = state == ElementState::Pressed;
        match kc {
            KeyCode::ArrowUp => self.slider_up_held = pressed,
            KeyCode::ArrowDown => self.slider_down_held = pressed,
            KeyCode::ArrowLeft => self.slider_left_held = pressed,
            KeyCode::ArrowRight => self.slider_right_held = pressed,
            KeyCode::KeyR if pressed => self.reset(),
            KeyCode::KeyH if pressed => self.show_controls = !self.show_controls,
            KeyCode::KeyT if pressed => {
                // Pause / resume only; rot_state is untouched so orientation
                // holds across the pause.
                self.rotate = !self.rotate;
            }
            // Space also toggles rotation, but not in freecam where it is the
            // move-up axis. T is the always-available toggle.
            KeyCode::Space if pressed && !matches!(self.camera_mode, CameraMode::FreeRoam) => {
                self.rotate = !self.rotate;
            }
            // Alt modulates the freecam cursor grab. Forward both edges so
            // Hold mode (release-on-hold, re-grab-on-release) sees them;
            // `Freecam::on_alt` dispatches on its `cursor_mode`.
            KeyCode::AltLeft | KeyCode::AltRight
                if matches!(self.camera_mode, CameraMode::FreeRoam) =>
            {
                self.freecam.on_alt(pressed);
            }
            // Plane toggles; the sum-of-bivectors composition is commutative,
            // so only the active set matters, not toggle order.
            KeyCode::Digit1 | KeyCode::Numpad1 if pressed => self.active[0] = !self.active[0],
            KeyCode::Digit2 | KeyCode::Numpad2 if pressed => self.active[1] = !self.active[1],
            KeyCode::Digit3 | KeyCode::Numpad3 if pressed => self.active[2] = !self.active[2],
            KeyCode::Digit4 | KeyCode::Numpad4 if pressed => self.active[3] = !self.active[3],
            KeyCode::Digit5 | KeyCode::Numpad5 if pressed => self.active[4] = !self.active[4],
            KeyCode::Digit6 | KeyCode::Numpad6 if pressed => self.active[5] = !self.active[5],
            _ => {}
        }
    }

    pub(crate) fn title(&self, _fps: f32) -> std::borrow::Cow<'static, str> {
        // Static so OS task switchers show a stable label; live state is in
        // the overlay.
        std::borrow::Cow::Borrowed("polytope playground")
    }
}

// ---------------------------------------------------------------------------
// Scene wrapper: Demo + Console<Demo>
// ---------------------------------------------------------------------------
//
// A wrapper, not a `console` field on `Demo`: `Console::ui` takes
// `(&mut self, &mut Ctx)`, so co-locating both would need a simultaneous
// `&mut self.console` + `&mut self` borrow. Separate fields give the
// dispatch a clean two-field borrow.

pub(crate) struct RotateScene {
    demo: Demo,
    console: Console<Demo>,
    /// Last frame's `egui::Context::wants_keyboard_input()`. Key events
    /// arrive before `ui`, so this one-frame-stale value gates demo hotkeys
    /// (Space / R / arrows): when egui holds keyboard focus they must not fire.
    last_egui_keyboard: bool,
}

/// Lower bound on a visible section-layer fill alpha; below this the cap reads
/// as off, so the grammar rejects it and steers to `0` (explicit off).
const SECTION_ALPHA_MIN_VISIBLE: f32 = 0.05;

/// Shared handler for `section cross-alpha` / `section cap-alpha`: bare queries,
/// else set `surface_alpha`. `0` is off; `[SECTION_ALPHA_MIN_VISIBLE, 1.0]` is a
/// visible fill. Takes `&mut SectionLayer` (not `&mut Demo`) so both
/// registrations share one body, unit-testable without a GPU-backed `Demo`.
fn run_section_alpha(
    layer_name: &str,
    layer: &mut state::SectionLayer,
    args: &[&str],
    out: &mut loam_egui::console::ConsoleWriter,
) -> anyhow::Result<()> {
    match args.first().copied() {
        None => {
            let state = if layer.fill_visible() {
                if layer.surface_alpha >= 1.0 {
                    "opaque"
                } else {
                    "translucent"
                }
            } else {
                "off"
            };
            out.line(format!(
                "section {layer_name}-alpha: {:.3} ({state})",
                layer.surface_alpha
            ));
        }
        Some(token) => {
            let parsed: f32 = token
                .parse()
                .map_err(|e| anyhow!("invalid alpha `{token}`: {e}"))?;
            // `0` is off; else a visible alpha in `[MIN_VISIBLE, 1.0]`. Reject
            // the faint `(0, MIN)` band rather than silently round it.
            let valid = parsed == 0.0 || (SECTION_ALPHA_MIN_VISIBLE..=1.0).contains(&parsed);
            if !valid {
                return Err(anyhow!(
                    "section {layer_name}-alpha {parsed} out of range; expected 0 (off) or {SECTION_ALPHA_MIN_VISIBLE}..=1.0"
                ));
            }
            layer.surface_alpha = parsed;
            out.line(format!(
                "section {layer_name}-alpha: set to {parsed:.3}{}",
                if parsed == 0.0 { " (off)" } else { "" }
            ));
        }
    }
    Ok(())
}

impl RotateScene {
    pub(crate) fn new(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        Ok(Self {
            demo: Demo::new(ctx)?,
            console: Self::build_console(),
            last_egui_keyboard: false,
        })
    }
}

impl shell::Scene for RotateScene {
    fn space(&self) -> &EuclideanR3 {
        self.demo.space()
    }

    fn menus(&mut self, ui: &mut egui::Ui) {
        self.demo.menu_contents(ui);
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        self.demo.update(ctx);
    }

    fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
        self.demo.ui(ctx, frame);
        // Pump pending tracing events into the scrollback before rendering it,
        // so mirrored log lines show this frame.
        loam_app::log::pump_into(&mut self.console);
        self.console.ui(ctx, &mut self.demo);
        // Stash for next frame's hotkey gating, captured after the console
        // renders so a freshly-focused input registers true.
        self.last_egui_keyboard = ctx.wants_keyboard_input();
    }

    fn on_key(
        &mut self,
        code: winit::keyboard::KeyCode,
        state: winit::event::ElementState,
        ctx: &mut FrameCtx<'_>,
    ) {
        // Suppress demo keybinds while egui captures the keyboard (a focused
        // TextEdit), so typing `reset` doesn't also fire the R hotkey.
        if !self.last_egui_keyboard {
            self.demo.on_key(code, state, ctx);
        }
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        self.demo.render(rd, view)
    }

    fn title(&self, fps: f32) -> std::borrow::Cow<'static, str> {
        self.demo.title(fps)
    }
}

fn main() -> Result<()> {
    // `loam_app::run` handles native + wasm dispatch off the page's `data-mode`
    // and WasmConfig IDs; the demo's `index.html` matches the default layout.
    loam_app::run::<shell::ShellApp>(RunConfig {
        window: WindowAttributes::default()
            .with_title("polytope playground")
            .with_visible(false),
        ..RunConfig::default()
    })
}

// ---------------------------------------------------------------------------
// Layout regression tests
// ---------------------------------------------------------------------------
//
// Headless-render the shape row through `egui::Context::run` (no GPU) and
// inspect placed-rect positions. They guard the "descending staircase"
// regression: a long-label shape (120/600-cell) wraps and grows its card,
// recomputing Center cross-alignment so earlier cards stayed at the old
// center while the new card centered higher.

#[cfg(test)]
mod color_tests {
    //! Tests for `compute_cell_strengths` (per-cell w-slice crossing strength).
    use super::*;

    // ---- compute_cell_strengths -----------------------------------------

    /// The into-form as an expression; the render path passes a retained buffer.
    fn strengths_of(cells: &[&[u32]], local_vertices: &[Vec4], w_slice: f32) -> Vec<f32> {
        let mut out = Vec::new();
        compute_cell_strengths(cells, local_vertices, w_slice, &mut out);
        out
    }

    /// Slice at the cell's w-midpoint produces strength = 1 (cap is widest there).
    #[test]
    fn cell_strength_at_midpoint_is_one() {
        // Single cell with two vertices at w = -0.5 and w = +0.5. Midpoint w = 0.
        let cells: [&[u32]; 1] = [&[0, 1]];
        let local_vertices = [
            glam::Vec4::new(0.0, 0.0, 0.0, -0.5),
            glam::Vec4::new(0.0, 0.0, 0.0, 0.5),
        ];
        let strengths = strengths_of(&cells, &local_vertices, 0.0);
        assert_eq!(strengths.len(), 1);
        assert!((strengths[0] - 1.0).abs() < 1e-5);
    }

    /// Slice outside the cell's w-range produces strength = 0 (cap doesn't exist).
    #[test]
    fn cell_strength_outside_range_is_zero() {
        let cells: [&[u32]; 1] = [&[0, 1]];
        let local_vertices = [
            glam::Vec4::new(0.0, 0.0, 0.0, -0.5),
            glam::Vec4::new(0.0, 0.0, 0.0, 0.5),
        ];
        let strengths = strengths_of(&cells, &local_vertices, 5.0);
        assert!(strengths[0].abs() < 1e-5);
    }

    /// Slice at the cell's w-boundary produces strength = 0 (cap is degenerate).
    #[test]
    fn cell_strength_at_boundary_is_zero() {
        let cells: [&[u32]; 1] = [&[0, 1]];
        let local_vertices = [
            glam::Vec4::new(0.0, 0.0, 0.0, -0.5),
            glam::Vec4::new(0.0, 0.0, 0.0, 0.5),
        ];
        // Slice exactly at the +w extreme: dist = 0.5, half_extent = 0.5,
        // strength = 1 - 1 = 0.
        let strengths = strengths_of(&cells, &local_vertices, 0.5);
        assert!(strengths[0].abs() < 1e-5);
    }

    /// Halfway between midpoint and boundary yields strength = 0.5 (linear in
    /// `|w_slice - mid| / half_extent`).
    #[test]
    fn cell_strength_is_linear() {
        let cells: [&[u32]; 1] = [&[0, 1]];
        let local_vertices = [
            glam::Vec4::new(0.0, 0.0, 0.0, -0.5),
            glam::Vec4::new(0.0, 0.0, 0.0, 0.5),
        ];
        // midpoint = 0, half_extent = 0.5; slice at 0.25 -> dist 0.25 -> 1 - 0.5 = 0.5.
        let strengths = strengths_of(&cells, &local_vertices, 0.25);
        assert!((strengths[0] - 0.5).abs() < 1e-5);
    }

    /// Degenerate cell (all vertices at the same w) yields strength = 0; the half-extent
    /// is zero so the gradient has nothing to interpolate. The function returns 0 rather
    /// than divide-by-zero, which is what the wireframe overlay path expects.
    #[test]
    fn cell_strength_degenerate_cell_is_zero() {
        let cells: [&[u32]; 1] = [&[0, 1]];
        let local_vertices = [
            glam::Vec4::new(0.0, 0.0, 0.0, 0.0),
            glam::Vec4::new(1.0, 0.0, 0.0, 0.0),
        ];
        let strengths = strengths_of(&cells, &local_vertices, 0.0);
        assert!(strengths[0].abs() < 1e-5);
    }
}

#[cfg(test)]
mod blended_edge_tests {
    //! Tests for `push_blended_edge`. The S³ slerp math is pinned in
    //! `loam_math::spherical_embedded`; these pin the demo-side contract: flat
    //! fast path, curved sub-segment count, and curved-edge-bows-off-chord.
    use super::*;

    fn flat_drop_w() -> loam_math::Projection<4> {
        loam_math::Projection::Identity
    }

    /// Total length of all segments in the mesh, in world R³.
    fn polyline_length(mesh: &LineMesh<3>) -> f32 {
        mesh.segments
            .iter()
            .map(|(p0, p1)| (Vec3::from_array(*p1) - Vec3::from_array(*p0)).length())
            .sum()
    }

    const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

    /// `blend == 0` emits one chord segment equal to the projected endpoints.
    #[test]
    fn blend_zero_emits_single_chord() {
        let a = Vec4::new(0.7, 0.0, 0.0, 0.0);
        let b = Vec4::new(0.0, 0.7, 0.0, 0.0);
        let mut mesh = LineMesh::<3>::default();
        let mut scratch = Vec::new();
        push_blended_edge(
            &mut mesh,
            a,
            b,
            WHITE,
            WHITE,
            1.0,
            0.0,
            &flat_drop_w(),
            Vec3::ZERO,
            &mut scratch,
            STEREOGRAPHIC_VIEW_RADIUS,
        );
        assert_eq!(mesh.segments.len(), 1);
        let chord = (Vec3::new(0.0, 0.7, 0.0) - Vec3::new(0.7, 0.0, 0.0)).length();
        assert!((polyline_length(&mesh) - chord).abs() < 1e-6);
    }

    /// `blend > 0` subdivides the edge into `SPACE_TESSELLATION_SAMPLES`
    /// sub-segments.
    #[test]
    fn blend_positive_emits_tessellated_segments() {
        let a = Vec4::new(0.7, 0.0, 0.0, 0.0);
        let b = Vec4::new(0.0, 0.7, 0.0, 0.0);
        let mut mesh = LineMesh::<3>::default();
        let mut scratch = Vec::new();
        push_blended_edge(
            &mut mesh,
            a,
            b,
            WHITE,
            WHITE,
            1.0,
            1.0,
            &flat_drop_w(),
            Vec3::ZERO,
            &mut scratch,
            STEREOGRAPHIC_VIEW_RADIUS,
        );
        assert_eq!(mesh.segments.len(), SPACE_TESSELLATION_SAMPLES);
    }

    /// A spherical edge bows off its chord: the tessellated polyline is strictly
    /// longer than the straight chord between the same endpoints.
    #[test]
    fn spherical_edge_is_longer_than_chord() {
        let a = Vec4::new(0.7, 0.0, 0.0, 0.0);
        let b = Vec4::new(0.0, 0.7, 0.0, 0.0);
        let chord = (Vec3::new(0.0, 0.7, 0.0) - Vec3::new(0.7, 0.0, 0.0)).length();

        let mut arc = LineMesh::<3>::default();
        let mut scratch = Vec::new();
        push_blended_edge(
            &mut arc,
            a,
            b,
            WHITE,
            WHITE,
            1.0,
            1.0,
            &flat_drop_w(),
            Vec3::ZERO,
            &mut scratch,
            STEREOGRAPHIC_VIEW_RADIUS,
        );
        let arc_len = polyline_length(&arc);
        // Quarter circle of radius 0.7 has arc length 0.7·π/2 ≈ 1.0996 vs chord
        // 0.7·√2 ≈ 0.9899. The 16-segment approximation undershoots the true arc
        // slightly but still clears the chord comfortably.
        assert!(
            arc_len > chord + 0.05,
            "arc {arc_len} should exceed chord {chord}"
        );
    }

    /// A half-blend's polyline length lies strictly between chord and full arc,
    /// pinning the morph as monotone, not a step.
    #[test]
    fn half_blend_is_between_flat_and_spherical() {
        let a = Vec4::new(0.7, 0.0, 0.0, 0.0);
        let b = Vec4::new(0.0, 0.7, 0.0, 0.0);
        let chord = (Vec3::new(0.0, 0.7, 0.0) - Vec3::new(0.7, 0.0, 0.0)).length();

        let make = |blend: f32| {
            let mut mesh = LineMesh::<3>::default();
            let mut scratch = Vec::new();
            push_blended_edge(
                &mut mesh,
                a,
                b,
                WHITE,
                WHITE,
                1.0,
                blend,
                &flat_drop_w(),
                Vec3::ZERO,
                &mut scratch,
                STEREOGRAPHIC_VIEW_RADIUS,
            );
            polyline_length(&mesh)
        };
        let half = make(0.5);
        let full = make(1.0);
        assert!(
            half > chord,
            "half-blend {half} should exceed chord {chord}"
        );
        assert!(
            half < full,
            "half-blend {half} should be under full arc {full}"
        );
    }

    /// A non-trivial Perspective4D projection; focal distance sits outside the
    /// unit-circumradius polytope so no vertex straddles the eye plane.
    fn perspective() -> loam_math::Projection<4> {
        loam_math::Projection::Perspective4D {
            focal_distance: 3.0,
        }
    }

    /// `blend == 0` through a non-identity affine projection (Perspective4D)
    /// emits one segment whose endpoints equal `project_to_world(a/b)` to the
    /// bit. Pins the fast path under a w-dependent perspective scale, which
    /// `blend_zero_emits_single_chord` (drop-w only) did not exercise.
    #[test]
    fn blend_zero_is_bit_identical_to_flat_chord() {
        // Adjacent tesseract vertices differing only in w, so the perspective
        // scale differs per endpoint.
        let a = Vec4::new(0.5, 0.5, 0.5, 0.5);
        let b = Vec4::new(0.5, 0.5, 0.5, -0.5);
        let proj = perspective();
        let body_pos = Vec3::new(1.0, -2.0, 0.5);

        let mut mesh = LineMesh::<3>::default();
        let mut scratch = Vec::new();
        push_blended_edge(
            &mut mesh,
            a,
            b,
            WHITE,
            WHITE,
            1.0,
            0.0,
            &proj,
            body_pos,
            &mut scratch,
            STEREOGRAPHIC_VIEW_RADIUS,
        );

        assert_eq!(mesh.segments.len(), 1, "affine flat chord is one segment");
        let expected_a = project_to_world(a, &proj, body_pos).to_array();
        let expected_b = project_to_world(b, &proj, body_pos).to_array();
        let (seg_a, seg_b) = mesh.segments[0];
        assert_eq!(seg_a, expected_a, "start equals projected a");
        assert_eq!(seg_b, expected_b, "end equals projected b");
    }

    /// At any blend in [0, 1] the first/last emitted point equals
    /// `project_to_world(a/b)` exactly. The morph bows only the interior; bit-
    /// exact endpoint glue keeps the section cap attached to the wireframe.
    #[test]
    fn blend_endpoints_exact_at_all_t() {
        let a = Vec4::new(0.5, 0.5, 0.5, 0.5);
        let b = Vec4::new(-0.5, 0.5, 0.5, -0.5);
        let proj = loam_math::Projection::Stereographic { pole: Vec4::W };
        let body_pos = Vec3::new(-0.25, 1.5, 0.0);
        let expected_a = project_to_world(a, &proj, body_pos).to_array();
        let expected_b = project_to_world(b, &proj, body_pos).to_array();

        for &blend in &[0.0_f32, 0.001, 0.25, 0.5, 0.75, 1.0] {
            let mut mesh = LineMesh::<3>::default();
            let mut scratch = Vec::new();
            push_blended_edge(
                &mut mesh,
                a,
                b,
                WHITE,
                WHITE,
                1.0,
                blend,
                &proj,
                body_pos,
                &mut scratch,
                STEREOGRAPHIC_VIEW_RADIUS,
            );
            assert!(!mesh.segments.is_empty(), "blend {blend}: emitted nothing");
            let first = mesh.segments.first().unwrap().0;
            let last = mesh.segments.last().unwrap().1;
            assert_eq!(
                first, expected_a,
                "blend {blend}: first point equals proj(a)"
            );
            assert_eq!(last, expected_b, "blend {blend}: last point equals proj(b)");
        }
    }
}

#[cfg(test)]
mod section_command_tests {
    //! Tests for shared console handlers unit-testable without a GPU-backed
    //! `Demo` by exercising the handler body directly.
    use super::*;
    use loam_egui::console::ConsoleWriter;

    /// `run_section_alpha` sets an in-range alpha, accepts `0` as off, rejects
    /// the faint `(0, MIN_VISIBLE)` band / over-range / unparseable input (no
    /// silent clamp), and leaves the field untouched on a bare query.
    #[test]
    fn section_alpha_sets_off_and_visible_rejects_faint_and_bad() {
        let run = |start: f32, args: &[&str]| -> (f32, bool) {
            let mut layer = state::SectionLayer {
                perimeter: true,
                surface_alpha: start,
            };
            let mut out = ConsoleWriter::new();
            let ok = run_section_alpha("cross", &mut layer, args, &mut out).is_ok();
            (layer.surface_alpha, ok)
        };

        // A visible alpha in [MIN_VISIBLE, 1.0] is set.
        assert_eq!(run(1.0, &["0.5"]), (0.5, true), "in-range alpha is set");
        assert_eq!(run(0.5, &["1.0"]), (1.0, true), "opaque alpha is set");
        // `0` is the explicit off state, accepted.
        assert_eq!(run(0.85, &["0"]), (0.0, true), "0 turns the layer off");
        // The faint sub-MIN band is rejected, not rounded.
        let (val, ok) = run(0.85, &["0.01"]);
        assert!(!ok, "faint (0, MIN) alpha must be rejected");
        assert_eq!(val, 0.85, "rejected faint alpha leaves the field untouched");
        // Over-range and unparseable are rejected, field untouched.
        assert_eq!(run(0.85, &["2.0"]).0, 0.85, "over-range alpha is rejected");
        assert_eq!(
            run(0.85, &["notafloat"]).0,
            0.85,
            "unparseable alpha is rejected"
        );
        // Bare query reports without mutating.
        assert_eq!(run(0.7, &[]), (0.7, true), "bare query leaves the field");
    }
}

#[cfg(test)]
mod formula_popup_tests {
    use super::*;

    /// The popup boots clear of the shell's menu-bar panel. Seating it against
    /// `content_rect` (viewport minus safe-area insets, panel-unaware) put its
    /// first rows behind the bar.
    #[test]
    fn formula_popup_seats_below_the_menu_bar_panel() {
        let ctx = egui::Context::default();
        let mut seat = None;
        let mut bar_bottom = 0.0;
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            // Any bar taller than the popup's top inset separates the two
            // rects; an exact height keeps that independent of font metrics.
            bar_bottom = egui::TopBottomPanel::top("shell-menu-bar")
                .exact_height(64.0)
                .show(ctx, |ui| {
                    ui.label("bar");
                })
                .response
                .rect
                .bottom();
            seat = Some(formula_popup_seat(ctx));
        });
        let seat = seat.expect("run closure sets the seat");
        assert!(
            seat.y >= bar_bottom,
            "popup seat y {} must clear the menu bar bottom {bar_bottom}",
            seat.y
        );
    }
}

#[cfg(test)]
mod hyperslice_filter_tests {
    //! Tests for the cell-level wireframe Hyperslice cull: an edge survives iff
    //! some cell holding both endpoints has its w-range overlapping the slab
    //! `[w_slice - t/2, w_slice + t/2]`. Split into `cell_w_range` and the 1D
    //! `slab_overlaps` predicate; tests pin band semantics plus agreement with
    //! the active-edge coloring.
    use super::*;

    /// Mirror of the production cull closure `edge_in_slab_cell`: keep `(i, j)`
    /// iff some cell holding both endpoints has a slab-overlapping w-range.
    fn kept_by_cull(
        i: u32,
        j: u32,
        cells: &[&[u32]],
        local_vertices: &[Vec4],
        w_slice: f32,
        thickness: f32,
    ) -> bool {
        cells.iter().any(|cell| {
            if !(cell.contains(&i) && cell.contains(&j)) {
                return false;
            }
            let (w_min, w_max) = cell_w_range(cell, local_vertices);
            slab_overlaps(w_min, w_max, w_slice, thickness)
        })
    }

    /// A w-range entirely outside the slab does not overlap, on either side.
    #[test]
    fn slab_overlaps_off_band_is_false() {
        assert!(!slab_overlaps(0.8, 0.9, 0.0, 0.2));
        assert!(!slab_overlaps(-0.9, -0.8, 0.0, 0.2));
    }

    /// A range straddling the slice, and a range wholly inside the slab, both
    /// overlap. The predicate is true whenever the range touches the band.
    #[test]
    fn slab_overlaps_on_band_is_true() {
        // Straddles w_slice = 0.
        assert!(slab_overlaps(-0.5, 0.5, 0.0, 0.2));
        // Wholly inside a wide slab.
        assert!(slab_overlaps(-0.05, 0.05, 0.0, 0.2));
        // Slab centered off-origin, range inside it.
        assert!(slab_overlaps(0.45, 0.55, 0.5, 0.2));
    }

    /// The band is closed (a range end exactly on `w_slice +/- t/2` overlaps)
    /// and deterministic across repeated evaluations.
    #[test]
    fn slab_overlaps_closed_boundary_and_deterministic() {
        let keep = slab_overlaps(-0.5, 0.5, 0.0, 1.0);
        assert!(keep, "range ends exactly on the closed band must overlap");

        // One end grazing the upper boundary, the other inside.
        assert!(slab_overlaps(0.0, 0.5, 0.0, 1.0));
        // One end grazing the lower boundary from outside.
        assert!(slab_overlaps(-0.6, -0.5, 0.0, 1.0));

        // Determinism: same inputs, same answer, every time.
        for _ in 0..16 {
            assert_eq!(slab_overlaps(-0.5, 0.5, 0.0, 1.0), keep);
        }
    }

    /// Thickness 0 is floored to [`HYPERSLICE_MIN_THICKNESS`]: the slab becomes
    /// a razor band where only a range crossing `w_slice` overlaps.
    #[test]
    fn slab_overlaps_zero_thickness_floor() {
        // Crosses w_slice = 0: overlaps even at thickness 0 (floor keeps the
        // band a hair wide).
        assert!(slab_overlaps(-0.3, 0.3, 0.0, 0.0));
        // Entirely on one side, just above the floor's reach: no overlap.
        assert!(!slab_overlaps(0.1, 0.3, 0.0, 0.0));
        // A range endpoint exactly at w_slice still counts (closed band).
        assert!(slab_overlaps(0.0, 0.3, 0.0, 0.0));
    }

    /// A negative thickness is floored the same as 0, staying a valid razor band
    /// rather than an inverted slab.
    #[test]
    fn slab_overlaps_negative_thickness_floor() {
        assert!(slab_overlaps(-0.3, 0.3, 0.0, -5.0));
        assert!(!slab_overlaps(0.1, 0.3, 0.0, -5.0));
    }

    /// The repro that motivated the cell-level cull: an edge with both endpoints
    /// on the far side of the slab is kept because its containing cell is sliced.
    /// An edge-level test would cull it; the cell-level cull matches the active-
    /// green coloring, which also reads the whole cell's w-range.
    #[test]
    fn far_side_edge_of_active_cell_is_kept() {
        let w_slice = -0.182_f32;
        let thickness = 0.2_f32;
        // 0,1 on the near side (w = -0.5), 2,3 on the far side (w = +0.5).
        let local_vertices = [
            Vec4::new(0.0, 0.0, 0.0, -0.5),
            Vec4::new(1.0, 0.0, 0.0, -0.5),
            Vec4::new(0.0, 1.0, 0.0, 0.5),
            Vec4::new(1.0, 1.0, 0.0, 0.5),
        ];
        let cell: &[u32] = &[0, 1, 2, 3];
        let cells: &[&[u32]] = &[cell];

        // The far-side edge's own w-interval [0.5, 0.5] misses the slab
        // [-0.282, -0.082]: the old edge-level rule would cull it.
        assert!(
            !slab_overlaps(0.5, 0.5, w_slice, thickness),
            "the far-side edge's own endpoints do not straddle the slab"
        );
        // The containing cell's w-range [-0.5, 0.5] DOES straddle the slab, so
        // the cell-level cull keeps the far-side edge.
        assert!(
            kept_by_cull(2, 3, cells, &local_vertices, w_slice, thickness),
            "far-side edge of a sliced cell must be kept by the cell-level cull"
        );
    }

    /// Agreement contract: every active-colored edge (cell strength above 0)
    /// is kept by the cull. The slab band is a superset of the strict-interior
    /// plane, so `active => kept` for any thickness at or above the floor.
    #[test]
    fn cull_keeps_every_active_edge() {
        let w_slice = -0.182_f32;
        let thickness = HYPERSLICE_MIN_THICKNESS; // razor band: the strictest cull
        let local_vertices = [
            Vec4::new(0.0, 0.0, 0.0, -0.5),
            Vec4::new(1.0, 0.0, 0.0, -0.5),
            Vec4::new(0.0, 1.0, 0.0, 0.5),
            Vec4::new(1.0, 1.0, 0.0, 0.5),
        ];
        let cell: &[u32] = &[0, 1, 2, 3];
        let cells: &[&[u32]] = &[cell];
        let edges: &[[u32; 2]] = &[[0, 1], [0, 2], [1, 3], [2, 3]];

        let mut strengths = Vec::new();
        compute_cell_strengths(cells, &local_vertices, w_slice, &mut strengths);
        assert!(
            strengths[0] > 0.0,
            "the cell must be active for this contract to mean anything"
        );
        for &[i, j] in edges {
            assert!(
                kept_by_cull(i, j, cells, &local_vertices, w_slice, thickness),
                "active cell's edge ({i},{j}) must be kept even at the razor band"
            );
        }
    }

    /// The cull still culls: an edge whose only containing cell sits entirely
    /// outside the slab is dropped.
    #[test]
    fn cull_drops_edge_when_no_containing_cell_overlaps() {
        let w_slice = 0.0_f32;
        let thickness = 0.2_f32; // slab [-0.1, 0.1]
        let local_vertices = [
            Vec4::new(0.0, 0.0, 0.0, 0.6),
            Vec4::new(1.0, 0.0, 0.0, 0.6),
            Vec4::new(0.0, 1.0, 0.0, 0.8),
            Vec4::new(1.0, 1.0, 0.0, 0.8),
        ];
        let cell: &[u32] = &[0, 1, 2, 3];
        let cells: &[&[u32]] = &[cell];
        // Cell w-range [0.6, 0.8] is entirely above the slab [-0.1, 0.1].
        assert!(!kept_by_cull(
            0,
            1,
            cells,
            &local_vertices,
            w_slice,
            thickness
        ));
        assert!(!kept_by_cull(
            2,
            3,
            cells,
            &local_vertices,
            w_slice,
            thickness
        ));
    }

    /// `cell_w_range` reproduces the `(w_min, w_max)` implicit in
    /// `compute_cell_strengths`, so the single-source split cannot drift.
    #[test]
    fn cell_w_range_matches_compute_cell_strengths() {
        let local_vertices = [
            Vec4::new(0.0, 0.0, 0.0, -0.3),
            Vec4::new(1.0, 0.0, 0.0, 0.1),
            Vec4::new(0.0, 1.0, 0.0, 0.7),
        ];
        let cell: &[u32] = &[0, 1, 2];
        let cells: &[&[u32]] = &[cell];

        let (w_min, w_max) = cell_w_range(cell, &local_vertices);
        assert_eq!((w_min, w_max), (-0.3, 0.7), "fold picks the w extremes");

        let mid = (w_min + w_max) * 0.5;
        let mut strengths = Vec::new();
        compute_cell_strengths(cells, &local_vertices, mid, &mut strengths);
        assert_eq!(
            strengths[0], 1.0,
            "strength at the cell's w-midpoint is the gradient peak"
        );
    }
}

#[cfg(test)]
mod alignment_tests {
    use super::*;

    /// Headless-render the `render_shapes_section` layout (minus the outer
    /// ScrollArea + Frame::popup) and capture each card + `+` button rect.
    fn capture_row_rects(row: &[ShapeEntry]) -> Vec<egui::Rect> {
        let ctx = egui::Context::default();
        let mut rects = Vec::new();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                // Top-align cross-axis (`Align::Min`) skips the
                // `frame_size.y = max(child, avail)` recursion Center uses, the
                // recursion that produced the converging staircase tops.
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

    /// All widgets share a top y (heights may vary; the `+` is 2pt shorter).
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

    /// Cards (all but the trailing `+`) have uniform height.
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

/// Drag-and-drop regression tests for `dnd_drag_source_collapsing`. The
/// headless driver simulates a press + drag-past-threshold and asserts drag
/// detection still wakes. Guards two regressions: (1) the helper works with
/// both `make_persistent_id` and `Id::new` keys; (2) wrapping the body in a
/// `Frame` does not eat the drag hit-test rect.
#[cfg(test)]
mod drag_tests {
    use super::*;

    fn screen() -> egui::Rect {
        egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 600.0))
    }

    /// Egui's drag detection compares `time - press_start_time` against
    /// `max_click_duration`, so frames must thread a monotonic clock or every
    /// press reads as still-within-click and never flips to dragging.
    fn pointer_press(time: f64, pos: egui::Pos2) -> egui::RawInput {
        egui::RawInput {
            screen_rect: Some(screen()),
            time: Some(time),
            events: vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: Default::default(),
                },
            ],
            ..Default::default()
        }
    }

    fn pointer_move(time: f64, pos: egui::Pos2) -> egui::RawInput {
        egui::RawInput {
            screen_rect: Some(screen()),
            time: Some(time),
            events: vec![egui::Event::PointerMoved(pos)],
            ..Default::default()
        }
    }

    fn warmup_input(time: f64) -> egui::RawInput {
        egui::RawInput {
            screen_rect: Some(screen()),
            time: Some(time),
            ..Default::default()
        }
    }

    /// Press + drag past the ~6 px threshold against
    /// `dnd_drag_source_collapsing`, then return the context so the caller can
    /// assert `is_being_dragged`.
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

    /// Baseline: stock `Ui::dnd_drag_source` starts a drag with the test
    /// driver. If this fails, the driver is wrong, not the helper.
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

    /// `egui::Id::new(...)` keys drive `dnd_drag_source_collapsing` as well as
    /// `make_persistent_id`. Regression: cards stopped responding to drags
    /// after a switch to `Id::new` per-row-index keys.
    #[test]
    fn id_new_starts_drag() {
        let id = egui::Id::new(("polytope-playground-shape-card-test", 0_usize));
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

    /// Regression: drag ids keyed by `egui::Id::new(...)` (not ui-scoped)
    /// collide across `BottomOverlay`'s two passes, breaking drag detection in
    /// release and tripping a `debug_assert!`. The fix scopes the id per pass
    /// via `make_persistent_id`. Area-routed input does not reach the headless
    /// interaction step, so this verifies the no-panic + distinct-id contract
    /// rather than drag detection directly.
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
        // A revert to `egui::Id::new(...)` would put the same id in two layers
        // and panic egui's `debug_assert!` here.
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

    /// Regression ("card snaps right for a frame"): the make-room gap's
    /// `open_width` must match the card slot's outer (Frame) width, else
    /// dropping a card shifts the row for one frame.
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

    /// The row's total width is invariant through the drag -> drop transition;
    /// a mismatch between the dragged card's drag-time and post-drop slot
    /// widths (or the make-room gap width) rubberbands the other cards on drop.
    #[test]
    fn shape_row_total_width_invariant_through_drop() {
        const N: usize = 4;
        const CARD_W: f32 = SHAPE_CARD_WIDTH + 8.0;
        const SPACING: f32 = 4.0;
        let ctx = egui::Context::default();
        // Compare widths with card 0 being dragged (drop at trailing slot N)
        // vs the post-drop scenario (no drag in flight).
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

        // Card 0 center = CARD_W/2 = 36.
        let card0_center = egui::pos2(CARD_W / 2.0, 9.0);
        let _ = ctx.run(warmup_input(0.0), |c| {
            render_during_drag(c, &mut total_during_drag)
        });
        let _ = ctx.run(pointer_press(0.05, card0_center), |c| {
            render_during_drag(c, &mut total_during_drag)
        });
        // Past the threshold and past the row to the trailing slot.
        let target_pos = egui::pos2(400.0, 9.0);
        let _ = ctx.run(pointer_move(0.10, target_pos), |c| {
            render_during_drag(c, &mut total_during_drag)
        });
        // Hold so the gap settles open at full width.
        for k in 0..15 {
            let t = 0.15 + (k as f64) * 0.02;
            let _ = ctx.run(pointer_move(t, target_pos), |c| {
                render_during_drag(c, &mut total_during_drag)
            });
        }
        let drag_total = total_during_drag;
        let dragged_id = ctx.dragged_id();
        // Release.
        let release_input = egui::RawInput {
            screen_rect: Some(screen()),
            time: Some(0.6),
            events: vec![
                egui::Event::PointerMoved(target_pos),
                egui::Event::PointerButton {
                    pos: target_pos,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: Default::default(),
                },
            ],
            ..Default::default()
        };
        let _ = ctx.run(release_input, |c| {
            render_during_drag(c, &mut total_during_drag)
        });
        // Post-drop frame: no gap, no collapse. Re-render to measure.
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

    /// `dnd_drag_source_collapsing` round-trips through a closure run in two
    /// egui layers without a same-id-in-two-layers panic.
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

    /// `make_persistent_id(...)` keys also start a drag; guards against
    /// hard-coding one id flavour.
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

#[cfg(test)]
mod section_cap_projection_tests {
    //! Tests for the section-cap world transform per wireframe projection. Affine
    //! modes take the scalar shim `perspective_scale_at_w -> Some(scale)`; non-
    //! affine modes return `None` and `cap_vertex_projected_and_world` projects
    //! the reconstructed 4D cap vertex per-vertex, matching the parent wireframe.
    use super::*;

    /// `perspective_scale_at_w` reports `Some(scale)` exactly for affine
    /// projections and `None` for non-affine ones; consumers branch on this, and
    /// a stray scalar on a non-affine arm would render a w-only-scaled ghost.
    #[test]
    fn perspective_scale_returns_none_for_non_affine() {
        // Affine: Identity at any w is unit scale.
        assert_eq!(
            perspective_scale_at_w(0.3, &loam_math::Projection::Identity),
            Some(1.0)
        );
        // Affine: Perspective4D at w_slice is `focal / (focal - w_slice)`.
        let focal = 2.0;
        let w_slice = 0.5;
        let got = perspective_scale_at_w(
            w_slice,
            &loam_math::Projection::Perspective4D {
                focal_distance: focal,
            },
        );
        assert_eq!(got, Some(focal / (focal - w_slice)));
        // Non-affine: both report `None`.
        assert_eq!(
            perspective_scale_at_w(0.0, &loam_math::Projection::Stereographic { pole: Vec4::W }),
            None
        );
        assert_eq!(
            perspective_scale_at_w(0.0, &loam_math::Projection::schlegel(Vec4::W, 0.5, 0.75)),
            None
        );
    }

    /// A cap vertex at `w = w_slice` lands at the same world R³ point via the
    /// affine shim and a direct per-vertex `Perspective4D` projection: scaling
    /// the dropped-w cap by `focal / (focal - w_slice)` IS the projection of
    /// `(x, y, z, w_slice)`, so the affine fast path is exact, not approximate.
    #[test]
    fn section_cap_matches_wireframe_under_perspective4d() {
        let focal = 2.0;
        let w_slice = 0.4;
        let proj = loam_math::Projection::Perspective4D {
            focal_distance: focal,
        };
        let body_pos = Vec3::new(1.3, -0.7, 0.2);
        let scale = perspective_scale_at_w(w_slice, &proj);
        assert!(scale.is_some(), "Perspective4D must take the affine shim");
        // Off-axis cap vertices sharing the slice's w.
        for cap_r3 in [[0.5, 0.0, 0.0], [0.0, 0.3, -0.2], [-0.4, 0.1, 0.6]] {
            // Affine shim path vs per-vertex projection of the reconstructed 4D
            // cap vertex (the wireframe path).
            let via_shim =
                cap_vertex_projected_and_world(cap_r3, w_slice, scale, &proj, body_pos).1;
            let p4 = Vec4::new(cap_r3[0], cap_r3[1], cap_r3[2], w_slice);
            let via_wireframe = (project_to_world(p4, &proj, body_pos)).to_array();
            for k in 0..3 {
                assert!(
                    (via_shim[k] - via_wireframe[k]).abs() < 1e-5,
                    "cap {cap_r3:?} component {k}: shim {} vs wireframe {}",
                    via_shim[k],
                    via_wireframe[k]
                );
            }
        }
    }

    /// Under Stereographic, the equatorial slice (`w_slice = 0`) is opposite the
    /// `+w` pole, so every reconstructed-and-projected cap vertex is finite,
    /// pinning the per-vertex non-affine path against NaN/Inf in the upload
    /// buffer. The test avoids the exact origin, which no cap vertex reaches and
    /// which `project_point` cannot normalize onto S³.
    #[test]
    fn section_cap_per_vertex_finite_under_stereographic() {
        let w_slice = 0.0;
        let proj = loam_math::Projection::Stereographic { pole: Vec4::W };
        let body_pos = Vec3::new(0.5, 0.0, -0.3);
        let scale = perspective_scale_at_w(w_slice, &proj);
        assert_eq!(scale, None, "Stereographic must take the per-vertex path");
        // Cap vertices across the equatorial 3-flat, small radius to near-shell.
        for cap_r3 in [
            [0.5, 0.0, 0.0],
            [0.0, -0.4, 0.3],
            [0.95, 0.0, 0.0],
            [0.02, -0.01, 0.015],
        ] {
            let world = cap_vertex_projected_and_world(cap_r3, w_slice, scale, &proj, body_pos).1;
            for (k, c) in world.iter().enumerate() {
                assert!(
                    c.is_finite(),
                    "cap {cap_r3:?} produced non-finite world component {k}: {c}"
                );
            }
        }
    }

    /// Edge-line preservation, flat endpoint chords, and cap scalar shortcuts
    /// are independent: Schlegel preserves straight edges yet needs per-vertex
    /// cap projection; Stereographic does not preserve a chord interior, but its
    /// flat wireframe is an endpoint-chord overlay.
    #[test]
    fn flat_edge_chord_policy_splits_from_cap_scale_policy() {
        let schlegel = loam_math::Projection::schlegel(Vec4::W, 0.5, 0.75);
        assert_eq!(
            perspective_scale_at_w(0.0, &schlegel),
            None,
            "Schlegel cap vertices need per-vertex projection"
        );
        assert!(
            projection_maps_chords_to_lines(&schlegel),
            "Schlegel central projection preserves straight edges"
        );
        assert!(
            flat_edge_uses_endpoint_chord(&schlegel),
            "flat Schlegel wireframe edges render as one chord"
        );

        let stereo = loam_math::Projection::Stereographic { pole: Vec4::W };
        assert_eq!(
            perspective_scale_at_w(0.0, &stereo),
            None,
            "stereographic cap vertices need per-vertex projection"
        );
        assert!(
            !projection_maps_chords_to_lines(&stereo),
            "stereographic does not preserve a sampled chord interior"
        );
        assert!(
            flat_edge_uses_endpoint_chord(&stereo),
            "flat stereographic wireframe edges render as comparison chords"
        );
    }

    /// Zero-blend stereographic is the comparison overlay: project endpoints and
    /// draw the R3 chord; the faithful S3 edge is sampled at blend one.
    #[test]
    fn stereographic_zero_blend_is_endpoint_chord_overlay() {
        let proj = loam_math::Projection::Stereographic { pole: Vec4::W };
        let body_pos = Vec3::ZERO;
        let a = Vec4::new(0.30, 0.60, 0.20, 0.50);
        let b = Vec4::new(0.70, 0.10, 0.40, -0.30);

        let mut mesh = LineMesh::<3>::default();
        let mut scratch = Vec::new();
        let white = [1.0, 1.0, 1.0, 1.0];
        push_blended_edge(
            &mut mesh,
            a,
            b,
            white,
            white,
            1.0,
            0.0,
            &proj,
            body_pos,
            &mut scratch,
            STEREOGRAPHIC_VIEW_RADIUS,
        );
        assert_eq!(
            mesh.segments.len(),
            1,
            "zero-blend stereographic should be one endpoint chord"
        );
        assert_eq!(
            scratch.len(),
            0,
            "zero-blend stereographic should not use the slerp scratch"
        );
        let expected_a = project_to_world(a, &proj, body_pos).to_array();
        let expected_b = project_to_world(b, &proj, body_pos).to_array();
        assert_eq!(mesh.segments[0].0, expected_a);
        assert_eq!(mesh.segments[0].1, expected_b);
    }

    /// Schlegel is not affine for cap scaling, but it is a central projection:
    /// flat R⁴ edges map to straight R³ chords. This catches the old
    /// "non-affine == must subdivide" mistake.
    #[test]
    fn schlegel_flat_wireframe_edge_is_endpoint_chord() {
        let proj = loam_math::Projection::schlegel(Vec4::W, 0.5, 1.0);
        assert_eq!(perspective_scale_at_w(0.0, &proj), None);
        let a = Vec4::new(0.25, 0.50, -0.25, 0.50);
        let b = Vec4::new(-0.25, 0.50, -0.25, -0.50);
        let mut mesh = LineMesh::<3>::default();
        let mut scratch = Vec::new();
        let white = [1.0, 1.0, 1.0, 1.0];
        push_blended_edge(
            &mut mesh,
            a,
            b,
            white,
            white,
            1.0,
            0.0,
            &proj,
            Vec3::ZERO,
            &mut scratch,
            STEREOGRAPHIC_VIEW_RADIUS,
        );
        assert_eq!(mesh.segments.len(), 1, "Schlegel edge is one chord");
        assert_eq!(
            mesh.segments[0].0,
            project_to_world(a, &proj, Vec3::ZERO).to_array()
        );
        assert_eq!(
            mesh.segments[0].1,
            project_to_world(b, &proj, Vec3::ZERO).to_array()
        );
    }

    /// Affine projections keep the single-segment fast path: a flat Perspective4D
    /// edge emits one segment and the cap vertex sits on it. Guards the perf-
    /// sensitive common case from the subdivided branch.
    #[test]
    fn affine_wireframe_keeps_single_segment_and_caps_land_on_it() {
        let proj = loam_math::Projection::Perspective4D {
            focal_distance: 2.0,
        };
        let body_pos = Vec3::ZERO;
        let w_slice = 0.0;
        let a = Vec4::new(0.5, 0.4, -0.3, 0.5);
        let b = Vec4::new(0.5, 0.4, -0.3, -0.5);
        let mut mesh = LineMesh::<3>::default();
        let mut scratch = Vec::new();
        let white = [1.0, 1.0, 1.0, 1.0];
        push_blended_edge(
            &mut mesh,
            a,
            b,
            white,
            white,
            1.0,
            0.0,
            &proj,
            body_pos,
            &mut scratch,
            STEREOGRAPHIC_VIEW_RADIUS,
        );
        assert_eq!(
            mesh.segments.len(),
            1,
            "affine flat edge must stay a single segment"
        );
        let mid = a.lerp(b, 0.5);
        let cap_r3 = [mid.x, mid.y, mid.z];
        let scale = perspective_scale_at_w(w_slice, &proj);
        let cap = Vec3::from_array(
            cap_vertex_projected_and_world(cap_r3, w_slice, scale, &proj, body_pos).1,
        );
        let (s, e) = mesh.segments[0];
        let gap = point_to_segment_distance(cap, Vec3::from_array(s), Vec3::from_array(e));
        assert!(
            gap < 1e-5,
            "affine cap must lie on its single-segment edge, gap {gap}"
        );
    }

    /// The two-layer world-transform invariant: the honest layer maps a cap
    /// vertex to the same world R³ point under every projection (forced drop-w by
    /// [`state::section_layer_projection`]), while the projected cap moves with
    /// the active projection. A projection change is non-destructive to the
    /// honest slice and effective on the projected cap.
    #[test]
    fn honest_section_cap_is_projection_invariant_projected_cap_is_not() {
        let body_pos = Vec3::new(0.7, -0.2, 0.4);
        let w_slice = 0.3;
        // Distinct spatial coords so a non-affine projection genuinely relocates
        // it (a pure-radial point could stay collinear).
        let cap_r3 = [0.4, -0.25, 0.15];
        let actives = [
            loam_math::Projection::Identity,
            loam_math::Projection::Perspective4D {
                focal_distance: 2.0,
            },
            loam_math::Projection::Stereographic { pole: Vec4::W },
            loam_math::Projection::schlegel(Vec4::W, 0.5, 0.9),
        ];

        // Honest layer (drop-w): identical under every active projection.
        let honest_reference = {
            let proj = state::section_layer_projection(true, loam_math::Projection::Identity);
            let scale = perspective_scale_at_w(w_slice, &proj);
            cap_vertex_projected_and_world(cap_r3, w_slice, scale, &proj, body_pos).1
        };
        let mut projected_caps = Vec::new();
        for active in actives {
            // Honest layer is drop-w regardless of `active`.
            let honest_proj = state::section_layer_projection(true, active);
            assert_eq!(
                honest_proj,
                loam_math::Projection::Identity,
                "honest layer must stay drop-w under {active:?}"
            );
            let honest_scale = perspective_scale_at_w(w_slice, &honest_proj);
            let honest = cap_vertex_projected_and_world(
                cap_r3,
                w_slice,
                honest_scale,
                &honest_proj,
                body_pos,
            )
            .1;
            for k in 0..3 {
                assert!(
                    (honest[k] - honest_reference[k]).abs() < 1e-6,
                    "honest cap drifted under {active:?}: {honest:?} vs {honest_reference:?}"
                );
            }

            // Projected layer follows `active`.
            let cap_proj = state::section_layer_projection(false, active);
            assert_eq!(cap_proj, active, "projected layer must follow {active:?}");
            let cap_scale = perspective_scale_at_w(w_slice, &cap_proj);
            projected_caps.push(
                cap_vertex_projected_and_world(cap_r3, w_slice, cap_scale, &cap_proj, body_pos).1,
            );
        }

        // At least one non-identity projection must relocate the projected cap
        // (Identity's equals the honest one by construction).
        let moved = projected_caps
            .iter()
            .any(|c| (0..3).any(|k| (c[k] - honest_reference[k]).abs() > 1e-4));
        assert!(
            moved,
            "projected cap must move under at least one active projection; \
             got {projected_caps:?} all equal to honest {honest_reference:?}"
        );
    }

    /// Distance from `p` to the segment `[s, e]` (clamped, not the infinite
    /// line); the "does the cap sit on this edge?" metric.
    fn point_to_segment_distance(p: Vec3, s: Vec3, e: Vec3) -> f32 {
        let d = e - s;
        let len_sq = d.length_squared();
        if len_sq < 1e-20 {
            return (p - s).length();
        }
        let t = ((p - s).dot(d) / len_sq).clamp(0.0, 1.0);
        (p - (s + t * d)).length()
    }

    // ---- Stereographic pole clip ----------------------------------------
    //
    // These pin the near-pole clip: a vertex on or near the projection pole
    // maps to the large-but-finite point the pole-denominator clamp produces,
    // and the wireframe builder DROPS the over-radius sub-segments (not a
    // magnitude rescale). The tests pin boundedness, finiteness, the no-rescale
    // segment-count discriminator, and non-perturbation off the pole.
    //
    // NOT pinned: flicker-freeness under rotation. A vertex crossing the pole is
    // a genuine projection discontinuity; the clip bounds and de-NaNs it but the
    // at-pole instant stays discontinuous, a visual property needing eyes-on.

    /// The per-shape, camera-adaptive clip radius. The 16-cell (the only shape
    /// with `+w`-pole vertices) stays below the camera distance (no rubberband),
    /// clears a unit-circumradius image (real geometry never clipped), stays
    /// under the pole-clamp ceiling, and saturates at
    /// [`STEREOGRAPHIC_CELL16_RADIUS_MAX`] on zoom-out. Every other shape is
    /// unclipped (`INFINITY`), its image naturally bounded.
    #[test]
    fn stereographic_view_radius_tracks_camera_distance() {
        // The worst non-pole vertex image (`+w`-cell corner at w = 0.5,
        // magnitude sqrt(3) ~ 1.73): the radius must always clear it.
        let legit = <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(
            Vec4::new(0.5, 0.5, 0.5, 0.5),
            &loam_math::Projection::Stereographic { pole: Vec4::W },
        )
        .length();
        let clamp_ceiling = (2.0 / loam_math::STEREOGRAPHIC_POLE_EPSILON).sqrt();

        // Across the zoom range the 16-cell radius stays a fixed fraction below
        // the camera distance, clears the figure, and sits under the clamp
        // ceiling. The strict-below-distance property (the rubberband fix) holds
        // above the floor's reach; at very close range the eye is inside.
        for distance in [2.0_f32, 4.0, 8.0, 16.0, 40.0] {
            let r = stereographic_view_radius(Polytope4::Cell16, distance);
            assert!(
                r > legit,
                "16-cell radius {r} at distance {distance} must clear the figure {legit}"
            );
            assert!(
                r < clamp_ceiling,
                "radius {r} must stay below the clamp ceiling"
            );
            if distance * STEREOGRAPHIC_VIEW_RADIUS_FRACTION >= STEREOGRAPHIC_VIEW_RADIUS_FLOOR {
                assert!(
                    r < distance,
                    "16-cell radius {r} must stay below camera distance {distance}"
                );
            }
            assert!(
                r <= STEREOGRAPHIC_CELL16_RADIUS_MAX,
                "16-cell radius {r} must never exceed the cap {STEREOGRAPHIC_CELL16_RADIUS_MAX}"
            );
        }

        // Zoom-out saturates at the cap, keeping far-zoom arcs smooth.
        assert_eq!(
            stereographic_view_radius(Polytope4::Cell16, 40.0),
            STEREOGRAPHIC_CELL16_RADIUS_MAX
        );
        // The test reference is the 16-cell value at an 8-unit camera distance.
        assert!(
            (stereographic_view_radius(Polytope4::Cell16, 8.0) - STEREOGRAPHIC_VIEW_RADIUS).abs()
                < 1e-5
        );

        // Every off-pole shape is unclipped at every distance: its image is
        // bounded, so we draw the full conformal extent (INFINITY).
        for polytope in [Polytope4::Tesseract, Polytope4::Cell24, Polytope4::Cell600] {
            for distance in [2.0_f32, 8.0, 40.0] {
                assert!(
                    stereographic_view_radius(polytope, distance).is_infinite(),
                    "{polytope:?} must be unclipped (no radius limit)"
                );
            }
        }
    }

    /// `stereographic_clip_radius` returns `Some(R)` only for Stereographic (the
    /// one projection with a point-at-infinity) and `None` elsewhere. A stray
    /// `Some` would clip legitimate geometry; a stray `None` would draw the pole
    /// blow-up.
    #[test]
    fn stereographic_clip_radius_only_for_stereographic() {
        assert_eq!(
            stereographic_clip_radius(
                &loam_math::Projection::Stereographic { pole: Vec4::W },
                STEREOGRAPHIC_VIEW_RADIUS
            ),
            Some(STEREOGRAPHIC_VIEW_RADIUS)
        );
        for proj in [
            loam_math::Projection::Identity,
            loam_math::Projection::Orthographic { drop_axis: 3 },
            loam_math::Projection::Perspective4D {
                focal_distance: 2.0,
            },
            loam_math::Projection::schlegel(Vec4::W, 0.5, 0.75),
        ] {
            assert_eq!(
                stereographic_clip_radius(&proj, STEREOGRAPHIC_VIEW_RADIUS),
                None,
                "non-stereographic projection {proj:?} must carry no clip"
            );
        }
    }

    /// Build the wireframe edge `a -> b` under `+w`-pole stereographic with
    /// `body_pos = ZERO`, so each world endpoint equals its body-local image.
    fn build_stereographic_edge_with_blend(
        a: Vec4,
        b: Vec4,
        blend: f32,
    ) -> Vec<([f32; 3], [f32; 3])> {
        let proj = loam_math::Projection::Stereographic { pole: Vec4::W };
        let mut mesh = LineMesh::<3>::default();
        let mut scratch = Vec::new();
        let white = [1.0, 1.0, 1.0, 1.0];
        push_blended_edge(
            &mut mesh,
            a,
            b,
            white,
            white,
            1.0,
            blend,
            &proj,
            Vec3::ZERO,
            &mut scratch,
            STEREOGRAPHIC_VIEW_RADIUS,
        );
        mesh.segments
    }

    fn build_stereographic_edge(a: Vec4, b: Vec4) -> Vec<([f32; 3], [f32; 3])> {
        build_stereographic_edge_with_blend(a, b, 0.0)
    }

    fn build_spherical_stereographic_edge(a: Vec4, b: Vec4) -> Vec<([f32; 3], [f32; 3])> {
        build_stereographic_edge_with_blend(a, b, 1.0)
    }

    /// A unit point at angular distance `theta_deg` from the `+w` pole, in the
    /// w-x plane. `theta_deg -> 0` approaches the pole singularity.
    fn near_pole(theta_deg: f32) -> Vec4 {
        let t = theta_deg.to_radians();
        Vec4::new(t.sin(), 0.0, 0.0, t.cos())
    }

    /// Zero-blend stereographic is an endpoint chord: if either endpoint clips
    /// out near the pole the flat chord drops, but the sampled S3 path resumes.
    #[test]
    fn stereographic_zero_blend_near_pole_uses_endpoint_clip() {
        let zero = build_stereographic_edge(near_pole(1.0), Vec4::new(1.0, 0.0, 0.0, 0.0));
        let spherical =
            build_spherical_stereographic_edge(near_pole(1.0), Vec4::new(1.0, 0.0, 0.0, 0.0));
        assert!(
            zero.is_empty(),
            "flat near-pole chord should drop when an endpoint clips out"
        );
        assert!(
            !spherical.is_empty(),
            "sampled S3 edge should resume after clipped near-pole samples"
        );
    }

    /// Boundedness: every emitted endpoint has magnitude <=
    /// `STEREOGRAPHIC_VIEW_RADIUS`, even on a pole-grazing edge (1 degree off the
    /// pole out to the equator), since the clip drops the near-pole blow-up.
    #[test]
    fn stereographic_clip_output_bounded_by_radius() {
        let segs =
            build_spherical_stereographic_edge(near_pole(1.0), Vec4::new(1.0, 0.0, 0.0, 0.0));
        assert!(
            !segs.is_empty(),
            "edge must emit at least one in-bounds segment"
        );
        let r = STEREOGRAPHIC_VIEW_RADIUS;
        for (s, e) in &segs {
            for end in [Vec3::from_array(*s), Vec3::from_array(*e)] {
                assert!(
                    end.length() <= r + 1e-3,
                    "emitted endpoint {end:?} (|.| = {}) exceeds clip radius {r}",
                    end.length()
                );
            }
        }
    }

    /// The clip cuts a straddling sub-segment AT the boundary and DROPS one
    /// running deep through the pole; neither a whole-segment drop nor a rescale.
    /// Fixture: an edge whose great circle passes through the `+w` pole
    /// (endpoints 30 degrees off, opposite sides). Three guarantees:
    /// (1) boundary cut, some endpoint within a hair of `R`, not stopped at the
    /// last in-radius sample; (2) deep-pole drop leaves a GAP (< samples), where
    /// a rescale-clamp would keep every segment; (3) every endpoint within `R`.
    #[test]
    fn stereographic_clip_cuts_to_boundary_and_drops_deep_pole() {
        let r = STEREOGRAPHIC_VIEW_RADIUS;
        // Endpoints 30 degrees off the pole, opposite sides; their great circle
        // passes through `+w` at its midpoint. Image magnitude cot(15 deg) ~ 3.73
        // < R keeps the endpoints while the midpoint samples blow up.
        let off = 30.0_f32.to_radians();
        let a = Vec4::new(off.sin(), 0.0, 0.0, off.cos());
        let b = Vec4::new(-off.sin(), 0.0, 0.0, off.cos());
        let segs = build_spherical_stereographic_edge(a, b);
        assert!(!segs.is_empty(), "kept endpoints must emit segments");

        // (1) Boundary cut: an endpoint sits within a hair of R.
        let max_extent = segs
            .iter()
            .flat_map(|(s, e)| [Vec3::from_array(*s).length(), Vec3::from_array(*e).length()])
            .fold(0.0_f32, f32::max);
        assert!(
            (max_extent - r).abs() < 1e-2,
            "straddling sub-segment must be cut to the boundary (max extent {max_extent}, R {r})"
        );

        // (2) Deep-pole drop: the through-pole samples vanish, leaving a gap.
        assert!(
            segs.len() < SPACE_TESSELLATION_SAMPLES,
            "deep-pole samples must drop (got {} of {}); a rescale-clamp would keep them all",
            segs.len(),
            SPACE_TESSELLATION_SAMPLES
        );

        // (3) Bounded.
        for (s, e) in &segs {
            for end in [Vec3::from_array(*s), Vec3::from_array(*e)] {
                assert!(
                    end.length() <= r + 1e-3,
                    "endpoint {end:?} (|.| = {}) exceeds the bound {r}",
                    end.length()
                );
            }
        }
    }

    /// Regression for the 16-cell near-pole artifacts under `xw` rotation: the
    /// visible arc tip must sit at the clip boundary and stay there as the vertex
    /// sweeps closer, with no popping and no dive toward the center. Fixture: the
    /// 16-cell edge `+e_w -> +e_y` (`+e_y` fixed, `+e_w` rotating to the pole), so
    /// the tip is the near-pole end. The `phi` sweep (3 deg, just past the
    /// view-radius crossing, down to 0.05 deg inside the clamp band) must hold `R`
    /// to within a unit; both prior defects (whole-segment drop, clamp-band
    /// deflation) leave the tip far below `R` somewhere in the sweep.
    #[test]
    fn stereographic_clip_arc_tip_holds_boundary_near_pole() {
        let r = STEREOGRAPHIC_VIEW_RADIUS;
        // `xw` rotation by `phi`: e_w -> (-sin phi, 0, 0, cos phi) sweeps to the
        // pole; e_y = (0, 1, 0, 0) is fixed.
        let tip_extent = |phi_deg: f32| -> f32 {
            let phi = phi_deg.to_radians();
            let a = Vec4::new(-phi.sin(), 0.0, 0.0, phi.cos());
            let b = Vec4::new(0.0, 1.0, 0.0, 0.0);
            build_spherical_stereographic_edge(a, b)
                .iter()
                .flat_map(|(s, e)| [Vec3::from_array(*s).length(), Vec3::from_array(*e).length()])
                .fold(0.0_f32, f32::max)
        };
        // Geometric sweep clusters steps near the pole, where the dive strikes.
        let samples = 60;
        let hi = 3.0_f32.ln();
        let lo = 0.05_f32.ln();
        for step in 0..=samples {
            let frac = step as f32 / samples as f32;
            let phi = (hi + (lo - hi) * frac).exp();
            let tip = tip_extent(phi);
            assert!(
                tip > r - 1.0 && tip <= r + 1e-2,
                "near-pole arc tip must hold the boundary R={r} at phi={phi} deg, got {tip}"
            );
        }
    }

    /// Finiteness: an edge with one endpoint exactly on the pole emits only
    /// finite, in-radius endpoints. The pole maps to the origin (zero numerator);
    /// its near-pole neighbors blow up and drop.
    #[test]
    fn stereographic_pole_endpoint_edge_is_finite_and_bounded() {
        let segs = build_spherical_stereographic_edge(Vec4::W, Vec4::new(1.0, 0.0, 0.0, 0.0));
        let r = STEREOGRAPHIC_VIEW_RADIUS;
        for (s, e) in &segs {
            for end in [Vec3::from_array(*s), Vec3::from_array(*e)] {
                assert!(
                    end.is_finite(),
                    "pole-edge endpoint must be finite: {end:?}"
                );
                assert!(
                    end.length() <= r + 1e-3,
                    "pole-edge endpoint {end:?} exceeds clip radius {r}"
                );
            }
        }
    }

    /// Non-perturbation off the pole: a well-clear spherical edge keeps every
    /// sub-segment, and each endpoint equals the raw `project_to_world` of its
    /// great-circle sample bit-for-bit. The clip is a pure post-filter; it never
    /// moves a retained sample.
    #[test]
    fn stereographic_clip_does_not_perturb_off_pole_edge() {
        let proj = loam_math::Projection::Stereographic { pole: Vec4::W };
        // A unit edge straddling w = 0, far from the +w pole on both ends.
        let a = Vec4::new(0.30, 0.60, 0.20, 0.10).normalize();
        let b = Vec4::new(0.70, 0.10, 0.40, -0.30).normalize();
        let segs = build_spherical_stereographic_edge(a, b);
        assert_eq!(
            segs.len(),
            SPACE_TESSELLATION_SAMPLES,
            "off-pole edge must retain every sub-segment (none clipped)"
        );
        // Reconstruct the un-clipped projected polyline directly and compare.
        let samples = SPACE_TESSELLATION_SAMPLES;
        let mut arc = Vec::new();
        <loam_math::SphericalS3Embedded as loam_math::RasterizableSpace<4>>::tessellate_segment(
            a, b, samples, &mut arc,
        );
        let mut prev = project_to_world(a, &proj, Vec3::ZERO).to_array();
        for (k, (seg, &sample)) in segs.iter().zip(arc.iter().skip(1)).enumerate() {
            let cur = project_to_world(sample, &proj, Vec3::ZERO).to_array();
            assert_eq!(seg.0, prev, "segment {k} start must match raw projection");
            assert_eq!(seg.1, cur, "segment {k} end must match raw projection");
            prev = cur;
        }
    }

    /// The clip adds no per-edge allocation: re-building a pole-grazing blended
    /// edge leaves `slerp_scratch` at its first-edge capacity. The clip is a
    /// streaming `continue`, not a `filter().collect()`.
    #[test]
    fn stereographic_clip_reuses_scratch_without_realloc() {
        let proj = loam_math::Projection::Stereographic { pole: Vec4::W };
        let white = [1.0, 1.0, 1.0, 1.0];
        let mut scratch = Vec::new();
        let mut mesh = LineMesh::<3>::default();
        // Blend > 0 populates the slerp buffer; one endpoint near the pole so the
        // clip drops interior samples.
        let a = near_pole(1.0);
        let b = Vec4::new(1.0, 0.0, 0.0, 0.0);
        push_blended_edge(
            &mut mesh,
            a,
            b,
            white,
            white,
            0.5,
            1.0,
            &proj,
            Vec3::ZERO,
            &mut scratch,
            STEREOGRAPHIC_VIEW_RADIUS,
        );
        let cap_after_first = scratch.capacity();
        // The slerp buffer holds `samples + 1` points.
        assert!(cap_after_first > SPACE_TESSELLATION_SAMPLES);
        // Re-run: cleared and refilled to the same length, no growth.
        push_blended_edge(
            &mut mesh,
            a,
            b,
            white,
            white,
            0.5,
            1.0,
            &proj,
            Vec3::ZERO,
            &mut scratch,
            STEREOGRAPHIC_VIEW_RADIUS,
        );
        assert_eq!(
            scratch.capacity(),
            cap_after_first,
            "clip must not grow the reused slerp scratch"
        );
    }

    // ---- Cap-fill + points-overlay near-pole drop ----------------------
    //
    // These pin gaps the perimeter outline already covered: the projected-cap
    // FILL (`retain_in_radius_triangles`, triangle granularity) and the points
    // overlay (`sample_in_radius` per vertex / cell-center). Both reuse the same
    // predicate as the wireframe edges and perimeter, so all four cull on one
    // drop cone. No flicker-freeness claim; see the note above.

    /// Demo default pole, kept in sync with the state constant by
    /// `state::stereographic_default_pole_is_unit_cell_center`.
    fn default_stereographic() -> loam_math::Projection<4> {
        loam_math::Projection::Stereographic {
            pole: state::STEREOGRAPHIC_DEFAULT_POLE,
        }
    }

    /// The body-local projected point a cap vertex maps to; the point the fill /
    /// perimeter / points clip all test.
    fn cap_projected(cap_r3: [f32; 3], w_slice: f32, proj: &loam_math::Projection<4>) -> Vec3 {
        let scale = perspective_scale_at_w(w_slice, proj);
        cap_vertex_projected_and_world(cap_r3, w_slice, scale, proj, Vec3::ZERO).0
    }

    /// The cap FILL drops, at triangle granularity, any triangle touching a
    /// near-pole vertex and keeps an all-far fan. A whole-triangle drop (no
    /// boundary cut), mirroring the deep-pole drop for the fill path.
    #[test]
    fn cap_fill_triangle_dropped_near_pole() {
        let r = STEREOGRAPHIC_VIEW_RADIUS;
        // Points 0,1 inside the radius, 2 outside (near-pole blow-up). Two fans
        // share centroid 0: [0,1,2] touches the near-pole vertex, [0,1,1] not.
        let projected = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(r * 2.0, 0.0, 0.0),
        ];
        let mut indices = vec![[0u32, 1, 2], [0u32, 1, 1]];
        retain_in_radius_triangles(&mut indices, 0, 0, &projected, Some(r));
        assert_eq!(
            indices,
            vec![[0u32, 1, 1]],
            "the triangle touching the near-pole vertex must be dropped, the far one kept"
        );

        // All-far fan: nothing dropped, bit-identical to the input.
        let all_far = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.0),
        ];
        let mut far_indices = vec![[0u32, 1, 2]];
        retain_in_radius_triangles(&mut far_indices, 0, 0, &all_far, Some(r));
        assert_eq!(
            far_indices,
            vec![[0u32, 1, 2]],
            "a fan entirely within the radius must keep every triangle"
        );

        // Affine layer (`None`): every triangle kept regardless of magnitude.
        let mut affine_indices = vec![[0u32, 1, 2]];
        retain_in_radius_triangles(&mut affine_indices, 0, 0, &projected, None);
        assert_eq!(
            affine_indices,
            vec![[0u32, 1, 2]],
            "no clip (affine) must keep every triangle even past the radius"
        );
    }

    /// Fill and perimeter cull in lockstep: the fill's
    /// `retain_in_radius_triangles` agrees with the perimeter's per-segment
    /// `sample_in_radius` on the same projected point and radius; both drop when
    /// the magnitude exceeds the radius.
    #[test]
    fn cap_fill_matches_perimeter_clip() {
        let proj = loam_math::Projection::Stereographic { pole: Vec4::W };
        let clip = stereographic_clip_radius(&proj, STEREOGRAPHIC_VIEW_RADIUS);
        // A near-pole cap vertex (off-axis, large finite image) and a far one.
        let near = cap_projected([0.05, 0.02, 0.01], 0.999, &proj);
        let far = cap_projected([0.5, 0.0, 0.0], 0.0, &proj);
        // The perimeter drops a segment when EITHER endpoint fails the test.
        let perimeter_keeps_near = sample_in_radius(near, clip);
        let perimeter_keeps_far = sample_in_radius(far, clip);
        assert!(
            !perimeter_keeps_near,
            "near-pole cap projected to {near:?} (|.| = {}) must fail the clip",
            near.length()
        );
        assert!(perimeter_keeps_far, "far cap must pass the clip");
        // The fill compaction must drop the near-touching fan iff the perimeter
        // would, on the same points + radius.
        let projected = [Vec3::ZERO, far, near];
        let mut indices = vec![[0u32, 1, 2]];
        retain_in_radius_triangles(&mut indices, 0, 0, &projected, clip);
        let fill_keeps = !indices.is_empty();
        assert_eq!(
            fill_keeps,
            perimeter_keeps_near && perimeter_keeps_far,
            "fill triangle keep/drop must match the perimeter's endpoint test"
        );
        assert!(!fill_keeps, "the near-pole fan must be dropped");
    }

    /// The points overlay drops a near-pole vertex and keeps a far one, the same
    /// `sample_in_radius` gate `render_points` applies per vertex / cell-center.
    /// Affine projection (`clip_radius == None`) keeps every point.
    #[test]
    fn points_overlay_drops_near_pole_vertex() {
        let proj = loam_math::Projection::Stereographic { pole: Vec4::W };
        let clip = stereographic_clip_radius(&proj, STEREOGRAPHIC_VIEW_RADIUS);
        // Project a near-+w vertex as render_points does, then apply the gate.
        let v_near = <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(
            near_pole(1.0),
            &proj,
        );
        let v_far = <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            &proj,
        );
        assert!(
            !sample_in_radius(v_near, clip),
            "near-pole vertex (|.| = {}) must be dropped from the points overlay",
            v_near.length()
        );
        assert!(
            sample_in_radius(v_far, clip),
            "far vertex (|.| = {}) must be kept",
            v_far.length()
        );
        // Affine projection carries no clip; its image is bounded anyway.
        let affine_clip =
            stereographic_clip_radius(&loam_math::Projection::Identity, STEREOGRAPHIC_VIEW_RADIUS);
        assert!(
            sample_in_radius(v_near, affine_clip),
            "affine projection must keep every point (no clip)"
        );
    }

    /// The default-pole render path applies the cap clip: a cap vertex near `+w`
    /// drops, a far one is kept. Pins that `resolved_wireframe_projection`'s pole
    /// substitution flows into the cap fill without re-deriving the projection.
    #[test]
    fn cap_fill_uses_default_plus_w_pole() {
        let proj = default_stereographic();
        let clip = stereographic_clip_radius(&proj, STEREOGRAPHIC_VIEW_RADIUS);
        // `(0.05, 0, 0, 1.0)` normalizes to dot ~ 0.99875 with +w, image ~40
        // past the ~35 radius, so it drops.
        let near = cap_projected([0.05, 0.0, 0.0], 1.0, &proj);
        assert!(
            !sample_in_radius(near, clip),
            "cap vertex near the +w pole (|.| = {}) must drop",
            near.length()
        );
        // A point far from +w (w = 0) stays bounded, finite, and kept.
        let far = cap_projected([-0.4, -0.3, 0.2], 0.0, &proj);
        assert!(
            far.is_finite() && sample_in_radius(far, clip),
            "off-pole cap vertex must stay finite + in-radius: {far:?}"
        );
    }

    /// The cap-fill clip scratch retains capacity across two compactions, so the
    /// hot path has no per-frame allocation. The fill-path counterpart to
    /// `stereographic_clip_reuses_scratch_without_realloc`.
    #[test]
    fn cap_fill_scratch_reused_without_realloc() {
        let r = STEREOGRAPHIC_VIEW_RADIUS;
        // Simulate `build_section_layer_meshes`'s per-body fill: push, compact,
        // clear + re-push, across two frames.
        let mut proj_scratch: Vec<Vec3> = Vec::new();
        let fill = |scratch: &mut Vec<Vec3>| {
            scratch.clear();
            scratch.push(Vec3::ZERO);
            scratch.push(Vec3::new(1.0, 0.0, 0.0));
            scratch.push(Vec3::new(r * 2.0, 0.0, 0.0));
            let mut indices = vec![[0u32, 1, 2], [0u32, 1, 1]];
            retain_in_radius_triangles(&mut indices, 0, 0, scratch, Some(r));
        };
        fill(&mut proj_scratch);
        let cap_after_first = proj_scratch.capacity();
        assert!(cap_after_first >= 3);
        fill(&mut proj_scratch);
        assert_eq!(
            proj_scratch.capacity(),
            cap_after_first,
            "fill clip must reuse the projected-point scratch without growth"
        );
    }
}
