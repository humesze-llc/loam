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
use glam::{Mat4, Vec2, Vec3, Vec4};
use rye_app::{
    egui,
    freecam::{CursorMode, Freecam},
    App, Camera, CameraController, FrameCtx, OrbitController, RunConfig, SetupCtx,
};
use rye_egui::Console;
use rye_math::WPlane;
use rye_math::{Bivector, EuclideanR3, Rotor, Rotor4};
use rye_physics::polytope::{
    polytope_section_faces_append, polytope_section_overlay_with_vertices, vertex_color_by_position,
};
use rye_render::{
    device::RenderDevice,
    raymarch::{
        polytope_extended_sdfs_wgsl, BodyUniform, Hyperslice4DNode, HYPERSLICE_KERNEL_WGSL,
    },
    DepthBuffer, DepthMode, LineRasterNode, PointRasterNode, TriangleRasterNode, Viewport,
};

/// Depth-attachment format for the rasterized section-faces pass. 32-bit float gives
/// enough precision to depth-sort cell caps from the 600-cell (which can have ~24
/// active caps within tenths of a unit of camera-z) without artifacting.
const SECTION_FACES_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

use rye_scene::{Scene4, SceneNode4};
use rye_shape::LineMesh;
use winit::window::WindowAttributes;

mod active;
mod catalog;
mod color;
mod composer;
mod consts;
mod filmstrip;
mod render;
mod shapes;
mod state;
mod ui;
mod wireframe_geom;

use active::combo_name;
use catalog::{parse_row_from_args, SHAPE_CATALOG};
use color::{unique_edge_palette, w_depth_color};
#[cfg(test)]
use consts::SPACE_TESSELLATION_SAMPLES;
use consts::{
    BODY_SIZE, BODY_Y, HYPERSLICE_MIN_THICKNESS, T_SCRUB_RATE, T_SLIDER_INITIAL, W_SCRUB_RATE,
};
#[cfg(test)]
use rye_physics::polytope::Polytope4;
use state::{
    body_position, CameraMode, Demo, RotationMode, SurfaceMode, ViewMode, WireframeColorMode,
    WireframeProjection,
};
use wireframe_geom::*;

/// Per-cell "crossing strength" in `[0, 1]`: 1 when `w_slice` sits at the
/// cell's w-midpoint (the cap face is widest); 0 when the slice is outside
/// the cell's w-range entirely. Linear in `|w_slice - midpoint|` normalized
/// by the cell's half-extent, a cheap proxy for the actual cap area that
/// gives the same visual gradient. Shared by `render_wireframe_overlay` and
/// `render_points` so both honor the same cell-activity definition.
fn compute_cell_strengths(cells: &[&[u32]], local_vertices: &[Vec4], w_slice: f32) -> Vec<f32> {
    cells
        .iter()
        .map(|cell| {
            let (w_min, w_max) = cell_w_range(cell, local_vertices);
            let half_extent = (w_max - w_min) * 0.5;
            if half_extent <= 0.0 {
                return 0.0;
            }
            let mid = (w_min + w_max) * 0.5;
            let dist = (w_slice - mid).abs();
            (1.0 - dist / half_extent).clamp(0.0, 1.0)
        })
        .collect()
}

#[cfg(test)]
use catalog::{parse_shape_name, ShapeEntry, DEFAULT_ROW};
#[cfg(test)]
use consts::{CONTROL_H, CONTROL_W, SHAPE_CARD_WIDTH};
#[cfg(test)]
use rye_egui::dnd::{drag_source_collapsing as dnd_drag_source_collapsing, make_room_gap};
#[cfg(test)]
use rye_egui::media::add_button;

impl Demo {
    pub(crate) fn new(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let row = parse_row_from_args()?;
        if row.is_empty() {
            return Err(anyhow!("--shapes produced an empty row"));
        }

        let scene = Scene4::new(SceneNode4::halfspace(Vec4::Y, 0.0));
        // Always include the extended polytope WGSL so any of the six
        // shapes can be added to the row at runtime via the panel.
        // The ~24 KB const-array cost is fixed per app and acceptable
        // for a viz/demo target.
        //
        // Floor visibility is gated at runtime via `u.params.x` (set by
        // [`Demo::floor_enabled`] each frame in `update`): when 0.0 the
        // halfspace SDF returns 1e9 + the ray-plane bound is skipped, so
        // the marcher never converges on the floor and the checkerboard
        // never paints. Engine-level support: see
        // [`Scene4::to_hyperslice_wgsl_gated`]. Zero per-frame cost; one
        // shader build per launch.
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

        // Initial body uniforms for the SDF kernel. With the surface default flipped to
        // raster, polychoral entries are emitted as `BodyUniform::default()` (kind =
        // Invalid) so the kernel skips them; the section-faces rasterizer draws them
        // instead. Mirrors `Demo::sdf_body_for_slot` for the initial upload; subsequent
        // re-uploads go through that helper directly.
        let n = row.len();
        let bodies: Vec<BodyUniform> = row
            .iter()
            .enumerate()
            .map(|(slot, entry)| {
                if entry.shape.polytope4().is_some() {
                    BodyUniform::default()
                } else {
                    BodyUniform::polytope_with_rotor(
                        body_position(slot, n),
                        entry.shape.shape_id(),
                        BODY_SIZE,
                        Rotor4::IDENTITY,
                        entry.body_color,
                    )
                }
            })
            .collect();
        node.set_bodies(&bodies);

        // Section perimeter (cyan outlines): depth-test ReadOnly against the shared
        // section-faces depth attachment. With multiple polychora in a row, polytope A's
        // perimeter must be occluded by polytope B's filled caps when A sits behind B.
        // `LineRasterNode` uses `CompareFunction::LessEqual`, so a perimeter line sitting
        // at exactly its own cap's depth still passes the test and draws on top, keeping
        // the intended "outline of this cap" visual.
        let section_edges = LineRasterNode::new(
            &ctx.rd.device,
            ctx.rd.target_format(),
            DepthMode::ReadOnly {
                format: SECTION_FACES_DEPTH_FORMAT,
            },
            ctx.rd.sample_count(),
        );
        // Parent wireframe: depth-test ReadOnly against the shared section-faces depth
        // attachment. Lines whose projected R³ position sits behind a section cap get
        // occluded by it; lines in front draw over. No depth-write so the wireframe
        // doesn't muddy the depth buffer for any downstream pass. In SDF mode the
        // depth buffer is cleared per frame but no pass writes to it, so the test
        // trivially passes everywhere -- the SDF visual stays unchanged.
        let parent_wireframe = LineRasterNode::new(
            &ctx.rd.device,
            ctx.rd.target_format(),
            DepthMode::ReadOnly {
                format: SECTION_FACES_DEPTH_FORMAT,
            },
            ctx.rd.sample_count(),
        );

        // Point-disc rasterizer for the optional vertex + cell-center sprites overlay.
        // No depth attachment: points are debug markers that must read as an overlay
        // regardless of where the section caps sit. With the previous ReadOnly setup,
        // a vertex at non-zero w would project (via drop-w) to the SAME (x, y, z) as
        // its enclosing cap's slice intersection but with a slightly farther camera
        // depth, so `LessEqual` failed and the sprite vanished behind the cap. The
        // overlay semantic ("show me where the vertices are") wants always-visible.
        let points_node = PointRasterNode::new(
            &ctx.rd.device,
            ctx.rd.target_format(),
            DepthMode::Off,
            ctx.rd.sample_count(),
        );

        // Rasterized cross-section faces: filled cell-caps with face-normal Lambert
        // shading. Uses a depth attachment so caps from different cells of the same
        // polychoron occlude each other correctly when projected to camera space. The
        // depth buffer is sized + cleared per-frame inside the render path (only when
        // surface mode is `Raster`); see `Demo::render_section_faces` in this file.
        let section_faces = TriangleRasterNode::new(
            &ctx.rd.device,
            ctx.rd.target_format(),
            DepthMode::ReadWrite {
                format: SECTION_FACES_DEPTH_FORMAT,
            },
            rye_render::FragmentShading::FaceNormalLambert,
            ctx.rd.sample_count(),
        );
        // Translucent variant: identical pipeline modulo depth-write. When
        // `surface_alpha < 1.0` we render through this one so the parent
        // wireframe (drawn AFTER section faces with `LessEqual` depth-test)
        // can show through the cap. The ReadWrite variant above writes
        // depth, which would otherwise hide wireframe edges sitting behind
        // a cap regardless of the cap's alpha.
        //
        // Tradeoff: with ReadOnly depth, caps within a single polytope can
        // overpaint each other in submission order when their R³
        // projections overlap. In practice the section-cap of one cell is
        // disjoint from the section-cap of another cell at the same
        // w_slice (each cell hosts at most one cap and they tile the
        // section), so overdraw is rare. If it surfaces on the 24-cell or
        // 600-cell at oblique angles, the fix is a depth-only prepass +
        // ReadOnly color pass; punted until we observe the artifact.
        let section_faces_translucent = TriangleRasterNode::new(
            &ctx.rd.device,
            ctx.rd.target_format(),
            DepthMode::ReadOnly {
                format: SECTION_FACES_DEPTH_FORMAT,
            },
            rye_render::FragmentShading::FaceNormalLambert,
            ctx.rd.sample_count(),
        );

        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 3.0, 9.0);
        let mut orbit: OrbitController<EuclideanR3> = OrbitController::default();
        // Default framing: all four bodies in the row visible at startup. `8.0`
        // is the original startup distance (the old code asked for 9.5 but the
        // pre-bump MAX_DISTANCE = 8 clamped it); the raised MAX_DISTANCE only
        // widens the scroll-out range, it does not push the startup view back.
        orbit.set_orbit(8.0, -0.25);

        // Freecam preset; inactive at startup. The `camera freecam` console
        // command calls `freecam.set_active(true, camera.position)` which
        // grabs the cursor + seeds the freecam position from the camera.
        let freecam = Freecam::new();

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
            // Default projection is drop-w, so no Schlegel cache is needed at
            // startup; it is resolved the moment the user selects Schlegel.
            schlegel_params: None,
            stereographic_pole: state::STEREOGRAPHIC_DEFAULT_POLE,
            wireframe_hyperslice: false,
            wireframe_hyperslice_thickness: consts::HYPERSLICE_DEFAULT_THICKNESS,
            wireframe_width_px: 1.8,
            wireframe_alpha: 1.0,
            unique_edge_palette_cache: std::collections::HashMap::new(),
            surface_scale: 1.0,
            floor_enabled: true,
            section_faces,
            section_faces_translucent,
            section_faces_projected_scratch: rye_shape::TriangleMesh::<3>::default(),
            section_clip_projected_scratch: Vec::new(),
            points_node,
            points_enabled: false,
            points_show_vertices: true,
            points_show_cell_centers: true,
            points_size_px: 4.0,
            points_mesh_scratch: rye_shape::PointMesh::<3>::default(),
            section_faces_depth: None,
            section_world_vertices_scratch: Vec::new(),
            section_faces_mesh_scratch: rye_shape::TriangleMesh::<3>::default(),
            body_uniform_scratch: Vec::new(),
            slerp_scratch: Vec::new(),
            surface_mode: SurfaceMode::default(),
            row,
            w_slice: initial_w,
            slider_up_held: false,
            slider_down_held: false,
            slider_left_held: false,
            slider_right_held: false,
            rotate: false,
            rot_state: Rotor4::IDENTITY,
            // Default: xw spin enabled (active[2] = Plane4::Xw). A
            // first-time user who hits "Spin" before toggling any
            // checkbox now sees motion immediately; the most
            // characteristic 4D rotation, pulling the visible x-axis
            // through the hidden w-axis.
            active: [false, false, true, false, false, false],
            base_angles: [0.0; 6],
            rate_scale: 1.0,
            rot_time: 0.0,
            t_slider_max: T_SLIDER_INITIAL,
            expanded: false,
            show_help: false,
            show_render_panel: false,
            example_callout: rye_egui::CalloutState {
                window_pos: egui::Pos2::new(220.0, 120.0),
                open: false,
            },
            // Unwired: the per-projection annotation callout is not shown for now
            // (defaults closed, and the View toggle is removed), so selecting a
            // wireframe type no longer spawns a panel. `render_mode_annotation`
            // and `state::mode_annotation` are kept; the render call early-returns
            // while `open` stays false. Default position retained for a re-wire.
            mode_annotation_open: rye_egui::CalloutState {
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

    pub(crate) fn space(&self) -> &EuclideanR3 {
        &self.space
    }

    pub(crate) fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        let dt_secs = ctx.n_ticks as f32 / 60.0;

        // Slice scrub (w axis, up/down arrow keys). Clamps against the
        // surface-scaled range so the keyboard scrub matches the slider
        // bounds after `surface scale`.
        let dir = (self.slider_up_held as i32 - self.slider_down_held as i32) as f32;
        if dir != 0.0 {
            let w_range = self.effective_w_range();
            self.w_slice = (self.w_slice + dir * W_SCRUB_RATE * dt_secs).clamp(-w_range, w_range);
        }

        // Time scrub (t axis, left/right arrow keys). Mirrors the
        // t-slider drag: rebuild `rot_state` from the new `rot_time`
        // via `rotor_at_time`, which dispatches Active (product) vs
        // Composer (sum) so the scrub matches the spin path's math.
        // Right = forward in time, left = back. Floors `rot_time`
        // at zero (the t slider's lower bound).
        let t_dir = (self.slider_right_held as i32 - self.slider_left_held as i32) as f32;
        if t_dir != 0.0 {
            self.rot_time = (self.rot_time + t_dir * T_SCRUB_RATE * dt_secs).max(0.0);
            // Same runaway guard as the spin path: grow the slider
            // range if scrub pushes us past current max.
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
            // (per-real-second).
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
        }
        // Recompose `rot_state` each frame so spin advances (Active mode
        // reads `rot_time` through `active_displayed_angle`; Composer
        // integrates the omega-bivector into rot_state directly via the
        // legacy path below).
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
        if !ctx.ui_has_focus {
            match self.camera_mode {
                CameraMode::Orbit => {
                    self.orbit
                        .advance(ctx.input, &mut self.camera, &EuclideanR3, dt_secs);
                }
                CameraMode::FreeRoam => {
                    // Freecam preset handles look + WASD + cursor-grab
                    // gating internally. No-ops when cursor is released
                    // (Alt-toggled UI-access mode), so the scene holds
                    // still while the user interacts with UI.
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
            // Floor-toggle gate read by the wrapper around `rye_scene_sdf`
            // we injected at App::setup time. `u.params[0]` is 1.0 = floor
            // on (the canonical halfspace SDF runs), 0.0 = floor off (the
            // wrapper short-circuits to 1e9 so the marcher never converges
            // on the floor and the checkerboard never paints).
            u.params[0] = if self.floor_enabled { 1.0 } else { 0.0 };
        }
        self.node.flush_uniforms(&ctx.rd.queue);
    }

    pub(crate) fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
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

        // Top-right: short git hash + dirty marker. Identifies the build at
        // a glance when a tester reloads the wasm bundle; the browser cache
        // can serve a stale page+script combination otherwise. F3's perf
        // overlay shows fps + framebuffer size when that data is wanted.
        // Build label: short git hash + dirty marker, rendered top-right with
        // visually symmetric padding (same gap from menu-bar bottom as from
        // window's right edge). `MENU_BAR_PAD + LABEL_INSET` lands the label
        // 14 px below the menu bar; `-LABEL_INSET` puts it 14 px in from the
        // right edge. Matching values keep the two whitespaces equal.
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
        // Floating render-settings modal (opened by the gear button in the bottom
        // overlay; off by default so the scene fills the window for first-launch
        // viewing).
        self.render_render_panel(ctx);
        // Example annotation callout (off by default; toggle via View > Example
        // callout). Demonstrates the rye_egui::callout primitive against the first
        // polychoron in the row.
        self.render_example_callout(ctx, frame);
        // Per-mode educational annotation (on by default; toggle via View > Mode
        // annotation). No-ops in the default drop-w + flat-space scene; otherwise
        // explains the active projection / edge-geometry combination.
        self.render_mode_annotation(ctx, frame);
    }

    /// Surface the per-projection / per-space-mode educational annotation via the
    /// `rye_egui::callout` primitive, anchored to the leading polychoron's body
    /// center. The text is the pure [`state::mode_annotation`] mapping of the
    /// active `wireframe_projection`, reprojected per frame so the leader line
    /// tracks the shape as the camera orbits. No-op when the toggle is off, the
    /// row has no polychoron, or the projection is the plain default (drop-w),
    /// where the mapping returns `None` and there is nothing to explain.
    ///
    /// Anchoring to the body center (not a single vertex like
    /// [`Self::render_example_callout`]) is deliberate: the annotation is about the
    /// whole shape's projection, not one vertex, so the body center is the honest
    /// anchor.
    fn render_mode_annotation(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
        if !self.mode_annotation_open.open {
            return;
        }
        let Some(annotation) = state::mode_annotation(self.wireframe_projection) else {
            return;
        };

        // Anchor: the leading polychoron's body center in world R³. Same
        // render-row selection the example callout and every per-body path use, so
        // the annotation tracks whichever shape the projection diagram is about.
        let render_row = state::render_row_entries(self.view_mode, &self.row, &self.strip_subject);
        let n = render_row.len();
        let Some((slot, _entry)) = render_row
            .iter()
            .enumerate()
            .find(|(_, e)| e.shape.polytope4().is_some())
        else {
            return;
        };
        let body_pos = body_position(slot, n);
        let world_pos = Vec3::new(body_pos[0], body_pos[1], body_pos[2]);

        let view_dir = self.camera.view();
        let cfg = &frame.rd.surface_bundle.config;
        let ppp = ctx.pixels_per_point();
        let vp_w = (cfg.width as f32 / ppp).round() as u32;
        let vp_h = (cfg.height as f32 / ppp).round() as u32;
        let Some(screen_pos) = rye_egui::world_to_screen(
            world_pos,
            &view_dir,
            60.0_f32.to_radians(),
            (vp_w, vp_h),
            0.1,
            100.0,
        ) else {
            return;
        };

        rye_egui::callout(
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

    /// Demonstrate the `rye_egui::callout` primitive against the first polychoron in
    /// the row. The anchor follows vertex 0 of that polytope's canonical topology
    /// through the current rotor + body position + wireframe projection chain,
    /// reprojected per frame so the line tracks live as the polytope rotates. No-op
    /// when the row is empty or contains no polychora.
    fn render_example_callout(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
        if !self.example_callout.open {
            return;
        }
        // Find the first polychoron in the RENDERED row (the lone `strip_subject`
        // in Single mode); its vertex 0 is the anchor target.
        let render_row = state::render_row_entries(self.view_mode, &self.row, &self.strip_subject);
        let n = render_row.len();
        let Some((slot, entry)) = render_row
            .iter()
            .enumerate()
            .find(|(_, e)| e.shape.polytope4().is_some())
        else {
            return;
        };
        let polytope = entry.shape.polytope4().expect("filter guarantees Some");
        let topo = polytope.topology();
        let canonical_v0 = topo.vertices[0];
        let v_local_4d = self.effective_body_size() * self.rot_state.apply(canonical_v0);
        let v_local_r3 = <rye_math::EuclideanR4 as rye_math::RasterizableSpace<4>>::project_point(
            v_local_4d,
            &self.resolved_wireframe_projection(),
        );
        let body_pos = body_position(slot, n);
        let world_pos = v_local_r3 + Vec3::new(body_pos[0], body_pos[1], body_pos[2]);

        // Reproject world R³ -> screen pixels via the same camera the rasterizer
        // chain uses. `world_to_screen` does the perspective + NDC + viewport-flip
        // math; it returns `None` when the anchor is offscreen (behind the camera or
        // outside the viewing frustum), in which case the callout draws nothing.
        let view_dir = self.camera.view();
        let cfg = &frame.rd.surface_bundle.config;
        let ppp = ctx.pixels_per_point();
        let vp_w = (cfg.width as f32 / ppp).round() as u32;
        let vp_h = (cfg.height as f32 / ppp).round() as u32;
        let Some(screen_pos) = rye_egui::world_to_screen(
            world_pos,
            &view_dir,
            60.0_f32.to_radians(),
            (vp_w, vp_h),
            0.1,
            100.0,
        ) else {
            return;
        };

        let title = format!("{} vertex 0", entry.label);
        rye_egui::callout(
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
                     (rye_egui::callout) is the foundation for future tutorial \
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
                // Pause / resume only, DO NOT touch rot_state. The bodies
                // keep their current orientation when paused and resume
                // from there when toggled back on.
                self.rotate = !self.rotate;
            }
            // Space ALSO toggles rotation, but only outside freecam mode.
            // In freecam Space is bound to the move-up axis (Space = +1
            // on `FrameInput::move_up`); rotating-on-Space would conflict.
            // T remains the always-available rotation toggle for freecam
            // users.
            KeyCode::Space if pressed && !matches!(self.camera_mode, CameraMode::FreeRoam) => {
                self.rotate = !self.rotate;
            }
            // Alt modulates the cursor grab while in freecam. Toggle mode
            // (default, FPS sticky): press flips the grab, release is
            // ignored. Hold mode (MMO-style): cursor released while Alt is
            // held, re-grabbed when Alt is released. Both are routed
            // through `Freecam::on_alt`, which inspects its `cursor_mode`
            // field. We forward press AND release so Hold mode sees both
            // edges; outside freecam the preset short-circuits to no-op.
            KeyCode::AltLeft | KeyCode::AltRight
                if matches!(self.camera_mode, CameraMode::FreeRoam) =>
            {
                self.freecam.on_alt(pressed);
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

    pub(crate) fn title(&self, _fps: f32) -> std::borrow::Cow<'static, str> {
        // Window title is now decorative, all live state is in the
        // overlay. Keep the title static so OS task switchers show
        // a stable label.
        std::borrow::Cow::Borrowed("polytope playground")
    }
}

// ---------------------------------------------------------------------------
// App wrapper: Demo + Console<Demo>
// ---------------------------------------------------------------------------
//
// Why a wrapper rather than `console` as a field of `Demo`: `Console::ui`
// takes `(&mut self, &mut Ctx)`. If both `console` and the rest of the
// state lived inside one struct, that call would require simultaneously
// borrowing `&mut self.console` and `&mut self`, which the borrow checker
// rejects. Splitting Demo + Console into a wrapper that owns both gives
// each its own field path, so the dispatch reads as a clean two-field
// borrow.

struct RotatePolytopesApp {
    demo: Demo,
    console: Console<Demo>,
    /// Cached `egui::Context::wants_keyboard_input()` from the
    /// previous frame's UI pass. `App::on_event` runs BEFORE
    /// `App::ui` each frame, so we can't read the current state
    /// during event dispatch; the cached value is one frame stale
    /// but reliably reflects whether an egui widget (the console
    /// input, a typed-formula bar, etc.) was holding keyboard
    /// focus when last drawn. Used to gate routing demo hotkeys
    /// like Space / R / arrows: when egui wants the keyboard,
    /// the demo's hotkeys must NOT fire on top.
    last_egui_keyboard: bool,
    /// Capture parameters panel (output dir, format, fps, scale, start/stop).
    /// Toggled via the `capture panel` console command or the F11 default bind.
    capture_panel: rye_app::capture::CapturePanel,
    /// F3-toggle live perf overlay: FPS, frame-time, sparkline. Reads from
    /// `rye_time::frame_trace`, so it surfaces the same numbers `trace summary`
    /// dumps but continuously. Cheap when hidden (just a key-press check).
    perf: rye_app::trace::PerfOverlay,
}

/// Lower bound on a VISIBLE section-layer fill alpha. Mirrors the old `surface
/// alpha` floor: below this the cap is so faint it reads as off, so the grammar
/// rejects it and steers the user to `0` (the explicit off state) instead, the
/// same open-lower-bound discipline `surface scale` uses.
const SECTION_ALPHA_MIN_VISIBLE: f32 = 0.05;

/// Shared handler for `section cross-alpha` / `section cap-alpha`: query (bare),
/// or set the layer's `surface_alpha`. `0` is the explicit off state (no fill
/// submitted); a value in `[SECTION_ALPHA_MIN_VISIBLE, 1.0]`
/// sets a visible fill. `layer_name` ("cross" / "cap") is only for the report
/// line. Takes `&mut SectionLayer` (not `&mut Demo`) so the two registrations
/// share one body and the handler stays unit-testable without a GPU-backed
/// `Demo`.
fn run_section_alpha(
    layer_name: &str,
    layer: &mut state::SectionLayer,
    args: &[&str],
    out: &mut rye_egui::console::ConsoleWriter,
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
            // `0` is the off state; any other value must be a visible alpha in
            // `[SECTION_ALPHA_MIN_VISIBLE, 1.0]`. A value in `(0, MIN)` is too
            // faint to read, so reject it rather than silently rounding.
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

impl RotatePolytopesApp {
    fn build_console() -> Console<Demo> {
        let mut c = Console::<Demo>::new();
        c.register(rye_egui::cmd(
            "reset",
            "full reset (R)",
            |_args, demo: &mut Demo, _out| {
                demo.reset();
                Ok(())
            },
        ));
        c.register(rye_egui::cmd(
            "spin",
            "toggle continuous rotation (Space / T)",
            |_args, demo: &mut Demo, _out| {
                demo.rotate = !demo.rotate;
                Ok(())
            },
        ));
        c.register(rye_egui::cmd(
            "controls",
            "toggle the bottom controls overlay (H)",
            |_args, demo: &mut Demo, _out| {
                demo.show_controls = !demo.show_controls;
                Ok(())
            },
        ));
        c.register(rye_egui::cmd(
            "formula",
            "toggle the top-right formula popup",
            |_args, demo: &mut Demo, _out| {
                demo.show_formula = !demo.show_formula;
                Ok(())
            },
        ));
        // Cross-section + parent-wireframe overlay. Tab-completion is context-aware
        // via [`SubcommandSet`]: each subcommand's value slot lists only that
        // subcommand's choices. Bare invocations flip:
        //   `wireframe`                 -> flips main on/off
        //   `wireframe nearest-active`  -> flips the alpha gradient toggle
        //   `wireframe color <mode>`    -> sets the color mode
        c.register(
            rye_egui::subcommands::<Demo>("wireframe", "wireframe + cross-section overlay")
                .on_bare(|d| {
                    d.wireframe_enabled = !d.wireframe_enabled;
                    Ok(())
                })
                .toggle(
                    "nearest-active",
                    "per-edge alpha gradient by cell-crossing strength (bare flips)",
                    |d, v| {
                        d.wireframe_nearest_active = v.unwrap_or(!d.wireframe_nearest_active);
                        Ok(())
                    },
                )
                .custom(
                    "width",
                    "parent-wireframe edge thickness in pixels (default 1.8)",
                    &[&[]],
                    &[],
                    |d, args, out| {
                        match args.first().copied() {
                            None => {
                                out.line(format!(
                                    "wireframe width: {:.2} px",
                                    d.wireframe_width_px
                                ));
                            }
                            Some(s) => match s.parse::<f32>() {
                                Ok(w) if w > 0.0 && w <= 16.0 => {
                                    d.wireframe_width_px = w;
                                    out.line(format!(
                                        "wireframe width: set to {w:.2} px"
                                    ));
                                }
                                _ => {
                                    out.line(format!(
                                        "wireframe width: invalid `{s}` (need a float in (0, 16])"
                                    ));
                                }
                            },
                        }
                        Ok(())
                    },
                )
                .custom(
                    "alpha",
                    "uniform edge alpha when nearest-active is off (default 1.0)",
                    &[&[]],
                    &[],
                    |d, args, out| {
                        match args.first().copied() {
                            None => {
                                out.line(format!(
                                    "wireframe alpha: {:.3} ({})",
                                    d.wireframe_alpha,
                                    if d.wireframe_nearest_active {
                                        "overridden by nearest-active gradient; toggle off to apply"
                                    } else {
                                        "active"
                                    }
                                ));
                            }
                            Some(s) => match s.parse::<f32>() {
                                Ok(a) if a > 0.0 && a <= 1.0 => {
                                    d.wireframe_alpha = a;
                                    out.line(format!(
                                        "wireframe alpha: set to {a:.3}"
                                    ));
                                }
                                _ => {
                                    out.line(format!(
                                        "wireframe alpha: invalid `{s}` (need a float in (0, 1])"
                                    ));
                                }
                            },
                        }
                        Ok(())
                    },
                )
                .choice(
                    "color",
                    "parent-edge color mode (bare cycles): vertex-gradient|unique-edge|w-depth|active",
                    &["vertex-gradient", "unique-edge", "w-depth", "active"],
                    |d, name| {
                        d.wireframe_color_mode = match name {
                            Some(n) => WireframeColorMode::from_token(n).ok_or_else(|| {
                                anyhow!(
                                    "unknown color mode `{n}` (try vertex-gradient|unique-edge|w-depth|active)"
                                )
                            })?,
                            None => {
                                // Cycle through the canonical order so bare
                                // `wireframe color` visits every mode in turn.
                                let all = WireframeColorMode::ALL;
                                let i = all
                                    .iter()
                                    .position(|m| *m == d.wireframe_color_mode)
                                    .unwrap_or(0);
                                all[(i + 1) % all.len()]
                            }
                        };
                        Ok(())
                    },
                )
                .custom(
                    "perspective",
                    "wireframe 4D->R³ projection (bare cycles): shadow | w-pinhole | stereographic | hyperslice",
                    &[&["shadow", "w-pinhole", "stereographic", "hyperslice"]],
                    &[],
                    |d, args, out| {
                        // Schlegel is intentionally not offered here (see
                        // `WireframeProjection::Schlegel` docs): it wants its own
                        // demo, not a wireframe-overlay mode. `from_token` rejects
                        // "schlegel" and `ALL` omits it, so neither path below can
                        // produce it.
                        let next = match args.first().copied() {
                            // Bare: cycle through ALL in variant order.
                            None => {
                                let all = WireframeProjection::ALL;
                                let i = all
                                    .iter()
                                    .position(|p| p.same_variant(d.wireframe_projection))
                                    .unwrap_or(0);
                                all[(i + 1) % all.len()]
                            }
                            Some(token) => WireframeProjection::from_token(token).ok_or_else(|| {
                                anyhow!(
                                    "unknown projection `{token}` (try shadow|w-pinhole|stereographic|hyperslice)"
                                )
                            })?,
                        };
                        d.wireframe_projection = next;
                        state::apply_projection_selection_defaults(
                            d.wireframe_projection,
                            &mut d.wireframe_enabled,
                        );
                        // No-op (clears any cached Schlegel params) for these modes;
                        // kept so re-wiring Schlegel later needs no console change.
                        d.resolve_schlegel_cache();
                        out.line(format!(
                            "wireframe perspective: {}",
                            d.wireframe_projection.label().to_lowercase()
                        ));
                        Ok(())
                    },
                )
                .custom(
                    "pole",
                    "stereographic projection pole (bare reports; sub: reset | +w | <x y z w>)",
                    &[&["reset", "+w"]],
                    &[],
                    |d, args, out| {
                        match args.first().copied() {
                            None => {
                                let p = d.stereographic_pole;
                                out.line(format!(
                                    "stereographic pole: ({:.3}, {:.3}, {:.3}, {:.3})",
                                    p.x, p.y, p.z, p.w
                                ));
                            }
                            // The default is a cell-center direction (off every
                            // 16-cell vertex; see STEREOGRAPHIC_DEFAULT_POLE).
                            Some("reset") | Some("default") => {
                                d.stereographic_pole = state::STEREOGRAPHIC_DEFAULT_POLE;
                                out.line("stereographic pole: reset to the cell-center default");
                            }
                            // The textbook `(x, y, z) / (1 - w)` pole, the old
                            // default; offered as a named shortcut so a user can
                            // recover the classic look without typing coordinates.
                            Some("+w") => {
                                d.stereographic_pole = Vec4::W;
                                out.line("stereographic pole: set to +w (textbook map)");
                            }
                            // Explicit pole: four floats, normalized onto S³ (the
                            // map only uses the direction). Reject a near-zero
                            // vector, which has no well-defined direction.
                            Some(_) => {
                                let coords: Result<Vec<f32>> = args
                                    .iter()
                                    .map(|t| {
                                        t.parse::<f32>()
                                            .map_err(|e| anyhow!("invalid pole component `{t}`: {e}"))
                                    })
                                    .collect();
                                let coords = coords?;
                                if coords.len() != 4 {
                                    return Err(anyhow!(
                                        "pole needs 4 components `<x y z w>`, got {}",
                                        coords.len()
                                    ));
                                }
                                let raw = Vec4::new(coords[0], coords[1], coords[2], coords[3]);
                                if raw.length() < MIN_EDGE_RADIUS {
                                    return Err(anyhow!(
                                        "pole vector is too close to zero to have a direction"
                                    ));
                                }
                                let pole = raw.normalize();
                                d.stereographic_pole = pole;
                                out.line(format!(
                                    "stereographic pole: set to ({:.3}, {:.3}, {:.3}, {:.3})",
                                    pole.x, pole.y, pole.z, pole.w
                                ));
                            }
                        }
                        Ok(())
                    },
                )
                .custom(
                    "hyperslice",
                    "cull parent edges to a w-slab around the slice (bare flips; sub: on|off|thickness <N>)",
                    &[&["on", "off", "thickness"]],
                    &[],
                    |d, args, out| {
                        match args.first().copied() {
                            None => {
                                d.wireframe_hyperslice = !d.wireframe_hyperslice;
                                out.line(format!(
                                    "wireframe hyperslice: {} (slab full-width {:.3})",
                                    if d.wireframe_hyperslice { "on" } else { "off" },
                                    d.wireframe_hyperslice_thickness
                                ));
                            }
                            Some("on") => {
                                d.wireframe_hyperslice = true;
                                out.line(format!(
                                    "wireframe hyperslice: on (slab full-width {:.3})",
                                    d.wireframe_hyperslice_thickness
                                ));
                            }
                            Some("off") => {
                                d.wireframe_hyperslice = false;
                                out.line("wireframe hyperslice: off (full edge graph)");
                            }
                            Some("thickness") => match args.get(1).copied() {
                                None => out.line(format!(
                                    "wireframe hyperslice thickness: {:.3}",
                                    d.wireframe_hyperslice_thickness
                                )),
                                Some(token) => {
                                    let t: f32 = token.parse().map_err(|e| {
                                        anyhow!("invalid thickness `{token}`: {e}")
                                    })?;
                                    // Lower bound is the predicate's own floor (a razor
                                    // band that still admits straddling edges); upper bound
                                    // is the full slider span `2 * W_RANGE`, at which the
                                    // slab covers every reachable w and the filter is a
                                    // no-op (equivalent to "off").
                                    let max = 2.0 * consts::W_RANGE;
                                    if !(HYPERSLICE_MIN_THICKNESS..=max).contains(&t) {
                                        return Err(anyhow!(
                                            "hyperslice thickness {t} out of range; expected {HYPERSLICE_MIN_THICKNESS}..={max}"
                                        ));
                                    }
                                    d.wireframe_hyperslice_thickness = t;
                                    out.line(format!(
                                        "wireframe hyperslice thickness: set to {t:.3}"
                                    ));
                                }
                            },
                            Some(other) => {
                                return Err(anyhow!(
                                    "unknown hyperslice subcommand `{other}` (try on|off|thickness)"
                                ));
                            }
                        }
                        Ok(())
                    },
                )
                .custom(
                    "points",
                    "vertex + cell-center sprite overlay (bare flips; sub: vertices|cell-centers|size <N>)",
                    &[&["vertices", "cell-centers", "size"]],
                    &[],
                    |d, args, out| {
                        match args.first().copied() {
                            None => {
                                d.points_enabled = !d.points_enabled;
                                out.line(format!(
                                    "points: {}",
                                    if d.points_enabled { "on" } else { "off" }
                                ));
                            }
                            Some("vertices") => {
                                d.points_show_vertices = !d.points_show_vertices;
                                out.line(format!(
                                    "points vertices: {}",
                                    if d.points_show_vertices { "on" } else { "off" }
                                ));
                            }
                            Some("cell-centers") => {
                                d.points_show_cell_centers = !d.points_show_cell_centers;
                                out.line(format!(
                                    "points cell-centers: {}",
                                    if d.points_show_cell_centers { "on" } else { "off" }
                                ));
                            }
                            Some("size") => match args.get(1) {
                                None => out.line(format!(
                                    "points size: {:.1} px",
                                    d.points_size_px
                                )),
                                Some(token) => {
                                    let px: f32 = token.parse().map_err(|e| {
                                        anyhow!("invalid pixel value `{token}`: {e}")
                                    })?;
                                    if !(1.0..=64.0).contains(&px) {
                                        return Err(anyhow!(
                                            "points size {px} out of range; expected 1..=64"
                                        ));
                                    }
                                    d.points_size_px = px;
                                    out.line(format!("points size: set to {px:.1} px"));
                                }
                            },
                            Some(other) => {
                                return Err(anyhow!(
                                    "unknown points subcommand `{other}` (try vertices|cell-centers|size)"
                                ));
                            }
                        }
                        Ok(())
                    },
                ),
        );

        // Polychoral surface renderer: raster (default) / SDF / off. Bare `surface` is
        // shorthand for "off" so the user can hide cap fills quickly when inspecting the
        // wireframe and cross-section perimeter on their own. Explicit `surface raster`
        // and `surface sdf` set those modes; `surface off` is the same as bare.
        // `surface scale <N>` rescales every polychoron in the row by multiplying the
        // canonical `BODY_SIZE` (see [`Demo::effective_body_size`]); default 1.0.
        c.register(
            rye_egui::cmd(
                "surface",
                "polychoral surface mode: raster | sdf | off (bare = off); `scale <N>` to resize (per-layer cap alpha lives under `section`)",
                |args, demo: &mut Demo, out| {
                    if matches!(args.first().copied(), Some("scale")) {
                        match args.get(1).copied() {
                            None => {
                                out.line(format!(
                                    "surface scale: {:.3} (multiplies BODY_SIZE)",
                                    demo.surface_scale
                                ));
                            }
                            Some(token) => {
                                let parsed: f32 = token.parse().map_err(|e| {
                                    anyhow!("invalid scale `{token}`: {e}")
                                })?;
                                if !(0.05..=10.0).contains(&parsed) {
                                    return Err(anyhow!(
                                        "surface scale {parsed} out of range; expected 0.05..=10.0"
                                    ));
                                }
                                demo.surface_scale = parsed;
                                // Rebuild SDF body uniforms so the kernel sees the new
                                // radius immediately; raster paths read effective_body_size()
                                // every frame and don't need a rebuild.
                                demo.rebuild_bodies();
                                // Clamp the current w-slice into the new scaled range so
                                // a shrink doesn't leave the slider off the visible body.
                                let w_range = demo.effective_w_range();
                                demo.w_slice = demo.w_slice.clamp(-w_range, w_range);
                                out.line(format!(
                                    "surface scale: set to {parsed:.3}"
                                ));
                            }
                        }
                        return Ok(());
                    }
                    let next = match args.first().copied() {
                        Some(token) => SurfaceMode::from_token(token).ok_or_else(|| {
                            anyhow!("unknown arg `{token}` (try raster|sdf|off|scale; cap alpha lives under `section`)")
                        })?,
                        None => SurfaceMode::Off,
                    };
                    if next == SurfaceMode::Sdf && demo.sdf_blocked_by_heavy_polychora() {
                        return Err(anyhow!(
                            "surface sdf disabled while 120-cell or 600-cell is in the row \
                             (the SDF kernel crashes the browser tab on those); remove the \
                             heavy polychora first, or use `surface raster`"
                        ));
                    }
                    if next != demo.surface_mode {
                        demo.surface_mode = next;
                        // Re-emit the SDF body list: switching INTO Sdf mode makes the
                        // polychora live in the kernel, switching OUT marks them inert.
                        demo.rebuild_bodies();
                    }
                    Ok(())
                },
            )
            .with_args(&[&["raster", "sdf", "off", "scale"]])
            .with_long_help(
                "Selects how the six regular convex 4-polytopes (5-cell, tesseract, 16-cell,\n\
                 24-cell, 120-cell, 600-cell) are rendered, plus a runtime scale knob.\n\
                 \n\
                 subcommands:\n  \
                 raster      Rasterized cross-section cell-caps (the default). Face-normal\n                             Lambert lit, per-body solid color. Much faster for the\n                             120-cell + 600-cell and exact (no SDF approximation).\n  \
                 sdf         SDF raymarch. The historical pre-rasterizer path; smoother\n                             shading but the 120-cell and 600-cell carry a face-plane\n                             approximation BUG. Kept for visual comparison.\n  \
                 off         No surface rendered. Wireframe overlay + cross-section\n                             perimeter stay visible if enabled; the cap interiors are\n                             blank. Useful for inspecting the wireframe on its own.\n  \
                 scale <N>   Multiply the canonical body radius by N (default 1.0; range\n                             0.05..=10.0). Affects SDF kernel, raster cross-section caps,\n                             wireframe overlay, perimeter, and points sprites uniformly.\n\
                 \n\
                 Bare `surface` (no argument) is shorthand for `surface off`.\n\
                 \n\
                 The rasterized cross-section splits into two overlaid layers with\n\
                 independent perimeter + fill alpha: see the `section` command (the honest\n\
                 drop-w cross-section and the projection-following cap).\n\
                 \n\
                 Smooth-surface shapes (Clifford torus, duocylinder, spherinder, 3-sphere)\n\
                 ignore the mode and always render via the SDF; they have no rasterizer\n\
                 path. Surface scale still applies to their SDF body radius.",
            ),
        );

        // Section layers: the rasterized cross-section is two overlaid layers in
        // one viewport, each with its own perimeter outline + fill alpha.
        //   - `cross`: the honest drop-w slice (NEVER reprojected; the geometry the
        //     SDF raymarch shows). On by default so selecting Schlegel /
        //     stereographic never silently distorts the slice.
        //   - `cap`: the same slice reprojected through the active wireframe
        //     projection, so it can sit on a Schlegel / stereographic wireframe.
        //     Off by default.
        // Alpha `0` is the layer's off state; `(0, 1]` sets a visible fill (below 1
        // composites through the depth-write-disabled pipeline). Side-by-side /
        // multi-viewport comparison is deferred to the multi-viewport milestone.
        c.register(
            rye_egui::subcommands::<Demo>(
                "section",
                "rasterized cross-section layers: cross (honest drop-w) + cap (projection-following), each with perimeter + alpha",
            )
            .toggle(
                "cross-perimeter",
                "honest drop-w cross-section perimeter outline (bare flips)",
                |d, v| {
                    d.cross_section.perimeter = v.unwrap_or(!d.cross_section.perimeter);
                    Ok(())
                },
            )
            .toggle(
                "cap-perimeter",
                "projected-cap perimeter outline (bare flips)",
                |d, v| {
                    d.projected_cap.perimeter = v.unwrap_or(!d.projected_cap.perimeter);
                    Ok(())
                },
            )
            .custom(
                "cross-alpha",
                "honest cross-section fill alpha (0 = off; range (0, 1])",
                &[&[]],
                &[],
                |d, args, out| run_section_alpha("cross", &mut d.cross_section, args, out),
            )
            .custom(
                "cap-alpha",
                "projected-cap fill alpha (0 = off; range (0, 1])",
                &[&[]],
                &[],
                |d, args, out| run_section_alpha("cap", &mut d.projected_cap, args, out),
            ),
        );

        // Framework-provided capture: `capture png [pre|post|both] [dir]`,
        // `capture frames [pre|post|both] [dir]`, `capture stop`. Bound to F12 (one-shot)
        // and F9 (sequence start; use `capture stop` to end). Requests push to a global
        // queue; the runner drains and processes them at the render-loop's two taps.
        rye_app::capture::register_commands(&mut c);
        rye_app::capture::bind_default_hotkeys(&mut c);

        // Framework-provided log mirror: `log on|off|toggle` toggles whether
        // `tracing::*` events show up in the console scrollback.
        rye_app::log::register_command(&mut c);

        // Framework-provided frame-timing surface: `trace [summary|last|clear|cap N]`.
        // The runner is already recording per-section scopes on every redraw; this
        // command lets the user read them. Surfaces the slowest hot-path sections,
        // which is the data the pipeline-warming + wireframe-cache decisions read
        // from.
        rye_app::trace::register_command(&mut c);
        rye_app::fps::register_command(&mut c);
        rye_app::vsync::register_command(&mut c);
        rye_app::version::register_command(
            &mut c,
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION"),
            env!("BUILD_HASH"),
            env!("BUILD_DIRTY"),
        );

        // Demo-side camera mode toggle. Bare `camera` cycles between Orbit
        // (the default scroll-zoom/drag camera) and FreeRoam (WASD + mouse-
        // look). Explicit `camera orbit` resets the orbit controller to its
        // default distance + pitch so the camera returns to a known framing
        // around the world origin; `camera freecam` seeds the free-roam
        // position from the camera's current location.
        //
        // Freecam tuning subcommands (do NOT change mode):
        //   `camera freecam speed=<N>`        WASD/Space/Shift units/sec.
        //   `camera freecam speed`            Print the current speed.
        //   `camera freecam cursor_mode <m>`  `toggle` (default, FPS) or `hold` (MMO).
        //   `camera freecam cursor_mode`      Print the current mode.
        c.register(
            rye_egui::cmd::<Demo, _>(
                "camera",
                "camera mode: orbit | freecam; bare cycles. `camera freecam speed=<N>` / `cursor_mode hold|toggle` tune the preset",
                |args, demo, out| {
                    // Freecam-tuning forms have a second positional token.
                    // `speed=<N>` is parsed as one token (matches the user's
                    // `speed=<N>` spec); `speed` alone queries; `cursor_mode
                    // <m>` is two tokens.
                    if matches!(args.first().copied(), Some("freecam")) && args.len() >= 2 {
                        let second = args[1];
                        // `speed=<N>` and `speed <N>` and bare `speed`.
                        if let Some(value) = second.strip_prefix("speed=") {
                            let parsed: f32 = value
                                .parse()
                                .map_err(|e| anyhow!("invalid speed `{value}`: {e}"))?;
                            if !(0.1..=200.0).contains(&parsed) {
                                return Err(anyhow!(
                                    "camera freecam speed {parsed} out of range; expected 0.1..=200.0"
                                ));
                            }
                            demo.freecam.speed = parsed;
                            out.line(format!("camera freecam speed: set to {parsed:.2} u/sec"));
                            return Ok(());
                        }
                        if second == "speed" {
                            if let Some(value) = args.get(2) {
                                let parsed: f32 = value.parse().map_err(|e| {
                                    anyhow!("invalid speed `{value}`: {e}")
                                })?;
                                if !(0.1..=200.0).contains(&parsed) {
                                    return Err(anyhow!(
                                        "camera freecam speed {parsed} out of range; expected 0.1..=200.0"
                                    ));
                                }
                                demo.freecam.speed = parsed;
                                out.line(format!(
                                    "camera freecam speed: set to {parsed:.2} u/sec"
                                ));
                            } else {
                                out.line(format!(
                                    "camera freecam speed: {:.2} u/sec",
                                    demo.freecam.speed
                                ));
                            }
                            return Ok(());
                        }
                        if second == "cursor_mode" {
                            match args.get(2).copied() {
                                None => {
                                    out.line(format!(
                                        "camera freecam cursor_mode: {}",
                                        demo.freecam.cursor_mode().token()
                                    ));
                                }
                                Some(token) => {
                                    let mode = CursorMode::from_token(token).ok_or_else(|| {
                                        anyhow!(
                                            "unknown cursor_mode `{token}` (try hold|toggle)"
                                        )
                                    })?;
                                    demo.freecam.set_cursor_mode(mode);
                                    out.line(format!(
                                        "camera freecam cursor_mode: set to {}",
                                        mode.token()
                                    ));
                                }
                            }
                            return Ok(());
                        }
                        // Unknown second token under `camera freecam`: fall
                        // through to the mode-switch path which will yell
                        // about it.
                    }

                    let next = match args.first().copied() {
                        None => match demo.camera_mode {
                            CameraMode::Orbit => CameraMode::FreeRoam,
                            CameraMode::FreeRoam => CameraMode::Orbit,
                        },
                        Some("orbit") => CameraMode::Orbit,
                        Some("freecam") => CameraMode::FreeRoam,
                        Some(other) => {
                            out.line(format!(
                                "camera: unknown mode `{other}` (try orbit|freecam)"
                            ));
                            return Ok(());
                        }
                    };
                    demo.camera_mode = next;
                    match next {
                        CameraMode::Orbit => {
                            // Reset orbit so the camera returns to a known
                            // framing around (0, 0, 0) regardless of where
                            // freecam left it. Freecam's `set_active(false)`
                            // releases the cursor grab.
                            demo.orbit = OrbitController::default();
                            demo.orbit.set_orbit(8.0, -0.25);
                            demo.freecam.set_active(false, demo.camera.position);
                            out.line("camera: orbit (reset to world origin)");
                        }
                        CameraMode::FreeRoam => {
                            // Preset grabs cursor + seeds position from the
                            // camera's current pose so the toggle is
                            // continuous, not a teleport.
                            demo.freecam.set_active(true, demo.camera.position);
                            out.line(
                                "camera: freecam (WASD + Space/Shift; mouse-look; Alt to free cursor)",
                            );
                        }
                    }
                    Ok(())
                },
            )
            .with_args(&[
                &["orbit", "freecam"],
                &["speed=", "cursor_mode"],
                &["hold", "toggle"],
            ]),
        );

        // Floor toggle for the y=0 hyperplane ground. On by default. The
        // SDF kernel reads `u.params[0]` (set in `Demo::update`); when 0.0
        // the wrapper around `rye_scene_sdf` (injected into the shader at
        // setup time) short-circuits to a huge distance, so the marcher
        // never converges on the floor and the checkerboard never paints.
        // Bare `floor` flips the flag; `floor on|off` is the explicit form.
        c.register(
            rye_egui::cmd::<Demo, _>(
                "floor",
                "toggle the y=0 hyperplane ground (on | off; bare flips)",
                |args, demo, out| {
                    let next = match args.first().copied() {
                        None => !demo.floor_enabled,
                        Some("on") => true,
                        Some("off") => false,
                        Some(other) => {
                            return Err(anyhow!("floor: unknown arg `{other}` (try on|off)"));
                        }
                    };
                    demo.floor_enabled = next;
                    out.line(format!("floor: {}", if next { "on" } else { "off" }));
                    Ok(())
                },
            )
            .with_args(&[&["on", "off"]]),
        );

        c
    }
}

impl App for RotatePolytopesApp {
    type Space = EuclideanR3;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let demo = Demo::new(ctx)?;
        let console = Self::build_console();
        Ok(Self {
            demo,
            console,
            last_egui_keyboard: false,
            capture_panel: rye_app::capture::CapturePanel::new(),
            perf: rye_app::trace::PerfOverlay::new(),
        })
    }

    fn space(&self) -> &EuclideanR3 {
        self.demo.space()
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        self.demo.update(ctx);
    }

    fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
        self.demo.ui(ctx, frame);
        self.capture_panel.show(ctx);
        // F3-toggle perf overlay (FPS / frame-time / between-frames). Reads
        // its hotkey state from the same egui Context the rest of the UI uses,
        // so the demo doesn't need to forward F3. Cheap when hidden.
        self.perf.show(ctx);
        // Pump any pending tracing events into the console scrollback BEFORE rendering
        // it, so the user sees mirrored log lines this frame instead of next.
        rye_app::log::pump_into(&mut self.console);
        self.console.ui(ctx, &mut self.demo);
        // Stash for next frame's `on_event` to gate hotkey routing.
        // Captured AFTER the console renders so a freshly-focused
        // console input registers true; captured BEFORE end_pass so
        // it reflects the state we want next-frame events to see.
        self.last_egui_keyboard = ctx.wants_keyboard_input();
    }

    fn on_key(
        &mut self,
        code: winit::keyboard::KeyCode,
        state: winit::event::ElementState,
        ctx: &mut FrameCtx<'_>,
    ) {
        // Suppress demo keybinds when egui is actively capturing
        // keyboard input (any TextEdit focused: console, formula
        // bar, etc.) so typing `reset` into the console doesn't
        // also fire the R hotkey, etc. When the user clicks
        // outside the egui widget that had focus, egui releases
        // keyboard focus and the next frame's `on_key` routes
        // hotkeys back to the demo as normal.
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
    // `rye_app::run` handles native + wasm dispatch (worker context vs
    // main-thread launch-on-click vs main-thread auto-launch fallback)
    // based on the page's `data-mode` attribute and the WasmConfig IDs.
    // Default WasmConfig uses our standard layout (`rye-canvas-host` /
    // `rye-launch` / `rye-canvas`); the demo's `index.html` matches.
    rye_app::run::<RotatePolytopesApp>(RunConfig {
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
// `cargo test --example polytope_playground` to run.
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
mod color_tests {
    //! Tests for `compute_cell_strengths` (per-cell w-slice crossing strength).
    use super::*;

    // ---- compute_cell_strengths -----------------------------------------

    /// Slice at the cell's w-midpoint produces strength = 1 (cap is widest there).
    #[test]
    fn cell_strength_at_midpoint_is_one() {
        // Single cell with two vertices at w = -0.5 and w = +0.5. Midpoint w = 0.
        let cells: [&[u32]; 1] = [&[0, 1]];
        let local_vertices = [
            glam::Vec4::new(0.0, 0.0, 0.0, -0.5),
            glam::Vec4::new(0.0, 0.0, 0.0, 0.5),
        ];
        let strengths = compute_cell_strengths(&cells, &local_vertices, 0.0);
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
        let strengths = compute_cell_strengths(&cells, &local_vertices, 5.0);
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
        let strengths = compute_cell_strengths(&cells, &local_vertices, 0.5);
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
        let strengths = compute_cell_strengths(&cells, &local_vertices, 0.25);
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
        let strengths = compute_cell_strengths(&cells, &local_vertices, 0.0);
        assert!(strengths[0].abs() < 1e-5);
    }
}

#[cfg(test)]
mod blended_edge_tests {
    //! Tests for `push_blended_edge`, the wireframe-edge tessellator behind the
    //! `space` command. The S³ slerp math itself is pinned in
    //! `rye_math::spherical_embedded`; these tests pin the demo-side contract:
    //! the flat fast path, the curved sub-segment count, and that a curved edge
    //! is actually longer than its chord (i.e. it bows out).
    use super::*;

    fn flat_drop_w() -> rye_math::Projection<4> {
        rye_math::Projection::Identity
    }

    /// Total length of all segments in the mesh, in world R³.
    fn polyline_length(mesh: &LineMesh<3>) -> f32 {
        mesh.segments
            .iter()
            .map(|(p0, p1)| (Vec3::from_array(*p1) - Vec3::from_array(*p0)).length())
            .sum()
    }

    const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

    /// `blend == 0` emits exactly one chord segment, equal to the projected
    /// endpoints. This is the historical (pre-`space`) wireframe behavior.
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
    /// longer than the straight chord between the same endpoints. Uses two
    /// equal-radius endpoints a quarter circle apart in the xy-plane, so drop-w
    /// preserves the bulge.
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

    /// A half-blend lands between flat and spherical: its polyline length is
    /// strictly between the chord and the full arc. Pins the morph as monotone,
    /// not a step.
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

    /// A representative non-trivial Perspective4D projection (the affine
    /// 4D->R³ map the wireframe selects for the curved/perspective view). Focal
    /// distance is comfortably outside the unit-circumradius polytope so no
    /// vertex straddles the eye plane.
    fn perspective() -> rye_math::Projection<4> {
        rye_math::Projection::Perspective4D {
            focal_distance: 3.0,
        }
    }

    /// `blend == 0` through an affine projection (Perspective4D) emits exactly
    /// one segment, and its two endpoints equal `project_to_world(a)` /
    /// `project_to_world(b)` to the bit. This pins the single-segment fast path
    /// at the top of `push_blended_edge` under a NON-identity affine projection:
    /// the existing `blend_zero_emits_single_chord` only exercised drop-w
    /// (Identity), so the w-dependent perspective scale was untested on the fast
    /// path. Uses a real tesseract edge (endpoints at w = +/- 0.5) so the
    /// perspective divide actually moves the projected points.
    #[test]
    fn blend_zero_is_bit_identical_to_flat_chord() {
        // Two adjacent tesseract vertices sharing the x edge: they differ only
        // in w, so the perspective scale differs per endpoint and the chord is
        // not w-invariant.
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

    /// At any blend in [0, 1] the FIRST emitted point equals `project_to_world(a)`
    /// and the LAST equals `project_to_world(b)`, exactly. The morph bows only the
    /// edge interior; the endpoints are shared by the flat chord and the S³ arc
    /// (the vertices already lie on the body's circumsphere), so the glue must be
    /// bit-exact at every t or the section cap would detach from the wireframe.
    /// Walks a stereographic projection: `blend == 0` takes the flat endpoint
    /// chord path, while `blend > 0` takes the sampled spherical path.
    #[test]
    fn blend_endpoints_exact_at_all_t() {
        let a = Vec4::new(0.5, 0.5, 0.5, 0.5);
        let b = Vec4::new(-0.5, 0.5, 0.5, -0.5);
        // Stereographic from the +w pole: zero blend is one endpoint chord, and
        // blend > 0 takes the sampled path. Both must still glue the endpoints.
        let proj = rye_math::Projection::Stereographic { pole: Vec4::W };
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
    use rye_egui::console::ConsoleWriter;

    /// `run_section_alpha` is the shared handler behind both `section cross-alpha`
    /// and `section cap-alpha`. It must: set a visible alpha in range, accept `0`
    /// as the explicit off state, reject the faint `(0, MIN_VISIBLE)` band and
    /// over-range / unparseable input (no silent clamp), and leave the field
    /// untouched on a bare query. Driving the registered console needs a
    /// GPU-backed `Demo`; exercising the handler directly IS the aliasing
    /// guarantee, since both registrations pass their layer to this one body.
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
mod hyperslice_filter_tests {
    //! Tests for the wireframe Hyperslice cull. The cull is CELL-level: an edge
    //! survives iff some cell containing BOTH its endpoints has its body-local
    //! w-range overlapping the slab `[w_slice - t/2, w_slice + t/2]`. The split
    //! is `cell_w_range` (the cell's w-interval over the rotated, scaled
    //! vertices, shared with `compute_cell_strengths`) and `slab_overlaps` (the
    //! 1D band-overlap predicate). The `slab_overlaps` tests pin the band
    //! semantics (closed boundary, zero/negative-thickness floor, determinism);
    //! the cell-level tests pin the agreement with the active-edge coloring and
    //! that the cull still culls.
    use super::*;

    /// Mirror of the production cull closure `edge_in_slab_cell` in
    /// `render_wireframe_overlay`: keep the edge `(i, j)` iff some cell holding
    /// both endpoints has a slab-overlapping w-range. Tests drive this so the
    /// invariant tracks the exact composition the renderer uses, while the
    /// renderer keeps the closure inline (no extra public surface, no per-frame
    /// allocation).
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

    /// A w-range lying entirely outside the slab does not overlap. With
    /// `w_slice = 0` and a thin slab, a range `[0.8, 0.9]` (well above the slab)
    /// and the symmetric `[-0.9, -0.8]` both return false.
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

    /// The band is CLOSED: a range endpoint sitting exactly on `w_slice +/- t/2`
    /// overlaps, and the result is identical across repeated evaluations (pure
    /// f32 arithmetic, no state). Uses the tesseract's canonical `w = +/- 0.5`
    /// w-range as the exact-boundary case: with `w_slice = 0` and `t = 1.0` the
    /// slab is `[-0.5, +0.5]`, so a range `[-0.5, +0.5]` lands both ends exactly
    /// on the boundary.
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

    /// Thickness 0 is floored to [`HYPERSLICE_MIN_THICKNESS`], so the slab
    /// degrades to a razor band around `w_slice`: only a range that CROSSES the
    /// slice overlaps, and the test neither panics nor produces an infinity. A
    /// range straddling `w_slice = 0` overlaps; one entirely to one side (even
    /// very close) does not.
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

    /// A negative thickness (nonsensical input, but possible from a future
    /// slider bug) is floored the same way as 0, so the predicate stays a valid
    /// razor band rather than an inverted slab that keeps nothing or everything.
    #[test]
    fn slab_overlaps_negative_thickness_floor() {
        assert!(slab_overlaps(-0.3, 0.3, 0.0, -5.0));
        assert!(!slab_overlaps(0.1, 0.3, 0.0, -5.0));
    }

    /// The repro that motivated the cell-level cull (the 16-cell at
    /// `w_slice = -0.182`): an edge whose BOTH endpoints sit on the far side of
    /// the slab is still kept, because the CELL it belongs to is being sliced.
    ///
    /// Minimal model: one cell with vertices spanning `w in [-0.5, +0.5]` so its
    /// w-range strictly straddles `w_slice = -0.182`. The edge under test
    /// (vertices 2,3) has both endpoints at `w = +0.5`, far outside the slab as
    /// an endpoint-pair. The OLD edge-level test on those endpoints would cull
    /// it; the cell-level cull keeps it, matching the active-green coloring,
    /// which also reads the whole cell's w-range.
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

    /// Agreement contract: every edge that the active-edge coloring lights up
    /// (its containing cell has `strength > 0`, i.e. the slice is strictly inside
    /// the cell's w-range) is kept by the cull. The slab band is a SUPERSET of
    /// the strict-interior plane, so `active => kept` holds for any thickness at
    /// or above the floor. Drives the same near/far cell as the repro but checks
    /// every edge of it.
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

        let strengths = compute_cell_strengths(cells, &local_vertices, w_slice);
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

    /// The cull still culls: an edge whose only containing cell has its w-range
    /// entirely outside the slab is dropped. Single cell far above the slice;
    /// its edges are all removed.
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

    /// The extracted `cell_w_range` reproduces the `(w_min, w_max)` implicit in
    /// `compute_cell_strengths`, so the single-source refactor cannot drift: the
    /// strength is `1 - |w_slice - mid| / half_extent` with `mid` and
    /// `half_extent` derived from exactly this range. Checked at the cell's
    /// w-midpoint, where the strength must be exactly 1.0.
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
        let strengths = compute_cell_strengths(cells, &local_vertices, mid);
        assert_eq!(
            strengths[0], 1.0,
            "strength at the cell's w-midpoint is the gradient peak"
        );
    }
}

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
    ///
    ///   1. Rendering the same source closure in two `Area`s with
    ///      different layers does NOT trigger the debug-assert when
    ///      ids are scoped per-ui (`make_persistent_id`).
    ///   2. The IDs actually ARE distinct between the two passes.
    ///
    /// The first part (running this test without panic in debug) is
    /// what catches a regression to globally-stable ids.
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

#[cfg(test)]
mod section_cap_projection_tests {
    //! Tests for the section-cap world transform under each wireframe projection.
    //! The affine modes (Identity/Orthographic/Perspective4D) take the scalar shim
    //! `perspective_scale_at_w` -> `Some(scale)`; the non-affine modes
    //! (Schlegel/Stereographic) return `None`, and `cap_vertex_projected_and_world`
    //! reconstructs the cap vertex's 4D coordinate at `w_slice` and projects it
    //! per-vertex through `EuclideanR4::project_point`, matching the parent
    //! wireframe so the flat cross-section lands on the projected edge graph.
    use super::*;

    /// `perspective_scale_at_w` reports `Some(scale)` exactly for the affine
    /// projections (where a single scalar at the slice's w is exact) and `None`
    /// for the non-affine ones (where no single scalar rescales the cap). This is
    /// the guard the consumers branch on; if a non-affine arm silently grew a
    /// scalar it would render the cross-section as a w-only-scaled ghost.
    #[test]
    fn perspective_scale_returns_none_for_non_affine() {
        // Affine: Identity at any w is unit scale.
        assert_eq!(
            perspective_scale_at_w(0.3, &rye_math::Projection::Identity),
            Some(1.0)
        );
        // Affine: Perspective4D at w_slice is `focal / (focal - w_slice)`.
        let focal = 2.0;
        let w_slice = 0.5;
        let got = perspective_scale_at_w(
            w_slice,
            &rye_math::Projection::Perspective4D {
                focal_distance: focal,
            },
        );
        assert_eq!(got, Some(focal / (focal - w_slice)));
        // Non-affine: both report `None`.
        assert_eq!(
            perspective_scale_at_w(0.0, &rye_math::Projection::Stereographic { pole: Vec4::W }),
            None
        );
        assert_eq!(
            perspective_scale_at_w(0.0, &rye_math::Projection::schlegel(Vec4::W, 0.5, 0.75)),
            None
        );
    }

    /// A cap vertex at `w = w_slice` lands at the same world R³ point whether the
    /// affine scalar shim or a direct per-vertex `EuclideanR4::project_point` with
    /// `Perspective4D` transforms it. This pins the equivalence the shim relies on:
    /// for an affine projection, scaling the dropped-w cap by `focal / (focal -
    /// w_slice)` IS the projection of `(x, y, z, w_slice)`, so the affine fast path
    /// is not an approximation. If they ever diverged, caps and wireframe would
    /// separate under W-depth.
    #[test]
    fn section_cap_matches_wireframe_under_perspective4d() {
        let focal = 2.0;
        let w_slice = 0.4;
        let proj = rye_math::Projection::Perspective4D {
            focal_distance: focal,
        };
        let body_pos = Vec3::new(1.3, -0.7, 0.2);
        let scale = perspective_scale_at_w(w_slice, &proj);
        assert!(scale.is_some(), "Perspective4D must take the affine shim");
        // A handful of off-axis cap vertices, all sharing the slice's w.
        for cap_r3 in [[0.5, 0.0, 0.0], [0.0, 0.3, -0.2], [-0.4, 0.1, 0.6]] {
            // Affine shim path (what the cap rendering uses).
            let via_shim =
                cap_vertex_projected_and_world(cap_r3, w_slice, scale, &proj, body_pos).1;
            // Per-vertex projection of the reconstructed 4D cap vertex (what the
            // wireframe path uses for its vertices).
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

    /// Under Stereographic, the equatorial slice (`w_slice = 0`) sits opposite the
    /// `+w` pole, so no cap vertex hits the projection's pole singularity: every
    /// reconstructed-and-projected cap vertex maps to an all-finite world R³ point.
    /// This pins the per-vertex non-affine path (`section_scale = None`) against
    /// NaN/Inf leaking into the upload buffer at the cross-section.
    ///
    /// Cap vertices are edge-slice intersections, so they live on the polytope's
    /// 1-skeleton at a radius bounded away from the center; the test stays away from
    /// the exact origin, which is not a reachable cap vertex (no convex-polytope edge
    /// passes through the interior center) and which `EuclideanR4::project_point`
    /// cannot normalize onto S³. A near-origin vertex is included to probe the small-
    /// radius end of the real range.
    #[test]
    fn section_cap_per_vertex_finite_under_stereographic() {
        let w_slice = 0.0;
        let proj = rye_math::Projection::Stereographic { pole: Vec4::W };
        let body_pos = Vec3::new(0.5, 0.0, -0.3);
        let scale = perspective_scale_at_w(w_slice, &proj);
        assert_eq!(scale, None, "Stereographic must take the per-vertex path");
        // Cap vertices spread across the equatorial 3-flat, from a small but nonzero
        // radius out to near the unit shell.
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

    /// Edge-line preservation, flat endpoint chords, and section-cap scalar
    /// shortcuts are separate questions. Schlegel preserves straight edges but
    /// still needs per-vertex cap projection. Stereographic does not preserve a
    /// sampled chord interior, but its flat wireframe mode is an endpoint-chord
    /// comparison overlay.
    #[test]
    fn flat_edge_chord_policy_splits_from_cap_scale_policy() {
        let schlegel = rye_math::Projection::schlegel(Vec4::W, 0.5, 0.75);
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

        let stereo = rye_math::Projection::Stereographic { pole: Vec4::W };
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

    /// Zero-blend stereographic is the comparison overlay: project endpoints,
    /// then draw the R3 chord between them. The faithful S3 edge is sampled at
    /// blend one.
    #[test]
    fn stereographic_zero_blend_is_endpoint_chord_overlay() {
        let proj = rye_math::Projection::Stereographic { pole: Vec4::W };
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
        let proj = rye_math::Projection::schlegel(Vec4::W, 0.5, 1.0);
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

    /// Affine projections keep the single-segment fast path: a flat edge under
    /// Perspective4D emits exactly one wireframe segment (no needless subdivision),
    /// and the cap vertex sits on it. Guards the perf-sensitive common case from
    /// accidentally taking the subdivided branch.
    #[test]
    fn affine_wireframe_keeps_single_segment_and_caps_land_on_it() {
        let proj = rye_math::Projection::Perspective4D {
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

    /// The two-layer split's world-transform invariant: the HONEST cross-section
    /// layer maps a cap vertex to the SAME world R³ point under every active
    /// wireframe projection (because [`state::section_layer_projection`] forces it
    /// to drop-w), while the PROJECTED cap layer moves with the active projection.
    /// This is the render-path counterpart to the state-model
    /// `section_layer_projection_honest_ignores_projected_follows` test: it pins
    /// that the projection override actually changes where the cap lands, so a
    /// projection change is provably non-destructive to the honest slice and
    /// provably effective on the projected cap.
    #[test]
    fn honest_section_cap_is_projection_invariant_projected_cap_is_not() {
        let body_pos = Vec3::new(0.7, -0.2, 0.4);
        let w_slice = 0.3;
        // A cap vertex with distinct spatial coords so a non-affine projection
        // genuinely relocates it (a pure-radial point could stay collinear).
        let cap_r3 = [0.4, -0.25, 0.15];
        let actives = [
            rye_math::Projection::Identity,
            rye_math::Projection::Perspective4D {
                focal_distance: 2.0,
            },
            rye_math::Projection::Stereographic { pole: Vec4::W },
            rye_math::Projection::schlegel(Vec4::W, 0.5, 0.9),
        ];

        // Honest layer (drop-w): the world cap is the body-local cap scaled by 1
        // and translated, identical under every active projection.
        let honest_reference = {
            let proj = state::section_layer_projection(true, rye_math::Projection::Identity);
            let scale = perspective_scale_at_w(w_slice, &proj);
            cap_vertex_projected_and_world(cap_r3, w_slice, scale, &proj, body_pos).1
        };
        let mut projected_caps = Vec::new();
        for active in actives {
            // Honest layer is drop-w regardless of `active`.
            let honest_proj = state::section_layer_projection(true, active);
            assert_eq!(
                honest_proj,
                rye_math::Projection::Identity,
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

        // The projected cap must NOT all collapse to the honest drop-w point: at
        // least one non-identity projection relocates it. (Identity's projected
        // cap equals the honest one by construction; the others must differ.)
        let moved = projected_caps
            .iter()
            .any(|c| (0..3).any(|k| (c[k] - honest_reference[k]).abs() > 1e-4));
        assert!(
            moved,
            "projected cap must move under at least one active projection; \
             got {projected_caps:?} all equal to honest {honest_reference:?}"
        );
    }

    /// Distance from `p` to the segment `[s, e]` (clamped to the segment, not the
    /// infinite line), the metric the polyline-tracking tests use to ask "does the
    /// cap sit on this edge?".
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
    // These pin the near-pole clip the stereographic wireframe needs: a vertex
    // landing on (or sweeping through) the projection pole maps to the
    // large-but-finite point the pole-denominator clamp produces, and the
    // wireframe builder drops the over-radius sub-segments rather than drawing
    // them. The clip is a DROP, not a magnitude rescale; the tests below
    // distinguish the two and pin boundedness, finiteness, the no-rescale
    // (segment-count) discriminator, and non-perturbation of off-pole edges.
    //
    // NOT pinned here, and NOT claimed by any test name: flicker-freeness under
    // continuous rotation. A vertex crossing the pole is a genuine projection
    // discontinuity (the pole-perpendicular numerator reverses sign across the
    // crossing); the clip bounds and de-NaNs the artifact and runs the edge out
    // to the view boundary, but the at-pole instant remains discontinuous. That
    // temporal behavior is a visual property and needs human eyes-on (see the
    // wireframe overlay note); a test named "flicker_free" would be a doc lie.

    /// The per-shape, camera-adaptive clip radius. For the 16-cell (the only shape
    /// with vertices on the `+w` pole) the radius stays below the camera distance
    /// at every reasonable zoom (no rubberband), clears the legitimate image of a
    /// unit-circumradius polytope (real geometry never clipped), stays below the
    /// pole-clamp magnitude ceiling (the near-pole blow-up still exceeds it), and
    /// saturates at [`STEREOGRAPHIC_CELL16_RADIUS_MAX`] on zoom-out (the arc never
    /// reaches the under-tessellated steep region). Every other shape is
    /// unclipped (`INFINITY`), since its image is naturally bounded.
    #[test]
    fn stereographic_view_radius_tracks_camera_distance() {
        // The legit image of the worst non-pole vertex (the `+w`-cell corner at
        // w = 0.5, image magnitude sqrt(3) ~ 1.73): the radius must always clear
        // it so real geometry is never clipped. Pinned against a genuine sample.
        let legit = <rye_math::EuclideanR4 as rye_math::RasterizableSpace<4>>::project_point(
            Vec4::new(0.5, 0.5, 0.5, 0.5),
            &rye_math::Projection::Stereographic { pole: Vec4::W },
        )
        .length();
        let clamp_ceiling = (2.0 / rye_math::STEREOGRAPHIC_POLE_EPSILON).sqrt();

        // Across the orbit's zoom range and beyond, the 16-cell radius stays a
        // fixed fraction below the camera distance, clears the figure, and sits
        // under the clamp ceiling. At very close range the figure floor can exceed
        // the distance (the eye is inside the figure); above the floor's reach the
        // strict-below-distance property holds, which is the rubberband fix.
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

        // Zoom-out saturates the 16-cell radius at its cap (no growth into the
        // under-tessellated steep region), keeping far-zoom arcs smooth.
        assert_eq!(
            stereographic_view_radius(Polytope4::Cell16, 40.0),
            STEREOGRAPHIC_CELL16_RADIUS_MAX
        );
        // The test reference is the 16-cell value at an 8-unit camera distance.
        assert!(
            (stereographic_view_radius(Polytope4::Cell16, 8.0) - STEREOGRAPHIC_VIEW_RADIUS).abs()
                < 1e-5
        );

        // Every other shape is unclipped at every distance: the tesseract,
        // 24-cell, etc. have their vertices off the pole, so their image is
        // bounded and we draw the full conformal extent (INFINITY -> no clip).
        for polytope in [Polytope4::Tesseract, Polytope4::Cell24, Polytope4::Cell600] {
            for distance in [2.0_f32, 8.0, 40.0] {
                assert!(
                    stereographic_view_radius(polytope, distance).is_infinite(),
                    "{polytope:?} must be unclipped (no radius limit)"
                );
            }
        }
    }

    /// `stereographic_clip_radius` returns `Some(R)` exactly for Stereographic and
    /// `None` for every other projection: only Stereographic has a genuine
    /// point-at-infinity (a vertex on the pole) in its image, so it is the only
    /// projection whose samples are clip-tested. Pins the gate the edge builders
    /// branch on; a stray `Some` on an affine projection would clip legitimate
    /// geometry, a stray `None` on Stereographic would draw the pole blow-up.
    #[test]
    fn stereographic_clip_radius_only_for_stereographic() {
        assert_eq!(
            stereographic_clip_radius(
                &rye_math::Projection::Stereographic { pole: Vec4::W },
                STEREOGRAPHIC_VIEW_RADIUS
            ),
            Some(STEREOGRAPHIC_VIEW_RADIUS)
        );
        for proj in [
            rye_math::Projection::Identity,
            rye_math::Projection::Orthographic { drop_axis: 3 },
            rye_math::Projection::Perspective4D {
                focal_distance: 2.0,
            },
            rye_math::Projection::schlegel(Vec4::W, 0.5, 0.75),
        ] {
            assert_eq!(
                stereographic_clip_radius(&proj, STEREOGRAPHIC_VIEW_RADIUS),
                None,
                "non-stereographic projection {proj:?} must carry no clip"
            );
        }
    }

    /// Build the parent wireframe edge `a -> b` under the `+w`-pole stereographic
    /// projection with `body_pos = ZERO`, so each emitted endpoint's world coord
    /// equals its body-local projected point. Returns the segment endpoints.
    fn build_stereographic_edge_with_blend(
        a: Vec4,
        b: Vec4,
        blend: f32,
    ) -> Vec<([f32; 3], [f32; 3])> {
        let proj = rye_math::Projection::Stereographic { pole: Vec4::W };
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

    /// Zero-blend stereographic is an endpoint chord overlay. If either endpoint
    /// clips out near the pole, the flat chord drops; the sampled S3 path can
    /// still resume after its clipped samples.
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

    /// Boundedness: every emitted endpoint of a stereographic wireframe edge has
    /// body-local projected magnitude <= `STEREOGRAPHIC_VIEW_RADIUS`, even for an
    /// edge that grazes the pole. The edge runs from 1 degree off the pole (well
    /// inside the clip band, image magnitude ~114) out to the equator; the clip
    /// drops the near-pole samples, so no emitted endpoint carries the blow-up.
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

    /// The clip cuts a straddling sub-segment AT the view boundary and DROPS a
    /// sub-segment whose interior runs deep through the pole; it is neither a
    /// whole-segment drop nor a magnitude rescale. The fixture is an edge whose
    /// great-circle interior passes straight through the `+w` pole (endpoints 30
    /// degrees off the pole on opposite sides), so the deep-pole samples blow up
    /// while both endpoints stay well inside `R`. Three guarantees:
    ///
    /// 1. **Boundary cut, not stop-short.** Some emitted endpoint sits within a
    ///    hair of `R`: the straddling sub-segment is cut to the clip sphere, so
    ///    the arc reaches the boundary instead of stopping at the last in-radius
    ///    tessellation sample (the old whole-segment drop left the tip at a random
    ///    interior sample, the source of the near-pole "bounce").
    /// 2. **Deep-pole drop, not rescale.** The polyline still has a GAP (fewer
    ///    than `SPACE_TESSELLATION_SAMPLES` segments): the both-outside samples
    ///    straddling the pole are dropped, never bridged. A radius rescale-clamp
    ///    would instead keep every sub-segment (pinned onto the sphere of radius
    ///    `R`), preserving the 180-degree direction flip across the pole as a
    ///    spurious segment sweeping the view; the gap proves we drop them.
    /// 3. **Bounded.** Every emitted endpoint stays within `R`.
    #[test]
    fn stereographic_clip_cuts_to_boundary_and_drops_deep_pole() {
        let r = STEREOGRAPHIC_VIEW_RADIUS;
        // Endpoints 30 degrees off the pole on opposite sides of the w-x plane;
        // their connecting great circle passes through the +w pole at its
        // midpoint. The endpoints' image magnitude is cot(15 deg) ~ 3.73 < R, so
        // they are kept, while the midpoint samples blow up past R.
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

    /// The regression test for the 16-cell near-pole artifacts under `xw`
    /// rotation: once a vertex is near the `+w` pole, the visible arc tip must sit
    /// at the clip boundary and STAY there as the vertex sweeps closer, with no
    /// popping (sample-granularity drop) and no diving toward the center (the
    /// conformal map's denominator-clamp deflation). The fixture is the genuine
    /// 16-cell edge `+e_w -> +e_y`: `+e_y` is FIXED by an `xw` rotation while
    /// `+e_w` rotates toward the pole, so the arc tip is purely the near-pole end.
    ///
    /// The sweep walks `phi` from 3 deg (just past the view-radius crossing,
    /// `cot(phi/2) = R` at `phi ~ 3.24 deg`, so `+e_w` is already beyond `R`) down
    /// to 0.05 deg, deep inside the pole-denominator clamp band (`phi < ~0.8 deg`).
    /// Across this whole regime the tip must hold the boundary `R` to within a
    /// unit. Two prior defects each break this: the whole-segment drop left the tip
    /// at the last in-radius sample (well below `R`), and the clamp-band deflation
    /// dove the tip from `R` toward the origin below `phi ~ 0.2 deg` (a >30-unit
    /// collapse). Both manifest as a tip far below `R` somewhere in the sweep.
    #[test]
    fn stereographic_clip_arc_tip_holds_boundary_near_pole() {
        let r = STEREOGRAPHIC_VIEW_RADIUS;
        // `xw` rotation of the +e_w -- +e_y edge by `phi`: e_w -> (-sin phi, 0, 0,
        // cos phi) sweeps toward the pole; e_y = (0, 1, 0, 0) is fixed.
        let tip_extent = |phi_deg: f32| -> f32 {
            let phi = phi_deg.to_radians();
            let a = Vec4::new(-phi.sin(), 0.0, 0.0, phi.cos());
            let b = Vec4::new(0.0, 1.0, 0.0, 0.0);
            build_spherical_stereographic_edge(a, b)
                .iter()
                .flat_map(|(s, e)| [Vec3::from_array(*s).length(), Vec3::from_array(*e).length()])
                .fold(0.0_f32, f32::max)
        };
        // Geometric sweep so steps cluster near the pole, where the deflation dive
        // strikes; every sample is in the beyond-R regime.
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

    /// Finiteness: an edge with one endpoint exactly on the pole produces only
    /// finite endpoints, never NaN/Inf, and bounded by the clip radius. The pole
    /// itself maps to the origin (the perpendicular numerator is zero there); its
    /// near-pole neighbors blow up and are dropped. Extends the rasterizer's
    /// finite-drop guard from "finite" to "finite AND bounded" for the
    /// stereographic case.
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

    /// Non-perturbation off the pole: a spherical edge well clear of the pole
    /// keeps every sub-segment (nothing dropped) and each emitted endpoint equals
    /// the raw `project_to_world` of the corresponding great-circle sample
    /// bit-for-bit. The clip is a pure post-filter on already-projected samples;
    /// it must not move a retained sample. This guards the conformal interior:
    /// the clip changes nothing where the projection is well-behaved.
    #[test]
    fn stereographic_clip_does_not_perturb_off_pole_edge() {
        let proj = rye_math::Projection::Stereographic { pole: Vec4::W };
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
        <rye_math::SphericalS3Embedded as rye_math::RasterizableSpace<4>>::tessellate_segment(
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

    /// The clip adds no per-edge allocation. Building a pole-grazing blended edge
    /// (which exercises the clip drop) then re-building it leaves `slerp_scratch`
    /// at the capacity it reached on the first edge: the clip is a streaming
    /// `continue`, not a `filter().collect()`, so the reused great-circle buffer
    /// never grows on account of dropped samples. Mirrors the rasterizer's
    /// `upload_drops_non_finite_without_reallocating`.
    #[test]
    fn stereographic_clip_reuses_scratch_without_realloc() {
        let proj = rye_math::Projection::Stereographic { pole: Vec4::W };
        let white = [1.0, 1.0, 1.0, 1.0];
        let mut scratch = Vec::new();
        let mut mesh = LineMesh::<3>::default();
        // Blended (blend > 0) edge so the slerp buffer is actually populated; one
        // endpoint near the pole so the clip drops interior samples.
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
        // The slerp buffer holds `samples + 1` points, so its capacity must be
        // at least that after the first build.
        assert!(cap_after_first > SPACE_TESSELLATION_SAMPLES);
        // Re-run: the buffer is cleared and refilled to the same length, so no
        // growth despite the clip dropping segments.
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
    // These pin GAP closures the perimeter outline already had: the
    // projected-cap FILL (`retain_in_radius_triangles`, triangle-granularity)
    // and the points overlay (`sample_in_radius` per vertex / cell-center).
    // Both reuse the SAME predicate the wireframe edges and the cap perimeter
    // use, so fill, outline, edges, and points all cull on one ~3.24-degree
    // drop cone. As with the edge clip, none of these claim flicker-freeness;
    // see the note above.

    /// Demo default pole, exercised by the render path via
    /// `resolved_wireframe_projection`. A literal here, kept in sync with the
    /// state constant by `state::stereographic_default_pole_is_unit_cell_center`.
    fn default_stereographic() -> rye_math::Projection<4> {
        rye_math::Projection::Stereographic {
            pole: state::STEREOGRAPHIC_DEFAULT_POLE,
        }
    }

    /// The body-local projected point a near-pole cap vertex maps to under the
    /// `+w` pole: a w-slice cap vertex within the angular epsilon of `+w`. Returns
    /// the projected point the fill / perimeter / points clip all test.
    fn cap_projected(cap_r3: [f32; 3], w_slice: f32, proj: &rye_math::Projection<4>) -> Vec3 {
        let scale = perspective_scale_at_w(w_slice, proj);
        cap_vertex_projected_and_world(cap_r3, w_slice, scale, proj, Vec3::ZERO).0
    }

    /// The cap FILL drops a triangle touching a near-pole vertex and keeps a
    /// triangle whose vertices are all far from the pole, at TRIANGLE granularity:
    /// a fan triangle with one near-pole vertex vanishes entirely (its index
    /// triple is removed), while a far fan keeps every triangle. The fill clip is
    /// a whole-triangle drop (no boundary cut), mirroring the deep-pole drop in
    /// `stereographic_clip_cuts_to_boundary_and_drops_deep_pole` for the fill path.
    #[test]
    fn cap_fill_triangle_dropped_near_pole() {
        let r = STEREOGRAPHIC_VIEW_RADIUS;
        // Three projected points: index 0 and 1 well inside the radius, index 2
        // far outside it (the near-pole blow-up). Two fan triangles share the
        // centroid (0): [0,1,2] touches the near-pole vertex, [0,1,1] does not.
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

    /// Fill and perimeter cull in LOCKSTEP: for a shared near-pole cap vertex,
    /// the predicate the fill uses (`retain_in_radius_triangles` via
    /// `sample_in_radius` on `cap_vertex_projected_and_world`'s projected point)
    /// agrees with the perimeter's per-segment `sample_in_radius` on the very same
    /// projected point and radius. Pins the RANK 1 fill/outline agreement: both
    /// drop exactly when the body-local projected magnitude exceeds the radius.
    #[test]
    fn cap_fill_matches_perimeter_clip() {
        let proj = rye_math::Projection::Stereographic { pole: Vec4::W };
        let clip = stereographic_clip_radius(&proj, STEREOGRAPHIC_VIEW_RADIUS);
        // A near-pole cap vertex (w-slice close to +w, off-axis so it projects to
        // a large finite point) and a far one on the equatorial slice.
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
        // The fill compaction over a single fan touching the near vertex must
        // drop it iff the perimeter would, on the same projected points + radius.
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
    /// `sample_in_radius` gate `render_points` applies after projecting each
    /// vertex / cell-center. Pins the RANK 2 consistency: a giant near-pole disc
    /// is culled (clean blink) just as the touching edge is. Affine projection
    /// (`clip_radius == None`) keeps every point.
    #[test]
    fn points_overlay_drops_near_pole_vertex() {
        let proj = rye_math::Projection::Stereographic { pole: Vec4::W };
        let clip = stereographic_clip_radius(&proj, STEREOGRAPHIC_VIEW_RADIUS);
        // Body-local vertex within the angular epsilon of +w: project it the same
        // way render_points does, then apply the gate.
        let v_near = <rye_math::EuclideanR4 as rye_math::RasterizableSpace<4>>::project_point(
            near_pole(1.0),
            &proj,
        );
        let v_far = <rye_math::EuclideanR4 as rye_math::RasterizableSpace<4>>::project_point(
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
        // Affine projection carries no clip: even the near-pole image is kept
        // (Identity has no point-at-infinity, so the magnitude is bounded anyway).
        let affine_clip =
            stereographic_clip_radius(&rye_math::Projection::Identity, STEREOGRAPHIC_VIEW_RADIUS);
        assert!(
            sample_in_radius(v_near, affine_clip),
            "affine projection must keep every point (no clip)"
        );
    }

    /// The default-pole render path projects through the `+w` pole, and the cap
    /// clip applies to it: a cap vertex near `+w` is dropped, one far from it
    /// kept. Pins that `resolved_wireframe_projection`'s pole substitution flows
    /// into the cap fill without re-deriving the projection. Also a demo-level
    /// guard that the conformal map is the pure rye-math primitive (a cap vertex
    /// far from the pole projects to a bounded, finite, kept point).
    #[test]
    fn cap_fill_uses_default_plus_w_pole() {
        let proj = default_stereographic();
        let clip = stereographic_clip_radius(&proj, STEREOGRAPHIC_VIEW_RADIUS);
        // A 4D point in the +w pole's near neighborhood: `(0.05, 0, 0, 1.0)`
        // normalizes to dot ~ 0.99875 with +w, image magnitude ~40, past the
        // ~35 radius, so it drops.
        let near = cap_projected([0.05, 0.0, 0.0], 1.0, &proj);
        assert!(
            !sample_in_radius(near, clip),
            "cap vertex near the +w pole (|.| = {}) must drop",
            near.length()
        );
        // A point far from +w (w = 0) stays bounded and finite, the pure
        // conformal image, and is kept.
        let far = cap_projected([-0.4, -0.3, 0.2], 0.0, &proj);
        assert!(
            far.is_finite() && sample_in_radius(far, clip),
            "off-pole cap vertex must stay finite + in-radius: {far:?}"
        );
    }

    /// The cap-fill clip scratch (`section_clip_projected_scratch`, taken via
    /// `std::mem::take`) retains capacity across two compactions, so the hot path
    /// has no per-frame allocation. Mirrors
    /// `stereographic_clip_reuses_scratch_without_realloc` for the fill path:
    /// fill the buffer, drop triangles, re-fill, and assert no growth.
    #[test]
    fn cap_fill_scratch_reused_without_realloc() {
        let r = STEREOGRAPHIC_VIEW_RADIUS;
        // Simulate the per-append fill of `proj_scratch`: push projected points,
        // compact, then clear + re-push (what build_section_layer_meshes does per
        // body, twice across two frames).
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
