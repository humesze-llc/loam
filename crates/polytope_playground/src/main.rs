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
    egui, App, Camera, CameraController, FirstPersonController, FrameCtx, OrbitController,
    RunConfig, SetupCtx,
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

/// Uniform R³ scale factor that `projection` applies to a 4D point with `w = w_slice`.
/// For `Projection::Identity` (drop-w) the result is `1.0`; for `Projection::Perspective4D`
/// it's `focal_distance / (focal_distance - w_slice)`, clamped against the same epsilon
/// the impl uses internally. Used by the wireframe overlay to translate section caps (whose
/// vertices all share `w = w_slice`) into the perspective-scaled R³ frame without re-running
/// the cap algorithm in 4D.
fn perspective_scale_at_w(w_slice: f32, projection: &rye_math::Projection<4>) -> f32 {
    match *projection {
        rye_math::Projection::Identity | rye_math::Projection::Orthographic { .. } => 1.0,
        rye_math::Projection::Perspective4D { focal_distance } => {
            focal_distance / (focal_distance - w_slice).max(1e-4)
        }
    }
}

/// Map a body-local R³ point to world R³: scale by `section_scale` (the perspective scale at
/// the cap's w-coordinate) then translate by the body's R³ position. Cap rendering uses this
/// because the cross-section algorithm internally drops w and emits body-local R³; the world
/// transform happens here.
fn local_r3_to_world(p: [f32; 3], section_scale: f32, body_pos_r3: Vec3) -> [f32; 3] {
    let scaled = Vec3::from_array(p) * section_scale;
    (scaled + body_pos_r3).to_array()
}
use rye_scene::{Scene4, SceneNode4};
use rye_shape::LineMesh;
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
use consts::{BODY_SIZE, BODY_Y, T_SCRUB_RATE, T_SLIDER_INITIAL, W_RANGE, W_SCRUB_RATE};
use state::{
    body_position, CameraMode, Demo, RotationMode, SurfaceMode, ViewMode, WireframeColorMode,
    WireframeProjection,
};

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
        // Same depth-attachment setup as the other rasterizer nodes so points respect the
        // shared section-faces depth buffer (sprites that sit behind a cap get occluded).
        let points_node = PointRasterNode::new(
            &ctx.rd.device,
            ctx.rd.target_format(),
            DepthMode::ReadOnly {
                format: SECTION_FACES_DEPTH_FORMAT,
            },
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

        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 3.0, 9.0);
        let mut orbit: OrbitController<EuclideanR3> = OrbitController::default();
        // Wider orbit so all four bodies in the row are visible at
        // default zoom; user can scroll-zoom in.
        orbit.set_orbit(9.5, -0.25);

        // Free-roam controller is constructed but unused until the user
        // toggles via `camera freecam`. Initial yaw 0 / pitch 0 matches the
        // tesseract_demo pattern; the camera's actual orientation seeds
        // from the orbit's last frame at toggle time.
        let free_roam = FirstPersonController::<EuclideanR3>::new(0.0, 0.0);
        let free_roam_pos = camera.position;

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
            free_roam,
            free_roam_pos,
            camera_mode: CameraMode::default(),
            cursor_grabbed: false,
            node,
            section_edges,
            parent_wireframe,
            wireframe_enabled: false,
            wireframe_nearest_active: true,
            wireframe_perimeter: true,
            wireframe_color_mode: WireframeColorMode::default(),
            wireframe_projection: WireframeProjection::default(),
            section_faces,
            points_node,
            points_enabled: false,
            points_show_vertices: true,
            points_show_cell_centers: true,
            points_size_px: 4.0,
            points_mesh_scratch: rye_shape::PointMesh::<3>::default(),
            section_faces_depth: None,
            section_world_vertices_scratch: Vec::new(),
            section_faces_mesh_scratch: rye_shape::TriangleMesh::<3>::default(),
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

    pub(crate) fn space(&self) -> &EuclideanR3 {
        &self.space
    }

    pub(crate) fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        let dt_secs = ctx.n_ticks as f32 / 60.0;

        // Slice scrub (w axis, up/down arrow keys).
        let dir = (self.slider_up_held as i32 - self.slider_down_held as i32) as f32;
        if dir != 0.0 {
            self.w_slice = (self.w_slice + dir * W_SCRUB_RATE * dt_secs).clamp(-W_RANGE, W_RANGE);
        }

        // Time scrub (t axis, left/right arrow keys). Mirrors the
        // t-slider drag: rebuild `rot_state` from the new
        // `rot_time` via `exp(omega_animation * rot_time)`.
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
            let omega = self.omega_animation();
            self.rot_state = (omega * self.rot_time).exp().normalize();
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
        if !ctx.ui_has_focus {
            match self.camera_mode {
                CameraMode::Orbit => {
                    self.orbit
                        .advance(ctx.input, &mut self.camera, &EuclideanR3, dt_secs);
                }
                // Freecam advances only while the cursor is grabbed. Alt
                // releases the grab for UI interaction; while ungrabbed,
                // the camera holds still so the user can click + drag UI
                // widgets without the scene panning underneath them.
                CameraMode::FreeRoam if self.cursor_grabbed => {
                    // Mouse-look + WASD translation. Controller handles
                    // look (consuming raw mouse delta via use_raw_delta);
                    // position integrates from the drained input axes.
                    self.free_roam
                        .advance(ctx.input, &mut self.camera, &EuclideanR3, dt_secs);
                    const FREECAM_SPEED: f32 = 4.5; // units/sec
                    let mut delta = self.camera.forward * ctx.input.move_forward
                        + self.camera.right * ctx.input.move_right
                        + Vec3::Y * ctx.input.move_up;
                    if delta.length_squared() > 1e-6 {
                        delta = delta.normalize();
                        self.free_roam_pos += delta * FREECAM_SPEED * dt_secs;
                        self.camera.position = self.free_roam_pos;
                    }
                }
                CameraMode::FreeRoam => {
                    // Cursor released (Alt held the toggle): no-op so the
                    // user can interact with UI without the scene moving.
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
        let build_label = format!(
            "version: {}{}",
            env!("BUILD_HASH"),
            env!("BUILD_DIRTY"),
        );
        egui::Area::new(egui::Id::new("polytope-playground-build"))
            .anchor(egui::Align2::RIGHT_TOP, [-12.0, 50.0])
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
        // Find the first polychoron in the row; its vertex 0 is the anchor target.
        let Some((slot, entry)) = self
            .row
            .iter()
            .enumerate()
            .find(|(_, e)| e.shape.polytope4().is_some())
        else {
            return;
        };
        let polytope = entry.shape.polytope4().expect("filter guarantees Some");
        let topo = polytope.topology();
        let canonical_v0 = topo.vertices[0];
        let v_local_4d = BODY_SIZE * self.rot_state.apply(canonical_v0);
        let v_local_r3 = <rye_math::EuclideanR4 as rye_math::RasterizableSpace<4>>::project_point(
            v_local_4d,
            &self.wireframe_projection.to_projection(),
        );
        let n = self.row.len();
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

    pub(crate) fn on_event(&mut self, ev: &winit::event::WindowEvent, _ctx: &mut FrameCtx<'_>) {
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
            KeyCode::Space
                if pressed && !matches!(self.camera_mode, CameraMode::FreeRoam) =>
            {
                self.rotate = !self.rotate;
            }
            // Alt toggles the cursor grab while in freecam: hidden +
            // confined for mouse-look, visible + free for UI interaction.
            // Mode stays FreeRoam either way; only the grab flips. Outside
            // freecam Alt is a no-op (cursor is never grabbed there).
            KeyCode::AltLeft | KeyCode::AltRight
                if pressed && matches!(self.camera_mode, CameraMode::FreeRoam) =>
            {
                self.cursor_grabbed = !self.cursor_grabbed;
                if self.cursor_grabbed {
                    rye_app::cursor::request_grab();
                } else {
                    rye_app::cursor::request_release();
                }
                self.free_roam.use_raw_delta = self.cursor_grabbed;
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

    pub(crate) fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        // Scene renders to the full window. The bottom controls overlay floats on top
        // (Area, not a docked panel; doesn't reserve pixels); the Render settings modal
        // is also free-floating, so the scene viewport is always the framebuffer.
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
                        entry.shape.shape_id(),
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
                let _scope = rye_time::frame_trace::scope("pp-sdf");
                {
                    let u = self.node.uniforms_mut();
                    u.resolution = viewport.resolution_f32();
                    u.viewport_origin = [viewport.x as f32, viewport.y as f32];
                }
                self.node.flush_uniforms(&rd.queue);
                self.node.execute_in_viewport(rd, view, viewport)?;
            }
            // Shared depth attachment for the rasterized section pass + the parent
            // wireframe's depth-test. Ensured + cleared once per Shapes-view frame so
            // the order is: SDF (color only) -> section_faces (writes depth + color
            // when raster mode is on) -> wireframe (tests depth, no write). In SDF
            // mode no pass writes depth, so the cleared `1.0` buffer makes every
            // wireframe fragment pass the depth-test trivially -- visual unchanged.
            self.ensure_and_clear_shared_depth(rd)?;
            if matches!(self.surface_mode, SurfaceMode::Raster) {
                let _scope = rye_time::frame_trace::scope("pp-section-faces");
                self.render_section_faces(rd, view)?;
            }
            // Cross-section + parent-wireframe overlay (when toggled). Only in Shapes
            // view since Filmstrip's per-cell viewport composition would require
            // per-cell depth-clear + per-cell uploads that aren't worth the v1 plumbing.
            if self.wireframe_enabled {
                // pp-wireframe is the project-memory-flagged hot path
                // (`project_polychoral_raster_perf`). Want a per-frame number here so
                // we can confirm (or refute) that hypothesis with browser data.
                let _scope = rye_time::frame_trace::scope("pp-wireframe");
                self.render_wireframe_overlay(rd, view)?;
            }
            // Points overlay (vertex markers + cell-center sprites). Drawn last so the
            // discs sit on top of wireframe edges and section caps at the same depth.
            if self.points_enabled {
                let _scope = rye_time::frame_trace::scope("pp-points");
                self.render_points(rd, view)?;
            }
            Ok(())
        }
    }

    /// Ensure the shared section-faces depth attachment exists at the current
    /// swapchain size + sample count, then clear it to `1.0`. Called once per
    /// Shapes-view frame at the top of the rasterizer chain.
    ///
    /// The buffer is shared between `section_faces` (which writes depth when raster
    /// mode is on) and `parent_wireframe` (which depth-tests against it without
    /// writing). Sharing means we pay one ensure + one clear per frame regardless of
    /// surface mode.
    fn ensure_and_clear_shared_depth(&mut self, rd: &RenderDevice) -> Result<()> {
        let cfg = &rd.surface_bundle.config;
        DepthBuffer::ensure(
            &mut self.section_faces_depth,
            &rd.device,
            SECTION_FACES_DEPTH_FORMAT,
            (cfg.width, cfg.height),
            rd.sample_count(),
        );
        let depth = self
            .section_faces_depth
            .as_ref()
            .expect("ensure() guarantees Some");
        let mut encoder = rd
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shared depth clear"),
            });
        let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shared depth clear pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rd.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Build a combined section-faces mesh across every polychoral body in `self.row`,
    /// upload it, clear the section-faces depth attachment, and execute the triangle
    /// raster pass. No-op when no polychoral body is present in the row.
    ///
    /// Per-body world transform mirrors `render_wireframe_overlay`: canonical vertices
    /// `v` become `body_position + BODY_SIZE * rot_state.apply(v)`. The section
    /// algorithm then runs on these world vertices against the demo's `w_slice`,
    /// producing R³ geometry that composes with the camera the SDF raymarcher uses.
    ///
    /// Depth: within a single cap, the fan triangles are coplanar (the cap is a 2D
    /// polygon in R³) and don't occlude one another. Different caps within one
    /// polytope, and caps across different polychoral bodies, all project to
    /// different camera depths after the view-projection step, so they DO require a
    /// depth attachment to resolve front-to-back occlusion correctly. The shared
    /// depth buffer is ensured + cleared at the top of the Shapes-view render path
    /// (see `Demo::ensure_and_clear_shared_depth`); this pass writes into it.
    /// Build the combined point sprites mesh (vertex markers + cell-center sprites) across
    /// every polychoral body in the row, upload it, and execute the point-disc raster pass.
    ///
    /// Same body-local + perspective + world-translate pattern as the wireframe and section-
    /// faces paths: each body's vertices and cell centers are computed in body-local 4D,
    /// projected through `wireframe_projection`, and translated by the body's R³ position so
    /// the perspective scale doesn't smear the body across its row x-offset.
    ///
    /// Coloring: vertex sprites use the same per-vertex position-derived RGB scheme as the
    /// "unique" wireframe color mode (`vertex_color_by_position`), so vertex sprites visually
    /// belong with the colored wireframe. Cell-center sprites use a uniform white with
    /// reduced alpha to read as a secondary structural marker rather than competing with the
    /// vertex sprites.
    fn render_points(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        let cfg = &rd.surface_bundle.config;
        let n = self.row.len();
        let wireframe_projection = self.wireframe_projection.to_projection();
        // Cell-center sprite color: a dim warm white. Pre-multiplied alpha makes the sprite
        // read as a "second-tier" marker behind the brighter vertex discs.
        const CELL_CENTER_COLOR: [f32; 4] = [0.92, 0.88, 0.78, 0.65];

        let mesh = &mut self.points_mesh_scratch;
        mesh.positions.clear();
        mesh.colors.clear();
        mesh.sizes.clear();

        for (slot, entry) in self.row.iter().enumerate() {
            let Some(polytope) = entry.shape.polytope4() else {
                continue;
            };
            let topo = polytope.topology();
            let body_pos = body_position(slot, n);
            let body_pos_r3 = Vec3::new(body_pos[0], body_pos[1], body_pos[2]);

            if self.points_show_vertices {
                for v in topo.vertices {
                    let v_local = BODY_SIZE * self.rot_state.apply(*v);
                    let v3_local =
                        <rye_math::EuclideanR4 as rye_math::RasterizableSpace<4>>::project_point(
                            v_local,
                            &wireframe_projection,
                        );
                    let v_world = v3_local + body_pos_r3;
                    mesh.positions.push(v_world.to_array());
                    // Color by the canonical (unrotated) vertex position so the sprite hue
                    // matches its corresponding wireframe edge in `WireframeColorMode::Unique`.
                    mesh.colors.push(vertex_color_by_position(*v));
                    mesh.sizes.push(self.points_size_px);
                }
            }
            if self.points_show_cell_centers {
                for c in polytope.cell_centers() {
                    let c_local = BODY_SIZE * self.rot_state.apply(c);
                    let c3_local =
                        <rye_math::EuclideanR4 as rye_math::RasterizableSpace<4>>::project_point(
                            c_local,
                            &wireframe_projection,
                        );
                    let c_world = c3_local + body_pos_r3;
                    mesh.positions.push(c_world.to_array());
                    mesh.colors.push(CELL_CENTER_COLOR);
                    // Cell-center sprites half-sized so they don't compete visually with the
                    // brighter vertex discs.
                    mesh.sizes.push(self.points_size_px * 0.5);
                }
            }
        }

        // Camera matches the wireframe overlay / section faces (same view-projection).
        let view_dir = self.camera.view();
        let aspect = cfg.width as f32 / cfg.height as f32;
        let view_mat = Mat4::look_to_rh(view_dir.position, view_dir.forward, view_dir.up);
        let proj_mat = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.1, 100.0);
        let view_proj = proj_mat * view_mat;
        let vp_size = Vec2::new(cfg.width as f32, cfg.height as f32);
        self.points_node.set_camera(&rd.queue, view_proj, vp_size);
        self.points_node.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            mesh,
            &rye_math::Projection::Identity,
        );
        let depth_view = self
            .section_faces_depth
            .as_ref()
            .map(|b| &b.view)
            .expect("shared depth buffer must be ensured before points overlay");
        self.points_node.execute(rd, view, Some(depth_view), None)?;
        Ok(())
    }

    fn render_section_faces(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        let cfg = &rd.surface_bundle.config;
        let n = self.row.len();

        // Reuse the per-Demo scratch mesh; capacity grows once to fit the largest
        // polychoron and stays there. Each frame's `clear()` keeps the underlying
        // allocations.
        let combined = &mut self.section_faces_mesh_scratch;
        combined.vertices.clear();
        combined.colors.clear();
        combined.indices.clear();

        // Same perspective scaling logic the wireframe path uses (see render_wireframe_overlay):
        // section the body in body-local 4D, then translate the produced R³ caps by the body's
        // R³ position with the active perspective scale at the slice's w. With drop-w this
        // collapses to identity scaling; with Perspective4D the cap scales by
        // `focal / (focal - w_slice)`.
        let wireframe_projection = self.wireframe_projection.to_projection();
        let section_scale = perspective_scale_at_w(self.w_slice, &wireframe_projection);

        for (slot, entry) in self.row.iter().enumerate() {
            let Some(polytope) = entry.shape.polytope4() else {
                continue;
            };
            let topo = polytope.topology();
            let body_pos = body_position(slot, n);
            let body_pos_r3 = Vec3::new(body_pos[0], body_pos[1], body_pos[2]);

            // Body-local 4D scratch (rotor-rotated, scaled, NO world translate). Same
            // rationale as the wireframe path: keep the body's R³ position out of the 4D
            // perspective math so it doesn't get scaled by `focal / (focal - w)`.
            self.section_world_vertices_scratch.clear();
            self.section_world_vertices_scratch.extend(
                topo.vertices
                    .iter()
                    .map(|v| BODY_SIZE * self.rot_state.apply(*v)),
            );

            // Match the SDF's per-body solid coloring: every cap of this polychoron uses
            // the body's identity color from the catalog. Per-face Lambert in the fragment
            // shader adds the geometric depth; the underlying color is flat.
            let [r, g, b] = entry.body_color;
            let start = combined.vertices.len();
            polytope_section_faces_append(
                topo.edges,
                topo.cells,
                &self.section_world_vertices_scratch,
                WPlane::new(self.w_slice),
                [r, g, b, 1.0],
                combined,
            );
            // Translate this body's body-local cap vertices into world R³. Indices were
            // emitted with the correct vertex offset already (the `_append` API handles
            // that internally); only the vertex positions need rebasing.
            for v in &mut combined.vertices[start..] {
                *v = local_r3_to_world(*v, section_scale, body_pos_r3);
            }
        }

        // Empty mesh handling lives in `TriangleRasterNode::execute` (it short-circuits
        // when `index_count == 0`); no need for a redundant early-return here.

        // Camera matches the SDF raymarcher's effective view-projection (same as the
        // wireframe overlay uses), so pixel-aligned composition over the SDF pass.
        let view_dir = self.camera.view();
        let aspect = cfg.width as f32 / cfg.height as f32;
        let view_mat = Mat4::look_to_rh(view_dir.position, view_dir.forward, view_dir.up);
        let proj_mat = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.1, 100.0);
        let view_proj = proj_mat * view_mat;

        // The shared depth attachment is ensured + cleared once per frame by
        // `ensure_and_clear_shared_depth` at the top of the Shapes-view render path;
        // here we just consume the view for the triangle pass's depth-write.
        let depth = self
            .section_faces_depth
            .as_ref()
            .expect("shared depth buffer must be ensured before section_faces");

        self.section_faces.set_camera(&rd.queue, view_proj);
        self.section_faces.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            combined,
            &rye_math::Projection::Identity,
        );
        self.section_faces
            .execute(rd, view, Some(&depth.view), None)?;
        Ok(())
    }

    /// Build the three overlay meshes (section triangles, section perimeter edges, parent
    /// wireframe) from the current row + rotor + w_slice, upload them, clear the overlay
    /// depth buffer, and execute the three raster passes on top of the existing SDF render.
    ///
    /// Per-body transform: each canonical Polytope4 vertex `v` becomes the world Vec4
    /// `body.position + BODY_SIZE * rot_state.apply(v)`. The section algorithm then runs
    /// on these world vertices against the demo's `w_slice`, producing geometry in world
    /// R³ that composes cleanly with the SDF camera frame.
    ///
    /// Non-polychoral shapes (Clifford torus, duocylinder, etc.) in the row are skipped:
    /// they have no [`rye_physics::polytope::Polytope4`] mapping and the cross-section
    /// algorithm doesn't apply to smooth surfaces.
    fn render_wireframe_overlay(
        &mut self,
        rd: &RenderDevice,
        view: &wgpu::TextureView,
    ) -> Result<()> {
        let cfg = &rd.surface_bundle.config;
        let n = self.row.len();

        // Build combined meshes across the entire row.
        let mut section_edges = LineMesh::<3>::default();
        let mut parent_lines = LineMesh::<3>::default();
        // Uniform-alpha endpoints when `nearest-active` is off; the active-mode mapping
        // interpolates between DIM (cells the slice misses entirely) and BRIGHT (cells
        // the slice is at the midpoint of).
        const PARENT_ALPHA_UNIFORM: f32 = 0.55;
        const PARENT_ALPHA_DIM: f32 = 0.10;
        const PARENT_ALPHA_BRIGHT: f32 = 0.85;
        const PARENT_WIDTH: f32 = 1.2;
        // Active-mode palette. Green for edges in any currently-intersected cell;
        // neutral gray for the rest. Chosen for clear binary contrast against the
        // grayish-blue scene backdrop and the dim ground checkerboard.
        const ACTIVE_GREEN: [f32; 4] = [0.40, 1.00, 0.55, 1.0];
        const INACTIVE_GRAY: [f32; 4] = [0.55, 0.55, 0.58, 1.0];
        let nearest_active = self.wireframe_nearest_active;
        let color_mode = self.wireframe_color_mode;
        // Resolve once per frame; same projection applied to every body's wireframe so all
        // bodies share a consistent R³ embedding.
        let wireframe_projection = self.wireframe_projection.to_projection();

        for (slot, entry) in self.row.iter().enumerate() {
            let Some(polytope) = entry.shape.polytope4() else {
                continue;
            };
            let topo = polytope.topology();
            let body_pos = body_position(slot, n);
            // Body's R³ position. The body sits at body_pos.w = 0 in 4D, so the perspective
            // projection at this body's location collapses to a pure R³ translation; doing
            // the projection in body-local 4D and translating in R³ AFTER is the only way
            // to keep the body's apparent x-position stable when Perspective4D scales the
            // (x, y, z) channel by `focal / (focal - w)`.
            let body_pos_r3 = Vec3::new(body_pos[0], body_pos[1], body_pos[2]);
            // Body-local 4D vertices: rotor-rotated, scaled, NO world translate yet. The
            // translate happens after the chosen `wireframe_projection` maps each vertex
            // from R⁴ to R³.
            let local_vertices: Vec<Vec4> = topo
                .vertices
                .iter()
                .map(|v| BODY_SIZE * self.rot_state.apply(*v))
                .collect();

            // Cross-section perimeter edges. The cyan outlines bound each cell cap on the
            // slice; gated by `wireframe_perimeter` so users can show the parent edge graph
            // on its own. `polytope_section_overlay_with_vertices` uses drop-w internally,
            // so its R³ output is in body-local frame; we apply the perspective scale at
            // w_slice (uniform across the cap because every cap point shares the slice's w)
            // and translate to world R³.
            if self.wireframe_perimeter {
                let section_scale = perspective_scale_at_w(self.w_slice, &wireframe_projection);
                let (_tri, mut perim) = polytope_section_overlay_with_vertices(
                    topo.edges,
                    topo.cells,
                    &local_vertices,
                    WPlane::new(self.w_slice),
                );
                for (a, b) in &mut perim.segments {
                    *a = local_r3_to_world(*a, section_scale, body_pos_r3);
                    *b = local_r3_to_world(*b, section_scale, body_pos_r3);
                }
                section_edges.segments.append(&mut perim.segments);
                section_edges.colors.append(&mut perim.colors);
                section_edges.widths.append(&mut perim.widths);
            }

            // Per-cell "crossing strength" in [0, 1]: 1 when `w_slice` is at the cell's
            // w-midpoint (cap face is widest), 0 when the slice is at the cell's w-boundary
            // or outside it entirely. A linear w-midpoint proxy avoids computing actual
            // cap areas while giving the same visual gradient: cells the slice is *deep
            // in* peak in brightness, cells the slice barely touches taper to dim.
            let cell_strengths: Vec<f32> = topo
                .cells
                .iter()
                .map(|cell| {
                    let (w_min, w_max) = cell
                        .iter()
                        .map(|&i| local_vertices[i as usize].w)
                        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), w| {
                            (lo.min(w), hi.max(w))
                        });
                    let half_extent = (w_max - w_min) * 0.5;
                    if half_extent <= 0.0 {
                        return 0.0;
                    }
                    let mid = (w_min + w_max) * 0.5;
                    let dist = (self.w_slice - mid).abs();
                    (1.0 - dist / half_extent).clamp(0.0, 1.0)
                })
                .collect();

            // Per-edge brightness: max strength across cells the edge belongs to. An edge
            // belongs to a cell when both endpoints sit in that cell's vertex list. This
            // lets a single edge "light up" as soon as ANY containing cell is being
            // crossed deep; doesn't require the slice to specifically hit the cell whose
            // boundary the edge sits on.
            let edge_strength = |i: u32, j: u32| -> f32 {
                let mut best = 0.0_f32;
                for (cell, strength) in topo.cells.iter().zip(cell_strengths.iter()) {
                    if cell.contains(&i) && cell.contains(&j) && *strength > best {
                        best = *strength;
                    }
                }
                best
            };

            // Active-mode binary classification: an edge is "active" if at least one of
            // its containing cells has the slice strictly between its w-range endpoints
            // (i.e., the slice is currently producing a cap from that cell). Uses the
            // same `edges-in-cell` membership rule as `edge_strength`; threshold is
            // `cell_strength > 0.0`, which corresponds exactly to `w_min < w_slice <
            // w_max` after the `(1 - dist / half_extent)` clamp.
            let edge_is_active = |i: u32, j: u32| -> bool {
                topo.cells
                    .iter()
                    .zip(cell_strengths.iter())
                    .any(|(cell, &s)| s > 0.0 && cell.contains(&i) && cell.contains(&j))
            };

            // Parent wireframe: every polytope edge as a world-R³ line. Base RGB is
            // picked by `wireframe_color_mode`:
            // - `Unique`: per-vertex position-derived RGB from the canonical vertex
            //   set so each vertex gets a distinct hue from its 4D coordinates and
            //   the polytope's symmetry shows as smooth gradients (same scheme as
            //   `Polytope4::lines_colored_by_position`).
            // - `Active`: binary green/gray by cell-activity (see `edge_is_active`).
            // Alpha is then modulated per-edge by the `nearest-active` strength (when
            // that toggle is on) or held uniform.
            for &[i, j] in topo.edges {
                let ia = i as usize;
                let ja = j as usize;
                let a = local_vertices[ia];
                let b = local_vertices[ja];
                let (mut color_a, mut color_b) = match color_mode {
                    WireframeColorMode::Unique => (
                        vertex_color_by_position(topo.vertices[ia]),
                        vertex_color_by_position(topo.vertices[ja]),
                    ),
                    WireframeColorMode::Active => {
                        let c = if edge_is_active(i, j) {
                            ACTIVE_GREEN
                        } else {
                            INACTIVE_GRAY
                        };
                        (c, c)
                    }
                };
                let alpha = if nearest_active {
                    let s = edge_strength(i, j);
                    PARENT_ALPHA_DIM + (PARENT_ALPHA_BRIGHT - PARENT_ALPHA_DIM) * s
                } else {
                    PARENT_ALPHA_UNIFORM
                };
                color_a[3] = alpha;
                color_b[3] = alpha;
                // Project 4D endpoints to R³ through the active wireframe projection (in
                // body-local frame), then translate by `body_pos_r3` to land in world R³.
                // For DropW the projection is identity-on-(x, y, z); for Perspective4D each
                // component scales by `focal / (focal - w)`.
                let a3_local =
                    <rye_math::EuclideanR4 as rye_math::RasterizableSpace<4>>::project_point(
                        a,
                        &wireframe_projection,
                    );
                let b3_local =
                    <rye_math::EuclideanR4 as rye_math::RasterizableSpace<4>>::project_point(
                        b,
                        &wireframe_projection,
                    );
                let a3 = a3_local + body_pos_r3;
                let b3 = b3_local + body_pos_r3;
                parent_lines.segments.push((a3.to_array(), b3.to_array()));
                parent_lines.colors.push((color_a, color_b));
                parent_lines.widths.push(PARENT_WIDTH);
            }
        }

        // Upload (each call is a no-op when its mesh is empty).
        self.section_edges.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &section_edges,
            &rye_math::Projection::Identity,
            1,
        );
        self.parent_wireframe.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &parent_lines,
            &rye_math::Projection::Identity,
            1,
        );

        // Camera. Build the same view+projection matrix the SDF raymarcher uses
        // implicitly via its ray basis, so the rasterized overlay aligns pixel-for-pixel
        // with the raymarched scene.
        let view_dir = self.camera.view();
        let aspect = cfg.width as f32 / cfg.height as f32;
        let view_mat = Mat4::look_to_rh(view_dir.position, view_dir.forward, view_dir.up);
        let proj_mat = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.1, 100.0);
        let view_proj = proj_mat * view_mat;
        let vp_size = Vec2::new(cfg.width as f32, cfg.height as f32);
        self.section_edges.set_camera(&rd.queue, view_proj, vp_size);
        self.parent_wireframe
            .set_camera(&rd.queue, view_proj, vp_size);

        // Section perimeter edges then dim parent wireframe. Both depth-test (no write)
        // against the shared section-faces depth attachment so lines behind a cap get
        // correctly occluded across polytopes in a row. The shared buffer is ensured +
        // cleared earlier in the frame; here we just borrow the view. In SDF mode no
        // pass writes depth, so the cleared `1.0` buffer makes every fragment pass the
        // test, preserving the historical visual.
        let depth_view = self
            .section_faces_depth
            .as_ref()
            .map(|b| &b.view)
            .expect("shared depth buffer must be ensured before wireframe overlay");
        self.section_edges
            .execute(rd, view, Some(depth_view), None)?;
        self.parent_wireframe
            .execute(rd, view, Some(depth_view), None)?;
        Ok(())
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
                .toggle(
                    "perimeter",
                    "cyan cross-section perimeter outlines (bare flips)",
                    |d, v| {
                        d.wireframe_perimeter = v.unwrap_or(!d.wireframe_perimeter);
                        Ok(())
                    },
                )
                .choice(
                    "color",
                    "base RGB for parent edges (bare cycles): unique (default) or active",
                    &["unique", "active"],
                    |d, name| {
                        d.wireframe_color_mode = match name {
                            Some(n) => WireframeColorMode::from_token(n).ok_or_else(|| {
                                anyhow!("unknown color mode `{n}` (try unique|active)")
                            })?,
                            None => match d.wireframe_color_mode {
                                WireframeColorMode::Unique => WireframeColorMode::Active,
                                WireframeColorMode::Active => WireframeColorMode::Unique,
                            },
                        };
                        Ok(())
                    },
                )
                .choice(
                    "perspective",
                    "wireframe 4D->R³ projection (bare cycles): drop-w (default) or w-depth",
                    &["drop-w", "w-depth"],
                    |d, name| {
                        d.wireframe_projection = match name {
                            Some(n) => WireframeProjection::from_token(n).ok_or_else(|| {
                                anyhow!("unknown projection `{n}` (try drop-w|w-depth)")
                            })?,
                            None => match d.wireframe_projection {
                                WireframeProjection::DropW => WireframeProjection::WDepth,
                                WireframeProjection::WDepth => WireframeProjection::DropW,
                            },
                        };
                        Ok(())
                    },
                ),
        );

        // Points overlay: vertex markers + cell-center sprites. Same SubcommandSet shape as
        // `wireframe`: bare flips main on/off, subcommands gate per-category visibility, and
        // a size knob for the disc radius. Off by default so first-launch readers see the
        // demo's identity (SDF / raster + wireframe) rather than a vertex cloud.
        c.register(
            rye_egui::subcommands::<Demo>("points", "vertex + cell-center sprite overlay")
                .on_bare(|d| {
                    d.points_enabled = !d.points_enabled;
                    Ok(())
                })
                .toggle(
                    "vertices",
                    "render a disc at each polytope vertex (bare flips)",
                    |d, v| {
                        d.points_show_vertices = v.unwrap_or(!d.points_show_vertices);
                        Ok(())
                    },
                )
                .toggle(
                    "cell-centers",
                    "render a dim disc at each cell's centroid (bare flips)",
                    |d, v| {
                        d.points_show_cell_centers = v.unwrap_or(!d.points_show_cell_centers);
                        Ok(())
                    },
                )
                .custom(
                    "size",
                    "set the disc radius in pixels (e.g. `points size 8`)",
                    &[],
                    &[],
                    |d, rest, _out| {
                        let Some(token) = rest.first() else {
                            return Err(anyhow!("usage: points size <pixels>"));
                        };
                        let px: f32 = token
                            .parse()
                            .map_err(|e| anyhow!("invalid pixel value `{token}`: {e}"))?;
                        if !(1.0..=64.0).contains(&px) {
                            return Err(anyhow!("points size {px} out of range; expected 1..=64"));
                        }
                        d.points_size_px = px;
                        Ok(())
                    },
                ),
        );

        // Polychoral surface renderer: raster (default) / SDF / off. Bare `surface` is
        // shorthand for "off" so the user can hide cap fills quickly when inspecting the
        // wireframe and cross-section perimeter on their own. Explicit `surface raster`
        // and `surface sdf` set those modes; `surface off` is the same as bare.
        c.register(
            rye_egui::cmd(
                "surface",
                "polychoral surface mode: raster | sdf | off (bare = off)",
                |args, demo: &mut Demo, _out| {
                    let next = match args.first().copied() {
                        Some(token) => SurfaceMode::from_token(token).ok_or_else(|| {
                            anyhow!("unknown arg `{token}` (try raster|sdf|off)")
                        })?,
                        None => SurfaceMode::Off,
                    };
                    if next != demo.surface_mode {
                        demo.surface_mode = next;
                        // Re-emit the SDF body list: switching INTO Sdf mode makes the
                        // polychora live in the kernel, switching OUT marks them inert.
                        demo.rebuild_bodies();
                    }
                    Ok(())
                },
            )
            .with_args(&[&["raster", "sdf", "off"]])
            .with_long_help(
                "Selects how the six regular convex 4-polytopes (5-cell, tesseract, 16-cell,\n\
                 24-cell, 120-cell, 600-cell) are rendered.\n\
                 \n\
                 subcommands:\n  \
                 raster  Rasterized cross-section cell-caps (the default). Face-normal Lambert\n                         lit, per-body solid color. Much faster for the 120-cell + 600-cell\n                         and exact (no SDF approximation).\n  \
                 sdf     SDF raymarch. The historical pre-rasterizer path; smoother shading\n                         but the 120-cell and 600-cell carry a face-plane approximation BUG.\n                         Kept for visual comparison.\n  \
                 off     No surface rendered. Wireframe overlay + cross-section perimeter\n                         stay visible if enabled; the cap interiors are blank. Useful for\n                         inspecting the wireframe on its own.\n\
                 \n\
                 Bare `surface` (no argument) is shorthand for `surface off`.\n\
                 \n\
                 Smooth-surface shapes (Clifford torus, duocylinder, spherinder, 3-sphere)\n\
                 ignore this and always render via the SDF; they have no rasterizer path.",
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
        c.register(
            rye_egui::cmd::<Demo, _>(
                "camera",
                "camera mode: orbit (default) or freecam (WASD + mouse-look). Bare cycles.",
                |args, demo, out| {
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
                            // freecam left it. Release the cursor grab; the
                            // UI is interactive again, mouse-look is off.
                            demo.orbit = OrbitController::default();
                            demo.orbit.set_orbit(9.5, -0.25);
                            demo.cursor_grabbed = false;
                            demo.free_roam.use_raw_delta = false;
                            rye_app::cursor::request_release();
                            out.line("camera: orbit (reset to world origin)");
                        }
                        CameraMode::FreeRoam => {
                            // Seed freecam from the camera's current pose so
                            // the toggle feels continuous instead of
                            // teleporting. Grab the cursor so panning works
                            // past the screen edge; user presses Alt to
                            // release for UI access.
                            demo.free_roam_pos = demo.camera.position;
                            demo.cursor_grabbed = true;
                            demo.free_roam.use_raw_delta = true;
                            rye_app::cursor::request_grab();
                            out.line(
                                "camera: freecam (WASD + Space/Shift; mouse-look; Alt to free cursor)",
                            );
                        }
                    }
                    Ok(())
                },
            )
            .with_args(&[&["orbit", "freecam"]]),
        );

        // The `floor` toggle the user asked for is deferred. The SDF ground
        // (a y=0 HalfSpace4D in the scene composition) is baked into the
        // shader module at App::setup time via `Scene4::to_hyperslice_wgsl`,
        // so toggling visibility at runtime requires either recompiling the
        // shader (cheap, ~100-300ms wasm) or adding a kernel-uniform flag to
        // rye-scene (cheaper per-toggle but engine-level work). Tracked as
        // follow-up; not part of this polish pass.

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

    fn on_event(&mut self, ev: &winit::event::WindowEvent, ctx: &mut FrameCtx<'_>) {
        // Suppress demo keybinds when egui is actively capturing
        // keyboard input (any TextEdit focused: console, formula
        // bar, etc.) so typing `reset` into the console doesn't
        // also fire the R hotkey, etc. When the user clicks
        // outside the egui widget that had focus, egui releases
        // keyboard focus and the next frame's `on_event` routes
        // hotkeys back to the demo as normal.
        if !self.last_egui_keyboard {
            self.demo.on_event(ev, ctx);
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
