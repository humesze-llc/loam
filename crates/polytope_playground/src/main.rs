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
use consts::{BODY_SIZE, BODY_Y, T_SCRUB_RATE, T_SLIDER_INITIAL, W_SCRUB_RATE};
use state::{
    body_position, CameraMode, Demo, RotationMode, SurfaceMode, ViewMode, WireframeColorMode,
    WireframeProjection,
};

// Cool-to-warm diverging palette for the `w-depth` wireframe color mode.
// Tracks SIGNED w in the body-local frame: cool blue at extreme -w (the
// vertex sits "behind" the slice plane in 4D), warm orange at extreme +w
// (vertex sits "in front"), and a near-neutral midpoint at w = 0 (vertex
// sits on the slice plane). This is the same palette + scheme the
// `LineRasterStaticR4` shader uses in `tesseract_demo`, where it reads as
// "which edge is in front of which in the rotating tesseract." Picking up
// the same colors here lets the playground demonstrate the same w-depth
// cue across every polytope.
//
// Signed (not `|w|`) is the key: a tesseract vertex at +0.5 and one at
// -0.5 are visually distinguishable, so the viewer can track a vertex
// migrating between "near" and "far" camps as the rotor swings. The
// previous |w|-based scheme collapsed both into the same color and made
// 4D rotation visually indistinguishable from a 3D twist at certain
// angles.
const W_DEPTH_BACK: [f32; 3] = [0.30, 0.42, 0.58];
const W_DEPTH_FRONT: [f32; 3] = [1.00, 0.78, 0.45];

/// Convert HSV to RGB. h, s, v in [0, 1]; output channels in [0, 1].
/// See e.g. Foley, van Dam et al., *Computer Graphics: Principles and
/// Practice*, 2nd ed., section 13.3.4.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h6 = h.fract() * 6.0;
    let c = v * s;
    let x = c * (1.0 - (h6 % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match h6.floor() as i32 % 6 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [r + m, g + m, b + m]
}

/// Deterministic palette: golden-ratio hue spacing + saturation/value
/// modulation so consecutive palette indices land far apart in HSV. The
/// greedy graph-coloring in [`unique_edge_palette`] only ever needs the
/// first few indices for any of the regular 4-polytopes, so the first ~12
/// entries are the perceptually-important ones.
fn unique_edge_palette_color(idx: usize) -> [f32; 4] {
    // Golden-ratio conjugate: maximally irrational, spreads hues evenly
    // even for small N. See Knuth, TAOCP vol. 3, section 6.4.
    const PHI_INV: f32 = 0.618_034;
    let h = ((idx as f32) * PHI_INV).fract();
    // 3-cycle on saturation and 2-cycle on value so adjacent indices (which
    // already differ in hue by ~137 degrees) also differ in S/V.
    let s = 0.78 + 0.18 * ((idx % 3) as f32 / 2.0);
    let v = 0.92 - 0.18 * (((idx / 3) % 2) as f32);
    let [r, g, b] = hsv_to_rgb(h, s, v);
    [r, g, b, 1.0]
}

/// Per-edge RGBA palette via greedy graph-coloring on the line graph of
/// `topo.edges`: two edges sharing a vertex are forbidden the same palette
/// index, so locally-adjacent edges always read as different colors. The
/// coloring is deterministic in the input edge order (which is itself
/// deterministic per [`rye_physics::polytope::Polytope4Topology::edges`]),
/// so the same polytope paints identically across runs.
///
/// Greedy first-fit coloring matches the line graph's chromatic number to
/// within a factor that depends on vertex ordering; for the six regular
/// convex 4-polytopes' edge graphs the result is within 1-2 colors of
/// optimal in practice, which is fine for visual identification.
fn unique_edge_palette(edges: &[[u32; 2]]) -> Vec<[f32; 4]> {
    let n = edges.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            let [a0, a1] = edges[i];
            let [b0, b1] = edges[j];
            if a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1 {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    let mut color_idx = vec![usize::MAX; n];
    let mut used = std::collections::HashSet::<usize>::new();
    for i in 0..n {
        used.clear();
        for &nbr in &adj[i] {
            if color_idx[nbr] != usize::MAX {
                used.insert(color_idx[nbr]);
            }
        }
        let mut c = 0;
        while used.contains(&c) {
            c += 1;
        }
        color_idx[i] = c;
    }
    color_idx
        .into_iter()
        .map(unique_edge_palette_color)
        .collect()
}

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
            let dist = (w_slice - mid).abs();
            (1.0 - dist / half_extent).clamp(0.0, 1.0)
        })
        .collect()
}

/// Per-vertex `w-depth` color: cool blue at -w extreme, warm orange at
/// +w extreme, near-neutral on the slice plane. `w_extent_local` is the
/// FIXED post-scale bound on `|w|` for this polytope's canonical vertex
/// set (`canonical_max_w * body_size`), so the gradient stays stable as
/// the rotor spins: a vertex that lands at `w = +0.4 * body_size` paints
/// the same color regardless of orientation. Per-vertex (not per-edge)
/// so edges that span the slice plane visibly fade cool to warm along their
/// length, surfacing the w-depth migration directly.
fn w_depth_color(w: f32, w_extent_local: f32) -> [f32; 4] {
    let denom = w_extent_local.max(1e-6);
    // Shift signed w from [-extent, +extent] into [0, 1]; clamp guards
    // against any vertex slightly past the canonical extent (numeric drift
    // or non-unit-circumradius polytopes).
    let t = ((w / denom) * 0.5 + 0.5).clamp(0.0, 1.0);
    [
        W_DEPTH_BACK[0] + (W_DEPTH_FRONT[0] - W_DEPTH_BACK[0]) * t,
        W_DEPTH_BACK[1] + (W_DEPTH_FRONT[1] - W_DEPTH_BACK[1]) * t,
        W_DEPTH_BACK[2] + (W_DEPTH_FRONT[2] - W_DEPTH_BACK[2]) * t,
        1.0,
    ]
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
        // Wider orbit so all four bodies in the row are visible at
        // default zoom; user can scroll-zoom in.
        orbit.set_orbit(9.5, -0.25);

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
            wireframe_perimeter: true,
            wireframe_color_mode: WireframeColorMode::default(),
            wireframe_projection: WireframeProjection::default(),
            wireframe_width_px: 1.8,
            wireframe_alpha: 1.0,
            unique_edge_palette_cache: std::collections::HashMap::new(),
            surface_scale: 1.0,
            surface_alpha: 1.0,
            floor_enabled: true,
            section_faces,
            section_faces_translucent,
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

        // Slice scrub (w axis, up/down arrow keys). Clamps against the
        // surface-scaled range so the keyboard scrub matches the slider
        // bounds after `surface scale`.
        let dir = (self.slider_up_held as i32 - self.slider_down_held as i32) as f32;
        if dir != 0.0 {
            let w_range = self.effective_w_range();
            self.w_slice = (self.w_slice + dir * W_SCRUB_RATE * dt_secs).clamp(-w_range, w_range);
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
        let v_local_4d = self.effective_body_size() * self.rot_state.apply(canonical_v0);
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
            let strip_w_extent = self.effective_body_size();
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
                        self.effective_body_size(),
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
    /// `v` become `body_position + effective_body_size() * rot_state.apply(v)`. The section
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
    /// Coloring: sprites honor the active [`WireframeColorMode`] so the
    /// points overlay reads as part of the same wireframe rendering pass.
    /// Per-vertex sprites get the same color the matching wireframe vertex
    /// would carry; per-cell-center sprites get the dominant-cell color
    /// (cell strength for `Active`, cell-center w for `WDepth`, position-
    /// derived gradient otherwise). For `UniqueEdge` mode the points fall
    /// back to the position-gradient because the unique-edge palette is
    /// edge-indexed and has no canonical vertex assignment.
    fn render_points(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        let cfg = &rd.surface_bundle.config;
        let n = self.row.len();
        let wireframe_projection = self.wireframe_projection.to_projection();
        // Active-mode palette: bright green for vertices that belong to a
        // currently-intersected cell, dim gray otherwise. Same hues the
        // wireframe overlay uses so the visual identity stays consistent.
        const ACTIVE_GREEN: [f32; 4] = [0.40, 1.00, 0.55, 1.0];
        const INACTIVE_GRAY: [f32; 4] = [0.55, 0.55, 0.58, 0.85];
        let color_mode = self.wireframe_color_mode;

        let body_size = self.effective_body_size();
        let w_slice = self.w_slice;
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

            // Per-frame, per-polytope body-local 4D vertices (rotated + scaled).
            // Shared by the vertex AND cell-center loops: for WDepth we need
            // the maximum |w| across them to normalize the gradient, and for
            // Active we need them to derive cell strengths.
            let local_vertices: Vec<Vec4> = topo
                .vertices
                .iter()
                .map(|v| body_size * self.rot_state.apply(*v))
                .collect();
            // Same canonical-max-w normalization the wireframe overlay uses
            // (see render_wireframe_overlay), keeping the points' color in
            // step with the edges across the same w-depth scheme.
            let w_extent_local: f32 = if matches!(color_mode, WireframeColorMode::WDepth) {
                let canonical_max_w = topo
                    .vertices
                    .iter()
                    .map(|v| v.w.abs())
                    .fold(0.0_f32, f32::max)
                    .max(1e-6);
                canonical_max_w * body_size
            } else {
                1.0
            };
            // Cell strengths only computed for the Active mode; saves work in
            // the common case. Same definition the wireframe overlay uses
            // (`compute_cell_strengths`) so "vertex in an active cell" reads
            // consistently across edges + dots.
            let cell_strengths: Vec<f32> = if matches!(color_mode, WireframeColorMode::Active) {
                compute_cell_strengths(topo.cells, &local_vertices, w_slice)
            } else {
                Vec::new()
            };
            let vertex_is_active = |vi: usize| -> bool {
                topo.cells
                    .iter()
                    .zip(cell_strengths.iter())
                    .any(|(cell, &s)| s > 0.0 && cell.contains(&(vi as u32)))
            };

            if self.points_show_vertices {
                for (vi, v) in topo.vertices.iter().enumerate() {
                    let v_local = local_vertices[vi];
                    let v3_local =
                        <rye_math::EuclideanR4 as rye_math::RasterizableSpace<4>>::project_point(
                            v_local,
                            &wireframe_projection,
                        );
                    let v_world = v3_local + body_pos_r3;
                    let color = match color_mode {
                        // Position-gradient also covers UniqueEdge, since the
                        // unique-edge palette is edge-indexed (no canonical
                        // assignment for a vertex shared by several edges).
                        WireframeColorMode::VertexGradient | WireframeColorMode::UniqueEdge => {
                            vertex_color_by_position(*v)
                        }
                        WireframeColorMode::WDepth => w_depth_color(v_local.w, w_extent_local),
                        WireframeColorMode::Active => {
                            if vertex_is_active(vi) {
                                ACTIVE_GREEN
                            } else {
                                INACTIVE_GRAY
                            }
                        }
                    };
                    mesh.positions.push(v_world.to_array());
                    mesh.colors.push(color);
                    mesh.sizes.push(self.points_size_px);
                }
            }
            if self.points_show_cell_centers {
                // Pull cell centroids radially inward by `CELL_CENTER_INSET` so
                // they read as "interior markers" instead of coinciding with the
                // DUAL polytope's vertices. Mathematically `polytope.cell_centers()`
                // returns each cell's centroid at the inradius, which IS the
                // dual's vertex set (16-cell's cell-centroids form a tesseract,
                // tesseract's form a 16-cell, etc). At full inradius the sprites
                // look like a smaller dual polytope sitting at the inradius; the
                // inset puts them visibly INSIDE the body's cap so a viewer reads
                // "one dot per cell, in the cell's direction" rather than
                // "vertices of the wrong polytope."
                const CELL_CENTER_INSET: f32 = 0.5;
                let centers = polytope.cell_centers();
                for (ci, c) in centers.iter().enumerate() {
                    let c_local = body_size * CELL_CENTER_INSET * self.rot_state.apply(*c);
                    let c3_local =
                        <rye_math::EuclideanR4 as rye_math::RasterizableSpace<4>>::project_point(
                            c_local,
                            &wireframe_projection,
                        );
                    let c_world = c3_local + body_pos_r3;
                    let color = match color_mode {
                        WireframeColorMode::VertexGradient | WireframeColorMode::UniqueEdge => {
                            vertex_color_by_position(*c)
                        }
                        WireframeColorMode::WDepth => w_depth_color(c_local.w, w_extent_local),
                        WireframeColorMode::Active => {
                            let s = cell_strengths.get(ci).copied().unwrap_or(0.0);
                            if s > 0.0 {
                                ACTIVE_GREEN
                            } else {
                                INACTIVE_GRAY
                            }
                        }
                    };
                    mesh.positions.push(c_world.to_array());
                    mesh.colors.push(color);
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
        // No depth attachment: see `PointRasterNode::new` site for the rationale
        // (drop-w + ReadOnly LessEqual was occluding non-w=0 vertices behind their
        // own caps).
        self.points_node.execute(rd, view, None, None)?;
        Ok(())
    }

    fn render_section_faces(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        let cfg = &rd.surface_bundle.config;
        let n = self.row.len();

        // Reuse the per-Demo scratch mesh; capacity grows once to fit the largest
        // polychoron and stays there. Each frame's `clear()` keeps the underlying
        // allocations.
        let body_size = self.effective_body_size();
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
                    .map(|v| body_size * self.rot_state.apply(*v)),
            );

            // Match the SDF's per-body solid coloring: every cap of this polychoron uses
            // the body's identity color from the catalog. Per-face Lambert in the fragment
            // shader adds the geometric depth; the underlying color is flat. Alpha is
            // the user-tuneable `surface_alpha` (default 1.0); below 1.0 the wireframe
            // overlay behind composites through via `SrcAlpha/OneMinusSrcAlpha` blending.
            let [r, g, b] = entry.body_color;
            let start = combined.vertices.len();
            polytope_section_faces_append(
                topo.edges,
                topo.cells,
                &self.section_world_vertices_scratch,
                WPlane::new(self.w_slice),
                [r, g, b, self.surface_alpha],
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

        // Pick the depth-write variant based on `surface_alpha`: opaque
        // (1.0) writes depth so caps occlude one another correctly within
        // a polytope; translucent (< 1.0) skips depth-write so the parent
        // wireframe drawn after sees through. The two nodes carry their
        // own GPU buffers, so we upload the mesh into whichever path
        // we're about to execute.
        if self.surface_alpha >= 1.0 {
            self.section_faces.set_camera(&rd.queue, view_proj);
            self.section_faces.upload::<EuclideanR3, 3>(
                &rd.device,
                &rd.queue,
                combined,
                &rye_math::Projection::Identity,
            );
            self.section_faces
                .execute(rd, view, Some(&depth.view), None)?;
        } else {
            self.section_faces_translucent
                .set_camera(&rd.queue, view_proj);
            self.section_faces_translucent.upload::<EuclideanR3, 3>(
                &rd.device,
                &rd.queue,
                combined,
                &rye_math::Projection::Identity,
            );
            self.section_faces_translucent
                .execute(rd, view, Some(&depth.view), None)?;
        }
        Ok(())
    }

    /// Build the three overlay meshes (section triangles, section perimeter edges, parent
    /// wireframe) from the current row + rotor + w_slice, upload them, clear the overlay
    /// depth buffer, and execute the three raster passes on top of the existing SDF render.
    ///
    /// Per-body transform: each canonical Polytope4 vertex `v` becomes the world Vec4
    /// `body.position + effective_body_size() * rot_state.apply(v)`. The section algorithm
    /// then runs on these world vertices against the demo's `w_slice`, producing geometry
    /// in world R³ that composes cleanly with the SDF camera frame.
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
        // When `wireframe nearest-active` is OFF, every edge gets the user-
        // tuneable uniform alpha (`wireframe_alpha`, default 1.0). When ON,
        // edges interpolate between DIM (cells the slice misses entirely)
        // and BRIGHT (cells the slice is at the midpoint of). DIM/BRIGHT
        // stay hardcoded because they encode "very off" / "very on" peaks
        // of the activity gradient, not a global opacity setting.
        const PARENT_ALPHA_DIM: f32 = 0.10;
        const PARENT_ALPHA_BRIGHT: f32 = 0.85;
        let parent_alpha_uniform = self.wireframe_alpha;
        let parent_width = self.wireframe_width_px;
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
            let body_size = self.effective_body_size();
            let local_vertices: Vec<Vec4> = topo
                .vertices
                .iter()
                .map(|v| body_size * self.rot_state.apply(*v))
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

            // Per-cell "crossing strength" in [0, 1] - shared with render_points
            // via `compute_cell_strengths` so both passes agree on what "active"
            // means for a given w_slice + rotated polytope.
            let cell_strengths = compute_cell_strengths(topo.cells, &local_vertices, self.w_slice);

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
            // - `VertexGradient`: per-vertex position-derived RGB from the canonical
            //   vertex set so each vertex gets a distinct hue from its 4D coordinates
            //   and the polytope's symmetry shows as smooth gradients (same scheme as
            //   `Polytope4::lines_colored_by_position`).
            // - `UniqueEdge`: each edge gets a distinct palette color via greedy
            //   graph-coloring on the line graph (see `unique_edge_palette`).
            // - `WDepth`: per-endpoint cool-blue-to-warm-orange by SIGNED w
            //   in body-local frame, normalized against the polytope's canonical
            //   max |w| (fixed-band, not per-frame), so a vertex paints the same
            //   color at the same w regardless of rotor orientation.
            // - `Active`: binary green/gray by cell-activity (see `edge_is_active`).
            // Alpha is then modulated per-edge by the `nearest-active` strength (when
            // that toggle is on) or held uniform.
            // UniqueEdge palette is topology-only (greedy graph-coloring on the line
            // graph). Memoize by `Polytope4` variant so the 600-cell's ~520k pair-checks
            // run once per launch instead of once per frame. Cache lives on Demo; an
            // empty cache simply triggers first-use population for the visited variants.
            let edge_palette: &[[f32; 4]] = if matches!(color_mode, WireframeColorMode::UniqueEdge)
            {
                self.unique_edge_palette_cache
                    .entry(polytope)
                    .or_insert_with(|| unique_edge_palette(topo.edges))
            } else {
                &[]
            };
            // For `WDepth` we normalize against this polytope's CANONICAL max
            // |w| (NOT the per-frame rotated max). The rotor preserves
            // magnitudes, so the maximum |w| any rotated vertex can reach is
            // bounded by the canonical max times `body_size`. Holding the
            // gradient endpoints fixed across frames is what makes the color
            // cue temporally stable: a vertex migrating from -w to +w as
            // the rotor swings paints a continuous cool-to-warm shift rather
            // than a per-frame normalized re-saturation. Mirrors the
            // `LineRasterStaticR4` shader's hardcoded `[-0.5, +0.5]` band
            // (which is correct for the tesseract); here it's per-polytope
            // because the 16-cell, 5-cell, etc. don't share that band.
            let w_extent_local: f32 = if matches!(color_mode, WireframeColorMode::WDepth) {
                let canonical_max_w = topo
                    .vertices
                    .iter()
                    .map(|v| v.w.abs())
                    .fold(0.0_f32, f32::max)
                    .max(1e-6);
                canonical_max_w * body_size
            } else {
                1.0
            };
            for (edge_idx, &[i, j]) in topo.edges.iter().enumerate() {
                let ia = i as usize;
                let ja = j as usize;
                let a = local_vertices[ia];
                let b = local_vertices[ja];
                let (mut color_a, mut color_b) = match color_mode {
                    WireframeColorMode::VertexGradient => (
                        vertex_color_by_position(topo.vertices[ia]),
                        vertex_color_by_position(topo.vertices[ja]),
                    ),
                    WireframeColorMode::UniqueEdge => {
                        let c = edge_palette[edge_idx];
                        (c, c)
                    }
                    WireframeColorMode::WDepth => (
                        w_depth_color(a.w, w_extent_local),
                        w_depth_color(b.w, w_extent_local),
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
                    parent_alpha_uniform
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
                parent_lines.widths.push(parent_width);
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
    /// F3-toggle live perf overlay: FPS, frame-time, sparkline. Reads from
    /// `rye_time::frame_trace`, so it surfaces the same numbers `trace summary`
    /// dumps but continuously. Cheap when hidden (just a key-press check).
    perf: rye_app::trace::PerfOverlay,
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
                "polychoral surface mode: raster | sdf | off (bare = off); `scale <N>` to resize",
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
                    if matches!(args.first().copied(), Some("alpha")) {
                        match args.get(1).copied() {
                            None => {
                                out.line(format!(
                                    "surface alpha: {:.3} ({} pipeline)",
                                    demo.surface_alpha,
                                    if demo.surface_alpha >= 1.0 {
                                        "opaque"
                                    } else {
                                        "translucent"
                                    }
                                ));
                            }
                            Some(token) => {
                                let parsed: f32 = token.parse().map_err(|e| {
                                    anyhow!("invalid alpha `{token}`: {e}")
                                })?;
                                if !(0.05..=1.0).contains(&parsed) {
                                    return Err(anyhow!(
                                        "surface alpha {parsed} out of range; expected 0.05..=1.0 (use `surface off` for invisible)"
                                    ));
                                }
                                demo.surface_alpha = parsed;
                                out.line(format!("surface alpha: set to {parsed:.3}"));
                            }
                        }
                        return Ok(());
                    }
                    let next = match args.first().copied() {
                        Some(token) => SurfaceMode::from_token(token).ok_or_else(|| {
                            anyhow!("unknown arg `{token}` (try raster|sdf|off|scale|alpha)")
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
            .with_args(&[&["raster", "sdf", "off", "scale", "alpha"]])
            .with_long_help(
                "Selects how the six regular convex 4-polytopes (5-cell, tesseract, 16-cell,\n\
                 24-cell, 120-cell, 600-cell) are rendered, plus runtime scale + alpha knobs.\n\
                 \n\
                 subcommands:\n  \
                 raster      Rasterized cross-section cell-caps (the default). Face-normal\n                             Lambert lit, per-body solid color. Much faster for the\n                             120-cell + 600-cell and exact (no SDF approximation).\n  \
                 sdf         SDF raymarch. The historical pre-rasterizer path; smoother\n                             shading but the 120-cell and 600-cell carry a face-plane\n                             approximation BUG. Kept for visual comparison.\n  \
                 off         No surface rendered. Wireframe overlay + cross-section\n                             perimeter stay visible if enabled; the cap interiors are\n                             blank. Useful for inspecting the wireframe on its own.\n  \
                 scale <N>   Multiply the canonical body radius by N (default 1.0; range\n                             0.05..=10.0). Affects SDF kernel, raster cross-section caps,\n                             wireframe overlay, perimeter, and points sprites uniformly.\n  \
                 alpha <N>   Section-faces opacity (default 1.0; range 0.05..=1.0). Below\n                             1.0 the cap renders through a no-depth-write pipeline so\n                             the parent wireframe behind composes through. Use `surface\n                             off` for fully invisible caps.\n\
                 \n\
                 Bare `surface` (no argument) is shorthand for `surface off`.\n\
                 \n\
                 Smooth-surface shapes (Clifford torus, duocylinder, spherinder, 3-sphere)\n\
                 ignore the mode and always render via the SDF; they have no rasterizer\n\
                 path. Surface scale still applies to their SDF body radius.",
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
                            demo.orbit.set_orbit(9.5, -0.25);
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
mod color_tests {
    //! Unit tests for the pure color/topology helpers used by `render_wireframe_overlay`
    //! and `render_points`. These are load-bearing primitives behind every wireframe
    //! color mode; a sign flip or off-by-one would slip past clippy + the GPU-side
    //! kernel parse tests.
    use super::*;

    // ---- hsv_to_rgb ------------------------------------------------------

    /// HSV anchor cases against the standard reference. h=0 -> red, h=1/3 -> green,
    /// h=2/3 -> blue. Saturation 1, value 1, so output is the pure primary.
    #[test]
    fn hsv_to_rgb_primaries() {
        let red = hsv_to_rgb(0.0, 1.0, 1.0);
        assert!((red[0] - 1.0).abs() < 1e-5);
        assert!(red[1].abs() < 1e-5);
        assert!(red[2].abs() < 1e-5);

        let green = hsv_to_rgb(1.0 / 3.0, 1.0, 1.0);
        assert!(green[0].abs() < 1e-5);
        assert!((green[1] - 1.0).abs() < 1e-5);
        assert!(green[2].abs() < 1e-5);

        let blue = hsv_to_rgb(2.0 / 3.0, 1.0, 1.0);
        assert!(blue[0].abs() < 1e-5);
        assert!(blue[1].abs() < 1e-5);
        assert!((blue[2] - 1.0).abs() < 1e-5);
    }

    /// Zero saturation collapses to gray regardless of hue: r == g == b == value.
    #[test]
    fn hsv_to_rgb_zero_saturation_is_gray() {
        for h in [0.0, 0.25, 0.5, 0.75, 0.999_f32] {
            let rgb = hsv_to_rgb(h, 0.0, 0.7);
            assert!((rgb[0] - 0.7).abs() < 1e-5, "h={h}: r should be 0.7");
            assert!((rgb[1] - 0.7).abs() < 1e-5, "h={h}: g should be 0.7");
            assert!((rgb[2] - 0.7).abs() < 1e-5, "h={h}: b should be 0.7");
        }
    }

    /// Zero value collapses to black regardless of hue/saturation.
    #[test]
    fn hsv_to_rgb_zero_value_is_black() {
        let rgb = hsv_to_rgb(0.5, 0.8, 0.0);
        assert!(rgb.iter().all(|c| c.abs() < 1e-5));
    }

    // ---- unique_edge_palette --------------------------------------------

    /// Adjacent edges (sharing a vertex) get different palette colors. This is the
    /// defining invariant of the greedy graph-coloring on the line graph; if it
    /// fails, the whole point of the mode is broken.
    #[test]
    fn unique_edge_palette_separates_adjacent_edges() {
        // A simple line graph: three edges meeting at vertex 0.
        //     1
        //     |
        // 2 - 0 - 3
        let edges: &[[u32; 2]] = &[[0, 1], [0, 2], [0, 3]];
        let palette = unique_edge_palette(edges);
        assert_eq!(palette.len(), 3, "one color per edge");
        // All three edges share vertex 0, so all three must be distinct in the line graph.
        for i in 0..palette.len() {
            for j in (i + 1)..palette.len() {
                assert_ne!(
                    palette[i], palette[j],
                    "edges {i} and {j} share vertex 0; palette must differ"
                );
            }
        }
    }

    /// Edges with no shared vertex are NOT adjacent in the line graph and CAN end up
    /// sharing palette indices (greedy first-fit will reuse index 0 for both).
    #[test]
    fn unique_edge_palette_non_adjacent_edges_may_share_color() {
        // Two disconnected edges: (0,1) and (2,3). No shared vertex.
        let edges: &[[u32; 2]] = &[[0, 1], [2, 3]];
        let palette = unique_edge_palette(edges);
        assert_eq!(palette.len(), 2);
        // Greedy first-fit assigns index 0 to both since they're not adjacent.
        assert_eq!(palette[0], palette[1]);
    }

    /// Determinism: calling twice on the same edge slice yields identical output.
    /// Critical for caching by `Polytope4` variant (`unique_edge_palette_cache`).
    #[test]
    fn unique_edge_palette_is_deterministic() {
        let edges: &[[u32; 2]] = &[[0, 1], [1, 2], [2, 3], [0, 3]];
        let a = unique_edge_palette(edges);
        let b = unique_edge_palette(edges);
        assert_eq!(a, b);
    }

    // ---- w_depth_color --------------------------------------------------

    /// At w = 0 (the slice plane), the color sits exactly midway between back and front.
    /// `t = 0.5` means RGB is the literal midpoint of W_DEPTH_BACK and W_DEPTH_FRONT.
    #[test]
    fn w_depth_color_zero_w_is_midpoint() {
        let c = w_depth_color(0.0, 1.0);
        for ch in 0..3 {
            let expected = (W_DEPTH_BACK[ch] + W_DEPTH_FRONT[ch]) * 0.5;
            assert!(
                (c[ch] - expected).abs() < 1e-5,
                "channel {ch}: expected {expected}, got {}",
                c[ch],
            );
        }
        assert!((c[3] - 1.0).abs() < 1e-5, "alpha is 1.0");
    }

    /// At w = -extent the color is the back tint (cool blue).
    #[test]
    fn w_depth_color_neg_extent_is_back() {
        let c = w_depth_color(-1.0, 1.0);
        for ch in 0..3 {
            assert!(
                (c[ch] - W_DEPTH_BACK[ch]).abs() < 1e-5,
                "channel {ch}: expected back tint",
            );
        }
    }

    /// At w = +extent the color is the front tint (warm orange).
    #[test]
    fn w_depth_color_pos_extent_is_front() {
        let c = w_depth_color(1.0, 1.0);
        for ch in 0..3 {
            assert!(
                (c[ch] - W_DEPTH_FRONT[ch]).abs() < 1e-5,
                "channel {ch}: expected front tint",
            );
        }
    }

    /// Vertices past the extent (a polytope canonical max underestimate, or
    /// numeric drift) clamp to the endpoint colors rather than over-saturating.
    #[test]
    fn w_depth_color_clamps_past_extent() {
        let c_far = w_depth_color(5.0, 1.0);
        let c_at_extent = w_depth_color(1.0, 1.0);
        assert_eq!(c_far, c_at_extent, "+w past extent should clamp to front");

        let c_back = w_depth_color(-5.0, 1.0);
        let c_at_neg_extent = w_depth_color(-1.0, 1.0);
        assert_eq!(
            c_back, c_at_neg_extent,
            "-w past extent should clamp to back"
        );
    }

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
