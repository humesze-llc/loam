//! Demo state: the [`Demo`] struct, the mode/view/deferred-action enums, the
//! [`RotorTerm`] data type and its display helpers, the angular-velocity
//! derivation, body layout, and full reset.
//!
//! Per-mode UI rendering lives in `modes/{active,composer,filmstrip,shapes}.rs`
//! as additional `impl Demo` blocks; cross-cutting overlay UI lives in `ui.rs`.
//! All struct fields are `pub(crate)` so those sibling impls can access them.

use std::collections::HashMap;

use glam::{Vec3, Vec4};
use loam_app::{freecam::Freecam, Camera, OrbitController};
use loam_math::{Bivector, Bivector4, EuclideanR3, Plane4, Projection, Rotor, Rotor4};
use loam_physics::polytope::Polytope4;
use loam_render::raymarch::{BodyUniform, Hyperslice4DNode};

use crate::catalog::ShapeEntry;
use crate::consts::{BASE_ROTATION_RATE, BODY_SIZE, BODY_X_SPACING, BODY_Y, T_SLIDER_INITIAL};
use crate::physics::{BodyPose, PlaygroundPhysics};

// Projection modes live in `projections.rs`; re-export so `impl Demo`, the test
// module, and the other playground modules keep importing them from `state`.
pub(crate) use crate::projections::*;
pub(crate) use crate::sections::*;

// ---------------------------------------------------------------------------
// Mode + view enums
// ---------------------------------------------------------------------------

/// Continuous-rotation source driving `omega`, picked via the rotation tab row.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum RotationMode {
    /// Sum of unit bivectors of planes whose checkboxes are on (1..6 keys).
    Active,
    /// Sum of bivectors from the composed seq: each term contributes
    /// `scalar.unwrap_or(1.0) * sum_of_unit_bivectors`.
    Composer,
}

/// Visualisation mode (what the scene shows). Orthogonal to [`RotationMode`];
/// picked by a top-level tab row above the rotation tabs.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ViewMode {
    /// Multi-shape comparison: `self.row` of [`ShapeEntry`]s side-by-side at one
    /// common `w_slice`. Drag-and-drop rearranges the left-to-right layout.
    Shapes,
    /// Single-shape inspection: exactly the `strip_subject` (independent of the
    /// row) rendered at one `w_slice` through the Shapes render path over a
    /// one-element row ([`Demo::render_row`]). Stereographic and the
    /// cross-section read better on one body than across a mixed row.
    Single,
    /// Single-shape filmstrip: the `strip_subject` rendered N times across
    /// evenly-spaced `w_slice` values around the slider's current `w`. The row UI
    /// is hidden.
    Filmstrip,
}

/// The slice of [`ShapeEntry`]s the scene renders for `view_mode`.
/// [`ViewMode::Single`] yields exactly the `strip_subject` (no allocation);
/// every other mode renders the full `row`. Filmstrip draws `strip_subject`
/// through its own grid path and is not a caller here.
///
/// Free function so the row-selection invariant is unit-testable without a
/// GPU-backed [`Demo`]; [`Demo::render_row`] is the one caller.
pub(crate) fn render_row_entries<'a>(
    view_mode: ViewMode,
    row: &'a [ShapeEntry],
    strip_subject: &'a ShapeEntry,
) -> &'a [ShapeEntry] {
    match view_mode {
        ViewMode::Single => std::slice::from_ref(strip_subject),
        ViewMode::Shapes | ViewMode::Filmstrip => row,
    }
}

/// How the parent-wireframe edges are colored. Orthogonal to
/// [`Demo::wireframe_nearest_active`]: the color mode picks the hue, the
/// nearest-active toggle modulates alpha on top.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub(crate) enum WireframeColorMode {
    /// Per-vertex RGB from [`loam_physics::polytope::vertex_color_by_position`];
    /// each edge is a gradient between its endpoint hues (same scheme as
    /// `Polytope4::lines_colored_by_position`).
    #[default]
    VertexGradient,
    /// Distinct solid RGB per edge via greedy graph-coloring on the line graph,
    /// so edges sharing a vertex differ. Deterministic golden-ratio hue spacing.
    UniqueEdge,
    /// Per-vertex color by SIGNED body-local `w`: blue at `-w`, orange at `+w`,
    /// neutral at the slice. Normalized against canonical max `|w|` (a fixed band,
    /// NOT a per-frame rotated extent) so the gradient is temporally stable.
    /// Mirrors `LineRasterStaticR4`'s depth cue in `tesseract_demo`.
    WDepth,
    /// Binary green/gray by cell activity: edges of a cell the slice is currently
    /// intersecting are green, the rest gray. Complements the continuous
    /// `nearest-active` gradient (which shows how strongly each cell is crossed).
    Active,
}

impl WireframeColorMode {
    /// All four modes in console-cycle order.
    pub(crate) const ALL: [Self; 4] = [
        Self::VertexGradient,
        Self::UniqueEdge,
        Self::WDepth,
        Self::Active,
    ];

    /// Parse a console-arg spelling. `None` for unknown input.
    pub(crate) fn from_token(token: &str) -> Option<Self> {
        match token {
            "vertex-gradient" => Some(Self::VertexGradient),
            "unique-edge" => Some(Self::UniqueEdge),
            "w-depth" => Some(Self::WDepth),
            "active" => Some(Self::Active),
            _ => None,
        }
    }

    /// Display label for the egui radio buttons.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::VertexGradient => "Vertex gradient",
            Self::UniqueEdge => "Unique edge",
            Self::WDepth => "W-depth",
            Self::Active => "Active",
        }
    }
}

/// Camera control mode. `Orbit` (default) is the origin-focused
/// scroll-zoom/drag-to-rotate camera; `FreeRoam` flies via WASD + mouse-look.
/// Toggle via the `camera` console command. Switching to `Orbit` resets the
/// orbit controller to its default distance + pitch.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(crate) enum CameraMode {
    #[default]
    Orbit,
    FreeRoam,
}

// ---------------------------------------------------------------------------
// RotorTerm + display helpers
// ---------------------------------------------------------------------------

/// One term in the rotor-composition sequence: `exp(phi * sum_of_unit_bivectors)`
/// with an optional scalar `phi` in radians (`None` defaults to unit magnitude).
///
/// Bivector addition within a term is commutative (plane order inside a term is
/// irrelevant); rotor multiplication between terms is not (seq term order
/// matters).
#[derive(Clone, Debug, Default)]
pub(crate) struct RotorTerm {
    /// Unit-bivector planes summed inside `exp(...)`. An empty term is dropped.
    pub(crate) planes: Vec<Plane4>,
    /// Optional scalar prefix `phi` in radians. `None` is unit magnitude;
    /// `Some(phi)` scales the sum before `exp()`. "Add scalar" inits to
    /// `FRAC_PI_2`; `Default` is `None`.
    pub(crate) scalar: Option<f32>,
}

/// Render `(p_0 + p_1 + ...)` (parens iff multi-plane) into the current ui, each
/// plane through `render_plane`. Shared paren + `+` logic keeps a bivector sum
/// reading identically across all callsites.
pub(crate) fn render_plane_sum(
    ui: &mut loam_app::egui::Ui,
    planes: &[Plane4],
    mut render_plane: impl FnMut(&mut loam_app::egui::Ui, usize, Plane4),
) {
    let multi = planes.len() > 1;
    if multi {
        ui.monospace("(");
    }
    for (i, plane) in planes.iter().enumerate() {
        if i > 0 {
            ui.monospace("+");
        }
        render_plane(ui, i, *plane);
    }
    if multi {
        ui.monospace(")");
    }
}

/// Render a [`RotorTerm`] as the `scalar · bivec` form inside `exp(...)`.
/// Multi-plane terms get inner parens; an absent scalar is dropped.
pub(crate) fn render_term(term: &RotorTerm) -> String {
    let plane_str = term
        .planes
        .iter()
        .map(|p| p.label())
        .collect::<Vec<_>>()
        .join(" + ");
    let bivec = if term.planes.len() > 1 {
        format!("({plane_str})")
    } else {
        plane_str
    };
    match term.scalar {
        Some(phi) => format!("{:.0}° · {}", phi.to_degrees(), bivec),
        None => bivec,
    }
}

/// Wrap bivector-expression parts into one expression (paren-grouped when
/// multiple). `None` for an empty list so the caller can return early.
pub(crate) fn render_bivector_sum(parts: &[String]) -> Option<String> {
    match parts {
        [] => None,
        [only] => Some(only.clone()),
        many => Some(format!("({})", many.join(" + "))),
    }
}

/// Angular velocity from a composed seq: sum over terms of
/// `scalar * sum_of_unit_bivectors_in_term`, scaled by `rate_scale`. Term order
/// is irrelevant here (bivector addition commutes); it only matters for the
/// one-shot multiplicative `Apply`.
/// Displayed angle of one Active-mode plane at time `t`: the baseline plus the
/// spin `t * BASE_ROTATION_RATE` when active. Free function so the composition
/// is unit-testable without a GPU-backed `Demo`.
pub(crate) fn active_plane_angle(base: f32, active: bool, t: f32) -> f32 {
    base + if active { t * BASE_ROTATION_RATE } else { 0.0 }
}

/// Active-mode rotor at time `t`: the ORDERED PRODUCT
/// `∏ᵢ exp(planeᵢ · active_plane_angle(base[i], active[i], t))` in `Plane4::ALL`
/// order. A product (independent sliders), not `exp(sum)`, which would
/// reintroduce BCH coupling (see the `active.rs` module doc).
pub(crate) fn compose_active_rotor(base_angles: &[f32; 6], active: &[bool; 6], t: f32) -> Rotor4 {
    let mut r = Rotor4::IDENTITY;
    for i in 0..6 {
        let angle = active_plane_angle(base_angles[i], active[i], t);
        if angle != 0.0 {
            let bivec = Plane4::ALL[i].unit_bivector() * angle;
            r = bivec.exp() * r;
        }
    }
    r.normalize()
}

pub(crate) fn angular_velocity_from_seq(seq: &[RotorTerm], rate_scale: f32) -> Bivector4 {
    let mut omega = Bivector4::ZERO;
    for term in seq {
        let phi = term.scalar.unwrap_or(1.0);
        for plane in &term.planes {
            omega = omega + plane.unit_bivector() * phi;
        }
    }
    omega * (BASE_ROTATION_RATE * rate_scale)
}

// ---------------------------------------------------------------------------
// Deferred action queue
// ---------------------------------------------------------------------------

/// State mutations queued during overlay rendering and applied AFTER the
/// overlay's measure + visible passes. Anything that changes the overlay's
/// content height must defer; mutating mid-frame makes the two `BottomOverlay`
/// passes disagree on height and flicker.
#[derive(Clone, Debug)]
pub(crate) enum DeferredAction {
    /// `+xy` etc. button on the plane row: append to draft.
    DraftPush(Plane4),
    /// `Add` button on the draft preview: commit current draft as a new RotorTerm in
    /// seq, clear draft.
    SeqCommitDraft,
    /// `×` button on the draft preview: discard the draft.
    DraftClear,
    /// Typed-formula bar: push a fully-formed term to seq.
    SeqPushTerm(RotorTerm),
}

/// Drag-and-drop payload for the rotor sequence UI. One enum so a term card is a
/// single drop zone branching on the variant: `Term` reorders the seq, `Entry`
/// migrates a plane into this term.
#[derive(Clone, Copy, Debug)]
pub(crate) enum DragPayload {
    /// The whole term at this seq index is being dragged.
    Term(usize),
    /// `Entry(term_idx, plane_idx)`: a single plane pill from the given term is being
    /// dragged.
    Entry(usize, usize),
}

// ---------------------------------------------------------------------------
// Body layout helper
// ---------------------------------------------------------------------------

/// Spawn position of the `slot`-th of `n` bodies, centred on the world origin
/// and spaced by [`BODY_X_SPACING`]. The static layout only: once a body is
/// in a [`PlaygroundPhysics`] world its live position comes from there.
pub(crate) fn body_position(slot: usize, n: usize) -> [f32; 4] {
    let x = (slot as f32 - (n as f32 - 1.0) * 0.5) * BODY_X_SPACING;
    [x, BODY_Y, 0.0, 0.0]
}

/// SDF body uniform for one entry of a rendered row of `slots`, with polychora
/// opt-out per [`SurfaceMode`]: in Raster / Off the returned uniform is
/// `BodyUniform::default()` (kind = Invalid), which the kernel skips, and the
/// surface comes from the section-face raster (Raster) or nowhere (Off).
/// Smooth-surface shapes ignore the mode and always produce a live SDF body.
///
/// The raymarched body's centre and orientation come from `physics`, not from
/// `spin` over the static layout: the kernel and the raster passes read the
/// same pose, so the SDF surface and the wireframe around it cannot separate.
pub(crate) fn sdf_body_uniform(
    physics: &PlaygroundPhysics,
    entry: &ShapeEntry,
    slot: usize,
    slots: usize,
    spin: Rotor4,
    size: f32,
    surface_mode: SurfaceMode,
) -> BodyUniform {
    // The 120-cell and 600-cell have NO authoritative SDF: their
    // `cell{120,600}_face_planes` are the known-wrong dual-vertex
    // approximation (see `loam_shape::polytope_geom`), wrong on 96 normals.
    // Never raymarch them, on any platform or mode; the raster section +
    // wireframe paths are their correct surfaces. (`row_blocks_sdf` also
    // refuses to enter Sdf mode with them; this is the belt-and-suspenders.)
    if matches!(
        entry.shape.polytope4(),
        Some(Polytope4::Cell120 | Polytope4::Cell600)
    ) {
        return BodyUniform::default();
    }
    if !surface_mode.uses_sdf_for_polychora() && entry.shape.polytope4().is_some() {
        return BodyUniform::default();
    }
    let pose = physics.pose(slot, slots, spin);
    BodyUniform::polytope_with_rotor(
        pose.position.to_array(),
        entry.shape.shape_id(),
        size,
        pose.rotor,
        entry.body_color,
    )
}

// ---------------------------------------------------------------------------
// Rendered-row pose seam
// ---------------------------------------------------------------------------

/// One frame's rendered row (which shape sits in which slot, where the bodies
/// actually are, how 4D maps to R³) as its readers see it: the three raster
/// passes (section caps, wireframe overlay, point sprites) and the egui
/// overlay anchors. The SDF upload is the Shapes-view path that does NOT read
/// it; [`sdf_body_uniform`] takes the same poses from the same
/// [`PlaygroundPhysics`] directly. Filmstrip has no body behind it at all (see
/// [`crate::physics`]).
///
/// A value cannot exist without a [`PlaygroundPhysics`], and each reader takes
/// ALL of its per-body geometry from one, so no pass can quietly fall back to
/// the authored spin over the static layout while the others follow the thrown
/// bodies. [`Demo::row_frame`] is the only production constructor.
pub(crate) struct RowFrame<'a> {
    pub(crate) physics: &'a PlaygroundPhysics,
    /// The rendered row (see [`render_row_entries`]); its length is the slot
    /// count [`PlaygroundPhysics::pose`] checks the world against.
    pub(crate) row: &'a [ShapeEntry],
    /// UI spin, applied before each body's physics orientation.
    pub(crate) spin: Rotor4,
    pub(crate) body_size: f32,
    /// The live 4D -> R³ map ([`Demo::resolved_wireframe_projection`]).
    pub(crate) projection: Projection<4>,
    pub(crate) w_slice: f32,
    /// Eye-to-focus distance; only the stereographic clip radius reads it.
    pub(crate) camera_distance: f32,
}

impl RowFrame<'_> {
    /// Live pose of `slot`.
    pub(crate) fn pose(&self, slot: usize) -> BodyPose {
        self.physics.pose(slot, self.row.len(), self.spin)
    }

    /// `canonical` carried into `slot`'s live body frame at `scale` (refilling
    /// `out`), returning the R³ translate to apply AFTER projection. See
    /// [`PlaygroundPhysics::body_frame`].
    pub(crate) fn body_local(
        &self,
        slot: usize,
        canonical: &[Vec4],
        scale: f32,
        out: &mut Vec<Vec4>,
    ) -> Vec3 {
        self.physics
            .body_frame(slot, self.row.len(), self.spin, canonical, scale, out)
    }

    /// World-R³ anchor for one canonical point of `slot`: body frame, then
    /// projection, then translate. Same order as the raster passes, so a
    /// callout's leader line lands on the vertex the wireframe drew.
    pub(crate) fn anchor_r3(&self, slot: usize, canonical: Vec4) -> Vec3 {
        let pose = self.pose(slot);
        <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(
            pose.body_local(canonical, self.body_size),
            &self.projection,
        ) + pose.position_r3()
    }
}

// ---------------------------------------------------------------------------
// The App struct
// ---------------------------------------------------------------------------

pub(crate) struct Demo {
    pub(crate) space: EuclideanR3,
    /// Rigid-body state for the rendered row, one body per slot. Drives every
    /// render path's pose; see [`crate::physics`].
    pub(crate) physics: PlaygroundPhysics,
    pub(crate) camera: Camera<EuclideanR3>,
    pub(crate) orbit: OrbitController<EuclideanR3>,
    /// Freecam preset (mouse-look + WASD + cursor grab); drives the camera in
    /// `CameraMode::FreeRoam`. Owns its own yaw/pitch/position/grab state.
    pub(crate) freecam: Freecam,
    /// Active camera control mode (default `Orbit`).
    pub(crate) camera_mode: CameraMode,
    pub(crate) node: Hyperslice4DNode,
    /// Rasterizer node for the cross-section perimeter (cyan edges around each cap
    /// polygon). Filled caps are not drawn here; this only outlines the boundaries
    /// between adjacent cell contributions.
    pub(crate) section_edges: loam_render::LineRasterNode,
    /// Rasterizer node for the dim "parent wireframe" overlay: the full edge graph
    /// per body, projected via drop-w.
    pub(crate) parent_wireframe: loam_render::LineRasterNode,
    /// Whether the cross-section + parent-wireframe overlay renders. Off by
    /// default; toggle via `wireframe on|off`.
    pub(crate) wireframe_enabled: bool,
    /// When `true`, parent-wireframe edges are alpha-graded by how close `w_slice`
    /// is to the midpoint of each cell they belong to, so brightness propagates as
    /// a wave as the slice scrubs. When `false`, every edge uses a uniform alpha.
    pub(crate) wireframe_nearest_active: bool,
    /// The honest cross-section layer: the drop-w slice 3-flat, NEVER reprojected
    /// through [`Self::wireframe_projection`] (the same geometry the SDF raymarch
    /// shows), so a projection change never distorts the slice. On by default. See
    /// [`SectionLayer`].
    pub(crate) cross_section: SectionLayer,
    /// The projected-cap layer: the slice reprojected through the active
    /// [`Self::wireframe_projection`] so it can sit on a Schlegel / stereographic
    /// wireframe. Off by default; overlaid with [`Self::cross_section`]. See
    /// [`SectionLayer`].
    pub(crate) projected_cap: SectionLayer,
    /// Base RGB for wireframe edges. See [`WireframeColorMode`].
    pub(crate) wireframe_color_mode: WireframeColorMode,
    /// How the parent wireframe's 4D vertices project to R³ (the cross-section is
    /// always drop-w; this only affects the overlay).
    pub(crate) wireframe_projection: WireframeProjection,
    /// Cached canonical Schlegel parameters for the current `(polytope,
    /// cell_index)`; `Some` only while `wireframe_projection` is `Schlegel` over a
    /// polychoral row. Resolved at cell-select time via
    /// [`Demo::resolve_schlegel_cache`], never per frame:
    /// [`Polytope4::face_planes`] runs a `LazyLock` O(V·D³) fit that must stay off
    /// the hot path.
    pub(crate) schlegel_params: Option<SchlegelParams>,
    /// Live pole for the Stereographic projection (default
    /// [`STEREOGRAPHIC_DEFAULT_POLE`]). A field rather than baked into the
    /// payload-free [`WireframeProjection::Stereographic`] variant so the enum
    /// stays a plain marker; [`Self::resolved_wireframe_projection`] substitutes
    /// it per frame. Console-settable via `wireframe`.
    pub(crate) stereographic_pole: glam::Vec4,
    /// Wireframe Hyperslice toggle: when `true`, the parent wireframe is culled to
    /// edges of cells whose body-local w-range intersects a slab of width
    /// [`Self::wireframe_hyperslice_thickness`] around `w_slice`. Off by default. A
    /// CPU cell-level cull (so it agrees with the active-edge coloring and the
    /// cross-section), composable with the SDF and the cyan perimeter.
    pub(crate) wireframe_hyperslice: bool,
    /// Full width of the Hyperslice slab. An edge survives iff a cell containing
    /// both endpoints has a w-range intersecting `[w_slice - t/2, w_slice + t/2]`.
    /// Floored at [`crate::consts::HYPERSLICE_MIN_THICKNESS`] at the test site so
    /// `0` degrades to "cells straddling `w_slice`." Default
    /// [`crate::consts::HYPERSLICE_DEFAULT_THICKNESS`].
    pub(crate) wireframe_hyperslice_thickness: f32,
    /// Pixel width of parent-wireframe edges (default 1.8 px; finer reads poorly
    /// against the SDF backdrop on hi-DPI). Tuneable via `wireframe width <N>`.
    pub(crate) wireframe_width_px: f32,
    /// Uniform edge alpha when [`Self::wireframe_nearest_active`] is OFF (default
    /// 1.0); tuneable via `wireframe alpha <N>`. Ignored when `nearest_active` is
    /// ON, where alpha follows the per-cell crossing strength (0.10 to 0.85).
    pub(crate) wireframe_alpha: f32,
    /// Memoized per-edge palette for the `unique-edge` color mode, keyed by
    /// [`Polytope4`]. A function of topology alone (greedy line-graph coloring), so
    /// valid for the process lifetime once computed.
    pub(crate) unique_edge_palette_cache: HashMap<Polytope4, Vec<[f32; 4]>>,
    /// Memoized canonical cell centroids, keyed by [`Polytope4`]. Topology-only
    /// like [`Self::unique_edge_palette_cache`], so the 600-cell's 600 centroid
    /// folds run once per launch rather than once per body per frame.
    pub(crate) cell_centers_cache: HashMap<Polytope4, Vec<Vec4>>,
    /// Runtime multiplier on [`BODY_SIZE`] for all polychora (default 1.0, range
    /// `(0, 10]`; the bound preserves the SDF marcher's bounded-w assumption). Set
    /// via `surface scale <N>`; applies uniformly to wireframe, SDF, perimeter, and
    /// cap-fill geometry.
    pub(crate) surface_scale: f32,
    /// `y = 0` gridded floor visibility (default on; `floor` command). Gated at the
    /// kernel via `u.params[0]` so off is zero-cost (the halfspace SDF never
    /// converges and the checkerboard never paints).
    pub(crate) floor_enabled: bool,
    /// Filled-faces rasterizer for the polychoral cross-section. When raster mode
    /// is on it replaces the SDF for the six regular 4-polytopes (those SDF slots
    /// get `BodyUniform::default()`, which the kernel skips). Per-body solid color
    /// + face-normal Lambert.
    pub(crate) section_faces: loam_render::TriangleRasterNode,
    /// Translucent, depth-write-disabled variant of [`Self::section_faces`], used
    /// when a layer's `surface_alpha < 1.0` so the wireframe shows through caps.
    /// Same shaders + blend; only `DepthMode::ReadOnly` differs. Both section
    /// layers route through this pair, picking opaque vs translucent per layer.
    pub(crate) section_faces_translucent: loam_render::TriangleRasterNode,
    /// Antialiased point-disc rasterizer for vertex + cell-center sprites.
    /// Uploaded with the combined point mesh each frame the overlay is enabled.
    pub(crate) points_node: loam_render::PointRasterNode,
    /// Master toggle for the points overlay (off by default).
    pub(crate) points_enabled: bool,
    /// When [`Self::points_enabled`] is on, render a sprite at each vertex.
    pub(crate) points_show_vertices: bool,
    /// When [`Self::points_enabled`] is on, render a sprite at each cell centroid.
    /// Toggleable independently of vertices (600 sprites read as clutter on the
    /// 600-cell).
    pub(crate) points_show_cell_centers: bool,
    /// Screen-space radius (px) for both vertex and cell-center sprites.
    pub(crate) points_size_px: f32,
    /// Scratch buffer reused across frames + bodies inside `render_points`.
    pub(crate) points_mesh_scratch: loam_shape::PointMesh<3>,
    /// Shared depth attachment for the Shapes-view rasterizer chain, sized to the
    /// swapchain via [`loam_render::DepthBuffer::ensure`]. Cleared once per frame in
    /// [`crate::Demo::ensure_and_clear_shared_depth`]; `section_faces` writes it,
    /// `parent_wireframe` reads it for occlusion. In SDF mode the cleared `1.0`
    /// leaves every wireframe fragment passing, preserving the historical visual.
    pub(crate) section_faces_depth: Option<loam_render::DepthBuffer>,
    /// Scratch reused across frames + bodies inside `render_section_faces` to avoid
    /// per-body allocation on the 240 fps hot path.
    pub(crate) section_world_vertices_scratch: Vec<glam::Vec4>,
    /// Combined-mesh scratch for the honest drop-w cross-section layer.
    pub(crate) section_faces_mesh_scratch: loam_shape::TriangleMesh<3>,
    /// Combined-mesh scratch for the projected-cap layer, separate from
    /// [`Self::section_faces_mesh_scratch`] so both layers build in one pass over
    /// the row without clobbering each other's allocation.
    pub(crate) section_faces_projected_scratch: loam_shape::TriangleMesh<3>,
    /// Per-vertex body-local projected points for the cap-fill near-pole clip,
    /// reused inside `build_section_layer_meshes`. Lets the triangle-granularity
    /// Stereographic drop reuse the same `sample_in_radius` predicate the wireframe
    /// and cap perimeter use, keeping fill and outline culling in lockstep.
    pub(crate) section_clip_projected_scratch: Vec<glam::Vec3>,
    /// Reused buffer for per-frame body-uniform uploads (see
    /// `upload_render_row_bodies`).
    pub(crate) body_uniform_scratch: Vec<BodyUniform>,
    /// Reused great-circle sampling buffer for `push_blended_edge`, taken via
    /// `mem::take` so the Stereographic arc path does not allocate per frame.
    pub(crate) slerp_scratch: Vec<glam::Vec4>,
    /// Combined section-perimeter mesh for the wireframe overlay, reused across
    /// frames. Worst measured case is an eight-slot row of 600-cells with both
    /// perimeters on: ~12k segments, about 0.7 MB and a dozen grow-and-copy
    /// rounds if rebuilt from empty each frame. Unlike the parent mesh this
    /// does not scale with the projection, since the perimeter emits at most
    /// one segment per cap edge whatever the projection is.
    pub(crate) wireframe_section_edges_scratch: loam_shape::LineMesh<3>,
    /// One body's body-local section perimeter, refilled per body and consumed
    /// once per enabled section layer (the two layers project the same outline
    /// differently), then folded into
    /// [`Self::wireframe_section_edges_scratch`].
    pub(crate) body_perimeter_scratch: loam_shape::LineMesh<3>,
    /// Per-cell working set of the section core, handed to both
    /// `polytope_section_perimeter_append` and `polytope_section_faces_append`
    /// so the cap fit runs out of retained buffers instead of allocating per
    /// crossed cell. The two are called from different render passes, which is
    /// why `Demo::render_section_faces` running before
    /// `Demo::render_wireframe_overlay` is load-bearing and not incidental:
    /// each takes this buffer and restores it before the other runs.
    pub(crate) section_cap_scratch: loam_shape::polytope::SectionScratch,
    /// Combined parent-wireframe edge mesh, reused across frames. Separate from
    /// [`Self::wireframe_section_edges_scratch`] because both are built in one
    /// pass over the row and uploaded to different raster nodes.
    pub(crate) wireframe_parent_lines_scratch: loam_shape::LineMesh<3>,
    /// Body-local 4D vertex buffer shared by the wireframe and points builders
    /// (they run sequentially within a frame); refilled per body by
    /// `RowFrame::body_local`.
    pub(crate) overlay_local_vertices_scratch: Vec<glam::Vec4>,
    /// Second body-local buffer for the point builder's inset cell centres,
    /// which are live at the same time as [`Self::overlay_local_vertices_scratch`].
    pub(crate) overlay_center_locals_scratch: Vec<glam::Vec4>,
    /// Per-cell crossing strengths for the frame's current body, shared by the
    /// wireframe and points builders.
    pub(crate) overlay_cell_strengths_scratch: Vec<f32>,
    /// How the six regular convex 4-polytopes are rendered. Smooth-surface shapes
    /// (Clifford torus, duocylinder) ignore this and always use the SDF.
    pub(crate) surface_mode: SurfaceMode,
    /// Polytope row from the `shapes` argument (or `DEFAULT_ROW`); drives body
    /// uniforms and per-body label lookups.
    pub(crate) row: Vec<ShapeEntry>,

    pub(crate) w_slice: f32,
    pub(crate) slider_up_held: bool,
    pub(crate) slider_down_held: bool,
    pub(crate) slider_left_held: bool,
    pub(crate) slider_right_held: bool,

    pub(crate) rotate: bool,
    pub(crate) rot_state: Rotor4,
    /// Toggle bitmap for the six rotation planes; an active plane participates in
    /// the spin. See [`Plane4::ALL`] for the index -> plane mapping.
    pub(crate) active: [bool; 6],
    /// User-set baseline angle per plane in radians. Active mode treats plane i's
    /// displayed angle as `base_angles[i] + rot_time * RATE * active[i]` and
    /// composes `rot_state` as the ORDERED PRODUCT `∏ᵢ exp(planeᵢ · angle[i])`:
    /// each plane is its own factor, so a slider only mutates its own plane. Still
    /// BCH-coupled in the rotor; the UI is the source of truth instead of reading
    /// back through `log(rot_state)` (which has no faithful 6-plane decomposition).
    pub(crate) base_angles: [f32; 6],
    pub(crate) rate_scale: f32,
    /// Accumulated rotating time (advances only while `rotate`; resets on **R**).
    pub(crate) rot_time: f32,
    /// Upper bound on the `t` slider; doubles whenever `rot_time` exceeds it so the
    /// handle stays meaningful at long elapsed times. Reset on `R`.
    pub(crate) t_slider_max: f32,

    /// Whether the bottom controls overlay is expanded. `false` shows only the
    /// slider strip + rate row; `true` extends to the mode tabs, mode-specific UI,
    /// and shape row. Toggle via the chevron button or **H**.
    pub(crate) expanded: bool,

    /// Whether the "About / help" modal is open (the `?` button).
    pub(crate) show_help: bool,
    /// Whether the floating `Render` settings modal is open (off by default; the
    /// gear button). A discoverability aid for the console-driven render settings.
    pub(crate) show_render_panel: bool,
    /// Persistent state for the example annotation callout, anchored to the first
    /// polychoron's vertex 0. Off by default; `View > Example callout` or the
    /// `callout` command. Hosts the `loam_egui::callout` primitive.
    pub(crate) example_callout: loam_egui::CalloutState,

    /// Persistent state for the per-mode annotation callout: a short explanation of
    /// the active projection, anchored to the leading polychoron. Draws only when
    /// [`mode_annotation`] returns `Some` AND this flag is on (on by default).
    /// Toggle via `View > Mode annotation`.
    pub(crate) mode_annotation_open: loam_egui::CalloutState,

    /// Whether the top-right rotation-formula popup is rendered (off by default;
    /// checkbox in the expanded section).
    pub(crate) show_formula: bool,

    /// Whether the bottom controls overlay is rendered (on by default). Toggle via
    /// `View > Rotation controls` or `H` for an unobstructed scene.
    pub(crate) show_controls: bool,

    /// Top-level visualisation mode. See [`ViewMode`].
    pub(crate) view_mode: ViewMode,
    /// Filmstrip-axis toggles; at least one MUST be active when `view_mode ==
    /// Filmstrip` (the UI enforces this). `strip_w` alone fans across the w slider,
    /// `strip_t` alone across `rot_time`, both a 2D grid (axis assignment swappable
    /// via `strip_swap_axes`).
    pub(crate) strip_w: bool,
    pub(crate) strip_t: bool,
    /// With both `strip_w` and `strip_t` active, swap the default axis assignment
    /// (w-on-columns / t-on-rows becomes the reverse).
    pub(crate) strip_swap_axes: bool,
    /// Cell counts along each filmstrip axis. Range 3..=21.
    pub(crate) strip_count_w: usize,
    pub(crate) strip_count_t: usize,
    /// Forward extent of the t-axis fan in animation seconds.
    pub(crate) strip_t_extent: f32,
    /// Polytope rendered in each filmstrip cell. Independent of `self.row`.
    pub(crate) strip_subject: ShapeEntry,

    /// Which rotation source drives the continuous spin.
    pub(crate) rotation_mode: RotationMode,

    /// Mode change requested this frame by the mode tabs, applied after the overlay
    /// renders so this frame's body still sees the old `rotation_mode`.
    pub(crate) pending_mode: Option<RotationMode>,

    /// View change requested this frame by the view tab row.
    pub(crate) pending_view_mode: Option<ViewMode>,

    /// Composer-mode actions deferred to end-of-frame (as for `pending_mode`).
    pub(crate) pending_actions: Vec<DeferredAction>,

    /// Sequence of [`RotorTerm`]s the user is building in the panel.
    pub(crate) seq: Vec<RotorTerm>,
    /// In-progress draft for the next term; "Add" commits it to `seq` and clears.
    pub(crate) draft: Vec<Plane4>,

    /// Typed-formula input for the Composer's text bar.
    pub(crate) formula_input: String,
    /// Last parse error from the formula bar.
    pub(crate) formula_error: Option<String>,
}

// ---------------------------------------------------------------------------
// State methods
// ---------------------------------------------------------------------------

impl Demo {
    /// The composer seq's net bivector direction (unscaled): sum over terms of
    /// `scalar * sum_planes`. The scrub slider uses this as its rotation axis.
    pub(crate) fn compose_omega(&self) -> Bivector4 {
        let mut omega = Bivector4::ZERO;
        for term in &self.seq {
            let phi = term.scalar.unwrap_or(1.0);
            for plane in &term.planes {
                omega = omega + plane.unit_bivector() * phi;
            }
        }
        omega
    }

    /// Per-animation-second angular velocity (`rate_scale`-independent). Active
    /// mode sums the toggled basis bivectors; Composer delegates to the seq walker.
    /// Active composes its rotor as a *product* ([`Self::active_rotor`]), so this
    /// bivector is exact only for a single active plane (BCH-trivial); Composer's
    /// sum semantics make its omega exact.
    pub(crate) fn omega_animation(&self) -> Bivector4 {
        match self.rotation_mode {
            RotationMode::Active => {
                let mut omega = Bivector4::ZERO;
                for (i, &on) in self.active.iter().enumerate() {
                    if on {
                        omega = omega + Plane4::ALL[i].unit_bivector();
                    }
                }
                omega * BASE_ROTATION_RATE
            }
            RotationMode::Composer => angular_velocity_from_seq(&self.seq, 1.0),
        }
    }

    /// Active-mode angle for plane `i` at time `t`. Parameterized over `t` so the
    /// filmstrip can sample future times; [`Self::active_displayed_angle`] is the
    /// `t = rot_time` specialization the sliders read.
    pub(crate) fn active_angle_at(&self, plane_idx: usize, t: f32) -> f32 {
        active_plane_angle(self.base_angles[plane_idx], self.active[plane_idx], t)
    }

    /// Displayed Active-mode angle for plane `i` at the current `rot_time`. The
    /// slider reads this; writing it sets `base_angles[i]` so a drag on a spinning
    /// slider does not snap back to the pre-drag baseline.
    pub(crate) fn active_displayed_angle(&self, plane_idx: usize) -> f32 {
        self.active_angle_at(plane_idx, self.rot_time)
    }

    /// Active-mode rotor at time `t`: ORDERED PRODUCT of per-plane simple rotations
    /// in `Plane4::ALL` order. Each `base_angles[i]` is one factor, so sliders are
    /// independent. See [`compose_active_rotor`].
    pub(crate) fn active_rotor_at(&self, t: f32) -> Rotor4 {
        compose_active_rotor(&self.base_angles, &self.active, t)
    }

    /// Active-mode rotor at the current `rot_time` (specializes
    /// [`Self::active_rotor_at`]).
    pub(crate) fn active_rotor(&self) -> Rotor4 {
        self.active_rotor_at(self.rot_time)
    }

    /// Orientation rotor at time `t`, dispatched on the rotation mode. The single
    /// source of truth every t-scrub and filmstrip-offset site MUST use so they
    /// agree with the spin path:
    ///
    /// - **Active**: product-of-exp via [`Self::active_rotor_at`] (summing would
    ///   reintroduce BCH coupling).
    /// - **Composer**: `exp(omega_animation * t)`.
    ///
    /// The two coincide for a single active plane; they diverge only with two or
    /// more non-commuting active planes.
    pub(crate) fn rotor_at_time(&self, t: f32) -> Rotor4 {
        match self.rotation_mode {
            RotationMode::Active => self.active_rotor_at(t),
            RotationMode::Composer => (self.omega_animation() * t).exp().normalize(),
        }
    }

    /// Effective body radius after the [`Self::surface_scale`] multiplier. All
    /// `BODY_SIZE` consumers route through here so `surface scale` applies
    /// uniformly.
    pub(crate) fn effective_body_size(&self) -> f32 {
        BODY_SIZE * self.surface_scale
    }

    /// True when entering `SurfaceMode::Sdf` would put a 120-cell or 600-cell into
    /// the live SDF kernel (which crashes the browser tab; see [`row_blocks_sdf`]).
    /// The `surface sdf` command and the UI radio gate on this.
    ///
    /// Tests the RENDERED row (see [`Self::render_row`]), not `self.row`: in
    /// [`ViewMode::Single`] only the `strip_subject` uploads, so gating on the
    /// stored row would both falsely block a light subject and falsely ALLOW a
    /// heavy subject over a light row.
    pub(crate) fn sdf_blocked_by_heavy_polychora(&self) -> bool {
        row_blocks_sdf(self.render_row())
    }

    /// Effective `w` slider half-range after [`Self::surface_scale`]. Scaling the
    /// polytope up scales [`crate::consts::W_RANGE`] too so the slice still clears
    /// the body's hull at the slider extremes.
    pub(crate) fn effective_w_range(&self) -> f32 {
        crate::consts::W_RANGE * self.surface_scale
    }

    /// The slice of [`ShapeEntry`]s the scene renders this frame: the
    /// `strip_subject` in [`ViewMode::Single`], else the full `row`. Every per-body
    /// render path and the SDF upload read this. See [`render_row_entries`].
    pub(crate) fn render_row(&self) -> &[ShapeEntry] {
        render_row_entries(self.view_mode, &self.row, &self.strip_subject)
    }

    /// This frame's [`RowFrame`]: the seam its readers take a body pose
    /// through. Rebuilt per pass rather than cached; the arithmetic is
    /// [`Self::effective_body_size`]'s multiply,
    /// [`Self::camera_distance_to_focus`]'s subtract and square root, and,
    /// under Schlegel, [`Self::resolved_wireframe_projection`]'s rotated normal
    /// and basis plus its two body-size scalings.
    pub(crate) fn row_frame(&self) -> RowFrame<'_> {
        RowFrame {
            physics: &self.physics,
            row: self.render_row(),
            spin: self.rot_state,
            body_size: self.effective_body_size(),
            projection: self.resolved_wireframe_projection(),
            w_slice: self.w_slice,
            camera_distance: self.camera_distance_to_focus(),
        }
    }

    /// The polytope a Schlegel cell index refers to: the first polychoron in the
    /// rendered row (a cell index is ambiguous across a mixed row). In
    /// [`ViewMode::Single`] this is the `strip_subject`.
    pub(crate) fn schlegel_subject(&self) -> Option<Polytope4> {
        self.render_row().iter().find_map(|e| e.shape.polytope4())
    }

    /// Resolve and cache the canonical Schlegel parameters. Call at every
    /// cell-SELECT point (console, UI stepper, switching the radio to Schlegel, any
    /// row edit changing the leading polychoron) so the per-frame path never reruns
    /// the `LazyLock`-backed [`Polytope4::face_planes`] fit. Idempotent (the fit is
    /// deterministic); clears the cache when the mode is not Schlegel or the row
    /// has no polychoron. Writes the clamped `cell_index` back into the projection
    /// so the enum, cache, UI, and console agree. See [`synced_schlegel_projection`].
    pub(crate) fn resolve_schlegel_cache(&mut self) {
        let (projection, cache) =
            synced_schlegel_projection(self.wireframe_projection, self.schlegel_subject());
        self.wireframe_projection = projection;
        self.schlegel_params = cache;
    }

    /// The live [`loam_math::Projection<4>`] for the wireframe overlay this frame.
    /// For Schlegel it builds the engine projection from the cached
    /// [`SchlegelParams`], rotating the normal/basis by `rot_state` (so the chosen
    /// cell stays the outer boundary) and scaling the offsets by
    /// [`Self::effective_body_size`]. No allocation, no `face_planes` call: the
    /// hot-path-safe counterpart to [`Self::resolve_schlegel_cache`]. Other modes
    /// delegate to [`WireframeProjection::to_projection`].
    pub(crate) fn resolved_wireframe_projection(&self) -> Projection<4> {
        match self.wireframe_projection {
            WireframeProjection::Schlegel { .. } => match self.schlegel_params {
                Some(p) => {
                    let body_size = self.effective_body_size();
                    let cell_normal = self.rot_state.apply(p.cell_normal);
                    let basis = p.cell_basis.map(|axis| self.rot_state.apply(axis));
                    Projection::schlegel_with_basis(
                        cell_normal,
                        p.cell_offset * body_size,
                        p.viewpoint_distance * body_size,
                        basis,
                    )
                }
                // No polychoron in row: drop-w fallback (the wireframe draws
                // nothing anyway).
                None => Projection::Identity,
            },
            // Substitute the live pole (`to_projection` returns the default).
            WireframeProjection::Stereographic => Projection::Stereographic {
                pole: self.stereographic_pole,
            },
            other => other.to_projection(),
        }
    }

    /// Whether the wireframe Hyperslice cull should run this frame. See the free
    /// [`hyperslice_cull_active`].
    pub(crate) fn hyperslice_cull_active(&self) -> bool {
        hyperslice_cull_active(self.wireframe_hyperslice, self.wireframe_projection)
    }

    /// Drive every body in the RENDERED row (see [`Self::render_row`]) with the
    /// same rotor. Uploads via `set_bodies` (not slot-wise) so a stale row from a
    /// previous mode cannot keep rendering.
    pub(crate) fn write_all(&mut self, rotor: Rotor4) {
        self.upload_render_row_bodies(rotor);
    }

    /// Re-emit every rendered body's uniform. Called after row mutations, rotor
    /// changes, view-mode changes, and surface-mode changes.
    pub(crate) fn rebuild_bodies(&mut self) {
        self.upload_render_row_bodies(self.rot_state);
    }

    /// Build each rendered body's SDF uniform and upload via `set_bodies` (which
    /// also sets the kernel's `body_count`). Shared by [`Self::write_all`] and
    /// [`Self::rebuild_bodies`].
    ///
    /// `body_uniform_scratch` is taken out of `self` for the build (so it can
    /// borrow `&self`) and put back, keeping its capacity so the steady-state spin
    /// upload does not allocate.
    ///
    /// The single choke point where the rendered row's slot count is
    /// materialized, so it is also where the physics world is reconciled with
    /// it ([`PlaygroundPhysics::sync`]).
    fn upload_render_row_bodies(&mut self, rotor: Rotor4) {
        let n = self.render_row().len();
        let size = self.effective_body_size();
        self.physics.sync(n, size);
        let mut scratch = std::mem::take(&mut self.body_uniform_scratch);
        scratch.clear();
        for slot in 0..n {
            let entry = &self.render_row()[slot];
            scratch.push(sdf_body_uniform(
                &self.physics,
                entry,
                slot,
                n,
                rotor,
                size,
                self.surface_mode,
            ));
        }
        self.node.set_bodies(&scratch);
        self.body_uniform_scratch = scratch;
    }

    /// Compact `exp(B · 0.30·t)` form for whichever mode drives the spin, where `B`
    /// is the bivector velocity (Active: enabled-plane terms; Composer: seq terms).
    /// Empty string when nothing contributes.
    pub(crate) fn formula_string(&self) -> String {
        let parts: Vec<String> = match self.rotation_mode {
            RotationMode::Active => Plane4::ALL
                .iter()
                .zip(self.active.iter())
                .filter(|(_, on)| **on)
                .map(|(p, _)| p.label().to_string())
                .collect(),
            RotationMode::Composer => self.seq.iter().map(render_term).collect(),
        };
        match render_bivector_sum(&parts) {
            Some(bivec) => format!(
                "exp({} · {:.2}·t)",
                bivec,
                BASE_ROTATION_RATE / std::f32::consts::TAU
            ),
            None => String::new(),
        }
    }

    /// Full reset: pause spin, slice, rate, active set, orientation, time,
    /// draft, and thrown bodies.
    /// `rotate` flips off too so the next `update()` does not immediately respin.
    pub(crate) fn reset(&mut self) {
        self.rotate = false;
        self.w_slice = 0.0;
        self.rate_scale = 1.0;
        // xw-only default so a first-time user who resets then spins sees motion.
        self.active = [false, false, true, false, false, false];
        self.base_angles = [0.0; 6];
        self.rot_state = Rotor4::IDENTITY;
        self.rot_time = 0.0;
        self.t_slider_max = T_SLIDER_INITIAL;
        // Honest-slice baseline: drop-w cross-section on, reprojected cap off.
        self.cross_section = SectionLayer::CROSS_SECTION_DEFAULT;
        self.projected_cap = SectionLayer::PROJECTED_CAP_DEFAULT;
        self.draft.clear();
        let slots = self.render_row().len();
        self.physics.respawn(slots, self.effective_body_size());
        self.write_all(Rotor4::IDENTITY);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        active_plane_angle, apply_projection_selection_defaults, body_position,
        compose_active_rotor, default_edge_blend, hyperslice_cull_active, mode_annotation,
        render_row_entries, resolve_schlegel_params, row_blocks_sdf, sdf_body_uniform,
        section_layer_projection, synced_schlegel_projection, SectionLayer, SurfaceMode, ViewMode,
        WireframeProjection, BASE_ROTATION_RATE, BODY_SIZE, STEREOGRAPHIC_DEFAULT_POLE,
    };
    use crate::catalog::ShapeEntry;
    use crate::physics::{composed_rotor, PlaygroundPhysics};
    use glam::Vec4;
    use loam_math::{Bivector, EuclideanR4, Plane4, Projection, Rotor, Rotor4};
    use loam_physics::polytope::Polytope4;
    use loam_render::raymarch::{BodyUniform, RaymarchShape};
    use std::collections::HashSet;

    fn entry(shape: RaymarchShape) -> ShapeEntry {
        ShapeEntry {
            shape,
            body_color: [0.5, 0.5, 0.5],
            label: "test",
            long_name: "test shape",
        }
    }

    const NONE: [bool; 6] = [false; 6];

    fn rotor_close(a: Rotor4, b: Rotor4, eps: f32) -> bool {
        (a.s - b.s).abs() < eps
            && (a.xy - b.xy).abs() < eps
            && (a.xz - b.xz).abs() < eps
            && (a.xw - b.xw).abs() < eps
            && (a.yz - b.yz).abs() < eps
            && (a.yw - b.yw).abs() < eps
            && (a.zw - b.zw).abs() < eps
            && (a.xyzw - b.xyzw).abs() < eps
    }

    #[test]
    fn active_plane_angle_adds_spin_only_when_active() {
        // Inactive plane: angle is the baseline regardless of t.
        assert_eq!(active_plane_angle(0.5, false, 3.0), 0.5);
        // Active plane: baseline plus t * rate.
        assert_eq!(
            active_plane_angle(0.5, true, 2.0),
            0.5 + 2.0 * BASE_ROTATION_RATE
        );
        // t = 0 collapses to the baseline even when active.
        assert_eq!(active_plane_angle(0.5, true, 0.0), 0.5);
    }

    #[test]
    fn toggling_active_preserves_displayed_angle() {
        // Flipping `active` re-solves `base` so the displayed angle is continuous
        // across the toggle: `base = displayed_before - spin(active_after)`.
        let t = 7.5_f32;
        let resolve = |base_old: f32, active_before: bool| {
            let displayed_before = active_plane_angle(base_old, active_before, t);
            let active_after = !active_before;
            let spin_after = if active_after {
                t * BASE_ROTATION_RATE
            } else {
                0.0
            };
            let base_new = displayed_before - spin_after;
            // The new displayed angle must equal the pre-toggle one.
            active_plane_angle(base_new, active_after, t)
        };
        // Switching ON: the inactive baseline must survive unchanged.
        let base_off = 0.42_f32;
        let displayed_off = active_plane_angle(base_off, false, t);
        assert!(
            (resolve(base_off, false) - displayed_off).abs() < 1e-6,
            "toggle on changed the displayed angle"
        );
        // Switching OFF: the accumulated spin must freeze into the baseline.
        let base_on = -1.1_f32;
        let displayed_on = active_plane_angle(base_on, true, t);
        assert!(
            (resolve(base_on, true) - displayed_on).abs() < 1e-6,
            "toggle off changed the displayed angle"
        );
    }

    #[test]
    fn compose_all_zero_is_identity() {
        let r = compose_active_rotor(&[0.0; 6], &NONE, 0.0);
        assert!(rotor_close(r, Rotor4::IDENTITY, 1e-6), "got {r:?}");
    }

    #[test]
    fn compose_is_always_unit_norm() {
        // A messy multi-plane configuration must still produce a unit rotor
        // (the function normalizes). Norm-squared within 1e-5 of 1.
        let base = [0.3, -1.1, 2.0, 0.7, -0.4, 1.6];
        let active = [true, false, true, true, false, true];
        for &t in &[0.0_f32, 0.5, 3.0, 50.0] {
            let n2 = compose_active_rotor(&base, &active, t).norm_squared();
            assert!((n2 - 1.0).abs() < 1e-5, "t={t} norm_squared={n2}");
        }
    }

    #[test]
    fn compose_single_plane_equals_direct_exp() {
        // One plane, no spin: the product collapses to exp(plane * angle)
        // (the BCH-trivial case).
        let theta = 0.8_f32;
        let mut base = [0.0; 6];
        base[2] = theta; // Plane4::Xw
        let composed = compose_active_rotor(&base, &NONE, 0.0);
        let direct = (Plane4::Xw.unit_bivector() * theta).exp().normalize();
        assert!(
            rotor_close(composed, direct, 1e-6),
            "{composed:?} vs {direct:?}"
        );
    }

    #[test]
    fn compose_orthogonal_pair_is_order_independent() {
        // xy and zw are absolutely orthogonal, so their bivectors commute and the
        // product equals the reverse-order product.
        let (a, b) = (0.6_f32, -0.9_f32);
        let mut base = [0.0; 6];
        base[0] = a; // xy
        base[5] = b; // zw
        let composed = compose_active_rotor(&base, &NONE, 0.0);
        let xy = (Plane4::Xy.unit_bivector() * a).exp();
        let zw = (Plane4::Zw.unit_bivector() * b).exp();
        let reverse = (xy * zw).normalize();
        assert!(
            rotor_close(composed, reverse, 1e-6),
            "{composed:?} vs {reverse:?}"
        );
    }

    #[test]
    fn row_blocks_sdf_only_for_heavy_polychora() {
        // Empty row: nothing to block.
        assert!(!row_blocks_sdf(&[]));
        // Lighter shapes (default-row members): SDF stays available.
        let light = [
            entry(RaymarchShape::Polytope(Polytope4::Tesseract)),
            entry(RaymarchShape::Polytope(Polytope4::Cell24)),
        ];
        assert!(!row_blocks_sdf(&light));
        // A 120-cell anywhere in the row blocks SDF.
        let with_120 = [
            entry(RaymarchShape::Polytope(Polytope4::Tesseract)),
            entry(RaymarchShape::Polytope(Polytope4::Cell120)),
        ];
        assert!(row_blocks_sdf(&with_120));
        // A 600-cell does too.
        let with_600 = [entry(RaymarchShape::Polytope(Polytope4::Cell600))];
        assert!(row_blocks_sdf(&with_600));
    }

    /// drop-w yields no annotation; every other projection yields a non-empty
    /// title and body, and the bodies are pairwise distinct.
    #[test]
    fn annotation_text_present_per_mode() {
        assert!(
            mode_annotation(WireframeProjection::Shadow).is_none(),
            "drop-w must not annotate the default view"
        );

        let cases = [
            WireframeProjection::WPinhole,
            WireframeProjection::Schlegel { cell_index: 0 },
            WireframeProjection::Stereographic,
            WireframeProjection::Hyperslice,
        ];
        let mut bodies = HashSet::new();
        for projection in cases {
            let annotation = mode_annotation(projection)
                .unwrap_or_else(|| panic!("{projection:?} must annotate"));
            assert!(
                !annotation.title.is_empty(),
                "{projection:?}: title must be non-empty"
            );
            assert!(
                !annotation.body.is_empty(),
                "{projection:?}: body must be non-empty"
            );
            assert!(
                bodies.insert(annotation.body.clone()),
                "{projection:?}: body duplicates another mode's copy"
            );
        }
    }

    /// The SDF crash-safety gate keys off the RENDERED row, not `self.row`: in
    /// [`ViewMode::Single`] a heavy subject over a light row must BLOCK and a light
    /// subject under a heavy row must NOT, while Shapes keeps the row-wide gate.
    #[test]
    fn sdf_gate_follows_single_subject_not_row() {
        let heavy = entry(RaymarchShape::Polytope(Polytope4::Cell600));
        let light = entry(RaymarchShape::Polytope(Polytope4::Tesseract));

        // Heavy subject, all-light row: blocked, because Single renders the
        // 600-cell. Reading the row alone here would WRONGLY allow SDF.
        let light_row = [light, entry(RaymarchShape::Polytope(Polytope4::Cell24))];
        assert!(
            row_blocks_sdf(render_row_entries(ViewMode::Single, &light_row, &heavy)),
            "heavy Single subject blocks SDF even over an all-light row",
        );
        // Light subject, heavy row: NOT blocked, because Single renders the
        // tesseract. Reading the row alone here would WRONGLY block SDF.
        let heavy_row = [entry(RaymarchShape::Polytope(Polytope4::Cell120))];
        assert!(
            !row_blocks_sdf(render_row_entries(ViewMode::Single, &heavy_row, &light)),
            "light Single subject keeps SDF available even over a heavy row",
        );
        // Shapes mode keeps the row-wide gate: the same heavy row blocks SDF
        // regardless of the (unrendered, in this mode) subject.
        assert!(
            row_blocks_sdf(render_row_entries(ViewMode::Shapes, &heavy_row, &light)),
            "Shapes mode blocks SDF on a heavy row member",
        );
    }

    /// Every selectable mode's token round-trips through `from_token`, and
    /// `to_projection` produces the matching engine variant (or the `Identity`
    /// fallback for Hyperslice / Schlegel). Schlegel is excluded from `ALL` /
    /// `from_token` (see `WireframeProjection::Schlegel`).
    #[test]
    fn wireframe_projection_from_token_round_trips() {
        // Pin the count so an enum addition that skips `ALL`, or a Schlegel
        // re-wire, is loud.
        assert_eq!(
            WireframeProjection::ALL.len(),
            4,
            "ALL must list every selectable projection mode (Schlegel is excluded)"
        );
        // Schlegel is intentionally not parseable from the console.
        assert_eq!(
            WireframeProjection::from_token("schlegel"),
            None,
            "Schlegel is deliberately not a selectable playground mode"
        );
        for mode in WireframeProjection::ALL {
            let token = match mode {
                WireframeProjection::Shadow => "shadow",
                WireframeProjection::WPinhole => "w-pinhole",
                WireframeProjection::Stereographic => "stereographic",
                WireframeProjection::Hyperslice => "hyperslice",
                // Excluded from ALL; if a re-wire puts it back, this fires loudly.
                WireframeProjection::Schlegel { .. } => {
                    unreachable!("Schlegel must not be in ALL while it is unwired")
                }
            };
            assert_eq!(
                WireframeProjection::from_token(token),
                Some(mode),
                "token `{token}` must parse back to {mode:?}"
            );
        }
        // Context-free engine projections per the documented contract.
        assert_eq!(
            WireframeProjection::Shadow.to_projection(),
            Projection::Identity
        );
        assert_eq!(
            WireframeProjection::WPinhole.to_projection(),
            Projection::Perspective4D {
                focal_distance: 2.0
            }
        );
        assert_eq!(
            WireframeProjection::Stereographic.to_projection(),
            Projection::Stereographic {
                pole: STEREOGRAPHIC_DEFAULT_POLE
            }
        );
        // Hyperslice: drop-w projection, the cull does the slicing.
        assert_eq!(
            WireframeProjection::Hyperslice.to_projection(),
            Projection::Identity
        );
        // Schlegel context-free fallback is Identity; the real Schlegel comes
        // from `Demo::resolved_wireframe_projection` with the cached params.
        assert_eq!(
            WireframeProjection::Schlegel { cell_index: 0 }.to_projection(),
            Projection::Identity
        );
    }

    /// Edge geometry is derived from the projection: Stereographic renders S3 arcs
    /// (`blend == 1`), every affine projection renders chords (`blend == 0`).
    #[test]
    fn edge_geometry_is_derived_from_projection() {
        assert_eq!(default_edge_blend(WireframeProjection::Stereographic), 1.0);
        for p in [
            WireframeProjection::Shadow,
            WireframeProjection::WPinhole,
            WireframeProjection::Hyperslice,
        ] {
            assert_eq!(default_edge_blend(p), 0.0, "{p:?} should draw chords");
        }
    }

    /// Selecting Stereographic turns the wireframe overlay on (its arcs are drawn
    /// only there); every other projection leaves the toggle alone.
    #[test]
    fn stereographic_selection_enables_wireframe() {
        let mut wireframe = false;
        apply_projection_selection_defaults(WireframeProjection::Stereographic, &mut wireframe);
        assert!(
            wireframe,
            "stereographic arcs require the wireframe overlay"
        );

        let mut wireframe = false;
        apply_projection_selection_defaults(WireframeProjection::WPinhole, &mut wireframe);
        assert!(
            !wireframe,
            "non-stereographic selection must not force the overlay"
        );
    }

    /// The default Stereographic pole is the `+w` axis, exactly unit. See
    /// [`stereographic_plus_w_pole_scale_depends_only_on_w`].
    #[test]
    fn stereographic_default_pole_is_plus_w() {
        assert_eq!(STEREOGRAPHIC_DEFAULT_POLE, Vec4::W);
        assert_eq!(
            STEREOGRAPHIC_DEFAULT_POLE.length_squared(),
            1.0,
            "default pole must be exactly unit"
        );
    }

    /// The centering property: under the `+w` pole the conformal scale
    /// `1 / (1 - dot(p, pole))` reduces to `1 / (1 - p.w)`, so points at equal `w`
    /// share a denominator; an off-w pole (scale depends on `x + y + z + w`) does
    /// not.
    #[test]
    fn stereographic_plus_w_pole_scale_depends_only_on_w() {
        let pole = STEREOGRAPHIC_DEFAULT_POLE;
        // Two unit directions at w = 0.6 with different x + y + z, so an off-w
        // pole tells them apart while +w cannot.
        let p = Vec4::new(0.8, 0.0, 0.0, 0.6);
        let q = Vec4::new(0.0, 0.48, 0.64, 0.6);
        assert!((p.w - q.w).abs() < 1e-6, "fixture points must share w");
        let denom_p = 1.0 - p.dot(pole);
        let denom_q = 1.0 - q.dot(pole);
        assert!(
            (denom_p - denom_q).abs() < 1e-6,
            "+w pole must give equal scale at equal w ({denom_p} vs {denom_q})"
        );
        // An off-w pole (a cell center) breaks the equal-w invariant, which is
        // exactly the off-center pull the +w pole was reverted to avoid.
        let off_w = Vec4::splat(0.5);
        assert!(
            (1.0 - p.dot(off_w) - (1.0 - q.dot(off_w))).abs() > 1e-3,
            "off-w pole must spread a single w-slice (the skew we reverted)"
        );
    }

    /// The accepted tradeoff: `+w` IS a 16-cell vertex (`+e_w`), so a vertex can
    /// sweep through the pole and flick to infinity. Pinned as deliberate.
    #[test]
    fn stereographic_default_pole_is_a_16cell_vertex_by_design() {
        let pole = STEREOGRAPHIC_DEFAULT_POLE;
        let on_a_vertex = Polytope4::Cell16
            .topology()
            .vertices
            .iter()
            .any(|v| (v.normalize() - pole).length() < 1e-6);
        assert!(
            on_a_vertex,
            "the slice-aligned +w pole sits on a 16-cell vertex by design"
        );
    }

    /// The Hyperslice cull runs when EITHER the toggle is on OR the projection is
    /// `Hyperslice`; every other projection leaves it off unless the toggle is set.
    #[test]
    fn hyperslice_cull_active_fires_for_toggle_or_projection() {
        // Projection mode alone activates the cull, toggle off.
        assert!(hyperslice_cull_active(
            false,
            WireframeProjection::Hyperslice
        ));
        // Toggle alone activates it under any other projection.
        for mode in WireframeProjection::ALL {
            assert!(
                hyperslice_cull_active(true, mode),
                "toggle on must activate the cull under {mode:?}"
            );
        }
        // Neither set: only the Hyperslice projection mode keeps it on.
        for mode in WireframeProjection::ALL {
            let expected = matches!(mode, WireframeProjection::Hyperslice);
            assert_eq!(
                hyperslice_cull_active(false, mode),
                expected,
                "with the toggle off, only the Hyperslice projection activates the cull ({mode:?})"
            );
        }
    }

    /// `resolve_schlegel_params` feeds the topology-derived `face_planes` normal,
    /// NOT the buggy dual-vertex `cell{120,600}_face_planes`. For the 600-cell the
    /// resolved normal (a) equals the `face_planes` direction and (b) is, for at
    /// least one cell, far from every dual normal.
    #[test]
    fn schlegel_params_from_face_planes_not_dual() {
        use loam_physics::euclidean_r4::cell600_face_planes;
        let polytope = Polytope4::Cell600;
        let (topo_normals, _) = polytope.face_planes();
        let (dual_normals, _) = cell600_face_planes();
        // A cell whose topology normal is far from every dual normal; the 96
        // golden-ratio orbits guarantee one exists (the dual set has only 120
        // entries for a 600-faced polytope).
        let divergent = (0..topo_normals.len() as u32).find(|&i| {
            let n = topo_normals[i as usize];
            dual_normals
                .iter()
                .all(|d| (n - *d).length() > 1e-3 && (n + *d).length() > 1e-3)
        });
        let cell_index = divergent.expect(
            "the 600-cell must have a golden-ratio face that diverges from the dual-vertex set",
        );
        let params = resolve_schlegel_params(polytope, cell_index);
        // (a) The resolved normal is exactly the topology face-plane direction.
        assert_eq!(params.cell_normal, topo_normals[cell_index as usize]);
        // (b) It is far from every dual normal (the buggy path is not the source).
        for d in &dual_normals {
            let n = params.cell_normal;
            assert!(
                (n - *d).length() > 1e-3 && (n + *d).length() > 1e-3,
                "resolved Schlegel normal must not coincide with any dual-vertex normal"
            );
        }
    }

    /// An out-of-range `cell_index` clamps to `[0, cell_count - 1]` without panic.
    #[test]
    fn schlegel_cell_index_clamped_to_cell_count() {
        let polytope = Polytope4::Pentatope; // 5 cells.
        let last = polytope.cell_count() as u32 - 1;
        let params = resolve_schlegel_params(polytope, 9999);
        assert_eq!(params.cell_index, last);
        // The clamped resolution equals an explicit last-cell request.
        let at_last = resolve_schlegel_params(polytope, last);
        assert_eq!(params, at_last);
    }

    /// An overrunning `cell_index` writes the CLAMPED index back into the
    /// projection so the enum, cache, UI stepper, and console report all name the
    /// same cell. Re-resolving is a fixed point.
    #[test]
    fn schlegel_resolve_syncs_carried_index_to_clamp() {
        let polytope = Polytope4::Pentatope; // 5 cells; indices 0..=4.
        let last = polytope.cell_count() as u32 - 1;
        let overrun = WireframeProjection::Schlegel { cell_index: 300 };
        let (projection, cache) = synced_schlegel_projection(overrun, Some(polytope));
        let cache = cache.expect("a Schlegel mode with a subject must produce a cache");
        // The carried index now equals the cache's clamped index (no desync).
        match projection {
            WireframeProjection::Schlegel { cell_index } => {
                assert_eq!(
                    cell_index, cache.cell_index,
                    "carried index must match cache"
                );
                assert_eq!(cell_index, last, "overrun must clamp to the last cell");
            }
            other => panic!("Schlegel input must stay Schlegel, got {other:?}"),
        }
        // Idempotent: feeding the synced projection back yields the same index.
        let (again, _) = synced_schlegel_projection(projection, Some(polytope));
        assert_eq!(again, projection, "re-resolve must be a fixed point");
    }

    /// A non-Schlegel mode passes through and clears the cache; a subjectless
    /// Schlegel keeps its index verbatim with a `None` cache.
    #[test]
    fn synced_schlegel_passes_through_non_schlegel_and_subjectless() {
        // Non-Schlegel: unchanged projection, no cache.
        let (proj, cache) =
            synced_schlegel_projection(WireframeProjection::Stereographic, Some(Polytope4::Cell24));
        assert_eq!(proj, WireframeProjection::Stereographic);
        assert!(cache.is_none(), "non-Schlegel mode must clear the cache");
        // Schlegel with no subject: index untouched, no cache (the wireframe
        // draws nothing for an empty / non-polychoral row anyway).
        let schlegel = WireframeProjection::Schlegel { cell_index: 7 };
        let (proj, cache) = synced_schlegel_projection(schlegel, None);
        assert_eq!(proj, schlegel, "no subject means no clamp, index stays put");
        assert!(cache.is_none(), "no subject means no cache");
    }

    /// Resolving a fixed `(polytope, cell_index)` twice yields BIT-identical f32,
    /// so the cache can be rebuilt on any select without projection jitter.
    #[test]
    fn schlegel_resolution_is_bit_deterministic() {
        for polytope in Polytope4::ALL {
            let cell_index = (polytope.cell_count() / 2) as u32;
            let a = resolve_schlegel_params(polytope, cell_index);
            let b = resolve_schlegel_params(polytope, cell_index);
            assert_eq!(a.cell_normal, b.cell_normal, "{polytope:?} normal");
            assert_eq!(a.cell_basis, b.cell_basis, "{polytope:?} basis");
            assert_eq!(a.cell_offset, b.cell_offset, "{polytope:?} offset");
            assert_eq!(
                a.viewpoint_distance, b.viewpoint_distance,
                "{polytope:?} viewpoint"
            );
        }
    }

    /// The cached Schlegel basis is an orthonormal frame spanning the chosen
    /// cell's 3-flat (the no-snap invariant: derived once in canonical coords).
    #[test]
    fn schlegel_cell_basis_spans_chosen_cell() {
        for polytope in Polytope4::ALL {
            let cell_index = (polytope.cell_count() / 2) as u32;
            let params = resolve_schlegel_params(polytope, cell_index);

            for (i, axis) in params.cell_basis.iter().enumerate() {
                assert!(
                    (axis.length() - 1.0).abs() < 1e-5,
                    "{polytope:?} basis axis {i} must be unit"
                );
                assert!(
                    axis.dot(params.cell_normal).abs() < 1e-5,
                    "{polytope:?} basis axis {i} must be in the cell plane"
                );
            }
            for i in 0..3 {
                for j in (i + 1)..3 {
                    assert!(
                        params.cell_basis[i].dot(params.cell_basis[j]).abs() < 1e-5,
                        "{polytope:?} basis axes {i}/{j} must be orthogonal"
                    );
                }
            }

            let topo = polytope.topology();
            let cell = topo.cells[params.cell_index as usize];
            let anchor = topo.vertices[cell[0] as usize];
            for &vi in cell {
                let delta = topo.vertices[vi as usize] - anchor;
                let reconstructed = params
                    .cell_basis
                    .iter()
                    .fold(Vec4::ZERO, |acc, &axis| acc + delta.dot(axis) * axis);
                let residual = delta - reconstructed;
                assert!(
                    residual.length() < 5e-4,
                    "{polytope:?} cell vertex {vi} must reconstruct in the basis"
                );
            }
        }
    }

    /// Under a non-identity `rot_state` the effective Schlegel normal and basis are
    /// the canonical ones rotated by that rotor, so the chosen cell stays the outer
    /// boundary as the body spins. Verified at the math level (the rotor apply).
    #[test]
    fn schlegel_frame_rotates_with_body() {
        let polytope = Polytope4::Tesseract;
        let cell_index = 0;
        let params = resolve_schlegel_params(polytope, cell_index);
        // A non-trivial xw rotation, so the canonical normal actually moves.
        let rot = (Plane4::Xw.unit_bivector() * 0.7).exp().normalize();
        let rotated = rot.apply(params.cell_normal);
        assert!(
            (rotated - params.cell_normal).length() > 1e-3,
            "rotation must actually move the normal for this test to bite"
        );
        // Rotation preserves unit length, so the rotated normal is still a valid
        // outward unit normal for the engine `Projection::Schlegel`.
        assert!((rotated.length() - 1.0).abs() < 1e-5);
        let rotated_basis = params.cell_basis.map(|axis| rot.apply(axis));
        for (i, axis) in rotated_basis.iter().enumerate() {
            assert!((axis.length() - 1.0).abs() < 1e-5, "axis {i} unit");
            assert!(
                axis.dot(rotated).abs() < 1e-5,
                "axis {i} remains perpendicular to the rotated normal"
            );
        }
        // Every chosen-cell vertex stays on the rotated boundary hyperplane:
        // `dot(rotated_normal, rot.apply(v))` still equals `cell_offset`.
        let topo = polytope.topology();
        let cell = topo.cells[cell_index as usize];
        let anchor = topo.vertices[cell[0] as usize];
        let rotated_anchor = rot.apply(anchor);
        for &vi in cell {
            let v = topo.vertices[vi as usize];
            let lhs = rotated.dot(rot.apply(v));
            assert!(
                (lhs - params.cell_offset).abs() < 1e-4,
                "rotated cell vertex must stay on the rotated boundary hyperplane: {lhs} vs {}",
                params.cell_offset
            );
            let delta = v - anchor;
            let rotated_delta = rot.apply(v) - rotated_anchor;
            for (axis, rotated_axis) in params.cell_basis.iter().zip(rotated_basis) {
                let want = delta.dot(*axis);
                let got = rotated_delta.dot(rotated_axis);
                assert!(
                    (got - want).abs() < 1e-5,
                    "rotated basis coordinates must match canonical coordinates"
                );
            }
        }
    }

    /// [`ViewMode::Single`] renders EXACTLY the `strip_subject`; Shapes / Filmstrip
    /// render the full row.
    #[test]
    fn single_mode_renders_one_subject() {
        let subject = entry(RaymarchShape::Polytope(Polytope4::Cell600));
        let row = [
            entry(RaymarchShape::Polytope(Polytope4::Tesseract)),
            entry(RaymarchShape::Polytope(Polytope4::Cell24)),
            entry(RaymarchShape::Polytope(Polytope4::Pentatope)),
        ];
        // Single: exactly one body, and it is the subject (not any row member).
        let single = render_row_entries(ViewMode::Single, &row, &subject);
        assert_eq!(single.len(), 1, "Single renders exactly one body");
        assert_eq!(single[0], subject, "the single body is the strip_subject");
        // The subject is deliberately absent from the row, so a length-1 result
        // alone cannot accidentally pass by aliasing a row entry.
        assert!(
            !row.contains(&subject),
            "test setup: subject must differ from every row entry",
        );
        // Shapes / Filmstrip render the whole row verbatim (same pointer + len).
        for mode in [ViewMode::Shapes, ViewMode::Filmstrip] {
            let full = render_row_entries(mode, &row, &subject);
            assert_eq!(full, &row[..], "{mode:?} renders the full row");
        }
    }

    /// In Single mode the Schlegel cell-index bound is the `strip_subject`'s cell
    /// count, NOT any row member's (the bound walks the rendered row's leading
    /// polychoron, as [`Demo::schlegel_subject`] does).
    #[test]
    fn single_mode_schlegel_cell_bound_from_subject() {
        let subject = entry(RaymarchShape::Polytope(Polytope4::Cell600));
        // A row whose leading polychoron has a DIFFERENT cell count, so reading
        // the row instead of the subject would give the wrong bound.
        let row = [
            entry(RaymarchShape::Polytope(Polytope4::Pentatope)), // 5 cells
            entry(RaymarchShape::Polytope(Polytope4::Cell24)),
        ];
        // Mirror `Demo::schlegel_subject`: first polychoron of the rendered row.
        let subject_poly = render_row_entries(ViewMode::Single, &row, &subject)
            .iter()
            .find_map(|e| e.shape.polytope4())
            .expect("the single subject is a polychoron");
        assert_eq!(subject_poly, Polytope4::Cell600);
        assert_eq!(
            subject_poly.cell_count(),
            Polytope4::Cell600.cell_count(),
            "the cell-index bound is the subject's cell count (600), not the row's leading 5",
        );
        // And in Shapes mode the same walk yields the ROW's leading polychoron
        // (the 5-cell), confirming the two modes resolve different subjects.
        let row_poly = render_row_entries(ViewMode::Shapes, &row, &subject)
            .iter()
            .find_map(|e| e.shape.polytope4())
            .expect("the row has a polychoron");
        assert_eq!(row_poly, Polytope4::Pentatope);
        assert_ne!(
            subject_poly.cell_count(),
            row_poly.cell_count(),
            "test setup: subject and row-leader must differ in cell count",
        );
    }

    /// Each `ViewMode` round-trips through the tab's stage-then-apply shape:
    /// staging a different value and applying it lands `view_mode` there and the
    /// rendered row matches; re-staging the same mode is a no-op.
    #[test]
    fn view_mode_tab_round_trips() {
        let subject = entry(RaymarchShape::Polytope(Polytope4::Cell24));
        let row = [entry(RaymarchShape::Polytope(Polytope4::Tesseract))];

        // Replicate the tab's stage rule: stage iff the clicked value differs.
        let stage = |current: ViewMode, clicked: ViewMode| -> Option<ViewMode> {
            (clicked != current).then_some(clicked)
        };

        for &target in &[ViewMode::Shapes, ViewMode::Single, ViewMode::Filmstrip] {
            // Start from a mode that is guaranteed different from `target` so the
            // stage fires; Single <-> Shapes covers both directions.
            let start = if target == ViewMode::Shapes {
                ViewMode::Single
            } else {
                ViewMode::Shapes
            };
            let pending = stage(start, target);
            assert_eq!(pending, Some(target), "clicking {target:?} stages it");
            // Apply (the `pending_view_mode.take()` arm in render_overlay).
            let applied = pending.unwrap_or(start);
            assert_eq!(applied, target, "applying the pending mode lands on it");
            // The rendered row reflects the applied mode.
            let rendered = render_row_entries(applied, &row, &subject);
            match applied {
                ViewMode::Single => assert_eq!(rendered, std::slice::from_ref(&subject)),
                ViewMode::Shapes | ViewMode::Filmstrip => assert_eq!(rendered, &row[..]),
            }
            // Re-staging the same mode is a no-op: nothing to apply.
            assert_eq!(
                stage(target, target),
                None,
                "{target:?} re-stage is a no-op"
            );
        }
    }

    // ---- Section layers (cross-section + projected cap) ------------------

    /// The two-layer split: the honest cross-section ALWAYS resolves to drop-w
    /// (`Identity`); the projected cap follows the active projection.
    #[test]
    fn section_layer_projection_honest_ignores_projected_follows() {
        let actives = [
            Projection::Identity,
            Projection::Perspective4D {
                focal_distance: 2.0,
            },
            Projection::Stereographic { pole: Vec4::W },
            Projection::schlegel(Vec4::W, 0.5, 0.75),
        ];
        for active in actives {
            // Honest layer: drop-w no matter what the active projection is.
            assert_eq!(
                section_layer_projection(true, active),
                Projection::Identity,
                "honest cross-section must stay drop-w under active {active:?}"
            );
            // Projected cap: exactly the active projection, passed through.
            assert_eq!(
                section_layer_projection(false, active),
                active,
                "projected cap must follow the active projection {active:?}"
            );
        }
    }

    /// `fill_visible` is the layer's on/off switch: positive alpha draws,
    /// `0` (or below) skips the pass.
    #[test]
    fn section_layer_fill_visible_at_positive_alpha_only() {
        assert!(!SectionLayer {
            perimeter: true,
            surface_alpha: 0.0
        }
        .fill_visible());
        assert!(SectionLayer {
            perimeter: false,
            surface_alpha: 0.01
        }
        .fill_visible());
        assert!(SectionLayer {
            perimeter: false,
            surface_alpha: 1.0
        }
        .fill_visible());
    }

    /// The defaults encode "honest slice visible, reprojected cap off": the
    /// cross-section's perimeter + fill on, the projected cap off.
    #[test]
    fn section_layer_defaults_match_spec() {
        let cross = SectionLayer::CROSS_SECTION_DEFAULT;
        assert!(cross.perimeter, "honest perimeter on by default");
        assert!(cross.fill_visible(), "honest fill on by default");
        assert!(
            cross.surface_alpha > 0.5 && cross.surface_alpha <= 1.0,
            "honest default alpha should be full-ish, got {}",
            cross.surface_alpha
        );

        let cap = SectionLayer::PROJECTED_CAP_DEFAULT;
        assert!(!cap.perimeter, "projected-cap perimeter off by default");
        assert!(!cap.fill_visible(), "projected-cap fill off by default");
    }

    /// Throw slot 1 along +w, the one axis on which it cannot reach a
    /// neighbour, from a +x lever point so it picks up an angular velocity
    /// too. The rest of the row stays a clean control group.
    fn tumbling(slots: usize) -> PlaygroundPhysics {
        let mut physics = PlaygroundPhysics::new(slots, BODY_SIZE);
        let layout = Vec4::from_array(body_position(1, slots));
        physics.world.bodies[1].apply_impulse_at_point(
            &EuclideanR4,
            Vec4::new(0.0, 0.0, 0.0, 1.2),
            layout + Vec4::X * 0.5,
        );
        physics.step(24);
        physics
    }

    /// The raymarched body reads the PHYSICS pose, not the authored layout
    /// under the UI spin: a thrown, tumbling slot's uniform carries its live
    /// centre and `spin · orientation`. Reverting either to the authored value
    /// would strand the SDF surface while every raster pass followed the body.
    #[test]
    fn sdf_body_uniform_reads_the_physics_pose_not_the_authored_spin() {
        let slots = 3;
        let physics = tumbling(slots);
        let shape = entry(RaymarchShape::Polytope(Polytope4::Cell24));
        let spin = (Plane4::Xy.unit_bivector() * 0.7).exp().normalize();

        let uniform = sdf_body_uniform(
            &physics,
            &shape,
            1,
            slots,
            spin,
            BODY_SIZE,
            SurfaceMode::Sdf,
        );

        let body = &physics.world.bodies[1];
        assert_ne!(
            body.orientation.rotation,
            Rotor4::IDENTITY,
            "throw produced no rotation, so the rotor pin below is vacuous"
        );
        assert_eq!(uniform.position, body.position.to_array());
        assert_ne!(
            uniform.position,
            body_position(1, slots),
            "uniform centre still reads the static layout"
        );
        assert_eq!(
            uniform.rotor,
            <[f32; 8]>::from(composed_rotor(spin, body.orientation.rotation))
        );
        assert_ne!(
            uniform.rotor,
            <[f32; 8]>::from(spin),
            "uniform rotor still reads the authored spin alone"
        );
    }

    /// An untouched slot's uniform is the authored layout and spin exactly:
    /// routing through physics adds no drift to a body nobody threw, which is
    /// what lets the pin above use exact equality.
    #[test]
    fn sdf_body_uniform_of_an_untouched_slot_is_the_authored_layout_and_spin() {
        let slots = 3;
        let physics = tumbling(slots);
        let shape = entry(RaymarchShape::Polytope(Polytope4::Cell24));
        let spin = (Plane4::Zw.unit_bivector() * -1.1).exp().normalize();

        let uniform = sdf_body_uniform(
            &physics,
            &shape,
            2,
            slots,
            spin,
            BODY_SIZE,
            SurfaceMode::Sdf,
        );
        assert_eq!(uniform.position, body_position(2, slots));
        assert_eq!(uniform.rotor, <[f32; 8]>::from(spin));
    }

    /// A callout anchor rides the live body: body frame, then projection, then
    /// the live R³ centre. Hand-rolling it from the authored spin over the
    /// static layout, which is what unwiring the callout means, lands the
    /// leader line on a vertex nothing drew.
    #[test]
    fn row_frame_anchor_reads_the_live_pose_not_the_authored_spin() {
        let slots = 3;
        let physics = tumbling(slots);
        let row = [entry(RaymarchShape::Polytope(Polytope4::Cell24)); 3];
        let spin = (Plane4::Xy.unit_bivector() * 0.7).exp().normalize();
        let frame = super::RowFrame {
            physics: &physics,
            row: &row,
            spin,
            body_size: BODY_SIZE,
            projection: Projection::Identity,
            w_slice: 0.0,
            camera_distance: 4.0,
        };

        let canonical = Polytope4::Cell24.topology().vertices[0];
        let pose = physics.pose(1, slots, spin);
        // Drop-w projection, so the anchor is the body-local vertex truncated
        // and then carried by the body's R³ centre.
        assert_eq!(
            frame.anchor_r3(1, canonical),
            pose.body_local(canonical, BODY_SIZE).truncate() + pose.position_r3()
        );
        assert_ne!(
            frame.anchor_r3(1, canonical),
            (BODY_SIZE * spin.apply(canonical)).truncate()
                + Vec4::from_array(body_position(1, slots)).truncate(),
            "anchor still reads the authored spin over the static layout"
        );
    }

    /// The polychora opt-out survives the pose plumbing: the 120/600-cell go
    /// inert in every mode (their face planes are the known-wrong dual-vertex
    /// fit), the other polychora only outside Sdf mode, and a smooth-surface
    /// shape never opts out.
    #[test]
    fn sdf_body_uniform_keeps_the_polychora_opt_out() {
        let slots = 3;
        let physics = tumbling(slots);
        let inert = BodyUniform::default().kind;
        let live = |shape, mode| {
            sdf_body_uniform(
                &physics,
                &entry(shape),
                1,
                slots,
                Rotor4::IDENTITY,
                BODY_SIZE,
                mode,
            )
            .kind
                != inert
        };

        for mode in [SurfaceMode::Sdf, SurfaceMode::Raster, SurfaceMode::Off] {
            for heavy in [Polytope4::Cell120, Polytope4::Cell600] {
                assert!(
                    !live(RaymarchShape::Polytope(heavy), mode),
                    "{heavy:?} raymarched in {mode:?}"
                );
            }
            assert_eq!(
                live(RaymarchShape::Polytope(Polytope4::Cell24), mode),
                mode.uses_sdf_for_polychora(),
                "24-cell SDF dispatch disagrees with {mode:?}"
            );
            assert!(
                live(RaymarchShape::CliffordTorus, mode),
                "smooth surface opted out in {mode:?}"
            );
        }
    }
}
