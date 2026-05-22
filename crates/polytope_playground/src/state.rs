//! Demo state: the [`Demo`] struct, the mode/view/deferred-action enums, the
//! [`RotorTerm`] data type and its display helpers, the angular-velocity
//! derivation, body layout, and full reset.
//!
//! This module owns the data model. Per-mode UI rendering lives in
//! `modes/{active,composer,filmstrip,shapes}.rs` as additional `impl Demo`
//! blocks; cross-cutting overlay UI lives in `ui.rs`. All struct fields are
//! `pub(crate)` so those sibling impls can access them directly without
//! per-field accessors.

use rye_app::{Camera, OrbitController};
use rye_math::{Bivector4, EuclideanR3, Plane4, Rotor4};
use rye_render::raymarch::{BodyUniform, Hyperslice4DNode};

use crate::catalog::ShapeEntry;
use crate::consts::{BASE_ROTATION_RATE, BODY_SIZE, BODY_X_SPACING, BODY_Y, T_SLIDER_INITIAL};

// ---------------------------------------------------------------------------
// Mode + view enums
// ---------------------------------------------------------------------------

/// Continuous-rotation source. Two distinct UIs (active-set checkboxes vs composed
/// sequence) populate the angular velocity independently; the user picks which one drives
/// `omega` for the spin animation via a tab in the rotation tab row.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum RotationMode {
    /// Sum of unit bivectors of planes whose checkboxes are on. The classic
    /// toggleable mode: 1..6 keys / panel checkboxes.
    Active,
    /// Sum of bivectors derived from the composed seq: each term contributes
    /// `scalar.unwrap_or(1.0) * sum_of_unit_bivectors`. Apply (one-shot rotor
    /// multiplication) is still available in this mode and is independent of the spin
    /// animation.
    Composer,
}

/// Visualisation mode. Orthogonal to [`RotationMode`]: rotation configures *how* the
/// rotor evolves, view configures *what* the scene shows. Two distinct visual demos live
/// here, picked by a top-level tab row above the rotation tabs.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ViewMode {
    /// Multi-shape comparison: `self.row` of [`ShapeEntry`]s rendered side-by-side at
    /// one common `w_slice`. Shape order in the row is meaningful; drag-and-drop
    /// rearranges the scene's left-to-right layout.
    Shapes,
    /// Single-shape filmstrip: one [`ShapeEntry`] (independent of the row) rendered N
    /// times across evenly-spaced `w_slice` values around the slider's current `w`.
    /// Order of the scene's row is irrelevant in this mode; the row UI is hidden
    /// entirely.
    Filmstrip,
}

/// How the six regular convex 4-polytopes have their surface rendered.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub(crate) enum SurfaceMode {
    /// Rasterized filled cross-section cell-caps. The default. Much faster for the 120-cell
    /// and 600-cell than the SDF, exact (no Wolfe-greedy approximation), and sidesteps the
    /// cell120/600 face-plane BUG in `rye_physics::euclidean_r4`.
    #[default]
    Raster,
    /// SDF raymarch via [`Demo::node`]. The pre-rasterizer behavior, kept for visual
    /// comparison. Slower on the 120/600-cell and carries the documented face-plane BUG.
    Sdf,
    /// No surface rendered for the polychora. The wireframe overlay (if enabled) still
    /// shows the polytope's edge graph, but the cap interiors stay empty. Useful for
    /// inspecting the wireframe + cross-section perimeter on their own without the cap
    /// fill competing for attention.
    Off,
}

impl SurfaceMode {
    /// Parse the console-arg spelling. Returns `None` for any other input.
    pub(crate) fn from_token(token: &str) -> Option<Self> {
        match token {
            "raster" => Some(SurfaceMode::Raster),
            "sdf" => Some(SurfaceMode::Sdf),
            "off" => Some(SurfaceMode::Off),
            _ => None,
        }
    }

    /// `true` when the polychoral SDF dispatch needs to be live (so that body uniforms
    /// stay populated for the kernel). False for Raster (the rasterizer draws those
    /// polytopes) and Off (nothing draws them).
    pub(crate) fn uses_sdf_for_polychora(self) -> bool {
        matches!(self, SurfaceMode::Sdf)
    }
}

/// How the parent wireframe's 4D vertex positions project to R³ for rendering.
/// Independent of the cross-section's projection (which is always drop-w because the
/// slice IS a 3-flat and drop-w is the inhabitant's natural view of it).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub(crate) enum WireframeProjection {
    /// Axis-aligned drop-w: `(x, y, z, w) -> (x, y, z)`. The default. Collapses every
    /// pair of w-opposite vertices to the same R³ point, so axis-aligned polytopes
    /// (especially the tesseract) render as visually degenerate "flat" shapes; the
    /// 5-cell and 24-cell read more naturally because their cells aren't w-aligned.
    #[default]
    DropW,
    /// 4D pinhole perspective from a viewer at `(0, 0, 0, focal_distance)` looking
    /// in -w. Produces the classical "cube within a cube" tesseract view: +w face
    /// renders as the outer (larger) shape, -w face as the inner (smaller) shape,
    /// connecting edges as the frustum lines. Brings axis-aligned polytopes to life
    /// at the cost of slight distortion on the polytopes that already read well
    /// under drop-w.
    WDepth,
}

impl WireframeProjection {
    /// Parse the console-arg spelling. Hyphens because the console grammar lexes on
    /// whitespace and `drop-w` / `w-depth` read as single tokens.
    pub(crate) fn from_token(token: &str) -> Option<Self> {
        match token {
            "drop-w" => Some(WireframeProjection::DropW),
            "w-depth" => Some(WireframeProjection::WDepth),
            _ => None,
        }
    }

    /// Resolve to a [`rye_math::Projection<4>`]. The focal distance is hardcoded to
    /// `2.0`, sized to comfortably exceed the unit-circumradius polytope's w-extent
    /// after the demo's `BODY_SIZE` scaling, so the perspective denominator never
    /// approaches zero in normal use.
    pub(crate) fn to_projection(self) -> rye_math::Projection<4> {
        match self {
            WireframeProjection::DropW => rye_math::Projection::Identity,
            WireframeProjection::WDepth => rye_math::Projection::Perspective4D {
                focal_distance: 2.0,
            },
        }
    }
}

/// How the parent-wireframe edges are colored. Orthogonal to the alpha-modulation
/// toggle [`Demo::wireframe_nearest_active`]: the color mode picks the hue, the
/// nearest-active toggle then modulates alpha on top.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub(crate) enum WireframeColorMode {
    /// Per-vertex RGB from [`rye_physics::polytope::vertex_color_by_position`]: a
    /// continuous color field over the polytope's vertex set that flows smoothly
    /// across the edge graph and reveals symmetry as color gradients. Every vertex
    /// picks up a distinct hue from its 4D coordinates, so the polytope reads as a
    /// stylized colorful identity rather than a uniform mass of edges.
    #[default]
    Unique,
    /// Binary green/gray by cell activity: edges that belong to at least one cell
    /// the slice is *currently* intersecting are bright green; all other edges are
    /// dim neutral gray. Reads as "which cells of the polytope am I looking at right
    /// now" at a glance, complementing the gradient `nearest-active` mode (which is
    /// continuous and shows *how strongly* each cell is being crossed).
    Active,
}

impl WireframeColorMode {
    /// Parse a console-arg spelling (`unique` or `active`). Returns `None` for any
    /// other input; the caller surfaces a usage error.
    pub(crate) fn from_token(token: &str) -> Option<Self> {
        match token {
            "unique" => Some(WireframeColorMode::Unique),
            "active" => Some(WireframeColorMode::Active),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// RotorTerm + display helpers
// ---------------------------------------------------------------------------

/// One term in the rotor-composition sequence: a sum of unit bivectors with an optional
/// leading scalar (angle in radians).
///
/// Without a scalar the term is `exp(sum_of_unit_bivectors)`, which is the natural
/// unit-magnitude rotation along the term's bivector direction. With a scalar `phi` it
/// becomes `exp(phi * sum_of_unit_bivectors)`. The scalar is optional by design: most
/// uses ("rotate 90° in xy") want a scalar, but the "raw direction" form (just the
/// bivector itself) is useful for composing isoclinics where the magnitude is implicit.
///
/// Bivector addition within a term is commutative, so plane order inside a term doesn't
/// matter. Rotor multiplication between terms is non-commutative, so the seq's term
/// order does.
#[derive(Clone, Debug, Default)]
pub(crate) struct RotorTerm {
    /// Unit-bivector planes summed inside `exp(...)`. Non-empty for a term to display;
    /// an empty term is dropped.
    pub(crate) planes: Vec<Plane4>,
    /// Optional scalar prefix `phi` in radians. `None` means the raw bivector sum (unit
    /// magnitude); `Some(phi)` scales the whole sum before `exp()`. The panel's "Add
    /// scalar" action initialises this to `FRAC_PI_2`; `Default::default()` is `None`
    /// so an empty draft commits as a unit-magnitude term.
    pub(crate) scalar: Option<f32>,
}

/// Render `(p_0 + p_1 + ...)` (with parens iff multi-plane) into the current ui. Each
/// plane goes through `render_plane`, which decides whether it's an interactive drag pill
/// (term card), plain monospace (draft card), or anything else. The paren logic and `+`
/// separators are shared so the visual reading of a bivector sum stays identical across
/// all callsites.
pub(crate) fn render_plane_sum(
    ui: &mut rye_app::egui::Ui,
    planes: &[Plane4],
    mut render_plane: impl FnMut(&mut rye_app::egui::Ui, usize, Plane4),
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

/// Render a single [`RotorTerm`] as the `scalar · bivec` form that appears
/// inside `exp(...)`. Multi-plane terms get inner parens; the lone scalar
/// prefix is dropped when absent. Pure presentation, no math.
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

/// Wrap a list of bivector-expression parts into a single bivector expression
/// (paren-grouped when there's more than one part). None when the list is empty so the
/// caller can return early.
pub(crate) fn render_bivector_sum(parts: &[String]) -> Option<String> {
    match parts {
        [] => None,
        [only] => Some(only.clone()),
        many => Some(format!("({})", many.join(" + "))),
    }
}

/// Angular velocity from a composed seq: sum over terms of
/// `scalar * sum_of_unit_bivectors_in_term`, scaled by rate_scale. Bivector addition is
/// commutative, so term order is irrelevant in this continuous mode (it matters for the
/// multiplicative `Apply` action, but that's a separate one-shot path).
///
/// The Active-mode angular velocity is structurally a special case: each active plane is
/// one unit term with `scalar = None`. The app-level `omega_per_sec` dispatcher inlines
/// that walk over the `[bool; 6]` directly to avoid allocating a transient seq each
/// frame.
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

/// State mutations queued during overlay rendering and applied AFTER the overlay's
/// measure + visible passes finish. Any mutation that changes the overlay's natural
/// content height must go through this; applying mid-frame would make the two
/// `BottomOverlay` passes disagree on body height and the user would see a one-frame
/// layout mismatch as flicker.
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

/// Drag-and-drop payload for the rotor sequence UI. Terms (whole cards) and plane entries
/// (pills inside cards) both ride this single enum so a term card can be a single drop
/// zone that branches on the variant: a `Term` payload reorders the seq, an `Entry`
/// payload migrates a plane into this term.
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

/// Position of the `slot`-th body in a row of `n` bodies, centred on the world origin
/// and spaced by [`BODY_X_SPACING`]. Used by both initial body layout and per-frame body
/// uniforms.
pub(crate) fn body_position(slot: usize, n: usize) -> [f32; 4] {
    let x = (slot as f32 - (n as f32 - 1.0) * 0.5) * BODY_X_SPACING;
    [x, BODY_Y, 0.0, 0.0]
}

// ---------------------------------------------------------------------------
// The App struct
// ---------------------------------------------------------------------------

pub(crate) struct Demo {
    pub(crate) space: EuclideanR3,
    pub(crate) camera: Camera<EuclideanR3>,
    pub(crate) orbit: OrbitController<EuclideanR3>,
    pub(crate) node: Hyperslice4DNode,
    /// Rasterizer node for the cross-section perimeter (bright cyan edges around each
    /// cap polygon). Filled caps are NOT drawn -- the SDF raymarcher already renders the
    /// section polytope as a solid volume, so the rasterizer's job is to outline the
    /// boundaries between adjacent cell contributions, not to fill them.
    pub(crate) section_edges: rye_render::LineRasterNode,
    /// Rasterizer node for the dim "parent wireframe" overlay: the full polytope's edge
    /// graph (per body) projected via drop-w. Conveys polytope structure independent of
    /// which slice is currently shown.
    pub(crate) parent_wireframe: rye_render::LineRasterNode,
    /// Whether the cross-section + parent-wireframe overlay renders. Off by default so
    /// the existing SDF-only demo is unchanged; toggle via the `wireframe on|off` console
    /// subcommand.
    pub(crate) wireframe_enabled: bool,
    /// When `true`, parent-wireframe edges are alpha-graded by how close the current
    /// `w_slice` is to the midpoint of each cell they belong to: edges of cells the slice
    /// is *deep in* glow at full alpha, edges of cells the slice doesn't touch fade to the
    /// dim "context" alpha. As the slice scrubs, brightness propagates through the
    /// wireframe as a wave, visually identifying which cells are contributing caps at
    /// each moment. When `false`, every edge uses the same uniform dim alpha (the
    /// previous behavior).
    pub(crate) wireframe_nearest_active: bool,
    /// Whether the cyan section-perimeter outlines render. Independent of the parent
    /// wireframe edges; turning this off leaves the polytope's full edge graph visible
    /// but hides the cap-boundary highlight, useful for inspecting the parent structure
    /// without the slice's cyan trace competing for attention. On by default.
    pub(crate) wireframe_perimeter: bool,
    /// Base RGB for wireframe edges. Orthogonal to [`Self::wireframe_nearest_active`]:
    /// the color mode picks the hue, the nearest-active toggle then modulates alpha on
    /// top.
    pub(crate) wireframe_color_mode: WireframeColorMode,
    /// How the parent wireframe's 4D vertex positions project to R³. The cross-section
    /// always uses drop-w (mathematically the inhabitant's view of the slice 3-flat);
    /// this toggle only affects the dim wireframe overlay on top.
    pub(crate) wireframe_projection: WireframeProjection,
    /// Filled-faces rasterizer for the cross-section of every polychoral body. When
    /// [`Self::surface_raster_enabled`] is `true`, this replaces the SDF raymarch for the
    /// six regular convex 4-polytopes: the SDF gets `BodyUniform::default()` for those
    /// slots (which the kernel skips) and the section's filled cell-caps come through
    /// here instead. Per-body solid color + face-normal Lambert in the fragment shader.
    pub(crate) section_faces: rye_render::TriangleRasterNode,
    /// Antialiased point-disc rasterizer for vertex markers and cell-center sprites.
    /// Constructed once during demo setup; uploaded with the combined point mesh each
    /// frame the points overlay is enabled.
    pub(crate) points_node: rye_render::PointRasterNode,
    /// Master toggle for the points overlay. Off by default; the demo's identity is the
    /// SDF / wireframe / cross-section composition. Enable to layer vertex + cell-center
    /// sprites on top.
    pub(crate) points_enabled: bool,
    /// When [`Self::points_enabled`] is on, render a sprite at each polytope vertex.
    pub(crate) points_show_vertices: bool,
    /// When [`Self::points_enabled`] is on, render a sprite at each cell's centroid
    /// (mean of the cell's vertex positions). The 600 sprites for the 600-cell can read
    /// as a cluttered point cloud; toggle off independently of vertices for a cleaner
    /// look when only the polytope's vertex structure matters.
    pub(crate) points_show_cell_centers: bool,
    /// Screen-space radius (pixels) for both vertex and cell-center sprites. Single
    /// uniform size keeps the UX simple; per-category sizes are an easy follow-up if a
    /// real need emerges.
    pub(crate) points_size_px: f32,
    /// Scratch buffer reused across frames + bodies inside `render_points`. Cleared at
    /// the start of each invocation; capacity grows monotonically with the maximum
    /// combined vertex + cell-center count across all polychora in the row.
    pub(crate) points_mesh_scratch: rye_shape::PointMesh<3>,
    /// Shared depth attachment for the rasterizer chain in Shapes view. Sized to the
    /// swapchain and recreated on resize via [`rye_render::DepthBuffer::ensure`].
    ///
    /// Cleared once per frame at the top of the Shapes-view render path
    /// ([`crate::Demo::ensure_and_clear_shared_depth`]). Two passes consume it:
    /// - `section_faces` writes depth + color when raster mode is on (no-op in SDF
    ///   mode, so the buffer stays at the cleared `1.0` value).
    /// - `parent_wireframe` reads depth (no write) so lines behind a section cap are
    ///   correctly occluded. In SDF mode the cleared depth makes every wireframe
    ///   fragment pass the test trivially, preserving the historical visual.
    pub(crate) section_faces_depth: Option<rye_render::DepthBuffer>,
    /// Scratch buffers reused across frames + bodies inside `render_section_faces` to
    /// avoid per-body heap allocations on the 240 fps hot path. Both are cleared at
    /// the start of each invocation; capacity grows monotonically with the largest
    /// polychoron's vertex / triangle count seen so far.
    pub(crate) section_world_vertices_scratch: Vec<glam::Vec4>,
    pub(crate) section_faces_mesh_scratch: rye_shape::TriangleMesh<3>,
    /// Selects how the six regular convex 4-polytopes are rendered. Smooth-surface shapes
    /// (Clifford torus, duocylinder, etc.) ignore this and always render via the SDF since
    /// they have no polytope topology to section.
    pub(crate) surface_mode: SurfaceMode,
    /// Polytope row built at startup from `--shapes` CLI args (or `DEFAULT_ROW`); drives
    /// both the body uniforms and per-body label lookups in the overlay.
    pub(crate) row: Vec<ShapeEntry>,

    pub(crate) w_slice: f32,
    pub(crate) slider_up_held: bool,
    pub(crate) slider_down_held: bool,
    pub(crate) slider_left_held: bool,
    pub(crate) slider_right_held: bool,

    pub(crate) rotate: bool,
    pub(crate) rot_state: Rotor4,
    /// Toggle bitmap for the six rotation planes; sum of active planes' unit bivectors
    /// becomes the per-frame angular velocity. See [`Plane4::ALL`] for the index ->
    /// plane mapping.
    pub(crate) active: [bool; 6],
    pub(crate) rate_scale: f32,
    /// Accumulated time spent rotating (advances only while `rotate == true`; resets on
    /// **R**). Useful for spotting periodicities in compound-bivector animations.
    pub(crate) rot_time: f32,
    /// Upper bound on the `t` slider's range. Doubles every time the spin's accumulated
    /// `rot_time` exceeds the current bound, so the slider's handle stays meaningful at
    /// long elapsed times instead of pinning at the right edge. Reset to the initial
    /// bound on `R`.
    pub(crate) t_slider_max: f32,

    /// Whether the bottom controls overlay is expanded. When `false` only the always-on
    /// slider strip + rate row is shown at the bottom; when `true` the strip extends
    /// upward to also show the rotation-mode tabs, mode-specific UI, and shape row.
    /// Toggle via the `^` / `v` chevron button or the **H** key. There is no longer a
    /// side panel: the scene renders to the full window and the overlay floats over it.
    pub(crate) expanded: bool,

    /// Whether the modal "About / help" window is open. Triggered by clicking the `?`
    /// button; closes via the window's title-bar X (egui's `Window::open(&mut bool)`
    /// flips it).
    pub(crate) show_help: bool,
    /// Whether the floating `Render` settings modal is open. Off by default; opened
    /// from the gear button in the bottom overlay. The console is the primary UX for
    /// changing render settings; this modal is the discoverability aid for new
    /// readers who haven't found the console yet.
    pub(crate) show_render_panel: bool,
    /// Persistent state for the example annotation callout. Anchored to the first
    /// polychoron-in-row's vertex 0 (the 5-cell's +w apex when the demo opens with
    /// the default row); leader line + panel position track the anchor each frame as
    /// the polytope rotates. Off by default; opened from `View > Example callout`
    /// (and toggleable via the console `callout` command).
    ///
    /// Hosts the `rye_egui::callout` primitive added in the M4-close mini-sprint;
    /// future tutorial / explanation overlays in the playground will instantiate
    /// additional `CalloutState`s the same way.
    pub(crate) example_callout: rye_egui::CalloutState,

    /// Cached natural overlay width on first frame. Used as the fixed width of the
    /// overlay regardless of the current window size, so resizing the demo window
    /// doesn't stretch the controls. Set lazily on first render.
    pub(crate) overlay_pinned_width: Option<f32>,

    /// Whether the top-right rotation-formula popup is rendered. Off by default; the
    /// formula is dense for newcomers; the expanded section has a checkbox to turn it on
    /// for users who want to see exactly which bivectors and scalars compose into the
    /// current orientation.
    pub(crate) show_formula: bool,

    /// Whether the bottom controls overlay is rendered. On by default so first-time users
    /// see all the demo's state at once; toggle off via `View > Rotation controls` or
    /// the `H` key for an unobstructed scene (e.g., for screenshots or focused viewing).
    pub(crate) show_controls: bool,

    /// Top-level visualisation mode. `Shapes` shows `self.row` side-by-side at one
    /// `w_slice`; `Filmstrip` shows one polytope (`self.strip_subject`) sampled across
    /// an axis of w, an axis of t, or both at once (a 2D grid).
    pub(crate) view_mode: ViewMode,
    /// Filmstrip-axis toggles. At least one MUST be active when `view_mode == Filmstrip`
    /// (UI prevents both being off); when only `strip_w` is on the panel renders a
    /// horizontal row of cells across the w slider's value, when only `strip_t` is on
    /// it renders a vertical column across the rotation animation's `rot_time`, and when
    /// both are on it renders a 2D grid (w on one axis, t on the other; default
    /// orientation has w on columns and t on rows, swappable via `strip_swap_axes`).
    pub(crate) strip_w: bool,
    pub(crate) strip_t: bool,
    /// When both `strip_w` and `strip_t` are active, swap the default axis assignment
    /// (w-on-columns / t-on-rows becomes t-on-columns / w-on-rows).
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

    /// Mode change requested this frame by the mode tabs. Applied after the overlay
    /// finishes rendering so that the body that renders this frame still sees
    /// `rotation_mode` (the OLD value), and only the next frame swaps to the new mode.
    pub(crate) pending_mode: Option<RotationMode>,

    /// View change requested this frame by the view tab row.
    pub(crate) pending_view_mode: Option<ViewMode>,

    /// Composer-mode actions deferred to end-of-frame for the same reason as
    /// `pending_mode`.
    pub(crate) pending_actions: Vec<DeferredAction>,

    /// Sequence of [`RotorTerm`]s the user is building in the panel.
    pub(crate) seq: Vec<RotorTerm>,
    /// In-progress draft for the next term. Plane buttons append here; "Add" commits
    /// this list as a new term in `seq` and clears the draft.
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
    /// The composer seq's net bivector direction (no rate or base-rate scaling). This is
    /// the "function" the seq describes: sum over terms of `scalar * sum_planes`. The
    /// scrub slider uses this as its rotation axis-bivector; the projection of
    /// `log(rot_state)` onto this direction is the slider's value.
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

    /// Per-animation-second angular velocity (the bivector that, integrated over
    /// animation time, produces `rot_state`). Independent of `rate_scale`. Active mode
    /// sums the toggled basis bivectors; Composer mode delegates to the seq walker.
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

    /// Build the SDF body uniform for a single row entry, with polychora opt-out based on
    /// the active [`SurfaceMode`]: when the polychora are being rendered outside the SDF
    /// (Raster, or Off entirely), the returned uniform is `BodyUniform::default()` (kind =
    /// Invalid), which the kernel's dispatch chain skips. The slot is preserved (so the
    /// visual layout doesn't shift); the polychoral surface is rendered separately via
    /// [`Self::section_faces`] in `main.rs` (Raster) or simply not at all (Off).
    ///
    /// Smooth-surface shapes (Clifford torus, etc.) ignore the surface mode and always
    /// produce a live SDF body; they have no rasterizer path.
    fn sdf_body_for_slot(&self, slot: usize, n: usize, rotor: Rotor4) -> BodyUniform {
        let entry = &self.row[slot];
        if !self.surface_mode.uses_sdf_for_polychora() && entry.shape.polytope4().is_some() {
            return BodyUniform::default();
        }
        BodyUniform::polytope_with_rotor(
            body_position(slot, n),
            entry.shape.shape_id(),
            BODY_SIZE,
            rotor,
            entry.body_color,
        )
    }

    /// Drive every body in the row with the same rotor, lets the user directly compare
    /// slice signatures under identical 4D motion.
    pub(crate) fn write_all(&mut self, rotor: Rotor4) {
        let n = self.row.len();
        for slot in 0..n {
            let body = self.sdf_body_for_slot(slot, n, rotor);
            self.node.set_body(slot, body);
        }
    }

    /// Re-emit every body's uniform from the current row + rotor state. Called after row
    /// mutations (add/remove/reorder), rotor changes during spin, and surface-mode changes
    /// (any time the polychora switch between SDF-live and SDF-inert, which happens at
    /// every transition involving SDF mode).
    pub(crate) fn rebuild_bodies(&mut self) {
        let n = self.row.len();
        let rotor = self.rot_state;
        let bodies: Vec<BodyUniform> = (0..n)
            .map(|slot| self.sdf_body_for_slot(slot, n, rotor))
            .collect();
        self.node.set_bodies(&bodies);
    }

    /// Render a compact `exp(B · 0.30·t)` form for whichever mode drives the spin. `B`
    /// is the bivector velocity expression: a sum of plane terms (Active mode: each
    /// enabled plane is one unit-bivector term; Composer mode: each seq entry is its
    /// scalar-weighted bivector). Empty string when nothing is contributing.
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

    /// Full reset: pause spin, slice, rate, active set, orientation, time, draft. Reset
    /// implies "stop", so `rotate` flips off too; otherwise the next frame's `update()`
    /// would immediately start spinning the freshly-reset state, which the user almost
    /// never wants.
    pub(crate) fn reset(&mut self) {
        self.rotate = false;
        self.w_slice = 0.0;
        self.rate_scale = 1.0;
        // Restore the xw-only default active set so a first-time
        // user resetting and then hitting Spin sees motion.
        self.active = [false, false, true, false, false, false];
        self.rot_state = Rotor4::IDENTITY;
        self.rot_time = 0.0;
        self.t_slider_max = T_SLIDER_INITIAL;
        self.draft.clear();
        self.write_all(Rotor4::IDENTITY);
    }
}
