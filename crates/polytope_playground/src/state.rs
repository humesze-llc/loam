//! Demo state: the [`Demo`] struct, the mode/view/deferred-action enums, the
//! [`RotorTerm`] data type and its display helpers, the angular-velocity
//! derivation, body layout, and full reset.
//!
//! This module owns the data model. Per-mode UI rendering lives in
//! `modes/{active,composer,filmstrip,shapes}.rs` as additional `impl Demo`
//! blocks; cross-cutting overlay UI lives in `ui.rs`. All struct fields are
//! `pub(crate)` so those sibling impls can access them directly without
//! per-field accessors.

use std::collections::HashMap;

use rye_app::{freecam::Freecam, Camera, OrbitController};
use rye_math::{Bivector, Bivector4, EuclideanR3, Plane4, Rotor4};
use rye_physics::polytope::Polytope4;
use rye_render::raymarch::{BodyUniform, Hyperslice4DNode, RaymarchShape};

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
    /// Per-vertex RGB from [`rye_physics::polytope::vertex_color_by_position`]:
    /// each edge is shaded as a smooth gradient between its two endpoint vertex
    /// colors, and the canonical vertex hue is derived from the vertex's 4D
    /// coordinate. The polytope's symmetry shows up as continuous color flow
    /// across the edge graph (same scheme as `Polytope4::lines_colored_by_position`).
    #[default]
    VertexGradient,
    /// Each edge gets a distinct solid RGB color via greedy graph-coloring on
    /// the polytope's line graph: edges sharing a vertex always end up with
    /// different palette indices, so the local edge structure stays visually
    /// separable at any zoom level. Palette is deterministic (golden-ratio
    /// hue spacing) so the same shape always paints the same edges the same way.
    UniqueEdge,
    /// Per-vertex color by SIGNED `w` in the body-local frame: cool blue
    /// at extreme `-w`, warm orange at extreme `+w`, near-neutral at the
    /// slice plane. Normalized against the polytope's canonical max `|w|`
    /// (a fixed band per shape, NOT a per-frame rotated extent), so the
    /// gradient stays temporally stable as the rotor spins. Mirrors the
    /// `LineRasterStaticR4` shader's depth cue in `tesseract_demo`: a
    /// tesseract under xy/zw rotation paints inner cube blue + outer
    /// cube orange + connecting edges as smooth blue-to-orange gradients,
    /// making the w-depth migration visible regardless of camera angle.
    WDepth,
    /// Binary green/gray by cell activity: edges that belong to at least one cell
    /// the slice is *currently* intersecting are bright green; all other edges are
    /// dim neutral gray. Reads as "which cells of the polytope am I looking at right
    /// now" at a glance, complementing the gradient `nearest-active` mode (which is
    /// continuous and shows *how strongly* each cell is being crossed).
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

    /// Parse a console-arg spelling. Returns `None` for any unknown input;
    /// the caller surfaces a usage error.
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

/// Camera control mode. `Orbit` is the default scroll-zoom/drag-to-rotate camera that
/// stays focused on the world origin (where the polytope bodies sit). `FreeRoam` lets
/// the user fly the camera around via WASD + mouse-look; useful for inspecting the
/// 120-cell / 600-cell from arbitrary angles without orbiting through the floor.
///
/// Toggle via the `camera` console command: bare `camera` cycles, `camera orbit` /
/// `camera freecam` set explicitly. Switching to `Orbit` resets the orbit controller
/// to its default distance + pitch so the camera returns to a known framing instead
/// of inheriting wherever FreeRoam ended.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(crate) enum CameraMode {
    #[default]
    Orbit,
    FreeRoam,
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
/// Displayed angle of one Active-mode plane at animation time `t`: the user's
/// baseline plus the spin contribution `t * BASE_ROTATION_RATE` when the plane
/// is active. Free function (not a `Demo` method) so the rotor composition is
/// unit-testable without a GPU-backed `Demo`.
pub(crate) fn active_plane_angle(base: f32, active: bool, t: f32) -> f32 {
    base + if active { t * BASE_ROTATION_RATE } else { 0.0 }
}

/// Active-mode rotor at animation time `t`: the ORDERED PRODUCT
/// `∏ᵢ exp(planeᵢ · active_plane_angle(base[i], active[i], t))` over the six
/// planes in `Plane4::ALL` order. See the module doc of `active.rs` for why
/// this is a product (independent sliders) rather than a single
/// `exp(sum)` (which would reintroduce BCH coupling). Free function so the
/// composition is testable without constructing a `Demo`.
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

/// True when `row` contains a 120-cell or 600-cell, the two polychora whose
/// SDFs overrun the browser WebGPU shader budget (120 / 600 face hyperplanes
/// each, against the per-pixel Wolfe-greedy projection) and crash the tab.
/// Free function so the gate is unit-testable without a GPU-backed `Demo`;
/// [`Demo::sdf_blocked_by_heavy_polychora`] is the `self.row` specialization.
pub(crate) fn row_blocks_sdf(row: &[ShapeEntry]) -> bool {
    row.iter().any(|e| {
        matches!(
            e.shape,
            RaymarchShape::Polytope(Polytope4::Cell120 | Polytope4::Cell600)
        )
    })
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
    /// Freecam preset (mouse-look + WASD + cursor grab). Drives the
    /// camera in `CameraMode::FreeRoam`; the orbit controller drives it
    /// in `CameraMode::Orbit`. The preset owns its own yaw, pitch,
    /// position, and cursor-grab state internally; the demo reads
    /// `freecam.active()` / `freecam.cursor_grabbed()` rather than
    /// mirroring those flags.
    pub(crate) freecam: Freecam,
    /// Active camera control mode. Default `Orbit` matches the long-
    /// standing behavior; `FreeRoam` is opt-in via the `camera` console
    /// command.
    pub(crate) camera_mode: CameraMode,
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
    /// Pixel width of parent-wireframe edges. Tuneable via `wireframe width <N>`
    /// for thicker lines on screenshots. Default bumped from 1.2 px to 1.8 px
    /// after side-by-side comparison: 1.2 px was too fine to read clearly
    /// against the SDF backdrop on hi-DPI displays.
    pub(crate) wireframe_width_px: f32,
    /// Uniform alpha applied to every parent-wireframe edge when
    /// [`Self::wireframe_nearest_active`] is OFF. Default 1.0 (fully
    /// opaque). Tuneable via `wireframe alpha <N>` for a low-key
    /// "wireframe as background layer" look. Ignored when
    /// `nearest_active` is ON; in that mode the alpha is driven by the
    /// per-cell crossing strength (DIM 0.10 to BRIGHT 0.85) regardless
    /// of this field.
    pub(crate) wireframe_alpha: f32,
    /// Memoized per-edge palette for the `unique-edge` wireframe color
    /// mode, keyed by [`Polytope4`] variant. The palette is a function of
    /// topology alone (greedy graph-coloring on the line graph), so once
    /// computed it stays valid for the process lifetime regardless of
    /// rotor / w_slice / surface scale. Computed on first use; an empty
    /// cache means the demo has never visited `unique-edge` mode for the
    /// row's current shapes.
    pub(crate) unique_edge_palette_cache: HashMap<Polytope4, Vec<[f32; 4]>>,
    /// Runtime multiplier on [`BODY_SIZE`] for all polychora in the row.
    /// Set via `surface scale <N>` (default 1.0). Multiplies wireframe, SDF,
    /// section perimeter, and cross-section cap-fill geometry uniformly so
    /// the slice-of-the-same-shape stays consistent at any scale. Values in
    /// (0, 10] are accepted; the upper bound exists to keep the SDF
    /// marcher's bounded-w-slice assumption intact.
    pub(crate) surface_scale: f32,
    /// Per-fragment alpha for the rasterized section-faces (filled cross-
    /// section cell-caps). Default 1.0 (fully opaque); `surface alpha <N>`
    /// lowers it so the parent wireframe shows through. Independent of
    /// `wireframe_nearest_active` (which only modulates wireframe-edge
    /// alpha, not the surface). Accepted range `(0, 1]`; the lower bound
    /// is open since 0.0 would just hide the surface entirely (use
    /// `surface off` for that).
    pub(crate) surface_alpha: f32,
    /// `y = 0` hyperplane floor visibility (the gridded ground). On by
    /// default; toggled via the `floor` console command. Gated at the
    /// kernel via `u.params[0]` so the toggle is zero-cost: when off, the
    /// scene's halfspace SDF returns a huge distance and the marcher
    /// never converges on the floor, so the checkerboard never paints.
    pub(crate) floor_enabled: bool,
    /// Filled-faces rasterizer for the cross-section of every polychoral body. When
    /// `Self::surface_raster_enabled` is `true`, this replaces the SDF raymarch for the
    /// six regular convex 4-polytopes: the SDF gets `BodyUniform::default()` for those
    /// slots (which the kernel skips) and the section's filled cell-caps come through
    /// here instead. Per-body solid color + face-normal Lambert in the fragment shader.
    pub(crate) section_faces: rye_render::TriangleRasterNode,
    /// Translucent variant of [`Self::section_faces`] with depth-write
    /// disabled. Used in place of `section_faces` when `surface_alpha`
    /// drops below 1.0 so the parent wireframe can show through caps.
    /// Same vertex/fragment shaders + blend state; the only delta is the
    /// `DepthMode::ReadOnly` pipeline-bake.
    pub(crate) section_faces_translucent: rye_render::TriangleRasterNode,
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
    /// Toggle bitmap for the six rotation planes; an active plane participates in the
    /// spin (`rot_time` advances its displayed angle). See [`Plane4::ALL`] for the
    /// index -> plane mapping.
    pub(crate) active: [bool; 6],
    /// User-set baseline angle per plane in radians. Active mode treats the displayed
    /// angle of plane i as `base_angles[i] + rot_time * RATE * active[i]` and composes
    /// `rot_state` as the ORDERED PRODUCT `∏ᵢ exp(planeᵢ · displayed_angle[i])`. This
    /// is the "comprehensive set of rotors" parameterization: each plane is its own
    /// simple-rotation factor in a product instead of a term in a single summed
    /// bivector. The sliders read/write `base_angles` directly, so changing one
    /// slider only mutates that plane's factor; the others stay put. The math
    /// underneath is still BCH-coupled (the product of exps of non-commuting plane
    /// bivectors isn't itself an exp of a sum), but the UI commits to "what the user
    /// set is what we use" instead of trying to read back from `log(rot_state)`,
    /// which has no faithful decomposition into 6 plane angles.
    pub(crate) base_angles: [f32; 6],
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
    /// Hosts the `rye_egui::callout` primitive; future tutorial / explanation
    /// overlays in the playground will instantiate additional `CalloutState`s
    /// the same way.
    pub(crate) example_callout: rye_egui::CalloutState,

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
    ///
    /// Note: Active mode composes its rotor as a *product* (see [`Self::active_rotor`]), so
    /// the returned bivector is only meaningful in the BCH-trivial direction (single
    /// active plane) or as a coarse "this is the direction the rotation is going."
    /// Composer mode uses the sum semantics throughout, so its omega is exact.
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

    /// Angle for plane `i` in Active mode at animation time `t`: the user's
    /// stored baseline plus, if the plane is active, the spin's accumulated
    /// contribution `t * BASE_ROTATION_RATE`. Parameterized over `t` so the
    /// filmstrip can sample future times; [`Self::active_displayed_angle`]
    /// is the `t = rot_time` specialization the sliders read.
    pub(crate) fn active_angle_at(&self, plane_idx: usize, t: f32) -> f32 {
        active_plane_angle(self.base_angles[plane_idx], self.active[plane_idx], t)
    }

    /// Displayed angle for plane `i` in Active mode at the current `rot_time`.
    /// The slider reads this; writing the slider sets `base_angles[i]` so the
    /// new displayed value matches where the user dropped the handle (dragging
    /// a spinning slider doesn't snap back to the pre-drag baseline).
    pub(crate) fn active_displayed_angle(&self, plane_idx: usize) -> f32 {
        self.active_angle_at(plane_idx, self.rot_time)
    }

    /// Active-mode rotor at animation time `t`: ORDERED PRODUCT of per-plane
    /// simple rotations in `Plane4::ALL` order. Sliders are independent because
    /// each `base_angles[i]` contributes to exactly one factor; changing one
    /// slider only mutates one factor in the product. The product across
    /// non-commuting planes is still BCH-coupled in the rotor itself (so
    /// `log(rot_state).component(plane)` won't give back the user-set angles),
    /// but we don't read back through `log` in Active mode -- the sliders are
    /// the source of truth.
    pub(crate) fn active_rotor_at(&self, t: f32) -> Rotor4 {
        compose_active_rotor(&self.base_angles, &self.active, t)
    }

    /// Active-mode rotor at the current `rot_time`. The `t = rot_time`
    /// specialization of [`Self::active_rotor_at`].
    pub(crate) fn active_rotor(&self) -> Rotor4 {
        self.active_rotor_at(self.rot_time)
    }

    /// Orientation rotor at animation time `t`, dispatched on the active
    /// rotation mode. This is the single source of truth for "what does the
    /// orientation look like at animation time `t`" and MUST be used by every
    /// t-scrub and filmstrip-offset site so they agree with the spin path:
    ///
    /// - **Active**: product-of-exp via [`Self::active_rotor_at`]. The 6 plane
    ///   sliders are independent factors; summing them (the Composer model)
    ///   would reintroduce the BCH coupling the Active rework removed.
    /// - **Composer**: `exp(omega_animation * t)`, the sum-of-bivectors model
    ///   the Composer UI is built around.
    ///
    /// For a single active plane the two modes coincide (no BCH coupling); the
    /// distinction only bites with two or more non-commuting active planes,
    /// which is exactly where a naive sum-everywhere t-scrub diverged from the
    /// product-based spin.
    pub(crate) fn rotor_at_time(&self, t: f32) -> Rotor4 {
        match self.rotation_mode {
            RotationMode::Active => self.active_rotor_at(t),
            RotationMode::Composer => (self.omega_animation() * t).exp().normalize(),
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
            self.effective_body_size(),
            rotor,
            entry.body_color,
        )
    }

    /// Effective body radius after the runtime [`Self::surface_scale`]
    /// multiplier. All consumers that previously read `BODY_SIZE` directly
    /// should route through here so the `surface scale` command takes
    /// effect uniformly across SDF, wireframe, section perimeter, and
    /// cross-section caps.
    pub(crate) fn effective_body_size(&self) -> f32 {
        BODY_SIZE * self.surface_scale
    }

    /// True when entering `SurfaceMode::Sdf` would put a 120-cell or
    /// 600-cell into the live SDF kernel. The 120-cell carries 120 face
    /// hyperplanes and the 600-cell carries 600; combined with the
    /// per-pixel Wolfe-greedy projection they exhaust the browser
    /// WebGPU shader budget (Chrome crashed the tab on first attempt).
    /// The console `surface sdf` command and the UI radio gate on this
    /// to keep the demo crash-free.
    pub(crate) fn sdf_blocked_by_heavy_polychora(&self) -> bool {
        row_blocks_sdf(&self.row)
    }

    /// Effective `w` slider half-range after [`Self::surface_scale`]. The
    /// canonical [`crate::consts::W_RANGE`] covers a unit-circumradius
    /// polytope's full w-extent with a small margin; scaling the polytope up
    /// requires the same scaling on the slider so the slice plane still leaves
    /// the body at the extremes (otherwise `surface scale 4.0` would cap the
    /// slider before w reaches the body's hull).
    pub(crate) fn effective_w_range(&self) -> f32 {
        crate::consts::W_RANGE * self.surface_scale
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
        self.base_angles = [0.0; 6];
        self.rot_state = Rotor4::IDENTITY;
        self.rot_time = 0.0;
        self.t_slider_max = T_SLIDER_INITIAL;
        self.draft.clear();
        self.write_all(Rotor4::IDENTITY);
    }
}

#[cfg(test)]
mod tests {
    use super::{active_plane_angle, compose_active_rotor, row_blocks_sdf, BASE_ROTATION_RATE};
    use crate::catalog::ShapeEntry;
    use rye_math::{Bivector, Plane4, Rotor4};
    use rye_physics::polytope::Polytope4;
    use rye_render::raymarch::RaymarchShape;

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
        // One plane (xw = index 2) at a baseline angle, no spin: the product
        // collapses to a single factor, which must equal exp(plane * angle)
        // directly. This is the BCH-trivial case where product == the obvious
        // single rotation.
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
        // xy (index 0) and zw (index 5) are absolutely orthogonal: their
        // bivectors commute, so exp(a*xy) * exp(b*zw) == exp(b*zw) * exp(a*xy).
        // `compose_active_rotor` walks Plane4::ALL order (xy before zw); build
        // the reverse by hand and confirm they match. (This is the case where
        // the product genuinely equals exp(sum); the non-commuting case is
        // exactly why Active mode uses the product, tested implicitly by the
        // unit-norm + single-plane invariants above.)
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
}
