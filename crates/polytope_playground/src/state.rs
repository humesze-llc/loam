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
use rye_math::{Bivector, Bivector4, EuclideanR3, Plane4, Projection, Rotor, Rotor4};
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
    /// Single-shape inspection: exactly one [`ShapeEntry`] (the `strip_subject`,
    /// independent of the row) rendered at one `w_slice` with the full
    /// surface / wireframe / projection / points stack. Shares the Shapes-view
    /// render path verbatim, only over a one-element row
    /// ([`Demo::render_row`]), so every overlay affordance carries over.
    ///
    /// REQUIRED for Schlegel: a cell-index boundary selection is meaningless
    /// across a row of different polytopes (a 5-cell has 5 cells, a 600-cell
    /// has 600), so the unambiguous single subject is what makes the
    /// cell-index stepper well-defined. Stereographic reads far better on one
    /// body too. Stereographic / Hyperslice still work in [`Self::Shapes`]
    /// (they are per-vertex maps that apply to every body uniformly); only
    /// Schlegel's cell-index strictly needs Single.
    Single,
    /// Single-shape filmstrip: one [`ShapeEntry`] (independent of the row) rendered N
    /// times across evenly-spaced `w_slice` values around the slider's current `w`.
    /// Order of the scene's row is irrelevant in this mode; the row UI is hidden
    /// entirely.
    Filmstrip,
}

/// The slice of [`ShapeEntry`]s the scene actually renders for `view_mode`, the
/// single source of truth every per-body render path and the SDF body upload
/// reads. [`ViewMode::Single`] yields exactly the `strip_subject` (a one-element
/// borrow of the subject, no allocation); every other mode renders the full
/// `row`. [`ViewMode::Filmstrip`] also draws only `strip_subject`, but through
/// its own per-cell grid path in `render`, so it is not a caller here and falls
/// through to the row arm without effect.
///
/// Free function (no `&self`) so the row-selection invariant is unit-testable
/// without a GPU-backed [`Demo`]; [`Demo::render_row`] is the one caller.
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

/// One overlaid layer of the rasterized cross-section: a perimeter-outline
/// toggle plus a surface-fill alpha that doubles as the layer's on/off switch
/// (`0.0` draws no fill). The slice geometry is identical for both layers (the
/// honest drop-w 3-flat cut); the layers differ only in how that geometry maps
/// to R³ for display. See [`Demo::cross_section`] / [`Demo::projected_cap`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct SectionLayer {
    /// Whether this layer's cap-boundary perimeter outline renders. Independent
    /// of [`Self::surface_alpha`]: a user can show the outline with no fill, or a
    /// fill with no outline.
    pub(crate) perimeter: bool,
    /// Per-fragment fill alpha in `[0, 1]`. `0.0` is the layer's off state (no
    /// fill submitted); `(0, 1)` renders through the depth-write-disabled
    /// translucent pipeline so layers behind composite through; `1.0` renders
    /// opaque with depth-write. See [`Demo::section_faces`] /
    /// [`Demo::section_faces_translucent`].
    pub(crate) surface_alpha: f32,
}

impl SectionLayer {
    /// Whether the surface fill draws at all (`surface_alpha > 0`). Below this an
    /// alpha-zero fill is fully transparent, so the layer skips its triangle pass
    /// entirely rather than submitting an invisible mesh.
    pub(crate) fn fill_visible(self) -> bool {
        self.surface_alpha > 0.0
    }
}

/// Default surface-fill alpha for the honest cross-section layer: visible but
/// slightly translucent so the parent wireframe (when enabled) reads through the
/// cap without a separate `surface alpha` command. "Full-ish" rather than fully
/// opaque per the layer's default-on role as the always-honest slice.
const CROSS_SECTION_DEFAULT_ALPHA: f32 = 0.85;

impl SectionLayer {
    /// The honest cross-section's default: perimeter + fill on at a full-ish
    /// alpha, so selecting a distorting wireframe projection never silently
    /// reshapes the slice the user reads as "the cross-section."
    pub(crate) const CROSS_SECTION_DEFAULT: SectionLayer = SectionLayer {
        perimeter: true,
        surface_alpha: CROSS_SECTION_DEFAULT_ALPHA,
    };
    /// The projected-cap's default: fully off. It reprojects the slice through
    /// the active wireframe projection, which is opt-in (the user asks for it to
    /// sit the cap on a Schlegel / stereographic wireframe).
    pub(crate) const PROJECTED_CAP_DEFAULT: SectionLayer = SectionLayer {
        perimeter: false,
        surface_alpha: 0.0,
    };
}

/// The 4D->R³ projection a section layer renders its slice through. The honest
/// cross-section is ALWAYS drop-w ([`rye_math::Projection::Identity`]) regardless
/// of the active wireframe projection: the slice IS a 3-flat and drop-w is the
/// inhabitant's undistorted view of it, the same geometry the SDF raymarch shows.
/// The projected cap follows the active wireframe `projection`, so it can sit on a
/// Schlegel / stereographic wireframe.
///
/// Free function (no `&Demo`) so the "honest layer ignores the projection,
/// projected layer follows it" invariant is unit-testable without a GPU-backed
/// [`Demo`]; the two section render paths in `main.rs` are the callers.
pub(crate) fn section_layer_projection(
    is_cross_section: bool,
    projection: Projection<4>,
) -> Projection<4> {
    if is_cross_section {
        Projection::Identity
    } else {
        projection
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
    /// 4D Schlegel diagram: central projection from a viewpoint just outside the
    /// chosen boundary cell onto that cell's bounding 3-flat (Coxeter, *Regular
    /// Polytopes*, ch. 13). The chosen cell becomes the diagram's outer boundary;
    /// every other cell nests inside it. `cell_index` selects which of the
    /// polytope's cells is the boundary, in the canonical [`Polytope4::topology`]
    /// cell order; it is clamped to the polytope's cell count at resolve time.
    ///
    /// The variant carries only the index. The `cell_normal` / `cell_offset` /
    /// `viewpoint_distance` scalars that [`rye_math::Projection::Schlegel`] needs
    /// are resolved from the *selected polytope's* topology (cell centroids via
    /// [`Polytope4::face_planes`], the CORRECT path, NOT the buggy dual-vertex
    /// `cell{120,600}_face_planes`) and cached on [`Demo::schlegel_params`] at
    /// cell-select time, never inside the per-frame upload. See
    /// [`resolve_schlegel_params`] and [`Demo::resolved_wireframe_projection`].
    Schlegel {
        /// Index of the boundary cell in the polytope's canonical cell order.
        cell_index: u32,
    },
    /// Conformal stereographic projection of the polytope from S³ (where its
    /// unit-circumradius vertices live) to R³, casting away from a configurable
    /// pole (default [`STEREOGRAPHIC_DEFAULT_POLE`], a cell-center direction; live
    /// value is [`Demo::stereographic_pole`]). Angle-preserving, distance-
    /// distorting: the cell facing the pole balloons to the outer boundary and the
    /// opposite cell shrinks to the interior, the same nesting Schlegel shows but
    /// with the round, angle-faithful look that pairs with spherical-space mode.
    /// The `EuclideanR4` projection normalizes each vertex onto S³ first, so this
    /// reads correctly for the demo's `BODY_SIZE`-scaled vertices.
    Stereographic,
    /// Drop-w projection paired with a demo-side CELL-level w-range cull (the
    /// `wireframe_hyperslice` filter): the wireframe thins to the edges that
    /// belong to a cell whose body-local w-range overlaps a slab around
    /// `w_slice`. The cull is cell-level (not edge-level) so a kept edge agrees
    /// with the cell-level active-edge coloring and the cross-section: a far-side
    /// edge of a sliced cell stays because its cell is being cut, even though its
    /// own endpoints sit outside the slab. The projection itself stays drop-w
    /// ([`rye_math::Projection::Identity`]); the slicing is done by the cull in
    /// the wireframe builder, not by the projection (a projection has already
    /// discarded w and cannot honestly carry a keep/drop signal). Selecting this
    /// mode turns the cull on; the slab thickness is the existing
    /// [`Demo::wireframe_hyperslice_thickness`] control.
    Hyperslice,
}

/// Resolved canonical (unit-circumradius) Schlegel parameters for one
/// `(polytope, cell_index)` choice. Cached on [`Demo::schlegel_params`] so the
/// O(V·D³) `LazyLock` cell-table fit behind [`Polytope4::face_planes`] runs once
/// per cell selection rather than once per frame.
///
/// Stored in CANONICAL coordinates (the polytope's unit-circumradius topology):
/// the per-frame [`Demo::resolved_wireframe_projection`] scales `cell_offset` and
/// `viewpoint_distance` by the live [`Demo::effective_body_size`] and rotates
/// `cell_normal` by the live `rot_state`. Caching canonical (not body-scaled)
/// params keeps the cache valid across `surface scale` changes and is what lets
/// the chosen cell stay the outer boundary as the polytope spins.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SchlegelParams {
    /// The polytope these params were resolved against. The cache is invalid if
    /// the selected polytope changes (row edit, different first polychoron).
    pub(crate) polytope: Polytope4,
    /// The (already-clamped) boundary cell index.
    pub(crate) cell_index: u32,
    /// Outward unit normal of the chosen cell's hyperplane, in CANONICAL coords.
    /// Per-frame rotated by `rot_state`; rotation preserves unit length.
    pub(crate) cell_normal: glam::Vec4,
    /// Signed plane offset in CANONICAL coords (the cell's inradius): the chosen
    /// cell lies in `{x : dot(cell_normal, x) = cell_offset}`. Per-frame scaled by
    /// `effective_body_size`.
    pub(crate) cell_offset: f32,
    /// Eye distance along `cell_normal` from the origin, in CANONICAL coords. The
    /// chosen cell's farthest vertex along `+normal` plus a fixed margin, so the
    /// eye sits just outside the cell for every polytope (a blanket `1.5 *
    /// cell_offset` crowds the eye against the small-inradius 5-cell). Per-frame
    /// scaled by `effective_body_size`.
    pub(crate) viewpoint_distance: f32,
}

/// Additive eye clearance beyond the chosen cell's far edge, in canonical
/// (unit-circumradius) units. Fixed and additive rather than a multiple of the
/// cell offset so the absolute clearance does not collapse for small-inradius
/// polytopes: the 5-cell's inradius is 0.25, where a `1.5 * cell_offset`
/// viewpoint leaves the eye only 0.125 clear of the cell and the diagram folds
/// to a near-degenerate sliver. 0.5 of the unit circumradius keeps the eye a
/// comfortable, polytope-independent distance outside the boundary cell.
const SCHLEGEL_EYE_MARGIN: f32 = 0.5;

/// Resolve the canonical Schlegel parameters for the chosen `(polytope,
/// cell_index)`. `cell_index` is clamped to `[0, cell_count - 1]` so an
/// out-of-range index never panics or indexes out of bounds.
///
/// The cell normal and inradius come from [`Polytope4::face_planes`], which
/// derives them from cell centroids via topology (Coxeter, *Regular Polytopes*,
/// ch. 13: a regular polytope's cell centroid lies along the outward face normal
/// at the inradius). This is the CORRECT path: the dual-vertex
/// `cell{120,600}_face_planes` helpers in `rye_physics::euclidean_r4` are wrong
/// for 96 of the 120/600-cell normals (the documented BUG) and are deliberately
/// NOT used here.
///
/// `viewpoint_distance` is the chosen cell's farthest vertex-projection along
/// `+cell_normal` plus [`SCHLEGEL_EYE_MARGIN`], placing the eye just outside the
/// boundary cell for any polytope. The result is in canonical coordinates; the
/// caller scales it by the live body size.
pub(crate) fn resolve_schlegel_params(polytope: Polytope4, cell_index: u32) -> SchlegelParams {
    let cell_count = polytope.cell_count() as u32;
    // `cell_count >= 1` for every polytope, so `cell_count - 1` never underflows.
    let clamped = cell_index.min(cell_count - 1);
    let (normals, cell_offset) = polytope.face_planes();
    let cell_normal = normals[clamped as usize];
    // Farthest vertex-projection along the outward normal. For the chosen cell
    // this is its own boundary plane (`= cell_offset`), but compute it over all
    // vertices so the eye clearance is robust to any topology quirk rather than
    // assuming the cell vertices are the extreme set.
    let max_dot = polytope
        .topology()
        .vertices
        .iter()
        .map(|v| cell_normal.dot(*v))
        .fold(f32::NEG_INFINITY, f32::max);
    SchlegelParams {
        polytope,
        cell_index: clamped,
        cell_normal,
        cell_offset,
        viewpoint_distance: max_dot + SCHLEGEL_EYE_MARGIN,
    }
}

/// Resolve `projection` against the current Schlegel `subject` (the leading
/// polychoron, or `None` for a row with no polytope), returning the projection
/// to store plus the cache to attach.
///
/// For a `Schlegel` mode with a subject this returns a projection whose
/// `cell_index` is the SAME clamped value the cache carries: [`resolve_schlegel_params`]
/// clamps an out-of-range index (a row edit can shrink the subject from a
/// 600-cell to a 5-cell while the mode still names `cell_index: 300`), and the
/// returned projection is rewritten to that clamp so the enum, the cache, the UI
/// cell-index stepper, and the console's "schlegel (cell N)" report never
/// disagree about which cell is the diagram's boundary. Without the rewrite the
/// projection rendered correctly (it reads the clamped cache) but the stepper
/// showed an out-of-range index and the report named a cell the diagram did not
/// use. Every other mode (and `Schlegel` with no subject) passes the projection
/// through unchanged and clears the cache.
///
/// Pure (no `&mut self`) so the index-sync invariant is unit-testable without a
/// GPU-backed [`Demo`]; [`Demo::resolve_schlegel_cache`] is the one caller.
pub(crate) fn synced_schlegel_projection(
    projection: WireframeProjection,
    subject: Option<Polytope4>,
) -> (WireframeProjection, Option<SchlegelParams>) {
    match (projection, subject) {
        (WireframeProjection::Schlegel { cell_index }, Some(polytope)) => {
            let params = resolve_schlegel_params(polytope, cell_index);
            let synced = WireframeProjection::Schlegel {
                cell_index: params.cell_index,
            };
            (synced, Some(params))
        }
        (other, _) => (other, None),
    }
}

/// A short educational annotation for the active projection / space-mode
/// combination: a `title` for the callout window plus a `body` of one to three
/// sentences explaining what the user is looking at. Surfaced via the
/// `rye_egui::callout` primitive (see [`Demo::render_mode_annotation`]) so a
/// reader can understand each non-default mode without opening the source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ModeAnnotation {
    /// Callout window title: the mode's short name.
    pub(crate) title: &'static str,
    /// One to three sentences. Composed from the projection's own explanation
    /// plus, when spherical curvature is active, a curvature sentence and (for
    /// the flat-cap overlap) a disambiguating note.
    pub(crate) body: String,
}

/// The `space_blend` argument above this counts as "spherical curvature is on"
/// for the annotation: any visible bow of the edges toward S³ warrants the
/// curvature sentence. Matches the wireframe builder's own `blend <= 0.0` flat
/// fast-path cutoff (the per-edge morph in the demo's `push_blended_edge`), so
/// the curvature sentence appears for exactly the blend values that bow the
/// edges. The wireframe-overlay-enabled gate is the caller's responsibility:
/// `Demo::render_mode_annotation` passes `0.0` here while the overlay is off,
/// since the bowed edges render only in that overlay, so this threshold is the
/// last condition rather than the only one.
const SPHERICAL_ANNOTATION_THRESHOLD: f32 = 0.0;

/// Educational annotation for the active `(projection, space curvature, flat-cap)`
/// combination, or `None` when the scene is in its plain default state (drop-w
/// projection AND flat space): there is nothing non-obvious to explain, so no
/// callout is shown. Every other combination returns `Some` with distinct,
/// non-empty copy.
///
/// `flat_cap_drawn` is `true` when the rasterized cross-section caps are being
/// drawn (`SurfaceMode::Raster`). It only affects the body text in the THREE-WAY
/// overlap case (spherical curvature on, with a flat cap and/or Stereographic),
/// where the cap stays a flat 3-flat slice while the edges bow onto S³; the note
/// tells the user the flat cap is expected (a curved cross-section is future
/// work) so a flat cap under curved edges does not read as a bug.
///
/// Pure (no `&Demo`) so the `(mode) -> copy` mapping is unit-testable without a
/// GPU-backed [`Demo`]; [`Demo::render_mode_annotation`] is the one caller.
///
/// Projection explanations follow Coxeter, *Regular Polytopes*, ch. 13
/// (Schlegel diagrams) and the standard conformal stereographic map
/// `(x, y, z) / (1 - w)` (Wikipedia, "Stereographic projection").
pub(crate) fn mode_annotation(
    projection: WireframeProjection,
    space_blend: f32,
    flat_cap_drawn: bool,
) -> Option<ModeAnnotation> {
    let spherical = space_blend > SPHERICAL_ANNOTATION_THRESHOLD;

    // Per-projection lead sentence. `None` for drop-w, the default projection:
    // it has no distortion to explain, so a drop-w + flat scene shows nothing.
    let (title, projection_body): (&'static str, Option<&str>) = match projection {
        WireframeProjection::DropW => ("Drop-w", None),
        WireframeProjection::WDepth => (
            "W-depth perspective",
            Some(
                "4D pinhole perspective from a viewer down the w-axis: the +w face \
                 projects to the outer shape and the -w face to the inner one, the \
                 classic cube-within-a-cube view of the tesseract.",
            ),
        ),
        WireframeProjection::Schlegel { .. } => (
            "Schlegel diagram",
            Some(
                "Central projection from a viewpoint just outside one chosen cell \
                 onto that cell's hyperplane (Coxeter, Regular Polytopes, ch. 13): \
                 the chosen cell becomes the outer boundary and every other cell \
                 nests inside it.",
            ),
        ),
        WireframeProjection::Stereographic => (
            "Stereographic projection",
            Some(
                "Conformal S^3 -> R^3 map, (x, y, z) / (1 - w): angles are preserved \
                 but distances are not, so the polytope distorts globally and the \
                 cell facing the +w pole balloons outward as its vertices approach \
                 the pole.",
            ),
        ),
        WireframeProjection::Hyperslice => (
            "Hyperslice",
            Some(
                "Shows only the edges of cells the current 4D cut passes through: \
                 an edge survives when a cell containing it has its w-range within \
                 the thin slab around the w-slice, so the wireframe thins to the \
                 cells being sliced.",
            ),
        ),
    };

    // The default scene (drop-w, flat space) has nothing to annotate.
    let Some(projection_lead) = projection_body else {
        if !spherical {
            return None;
        }
        // Spherical curvature on but the projection is plain drop-w: the
        // curvature is the only thing to explain, so the callout is titled for
        // the space mode rather than the projection.
        return Some(ModeAnnotation {
            title: "Spherical space",
            body: spherical_sentence(flat_cap_drawn, projection).to_string(),
        });
    };

    let mut body = projection_lead.to_string();
    if spherical {
        body.push(' ');
        body.push_str(spherical_sentence(flat_cap_drawn, projection));
    }
    Some(ModeAnnotation { title, body })
}

/// The `space_blend` value the annotation should describe, given the raw blend
/// and whether the wireframe overlay is on. The flat-to-spherical edge morph
/// lives entirely in the wireframe builder (`push_blended_edge`, reached only
/// from `Demo::render_wireframe_overlay`), which the frame loop skips while the
/// overlay is disabled. With the overlay off there is no bowed edge on screen,
/// so the annotation must not claim curvature: report the blend as `0.0` so
/// [`mode_annotation`] drops the spherical sentence (or the whole "Spherical
/// space" callout, for plain drop-w). The projection annotation is unaffected
/// because the rasterized section caps reproject through the same projection
/// whether or not the wireframe overlay draws.
///
/// Split out so the "no curvature claim without a visible bow" invariant is
/// unit-testable without a GPU-backed [`Demo`]; [`Demo::render_mode_annotation`]
/// is the one caller.
pub(crate) fn annotation_effective_blend(space_blend: f32, wireframe_enabled: bool) -> f32 {
    if wireframe_enabled {
        space_blend
    } else {
        0.0
    }
}

/// The spherical-curvature sentence appended when `space_blend > 0`, plus the
/// three-way-overlap note when a flat cross-section cap and/or Stereographic is
/// also in play. The flat cap stays a flat 3-flat slice while the edges bow onto
/// S^3; the note flags that as expected (curved cross-sections are future work)
/// so the user does not read the flat cap under curved edges as a bug.
fn spherical_sentence(flat_cap_drawn: bool, projection: WireframeProjection) -> &'static str {
    let stereographic = matches!(projection, WireframeProjection::Stereographic);
    if flat_cap_drawn || stereographic {
        "Spherical space bows the edges onto S^3 great-circle arcs, so the \
         polytope curves; the filled cross-section cap stays flat (a flat 3-flat \
         slice), which is expected here -- a curved cross-section is not yet \
         implemented."
    } else {
        "Spherical space bows the edges onto S^3 great-circle arcs, so the \
         polytope curves rather than rendering as straight chords."
    }
}

/// Default pole for the Stereographic wireframe projection: the unit
/// direction `(1, 1, 1, 1)/2`, a cell-center direction of the 16-cell (and of
/// the tesseract and 24-cell). Casting away from a cell center, rather than the
/// `+w` axis, keeps an *axis-aligned* polytope's vertices off the pole: the
/// 16-cell's vertices are the `±e_i` axes (Coxeter, *Regular Polytopes*, §8.2),
/// and a cell centroid lies along the outward face normal between four such
/// axes, so no 16-cell vertex sits on this pole and a pure `xw` rotation (which
/// fixes `y, z`) can never sweep one onto it. That removes the common-case
/// 16-cell pole flicker the `+w` pole exhibits.
///
/// Tradeoff, by 16-cell / tesseract duality: this direction *is* a tesseract
/// vertex direction (the tesseract vertex `(½,½,½,½)`), so under this pole a
/// tesseract presents a vertex toward the pole at rest, where the `+w` pole
/// presented a cell. A single pole cannot avoid every polytope's vertices in a
/// mixed row; this constant trades the tesseract's static look for the
/// 16-cell's continuous-rotation robustness, and the pole stays configurable
/// (see [`Demo::stereographic_pole`]). Written as the exact, exactly-unit f32
/// literal `0.5` per lane so no runtime `normalize` is needed and the constant
/// is bit-reproducible.
pub(crate) const STEREOGRAPHIC_DEFAULT_POLE: glam::Vec4 = glam::Vec4::new(0.5, 0.5, 0.5, 0.5);

impl WireframeProjection {
    /// Parse the console-arg spelling. Hyphens because the console grammar lexes on
    /// whitespace and `drop-w` / `w-depth` read as single tokens. `schlegel` parses
    /// to `cell_index = 0`; the console handler reads the trailing `<cell-index>`
    /// token separately (the grammar carries it as its own positional arg).
    pub(crate) fn from_token(token: &str) -> Option<Self> {
        match token {
            "drop-w" => Some(WireframeProjection::DropW),
            "w-depth" => Some(WireframeProjection::WDepth),
            "schlegel" => Some(WireframeProjection::Schlegel { cell_index: 0 }),
            "stereographic" => Some(WireframeProjection::Stereographic),
            "hyperslice" => Some(WireframeProjection::Hyperslice),
            _ => None,
        }
    }

    /// Cycle order for the bare `wireframe perspective` console command and the
    /// UI radio: drop-w -> w-depth -> schlegel -> stereographic -> hyperslice ->
    /// drop-w. Schlegel cycles in at `cell_index = 0`; the cell-index stepper then
    /// picks the boundary cell.
    pub(crate) const ALL: [Self; 5] = [
        WireframeProjection::DropW,
        WireframeProjection::WDepth,
        WireframeProjection::Schlegel { cell_index: 0 },
        WireframeProjection::Stereographic,
        WireframeProjection::Hyperslice,
    ];

    /// Display label for the egui projection radio.
    pub(crate) fn label(self) -> &'static str {
        match self {
            WireframeProjection::DropW => "Drop-w",
            WireframeProjection::WDepth => "W-depth",
            WireframeProjection::Schlegel { .. } => "Schlegel",
            WireframeProjection::Stereographic => "Stereographic",
            WireframeProjection::Hyperslice => "Hyperslice",
        }
    }

    /// Whether two modes are the same VARIANT, ignoring the Schlegel cell index.
    /// The radio compares modes by variant (a cell-index change must not deselect
    /// the Schlegel button); `PartialEq` on the enum would treat
    /// `Schlegel { cell_index: 1 }` as a different button than
    /// `Schlegel { cell_index: 2 }`.
    pub(crate) fn same_variant(self, other: Self) -> bool {
        std::mem::discriminant(&self) == std::mem::discriminant(&other)
    }

    /// Context-free resolution to a [`rye_math::Projection<4>`]. Handles the modes
    /// that need no polytope or rotor context:
    /// - `DropW` -> `Identity` (drop-w),
    /// - `WDepth` -> `Perspective4D { focal_distance: 2.0 }` (focal sized to clear
    ///   the unit-circumradius polytope's `BODY_SIZE`-scaled w-extent so the
    ///   denominator never nears zero),
    /// - `Stereographic` -> `Stereographic { pole: STEREOGRAPHIC_DEFAULT_POLE }`
    ///   (a cell-center pole, off every 16-cell vertex; see the constant). The
    ///   live pole is [`Demo::stereographic_pole`], which
    ///   [`Demo::resolved_wireframe_projection`] substitutes; this context-free
    ///   resolution returns the default so a non-render caller still gets the
    ///   documented pole.
    /// - `Hyperslice` -> `Identity` (the projection stays drop-w; the demo-side
    ///   `wireframe_hyperslice` filter does the slicing).
    ///
    /// `Schlegel` returns `Identity` here as a SAFE FALLBACK only: the real
    /// Schlegel projection needs the selected polytope's cached
    /// [`SchlegelParams`] plus the live `rot_state`, which the context-free enum
    /// cannot supply. [`Demo::resolved_wireframe_projection`] is the per-frame
    /// entry point that builds the actual `Schlegel` variant; the four render
    /// sites call it, not this.
    pub(crate) fn to_projection(self) -> rye_math::Projection<4> {
        match self {
            WireframeProjection::DropW | WireframeProjection::Hyperslice => {
                rye_math::Projection::Identity
            }
            WireframeProjection::WDepth => rye_math::Projection::Perspective4D {
                focal_distance: 2.0,
            },
            // Schlegel needs cached params + rotor (see doc above); fall back to
            // drop-w until `Demo::resolved_wireframe_projection` resolves it.
            WireframeProjection::Schlegel { .. } => rye_math::Projection::Identity,
            WireframeProjection::Stereographic => rye_math::Projection::Stereographic {
                pole: STEREOGRAPHIC_DEFAULT_POLE,
            },
        }
    }
}

/// Whether the wireframe Hyperslice cull is active, given the standalone toggle
/// and the selected projection mode. The cull runs when EITHER the
/// [`Demo::wireframe_hyperslice`] toggle is on OR the projection is
/// [`WireframeProjection::Hyperslice`] (which resolves to drop-w and lets the
/// cull do the slicing). Both the wireframe builder (the cull's `continue`) and
/// the Render-modal slab-width control gate on this same predicate, so the
/// thickness stepper is reachable in exactly the cases where it has an effect:
/// gating the control on the bare toggle alone left it greyed out under the
/// Hyperslice projection even though the cull was running.
pub(crate) fn hyperslice_cull_active(toggle: bool, projection: WireframeProjection) -> bool {
    toggle || matches!(projection, WireframeProjection::Hyperslice)
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
    /// The honest cross-section layer: the drop-w slice 3-flat, NEVER reprojected
    /// through [`Self::wireframe_projection`]. This is the same geometry the SDF
    /// raymarch shows; the raster version renders it drop-w regardless of the
    /// active projection so selecting Schlegel / stereographic never silently
    /// distorts the slice the user reads as "the cross-section." On by default
    /// (perimeter + a full-ish fill). See [`SectionLayer`].
    pub(crate) cross_section: SectionLayer,
    /// The projected-cap layer: the same slice reprojected through the active
    /// [`Self::wireframe_projection`] (the `cap_vertex_projected_and_world` /
    /// `perspective_scale_at_w` behavior), so the cap can sit on a Schlegel /
    /// stereographic wireframe. Off by default; opt-in for inspecting the slice
    /// in the wireframe's own projected frame. Overlaid in the same viewport as
    /// [`Self::cross_section`]. See [`SectionLayer`].
    pub(crate) projected_cap: SectionLayer,
    /// Base RGB for wireframe edges. Orthogonal to [`Self::wireframe_nearest_active`]:
    /// the color mode picks the hue, the nearest-active toggle then modulates alpha on
    /// top.
    pub(crate) wireframe_color_mode: WireframeColorMode,
    /// How the parent wireframe's 4D vertex positions project to R³. The cross-section
    /// always uses drop-w (mathematically the inhabitant's view of the slice 3-flat);
    /// this toggle only affects the dim wireframe overlay on top.
    pub(crate) wireframe_projection: WireframeProjection,
    /// Cached canonical Schlegel parameters for the current `(selected polytope,
    /// cell_index)`. `Some` only while `wireframe_projection` is `Schlegel` and the
    /// row has a polychoron to project; `None` otherwise. Resolved at cell-select
    /// time via [`Demo::resolve_schlegel_cache`] (console + UI both call it), never
    /// inside the per-frame upload: [`Polytope4::face_planes`] runs a `LazyLock`
    /// O(V·D³) cell-table fit on first access that must not land on the hot path.
    /// The per-frame [`Demo::resolved_wireframe_projection`] only rotates the
    /// cached normal and scales the offsets.
    pub(crate) schlegel_params: Option<SchlegelParams>,
    /// Live pole for the Stereographic wireframe projection, the unit `Vec4` the
    /// conformal map casts away from. Defaults to [`STEREOGRAPHIC_DEFAULT_POLE`]
    /// (a cell-center direction, off every 16-cell vertex; see the constant for
    /// why and the tesseract tradeoff). Kept as a field rather than baked into the
    /// payload-free [`WireframeProjection::Stereographic`] variant so the enum
    /// stays a plain marker across `ALL` / `from_token` / `same_variant` / the UI
    /// radio; [`Self::resolved_wireframe_projection`] substitutes it per frame.
    /// Console-settable via the `wireframe` command.
    pub(crate) stereographic_pole: glam::Vec4,
    /// Wireframe Hyperslice toggle. When `true`, the parent wireframe is culled
    /// to only the edges belonging to a cell whose body-local 4D w-range
    /// intersects a slab of width [`Self::wireframe_hyperslice_thickness`]
    /// centered on `w_slice`, so the graph thins to "the edges of the cells the
    /// current 4D cut passes through." Off by default; the demo's identity is the
    /// full wireframe + cross-section composition. This is a CPU-side cell-level
    /// filter in the wireframe builder (cell-level so it agrees with the
    /// active-edge coloring and the cross-section), independent of (and
    /// composable with) the SDF raymarch's own w-slice and the cyan section
    /// perimeter; all three slice against the same `w_slice`.
    pub(crate) wireframe_hyperslice: bool,
    /// Full width of the wireframe Hyperslice slab (see
    /// [`Self::wireframe_hyperslice`]). An edge survives iff a cell containing
    /// both its endpoints has a w-range intersecting `[w_slice - t/2, w_slice +
    /// t/2]`. Floored at [`crate::consts::HYPERSLICE_MIN_THICKNESS`] at the test
    /// site so a 0 here degrades to "only edges of cells straddling `w_slice`
    /// survive" rather than an exact-equality test that never fires. Default
    /// [`crate::consts::HYPERSLICE_DEFAULT_THICKNESS`].
    pub(crate) wireframe_hyperslice_thickness: f32,
    /// Wireframe-edge geometry morph in `[0, 1]`: `0.0` draws straight chords
    /// in R⁴ (flat, `EuclideanR4` lerp); `1.0` draws great-circle arcs on S³
    /// (`SphericalS3Embedded` slerp); values between linearly blend the two
    /// per tessellation sample. The polytope's unit-circumradius vertices
    /// already lie on S³, so only the edge interiors move; endpoints are
    /// shared. Set via the `space` console command. At exactly `0.0` the
    /// wireframe takes a one-segment-per-edge fast path (no tessellation).
    pub(crate) space_blend: f32,
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
    /// disabled. Used in place of `section_faces` when a section layer's
    /// `surface_alpha` drops below 1.0 so the parent wireframe (and any
    /// layer drawn behind) can show through caps. Same vertex/fragment
    /// shaders + blend state; the only delta is the `DepthMode::ReadOnly`
    /// pipeline-bake. Both section layers (honest cross-section + projected
    /// cap) route through this pair of nodes, picking opaque vs translucent
    /// per layer alpha; each layer's pass is a self-contained submit, so the
    /// two nodes are reused across the two layers within one frame.
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
    /// Combined-mesh scratch for the honest drop-w cross-section layer.
    pub(crate) section_faces_mesh_scratch: rye_shape::TriangleMesh<3>,
    /// Combined-mesh scratch for the projected-cap layer. Separate from
    /// [`Self::section_faces_mesh_scratch`] so both section layers can be built
    /// in one pass over the row (the body-local 4D vertices are shared) without
    /// either layer's mesh clobbering the other's reused allocation.
    pub(crate) section_faces_projected_scratch: rye_shape::TriangleMesh<3>,
    /// Per-vertex body-local projected points for the cap-fill near-pole clip,
    /// reused across frames + bodies inside `build_section_layer_meshes` to avoid
    /// a per-body allocation. Holds the pre-translate projected point of each
    /// freshly-appended cap vertex so the triangle-granularity Stereographic drop
    /// (drop a fill triangle when any of its three projected vertices is past the
    /// clip radius) reuses the same `sample_in_radius` predicate the wireframe and
    /// cap-perimeter outline use, keeping fill and outline culling in lockstep.
    pub(crate) section_clip_projected_scratch: Vec<glam::Vec3>,
    /// Reused buffer for per-frame body-uniform uploads (see
    /// `upload_render_row_bodies`); kept to avoid a per-frame allocation on
    /// the steady-state spin path.
    pub(crate) body_uniform_scratch: Vec<BodyUniform>,
    /// Reused great-circle sampling buffer for `push_blended_edge`; taken via
    /// `mem::take` during the wireframe-overlay build and put back after, so the
    /// curved-mode (`space_blend > 0`) path does not allocate per frame.
    pub(crate) slerp_scratch: Vec<glam::Vec4>,
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

    /// Persistent state for the per-mode educational annotation callout: a short
    /// floating explanation of the active projection / space mode, anchored to
    /// the leading polychoron. Its text is the pure [`mode_annotation`] mapping
    /// reprojected each frame; the callout only draws when that mapping returns
    /// `Some` (a non-default projection or spherical curvature is active) AND
    /// this flag is on. On by default so a first-time user who switches to
    /// Schlegel / Stereographic / Hyperslice or turns up Curvature immediately
    /// sees what the mode does; toggle via `View > Mode annotation`.
    pub(crate) mode_annotation_open: rye_egui::CalloutState,

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
    fn sdf_body_for_slot(
        &self,
        entry: &ShapeEntry,
        slot: usize,
        n: usize,
        rotor: Rotor4,
    ) -> BodyUniform {
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
    ///
    /// Tests the RENDERED row (see [`Self::render_row`]), not the stored
    /// `self.row`: the SDF kernel only ever uploads what `write_all` /
    /// `rebuild_bodies` emit, which in [`ViewMode::Single`] is the lone
    /// `strip_subject`, independent of the multi-shape row. Gating on
    /// `self.row` would both falsely block SDF on a light Single subject
    /// (because a hidden row member is heavy) and, worse, falsely ALLOW it
    /// for a heavy Single subject sitting over an all-light row, which would
    /// hand the crash-prone 120/600-cell SDF straight to the kernel. This
    /// matches the heavy-shape warning `render_single_body` already shows
    /// against `strip_subject`.
    pub(crate) fn sdf_blocked_by_heavy_polychora(&self) -> bool {
        row_blocks_sdf(self.render_row())
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

    /// The slice of [`ShapeEntry`]s the scene renders this frame. In
    /// [`ViewMode::Single`] this is exactly the `strip_subject`; otherwise the
    /// full `row`. Every per-body render path (section faces, wireframe overlay,
    /// points) and the SDF body upload (`write_all` / `rebuild_bodies`) reads
    /// this, so Single mode draws one body without disturbing the user's row.
    /// See [`render_row_entries`].
    pub(crate) fn render_row(&self) -> &[ShapeEntry] {
        render_row_entries(self.view_mode, &self.row, &self.strip_subject)
    }

    /// The polytope a Schlegel cell index refers to: the first polychoron in the
    /// rendered row, the same "representative shape" convention the example
    /// callout anchors to. A Schlegel cell index has no single meaning across a
    /// row of different polytopes (a 5-cell has 5 cells, a 600-cell has 600), so
    /// the diagram picks the leading polychoron and projects through its chosen
    /// cell's frame. In [`ViewMode::Single`] the rendered row is exactly the
    /// `strip_subject`, so the cell-index bound is that subject's cell count
    /// (the unambiguous selection Schlegel needs).
    pub(crate) fn schlegel_subject(&self) -> Option<Polytope4> {
        self.render_row().iter().find_map(|e| e.shape.polytope4())
    }

    /// Resolve and cache the canonical Schlegel parameters for the current
    /// projection mode + selected polytope. Call this at every cell-SELECT point
    /// (console `wireframe perspective schlegel <n>`, the UI cell-index stepper,
    /// switching the projection radio to Schlegel, and any row edit that changes
    /// the leading polychoron) so the per-frame path never re-runs the
    /// `LazyLock`-backed [`Polytope4::face_planes`] fit.
    ///
    /// Clears the cache (`None`) when the mode is not Schlegel or the row has no
    /// polychoron to project. Idempotent: re-resolving the same `(polytope,
    /// cell_index)` recomputes bit-identical params (the fit is deterministic), so
    /// callers may invoke it unconditionally.
    ///
    /// When the leading polychoron has fewer cells than the carried
    /// `cell_index` (a row edit can swap a 600-cell out for a 5-cell while the
    /// mode stays `Schlegel { cell_index: 300 }`), [`resolve_schlegel_params`]
    /// clamps the index for the cache. The clamped value is written back into
    /// `wireframe_projection` so the enum, the cache, the UI cell-index stepper,
    /// and the console's "schlegel (cell N)" report all name the same cell the
    /// projection actually renders, rather than the stepper showing an
    /// out-of-range index and the report lying about the boundary cell.
    pub(crate) fn resolve_schlegel_cache(&mut self) {
        let (projection, cache) =
            synced_schlegel_projection(self.wireframe_projection, self.schlegel_subject());
        self.wireframe_projection = projection;
        self.schlegel_params = cache;
    }

    /// The live [`rye_math::Projection<4>`] for the wireframe overlay this frame.
    /// For Schlegel it builds the engine projection from the cached canonical
    /// [`SchlegelParams`]: the canonical normal is rotated by the current
    /// `rot_state` (one rotor apply; rotation preserves unit length) so the chosen
    /// cell stays the outer boundary as the body spins, and the canonical offset +
    /// viewpoint distance are scaled by [`Self::effective_body_size`] to match the
    /// body-scaled vertices the wireframe feeds the projection. No allocation, no
    /// `face_planes` call: this is the hot-path-safe counterpart to
    /// [`Self::resolve_schlegel_cache`]. Every other mode delegates to the
    /// context-free [`WireframeProjection::to_projection`].
    pub(crate) fn resolved_wireframe_projection(&self) -> Projection<4> {
        match self.wireframe_projection {
            WireframeProjection::Schlegel { .. } => match self.schlegel_params {
                Some(p) => {
                    let body_size = self.effective_body_size();
                    Projection::Schlegel {
                        cell_normal: self.rot_state.apply(p.cell_normal),
                        cell_offset: p.cell_offset * body_size,
                        viewpoint_distance: p.viewpoint_distance * body_size,
                    }
                }
                // Unresolved (no polychoron in row): drop-w fallback. The cache is
                // resolved at every select point, so this only fires for a row with
                // no polytope, where the wireframe draws nothing anyway.
                None => Projection::Identity,
            },
            // Substitute the live pole. The context-free `to_projection` returns
            // the default pole; the per-frame path honors the configurable field.
            WireframeProjection::Stereographic => Projection::Stereographic {
                pole: self.stereographic_pole,
            },
            other => other.to_projection(),
        }
    }

    /// Whether the wireframe Hyperslice cull should run this frame. Single source
    /// for both the cull's per-edge `continue` in `render_wireframe_overlay` and
    /// the Render-modal slab-width enable-gate, so the thickness control is live
    /// in exactly the frames the cull is. See [`hyperslice_cull_active`].
    pub(crate) fn hyperslice_cull_active(&self) -> bool {
        hyperslice_cull_active(self.wireframe_hyperslice, self.wireframe_projection)
    }

    /// Drive every body in the RENDERED row (see [`Self::render_row`]) with the
    /// same rotor, letting the user directly compare slice signatures under
    /// identical 4D motion. In [`ViewMode::Single`] the rendered row is the lone
    /// `strip_subject`, so this uploads exactly one body and rewrites the active
    /// `body_count` accordingly (via `set_bodies`, not slot-wise patching, so a
    /// stale row of bodies from a previous mode can't keep rendering).
    pub(crate) fn write_all(&mut self, rotor: Rotor4) {
        self.upload_render_row_bodies(rotor);
    }

    /// Re-emit every rendered body's uniform from the current row + rotor state.
    /// Called after row mutations (add/remove/reorder), rotor changes during
    /// spin, view-mode changes (the rendered row swaps between the full row and
    /// the single subject), and surface-mode changes (any time the polychora
    /// switch between SDF-live and SDF-inert).
    pub(crate) fn rebuild_bodies(&mut self) {
        self.upload_render_row_bodies(self.rot_state);
    }

    /// Build each rendered body's SDF uniform from the current row + rotor and
    /// upload them via `set_bodies`, which also sets the kernel's active
    /// `body_count`. Shared by [`Self::write_all`] and [`Self::rebuild_bodies`].
    ///
    /// The uniforms are filled into `body_uniform_scratch`, taken out of `self`
    /// for the duration so the build can borrow `&self` (for `render_row` and
    /// `sdf_body_for_slot`) while writing an owned buffer, then put back. The
    /// buffer keeps its capacity across frames, so the steady-state spin upload
    /// does not allocate.
    fn upload_render_row_bodies(&mut self, rotor: Rotor4) {
        let mut scratch = std::mem::take(&mut self.body_uniform_scratch);
        scratch.clear();
        let n = self.render_row().len();
        for slot in 0..n {
            let entry = &self.render_row()[slot];
            scratch.push(self.sdf_body_for_slot(entry, slot, n, rotor));
        }
        self.node.set_bodies(&scratch);
        self.body_uniform_scratch = scratch;
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
        self.space_blend = 0.0;
        // Restore the honest-slice default: the drop-w cross-section visible,
        // the reprojected cap off, so a reset always returns to the "slice that
        // never distorts under a projection change" baseline.
        self.cross_section = SectionLayer::CROSS_SECTION_DEFAULT;
        self.projected_cap = SectionLayer::PROJECTED_CAP_DEFAULT;
        self.draft.clear();
        self.write_all(Rotor4::IDENTITY);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        active_plane_angle, annotation_effective_blend, compose_active_rotor,
        hyperslice_cull_active, mode_annotation, render_row_entries, resolve_schlegel_params,
        row_blocks_sdf, section_layer_projection, synced_schlegel_projection, SectionLayer,
        ViewMode, WireframeProjection, BASE_ROTATION_RATE, STEREOGRAPHIC_DEFAULT_POLE,
    };
    use crate::catalog::ShapeEntry;
    use glam::Vec4;
    use rye_math::{Bivector, Plane4, Projection, Rotor, Rotor4};
    use rye_physics::polytope::Polytope4;
    use rye_render::raymarch::RaymarchShape;
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
        // The Active-mode checkbox is decoupled from the live orientation:
        // flipping `active` adds or removes the spin term `t * RATE`, and the
        // cell re-solves `base` so the displayed angle is continuous across
        // the toggle (no teleport). This pins that re-solve, which the UI
        // performs inline as `base = displayed_before - spin(active_after)`.
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

    /// A curvature value set while the wireframe overlay is OFF must not produce a
    /// spherical annotation: the flat-to-spherical edge bow renders only in the
    /// wireframe overlay, so with the overlay off there is no curved geometry on
    /// screen and the callout would describe something the user cannot see. The
    /// caller routes `space_blend` through [`annotation_effective_blend`], which
    /// reports flat (`0.0`) while the overlay is off, so:
    ///  - drop-w + curvature-but-overlay-off collapses to the default no-op
    ///    (`None`), exactly as plain flat drop-w does; and
    ///  - a non-default projection keeps its projection annotation (the caps
    ///    reproject without the wireframe) but drops the spherical sentence.
    ///
    /// With the overlay on, the raw blend passes through unchanged.
    #[test]
    fn curvature_annotation_requires_visible_wireframe() {
        // Overlay off: the gate reports flat regardless of the raw blend.
        assert_eq!(annotation_effective_blend(1.0, false), 0.0);
        assert_eq!(annotation_effective_blend(0.5, false), 0.0);
        // Overlay on: the raw blend passes through untouched.
        assert_eq!(annotation_effective_blend(1.0, true), 1.0);
        assert_eq!(annotation_effective_blend(0.0, true), 0.0);

        // Drop-w with curvature set but the overlay off is indistinguishable from
        // the plain flat default scene: no annotation.
        let blend_off = annotation_effective_blend(1.0, false);
        assert!(
            mode_annotation(WireframeProjection::DropW, blend_off, false).is_none(),
            "spherical-over-drop-w must not annotate while the wireframe is off"
        );
        // Turning the overlay on resurrects the spherical-space callout.
        let blend_on = annotation_effective_blend(1.0, true);
        assert!(
            mode_annotation(WireframeProjection::DropW, blend_on, false).is_some(),
            "spherical-over-drop-w must annotate once the wireframe is on"
        );

        // A non-default projection keeps its projection annotation with the
        // overlay off, but the body must NOT mention the S^3 edge bow, since no
        // bowed edge is drawn. The projection lead (caps reproject regardless) is
        // still present.
        let stereo_off = mode_annotation(WireframeProjection::Stereographic, blend_off, false)
            .expect("stereographic still annotates its projection with the overlay off");
        assert!(
            !stereo_off.body.contains("S^3 great-circle arcs"),
            "no spherical sentence without a visible wireframe bow: {}",
            stereo_off.body
        );
        assert!(
            stereo_off.body.contains("Conformal S^3 -> R^3"),
            "the stereographic projection lead must still be present: {}",
            stereo_off.body
        );
    }

    /// The educational-annotation mapping pins three invariants the UI relies on:
    /// (1) the plain default scene (drop-w projection, flat space) yields no
    /// annotation, so nothing floats over an undistorted view; (2) every
    /// non-default projection and the spherical-space-over-drop-w case yields a
    /// non-empty body; and (3) the bodies are pairwise DISTINCT across the
    /// non-default modes, so each mode reads as its own explanation rather than a
    /// shared placeholder. Pins the `(mode) -> copy` mapping, not the egui render.
    #[test]
    fn annotation_text_present_per_mode() {
        // Default scene: drop-w + flat space + (cap state irrelevant) -> nothing.
        assert!(
            mode_annotation(WireframeProjection::DropW, 0.0, true).is_none(),
            "drop-w + flat space must not annotate the default view"
        );
        assert!(
            mode_annotation(WireframeProjection::DropW, 0.0, false).is_none(),
            "cap state must not resurrect the default-scene annotation"
        );

        // Each distinct mode case the UI can surface. `space_blend` and the
        // flat-cap flag are chosen to exercise both the plain projection body and
        // the spherical/overlap branches.
        let cases: &[(WireframeProjection, f32, bool)] = &[
            (WireframeProjection::WDepth, 0.0, true),
            (WireframeProjection::Schlegel { cell_index: 0 }, 0.0, true),
            (WireframeProjection::Stereographic, 0.0, true),
            (WireframeProjection::Hyperslice, 0.0, true),
            // Spherical curvature over plain drop-w: the space mode is the only
            // thing to explain, so this is its own distinct annotation.
            (WireframeProjection::DropW, 1.0, true),
        ];

        let mut bodies = HashSet::new();
        for &(projection, blend, flat_cap) in cases {
            let annotation = mode_annotation(projection, blend, flat_cap)
                .unwrap_or_else(|| panic!("{projection:?} blend={blend} must annotate"));
            assert!(
                !annotation.title.is_empty(),
                "{projection:?} blend={blend}: title must be non-empty"
            );
            assert!(
                !annotation.body.is_empty(),
                "{projection:?} blend={blend}: body must be non-empty"
            );
            assert!(
                bodies.insert(annotation.body.clone()),
                "{projection:?} blend={blend}: body duplicates another mode's copy"
            );
        }

        // The three-way overlap note is conditional: spherical curvature with a
        // flat cap (or Stereographic) must mention the flat cross-section, and the
        // no-flat-cap / non-stereographic case must NOT, so the user is only warned
        // about the flat cap when one is actually drawn under curved edges.
        let with_cap = mode_annotation(WireframeProjection::WDepth, 1.0, true)
            .expect("wdepth + spherical annotates");
        assert!(
            with_cap.body.contains("cross-section cap stays flat"),
            "spherical + flat cap must flag the flat cross-section: {}",
            with_cap.body
        );
        let no_cap = mode_annotation(WireframeProjection::WDepth, 1.0, false)
            .expect("wdepth + spherical annotates");
        assert!(
            !no_cap.body.contains("cross-section cap stays flat"),
            "spherical without a flat cap must not mention the cap: {}",
            no_cap.body
        );
        // Stereographic forces the overlap note even with no raster cap, because
        // its own R^3 image is a flat conformal map under the curved edges.
        let stereo_no_cap = mode_annotation(WireframeProjection::Stereographic, 1.0, false)
            .expect("stereographic + spherical annotates");
        assert!(
            stereo_no_cap.body.contains("cross-section cap stays flat"),
            "stereographic + spherical must flag the flat slice even without a cap: {}",
            stereo_no_cap.body
        );
    }

    /// The SDF crash-safety gate keys off the RENDERED row, not the stored
    /// `self.row`. `Demo::sdf_blocked_by_heavy_polychora` is
    /// `row_blocks_sdf(self.render_row())`; in [`ViewMode::Single`] the rendered
    /// row is exactly the `strip_subject`, so the gate must follow the subject
    /// and ignore the multi-shape row entirely. Both failure directions matter:
    /// a heavy subject over a light row must still BLOCK (else the 120/600-cell
    /// SDF reaches the kernel and crashes the tab), and a light subject under a
    /// heavy row must NOT block (else SDF is needlessly disabled on a safe body).
    /// This pins the same `render_row()`-keyed contract the method now uses,
    /// which the row-only `row_blocks_sdf_only_for_heavy_polychora` test could
    /// not catch.
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

    /// Every token in the five-mode set parses to its `WireframeProjection`
    /// variant, and the context-free `to_projection` produces the matching engine
    /// `Projection<4>` variant or the documented `Identity` fallback. The fallback
    /// modes are `Hyperslice` (a demo-side w-cull, drop-w projection) and
    /// `Schlegel` (resolved Demo-side from the cached face-plane params + rotor,
    /// which the context-free enum cannot supply). `ALL` is the single cycle
    /// source the bare console command and the UI radio share, so a variant added
    /// to the enum but omitted from `ALL`/`from_token` fails here.
    #[test]
    fn wireframe_projection_from_token_round_trips() {
        // Pin the count so an enum addition that skips `ALL` is loud.
        assert_eq!(
            WireframeProjection::ALL.len(),
            5,
            "ALL must list every selectable projection mode"
        );
        for mode in WireframeProjection::ALL {
            let token = match mode {
                WireframeProjection::DropW => "drop-w",
                WireframeProjection::WDepth => "w-depth",
                WireframeProjection::Schlegel { .. } => "schlegel",
                WireframeProjection::Stereographic => "stereographic",
                WireframeProjection::Hyperslice => "hyperslice",
            };
            assert_eq!(
                WireframeProjection::from_token(token),
                Some(mode),
                "token `{token}` must parse back to {mode:?}"
            );
        }
        // Context-free engine projections per the documented contract.
        assert_eq!(
            WireframeProjection::DropW.to_projection(),
            Projection::Identity
        );
        assert_eq!(
            WireframeProjection::WDepth.to_projection(),
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

    /// The default Stereographic pole is the unit direction `(1, 1, 1, 1)/2` and
    /// equals (to f32 epsilon) the normalized centroid of a 16-cell cell, tying
    /// the literal constant to the polytope's topology rather than a bare number.
    /// If the cell-centroid derivation ever drifts from the recorded literal this
    /// fails loudly.
    #[test]
    fn stereographic_default_pole_is_unit_cell_center() {
        // Exactly unit by construction: 4 * 0.5^2 = 1.
        assert_eq!(
            STEREOGRAPHIC_DEFAULT_POLE.length_squared(),
            1.0,
            "default pole must be exactly unit"
        );
        // The 16-cell cell centroids are the (±½, ±½, ±½, ±½) directions; the
        // all-positive one, normalized, is the default pole. `cell_centers`
        // returns centroids at the inradius, so normalize before comparing.
        let centers = Polytope4::Cell16.cell_centers();
        let matches_a_centroid = centers
            .iter()
            .any(|c| (c.normalize() - STEREOGRAPHIC_DEFAULT_POLE).length() < 1e-6);
        assert!(
            matches_a_centroid,
            "default pole must be a normalized 16-cell cell centroid"
        );
    }

    /// No 16-cell vertex sits on the default pole: every vertex direction is
    /// strictly outside the pole-clamp band (`dot(v, pole) < 1 - eps`), so the
    /// stereographic denominator never reaches the clamp for an axis-aligned
    /// 16-cell at rest. Also pins the common-case sweep: a pure `xw` rotation
    /// leaves each vertex's `y, z` fixed, and the pole has `y = z = ½`, so no
    /// 16-cell vertex (whose `y, z` are each `0` or `±1`) can ever rotate onto it.
    #[test]
    fn stereographic_default_pole_is_never_a_16cell_vertex() {
        let eps = rye_math::STEREOGRAPHIC_POLE_EPSILON;
        let pole = STEREOGRAPHIC_DEFAULT_POLE;
        for v in Polytope4::Cell16.topology().vertices {
            // Vertices are unit `±e_i`; dot with the pole is exactly ±½.
            let d = v.normalize().dot(pole);
            assert!(
                d < 1.0 - eps,
                "16-cell vertex {v:?} has dot {d} with the default pole, inside \
                 the pole-clamp band"
            );
        }
        // The common-case xw sweep: rotate +x through the xw plane and confirm
        // the pole's fixed y = z = ½ keeps dot bounded away from 1 the whole way.
        for step in 0..360 {
            let theta = (step as f32).to_radians();
            // xw rotation of +x: (cos, 0, 0, sin). y and z stay 0.
            let rotated = Vec4::new(theta.cos(), 0.0, 0.0, theta.sin());
            let d = rotated.dot(pole);
            // Max over the sweep is cos*½ + sin*½ <= sqrt(2)/2 ~ 0.707 < 1 - eps.
            assert!(
                d < 1.0 - eps,
                "xw-rotated +x at {step} deg has dot {d}, inside the clamp band"
            );
        }
    }

    /// The Hyperslice cull runs when EITHER the standalone toggle is on OR the
    /// projection mode is `Hyperslice`. This is the single predicate the wireframe
    /// builder's per-edge `continue` and the Render-modal slab-width enable-gate
    /// both consume, so the thickness control is reachable in exactly the frames
    /// the cull is active. The bug this pins: the projection-mode arm must turn the
    /// cull on by itself, with EVERY other projection leaving it off unless the
    /// toggle is set.
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

    /// The Schlegel `cell_normal` `resolve_schlegel_params` feeds the projection is
    /// the topology-derived `Polytope4::face_planes` direction, NOT the buggy
    /// dual-vertex `cell{120,600}_face_planes`. For the 600-cell the dual helper
    /// returns the 120-cell's 120 vertices as "face normals", which are wrong for
    /// the 96 golden-ratio cell orbits. Pin that the resolved normal (a) equals the
    /// `face_planes` direction exactly and (b) is, for at least one cell, more than
    /// 1e-3 from EVERY dual normal: the only way that holds is if the dual path is
    /// not the source.
    #[test]
    fn schlegel_params_from_face_planes_not_dual() {
        use rye_physics::euclidean_r4::cell600_face_planes;
        let polytope = Polytope4::Cell600;
        let (topo_normals, _) = polytope.face_planes();
        let (dual_normals, _) = cell600_face_planes();
        // Find a cell whose topology normal is far from every dual normal. The 96
        // golden-ratio orbits guarantee at least one exists; the dual set has only
        // 120 entries for a 600-faced polytope, so most topology faces have no dual
        // counterpart at all.
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

    /// An out-of-range `cell_index` clamps to `[0, cell_count - 1]`: it never
    /// panics, never indexes the face-plane slice out of bounds, and resolves to
    /// the last cell rather than wrapping or saturating past the end.
    #[test]
    fn schlegel_cell_index_clamped_to_cell_count() {
        let polytope = Polytope4::Pentatope; // 5 cells.
        let last = polytope.cell_count() as u32 - 1;
        // Way past the end clamps to the last cell, no panic.
        let params = resolve_schlegel_params(polytope, 9999);
        assert_eq!(params.cell_index, last);
        // The clamped resolution equals an explicit last-cell request.
        let at_last = resolve_schlegel_params(polytope, last);
        assert_eq!(params, at_last);
    }

    /// Resolving a Schlegel selection whose `cell_index` overruns the leading
    /// polytope's cell count writes the CLAMPED index back into the projection,
    /// so the carried `WireframeProjection::Schlegel { cell_index }` names the
    /// exact cell the cache (and therefore the rendered diagram) uses. This is
    /// the row-shrink desync the cache-clamp alone left open: a 600-cell ->
    /// 5-cell row edit clamps the cache to cell 4 but, before the writeback, left
    /// the enum (and thus the UI stepper + console "schlegel (cell N)" report)
    /// naming cell 300. Pin that the returned projection's index equals the
    /// cache's index and is in range, and that re-resolving is a fixed point.
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

    /// A non-Schlegel mode passes through unchanged and clears the cache, and a
    /// Schlegel mode with no polychoron in the row (`subject == None`) keeps its
    /// carried index verbatim (no polytope to clamp against) with a `None` cache.
    /// Pins that the index-sync only fires where a subject can actually clamp it.
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

    /// Resolving the canonical params for a fixed `(polytope, cell_index)` twice
    /// yields BIT-identical f32 (not merely approximately equal). The face-plane
    /// fit is deterministic, so the Schlegel cache can be rebuilt on any select
    /// without introducing frame-to-frame jitter in the projection.
    #[test]
    fn schlegel_resolution_is_bit_deterministic() {
        for polytope in Polytope4::ALL {
            let cell_index = (polytope.cell_count() / 2) as u32;
            let a = resolve_schlegel_params(polytope, cell_index);
            let b = resolve_schlegel_params(polytope, cell_index);
            assert_eq!(a.cell_normal, b.cell_normal, "{polytope:?} normal");
            assert_eq!(a.cell_offset, b.cell_offset, "{polytope:?} offset");
            assert_eq!(
                a.viewpoint_distance, b.viewpoint_distance,
                "{polytope:?} viewpoint"
            );
        }
    }

    /// With a non-identity `rot_state`, the effective Schlegel boundary normal is
    /// the canonical normal rotated by that same rotor: `rot_state.apply(canonical)`.
    /// This is the determinism hazard the plan flags. `face_planes` returns
    /// CANONICAL unrotated normals, but the wireframe vertices are `rot_state`-
    /// rotated before projection, so the chosen cell only stays the outer boundary
    /// if its normal rotates with the body. Verified at the math level (the rotor
    /// apply) since `Demo::resolved_wireframe_projection` needs a GPU-backed `Demo`.
    #[test]
    fn schlegel_normal_rotates_with_body() {
        let polytope = Polytope4::Tesseract;
        let cell_index = 0;
        let params = resolve_schlegel_params(polytope, cell_index);
        // A non-trivial xw rotation. The canonical normal must not already be the
        // rotated one, else the test proves nothing.
        let rot = (Plane4::Xw.unit_bivector() * 0.7).exp().normalize();
        let rotated = rot.apply(params.cell_normal);
        assert!(
            (rotated - params.cell_normal).length() > 1e-3,
            "rotation must actually move the normal for this test to bite"
        );
        // Rotation preserves unit length, so the rotated normal is still a valid
        // outward unit normal for the engine `Projection::Schlegel`.
        assert!((rotated.length() - 1.0).abs() < 1e-5);
        // And it equals the body's own rotation of every chosen-cell vertex's
        // direction: a chosen-cell vertex `v` has `dot(canonical_normal, v) =
        // cell_offset`; after rotating both, `dot(rotated_normal, rot.apply(v))`
        // must still equal `cell_offset` (the cell stays on its hyperplane).
        let topo = polytope.topology();
        let cell = topo.cells[cell_index as usize];
        for &vi in cell {
            let v = topo.vertices[vi as usize];
            let lhs = rotated.dot(rot.apply(v));
            assert!(
                (lhs - params.cell_offset).abs() < 1e-4,
                "rotated cell vertex must stay on the rotated boundary hyperplane: {lhs} vs {}",
                params.cell_offset
            );
        }
    }

    /// In [`ViewMode::Single`] the rendered row is EXACTLY the `strip_subject`,
    /// not the multi-shape row. Every per-body render path (wireframe overlay,
    /// section faces, points) and the SDF body upload iterate the slice
    /// `render_row_entries` returns, so this pins that those passes build
    /// geometry for the single subject alone, independent of how many shapes the
    /// row holds. Shapes / Filmstrip keep returning the full row.
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

    /// The Schlegel cell-index upper bound in Single mode is the
    /// `strip_subject`'s cell count, NOT any row member's. The bound is derived
    /// from the leading polychoron of the rendered row (the same path
    /// `Demo::schlegel_subject` walks: `render_row().find_map(polytope4)`), so in
    /// Single mode it must resolve to the subject's polytope and therefore its
    /// `cell_count()`. This is the unambiguous single-polytope selection a
    /// boundary-cell index needs (a 5-cell has 5 cells, a 600-cell has 600), and
    /// the whole reason Single mode unblocks the Schlegel cell-index control.
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

    /// Each `ViewMode` variant round-trips through the tab's stage-then-apply
    /// shape: staging a different value into `pending_view_mode` and applying it
    /// lands `view_mode` on that variant, and the rendered row then matches the
    /// applied mode (Single -> subject, Shapes/Filmstrip -> row). Re-staging the
    /// SAME mode is a no-op (the UI only stages on `staged != view_mode`), so the
    /// pending slot stays `None`. This pins the `render_view_tab_row` ->
    /// `render_overlay` deferred-apply contract without a GPU-backed `Demo`.
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

    /// The headline invariant of the two-layer split: the honest cross-section
    /// ALWAYS resolves to drop-w (`Identity`) regardless of the active wireframe
    /// projection, while the projected cap follows whatever projection is active.
    /// This is what guarantees that selecting Schlegel / stereographic never
    /// silently distorts the slice the user reads as "the cross-section": the
    /// honest layer is pinned to the SDF's own drop-w view.
    #[test]
    fn section_layer_projection_honest_ignores_projected_follows() {
        let actives = [
            Projection::Identity,
            Projection::Perspective4D {
                focal_distance: 2.0,
            },
            Projection::Stereographic { pole: Vec4::W },
            Projection::Schlegel {
                cell_normal: Vec4::W,
                cell_offset: 0.5,
                viewpoint_distance: 0.75,
            },
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

    /// `fill_visible` is the layer's on/off switch: any positive alpha draws a
    /// fill, `0` (or below) skips the pass. Pins the boundary so a layer set to
    /// alpha 0 never submits an invisible mesh and a faint positive alpha still
    /// draws.
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

    /// The defaults encode the spec's "honest slice visible, reprojected cap off"
    /// baseline: the cross-section's perimeter + fill are on (the fill at a
    /// full-ish, sub-opaque alpha so the wireframe reads through), and the
    /// projected cap is fully off. This is the state that makes a projection
    /// change non-destructive to the slice.
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
}
