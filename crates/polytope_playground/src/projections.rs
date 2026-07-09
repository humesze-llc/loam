//! Wireframe projection modes: the `WireframeProjection` enum, Schlegel diagram
//! parameter resolution, per-mode edge-blend selection, and mode annotations.

use loam_physics::polytope::Polytope4;

/// How the parent wireframe's 4D vertex positions project to R³. Independent of
/// the cross-section's projection (always drop-w).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub(crate) enum WireframeProjection {
    /// Orthographic drop-w `(x, y, z, w) -> (x, y, z)`, the shadow down the
    /// w-axis. Collapses w-opposite vertices to one R³ point, so axis-aligned
    /// polytopes (the tesseract) render flat; the 5-cell and 24-cell read better.
    #[default]
    Shadow,
    /// 4D pinhole at `(0, 0, 0, focal_distance)` projecting onto the `w = 0`
    /// 3-flat, foreshortening by `focal / (focal - w)`. The classical
    /// cube-within-a-cube tesseract view (+w face outer, -w inner).
    WPinhole,
    /// 4D Schlegel diagram: central projection from just outside the chosen
    /// boundary cell onto its 3-flat (Coxeter, *Regular Polytopes*, ch. 13); the
    /// chosen cell becomes the outer boundary, every other cell nests inside.
    /// `cell_index` selects the boundary cell in canonical [`Polytope4::topology`]
    /// order, clamped at resolve time. The normal, basis, and distances are
    /// resolved from topology via [`Polytope4::face_planes`] (the CORRECT path,
    /// NOT the buggy dual-vertex `cell{120,600}_face_planes`) and cached on
    /// `Demo::schlegel_params`. See [`resolve_schlegel_params`] and
    /// `Demo::resolved_wireframe_projection`.
    ///
    /// NOT WIRED into the playground: deliberately absent from [`Self::ALL`] and
    /// [`Self::from_token`]. A faithful Schlegel diagram needs its own framing and
    /// cell-selection UX; in the shared single-viewport overlay it does not earn
    /// its keep. The math + (correct) param resolution are kept and tested for a
    /// future dedicated Schlegel demo. `allow(dead_code)` marks the variant as
    /// retained on purpose; removing it signals Schlegel is fully gone.
    #[allow(dead_code)]
    Schlegel {
        /// Index of the boundary cell in the polytope's canonical cell order.
        cell_index: u32,
    },
    /// Conformal stereographic projection from S³ (unit-circumradius vertices) to
    /// R³, casting away from a configurable pole (default
    /// [`STEREOGRAPHIC_DEFAULT_POLE`]; live value `Demo::stereographic_pole`).
    /// Edges render as S³ great-circle arcs (see [`default_edge_blend`]).
    /// Angle-preserving, distance-distorting: the pole-facing cell balloons
    /// outward. `EuclideanR4` normalizes each vertex onto S³ first, so the
    /// `BODY_SIZE`-scaled vertices read correctly.
    Stereographic,
    /// Drop-w paired with the demo-side cell-level w-range cull (the
    /// `wireframe_hyperslice` filter): the wireframe thins to edges of cells whose
    /// body-local w-range overlaps a slab around `w_slice`. Cell-level (not
    /// edge-level) so a kept edge agrees with the active-edge coloring and the
    /// cross-section. The projection stays drop-w
    /// ([`loam_math::Projection::Identity`]); the cull does the slicing. Slab width
    /// is `Demo::wireframe_hyperslice_thickness`.
    Hyperslice,
}

/// Canonical (unit-circumradius) Schlegel parameters for one
/// `(polytope, cell_index)`. Cached on `Demo::schlegel_params` so the O(V·D³)
/// `LazyLock` cell-table fit behind [`Polytope4::face_planes`] runs once per
/// selection, not per frame.
///
/// Stored in CANONICAL coords: `Demo::resolved_wireframe_projection` scales
/// the offsets by `Demo::effective_body_size` and rotates the normal/basis by
/// `rot_state` each frame. Canonical caching stays valid across `surface scale`
/// and keeps the chosen cell the outer boundary as the polytope spins.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SchlegelParams {
    /// Polytope resolved against; the cache is invalid if it changes.
    pub(crate) polytope: Polytope4,
    /// The (already-clamped) boundary cell index.
    pub(crate) cell_index: u32,
    /// Outward unit normal of the chosen cell's hyperplane, CANONICAL coords.
    pub(crate) cell_normal: glam::Vec4,
    /// Orthonormal readout basis spanning the chosen cell, CANONICAL coords.
    pub(crate) cell_basis: [glam::Vec4; 3],
    /// Signed plane offset (the cell's inradius), CANONICAL coords: the cell lies
    /// in `{x : dot(cell_normal, x) = cell_offset}`.
    pub(crate) cell_offset: f32,
    /// Eye distance along `cell_normal`, CANONICAL coords: the farthest vertex
    /// projection plus [`SCHLEGEL_EYE_MARGIN`].
    pub(crate) viewpoint_distance: f32,
}

const SCHLEGEL_BASIS_EPSILON: f32 = 1e-6;

fn push_schlegel_basis_vector(
    candidate: glam::Vec4,
    cell_normal: glam::Vec4,
    basis: &mut [glam::Vec4; 3],
    count: &mut usize,
) {
    if *count == basis.len() {
        return;
    }
    let mut v = candidate - candidate.dot(cell_normal) * cell_normal;
    for b in basis.iter().take(*count) {
        v -= v.dot(*b) * *b;
    }
    let len = v.length();
    if len > SCHLEGEL_BASIS_EPSILON {
        basis[*count] = v / len;
        *count += 1;
    }
}

fn resolve_schlegel_cell_basis(
    polytope: Polytope4,
    cell_index: usize,
    cell_normal: glam::Vec4,
) -> [glam::Vec4; 3] {
    let topo = polytope.topology();
    let cell = topo.cells[cell_index];
    let anchor = topo.vertices[cell[0] as usize];
    let mut basis = [glam::Vec4::ZERO; 3];
    let mut count = 0usize;

    for &vi in cell.iter().skip(1) {
        push_schlegel_basis_vector(
            topo.vertices[vi as usize] - anchor,
            cell_normal,
            &mut basis,
            &mut count,
        );
        if count == 3 {
            return basis;
        }
    }

    debug_assert_eq!(count, 3, "Schlegel boundary cell must span a 3-flat");
    for seed in [glam::Vec4::X, glam::Vec4::Y, glam::Vec4::Z, glam::Vec4::W] {
        push_schlegel_basis_vector(seed, cell_normal, &mut basis, &mut count);
        if count == 3 {
            break;
        }
    }
    basis
}

/// Additive eye clearance beyond the chosen cell's far edge, canonical units.
/// Additive (not a multiple of cell offset) so the clearance does not collapse
/// for small-inradius polytopes: the 5-cell's 0.25 inradius under `1.5 *
/// cell_offset` leaves the eye only 0.125 clear and the diagram folds to a
/// sliver. 0.5 of the unit circumradius keeps a polytope-independent clearance.
const SCHLEGEL_EYE_MARGIN: f32 = 0.5;

/// Resolve the canonical Schlegel parameters for `(polytope, cell_index)`.
/// `cell_index` is clamped to `[0, cell_count - 1]`.
///
/// The cell normal and inradius come from [`Polytope4::face_planes`] (cell
/// centroids via topology; Coxeter, *Regular Polytopes*, ch. 13). This is the
/// CORRECT path: the dual-vertex `cell{120,600}_face_planes` helpers in
/// `loam_physics::euclidean_r4` are wrong for 96 of the 120/600-cell normals (the
/// documented BUG) and are NOT used here. The result is canonical; the caller
/// scales by the live body size.
pub(crate) fn resolve_schlegel_params(polytope: Polytope4, cell_index: u32) -> SchlegelParams {
    let cell_count = polytope.cell_count() as u32;
    // `cell_count >= 1`, so `cell_count - 1` never underflows.
    let clamped = cell_index.min(cell_count - 1);
    let (normals, cell_offset) = polytope.face_planes();
    let cell_normal = normals[clamped as usize];
    let cell_basis = resolve_schlegel_cell_basis(polytope, clamped as usize, cell_normal);
    // Farthest vertex-projection along the outward normal, computed over all
    // vertices so the eye clearance is robust to topology quirks.
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
        cell_basis,
        cell_offset,
        viewpoint_distance: max_dot + SCHLEGEL_EYE_MARGIN,
    }
}

/// Resolve `projection` against the Schlegel `subject` (the leading polychoron,
/// or `None`), returning the projection to store plus the cache to attach.
///
/// For `Schlegel` with a subject, the returned projection's `cell_index` is
/// rewritten to the SAME clamped value the cache carries (a row edit can shrink
/// a 600-cell to a 5-cell while the mode still names `cell_index: 300`), so the
/// enum, cache, UI stepper, and console report never disagree about the boundary
/// cell. Every other mode (and subjectless `Schlegel`) passes through unchanged
/// and clears the cache.
///
/// Pure so the index-sync invariant is unit-testable without a GPU-backed
/// `Demo`; `Demo::resolve_schlegel_cache` is the one caller.
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

/// Edge geometry for `projection`: S3 great-circle arcs (`blend == 1`) for
/// Stereographic (a conformal-map edge is a circular arc, not a chord), flat R4
/// chords (`blend == 0`) otherwise. Derived from the projection, not a separate
/// control, so it can never disagree with the map. Read per frame by the
/// wireframe builder (`Demo::render_wireframe_overlay`).
pub(crate) fn default_edge_blend(projection: WireframeProjection) -> f32 {
    match projection {
        WireframeProjection::Stereographic => 1.0,
        _ => 0.0,
    }
}

/// Secondary state changes implied by selecting a wireframe projection.
/// Stereographic's S3 arcs only draw in the overlay, so selecting it turns the
/// overlay on. Edge geometry is derived ([`default_edge_blend`]), not set here.
pub(crate) fn apply_projection_selection_defaults(
    projection: WireframeProjection,
    wireframe_enabled: &mut bool,
) {
    if matches!(projection, WireframeProjection::Stereographic) {
        *wireframe_enabled = true;
    }
}

/// Short educational annotation for the active projection: a callout `title`
/// plus a one-to-three-sentence `body`. Surfaced via `loam_egui::callout` (see
/// `Demo::render_mode_annotation`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ModeAnnotation {
    /// Callout window title: the mode's short name.
    pub(crate) title: &'static str,
    /// One to three sentences explaining the active projection.
    pub(crate) body: String,
}

/// Educational annotation for `projection`, or `None` for the default drop-w
/// (nothing non-obvious to explain). Every other projection returns `Some` with
/// distinct, non-empty copy.
///
/// Pure so the `(mode) -> copy` mapping is unit-testable without a GPU-backed
/// `Demo`; `Demo::render_mode_annotation` is the one caller. Explanations
/// follow Coxeter, *Regular Polytopes*, ch. 13 (Schlegel) and the conformal map
/// `(x, y, z) / (1 - w)` (Wikipedia, "Stereographic projection").
pub(crate) fn mode_annotation(projection: WireframeProjection) -> Option<ModeAnnotation> {
    let (title, projection_body): (&'static str, Option<&str>) = match projection {
        WireframeProjection::Shadow => ("Shadow", None),
        WireframeProjection::WPinhole => (
            "W-pinhole",
            Some(
                "4D pinhole camera with its eye down the w-axis: the +w face \
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
                "Conformal S^3 -> R^3 map, (x, y, z) / (1 - w): the polytope is \
                 cast onto S^3 and its edges render as great-circle arcs. Angles \
                 are preserved but distances are not, so the cell facing the +w \
                 pole balloons outward as its vertices approach the pole.",
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

    let projection_lead = projection_body?;
    Some(ModeAnnotation {
        title,
        body: projection_lead.to_string(),
    })
}

/// Default pole for the Stereographic wireframe projection: the `+w` axis
/// `(0, 0, 0, 1)`. Aligned with the w-slice axis so the conformal scale
/// `1 / (1 - dot(p, pole))` reduces to `1 / (1 - p.w)` and every cell sharing
/// the slice's w maps to one CENTERED radial shell; an off-axis pole makes the
/// scale depend on `x + y + z + w` and skews each slice.
///
/// Tradeoff: `+w` is a 16-cell vertex (`+e_w`, Coxeter, *Regular Polytopes*,
/// §8.2), so a vertex sweeping through the pole under an `xw` rotation flicks to
/// infinity (the standard stereographic singularity), accepted for the centered
/// image. Configurable via `Demo::stereographic_pole`. Exactly unit, so no
/// runtime `normalize` and the constant is bit-reproducible.
pub(crate) const STEREOGRAPHIC_DEFAULT_POLE: glam::Vec4 = glam::Vec4::new(0.0, 0.0, 0.0, 1.0);

impl WireframeProjection {
    /// Parse the console-arg spelling. Hyphens because `w-pinhole` lexes as one
    /// token. `schlegel` is NOT parsed (see [`Self::ALL`] / [`Self::Schlegel`]).
    pub(crate) fn from_token(token: &str) -> Option<Self> {
        match token {
            "shadow" => Some(WireframeProjection::Shadow),
            "w-pinhole" => Some(WireframeProjection::WPinhole),
            "stereographic" => Some(WireframeProjection::Stereographic),
            "hyperslice" => Some(WireframeProjection::Hyperslice),
            _ => None,
        }
    }

    /// Cycle order for the bare `wireframe perspective` command and the UI radio.
    /// Schlegel is intentionally omitted (see [`Self::Schlegel`]).
    pub(crate) const ALL: [Self; 4] = [
        WireframeProjection::Shadow,
        WireframeProjection::WPinhole,
        WireframeProjection::Stereographic,
        WireframeProjection::Hyperslice,
    ];

    /// Display label for the egui projection radio.
    pub(crate) fn label(self) -> &'static str {
        match self {
            WireframeProjection::Shadow => "Shadow",
            WireframeProjection::WPinhole => "W-pinhole",
            WireframeProjection::Schlegel { .. } => "Schlegel",
            WireframeProjection::Stereographic => "Stereographic",
            WireframeProjection::Hyperslice => "Hyperslice",
        }
    }

    /// Whether two modes are the same VARIANT, ignoring the Schlegel cell index,
    /// so a cell-index change does not deselect the Schlegel radio button.
    pub(crate) fn same_variant(self, other: Self) -> bool {
        std::mem::discriminant(&self) == std::mem::discriminant(&other)
    }

    /// Context-free resolution to a [`loam_math::Projection<4>`] for the modes
    /// needing no polytope or rotor context:
    /// - `Shadow`, `Hyperslice` -> `Identity` (drop-w; the demo-side
    ///   `wireframe_hyperslice` filter does Hyperslice's slicing),
    /// - `WPinhole` -> `Perspective4D { focal_distance: 2.0 }` (focal clears the
    ///   `BODY_SIZE`-scaled w-extent so the denominator never nears zero),
    /// - `Stereographic` -> `Stereographic { pole: STEREOGRAPHIC_DEFAULT_POLE }`
    ///   (the default pole; `Demo::resolved_wireframe_projection` substitutes
    ///   the live `Demo::stereographic_pole` per frame).
    ///
    /// `Schlegel` returns `Identity` as a SAFE FALLBACK only: the real projection
    /// needs the cached [`SchlegelParams`] plus the live `rot_state`, built by
    /// `Demo::resolved_wireframe_projection` (the four render sites call it).
    pub(crate) fn to_projection(self) -> loam_math::Projection<4> {
        match self {
            WireframeProjection::Shadow | WireframeProjection::Hyperslice => {
                loam_math::Projection::Identity
            }
            WireframeProjection::WPinhole => loam_math::Projection::Perspective4D {
                focal_distance: 2.0,
            },
            // Schlegel needs cached params + rotor (see doc above); drop-w
            // fallback until `Demo::resolved_wireframe_projection` resolves it.
            WireframeProjection::Schlegel { .. } => loam_math::Projection::Identity,
            WireframeProjection::Stereographic => loam_math::Projection::Stereographic {
                pole: STEREOGRAPHIC_DEFAULT_POLE,
            },
        }
    }
}

/// Whether the wireframe Hyperslice cull is active: the
/// `Demo::wireframe_hyperslice` toggle is on OR the projection is
/// [`WireframeProjection::Hyperslice`]. Both the wireframe builder's `continue`
/// and the Render-modal slab-width gate share this predicate so the thickness
/// stepper is reachable in exactly the frames the cull runs.
pub(crate) fn hyperslice_cull_active(toggle: bool, projection: WireframeProjection) -> bool {
    toggle || matches!(projection, WireframeProjection::Hyperslice)
}
