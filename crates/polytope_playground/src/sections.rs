//! Cross-section and surface policy: the SurfaceMode enum, the SectionLayer
//! pass config, the cross-section projection rule, and the heavy-polychora SDF
//! gate.

use rye_math::Projection;
use rye_physics::polytope::Polytope4;
use rye_render::raymarch::RaymarchShape;

use crate::catalog::ShapeEntry;

/// How the six regular convex 4-polytopes have their surface rendered.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub(crate) enum SurfaceMode {
    /// Rasterized filled cross-section cell-caps. Exact, much faster on the
    /// 120/600-cell than the SDF, and sidesteps the cell120/600 face-plane BUG
    /// in `rye_physics::euclidean_r4`.
    #[default]
    Raster,
    /// SDF raymarch via `Demo::node`. Kept for visual comparison; slower on
    /// the 120/600-cell and carries the face-plane BUG.
    Sdf,
    /// No polychoral surface. The wireframe overlay (if enabled) still shows the
    /// edge graph; useful for inspecting wireframe + perimeter without the fill.
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

    /// `true` when the polychoral SDF dispatch needs to be live. False for
    /// Raster (the rasterizer draws them) and Off (nothing does).
    pub(crate) fn uses_sdf_for_polychora(self) -> bool {
        matches!(self, SurfaceMode::Sdf)
    }
}

/// One overlaid layer of the rasterized cross-section: a perimeter-outline
/// toggle plus a surface-fill alpha that doubles as the layer's on/off switch.
/// Both layers share the same slice geometry (the drop-w 3-flat cut) and differ
/// only in how it maps to R³. See `Demo::cross_section` /
/// `Demo::projected_cap`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct SectionLayer {
    /// Whether the cap-boundary perimeter outline renders. Independent of
    /// [`Self::surface_alpha`].
    pub(crate) perimeter: bool,
    /// Per-fragment fill alpha in `[0, 1]`. `0.0` skips the fill; `(0, 1)` goes
    /// through the depth-write-disabled translucent pipeline; `1.0` is opaque
    /// with depth-write. See `Demo::section_faces` /
    /// `Demo::section_faces_translucent`.
    pub(crate) surface_alpha: f32,
}

impl SectionLayer {
    /// Whether the surface fill draws at all (`surface_alpha > 0`); at or below
    /// zero the layer skips its triangle pass rather than submit invisible mesh.
    pub(crate) fn fill_visible(self) -> bool {
        self.surface_alpha > 0.0
    }
}

/// Default fill alpha for the honest cross-section: opaque so the slice reads
/// as solid surface geometry.
const CROSS_SECTION_DEFAULT_ALPHA: f32 = 1.0;

impl SectionLayer {
    /// Honest cross-section default: perimeter + opaque fill on, so selecting a
    /// distorting wireframe projection never silently reshapes the slice.
    pub(crate) const CROSS_SECTION_DEFAULT: SectionLayer = SectionLayer {
        perimeter: true,
        surface_alpha: CROSS_SECTION_DEFAULT_ALPHA,
    };
    /// Projected-cap default: fully off; reprojecting the slice through the
    /// active wireframe projection is opt-in.
    pub(crate) const PROJECTED_CAP_DEFAULT: SectionLayer = SectionLayer {
        perimeter: false,
        surface_alpha: 0.0,
    };
}

/// The 4D->R³ projection a section layer renders through. The honest
/// cross-section is ALWAYS drop-w ([`rye_math::Projection::Identity`]); the slice
/// is a 3-flat and drop-w is its undistorted view, matching the SDF raymarch.
/// The projected cap follows the active wireframe `projection`.
///
/// Free function so the "honest ignores, projected follows" invariant is
/// unit-testable without a GPU-backed `Demo`; the two section render paths in
/// `main.rs` are the callers.
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

/// True when `row` contains a 120-cell or 600-cell. Blocked from SDF for TWO
/// reasons, either sufficient: (1) their SDFs (120 / 600 face hyperplanes each,
/// against the per-pixel Wolfe-greedy projection) overrun the browser WebGPU
/// shader budget and crash the tab; (2) their `cell{120,600}_face_planes` are the
/// known-wrong dual-vertex approximation (see `rye_shape::polytope_geom`), so the
/// SDF is geometrically wrong even where it fits the budget. Do NOT re-enable on a
/// perf fix alone; the face planes must be corrected first. Free function so the
/// gate is unit-testable; `Demo::sdf_blocked_by_heavy_polychora` specializes it.
pub(crate) fn row_blocks_sdf(row: &[ShapeEntry]) -> bool {
    row.iter().any(|e| {
        matches!(
            e.shape,
            RaymarchShape::Polytope(Polytope4::Cell120 | Polytope4::Cell600)
        )
    })
}
