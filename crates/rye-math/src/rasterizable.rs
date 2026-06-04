//! [`RasterizableSpace<N>`] trait + [`Projection<N>`] enum + flat-Euclidean impls.
//!
//! Pairs with the `Visualizable<N>` trait in `rye-shape`. `Visualizable` answers "what mesh
//! data does this shape produce in R^N?"; `RasterizableSpace` answers "given that mesh data
//! in space `S`, how do we get screen-ready R³ vertices?". The rasterizer pipeline in
//! `rye-render` composes them.
//!
//! ## Unified for flat and curved spaces
//!
//! Existing [`Space`] impls in this crate use `glam::Vec3` / `Vec4` as their `Point` type, not
//! `[f32; N]`. So [`RasterizableSpace<N>`] is generic over [`Space`] rather than the array
//! type, and uses [`RasterizableSpace::point_to_array`] / [`RasterizableSpace::array_to_point`]
//! to bridge between the Space's native `Point` (math-friendly) and `[f32; N]`
//! (storage-friendly for mesh upload).
//!
//! The flat / curved distinction lives entirely in [`RasterizableSpace::tessellate_segment`]:
//! flat spaces use lerp; future curved spaces use `Space::exp` along the geodesic from `p0`
//! to `p1`. The rasterizer pipeline is identical for both, so geodesic-space wireframes drop
//! in as additional impls without changing call sites.
//!
//! ## Current scope
//!
//! Ships `RasterizableSpace<3> for EuclideanR3`. Other dimensions (`EuclideanR2`,
//! `EuclideanR4`) and curved spaces (`HyperbolicH3`, `SphericalS3`, `BlendedSpace`) are
//! additive extensions: add an `impl RasterizableSpace<N> for ...` block, no
//! rasterizer-pipeline changes required. The [`Projection<N>`] enum starts with
//! [`Projection::Identity`] only; more variants land alongside their consuming impls.

use glam::{Vec3, Vec4};

use crate::space::Space;
use crate::{EuclideanR3, EuclideanR4};

/// Sign-preserving floor for central-projection denominators (Perspective4D scale,
/// Schlegel ray parameter). A vertex on the viewer's 3-flat would otherwise divide by zero;
/// clamping the denominator to this magnitude yields a large-but-finite screen point instead
/// of NaN/Inf leaking into the upload buffer. The denominator is a difference of dot products
/// of unit-circumradius vertices with a unit normal, so its meaningful values are order 1;
/// `1e-4` sits comfortably above f32 roundoff on such a dot product yet far below any
/// geometrically real ray parameter, so the clamp engages only for a vertex essentially on the
/// viewer's 3-flat. The picture is meaningless at the clamp but the buffer stays finite.
const PROJECTION_DENOM_EPSILON: f32 = 1e-4;

/// Floor for the stereographic denominator `1 - dot(p, n)`. The pole is a real,
/// reachable input: the 16-cell has `±e_i` vertices and a cell-center pole such
/// as `(1,1,1,1)/2` coincides with a genuine tesseract vertex direction, so a
/// vertex landing on the pole gives `dot(p, n) = 1` and a bare divide NaNs the
/// whole upload buffer. Clamping the denominator to this magnitude yields a
/// large-but-finite point instead. Same order as `PROJECTION_DENOM_EPSILON`:
/// `dot(p, n)` is a dot of unit vectors so its meaningful denominator values are
/// order 1 (the antipode gives the maximum, 2); `1e-4` sits well above f32
/// roundoff yet far below any geometrically real value, so the clamp engages
/// only for a vertex essentially at the pole.
///
/// Exposed (`pub`) because a renderer that draws the stereographic image as a
/// finite polyline must clip the near-pole blow-up, and the clip radius is
/// interdependent with this floor: a sample inside the clamp band maps to
/// magnitude at most `sqrt((2 - eps)*eps) / eps`, i.e. on the order of
/// `sqrt(2 / eps)`, so a screen-space clip radius chosen strictly below that
/// ceiling reliably catches every clamp-saturated near-pole sample. The clip
/// itself lives in the rasterizer/demo layer, not here, so this map stays the
/// pure conformal projection (see the module-level note and `Projection::Stereographic`).
pub const STEREOGRAPHIC_POLE_EPSILON: f32 = 1e-4;

/// Orthonormal basis `(e1, e2, e3)` of the 3-flat perpendicular to the unit
/// vector `n`, for reading a point that lies in that 3-flat out as R³ (the
/// Schlegel diagram's screen coordinates). The basis is deterministic in `n`:
/// drop the world axis most aligned with `n`, then Gram-Schmidt the surviving
/// three axes (in x, y, z, w order) against `n` and each other (do Carmo,
/// *Differential Geometry of Curves and Surfaces*, §1.4, the standard
/// Gram-Schmidt construction). Dropping the most-aligned axis keeps the three
/// survivors well clear of `n`, so each residual is well-conditioned.
///
/// Returning the readout in this frame, rather than naively dropping the
/// `n`-aligned coordinate, is what makes the diagram faithful for a cell whose
/// normal is not axis-aligned: an oblique drop-w flattens the chosen boundary
/// cell (its `+n` vertex collapses toward the origin) and breaks the nesting
/// invariant.
///
/// The frame is a function of `n` alone, so it is identical for every vertex of
/// a fixed Schlegel projection. It is nonetheless rebuilt on every
/// [`RasterizableSpace::project_point`] call, which is once per vertex (per
/// tessellation sample) in the rasterizer upload path. That recompute is a
/// handful of f32 ops with no allocation and yields the byte-identical frame
/// each time (determinism intact), so it is left un-hoisted until Schlegel is
/// actually wired into the demo's per-frame wireframe rebuild and the redundant
/// rebuilds show up in a measurement. At that point the frame should be
/// resolved once per projection alongside `cell_normal` / `cell_offset` rather
/// than per vertex (see the note on [`Projection::Schlegel`]).
fn perp_frame(n: Vec4) -> (Vec4, Vec4, Vec4) {
    // Drop the world axis most aligned with `n` (its residual after projecting
    // out `n` is the shortest and worst-conditioned), then Gram-Schmidt the
    // other three against `n` and each other. Dropping the most-aligned axis
    // guarantees the remaining three stay linearly independent in the 3-flat.
    let ax = n.x.abs();
    let ay = n.y.abs();
    let az = n.z.abs();
    let aw = n.w.abs();
    let drop = ax.max(ay).max(az).max(aw);
    let mut seeds = [Vec4::X, Vec4::Y, Vec4::Z, Vec4::W];
    // Replace the most-aligned axis with a sentinel we then skip; ties resolve
    // toward the earliest axis so the choice is deterministic in `n`.
    let drop_idx = if drop == ax {
        0
    } else if drop == ay {
        1
    } else if drop == az {
        2
    } else {
        3
    };
    seeds[drop_idx] = Vec4::ZERO;

    let mut basis = [Vec4::ZERO; 3];
    let mut count = 0usize;
    for s in seeds {
        if s == Vec4::ZERO {
            continue;
        }
        let mut v = s - s.dot(n) * n;
        for b in basis.iter().take(count) {
            v -= v.dot(*b) * *b;
        }
        basis[count] = v.normalize();
        count += 1;
    }
    (basis[0], basis[1], basis[2])
}

/// Stereographic map of a point `p` on S³ to R³ from the unit `pole`
/// (Wikipedia, *Stereographic projection*). Shared by the `EuclideanR4` and
/// `SphericalS3Embedded` impls so the conformal map and its clamp discipline
/// have exactly one definition.
///
/// `image = (p - dot(p, pole)*pole) / (1 - dot(p, pole))`, the `pole`-perpendicular
/// component of `p` rescaled, then read out in the orthonormal frame of the
/// `pole`-perpendicular 3-flat. The numerator truncation is computed before the
/// divide so the result genuinely lies in that 3-flat (the bare `p / denom` form
/// leaks a `pole`-component); the readout against `perp_frame` then drops to the
/// in-3-flat coordinates.
///
/// `dot(p, pole)` is clamped to `[-1, 1]` so a slightly-off-unit `p` cannot push
/// the denominator outside `(0, 2]`, and the denominator is floored at
/// `STEREOGRAPHIC_POLE_EPSILON` so a vertex at the pole stays finite.
///
/// The default pole `Vec4::W` takes a closed-form fast path: the frame is the
/// literal `{x, y, z}` axes, so the readout is `(p.x, p.y, p.z) / (1 - p.w)` with
/// zero Gram-Schmidt. This is bit-identical to the general path for that pole
/// (the truncated numerator's first three components equal `p`'s, and dotting
/// against the axis frame just selects them), and it is the common case in the
/// demo, so it is pinned as the fast path. Non-default poles pay one stack-only
/// `perp_frame` build per call; the rebuild does not allocate and is left
/// un-hoisted until a trace shows the redundancy across the upload path is real.
pub(crate) fn stereographic_to_r3(p: Vec4, pole: Vec4) -> Vec3 {
    // Clamp matches `SphericalS3Embedded`'s dot-product discipline: a vertex a
    // hair off the unit sphere must not drive the denominator negative or past 2.
    let dot = p.dot(pole).clamp(-1.0, 1.0);
    let denom = (1.0 - dot).max(STEREOGRAPHIC_POLE_EPSILON);
    if pole == Vec4::W {
        // Closed-form drop-w: `perp_frame(W)` is exactly `(X, Y, Z)`, so the
        // frame readout reduces to dividing the first three components by `denom`.
        // No Gram-Schmidt, byte-identical to the general path below for this pole.
        return Vec3::new(p.x, p.y, p.z) / denom;
    }
    // Truncate to the pole-perpendicular component BEFORE dividing, so the image
    // lies in the pole-perpendicular 3-flat (the naive `p / denom` would carry a
    // nonzero pole-component into the readout).
    let perp = p - dot * pole;
    let scaled = perp / denom;
    let (e1, e2, e3) = perp_frame(pole);
    Vec3::new(scaled.dot(e1), scaled.dot(e2), scaled.dot(e3))
}

/// Projection from R^N to R³ for the rasterizer's screen-space transform.
///
/// All variants are dimension-generic in the type system, but each variant only makes sense for
/// specific `N`. Impls are expected to return `Vec3::ZERO` rather than panic when they receive
/// a variant they don't support; new variants are added alongside their first consuming impl
/// rather than speculatively.
///
/// - [`Identity`](Self::Identity): "use the first 3 components, zero-pad if `N < 3`, truncate
///   if `N > 3`." Default; works for `N == 3` (bitwise identity) and as a "natural" R⁴ to R³
///   view (drops `w`).
/// - [`Orthographic`](Self::Orthographic): drop one axis by index. The natural R⁴ wireframe
///   view is `Orthographic { drop_axis: 3 }` (drop `w`); other axis choices show alternative
///   3-flat slices. For `N == 3` this can drop one axis to produce a 2D-looking projection
///   (used later for Flatland-style reveals).
/// - [`Perspective4D`](Self::Perspective4D): R⁴ pinhole projection from a viewer at
///   `w = focal_distance` looking in -w. Produces the canonical "cube within a cube"
///   tesseract view; drop-w renders axis-aligned polytopes as degenerate flat shapes because
///   every pair of w-opposite vertices collapses to the same R³ point, and Perspective4D
///   separates them.
/// - [`Schlegel`](Self::Schlegel): R⁴ central projection of a 4-polytope from a viewpoint
///   just outside a chosen cell onto that cell's bounding 3-flat. The chosen cell becomes the
///   outer boundary; every other cell nests inside it (Coxeter, *Regular Polytopes*, ch. 13).
/// - [`Stereographic`](Self::Stereographic): conformal map of S³ to R³ from a chosen pole.
///   The natural projection for spherical-space polytopes; preserves angles, distorts
///   distances (Wikipedia, *Stereographic projection*).
///
/// Future variants under consideration: R⁴-specific `Hyperslice`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Projection<const N: usize> {
    /// Pass through: take the first 3 components, zero-pad if `N < 3`, truncate if `N > 3`.
    /// For `N == 3` this is bitwise identity. Default variant.
    #[default]
    Identity,

    /// Drop one axis by index (0-based). The remaining `N - 1` components fill R³ in their
    /// natural order, zero-padding if `N - 1 < 3`. For R⁴ with `drop_axis: 3` (the "drop-w"
    /// case) this produces `(x, y, z)`, the standard R⁴-into-R³ viewing convention. For R³
    /// with `drop_axis: 1` it produces `(x, z, 0)`, a 2D-looking projection in the XZ plane.
    ///
    /// Out-of-range `drop_axis` (>= `N`) returns `Vec3::ZERO`.
    Orthographic { drop_axis: usize },

    /// 4D pinhole projection from a viewer at `(0, 0, 0, focal_distance)` looking in the -w
    /// direction. A point `(x, y, z, w)` maps to `(x, y, z) * scale` where
    /// `scale = focal_distance / (focal_distance - w)`. Points at `w = 0` are unchanged;
    /// positive-w points (closer to the viewer in 4D) scale up; negative-w points scale down.
    /// For a unit-circumradius polytope (vertices in `[-1, 1]^4`) this produces the classical
    /// "cube within a cube" tesseract view: the +w face renders as the outer cube, the -w
    /// face as the inner cube, connecting edges as the frustum.
    ///
    /// **Precondition: `focal_distance > max(w)` across every input vertex.** Otherwise the
    /// denominator goes through zero (singularity at the viewer's eye) or negative (the
    /// projection flips inside-out). The impl clamps the denominator to a small positive
    /// epsilon rather than panicking, so a bad parameter degrades gracefully instead of NaN-
    /// ing the upload, but the resulting picture is meaningless.
    ///
    /// Only meaningful for `N == 4`; impls for other `N` return `Vec3::ZERO`.
    Perspective4D {
        /// Viewer position along the w-axis. Typical value for unit polytopes: `2.0`.
        focal_distance: f32,
    },

    /// 4D Schlegel diagram: central projection of a 4-polytope from a viewpoint placed just
    /// outside a chosen cell, onto that cell's bounding 3-flat (Coxeter, *Regular Polytopes*,
    /// 3rd ed., ch. 13, "Schlegel diagrams"). The chosen cell maps to the outer boundary of
    /// the diagram and every other cell projects into the interior, nested. This is the
    /// textbook 4D-to-3D picture: the 5-cell becomes a tetrahedron holding four nested
    /// tetrahedra, the tesseract a cube holding seven nested cubes.
    ///
    /// The chosen cell lies in the hyperplane `{x : dot(cell_normal, x) = cell_offset}` with
    /// `cell_normal` the outward unit normal of that cell. The eye sits on the outward radial
    /// axis at `E = viewpoint_distance * cell_normal`, beyond the cell (so
    /// `viewpoint_distance > cell_offset`). Each vertex `p` projects along the ray from `E`
    /// through `p` onto the cell hyperplane:
    /// `t = (cell_offset - dot(n, E)) / (dot(n, p) - dot(n, E))`,
    /// `result = E + t * (p - E)`.
    ///
    /// `result` lands in the 3-flat perpendicular to `cell_normal`; the readout expresses it
    /// in the supplied orthonormal `basis` for that 3-flat, not by dropping the w-coordinate.
    /// A naive w-drop is exact only for a w-aligned `n`; for any other cell normal it is an
    /// oblique projection that flattens the chosen boundary cell. The basis is part of the
    /// projection because its in-flat gauge is visible: a basis rebuilt from the live normal
    /// can snap during rotation when a "drop one axis" construction crosses a tie.
    ///
    /// **Precondition: `cell_normal` is the *outward* unit normal and
    /// `viewpoint_distance > cell_offset`.** With the inward normal the diagram fails to nest
    /// (interior cells escape the boundary). The denominator `dot(n, p) - dot(n, E)` is
    /// clamped sign-preservingly away from zero so a vertex sitting on the viewer's 3-flat
    /// yields a large-but-finite point rather than a NaN reaching the upload buffer.
    ///
    /// `cell_normal` / `cell_offset` are pre-resolved scalars; the variant carries no polytope
    /// reference. The demo resolves them once per cell selection from the polytope's topology
    /// (cell centroids), never inside the per-frame upload.
    ///
    /// Only meaningful for `N == 4`; impls for other `N` return `Vec3::ZERO`.
    Schlegel {
        /// Outward unit normal of the chosen boundary cell's hyperplane.
        cell_normal: Vec4,
        /// Signed plane offset: the chosen cell lies in the hyperplane
        /// `{x : dot(cell_normal, x) = cell_offset}`.
        cell_offset: f32,
        /// Eye distance along `cell_normal` from the origin; must exceed `cell_offset` so the
        /// viewpoint sits just outside the chosen cell.
        viewpoint_distance: f32,
        /// Orthonormal readout basis spanning the chosen cell's 3-flat.
        basis: [Vec4; 3],
    },

    /// Stereographic projection of the unit 3-sphere S³ onto R³ from a chosen `pole` (a unit
    /// `Vec4`), the canonical conformal S³ to R³ map (Wikipedia, *Stereographic projection*;
    /// the higher-dimensional generalization of the plane case). A point `p` on S³ maps to
    ///   `image = (p - dot(p, pole)*pole) / (1 - dot(p, pole))`,
    /// the component of `p` perpendicular to `pole` rescaled by the stereographic factor. The
    /// result lies in the 3-flat perpendicular to `pole`; the readout expresses it in an
    /// orthonormal basis of that 3-flat (see `perp_frame`). For pole `Vec4::W`
    /// this collapses to the closed form `(p.x, p.y, p.z) / (1 - p.w)` with zero
    /// frame math.
    ///
    /// **The truncated `n`-perpendicular numerator is load-bearing, not
    /// cosmetic.** Writing the naive `scaled = p / (1 - dot(p, pole))` as a full
    /// 4-vector and relying on a later "express in the frame" to drop the radial
    /// part silently leaks a nonzero `pole`-component into the readout (the bare
    /// 4-vector is not in the `pole`-perpendicular hyperplane), which corrupts the
    /// projection; subtract `dot(p, pole)*pole` *before* dividing.
    ///
    /// **Precondition: `pole` is a unit vector and `p` lies on S³ (`|p| = 1`).**
    /// `dot(p, pole)` is clamped into `[-1, 1]` before forming the denominator,
    /// matching the clamp discipline in [`crate::SphericalS3Embedded`], so a
    /// slightly-off-unit upstream vertex cannot push the denominator outside
    /// `(0, 2]`. The pole itself is reachable (the 16-cell has `±e_i` vertices;
    /// a cell-center pole such as `(1,1,1,1)/2` coincides with a real tesseract
    /// vertex direction), so the denominator is floored at `STEREOGRAPHIC_POLE_EPSILON`:
    /// a vertex at the pole yields a large-but-finite point rather than a NaN
    /// reaching the upload buffer. The antipode (`-pole`) is the safe far point,
    /// `dot = -1`, denominator `2`, mapping to the origin.
    ///
    /// `EuclideanR4`'s impl normalizes its input onto S³ first, since
    /// stereographic is only meaningful on the unit sphere and polytope vertices
    /// fed through the demo are `body_size`-scaled, not unit; that normalize is
    /// the precondition guard. The [`crate::SphericalS3Embedded`] impl computes
    /// the stereographic map directly (a true conformal S³ view), rather than
    /// delegating to the flat R⁴ projection as it does for the other variants.
    ///
    /// The frame is a function of `pole` alone, so it is identical across every
    /// vertex of a fixed projection, but [`RasterizableSpace::project_point`]
    /// rebuilds it per call (per vertex) like [`Schlegel`](Self::Schlegel); the
    /// `Vec4::W` closed form does zero frame math, while a general pole pays the
    /// `perp_frame` rebuild (see `perp_frame` for why it is left un-hoisted).
    /// Stereographic is non-affine, so it cannot be represented by an affine
    /// section-cap scalar shim.
    ///
    /// Only meaningful for `N == 4`; impls for other `N` return `Vec3::ZERO`.
    Stereographic {
        /// Unit `Vec4` pole the projection casts away from. Pole `Vec4::W` gives
        /// the textbook closed-form `(x, y, z) / (1 - w)` map; the pole the demo
        /// actually selects lives in the demo layer (this map stays pole-agnostic).
        pole: Vec4,
    },
}

impl Projection<4> {
    /// Build a Schlegel projection with the deterministic normal-only basis.
    pub fn schlegel(cell_normal: Vec4, cell_offset: f32, viewpoint_distance: f32) -> Projection<4> {
        let (e1, e2, e3) = perp_frame(cell_normal);
        Self::schlegel_with_basis(cell_normal, cell_offset, viewpoint_distance, [e1, e2, e3])
    }

    /// Build a Schlegel projection with an explicit 3-flat readout basis.
    pub fn schlegel_with_basis(
        cell_normal: Vec4,
        cell_offset: f32,
        viewpoint_distance: f32,
        basis: [Vec4; 3],
    ) -> Projection<4> {
        Projection::Schlegel {
            cell_normal,
            cell_offset,
            viewpoint_distance,
            basis,
        }
    }
}

/// A flat or curved space that can drive the rasterizer pipeline: provides projection from its
/// native point representation to R³, plus segment tessellation.
///
/// `N` is the const-generic ambient dimension matching the `Visualizable<N>` mesh data in
/// `rye-shape`. Implementations bridge between the Space's native `Point` type (typically
/// `glam::Vec3` or `Vec4`) and `[f32; N]` (mesh storage).
pub trait RasterizableSpace<const N: usize>: Space {
    /// Convert a space-native point to the mesh storage representation `[f32; N]`.
    fn point_to_array(p: Self::Point) -> [f32; N];

    /// Inverse of [`point_to_array`](Self::point_to_array).
    fn array_to_point(arr: [f32; N]) -> Self::Point;

    /// Project a point in this space to R³ for the camera's view-projection stage. The
    /// projection mode is given by `projection`; for `N == 3` and `Projection::Identity`
    /// this is the trivial pass-through.
    fn project_point(point: Self::Point, projection: &Projection<N>) -> Vec3;

    /// Tessellate a segment into space-native points and append them to `out`. Always called
    /// from the rasterizer's upload path, so the GPU receives pre-tessellated segments
    /// uniformly regardless of curvature.
    ///
    /// `samples` is the number of subdivisions (not the total point count): `samples == 1`
    /// appends `[p0, p1]`; `samples == 4` appends 5 points (`p0`, three interior lerps, `p1`).
    /// For flat spaces this is straight linear interpolation. For curved spaces (future) it
    /// samples along [`Space::exp`] / [`Space::log`].
    ///
    /// **Writer pattern, not return-by-value.** The upload loop reuses one `Vec` across all
    /// segments to keep allocations off the per-segment hot path. Implementors call
    /// `out.push` for each output point; they do not call `out.clear` (the caller owns the
    /// buffer and may want to accumulate across multiple segments).
    fn tessellate_segment(
        p0: Self::Point,
        p1: Self::Point,
        samples: usize,
        out: &mut Vec<Self::Point>,
    );
}

impl RasterizableSpace<3> for EuclideanR3 {
    fn point_to_array(p: Vec3) -> [f32; 3] {
        p.to_array()
    }

    fn array_to_point(arr: [f32; 3]) -> Vec3 {
        Vec3::from_array(arr)
    }

    fn project_point(point: Vec3, projection: &Projection<3>) -> Vec3 {
        match projection {
            Projection::Identity => point,
            Projection::Orthographic { drop_axis } => match *drop_axis {
                0 => Vec3::new(point.y, point.z, 0.0),
                1 => Vec3::new(point.x, point.z, 0.0),
                2 => Vec3::new(point.x, point.y, 0.0),
                _ => Vec3::ZERO,
            },
            // Perspective4D, Schlegel, and Stereographic are meaningful only for `N == 4`; on
            // R³ there's no w axis (nor a 3-sphere) to project from. Per the enum contract,
            // return zero rather than panic.
            Projection::Perspective4D { .. }
            | Projection::Schlegel { .. }
            | Projection::Stereographic { .. } => Vec3::ZERO,
        }
    }

    fn tessellate_segment(p0: Vec3, p1: Vec3, samples: usize, out: &mut Vec<Vec3>) {
        out.push(p0);
        for i in 1..samples {
            let t = i as f32 / samples as f32;
            out.push(p0.lerp(p1, t));
        }
        out.push(p1);
    }
}

impl RasterizableSpace<4> for EuclideanR4 {
    fn point_to_array(p: Vec4) -> [f32; 4] {
        p.to_array()
    }

    fn array_to_point(arr: [f32; 4]) -> Vec4 {
        Vec4::from_array(arr)
    }

    fn project_point(point: Vec4, projection: &Projection<4>) -> Vec3 {
        match projection {
            // Identity on R⁴: truncate to the first three components. Equivalent to drop-w.
            Projection::Identity => Vec3::new(point.x, point.y, point.z),
            Projection::Orthographic { drop_axis } => match *drop_axis {
                0 => Vec3::new(point.y, point.z, point.w),
                1 => Vec3::new(point.x, point.z, point.w),
                2 => Vec3::new(point.x, point.y, point.w),
                3 => Vec3::new(point.x, point.y, point.z),
                _ => Vec3::ZERO,
            },
            Projection::Perspective4D { focal_distance } => {
                // Pinhole from viewer at `(0, 0, 0, focal_distance)` looking in -w. The
                // denominator clamp guards against zero / negative values: callers should
                // pick `focal_distance > max(w)` so the clamp never engages on legitimate
                // input, but clamping lets a misconfigured projection degrade visibly
                // rather than emit NaN through the rest of the upload path.
                let denom = (focal_distance - point.w).max(PROJECTION_DENOM_EPSILON);
                let scale = focal_distance / denom;
                Vec3::new(point.x, point.y, point.z) * scale
            }
            Projection::Schlegel {
                cell_normal,
                cell_offset,
                viewpoint_distance,
                basis,
            } => {
                // Central projection from the eye `E = viewpoint_distance * n` through the
                // vertex `point` onto the chosen cell's 3-flat `{x : dot(n, x) = cell_offset}`
                // (Coxeter, *Regular Polytopes*, ch. 13). Ray: `E + t * (point - E)`; solve
                // `dot(n, E + t*(point - E)) = cell_offset` for the hit parameter
                //   t = (cell_offset - dot(n, E)) / (dot(n, point) - dot(n, E)).
                // A chosen-cell vertex has `dot(n, point) = cell_offset`, giving `t = 1` and
                // `result = point`: the cell maps to itself (the diagram's outer boundary).
                let n = *cell_normal;
                let eye = *viewpoint_distance * n;
                let n_dot_eye = n.dot(eye);
                // Sign-preserving clamp: a vertex on the viewer's 3-flat
                // (`dot(n, point) == dot(n, eye)`) would divide by zero. Keeping the sign
                // means the clamped point shoots off in the geometrically correct direction
                // (huge but finite) instead of flipping across the eye.
                let raw_denom = n.dot(point) - n_dot_eye;
                let denom = if raw_denom.abs() < PROJECTION_DENOM_EPSILON {
                    PROJECTION_DENOM_EPSILON.copysign(raw_denom)
                } else {
                    raw_denom
                };
                let t = (*cell_offset - n_dot_eye) / denom;
                let result = eye + t * (point - eye);
                // Read in the caller-supplied frame so interactive Schlegel diagrams keep a
                // stable in-flat orientation while the body rotates.
                let [e1, e2, e3] = *basis;
                Vec3::new(result.dot(e1), result.dot(e2), result.dot(e3))
            }
            Projection::Stereographic { pole } => {
                // Stereographic is only defined on the unit sphere, but demo
                // vertices are body-scaled. Normalize onto S3 first, then apply
                // the conformal map with the pole-denominator clamp.
                stereographic_to_r3(point.normalize(), *pole)
            }
        }
    }

    fn tessellate_segment(p0: Vec4, p1: Vec4, samples: usize, out: &mut Vec<Vec4>) {
        out.push(p0);
        for i in 1..samples {
            let t = i as f32 / samples as f32;
            out.push(p0.lerp(p1, t));
        }
        out.push(p1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SphericalS3Embedded;
    use approx::assert_relative_eq;

    /// Golden Vec3 for `stereographic_frame_is_deterministic_under_tie`.
    /// Hand-derived from `perp_frame` for pole `(0,0,1,1)/sqrt(2)` and input
    /// `(0.5, 0.5, -0.5, 0.5)`: the z/w tie drops z, giving the frame
    /// `(X, Y, (0,0,-1,1)/sqrt(2))`. With `dot(p, pole) = 0`, the readout is
    /// `(0.5, 0.5, 1/sqrt(2))`. A future tie-break or Gram-Schmidt change that
    /// flips the gauge changes this value and the test catches it.
    /// `1/sqrt(2)` is `FRAC_1_SQRT_2`, not a bare literal.
    const GOLDEN_TIE_FRAME: Vec3 = Vec3::new(0.5, 0.5, std::f32::consts::FRAC_1_SQRT_2);

    /// `point_to_array` is the inverse of `array_to_point` on `EuclideanR3` for any input.
    #[test]
    fn r3_array_round_trip() {
        let p = Vec3::new(1.0, -2.5, 0.7);
        let arr = <EuclideanR3 as RasterizableSpace<3>>::point_to_array(p);
        let back = <EuclideanR3 as RasterizableSpace<3>>::array_to_point(arr);
        assert_eq!(p, back);
    }

    /// The `EuclideanR4` Stereographic arm normalizes its input onto S³ before
    /// applying the conformal map, so a `body_size`-scaled vertex `k * p`
    /// projects to the same R³ point as the unit vertex `p`. This pins the
    /// `point.normalize()` precondition guard: without it the demo's non-unit
    /// vertices would project through a wrong-radius denominator, and every
    /// other test feeds already-unit input so none would catch its removal.
    #[test]
    fn stereographic_r4_normalizes_scaled_input() {
        let proj = Projection::Stereographic { pole: Vec4::W };
        for p in [
            Vec4::new(0.3, -0.1, 0.2, -0.5).normalize(),
            Vec4::new(-0.4, 0.6, 0.1, 0.3).normalize(),
            Vec4::new(0.0, 0.0, 0.0, -1.0),
        ] {
            let unit = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &proj);
            for k in [0.25_f32, 1.5, 3.0] {
                let scaled = <EuclideanR4 as RasterizableSpace<4>>::project_point(k * p, &proj);
                assert!(
                    scaled.abs_diff_eq(unit, 1e-5),
                    "scale {k}: {scaled:?} vs {unit:?}"
                );
            }
        }
    }

    /// `Projection::Identity` on R³ is bitwise identity: the projected point equals the input.
    #[test]
    fn r3_identity_projection_is_passthrough() {
        let p = Vec3::new(0.7, -1.3, 2.1);
        let projected =
            <EuclideanR3 as RasterizableSpace<3>>::project_point(p, &Projection::Identity);
        assert_eq!(p, projected);
    }

    /// `tessellate_segment(p0, p1, 1, out)` appends exactly
    /// `[p0, p1]` and nothing else. The "one subdivision" case is the
    /// default for flat spaces where no interior sampling is needed.
    #[test]
    fn r3_tessellate_one_sample_appends_endpoints() {
        let p0 = Vec3::new(0.0, 0.0, 0.0);
        let p1 = Vec3::new(2.0, 4.0, -6.0);
        let mut out = Vec::new();
        <EuclideanR3 as RasterizableSpace<3>>::tessellate_segment(p0, p1, 1, &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], p0);
        assert_eq!(out[1], p1);
    }

    /// `tessellate_segment(p0, p1, 4, out)` appends 5 points: `p0`, three interior lerps at
    /// t = 1/4, 2/4, 3/4, and `p1`. Verifies the lerp factor convention.
    #[test]
    fn r3_tessellate_four_samples_produces_five_points() {
        let p0 = Vec3::new(0.0, 0.0, 0.0);
        let p1 = Vec3::new(4.0, 0.0, 0.0);
        let mut out = Vec::new();
        <EuclideanR3 as RasterizableSpace<3>>::tessellate_segment(p0, p1, 4, &mut out);
        assert_eq!(out.len(), 5);
        assert_eq!(out[0], p0);
        assert_eq!(out[1], Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(out[2], Vec3::new(2.0, 0.0, 0.0));
        assert_eq!(out[3], Vec3::new(3.0, 0.0, 0.0));
        assert_eq!(out[4], p1);
    }

    /// `tessellate_segment` appends to an existing buffer instead of clearing it. This is the
    /// "writer pattern, not return-by-value" guarantee the upload loop depends on for
    /// allocation reuse.
    #[test]
    fn r3_tessellate_appends_does_not_clear() {
        let mut out = vec![Vec3::new(9.0, 9.0, 9.0)];
        <EuclideanR3 as RasterizableSpace<3>>::tessellate_segment(Vec3::ZERO, Vec3::X, 1, &mut out);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], Vec3::new(9.0, 9.0, 9.0));
        assert_eq!(out[1], Vec3::ZERO);
        assert_eq!(out[2], Vec3::X);
    }

    /// Default `Projection<N>` is `Identity`. Pins the const-generic enum default so callers
    /// can `Projection::default()` without specifying a variant.
    #[test]
    fn projection_default_is_identity() {
        let p3: Projection<3> = Projection::default();
        assert_eq!(p3, Projection::Identity);
        let p4: Projection<4> = Projection::default();
        assert_eq!(p4, Projection::Identity);
    }

    /// `Orthographic { drop_axis: 1 }` on R³ produces the XZ plane embedded in R³ with Y
    /// zeroed. The 2D-looking projection used by camera tweens that want to dramatize a
    /// "flat to depth" reveal.
    #[test]
    fn r3_orthographic_drops_named_axis() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let pj = |drop_axis| {
            <EuclideanR3 as RasterizableSpace<3>>::project_point(
                p,
                &Projection::Orthographic { drop_axis },
            )
        };
        assert_eq!(pj(0), Vec3::new(2.0, 3.0, 0.0));
        assert_eq!(pj(1), Vec3::new(1.0, 3.0, 0.0));
        assert_eq!(pj(2), Vec3::new(1.0, 2.0, 0.0));
        // Out-of-range falls back to zero per the doc contract.
        assert_eq!(pj(3), Vec3::ZERO);
        assert_eq!(pj(99), Vec3::ZERO);
    }

    /// `EuclideanR4` round-trips through point/array conversion for any input.
    #[test]
    fn r4_array_round_trip() {
        let p = Vec4::new(1.0, -2.5, 0.7, 4.2);
        let arr = <EuclideanR4 as RasterizableSpace<4>>::point_to_array(p);
        let back = <EuclideanR4 as RasterizableSpace<4>>::array_to_point(arr);
        assert_eq!(p, back);
    }

    /// `Projection::Identity` on R⁴ truncates `w` (equivalent to `Orthographic` with
    /// `drop_axis = 3`). This is the "natural" 4D-to-3D viewing convention.
    #[test]
    fn r4_identity_drops_w() {
        let p = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let projected =
            <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &Projection::Identity);
        assert_eq!(projected, Vec3::new(1.0, 2.0, 3.0));
    }

    /// `Orthographic { drop_axis }` on R⁴ for each of the four axis choices produces the
    /// expected 3-flat view. drop_axis=3 (drop w) is the canonical wireframe-rendering case.
    #[test]
    fn r4_orthographic_drops_each_axis() {
        let p = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let pj = |drop_axis| {
            <EuclideanR4 as RasterizableSpace<4>>::project_point(
                p,
                &Projection::Orthographic { drop_axis },
            )
        };
        assert_eq!(pj(0), Vec3::new(2.0, 3.0, 4.0));
        assert_eq!(pj(1), Vec3::new(1.0, 3.0, 4.0));
        assert_eq!(pj(2), Vec3::new(1.0, 2.0, 4.0));
        assert_eq!(pj(3), Vec3::new(1.0, 2.0, 3.0));
        // Out-of-range falls back to zero per the doc contract.
        assert_eq!(pj(4), Vec3::ZERO);
    }

    /// `Projection::Perspective4D` projects a `w = 0` point unchanged: the viewer sits at
    /// `(0, 0, 0, focal_distance)` and a point on the `w = 0` 3-flat is at perpendicular
    /// distance `focal_distance` from the eye, giving `scale = focal_distance / focal_distance
    /// = 1`. Pins the boundary case where Perspective4D collapses to Identity.
    #[test]
    fn r4_perspective4d_w_zero_is_unchanged() {
        let p = Vec4::new(1.0, 2.0, 3.0, 0.0);
        let proj = Projection::Perspective4D {
            focal_distance: 2.0,
        };
        let got = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &proj);
        assert_eq!(got, Vec3::new(1.0, 2.0, 3.0));
    }

    /// `Projection::Perspective4D` on the tesseract's w-extreme vertices produces the
    /// canonical "cube within a cube" scale relationship: the +w face renders as the outer
    /// (larger) cube and the -w face as the inner (smaller) cube. Pins the qualitative
    /// behavior the demo depends on by checking the two end scales and their ratio.
    #[test]
    fn r4_perspective4d_cube_within_cube_scaling() {
        let focal = 2.0;
        let proj = Projection::Perspective4D {
            focal_distance: focal,
        };
        let near = Vec4::new(0.5, 0.5, 0.5, 0.5);
        let far = Vec4::new(0.5, 0.5, 0.5, -0.5);
        let pn = <EuclideanR4 as RasterizableSpace<4>>::project_point(near, &proj);
        let pf = <EuclideanR4 as RasterizableSpace<4>>::project_point(far, &proj);
        // Scale at w=+0.5 is `2 / (2 - 0.5) = 4/3`; at w=-0.5 it is
        // `2 / 2.5 = 4/5`. Each R3 component scales by the same factor, so
        // the magnitude ratio is `(4/3)/(4/5) = 5/3`.
        let r_near = (pn.length() / 0.5_f32.mul_add(3.0_f32.sqrt(), 0.0)).abs(); // |near|/|input|
        let r_far = (pf.length() / 0.5_f32.mul_add(3.0_f32.sqrt(), 0.0)).abs();
        assert!((r_near - 4.0 / 3.0).abs() < 1e-5, "near scale {r_near}");
        assert!((r_far - 4.0 / 5.0).abs() < 1e-5, "far scale {r_far}");
        // Outer cube (|+w face|) > inner cube (|-w face|).
        assert!(pn.length() > pf.length(), "near={pn:?} far={pf:?}");
    }

    /// `Projection::Perspective4D` clamps the denominator rather than dividing by zero or
    /// going negative. A point at the viewer's eye (`w == focal_distance`) would naively
    /// produce infinite scale; the impl returns a finite (large) result instead so a single
    /// misconfigured vertex doesn't NaN the entire upload buffer.
    #[test]
    fn r4_perspective4d_at_viewer_clamps_finite() {
        let p = Vec4::new(0.1, 0.2, 0.3, 2.0);
        let proj = Projection::Perspective4D {
            focal_distance: 2.0,
        };
        let got = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &proj);
        for c in [got.x, got.y, got.z] {
            assert!(
                c.is_finite(),
                "expected finite output at viewer, got {got:?}"
            );
        }
    }

    /// `Projection::Perspective4D` on R³ falls back to `Vec3::ZERO` per the enum's
    /// "unsupported variant returns zero" contract (R³ has no w axis to project from).
    #[test]
    fn r3_perspective4d_returns_zero() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let proj = Projection::Perspective4D {
            focal_distance: 2.0,
        };
        let got = <EuclideanR3 as RasterizableSpace<3>>::project_point(p, &proj);
        assert_eq!(got, Vec3::ZERO);
    }

    // ---- Schlegel ------------------------------------------------------

    /// The eight tesseract vertices, unit-circumradius (`±0.5` in every coordinate). Shared
    /// across the Schlegel tests; the cell at `w = +0.5` is the canonical boundary cell.
    const TESSERACT_VERTS: [Vec4; 16] = [
        Vec4::new(0.5, 0.5, 0.5, 0.5),
        Vec4::new(-0.5, 0.5, 0.5, 0.5),
        Vec4::new(0.5, -0.5, 0.5, 0.5),
        Vec4::new(-0.5, -0.5, 0.5, 0.5),
        Vec4::new(0.5, 0.5, -0.5, 0.5),
        Vec4::new(-0.5, 0.5, -0.5, 0.5),
        Vec4::new(0.5, -0.5, -0.5, 0.5),
        Vec4::new(-0.5, -0.5, -0.5, 0.5),
        Vec4::new(0.5, 0.5, 0.5, -0.5),
        Vec4::new(-0.5, 0.5, 0.5, -0.5),
        Vec4::new(0.5, -0.5, 0.5, -0.5),
        Vec4::new(-0.5, -0.5, 0.5, -0.5),
        Vec4::new(0.5, 0.5, -0.5, -0.5),
        Vec4::new(-0.5, 0.5, -0.5, -0.5),
        Vec4::new(0.5, -0.5, -0.5, -0.5),
        Vec4::new(-0.5, -0.5, -0.5, -0.5),
    ];

    /// Vertices of the chosen boundary cell map onto the diagram's outer boundary undistorted:
    /// they satisfy `dot(n, v) = cell_offset`, so the ray parameter is exactly `t = 1` and
    /// `result = v` (Coxeter, *Regular Polytopes*, ch. 13). The frame readout reports `result`
    /// in an orthonormal basis of the cell's 3-flat, which is an isometry of that 3-flat, so it
    /// preserves the cell's intrinsic shape. Pin that by checking pairwise distances among the
    /// chosen-cell vertices match the original cell's: the boundary cube renders at true size,
    /// not flattened. Checked over the `w = +0.5` cell of a synthetic unit cube.
    #[test]
    fn schlegel_chosen_cell_renders_undistorted() {
        let cell_offset = 0.5;
        let proj = Projection::schlegel(Vec4::W, cell_offset, 1.5 * cell_offset);
        // The eight vertices of the `w = +0.5` cell (the first half of TESSERACT_VERTS).
        let cell: Vec<Vec4> = TESSERACT_VERTS.iter().take(8).copied().collect();
        let projected: Vec<Vec3> = cell
            .iter()
            .map(|v| <EuclideanR4 as RasterizableSpace<4>>::project_point(*v, &proj))
            .collect();
        // The readout is an isometry, so every pairwise distance is preserved. The cube's
        // 3D distances ignore the shared `w`, so compare against the xyz distance.
        for i in 0..cell.len() {
            for j in (i + 1)..cell.len() {
                let orig = (Vec3::new(cell[i].x, cell[i].y, cell[i].z)
                    - Vec3::new(cell[j].x, cell[j].y, cell[j].z))
                .length();
                let got = (projected[i] - projected[j]).length();
                assert!(
                    (orig - got).abs() < 1e-5,
                    "chosen-cell distance v{i}-v{j} should be {orig}, got {got}"
                );
            }
        }
    }

    /// The chosen boundary cell stays a genuine 3D cell, not a flattened one. The frame readout
    /// fixed the w-drop bug where a cell normal with a w-component collapsed the boundary's
    /// `+n` vertex onto the origin. Pin non-degeneracy directly: the eight projected boundary
    /// vertices span all three R³ axes (every coordinate has nonzero spread). Uses the 16-cell
    /// cell `{+x,+y,+z,+w}`, whose normal `(1,1,1,1)/2` is the worst case for the old w-drop.
    #[test]
    fn schlegel_non_axis_aligned_cell_is_not_flattened() {
        // 16-cell vertices: the eight signed unit axis points.
        let verts = [
            Vec4::X,
            -Vec4::X,
            Vec4::Y,
            -Vec4::Y,
            Vec4::Z,
            -Vec4::Z,
            Vec4::W,
            -Vec4::W,
        ];
        // Chosen cell {+x,+y,+z,+w}; centroid direction is the outward normal.
        let centroid = (Vec4::X + Vec4::Y + Vec4::Z + Vec4::W) / 4.0;
        let cell_offset = centroid.length();
        let cell_normal = centroid / cell_offset;
        let proj = Projection::schlegel(cell_normal, cell_offset, 1.5 * cell_offset);
        // The four chosen-cell vertices (the +axes) are the boundary; the four -axes nest.
        let boundary = [Vec4::X, Vec4::Y, Vec4::Z, Vec4::W];
        let inner = [-Vec4::X, -Vec4::Y, -Vec4::Z, -Vec4::W];
        let proj_pt = |v: Vec4| <EuclideanR4 as RasterizableSpace<4>>::project_point(v, &proj);

        // Boundary vertices are equidistant from the diagram center (a regular tetrahedron),
        // and each strictly farther from center than every nested vertex.
        let boundary_r: Vec<f32> = boundary.iter().map(|&v| proj_pt(v).length()).collect();
        let inner_r: Vec<f32> = inner.iter().map(|&v| proj_pt(v).length()).collect();
        let r0 = boundary_r[0];
        assert!(r0 > 1e-3, "boundary must not collapse to the origin");
        for r in &boundary_r {
            assert!(
                (r - r0).abs() < 1e-5,
                "boundary tetrahedron must be regular, radii {boundary_r:?}"
            );
        }
        for r in &inner_r {
            assert!(
                *r < r0 - 1e-3,
                "every nested vertex must sit inside the boundary, inner {inner_r:?} vs {r0}"
            );
        }
        // Directly pin the oblique-normal isometry. The four boundary +axes are
        // mutually equidistant at `sqrt(2)`, and each lies on the cell hyperplane
        // (`t = 1`, projects to itself), so the orthonormal-frame readout must
        // preserve every pairwise distance. A naive oblique drop-w would not.
        let proj_boundary: Vec<Vec3> = boundary.iter().map(|&v| proj_pt(v)).collect();
        let edge_len = 2.0_f32.sqrt();
        for i in 0..proj_boundary.len() {
            for j in (i + 1)..proj_boundary.len() {
                let got = (proj_boundary[i] - proj_boundary[j]).length();
                assert!(
                    (got - edge_len).abs() < 1e-5,
                    "oblique boundary edge {i}-{j} should stay {edge_len}, got {got}"
                );
            }
        }
        // Sanity: the diagram has real 3D extent on every axis. The radius/nesting asserts
        // above are what actually discriminate the old w-drop bug (which sent `+n` to the
        // origin, failing `r0 > 1e-3` and the regular-tetrahedron check); this spread check is
        // only a coarse non-degeneracy guard, since a naive drop-w keeps the `±axis` extents
        // here too.
        let all: Vec<Vec3> = verts.iter().map(|&v| proj_pt(v)).collect();
        for axis in 0..3 {
            let comp = |p: Vec3| [p.x, p.y, p.z][axis];
            let spread = all
                .iter()
                .map(|&p| comp(p))
                .fold(f32::NEG_INFINITY, f32::max)
                - all.iter().map(|&p| comp(p)).fold(f32::INFINITY, f32::min);
            assert!(
                spread > 0.5,
                "axis {axis} should have real spread, got {spread}"
            );
        }
    }

    /// Schlegel readout uses the supplied basis, not a rebuilt normal-only frame.
    /// The point lies on the chosen boundary cell, so central projection returns
    /// it unchanged and only the basis ordering can affect the R3 coordinates.
    #[test]
    fn schlegel_uses_supplied_basis_for_readout() {
        let p = Vec4::new(0.5, -0.25, 0.125, 0.5);
        let xyz = Projection::schlegel_with_basis(Vec4::W, 0.5, 0.75, [Vec4::X, Vec4::Y, Vec4::Z]);
        let yxz = Projection::schlegel_with_basis(Vec4::W, 0.5, 0.75, [Vec4::Y, Vec4::X, Vec4::Z]);

        let a = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &xyz);
        let b = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &yxz);

        assert_eq!(a, Vec3::new(0.5, -0.25, 0.125));
        assert_eq!(b, Vec3::new(-0.25, 0.5, 0.125));
    }

    /// Every tesseract vertex projects to an all-finite R³ point under the default viewpoint
    /// (`viewpoint_distance = 1.5 * cell_offset`). The viewpoint clears the chosen cell, so no
    /// vertex sits on the viewer 3-flat and the denominator clamp never engages on this input;
    /// the assertion still guards the whole vertex set, including the `w = -0.5` cell that the
    /// eye looks through.
    #[test]
    fn schlegel_projection_is_always_finite() {
        let cell_offset = 0.5;
        let proj = Projection::schlegel(Vec4::W, cell_offset, 1.5 * cell_offset);
        for v in TESSERACT_VERTS {
            let got = <EuclideanR4 as RasterizableSpace<4>>::project_point(v, &proj);
            for c in [got.x, got.y, got.z] {
                assert!(
                    c.is_finite(),
                    "vertex {v:?} projected to non-finite {got:?}"
                );
            }
        }
    }

    /// A vertex on the viewer's 3-flat (`dot(n, p) == dot(n, E)`) would divide by zero in the
    /// ray parameter; the sign-preserving denominator clamp yields a finite (large) result
    /// instead, mirroring `r4_perspective4d_at_viewer_clamps_finite`. Constructed so
    /// `dot(W, p) == dot(W, E) == viewpoint_distance`.
    #[test]
    fn schlegel_zero_denominator_clamps_finite() {
        let viewpoint_distance = 0.75;
        let proj = Projection::schlegel(Vec4::W, 0.5, viewpoint_distance);
        // `w = viewpoint_distance` puts the vertex exactly on the eye's 3-flat.
        let p = Vec4::new(0.3, -0.2, 0.1, viewpoint_distance);
        let got = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &proj);
        for c in [got.x, got.y, got.z] {
            assert!(
                c.is_finite(),
                "degenerate denominator should clamp finite, got {got:?}"
            );
        }
    }

    /// The normal sign is load-bearing: with the *outward* normal the opposite cell nests
    /// inside the boundary (its projected radius is smaller than the boundary cube's
    /// half-diagonal); flipping to the *inward* normal (eye on the wrong side) blows that
    /// vertex out past the boundary. Pins that callers must supply the outward normal.
    /// Compares radii from the diagram center rather than per-axis extents because the frame
    /// readout rotates the xyz axes; radius is the rotation-invariant nesting measure.
    #[test]
    fn schlegel_outward_normal_sign_required() {
        let cell_offset = 0.5;
        let viewpoint_distance = 1.5 * cell_offset;
        let outward = Projection::schlegel(Vec4::W, cell_offset, viewpoint_distance);
        // Same physical `w = +0.5` cell described with the inward normal `-W`: its offset
        // flips sign, and the eye lands on the far side of the polytope.
        let inward = Projection::schlegel(-Vec4::W, -cell_offset, viewpoint_distance);
        // A vertex of the opposite (`w = -0.5`) cell.
        let opposite = Vec4::new(0.5, 0.5, 0.5, -0.5);
        // The boundary cube's corners sit at radius `sqrt(3)/2` from the diagram center.
        let boundary_radius = (0.75_f32).sqrt();
        let nested = <EuclideanR4 as RasterizableSpace<4>>::project_point(opposite, &outward);
        let escaped = <EuclideanR4 as RasterizableSpace<4>>::project_point(opposite, &inward);
        assert!(
            nested.length() < boundary_radius,
            "outward normal must nest the opposite cell (r {} < {boundary_radius}), got {nested:?}",
            nested.length()
        );
        assert!(
            escaped.length() > boundary_radius,
            "inward normal must push the opposite cell outside (r {} > {boundary_radius}), got {escaped:?}",
            escaped.length()
        );
    }

    /// `perp_frame(n)` returns an orthonormal triple spanning the 3-flat perpendicular to `n`,
    /// for a spread of normals including the axis-aligned and the all-equal worst cases. This
    /// is the invariant the Schlegel readout relies on; if it breaks, the diagram skews.
    #[test]
    fn perp_frame_is_orthonormal_and_perpendicular_to_normal() {
        let normals = [
            Vec4::W,
            Vec4::X,
            Vec4::new(0.5, 0.5, 0.5, 0.5),
            Vec4::new(0.1, -0.2, 0.3, 0.9).normalize(),
            Vec4::new(-0.6, 0.0, 0.8, 0.0).normalize(),
        ];
        for n in normals {
            let (e1, e2, e3) = perp_frame(n);
            for (label, e) in [("e1", e1), ("e2", e2), ("e3", e3)] {
                assert!(
                    (e.length() - 1.0).abs() < 1e-5,
                    "{label} not unit for n {n:?}"
                );
                assert!(
                    e.dot(n).abs() < 1e-5,
                    "{label} not perpendicular to n {n:?}"
                );
            }
            assert!(e1.dot(e2).abs() < 1e-5, "e1·e2 != 0 for n {n:?}");
            assert!(e1.dot(e3).abs() < 1e-5, "e1·e3 != 0 for n {n:?}");
            assert!(e2.dot(e3).abs() < 1e-5, "e2·e3 != 0 for n {n:?}");
        }
    }

    /// `Projection::Schlegel` on R³ falls back to `Vec3::ZERO` per the enum's "unsupported
    /// variant returns zero" contract (R³ has no fourth axis for the central projection).
    #[test]
    fn schlegel_unsupported_on_r3_returns_zero() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let proj = Projection::<3>::Schlegel {
            cell_normal: Vec4::W,
            cell_offset: 0.5,
            viewpoint_distance: 0.75,
            basis: [Vec4::X, Vec4::Y, Vec4::Z],
        };
        let got = <EuclideanR3 as RasterizableSpace<3>>::project_point(p, &proj);
        assert_eq!(got, Vec3::ZERO);
    }

    // ---- Stereographic -------------------------------------------------

    /// Closed-form inverse of the stereographic map for the default pole `Vec4::W`: given an
    /// R³ image `q`, recover the S³ point. Derived by inverting
    /// `q = (x, y, z) / (1 - w)` together with `|p| = 1` (Wikipedia, *Stereographic
    /// projection*): with `s = |q|²`, `w = (s - 1) / (s + 1)` and `(x, y, z) = q * (1 - w)`.
    /// Used only by the round-trip test to pin that the truncated forward map is invertible.
    fn stereo_inverse_w_pole(q: Vec3) -> Vec4 {
        let s = q.length_squared();
        let w = (s - 1.0) / (s + 1.0);
        let xyz = q * (1.0 - w);
        Vec4::new(xyz.x, xyz.y, xyz.z, w)
    }

    /// General-pole inverse: lift `q` (expressed in the `perp_frame(pole)` basis) back into
    /// ambient R⁴, then invert the radial stereographic scaling against `|p| = 1`. For a unit
    /// `pole` and `q` the frame readout of `stereographic_to_r3(p, pole)`, this returns `p`.
    fn stereo_inverse_general(q: Vec3, pole: Vec4) -> Vec4 {
        // Re-embed the in-3-flat coordinates as an ambient pole-perpendicular vector.
        let (e1, e2, e3) = perp_frame(pole);
        let perp = q.x * e1 + q.y * e2 + q.z * e3;
        // Forward: perp = (p - dot*pole) / (1 - dot) with |perp_ambient(p)| =
        // sqrt(1 - dot²)/(1 - dot) = sqrt((1+dot)/(1-dot)). Solve for dot from |q|=|perp|.
        let s = perp.length_squared();
        let dot = (s - 1.0) / (s + 1.0);
        // p = dot*pole + (1 - dot)*perp  (undo the radial scaling, restore the pole part).
        dot * pole + (1.0 - dot) * perp
    }

    /// Default pole `Vec4::W`: the closed-form fast path equals the canonical stereographic
    /// formula `(x, y, z) / (1 - w)` bitwise, with zero frame math (Wikipedia, *Stereographic
    /// projection*). Tested on `stereographic_to_r3` directly with exactly-unit inputs, since
    /// `EuclideanR4::project_point` re-normalizes its input (its precondition guard) and a
    /// second f32 normalize of an already-unit vector is not bit-idempotent; the closed-form
    /// equality is a property of the map, not of the normalize guard.
    #[test]
    fn stereographic_default_pole_is_drop_w_of_scaled() {
        for p in [
            Vec4::new(0.5, 0.5, 0.5, 0.5),   // unit by construction
            Vec4::new(-0.5, 0.5, 0.5, -0.5), // unit by construction
            Vec4::new(-0.6, 0.0, 0.8, 0.0),  // unit, w = 0
        ] {
            let got = stereographic_to_r3(p, Vec4::W);
            let want = Vec3::new(p.x, p.y, p.z) / (1.0 - p.w);
            assert_eq!(
                got, want,
                "fast path must match canonical formula for {p:?}"
            );
        }
        // The fast path is also bit-identical to the general frame-readout path for the W pole
        // (the property the closed form trades zero Gram-Schmidt for). Confirm by forcing the
        // general path on a pole numerically equal to W up to f32 (it is exactly W here).
        let general = {
            let p = Vec4::new(0.5, 0.5, 0.5, 0.5);
            let dot = p.dot(Vec4::W).clamp(-1.0, 1.0);
            let denom = (1.0 - dot).max(STEREOGRAPHIC_POLE_EPSILON);
            let perp = p - dot * Vec4::W;
            let scaled = perp / denom;
            let (e1, e2, e3) = perp_frame(Vec4::W);
            Vec3::new(scaled.dot(e1), scaled.dot(e2), scaled.dot(e3))
        };
        assert_eq!(
            general,
            stereographic_to_r3(Vec4::new(0.5, 0.5, 0.5, 0.5), Vec4::W)
        );
    }

    /// `stereo_inverse(stereo(p)) ~= p` for unit `p` clear of the pole, for the
    /// default pole and an off-axis pole. Written against the TRUNCATED image
    /// (the inverse re-embeds the
    /// in-3-flat readout), so the `n`-perpendicular truncation is pinned as load-bearing: a
    /// leaked pole-component would break this round-trip.
    #[test]
    fn stereographic_inverts_off_pole() {
        // Default pole.
        let proj_w = Projection::Stereographic { pole: Vec4::W };
        for p in [
            Vec4::new(0.2, 0.1, -0.3, 0.4).normalize(),
            Vec4::new(-0.5, 0.5, 0.5, -0.5),
            Vec4::new(0.7, -0.2, 0.1, 0.1).normalize(),
        ] {
            let img = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &proj_w);
            let back = stereo_inverse_w_pole(img);
            assert_relative_eq!(back.x, p.x, epsilon = 1e-5);
            assert_relative_eq!(back.y, p.y, epsilon = 1e-5);
            assert_relative_eq!(back.z, p.z, epsilon = 1e-5);
            assert_relative_eq!(back.w, p.w, epsilon = 1e-5);
        }
        // Off-axis pole, well clear of the inputs.
        let pole = Vec4::new(0.1, -0.2, 0.3, 0.9).normalize();
        let proj_n = Projection::Stereographic { pole };
        for p in [
            Vec4::new(0.6, 0.5, -0.2, 0.0).normalize(),
            Vec4::new(-0.3, 0.4, 0.5, -0.2).normalize(),
        ] {
            let img = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &proj_n);
            let back = stereo_inverse_general(img, pole);
            assert_relative_eq!(back.x, p.x, epsilon = 1e-5);
            assert_relative_eq!(back.y, p.y, epsilon = 1e-5);
            assert_relative_eq!(back.z, p.z, epsilon = 1e-5);
            assert_relative_eq!(back.w, p.w, epsilon = 1e-5);
        }
    }

    /// The image lies in the pole-perpendicular 3-flat: re-embedding the R³ readout against
    /// `perp_frame(pole)` gives an ambient vector with zero pole-component. This is the test
    /// that catches the naive `p / (1 - dot)` leak (the refuted form carries a nonzero
    /// pole-component, measured round-trip error 0.176 in the plan's degeneracy analysis).
    #[test]
    fn stereographic_image_in_n_perp_hyperplane() {
        for pole in [
            Vec4::W,
            Vec4::new(0.1, -0.2, 0.3, 0.9).normalize(),
            Vec4::new(0.5, 0.5, 0.5, 0.5),
        ] {
            let (e1, e2, e3) = perp_frame(pole);
            for p in [
                Vec4::new(0.6, 0.5, -0.2, 0.0).normalize(),
                Vec4::new(-0.3, 0.4, 0.5, -0.2).normalize(),
                Vec4::new(0.2, 0.1, -0.3, 0.4).normalize(),
            ] {
                let proj = Projection::Stereographic { pole };
                let img = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &proj);
                let ambient = img.x * e1 + img.y * e2 + img.z * e3;
                assert!(
                    ambient.dot(pole).abs() < 1e-5,
                    "image must lie in pole-perp 3-flat: pole {pole:?} p {p:?} leak {}",
                    ambient.dot(pole)
                );
            }
        }
    }

    /// A vertex exactly at the pole (`p == pole`) and a tangential near-pole point both return
    /// finite R³, never NaN/Inf. The denominator `1 - dot(p, pole)` is `0` at the pole; the
    /// `STEREOGRAPHIC_POLE_EPSILON` floor keeps the buffer finite (mirrors the Schlegel and
    /// Perspective4D denominator clamps).
    #[test]
    fn stereographic_pole_denominator_clamped_finite() {
        for pole in [Vec4::W, Vec4::new(0.5, 0.5, 0.5, 0.5)] {
            let proj = Projection::Stereographic { pole };
            // Exactly at the pole.
            let at_pole = <EuclideanR4 as RasterizableSpace<4>>::project_point(pole, &proj);
            for c in [at_pole.x, at_pole.y, at_pole.z] {
                assert!(
                    c.is_finite(),
                    "pole input must clamp finite, got {at_pole:?}"
                );
            }
            // Just off the pole, mostly tangential: dot(p, pole) ≈ 1 but < 1.
            let (e1, _, _) = perp_frame(pole);
            let near = (pole * 0.9999 + e1 * 0.01).normalize();
            let near_img = <EuclideanR4 as RasterizableSpace<4>>::project_point(near, &proj);
            for c in [near_img.x, near_img.y, near_img.z] {
                assert!(
                    c.is_finite(),
                    "near-pole input must stay finite, got {near_img:?}"
                );
            }
        }
    }

    /// The antipode `-pole` maps to the origin with denominator exactly `2`: `dot(-pole, pole)
    /// = -1`, the pole-perpendicular component is zero, so the image is `Vec3::ZERO`. This is
    /// the safe far point that distinguishes the antipode from the singular pole.
    #[test]
    fn stereographic_antipode_maps_to_origin() {
        for pole in [Vec4::W, Vec4::new(0.1, -0.2, 0.3, 0.9).normalize()] {
            let proj = Projection::Stereographic { pole };
            let got = <EuclideanR4 as RasterizableSpace<4>>::project_point(-pole, &proj);
            assert_relative_eq!(got.x, 0.0, epsilon = 1e-6);
            assert_relative_eq!(got.y, 0.0, epsilon = 1e-6);
            assert_relative_eq!(got.z, 0.0, epsilon = 1e-6);
        }
    }

    /// A pole at 45° between two axes (so two Gram-Schmidt seeds are near-equidistant) yields
    /// the same Vec3 across repeated calls and matches a golden value. Tier-0 bit-repro guard:
    /// the frame must not flip gauge under the tie-break, or the projected wireframe would
    /// silently rotate between machines or runs.
    #[test]
    fn stereographic_frame_is_deterministic_under_tie() {
        // Pole equidistant from the z and w axes; perp_frame must tie-break deterministically
        // toward the lowest index.
        let pole = Vec4::new(0.0, 0.0, 1.0, 1.0).normalize();
        let proj = Projection::Stereographic { pole };
        let p = Vec4::new(0.5, 0.5, -0.5, 0.5); // unit
        let first = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &proj);
        for _ in 0..16 {
            let again = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &proj);
            assert_eq!(first, again, "frame must be byte-stable across calls");
        }
        // Golden value: locks the chosen gauge so a future tie-break change is caught. The
        // frame for this pole drops the w axis (largest |component| ties z and w; the max-pick
        // resolves to the lowest matching index, w at index 3 only if it is the unique max --
        // here z and w tie at the max, and the `==` ladder selects z, index 2). Recorded from
        // the current deterministic construction.
        assert_relative_eq!(first.x, GOLDEN_TIE_FRAME.x, epsilon = 1e-6);
        assert_relative_eq!(first.y, GOLDEN_TIE_FRAME.y, epsilon = 1e-6);
        assert_relative_eq!(first.z, GOLDEN_TIE_FRAME.z, epsilon = 1e-6);
    }

    /// `perp_frame(pole)` is orthonormal for poles swept toward each axis, including exactly an
    /// axis: the Gram matrix is approximately the identity and every entry is finite. The
    /// stereographic readout relies on this; a degenerate frame would skew or NaN the image.
    #[test]
    fn stereographic_frame_orthonormal_for_every_pole() {
        let mut poles = vec![Vec4::X, Vec4::Y, Vec4::Z, Vec4::W];
        // Sweep toward each axis from the all-equal direction.
        for axis in [Vec4::X, Vec4::Y, Vec4::Z, Vec4::W] {
            for k in 1..6 {
                let t = k as f32 / 6.0;
                poles.push((Vec4::splat(0.5) * (1.0 - t) + axis * t).normalize());
            }
        }
        for pole in poles {
            let (e1, e2, e3) = perp_frame(pole);
            for e in [e1, e2, e3] {
                assert!(e.is_finite(), "frame must be finite for pole {pole:?}");
            }
            // Gram matrix approx I.
            assert_relative_eq!(e1.dot(e1), 1.0, epsilon = 1e-5);
            assert_relative_eq!(e2.dot(e2), 1.0, epsilon = 1e-5);
            assert_relative_eq!(e3.dot(e3), 1.0, epsilon = 1e-5);
            assert!(e1.dot(e2).abs() < 1e-5, "e1·e2 for pole {pole:?}");
            assert!(e1.dot(e3).abs() < 1e-5, "e1·e3 for pole {pole:?}");
            assert!(e2.dot(e3).abs() < 1e-5, "e2·e3 for pole {pole:?}");
            // Each basis vector lies in the pole-perp 3-flat.
            assert!(e1.dot(pole).abs() < 1e-5, "e1 ⟂ pole {pole:?}");
            assert!(e2.dot(pole).abs() < 1e-5, "e2 ⟂ pole {pole:?}");
            assert!(e3.dot(pole).abs() < 1e-5, "e3 ⟂ pole {pole:?}");
        }
    }

    /// Stereographic projection is conformal (angle-preserving): two edges meeting at a shared
    /// non-pole vertex keep the angle between them after projection, within tolerance
    /// (Wikipedia, *Stereographic projection*: the map is conformal). Uses the geodesic tangent
    /// directions on S³ as the "edges" and compares to the angle between their
    /// projected images.
    #[test]
    fn stereographic_is_conformal() {
        let s = SphericalS3Embedded;
        let pole = Vec4::W;
        let proj = Projection::Stereographic { pole };
        // Shared vertex well off the pole; two neighbors define two great-circle edges.
        let v = Vec4::new(0.3, -0.1, 0.2, -0.5).normalize();
        let a = Vec4::new(0.5, 0.4, -0.1, -0.3).normalize();
        let b = Vec4::new(-0.2, 0.3, 0.6, -0.4).normalize();
        // Angle between the two geodesic tangents at `v` (the intrinsic edge angle on S³).
        let ta = s.log(v, a);
        let tb = s.log(v, b);
        let intrinsic = (ta.dot(tb) / (ta.length() * tb.length()))
            .clamp(-1.0, 1.0)
            .acos();
        // Angle between the projected edges in R3. Step a small distance along
        // each geodesic so the secant approximates the projected tangent.
        let step = 1e-3;
        let pv = <EuclideanR4 as RasterizableSpace<4>>::project_point(v, &proj);
        let pa = <EuclideanR4 as RasterizableSpace<4>>::project_point(
            s.exp(v, ta.normalize() * step),
            &proj,
        );
        let pb = <EuclideanR4 as RasterizableSpace<4>>::project_point(
            s.exp(v, tb.normalize() * step),
            &proj,
        );
        let da = pa - pv;
        let db = pb - pv;
        let projected = (da.dot(db) / (da.length() * db.length()))
            .clamp(-1.0, 1.0)
            .acos();
        assert_relative_eq!(projected, intrinsic, epsilon = 1e-2);
    }

    /// `Projection::Stereographic` on R3 falls back to `Vec3::ZERO` per the
    /// enum's "unsupported variant returns zero" contract.
    #[test]
    fn stereographic_unsupported_on_r3_returns_zero() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let proj = Projection::Stereographic { pole: Vec4::W };
        let got = <EuclideanR3 as RasterizableSpace<3>>::project_point(p, &proj);
        assert_eq!(got, Vec3::ZERO);
    }

    /// `tessellate_segment` on R⁴ is plain lerp (since `EuclideanR4` is flat). Each sample
    /// gives the expected linear interpolation across all four components.
    #[test]
    fn r4_tessellate_lerps_all_components() {
        let p0 = Vec4::new(0.0, 0.0, 0.0, 0.0);
        let p1 = Vec4::new(4.0, 8.0, 12.0, 16.0);
        let mut out = Vec::new();
        <EuclideanR4 as RasterizableSpace<4>>::tessellate_segment(p0, p1, 2, &mut out);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], p0);
        assert_eq!(out[1], Vec4::new(2.0, 4.0, 6.0, 8.0));
        assert_eq!(out[2], p1);
    }
}
