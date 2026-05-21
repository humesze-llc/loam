//! [`SectionableSpace<N>`] trait + axis-aligned hyperplane types + flat-Euclidean impls.
//!
//! A cross-section is the geometry "an inhabitant at the slice would actually see": the
//! `N`-dimensional space's intersection with a hyperplane, expressed in the `(N-1)`-
//! dimensional space of the slice. This is the load-bearing primitive for the 4D throughline
//! demos -- slicing R⁴ into R³ exposes a dimension the viewer can't directly perceive.
//!
//! ## Why a Space-level trait, not a polytope helper
//!
//! Cross-section is the same conceptual operation regardless of what's being sliced. The
//! polytope cross-section we ship first is one consumer; future curved-space and
//! BlendedSpace cross-sections share the algorithm above [`SectionableSpace::edge_section`]
//! (cap-polygon assembly, plane fit, fan triangulation are all space-agnostic). The trait
//! shape mirrors [`crate::rasterizable::RasterizableSpace`]: the flat-vs-curved distinction
//! lives in a single method.
//!
//! ## Hyperplane representation
//!
//! For flat R⁴ at v1 we use an axis-aligned `w`-slice ([`WPlane`]): the simplest case, which
//! is also what the rotate_polytopes demo sweeps. Arbitrary-normal hyperplanes for R⁴ and
//! geodesic hyperplanes for curved spaces are additive extensions; the trait has no
//! commitment to a specific hyperplane geometry beyond `type Hyperplane`.
//!
//! ## Numerical care
//!
//! [`SectionableSpace::edge_section`] uses an FMA-friendly lerp ordering and rejects edges
//! whose endpoints are too close to the slice plane to give a numerically stable intersection
//! ([`EDGE_PARALLEL_EPSILON`]). Callers handling the surrounding cell-assembly algorithm
//! perturb the slice value when any polytope vertex sits within [`SLICE_PERTURBATION_EPSILON`]
//! of it; that single perturbation kills three degeneracies at once (vertex on slice, edge in
//! slice plane, slice grazes a face).

use glam::{Vec3, Vec4};

use crate::space::Space;
use crate::EuclideanR4;

/// Smallest `|dw|` between an edge's endpoints for which [`SectionableSpace::edge_section`]
/// will return an intersection. Edges with smaller dw are treated as parallel to the slice
/// hyperplane (so the intersection is either the whole edge or empty, both handled by the
/// cell-assembly caller rather than by returning a single point).
///
/// Chosen to be larger than the f32 roundoff on a unit-circumradius `dw` computation
/// (~1e-7) by a 10x margin. See `SLICE_PERTURBATION_EPSILON` for the dual epsilon used by
/// the caller's slice perturbation.
pub const EDGE_PARALLEL_EPSILON: f32 = 1e-6;

/// Recommended `epsilon` for slice perturbation: when any polytope vertex's slice-axis
/// coordinate sits within this value of the slice, the caller (cell-assembly algorithm)
/// shifts the slice by this amount to avoid degenerate cases. A single perturbation kills
/// three degeneracies at once: vertex on slice, edge in slice plane, slice grazes a face.
pub const SLICE_PERTURBATION_EPSILON: f32 = 1e-5;

/// Axis-aligned w-slice hyperplane for R⁴ (the standard 4D demo case). Encodes "the
/// 3-flat where `w = w_slice`." A newtype rather than a bare `f32` so future hyperplane
/// variants (arbitrary normal, geodesic) extend the same trait without breaking the
/// `type Hyperplane` boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WPlane {
    pub w_slice: f32,
}

impl WPlane {
    pub const fn new(w_slice: f32) -> Self {
        Self { w_slice }
    }
}

/// A Space whose inhabitants live in `N` dimensions and which supports taking a
/// cross-section: an `(N - 1)`-dimensional Space containing the intersection geometry of
/// the parent with a hyperplane.
///
/// `N` is the const-generic ambient dimension, matching the
/// [`crate::rasterizable::RasterizableSpace`] convention. The associated `SectionSpace` is
/// the Space the section *lives in* (its
/// `Point` type is the cap-vertex coordinate type used by the cell-assembly algorithm).
/// The associated `Hyperplane` lets curved-space impls carry the extra structure
/// (geodesic basis) that a flat-space `(point, normal)` pair doesn't need.
pub trait SectionableSpace<const N: usize>: Space {
    /// The `(N - 1)`-dimensional Space the section lives in. For flat R⁴, this is
    /// [`crate::EuclideanR3`]. For S³, it would be [`crate::SphericalS3`]-restricted-to-a-
    /// great-2-sphere, which is `SphericalS2` (a future addition).
    type SectionSpace: Space;

    /// Hyperplane identifier. Flat spaces use simple variants like [`WPlane`] (axis-aligned)
    /// or a future `(point, normal)` pair; curved spaces will use geodesic-hyperplane
    /// descriptors (point on the parent + tangent basis spanning the slice).
    type Hyperplane;

    /// Intersect a geodesic edge `(p0, p1)` with `slice`. Returns the lerp parameter
    /// `t in [0, 1]` and the intersection point expressed in [`Self::SectionSpace`], or
    /// `None` if the edge does not cross the slice or is parallel to it within
    /// [`EDGE_PARALLEL_EPSILON`].
    ///
    /// For flat spaces this is a linear solve. For curved spaces it walks the geodesic from
    /// `p0` to `p1` and bisects on side-of-slice; closed-form solves exist for the standard
    /// chart parameterizations of S³ and H³.
    fn edge_section(
        slice: &Self::Hyperplane,
        p0: Self::Point,
        p1: Self::Point,
    ) -> Option<(f32, <Self::SectionSpace as Space>::Point)>;
}

impl SectionableSpace<4> for EuclideanR4 {
    type SectionSpace = crate::EuclideanR3;
    type Hyperplane = WPlane;

    fn edge_section(slice: &WPlane, p0: Vec4, p1: Vec4) -> Option<(f32, Vec3)> {
        let dw = p1.w - p0.w;
        // Edge parallel to the w-slice (within roundoff): the intersection is either the
        // whole edge or empty. Both are degenerate from this method's "single point" point
        // of view; the cell-assembly caller handles them by perturbing the slice (per
        // SLICE_PERTURBATION_EPSILON) before the per-edge loop.
        if dw.abs() < EDGE_PARALLEL_EPSILON {
            return None;
        }
        let t = (slice.w_slice - p0.w) / dw;
        // Strictly inside the edge. `<` rather than `<=` at the boundaries prevents
        // double-counting at shared cell vertices when the slice grazes a vertex's w.
        if !(0.0..=1.0).contains(&t) {
            return None;
        }
        // FMA-friendly lerp ordering: `p0 + t * (p1 - p0)` preserves precision when t is
        // near 0 or 1, vs. the algebraically equivalent `(1 - t) * p0 + t * p1`.
        let p = p0 + t * (p1 - p0);
        Some((t, Vec3::new(p.x, p.y, p.z)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Edge straddling w = 0 yields the lerp midpoint when endpoints have equal-magnitude
    /// opposite w. Sanity check on both the t value and the dropped-w R³ coordinate.
    #[test]
    fn r4_edge_section_midpoint() {
        let slice = WPlane::new(0.0);
        let p0 = Vec4::new(1.0, 2.0, 3.0, -1.0);
        let p1 = Vec4::new(5.0, 6.0, 7.0, 1.0);
        let (t, p3) = <EuclideanR4 as SectionableSpace<4>>::edge_section(&slice, p0, p1).unwrap();
        assert!((t - 0.5).abs() < 1e-6);
        assert_eq!(p3, Vec3::new(3.0, 4.0, 5.0));
    }

    /// Edge with both endpoints on the same side of the slice returns `None`.
    #[test]
    fn r4_edge_section_no_crossing_returns_none() {
        let slice = WPlane::new(0.0);
        let p0 = Vec4::new(0.0, 0.0, 0.0, 0.5);
        let p1 = Vec4::new(0.0, 0.0, 0.0, 1.5);
        assert!(<EuclideanR4 as SectionableSpace<4>>::edge_section(&slice, p0, p1).is_none());
    }

    /// Edge parallel to the slice (both endpoints share w within the parallel epsilon)
    /// returns `None`; the cell-assembly caller handles the whole-edge-in-plane case by
    /// perturbing the slice.
    #[test]
    fn r4_edge_section_parallel_edge_returns_none() {
        let slice = WPlane::new(0.0);
        let p0 = Vec4::new(0.0, 0.0, 0.0, 0.0);
        let p1 = Vec4::new(1.0, 1.0, 1.0, 1e-7);
        assert!(<EuclideanR4 as SectionableSpace<4>>::edge_section(&slice, p0, p1).is_none());
    }

    /// Edge whose first endpoint sits exactly on the slice returns `t = 0` and the
    /// endpoint's R³ coordinates. The cell-assembly caller is expected to perturb the
    /// slice to avoid this boundary, but `edge_section` itself doesn't reject it.
    #[test]
    fn r4_edge_section_endpoint_on_slice_returns_t_zero() {
        let slice = WPlane::new(0.0);
        let p0 = Vec4::new(2.0, 2.0, 2.0, 0.0);
        let p1 = Vec4::new(5.0, 5.0, 5.0, 1.0);
        let (t, p3) = <EuclideanR4 as SectionableSpace<4>>::edge_section(&slice, p0, p1).unwrap();
        assert!(t.abs() < 1e-6);
        assert_eq!(p3, Vec3::new(2.0, 2.0, 2.0));
    }

    /// 5-cell midpoint slice fixture: one apex edge crosses the w=0 hyperplane at lerp
    /// parameter t = 0.8 (solves `1 + t * -1.25 = 0` for the standard apex w=1, base w=-0.25
    /// pentatope generators) and the R³ intersection is at `0.8 * v_i` where `v_i` is the
    /// base vertex's R³ coords. Matches Coxeter's classical result for the pentatope's
    /// midpoint cross-section being a regular tetrahedron.
    #[test]
    fn r4_edge_section_matches_pentatope_midpoint_worked_example() {
        let slice = WPlane::new(0.0);
        // Apex v0 = (0, 0, 0, 1); base vertex v1 = (t, t, t, -0.25) with t = sqrt(15)/(4*sqrt(3)).
        let t_base = (15.0f32).sqrt() / (4.0 * (3.0f32).sqrt());
        let apex = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let v1 = Vec4::new(t_base, t_base, t_base, -0.25);
        let (t, p3) = <EuclideanR4 as SectionableSpace<4>>::edge_section(&slice, apex, v1)
            .expect("apex edge straddles w = 0");
        // Lerp parameter solves 1 + t*(-1.25) = 0 -> t = 0.8.
        assert!((t - 0.8).abs() < 1e-6);
        // Intersection point P1 = (0.8 * t_base, 0.8 * t_base, 0.8 * t_base).
        let expected = 0.8 * t_base;
        assert!((p3.x - expected).abs() < 1e-5);
        assert!((p3.y - expected).abs() < 1e-5);
        assert!((p3.z - expected).abs() < 1e-5);
    }
}
