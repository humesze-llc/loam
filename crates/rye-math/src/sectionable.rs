//! [`SectionableSpace<N>`] trait + axis-aligned hyperplane types + flat-Euclidean
//! impls.
//!
//! A cross-section is what an inhabitant at the slice would see: the `N`-space's
//! intersection with a hyperplane, expressed in the `(N-1)`-space of the slice.
//! It is a Space-level trait, not a polytope helper, because the assembly above
//! [`SectionableSpace::edge_section`] (cap-polygon, plane fit, fan triangulation)
//! is space-agnostic; only the per-edge solve is flat-vs-curved.
//!
//! [`SectionableSpace::edge_section`] uses an FMA-friendly lerp and rejects edges
//! within [`EDGE_PARALLEL_EPSILON`] of parallel to the slice. The cell-assembly
//! caller perturbs the slice by [`SLICE_PERTURBATION_EPSILON`] when any vertex
//! sits that close; one perturbation kills three degeneracies (vertex on slice,
//! edge in slice plane, slice grazes a face).

use glam::{Vec3, Vec4};

use crate::space::Space;
use crate::EuclideanR4;

/// Smallest `|dw|` for which [`SectionableSpace::edge_section`] returns an
/// intersection; below it the edge is treated as parallel to the slice (whole
/// edge or empty, handled by the caller). 10x above f32 roundoff (~1e-7) on a
/// unit-circumradius `dw`.
pub const EDGE_PARALLEL_EPSILON: f32 = 1e-6;

/// Slice-perturbation epsilon: when a vertex's slice-axis coord sits this close
/// to the slice, the cell-assembly caller shifts the slice by this amount. One
/// shift kills three degeneracies: vertex on slice, edge in slice plane, slice
/// grazes a face.
pub const SLICE_PERTURBATION_EPSILON: f32 = 1e-5;

/// Axis-aligned w-slice hyperplane for R⁴: the 3-flat where `w = w_slice`. A
/// newtype, not a bare `f32`, so future variants (arbitrary normal, geodesic)
/// extend the same `type Hyperplane`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WPlane {
    pub w_slice: f32,
}

impl WPlane {
    pub const fn new(w_slice: f32) -> Self {
        Self { w_slice }
    }
}

/// A Space of ambient dimension `N` that supports cross-section: an `(N - 1)`-Space
/// holding the intersection of the parent with a hyperplane. `N` matches the
/// [`crate::rasterizable::RasterizableSpace`] convention.
pub trait SectionableSpace<const N: usize>: Space {
    /// The `(N - 1)`-Space the section lives in. Flat R⁴ -> [`crate::EuclideanR3`];
    /// S³ -> a great-2-sphere `SphericalS2` (future).
    type SectionSpace: Space;

    /// Hyperplane identifier. Flat spaces use [`WPlane`] (or a future
    /// `(point, normal)`); curved spaces carry a geodesic basis.
    type Hyperplane;

    /// Intersect geodesic edge `(p0, p1)` with `slice`. Returns the lerp `t in
    /// [0, 1]` and the point in [`Self::SectionSpace`], or `None` if the edge
    /// misses the slice or is parallel within [`EDGE_PARALLEL_EPSILON`].
    ///
    /// Flat spaces solve linearly; curved spaces bisect along the geodesic
    /// (closed forms exist for the standard S³/H³ charts).
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
        // Parallel within roundoff: whole-edge or empty, both left to the
        // caller's slice perturbation.
        if dw.abs() < EDGE_PARALLEL_EPSILON {
            return None;
        }
        let t = (slice.w_slice - p0.w) / dw;
        // Closed `[0, 1]`: an endpoint on the slice counts as an intersection.
        if !(0.0..=1.0).contains(&t) {
            return None;
        }
        // FMA-friendly lerp; preserves precision near t = 0 or 1 vs.
        // `(1 - t)·p0 + t·p1`.
        let p = p0 + t * (p1 - p0);
        Some((t, Vec3::new(p.x, p.y, p.z)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Edge straddling w = 0 with equal-magnitude opposite w yields the midpoint.
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

    /// Edge parallel to the slice (shared w within the epsilon) returns `None`.
    #[test]
    fn r4_edge_section_parallel_edge_returns_none() {
        let slice = WPlane::new(0.0);
        let p0 = Vec4::new(0.0, 0.0, 0.0, 0.0);
        let p1 = Vec4::new(1.0, 1.0, 1.0, 1e-7);
        assert!(<EuclideanR4 as SectionableSpace<4>>::edge_section(&slice, p0, p1).is_none());
    }

    /// First endpoint on the slice returns `t = 0`; `edge_section` does not reject
    /// the boundary even though the caller normally perturbs it away.
    #[test]
    fn r4_edge_section_endpoint_on_slice_returns_t_zero() {
        let slice = WPlane::new(0.0);
        let p0 = Vec4::new(2.0, 2.0, 2.0, 0.0);
        let p1 = Vec4::new(5.0, 5.0, 5.0, 1.0);
        let (t, p3) = <EuclideanR4 as SectionableSpace<4>>::edge_section(&slice, p0, p1).unwrap();
        assert!(t.abs() < 1e-6);
        assert_eq!(p3, Vec3::new(2.0, 2.0, 2.0));
    }

    /// Pentatope midpoint slice: the apex edge crosses w = 0 at t = 0.8 and the R³
    /// point lands at `0.8·v_i`. Coxeter: the pentatope's midpoint section is a
    /// regular tetrahedron.
    #[test]
    fn r4_edge_section_matches_pentatope_midpoint_worked_example() {
        let slice = WPlane::new(0.0);
        let t_base = (15.0f32).sqrt() / (4.0 * (3.0f32).sqrt());
        let apex = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let v1 = Vec4::new(t_base, t_base, t_base, -0.25);
        let (t, p3) = <EuclideanR4 as SectionableSpace<4>>::edge_section(&slice, apex, v1)
            .expect("apex edge straddles w = 0");
        // 1 + t·(-1.25) = 0 -> t = 0.8.
        assert!((t - 0.8).abs() < 1e-6);
        let expected = 0.8 * t_base;
        assert!((p3.x - expected).abs() < 1e-5);
        assert!((p3.y - expected).abs() < 1e-5);
        assert!((p3.z - expected).abs() < 1e-5);
    }
}
