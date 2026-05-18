//! [`RasterizableSpace<N>`] trait + [`Projection<N>`] enum + flat-Euclidean impls.
//!
//! Pairs with the `Visualizable<N>` trait in `rye-shape`. `Visualizable` answers "what mesh
//! data does this shape produce in R^N?"; `RasterizableSpace` answers "given that mesh data in
//! space `S`, how do we get screen-ready R³ vertices?". The rasterizer pipeline in `rye-render`
//! composes them.
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
///
/// Future variants under consideration: R⁴-specific `Schlegel`, `Stereographic`, `Hyperslice`.
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
    Orthographic {
        drop_axis: usize,
    },
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

    /// `point_to_array` is the inverse of `array_to_point` on `EuclideanR3` for any input.
    #[test]
    fn r3_array_round_trip() {
        let p = Vec3::new(1.0, -2.5, 0.7);
        let arr = <EuclideanR3 as RasterizableSpace<3>>::point_to_array(p);
        let back = <EuclideanR3 as RasterizableSpace<3>>::array_to_point(arr);
        assert_eq!(p, back);
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
