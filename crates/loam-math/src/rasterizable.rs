//! [`RasterizableSpace<N>`] trait + [`Projection<N>`] enum + flat-Euclidean impls.
//!
//! Pairs with the `Visualizable<N>` trait in `loam-shape`: that produces mesh
//! data in R^N, this maps it to screen-ready R³ vertices. The rasterizer
//! pipeline in `loam-render` composes them.
//!
//! [`RasterizableSpace<N>`] is generic over [`Space`] rather than the array
//! type because the [`Space`] impls use `glam::Vec3` / `Vec4` as `Point`;
//! [`RasterizableSpace::point_to_array`] / [`RasterizableSpace::array_to_point`]
//! bridge to `[f32; N]` mesh storage. The flat / curved distinction lives only
//! in [`RasterizableSpace::tessellate_segment`] (lerp vs geodesic), so the
//! pipeline is identical for both and curved-space impls drop in additively.

use glam::{Vec3, Vec4};

use crate::space::Space;
use crate::{EuclideanR3, EuclideanR4};

/// Sign-preserving floor for central-projection denominators (Perspective4D
/// scale, Schlegel ray parameter): keeps a vertex on the viewer's 3-flat from
/// dividing by zero. The denominator is a dot of order-1 unit vectors, so `1e-4`
/// sits above f32 roundoff yet below any real ray parameter; it engages only at
/// the singularity, where the buffer stays finite but the picture is meaningless.
const PROJECTION_DENOM_EPSILON: f32 = 1e-4;

/// Floor for the stereographic denominator `1 - dot(p, n)`. The pole is a
/// reachable input (the 16-cell has `±e_i` vertices; a cell-center pole such as
/// `(1,1,1,1)/2` is a real tesseract vertex direction), so a vertex at the pole
/// gives `dot = 1` and a bare divide NaNs the buffer. Same order-1 reasoning as
/// `PROJECTION_DENOM_EPSILON`.
///
/// `pub` because a renderer drawing the image as a finite polyline must clip the
/// near-pole blow-up, and the clip radius is interdependent with this floor: a
/// clamped sample maps to magnitude at most `sqrt(2 / eps)`, so a clip radius
/// below that ceiling catches every saturated near-pole sample. The clip lives
/// in the demo layer so this map stays the pure conformal projection.
pub const STEREOGRAPHIC_POLE_EPSILON: f32 = 1e-4;

/// Orthonormal basis `(e1, e2, e3)` of the 3-flat perpendicular to unit `n`, for
/// reading an in-3-flat point out as R³ (the Schlegel screen coordinates).
/// Deterministic in `n`: drop the world axis most aligned with `n`, then
/// Gram-Schmidt the surviving three in x, y, z, w order (do Carmo, *Differential
/// Geometry of Curves and Surfaces*, §1.4). Dropping the most-aligned axis keeps
/// every residual well clear of zero.
///
/// Using this frame instead of a naive drop of the `n`-aligned coordinate is
/// what keeps the diagram faithful for a non-axis-aligned cell normal: an oblique
/// drop-w flattens the boundary cell and breaks nesting.
///
/// The frame depends on `n` alone, so it is identical across a projection's
/// vertices, yet rebuilt per [`RasterizableSpace::project_point`] call. The
/// rebuild is a few allocation-free f32 ops yielding a byte-identical frame, so
/// it is left un-hoisted until a measurement shows the redundancy is real (see
/// [`Projection::Schlegel`]).
fn perp_frame(n: Vec4) -> (Vec4, Vec4, Vec4) {
    // Dropping the most-aligned axis keeps the remaining three independent in the
    // 3-flat; ties resolve toward the earliest axis for determinism.
    let ax = n.x.abs();
    let ay = n.y.abs();
    let az = n.z.abs();
    let aw = n.w.abs();
    let drop = ax.max(ay).max(az).max(aw);
    let mut seeds = [Vec4::X, Vec4::Y, Vec4::Z, Vec4::W];
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

/// Stereographic map of `p` on S³ to R³ from unit `pole` (Wikipedia,
/// *Stereographic projection*). Shared by the `EuclideanR4` and
/// `SphericalS3Embedded` impls so the map and its clamp discipline have one
/// definition.
///
/// `image = (p - dot(p, pole)*pole) / (1 - dot(p, pole))`, read out in the
/// `perp_frame` of the `pole`-perpendicular 3-flat. The numerator is truncated
/// before the divide so the image lies in that 3-flat; the bare `p / denom` form
/// leaks a `pole`-component. `dot` is clamped to `[-1, 1]` and the denominator
/// floored at `STEREOGRAPHIC_POLE_EPSILON` (see those items).
///
/// Pole `Vec4::W` takes the closed-form fast path `(p.x, p.y, p.z) / (1 - p.w)`,
/// bit-identical to the general path since `perp_frame(W)` is `(X, Y, Z)`.
pub(crate) fn stereographic_to_r3(p: Vec4, pole: Vec4) -> Vec3 {
    let dot = p.dot(pole).clamp(-1.0, 1.0);
    let denom = (1.0 - dot).max(STEREOGRAPHIC_POLE_EPSILON);
    if pole == Vec4::W {
        return Vec3::new(p.x, p.y, p.z) / denom;
    }
    let perp = p - dot * pole;
    let scaled = perp / denom;
    let (e1, e2, e3) = perp_frame(pole);
    Vec3::new(scaled.dot(e1), scaled.dot(e2), scaled.dot(e3))
}

/// Projection from R^N to R³ for the rasterizer's screen-space transform.
///
/// Variants are dimension-generic in the type system but each makes sense only
/// for specific `N`; impls return `Vec3::ZERO` rather than panic on an
/// unsupported variant.
///
/// - [`Identity`](Self::Identity): first 3 components, zero-pad if `N < 3`,
///   truncate if `N > 3`. Bitwise identity at `N == 3`; drop-w at `N == 4`.
/// - [`Orthographic`](Self::Orthographic): drop one axis by index.
/// - [`Perspective4D`](Self::Perspective4D): R⁴ pinhole from `w = focal_distance`
///   looking in -w; the canonical "cube within a cube" tesseract view.
/// - [`Schlegel`](Self::Schlegel): R⁴ central projection from just outside a
///   chosen cell onto its 3-flat; that cell becomes the outer boundary and the
///   rest nest inside (Coxeter, *Regular Polytopes*, ch. 13).
/// - [`Stereographic`](Self::Stereographic): conformal S³ to R³ map from a chosen
///   pole (Wikipedia, *Stereographic projection*).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Projection<const N: usize> {
    /// First 3 components, zero-pad if `N < 3`, truncate if `N > 3`. Bitwise
    /// identity at `N == 3`. Default variant.
    #[default]
    Identity,

    /// Drop one axis by 0-based index; remaining `N - 1` components fill R³ in
    /// order, zero-padded if short. R⁴ `drop_axis: 3` gives `(x, y, z)`, the
    /// standard R⁴-into-R³ convention. Out-of-range (>= `N`) returns `Vec3::ZERO`.
    Orthographic { drop_axis: usize },

    /// 4D pinhole from a viewer at `(0, 0, 0, focal_distance)` looking in -w:
    /// `(x, y, z, w) -> (x, y, z) * focal_distance / (focal_distance - w)`. For a
    /// unit-circumradius polytope this is the "cube within a cube" tesseract view.
    ///
    /// **Precondition: `focal_distance > max(w)` over every vertex.** Otherwise
    /// the denominator hits zero (eye singularity) or flips sign; the impl clamps
    /// it rather than NaN-ing the upload, but the picture is then meaningless.
    ///
    /// Only meaningful for `N == 4`; other `N` return `Vec3::ZERO`.
    Perspective4D {
        /// Viewer position along the w-axis. Typical for unit polytopes: `2.0`.
        focal_distance: f32,
    },

    /// 4D Schlegel diagram: central projection from a viewpoint just outside a
    /// chosen cell onto that cell's bounding 3-flat (Coxeter, *Regular Polytopes*,
    /// 3rd ed., ch. 13). The chosen cell maps to the outer boundary and the rest
    /// nest inside.
    ///
    /// The cell lies in `{x : dot(cell_normal, x) = cell_offset}`; the eye sits at
    /// `E = viewpoint_distance * cell_normal`. Each vertex `p` projects along the
    /// ray `E + t*(p - E)` onto the hyperplane,
    /// `t = (cell_offset - dot(n, E)) / (dot(n, p) - dot(n, E))`.
    ///
    /// The result is read out in the supplied orthonormal `basis`, not by dropping
    /// w: an oblique w-drop is exact only for a w-aligned `n` and otherwise
    /// flattens the boundary cell. The basis is carried (not rebuilt from the live
    /// normal) so its in-flat gauge cannot snap across a tie during rotation.
    ///
    /// **Precondition: `cell_normal` is the *outward* unit normal and
    /// `viewpoint_distance > cell_offset`.** The inward normal breaks nesting. The
    /// denominator is sign-preservingly clamped so a vertex on the viewer's 3-flat
    /// stays finite. `cell_normal` / `cell_offset` are pre-resolved by the demo
    /// once per cell selection, never in the per-frame upload.
    ///
    /// Only meaningful for `N == 4`; other `N` return `Vec3::ZERO`.
    Schlegel {
        /// Outward unit normal of the chosen boundary cell's hyperplane.
        cell_normal: Vec4,
        /// Signed plane offset: the cell lies in
        /// `{x : dot(cell_normal, x) = cell_offset}`.
        cell_offset: f32,
        /// Eye distance along `cell_normal`; must exceed `cell_offset`.
        viewpoint_distance: f32,
        /// Orthonormal readout basis spanning the chosen cell's 3-flat.
        basis: [Vec4; 3],
    },

    /// Conformal projection of S³ onto R³ from a unit `pole` (Wikipedia,
    /// *Stereographic projection*):
    ///   `image = (p - dot(p, pole)*pole) / (1 - dot(p, pole))`,
    /// read out in an orthonormal basis of the `pole`-perpendicular 3-flat (see
    /// `perp_frame`). Pole `Vec4::W` collapses to `(p.x, p.y, p.z) / (1 - p.w)`.
    ///
    /// **The truncated `pole`-perpendicular numerator is load-bearing:**
    /// dividing the full 4-vector and dropping the radial part in the readout
    /// leaks a `pole`-component and corrupts the projection. Subtract
    /// `dot(p, pole)*pole` before dividing.
    ///
    /// **Precondition: `pole` and `p` are unit.** `dot` is clamped to `[-1, 1]`
    /// (matching [`crate::SphericalS3Embedded`]) and the denominator floored at
    /// `STEREOGRAPHIC_POLE_EPSILON` so a vertex at the reachable pole stays
    /// finite. The antipode `-pole` is the safe far point (`dot = -1`, maps to
    /// the origin).
    ///
    /// `EuclideanR4`'s impl normalizes onto S³ first (demo vertices are
    /// `body_size`-scaled); [`crate::SphericalS3Embedded`] computes the map
    /// directly. The frame is rebuilt per call like [`Schlegel`](Self::Schlegel).
    ///
    /// Only meaningful for `N == 4`; other `N` return `Vec3::ZERO`.
    Stereographic {
        /// Unit `Vec4` pole the projection casts away from. `Vec4::W` gives the
        /// closed-form `(x, y, z) / (1 - w)` map.
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

/// A flat or curved space that can drive the rasterizer pipeline: projection to
/// R³ plus segment tessellation.
///
/// `N` is the const-generic ambient dimension matching the `Visualizable<N>` mesh
/// data in `loam-shape`. Impls bridge the Space's native `Point` (typically `Vec3`
/// or `Vec4`) and `[f32; N]` mesh storage.
pub trait RasterizableSpace<const N: usize>: Space {
    /// Convert a space-native point to the mesh storage representation `[f32; N]`.
    fn point_to_array(p: Self::Point) -> [f32; N];

    /// Inverse of [`point_to_array`](Self::point_to_array).
    fn array_to_point(arr: [f32; N]) -> Self::Point;

    /// Project a point in this space to R³ under `projection`.
    fn project_point(point: Self::Point, projection: &Projection<N>) -> Vec3;

    /// Tessellate a segment into space-native points and append them to `out`.
    ///
    /// `samples` is the subdivision count, not the point count: `samples == 1`
    /// appends `[p0, p1]`; `samples == 4` appends 5 points. Flat spaces lerp;
    /// curved spaces sample along [`Space::exp`] / [`Space::log`].
    ///
    /// **Writer pattern:** the upload loop reuses one `Vec` across segments, so
    /// impls `push` and never `clear` (the caller owns the buffer).
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
            // Meaningful only for `N == 4`; zero per the enum contract.
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
            // Drop-w.
            Projection::Identity => Vec3::new(point.x, point.y, point.z),
            Projection::Orthographic { drop_axis } => match *drop_axis {
                0 => Vec3::new(point.y, point.z, point.w),
                1 => Vec3::new(point.x, point.z, point.w),
                2 => Vec3::new(point.x, point.y, point.w),
                3 => Vec3::new(point.x, point.y, point.z),
                _ => Vec3::ZERO,
            },
            Projection::Perspective4D { focal_distance } => {
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
                // Central projection from `E = viewpoint_distance * n` onto the
                // cell 3-flat (Coxeter, *Regular Polytopes*, ch. 13). A
                // chosen-cell vertex has `t = 1` and maps to itself.
                let n = *cell_normal;
                let eye = *viewpoint_distance * n;
                let n_dot_eye = n.dot(eye);
                // Sign-preserving clamp keeps a vertex on the viewer's 3-flat
                // shooting off in the correct direction rather than dividing by
                // zero or flipping across the eye.
                let raw_denom = n.dot(point) - n_dot_eye;
                let denom = if raw_denom.abs() < PROJECTION_DENOM_EPSILON {
                    PROJECTION_DENOM_EPSILON.copysign(raw_denom)
                } else {
                    raw_denom
                };
                let t = (*cell_offset - n_dot_eye) / denom;
                let result = eye + t * (point - eye);
                // Caller-supplied frame keeps a stable in-flat orientation while
                // the body rotates.
                let [e1, e2, e3] = *basis;
                Vec3::new(result.dot(e1), result.dot(e2), result.dot(e3))
            }
            Projection::Stereographic { pole } => {
                // Normalize onto S³ first: demo vertices are body-scaled, and the
                // map is only defined on the unit sphere.
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

    /// Golden Vec3 for `stereographic_frame_is_deterministic_under_tie`: the
    /// readout of pole `(0,0,1,1)/sqrt(2)`, input `(0.5,0.5,-0.5,0.5)`. A gauge
    /// flip from a tie-break or Gram-Schmidt change moves it.
    const GOLDEN_TIE_FRAME: Vec3 = Vec3::new(0.5, 0.5, std::f32::consts::FRAC_1_SQRT_2);

    #[test]
    fn r3_array_round_trip() {
        let p = Vec3::new(1.0, -2.5, 0.7);
        let arr = <EuclideanR3 as RasterizableSpace<3>>::point_to_array(p);
        let back = <EuclideanR3 as RasterizableSpace<3>>::array_to_point(arr);
        assert_eq!(p, back);
    }

    /// A scaled vertex `k * p` projects to the same R³ point as unit `p`. Pins
    /// the `point.normalize()` guard; every other test feeds unit input.
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

    #[test]
    fn r3_identity_projection_is_passthrough() {
        let p = Vec3::new(0.7, -1.3, 2.1);
        let projected =
            <EuclideanR3 as RasterizableSpace<3>>::project_point(p, &Projection::Identity);
        assert_eq!(p, projected);
    }

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

    /// Pins the lerp-factor convention: t = i/samples for i in 1..samples.
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

    /// Pins the writer-pattern guarantee the upload loop relies on for reuse.
    #[test]
    fn r3_tessellate_appends_does_not_clear() {
        let mut out = vec![Vec3::new(9.0, 9.0, 9.0)];
        <EuclideanR3 as RasterizableSpace<3>>::tessellate_segment(Vec3::ZERO, Vec3::X, 1, &mut out);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], Vec3::new(9.0, 9.0, 9.0));
        assert_eq!(out[1], Vec3::ZERO);
        assert_eq!(out[2], Vec3::X);
    }

    #[test]
    fn projection_default_is_identity() {
        let p3: Projection<3> = Projection::default();
        assert_eq!(p3, Projection::Identity);
        let p4: Projection<4> = Projection::default();
        assert_eq!(p4, Projection::Identity);
    }

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

    #[test]
    fn r4_array_round_trip() {
        let p = Vec4::new(1.0, -2.5, 0.7, 4.2);
        let arr = <EuclideanR4 as RasterizableSpace<4>>::point_to_array(p);
        let back = <EuclideanR4 as RasterizableSpace<4>>::array_to_point(arr);
        assert_eq!(p, back);
    }

    #[test]
    fn r4_identity_drops_w() {
        let p = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let projected =
            <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &Projection::Identity);
        assert_eq!(projected, Vec3::new(1.0, 2.0, 3.0));
    }

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

    /// `w = 0` gives `scale = 1`: Perspective4D collapses to Identity on the
    /// `w = 0` 3-flat.
    #[test]
    fn r4_perspective4d_w_zero_is_unchanged() {
        let p = Vec4::new(1.0, 2.0, 3.0, 0.0);
        let proj = Projection::Perspective4D {
            focal_distance: 2.0,
        };
        let got = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &proj);
        assert_eq!(got, Vec3::new(1.0, 2.0, 3.0));
    }

    /// Pins the "cube within a cube" scaling: +w face outer, -w face inner.
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
        // Scale is `4/3` at w=+0.5 and `4/5` at w=-0.5.
        let r_near = (pn.length() / 0.5_f32.mul_add(3.0_f32.sqrt(), 0.0)).abs();
        let r_far = (pf.length() / 0.5_f32.mul_add(3.0_f32.sqrt(), 0.0)).abs();
        assert!((r_near - 4.0 / 3.0).abs() < 1e-5, "near scale {r_near}");
        assert!((r_far - 4.0 / 5.0).abs() < 1e-5, "far scale {r_far}");
        assert!(pn.length() > pf.length(), "near={pn:?} far={pf:?}");
    }

    /// A vertex at the eye (`w == focal_distance`) clamps to a finite result
    /// rather than NaN-ing the buffer.
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

    /// Unsupported on R³: `Vec3::ZERO` per the enum contract.
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

    /// Tesseract vertices, unit-circumradius (`±0.5` each). The `w = +0.5` cell
    /// is the canonical boundary cell.
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

    /// The chosen boundary cell maps to itself (`t = 1`) and the frame readout is
    /// an isometry of its 3-flat, so pairwise distances are preserved: the
    /// boundary cube renders at true size (Coxeter, *Regular Polytopes*, ch. 13).
    #[test]
    fn schlegel_chosen_cell_renders_undistorted() {
        let cell_offset = 0.5;
        let proj = Projection::schlegel(Vec4::W, cell_offset, 1.5 * cell_offset);
        // The `w = +0.5` cell (first half of TESSERACT_VERTS).
        let cell: Vec<Vec4> = TESSERACT_VERTS.iter().take(8).copied().collect();
        let projected: Vec<Vec3> = cell
            .iter()
            .map(|v| <EuclideanR4 as RasterizableSpace<4>>::project_point(*v, &proj))
            .collect();
        // The cube's 3D distances ignore the shared `w`; compare against xyz.
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

    /// An oblique cell normal (the 16-cell `{+x,+y,+z,+w}` cell, normal
    /// `(1,1,1,1)/2`) must not flatten the boundary cell. Regression for the old
    /// w-drop that collapsed the `+n` vertex onto the origin.
    #[test]
    fn schlegel_non_axis_aligned_cell_is_not_flattened() {
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
        // The +axes are the boundary; the -axes nest.
        let boundary = [Vec4::X, Vec4::Y, Vec4::Z, Vec4::W];
        let inner = [-Vec4::X, -Vec4::Y, -Vec4::Z, -Vec4::W];
        let proj_pt = |v: Vec4| <EuclideanR4 as RasterizableSpace<4>>::project_point(v, &proj);

        // Boundary vertices form a regular tetrahedron, each farther from center
        // than every nested vertex.
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
        // The four boundary +axes are mutually equidistant at `sqrt(2)`; the
        // isometric frame readout must preserve that, where a drop-w would not.
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
        // Coarse non-degeneracy guard; the radius/nesting asserts above are what
        // actually discriminate the old w-drop bug.
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

    /// The supplied basis (not a rebuilt frame) drives the readout: the point is
    /// on the boundary cell, so only basis ordering can change the R³ coordinates.
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

    /// Every tesseract vertex projects to a finite R³ point under the default
    /// viewpoint, including the `w = -0.5` cell the eye looks through.
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

    /// A vertex on the viewer's 3-flat clamps to a finite result, mirroring
    /// `r4_perspective4d_at_viewer_clamps_finite`.
    #[test]
    fn schlegel_zero_denominator_clamps_finite() {
        let viewpoint_distance = 0.75;
        let proj = Projection::schlegel(Vec4::W, 0.5, viewpoint_distance);
        // `w = viewpoint_distance` puts the vertex on the eye's 3-flat.
        let p = Vec4::new(0.3, -0.2, 0.1, viewpoint_distance);
        let got = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &proj);
        for c in [got.x, got.y, got.z] {
            assert!(
                c.is_finite(),
                "degenerate denominator should clamp finite, got {got:?}"
            );
        }
    }

    /// The normal sign is load-bearing: the outward normal nests the opposite
    /// cell, the inward normal blows it past the boundary. Compares center radii
    /// (rotation-invariant) since the frame readout rotates the xyz axes.
    #[test]
    fn schlegel_outward_normal_sign_required() {
        let cell_offset = 0.5;
        let viewpoint_distance = 1.5 * cell_offset;
        let outward = Projection::schlegel(Vec4::W, cell_offset, viewpoint_distance);
        // Same `w = +0.5` cell via the inward normal `-W`: offset flips, eye lands
        // on the far side.
        let inward = Projection::schlegel(-Vec4::W, -cell_offset, viewpoint_distance);
        let opposite = Vec4::new(0.5, 0.5, 0.5, -0.5);
        // Boundary cube corners sit at radius `sqrt(3)/2`.
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

    /// `perp_frame(n)` is orthonormal and perpendicular to `n` across a spread of
    /// normals (axis-aligned and all-equal worst cases). The Schlegel readout
    /// relies on this invariant.
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

    /// Unsupported on R³: `Vec3::ZERO` per the enum contract.
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

    /// Inverse stereographic map for pole `Vec4::W`: with `s = |q|²`,
    /// `w = (s - 1)/(s + 1)`, `(x, y, z) = q*(1 - w)` (Wikipedia, *Stereographic
    /// projection*).
    fn stereo_inverse_w_pole(q: Vec3) -> Vec4 {
        let s = q.length_squared();
        let w = (s - 1.0) / (s + 1.0);
        let xyz = q * (1.0 - w);
        Vec4::new(xyz.x, xyz.y, xyz.z, w)
    }

    /// General-pole inverse: re-embed `q` against `perp_frame(pole)`, then invert
    /// the radial scaling against `|p| = 1`. Inverts `stereographic_to_r3`.
    fn stereo_inverse_general(q: Vec3, pole: Vec4) -> Vec4 {
        let (e1, e2, e3) = perp_frame(pole);
        let perp = q.x * e1 + q.y * e2 + q.z * e3;
        // |perp| = sqrt((1 + dot)/(1 - dot)); solve for dot, then restore the
        // pole part.
        let s = perp.length_squared();
        let dot = (s - 1.0) / (s + 1.0);
        dot * pole + (1.0 - dot) * perp
    }

    /// Pole `Vec4::W`: the fast path equals `(x, y, z) / (1 - w)` bitwise
    /// (Wikipedia, *Stereographic projection*). Tested on `stereographic_to_r3`
    /// directly because `project_point` re-normalizes and a second normalize of a
    /// unit vector is not bit-idempotent.
    #[test]
    fn stereographic_default_pole_is_drop_w_of_scaled() {
        for p in [
            Vec4::new(0.5, 0.5, 0.5, 0.5),
            Vec4::new(-0.5, 0.5, 0.5, -0.5),
            Vec4::new(-0.6, 0.0, 0.8, 0.0), // unit, w = 0
        ] {
            let got = stereographic_to_r3(p, Vec4::W);
            let want = Vec3::new(p.x, p.y, p.z) / (1.0 - p.w);
            assert_eq!(
                got, want,
                "fast path must match canonical formula for {p:?}"
            );
        }
        // Also bit-identical to the general frame-readout path for the W pole.
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

    /// `stereo_inverse(stereo(p)) ~= p` for unit `p` clear of the pole, default
    /// and off-axis poles. Pins the `n`-perpendicular truncation: a leaked
    /// pole-component would break the round-trip.
    #[test]
    fn stereographic_inverts_off_pole() {
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
        // Off-axis pole.
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

    /// The image lies in the pole-perpendicular 3-flat (zero pole-component after
    /// re-embedding). Catches the naive `p / (1 - dot)` leak.
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

    /// A vertex at the pole and a near-pole tangential point both stay finite via
    /// the `STEREOGRAPHIC_POLE_EPSILON` floor.
    #[test]
    fn stereographic_pole_denominator_clamped_finite() {
        for pole in [Vec4::W, Vec4::new(0.5, 0.5, 0.5, 0.5)] {
            let proj = Projection::Stereographic { pole };
            let at_pole = <EuclideanR4 as RasterizableSpace<4>>::project_point(pole, &proj);
            for c in [at_pole.x, at_pole.y, at_pole.z] {
                assert!(
                    c.is_finite(),
                    "pole input must clamp finite, got {at_pole:?}"
                );
            }
            // Just off the pole, mostly tangential.
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

    /// The antipode `-pole` maps to the origin (`dot = -1`, denominator `2`): the
    /// safe far point, distinct from the singular pole.
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

    /// A pole tied between two axes yields the same Vec3 across calls and matches
    /// a golden value. Tier-0 bit-repro guard: the frame must not flip gauge under
    /// the tie-break.
    #[test]
    fn stereographic_frame_is_deterministic_under_tie() {
        // Pole equidistant from the z and w axes.
        let pole = Vec4::new(0.0, 0.0, 1.0, 1.0).normalize();
        let proj = Projection::Stereographic { pole };
        let p = Vec4::new(0.5, 0.5, -0.5, 0.5); // unit
        let first = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &proj);
        for _ in 0..16 {
            let again = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &proj);
            assert_eq!(first, again, "frame must be byte-stable across calls");
        }
        // Golden value locks the chosen gauge; the `==` ladder drops z (index 2)
        // on the z/w max tie.
        assert_relative_eq!(first.x, GOLDEN_TIE_FRAME.x, epsilon = 1e-6);
        assert_relative_eq!(first.y, GOLDEN_TIE_FRAME.y, epsilon = 1e-6);
        assert_relative_eq!(first.z, GOLDEN_TIE_FRAME.z, epsilon = 1e-6);
    }

    /// `perp_frame(pole)` is orthonormal and finite for poles swept toward each
    /// axis, including exactly an axis. The stereographic readout relies on this.
    #[test]
    fn stereographic_frame_orthonormal_for_every_pole() {
        let mut poles = vec![Vec4::X, Vec4::Y, Vec4::Z, Vec4::W];
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

    /// Stereographic is conformal: the angle between two edges at a shared
    /// non-pole vertex is preserved (Wikipedia, *Stereographic projection*).
    #[test]
    fn stereographic_is_conformal() {
        let s = SphericalS3Embedded;
        let pole = Vec4::W;
        let proj = Projection::Stereographic { pole };
        let v = Vec4::new(0.3, -0.1, 0.2, -0.5).normalize();
        let a = Vec4::new(0.5, 0.4, -0.1, -0.3).normalize();
        let b = Vec4::new(-0.2, 0.3, 0.6, -0.4).normalize();
        // Intrinsic edge angle at `v` from the geodesic tangents.
        let ta = s.log(v, a);
        let tb = s.log(v, b);
        let intrinsic = (ta.dot(tb) / (ta.length() * tb.length()))
            .clamp(-1.0, 1.0)
            .acos();
        // Secant a small step along each geodesic approximates the projected
        // tangent.
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

    /// Unsupported on R³: `Vec3::ZERO` per the enum contract.
    #[test]
    fn stereographic_unsupported_on_r3_returns_zero() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let proj = Projection::Stereographic { pole: Vec4::W };
        let got = <EuclideanR3 as RasterizableSpace<3>>::project_point(p, &proj);
        assert_eq!(got, Vec3::ZERO);
    }

    /// `tessellate_segment` on flat R⁴ is plain lerp across all four components.
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
