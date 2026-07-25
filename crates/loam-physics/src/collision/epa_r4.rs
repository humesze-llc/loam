//! EPA in R⁴: Expanding Polytope for 4D penetration depth.
//!
//! Parallel to [`super::epa`][mod@super::epa] (3D), with three changes:
//!
//! 1. **Faces** are tetrahedra (4 vertex indices), not triangles.
//! 2. **Normals** are the Hodge dual of `(b−a) ∧ (c−a) ∧ (d−a)`, the 4D
//!    generalized cross product, perpendicular to all three edge vectors.
//! 3. **Horizon** triangles are those unique to one removed tetra; each plus the
//!    new support point forms a new tetrahedral face.
//!
//! Contact-point reconstruction uses the Gram-matrix projection from
//! [`super::simplex_r4`] on the terminating face's four vertices.

use glam::Vec4;

use super::gjk_r4::{minkowski_support_r4, MinkowskiPoint4, SupportFn4};
use super::simplex_r4::closest_to_origin;

const EPA_MAX_ITERATIONS: u32 = 96;
const EPA_TOLERANCE: f32 = 1e-3;
const EPA_MAX_VERTICES: usize = 192;

/// Resolved 4D contact info; the 3D [`super::epa::ContactInfo`] with `Vec4` fields.
#[derive(Clone, Copy, Debug)]
pub struct ContactInfo4 {
    pub normal: Vec4,
    pub penetration: f32,
    pub point: Vec4,
}

/// Tetrahedral face of the expanding polytope.
#[derive(Clone, Copy, Debug)]
struct Face4 {
    v: [usize; 4],
    normal: Vec4,
    distance: f32,
}

struct Polytope4 {
    vertices: Vec<MinkowskiPoint4>,
    faces: Vec<Face4>,
    /// Centroid of the seed 5-simplex; stays interior under all convex expansions,
    /// so it tiebreaks face orientation when the origin sits on a face plane
    /// (common for symmetric Minkowski differences).
    centroid: Vec4,
}

impl Polytope4 {
    fn from_simplex(simplex: [MinkowskiPoint4; 5]) -> Self {
        let vertices = simplex.to_vec();
        let centroid = (simplex[0].point
            + simplex[1].point
            + simplex[2].point
            + simplex[3].point
            + simplex[4].point)
            * 0.2;

        // Five tetrahedral faces of the 4-simplex: each excludes the `l`-th vertex.
        // Orientation rule lives in `build_face`.
        let mut faces = Vec::with_capacity(5);
        for l in 0..5 {
            let mut tet = [0usize; 4];
            let mut idx = 0;
            for i in 0..5 {
                if i != l {
                    tet[idx] = i;
                    idx += 1;
                }
            }
            if let Some(face) = build_face(&vertices, tet[0], tet[1], tet[2], tet[3], centroid) {
                faces.push(face);
            }
        }
        Self {
            vertices,
            faces,
            centroid,
        }
    }

    /// Closest face to the origin, preferring strictly positive-distance faces.
    ///
    /// Distance-0 faces are common in 4D (coplanar Minkowski-diff vertices spawn
    /// through-origin faces); chasing the global minimum never converges. Prefer
    /// the smallest positive-distance face, falling back to distance-0 only when
    /// none exists (genuine tangency, or a fully symmetric seed needing expansion).
    fn closest_face(&self) -> Option<usize> {
        if let Some((idx, _)) = self
            .faces
            .iter()
            .enumerate()
            .filter(|(_, f)| f.distance > ORIGIN_ON_PLANE_EPS)
            .min_by(|a, b| a.1.distance.total_cmp(&b.1.distance))
        {
            return Some(idx);
        }
        self.faces
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.distance.total_cmp(&b.1.distance))
            .map(|(idx, _)| idx)
    }

    fn expand(&mut self, support: MinkowskiPoint4) {
        let new_idx = self.vertices.len();
        self.vertices.push(support);

        // Horizon: triangles unique to one removed tetra. Shared triangles cancel.
        let mut horizon: Vec<Triangle> = Vec::new();
        let mut keep = Vec::with_capacity(self.faces.len());

        for f in self.faces.drain(..) {
            let view = support.point - self.vertices[f.v[0]].point;
            if f.normal.dot(view) > -FACE_COPLANAR_EPS {
                for tri in tet_triangles(&f.v) {
                    add_or_remove_triangle(&mut horizon, tri);
                }
            } else {
                keep.push(f);
            }
        }
        self.faces = keep;

        // Each horizon triangle + new vertex -> a new tetrahedral face, oriented
        // against the seed centroid (still interior by convexity).
        let centroid = self.centroid;
        for tri in &horizon {
            if let Some(face) = build_face(&self.vertices, tri.0, tri.1, tri.2, new_idx, centroid) {
                self.faces.push(face);
            }
        }
    }
}

/// Threshold below which the origin counts as on a face plane, switching the
/// interior reference to the seed centroid. Empirical: above f32 noise (~1e-7 at
/// unit scale), below any real penetration depth.
const ORIGIN_ON_PLANE_EPS: f32 = 1e-4;

/// Band around a face plane in which a support point counts as lying on it, so
/// that `expand` retires the face instead of keeping it.
///
/// A facet of a Minkowski difference of polytopes is generally not a simplex
/// (the difference body of a 4-simplex has prism facets), so it is tiled by
/// several coplanar tetrahedral faces and support points land exactly on their
/// shared plane. Keeping those tiles while their neighbours are retired stitches
/// the new vertex into a facet that is already covered: the surface stops being
/// convex, later iterations chase interior faces, and EPA returns a normal that
/// can point from B toward A. Same conditioning class as [`ORIGIN_ON_PLANE_EPS`]:
/// the plane residual of a unit-scale 4-term dot product is a few ULPs (~4e-7,
/// and 1e-7 measurably misses coplanar tiles), while the gap that is the
/// precondition for calling `expand` at all is `EPA_TOLERANCE`, two orders up.
const FACE_COPLANAR_EPS: f32 = 1e-5;

/// A triangle index triple; winding implicit in order.
type Triangle = (usize, usize, usize);

/// The four triangular faces of a tetrahedron, each excluding one vertex. Winding
/// is irrelevant here: matching is order-insensitive (see `add_or_remove_triangle`).
fn tet_triangles(tet: &[usize; 4]) -> [Triangle; 4] {
    let (a, b, c, d) = (tet[0], tet[1], tet[2], tet[3]);
    [(a, b, c), (a, b, d), (a, c, d), (b, c, d)]
}

/// Add a triangle to the horizon, or cancel it if its index-set is already
/// present (a triangle shared by two removed tetra is interior).
fn add_or_remove_triangle(horizon: &mut Vec<Triangle>, tri: Triangle) {
    let key = sort_triangle(tri);
    if let Some(pos) = horizon.iter().position(|t| sort_triangle(*t) == key) {
        horizon.swap_remove(pos);
    } else {
        horizon.push(tri);
    }
}

fn sort_triangle(t: Triangle) -> (usize, usize, usize) {
    let mut a = [t.0, t.1, t.2];
    a.sort_unstable();
    (a[0], a[1], a[2])
}

/// Build a tetrahedral face `(a, b, c, d)` with outward unit normal and distance
/// from origin. `None` when degenerate (near-coplanar edges, tiny normal).
///
/// Orientation: keep both the origin and the centroid on the interior side. They
/// agree except when the origin lies on the face plane (symmetric Minkowski
/// differences); there the origin test gives no signal and the centroid decides.
fn build_face(
    verts: &[MinkowskiPoint4],
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    centroid: Vec4,
) -> Option<Face4> {
    let pa = verts[a].point;
    let pb = verts[b].point;
    let pc = verts[c].point;
    let pd = verts[d].point;

    let raw_normal = hodge_dual_of_trivector_wedge(pb - pa, pc - pa, pd - pa);
    let len = raw_normal.length();
    if len < 1e-8 {
        return None;
    }
    let normal = raw_normal / len;

    // `normal · pa` is the plane offset; positive means origin is on the interior
    // (`-normal`) side, so `+normal` is already outward.
    let signed_origin = normal.dot(pa);
    let flip = if signed_origin.abs() > ORIGIN_ON_PLANE_EPS {
        signed_origin < 0.0
    } else {
        // Origin on the plane: the centroid decides. Interior on `+normal` -> flip.
        let signed_c = normal.dot(centroid - pa);
        signed_c > 0.0
    };

    let (outward, v_order) = if flip {
        (-normal, [a, b, d, c])
    } else {
        (normal, [a, b, c, d])
    };

    // Clamp at 0: the origin-on-plane case can dip slightly negative from noise.
    let distance = outward.dot(pa).max(0.0);

    Some(Face4 {
        v: v_order,
        normal: outward,
        distance,
    })
}

/// Generalized 4D cross product: the vector perpendicular to `u`, `v`, `w`. The
/// Hodge dual of `u ∧ v ∧ w`, mapping `e_123 -> −e_4, e_124 -> +e_3,
/// e_134 -> −e_2, e_234 -> +e_1`. Components are four signed 3×3 determinants.
fn hodge_dual_of_trivector_wedge(u: Vec4, v: Vec4, w: Vec4) -> Vec4 {
    // t_ijk = det of (u, v, w) on columns (i, j, k).
    let t_234 = det3(u.y, u.z, u.w, v.y, v.z, v.w, w.y, w.z, w.w);
    let t_134 = det3(u.x, u.z, u.w, v.x, v.z, v.w, w.x, w.z, w.w);
    let t_124 = det3(u.x, u.y, u.w, v.x, v.y, v.w, w.x, w.y, w.w);
    let t_123 = det3(u.x, u.y, u.z, v.x, v.y, v.z, w.x, w.y, w.z);

    Vec4::new(t_234, -t_134, t_124, -t_123)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn det3(
    a00: f32,
    a01: f32,
    a02: f32,
    a10: f32,
    a11: f32,
    a12: f32,
    a20: f32,
    a21: f32,
    a22: f32,
) -> f32 {
    a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)
}

/// Resolve penetration for overlapping 4D shapes given GJK's terminating 5-simplex.
pub fn epa_r4<A: SupportFn4, B: SupportFn4>(
    a: &A,
    b: &B,
    initial_simplex: [MinkowskiPoint4; 5],
) -> Option<ContactInfo4> {
    // Reject a zero-4-volume seed: |det([p1-p0; p2-p0; p3-p0; p4-p0])|.
    let p0 = initial_simplex[0].point;
    let d1 = initial_simplex[1].point - p0;
    let d2 = initial_simplex[2].point - p0;
    let d3 = initial_simplex[3].point - p0;
    let d4 = initial_simplex[4].point - p0;
    let volume = det4(d1, d2, d3, d4).abs();
    if volume < 1e-8 {
        return None;
    }

    let mut polytope = Polytope4::from_simplex(initial_simplex);

    for _ in 0..EPA_MAX_ITERATIONS {
        let face_idx = polytope.closest_face()?;
        let face = polytope.faces[face_idx];

        let support = minkowski_support_r4(a, b, face.normal);
        let new_distance = support.point.dot(face.normal);

        if !new_distance.is_finite() || !support.point.is_finite() {
            return None;
        }

        if (new_distance - face.distance).abs() < EPA_TOLERANCE {
            return contact_from_face(&polytope, face);
        }

        polytope.expand(support);
        if polytope.vertices.len() > EPA_MAX_VERTICES {
            break;
        }
    }

    // Cap hit: return the best-estimate contact rather than failing.
    tracing::debug!(
        max_iterations = EPA_MAX_ITERATIONS,
        vertices = polytope.vertices.len(),
        "EPA 4D hit iteration cap; returning best-estimate contact",
    );
    let face_idx = polytope.closest_face()?;
    contact_from_face(&polytope, polytope.faces[face_idx])
}

/// 4×4 determinant of the matrix with rows `r0..r3` (Laplace expansion, first row).
fn det4(r0: Vec4, r1: Vec4, r2: Vec4, r3: Vec4) -> f32 {
    r0.x * det3(r1.y, r1.z, r1.w, r2.y, r2.z, r2.w, r3.y, r3.z, r3.w)
        - r0.y * det3(r1.x, r1.z, r1.w, r2.x, r2.z, r2.w, r3.x, r3.z, r3.w)
        + r0.z * det3(r1.x, r1.y, r1.w, r2.x, r2.y, r2.w, r3.x, r3.y, r3.w)
        - r0.w * det3(r1.x, r1.y, r1.z, r2.x, r2.y, r2.z, r3.x, r3.y, r3.z)
}

fn contact_from_face(polytope: &Polytope4, face: Face4) -> Option<ContactInfo4> {
    let v0 = polytope.vertices[face.v[0]];
    let v1 = polytope.vertices[face.v[1]];
    let v2 = polytope.vertices[face.v[2]];
    let v3 = polytope.vertices[face.v[3]];

    // Closest point on the face hyperplane to the origin, in Minkowski-diff space.
    let closest = face.normal * face.distance;

    // Barycentric weights of `closest` on the tetra via Gram-matrix projection.
    let simplex_points = [v0.point, v1.point, v2.point, v3.point];
    let proj = closest_to_origin(
        &simplex_points
            .iter()
            .map(|p| *p - closest)
            .collect::<Vec<_>>(),
    );
    let weights = &proj.weights;

    let point_a = v0.sa * weights[0] + v1.sa * weights[1] + v2.sa * weights[2] + v3.sa * weights[3];
    let point_b = v0.sb * weights[0] + v1.sb * weights[1] + v2.sb * weights[2] + v3.sb * weights[3];

    Some(ContactInfo4 {
        normal: face.normal,
        penetration: face.distance,
        point: (point_a + point_b) * 0.5,
    })
}

#[cfg(test)]
mod tests {
    use super::super::gjk_r4::{gjk_intersect_r4, GjkResult4, Sphere4};
    use super::*;

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol,
            "{a} not close to {b} (tol {tol}, diff {})",
            (a - b).abs()
        );
    }

    #[test]
    fn sphere_sphere_penetration_matches_analytical() {
        // Centers 0.8 apart, radius 0.5 each: penetration = 1.0 − 0.8 = 0.2.
        let a = Sphere4 {
            center: Vec4::new(0.0, 0.0, 0.0, 0.0),
            radius: 0.5,
        };
        let b = Sphere4 {
            center: Vec4::new(0.8, 0.0, 0.0, 0.0),
            radius: 0.5,
        };
        let simplex = match gjk_intersect_r4(&a, &b, Vec4::X) {
            GjkResult4::Intersecting { simplex } => simplex,
            _ => panic!("spheres should overlap"),
        };
        let contact = epa_r4(&a, &b, simplex).expect("EPA should succeed");
        assert_close(contact.penetration, 0.2, 5e-3);
        assert!(
            contact.normal.dot(Vec4::X) > 0.99,
            "normal must run from A toward B along +x, got {:?}",
            contact.normal
        );
    }

    #[test]
    fn sphere_sphere_penetration_moderate() {
        // Analytical depth 0.5.
        let a = Sphere4 {
            center: Vec4::ZERO,
            radius: 0.5,
        };
        let b = Sphere4 {
            center: Vec4::new(0.5, 0.0, 0.0, 0.0),
            radius: 0.5,
        };
        let simplex = match gjk_intersect_r4(&a, &b, Vec4::X) {
            GjkResult4::Intersecting { simplex } => simplex,
            _ => panic!("spheres should overlap"),
        };
        let contact = epa_r4(&a, &b, simplex).expect("EPA should succeed");
        assert_close(contact.penetration, 0.5, 5e-2);
        assert!(
            contact.normal.dot(Vec4::X) > 0.99,
            "normal must run from A toward B along +x, got {:?}",
            contact.normal
        );
    }

    #[test]
    fn sphere_sphere_contact_point_between_centers() {
        // Shallow overlap: contact point should sit on or near the line between centers.
        let a = Sphere4 {
            center: Vec4::ZERO,
            radius: 0.5,
        };
        let b = Sphere4 {
            center: Vec4::new(0.8, 0.0, 0.0, 0.0),
            radius: 0.5,
        };
        let simplex = match gjk_intersect_r4(&a, &b, Vec4::X) {
            GjkResult4::Intersecting { simplex } => simplex,
            _ => panic!("spheres should overlap"),
        };
        let contact = epa_r4(&a, &b, simplex).expect("EPA should succeed");
        // Contact y/z/w should be near zero.
        assert!(contact.point.y.abs() < 0.1);
        assert!(contact.point.z.abs() < 0.1);
        assert!(contact.point.w.abs() < 0.1);
    }

    // The three polytope fixtures below all take `B = A + t`, so the Minkowski
    // difference is `K − t` with `K = A ⊕ (−A)` the difference body. For an
    // interior origin the depth is `min_j (h_K(u_j) − ⟨u_j, t⟩)` over K's
    // facet normals `u_j`, and the minimizing `u_j` is the contact normal.
    // Support functions add over Minkowski sums (Schneider 2014, *Convex
    // Bodies: The Brunn-Minkowski Theory*, §1.7) and the normal fan of a sum
    // is the common refinement of the summands' fans (Ziegler 1995, *Lectures
    // on Polytopes*, §7.1), which is what gives each K below its facet list.

    /// Deep pentatope overlap. K's facet normals are `ŵ_S ∝ Σ_{i∈S} v_i` over
    /// the proper nonempty subsets S, and at circumradius 1
    /// (`⟨v_i, v_j⟩ = −1/4` for `i ≠ j`) `h_K(ŵ_S) = 5 / (2·√(k(5−k)))` with
    /// `k = |S|`. For `t = 0.3·x̂` the minimizers are the two facets spanned by
    /// the vertices with `x = +√5/4`, tied at depth `5/(2√6) − 0.3·√5/√6` with
    /// x-component `√5/√6`.
    #[test]
    fn pentatope_pentatope_contact_matches_difference_body_facet() {
        use crate::collision::gjk_r4::ConvexHull4;
        use crate::euclidean_r4::pentatope_vertices;

        let va: Vec<Vec4> = pentatope_vertices(1.0);
        let vb: Vec<Vec4> = pentatope_vertices(1.0)
            .into_iter()
            .map(|v| v + Vec4::new(0.3, 0.0, 0.0, 0.0))
            .collect();

        let a = ConvexHull4 { vertices: &va };
        let b = ConvexHull4 { vertices: &vb };
        let simplex = match gjk_intersect_r4(&a, &b, Vec4::X) {
            GjkResult4::Intersecting { simplex } => simplex,
            _ => panic!("pentatopes should overlap"),
        };
        let contact = epa_r4(&a, &b, simplex).expect("EPA should succeed");

        let root6 = 6.0_f32.sqrt();
        let normal_x = 5.0_f32.sqrt() / root6;
        assert_close(
            contact.penetration,
            5.0 / (2.0 * root6) - 0.3 * normal_x,
            EPA_TOLERANCE,
        );
        // The tie leaves the facet unpinned but not the direction: both tied
        // facets carry the same x-component, and it is positive (A toward B).
        assert_close(contact.normal.x, normal_x, EPA_TOLERANCE);
        let n2 = contact.normal.length_squared();
        assert!(
            (n2 - 1.0).abs() < 1e-2,
            "normal should be unit-length: |n|² = {n2}"
        );
    }

    /// Two tesseracts overlapping along all four axes. K is the cube of
    /// half-extent `r` (the difference body of a half-extent-`r/2` cube), so
    /// `h_K(±ê_i) = r` and the minimizer is the axis carrying the largest
    /// `|t_i|`: depth `r − |t_x|`, normal `+x̂`, no tie.
    #[test]
    fn tesseract_tesseract_contact_matches_deepest_axis() {
        use crate::collision::gjk_r4::ConvexHull4;
        use crate::euclidean_r4::tesseract_vertices;

        let va: Vec<Vec4> = tesseract_vertices(1.0);
        let vb: Vec<Vec4> = tesseract_vertices(1.0)
            .into_iter()
            .map(|v| v + Vec4::new(0.4, 0.2, 0.1, 0.0))
            .collect();

        let a = ConvexHull4 { vertices: &va };
        let b = ConvexHull4 { vertices: &vb };
        let simplex = match gjk_intersect_r4(&a, &b, Vec4::X) {
            GjkResult4::Intersecting { simplex } => simplex,
            _ => panic!("tesseracts should overlap"),
        };
        let contact = epa_r4(&a, &b, simplex).expect("EPA should succeed");

        assert_close(contact.penetration, 1.0 - 0.4, EPA_TOLERANCE);
        assert!(
            contact.normal.dot(Vec4::X) > 0.999,
            "normal must run from A toward B along +x, got {:?}",
            contact.normal
        );
    }

    /// 16-cell vs 16-cell: the 8-vertex cross-polytope, fewer support points
    /// than the tesseract. K is the L¹ ball of radius `2r`, whose facet normals
    /// are the 16 sign patterns `(±1,±1,±1,±1)/2` with `h_K = r`. For
    /// `t = 0.5·x̂` the eight patterns with `+1` in x tie at depth `r − t_x/2`,
    /// all with x-component `1/2`.
    #[test]
    fn cell16_cell16_contact_matches_l1_facet() {
        use crate::collision::gjk_r4::ConvexHull4;
        use crate::euclidean_r4::cell16_vertices;

        let va: Vec<Vec4> = cell16_vertices(1.0);
        let vb: Vec<Vec4> = cell16_vertices(1.0)
            .into_iter()
            .map(|v| v + Vec4::new(0.5, 0.0, 0.0, 0.0))
            .collect();

        let a = ConvexHull4 { vertices: &va };
        let b = ConvexHull4 { vertices: &vb };
        let simplex = match gjk_intersect_r4(&a, &b, Vec4::X) {
            GjkResult4::Intersecting { simplex } => simplex,
            _ => panic!("16-cells should overlap"),
        };
        let contact = epa_r4(&a, &b, simplex).expect("EPA should succeed");

        assert_close(contact.penetration, 1.0 - 0.25, EPA_TOLERANCE);
        assert_close(contact.normal.x, 0.5, EPA_TOLERANCE);
    }
}
