//! EPA in R⁴, Expanding Polytope for 4D penetration depth.
//!
//! Parallel to [`super::epa`][mod@super::epa] (3D) with three dimensionality changes:
//!
//! 1. **Faces** are tetrahedra (3-simplices, 4 vertex indices), not triangles.
//! 2. **Face normals** come from the Hodge dual of the trivector `(b−a) ∧ (c−a) ∧ (d−a)`.
//!    Concretely this is the 4D "generalized cross product", four signed 3×3 determinants
//!    of the 3-row matrix `[b−a; c−a; d−a]`. The result is perpendicular to all three
//!    edge vectors.
//! 3. **Horizon** of a polytope expansion: triangles shared between removed tetrahedral
//!    faces are interior; unique triangles are the 3D horizon. Each horizon triangle,
//!    combined with the new support point, becomes a new tetrahedral face.
//!
//! Barycentric reconstruction for the contact point uses the Gram-matrix projection from
//! [`super::simplex_r4`] applied to the terminating face's four vertices.

use glam::Vec4;

use super::gjk_r4::{minkowski_support_r4, MinkowskiPoint4, SupportFn4};
use super::simplex_r4::closest_to_origin;

const EPA_MAX_ITERATIONS: u32 = 96;
const EPA_TOLERANCE: f32 = 1e-3;
const EPA_MAX_VERTICES: usize = 192;

/// Resolved contact information in 4D. Same shape as the 3D [`super::epa::ContactInfo`] but
/// with `Vec4` fields.
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
    /// Centroid of the seed 5-simplex. Guaranteed interior to the polytope for all subsequent
    /// convex expansions, so it's a reliable tiebreaker when the origin itself sits on a face
    /// plane (common for symmetric Minkowski differences, where the origin lies on an edge of the
    /// seed simplex and multiple initial faces pass through it).
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

        // Five tetrahedral faces of the 4-simplex: each one is the
        // tetra of all vertices except the `l`-th. Orientation uses
        // a hybrid "origin-first, centroid-fallback" rule, see
        // `build_face`.
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

    /// Face with smallest distance from origin.
    ///
    /// Distance-0 faces are common in 4D EPA: many Minkowski-diff vertices end up coplanar (e.g.
    /// pentatope-pentatope produces dozens of `w=0` points), which spawns "through-origin"
    /// faces during expansion. Naively picking the smallest distance always chases these
    /// spurious faces and never converges on the real Minkowski boundary.
    ///
    /// Strategy: if the polytope has any **strictly positive-distance** face, prefer the smallest
    /// of those — they're real boundary candidates. Only fall back to a distance-0 face when no
    /// positive face exists (the boundary genuinely touches the origin, e.g. tangent shapes; or
    /// the seed simplex is so symmetric that every face passes through origin and we need
    /// expansion to break the symmetry).
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

        // Horizon = set of triangles on the boundary of the region
        // being removed. Encoded as sorted-by-identity triples of
        // vertex indices; triangles shared between two removed
        // tetrahedra are interior and cancel, unique ones form the
        // 3D horizon "skin."
        let mut horizon: Vec<Triangle> = Vec::new();
        let mut keep = Vec::with_capacity(self.faces.len());

        for f in self.faces.drain(..) {
            let view = support.point - self.vertices[f.v[0]].point;
            if f.normal.dot(view) > 0.0 {
                // Visible from the new point -> remove.
                for tri in tet_triangles(&f.v) {
                    add_or_remove_triangle(&mut horizon, tri);
                }
            } else {
                keep.push(f);
            }
        }
        self.faces = keep;

        // Each horizon triangle + new vertex -> new tetrahedral face.
        // Re-uses the seed centroid as the interior reference,
        // still inside the (only-expanding) polytope by convexity.
        let centroid = self.centroid;
        for tri in &horizon {
            if let Some(face) = build_face(&self.vertices, tri.0, tri.1, tri.2, new_idx, centroid) {
                self.faces.push(face);
            }
        }
    }
}

/// Threshold below which the origin's signed distance to a face
/// plane is considered "on the plane" and the seed centroid is used
/// as the interior reference instead. Empirical, larger than f32
/// noise (~1e-7 for unit-scale geometry) but much smaller than any
/// real penetration depth.
const ORIGIN_ON_PLANE_EPS: f32 = 1e-4;

/// A triangle (3-vertex index tuple) with sign implicit in order.
type Triangle = (usize, usize, usize);

/// The four triangular faces of a tetrahedron `(a, b, c, d)`. Winding
/// is kept consistent with the tetrahedron's, each triangle excludes
/// one vertex in the rotation `(a, b, c, d)`.
fn tet_triangles(tet: &[usize; 4]) -> [Triangle; 4] {
    // The opposite-vertex-excluded triangles:
    //   exclude d -> (a, b, c)
    //   exclude c -> (a, b, d) but with flipped winding
    //   exclude b -> (a, c, d)
    //   exclude a -> (b, c, d) with flipped winding
    //
    // Signs don't matter here since we use an order-insensitive
    // match (see `add_or_remove_triangle`). Use the canonical
    // index-order subsets.
    let (a, b, c, d) = (tet[0], tet[1], tet[2], tet[3]);
    [(a, b, c), (a, b, d), (a, c, d), (b, c, d)]
}

/// Add a triangle to the horizon, or cancel it if the same (index-set) triangle is
/// already present. Two removed tetra share one triangle; that triangle is inside the
/// region being removed and contributes no horizon.
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

/// Build a tetrahedral face `(a, b, c, d)` with outward unit normal
/// and distance-from-origin.
///
/// # Orientation (hybrid rule)
///
/// EPA's invariant is that both the origin and the polytope's centroid are interior
/// points. The outward normal should put both on the same (negative-distance) side of the
/// face plane. Usually they agree; the tricky case is when **the origin lies on a face
/// plane**: common for symmetric Minkowski differences where the seed 5-simplex has an
/// edge passing through origin. Then the origin-based test gives no signal, and we fall
/// back to the seed centroid (guaranteed off-plane except for contrived full-symmetry
/// cases, where the face is degenerate anyway).
///
/// Returns `None` when the face is degenerate (three edges nearly coplanar, yielding a
/// tiny normal magnitude).
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

    // Signed position of the origin relative to the face plane along
    // `+normal`. `normal · pa` is the plane's offset from origin; if
    // positive, origin is on `-normal` (interior) side and we keep
    // `+normal` as outward.
    let signed_origin = normal.dot(pa);
    let flip = if signed_origin.abs() > ORIGIN_ON_PLANE_EPS {
        signed_origin < 0.0
    } else {
        // Origin lies on the face plane. Use the centroid as
        // tiebreaker: if centroid is on `+normal` side (relative to
        // pa), `+normal` points toward interior -> flip.
        let signed_c = normal.dot(centroid - pa);
        signed_c > 0.0
    };

    let (outward, v_order) = if flip {
        (-normal, [a, b, d, c])
    } else {
        (normal, [a, b, c, d])
    };

    // Face distance from origin along the outward normal. Clamp at 0
    // for the origin-on-plane case where the signed result could dip
    // slightly negative from f32 noise.
    let distance = outward.dot(pa).max(0.0);

    Some(Face4 {
        v: v_order,
        normal: outward,
        distance,
    })
}

/// Generalized 4D cross product: the vector perpendicular to three 4-vectors `u`, `v`,
/// `w`. Equal to the Hodge dual of the trivector `u ∧ v ∧ w`, which for basis `e_ijk`
/// maps `e_123 -> −e_4, e_124 -> +e_3, e_134 -> −e_2, e_234 -> +e_1`.
///
/// Components are four 3×3 determinants of the column-sub-matrices of `[u; v; w]`, with
/// alternating signs.
fn hodge_dual_of_trivector_wedge(u: Vec4, v: Vec4, w: Vec4) -> Vec4 {
    // u ∧ v ∧ w trivector components:
    //   t_ijk = det of (u, v, w) columns (i, j, k).
    let t_234 = det3(u.y, u.z, u.w, v.y, v.z, v.w, w.y, w.z, w.w);
    let t_134 = det3(u.x, u.z, u.w, v.x, v.z, v.w, w.x, w.z, w.w);
    let t_124 = det3(u.x, u.y, u.w, v.x, v.y, v.w, w.x, w.y, w.w);
    let t_123 = det3(u.x, u.y, u.z, v.x, v.y, v.z, w.x, w.y, w.z);

    // Hodge dual: e_123 -> −e_4, e_124 -> +e_3, e_134 -> −e_2, e_234 -> +e_1.
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

/// Main entry point: resolve penetration for overlapping 4D shapes
/// given GJK's terminating 5-simplex.
pub fn epa_r4<A: SupportFn4, B: SupportFn4>(
    a: &A,
    b: &B,
    initial_simplex: [MinkowskiPoint4; 5],
) -> Option<ContactInfo4> {
    // Reject a degenerate starting simplex (zero 4D volume). The
    // signed 4-volume is `det([p1-p0; p2-p0; p3-p0; p4-p0])`; we only
    // need the magnitude.
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

    // Iteration cap: return best-estimate contact from current closest face rather than
    // failing. Debug-level trace so 4D narrowphase tuning has the same observability as 3D.
    tracing::debug!(
        max_iterations = EPA_MAX_ITERATIONS,
        vertices = polytope.vertices.len(),
        "EPA 4D hit iteration cap; returning best-estimate contact",
    );
    let face_idx = polytope.closest_face()?;
    contact_from_face(&polytope, polytope.faces[face_idx])
}

/// 4×4 determinant of the matrix whose rows are `r0..r3`.
fn det4(r0: Vec4, r1: Vec4, r2: Vec4, r3: Vec4) -> f32 {
    // Laplace expansion along the first row.
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

    // Closest point on the face's hyperplane to the origin, in
    // Minkowski-diff space.
    let closest = face.normal * face.distance;

    // Barycentric coords of `closest` on the tetrahedron
    // (v0, v1, v2, v3). Solve via the Gram-matrix projection: project
    // `closest - v0` onto the span of `{vi - v0}`, recover weights.
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
        // Two spheres in 4D, centers 0.8 apart along +x, radius 0.5
        // each. Analytical penetration = 2·0.5 − 0.8 = 0.2.
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
        // Contact normal should point along +x (from A toward B).
        assert!(
            contact.normal.x.abs() > 0.9,
            "normal should be roughly along x, got {:?}",
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
    }

    #[test]
    fn sphere_sphere_contact_point_between_centers() {
        // Shallow overlap: contact point should sit on or near the
        // line between centers.
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

    /// Two overlapping pentatopes produce a finite penetration with a unit-length normal.
    /// This is the case that used to collapse to a zero-distance face under the fragile
    /// interior-reference orientation heuristic; robustified via the hybrid origin-first,
    /// centroid-fallback rule in `build_face`.
    #[test]
    fn pentatope_pentatope_penetration_nonzero() {
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
        assert!(
            contact.penetration > 0.0 && contact.penetration.is_finite(),
            "penetration = {}",
            contact.penetration
        );
        let n2 = contact.normal.length_squared();
        assert!(
            (n2 - 1.0).abs() < 1e-2,
            "normal should be unit-length: |n|² = {n2}"
        );
    }

    /// Two tesseracts sharing a corner region along all four axes. Sharper features than
    /// the pentatope; used to be another EPA-collapse source before the orientation fix.
    #[test]
    fn tesseract_tesseract_penetration_nonzero() {
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
        assert!(
            contact.penetration > 0.0 && contact.penetration.is_finite(),
            "penetration = {}",
            contact.penetration
        );
    }

    /// 16-cell vs 16-cell: the cross-polytope with 8 vertices. Tests the GJK->EPA
    /// pipeline on a sharp-vertexed polytope that has fewer support points than the
    /// tesseract.
    #[test]
    fn cell16_cell16_penetration_nonzero() {
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
        assert!(
            contact.penetration > 0.0 && contact.penetration.is_finite(),
            "penetration = {}",
            contact.penetration
        );
    }
}
