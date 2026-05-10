//! EPA: Expanding Polytope Algorithm.
//!
//! Given the tetrahedron simplex produced by [`super::gjk_intersect`] when two shapes
//! overlap, EPA expands that simplex into a convex polytope whose surface matches
//! (locally) the boundary of the Minkowski difference `A ⊖ B`. The point on that
//! surface closest to the origin gives the **minimum translation** needed to separate
//! the shapes, i.e. the contact normal and penetration depth.
//!
//! Contact point reconstruction uses the barycentric coordinates of the closest point
//! on the terminating face, applied to the original support points on A and B (which
//! GJK already cached in each [`MinkowskiPoint`]).
//!
//! Algorithm (repeated until convergence):
//! 1. Find the face of the current polytope closest to the origin.
//! 2. Query a new support point along that face's outward normal.
//! 3. If the support's distance from the origin ≈ the face's distance, the face is on
//!    the Minkowski boundary, we're done.
//! 4. Otherwise, add the support to the polytope: remove every face whose outward
//!    normal "sees" the new point, then stitch a fan of new triangles from each
//!    horizon edge to the new vertex.
//!
//! 3D only for now. The face-normal reconstruction step uses the cross product;
//! generalizing to 4D requires a "generalized cross" (the vector orthogonal to three
//! given vectors, via a 4D determinant expansion). When 4D lands, this module splits
//! into a dimension-specific normal helper with the rest of EPA shared.

use glam::Vec3;

use super::gjk::{minkowski_support, MinkowskiPoint, SupportFn};

const EPA_MAX_ITERATIONS: u32 = 48;
const EPA_TOLERANCE: f32 = 1e-4;
/// Sanity cap: a well-formed EPA typically finishes with < 30 vertices.
/// If we blow through this we're likely in a degenerate stall.
const EPA_MAX_VERTICES: usize = 96;

/// Output of [`epa`], the resolved contact info, ready to plug into
/// a [`crate::Contact`].
#[derive(Clone, Copy, Debug)]
pub struct ContactInfo {
    /// Unit vector from A toward B in world coordinates. Matches the `Contact::normal`
    /// convention the PGS solver expects.
    pub normal: Vec3,
    /// How far the shapes overlap along `normal`.
    pub penetration: f32,
    /// World-space contact point (midpoint between the surfaces of A and B at the
    /// closest feature).
    pub point: Vec3,
}

/// Triangle face of the expanding polytope, stored by vertex index.
#[derive(Clone, Copy, Debug)]
struct Face {
    /// Indices into `Polytope::vertices`. Winding is kept consistent with `normal`
    /// pointing outward from the origin-side interior.
    v: [usize; 3],
    /// Unit outward normal. `normal * distance` is the closest point on the face's
    /// plane to the origin.
    normal: Vec3,
    /// Distance from origin to the face's plane. Always ≥ 0 by
    /// construction.
    distance: f32,
}

struct Polytope {
    vertices: Vec<MinkowskiPoint>,
    faces: Vec<Face>,
    /// Centroid of the seed tetrahedron. Stays interior to the polytope for all
    /// subsequent (convex) expansions, so it's the reliable reference for orienting new
    /// faces outward, much more robust than "pick any old vertex," which can happen to
    /// lie on a degenerate-face plane and produce an inward orientation that cascades
    /// into a corrupted polytope.
    interior: glam::Vec3,
}

impl Polytope {
    fn from_tetra(tetra: [MinkowskiPoint; 4]) -> Self {
        let vertices = tetra.to_vec();
        let interior = (tetra[0].point + tetra[1].point + tetra[2].point + tetra[3].point) * 0.25;
        // Four triangular faces of the tetrahedron. Winding chosen so
        // that each face's cross-product normal points away from the
        // opposite vertex.
        let mut faces = Vec::with_capacity(4);
        for &(i, j, k, l) in &[(0, 1, 2, 3), (0, 3, 1, 2), (0, 2, 3, 1), (1, 3, 2, 0)] {
            faces.push(build_face_vs_point(&vertices, i, j, k, vertices[l].point));
        }
        Self {
            vertices,
            faces,
            interior,
        }
    }

    /// Index of the face with the smallest distance from origin, or `None` if the
    /// polytope has no faces (should not happen in a well-formed expansion, but a
    /// degenerate support sequence can remove every face without producing any stitch
    /// replacements).
    fn closest_face(&self) -> Option<usize> {
        let (idx, _) = self
            .faces
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.distance.total_cmp(&b.1.distance))?;
        Some(idx)
    }

    /// Add `support` to the polytope: remove all faces whose outward normal is oriented
    /// toward `support` (we can "see" the support from outside those faces), then
    /// rebuild by connecting `support` to every horizon edge.
    fn expand(&mut self, support: MinkowskiPoint) {
        let new_idx = self.vertices.len();
        self.vertices.push(support);

        // Collect horizon edges: edges of removed faces that are *not* shared with any
        // other removed face. Represent edges as sorted vertex-index pairs for
        // deduplication.
        let mut horizon: Vec<(usize, usize)> = Vec::new();
        let mut keep = Vec::with_capacity(self.faces.len());

        for f in self.faces.drain(..) {
            let view = support.point - self.vertices[f.v[0]].point;
            if f.normal.dot(view) > 0.0 {
                // Face faces the new point -> remove, record edges.
                add_or_remove_edge(&mut horizon, f.v[0], f.v[1]);
                add_or_remove_edge(&mut horizon, f.v[1], f.v[2]);
                add_or_remove_edge(&mut horizon, f.v[2], f.v[0]);
            } else {
                keep.push(f);
            }
        }
        self.faces = keep;

        // Stitch new faces from each horizon edge to the new vertex. Orientation uses
        // `self.interior`, the seed tetrahedron's centroid, as a guaranteed interior
        // reference. Using an arbitrary old vertex was the source of the "polytope face
        // count explodes" bug: when an old vertex happens to lie on the plane of a new
        // face, the sign test is ambiguous, the face gets an inward-facing normal,
        // downstream visibility tests lie about it, and faces multiply without bound.
        let interior = self.interior;
        for &(i, j) in &horizon {
            self.faces
                .push(build_face_vs_point(&self.vertices, i, j, new_idx, interior));
        }
    }
}

/// Build a face with outward normal, orienting it away from `interior_point`, which
/// the caller guarantees is inside the polytope (the seed tetrahedron's centroid
/// satisfies this).
fn build_face_vs_point(
    verts: &[MinkowskiPoint],
    a: usize,
    b: usize,
    c: usize,
    interior_point: Vec3,
) -> Face {
    let pa = verts[a].point;
    let pb = verts[b].point;
    let pc = verts[c].point;

    let mut normal = (pb - pa).cross(pc - pa);
    let len = normal.length();
    if len < 1e-8 {
        normal = Vec3::Y;
    } else {
        normal /= len;
    }

    // Orient away from the interior reference point. If the normal
    // currently points *toward* the interior, flip it.
    let to_interior = interior_point - pa;
    let (v_order, outward_normal) = if normal.dot(to_interior) > 0.0 {
        ([a, c, b], -normal)
    } else {
        ([a, b, c], normal)
    };

    // Signed distance from origin to the face's plane along the
    // outward normal. Clamped at zero for numerical noise near the
    // origin-on-boundary case.
    let raw_distance = outward_normal.dot(verts[v_order[0]].point);
    let distance = raw_distance.max(0.0);
    Face {
        v: v_order,
        normal: outward_normal,
        distance,
    }
}

/// Track a single horizon edge with its original winding direction. Two removed faces
/// that share an edge store it with opposite orientation (because the adjacent faces
/// wind in opposite directions along their shared edge); we detect that by looking for
/// `(b, a)` already in the list and cancelling it. The surviving edges are the horizon
/// in the winding direction that gives correct outward normals when stitched to the
/// new vertex.
fn add_or_remove_edge(horizon: &mut Vec<(usize, usize)>, a: usize, b: usize) {
    if let Some(pos) = horizon.iter().position(|&e| e == (b, a)) {
        horizon.swap_remove(pos);
    } else {
        horizon.push((a, b));
    }
}

/// Barycentric coordinates (u, v, w) of point `p` projected onto the triangle
/// `(a, b, c)`. Returns the triple such that `u·a + v·b + w·c` is the projection.
fn barycentric(a: Vec3, b: Vec3, c: Vec3, p: Vec3) -> (f32, f32, f32) {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-12 {
        return (1.0, 0.0, 0.0);
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    (u, v, w)
}

/// Compute penetration normal, depth, and contact point for two overlapping shapes,
/// given GJK's terminating tetrahedron simplex.
pub fn epa<A: SupportFn, B: SupportFn>(
    a: &A,
    b: &B,
    initial_simplex: [MinkowskiPoint; 4],
) -> Option<ContactInfo> {
    // Reject degenerate starting simplices: if the 4 GJK points are (nearly) coplanar,
    // the tetrahedron has ~zero volume and EPA cannot produce meaningful outward
    // normals. The signed volume is det([p1-p0, p2-p0, p3-p0])/6, and sign flips
    // depending on handedness, we only care about magnitude.
    let p0 = initial_simplex[0].point;
    let p1 = initial_simplex[1].point;
    let p2 = initial_simplex[2].point;
    let p3 = initial_simplex[3].point;
    let volume6 = (p1 - p0).dot((p2 - p0).cross(p3 - p0)).abs();
    if volume6 < 1e-8 {
        return None;
    }

    let mut polytope = Polytope::from_tetra(initial_simplex);

    for _ in 0..EPA_MAX_ITERATIONS {
        // If expansion has collapsed the polytope (no faces), we've left the domain
        // where EPA can give a meaningful answer. Bail cleanly rather than panicking
        // on the empty slice.
        let face_idx = polytope.closest_face()?;
        let face = polytope.faces[face_idx];

        let support = minkowski_support(a, b, face.normal);
        let new_distance = support.point.dot(face.normal);

        // Guard: if the support query produced non-finite values, the Minkowski diff
        // has pathological inputs, bail rather than feed bad data back into `expand()`
        // and grow the polytope with NaN vertices.
        if !new_distance.is_finite() || !support.point.is_finite() {
            return None;
        }

        if (new_distance - face.distance).abs() < EPA_TOLERANCE {
            // Face is on the Minkowski boundary; terminate.
            return contact_from_face(&polytope, face);
        }

        polytope.expand(support);

        if polytope.vertices.len() > EPA_MAX_VERTICES {
            break;
        }
    }

    // Iteration cap reached; return the current best estimate rather than failing
    // outright. Happens only for nearly-degenerate inputs. Emit a debug-level trace so
    // developers tuning narrowphase can count cap hits without spamming release logs.
    tracing::debug!(
        max_iterations = EPA_MAX_ITERATIONS,
        vertices = polytope.vertices.len(),
        "EPA 3D hit iteration cap; returning best-estimate contact",
    );
    let face_idx = polytope.closest_face()?;
    contact_from_face(&polytope, polytope.faces[face_idx])
}

fn contact_from_face(polytope: &Polytope, face: Face) -> Option<ContactInfo> {
    let v0 = polytope.vertices[face.v[0]];
    let v1 = polytope.vertices[face.v[1]];
    let v2 = polytope.vertices[face.v[2]];

    // Closest point on the face to the origin, in Minkowski-diff space.
    let closest = face.normal * face.distance;

    // Barycentric coords of `closest` on triangle (v0, v1, v2).
    let (u, v, w) = barycentric(v0.point, v1.point, v2.point, closest);

    // Reconstruct the matching points on A and B using the same
    // weights applied to the cached original supports.
    let point_a = v0.sa * u + v1.sa * v + v2.sa * w;
    let point_b = v0.sb * u + v1.sb * v + v2.sb * w;

    Some(ContactInfo {
        normal: face.normal,
        penetration: face.distance,
        point: (point_a + point_b) * 0.5,
    })
}

#[cfg(test)]
mod tests {
    use super::super::gjk::{gjk_intersect, ConvexHull, GjkResult, Sphere};
    use super::*;

    fn box_vertices(center: Vec3, half: Vec3) -> Vec<Vec3> {
        vec![
            center + Vec3::new(-half.x, -half.y, -half.z),
            center + Vec3::new(half.x, -half.y, -half.z),
            center + Vec3::new(half.x, half.y, -half.z),
            center + Vec3::new(-half.x, half.y, -half.z),
            center + Vec3::new(-half.x, -half.y, half.z),
            center + Vec3::new(half.x, -half.y, half.z),
            center + Vec3::new(half.x, half.y, half.z),
            center + Vec3::new(-half.x, half.y, half.z),
        ]
    }

    fn run(a: &impl SupportFn, b: &impl SupportFn, d: Vec3) -> ContactInfo {
        match gjk_intersect(a, b, d) {
            GjkResult::Intersecting { simplex } => epa(a, b, simplex).expect("EPA should converge"),
            GjkResult::Separated => panic!("GJK says separated, EPA can't run"),
        }
    }

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol,
            "expected {a} close to {b} (tol {tol})"
        );
    }

    #[test]
    fn sphere_sphere_penetration_matches_distance() {
        // Two unit-radius spheres with centres 1.5 apart overlap by 0.5.
        let a = Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        };
        let b = Sphere {
            center: Vec3::new(1.5, 0.0, 0.0),
            radius: 1.0,
        };
        let info = run(&a, &b, Vec3::new(1.5, 0.0, 0.0));
        assert_close(info.penetration, 0.5, 1e-3);
        assert!(info.normal.dot(Vec3::X) > 0.99, "normal: {:?}", info.normal);
    }

    #[test]
    fn box_box_axis_aligned_overlap_penetration_matches_axis() {
        // Unit boxes offset by 1.5 along X -> 0.5 overlap along +X.
        let va = box_vertices(Vec3::ZERO, Vec3::ONE);
        let vb = box_vertices(Vec3::new(1.5, 0.0, 0.0), Vec3::ONE);
        let a = ConvexHull { vertices: &va };
        let b = ConvexHull { vertices: &vb };

        let info = run(&a, &b, Vec3::new(1.5, 0.0, 0.0));
        assert!(
            info.normal.dot(Vec3::X).abs() > 0.99,
            "normal: {:?}",
            info.normal
        );
        assert!(
            info.normal.dot(Vec3::X) > 0.0,
            "normal not A->B: {:?}",
            info.normal
        );
        assert_close(info.penetration, 0.5, 1e-3);
    }

    #[test]
    fn sphere_box_corner_penetration_points_outward() {
        // Sphere at (1.2, 1.2, 1.2) with r=0.5 penetrates the corner
        // (1, 1, 1) of a unit box. Distance from corner to centre =
        // sqrt(3·0.04) ≈ 0.346. Penetration = 0.5 − 0.346 ≈ 0.154.
        // Contact normal should point along +(1,1,1)/√3 (from box
        // corner out toward sphere, i.e. from A to B).
        let vb = box_vertices(Vec3::ZERO, Vec3::ONE);
        let b = ConvexHull { vertices: &vb };
        let s = Sphere {
            center: Vec3::new(1.2, 1.2, 1.2),
            radius: 0.5,
        };

        // Put the box as A and the sphere as B so normal A->B points
        // toward (1, 1, 1)/√3.
        let info = run(&b, &s, Vec3::new(1.0, 1.0, 1.0));
        let expected = Vec3::new(1.0, 1.0, 1.0).normalize();
        assert!(
            info.normal.dot(expected) > 0.95,
            "normal {:?} not aligned with {:?}",
            info.normal,
            expected
        );
        assert_close(info.penetration, 0.5 - 3.0_f32.sqrt() * 0.2, 1e-2);
    }

    #[test]
    fn deeply_nested_boxes_report_positive_penetration() {
        let va = box_vertices(Vec3::ZERO, Vec3::ONE);
        let vb = box_vertices(Vec3::new(0.3, 0.1, 0.2), Vec3::ONE);
        let a = ConvexHull { vertices: &va };
        let b = ConvexHull { vertices: &vb };
        let info = run(&a, &b, Vec3::new(0.3, 0.1, 0.2));
        assert!(info.penetration > 0.0, "penetration: {}", info.penetration);
        assert!(info.penetration.is_finite());
    }
}
