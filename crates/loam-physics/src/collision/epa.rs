//! EPA: Expanding Polytope Algorithm (3D).
//!
//! Given GJK's terminating tetrahedron, EPA grows it into a polytope whose surface
//! approaches the boundary of `A ⊖ B`. The closest surface point to the origin is
//! the minimum translation: contact normal and penetration depth. The contact
//! point comes from the closest point's barycentric weights applied to the cached
//! per-shape supports in each [`MinkowskiPoint`].
//!
//! Each iteration: find the closest face, query a support along its outward normal,
//! terminate if that support's distance matches the face's, else add the support
//! (remove faces it sees, stitch new triangles from each horizon edge to it).

use glam::Vec3;

use super::gjk::{minkowski_support, MinkowskiPoint, SupportFn};

const EPA_MAX_ITERATIONS: u32 = 48;
const EPA_TOLERANCE: f32 = 1e-4;
/// Sanity cap: a well-formed EPA finishes with < 30 vertices; past this we are in
/// a degenerate stall.
const EPA_MAX_VERTICES: usize = 96;

/// Resolved contact info for a [`crate::Contact`].
#[derive(Clone, Copy, Debug)]
pub struct ContactInfo {
    /// Unit vector from A toward B (world), per the `Contact::normal` convention.
    pub normal: Vec3,
    /// Overlap depth along `normal`.
    pub penetration: f32,
    /// World-space contact point: midpoint of the closest features on A and B.
    pub point: Vec3,
}

/// Triangle face of the expanding polytope, stored by vertex index.
#[derive(Clone, Copy, Debug)]
struct Face {
    /// Indices into `Polytope::vertices`, wound so `normal` points outward.
    v: [usize; 3],
    /// Unit outward normal; `normal * distance` is the plane's closest point to
    /// the origin.
    normal: Vec3,
    /// Distance from origin to the face plane, always ≥ 0.
    distance: f32,
}

struct Polytope {
    vertices: Vec<MinkowskiPoint>,
    faces: Vec<Face>,
    /// Seed-tetrahedron centroid; stays interior under all convex expansions, so
    /// it orients new faces outward. An arbitrary old vertex can sit on a new
    /// face's plane and flip the orientation, corrupting the polytope.
    interior: glam::Vec3,
}

impl Polytope {
    fn from_tetra(tetra: [MinkowskiPoint; 4]) -> Self {
        let vertices = tetra.to_vec();
        let interior = (tetra[0].point + tetra[1].point + tetra[2].point + tetra[3].point) * 0.25;
        // Four faces, each wound so its normal points away from the opposite vertex.
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

    /// Closest face to the origin, or `None` if a degenerate expansion removed
    /// every face without restitching.
    fn closest_face(&self) -> Option<usize> {
        let (idx, _) = self
            .faces
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.distance.total_cmp(&b.1.distance))?;
        Some(idx)
    }

    /// Add `support`: remove every face whose outward normal faces it, then connect
    /// `support` to each horizon edge.
    fn expand(&mut self, support: MinkowskiPoint) {
        let new_idx = self.vertices.len();
        self.vertices.push(support);

        // Horizon = edges of removed faces not shared with another removed face.
        let mut horizon: Vec<(usize, usize)> = Vec::new();
        let mut keep = Vec::with_capacity(self.faces.len());

        for f in self.faces.drain(..) {
            let view = support.point - self.vertices[f.v[0]].point;
            if f.normal.dot(view) > 0.0 {
                add_or_remove_edge(&mut horizon, f.v[0], f.v[1]);
                add_or_remove_edge(&mut horizon, f.v[1], f.v[2]);
                add_or_remove_edge(&mut horizon, f.v[2], f.v[0]);
            } else {
                keep.push(f);
            }
        }
        self.faces = keep;

        // Stitch each horizon edge to the new vertex, oriented against the seed
        // centroid (a guaranteed interior reference; see the `interior` field).
        let interior = self.interior;
        for &(i, j) in &horizon {
            self.faces
                .push(build_face_vs_point(&self.vertices, i, j, new_idx, interior));
        }
    }
}

/// Build a face with outward normal, oriented away from `interior_point` (the
/// caller guarantees it is inside the polytope).
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

    // Flip if the normal points toward the interior reference.
    let to_interior = interior_point - pa;
    let (v_order, outward_normal) = if normal.dot(to_interior) > 0.0 {
        ([a, c, b], -normal)
    } else {
        ([a, b, c], normal)
    };

    // Clamp at zero for noise near the origin-on-boundary case.
    let raw_distance = outward_normal.dot(verts[v_order[0]].point);
    let distance = raw_distance.max(0.0);
    Face {
        v: v_order,
        normal: outward_normal,
        distance,
    }
}

/// Track a horizon edge by winding. Two removed faces sharing an edge store it as
/// `(a, b)` and `(b, a)`; finding the reverse cancels both. Survivors keep the
/// winding that gives correct outward normals when stitched to the new vertex.
fn add_or_remove_edge(horizon: &mut Vec<(usize, usize)>, a: usize, b: usize) {
    if let Some(pos) = horizon.iter().position(|&e| e == (b, a)) {
        horizon.swap_remove(pos);
    } else {
        horizon.push((a, b));
    }
}

/// Barycentric coords `(u, v, w)` of `p` projected onto triangle `(a, b, c)`, with
/// `u·a + v·b + w·c` the projection.
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

/// Penetration normal, depth, and contact point for two overlapping shapes, given
/// GJK's terminating tetrahedron.
pub fn epa<A: SupportFn, B: SupportFn>(
    a: &A,
    b: &B,
    initial_simplex: [MinkowskiPoint; 4],
) -> Option<ContactInfo> {
    // Reject a near-coplanar (zero-volume) seed: |det([p1-p0, p2-p0, p3-p0])|.
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
        // A collapsed polytope (no faces) is outside EPA's domain; bail cleanly.
        let face_idx = polytope.closest_face()?;
        let face = polytope.faces[face_idx];

        let support = minkowski_support(a, b, face.normal);
        let new_distance = support.point.dot(face.normal);

        // Bail on non-finite support rather than grow the polytope with NaN.
        if !new_distance.is_finite() || !support.point.is_finite() {
            return None;
        }

        if (new_distance - face.distance).abs() < EPA_TOLERANCE {
            // Face is on the Minkowski boundary.
            return contact_from_face(&polytope, face);
        }

        polytope.expand(support);

        if polytope.vertices.len() > EPA_MAX_VERTICES {
            break;
        }
    }

    // Cap hit (near-degenerate inputs only): return the best estimate.
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

    let (u, v, w) = barycentric(v0.point, v1.point, v2.point, closest);

    // Same weights against the cached supports recover the points on A and B.
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
        // Sphere at (1.2,1.2,1.2), r=0.5 vs box corner (1,1,1): corner-centre
        // distance √(3·0.04) ≈ 0.346, penetration ≈ 0.154, normal along +(1,1,1)/√3.
        let vb = box_vertices(Vec3::ZERO, Vec3::ONE);
        let b = ConvexHull { vertices: &vb };
        let s = Sphere {
            center: Vec3::new(1.2, 1.2, 1.2),
            radius: 0.5,
        };

        // Box as A, sphere as B so normal A->B points toward (1,1,1)/√3.
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
