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
use super::simplex_r4::project_origin_onto_affine_hull;

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
    /// Centroid of the seed 5-simplex, and the sole orientation reference for
    /// every face: expansion only adds vertices, so it stays interior.
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

    /// Face whose plane is nearest the origin, as in the 3D
    /// [`super::epa`][mod@super::epa].
    ///
    /// Distance-0 faces are the rule rather than the exception in 4D:
    /// [`gjk_intersect_r4`][super::gjk_r4::gjk_intersect_r4] grows its
    /// terminating sub-simplex to five vertices by adding supports on one side
    /// of it, so whenever that sub-simplex has fewer than five vertices the
    /// origin ends up on a proper face of the seed and every seed face through
    /// it starts at distance 0. Those are the faces EPA has to expand across,
    /// so they compete for the minimum on equal terms. Skipping them starts the
    /// expansion on the far side of the difference body, where it converges to
    /// a supporting face that is not the minimum: a normal that points through
    /// the obstacle instead of out of it, at whatever depth that far facet
    /// sits. Expanding across them terminates because [`FACE_COPLANAR_EPS`]
    /// retires the coplanar tiles they belong to.
    fn closest_face(&self) -> Option<usize> {
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

/// Band around a face plane in which a support point counts as lying on it, so
/// that `expand` retires the face instead of keeping it.
///
/// A facet of a Minkowski difference of polytopes is generally not a simplex
/// (the difference body of a 4-simplex has prism facets), so it is tiled by
/// several coplanar tetrahedral faces and support points land exactly on their
/// shared plane. Keeping those tiles while their neighbours are retired stitches
/// the new vertex into a facet that is already covered: the surface stops being
/// convex, later iterations chase interior faces, and EPA returns a normal that
/// can point from B toward A. The plane residual of a unit-scale 4-term dot
/// product is a few ULPs (~4e-7, and 1e-7 measurably misses coplanar tiles),
/// while the gap that is the precondition for calling `expand` at all is
/// `EPA_TOLERANCE`, two orders up.
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
/// Orientation comes from the seed centroid alone. The origin is the tempting
/// second reference and is wrong: it is interior only while GJK's containment
/// verdict holds exactly, and GJK accepts a simplex whose closest point is
/// within its own tolerance of the origin, so on a near-tangency the origin can
/// sit outside the seed. Orienting one face against it there inverts that face
/// and EPA converges on a plane that is not the boundary at all.
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

    // Interior on the `+normal` side means `+normal` points inward.
    let flip = normal.dot(centroid - pa) > 0.0;

    let (outward, v_order) = if flip {
        (-normal, [a, b, d, c])
    } else {
        (normal, [a, b, c, d])
    };

    // Clamp at 0: on a near-tangency the origin can sit a hair outside the
    // face, and a negative offset is not a penetration depth.
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
    let tetra = face.v.map(|i| polytope.vertices[i]);

    // Closest point on the face hyperplane to the origin, in Minkowski-diff space.
    let closest = face.normal * face.distance;
    let weights = face_barycentrics(&tetra.map(|p| p.point), closest);

    let mut point_a = Vec4::ZERO;
    let mut point_b = Vec4::ZERO;
    for (vertex, w) in tetra.iter().zip(weights) {
        point_a += vertex.sa * w;
        point_b += vertex.sb * w;
    }

    Some(ContactInfo4 {
        normal: face.normal,
        penetration: face.distance,
        point: (point_a + point_b) * 0.5,
    })
}

/// Barycentric weights of `closest` on all four vertices of the face tetra.
///
/// The face plane is the tetra's affine hull, so `closest` lies in it by
/// construction and the affine solve reproduces it: `Σ wᵢ·vᵢ = closest`, which
/// is what makes the reconstructed witnesses satisfy
/// `point_a − point_b = normal·penetration`. Weights go negative when the
/// projection leaves the tetra, which happens whenever a Minkowski-difference
/// facet is tiled by several coplanar faces (see [`FACE_COPLANAR_EPS`]) and the
/// projection lands on a neighbouring tile. Clamping to the nearest point of
/// this tile instead, as `simplex_r4::closest_to_origin` does, drops vertices
/// from the combination and breaks the identity.
///
/// A negative weight therefore makes `point_a` and `point_b` affine
/// extrapolations: each leaves the convex hull of its own shape's support
/// points, so neither witness is guaranteed to lie on its surface, and their
/// midpoint can sit outside both bodies. That is deliberate. The contact point
/// feeds a lever arm and a manifold merge radius, both of which want the
/// combination that reproduces `closest`; a witness clamped back onto the hull
/// would name a surface point whose difference is no longer the penetration
/// vector.
///
/// The tetra centroid is the fallback for a singular Gram system. `build_face`
/// floors the tetra's 3-volume, so reaching it takes a face anisotropic enough
/// to lose a pivot in f32, where no decomposition of the plane point would be
/// trustworthy.
fn face_barycentrics(points: &[Vec4; 4], closest: Vec4) -> [f32; 4] {
    let shifted = points.map(|p| p - closest);
    match project_origin_onto_affine_hull(&[0, 1, 2, 3], &shifted) {
        Some((_, w)) => [w[0], w[1], w[2], w[3]],
        None => [0.25; 4],
    }
}

#[cfg(test)]
mod tests {
    use super::super::gjk_r4::{gjk_intersect_r4, ConvexHull4, GjkResult4, Sphere4};
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

    /// A face whose plane projection falls outside its own tetra. The contact
    /// must be the affine combination that realizes the projection over all
    /// four vertices, so the witnesses keep
    /// `point_a − point_b = normal·penetration`; clamping to the nearest
    /// sub-simplex rebuilds the contact from three vertices and breaks that
    /// identity.
    ///
    /// The four weights are pairwise distinct and non-zero, so each vertex is
    /// separately observable: transposing any two of them permutes the solved
    /// weights and moves the contact, which a fixture with a repeated or zero
    /// weight would hide.
    #[test]
    fn contact_from_face_realizes_the_plane_projection_outside_the_tetra() {
        use super::super::simplex_r4::closest_to_origin;

        // Tetra in the hyperplane x = 1, so the plane projection is x̂; the
        // affine coordinates below are not all non-negative, which puts x̂
        // outside the tetra.
        let points = [
            Vec4::new(1.0, 1.0, 0.0, 0.0),
            Vec4::new(1.0, 0.0, 1.0, 0.0),
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            Vec4::new(1.0, 6.0, -3.0, 2.0),
        ];
        let pre_images_b = [
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
            Vec4::new(1.0, 0.0, 0.0, 0.0),
        ];
        let vertices: Vec<MinkowskiPoint4> = points
            .iter()
            .zip(pre_images_b)
            .map(|(&point, sb)| MinkowskiPoint4 {
                point,
                sa: point + sb,
                sb,
            })
            .collect();

        // Fixture premise: the convex solve clamps to the triangle v₀v₁v₂ and
        // drops v₃ from the combination.
        let clamped = closest_to_origin(&points.map(|p| p - Vec4::X));
        assert_eq!(clamped.kept, vec![0, 1, 2]);

        // Affine coordinates of x̂ on the tetra. In the yzw slice the first
        // three vertices are the standard basis and the fourth is
        // q₃ = (6, −3, 2), so Σ wᵢ·qᵢ = 0 forces (w₀, w₁, w₂) = −w₃·q₃, and
        // Σ wᵢ = 1 then gives −4·w₃ = 1.
        let weights = [1.5, -0.75, 0.5, -0.25];
        let realized = points
            .iter()
            .zip(weights)
            .fold(Vec4::ZERO, |acc, (&p, w)| acc + p * w);
        assert!((realized - Vec4::X).length() < 1e-6, "{realized:?}");

        let solved = face_barycentrics(&points, Vec4::X);
        assert!(
            solved
                .iter()
                .zip(weights)
                .all(|(&got, want)| (got - want).abs() < 1e-5),
            "weights {solved:?} should be {weights:?}"
        );

        let centroid = (points[0] + points[1] + points[2] + points[3]) * 0.25;
        let face = build_face(&vertices, 0, 1, 2, 3, centroid).expect("tetra is non-degenerate");
        assert_close(face.distance, 1.0, 1e-6);

        let polytope = Polytope4 {
            vertices,
            faces: vec![face],
            centroid,
        };
        let contact = contact_from_face(&polytope, face).expect("face resolves a contact");

        // Σ wᵢ·saᵢ = (0.75, 1.5, −0.75, 0.5) and Σ wᵢ·sbᵢ = (−0.25, 1.5,
        // −0.75, 0.5): the witnesses differ by x̂ = normal·penetration, and the
        // contact is their midpoint.
        let expected = Vec4::new(0.25, 1.5, -0.75, 0.5);
        assert!(
            (contact.point - expected).length() < 1e-5,
            "contact {:?} should be {expected:?}",
            contact.point
        );
    }

    /// Pre-images are irrelevant to a seed that never reaches contact
    /// reconstruction, so they are the difference points themselves.
    fn seed(points: [Vec4; 5]) -> [MinkowskiPoint4; 5] {
        points.map(|point| MinkowskiPoint4 {
            point,
            sa: point,
            sb: Vec4::ZERO,
        })
    }

    /// Base tetrahedron around the origin in `w = 0` plus an apex at height
    /// `h`, giving `|det| = 8·h` against the seed's 1e-8 volume floor: `h = 0`
    /// is the flat seed, everything above it is admissible.
    fn seed_of_height(h: f32) -> [MinkowskiPoint4; 5] {
        seed([
            Vec4::new(-1.0, -1.0, -1.0, 0.0),
            Vec4::new(1.0, -1.0, -1.0, 0.0),
            Vec4::new(0.0, 1.0, -1.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, h),
        ])
    }

    /// The 4D twin of the 3D floor. A seed confined to a hyperplane has no
    /// interior for `build_face` to orient against, and the Hodge dual of
    /// three dependent edges normalizes to NaN, which reaches the solver as a
    /// contact. One rung above the floor the answer has to be a number.
    #[test]
    fn seeds_across_the_volume_floor_resolve_finitely_or_not_at_all() {
        let a = Sphere4 {
            center: Vec4::ZERO,
            radius: 1.0,
        };
        let b = Sphere4 {
            center: Vec4::new(0.5, 0.0, 0.0, 0.0),
            radius: 1.0,
        };

        assert!(
            epa_r4(&a, &b, seed_of_height(0.0)).is_none(),
            "a seed inside a hyperplane has no interior to orient against"
        );
        for h in [1e-6, 1e-4, 1e-2, 1.0] {
            let contact = epa_r4(&a, &b, seed_of_height(h))
                .unwrap_or_else(|| panic!("seed of height {h} clears the volume floor"));
            assert!(
                contact.normal.is_finite()
                    && contact.point.is_finite()
                    && contact.penetration.is_finite(),
                "seed of height {h} resolved to {contact:?}"
            );
            assert!(
                contact.penetration >= 0.0,
                "seed of height {h}: negative depth"
            );
        }
    }

    /// The rest of the degeneracy ladder: a seed collapsed onto a line, and one
    /// with a repeated vertex. Both have zero 4-volume by a different route
    /// than lying in a hyperplane, and both take the same exit.
    #[test]
    fn collinear_and_repeated_vertex_seeds_are_rejected_rather_than_resolved() {
        let a = Sphere4 {
            center: Vec4::ZERO,
            radius: 1.0,
        };
        let b = Sphere4 {
            center: Vec4::new(0.5, 0.0, 0.0, 0.0),
            radius: 1.0,
        };
        let collinear = seed([
            Vec4::new(-1.0, 0.0, 0.0, 0.0),
            Vec4::new(-0.5, 0.0, 0.0, 0.0),
            Vec4::ZERO,
            Vec4::new(0.5, 0.0, 0.0, 0.0),
            Vec4::new(1.0, 0.0, 0.0, 0.0),
        ]);
        let mut repeated = seed_of_height(1.0);
        repeated[2] = repeated[1];
        assert!(epa_r4(&a, &b, collinear).is_none());
        assert!(epa_r4(&a, &b, repeated).is_none());
    }

    /// 16 corners of an axis-aligned R⁴ box centred at the origin.
    fn box4_vertices(half: Vec4) -> Vec<Vec4> {
        let mut vertices = Vec::with_capacity(16);
        for &x in &[-half.x, half.x] {
            for &y in &[-half.y, half.y] {
                for &z in &[-half.z, half.z] {
                    for &w in &[-half.w, half.w] {
                        vertices.push(Vec4::new(x, y, z, w));
                    }
                }
            }
        }
        vertices
    }

    const WALL_HALF: f32 = 0.05;
    const BALL_RADIUS: f32 = 0.1;
    /// Distance from the wall's midplane at which the ball stops touching it,
    /// and so the half-width of the interval the sweep below covers.
    const CAPTURE: f32 = WALL_HALF + BALL_RADIUS;

    fn ball_vs_wall(x: f32) -> ContactInfo4 {
        let vertices = box4_vertices(Vec4::new(WALL_HALF, 2.0, 2.0, 2.0));
        let wall = ConvexHull4 {
            vertices: &vertices,
        };
        let ball = Sphere4 {
            center: Vec4::new(x, 0.0, 0.0, 0.0),
            radius: BALL_RADIUS,
        };
        let simplex = match gjk_intersect_r4(&ball, &wall, -ball.center) {
            GjkResult4::Intersecting { simplex } => simplex,
            GjkResult4::Separated => panic!("ball at {x} overlaps the wall"),
        };
        epa_r4(&ball, &wall, simplex).expect("EPA should resolve an overlap")
    }

    /// A ball inside a wall thinner than itself overlaps both faces at once, so
    /// the wrong exit is always available, and a contact resolved against it
    /// drives the ball through instead of back. `Contact::normal` runs A -> B
    /// and the solver drives A along `−normal`, so `−normal` has to be the way
    /// back out of the face the ball entered.
    ///
    /// The depth is closed form: the difference body is the box of half extents
    /// `WALL_HALF + BALL_RADIUS` on the launch axis rounded by the radius, so on
    /// that axis the boundary is `CAPTURE − |x|` away.
    ///
    /// Swept rather than sampled because the failure this pins occupied
    /// isolated depths, roughly one sample in twenty, rather than a band.
    #[test]
    fn wall_contact_leaves_through_the_face_the_ball_entered() {
        for side in [-1.0_f32, 1.0] {
            for k in 1..(CAPTURE / 5e-4) as u32 {
                let x = side * (CAPTURE - 5e-4 * k as f32);
                let contact = ball_vs_wall(x);
                assert!(
                    contact.normal.x * side < -0.99,
                    "ball at {x} leaves along {:?}, not back out of its own face",
                    -contact.normal
                );
                assert_close(contact.penetration, CAPTURE - x.abs(), EPA_TOLERANCE);
            }
        }
    }

    /// The contact boundary. A ball exactly `CAPTURE` out is touching, not
    /// overlapping, so the depth is zero there and grows only as it comes in.
    /// A non-zero depth at a non-negative gap is a phantom contact the solver
    /// cannot tell from a real one, and it appears the moment face orientation
    /// takes the origin for an interior point: GJK's containment verdict has a
    /// tolerance, so on a near-tangency the origin sits outside the seed.
    #[test]
    fn wall_contact_depth_stays_zero_up_to_exact_touching() {
        for gap in [1e-3_f32, 1e-4, 1e-5, 0.0] {
            let contact = ball_vs_wall(-(CAPTURE + gap));
            assert_eq!(
                contact.penetration, 0.0,
                "a ball {gap} clear of the wall is not {} deep in it",
                contact.penetration
            );
        }
        for overlap in [1e-5_f32, 1e-4, 1e-3, 1e-2] {
            let contact = ball_vs_wall(-(CAPTURE - overlap));
            assert_close(contact.penetration, overlap, EPA_TOLERANCE);
            assert!(
                contact.normal.x > 0.99,
                "an overlap of {overlap} leaves along {:?}",
                -contact.normal
            );
        }
    }

    // The polytope fixtures below all take `B = A + t`, so the Minkowski
    // difference is `K − t` with `K = A ⊕ (−A)` the difference body. For an
    // interior origin the depth is `min_j (h_K(u_j) − ⟨u_j, t⟩)` over K's
    // facet normals `u_j`, and the minimizing `u_j` is the contact normal.
    // Support functions add over Minkowski sums (Schneider 2014, *Convex
    // Bodies: The Brunn-Minkowski Theory*, §1.7) and the normal fan of a sum
    // is the common refinement of the summands' fans (Ziegler 1995, *Lectures
    // on Polytopes*, §7.1), which is what gives each K below its facet list.
    // Every fixture but the pentatope uses a centrally symmetric polytope, so
    // there `−A = A`, `K = 2A`, and `h_K = 2·h_A` on the facet normals of A.
    // Pinning both the depth and the normal is what makes these regressions on
    // `FACE_COPLANAR_EPS`: drop it to 0 and the pentatope, 24-cell and
    // 600-cell all resolve against a face the expansion should have retired.

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

    /// 24-cell vs 24-cell, K's 24 octahedral facets each tiled by several
    /// coplanar EPA faces. The 24-cell is self-dual (Wikipedia "24-cell"), so
    /// its facet normals are the dual's vertex directions, the 8 axes `±ê_i`
    /// and the 16 patterns `(±1,±1,±1,±1)/2`. Both families
    /// support the vertex set `perms(±r/√2, ±r/√2, 0, 0)` at `h_A = r/√2`, so
    /// `h_K = √2·r`. For `t = d·x̂` only `x̂` attains `⟨u_j, t⟩ = d`, the sign
    /// patterns reaching `d/2`, so the minimizer is unique: depth `√2·r − d`
    /// with the normal exactly `x̂`.
    #[test]
    fn cell24_cell24_contact_matches_axis_facet() {
        use crate::collision::gjk_r4::ConvexHull4;
        use crate::euclidean_r4::cell24_vertices;

        let va: Vec<Vec4> = cell24_vertices(1.0);
        let vb: Vec<Vec4> = cell24_vertices(1.0)
            .into_iter()
            .map(|v| v + Vec4::new(0.3, 0.0, 0.0, 0.0))
            .collect();

        let a = ConvexHull4 { vertices: &va };
        let b = ConvexHull4 { vertices: &vb };
        let simplex = match gjk_intersect_r4(&a, &b, Vec4::X) {
            GjkResult4::Intersecting { simplex } => simplex,
            _ => panic!("24-cells should overlap"),
        };
        let contact = epa_r4(&a, &b, simplex).expect("EPA should succeed");

        assert_close(contact.penetration, 2.0_f32.sqrt() - 0.3, EPA_TOLERANCE);
        assert!(
            contact.normal.dot(Vec4::X) > 0.999,
            "normal must run from A toward B along +x, got {:?}",
            contact.normal
        );
    }

    /// 600-cell vs 600-cell, the most facets in the suite: K carries 600
    /// simplicial ones. In `cell600_vertices` at circumradius 1 the nearest
    /// neighbours sit at `⟨v_i, v_j⟩ = 1 − 1/(2φ²) = φ/2` (edge `1/φ`), so a
    /// cell's four mutually adjacent vertices have centroid norm
    /// `|c|² = (4 + 12·φ/2)/16 = φ⁴/8`, the inradius `r_in = φ²/(2√2)`.
    /// `x̂` is not a facet normal here: it is maximized by the single vertex
    /// `(1, 0, 0, 0)`. The 20 cells meeting that vertex are the closest
    /// facets to it, each with `c_x = (1 + 3·φ/2)/4 = φ⁴/8`, hence a unit
    /// normal with x-component `c_x / r_in = r_in`, the largest over all 600
    /// facets. Depth is `2·r_in − d·r_in`; the 20-fold tie pins the normal
    /// only through that x-component.
    #[test]
    fn cell600_cell600_contact_matches_vertex_incident_facet() {
        use crate::collision::gjk_r4::ConvexHull4;
        use crate::euclidean_r4::cell600_vertices;

        let va: Vec<Vec4> = cell600_vertices(1.0);
        let vb: Vec<Vec4> = cell600_vertices(1.0)
            .into_iter()
            .map(|v| v + Vec4::new(0.3, 0.0, 0.0, 0.0))
            .collect();

        let a = ConvexHull4 { vertices: &va };
        let b = ConvexHull4 { vertices: &vb };
        let simplex = match gjk_intersect_r4(&a, &b, Vec4::X) {
            GjkResult4::Intersecting { simplex } => simplex,
            _ => panic!("600-cells should overlap"),
        };
        let contact = epa_r4(&a, &b, simplex).expect("EPA should succeed");

        let phi = (1.0 + 5.0_f32.sqrt()) * 0.5;
        let inradius = phi * phi / (2.0 * 2.0_f32.sqrt());
        assert_close(contact.penetration, inradius * (2.0 - 0.3), EPA_TOLERANCE);
        assert_close(contact.normal.x, inradius, EPA_TOLERANCE);
    }

    /// 120-cell vs 120-cell, 600 vertices: the largest support set in the
    /// suite. Facet normals are the dual 600-cell's vertex directions
    /// (Wikipedia "120-cell"), which include the axes: in
    /// `cell120_vertices` the largest coordinate is `φ²` before the
    /// `1/(2√2)` rescale, and the 20 vertices attaining `x = φ²/(2√2)` are one
    /// dodecahedral cell, so `h_A(x̂) = φ²/(2√2)`. Being a facet normal, `x̂`
    /// maximizes `⟨u_j, t⟩` for `t = d·x̂` outright: depth `φ²/√2 − d`, normal
    /// exactly `x̂`, no tie.
    #[test]
    fn cell120_cell120_contact_matches_dodecahedral_facet() {
        use crate::collision::gjk_r4::ConvexHull4;
        use crate::euclidean_r4::cell120_vertices;

        let va: Vec<Vec4> = cell120_vertices(1.0);
        let vb: Vec<Vec4> = cell120_vertices(1.0)
            .into_iter()
            .map(|v| v + Vec4::new(0.3, 0.0, 0.0, 0.0))
            .collect();

        let a = ConvexHull4 { vertices: &va };
        let b = ConvexHull4 { vertices: &vb };
        let simplex = match gjk_intersect_r4(&a, &b, Vec4::X) {
            GjkResult4::Intersecting { simplex } => simplex,
            _ => panic!("120-cells should overlap"),
        };
        let contact = epa_r4(&a, &b, simplex).expect("EPA should succeed");

        let phi = (1.0 + 5.0_f32.sqrt()) * 0.5;
        assert_close(
            contact.penetration,
            phi * phi / 2.0_f32.sqrt() - 0.3,
            EPA_TOLERANCE,
        );
        assert!(
            contact.normal.dot(Vec4::X) > 0.999,
            "normal must run from A toward B along +x, got {:?}",
            contact.normal
        );
    }
}
