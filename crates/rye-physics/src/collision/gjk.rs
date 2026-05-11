//! GJK containment test for convex shapes.
//!
//! The algorithm, briefly:
//! - The *Minkowski difference* `A ⊖ B = { a − b : a ∈ A, b ∈ B }` contains the origin if and
//!   only if `A ∩ B` is non-empty. GJK tests exactly this origin-containment property.
//! - A *support function* `s(d)` returns the point of the shape farthest along direction `d`.
//!   The Minkowski difference has an easy support: `s_{A⊖B}(d) = s_A(d) − s_B(−d)`.
//! - GJK maintains a simplex of such support points inside `A ⊖ B` and iteratively refines it,
//!   always moving toward the origin, until it either encloses the origin (-> intersection) or
//!   finds a direction where no new support point makes progress (-> separation).
//!
//! This module is the 3D specialization: a simplex can be at most a tetrahedron (4 points). The
//! Voronoi-region logic for the line -> triangle -> tetrahedron cases is hand-written. The
//! support-function side is generic over `VectorOps`, so when 4D lands, only the simplex-case
//! logic needs a 4D cousin (pentachoron = 5 points); the support-function path, the iteration
//! loop, and all the numerics are shared.

use glam::Vec3;

/// Shape-side abstraction GJK operates on: a function from direction to the farthest point of
/// the shape in world coordinates.
pub trait SupportFn {
    fn support(&self, direction: Vec3) -> Vec3;
}

/// Convex hull of a finite vertex set in world coordinates. Used as the concrete support for
/// polytope colliders (caller transforms vertices to world space before constructing).
pub struct ConvexHull<'a> {
    pub vertices: &'a [Vec3],
}

impl<'a> SupportFn for ConvexHull<'a> {
    fn support(&self, direction: Vec3) -> Vec3 {
        let mut best = self.vertices[0];
        let mut best_d = best.dot(direction);
        for &v in &self.vertices[1..] {
            let d = v.dot(direction);
            if d > best_d {
                best_d = d;
                best = v;
            }
        }
        best
    }
}

/// Sphere support: centre + radius · normalized(direction).
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

impl SupportFn for Sphere {
    fn support(&self, direction: Vec3) -> Vec3 {
        let dir = direction.normalize_or(Vec3::Y);
        self.center + dir * self.radius
    }
}

/// Support point on the Minkowski difference `A ⊖ B` along `d`, along with the contributing
/// points on each original shape. The originating points are cached because EPA later needs them
/// to reconstruct contact positions on the original shapes.
#[derive(Clone, Copy, Debug)]
pub struct MinkowskiPoint {
    pub point: Vec3,
    pub sa: Vec3,
    pub sb: Vec3,
}

pub fn minkowski_support<A: SupportFn, B: SupportFn>(
    a: &A,
    b: &B,
    direction: Vec3,
) -> MinkowskiPoint {
    let sa = a.support(direction);
    let sb = b.support(-direction);
    MinkowskiPoint {
        point: sa - sb,
        sa,
        sb,
    }
}

/// Result of [`gjk_intersect`]. Either the shapes overlap (with the final tetrahedron simplex,
/// handed to EPA for penetration depth), or they don't (with the last search direction, useful
/// for closest-point queries but currently unused downstream).
#[derive(Debug)]
pub enum GjkResult {
    Intersecting { simplex: [MinkowskiPoint; 4] },
    Separated,
}

const GJK_MAX_ITERATIONS: u32 = 32;
const GJK_EPS: f32 = 1e-6;

/// Test whether shapes `a` and `b` overlap. Returns the final enclosing tetrahedron simplex on
/// intersection for EPA, or `Separated` when a separating direction is found.
pub fn gjk_intersect<A: SupportFn, B: SupportFn>(
    a: &A,
    b: &B,
    initial_direction: Vec3,
) -> GjkResult {
    // First support point, seeded with the caller's initial direction (typically
    // `b.center − a.center`). If that vector is zero we fall back to `+x`.
    let mut dir = if initial_direction.length_squared() > GJK_EPS {
        initial_direction
    } else {
        Vec3::X
    };

    let mut simplex: [MinkowskiPoint; 4] = [MinkowskiPoint {
        point: Vec3::ZERO,
        sa: Vec3::ZERO,
        sb: Vec3::ZERO,
    }; 4];
    // Seed the simplex with the first support and start searching from the side of that point
    // opposite the origin.
    simplex[0] = minkowski_support(a, b, dir);
    let mut n = 1usize;
    dir = -simplex[0].point;

    for _ in 0..GJK_MAX_ITERATIONS {
        // Reject if the new support doesn't cross the origin, the shapes are fully separated
        // along `dir`.
        let new_point = minkowski_support(a, b, dir);
        if new_point.point.dot(dir) < 0.0 {
            return GjkResult::Separated;
        }

        simplex[n] = new_point;
        n += 1;

        let (contains_origin, new_n, new_dir) = do_simplex(&mut simplex, n);
        n = new_n;
        if contains_origin {
            return GjkResult::Intersecting { simplex };
        }
        if new_dir.length_squared() < GJK_EPS {
            // Degenerate: origin sits on a boundary face/edge, or the simplex collapsed. Only
            // return Intersecting if we have a full 4-point tetrahedron — EPA can't operate on
            // anything less. Otherwise report Separated; missing a grazing touch is much safer
            // than crashing EPA with a degenerate simplex.
            if n >= 4 {
                return GjkResult::Intersecting { simplex };
            }
            return GjkResult::Separated;
        }
        dir = new_dir;
    }

    // Iteration cap hit without convergence. This is almost always numerical thrashing at a
    // tangent boundary; the simplex probably doesn't genuinely enclose the origin. Report
    // Separated — better to lose a marginal contact than to feed EPA a bad tetrahedron.
    GjkResult::Separated
}

/// Voronoi-region simplex logic. Reduces the simplex to the feature (vertex / edge / face /
/// volume) closest to the origin and returns a new search direction pointing *from that feature
/// toward the origin*. Also returns `true` if the simplex encloses the origin (only possible
/// with 4 points in 3D).
///
/// On entry `simplex[0..n]` holds `n` points with the most recently added at index `n-1`. On
/// exit `simplex[0..new_n]` holds the surviving points.
fn do_simplex(simplex: &mut [MinkowskiPoint; 4], n: usize) -> (bool, usize, Vec3) {
    match n {
        2 => do_line(simplex),
        3 => do_triangle(simplex),
        4 => do_tetrahedron(simplex),
        _ => unreachable!("simplex size {n} out of range"),
    }
}

/// Line case: simplex is [b, a] with `a` most recent. Decide whether origin is in the AB edge
/// region or past A (past B is impossible because `a` was chosen along `-b`).
fn do_line(simplex: &mut [MinkowskiPoint; 4]) -> (bool, usize, Vec3) {
    let a = simplex[1].point;
    let b = simplex[0].point;
    let ab = b - a;
    let ao = -a;

    if ab.dot(ao) > 0.0 {
        // Origin is between A and B (or past B, but that's ruled out by how `a` was produced).
        // Next search direction is perpendicular to AB, pointing toward origin.
        let dir = triple_product(ab, ao, ab);
        if dir.length_squared() < 1e-10 {
            // Degenerate: origin lies on the line A-B. Pick any vector perpendicular to AB and
            // recurse along that; the next support will either exit the line (and grow the
            // simplex to a triangle) or confirm containment along a different axis.
            (false, 2, any_perpendicular(ab))
        } else {
            (false, 2, dir)
        }
    } else {
        // Origin is past A, away from B. Discard B.
        simplex[0] = simplex[1];
        (false, 1, ao)
    }
}

/// An arbitrary unit-ish vector perpendicular to `v`. Used to escape collinear degeneracies in
/// GJK. The axis chosen depends on `v`'s dominant component to avoid near-zero cross products.
fn any_perpendicular(v: Vec3) -> Vec3 {
    // Cross with whichever cardinal axis `v` is *least* aligned with.
    if v.x.abs() <= v.y.abs() && v.x.abs() <= v.z.abs() {
        v.cross(Vec3::X)
    } else if v.y.abs() <= v.z.abs() {
        v.cross(Vec3::Y)
    } else {
        v.cross(Vec3::Z)
    }
}

/// Discard C from a triangle simplex, keep A and B, recurse into the line case on `[B, A]`.
/// Extracted so `do_triangle` can call it in multiple branches without nested-fn scope issues.
fn fall_back_to_ab(simplex: &mut [MinkowskiPoint; 4]) -> (bool, usize, Vec3) {
    simplex[0] = simplex[1];
    simplex[1] = simplex[2];
    do_line(simplex)
}

/// Triangle case: [c, b, a] with `a` most recent. Determine whether origin is in a vertex
/// region, an edge region, or the triangle's Voronoi face region (above or below the triangle
/// plane).
fn do_triangle(simplex: &mut [MinkowskiPoint; 4]) -> (bool, usize, Vec3) {
    let a = simplex[2].point;
    let b = simplex[1].point;
    let c = simplex[0].point;

    let ab = b - a;
    let ac = c - a;
    let ao = -a;
    let abc = ab.cross(ac); // triangle normal

    // Edge AC region?
    if abc.cross(ac).dot(ao) > 0.0 {
        if ac.dot(ao) > 0.0 {
            // Keep A and C.
            simplex[1] = simplex[2];
            let dir = triple_product(ac, ao, ac);
            return (false, 2, dir);
        }
        return fall_back_to_ab(simplex);
    }

    // Edge AB region?
    if ab.cross(abc).dot(ao) > 0.0 {
        return fall_back_to_ab(simplex);
    }

    // Above or below the triangle.
    if abc.dot(ao) > 0.0 {
        // Origin is on the "abc" side, keep winding [c, b, a], search normal is +abc.
        (false, 3, abc)
    } else {
        // Origin is on the other side, flip winding by swapping b and c, search normal is -abc.
        simplex.swap(0, 1);
        (false, 3, -abc)
    }
}

/// Tetrahedron case: [d, c, b, a] with `a` most recent. Either the origin is inside
/// (intersection!) or it lies in the Voronoi region of one of the three faces adjacent to `a`
/// (ABC / ACD / ADB).
///
/// Textbook formulations of this step assume the simplex winding is such that the bare cross
/// products `ab×ac`, `ac×ad`, `ad×ab` are already outward normals. That assumption doesn't hold
/// for every path through `do_triangle` (specifically the swap branch produces an inverted
/// winding). Rather than patch the winding invariant everywhere, we explicitly orient each face
/// normal using the opposite vertex of the tetrahedron; pointing away from that vertex is always
/// outward.
fn do_tetrahedron(simplex: &mut [MinkowskiPoint; 4]) -> (bool, usize, Vec3) {
    let a = simplex[3].point;
    let b = simplex[2].point;
    let c = simplex[1].point;
    let d = simplex[0].point;

    let ab = b - a;
    let ac = c - a;
    let ad = d - a;
    let ao = -a;

    // Outward normal of face ABC (adjacent to a, opposite d).
    let mut abc = ab.cross(ac);
    if abc.dot(ad) > 0.0 {
        abc = -abc;
    }
    // Outward normal of face ACD (adjacent to a, opposite b).
    let mut acd = ac.cross(ad);
    if acd.dot(ab) > 0.0 {
        acd = -acd;
    }
    // Outward normal of face ADB (adjacent to a, opposite c).
    let mut adb = ad.cross(ab);
    if adb.dot(ac) > 0.0 {
        adb = -adb;
    }

    if abc.dot(ao) > 0.0 {
        // Drop D, recurse on triangle [C, B, A].
        simplex[0] = simplex[1];
        simplex[1] = simplex[2];
        simplex[2] = simplex[3];
        return do_triangle(simplex);
    }
    if acd.dot(ao) > 0.0 {
        // Drop B, recurse on triangle [D, C, A].
        simplex[2] = simplex[3];
        return do_triangle(simplex);
    }
    if adb.dot(ao) > 0.0 {
        // Drop C, recurse on triangle [B, D, A].
        let d_point = simplex[0];
        simplex[0] = simplex[2]; // B -> [0]
        simplex[1] = d_point; // D -> [1]
        simplex[2] = simplex[3]; // A -> [2]
        return do_triangle(simplex);
    }

    // Origin is on the inward side of all three adjacent faces (opposite-d face is implicitly
    // the 4th face; origin's being inside the tetrahedron is exactly these three checks).
    (true, 4, Vec3::ZERO)
}

/// Vector triple product `(a × b) × c`, appears repeatedly in the edge-region search direction
/// formulas. In 3D this simplifies to `b·(a·c) − a·(b·c)` (the "BAC-CAB" identity) but the
/// direct form reads more clearly here.
fn triple_product(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    a.cross(b).cross(c)
}

#[cfg(test)]
mod tests {
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

    #[test]
    fn separated_boxes_report_no_intersection() {
        let va = box_vertices(Vec3::ZERO, Vec3::ONE);
        let vb = box_vertices(Vec3::new(3.0, 0.0, 0.0), Vec3::ONE);
        let a = ConvexHull { vertices: &va };
        let b = ConvexHull { vertices: &vb };

        match gjk_intersect(&a, &b, Vec3::new(3.0, 0.0, 0.0)) {
            GjkResult::Separated => {}
            GjkResult::Intersecting { .. } => panic!("should be separated"),
        }
    }

    #[test]
    fn overlapping_boxes_report_intersection() {
        let va = box_vertices(Vec3::ZERO, Vec3::ONE);
        let vb = box_vertices(Vec3::new(1.5, 0.0, 0.0), Vec3::ONE);
        let a = ConvexHull { vertices: &va };
        let b = ConvexHull { vertices: &vb };

        match gjk_intersect(&a, &b, Vec3::new(1.5, 0.0, 0.0)) {
            GjkResult::Intersecting { .. } => {}
            GjkResult::Separated => panic!("should intersect"),
        }
    }

    #[test]
    fn touching_boxes_report_intersection() {
        // Boundaries exactly meeting count as intersecting (GJK finds the origin on the boundary
        // of the Minkowski difference).
        let va = box_vertices(Vec3::ZERO, Vec3::ONE);
        let vb = box_vertices(Vec3::new(2.0, 0.0, 0.0), Vec3::ONE);
        let a = ConvexHull { vertices: &va };
        let b = ConvexHull { vertices: &vb };

        match gjk_intersect(&a, &b, Vec3::new(2.0, 0.0, 0.0)) {
            GjkResult::Intersecting { .. } => {}
            GjkResult::Separated => panic!("touching boundaries should count as intersecting"),
        }
    }

    #[test]
    fn deeply_overlapping_boxes_report_intersection() {
        let va = box_vertices(Vec3::ZERO, Vec3::ONE);
        let vb = box_vertices(Vec3::new(0.3, 0.1, 0.2), Vec3::ONE);
        let a = ConvexHull { vertices: &va };
        let b = ConvexHull { vertices: &vb };

        match gjk_intersect(&a, &b, Vec3::new(0.3, 0.1, 0.2)) {
            GjkResult::Intersecting { .. } => {}
            GjkResult::Separated => panic!("overlapping centres should intersect"),
        }
    }

    #[test]
    fn sphere_vs_sphere_matches_distance_test() {
        for &(ax, bx, overlap) in &[
            (0.0, 3.0, false), // 3 apart, radii 1 each -> gap of 1
            (0.0, 1.5, true),  // 1.5 apart, radii 1 each -> overlap
            (0.0, 2.0, true),  // exactly touching
        ] {
            let a = Sphere {
                center: Vec3::new(ax, 0.0, 0.0),
                radius: 1.0,
            };
            let b = Sphere {
                center: Vec3::new(bx, 0.0, 0.0),
                radius: 1.0,
            };
            let result = gjk_intersect(&a, &b, Vec3::new(bx - ax, 0.0, 0.0));
            let got = matches!(result, GjkResult::Intersecting { .. });
            assert_eq!(
                got, overlap,
                "centres at ({ax},0,0) and ({bx},0,0): expected intersecting={overlap}, got {got}"
            );
        }
    }

    #[test]
    fn box_vs_sphere_corner_contact() {
        // Unit box with corner at (1,1,1). Sphere at (1+d, 1+d, 1+d) reaches the corner when
        // d·√3 ≤ r, i.e. d ≤ r/√3. For r=0.5: threshold d ≈ 0.2887.
        //   d = 0.35 -> distance 0.606 > 0.5 -> no overlap
        //   d = 0.20 -> distance 0.346 < 0.5 -> overlap
        let vb = box_vertices(Vec3::ZERO, Vec3::ONE);
        let b = ConvexHull { vertices: &vb };

        let far = Sphere {
            center: Vec3::new(1.35, 1.35, 1.35),
            radius: 0.5,
        };
        assert!(matches!(
            gjk_intersect(&far, &b, Vec3::new(-1.0, -1.0, -1.0)),
            GjkResult::Separated
        ));

        let near = Sphere {
            center: Vec3::new(1.2, 1.2, 1.2),
            radius: 0.5,
        };
        assert!(matches!(
            gjk_intersect(&near, &b, Vec3::new(-1.0, -1.0, -1.0)),
            GjkResult::Intersecting { .. }
        ));
    }

    #[test]
    fn rotated_boxes_separate_as_axes_allow() {
        // Two unit boxes, one rotated 45° about Z. Axis-aligned box extends x∈[-1,1]; the
        // diamond (rotated box) has x-extent of ±√2 ≈ ±1.414. At centre distance 2.5 they
        // separate; at 2.2 they barely overlap.
        use glam::Quat;
        let va = box_vertices(Vec3::ZERO, Vec3::ONE);
        let rot = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
        let vb_rot: Vec<Vec3> = box_vertices(Vec3::ZERO, Vec3::ONE)
            .iter()
            .map(|&v| rot * v + Vec3::new(2.5, 0.0, 0.0))
            .collect();
        let a = ConvexHull { vertices: &va };
        let b = ConvexHull { vertices: &vb_rot };

        assert!(matches!(
            gjk_intersect(&a, &b, Vec3::new(2.5, 0.0, 0.0)),
            GjkResult::Separated
        ));

        let vb_close: Vec<Vec3> = box_vertices(Vec3::ZERO, Vec3::ONE)
            .iter()
            .map(|&v| rot * v + Vec3::new(2.2, 0.0, 0.0))
            .collect();
        let b_close = ConvexHull {
            vertices: &vb_close,
        };
        assert!(matches!(
            gjk_intersect(&a, &b_close, Vec3::new(2.2, 0.0, 0.0)),
            GjkResult::Intersecting { .. }
        ));
    }
}
