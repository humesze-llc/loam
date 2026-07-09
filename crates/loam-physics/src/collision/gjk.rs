//! GJK containment test for convex shapes.
//!
//! The Minkowski difference `A ⊖ B = { a − b }` contains the origin iff `A ∩ B`
//! is non-empty. GJK tests this by maintaining a simplex of support points
//! `s_{A⊖B}(d) = s_A(d) − s_B(−d)` and refining toward the origin until it
//! either encloses the origin (intersection) or finds a direction with no
//! progress (separation).
//!
//! 3D specialization: simplex up to a tetrahedron. The Voronoi-region logic for
//! line -> triangle -> tetrahedron is hand-written.

use glam::Vec3;

/// A function from direction to the shape's farthest point in world coordinates.
pub trait SupportFn {
    fn support(&self, direction: Vec3) -> Vec3;
}

/// Convex hull of a world-space vertex set; the support for polytope colliders.
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

/// Sphere collider; support is `center + radius·normalize(direction)`.
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

/// Support point on `A ⊖ B` plus the contributing points `sa`, `sb` on each shape,
/// cached because EPA reconstructs contact positions from them.
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

/// Result of [`gjk_intersect`]: overlapping (with the terminating tetrahedron for
/// EPA) or separated.
#[derive(Debug)]
pub enum GjkResult {
    Intersecting { simplex: [MinkowskiPoint; 4] },
    Separated,
}

const GJK_MAX_ITERATIONS: u32 = 32;
const GJK_EPS: f32 = 1e-6;

/// Test whether shapes `a` and `b` overlap, returning the enclosing tetrahedron
/// for EPA on intersection or `Separated` otherwise.
pub fn gjk_intersect<A: SupportFn, B: SupportFn>(
    a: &A,
    b: &B,
    initial_direction: Vec3,
) -> GjkResult {
    // Seed direction (typically `b.center − a.center`); fall back to `+x` if zero.
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
    simplex[0] = minkowski_support(a, b, dir);
    let mut n = 1usize;
    dir = -simplex[0].point;

    for _ in 0..GJK_MAX_ITERATIONS {
        // If the new support doesn't reach past the origin, separated along `dir`.
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
            // Origin on a boundary feature, or simplex collapsed. Only hand EPA a
            // full tetrahedron; otherwise report Separated rather than feed it a
            // degenerate simplex.
            if n >= 4 {
                return GjkResult::Intersecting { simplex };
            }
            return GjkResult::Separated;
        }
        dir = new_dir;
    }

    // Cap hit without convergence: almost always thrashing at a tangent boundary.
    // Report Separated rather than feed EPA a bad tetrahedron.
    GjkResult::Separated
}

/// Reduce the simplex to the feature closest to the origin and return a search
/// direction from that feature toward the origin; `true` iff the simplex encloses
/// the origin. On entry `simplex[0..n]` has the newest point at `n-1`; on exit
/// `simplex[0..new_n]` holds the survivors.
fn do_simplex(simplex: &mut [MinkowskiPoint; 4], n: usize) -> (bool, usize, Vec3) {
    match n {
        2 => do_line(simplex),
        3 => do_triangle(simplex),
        4 => do_tetrahedron(simplex),
        _ => unreachable!("simplex size {n} out of range"),
    }
}

/// Line case: simplex [b, a], `a` newest. Origin is in the AB edge region or past
/// A (past B is impossible since `a` was chosen along `-b`).
fn do_line(simplex: &mut [MinkowskiPoint; 4]) -> (bool, usize, Vec3) {
    let a = simplex[1].point;
    let b = simplex[0].point;
    let ab = b - a;
    let ao = -a;

    if ab.dot(ao) > 0.0 {
        // Edge region: search perpendicular to AB toward the origin.
        let dir = triple_product(ab, ao, ab);
        if dir.length_squared() < 1e-10 {
            // Origin on line AB: any perpendicular escapes the degeneracy.
            (false, 2, any_perpendicular(ab))
        } else {
            (false, 2, dir)
        }
    } else {
        // Past A: discard B.
        simplex[0] = simplex[1];
        (false, 1, ao)
    }
}

/// A vector perpendicular to `v`, crossing with the cardinal axis `v` is least
/// aligned with to avoid a near-zero cross product.
fn any_perpendicular(v: Vec3) -> Vec3 {
    if v.x.abs() <= v.y.abs() && v.x.abs() <= v.z.abs() {
        v.cross(Vec3::X)
    } else if v.y.abs() <= v.z.abs() {
        v.cross(Vec3::Y)
    } else {
        v.cross(Vec3::Z)
    }
}

/// Drop C, keep A and B, recurse into the line case on `[B, A]`.
fn fall_back_to_ab(simplex: &mut [MinkowskiPoint; 4]) -> (bool, usize, Vec3) {
    simplex[0] = simplex[1];
    simplex[1] = simplex[2];
    do_line(simplex)
}

/// Triangle case: [c, b, a], `a` newest. Origin is in a vertex, edge, or face
/// Voronoi region (above or below the triangle plane).
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

    if abc.dot(ao) > 0.0 {
        // Origin above: keep winding, search +abc.
        (false, 3, abc)
    } else {
        // Origin below: flip winding (swap b, c), search -abc.
        simplex.swap(0, 1);
        (false, 3, -abc)
    }
}

/// Tetrahedron case: [d, c, b, a], `a` newest. Origin is either inside
/// (intersection) or in the Voronoi region of a face adjacent to `a` (ABC/ACD/ADB).
///
/// We do not rely on the textbook winding invariant (the `do_triangle` swap branch
/// breaks it); each face normal is oriented away from the opposite vertex, which
/// is always outward.
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

    // Inside all three adjacent faces: origin is in the tetrahedron.
    (true, 4, Vec3::ZERO)
}

/// Vector triple product `(a × b) × c`, used in the edge-region search directions.
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
        // Boundaries exactly meeting count as intersecting.
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
