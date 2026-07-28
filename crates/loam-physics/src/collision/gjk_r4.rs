//! GJK in R⁴ using dimension-agnostic closest-point-on-simplex.
//!
//! Parallels [`crate::collision::gjk`] (3D) but substitutes the 3D's hand-rolled Voronoi-region
//! simplex analysis with the Gram-matrix projection from [`super::simplex_r4`]. That lets us
//! handle all simplex sizes (1 through 5) uniformly, which matters in 4D because a 4-simplex
//! (pentatope) is the first volume-enclosing case; the 3D "tetrahedron encloses when all three
//! face-normals point inward" logic has no direct 4D analogue worth hand-deriving.
//!
//! The support-function side (`Sphere`, `ConvexHull`, `SupportFn`) parallels the 3D module with
//! `Vec4` replacements.

use glam::Vec4;

use super::simplex_r4::{closest_to_origin, Closest};

/// Shape-side abstraction for 4D GJK.
pub trait SupportFn4 {
    fn support(&self, direction: Vec4) -> Vec4;
}

/// Convex-hull support from an explicit world-space vertex list.
pub struct ConvexHull4<'a> {
    pub vertices: &'a [Vec4],
}

impl<'a> SupportFn4 for ConvexHull4<'a> {
    fn support(&self, direction: Vec4) -> Vec4 {
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

/// Sphere support in 4D: centre + r·d̂.
pub struct Sphere4 {
    pub center: Vec4,
    pub radius: f32,
}

impl SupportFn4 for Sphere4 {
    fn support(&self, direction: Vec4) -> Vec4 {
        let d = direction.length_squared();
        let dir = if d > 1e-12 {
            direction / d.sqrt()
        } else {
            Vec4::Y
        };
        self.center + dir * self.radius
    }
}

/// Minkowski-difference support: the point on `A ⊖ B` farthest along `direction`, with the
/// contributing pre-image points on `A` and `B` cached for EPA's contact-point reconstruction.
#[derive(Clone, Copy, Debug)]
pub struct MinkowskiPoint4 {
    pub point: Vec4,
    pub sa: Vec4,
    pub sb: Vec4,
}

pub fn minkowski_support_r4<A: SupportFn4, B: SupportFn4>(
    a: &A,
    b: &B,
    direction: Vec4,
) -> MinkowskiPoint4 {
    let sa = a.support(direction);
    let sb = b.support(-direction);
    MinkowskiPoint4 {
        point: sa - sb,
        sa,
        sb,
    }
}

/// GJK result: either the shapes overlap, in which case we hand the final simplex plus its
/// surviving sub-simplex to EPA, or they don't. In 4D the enclosing simplex always has 5
/// vertices; EPA receives exactly that.
///
/// The variants are asymmetric in size (an inline `[MinkowskiPoint4; 5]` is ~240 bytes;
/// `Separated` is 0). We keep it inline; the enum is a short-lived stack return from
/// narrowphase, not a stored field, so the size asymmetry doesn't matter in practice.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum GjkResult4 {
    Intersecting {
        /// 4D simplex (5 points) whose convex hull encloses the origin. Used to seed 4D EPA's
        /// polytope expansion.
        simplex: [MinkowskiPoint4; 5],
    },
    Separated,
}

const GJK_MAX_ITERATIONS: u32 = 48;
const GJK_EPS: f32 = 1e-6;

/// Test whether `a` and `b` overlap via GJK on their Minkowski difference. Returns
/// `Intersecting` with an enclosing 4-simplex for downstream EPA, or `Separated`.
///
/// Strategy: maintain a growing simplex inside `A ⊖ B`; each iteration computes the closest
/// point on the current simplex to the origin (via [`closest_to_origin`]), drops any unused
/// vertices, then searches for a new support in the direction from the closest point toward the
/// origin. Terminates when (i) a new support can't advance toward the origin (-> separated),
/// (ii) the simplex's closest point reaches the origin (-> intersecting), or (iii) iteration cap
/// is hit.
pub fn gjk_intersect_r4<A: SupportFn4, B: SupportFn4>(
    a: &A,
    b: &B,
    initial_direction: Vec4,
) -> GjkResult4 {
    let mut dir = if initial_direction.length_squared() > GJK_EPS {
        initial_direction
    } else {
        Vec4::X
    };
    let mut simplex: Vec<MinkowskiPoint4> = Vec::with_capacity(5);
    simplex.push(minkowski_support_r4(a, b, dir));
    dir = -simplex[0].point;

    // ---- Stage 1: standard GJK, searching toward the origin.
    // Terminates when either (a) a new support fails to cross the origin along the search
    // direction (-> Separated) or (b) the current simplex's closest-point to origin is already
    // at the origin (-> shapes intersect, exit to stage 2 to grow the simplex to 5 points for
    // EPA).
    for _ in 0..GJK_MAX_ITERATIONS {
        if dir.length_squared() < GJK_EPS {
            break;
        }
        let new_point = minkowski_support_r4(a, b, dir);
        if new_point.point.dot(dir) < 0.0 {
            return GjkResult4::Separated;
        }
        // Duplicate support = can't make further toward-origin progress; treat as enclosure
        // confirmation.
        if simplex
            .iter()
            .any(|p| (p.point - new_point.point).length_squared() < 1e-10)
        {
            break;
        }
        simplex.push(new_point);

        // Reduce + advance.
        let points: Vec<Vec4> = simplex.iter().map(|p| p.point).collect();
        let Closest {
            point: closest,
            kept,
            ..
        } = closest_to_origin(&points);
        if closest.length_squared() < GJK_EPS {
            // Origin is in the current hull, enter growth phase.
            let pruned: Vec<MinkowskiPoint4> = kept.iter().map(|&i| simplex[i]).collect();
            simplex = pruned;
            break;
        }
        let pruned: Vec<MinkowskiPoint4> = kept.iter().map(|&i| simplex[i]).collect();
        simplex = pruned;
        dir = -closest;
    }

    // ---- Stage 2: grow the (already-enclosing) simplex to 5 points.
    // Each iteration picks a direction orthogonal to the current simplex's affine hull and adds
    // the support point there; either it's a genuine new hull vertex (simplex grows) or it's
    // co-located with an existing vertex (polytope is too thin along that axis, try the opposite
    // sign, then bail).
    let mut tried: Vec<Vec4> = Vec::new();
    while simplex.len() < 5 {
        let Some(probe) = orthogonal_to_hull(&simplex, &tried) else {
            break;
        };
        tried.push(probe);

        let sup = minkowski_support_r4(a, b, probe);
        if simplex
            .iter()
            .all(|p| (p.point - sup.point).length_squared() >= 1e-10)
        {
            simplex.push(sup);
            continue;
        }
        let sup_neg = minkowski_support_r4(a, b, -probe);
        if simplex
            .iter()
            .all(|p| (p.point - sup_neg.point).length_squared() >= 1e-10)
        {
            simplex.push(sup_neg);
            continue;
        }
        // Polytope is truly flat along this probe axis; mark as tried and move on.
    }

    if simplex.len() == 5 {
        finalize_intersecting(simplex)
    } else {
        GjkResult4::Separated
    }
}

fn finalize_intersecting(simplex: Vec<MinkowskiPoint4>) -> GjkResult4 {
    if simplex.len() == 5 {
        let arr: [MinkowskiPoint4; 5] =
            [simplex[0], simplex[1], simplex[2], simplex[3], simplex[4]];
        GjkResult4::Intersecting { simplex: arr }
    } else {
        GjkResult4::Separated
    }
}

/// A unit vector perpendicular to the affine hull of the current `simplex`, and not parallel to
/// any already-tried direction. Used to grow a partial simplex after GJK has already established
/// that the origin lies in its hull.
///
/// `tried` is consulted so we don't re-pick a direction the caller has already probed (which
/// would just return the same support and stall growth).
fn orthogonal_to_hull(simplex: &[MinkowskiPoint4], tried: &[Vec4]) -> Option<Vec4> {
    let points: Vec<Vec4> = simplex.iter().map(|p| p.point).collect();
    let basis: Vec<Vec4> = if points.len() <= 1 {
        Vec::new()
    } else {
        let v0 = points[0];
        points[1..].iter().map(|&p| p - v0).collect()
    };

    // Orthonormalize the basis (Gram-Schmidt). Skip near-zero rows.
    let mut onb: Vec<Vec4> = Vec::with_capacity(basis.len());
    for &b in &basis {
        let mut r = b;
        for o in &onb {
            r -= *o * r.dot(*o);
        }
        let m = r.length_squared();
        if m > 1e-10 {
            onb.push(r / m.sqrt());
        }
    }

    // For each cardinal axis, compute the residual after projecting out the basis and out every
    // already-tried direction. Rank candidates by residual magnitude; return the strongest.
    let axes = [Vec4::X, Vec4::Y, Vec4::Z, Vec4::W];
    let mut best: Option<(f32, Vec4)> = None;
    for &axis in &axes {
        let mut r = axis;
        for o in &onb {
            r -= *o * r.dot(*o);
        }
        for t in tried {
            let tl = t.length_squared();
            if tl > 1e-12 {
                r -= *t * (r.dot(*t) / tl);
            }
        }
        let mag_sq = r.length_squared();
        if mag_sq > 1e-8 && best.is_none_or(|(m, _)| mag_sq > m) {
            best = Some((mag_sq, r));
        }
    }
    best.map(|(_, v)| v.normalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separated_spheres() {
        let a = Sphere4 {
            center: Vec4::new(-5.0, 0.0, 0.0, 0.0),
            radius: 1.0,
        };
        let b = Sphere4 {
            center: Vec4::new(5.0, 0.0, 0.0, 0.0),
            radius: 1.0,
        };
        match gjk_intersect_r4(&a, &b, Vec4::X) {
            GjkResult4::Separated => {}
            _ => panic!("expected Separated"),
        }
    }

    #[test]
    fn overlapping_spheres() {
        let a = Sphere4 {
            center: Vec4::new(0.0, 0.0, 0.0, 0.0),
            radius: 2.0,
        };
        let b = Sphere4 {
            center: Vec4::new(1.0, 0.0, 0.0, 0.0),
            radius: 2.0,
        };
        // Overlapping spheres must report Intersecting, though the simplex shape itself we don't
        // inspect here.
        assert!(matches!(
            gjk_intersect_r4(&a, &b, Vec4::X),
            GjkResult4::Intersecting { .. }
        ));
    }

    #[test]
    fn tesseracts_overlap_past_touching() {
        use crate::euclidean_r4::tesseract_vertices;
        let va: Vec<Vec4> = tesseract_vertices(1.0);
        // Shift less than 1 so they overlap well past a single-corner touch. (Exact-touch at
        // `(1,1,1,1)` is a boundary case GJK handles probabilistically, dropped as a test case.)
        let vb: Vec<Vec4> = tesseract_vertices(1.0)
            .into_iter()
            .map(|v| v + Vec4::new(0.6, 0.6, 0.6, 0.6))
            .collect();
        let a = ConvexHull4 { vertices: &va };
        let b = ConvexHull4 { vertices: &vb };
        assert!(matches!(
            gjk_intersect_r4(&a, &b, Vec4::X),
            GjkResult4::Intersecting { .. }
        ));
    }

    #[test]
    fn deeply_overlapping_pentatopes() {
        use crate::euclidean_r4::pentatope_vertices;
        let va: Vec<Vec4> = pentatope_vertices(1.0);
        // Pentatope at origin vs pentatope shifted by a small vector; they should overlap
        // substantially.
        let vb: Vec<Vec4> = pentatope_vertices(1.0)
            .into_iter()
            .map(|v| v + Vec4::new(0.2, 0.0, 0.0, 0.0))
            .collect();
        let a = ConvexHull4 { vertices: &va };
        let b = ConvexHull4 { vertices: &vb };
        assert!(matches!(
            gjk_intersect_r4(&a, &b, Vec4::X),
            GjkResult4::Intersecting { .. }
        ));
    }

    #[test]
    fn fully_separated_pentatopes() {
        use crate::euclidean_r4::pentatope_vertices;
        let va: Vec<Vec4> = pentatope_vertices(1.0);
        let vb: Vec<Vec4> = pentatope_vertices(1.0)
            .into_iter()
            .map(|v| v + Vec4::new(10.0, 0.0, 0.0, 0.0))
            .collect();
        let a = ConvexHull4 { vertices: &va };
        let b = ConvexHull4 { vertices: &vb };
        assert!(matches!(
            gjk_intersect_r4(&a, &b, Vec4::X),
            GjkResult4::Separated
        ));
    }

    /// Exact touching, the case the overlap fixtures step around. GJK accepts
    /// a simplex whose closest point is within `GJK_EPS` of the origin, so a
    /// pair at zero separation can be called either way and pinning that
    /// verdict would pin the tolerance instead of the geometry. What has to
    /// hold is that the verdict is monotone in the gap, and that a gap never
    /// resolves to a depth: a phantom depth is indistinguishable from a real
    /// one by the time it reaches the solver.
    #[test]
    fn tesseract_face_touching_is_bracketed_and_never_resolves_to_a_depth() {
        use crate::collision::epa_r4::epa_r4;
        use crate::euclidean_r4::tesseract_vertices;

        let va: Vec<Vec4> = tesseract_vertices(1.0);
        let a = ConvexHull4 { vertices: &va };
        // `tesseract_vertices(r)` is the box of half extent `r/2`, so the
        // faces meet at a shift of `r`; below that the boxes overlap by the
        // difference, above it they are clear.
        for (shift, expect_overlap) in [
            (1.0 - 1e-1, true),
            (1.0 - 1e-2, true),
            (1.0, false),
            (1.0 + 1e-1, false),
            (1.0 + 1.0, false),
        ] {
            let vb: Vec<Vec4> = tesseract_vertices(1.0)
                .into_iter()
                .map(|v| v + Vec4::new(shift, 0.0, 0.0, 0.0))
                .collect();
            let b = ConvexHull4 { vertices: &vb };
            let depth = match gjk_intersect_r4(&a, &b, Vec4::X) {
                GjkResult4::Intersecting { simplex } => {
                    epa_r4(&a, &b, simplex).map_or(0.0, |c| c.penetration)
                }
                GjkResult4::Separated => 0.0,
            };
            if expect_overlap {
                assert!(
                    (depth - (1.0 - shift)).abs() < 1e-2,
                    "shift {shift} overlaps by {} but resolved to {depth}",
                    1.0 - shift
                );
            } else {
                assert_eq!(depth, 0.0, "shift {shift} is clear but resolved to {depth}");
            }
        }
    }

    #[test]
    fn sphere_and_tesseract_inside() {
        use crate::euclidean_r4::tesseract_vertices;
        let sphere = Sphere4 {
            center: Vec4::ZERO,
            radius: 0.1,
        };
        let vs: Vec<Vec4> = tesseract_vertices(1.0);
        let tess = ConvexHull4 { vertices: &vs };
        assert!(matches!(
            gjk_intersect_r4(&sphere, &tess, Vec4::X),
            GjkResult4::Intersecting { .. }
        ));
    }
}
