//! Closest-point-on-simplex for GJK in R⁴.
//!
//! In 3D the GJK simplex logic uses explicit Voronoi-region analysis (line/triangle/tetrahedron)
//! with hand-tuned cross products. That approach doesn't generalize cleanly to 4D, a triangle's
//! "normal" is now a 2D space, a tetrahedron's is a 1D line, and a 4-simplex is the first
//! volume-enclosing case.
//!
//! Instead, this module computes the closest point on the simplex's convex hull to the origin
//! via **Gram-matrix projection** onto each sub-simplex's affine hull, keeping the one with
//! smallest distance that has all barycentric weights non-negative. For a k-simplex this is an
//! O(2^k · k³) operation; with k ≤ 4 (4D) it reduces to at most 31 × 64 ≈ 2000 f32 ops,
//! dominated by collision check setup anyway.
//!
//! The approach trades absolute speed for obvious correctness and dimension-agnostic simplicity.
//! A future revision could swap in the signed-volumes method (Montanari-Petrinic 2018) once the
//! 4D collision layer has tests covering its edge cases.

use glam::Vec4;

/// Closest point on the simplex's convex hull to the origin, plus the sub-simplex that realizes
/// that closest point.
#[derive(Debug, Clone)]
pub struct Closest {
    /// World-space closest point, `Σ weight_i · simplex_i`.
    pub point: Vec4,
    /// Barycentric weights over the input simplex; entries not in `kept` are exactly zero.
    /// Weights in `kept` are ≥ 0 and sum to 1 within f32 tolerance.
    pub weights: Vec<f32>,
    /// Indices of the simplex vertices whose weights are non-zero, the sub-simplex GJK should
    /// carry forward into the next iter.
    pub kept: Vec<usize>,
}

/// Compute the closest point on the convex hull of `simplex` to the origin. `simplex.len()` must
/// be in `1..=5` (4D simplex can have at most 5 vertices). Returns the closest point with its
/// barycentric decomposition.
pub fn closest_to_origin(simplex: &[Vec4]) -> Closest {
    let n = simplex.len();
    debug_assert!((1..=5).contains(&n), "simplex size {n} out of 1..=5");

    let mut best_dist_sq = f32::MAX;
    let mut best = Closest {
        point: simplex[0],
        weights: {
            let mut w = vec![0.0; n];
            w[0] = 1.0;
            w
        },
        kept: vec![0],
    };

    // Enumerate all non-empty subsets (at most 2⁵ − 1 = 31 for n = 5).
    for mask in 1u32..(1u32 << n) {
        let subset: Vec<usize> = (0..n).filter(|i| mask & (1 << i) != 0).collect();

        let Some((pt, weights)) = project_origin_onto_affine_hull(&subset, simplex) else {
            continue;
        };

        // Accept only when every weight is non-negative (i.e. the projection lies inside, not
        // outside, the sub-simplex). Small tolerance for f32 boundary cases.
        if !weights.iter().all(|&w| w >= -1e-6) {
            continue;
        }

        let dist_sq = pt.length_squared();
        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            let mut full_weights = vec![0.0; n];
            for (k, &i) in subset.iter().enumerate() {
                full_weights[i] = weights[k].max(0.0);
            }
            best = Closest {
                point: pt,
                weights: full_weights,
                kept: subset,
            };
        }
    }

    best
}

/// Project the origin onto the affine hull of the sub-simplex with indices `subset` into
/// `simplex`. Returns the projected point and the barycentric weights over the sub-simplex, or
/// `None` when the sub-simplex is degenerate (e.g. two vertices at the same point).
///
/// The minimization:
/// ```text
///   min_α |v₀ + Σᵢ₌₁ᵏ αᵢ (vᵢ − v₀)|²   with weights w = (1 − Σα, α₁, …, αₖ)
/// ```
/// reduces to the normal equations `G α = −Dᵀv₀` where `Dᵢⱼ = (vᵢ − v₀) · (vⱼ − v₀)` is the
/// Gram matrix of edge vectors.
fn project_origin_onto_affine_hull(subset: &[usize], simplex: &[Vec4]) -> Option<(Vec4, Vec<f32>)> {
    let n = subset.len();
    if n == 0 {
        return None;
    }
    if n == 1 {
        return Some((simplex[subset[0]], vec![1.0]));
    }

    let v0 = simplex[subset[0]];
    let dirs: Vec<Vec4> = subset[1..].iter().map(|&i| simplex[i] - v0).collect();
    let k = dirs.len();

    // Build k×k Gram matrix `G` and right-hand side `b`.
    let mut g = [[0.0_f32; 4]; 4];
    let mut b = [0.0_f32; 4];
    for i in 0..k {
        b[i] = -dirs[i].dot(v0);
        for j in 0..k {
            g[i][j] = dirs[i].dot(dirs[j]);
        }
    }

    // Solve `G α = b` via Gauss-Jordan elimination. For k ≤ 4 this is tiny and the copy
    // overhead of more general linear-algebra machinery isn't worth it. Returns `None` when the
    // matrix is singular (collinear / coplanar sub-simplex).
    let alphas = solve_spd_system(&mut g, &mut b, k)?;

    let mut weights = Vec::with_capacity(n);
    let sum_alpha: f32 = alphas.iter().sum();
    weights.push(1.0 - sum_alpha);
    weights.extend_from_slice(&alphas);

    let mut point = v0;
    for (i, &a) in alphas.iter().enumerate() {
        point += dirs[i] * a;
    }

    Some((point, weights))
}

/// Gauss-Jordan solve of a small symmetric positive-semidefinite system `G · α = b`. In-place
/// on the augmented matrix; returns `None` if any pivot is too small to trust (degenerate
/// simplex).
fn solve_spd_system(g: &mut [[f32; 4]; 4], b: &mut [f32; 4], k: usize) -> Option<Vec<f32>> {
    // Augmented: `[G | b]`.
    for i in 0..k {
        // Partial-pivot: find the row with largest |g[row][i]| below the diagonal, swap to
        // position `i`.
        let mut pivot = i;
        for r in (i + 1)..k {
            if g[r][i].abs() > g[pivot][i].abs() {
                pivot = r;
            }
        }
        if pivot != i {
            g.swap(i, pivot);
            b.swap(i, pivot);
        }

        let piv = g[i][i];
        if piv.abs() < 1e-10 {
            return None; // degenerate
        }

        // Normalize pivot row.
        let inv_piv = 1.0 / piv;
        for x in g[i][i..k].iter_mut() {
            *x *= inv_piv;
        }
        b[i] *= inv_piv;

        // Eliminate other rows. `g[i]` is `[f32; 4]` (Copy), so cache it locally, that frees
        // the borrow checker to mutate other rows of `g` without splitting the outer borrow.
        let pivot_row = g[i];
        let pivot_b = b[i];
        for r in 0..k {
            if r == i {
                continue;
            }
            let factor = g[r][i];
            if factor == 0.0 {
                continue;
            }
            for (target, &p) in g[r][i..k].iter_mut().zip(pivot_row[i..k].iter()) {
                *target -= factor * p;
            }
            b[r] -= factor * pivot_b;
        }
    }

    Some((0..k).map(|i| b[i]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol,
            "{a} not close to {b} (tol {tol}, diff {})",
            (a - b).abs()
        );
    }

    #[test]
    fn single_point_returns_itself() {
        let c = closest_to_origin(&[Vec4::new(1.0, 2.0, 3.0, 4.0)]);
        assert_close(
            c.point.length(),
            Vec4::new(1.0, 2.0, 3.0, 4.0).length(),
            1e-6,
        );
        assert_eq!(c.kept, vec![0]);
        assert_close(c.weights[0], 1.0, 1e-6);
    }

    #[test]
    fn line_segment_containing_origin_returns_origin() {
        // Segment from (−1, 0, 0, 0) to (1, 0, 0, 0) contains origin at the midpoint with
        // equal weights.
        let c = closest_to_origin(&[
            Vec4::new(-1.0, 0.0, 0.0, 0.0),
            Vec4::new(1.0, 0.0, 0.0, 0.0),
        ]);
        assert_close(c.point.length(), 0.0, 1e-6);
        assert_eq!(c.kept.len(), 2);
        assert_close(c.weights[0], 0.5, 1e-4);
        assert_close(c.weights[1], 0.5, 1e-4);
    }

    #[test]
    fn line_segment_outside_origin_projects_to_endpoint() {
        // Segment entirely on positive x-axis: origin projects onto the endpoint closest to it,
        // which is x = 1.
        let c = closest_to_origin(&[Vec4::new(1.0, 0.0, 0.0, 0.0), Vec4::new(2.0, 0.0, 0.0, 0.0)]);
        assert_close(c.point.x, 1.0, 1e-4);
        assert_eq!(c.kept, vec![0]);
    }

    #[test]
    fn triangle_containing_origin() {
        // Three points in the xy-plane forming a triangle around the origin; closest point
        // should be the origin itself.
        let c = closest_to_origin(&[
            Vec4::new(-1.0, -1.0, 0.0, 0.0),
            Vec4::new(1.0, -1.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0, 0.0, 0.0),
        ]);
        assert_close(c.point.length(), 0.0, 1e-5);
        assert_eq!(c.kept.len(), 3);
    }

    #[test]
    fn tetrahedron_in_3d_subspace_containing_origin() {
        // Standard tetrahedron with origin inside, embedded in 4D (w = 0 throughout).
        let c = closest_to_origin(&[
            Vec4::new(1.0, 1.0, 1.0, 0.0),
            Vec4::new(1.0, -1.0, -1.0, 0.0),
            Vec4::new(-1.0, 1.0, -1.0, 0.0),
            Vec4::new(-1.0, -1.0, 1.0, 0.0),
        ]);
        assert_close(c.point.length(), 0.0, 1e-4);
    }

    #[test]
    fn pentatope_containing_origin() {
        // 5-simplex in 4D enclosing origin. Using the 16-cell's vertex set minus a few, actually
        // easier: use 5 vertices of a symmetric 4-simplex.
        let c = closest_to_origin(&[
            Vec4::new(1.0, 1.0, 1.0, -1.0 / 5.0_f32.sqrt()),
            Vec4::new(1.0, -1.0, -1.0, -1.0 / 5.0_f32.sqrt()),
            Vec4::new(-1.0, 1.0, -1.0, -1.0 / 5.0_f32.sqrt()),
            Vec4::new(-1.0, -1.0, 1.0, -1.0 / 5.0_f32.sqrt()),
            Vec4::new(0.0, 0.0, 0.0, 4.0 / 5.0_f32.sqrt()),
        ]);
        assert_close(c.point.length(), 0.0, 1e-3);
    }

    #[test]
    fn triangle_projects_to_edge() {
        // Triangle whose closest face to origin is an edge (origin lies outside the triangle,
        // nearest the AB edge).
        let a = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let b = Vec4::new(0.0, 1.0, 0.0, 0.0);
        let c_vert = Vec4::new(2.0, 2.0, 0.0, 0.0); // far away
        let c = closest_to_origin(&[a, b, c_vert]);
        // Closest point on the AB edge to origin is (0.5, 0.5, 0, 0).
        assert_close(c.point.x, 0.5, 1e-4);
        assert_close(c.point.y, 0.5, 1e-4);
        assert_close(c.point.z, 0.0, 1e-4);
        assert_close(c.point.w, 0.0, 1e-4);
        assert_eq!(c.kept.len(), 2);
    }
}
