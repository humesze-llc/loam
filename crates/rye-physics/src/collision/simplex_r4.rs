//! Closest-point-on-simplex for GJK in R⁴.
//!
//! The 3D Voronoi-region approach doesn't generalize to 4D, so this computes
//! the closest point on the simplex hull to the origin via Gram-matrix
//! projection onto each sub-simplex's affine hull, keeping the smallest-distance
//! one with all barycentric weights non-negative. O(2^k · k³), at most ~2000
//! f32 ops for k ≤ 4. Trades speed for dimension-agnostic correctness; the
//! signed-volumes method (Montanari-Petrinic 2018) is the faster alternative.

use glam::Vec4;

/// Closest point on the simplex hull to the origin, plus the realizing
/// sub-simplex.
#[derive(Debug, Clone)]
pub struct Closest {
    /// World-space closest point, `Σ weight_i · simplex_i`.
    pub point: Vec4,
    /// Barycentric weights over the input simplex; entries not in `kept` are
    /// zero. Weights in `kept` are ≥ 0 and sum to 1 within f32 tolerance.
    pub weights: Vec<f32>,
    /// Non-zero-weight vertex indices; the sub-simplex GJK carries forward.
    pub kept: Vec<usize>,
}

/// Closest point on the convex hull of `simplex` to the origin, with its
/// barycentric decomposition. `simplex.len()` must be in `1..=5`.
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

    // All non-empty subsets (at most 2⁵ − 1 = 31).
    for mask in 1u32..(1u32 << n) {
        let subset: Vec<usize> = (0..n).filter(|i| mask & (1 << i) != 0).collect();

        let Some((pt, weights)) = project_origin_onto_affine_hull(&subset, simplex) else {
            continue;
        };

        // All-non-negative weights means the projection lies inside the
        // sub-simplex. Tolerance covers f32 boundary cases.
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

/// Project the origin onto the affine hull of sub-simplex `subset`. Returns the
/// point and barycentric weights, or `None` when the sub-simplex is degenerate.
///
/// `min_α |v₀ + Σ αᵢ (vᵢ − v₀)|²` reduces to the normal equations `G α = −Dᵀv₀`
/// with Gram matrix `Dᵢⱼ = (vᵢ − v₀) · (vⱼ − v₀)`.
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

    let mut g = [[0.0_f32; 4]; 4];
    let mut b = [0.0_f32; 4];
    for i in 0..k {
        b[i] = -dirs[i].dot(v0);
        for j in 0..k {
            g[i][j] = dirs[i].dot(dirs[j]);
        }
    }

    // `None` when singular (collinear / coplanar sub-simplex).
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

/// Gauss-Jordan solve of a small SPSD system `G · α = b`, in-place on the
/// augmented matrix. `None` if any pivot is too small to trust (degenerate).
fn solve_spd_system(g: &mut [[f32; 4]; 4], b: &mut [f32; 4], k: usize) -> Option<Vec<f32>> {
    for i in 0..k {
        // Partial pivot: largest |g[row][i]| at or below the diagonal.
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

        let inv_piv = 1.0 / piv;
        for x in g[i][i..k].iter_mut() {
            *x *= inv_piv;
        }
        b[i] *= inv_piv;

        // Cache the pivot row (Copy) so other rows can be mutated without
        // splitting the borrow.
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
        let c = closest_to_origin(&[Vec4::new(1.0, 0.0, 0.0, 0.0), Vec4::new(2.0, 0.0, 0.0, 0.0)]);
        assert_close(c.point.x, 1.0, 1e-4);
        assert_eq!(c.kept, vec![0]);
    }

    #[test]
    fn triangle_containing_origin() {
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
        // Standard tetrahedron containing origin, embedded at w = 0.
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
        // Symmetric 4-simplex (5 vertices) enclosing the origin.
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
        // Origin outside the triangle, nearest the AB edge.
        let a = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let b = Vec4::new(0.0, 1.0, 0.0, 0.0);
        let c_vert = Vec4::new(2.0, 2.0, 0.0, 0.0);
        let c = closest_to_origin(&[a, b, c_vert]);
        // Closest point on AB to origin is (0.5, 0.5, 0, 0).
        assert_close(c.point.x, 0.5, 1e-4);
        assert_close(c.point.y, 0.5, 1e-4);
        assert_close(c.point.z, 0.0, 1e-4);
        assert_close(c.point.w, 0.0, 1e-4);
        assert_eq!(c.kept.len(), 2);
    }
}
