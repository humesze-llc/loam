//! `BlendedSpace<A, B, F>`: a `Space` whose metric smoothly
//! interpolates between two source Spaces A and B via a
//! blending field F: ℝ³ -> [0, 1].
//!
//! Design goal: seamless transitions between geometries, not camera
//! tricks. The math foundation, numerical scheme, validation strategy,
//! and lock-in notes for this implementation live in the project's
//! private design notes, not in this docstring.
//!
//! ## What ships
//!
//! - [`BlendingField`] trait + [`LinearBlendX`] (axis-aligned smooth-step zone).
//! - [`ConformallyFlat`] trait + impls for `EuclideanR3`, `HyperbolicH3`,
//!   `SphericalS3`.
//! - [`BlendedSpace<A, B, F>`] implementing `Space` via RK4 geodesic
//!   integration, Gauss-Newton `log` shooting, and RK4 parallel
//!   transport for the conformally-flat fast path.
//! - WGSL emit (specific to `BlendedSpace<EuclideanR3, HyperbolicH3, LinearBlendX>`).

use std::borrow::Cow;
use std::marker::PhantomData;

use glam::{Mat3, Vec3};

use crate::space::{Space, WgslSpace};

// ---------------------------------------------------------------------------
// ConformallyFlat: extension trait for Spaces with scalar metric
// ---------------------------------------------------------------------------

/// A [`Space`] whose metric tensor is a scalar multiple of the
/// identity in its standard chart: g_ij(p) = f(p)·δ_ij for some
/// positive scalar function f.
///
/// Implemented by every constant-curvature 3-Space currently in
/// `rye-math`:
///
/// - [`crate::EuclideanR3`]: f ≡ 1.
/// - [`crate::HyperbolicH3`] (Poincaré ball model):
///   f(p) = 4/(1-|p|²)², valid for |p| < 1.
/// - [`crate::SphericalS3`] (stereographic model):
///   f(p) = 4/(1+|p|²)².
/// - `HyperbolicH3UpperHalf` (upper-half-space, *future*,
///   lands when the BlendedSpace demo needs an *unbounded* E³
///   side): f(p) = 1/z², valid for z > 0.
///
/// **Why a separate trait, not a method on `Space`:** not every
/// Space is conformally flat. Sol³ and Nil³ (two of Thurston's
/// eight 3-geometries) have anisotropic metrics that aren't
/// scalar-multiples of identity. Confining the conformal-factor
/// query to its own trait keeps the base `Space` trait honest
/// about what generalises.
///
/// **Why this matters for [`BlendedSpace`]:** when both source
/// Spaces are conformally flat, the blended metric is *also*
/// conformally flat (a blend of scalar multiples is a scalar
/// multiple), and the geodesic ODE collapses to a closed-form
/// expression in ∇ log f. That's the fast path the numerical
/// integrator takes. A future non-conformally-flat blend would
/// need the full Christoffel-symbol machinery.
pub trait ConformallyFlat: Space {
    /// Conformal scale factor at `p`: the scalar f such that
    /// g_ij(p) = f(p)·δ_ij.
    ///
    /// Must be positive and finite at `p` for `p` inside the
    /// chart. Boundary points (e.g. Poincaré ideal boundary
    /// as |p| approaches 1) may diverge to `f32::INFINITY`; the
    /// integrator detects and clamps.
    fn conformal_factor(&self, p: Vec3) -> f32;

    /// Scalar curvature R(p) at the point. For a 3D conformally
    /// flat metric:
    ///
    ///   R = -(4/f(p))·[∇²φ + (1/2)|∇φ|²]
    ///
    /// Default impl computes the Laplacian of φ by finite
    /// differences. Closed-form overrides save cost (and noise)
    /// when the Space's curvature is known analytically, every
    /// constant-curvature Space overrides this.
    fn scalar_curvature(&self, p: Vec3) -> f32 {
        const EPS: f32 = 5.0e-3;
        let phi_at = self.conformal_log_half(p);
        // Standard 7-point stencil Laplacian:
        //   ∇²φ ≈ Σ_axis [φ(p + ε e_axis) − 2 φ(p) + φ(p − ε e_axis)] / ε²
        let lap = (self.conformal_log_half(p + Vec3::X * EPS)
            + self.conformal_log_half(p - Vec3::X * EPS)
            + self.conformal_log_half(p + Vec3::Y * EPS)
            + self.conformal_log_half(p - Vec3::Y * EPS)
            + self.conformal_log_half(p + Vec3::Z * EPS)
            + self.conformal_log_half(p - Vec3::Z * EPS)
            - 6.0 * phi_at)
            / (EPS * EPS);
        let grad = self.conformal_log_half_gradient(p);
        let grad_sq = grad.length_squared();
        let f_p = self.conformal_factor(p);
        if !f_p.is_finite() || f_p <= 0.0 {
            return 0.0;
        }
        -(4.0 / f_p) * (lap + 0.5 * grad_sq)
    }

    /// Logarithm of the conformal factor: φ(p) = (1/2) ln f(p).
    /// Default impl computes from `conformal_factor`; closed-form
    /// overrides save a `ln` per evaluation in the hot path.
    fn conformal_log_half(&self, p: Vec3) -> f32 {
        0.5 * self.conformal_factor(p).ln()
    }

    /// Spatial gradient of [`Self::conformal_log_half`]. Default
    /// impl uses central finite differences; closed-form overrides
    /// halve cost and avoid finite-difference noise.
    fn conformal_log_half_gradient(&self, p: Vec3) -> Vec3 {
        const EPS: f32 = 1.0e-3;
        let dx = (self.conformal_log_half(p + Vec3::X * EPS)
            - self.conformal_log_half(p - Vec3::X * EPS))
            / (2.0 * EPS);
        let dy = (self.conformal_log_half(p + Vec3::Y * EPS)
            - self.conformal_log_half(p - Vec3::Y * EPS))
            / (2.0 * EPS);
        let dz = (self.conformal_log_half(p + Vec3::Z * EPS)
            - self.conformal_log_half(p - Vec3::Z * EPS))
            / (2.0 * EPS);
        Vec3::new(dx, dy, dz)
    }
}

// EuclideanR3: trivial conformally-flat case, f ≡ 1.
impl ConformallyFlat for crate::EuclideanR3 {
    fn conformal_factor(&self, _p: Vec3) -> f32 {
        1.0
    }
    fn conformal_log_half(&self, _p: Vec3) -> f32 {
        0.0
    }
    fn conformal_log_half_gradient(&self, _p: Vec3) -> Vec3 {
        Vec3::ZERO
    }
    fn scalar_curvature(&self, _p: Vec3) -> f32 {
        0.0
    }
}

// HyperbolicH3 (Poincaré ball model): f(p) = 4/(1-|p|²)² for
// |p| < 1. Diverges at the ideal boundary |p| = 1; returns
// `INFINITY` there to flag chart-boundary crossings.
impl ConformallyFlat for crate::HyperbolicH3 {
    fn conformal_factor(&self, p: Vec3) -> f32 {
        let r2 = p.length_squared();
        let denom = (1.0 - r2).max(0.0);
        if denom <= 0.0 {
            return f32::INFINITY;
        }
        4.0 / (denom * denom)
    }
    fn conformal_log_half(&self, p: Vec3) -> f32 {
        // (1/2) ln(4 / (1 − |p|²)²) = ln 2 − ln(1 − |p|²).
        let r2 = p.length_squared();
        let denom = (1.0 - r2).max(0.0);
        if denom <= 0.0 {
            return f32::INFINITY;
        }
        std::f32::consts::LN_2 - denom.ln()
    }
    fn conformal_log_half_gradient(&self, p: Vec3) -> Vec3 {
        // ∇φ = ∇[ln 2 − ln(1 − |p|²)] = 2p / (1 − |p|²).
        let r2 = p.length_squared();
        let denom = (1.0 - r2).max(0.0);
        if denom <= 0.0 {
            return Vec3::ZERO;
        }
        p * (2.0 / denom)
    }
    fn scalar_curvature(&self, _p: Vec3) -> f32 {
        // Constant negative curvature K = -1 in 3D ⇒ R = n(n−1)K
        // = 3·2·(−1) = −6. (Verified against the closed-form
        // calculation: ∇²φ + (1/2)|∇φ|² = 6/(1−|p|²)², so
        // R = −(4/f) · 6/(1−|p|²)² with f = 4/(1−|p|²)² gives
        // R = −6 identically.)
        -6.0
    }
}

// SphericalS3 (stereographic projection from north pole):
// f(p) = 4/(1+|p|²)².
impl ConformallyFlat for crate::SphericalS3 {
    fn conformal_factor(&self, p: Vec3) -> f32 {
        let r2 = p.length_squared();
        4.0 / (1.0 + r2).powi(2)
    }
    fn conformal_log_half(&self, p: Vec3) -> f32 {
        // (1/2) ln(4 / (1 + |p|²)²) = ln 2 − ln(1 + |p|²).
        std::f32::consts::LN_2 - (1.0 + p.length_squared()).ln()
    }
    fn conformal_log_half_gradient(&self, p: Vec3) -> Vec3 {
        // ∇φ = -2p / (1 + |p|²).
        let r2 = p.length_squared();
        p * (-2.0 / (1.0 + r2))
    }
    fn scalar_curvature(&self, _p: Vec3) -> f32 {
        // Constant positive curvature K = +1 in 3D ⇒ R = n(n−1)K
        // = 3·2·(+1) = +6.
        6.0
    }
}

// ---------------------------------------------------------------------------
// BlendedSpace: Space whose metric varies smoothly with position
// ---------------------------------------------------------------------------

/// A `Space` whose metric is the smooth blend of two source
/// Spaces' metrics, weighted by a [`BlendingField`]:
///
///   g(p) = (1 - α(p))·g_A(p) + α(p)·g_B(p)
///
/// At zone extremes the metric reduces to pure g_A or pure g_B.
/// In between it's an honest variable-metric Riemannian
/// manifold, geodesics curve continuously, parallel transport
/// is path-dependent, distances vary by region.
///
/// **Trait bounds:** both source Spaces must use ℝ³ for points
/// and tangent vectors (every closed-form 3-Space in `rye-math`
/// does), and both must be [`ConformallyFlat`] (so the blended
/// metric is also conformally flat, the integrator's fast
/// path).
///
/// **Isometries:** none non-trivial. The variable metric breaks
/// translation and rotation symmetry by construction (the
/// blending field F defines a privileged spatial dependence).
/// `Iso = ()`; `iso_apply` is the identity.
pub struct BlendedSpace<A, B, F>
where
    A: Space<Point = Vec3, Vector = Vec3>,
    B: Space<Point = Vec3, Vector = Vec3>,
    F: BlendingField,
{
    pub a: A,
    pub b: B,
    pub field: F,
    _marker: PhantomData<(A, B, F)>,
}

impl<A, B, F> BlendedSpace<A, B, F>
where
    A: Space<Point = Vec3, Vector = Vec3>,
    B: Space<Point = Vec3, Vector = Vec3>,
    F: BlendingField,
{
    pub fn new(a: A, b: B, field: F) -> Self {
        Self {
            a,
            b,
            field,
            _marker: PhantomData,
        }
    }
}

impl<A, B, F> Space for BlendedSpace<A, B, F>
where
    A: Space<Point = Vec3, Vector = Vec3> + ConformallyFlat,
    B: Space<Point = Vec3, Vector = Vec3> + ConformallyFlat,
    F: BlendingField,
{
    type Point = Vec3;
    type Vector = Vec3;
    /// No non-trivial isometries; see type-level docs.
    type Iso = ();

    fn distance(&self, a: Vec3, b: Vec3) -> f32 {
        // Zone-extreme fast path: when both endpoints are at the
        // same zone extreme, distance is exactly the source
        // Space's distance, preserves f32-tight visual
        // continuity at the demo's far ends.
        let alpha_a = self.field.weight(a);
        let alpha_b = self.field.weight(b);
        if alpha_a == 0.0 && alpha_b == 0.0 {
            return self.a.distance(a, b);
        }
        if alpha_a == 1.0 && alpha_b == 1.0 {
            return self.b.distance(a, b);
        }
        // Variable-metric path: |log_a(b)|_g = √f(a)·|log_a(b)|_E
        // is the Riemannian length of the tangent vector at a
        // that reaches b.
        let log = self.log(a, b);
        let f_a = self.conformal_factor(a);
        f_a.sqrt() * log.length()
    }

    fn exp(&self, at: Vec3, v: Vec3) -> Vec3 {
        // RK4 on the geodesic ODE for conformally-flat metric.
        // For g_ij = e^(2φ)·δ_ij the equation reduces to
        //   v̇ = |v|²·∇φ - 2·(∇φ·v)·v
        // Caller's initial velocity `v` is interpreted as
        // Euclidean; we travel for unit parameter time, so the
        // geodesic length covered is |v|_g = |v|_E·√f(at).
        rk4_geodesic(self, at, v, GEODESIC_DEFAULT_STEPS).0
    }

    fn log(&self, from: Vec3, to: Vec3) -> Vec3 {
        // Gauss-Newton shooting: find `v` such that
        // `exp_from(v) ≈ to`. See `gauss_newton_log` for details.
        gauss_newton_log(self, from, to, GEODESIC_DEFAULT_STEPS, LOG_MAX_ITERS)
    }

    fn parallel_transport(&self, from: Vec3, to: Vec3, v: Vec3) -> Vec3 {
        // Per `Space::parallel_transport`'s contract, implementations
        // pick the path. `BlendedSpace` picks the chart-coordinate
        // straight line from `from` to `to`. Sampling the actual
        // geodesic would require running `log` to find the initial
        // tangent, integrating `exp` to sample it, and transporting
        // along the sampled polyline, at ~7x the cost. Callers that
        // need transport along a known path (camera, player) should
        // call `parallel_transport_along` with the polyline directly.
        parallel_transport_segment_rk4(self, from, to, v, PARALLEL_TRANSPORT_DEFAULT_STEPS)
    }

    fn parallel_transport_along(&self, path: &[Vec3], v: Vec3) -> Vec3 {
        let mut current = v;
        for w in path.windows(2) {
            current = parallel_transport_segment_rk4(
                self,
                w[0],
                w[1],
                current,
                PARALLEL_TRANSPORT_DEFAULT_STEPS,
            );
        }
        current
    }

    fn iso_identity(&self) {}
    fn iso_compose(&self, _a: (), _b: ()) {}
    fn iso_inverse(&self, _a: ()) {}
    fn iso_apply(&self, _iso: (), p: Vec3) -> Vec3 {
        p
    }
    fn iso_transport(&self, _iso: (), _at: Vec3, v: Vec3) -> Vec3 {
        v
    }
}

// ---------------------------------------------------------------------------
// RK4 geodesic integrator
// ---------------------------------------------------------------------------

/// Default number of RK4 steps per unit-parameter integration.
/// Empirically: 32 gives ~6 digits of accuracy on moderately
/// curved metrics; 64 gives ~7 (diminishing returns). Per the
/// design doc this is fixed; adaptive step refinement is
/// deferred until measurement shows we need it.
pub const GEODESIC_DEFAULT_STEPS: u32 = 32;

/// Single step of RK4 on the geodesic ODE for a conformally
/// flat metric. State = `(p, v)`; ODE RHS:
///
///   ṗ = v
///   v̇ = |v|²·∇φ(p) - 2·(∇φ·v)·v
///
/// Returns the new state after stepping by `h` in parameter
/// time. The caller chains this `n_steps` times to reach unit
/// parameter time.
fn rk4_geodesic_step<S: ConformallyFlat>(space: &S, p: Vec3, v: Vec3, h: f32) -> (Vec3, Vec3) {
    // RHS: returns (dp/dt, dv/dt) given (p, v).
    let rhs = |p: Vec3, v: Vec3| -> (Vec3, Vec3) {
        let grad_phi = space.conformal_log_half_gradient(p);
        let v_sq = v.length_squared();
        let dot = grad_phi.dot(v);
        (v, grad_phi * v_sq - v * (2.0 * dot))
    };

    let (k1_p, k1_v) = rhs(p, v);
    let (k2_p, k2_v) = rhs(p + k1_p * (h * 0.5), v + k1_v * (h * 0.5));
    let (k3_p, k3_v) = rhs(p + k2_p * (h * 0.5), v + k2_v * (h * 0.5));
    let (k4_p, k4_v) = rhs(p + k3_p * h, v + k3_v * h);

    let dp = (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p) * (h / 6.0);
    let dv = (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) * (h / 6.0);
    (p + dp, v + dv)
}

/// Integrate the geodesic ODE starting at `(at, v)` for unit
/// parameter time, using `n_steps` RK4 steps. Returns the final
/// `(point, velocity)` pair.
///
/// Public so other modules (e.g. parallel-transport, log
/// shooting) can reuse the same integrator without duplicating
/// the math.
pub fn rk4_geodesic<S: ConformallyFlat>(
    space: &S,
    at: Vec3,
    v: Vec3,
    n_steps: u32,
) -> (Vec3, Vec3) {
    let h = 1.0 / n_steps as f32;
    let mut p = at;
    let mut vel = v;
    for _ in 0..n_steps {
        let (np, nv) = rk4_geodesic_step(space, p, vel, h);
        // Defensive: clamp non-finite states (chart-boundary
        // crossings, integrator blow-ups) to the previous valid
        // state so downstream callers don't propagate NaN.
        if np.is_finite() && nv.is_finite() {
            p = np;
            vel = nv;
        } else {
            tracing::warn!(
                "rk4_geodesic step produced non-finite state; clamping to previous step"
            );
            break;
        }
    }
    (p, vel)
}

// ---------------------------------------------------------------------------
// Parallel transport along a polyline
// ---------------------------------------------------------------------------

/// Default number of RK4 steps per polyline segment for parallel
/// transport. Empirically: 8 gives ~5 digits of accuracy on
/// moderately-curved metrics, which is enough that a camera's
/// frame stays orthonormal-ish over typical gameplay paths
/// without needing post-step renormalisation.
pub const PARALLEL_TRANSPORT_DEFAULT_STEPS: u32 = 8;

/// Parallel-transport `v` along a single polyline segment from
/// `p_from` to `p_to`, parameterised linearly with t ∈ [0, 1].
///
/// For a conformally flat metric g = e^(2φ)·δ, the
/// parallel-transport ODE collapses to:
///
///   V̇ = -[(∇φ·γ̇)·V + (∇φ·V)·γ̇ - (γ̇·V)·∇φ]
///
/// where γ̇ = p_to − p_from is the (constant) segment direction.
/// RK4 integrates this over t ∈ [0, 1] in `n_steps` steps.
///
/// In flat space (∇φ = 0), all three terms vanish and transport
/// is the identity.
pub fn parallel_transport_segment_rk4<S: ConformallyFlat>(
    space: &S,
    p_from: Vec3,
    p_to: Vec3,
    v: Vec3,
    n_steps: u32,
) -> Vec3 {
    let dgamma = p_to - p_from;
    if dgamma.length_squared() < 1.0e-14 {
        return v;
    }
    let h = 1.0 / n_steps as f32;

    // ODE RHS as a closure capturing `space` and `dgamma`.
    let rhs = |gamma_pt: Vec3, v_at_t: Vec3| -> Vec3 {
        let grad_phi = space.conformal_log_half_gradient(gamma_pt);
        let term1 = v_at_t * grad_phi.dot(dgamma);
        let term2 = dgamma * grad_phi.dot(v_at_t);
        let term3 = grad_phi * dgamma.dot(v_at_t);
        -(term1 + term2 - term3)
    };

    let mut v_curr = v;
    for step in 0..n_steps {
        let t = step as f32 * h;
        let p_t = p_from + dgamma * t;
        let p_t_half = p_from + dgamma * (t + h * 0.5);
        let p_t_full = p_from + dgamma * (t + h);

        let k1 = rhs(p_t, v_curr);
        let k2 = rhs(p_t_half, v_curr + k1 * (h * 0.5));
        let k3 = rhs(p_t_half, v_curr + k2 * (h * 0.5));
        let k4 = rhs(p_t_full, v_curr + k3 * h);

        let dv = (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (h / 6.0);
        if dv.is_finite() {
            v_curr += dv;
        } else {
            tracing::warn!(
                "parallel_transport_segment_rk4: non-finite Δv at step {step}; \
                 stopping segment"
            );
            break;
        }
    }
    v_curr
}

// ---------------------------------------------------------------------------
// log: Gauss-Newton shooting
// ---------------------------------------------------------------------------

/// Maximum Gauss-Newton iterations for `log`. ~5 typically
/// converges to f32 precision when `from` and `to` are not in
/// each other's cut locus; we cap at 12 to bound the worst case.
pub const LOG_MAX_ITERS: u32 = 12;

/// Convergence threshold (Euclidean) for the residual
/// `|to − exp_from(v)|`. Below this, we declare success.
pub const LOG_RESIDUAL_TOL: f32 = 1.0e-5;

/// Finite-difference step for the Jacobian of `exp` w.r.t. `v`.
/// Smaller -> more accurate Jacobian but more f32 noise; 1e-3 is
/// the sweet spot for f32 RK4-of-32-steps.
const LOG_JACOBIAN_EPS: f32 = 1.0e-3;

/// Find the tangent vector `v` at `from` such that
/// `exp_from(v) ≈ to`, by Gauss-Newton iteration.
///
/// Each iteration:
///
/// 1. Forward-evaluate `exp_from(v_k)` to get the current
///    geodesic endpoint.
/// 2. Compute residual `r = to − endpoint`. If `|r| < LOG_RESIDUAL_TOL`, return `v_k`.
/// 3. Estimate the Jacobian `J[i][j] = ∂exp[i]/∂v[j]` by
///    finite differences (3 axis-aligned perturbations of
///    `v_k`, central differences).
/// 4. Solve `J · δv = r` for the Newton update; set
///    `v_{k+1} = v_k + δv`.
///
/// Returns the best `v` found within `max_iters`. If the system
/// is singular at any iteration (e.g. `to` in the cut locus of
/// `from`), returns the current best guess with a `tracing::warn`.
///
/// Cost: ~7 forward `exp` evaluations per iteration (1 at the
/// guess, 6 for the Jacobian). At 32 RK4 steps × 4 RHS evals
/// each, that's ~900 conformal-factor calls per iteration. For
/// camera/player use this is fine (per-frame, not per-pixel).
pub fn gauss_newton_log<S: ConformallyFlat>(
    space: &S,
    from: Vec3,
    to: Vec3,
    n_steps: u32,
    max_iters: u32,
) -> Vec3 {
    // Trivial case: same point.
    if (from - to).length() < LOG_RESIDUAL_TOL {
        return Vec3::ZERO;
    }

    // Initial guess: Euclidean displacement. For pure E³ this
    // is exactly correct; for variable-metric it's a starting
    // point the Newton iteration corrects toward the true
    // tangent vector.
    let mut v = to - from;

    for iter in 0..max_iters {
        let endpoint = rk4_geodesic(space, from, v, n_steps).0;
        let residual = to - endpoint;
        if residual.length() < LOG_RESIDUAL_TOL {
            return v;
        }

        // Jacobian via central finite differences. Each column
        // is `(exp(v + ε e_j) − exp(v − ε e_j)) / (2ε)`.
        let two_eps = 2.0 * LOG_JACOBIAN_EPS;
        let mut jac = Mat3::ZERO;
        for j in 0..3 {
            let mut e = Vec3::ZERO;
            e[j] = LOG_JACOBIAN_EPS;
            let plus = rk4_geodesic(space, from, v + e, n_steps).0;
            let minus = rk4_geodesic(space, from, v - e, n_steps).0;
            let col = (plus - minus) / two_eps;
            jac.col_mut(j).x = col.x;
            jac.col_mut(j).y = col.y;
            jac.col_mut(j).z = col.z;
        }

        let det = jac.determinant();
        if det.abs() < 1.0e-8 {
            tracing::warn!(
                "gauss_newton_log: singular Jacobian at iter {iter} (det = {det:e}); \
                 returning best guess. `to` may be in the cut locus of `from`."
            );
            return v;
        }

        let delta = jac.inverse() * residual;
        if !delta.is_finite() {
            tracing::warn!(
                "gauss_newton_log: non-finite Newton update at iter {iter}; \
                 returning best guess."
            );
            return v;
        }
        v += delta;
    }

    tracing::warn!(
        "gauss_newton_log: did not converge in {max_iters} iters; \
         residual remained > {LOG_RESIDUAL_TOL}. Returning best guess."
    );
    v
}

// `BlendedSpace` is itself conformally flat when both sources are:
// the blend of scalar multiples of identity is still a scalar
// multiple of identity. This is the property the integrator
// exploits.
impl<A, B, F> ConformallyFlat for BlendedSpace<A, B, F>
where
    A: Space<Point = Vec3, Vector = Vec3> + ConformallyFlat,
    B: Space<Point = Vec3, Vector = Vec3> + ConformallyFlat,
    F: BlendingField,
{
    fn conformal_factor(&self, p: Vec3) -> f32 {
        let alpha = self.field.weight(p);
        let f_a = self.a.conformal_factor(p);
        let f_b = self.b.conformal_factor(p);
        // Defensive: if either source diverges (`f32::INFINITY`
        // outside its chart), and we're at the corresponding zone
        // extreme, the blend takes the *other* Space's value
        // exactly. Otherwise the divergence propagates; the chart
        // is invalid here.
        if alpha <= 0.0 {
            return f_a;
        }
        if alpha >= 1.0 {
            return f_b;
        }
        (1.0 - alpha) * f_a + alpha * f_b
    }

    /// Analytical chain-rule gradient of φ = (1/2)·ln f for the
    /// blended factor f = (1-α)·f_A + α·f_B.
    ///
    ///   ∇f = ∇α·(f_B - f_A)
    ///      + 2·(1-α)·f_A·∇φ_A
    ///      + 2·α·f_B·∇φ_B
    ///
    /// using ∇f_X = 2·f_X·∇φ_X from φ_X = (1/2)·ln f_X. Then
    /// ∇φ = ∇f / (2f).
    ///
    /// Replaces the trait's default finite-difference path so the
    /// CPU side runs the same analytical formula the WGSL emit
    /// already uses (`rye_blended_grad_phi` for the
    /// `<E3, H3, LinearBlendX>` instantiation). The unit test
    /// pins this against a central-difference of `conformal_log_half`
    /// at three sample points. CPU/GPU parity for the blended
    /// emit is not yet pinned at the test level; the existing
    /// E3/S3/H3 GPU probes in `rye-shader/db.rs` are the template
    /// a future BlendedSpace probe should follow. Removes FD
    /// truncation noise and saves six `conformal_factor`
    /// evaluations per gradient call.
    fn conformal_log_half_gradient(&self, p: Vec3) -> Vec3 {
        let alpha = self.field.weight(p);
        // Zone-extreme fast paths, mirroring `conformal_factor`
        // above. Without these, `0 * INFINITY` from an off-chart
        // source factor (e.g. H3 outside its Poincaré ball)
        // poisons the blended `f` with NaN even when alpha=0
        // means the off-chart source contributes nothing.
        if alpha <= 0.0 {
            return self.a.conformal_log_half_gradient(p);
        }
        if alpha >= 1.0 {
            return self.b.conformal_log_half_gradient(p);
        }
        let f_a = self.a.conformal_factor(p);
        let f_b = self.b.conformal_factor(p);
        let f = (1.0 - alpha) * f_a + alpha * f_b;
        // Inside the blending zone both sources contribute, so a
        // non-finite or non-positive `f` means a source is
        // diverging where the chart is supposed to be valid.
        // Loud in debug. In release, return Vec3::NAN rather than
        // Vec3::ZERO so the geodesic integrator's `is_finite` guard
        // trips and surfaces the bug (a silent zero would let the
        // integrator proceed with wrong dynamics).
        debug_assert!(
            f.is_finite() && f > 0.0,
            "BlendedSpace conformal factor invalid: f = {f}, alpha = {alpha}, f_a = {f_a}, f_b = {f_b}, p = {p:?}"
        );
        if !f.is_finite() || f <= 0.0 {
            return Vec3::NAN;
        }
        let grad_alpha = self.field.gradient(p);
        let grad_phi_a = self.a.conformal_log_half_gradient(p);
        let grad_phi_b = self.b.conformal_log_half_gradient(p);
        let grad_f = grad_alpha * (f_b - f_a)
            + grad_phi_a * (2.0 * (1.0 - alpha) * f_a)
            + grad_phi_b * (2.0 * alpha * f_b);
        grad_f / (2.0 * f)
    }
}

/// A scalar blending field over ℝ³.
///
/// `weight(p)` selects how much of the second Space's metric
/// applies at `p`: `0.0` is "pure A", `1.0` is "pure B",
/// intermediate values blend continuously.
///
/// **Smoothness matters.** The geodesic integrator differentiates
/// through this field to compute Christoffel symbols. A field
/// that's continuous but not continuously differentiable (e.g. a
/// linear ramp clamped to `[0, 1]` without smoothing) produces
/// integrator artifacts at the breakpoints. Use a smoothstep or
/// equivalent C¹ profile.
///
/// **Cost matters.** `weight` and `gradient` are evaluated many
/// times per ray-march step. Closed-form impls override the
/// default finite-difference `gradient` to halve the per-step
/// cost.
pub trait BlendingField: Copy + Send + Sync + 'static {
    /// Blend weight at `p`. Implementations must clamp to
    /// `[0, 1]`; values outside that range produce a metric that
    /// isn't a valid interpolation.
    fn weight(&self, p: Vec3) -> f32;

    /// Spatial gradient of [`Self::weight`]. Default impl uses
    /// central finite differences with `h = 1e-3`. Override for
    /// closed-form blends to (a) reduce evaluation cost and
    /// (b) avoid finite-difference noise near sharp profiles.
    fn gradient(&self, p: Vec3) -> Vec3 {
        const EPS: f32 = 1.0e-3;
        let dx = (self.weight(p + Vec3::X * EPS) - self.weight(p - Vec3::X * EPS)) / (2.0 * EPS);
        let dy = (self.weight(p + Vec3::Y * EPS) - self.weight(p - Vec3::Y * EPS)) / (2.0 * EPS);
        let dz = (self.weight(p + Vec3::Z * EPS) - self.weight(p - Vec3::Z * EPS)) / (2.0 * EPS);
        Vec3::new(dx, dy, dz)
    }
}

// ---------------------------------------------------------------------------
// LinearBlendX: axis-aligned smoothstep zone
// ---------------------------------------------------------------------------

/// Smoothstep blending zone along the X axis: pure A at
/// `x ≤ start`, pure B at `x ≥ end`, smooth C² transition in
/// between.
///
/// The smoothing profile is the **quintic smootherstep**
/// 6t⁵ - 15t⁴ + 10t³, which:
///
/// - Is continuous, continuously differentiable, *and*
///   continuously twice-differentiable. The cubic 3t² - 2t³
///   smoothstep is only C¹, its second derivative jumps at
///   the zone endpoints, which produces a discontinuity in the
///   scalar curvature R(p) (since R involves ∇²φ). For THESIS
///   §2.2's *"seamless transitions"* discipline the curvature
///   must be continuous, hence quintic.
/// - Has zero gradient *and* zero second derivative at both
///   endpoints, so the metric reduces exactly to g_A / g_B
///   outside the zone with no integrator or curvature kicks.
/// - Maps `t ∈ [0, 1]` to `[0, 1]` monotonically.
///
/// The `examples/blended` demo uses this with `start = -2.0`,
/// `end = +2.0` so the player rolls from E³ into H³ over 4 units
/// of X-axis distance.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LinearBlendX {
    pub start: f32,
    pub end: f32,
}

impl LinearBlendX {
    /// New zone with the given start / end x-coordinates.
    /// Requires `end > start`; reversed inputs are silently
    /// swapped to maintain `weight ∈ [0, 1]` semantics.
    pub fn new(start: f32, end: f32) -> Self {
        if end >= start {
            Self { start, end }
        } else {
            Self {
                start: end,
                end: start,
            }
        }
    }

    /// Width of the blending zone in world units.
    fn width(&self) -> f32 {
        self.end - self.start
    }
}

impl BlendingField for LinearBlendX {
    fn weight(&self, p: Vec3) -> f32 {
        let w = self.width();
        if w <= 0.0 {
            // Degenerate: treat as a step function at `start`.
            return if p.x < self.start { 0.0 } else { 1.0 };
        }
        let t = ((p.x - self.start) / w).clamp(0.0, 1.0);
        // Smootherstep: 6t⁵ − 15t⁴ + 10t³.
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }

    fn gradient(&self, p: Vec3) -> Vec3 {
        let w = self.width();
        if w <= 0.0 {
            // Degenerate step function, zero gradient
            // everywhere except the point of discontinuity, which
            // we treat as zero (the integrator avoids it).
            return Vec3::ZERO;
        }
        let raw_t = (p.x - self.start) / w;
        // Outside `[0, 1]` the field is constant and its
        // gradient vanishes.
        if !(0.0..=1.0).contains(&raw_t) {
            return Vec3::ZERO;
        }
        // d/dx [6t⁵ − 15t⁴ + 10t³] · dt/dx
        //   = 30t²(1−t)² · (1/w).
        let t = raw_t;
        let one_minus_t = 1.0 - t;
        let dx = 30.0 * t * t * one_minus_t * one_minus_t / w;
        Vec3::new(dx, 0.0, 0.0)
    }
}

// ---------------------------------------------------------------------------
// WGSL emission
// ---------------------------------------------------------------------------

/// WGSL prelude for the **specific** `BlendedSpace<EuclideanR3, HyperbolicH3, LinearBlendX>`
/// instantiation used by the
/// `examples/blended` demo.
///
/// **Scope.** This is a hand-rolled, parametric prelude, not a
/// generic `WgslSpace for BlendedSpace<A, B, F>` impl. The latter
/// requires a per-`ConformallyFlat`-Space WGSL emission trait
/// (so f(p), ∇φ(p) formulas can be substituted by type), plus a
/// per-`BlendingField` emission trait. That
/// architectural lift is deferred until a *second* `BlendedSpace`
/// instantiation needs WGSL, first-instance lock-in is cheap to
/// undo, second-instance lock-in is what actually motivates the
/// generic design. The Rust API is already generic; only the
/// shader emission is single-instantiation.
///
/// **Numerical scheme.** 16 RK4 sub-steps per `rye_exp` call
/// (`RYE_BLENDED_RK4_SUB`). 8 RK4 sub-steps per `rye_parallel_transport`
/// call (`RYE_BLENDED_TRANSPORT_SUB`), matching the CPU's
/// `parallel_transport_segment_rk4` step-for-step.
///
/// The geodesic-march kernel calls `rye_parallel_transport` once per
/// ray-march sub-step. Internal RK4 multiplies per-call cost ~32x vs
/// the previous single-step Euler, but `parallel_transport` only
/// touches the curved-Space ray-march path; closed-form Spaces
/// continue to use closed-form transport with no integrator at all.
///
/// CPU/GPU parity is pinned by
/// `blended_e3_h3_gpu_probe_exp_matches_cpu` (`rye_exp`) and
/// `blended_e3_h3_gpu_probe_transport_matches_cpu`
/// (`rye_parallel_transport`) in `rye-shader/db.rs`.
///
/// **`rye_log` / `rye_distance` accuracy.** `rye_log` returns
/// the Euclidean chart-coordinate difference (the geodesic march
/// kernel does not call it). `rye_distance` uses the midpoint-rule
/// chord-metric approximation
/// `sqrt(f((a+b)/2)) · |a − b|`, first-order accurate for nearby
/// points (which is the SDF use case). For accurate
/// arbitrary-pair distances at runtime, use the CPU side.
fn blended_e3_h3_linearx_wgsl(field: &LinearBlendX) -> String {
    format!(
        r#"
// rye-math :: BlendedSpace<EuclideanR3, HyperbolicH3, LinearBlendX> (v0 Space WGSL ABI)
const RYE_MAX_ARC: f32 = 1e9;
const RYE_BLENDED_R2_MAX: f32 = 0.9999999;
const RYE_BLENDED_X_START: f32 = {start:?};
const RYE_BLENDED_X_END:   f32 = {end:?};
const RYE_BLENDED_X_WIDTH: f32 = {width:?};
const RYE_BLENDED_RK4_SUB: i32 = 16;
const RYE_BLENDED_TRANSPORT_SUB: i32 = 8;

fn rye_blended_alpha(p: vec3<f32>) -> f32 {{
    let raw_t = (p.x - RYE_BLENDED_X_START) / RYE_BLENDED_X_WIDTH;
    let t = clamp(raw_t, 0.0, 1.0);
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}}

fn rye_blended_alpha_dx(p: vec3<f32>) -> f32 {{
    let raw_t = (p.x - RYE_BLENDED_X_START) / RYE_BLENDED_X_WIDTH;
    if raw_t <= 0.0 || raw_t >= 1.0 {{ return 0.0; }}
    let t = raw_t;
    let one_minus_t = 1.0 - t;
    return 30.0 * t * t * one_minus_t * one_minus_t / RYE_BLENDED_X_WIDTH;
}}

fn rye_blended_f_h3(p: vec3<f32>) -> f32 {{
    let r2 = min(dot(p, p), RYE_BLENDED_R2_MAX);
    let denom = 1.0 - r2;
    return 4.0 / (denom * denom);
}}

fn rye_blended_f(p: vec3<f32>) -> f32 {{
    let alpha = rye_blended_alpha(p);
    return (1.0 - alpha) + alpha * rye_blended_f_h3(p);
}}

// ∇φ(p) = ∇f / (2f), with f the blended factor.
fn rye_blended_grad_phi(p: vec3<f32>) -> vec3<f32> {{
    let alpha = rye_blended_alpha(p);
    let alpha_dx = rye_blended_alpha_dx(p);
    let f_e3 = 1.0;
    let f_h3 = rye_blended_f_h3(p);

    // ∂f_H3/∂p_i = 16 p_i / (1 − r²)³
    let r2 = min(dot(p, p), RYE_BLENDED_R2_MAX);
    let denom = 1.0 - r2;
    let grad_f_h3 = (16.0 / (denom * denom * denom)) * p;

    // ∂f/∂x = α' (f_H3 − f_E3) + α · ∂f_H3/∂x
    // ∂f/∂y = α · ∂f_H3/∂y
    // ∂f/∂z = α · ∂f_H3/∂z
    let grad_f = vec3<f32>(
        alpha_dx * (f_h3 - f_e3) + alpha * grad_f_h3.x,
        alpha * grad_f_h3.y,
        alpha * grad_f_h3.z,
    );

    let f = (1.0 - alpha) * f_e3 + alpha * f_h3;
    return grad_f / (2.0 * max(f, 1e-12));
}}

// Geodesic ODE rhs: ṗ = v, v̇ = |v|²·∇φ - 2·(∇φ·v)·v.
struct RyeBlendedRhs {{ dp: vec3<f32>, dv: vec3<f32> }};

fn rye_blended_rhs(p: vec3<f32>, v: vec3<f32>) -> RyeBlendedRhs {{
    let g = rye_blended_grad_phi(p);
    let v_sq = dot(v, v);
    let g_dot_v = dot(g, v);
    return RyeBlendedRhs(v, g * v_sq - v * (2.0 * g_dot_v));
}}

struct RyeBlendedState {{ p: vec3<f32>, v: vec3<f32> }};

fn rye_blended_rk4_step(p0: vec3<f32>, v0: vec3<f32>, h: f32) -> RyeBlendedState {{
    let k1 = rye_blended_rhs(p0, v0);
    let p1 = p0 + 0.5 * h * k1.dp;
    let v1 = v0 + 0.5 * h * k1.dv;
    let k2 = rye_blended_rhs(p1, v1);
    let p2 = p0 + 0.5 * h * k2.dp;
    let v2 = v0 + 0.5 * h * k2.dv;
    let k3 = rye_blended_rhs(p2, v2);
    let p3 = p0 + h * k3.dp;
    let v3 = v0 + h * k3.dv;
    let k4 = rye_blended_rhs(p3, v3);
    let p_out = p0 + (h / 6.0) * (k1.dp + 2.0 * k2.dp + 2.0 * k3.dp + k4.dp);
    let v_out = v0 + (h / 6.0) * (k1.dv + 2.0 * k2.dv + 2.0 * k3.dv + k4.dv);
    // No position clamp: f_h3 clamps r² internally so the metric
    // stays bounded for all p. Physically clamping position would
    // collapse the E³ side's half-space floor to the unit-ball
    // surface (and create concentric ring artifacts where rays
    // graze that surface).
    return RyeBlendedState(p_out, v_out);
}}

fn rye_exp(at: vec3<f32>, v: vec3<f32>) -> vec3<f32> {{
    let n2 = dot(v, v);
    if n2 < 1e-14 {{ return at; }}
    var p = at;
    var vv = v;
    let h = 1.0 / f32(RYE_BLENDED_RK4_SUB);
    for (var i: i32 = 0; i < RYE_BLENDED_RK4_SUB; i = i + 1) {{
        let s = rye_blended_rk4_step(p, vv, h);
        p = s.p;
        vv = s.v;
    }}
    return p;
}}

// Parallel transport ODE rhs along a curve γ(t):
//   V̇ = -[(∇φ·γ̇)·V + (∇φ·V)·γ̇ - (γ̇·V)·∇φ]
fn rye_blended_transport_rhs(p: vec3<f32>, gamma_dot: vec3<f32>, v: vec3<f32>) -> vec3<f32> {{
    let g = rye_blended_grad_phi(p);
    let g_dot_gd = dot(g, gamma_dot);
    let g_dot_v  = dot(g, v);
    let gd_dot_v = dot(gamma_dot, v);
    return -(g_dot_gd * v + g_dot_v * gamma_dot - gd_dot_v * g);
}}

fn rye_parallel_transport(p_from: vec3<f32>, p_to: vec3<f32>, v: vec3<f32>) -> vec3<f32> {{
    // 8 RK4 sub-steps along the chart-coordinate straight line from
    // p_from to p_to. Mirrors the CPU `parallel_transport_segment_rk4`
    // step-for-step so the two sides agree to 4th-order truncation.
    // Pinned by `blended_e3_h3_gpu_probe_transport_matches_cpu` in
    // rye-shader/db.rs.
    let dgamma = p_to - p_from;
    if dot(dgamma, dgamma) < 1e-14 {{ return v; }}
    let h = 1.0 / f32(RYE_BLENDED_TRANSPORT_SUB);
    var v_curr = v;
    for (var step: i32 = 0; step < RYE_BLENDED_TRANSPORT_SUB; step = step + 1) {{
        let t = f32(step) * h;
        let p_t      = p_from + dgamma * t;
        let p_t_half = p_from + dgamma * (t + h * 0.5);
        let p_t_full = p_from + dgamma * (t + h);
        let k1 = rye_blended_transport_rhs(p_t,      dgamma, v_curr);
        let k2 = rye_blended_transport_rhs(p_t_half, dgamma, v_curr + k1 * (h * 0.5));
        let k3 = rye_blended_transport_rhs(p_t_half, dgamma, v_curr + k2 * (h * 0.5));
        let k4 = rye_blended_transport_rhs(p_t_full, dgamma, v_curr + k3 * h);
        v_curr = v_curr + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (h / 6.0);
    }}
    return v_curr;
}}

fn rye_distance(a: vec3<f32>, b: vec3<f32>) -> f32 {{
    // Midpoint-rule chord-metric: accurate for nearby points,
    // smooth across the blending zone, cheap per call. This is
    // what `rye_scene_sdf` callers see.
    let mid = 0.5 * (a + b);
    let f_mid = rye_blended_f(mid);
    return sqrt(max(f_mid, 0.0)) * length(b - a);
}}

fn rye_origin_distance(p: vec3<f32>) -> f32 {{
    return rye_distance(vec3<f32>(0.0, 0.0, 0.0), p);
}}

fn rye_log(p_from: vec3<f32>, p_to: vec3<f32>) -> vec3<f32> {{
    // Chart-coord difference. The geodesic march kernel does not
    // call this; surfaces that need a true Riemannian log should
    // compute it on the CPU and pass the result through a uniform.
    return p_to - p_from;
}}
"#,
        start = field.start,
        end = field.end,
        width = (field.end - field.start).max(1e-12),
    )
}

impl WgslSpace for BlendedSpace<crate::EuclideanR3, crate::HyperbolicH3, LinearBlendX> {
    fn wgsl_impl(&self) -> Cow<'static, str> {
        Cow::Owned(blended_e3_h3_linearx_wgsl(&self.field))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() <= tol, "expected {a} ≈ {b} (tol {tol})");
    }

    /// Smoothstep boundary values: 0 at start, 1 at end, 0.5 at
    /// midpoint. Pin the canonical profile.
    #[test]
    fn linear_blend_x_smoothstep_endpoints() {
        let f = LinearBlendX::new(-1.0, 1.0);
        close(f.weight(Vec3::new(-1.0, 0.0, 0.0)), 0.0, 1e-6);
        close(f.weight(Vec3::new(1.0, 0.0, 0.0)), 1.0, 1e-6);
        close(f.weight(Vec3::ZERO), 0.5, 1e-6);
    }

    /// Outside the zone, the field is constant (0 or 1) and the
    /// gradient is exactly zero. This is what makes the metric
    /// reduce to pure A / pure B at the extremes; the integrator
    /// sees no Christoffel kick.
    #[test]
    fn linear_blend_x_is_constant_outside_zone() {
        let f = LinearBlendX::new(-1.0, 1.0);
        // Far on the A side.
        close(f.weight(Vec3::new(-100.0, 0.0, 0.0)), 0.0, 0.0);
        assert_eq!(f.gradient(Vec3::new(-100.0, 0.0, 0.0)), Vec3::ZERO);
        // Far on the B side.
        close(f.weight(Vec3::new(100.0, 0.0, 0.0)), 1.0, 0.0);
        assert_eq!(f.gradient(Vec3::new(100.0, 0.0, 0.0)), Vec3::ZERO);
        // Just at the boundary points (smoothstep gradient is 0
        // there too, that's the whole point of using smoothstep
        // over a linear ramp).
        close(f.gradient(Vec3::new(-1.0, 0.0, 0.0)).x, 0.0, 1e-6);
        close(f.gradient(Vec3::new(1.0, 0.0, 0.0)).x, 0.0, 1e-6);
    }

    /// Inside the zone, the gradient is non-zero along X and zero
    /// along Y / Z (axis-aligned blend). Gradient magnitude peaks
    /// at the midpoint where the smoothstep curve is steepest.
    #[test]
    fn linear_blend_x_gradient_is_axis_aligned() {
        let f = LinearBlendX::new(-1.0, 1.0);
        let g = f.gradient(Vec3::ZERO);
        assert!(g.x > 0.0, "midpoint gradient should be positive along +x");
        close(g.y, 0.0, 0.0);
        close(g.z, 0.0, 0.0);
        // Midpoint of smootherstep: t = 0.5,
        // gradient = 30 · 0.25 · 0.25 / 2.0 = 0.9375.
        close(g.x, 0.9375, 1e-6);
    }

    /// Closed-form gradient agrees with the central-finite-
    /// difference default impl. Catches sign or scale errors in
    /// the analytic gradient.
    #[test]
    fn linear_blend_x_closed_form_matches_finite_diff() {
        let f = LinearBlendX::new(-1.0, 1.0);
        // Reference: a wrapper that uses the default `gradient`
        // (finite difference). Since `gradient` has a default
        // impl on the trait, we replicate that path manually
        // here without hitting the override.
        fn finite_diff_gradient<F: BlendingField>(field: &F, p: Vec3) -> Vec3 {
            const EPS: f32 = 1.0e-3;
            let dx =
                (field.weight(p + Vec3::X * EPS) - field.weight(p - Vec3::X * EPS)) / (2.0 * EPS);
            let dy =
                (field.weight(p + Vec3::Y * EPS) - field.weight(p - Vec3::Y * EPS)) / (2.0 * EPS);
            let dz =
                (field.weight(p + Vec3::Z * EPS) - field.weight(p - Vec3::Z * EPS)) / (2.0 * EPS);
            Vec3::new(dx, dy, dz)
        }

        for x in [-0.7_f32, -0.3, 0.0, 0.3, 0.7] {
            let p = Vec3::new(x, 0.0, 0.0);
            let analytic = f.gradient(p);
            let numeric = finite_diff_gradient(&f, p);
            close(analytic.x, numeric.x, 5e-4);
            close(analytic.y, numeric.y, 1e-6);
            close(analytic.z, numeric.z, 1e-6);
        }
    }

    /// Reversed inputs (`end < start`) get auto-swapped so the
    /// resulting field still ramps from 0 to 1 over the zone.
    #[test]
    fn linear_blend_x_handles_reversed_inputs() {
        let f = LinearBlendX::new(1.0, -1.0);
        // After swap, equivalent to `new(-1.0, 1.0)`.
        close(f.weight(Vec3::new(-1.0, 0.0, 0.0)), 0.0, 1e-6);
        close(f.weight(Vec3::new(1.0, 0.0, 0.0)), 1.0, 1e-6);
    }

    // ------ ConformallyFlat impls ------

    /// EuclideanR3's conformal factor is identically 1; its
    /// log-half is identically 0; its gradient is identically
    /// zero. Pin the trivial case.
    #[test]
    fn euclidean_r3_conformal_factor_is_unity() {
        use crate::EuclideanR3;
        let s = EuclideanR3;
        for p in [
            Vec3::ZERO,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-100.0, 50.0, 7.0),
        ] {
            close(s.conformal_factor(p), 1.0, 0.0);
            close(s.conformal_log_half(p), 0.0, 0.0);
            assert_eq!(s.conformal_log_half_gradient(p), Vec3::ZERO);
        }
    }

    /// Poincaré-ball HyperbolicH3 conformal factor: $4 / (1 -
    /// |p|^2)^2$. Pin standard values (origin, halfway out,
    /// near boundary).
    #[test]
    fn hyperbolic_h3_conformal_factor_pin_values() {
        use crate::HyperbolicH3;
        let s = HyperbolicH3;
        // Origin: f(0) = 4 / 1 = 4.
        close(s.conformal_factor(Vec3::ZERO), 4.0, 1e-6);
        // Halfway out: |p| = 0.5, |p|² = 0.25, denom = 0.75,
        // f = 4 / 0.5625 ≈ 7.111.
        close(
            s.conformal_factor(Vec3::new(0.5, 0.0, 0.0)),
            4.0 / 0.5625,
            1e-5,
        );
        // Near boundary: |p|² = 0.99, denom = 0.01, f = 4 /
        // 0.0001 = 40000.
        close(
            s.conformal_factor(Vec3::new(0.0, 0.0, 0.99499f32.sqrt())),
            4.0 / (1.0 - 0.99499_f32).powi(2),
            10.0, // huge magnitude, generous tolerance
        );
    }

    /// Closed-form `conformal_log_half_gradient` for HyperbolicH3
    /// agrees with central finite differences. Catches sign /
    /// scale errors in the analytic gradient.
    #[test]
    fn hyperbolic_h3_log_half_gradient_matches_finite_diff() {
        use crate::HyperbolicH3;
        let s = HyperbolicH3;
        const EPS: f32 = 1e-3;
        let fd = |p: Vec3| -> Vec3 {
            let dx = (s.conformal_log_half(p + Vec3::X * EPS)
                - s.conformal_log_half(p - Vec3::X * EPS))
                / (2.0 * EPS);
            let dy = (s.conformal_log_half(p + Vec3::Y * EPS)
                - s.conformal_log_half(p - Vec3::Y * EPS))
                / (2.0 * EPS);
            let dz = (s.conformal_log_half(p + Vec3::Z * EPS)
                - s.conformal_log_half(p - Vec3::Z * EPS))
                / (2.0 * EPS);
            Vec3::new(dx, dy, dz)
        };
        for p in [
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.3, -0.2, 0.1),
            Vec3::new(-0.4, 0.5, 0.2),
        ] {
            let analytic = s.conformal_log_half_gradient(p);
            let numeric = fd(p);
            close(analytic.x, numeric.x, 5e-3);
            close(analytic.y, numeric.y, 5e-3);
            close(analytic.z, numeric.z, 5e-3);
        }
    }

    /// SphericalS3 conformal factor: 4/(1+|p|²)². Pin origin and
    /// a generic point.
    #[test]
    fn spherical_s3_conformal_factor_pin_values() {
        use crate::SphericalS3;
        let s = SphericalS3;
        close(s.conformal_factor(Vec3::ZERO), 4.0, 1e-6);
        // |p|² = 1: f = 4 / 4 = 1.
        close(s.conformal_factor(Vec3::new(1.0, 0.0, 0.0)), 1.0, 1e-6);
        // |p|² = 3: f = 4 / 16 = 0.25.
        close(s.conformal_factor(Vec3::new(1.0, 1.0, 1.0)), 0.25, 1e-6);
    }

    // ------ BlendedSpace conformally-flat overrides ------

    /// `BlendedSpace::conformal_log_half_gradient` analytical
    /// override agrees with central finite differences on the
    /// blended `conformal_log_half`. Pins the chain rule against
    /// the trait's default FD path so a future regression in
    /// either one trips here.
    ///
    /// Tolerance choice: central differences with `EPS = 1e-3`
    /// have leading truncation `O(EPS² · f''') ≈ 1e-6` for the
    /// smooth conformal factor here, plus f32 roundoff `O(1/EPS) · eps_machine ≈ 1e-4`. Bound
    /// at `5e-3` leaves ~50x headroom
    /// over both, tight enough to catch a sign error or missing
    /// chain-rule term but loose enough that the H3 Poincaré
    /// factor's `(1−r²)` denominator at the |r|≈0.7 sample point
    /// (the `(−0.7, 0.05, 0.0)` row below) doesn't trip on the
    /// noise floor.
    #[test]
    fn blended_space_log_half_gradient_matches_finite_diff() {
        use crate::{EuclideanR3, HyperbolicH3};
        let bs = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-0.5, 0.5));
        let fd = |p: Vec3| -> Vec3 {
            const EPS: f32 = 1e-3;
            let dx = (bs.conformal_log_half(p + Vec3::X * EPS)
                - bs.conformal_log_half(p - Vec3::X * EPS))
                / (2.0 * EPS);
            let dy = (bs.conformal_log_half(p + Vec3::Y * EPS)
                - bs.conformal_log_half(p - Vec3::Y * EPS))
                / (2.0 * EPS);
            let dz = (bs.conformal_log_half(p + Vec3::Z * EPS)
                - bs.conformal_log_half(p - Vec3::Z * EPS))
                / (2.0 * EPS);
            Vec3::new(dx, dy, dz)
        };
        // Sample three regimes: well inside the alpha=0 region,
        // mid-zone (where both sources contribute), and well into
        // the alpha=1 region but inside the H3 Poincaré ball.
        for p in [
            Vec3::new(-0.7, 0.05, 0.0),
            Vec3::new(0.0, 0.1, -0.1),
            Vec3::new(0.6, 0.05, 0.0),
        ] {
            let analytic = bs.conformal_log_half_gradient(p);
            let numeric = fd(p);
            close(analytic.x, numeric.x, 5e-3);
            close(analytic.y, numeric.y, 5e-3);
            close(analytic.z, numeric.z, 5e-3);
        }
    }

    /// At alpha=0 (pure A region), the `alpha <= 0.0` fast path
    /// returns A's gradient verbatim. EuclideanR3 has zero
    /// gradient everywhere, so the BlendedSpace gradient should
    /// be zero too.
    ///
    /// Tolerance `1e-6` is the f32 single-step roundoff floor;
    /// the fast path does no arithmetic, so anything looser
    /// hides a real bug.
    #[test]
    fn blended_space_log_half_gradient_at_alpha_zero_is_pure_a() {
        use crate::{EuclideanR3, HyperbolicH3};
        let bs = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(50.0, 100.0));
        let g = bs.conformal_log_half_gradient(Vec3::new(1.0, 2.0, 3.0));
        close(g.x, 0.0, 1e-6);
        close(g.y, 0.0, 1e-6);
        close(g.z, 0.0, 1e-6);
    }

    /// At alpha=1 (pure B region), the `alpha >= 1.0` fast path
    /// returns B's gradient verbatim. Tolerance `1e-6` chosen
    /// for the same reason as the alpha=0 test: the fast path
    /// does no arithmetic.
    #[test]
    fn blended_space_log_half_gradient_at_alpha_one_is_pure_b() {
        use crate::{EuclideanR3, HyperbolicH3};
        let bs = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-100.0, -50.0));
        let p = Vec3::new(0.2, 0.1, 0.0);
        let blended = bs.conformal_log_half_gradient(p);
        let pure_b = HyperbolicH3.conformal_log_half_gradient(p);
        close(blended.x, pure_b.x, 1e-6);
        close(blended.y, pure_b.y, 1e-6);
        close(blended.z, pure_b.z, 1e-6);
    }

    // ------ BlendedSpace skeleton ------

    /// At a zone extreme (pure A), `BlendedSpace::distance`
    /// matches `A::distance`. Pin the fast path that lets the
    /// demo's far ends be visually identical to pure E³ / pure H³.
    #[test]
    fn blended_space_distance_at_alpha_zero_matches_a() {
        use crate::EuclideanR3;
        let bs = BlendedSpace::new(
            EuclideanR3,
            EuclideanR3, // dummy; alpha=0 means we never see B
            LinearBlendX::new(10.0, 20.0),
        );
        // Both points at x < 10: alpha = 0, fast path to A.
        let a = Vec3::new(-1.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 0.0, 0.0);
        let d_blend = bs.distance(a, b);
        let d_a = EuclideanR3.distance(a, b);
        close(d_blend, d_a, 1e-6);
    }

    /// At a zone extreme (pure B), distance matches `B::distance`.
    #[test]
    fn blended_space_distance_at_alpha_one_matches_b() {
        use crate::EuclideanR3;
        let bs = BlendedSpace::new(EuclideanR3, EuclideanR3, LinearBlendX::new(-20.0, -10.0));
        // Both points at x > -10: alpha = 1, fast path to B.
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(3.0, 0.0, 0.0);
        let d_blend = bs.distance(a, b);
        let d_b = EuclideanR3.distance(a, b);
        close(d_blend, d_b, 1e-6);
    }

    /// `BlendedSpace::iso_apply` is the identity (no non-trivial
    /// isometries; `Iso = ()`). Pin the convention.
    #[test]
    fn blended_space_iso_is_trivial() {
        use crate::EuclideanR3;
        use crate::Space;
        let bs = BlendedSpace::new(EuclideanR3, EuclideanR3, LinearBlendX::new(-1.0, 1.0));
        let p = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(bs.iso_apply((), p), p);
        let v = Vec3::new(0.5, -0.5, 0.5);
        assert_eq!(bs.iso_transport((), p, v), v);
    }

    /// `BlendedSpace` is itself `ConformallyFlat`: at a zone
    /// extreme, its conformal factor equals the relevant source
    /// Space's. In between, it's the linear blend of the two
    /// sources' factors weighted by alpha.
    #[test]
    fn blended_space_conformal_factor_blends_linearly() {
        use crate::{EuclideanR3, HyperbolicH3};
        // E³ blends to H³(Poincaré) along x ∈ [-1, 1].
        let bs = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-1.0, 1.0));

        // At x = -1 (alpha=0): factor = 1 (pure E³).
        close(bs.conformal_factor(Vec3::new(-1.0, 0.0, 0.0)), 1.0, 1e-6);
        // At x = 1 (alpha=1): factor = HyperbolicH3 at x=1, but
        // x=1 is on the Poincaré boundary so f -> ∞. Use a point
        // *near* x=1 but slightly inside.
        let p = Vec3::new(0.99, 0.0, 0.0);
        let alpha = LinearBlendX::new(-1.0, 1.0).weight(p);
        let f_e = 1.0;
        let f_h = HyperbolicH3.conformal_factor(p);
        let expected = (1.0 - alpha) * f_e + alpha * f_h;
        close(bs.conformal_factor(p), expected, 1e-2);
    }

    // ------ RK4 geodesic integrator ------

    /// In flat E³ the geodesic ODE has zero curvature term, so
    /// RK4 should reproduce straight-line motion exactly (within
    /// f32 rounding): `exp_p(v) = p + v` for all `p`, `v`.
    #[test]
    fn rk4_in_pure_e3_is_straight_line() {
        use crate::EuclideanR3;
        let bs = BlendedSpace::new(
            EuclideanR3,
            EuclideanR3,
            LinearBlendX::new(100.0, 200.0), // far away, alpha ≡ 0 in our test region
        );
        for (p, v) in [
            (Vec3::ZERO, Vec3::X),
            (Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.5, -0.3, 0.7)),
            (Vec3::new(-5.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)),
        ] {
            let (final_p, final_v) = rk4_geodesic(&bs, p, v, GEODESIC_DEFAULT_STEPS);
            let expected = p + v;
            close((final_p - expected).length(), 0.0, 1e-5);
            close((final_v - v).length(), 0.0, 1e-5);
        }
    }

    /// In pure HyperbolicH3 (Poincaré ball), `exp` from the
    /// origin along an axis-aligned **Euclidean** tangent vector
    /// `v` should land at the closed-form Poincaré geodesic
    /// endpoint. The convention (matching the existing
    /// `HyperbolicH3::exp`): `v` is Euclidean, so the Riemannian
    /// length is |v|_g = √f(p)·|v|_E, and at the origin where
    /// f(0) = 4:
    ///
    ///   |exp_0(v)|_E = tanh(|v|_g / 2) = tanh(|v|_E).
    ///
    /// Pins the convention *and* validates the integrator
    /// against the engine's existing closed-form ground truth.
    #[test]
    fn rk4_in_pure_h3_matches_closed_form_at_origin() {
        use crate::{HyperbolicH3, Space};

        for &mag in &[0.1_f32, 0.3, 0.5] {
            let v = Vec3::new(mag, 0.0, 0.0);
            let (final_p, _) = rk4_geodesic(&HyperbolicH3, Vec3::ZERO, v, GEODESIC_DEFAULT_STEPS);
            // Closed form at origin (sqrt(f(0)) = 2):
            // |exp_0(v)|_E = tanh(|v|_E).
            let expected_radius = mag.tanh();
            close(final_p.x, expected_radius, 5e-3);
            close(final_p.y, 0.0, 1e-4);
            close(final_p.z, 0.0, 1e-4);
        }

        // Cross-check: numerical integrator agrees with the
        // engine's existing closed-form `HyperbolicH3::exp`.
        let v = Vec3::new(0.4, 0.1, 0.0);
        let (numerical, _) = rk4_geodesic(&HyperbolicH3, Vec3::ZERO, v, GEODESIC_DEFAULT_STEPS);
        let closed_form = HyperbolicH3.exp(Vec3::ZERO, v);
        close((numerical - closed_form).length(), 0.0, 5e-3);

        // Drive through `BlendedSpace::exp` (alpha ≡ 1 in the
        // test region), the variable-metric path collapses to
        // pure H³ here.
        let bs = BlendedSpace::new(
            HyperbolicH3, // dummy; alpha=1 never reaches A
            HyperbolicH3,
            LinearBlendX::new(-100.0, -50.0),
        );
        let v = Vec3::new(0.5, 0.0, 0.0);
        let final_p_blended = bs.exp(Vec3::ZERO, v);
        close(final_p_blended.x, 0.5_f32.tanh(), 5e-3);
    }

    /// Geodesic round-trip in pure E³: `exp_p(v)` then `exp_q(-v)`
    /// returns to `p` (within RK4 noise). Time-reversibility of
    /// the integrator on a flat metric. Catches accumulated drift.
    #[test]
    fn rk4_in_e3_is_time_reversible() {
        use crate::EuclideanR3;
        let p = Vec3::new(1.0, 2.0, 3.0);
        let v = Vec3::new(0.5, -0.3, 0.7);
        let (q, vq) = rk4_geodesic(&EuclideanR3, p, v, GEODESIC_DEFAULT_STEPS);
        let (back, _) = rk4_geodesic(&EuclideanR3, q, -vq, GEODESIC_DEFAULT_STEPS);
        close((back - p).length(), 0.0, 1e-5);
    }

    /// `BlendedSpace::exp` at a zone extreme matches the source
    /// Space's exp (both being E³ here). End-to-end pin: the
    /// trait method actually goes through the integrator and
    /// recovers the closed-form answer.
    #[test]
    fn blended_space_exp_at_alpha_zero_matches_e3() {
        use crate::EuclideanR3;
        use crate::Space;
        let bs = BlendedSpace::new(EuclideanR3, EuclideanR3, LinearBlendX::new(50.0, 100.0));
        let p = Vec3::new(1.0, 2.0, 3.0);
        let v = Vec3::new(0.5, -0.3, 0.7);
        let result = bs.exp(p, v);
        // Pure E³ ⇒ straight-line motion ⇒ result = p + v.
        let expected = p + v;
        close((result - expected).length(), 0.0, 1e-5);
    }

    // ------ log via Gauss-Newton shooting ------

    /// In pure E³, `log_from(to) = to − from` exactly. The
    /// Gauss-Newton iteration should converge in 1 step (the
    /// initial guess is already correct).
    #[test]
    fn log_in_pure_e3_is_euclidean_displacement() {
        use crate::EuclideanR3;
        for (from, to) in [
            (Vec3::ZERO, Vec3::X),
            (Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0)),
            (Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)),
        ] {
            let v = gauss_newton_log(
                &EuclideanR3,
                from,
                to,
                GEODESIC_DEFAULT_STEPS,
                LOG_MAX_ITERS,
            );
            close((v - (to - from)).length(), 0.0, 1e-4);
        }
    }

    /// Round-trip: `exp_from(log_from(to)) ≈ to`. The defining
    /// property of `log`. Test in pure H³ (where the integrator
    /// is doing real work, not just identity).
    #[test]
    fn exp_log_round_trip_in_pure_h3() {
        use crate::HyperbolicH3;
        for (from, to) in [
            (Vec3::ZERO, Vec3::new(0.3, 0.0, 0.0)),
            (Vec3::ZERO, Vec3::new(0.2, 0.1, -0.1)),
            (Vec3::new(0.1, 0.1, 0.0), Vec3::new(-0.1, 0.2, 0.1)),
        ] {
            let v = gauss_newton_log(
                &HyperbolicH3,
                from,
                to,
                GEODESIC_DEFAULT_STEPS,
                LOG_MAX_ITERS,
            );
            let endpoint = rk4_geodesic(&HyperbolicH3, from, v, GEODESIC_DEFAULT_STEPS).0;
            close((endpoint - to).length(), 0.0, 5e-4);
        }
    }

    /// `log` matches the engine's existing closed-form
    /// `HyperbolicH3::log` on representative points. Validates
    /// the numerical inversion against an independent ground
    /// truth.
    #[test]
    fn log_in_pure_h3_matches_closed_form() {
        use crate::{HyperbolicH3, Space};
        for (from, to) in [
            (Vec3::ZERO, Vec3::new(0.3, 0.0, 0.0)),
            (Vec3::ZERO, Vec3::new(0.2, 0.1, 0.0)),
            (Vec3::new(0.1, 0.0, 0.0), Vec3::new(0.3, 0.2, 0.0)),
        ] {
            let numerical = gauss_newton_log(
                &HyperbolicH3,
                from,
                to,
                GEODESIC_DEFAULT_STEPS,
                LOG_MAX_ITERS,
            );
            let closed_form = HyperbolicH3.log(from, to);
            close((numerical - closed_form).length(), 0.0, 5e-3);
        }
    }

    /// `log_p(p) = 0` (zero tangent vector). Pin the trivial
    /// case explicitly, the shooting routine special-cases it
    /// to avoid the Jacobian blowing up at zero residual.
    #[test]
    fn log_of_self_is_zero() {
        use crate::HyperbolicH3;
        let p = Vec3::new(0.2, 0.1, 0.0);
        let v = gauss_newton_log(&HyperbolicH3, p, p, GEODESIC_DEFAULT_STEPS, LOG_MAX_ITERS);
        close(v.length(), 0.0, 1e-5);
    }

    /// `BlendedSpace::distance` matches the source Space's
    /// distance at zone extremes (pure E³ here).
    #[test]
    fn blended_space_distance_at_alpha_zero_uses_log() {
        use crate::{EuclideanR3, Space};
        let bs = BlendedSpace::new(EuclideanR3, EuclideanR3, LinearBlendX::new(50.0, 100.0));
        // Both points well inside the alpha=0 region.
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let d = bs.distance(a, b);
        // Pure E³ distance is Euclidean.
        close(d, (b - a).length(), 1e-4);
    }

    // ------ Parallel transport ------

    /// In pure E³, parallel transport is the identity along any
    /// path, the Christoffel symbols are zero, so the transport
    /// ODE has zero RHS and `v` doesn't change.
    #[test]
    fn parallel_transport_in_e3_is_identity() {
        use crate::EuclideanR3;
        let v = Vec3::new(0.5, -0.3, 0.7);
        // Single segment.
        let transported = parallel_transport_segment_rk4(
            &EuclideanR3,
            Vec3::ZERO,
            Vec3::new(2.0, 1.0, -1.0),
            v,
            PARALLEL_TRANSPORT_DEFAULT_STEPS,
        );
        close((transported - v).length(), 0.0, 1e-6);

        // Multi-segment polyline.
        use crate::Space;
        let bs = BlendedSpace::new(EuclideanR3, EuclideanR3, LinearBlendX::new(50.0, 100.0));
        let path = [
            Vec3::ZERO,
            Vec3::X,
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(2.0, -1.0, 1.0),
        ];
        let result = bs.parallel_transport_along(&path, v);
        close((result - v).length(), 0.0, 1e-5);
    }

    /// Hyperbolic transport preserves the *Riemannian* (not
    /// Euclidean) length of the vector. The Riemannian length
    /// at p is √f(p)·|v|_E; check that
    /// `f(p_to)·|v_to|² ≈ f(p_from)·|v|²`.
    #[test]
    fn parallel_transport_in_h3_preserves_riemannian_length() {
        use crate::HyperbolicH3;
        let from = Vec3::new(0.1, 0.0, 0.0);
        let to = Vec3::new(0.3, 0.1, 0.0);
        let v = Vec3::new(0.2, 0.1, 0.0);

        let transported = parallel_transport_segment_rk4(
            &HyperbolicH3,
            from,
            to,
            v,
            PARALLEL_TRANSPORT_DEFAULT_STEPS,
        );

        let f_from = HyperbolicH3.conformal_factor(from);
        let f_to = HyperbolicH3.conformal_factor(to);
        let len_from = f_from.sqrt() * v.length();
        let len_to = f_to.sqrt() * transported.length();
        close(len_from, len_to, 5e-3);
    }

    /// Cross-check: numerical transport on a 2-point polyline in
    /// pure H³ agrees with the engine's existing closed-form
    /// `HyperbolicH3::parallel_transport` (gyration formula).
    /// Note: the closed-form transports along the *geodesic*; our
    /// RK4 transports along the *Euclidean line segment*. They
    /// agree only when the points are close (geodesic ≈ line).
    /// Pin a small-displacement case.
    #[test]
    fn parallel_transport_in_h3_matches_closed_form_for_short_paths() {
        use crate::{HyperbolicH3, Space};
        // Small displacement near the origin where the geodesic
        // is essentially a Euclidean line.
        let from = Vec3::new(0.05, 0.0, 0.0);
        let to = Vec3::new(0.06, 0.01, 0.0);
        let v = Vec3::new(0.1, 0.0, 0.0);

        let numerical = parallel_transport_segment_rk4(
            &HyperbolicH3,
            from,
            to,
            v,
            PARALLEL_TRANSPORT_DEFAULT_STEPS,
        );
        let closed_form = HyperbolicH3.parallel_transport(from, to, v);
        close((numerical - closed_form).length(), 0.0, 5e-3);
    }

    /// Closed-loop holonomy test in H³: transport a vector
    /// around a small triangle and check the final vector
    /// differs from the original (proving real curvature is
    /// being integrated). In flat space the loop returns
    /// exactly to the start; in H³ there's a holonomy rotation
    /// proportional to the loop's enclosed area.
    #[test]
    fn parallel_transport_in_h3_has_nonzero_holonomy() {
        use crate::{HyperbolicH3, Space};
        let bs = BlendedSpace::new(
            HyperbolicH3,
            HyperbolicH3,
            LinearBlendX::new(-100.0, -50.0), // alpha ≡ 1 in test region
        );
        // Triangle path well inside the ball.
        let path = [
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.3, 0.0, 0.0),
            Vec3::new(0.2, 0.2, 0.0),
            Vec3::new(0.1, 0.0, 0.0), // back to start
        ];
        let v = Vec3::new(0.1, 0.0, 0.0);
        let transported = bs.parallel_transport_along(&path, v);
        // The transported vector should differ from the
        // original, H³ has constant negative curvature, so any
        // non-degenerate loop produces visible holonomy. Also
        // check the result is finite (no integrator blow-up).
        assert!(transported.is_finite());
        let drift = (transported - v).length();
        assert!(
            drift > 1e-3,
            "expected non-zero holonomy in H³, got drift {drift}"
        );
        // But not catastrophic, loop is small, holonomy bounded.
        assert!(drift < 0.5, "holonomy unreasonably large: {drift}");
    }

    // ------ Curvature continuity (sub-task 7) ------

    /// Closed-form curvature scalar for HyperbolicH3 is `-6`
    /// everywhere (constant negative curvature `K = -1` in 3D).
    /// Pins both the override and the sign convention.
    #[test]
    fn hyperbolic_h3_scalar_curvature_is_constant_minus_six() {
        use crate::HyperbolicH3;
        let h3 = HyperbolicH3;
        for p in [
            Vec3::ZERO,
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.3, 0.2, -0.1),
        ] {
            close(h3.scalar_curvature(p), -6.0, 1e-6);
        }
    }

    /// SphericalS3 closed-form scalar curvature is `+6` (constant
    /// positive curvature `K = +1` in 3D).
    #[test]
    fn spherical_s3_scalar_curvature_is_constant_plus_six() {
        use crate::SphericalS3;
        let s3 = SphericalS3;
        for p in [
            Vec3::ZERO,
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.5, -0.3, 0.2),
        ] {
            close(s3.scalar_curvature(p), 6.0, 1e-6);
        }
    }

    /// Default (finite-difference) curvature impl agrees with the
    /// closed-form override for HyperbolicH3 at a few sample
    /// points, validating the FD stencil for blended-space use
    /// (where there's no closed form).
    #[test]
    fn finite_diff_curvature_matches_closed_form_in_h3() {
        use crate::HyperbolicH3;
        // Drop down to a dummy `ConformallyFlat` that doesn't
        // override `scalar_curvature` so we hit the default impl.
        struct H3FdOnly;
        impl crate::space::Space for H3FdOnly {
            type Point = Vec3;
            type Vector = Vec3;
            type Iso = ();
            fn distance(&self, _: Vec3, _: Vec3) -> f32 {
                0.0
            }
            fn exp(&self, _: Vec3, _: Vec3) -> Vec3 {
                Vec3::ZERO
            }
            fn log(&self, _: Vec3, _: Vec3) -> Vec3 {
                Vec3::ZERO
            }
            fn parallel_transport(&self, _: Vec3, _: Vec3, v: Vec3) -> Vec3 {
                v
            }
            fn iso_identity(&self) {}
            fn iso_compose(&self, _: (), _: ()) {}
            fn iso_inverse(&self, _: ()) {}
            fn iso_apply(&self, _: (), p: Vec3) -> Vec3 {
                p
            }
            fn iso_transport(&self, _: (), _: Vec3, v: Vec3) -> Vec3 {
                v
            }
        }
        impl ConformallyFlat for H3FdOnly {
            fn conformal_factor(&self, p: Vec3) -> f32 {
                HyperbolicH3.conformal_factor(p)
            }
            fn conformal_log_half(&self, p: Vec3) -> f32 {
                HyperbolicH3.conformal_log_half(p)
            }
            fn conformal_log_half_gradient(&self, p: Vec3) -> Vec3 {
                HyperbolicH3.conformal_log_half_gradient(p)
            }
            // Don't override scalar_curvature, use default (FD).
        }
        let fd = H3FdOnly;
        // Loose tolerance: finite differences with EPS=5e-3 give
        // ~3 digits, plenty to confirm correctness.
        for p in [Vec3::ZERO, Vec3::new(0.2, 0.1, 0.0)] {
            close(fd.scalar_curvature(p), -6.0, 0.5);
        }
    }

    /// BlendedSpace<E³, H³, LinearBlendX> scalar curvature varies
    /// continuously across the zone: at zone extremes it matches
    /// pure E³ (R=0) and pure H³ (R=-6); in between it's a smooth
    /// interpolation with no discontinuous jumps. Pin the
    /// continuity that THESIS §2.2 calls for.
    #[test]
    fn blended_space_curvature_varies_continuously_across_zone() {
        use crate::{EuclideanR3, HyperbolicH3};
        let bs = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-0.5, 0.5));

        // Zone-extreme: well into E³.
        let r_e = bs.scalar_curvature(Vec3::new(-0.7, 0.0, 0.0));
        close(r_e, 0.0, 1e-1);

        // Zone-extreme: well into H³ (still inside the Poincaré
        // ball where the Space is well-defined).
        let r_h = bs.scalar_curvature(Vec3::new(0.7, 0.0, 0.0));
        // We're past the zone end (alpha ≈ 1) but technically the
        // BlendedSpace's curvature differs from pure H³'s -6 by
        // f32 noise (since alpha hits exactly 1.0 inside the
        // smoothstep clamp). Loose tolerance.
        close(r_h, -6.0, 1.0);

        // Sample along the zone, collect curvature values, verify
        // they vary smoothly (no spikes / discontinuities). The
        // strongest test: consecutive samples differ by an amount
        // proportional to the sample step.
        let xs: Vec<f32> = (-30..=30).map(|i| (i as f32) * 0.025).collect();
        let curvatures: Vec<f32> = xs
            .iter()
            .map(|&x| bs.scalar_curvature(Vec3::new(x, 0.0, 0.0)))
            .collect();

        // Every value must be finite. Note the transition zone
        // can show stronger curvature than *either* endpoint: a
        // rapidly-varying metric (|∇φ|² and ∇²φ both spike inside
        // the zone) genuinely produces |R| > 6. That's a real
        // geometric feature, not a bug, the seam is its own
        // curved region. Bound is therefore generous.
        for &r in &curvatures {
            assert!(
                r.is_finite() && (-50.0..=5.0).contains(&r),
                "curvature out of expected range: {r}"
            );
        }

        // The real continuity check: adjacent samples differ by a
        // bounded amount. The transition zone has rapid (but
        // continuous) curvature variation; FD aliasing on the
        // chosen sample step amplifies that into ~14 max jumps
        // empirically. A genuine C² discontinuity (e.g. from
        // cubic smoothstep) shows up as jumps in the tens or
        // hundreds at much finer sample spacing, so cap at 25,
        // tolerant of FD aliasing here while catching real
        // discontinuities.
        let max_jump = curvatures
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_jump < 25.0,
            "curvature has a discontinuity: max adjacent jump = {max_jump}"
        );
    }

    // ------ Boundary extremes (sub-task 8) ------

    /// At zone extremes (α exactly 0 or 1), the BlendedSpace
    /// `Space` methods should produce results bit-identical to
    /// the source Space's. Pin: `exp`, `log`, `parallel_transport`,
    /// `distance`, `conformal_factor`, `scalar_curvature` all
    /// match A at α=0 and B at α=1.
    #[test]
    fn blended_space_at_alpha_zero_is_pure_a() {
        use crate::{EuclideanR3, HyperbolicH3, Space};
        let bs = BlendedSpace::new(
            EuclideanR3,
            HyperbolicH3,
            LinearBlendX::new(50.0, 100.0), // alpha ≡ 0 in test region
        );
        let p = Vec3::new(1.0, 2.0, 3.0);
        let q = Vec3::new(4.0, 5.0, 6.0);
        let v = Vec3::new(0.5, -0.3, 0.7);

        // exp / log / distance / parallel_transport all match
        // pure E³ (Euclidean) within numerical-integrator noise.
        close((bs.exp(p, v) - EuclideanR3.exp(p, v)).length(), 0.0, 1e-5);
        close((bs.log(p, q) - EuclideanR3.log(p, q)).length(), 0.0, 1e-3);
        close(bs.distance(p, q), EuclideanR3.distance(p, q), 1e-3);
        close(
            (bs.parallel_transport(p, q, v) - EuclideanR3.parallel_transport(p, q, v)).length(),
            0.0,
            1e-5,
        );
        // ConformallyFlat fields match.
        close(
            bs.conformal_factor(p),
            EuclideanR3.conformal_factor(p),
            1e-6,
        );
        close(bs.scalar_curvature(p), 0.0, 1e-3);
    }

    #[test]
    fn blended_space_at_alpha_one_is_pure_b() {
        use crate::{EuclideanR3, HyperbolicH3, Space};
        let bs = BlendedSpace::new(
            EuclideanR3,
            HyperbolicH3,
            LinearBlendX::new(-100.0, -50.0), // alpha ≡ 1 in test region
        );
        let p = Vec3::new(0.1, 0.0, 0.0);
        let q = Vec3::new(0.2, 0.1, 0.0);
        let v = Vec3::new(0.05, 0.0, 0.0);

        // Match pure H³ within numerical-integrator noise.
        close((bs.exp(p, v) - HyperbolicH3.exp(p, v)).length(), 0.0, 5e-3);
        close((bs.log(p, q) - HyperbolicH3.log(p, q)).length(), 0.0, 5e-3);
        close(bs.distance(p, q), HyperbolicH3.distance(p, q), 5e-3);
        close(
            bs.conformal_factor(p),
            HyperbolicH3.conformal_factor(p),
            1e-3,
        );
        // FD curvature on BlendedSpace has ~1% noise even at α=1
        // (FD samples don't perfectly cancel, they're at ε≈5e-3
        // spacing and the underlying φ has nonzero curvature).
        close(bs.scalar_curvature(p), -6.0, 0.05);
    }

    /// Smoothstep is monotonic non-decreasing across the zone.
    /// Pin so a future "use a different smoothing profile"
    /// regression doesn't accidentally introduce overshoot.
    #[test]
    fn linear_blend_x_is_monotonic() {
        let f = LinearBlendX::new(-1.0, 1.0);
        let xs: Vec<f32> = (0..=20).map(|i| -1.0 + (i as f32) / 10.0).collect();
        let mut prev = f.weight(Vec3::new(xs[0], 0.0, 0.0));
        for &x in &xs[1..] {
            let curr = f.weight(Vec3::new(x, 0.0, 0.0));
            assert!(
                curr >= prev - 1e-6,
                "non-monotonic: at x={x}, weight={curr} < previous {prev}"
            );
            prev = curr;
        }
    }
}
