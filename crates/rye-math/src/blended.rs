//! `BlendedSpace<A, B, F>`: a `Space` whose metric smoothly interpolates
//! between two source Spaces A and B via a blending field F: ℝ³ -> [0, 1].
//!
//! - [`BlendingField`] trait + [`LinearBlendX`] (axis-aligned smooth-step
//!   zone).
//! - [`ConformallyFlat`] trait + impls for `EuclideanR3`, `HyperbolicH3`,
//!   `SphericalS3`.
//! - [`BlendedSpace<A, B, F>`] implementing `Space` via RK4 geodesic
//!   integration, Gauss-Newton `log` shooting, and RK4 parallel transport for
//!   the conformally-flat fast path.
//! - WGSL emit (specific to
//!   `BlendedSpace<EuclideanR3, HyperbolicH3, LinearBlendX>`).

use std::borrow::Cow;
use std::marker::PhantomData;

use glam::{Mat3, Vec3};

use crate::space::{Space, WgslSpace};

// ---------------------------------------------------------------------------
// ConformallyFlat: extension trait for Spaces with scalar metric
// ---------------------------------------------------------------------------

/// A [`Space`] whose metric tensor is a scalar multiple of the identity in its
/// standard chart: g_ij(p) = f(p)·δ_ij for some positive scalar function f.
///
/// - [`crate::EuclideanR3`]: f ≡ 1.
/// - [`crate::HyperbolicH3`] (Poincaré ball): f(p) = 4/(1-|p|²)², for |p| < 1.
/// - [`crate::SphericalS3`] (stereographic): f(p) = 4/(1+|p|²)².
///
/// Separate trait, not a `Space` method: not every Space is conformally flat
/// (Sol³, Nil³ have anisotropic metrics). When both sources are conformally
/// flat the blend is too, and the geodesic ODE collapses to a closed form in
/// ∇ log f; that is the integrator's fast path.
pub trait ConformallyFlat: Space {
    /// Conformal scale factor at `p`: the scalar f with g_ij(p) = f(p)·δ_ij.
    ///
    /// Positive and finite inside the chart; chart-boundary points (Poincaré
    /// |p| -> 1) may return `f32::INFINITY`, which the integrator clamps.
    fn conformal_factor(&self, p: Vec3) -> f32;

    /// Scalar curvature R(p). For a 3D conformally flat metric:
    /// R = -(4/f(p))·[∇²φ + (1/2)|∇φ|²]. Default is finite-difference; every
    /// constant-curvature Space overrides with the closed form.
    fn scalar_curvature(&self, p: Vec3) -> f32 {
        const EPS: f32 = 5.0e-3;
        let phi_at = self.conformal_log_half(p);
        // 7-point stencil Laplacian.
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
    fn conformal_log_half(&self, p: Vec3) -> f32 {
        0.5 * self.conformal_factor(p).ln()
    }

    /// Spatial gradient of [`Self::conformal_log_half`]. Default is central
    /// finite differences; closed-form overrides avoid the FD noise.
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

// HyperbolicH3 (Poincaré ball): f(p) = 4/(1-|p|²)² for |p| < 1; INFINITY at
// the ideal boundary |p| = 1 to flag chart-boundary crossings.
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
        // ln 2 − ln(1 − |p|²).
        let r2 = p.length_squared();
        let denom = (1.0 - r2).max(0.0);
        if denom <= 0.0 {
            return f32::INFINITY;
        }
        std::f32::consts::LN_2 - denom.ln()
    }
    fn conformal_log_half_gradient(&self, p: Vec3) -> Vec3 {
        // ∇φ = 2p / (1 − |p|²).
        let r2 = p.length_squared();
        let denom = (1.0 - r2).max(0.0);
        if denom <= 0.0 {
            return Vec3::ZERO;
        }
        p * (2.0 / denom)
    }
    fn scalar_curvature(&self, _p: Vec3) -> f32 {
        // Constant K = -1 in 3D: R = n(n−1)K = −6.
        -6.0
    }
}

// SphericalS3 (stereographic from north pole): f(p) = 4/(1+|p|²)².
impl ConformallyFlat for crate::SphericalS3 {
    fn conformal_factor(&self, p: Vec3) -> f32 {
        let r2 = p.length_squared();
        4.0 / (1.0 + r2).powi(2)
    }
    fn conformal_log_half(&self, p: Vec3) -> f32 {
        // ln 2 − ln(1 + |p|²).
        std::f32::consts::LN_2 - (1.0 + p.length_squared()).ln()
    }
    fn conformal_log_half_gradient(&self, p: Vec3) -> Vec3 {
        // ∇φ = -2p / (1 + |p|²).
        let r2 = p.length_squared();
        p * (-2.0 / (1.0 + r2))
    }
    fn scalar_curvature(&self, _p: Vec3) -> f32 {
        // Constant K = +1 in 3D: R = n(n−1)K = +6.
        6.0
    }
}

// ---------------------------------------------------------------------------
// BlendedSpace: Space whose metric varies smoothly with position
// ---------------------------------------------------------------------------

/// A `Space` whose metric is the smooth blend of two source Spaces' metrics,
/// weighted by a [`BlendingField`]:
/// g(p) = (1 - α(p))·g_A(p) + α(p)·g_B(p).
///
/// At zone extremes the metric reduces to pure g_A or g_B; in between it is a
/// variable-metric Riemannian manifold. Sources must use ℝ³ points/vectors and
/// be [`ConformallyFlat`] (so the blend is too, the integrator's fast path).
/// No non-trivial isometries (the field breaks translation/rotation symmetry):
/// `Iso = ()`, `iso_apply` is the identity.
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
    /// No non-trivial isometries.
    type Iso = ();

    fn distance(&self, a: Vec3, b: Vec3) -> f32 {
        // Both endpoints at the same zone extreme: exact source-Space distance.
        let alpha_a = self.field.weight(a);
        let alpha_b = self.field.weight(b);
        if alpha_a == 0.0 && alpha_b == 0.0 {
            return self.a.distance(a, b);
        }
        if alpha_a == 1.0 && alpha_b == 1.0 {
            return self.b.distance(a, b);
        }
        // |log_a(b)|_g = √f(a)·|log_a(b)|_E.
        let log = self.log(a, b);
        let f_a = self.conformal_factor(a);
        f_a.sqrt() * log.length()
    }

    fn exp(&self, at: Vec3, v: Vec3) -> Vec3 {
        // `v` is Euclidean; over unit parameter time the geodesic length covered
        // is |v|_g = |v|_E·√f(at).
        rk4_geodesic(self, at, v, GEODESIC_DEFAULT_STEPS).0
    }

    fn log(&self, from: Vec3, to: Vec3) -> Vec3 {
        gauss_newton_log(self, from, to, GEODESIC_DEFAULT_STEPS, LOG_MAX_ITERS)
    }

    fn parallel_transport(&self, from: Vec3, to: Vec3, v: Vec3) -> Vec3 {
        // Transports along the chart-coordinate straight line, not the geodesic
        // (the latter costs ~7x). Callers needing a known path should use
        // `parallel_transport_along` with the polyline directly.
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

/// RK4 steps per unit-parameter integration. 32 gives ~6 digits on moderately
/// curved metrics; 64 gives ~7.
pub const GEODESIC_DEFAULT_STEPS: u32 = 32;

/// Single RK4 step on the geodesic ODE for a conformally flat metric, state
/// `(p, v)`: ṗ = v, v̇ = |v|²·∇φ(p) - 2·(∇φ·v)·v. Steps by `h` in parameter
/// time.
fn rk4_geodesic_step<S: ConformallyFlat>(space: &S, p: Vec3, v: Vec3, h: f32) -> (Vec3, Vec3) {
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

/// Integrate the geodesic ODE from `(at, v)` for unit parameter time in
/// `n_steps` RK4 steps; returns the final `(point, velocity)`.
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
        // Stop at the last finite state so chart-boundary crossings and blow-ups
        // don't propagate NaN downstream.
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

/// RK4 steps per polyline segment for parallel transport. 8 gives ~5 digits on
/// moderately curved metrics, enough that a camera frame stays orthonormal over
/// typical paths without post-step renormalisation.
pub const PARALLEL_TRANSPORT_DEFAULT_STEPS: u32 = 8;

/// Parallel-transport `v` along the segment `p_from` -> `p_to`, parameterised
/// linearly over t ∈ [0, 1]. For a conformally flat metric g = e^(2φ)·δ the ODE
/// is V̇ = -[(∇φ·γ̇)·V + (∇φ·V)·γ̇ - (γ̇·V)·∇φ] with γ̇ = p_to − p_from.
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

/// Maximum Gauss-Newton iterations for `log`. ~5 converges to f32 precision off
/// the cut locus; cap at 12 to bound the worst case.
pub const LOG_MAX_ITERS: u32 = 12;

/// Euclidean convergence threshold for the residual `|to − exp_from(v)|`.
pub const LOG_RESIDUAL_TOL: f32 = 1.0e-5;

/// Finite-difference step for the Jacobian of `exp` w.r.t. `v`. Smaller is more
/// accurate but noisier; 1e-3 is the sweet spot for f32 RK4-of-32-steps.
const LOG_JACOBIAN_EPS: f32 = 1.0e-3;

/// Find the tangent `v` at `from` with `exp_from(v) ≈ to`, by Gauss-Newton
/// shooting: forward-evaluate `exp`, take the residual, estimate the Jacobian
/// `∂exp/∂v` by central differences, solve for the Newton update.
///
/// Returns the best `v` within `max_iters`. A singular Jacobian (e.g. `to` in
/// the cut locus of `from`) returns the current guess with a `tracing::warn`.
pub fn gauss_newton_log<S: ConformallyFlat>(
    space: &S,
    from: Vec3,
    to: Vec3,
    n_steps: u32,
    max_iters: u32,
) -> Vec3 {
    if (from - to).length() < LOG_RESIDUAL_TOL {
        return Vec3::ZERO;
    }

    // Euclidean displacement: exact for pure E³, a Newton seed otherwise.
    let mut v = to - from;

    for iter in 0..max_iters {
        let endpoint = rk4_geodesic(space, from, v, n_steps).0;
        let residual = to - endpoint;
        if residual.length() < LOG_RESIDUAL_TOL {
            return v;
        }

        // Jacobian column j: (exp(v + ε e_j) − exp(v − ε e_j)) / (2ε).
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

// Blend of scalar-multiple metrics is itself a scalar multiple, so
// `BlendedSpace` is conformally flat when both sources are.
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
        // At a zone extreme take the live source's value exactly, so an
        // off-chart source's INFINITY does not poison the blend.
        if alpha <= 0.0 {
            return f_a;
        }
        if alpha >= 1.0 {
            return f_b;
        }
        (1.0 - alpha) * f_a + alpha * f_b
    }

    /// Analytical chain-rule gradient of φ = (1/2)·ln f for the blended factor
    /// f = (1-α)·f_A + α·f_B:
    ///
    ///   ∇f = ∇α·(f_B - f_A) + 2·(1-α)·f_A·∇φ_A + 2·α·f_B·∇φ_B,
    ///
    /// using ∇f_X = 2·f_X·∇φ_X, then ∇φ = ∇f / (2f). Mirrors the WGSL emit's
    /// `rye_blended_grad_phi`; avoids the default's FD truncation noise.
    fn conformal_log_half_gradient(&self, p: Vec3) -> Vec3 {
        let alpha = self.field.weight(p);
        // Zone-extreme fast paths: without them a `0 * INFINITY` from an
        // off-chart source factor poisons the blend with NaN.
        if alpha <= 0.0 {
            return self.a.conformal_log_half_gradient(p);
        }
        if alpha >= 1.0 {
            return self.b.conformal_log_half_gradient(p);
        }
        let f_a = self.a.conformal_factor(p);
        let f_b = self.b.conformal_factor(p);
        let f = (1.0 - alpha) * f_a + alpha * f_b;
        // Inside the zone a non-finite or non-positive `f` means a source
        // diverged where its chart should be valid. Return NaN (not ZERO) so
        // the integrator's `is_finite` guard surfaces the bug rather than
        // marching on with wrong dynamics.
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

/// A scalar blending field over ℝ³. `weight(p)` is `0.0` for pure A, `1.0` for
/// pure B, blending continuously between.
///
/// The integrator differentiates this field for the Christoffel symbols, so a
/// merely-continuous field (e.g. a clamped linear ramp) produces artifacts at
/// its breakpoints; use a smoothstep or equivalent C¹ profile.
pub trait BlendingField: Copy + Send + Sync + 'static {
    /// Blend weight at `p`. Implementations must clamp to `[0, 1]`.
    fn weight(&self, p: Vec3) -> f32;

    /// Spatial gradient of [`Self::weight`]. Default is central finite
    /// differences; closed-form blends override to avoid the FD noise.
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

/// Smoothstep blending zone along the X axis: pure A at `x ≤ start`, pure B at
/// `x ≥ end`, smooth C² transition between.
///
/// Uses the quintic smootherstep 6t⁵ - 15t⁴ + 10t³ (Perlin 2002). It is C²,
/// so the scalar curvature R(p) (which involves ∇²φ) stays continuous across
/// the seam; the cubic 3t² - 2t³ is only C¹ and jumps R at the endpoints. Zero
/// first and second derivative at both endpoints means the metric reduces
/// exactly to g_A / g_B outside the zone with no curvature kick.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LinearBlendX {
    pub start: f32,
    pub end: f32,
}

impl LinearBlendX {
    /// New zone over the given start / end x-coordinates; reversed inputs are
    /// swapped so `weight` still ramps 0 -> 1.
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
            // Degenerate zero-width zone: step function at `start`.
            return if p.x < self.start { 0.0 } else { 1.0 };
        }
        let t = ((p.x - self.start) / w).clamp(0.0, 1.0);
        // Smootherstep: 6t⁵ − 15t⁴ + 10t³.
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }

    fn gradient(&self, p: Vec3) -> Vec3 {
        let w = self.width();
        if w <= 0.0 {
            return Vec3::ZERO;
        }
        let raw_t = (p.x - self.start) / w;
        if !(0.0..=1.0).contains(&raw_t) {
            return Vec3::ZERO;
        }
        // 30t²(1−t)² · (1/w).
        let t = raw_t;
        let one_minus_t = 1.0 - t;
        let dx = 30.0 * t * t * one_minus_t * one_minus_t / w;
        Vec3::new(dx, 0.0, 0.0)
    }
}

// ---------------------------------------------------------------------------
// WGSL emission
// ---------------------------------------------------------------------------

/// WGSL prelude for the specific
/// `BlendedSpace<EuclideanR3, HyperbolicH3, LinearBlendX>` instantiation. A
/// hand-rolled parametric prelude, not a generic `WgslSpace` impl; only the
/// shader emission is single-instantiation, the Rust API is already generic.
///
/// 16 RK4 sub-steps per `rye_exp`, 8 per `rye_parallel_transport`, matching the
/// CPU's `parallel_transport_segment_rk4` step-for-step.
///
/// `rye_log` returns the chart-coordinate difference (the geodesic march kernel
/// does not call it). `rye_distance` uses the midpoint chord-metric
/// `sqrt(f((a+b)/2)) · |a − b|`, first-order accurate for nearby points (the
/// SDF use case); use the CPU side for accurate arbitrary-pair distances.
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

    /// Smoothstep boundary values: 0 at start, 1 at end, 0.5 at midpoint.
    #[test]
    fn linear_blend_x_smoothstep_endpoints() {
        let f = LinearBlendX::new(-1.0, 1.0);
        close(f.weight(Vec3::new(-1.0, 0.0, 0.0)), 0.0, 1e-6);
        close(f.weight(Vec3::new(1.0, 0.0, 0.0)), 1.0, 1e-6);
        close(f.weight(Vec3::ZERO), 0.5, 1e-6);
    }

    /// Outside the zone the field is constant and the gradient is exactly zero,
    /// so the metric reduces to pure A / pure B with no Christoffel kick.
    #[test]
    fn linear_blend_x_is_constant_outside_zone() {
        let f = LinearBlendX::new(-1.0, 1.0);
        close(f.weight(Vec3::new(-100.0, 0.0, 0.0)), 0.0, 0.0);
        assert_eq!(f.gradient(Vec3::new(-100.0, 0.0, 0.0)), Vec3::ZERO);
        close(f.weight(Vec3::new(100.0, 0.0, 0.0)), 1.0, 0.0);
        assert_eq!(f.gradient(Vec3::new(100.0, 0.0, 0.0)), Vec3::ZERO);
        // Smoothstep gradient is also zero at the endpoints.
        close(f.gradient(Vec3::new(-1.0, 0.0, 0.0)).x, 0.0, 1e-6);
        close(f.gradient(Vec3::new(1.0, 0.0, 0.0)).x, 0.0, 1e-6);
    }

    /// Inside the zone the gradient is along X only (axis-aligned blend), peak
    /// magnitude at the midpoint.
    #[test]
    fn linear_blend_x_gradient_is_axis_aligned() {
        let f = LinearBlendX::new(-1.0, 1.0);
        let g = f.gradient(Vec3::ZERO);
        assert!(g.x > 0.0, "midpoint gradient should be positive along +x");
        close(g.y, 0.0, 0.0);
        close(g.z, 0.0, 0.0);
        // t = 0.5: 30 · 0.25 · 0.25 / 2.0 = 0.9375.
        close(g.x, 0.9375, 1e-6);
    }

    /// Closed-form gradient agrees with central finite differences. Catches a
    /// sign or scale error in the analytic gradient.
    #[test]
    fn linear_blend_x_closed_form_matches_finite_diff() {
        let f = LinearBlendX::new(-1.0, 1.0);
        // Replicate the trait's default FD path so the override isn't hit.
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

    /// Reversed inputs get auto-swapped so the field still ramps 0 -> 1.
    #[test]
    fn linear_blend_x_handles_reversed_inputs() {
        let f = LinearBlendX::new(1.0, -1.0);
        close(f.weight(Vec3::new(-1.0, 0.0, 0.0)), 0.0, 1e-6);
        close(f.weight(Vec3::new(1.0, 0.0, 0.0)), 1.0, 1e-6);
    }

    // ------ ConformallyFlat impls ------

    /// EuclideanR3: f ≡ 1, log-half ≡ 0, gradient ≡ 0.
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

    /// Poincaré-ball HyperbolicH3 conformal factor 4/(1-|p|²)² at origin,
    /// halfway out, and near the boundary.
    #[test]
    fn hyperbolic_h3_conformal_factor_pin_values() {
        use crate::HyperbolicH3;
        let s = HyperbolicH3;
        // f(0) = 4.
        close(s.conformal_factor(Vec3::ZERO), 4.0, 1e-6);
        // |p| = 0.5: f = 4 / 0.5625.
        close(
            s.conformal_factor(Vec3::new(0.5, 0.0, 0.0)),
            4.0 / 0.5625,
            1e-5,
        );
        // Near boundary, huge magnitude.
        close(
            s.conformal_factor(Vec3::new(0.0, 0.0, 0.99499f32.sqrt())),
            4.0 / (1.0 - 0.99499_f32).powi(2),
            10.0,
        );
    }

    /// HyperbolicH3 closed-form `conformal_log_half_gradient` agrees with central
    /// finite differences.
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

    /// SphericalS3 conformal factor 4/(1+|p|²)² at origin and a generic point.
    #[test]
    fn spherical_s3_conformal_factor_pin_values() {
        use crate::SphericalS3;
        let s = SphericalS3;
        close(s.conformal_factor(Vec3::ZERO), 4.0, 1e-6);
        // |p|² = 1: f = 1.
        close(s.conformal_factor(Vec3::new(1.0, 0.0, 0.0)), 1.0, 1e-6);
        // |p|² = 3: f = 0.25.
        close(s.conformal_factor(Vec3::new(1.0, 1.0, 1.0)), 0.25, 1e-6);
    }

    // ------ BlendedSpace conformally-flat overrides ------

    /// `BlendedSpace::conformal_log_half_gradient` analytic override agrees with
    /// central finite differences on the blended `conformal_log_half`, pinning
    /// the chain rule against the default FD path. Tolerance `5e-3` clears both
    /// the FD truncation and roundoff floors near the |r|≈0.7 H3 sample.
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
        // alpha=0 region, mid-zone, alpha=1 region (inside the Poincaré ball).
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

    /// At alpha=0 the fast path returns A's gradient verbatim; with A = E³ that
    /// is exactly zero.
    #[test]
    fn blended_space_log_half_gradient_at_alpha_zero_is_pure_a() {
        use crate::{EuclideanR3, HyperbolicH3};
        let bs = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(50.0, 100.0));
        let g = bs.conformal_log_half_gradient(Vec3::new(1.0, 2.0, 3.0));
        close(g.x, 0.0, 1e-6);
        close(g.y, 0.0, 1e-6);
        close(g.z, 0.0, 1e-6);
    }

    /// At alpha=1 the fast path returns B's gradient verbatim.
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

    /// At a zone extreme (pure A), `BlendedSpace::distance` matches
    /// `A::distance`.
    #[test]
    fn blended_space_distance_at_alpha_zero_matches_a() {
        use crate::EuclideanR3;
        let bs = BlendedSpace::new(
            EuclideanR3,
            EuclideanR3, // dummy; alpha=0 means we never see B
            LinearBlendX::new(10.0, 20.0),
        );
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
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(3.0, 0.0, 0.0);
        let d_blend = bs.distance(a, b);
        let d_b = EuclideanR3.distance(a, b);
        close(d_blend, d_b, 1e-6);
    }

    /// `BlendedSpace::iso_apply` is the identity (`Iso = ()`).
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

    /// `BlendedSpace` conformal factor: source value at a zone extreme,
    /// alpha-weighted blend of the two in between.
    #[test]
    fn blended_space_conformal_factor_blends_linearly() {
        use crate::{EuclideanR3, HyperbolicH3};
        let bs = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-1.0, 1.0));

        // alpha=0: pure E³, factor = 1.
        close(bs.conformal_factor(Vec3::new(-1.0, 0.0, 0.0)), 1.0, 1e-6);
        // Near x=1 but inside; x=1 itself is the Poincaré boundary (f -> ∞).
        let p = Vec3::new(0.99, 0.0, 0.0);
        let alpha = LinearBlendX::new(-1.0, 1.0).weight(p);
        let f_e = 1.0;
        let f_h = HyperbolicH3.conformal_factor(p);
        let expected = (1.0 - alpha) * f_e + alpha * f_h;
        close(bs.conformal_factor(p), expected, 1e-2);
    }

    // ------ RK4 geodesic integrator ------

    /// In flat E³ the geodesic ODE has zero curvature term: `exp_p(v) = p + v`.
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

    /// In pure HyperbolicH3, `exp` from the origin along a Euclidean tangent `v`
    /// lands at the closed-form endpoint. Convention (matching `HyperbolicH3::
    /// exp`): `v` is Euclidean, so at the origin |exp_0(v)|_E = tanh(|v|_E).
    #[test]
    fn rk4_in_pure_h3_matches_closed_form_at_origin() {
        use crate::{HyperbolicH3, Space};

        for &mag in &[0.1_f32, 0.3, 0.5] {
            let v = Vec3::new(mag, 0.0, 0.0);
            let (final_p, _) = rk4_geodesic(&HyperbolicH3, Vec3::ZERO, v, GEODESIC_DEFAULT_STEPS);
            // |exp_0(v)|_E = tanh(|v|_E).
            let expected_radius = mag.tanh();
            close(final_p.x, expected_radius, 5e-3);
            close(final_p.y, 0.0, 1e-4);
            close(final_p.z, 0.0, 1e-4);
        }

        // Cross-check against the closed-form `HyperbolicH3::exp`.
        let v = Vec3::new(0.4, 0.1, 0.0);
        let (numerical, _) = rk4_geodesic(&HyperbolicH3, Vec3::ZERO, v, GEODESIC_DEFAULT_STEPS);
        let closed_form = HyperbolicH3.exp(Vec3::ZERO, v);
        close((numerical - closed_form).length(), 0.0, 5e-3);

        // Through `BlendedSpace::exp` with alpha ≡ 1, collapsing to pure H³.
        let bs = BlendedSpace::new(
            HyperbolicH3, // dummy; alpha=1 never reaches A
            HyperbolicH3,
            LinearBlendX::new(-100.0, -50.0),
        );
        let v = Vec3::new(0.5, 0.0, 0.0);
        let final_p_blended = bs.exp(Vec3::ZERO, v);
        close(final_p_blended.x, 0.5_f32.tanh(), 5e-3);
    }

    /// Geodesic round-trip in pure E³: `exp_p(v)` then `exp_q(-v)` returns to
    /// `p`. Integrator time-reversibility on a flat metric.
    #[test]
    fn rk4_in_e3_is_time_reversible() {
        use crate::EuclideanR3;
        let p = Vec3::new(1.0, 2.0, 3.0);
        let v = Vec3::new(0.5, -0.3, 0.7);
        let (q, vq) = rk4_geodesic(&EuclideanR3, p, v, GEODESIC_DEFAULT_STEPS);
        let (back, _) = rk4_geodesic(&EuclideanR3, q, -vq, GEODESIC_DEFAULT_STEPS);
        close((back - p).length(), 0.0, 1e-5);
    }

    /// `BlendedSpace::exp` at a zone extreme matches the source Space's exp,
    /// end-to-end through the integrator.
    #[test]
    fn blended_space_exp_at_alpha_zero_matches_e3() {
        use crate::EuclideanR3;
        use crate::Space;
        let bs = BlendedSpace::new(EuclideanR3, EuclideanR3, LinearBlendX::new(50.0, 100.0));
        let p = Vec3::new(1.0, 2.0, 3.0);
        let v = Vec3::new(0.5, -0.3, 0.7);
        let result = bs.exp(p, v);
        let expected = p + v;
        close((result - expected).length(), 0.0, 1e-5);
    }

    // ------ log via Gauss-Newton shooting ------

    /// In pure E³, `log_from(to) = to − from` exactly (Gauss-Newton converges
    /// in one step).
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

    /// Round-trip `exp_from(log_from(to)) ≈ to` in pure H³, where the integrator
    /// does real work.
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

    /// `log` matches the closed-form `HyperbolicH3::log`, validating the
    /// numerical inversion against independent ground truth.
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

    /// `log_p(p) = 0`; the shooting routine special-cases this to avoid a
    /// singular Jacobian at zero residual.
    #[test]
    fn log_of_self_is_zero() {
        use crate::HyperbolicH3;
        let p = Vec3::new(0.2, 0.1, 0.0);
        let v = gauss_newton_log(&HyperbolicH3, p, p, GEODESIC_DEFAULT_STEPS, LOG_MAX_ITERS);
        close(v.length(), 0.0, 1e-5);
    }

    /// `BlendedSpace::distance` matches the source distance at a zone extreme.
    #[test]
    fn blended_space_distance_at_alpha_zero_uses_log() {
        use crate::{EuclideanR3, Space};
        let bs = BlendedSpace::new(EuclideanR3, EuclideanR3, LinearBlendX::new(50.0, 100.0));
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let d = bs.distance(a, b);
        close(d, (b - a).length(), 1e-4);
    }

    // ------ Parallel transport ------

    /// In pure E³ parallel transport is the identity along any path (zero
    /// Christoffel symbols, zero transport RHS).
    #[test]
    fn parallel_transport_in_e3_is_identity() {
        use crate::EuclideanR3;
        let v = Vec3::new(0.5, -0.3, 0.7);
        let transported = parallel_transport_segment_rk4(
            &EuclideanR3,
            Vec3::ZERO,
            Vec3::new(2.0, 1.0, -1.0),
            v,
            PARALLEL_TRANSPORT_DEFAULT_STEPS,
        );
        close((transported - v).length(), 0.0, 1e-6);

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

    /// Hyperbolic transport preserves the Riemannian length √f(p)·|v|_E, not
    /// the Euclidean length.
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

    /// Numerical transport agrees with the closed-form
    /// `HyperbolicH3::parallel_transport` (gyration formula) only for short
    /// paths, where the geodesic ≈ the Euclidean segment our RK4 follows.
    #[test]
    fn parallel_transport_in_h3_matches_closed_form_for_short_paths() {
        use crate::{HyperbolicH3, Space};
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

    /// Closed-loop holonomy in H³: transport around a small triangle returns a
    /// vector differing from the original, proving real curvature is integrated
    /// (flat space would return it exactly).
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
        assert!(transported.is_finite());
        let drift = (transported - v).length();
        assert!(
            drift > 1e-3,
            "expected non-zero holonomy in H³, got drift {drift}"
        );
        // Small loop, so holonomy stays bounded.
        assert!(drift < 0.5, "holonomy unreasonably large: {drift}");
    }

    // ------ Curvature continuity ------

    /// HyperbolicH3 scalar curvature is -6 everywhere (constant K = -1 in 3D).
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

    /// SphericalS3 scalar curvature is +6 (constant K = +1 in 3D).
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

    /// Default FD curvature agrees with the closed-form override for H3,
    /// validating the FD stencil for blended-space use (no closed form there).
    #[test]
    fn finite_diff_curvature_matches_closed_form_in_h3() {
        use crate::HyperbolicH3;
        // Dummy that doesn't override `scalar_curvature`, hitting the FD default.
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
        }
        let fd = H3FdOnly;
        // FD with EPS=5e-3 gives ~3 digits.
        for p in [Vec3::ZERO, Vec3::new(0.2, 0.1, 0.0)] {
            close(fd.scalar_curvature(p), -6.0, 0.5);
        }
    }

    /// BlendedSpace<E³, H³, LinearBlendX> scalar curvature varies continuously
    /// across the zone: R=0 (E³) and R=-6 (H³) at the extremes, smooth between.
    #[test]
    fn blended_space_curvature_varies_continuously_across_zone() {
        use crate::{EuclideanR3, HyperbolicH3};
        let bs = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-0.5, 0.5));

        // Well into E³.
        let r_e = bs.scalar_curvature(Vec3::new(-0.7, 0.0, 0.0));
        close(r_e, 0.0, 1e-1);

        // Well into H³, inside the Poincaré ball. Loose tolerance: FD noise.
        let r_h = bs.scalar_curvature(Vec3::new(0.7, 0.0, 0.0));
        close(r_h, -6.0, 1.0);

        // Sample across the zone and check the values vary smoothly.
        let xs: Vec<f32> = (-30..=30).map(|i| (i as f32) * 0.025).collect();
        let curvatures: Vec<f32> = xs
            .iter()
            .map(|&x| bs.scalar_curvature(Vec3::new(x, 0.0, 0.0)))
            .collect();

        // The seam is its own curved region, so |R| > 6 inside the zone is a
        // real feature (|∇φ|² and ∇²φ both spike); bound generously.
        for &r in &curvatures {
            assert!(
                r.is_finite() && (-50.0..=5.0).contains(&r),
                "curvature out of expected range: {r}"
            );
        }

        // Adjacent samples differ by a bounded amount. FD aliasing on this
        // sample step reaches ~14 empirically; a C² discontinuity (e.g. cubic
        // smoothstep) jumps into the tens-hundreds, so 25 catches it.
        let max_jump = curvatures
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_jump < 25.0,
            "curvature has a discontinuity: max adjacent jump = {max_jump}"
        );
    }

    // ------ Boundary extremes ------

    /// At α=0 every BlendedSpace `Space` method matches source A; `exp`, `log`,
    /// `parallel_transport`, `distance`, `conformal_factor`, `scalar_curvature`.
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

        close((bs.exp(p, v) - EuclideanR3.exp(p, v)).length(), 0.0, 1e-5);
        close((bs.log(p, q) - EuclideanR3.log(p, q)).length(), 0.0, 1e-3);
        close(bs.distance(p, q), EuclideanR3.distance(p, q), 1e-3);
        close(
            (bs.parallel_transport(p, q, v) - EuclideanR3.parallel_transport(p, q, v)).length(),
            0.0,
            1e-5,
        );
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

        close((bs.exp(p, v) - HyperbolicH3.exp(p, v)).length(), 0.0, 5e-3);
        close((bs.log(p, q) - HyperbolicH3.log(p, q)).length(), 0.0, 5e-3);
        close(bs.distance(p, q), HyperbolicH3.distance(p, q), 5e-3);
        close(
            bs.conformal_factor(p),
            HyperbolicH3.conformal_factor(p),
            1e-3,
        );
        // FD curvature carries ~1% noise even at α=1.
        close(bs.scalar_curvature(p), -6.0, 0.05);
    }

    /// Smoothstep is monotonic non-decreasing across the zone (no overshoot).
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
