//! The [`Space`] trait, Rye's interface to geometry.
//!
//! A `Space` is a Riemannian manifold equipped with an isometry group. The
//! GPU-facing half of the contract lives on [`WgslSpace`].
//!
//! Methods take `&self` so parametric geometries (curvature scalar, radius) can
//! store their parameter; stateless ones monomorphize to direct calls.
//!
//! `Self::Vector` is a tangent vector at *some* point; the trait does not track
//! which. Prefer [`crate::Tangent`] outside tight kernels, it bundles the base
//! point with the vector.

use std::borrow::Cow;

/// A Riemannian manifold with a transitive isometry group. Shader integration
/// lives on [`WgslSpace`]. All methods must be deterministic and side-effect-free.
pub trait Space {
    /// A point on the manifold.
    type Point: Copy + Send + Sync + 'static;
    /// A tangent vector at *some* point; the base point is tracked by the caller.
    /// Use [`crate::Tangent`] to enforce that tracking.
    type Vector: Copy + Send + Sync + 'static;
    /// An orientation-preserving isometry of the manifold.
    type Iso: Copy + Send + Sync + 'static;

    // ---- Riemannian primitives ----------------------------------------

    /// Geodesic distance between two points.
    fn distance(&self, a: Self::Point, b: Self::Point) -> f32;

    /// Exponential map: travel from `at` along the geodesic with initial velocity `v` for
    /// unit time. Inverse of [`Self::log`].
    fn exp(&self, at: Self::Point, v: Self::Vector) -> Self::Point;

    /// Logarithm map: the tangent vector at `from` whose [`Self::exp`] reaches `to`. Inverse
    /// of [`Self::exp`]. Undefined if `to` is in the cut locus of `from` (e.g. antipode on a
    /// sphere); impls should document their handling.
    fn log(&self, from: Self::Point, to: Self::Point) -> Self::Vector;

    /// Parallel-transport `v` (a tangent vector at `from`) to `to`, returning the
    /// tangent vector at `to`.
    ///
    /// The path is implementation-defined: parallel transport is path-dependent
    /// in any non-flat geometry and this signature names no path. Each impl
    /// documents its choice. Callers needing a *specific* path should call
    /// [`Self::parallel_transport_along`] with the polyline explicitly.
    fn parallel_transport(
        &self,
        from: Self::Point,
        to: Self::Point,
        v: Self::Vector,
    ) -> Self::Vector;

    /// Parallel-transport `v` along the polyline through `path`, segment by
    /// segment, returning the vector at the final point.
    ///
    /// The path-aware primitive: pinning the macro path makes the integrated
    /// result independent of how the caller batched the journey. Contract: finer
    /// subdivision converges to true parallel transport along the polyline.
    /// `path.len() < 2` returns `v` unchanged. The default chains
    /// [`Self::parallel_transport`] over consecutive pairs.
    fn parallel_transport_along(&self, path: &[Self::Point], v: Self::Vector) -> Self::Vector {
        let mut current = v;
        for w in path.windows(2) {
            current = self.parallel_transport(w[0], w[1], current);
        }
        current
    }

    // ---- Isometry group -----------------------------------------------

    /// The identity isometry.
    fn iso_identity(&self) -> Self::Iso;

    /// `a ∘ b`, apply `b` first, then `a`.
    fn iso_compose(&self, a: Self::Iso, b: Self::Iso) -> Self::Iso;

    /// Inverse isometry: `iso_compose(a, iso_inverse(a)) == iso_identity()`.
    fn iso_inverse(&self, a: Self::Iso) -> Self::Iso;

    /// Apply an isometry to a point.
    fn iso_apply(&self, iso: Self::Iso, p: Self::Point) -> Self::Point;

    /// Apply an isometry's differential to a tangent vector at `at`. The result is a tangent
    /// vector at `iso_apply(iso, at)`.
    fn iso_transport(&self, iso: Self::Iso, at: Self::Point, v: Self::Vector) -> Self::Vector;
}

/// A [`Space`] that additionally exposes its primitives as WGSL for inlining
/// into shaders by `rye-shader`.
///
/// Split from [`Space`] so the stable math trait and the volatile shader ABI do
/// not share a release cadence, and so CPU-only consumers can depend on
/// `rye-math` without WGSL.
pub trait WgslSpace: Space {
    /// WGSL source providing this space's primitives. The v0 ABI is tiny and
    /// single-space (`vec3<f32>` point/vector only):
    ///
    /// ```wgsl
    /// fn rye_distance(a: vec3<f32>, b: vec3<f32>) -> f32
    /// fn rye_exp(at: vec3<f32>, v: vec3<f32>) -> vec3<f32>
    /// fn rye_log(p_from: vec3<f32>, p_to: vec3<f32>) -> vec3<f32>
    /// fn rye_parallel_transport(p_from: vec3<f32>, p_to: vec3<f32>, v: vec3<f32>) -> vec3<f32>
    /// ```
    ///
    /// Stateless geometries return `Cow::Borrowed`; parametric ones `format!`
    /// constants in and return `Cow::Owned`.
    fn wgsl_impl(&self) -> Cow<'static, str>;

    /// Whether the chart is globally flat: chart-coord arithmetic computes the
    /// correct geometry without the Riemannian `rye_*` machinery. False for
    /// curved Spaces (Poincaré ball H³, stereographic S³, `BlendedSpace`).
    /// Defaults to `false` so a new Space must opt in to chart-coord SDF fast
    /// paths.
    fn is_chart_flat(&self) -> bool {
        false
    }
}
