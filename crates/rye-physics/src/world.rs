//! [`World<S>`], top-level container. Owns bodies, force fields, narrowphase dispatch, and
//! persistent contact manifolds; runs one simulation tick per [`World::step`].
//!
//! ## Step pipeline
//!
//! Each tick runs phases in a fixed order:
//!
//! 1. **Apply forces**: sample every [`ForceField`] at each body's position, accumulate
//!    `F·dt/m` into velocities.
//! 2. **Integrate**: advance position and orientation via [`crate::integrate_body`]. Position
//!    uses `space.exp`, velocity parallel-transports, orientation integrates per the space's
//!    rule.
//! 3. **Broadphase**: O(n²) all-pairs for now. Grid / BVH come in when body counts demand it.
//! 4. **Narrowphase**: dispatch through [`crate::Narrowphase`] for each candidate pair. New
//!    contacts are merged into the persistent [`crate::Manifold`] for that pair.
//! 5. **Manifold maintenance**: drop manifolds whose pair didn't generate a contact this frame.
//! 6. **Warm start**: re-apply each cached contact's previous-frame accumulated impulses to the
//!    bodies. The PGS loop then converges in a handful of iterations.
//! 7. **PGS solve**: `pgs_iters` passes of normal-then-tangent impulses per contact, with
//!    accumulated-impulse clamping (`jn ≥ 0`, `|jt| ≤ μ·jn`) and a Baumgarte velocity bias
//!    for positional correction.
//!
//! Each phase is exposed as a method so games / test harnesses can substitute or inspect
//! individual phases without forking the step loop.

use std::collections::{BTreeMap, HashSet};

use crate::body::RigidBody;
use crate::collision::VectorOps;
use crate::field::ForceField;
use crate::integrator::{integrate_body, PhysicsSpace};
use crate::manifold::{
    ContactPoint, Manifold, BAUMGARTE_BETA, DEFAULT_PGS_ITERS, MAX_LINEAR_CORRECTION,
    PENETRATION_SLOP, RESTITUTION_THRESHOLD,
};
use crate::narrowphase::Narrowphase;
use crate::response::FRICTION_COEFF;

/// Pair key for the persistent manifold cache. Convention: `(small, large)` so a pair has one
/// canonical key regardless of iteration order in broadphase.
pub type PairKey = (usize, usize);

pub struct World<S: PhysicsSpace> {
    pub space: S,
    pub bodies: Vec<RigidBody<S>>,
    pub fields: Vec<Box<dyn ForceField<S>>>,
    pub narrowphase: Narrowphase<S>,
    /// Persistent contact manifolds, keyed by `(body_a, body_b)` with `body_a < body_b`.
    /// Refreshed each step from new narrowphase contacts; pairs that didn't touch are evicted
    /// before the solver runs so no stale impulses get warm-started.
    ///
    /// `BTreeMap` (not `HashMap`) for deterministic iteration order: PGS convergence is
    /// sensitive to the order in which constraints are visited, and sim-path data structures
    /// must not iterate in hash order (determinism is a Tier-0 invariant).
    pub manifolds: BTreeMap<PairKey, Manifold<S>>,
    /// Number of PGS iterations per step. Defaults to [`DEFAULT_PGS_ITERS`]; raise for stiff
    /// stacks, lower for cheap scenes.
    pub pgs_iters: usize,
    pub time: f32,
}

impl<S: PhysicsSpace> World<S> {
    pub fn new(space: S) -> Self {
        Self {
            space,
            bodies: Vec::new(),
            fields: Vec::new(),
            narrowphase: Narrowphase::new(),
            manifolds: BTreeMap::new(),
            pgs_iters: DEFAULT_PGS_ITERS,
            time: 0.0,
        }
    }

    /// Add a body to the world; returns its index.
    pub fn push_body(&mut self, body: RigidBody<S>) -> usize {
        let id = self.bodies.len();
        self.bodies.push(body);
        id
    }

    /// Add a force field to the world.
    pub fn push_field(&mut self, field: Box<dyn ForceField<S>>) {
        self.fields.push(field);
    }

    /// Advance the simulation by `dt` seconds.
    pub fn step(&mut self, dt: f32)
    where
        S::Vector: VectorOps,
        S::Point: Copy + std::ops::Sub<Output = S::Vector>,
    {
        self.apply_forces(dt);
        self.integrate(dt);
        self.update_manifolds();
        self.prepare_solve(dt);
        self.warm_start();
        self.solve();

        self.time += dt;
    }

    fn apply_forces(&mut self, dt: f32)
    where
        S::Vector: VectorOps,
    {
        for body in &mut self.bodies {
            if body.inv_mass == 0.0 {
                continue;
            }
            for field in &self.fields {
                let f = field.force_at(body, self.time);
                body.velocity = body.velocity + f * (dt * body.inv_mass);
            }
        }
    }

    fn integrate(&mut self, dt: f32)
    where
        S::Vector: VectorOps,
    {
        for body in &mut self.bodies {
            integrate_body(&self.space, body, dt);
        }
    }

    /// Run broadphase + narrowphase, merging each new contact into its pair's persistent
    /// manifold (or creating one). Manifolds whose pair didn't generate a contact this frame are
    /// dropped so their stale warm-start impulses can't leak into the next solve.
    fn update_manifolds(&mut self)
    where
        S::Vector: VectorOps,
        S::Point: Copy + std::ops::Sub<Output = S::Vector>,
    {
        let pairs = self.broadphase();
        let mut touched: HashSet<PairKey> = HashSet::with_capacity(pairs.len());

        for (i, j) in pairs {
            let (a, b) = split_two_mut(&mut self.bodies, i, j);
            let Some(contact) = self.narrowphase.test(a, b, &self.space) else {
                continue;
            };
            let key = (i, j);
            touched.insert(key);
            let restitution = contact.restitution;
            let manifold = self
                .manifolds
                .entry(key)
                .or_insert_with(|| Manifold::new(i, j, restitution));
            manifold.add_or_update(contact);
        }

        self.manifolds.retain(|k, _| touched.contains(k));
    }

    /// Snapshot per-contact `velocity_bias` (restitution + Baumgarte combined) and reset
    /// within-step tangent accumulators. Runs **before** warm-start so the bias reflects the
    /// actual approach velocity from physics, not the post-warm-start v_n. Without this,
    /// restitution would be recomputed against a moving target each iteration and converge to
    /// zero bounce.
    fn prepare_solve(&mut self, dt: f32)
    where
        S::Vector: VectorOps,
    {
        for manifold in self.manifolds.values_mut() {
            let (a, b) = split_two_mut(&mut self.bodies, manifold.body_a, manifold.body_b);
            for cp in &mut manifold.points {
                let v_rel = self.space.velocity_at_point(b, cp.world_point)
                    - self.space.velocity_at_point(a, cp.world_point);
                let v_n = VectorOps::dot(v_rel, cp.normal);

                let restitution_bias = if v_n < -RESTITUTION_THRESHOLD {
                    manifold.restitution * v_n
                } else {
                    0.0
                };

                let baumgarte_bias = if dt > 0.0 {
                    let target = (cp.penetration - PENETRATION_SLOP).max(0.0) * BAUMGARTE_BETA / dt;
                    -target.min(MAX_LINEAR_CORRECTION / dt)
                } else {
                    0.0
                };

                cp.velocity_bias = restitution_bias + baumgarte_bias;

                // Reset within-step tangent state. The slide direction can flip between frames;
                // warm-starting friction with a stale signed magnitude would push in the wrong
                // direction. Normal impulse is the meaningful warm-start for stacking; tangent
                // re-converges in 1-2 iterations.
                cp.tangent_impulse = 0.0;
                cp.tangent_dir = VectorOps::zero();
            }
        }
    }

    /// Re-apply each cached contact's previous-frame accumulated normal impulse. Tangent impulse
    /// was reset in `prepare_solve` because slide direction is not stable across frames.
    fn warm_start(&mut self)
    where
        S::Vector: VectorOps,
    {
        for manifold in self.manifolds.values() {
            let (a, b) = split_two_mut(&mut self.bodies, manifold.body_a, manifold.body_b);
            for cp in &manifold.points {
                if cp.normal_impulse > 0.0 {
                    self.space.apply_contact_impulse(
                        a,
                        b,
                        cp.world_point,
                        cp.normal,
                        cp.normal_impulse,
                    );
                }
            }
        }
    }

    /// PGS solve: `pgs_iters` passes over every contact in every manifold, applying clamped
    /// incremental normal-then-tangent impulses. The pre-snapshotted `velocity_bias` on each
    /// contact drives both restitution and positional correction, so this loop never recomputes
    /// either, it just chases the fixed target.
    fn solve(&mut self)
    where
        S::Vector: VectorOps,
    {
        let keys: Vec<PairKey> = self.manifolds.keys().copied().collect();

        for _ in 0..self.pgs_iters {
            for &key in &keys {
                let manifold = match self.manifolds.get_mut(&key) {
                    Some(m) => m,
                    None => continue,
                };
                let (a, b) = split_two_mut(&mut self.bodies, manifold.body_a, manifold.body_b);
                for cp in &mut manifold.points {
                    solve_normal_then_tangent(&self.space, a, b, cp);
                }
            }
        }
    }

    /// All-pairs broadphase. Returns `(i, j)` pairs with `i < j`. Replace with a grid / BVH
    /// when body counts demand it.
    pub fn broadphase(&self) -> Vec<PairKey> {
        let n = self.bodies.len();
        let mut pairs = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if self.bodies[i].inv_mass == 0.0 && self.bodies[j].inv_mass == 0.0 {
                    continue;
                }
                pairs.push((i, j));
            }
        }
        pairs
    }
}

/// Split-borrow helper: get `&mut bodies[i]` and `&mut bodies[j]` simultaneously, given `i < j`.
/// The caller must ensure that invariant.
fn split_two_mut<T>(slice: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    debug_assert!(i < j, "split_two_mut requires i < j (got {i}, {j})");
    let (left, right) = slice.split_at_mut(j);
    (&mut left[i], &mut right[0])
}

/// One PGS iteration over a single contact: normal solve, then tangent (friction) solve. Both
/// use accumulated-impulse clamping (`jn ≥ 0`, `|jt| ≤ μ·jn`) so the solver can revisit the
/// same contact across iterations and converge to the constraint's fixed `velocity_bias` target.
fn solve_normal_then_tangent<S>(
    space: &S,
    a: &mut RigidBody<S>,
    b: &mut RigidBody<S>,
    cp: &mut ContactPoint<S>,
) where
    S: PhysicsSpace,
    S::Vector: VectorOps,
{
    // Contacts reach the solver only after narrowphase validation; a non-finite slot here means
    // a bug upstream, not a runtime case to silently skip. Catch it in debug; release trusts
    // narrowphase.
    debug_assert!(
        VectorOps::is_finite(cp.normal) && cp.penetration.is_finite(),
        "non-finite contact in solve_normal_then_tangent",
    );

    // ---- Normal solve ----
    let v_rel_n_vec =
        space.velocity_at_point(b, cp.world_point) - space.velocity_at_point(a, cp.world_point);
    let v_n = VectorOps::dot(v_rel_n_vec, cp.normal);
    let k_n = space.effective_mass_inv(a, b, cp.world_point, cp.normal);

    if k_n > 0.0 {
        // Target post-impulse v_n is `−velocity_bias`. dj corrects by exactly that amount per
        // iteration, clamped so the accumulated normal impulse stays ≥ 0.
        let dj = -(v_n + cp.velocity_bias) / k_n;
        let new_acc = (cp.normal_impulse + dj).max(0.0);
        let actual = new_acc - cp.normal_impulse;
        cp.normal_impulse = new_acc;
        if actual.abs() > 0.0 {
            space.apply_contact_impulse(a, b, cp.world_point, cp.normal, actual);
        }
    }

    // ---- Tangent (friction) solve ----
    let v_rel_t_vec =
        space.velocity_at_point(b, cp.world_point) - space.velocity_at_point(a, cp.world_point);
    let v_t_vec = v_rel_t_vec - cp.normal * VectorOps::dot(v_rel_t_vec, cp.normal);
    let v_t_mag = VectorOps::length(v_t_vec);

    if v_t_mag < 1e-8 {
        return;
    }

    let tangent = v_t_vec * (1.0 / v_t_mag);
    let k_t = space.effective_mass_inv(a, b, cp.world_point, tangent);
    if k_t <= 0.0 {
        return;
    }

    // Desired tangent impulse to zero v_t this iteration. We accumulate as a magnitude-only
    // positive scalar within the step (cleared in `prepare_solve`); the snapshotted
    // `tangent_dir` keeps the direction consistent across iterations even though we recompute
    // the current iteration's tangent from v_t.
    let dj_t = v_t_mag / k_t;
    let max_friction = cp.normal_impulse * FRICTION_COEFF;
    let new_acc = (cp.tangent_impulse + dj_t).min(max_friction);
    let actual = new_acc - cp.tangent_impulse;
    cp.tangent_impulse = new_acc;
    cp.tangent_dir = tangent;

    if actual > 0.0 {
        // tangent points along v_t (the slide direction); applying along −tangent brakes the
        // slide.
        space.apply_contact_impulse(a, b, cp.world_point, tangent, -actual);
    }
}
