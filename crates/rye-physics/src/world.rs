//! [`World<S>`], top-level container. Owns bodies, force fields, narrowphase
//! dispatch, and persistent contact manifolds; runs one tick per
//! [`World::step`].
//!
//! ## Step pipeline
//!
//! Each tick runs a fixed phase order: apply forces, integrate, broadphase,
//! narrowphase, manifold maintenance, warm start, PGS solve. Each phase is a
//! method so harnesses can substitute or inspect it without forking the loop.

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

/// Pair key for the manifold cache. Convention: `(small, large)` so a pair has
/// one canonical key regardless of broadphase iteration order.
pub type PairKey = (usize, usize);

pub struct World<S: PhysicsSpace> {
    pub space: S,
    pub bodies: Vec<RigidBody<S>>,
    pub fields: Vec<Box<dyn ForceField<S>>>,
    pub narrowphase: Narrowphase<S>,
    /// Persistent contact manifolds, keyed `(body_a, body_b)` with `a < b`.
    /// `BTreeMap` for deterministic iteration: PGS convergence depends on
    /// constraint visit order, which must not be hash order (Tier-0
    /// determinism invariant).
    pub manifolds: BTreeMap<PairKey, Manifold<S>>,
    /// PGS iterations per step. Defaults to [`DEFAULT_PGS_ITERS`].
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

    /// Broadphase + narrowphase, merging each contact into its pair's manifold.
    /// Untouched pairs are evicted so stale warm-start impulses can't leak into
    /// the next solve.
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

    /// Snapshot per-contact `velocity_bias` (restitution + Baumgarte) and reset
    /// tangent accumulators. Must run before warm-start so the bias reflects the
    /// true approach velocity, not the post-warm-start v_n; otherwise
    /// restitution chases a moving target and converges to zero bounce.
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

                // Slide direction can flip between frames, so a stale tangent
                // magnitude would brake the wrong way; re-converges in 1-2 iters.
                cp.tangent_impulse = 0.0;
                cp.tangent_dir = VectorOps::zero();
            }
        }
    }

    /// Re-apply each contact's previous-frame normal impulse. Tangent was reset
    /// in `prepare_solve` (slide direction is not stable across frames).
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

    /// PGS solve: `pgs_iters` passes of clamped incremental normal-then-tangent
    /// impulses. The pre-snapshotted `velocity_bias` is the fixed target; this
    /// loop chases it and never recomputes restitution or correction.
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

    /// All-pairs broadphase. Returns `(i, j)` pairs with `i < j`.
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

/// Split-borrow `&mut slice[i]` and `&mut slice[j]` simultaneously. Caller must
/// ensure `i < j`.
fn split_two_mut<T>(slice: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    debug_assert!(i < j, "split_two_mut requires i < j (got {i}, {j})");
    let (left, right) = slice.split_at_mut(j);
    (&mut left[i], &mut right[0])
}

/// One PGS iteration over a contact: normal then tangent (friction) solve, both
/// with accumulated-impulse clamping (`jn ≥ 0`, `|jt| ≤ μ·jn`) so repeated
/// passes converge to the fixed `velocity_bias` target.
fn solve_normal_then_tangent<S>(
    space: &S,
    a: &mut RigidBody<S>,
    b: &mut RigidBody<S>,
    cp: &mut ContactPoint<S>,
) where
    S: PhysicsSpace,
    S::Vector: VectorOps,
{
    // A non-finite slot here is an upstream narrowphase bug, not a runtime case
    // to skip; release trusts narrowphase validation.
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
        // Target post-impulse v_n is `−velocity_bias`, clamped so accumulated
        // normal impulse stays ≥ 0.
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

    // Accumulated as a magnitude-only positive scalar within the step (cleared
    // in `prepare_solve`); `tangent_dir` snapshots the direction.
    let dj_t = v_t_mag / k_t;
    let max_friction = cp.normal_impulse * FRICTION_COEFF;
    let new_acc = (cp.tangent_impulse + dj_t).min(max_friction);
    let actual = new_acc - cp.tangent_impulse;
    cp.tangent_impulse = new_acc;
    cp.tangent_dir = tangent;

    if actual > 0.0 {
        // tangent points along the slide; apply along −tangent to brake it.
        space.apply_contact_impulse(a, b, cp.world_point, tangent, -actual);
    }
}
