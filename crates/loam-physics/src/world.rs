//! [`World<S>`], top-level container. Owns bodies, force fields, narrowphase
//! dispatch, and persistent contact manifolds; runs one tick per
//! [`World::step`].
//!
//! ## Step pipeline
//!
//! Each tick runs a fixed phase order: apply forces, integrate, broadphase,
//! narrowphase, manifold maintenance, warm start, PGS solve. Each phase is a
//! method so harnesses can substitute or inspect it without forking the loop.
//!
//! ## Schedule seam
//!
//! Every phase materialises its work units into a reused buffer and runs the
//! buffer, so [`Schedule`] can reorder a phase without the phase knowing. For
//! work units that are independent, the orders a thread pool can produce are a
//! subset of the permutations of that buffer, which is what makes
//! permutation invariance testable before an executor exists.

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

/// How a step's work units are executed. Ships in release rather than behind
/// `cfg(test)`: the determinism contract is a claim about the shipping binary,
/// so the instrument that checks it has to live in the same binary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Schedule {
    /// Worker count. Fixed at 1 until an executor lands, and permanently 1 on
    /// wasm32.
    pub threads: usize,
    pub order: OrderPolicy,
}

impl Default for Schedule {
    fn default() -> Self {
        Self {
            threads: 1,
            order: OrderPolicy::Canonical,
        }
    }
}

/// Visit order for one phase's work-unit buffer. Exactly one phase is
/// reordered so a fixture varies one axis with every other held canonical;
/// a hash that moves then names the phase responsible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderPolicy {
    Canonical,
    /// The adversarial case for a Gauss-Seidel sweep: every dependency edge
    /// traversed against the canonical direction.
    Reversed {
        phase: SchedulePhase,
    },
    Permuted {
        phase: SchedulePhase,
        seed: u64,
    },
}

/// A phase group sharing one work-unit buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedulePhase {
    /// Body visit order, shared by `apply_forces` and `integrate`.
    Body,
    /// Pair visit order in `update_manifolds`.
    BroadphasePair,
    /// Constraint visit order, shared by `prepare_solve`, `warm_start`, and
    /// every `solve` sweep.
    Constraint,
}

impl OrderPolicy {
    fn apply<T>(self, phase: SchedulePhase, units: &mut [T]) {
        match self {
            OrderPolicy::Canonical => {}
            OrderPolicy::Reversed { phase: target } if target == phase => units.reverse(),
            OrderPolicy::Permuted {
                phase: target,
                seed,
            } if target == phase => shuffle(units, seed),
            _ => {}
        }
    }
}

/// Durstenfeld's in-place Fisher-Yates shuffle (Fisher and Yates 1938, table
/// XXXIII; Durstenfeld 1964, CACM 7(7):420) driven by xorshift64 (Marsaglia
/// 2003, "Xorshift RNGs", J. Stat. Soft. 8(14), the 13/7/17 triple). Modulo
/// bias is accepted: the requirement is a reproducible permutation reportable
/// by seed, not a uniform one.
fn shuffle<T>(units: &mut [T], seed: u64) {
    // xorshift64 is absorbing at zero, so a zero seed must not reach it.
    let mut state = seed | 1;
    for i in (1..units.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        units.swap(i, (state % (i as u64 + 1)) as usize);
    }
}

const STALE_CONSTRAINT_KEY: &str = "constraint buffer outlived its manifold";

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
    pub schedule: Schedule,
    /// Work-unit buffers, refilled and reordered at the head of their phase
    /// group and retained across steps so the seam allocates once. Each phase
    /// loop swaps its buffer out with `mem::take` and swaps it back, which is
    /// what keeps the allocation while the loop holds `&mut self`.
    body_order: Vec<usize>,
    pair_order: Vec<PairKey>,
    constraint_keys: Vec<PairKey>,
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
            schedule: Schedule::default(),
            body_order: Vec::new(),
            pair_order: Vec::new(),
            constraint_keys: Vec::new(),
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
        self.collect_bodies();
        self.apply_forces(dt);
        self.integrate(dt);
        self.update_manifolds();
        self.collect_constraints();
        self.prepare_solve(dt);
        self.warm_start();
        self.solve();

        self.time += dt;
    }

    /// Refill the body buffer in slot order, then hand it to the schedule.
    /// Refilled rather than reused in place so a permutation cannot compound
    /// across steps.
    fn collect_bodies(&mut self) {
        self.body_order.clear();
        self.body_order.extend(0..self.bodies.len());
        self.schedule
            .order
            .apply(SchedulePhase::Body, &mut self.body_order);
    }

    fn apply_forces(&mut self, dt: f32)
    where
        S::Vector: VectorOps,
    {
        let order = std::mem::take(&mut self.body_order);
        for &i in &order {
            let body = &mut self.bodies[i];
            if body.inv_mass == 0.0 {
                continue;
            }
            for field in &self.fields {
                let f = field.force_at(body, self.time);
                body.velocity = body.velocity + f * (dt * body.inv_mass);
            }
        }
        self.body_order = order;
    }

    fn integrate(&mut self, dt: f32)
    where
        S::Vector: VectorOps,
    {
        let order = std::mem::take(&mut self.body_order);
        for &i in &order {
            integrate_body(&self.space, &mut self.bodies[i], dt);
        }
        self.body_order = order;
    }

    /// Broadphase + narrowphase, merging each contact into its pair's manifold.
    /// Untouched pairs are evicted so stale warm-start impulses can't leak into
    /// the next solve.
    fn update_manifolds(&mut self)
    where
        S::Vector: VectorOps,
        S::Point: Copy + std::ops::Sub<Output = S::Vector>,
    {
        let mut pairs = std::mem::take(&mut self.pair_order);
        self.fill_broadphase(&mut pairs);
        self.schedule
            .order
            .apply(SchedulePhase::BroadphasePair, &mut pairs);
        let mut touched: HashSet<PairKey> = HashSet::with_capacity(pairs.len());

        for &(i, j) in &pairs {
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
        self.pair_order = pairs;
    }

    /// Refill the constraint buffer in `manifolds` key order, then hand it to
    /// the schedule. One buffer serves `prepare_solve`, `warm_start`, and
    /// `solve`, so those three always agree on the constraint order. Nothing
    /// between here and the end of the solve inserts or removes a manifold,
    /// which is why those three phases can index by key without a fallback.
    fn collect_constraints(&mut self) {
        self.constraint_keys.clear();
        self.constraint_keys.extend(self.manifolds.keys().copied());
        self.schedule
            .order
            .apply(SchedulePhase::Constraint, &mut self.constraint_keys);
    }

    /// Snapshot per-contact `velocity_bias` (restitution + Baumgarte) and reset
    /// tangent accumulators. Must run before warm-start so the bias reflects the
    /// true approach velocity, not the post-warm-start v_n; otherwise
    /// restitution chases a moving target and converges to zero bounce.
    fn prepare_solve(&mut self, dt: f32)
    where
        S::Vector: VectorOps,
    {
        let keys = std::mem::take(&mut self.constraint_keys);
        for key in &keys {
            let manifold = self.manifolds.get_mut(key).expect(STALE_CONSTRAINT_KEY);
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
        self.constraint_keys = keys;
    }

    /// Re-apply each contact's previous-frame normal impulse. Tangent was reset
    /// in `prepare_solve` (slide direction is not stable across frames).
    fn warm_start(&mut self)
    where
        S::Vector: VectorOps,
    {
        let keys = std::mem::take(&mut self.constraint_keys);
        for key in &keys {
            let manifold = self.manifolds.get(key).expect(STALE_CONSTRAINT_KEY);
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
        self.constraint_keys = keys;
    }

    /// PGS solve: `pgs_iters` passes of clamped incremental normal-then-tangent
    /// impulses. The pre-snapshotted `velocity_bias` is the fixed target; this
    /// loop chases it and never recomputes restitution or correction.
    fn solve(&mut self)
    where
        S::Vector: VectorOps,
    {
        let keys = std::mem::take(&mut self.constraint_keys);
        for _ in 0..self.pgs_iters {
            for key in &keys {
                let manifold = self.manifolds.get_mut(key).expect(STALE_CONSTRAINT_KEY);
                let (a, b) = split_two_mut(&mut self.bodies, manifold.body_a, manifold.body_b);
                for cp in &mut manifold.points {
                    solve_normal_then_tangent(&self.space, a, b, cp);
                }
            }
        }
        self.constraint_keys = keys;
    }

    /// All-pairs broadphase. Returns `(i, j)` pairs with `i < j`.
    pub fn broadphase(&self) -> Vec<PairKey> {
        let mut pairs = Vec::new();
        self.fill_broadphase(&mut pairs);
        pairs
    }

    fn fill_broadphase(&self, pairs: &mut Vec<PairKey>) {
        pairs.clear();
        let n = self.bodies.len();
        for i in 0..n {
            for j in (i + 1)..n {
                if self.bodies[i].inv_mass == 0.0 && self.bodies[j].inv_mass == 0.0 {
                    continue;
                }
                pairs.push((i, j));
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::determinism_fixture::{
        determinism_scenario_run, first_divergent_step, fnv1a64, ScenarioRun,
        GOLDEN_TRAJECTORY_HASH,
    };
    use crate::euclidean_r3::{halfspace_body_r3, register_default_narrowphase, sphere_body_r3};
    use crate::field::Gravity;
    use glam::Vec3;
    use loam_math::EuclideanR3;

    /// Arbitrary but fixed, so a failure is reproducible from its message.
    const PERMUTATION_SEEDS: [u64; 4] = [1, 0x9e37_79b9_7f4a_7c15, 0xdead_beef_cafe_f00d, 424_242];

    /// Reversal first: for a Gauss-Seidel sweep it is the adversarial order,
    /// not just another sample.
    fn order_variants(phase: SchedulePhase) -> Vec<OrderPolicy> {
        let mut variants = vec![OrderPolicy::Reversed { phase }];
        variants.extend(
            PERMUTATION_SEEDS
                .iter()
                .map(|&seed| OrderPolicy::Permuted { phase, seed }),
        );
        variants
    }

    fn run_with(order: OrderPolicy) -> ScenarioRun {
        determinism_scenario_run(Schedule { threads: 1, order })
    }

    /// The harness's sensitivity control. Global PGS is Gauss-Seidel, so its
    /// converged state depends on constraint visit order; if permuting that
    /// order left the hash untouched, the hash could not see the solver and
    /// every invariance assertion built on it would be a rubber stamp.
    #[test]
    fn global_solve_order_permutation_changes_state_hash_determinism() {
        let canonical = run_with(OrderPolicy::Canonical);
        assert!(
            canonical.step_hashes.len() > 1,
            "fixture produced no steps to compare"
        );
        for order in order_variants(SchedulePhase::Constraint) {
            let permuted = run_with(order);
            assert!(
                first_divergent_step(&canonical, &permuted).is_some(),
                "{order:?} left the state hash identical: the hash cannot see \
                 constraint visit order, so the positive axes below prove nothing"
            );
        }
    }

    /// Both invariance axes, asserted as `permuted == canonical == golden`.
    /// Mutual agreement among variants would certify a schedule that is
    /// self-consistently wrong, so the committed constant is the third link
    /// and not a redundant one.
    ///
    /// Non-vacuity rests on
    /// [`global_solve_order_permutation_changes_state_hash_determinism`]: that
    /// the same fixture's hash moves under a constraint permutation is what
    /// establishes it reaches contacts and that the hash observes them.
    fn assert_phase_order_does_not_reach_the_state_hash(phase: SchedulePhase) {
        let canonical = run_with(OrderPolicy::Canonical);
        assert_eq!(
            fnv1a64(&canonical.trajectory),
            GOLDEN_TRAJECTORY_HASH,
            "canonical run no longer matches the committed golden hash"
        );

        for order in order_variants(phase) {
            let permuted = run_with(order);
            if let Some(step) = first_divergent_step(&canonical, &permuted) {
                panic!("{order:?} diverged from the canonical schedule at step {step}");
            }
            let word_gap = canonical
                .trajectory
                .iter()
                .zip(&permuted.trajectory)
                .position(|(a, b)| a != b);
            assert!(
                word_gap.is_none() && permuted.trajectory.len() == canonical.trajectory.len(),
                "{order:?} moved trajectory word {word_gap:?}"
            );
            let hash = fnv1a64(&permuted.trajectory);
            assert_eq!(
                hash, GOLDEN_TRAJECTORY_HASH,
                "{order:?} produced {hash:#018x} against the committed golden \
                 {GOLDEN_TRAJECTORY_HASH:#018x}"
            );
        }
    }

    /// `apply_forces` and `integrate` read and write one body each, and
    /// `force_at` is a pure function of body state and `time`, so body visit
    /// order must not reach the state hash. Vacuous by construction today and
    /// deliberately so: it is the tripwire that fires the moment force
    /// accumulation grows a shared buffer.
    #[test]
    fn body_visit_order_permutation_preserves_state_hash_determinism() {
        assert_phase_order_does_not_reach_the_state_hash(SchedulePhase::Body);
    }

    /// Narrowphase runs once per pair, results land in a `BTreeMap` keyed
    /// canonically, and each pair contributes one contact per step, so pair
    /// emission order must not reach the solve. This is the property a
    /// parallel narrowphase would depend on, and unlike the body axis it is
    /// not true by construction.
    #[test]
    fn broadphase_pair_order_permutation_preserves_state_hash_determinism() {
        assert_phase_order_does_not_reach_the_state_hash(SchedulePhase::BroadphasePair);
    }

    /// The invariance axes are only evidence if the policy actually reorders
    /// the buffers the fixture builds: 7 bodies and 21 broadphase pairs.
    #[test]
    fn order_policy_permutes_reproducibly_and_never_to_identity_determinism() {
        for len in [7usize, 21] {
            let canonical: Vec<usize> = (0..len).collect();
            for phase in [
                SchedulePhase::Body,
                SchedulePhase::BroadphasePair,
                SchedulePhase::Constraint,
            ] {
                for order in order_variants(phase) {
                    let mut units = canonical.clone();
                    order.apply(phase, &mut units);
                    assert_ne!(units, canonical, "{order:?} on {len} units is the identity");
                    let mut sorted = units.clone();
                    sorted.sort_unstable();
                    assert_eq!(
                        sorted, canonical,
                        "{order:?} on {len} units lost or duplicated a unit"
                    );

                    let mut repeat = canonical.clone();
                    order.apply(phase, &mut repeat);
                    assert_eq!(repeat, units, "{order:?} is not reproducible");

                    // A policy naming one phase must leave the others alone,
                    // or the axes are not independent.
                    for other in [
                        SchedulePhase::Body,
                        SchedulePhase::BroadphasePair,
                        SchedulePhase::Constraint,
                    ] {
                        if other == phase {
                            continue;
                        }
                        let mut untouched = canonical.clone();
                        order.apply(other, &mut untouched);
                        assert_eq!(untouched, canonical, "{order:?} reordered {other:?}");
                    }
                }
            }
        }
    }

    const SPHERE_RADIUS: f32 = 0.5;
    const GRAVITY_Y: f32 = -9.8;

    /// Translational plus rotational kinetic energy. Inertia is the scalar
    /// isotropic moment `EuclideanR3` uses, so the rotational term is
    /// `½·I·|ω|²`.
    fn kinetic_energy(body: &RigidBody<EuclideanR3>) -> f32 {
        let omega = body.angular_velocity.magnitude();
        0.5 * body.mass * body.velocity.length_squared() + 0.5 * body.inertia * omega * omega
    }

    /// A perfectly elastic (`e = 1`) impact must return exactly the incoming
    /// kinetic energy: no loss, and no gain from the Baumgarte term riding
    /// along in `velocity_bias`.
    ///
    /// `dt` is chosen so one step of approach buries less than
    /// [`PENETRATION_SLOP`]; the positional bias is then identically zero and
    /// the rebound is pure restitution. A coarser step would legitimately add
    /// energy, which is why this contract is stated per-impact rather than as a
    /// global energy budget.
    #[test]
    fn perfectly_elastic_rebound_conserves_kinetic_energy() {
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);

        let approach = 2.0;
        let dt = 1.0 / 1000.0;
        assert!(
            approach > RESTITUTION_THRESHOLD,
            "below the threshold restitution is deliberately suppressed",
        );
        assert!(
            approach * dt < PENETRATION_SLOP,
            "a deeper first-frame burial admits a Baumgarte contribution",
        );

        // Start clear of the floor so the impact is produced by the sim rather
        // than by an initial condition already inside the plane.
        let start_gap = 0.01;
        let sphere = world.push_body(sphere_body_r3(
            Vec3::new(0.0, SPHERE_RADIUS + start_gap, 0.0),
            Vec3::new(0.0, -approach, 0.0),
            SPHERE_RADIUS,
            1.0,
        ));
        let floor = world.push_body(halfspace_body_r3(Vec3::Y, 0.0));
        world.bodies[sphere].restitution = 1.0;
        world.bodies[floor].restitution = 1.0;

        let energy_before = kinetic_energy(&world.bodies[sphere]);
        // Long enough to close `start_gap`, bounce, and separate again.
        let steps = (4.0 * start_gap / (approach * dt)).ceil() as usize;
        for _ in 0..steps {
            world.step(dt);
        }

        let body = &world.bodies[sphere];
        assert!(
            body.velocity.y > 0.0,
            "sphere did not rebound: v_y = {}",
            body.velocity.y
        );
        // The contact lies on the line through the centre of mass, so friction
        // and torque have no lever arm and every joule stays translational.
        assert!(
            body.angular_velocity.magnitude() < 1e-6,
            "central impact spun the sphere: |ω| = {}",
            body.angular_velocity.magnitude()
        );

        let energy_after = kinetic_energy(body);
        let ratio = energy_after / energy_before;
        assert!(
            ratio <= 1.0 + 1e-4,
            "e = 1 impact added energy: {energy_before} -> {energy_after}"
        );
        assert!(
            ratio >= 1.0 - 1e-4,
            "e = 1 impact lost energy: {energy_before} -> {energy_after}"
        );
    }

    /// Coulomb's cone, `|jt| ≤ μ·jn`, must hold at every contact on every step,
    /// and must actually bind at least once: a solver that never applied
    /// friction would satisfy the inequality vacuously.
    #[test]
    fn tangent_impulse_stays_inside_the_coulomb_cone() {
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        world.push_field(Box::new(Gravity::new(Vec3::new(0.0, GRAVITY_Y, 0.0))));

        let slide_speed = 5.0;
        let sphere = world.push_body(sphere_body_r3(
            Vec3::new(0.0, SPHERE_RADIUS, 0.0),
            Vec3::new(slide_speed, 0.0, 0.0),
            SPHERE_RADIUS,
            1.0,
        ));
        let floor = world.push_body(halfspace_body_r3(Vec3::Y, 0.0));
        world.bodies[sphere].restitution = 0.0;
        world.bodies[floor].restitution = 0.0;

        let dt = 1.0 / 240.0;
        let mut cone_ever_binds = false;
        for _ in 0..240 {
            world.step(dt);
            for manifold in world.manifolds.values() {
                for cp in &manifold.points {
                    let cap = cp.normal_impulse * FRICTION_COEFF;
                    // The clamp is applied against the normal impulse of the
                    // same iteration, so the bound holds on the state the step
                    // leaves behind, not merely in expectation. The 1e-6 slack
                    // covers f32 accumulation across `pgs_iters` passes; the
                    // impulses here are of order 1e-2, so it cannot hide a
                    // widened cone.
                    assert!(
                        cp.tangent_impulse <= cap + 1e-6,
                        "friction escaped the cone: jt = {}, μ·jn = {cap}",
                        cp.tangent_impulse
                    );
                    assert!(
                        cp.tangent_impulse >= 0.0,
                        "tangent accumulator went negative: {}",
                        cp.tangent_impulse
                    );
                    if cap > 1e-6 && cp.tangent_impulse >= cap - 1e-6 {
                        cone_ever_binds = true;
                    }
                }
            }
        }

        assert!(
            cone_ever_binds,
            "friction never saturated, so the clamp was never exercised"
        );
        let body = &world.bodies[sphere];
        assert!(
            body.velocity.x < slide_speed,
            "friction did not brake the slide: v_x = {}",
            body.velocity.x
        );
        assert!(
            body.angular_velocity.magnitude() > 1e-3,
            "friction applied no torque: |ω| = {}",
            body.angular_velocity.magnitude()
        );
    }

    /// Three spheres resting on the floor, run long enough that the manifolds
    /// carry converged accumulated impulses.
    fn settled_sphere_stack(dt: f32, settle_steps: usize) -> World<EuclideanR3> {
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        world.push_field(Box::new(Gravity::new(Vec3::new(0.0, GRAVITY_Y, 0.0))));

        for level in 0..3 {
            let y = SPHERE_RADIUS + level as f32 * 2.0 * SPHERE_RADIUS;
            let id = world.push_body(sphere_body_r3(
                Vec3::new(0.0, y, 0.0),
                Vec3::ZERO,
                SPHERE_RADIUS,
                1.0,
            ));
            world.bodies[id].restitution = 0.0;
        }
        let floor = world.push_body(halfspace_body_r3(Vec3::Y, 0.0));
        world.bodies[floor].restitution = 0.0;

        for _ in 0..settle_steps {
            world.step(dt);
        }
        world
    }

    /// Discard the cached normal impulses so the next step solves from zero.
    fn clear_warm_start(world: &mut World<EuclideanR3>) {
        for manifold in world.manifolds.values_mut() {
            for cp in &mut manifold.points {
                cp.normal_impulse = 0.0;
            }
        }
    }

    fn stack_velocities(world: &World<EuclideanR3>) -> Vec<Vec3> {
        world.bodies.iter().map(|b| b.velocity).collect()
    }

    /// Accumulated normal impulses in `BTreeMap` then slot order, which is the
    /// same order in two worlds built and settled by the same code path.
    fn stack_normal_impulses(world: &World<EuclideanR3>) -> Vec<f32> {
        world
            .manifolds
            .values()
            .flat_map(|m| m.points.iter().map(|cp| cp.normal_impulse))
            .collect()
    }

    fn max_component_gap(a: &[Vec3], b: &[Vec3]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x - *y).abs().max_element())
            .fold(0.0_f32, f32::max)
    }

    fn max_scalar_gap(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "manifold layouts diverged");
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    /// Warm-starting is an initial guess, not a different constraint problem:
    /// re-applying the cached impulses must leave the default-iteration solve
    /// at the same fixed point a cold solve reaches with iterations to spare.
    /// If it did not, parallelizing the solver would be chasing a moving
    /// target.
    ///
    /// The second assert pins the reason warm-starting exists: at equal
    /// iteration count it must be strictly closer to the converged answer than
    /// a cold start is.
    #[test]
    fn warm_started_step_matches_cold_started_converged_step() {
        let dt = 1.0 / 240.0;
        let settle_steps = 400;

        let mut warm = settled_sphere_stack(dt, settle_steps);
        let mut cold_converged = settled_sphere_stack(dt, settle_steps);
        let mut cold_default = settled_sphere_stack(dt, settle_steps);

        assert!(
            warm.manifolds
                .values()
                .flat_map(|m| &m.points)
                .any(|cp| cp.normal_impulse > 0.0),
            "fixture carries no warm-start payload"
        );

        clear_warm_start(&mut cold_converged);
        clear_warm_start(&mut cold_default);
        // The reference solution, not another partial solve: raising this to
        // 4000 moves neither assert below.
        cold_converged.pgs_iters = 400;

        warm.step(dt);
        cold_converged.step(dt);
        cold_default.step(dt);

        let converged = stack_velocities(&cold_converged);
        let warm_gap = max_component_gap(&stack_velocities(&warm), &converged);
        let cold_gap = max_component_gap(&stack_velocities(&cold_default), &converged);

        // Tolerance is on the velocity a single 1/240 s step imparts; gravity
        // alone contributes 0.04 m/s per step, so 1e-5 is a tight fraction of
        // the quantity under test and three orders below the cold-start
        // residual it is distinguishing itself from.
        assert!(
            warm_gap < 1e-5,
            "warm-started step diverged from the converged solve by {warm_gap} m/s"
        );
        assert!(
            warm_gap < cold_gap,
            "warm start bought no convergence: warm {warm_gap} vs cold {cold_gap}"
        );

        // Velocities can land on target while the accumulator that produced
        // them is wrong, because the solve corrects whatever the warm start
        // applied. Pinning the accumulator too is what makes the carried state
        // safe to reuse next step, which is the property a per-island parallel
        // solve has to preserve.
        let impulse_gap = max_scalar_gap(
            &stack_normal_impulses(&warm),
            &stack_normal_impulses(&cold_converged),
        );
        let reference = stack_normal_impulses(&cold_converged)
            .into_iter()
            .fold(0.0_f32, f32::max);
        assert!(
            impulse_gap < 1e-3 * reference,
            "warm-started accumulator diverged by {impulse_gap} against a peak impulse of {reference}"
        );
    }
}
