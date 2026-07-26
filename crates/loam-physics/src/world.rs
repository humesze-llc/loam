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

use crate::body::{BodyArena, BodyId, RigidBody};
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
/// one canonical key regardless of broadphase iteration order. Keyed on
/// [`BodyId`] rather than on storage position, so a manifold and its
/// warm-start impulses survive a despawn that compacts the arena.
pub type PairKey = (BodyId, BodyId);

fn canonical_pair(a: BodyId, b: BodyId) -> PairKey {
    debug_assert_ne!(a, b, "a body cannot pair with itself");
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

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
const STALE_MANIFOLD_BODY: &str = "manifold key names a body that is gone";

/// What each phase loop actually iterated, pushed from the loop's own control
/// variable. Reading the retained buffer instead would agree with the schedule
/// by construction and could not catch a loop head that walks a freshly built
/// list, which is the failure the buffer-level pin cannot see.
#[cfg(test)]
#[derive(Default)]
struct VisitLog {
    apply_forces: Vec<usize>,
    integrate: Vec<usize>,
    update_manifolds: Vec<PairKey>,
    prepare_solve: Vec<PairKey>,
    warm_start: Vec<PairKey>,
    /// Every PGS sweep, concatenated, rather than one sampled sweep: a loop
    /// head that reads the ordered buffer on the first pass and a rebuilt list
    /// afterwards is a live failure mode that a first-only or last-only log
    /// cannot see. Sweep boundaries are recoverable from the key count, so a
    /// flat buffer avoids a per-sweep allocation.
    solve_sweeps: Vec<PairKey>,
}

pub struct World<S: PhysicsSpace> {
    pub space: S,
    pub bodies: BodyArena<S>,
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
    #[cfg(test)]
    visit_log: VisitLog,
}

impl<S: PhysicsSpace> World<S> {
    pub fn new(space: S) -> Self {
        Self {
            space,
            bodies: BodyArena::new(),
            fields: Vec::new(),
            narrowphase: Narrowphase::new(),
            manifolds: BTreeMap::new(),
            pgs_iters: DEFAULT_PGS_ITERS,
            time: 0.0,
            schedule: Schedule::default(),
            body_order: Vec::new(),
            pair_order: Vec::new(),
            constraint_keys: Vec::new(),
            #[cfg(test)]
            visit_log: VisitLog::default(),
        }
    }

    /// Add a body to the world; returns its handle.
    pub fn push_body(&mut self, body: RigidBody<S>) -> BodyId {
        self.bodies.spawn(body)
    }

    /// Remove a body and every manifold it takes part in. Returns false if the
    /// handle is stale. Dropping the manifolds here rather than leaving them
    /// for the next step's eviction keeps `manifolds` free of keys that name
    /// no live body, so a caller inspecting it between steps sees the world it
    /// actually has.
    pub fn despawn_body(&mut self, id: BodyId) -> bool {
        if self.bodies.despawn(id).is_none() {
            return false;
        }
        self.manifolds.retain(|&(a, b), _| a != id && b != id);
        true
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
        #[cfg(test)]
        self.visit_log.apply_forces.clear();
        let order = std::mem::take(&mut self.body_order);
        for &i in &order {
            #[cfg(test)]
            self.visit_log.apply_forces.push(i);
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
        #[cfg(test)]
        self.visit_log.integrate.clear();
        let order = std::mem::take(&mut self.body_order);
        for &i in &order {
            #[cfg(test)]
            self.visit_log.integrate.push(i);
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
        #[cfg(test)]
        self.visit_log.update_manifolds.clear();

        for &key in &pairs {
            #[cfg(test)]
            self.visit_log.update_manifolds.push(key);
            let (i, j) = self.dense_pair(key);
            let (a, b) = split_two_mut(&mut self.bodies, i, j);
            let Some(contact) = self.narrowphase.test(a, b, &self.space) else {
                continue;
            };
            touched.insert(key);
            let restitution = contact.restitution;
            let manifold = self
                .manifolds
                .entry(key)
                .or_insert_with(|| Manifold::new(key.0, key.1, restitution));
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
        #[cfg(test)]
        self.visit_log.prepare_solve.clear();
        let keys = std::mem::take(&mut self.constraint_keys);
        for key in &keys {
            #[cfg(test)]
            self.visit_log.prepare_solve.push(*key);
            let (i, j) = self.dense_pair(*key);
            let manifold = self.manifolds.get_mut(key).expect(STALE_CONSTRAINT_KEY);
            let (a, b) = split_two_mut(&mut self.bodies, i, j);
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
        #[cfg(test)]
        self.visit_log.warm_start.clear();
        let keys = std::mem::take(&mut self.constraint_keys);
        for key in &keys {
            #[cfg(test)]
            self.visit_log.warm_start.push(*key);
            let (i, j) = self.dense_pair(*key);
            let manifold = self.manifolds.get(key).expect(STALE_CONSTRAINT_KEY);
            let (a, b) = split_two_mut(&mut self.bodies, i, j);
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
        #[cfg(test)]
        self.visit_log.solve_sweeps.clear();
        let keys = std::mem::take(&mut self.constraint_keys);
        for _ in 0..self.pgs_iters {
            for key in &keys {
                #[cfg(test)]
                self.visit_log.solve_sweeps.push(*key);
                let (i, j) = self.dense_pair(*key);
                let manifold = self.manifolds.get_mut(key).expect(STALE_CONSTRAINT_KEY);
                let (a, b) = split_two_mut(&mut self.bodies, i, j);
                for cp in &mut manifold.points {
                    solve_normal_then_tangent(&self.space, a, b, cp);
                }
            }
        }
        self.constraint_keys = keys;
    }

    /// All-pairs broadphase: one canonical [`PairKey`] per candidate pair,
    /// ordered by [`BodyId`] and not by storage position. Pairs of two static
    /// bodies are skipped.
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
                pairs.push(canonical_pair(self.bodies.id_at(i), self.bodies.id_at(j)));
            }
        }
    }

    /// Storage positions of a key's two bodies, in the key's own order. Both
    /// resolve: a manifold outlives neither its bodies (`despawn_body` drops
    /// it) nor the step that stopped touching it (`update_manifolds` evicts
    /// it).
    fn dense_pair(&self, key: PairKey) -> (usize, usize) {
        (
            self.bodies.dense_index(key.0).expect(STALE_MANIFOLD_BODY),
            self.bodies.dense_index(key.1).expect(STALE_MANIFOLD_BODY),
        )
    }
}

/// Split-borrow `&mut slice[i]` and `&mut slice[j]` simultaneously, returned in
/// argument order. Caller must ensure `i != j`. The two are ordered by
/// [`BodyId`], not by storage position, so either may be the lower index.
fn split_two_mut<T>(slice: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    debug_assert_ne!(i, j, "split_two_mut requires distinct indices");
    if i < j {
        let (left, right) = slice.split_at_mut(j);
        (&mut left[i], &mut right[0])
    } else {
        let (left, right) = slice.split_at_mut(i);
        (&mut right[0], &mut left[j])
    }
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
        determinism_scenario_run, first_divergent_step, fnv1a64, multi_island_groups,
        multi_island_scenario_run, multi_island_world, ScenarioRun, GOLDEN_MULTI_ISLAND_HASH,
        GOLDEN_TRAJECTORY_HASH, MULTI_ISLAND_DT, MULTI_ISLAND_STEPS,
    };
    use crate::euclidean_r3::{
        box_body, halfspace_body_r3, register_default_narrowphase, sphere_body_r3,
    };
    use crate::field::Gravity;
    use glam::Vec3;
    use loam_math::{Bivector3, EuclideanR3};

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

    /// `expected` is `canonical` reversed exactly when `order` names `owner`,
    /// so one helper carries both halves of the contract: the named phase's
    /// buffer moves, and no other phase's buffer does.
    fn assert_buffer_matches_policy<T>(
        order: OrderPolicy,
        owner: SchedulePhase,
        buffer: &[T],
        canonical: &[T],
    ) where
        T: Clone + PartialEq + std::fmt::Debug,
    {
        let mut expected = canonical.to_vec();
        if matches!(order, OrderPolicy::Reversed { phase } if phase == owner) {
            expected.reverse();
        }
        assert_eq!(
            buffer,
            expected.as_slice(),
            "under {order:?} the retained {owner:?} buffer is wrong: the policy \
             either never reached that phase or reached a different one"
        );
    }

    /// The invariance axes above compare a canonical run against a canonical
    /// run whenever a policy fails to reach the buffer its phase executes, so
    /// on their own they cannot tell "this order does not matter" apart from
    /// "this order was never applied". This pins the seam directly: after a
    /// step, each phase's retained buffer is reversed exactly when the policy
    /// names that phase, and identical to its canonical fill otherwise.
    ///
    /// `Reversed` rather than a seeded permutation because its expected buffer
    /// is computable here without re-implementing `shuffle`.
    #[test]
    fn schedule_reordering_reaches_its_named_phase_buffer_determinism() {
        let dt = 1.0 / 240.0;
        let settle_steps = 200;

        for order in [
            OrderPolicy::Canonical,
            OrderPolicy::Reversed {
                phase: SchedulePhase::Body,
            },
            OrderPolicy::Reversed {
                phase: SchedulePhase::BroadphasePair,
            },
            OrderPolicy::Reversed {
                phase: SchedulePhase::Constraint,
            },
        ] {
            let mut world = settled_sphere_stack(dt, 0);
            world.schedule = Schedule { threads: 1, order };
            for _ in 0..settle_steps {
                world.step(dt);
            }

            let canonical_bodies: Vec<usize> = (0..world.bodies.len()).collect();
            let canonical_pairs = world.broadphase();
            let canonical_constraints: Vec<PairKey> = world.manifolds.keys().copied().collect();
            // A buffer of fewer than two units reverses to itself, which would
            // satisfy every assertion below without the seam existing.
            assert!(
                canonical_bodies.len() >= 2
                    && canonical_pairs.len() >= 2
                    && canonical_constraints.len() >= 2,
                "{order:?} left a buffer too short for a reversal to be visible: \
                 {} bodies, {} pairs, {} constraints",
                canonical_bodies.len(),
                canonical_pairs.len(),
                canonical_constraints.len()
            );

            assert_buffer_matches_policy(
                order,
                SchedulePhase::Body,
                &world.body_order,
                &canonical_bodies,
            );
            assert_buffer_matches_policy(
                order,
                SchedulePhase::BroadphasePair,
                &world.pair_order,
                &canonical_pairs,
            );
            assert_buffer_matches_policy(
                order,
                SchedulePhase::Constraint,
                &world.constraint_keys,
                &canonical_constraints,
            );
        }
    }

    /// [`schedule_reordering_reaches_its_named_phase_buffer_determinism`] shows
    /// the retained buffer was reordered, not that the phase loop read it. A
    /// loop head swapped for a freshly built list
    /// (`let pairs = self.broadphase();` in `update_manifolds`, `0..len` in
    /// `apply_forces` or `integrate`) leaves that pin and every invariance axis
    /// green while the ordered buffer goes unread. [`VisitLog`] records each
    /// loop's own control variable, so the visit order is observed from inside
    /// the loop and cannot agree with the buffer by construction.
    ///
    /// `Reversed` for the same reason as that pin: the expected order is
    /// computable here without re-implementing `shuffle`.
    ///
    /// The Constraint consumers are where the hole actually costs something:
    /// PGS is Gauss-Seidel, so a rebuilt key list there silently restores
    /// `BTreeMap` order and changes the converged answer. `solve` is checked on
    /// every sweep, not the first or the last, because a head that reads the
    /// ordered buffer once and rebuilds on later passes is exactly the
    /// half-broken case a sampled sweep would clear.
    #[test]
    fn phase_loops_visit_the_buffer_the_schedule_ordered_determinism() {
        let dt = 1.0 / 240.0;
        let settle_steps = 200;

        for order in [
            OrderPolicy::Canonical,
            OrderPolicy::Reversed {
                phase: SchedulePhase::Body,
            },
            OrderPolicy::Reversed {
                phase: SchedulePhase::BroadphasePair,
            },
            OrderPolicy::Reversed {
                phase: SchedulePhase::Constraint,
            },
        ] {
            let mut world = settled_sphere_stack(dt, 0);
            world.schedule = Schedule { threads: 1, order };
            for _ in 0..settle_steps {
                world.step(dt);
            }

            let canonical_bodies: Vec<usize> = (0..world.bodies.len()).collect();
            let canonical_pairs = world.broadphase();
            let canonical_constraints: Vec<PairKey> = world.manifolds.keys().copied().collect();
            assert!(
                canonical_bodies.len() >= 2
                    && canonical_pairs.len() >= 2
                    && canonical_constraints.len() >= 2,
                "{order:?} left a buffer too short for a reversal to be visible: \
                 {} bodies, {} pairs, {} constraints",
                canonical_bodies.len(),
                canonical_pairs.len(),
                canonical_constraints.len()
            );

            // Both Body-phase consumers, because each holds the buffer through
            // its own loop and either one can stop reading it alone.
            for (phase, visited) in [
                ("apply_forces", &world.visit_log.apply_forces),
                ("integrate", &world.visit_log.integrate),
            ] {
                assert_eq!(
                    visited, &world.body_order,
                    "{phase} under {order:?} visited a list other than the \
                     ordered body buffer"
                );
                assert_buffer_matches_policy(
                    order,
                    SchedulePhase::Body,
                    visited,
                    &canonical_bodies,
                );
            }

            assert_eq!(
                world.visit_log.update_manifolds, world.pair_order,
                "update_manifolds under {order:?} visited a list other than the \
                 ordered pair buffer"
            );
            assert_buffer_matches_policy(
                order,
                SchedulePhase::BroadphasePair,
                &world.visit_log.update_manifolds,
                &canonical_pairs,
            );

            // The two single-pass Constraint consumers. Each takes the buffer
            // for its own loop, so either can stop reading it alone.
            for (phase, visited) in [
                ("prepare_solve", &world.visit_log.prepare_solve),
                ("warm_start", &world.visit_log.warm_start),
            ] {
                assert_eq!(
                    visited, &world.constraint_keys,
                    "{phase} under {order:?} visited a list other than the \
                     ordered constraint buffer"
                );
                assert_buffer_matches_policy(
                    order,
                    SchedulePhase::Constraint,
                    visited,
                    &canonical_constraints,
                );
            }

            let sweep_len = world.constraint_keys.len();
            assert_eq!(
                world.visit_log.solve_sweeps.len(),
                sweep_len * world.pgs_iters,
                "solve under {order:?} logged {} visits, not {} sweeps of {sweep_len}",
                world.visit_log.solve_sweeps.len(),
                world.pgs_iters
            );
            for (sweep, visited) in world
                .visit_log
                .solve_sweeps
                .chunks_exact(sweep_len)
                .enumerate()
            {
                assert_eq!(
                    visited,
                    world.constraint_keys.as_slice(),
                    "solve sweep {sweep} under {order:?} visited a list other \
                     than the ordered constraint buffer"
                );
                assert_buffer_matches_policy(
                    order,
                    SchedulePhase::Constraint,
                    visited,
                    &canonical_constraints,
                );
            }
        }
    }

    /// The multi-island fixture's behaviour pin, on the same terms as the R4
    /// golden: deterministic-but-changed integration, solve, or contact
    /// constants move it.
    #[test]
    fn multi_island_scenario_matches_golden_determinism_hash() {
        let hash = fnv1a64(&multi_island_scenario_run(Schedule::default()).trajectory);
        assert_eq!(
            hash, GOLDEN_MULTI_ISLAND_HASH,
            "multi-island trajectory hashed {hash:#018x} against the committed \
             {GOLDEN_MULTI_ISLAND_HASH:#018x}"
        );
    }

    /// The fixture earns its name only if the contact graph really splits into
    /// the three groups it lays out and the four-body chain really rests as a
    /// chain. A layout edit that lets two groups touch, or that leaves the
    /// chain short of four simultaneous contacts, fails here rather than
    /// silently making the island-order and colour-order axes vacuous on the
    /// day they land.
    #[test]
    fn multi_island_contact_graph_stays_three_disjoint_islands_determinism() {
        let groups = multi_island_groups();
        let mut world = multi_island_world(Schedule::default());
        let start_x: Vec<f32> = world.bodies.iter().map(|b| b.position.x).collect();
        let mut contacts_per_group = [0usize; 3];
        let mut chain_contacts_peak = 0usize;

        for _ in 0..MULTI_ISLAND_STEPS {
            world.step(MULTI_ISLAND_DT);
            let mut this_step = [0usize; 3];
            for &(id_a, id_b) in world.manifolds.keys() {
                let (i, j) = (id_a.slot() as usize, id_b.slot() as usize);
                let a = groups.iter().position(|g| g.contains(&i));
                let b = groups.iter().position(|g| g.contains(&j));
                let group = match (a, b) {
                    (Some(x), Some(y)) => {
                        assert_eq!(x, y, "contact ({i}, {j}) joined islands {x} and {y}");
                        x
                    }
                    // The floor sits in every island's contact set and merges
                    // none of them: static, so it transmits no impulse.
                    (Some(x), None) | (None, Some(x)) => x,
                    (None, None) => panic!("contact ({i}, {j}) between two static bodies"),
                };
                this_step[group] += 1;
            }
            for (group, count) in this_step.iter().enumerate() {
                contacts_per_group[group] += count;
            }
            chain_contacts_peak = chain_contacts_peak.max(this_step[0]);
        }

        for (group, count) in contacts_per_group.iter().enumerate() {
            assert!(*count > 0, "island {group} never made contact");
        }
        assert_eq!(
            chain_contacts_peak, 4,
            "the four-body chain never rested as floor-A0, A0-A1, A1-A2, A2-A3"
        );
        // Bit equality, not a tolerance: the fixture's claim is that lateral
        // motion is identically absent, not merely small. A tolerance would let
        // a slow drift accumulate until the islands do meet.
        let end_x: Vec<f32> = world.bodies.iter().map(|b| b.position.x).collect();
        assert_eq!(
            end_x, start_x,
            "a body left its group's vertical axis, so the island partition is \
             not constant by construction"
        );
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

    /// Sphere centres on the x axis, several diameters apart, so no two
    /// spheres can reach each other for the length of a test.
    const ISLAND_X: [f32; 4] = [-4.0, 0.0, 4.0, 8.0];

    /// One sphere per island over one shared static floor, settled until every
    /// manifold carries a converged warm-start impulse. The floor is static,
    /// so it transmits no impulse and merges no islands: any influence one
    /// island shows from another is a defect, which is what makes bit equality
    /// the right assertion below.
    fn settled_islands(dt: f32, settle_steps: usize) -> (World<EuclideanR3>, BodyId, Vec<BodyId>) {
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        world.push_field(Box::new(Gravity::new(Vec3::new(0.0, GRAVITY_Y, 0.0))));

        let floor = world.push_body(halfspace_body_r3(Vec3::Y, 0.0));
        world.bodies[floor].restitution = 0.0;
        let mut spheres = Vec::with_capacity(ISLAND_X.len());
        for x in ISLAND_X {
            let id = world.push_body(island_sphere(x));
            world.bodies[id].restitution = 0.0;
            spheres.push(id);
        }

        for _ in 0..settle_steps {
            world.step(dt);
        }
        (world, floor, spheres)
    }

    fn island_sphere(x: f32) -> RigidBody<EuclideanR3> {
        sphere_body_r3(
            Vec3::new(x, SPHERE_RADIUS, 0.0),
            Vec3::ZERO,
            SPHERE_RADIUS,
            1.0,
        )
    }

    fn body_state(world: &World<EuclideanR3>, id: BodyId) -> (Vec3, Vec3, Bivector3) {
        let body = &world.bodies[id];
        (body.position, body.velocity, body.angular_velocity)
    }

    fn normal_impulses(world: &World<EuclideanR3>, key: PairKey) -> Vec<f32> {
        world.manifolds[&key]
            .points
            .iter()
            .map(|cp| cp.normal_impulse)
            .collect()
    }

    /// The property positional indices cannot have: despawning a body
    /// compacts storage, and every surviving manifold must keep both its key
    /// and its accumulated impulses across that move. Bit equality against a
    /// world that despawned nothing, not a tolerance: a disjoint island's
    /// trajectory is identically unchanged, and a tolerance would let a
    /// rebound from a rebound key hide inside it.
    #[test]
    fn despawn_preserves_surviving_manifold_keys_and_warm_start_impulses() {
        let dt = 1.0 / 240.0;
        let settle_steps = 400;
        let (mut world, floor, spheres) = settled_islands(dt, settle_steps);
        let (mut control, _, control_spheres) = settled_islands(dt, settle_steps);

        let doomed = spheres[1];
        let keeper = spheres[3];
        let keeper_key = (floor, keeper);
        let control_key = (floor, control_spheres[3]);

        let impulses_before = normal_impulses(&world, keeper_key);
        assert!(
            impulses_before.iter().any(|&jn| jn > 0.0),
            "fixture carries no warm-start payload, so nothing below is being preserved"
        );
        let keeper_position_before = world.bodies.dense_index(keeper).unwrap();

        assert!(world.despawn_body(doomed));

        assert_ne!(
            world.bodies.dense_index(keeper).unwrap(),
            keeper_position_before,
            "despawn moved no surviving body, so this test never reached the compaction case"
        );
        assert!(
            !world
                .manifolds
                .keys()
                .any(|&(a, b)| a == doomed || b == doomed),
            "the removed body's manifolds outlived it"
        );
        assert_eq!(
            normal_impulses(&world, keeper_key),
            impulses_before,
            "compaction disturbed a surviving manifold's warm-start impulses"
        );

        for _ in 0..60 {
            world.step(dt);
            control.step(dt);
        }

        assert_eq!(
            body_state(&world, keeper),
            body_state(&control, control_spheres[3]),
            "an unrelated despawn perturbed a surviving island"
        );
        let impulses_after = normal_impulses(&world, keeper_key);
        assert_eq!(
            impulses_after,
            normal_impulses(&control, control_key),
            "an unrelated despawn moved a surviving manifold's impulses"
        );
        assert!(
            impulses_after.iter().any(|&jn| jn > 0.0),
            "the surviving contact stopped carrying an impulse"
        );
    }

    /// The spawn half of the same contract: a body arriving mid-simulation
    /// gets its own manifold and leaves every existing one alone.
    #[test]
    fn spawn_mid_simulation_leaves_existing_islands_bit_identical() {
        let dt = 1.0 / 240.0;
        let settle_steps = 400;
        let (mut world, floor, spheres) = settled_islands(dt, settle_steps);
        let (mut control, _, control_spheres) = settled_islands(dt, settle_steps);

        let keeper = spheres[0];
        let keeper_key = (floor, keeper);
        let newcomer = world.push_body(island_sphere(12.0));
        world.bodies[newcomer].restitution = 0.0;

        for _ in 0..60 {
            world.step(dt);
            control.step(dt);
        }

        assert!(
            world.manifolds.contains_key(&(floor, newcomer)),
            "the spawned body never made contact, so it exercised no solver state"
        );
        assert_eq!(
            body_state(&world, keeper),
            body_state(&control, control_spheres[0]),
            "a spawn perturbed an existing island"
        );
        assert_eq!(
            normal_impulses(&world, keeper_key),
            normal_impulses(&control, (floor, control_spheres[0])),
            "a spawn moved an existing manifold's warm-start impulses"
        );
    }

    /// The aliasing failure the generation exists to prevent, at world scope:
    /// a recycled slot must not inherit the previous occupant's contacts, and
    /// the old handle must be rejected rather than resolve to the new body.
    #[test]
    fn a_recycled_slot_inherits_no_manifold_from_the_previous_body() {
        let dt = 1.0 / 240.0;
        let (mut world, floor, spheres) = settled_islands(dt, 400);
        let doomed = spheres[1];
        assert!(world.manifolds.contains_key(&(floor, doomed)));

        assert!(world.despawn_body(doomed));
        assert!(
            !world.despawn_body(doomed),
            "a stale handle despawned a second body"
        );
        assert!(world.bodies.get(doomed).is_none());

        let reborn = world.push_body(island_sphere(ISLAND_X[1]));
        assert_eq!(
            reborn.slot(),
            doomed.slot(),
            "the slot was not recycled, so this test is not exercising aliasing"
        );
        assert_ne!(reborn, doomed);
        assert!(
            world.bodies.get(doomed).is_none(),
            "the stale handle resolved to the body that took its slot"
        );

        world.step(dt);
        assert!(
            world.manifolds.contains_key(&(floor, reborn)),
            "the respawned body made no contact"
        );
        assert!(
            !world.manifolds.contains_key(&(floor, doomed)),
            "a manifold keyed on the despawned body came back with the slot"
        );
    }

    /// `dense_pair` hands back two storage positions in its key's order, so
    /// the caller's first index has to come back as the first borrow whichever
    /// side of the split it lands on.
    #[test]
    fn split_two_mut_returns_borrows_in_argument_order() {
        let mut slice = [0u32, 1, 2, 3];
        for (i, j) in [(1usize, 3usize), (3, 1)] {
            let (a, b) = split_two_mut(&mut slice, i, j);
            assert_eq!((*a, *b), (i as u32, j as u32), "split_two_mut({i}, {j})");
        }
    }

    /// Two dynamic boxes resting on the static floor, plus a disjoint fourth
    /// body whose despawn decides the surviving pair's storage order: spawned
    /// before the pair, it sits below them and its removal swaps the upper box
    /// down past the lower one; spawned after, its removal moves nothing.
    /// Either way the world ends up holding the same three bodies in the same
    /// configuration, so storage order is the only variable between the two.
    ///
    /// Boxes rather than spheres because the box pair runs GJK + EPA, whose
    /// result depends on which hull is the Minkowski minuend. The sphere
    /// narrowphases and the impulse response are exactly antisymmetric under an
    /// operand swap, so a sphere pair would settle to the same state either way
    /// and could not tell the two orders apart.
    fn stacked_pair_world(doomed_first: bool) -> (World<EuclideanR3>, BodyId, BodyId, BodyId) {
        const LOWER_HALF_EXTENT: f32 = 0.5;
        const UPPER_HALF_EXTENT: f32 = 0.35;
        const DOOMED_X: f32 = -6.0;

        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        world.push_field(Box::new(Gravity::new(Vec3::new(0.0, GRAVITY_Y, 0.0))));

        let floor = world.push_body(halfspace_body_r3(Vec3::Y, 0.0));
        world.bodies[floor].restitution = 0.0;

        let spawned_first = doomed_first.then(|| world.push_body(island_sphere(DOOMED_X)));
        let lower = world.push_body(box_body(
            Vec3::new(0.0, LOWER_HALF_EXTENT, 0.0),
            Vec3::ZERO,
            Vec3::splat(LOWER_HALF_EXTENT),
            1.0,
        ));
        let upper = world.push_body(box_body(
            Vec3::new(0.0, 2.0 * LOWER_HALF_EXTENT + UPPER_HALF_EXTENT, 0.0),
            Vec3::ZERO,
            Vec3::splat(UPPER_HALF_EXTENT),
            3.0,
        ));
        let doomed = spawned_first.unwrap_or_else(|| world.push_body(island_sphere(DOOMED_X)));

        for id in [lower, upper, doomed] {
            world.bodies[id].restitution = 0.0;
        }
        (world, lower, upper, doomed)
    }

    /// Settle the pair, drop the fourth body, and return the world with the
    /// pair's key. Asserts the pair is in contact and that the despawn left
    /// storage order in the state the caller asked for, so a fixture that stops
    /// reaching the disagreeing case fails instead of going quiet.
    fn despawned_pair_world(
        doomed_first: bool,
        dt: f32,
    ) -> (World<EuclideanR3>, BodyId, BodyId, PairKey) {
        const SETTLE_STEPS: usize = 40;

        let (mut world, lower, upper, doomed) = stacked_pair_world(doomed_first);
        for _ in 0..SETTLE_STEPS {
            world.step(dt);
        }

        let key = canonical_pair(lower, upper);
        assert!(
            world.manifolds.contains_key(&key),
            "the two dynamic bodies never settled into contact"
        );
        assert!(world.despawn_body(doomed));

        let (i, j) = world.dense_pair(key);
        assert_eq!(
            i > j,
            doomed_first,
            "the pair is stored at {i}, {j}, which is not the order this fixture \
             was built to produce"
        );
        (world, lower, upper, key)
    }

    /// A manifold's contact normal is documented as pointing from `body_a`
    /// toward `body_b`, and `body_a` is its key's low handle. Nothing ties the
    /// key to storage, so taking the pair in storage order flips every normal a
    /// manifold carries whenever a despawn has moved one body below its
    /// partner.
    #[test]
    fn contact_normal_points_from_the_pair_key_low_body_to_the_high_one() {
        let dt = 1.0 / 240.0;
        for doomed_first in [false, true] {
            let (mut world, _, _, key) = despawned_pair_world(doomed_first, dt);
            world.step(dt);

            let manifold = world.manifolds.get(&key).expect("the pair separated");
            let key_axis = world.bodies[key.1].position - world.bodies[key.0].position;
            assert!(!manifold.points.is_empty(), "manifold carries no contact");
            for cp in &manifold.points {
                assert!(
                    cp.normal.dot(key_axis) > 0.0,
                    "normal {} points back toward the key's low body",
                    cp.normal
                );
            }
        }
    }

    /// Which slot the arena happens to store a body in is not physics, so it
    /// must not reach the pair's state at all. Bit equality against the
    /// control, not a tolerance: both worlds run the same arithmetic on the
    /// same inputs, so a correct solver leaves no error term to bound.
    #[test]
    fn storage_order_does_not_reach_a_contacting_pairs_trajectory() {
        let dt = 1.0 / 240.0;
        let steps = 120;
        let mut trajectories = Vec::new();
        for doomed_first in [false, true] {
            let (mut world, lower, upper, key) = despawned_pair_world(doomed_first, dt);
            let mut contact_steps = 0;
            let mut trajectory = Vec::with_capacity(steps);
            for _ in 0..steps {
                world.step(dt);
                if world.manifolds.contains_key(&key) {
                    contact_steps += 1;
                }
                trajectory.push((body_state(&world, lower), body_state(&world, upper)));
            }
            // A pair that separates stops reaching the solver, and the rest of
            // the comparison is then two ballistic arcs agreeing for free.
            assert_eq!(
                contact_steps, steps,
                "the pair held contact for only {contact_steps} of {steps} steps"
            );
            trajectories.push(trajectory);
        }

        let step = trajectories[0]
            .iter()
            .zip(&trajectories[1])
            .position(|(a, b)| a != b);
        assert!(
            step.is_none(),
            "storage order reached the solve: the pair diverged from the control \
             at step {step:?}"
        );
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
