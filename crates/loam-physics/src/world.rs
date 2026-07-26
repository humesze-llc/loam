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
//!
//! ## Islands
//!
//! The constraint buffer is grouped into [`Island`]s, the connected components
//! of the contact graph over dynamic bodies, so a solve pass over one island
//! reads and writes no body another island touches. Grouping is a reordering
//! of independent work and leaves the solve bit-identical; it is what makes an
//! island the unit a parallel solver can take whole.

use std::collections::{BTreeMap, HashSet};

use crate::body::{BodyArena, BodyId, RigidBody};
use crate::collider::Collider;
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

/// One connected component of the contact graph: a set of bodies no other
/// island's solve can reach, and the constraints coupling them.
///
/// Membership is over dynamic bodies only. A static body absorbs no impulse and
/// its state is invariant under the solve, so two groups resting on one floor
/// are two islands rather than one.
///
/// Instrumentation, on [`Schedule`]'s terms: the partition is the claim that a
/// parallel solver can take one island whole, and the three fields below are
/// the readout that checks it, so both ship in the binary the claim is about
/// rather than behind `cfg(test)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Island {
    /// Lowest handle among [`Self::bodies`]. A function of the partition alone,
    /// so one contact set names its islands the same way however the pairs
    /// producing it were discovered or stored.
    pub id: BodyId,
    /// The island's dynamic bodies, ascending.
    pub bodies: Vec<BodyId>,
    /// The manifolds coupling them, ascending. A contact against a static body
    /// belongs to the island of its dynamic side.
    pub constraints: Vec<PairKey>,
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

/// Relative widening applied to each sweep interval. The cull rests on
/// `|d(anchor, a) − d(anchor, b)| ≤ d(a, b)`, the triangle inequality for the
/// Riemannian distance function (do Carmo 1992, *Riemannian Geometry*, ch. 7,
/// prop. 3.6), which holds exactly in R but not in f32: each of the three
/// distances carries a few ulps of error, so a pair within an ulp of tangency
/// could be culled and the emitted set would stop being a function of the body
/// set alone. Four eps per side covers all three error terms.
const BROADPHASE_TRIANGLE_SLACK: f32 = 4.0 * f32::EPSILON;

/// One body's interval on the sweep axis: geodesic distance to the anchor,
/// widened by the body's bounding radius.
#[derive(Clone, Copy)]
struct RadialInterval {
    lo: f32,
    hi: f32,
    radius: f32,
    /// Storage position at fill time. The arena cannot change mid-sweep, so
    /// this stays valid without carrying `S::Point` through a generic entry.
    dense: u32,
    id: BodyId,
    dynamic: bool,
}

/// Radius of the smallest ball about a body's position that contains its
/// collider; infinite for a collider of unbounded extent. Every narrowphase
/// poses local geometry as `rotation · v + position` and a rotation preserves
/// norms, so the largest local vertex norm bounds the body at any orientation.
/// `Sphere`'s and `HyperSphere4D`'s `center` is ignored for the same reason the
/// narrowphases ignore it: in physics the body position is the centre.
fn bounding_radius(collider: &Collider) -> f32 {
    match collider {
        Collider::Sphere { radius, .. } | Collider::HyperSphere4D { radius, .. } => *radius,
        Collider::Box3 { half_extents } => half_extents.length(),
        Collider::Polygon2D { vertices } => max_norm(vertices.iter().map(|v| v.length_squared())),
        Collider::ConvexPolytope3D { vertices } => {
            max_norm(vertices.iter().map(|v| v.length_squared()))
        }
        Collider::ConvexPolytope4D { vertices } => {
            max_norm(vertices.iter().map(|v| v.length_squared()))
        }
        Collider::HalfSpace { .. } | Collider::HalfSpace4D { .. } => f32::INFINITY,
    }
}

fn max_norm(norms_squared: impl Iterator<Item = f32>) -> f32 {
    norms_squared.fold(0.0_f32, f32::max).sqrt()
}

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
    /// Broadphase scratch: the sweep's sorted intervals and its active list.
    /// Retained on the same terms as the work-unit buffers above, so a step
    /// with a steady body count never reaches the allocator.
    broadphase_intervals: Vec<RadialInterval>,
    broadphase_active: Vec<u32>,
    /// Island scratch: the union-find forest and the per-body island label,
    /// both indexed by dense position and both retained for the same reason.
    island_parent: Vec<u32>,
    island_labels: Vec<BodyId>,
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
            broadphase_intervals: Vec::new(),
            broadphase_active: Vec::new(),
            island_parent: Vec::new(),
            island_labels: Vec::new(),
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
    ///
    /// API, not instrumentation: it is the inverse of [`Self::push_body`] and
    /// the only removal that leaves the world's own state consistent, so a
    /// caller that can spawn has to be able to reach it.
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
        let mut intervals = std::mem::take(&mut self.broadphase_intervals);
        let mut active = std::mem::take(&mut self.broadphase_active);
        Self::fill_broadphase(
            &self.bodies,
            &self.space,
            &mut intervals,
            &mut active,
            &mut pairs,
        );
        self.broadphase_intervals = intervals;
        self.broadphase_active = active;
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
            let (a, b) = split_two_mut(self.bodies.dense_mut(), i, j);
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

    /// Refill the constraint buffer grouped by island, islands ascending by id
    /// and constraints ascending by key inside each, then hand it to the
    /// schedule. One buffer serves `prepare_solve`, `warm_start`, and `solve`,
    /// so those three always agree on the constraint order. Nothing between
    /// here and the end of the solve inserts or removes a manifold, which is
    /// why those three phases can index by key without a fallback.
    ///
    /// Grouping only moves constraints across island boundaries, and the
    /// bodies two islands write are disjoint, so the solve it produces is the
    /// one the ungrouped buffer produced, bit for bit.
    fn collect_constraints(&mut self) {
        let mut parent = std::mem::take(&mut self.island_parent);
        let mut labels = std::mem::take(&mut self.island_labels);
        Self::fill_islands(
            &self.bodies,
            self.manifolds.keys().copied(),
            &mut parent,
            &mut labels,
        );
        self.constraint_keys.clear();
        self.constraint_keys.extend(self.manifolds.keys().copied());
        let bodies = &self.bodies;
        self.constraint_keys
            .sort_unstable_by_key(|&key| (constraint_island(bodies, &labels, key), key));
        self.island_parent = parent;
        self.island_labels = labels;
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
            let (a, b) = split_two_mut(self.bodies.dense_mut(), i, j);
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
            let (a, b) = split_two_mut(self.bodies.dense_mut(), i, j);
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
                let (a, b) = split_two_mut(self.bodies.dense_mut(), i, j);
                for cp in &mut manifold.points {
                    solve_normal_then_tangent(&self.space, a, b, cp);
                }
            }
        }
        self.constraint_keys = keys;
    }

    /// Candidate pairs for the current body configuration: one canonical
    /// [`PairKey`] per pair whose bounding balls overlap, in ascending key
    /// order, skipping pairs of two static bodies. Allocating form, for callers
    /// outside the step loop; the step sweeps into buffers the world retains.
    pub fn broadphase(&self) -> Vec<PairKey> {
        let mut pairs = Vec::new();
        Self::fill_broadphase(
            &self.bodies,
            &self.space,
            &mut Vec::new(),
            &mut Vec::new(),
            &mut pairs,
        );
        pairs
    }

    /// Sort-and-sweep broadphase, emitting one canonical [`PairKey`] per
    /// candidate pair in ascending key order.
    ///
    /// A candidate is a pair that is not two static bodies and whose bounding
    /// balls overlap, `d(a, b) ≤ r_a + r_b`; the same test the polytope
    /// narrowphases already apply before entering GJK. That predicate, not the
    /// acceleration structure, defines the emitted set, so the set is a
    /// function of the body set alone and the sweep is free to prune however it
    /// likes. Emission order is likewise a function of the handles and not of
    /// storage position or discovery order, which is what lets a partitioned
    /// executor reproduce it.
    ///
    /// The sweep axis is geodesic distance to the lowest-handle body: one
    /// `distance` call per body, and defined in a curved space where a
    /// coordinate axis is not. Interval overlap along it is necessary for ball
    /// overlap by the triangle inequality, so the one-axis sweep (Cohen, Lin,
    /// Manocha, Ponamgi 1995, "I-COLLIDE", sec. 3) carries over unchanged. It
    /// degenerates to all-pairs when every body is equidistant from the anchor,
    /// which a coordinate grid would not; the grid is the upgrade once a space
    /// can hand out chart coordinates.
    fn fill_broadphase(
        bodies: &BodyArena<S>,
        space: &S,
        intervals: &mut Vec<RadialInterval>,
        active: &mut Vec<u32>,
        pairs: &mut Vec<PairKey>,
    ) {
        pairs.clear();
        intervals.clear();
        active.clear();
        let n = bodies.len();
        if n < 2 {
            return;
        }

        let anchor = (0..n)
            .min_by_key(|&dense| bodies.id_at(dense))
            .expect("a non-empty arena has a lowest handle");
        let origin = bodies[anchor].position;

        for dense in 0..n {
            let body = &bodies[dense];
            let radius = bounding_radius(&body.collider);
            let d = space.distance(origin, body.position);
            let slack = d * BROADPHASE_TRIANGLE_SLACK;
            intervals.push(RadialInterval {
                lo: d - radius - slack,
                hi: d + radius + slack,
                radius,
                dense: dense as u32,
                id: bodies.id_at(dense),
                dynamic: body.inv_mass != 0.0,
            });
        }
        // Unstable sort: the stable one allocates a scratch buffer, and the
        // handle tie-break already makes the order total.
        intervals.sort_unstable_by(|a, b| a.lo.total_cmp(&b.lo).then(a.id.cmp(&b.id)));

        for i in 0..n {
            let entry = intervals[i];
            // `lo` is non-decreasing, so an interval that ends before this one
            // starts also ends before every later one starts.
            active.retain(|&open| intervals[open as usize].hi >= entry.lo);
            for &open in active.iter() {
                let other = intervals[open as usize];
                if !entry.dynamic && !other.dynamic {
                    continue;
                }
                let gap = space.distance(
                    bodies[other.dense as usize].position,
                    bodies[entry.dense as usize].position,
                );
                if gap <= other.radius + entry.radius {
                    pairs.push(canonical_pair(other.id, entry.id));
                }
            }
            active.push(i as u32);
        }

        pairs.sort_unstable();
    }

    /// The islands of the current manifold set, ascending by island id.
    /// Allocating form, for callers outside the step loop; the step groups its
    /// constraint buffer through the same partition without allocating. Public
    /// as the read side of [`Island`]'s instrumentation, not as a step API.
    ///
    /// Panics if `manifolds` names a body the arena no longer holds, which is
    /// reachable only between a bare [`BodyArena::despawn`] and the next
    /// [`Self::step`]; [`Self::despawn_body`] is the entry point that keeps the
    /// two consistent.
    pub fn islands(&self) -> Vec<Island> {
        let mut parent = Vec::new();
        let mut labels = Vec::new();
        Self::fill_islands(
            &self.bodies,
            self.manifolds.keys().copied(),
            &mut parent,
            &mut labels,
        );

        let mut by_id: BTreeMap<BodyId, Island> = BTreeMap::new();
        for &key in self.manifolds.keys() {
            let id = constraint_island(&self.bodies, &labels, key);
            let island = by_id.entry(id).or_insert_with(|| Island {
                id,
                bodies: Vec::new(),
                constraints: Vec::new(),
            });
            island.constraints.push(key);
            for member in [key.0, key.1] {
                if self.bodies[member].inv_mass != 0.0 {
                    island.bodies.push(member);
                }
            }
        }

        let mut islands: Vec<Island> = by_id.into_values().collect();
        for island in &mut islands {
            island.bodies.sort_unstable();
            island.bodies.dedup();
        }
        islands
    }

    /// Union-find over the touched pairs, writing each body's island id to
    /// `labels[dense]`. A body in no touched pair is its own singleton.
    ///
    /// A pair with a static body merges nothing: that body absorbs no impulse,
    /// so the two sides of it are independent and joining them would hand a
    /// parallel solver one island where it has two. The label is a post-pass
    /// minimum over each component rather than whichever root the unions
    /// happened to leave, which is what makes an island's identity a function
    /// of the handles in it and not of the order the pairs arrived in.
    fn fill_islands(
        bodies: &BodyArena<S>,
        touched: impl Iterator<Item = PairKey>,
        parent: &mut Vec<u32>,
        labels: &mut Vec<BodyId>,
    ) {
        let n = bodies.len();
        parent.clear();
        parent.extend(0..n as u32);
        labels.clear();
        labels.extend((0..n).map(|dense| bodies.id_at(dense)));

        for key in touched {
            let (i, j) = (
                bodies.dense_index(key.0).expect(STALE_MANIFOLD_BODY),
                bodies.dense_index(key.1).expect(STALE_MANIFOLD_BODY),
            );
            if bodies[i].inv_mass == 0.0 || bodies[j].inv_mass == 0.0 {
                continue;
            }
            let (a, b) = (find_root(parent, i), find_root(parent, j));
            if a != b {
                // Which root survives only shapes the forest; the component and
                // the label below are the same either way.
                parent[a.max(b)] = a.min(b) as u32;
            }
        }

        for dense in 0..n {
            let root = find_root(parent, dense);
            labels[root] = labels[root].min(bodies.id_at(dense));
        }
        for dense in 0..n {
            let label = labels[find_root(parent, dense)];
            labels[dense] = label;
        }
    }

    /// Storage positions of a key's two bodies, in the key's own order. Both
    /// resolve, but as a property of the four callers rather than of the
    /// manifold map: `update_manifolds` passes keys its own broadphase minted
    /// from the live arena this step, and the three solve phases pass
    /// `constraint_keys`, filled after that phase evicted every key it did not
    /// touch.
    ///
    /// The map itself carries no such guarantee. `despawn_body` prunes it, but
    /// [`BodyArena::despawn`] is reachable on `bodies` directly and does not,
    /// so between one of those and the next `update_manifolds` the map can name
    /// a dead body. Nothing in that window reaches here; [`Self::islands`],
    /// which walks the map, is where it surfaces as a panic.
    fn dense_pair(&self, key: PairKey) -> (usize, usize) {
        (
            self.bodies.dense_index(key.0).expect(STALE_MANIFOLD_BODY),
            self.bodies.dense_index(key.1).expect(STALE_MANIFOLD_BODY),
        )
    }
}

/// Representative of `dense`'s component, halving the path it walks on the way
/// (Tarjan and van Leeuwen 1984, JACM 31(2), sec. 2: path halving carries the
/// same amortized bound as full compression in one pass).
fn find_root(parent: &mut [u32], mut dense: usize) -> usize {
    while parent[dense] as usize != dense {
        parent[dense] = parent[parent[dense] as usize];
        dense = parent[dense] as usize;
    }
    dense
}

/// The island a constraint is solved in: the island of its dynamic body. Every
/// constraint has one, since the broadphase never emits a pair of two statics.
fn constraint_island<S: PhysicsSpace>(
    bodies: &BodyArena<S>,
    labels: &[BodyId],
    key: PairKey,
) -> BodyId {
    let (i, j) = (
        bodies.dense_index(key.0).expect(STALE_MANIFOLD_BODY),
        bodies.dense_index(key.1).expect(STALE_MANIFOLD_BODY),
    );
    debug_assert!(
        bodies[i].inv_mass != 0.0 || bodies[j].inv_mass != 0.0,
        "a contact between two static bodies has no island to solve in",
    );
    if bodies[i].inv_mass != 0.0 {
        labels[i]
    } else {
        labels[j]
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
    use std::collections::BTreeSet;

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
    use loam_math::{Bivector3, EuclideanR3, Space};

    /// Arbitrary but fixed, so a failure is reproducible from its message.
    const PERMUTATION_SEEDS: [u64; 4] = [1, 0x9e37_79b9_7f4a_7c15, 0xdead_beef_cafe_f00d, 424_242];

    /// Counts what the thread running a probe asks of the allocator. The test
    /// runner gives each test its own thread, so a probe never sees a
    /// concurrent test's allocations; the counter is thread-local rather than
    /// global for exactly that reason. Const-initialised so reading it inside
    /// `alloc` cannot itself allocate.
    mod alloc_probe {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;

        thread_local! {
            static BYTES: Cell<usize> = const { Cell::new(0) };
        }

        pub struct Counting;

        unsafe impl GlobalAlloc for Counting {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                let _ = BYTES.try_with(|bytes| bytes.set(bytes.get() + layout.size()));
                System.alloc(layout)
            }

            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                System.dealloc(ptr, layout)
            }

            unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
                let _ = BYTES.try_with(|bytes| bytes.set(bytes.get() + new_size));
                System.realloc(ptr, layout, new_size)
            }
        }

        pub fn bytes_allocated_by(body: impl FnOnce()) -> usize {
            let before = BYTES.with(Cell::get);
            body();
            BYTES.with(Cell::get) - before
        }
    }

    #[global_allocator]
    static COUNTING_ALLOCATOR: alloc_probe::Counting = alloc_probe::Counting;

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

    /// `dense_pair` states its precondition as a property of its callers, not
    /// of `manifolds`, because [`BodyArena::despawn`] is reachable on `bodies`
    /// and prunes nothing. This pins the consequence the doc names: the map
    /// keeps naming the dead body, `islands` panics on it, and the next step's
    /// eviction is what clears it. `despawn_body` is the control, and the pair
    /// of them is what would fail if either half of the doc drifted.
    #[test]
    fn bare_arena_despawn_strands_a_manifold_key_until_the_next_step() {
        let dt = 1.0 / 240.0;
        let settle_steps = 400;
        let (mut world, floor, spheres) = settled_islands(dt, settle_steps);
        let doomed = spheres[1];
        assert!(
            world.manifolds.contains_key(&(floor, doomed)),
            "fixture has no manifold on the doomed body, so nothing is stranded"
        );

        assert!(world.bodies.despawn(doomed).is_some());
        assert!(
            world.manifolds.contains_key(&(floor, doomed)),
            "the arena despawn pruned manifolds, so despawn_body is no longer \
             the only removal that keeps the world consistent"
        );
        let resolved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| world.islands()));
        assert!(
            resolved.is_err(),
            "islands resolved a key naming a despawned body"
        );

        world.step(dt);
        assert!(
            !world
                .manifolds
                .keys()
                .any(|&(a, b)| a == doomed || b == doomed),
            "the step did not evict the stranded key"
        );
        assert_eq!(
            world.islands().len(),
            ISLAND_X.len() - 1,
            "the eviction did not close the panic window"
        );

        // The control: the same removal through the world drops the manifold
        // with the body, so no window exists in the first place.
        let (mut control, control_floor, control_spheres) = settled_islands(dt, settle_steps);
        assert!(control.despawn_body(control_spheres[1]));
        assert!(!control
            .manifolds
            .contains_key(&(control_floor, control_spheres[1])));
        assert_eq!(control.islands().len(), ISLAND_X.len() - 1);
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

    /// xorshift64 (Marsaglia 2003, "Xorshift RNGs", J. Stat. Soft. 8(14), the
    /// 13/7/17 triple) so a randomized scene replays from the seed in the
    /// failure message.
    struct Xorshift(u64);

    impl Xorshift {
        fn new(seed: u64) -> Self {
            // Absorbing at zero.
            Self(seed | 1)
        }

        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        /// Uniform in `[lo, hi)` off the top 24 bits, which is the whole f32
        /// significand.
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            let unit = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
            lo + (hi - lo) * unit
        }
    }

    /// Spheres and boxes at seeded positions over a static floor, a few of them
    /// pinned static so the two-static skip is exercised, then thinned by
    /// despawns so storage order and handle order disagree. The despawns are
    /// what make the fixture adversarial: a broadphase keyed on storage
    /// position agrees with one keyed on handles until the arena compacts.
    fn random_scene(seed: u64, count: usize, spread: f32) -> World<EuclideanR3> {
        let mut rng = Xorshift::new(seed);
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        world.push_field(Box::new(Gravity::new(Vec3::new(0.0, GRAVITY_Y, 0.0))));
        world.push_body(halfspace_body_r3(Vec3::Y, 0.0));

        let mut spawned = Vec::with_capacity(count);
        for _ in 0..count {
            let position = Vec3::new(
                rng.range(-spread, spread),
                rng.range(0.5, spread + 0.5),
                rng.range(-spread, spread),
            );
            let id = if rng.next_u64() & 1 == 0 {
                world.push_body(sphere_body_r3(
                    position,
                    Vec3::ZERO,
                    rng.range(0.2, 0.8),
                    1.0,
                ))
            } else {
                world.push_body(box_body(
                    position,
                    Vec3::ZERO,
                    Vec3::splat(rng.range(0.2, 0.6)),
                    1.0,
                ))
            };
            if rng.next_u64().is_multiple_of(8) {
                world.bodies[id].mass = 0.0;
                world.bodies[id].inv_mass = 0.0;
            }
            world.bodies[id].restitution = 0.0;
            spawned.push(id);
        }
        for doomed in spawned.iter().step_by(5) {
            assert!(world.despawn_body(*doomed));
        }
        world
    }

    /// The O(n²) definition of the candidate set: every pair that is not two
    /// static bodies and whose bounding balls overlap. The sweep is only an
    /// acceleration structure over this, so it owes exact agreement rather
    /// than a superset.
    fn all_pairs_reference(world: &World<EuclideanR3>) -> Vec<PairKey> {
        let mut pairs = Vec::new();
        let n = world.bodies.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let (a, b) = (&world.bodies[i], &world.bodies[j]);
                if a.inv_mass == 0.0 && b.inv_mass == 0.0 {
                    continue;
                }
                let reach = bounding_radius(&a.collider) + bounding_radius(&b.collider);
                if world.space.distance(a.position, b.position) <= reach {
                    pairs.push(canonical_pair(world.bodies.id_at(i), world.bodies.id_at(j)));
                }
            }
        }
        pairs.sort_unstable();
        pairs
    }

    fn dynamic_body_count(world: &World<EuclideanR3>) -> usize {
        world.bodies.iter().filter(|b| b.inv_mass != 0.0).count()
    }

    /// Sweep sizes chosen so the sweep is exercised on a crowded scene, a
    /// sparse one, and one large enough for the active list to turn over many
    /// times.
    const RANDOM_SCENE_SHAPES: [(usize, f32); 3] = [(12, 2.0), (40, 6.0), (80, 3.0)];

    /// The sweep's contract: it emits the candidate set the O(n²) reference
    /// defines, exactly, on every step of every seeded scene. Set equality and
    /// not containment in either direction, because a sweep that emits a
    /// superset has stopped pruning and one that emits a subset has dropped a
    /// contact.
    #[test]
    fn sweep_broadphase_emits_exactly_the_all_pairs_candidate_set_determinism() {
        let mut ever_beyond_the_floor = false;
        for seed in PERMUTATION_SEEDS {
            for (count, spread) in RANDOM_SCENE_SHAPES {
                let mut world = random_scene(seed, count, spread);
                // Stepped so the comparison covers configurations gravity and
                // the solver produce, not only the one the seed laid out.
                for step in 0..8 {
                    let expected = all_pairs_reference(&world);
                    assert_eq!(
                        world.broadphase(),
                        expected,
                        "seed {seed}, {count} bodies, spread {spread}, step {step}"
                    );
                    ever_beyond_the_floor |= expected.len() > dynamic_body_count(&world);
                    world.step(1.0 / 240.0);
                }
            }
        }
        // The floor is unbounded and pairs with every dynamic body, so a run
        // that only ever emitted those pairs would have compared two trivial
        // sets.
        assert!(
            ever_beyond_the_floor,
            "no scene ever produced a candidate pair between two finite colliders"
        );
    }

    /// The physics-safety half, stated against the narrowphase rather than
    /// against the reference: a culled pair must be one the narrowphase would
    /// have rejected anyway. This is what lets the golden hashes survive a
    /// broadphase that emits fewer pairs than the old all-pairs loop.
    #[test]
    fn broadphase_culls_only_pairs_the_narrowphase_would_reject() {
        for seed in PERMUTATION_SEEDS {
            let mut world = random_scene(seed, 40, 3.0);
            for step in 0..8 {
                let emitted = world.broadphase();
                let n = world.bodies.len();
                let mut culled = 0usize;
                for i in 0..n {
                    for j in (i + 1)..n {
                        // Two static bodies are excluded by the candidate
                        // definition, not by the cull: neither can move, so a
                        // contact between them has nothing to solve.
                        if world.bodies[i].inv_mass == 0.0 && world.bodies[j].inv_mass == 0.0 {
                            continue;
                        }
                        let key = canonical_pair(world.bodies.id_at(i), world.bodies.id_at(j));
                        if emitted.binary_search(&key).is_ok() {
                            continue;
                        }
                        culled += 1;
                        let contact = world.narrowphase.test(
                            &world.bodies[i],
                            &world.bodies[j],
                            &world.space,
                        );
                        assert!(
                            contact.is_none(),
                            "seed {seed} step {step}: the sweep culled {key:?}, which the \
                             narrowphase reports in contact"
                        );
                    }
                }
                assert!(
                    culled > 0,
                    "seed {seed} step {step}: nothing was culled, so this pass proved nothing"
                );
                world.step(1.0 / 240.0);
            }
        }
    }

    /// Emission order is a function of the handles alone. Ascending and
    /// strictly so, which also pins that no pair is emitted twice; a sweep
    /// whose active list double-counted an interval would fail here rather
    /// than by quietly solving a contact twice.
    #[test]
    fn broadphase_emits_strictly_ascending_keys_under_disagreeing_storage_order_determinism() {
        for seed in PERMUTATION_SEEDS {
            let world = random_scene(seed, 40, 3.0);
            let disagrees = (1..world.bodies.len())
                .any(|dense| world.bodies.id_at(dense) < world.bodies.id_at(dense - 1));
            assert!(
                disagrees,
                "seed {seed}: storage order still agrees with handle order, so this \
                 scene cannot tell the two apart"
            );

            let pairs = world.broadphase();
            assert!(pairs.len() > 1, "seed {seed}: too few pairs to be ordered");
            assert!(
                pairs.windows(2).all(|w| w[0] < w[1]),
                "seed {seed}: emission order is not strictly ascending in BodyId"
            );
        }
    }

    /// The scale claim. At 200 bodies the all-pairs loop is ~20k pairs; the
    /// unbounded floor alone contributes one per dynamic body, so the floor is
    /// the sweep's own lower bound and the assertion is that it lands near it.
    #[test]
    fn broadphase_prunes_the_quadratic_pair_set_at_scale() {
        let world = random_scene(PERMUTATION_SEEDS[1], 200, 20.0);
        let n = world.bodies.len();
        assert!(
            n >= 100,
            "the scale case needs at least 100 bodies, got {n}"
        );
        let all_pairs = n * (n - 1) / 2;
        let emitted = world.broadphase().len();
        assert!(
            emitted * 10 < all_pairs,
            "the sweep emitted {emitted} of {all_pairs} pairs, which is no better than \
             a constant-factor cull"
        );
    }

    /// The buffers the step hands the sweep are reused, so a steady body count
    /// must not reach the allocator at all. Measured on the sweep rather than
    /// on the whole step because the phases downstream of it still allocate.
    #[test]
    fn broadphase_fill_allocates_nothing_after_the_first_pass() {
        let world = random_scene(PERMUTATION_SEEDS[0], 120, 8.0);
        let mut intervals = Vec::new();
        let mut active = Vec::new();
        let mut pairs = Vec::new();

        // The first passes grow the three buffers to their steady size.
        for _ in 0..2 {
            World::fill_broadphase(
                &world.bodies,
                &world.space,
                &mut intervals,
                &mut active,
                &mut pairs,
            );
        }
        assert!(!pairs.is_empty(), "the fixture produced no pairs to emit");

        let bytes = alloc_probe::bytes_allocated_by(|| {
            for _ in 0..16 {
                World::fill_broadphase(
                    &world.bodies,
                    &world.space,
                    &mut intervals,
                    &mut active,
                    &mut pairs,
                );
            }
        });
        assert_eq!(
            bytes, 0,
            "16 sweeps over a steady body set asked the allocator for {bytes} bytes"
        );
    }

    /// The bound the cull rests on: no posed vertex of a collider can lie
    /// further from the body position than its bounding radius, at any
    /// orientation. A bound that under-reported would cull contacting pairs.
    #[test]
    fn bounding_radius_contains_every_posed_vertex_of_its_collider() {
        let half_extents = Vec3::new(0.5, 1.25, 0.25);
        let vertices = crate::euclidean_r3::box_vertices(half_extents);
        let radius = bounding_radius(&Collider::ConvexPolytope3D {
            vertices: vertices.clone(),
        });
        assert_eq!(radius, half_extents.length());

        let rotation = glam::Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalize(), 0.7);
        for v in &vertices {
            let posed = rotation * *v;
            assert!(
                posed.length() <= radius + 1e-6,
                "posed vertex {posed} escaped the bounding radius {radius}"
            );
        }

        assert_eq!(bounding_radius(&Collider::sphere_at_origin(0.75)), 0.75);
        assert_eq!(
            bounding_radius(&Collider::HalfSpace {
                normal: Vec3::Y,
                offset: 0.0
            }),
            f32::INFINITY,
            "a half-space is unbounded and must never be culled"
        );
    }

    /// Long enough for every column to fall, land, and rest in contact.
    const ISLAND_SETTLE_STEPS: usize = 400;
    /// Columns far enough apart in x that no column can reach its neighbour.
    const ISLAND_COLUMN_PITCH: f32 = 4.0;
    const ISLAND_COLUMNS: usize = 6;
    const ISLAND_COLUMN_HEIGHT: usize = 3;
    /// The column whose middle sphere is static. Not one of the two the
    /// despawns below take from, so the wedge survives the thinning.
    const ISLAND_PINNED_COLUMN: usize = 2;

    /// Seeded columns of spheres over one shared static floor, settled, then
    /// thinned so storage order and handle order disagree inside the surviving
    /// islands.
    ///
    /// Columns rather than the scattered `random_scene` layout because a stack
    /// is what holds contact: bodies dropped side by side settle into gaps and
    /// leave every island a singleton, which would make every assertion about a
    /// union vacuous. The despawns take low-slot bodies, so `swap_remove` moves
    /// the last-spawned bodies, which carry the highest handles, into low
    /// storage positions.
    fn settled_columns(seed: u64) -> World<EuclideanR3> {
        const GAP: f32 = 0.05;
        /// Overlap that puts the pinned sphere inside both neighbours' reach
        /// once the column has settled around it.
        const PINNED_TOUCH: f32 = 0.01;

        let mut rng = Xorshift::new(seed);
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        world.push_field(Box::new(Gravity::new(Vec3::new(0.0, GRAVITY_Y, 0.0))));
        world.push_body(halfspace_body_r3(Vec3::Y, 0.0));

        let mut columns: Vec<Vec<BodyId>> = Vec::with_capacity(ISLAND_COLUMNS);
        for column in 0..ISLAND_COLUMNS {
            let x = column as f32 * ISLAND_COLUMN_PITCH + rng.range(-0.5, 0.5);
            // One radius per column: a stack of unequal spheres rolls off
            // itself and the island stops being a column.
            let radius = rng.range(0.3, 0.6);
            let mut ids = Vec::with_capacity(ISLAND_COLUMN_HEIGHT);
            for level in 0..ISLAND_COLUMN_HEIGHT {
                // One column carries a static sphere placed to touch both its
                // neighbours: the shared floor already covers the rule that a
                // static body merges no islands, but it covers it where every
                // candidate rule agrees. Wedged mid-column, a rule that merged
                // through statics would visibly join the two dynamic spheres.
                let pinned = column == ISLAND_PINNED_COLUMN && level == 1;
                let y = if pinned {
                    3.0 * radius - PINNED_TOUCH
                } else {
                    radius + GAP + level as f32 * (2.0 * radius + GAP)
                };
                let id = world.push_body(sphere_body_r3(
                    Vec3::new(x, y, 0.0),
                    Vec3::ZERO,
                    radius,
                    1.0,
                ));
                world.bodies[id].restitution = 0.0;
                if pinned {
                    world.bodies[id].mass = 0.0;
                    world.bodies[id].inv_mass = 0.0;
                }
                ids.push(id);
            }
            columns.push(ids);
        }

        for column in columns.iter().take(2) {
            assert!(world.despawn_body(column[0]));
        }
        for _ in 0..ISLAND_SETTLE_STEPS {
            world.step(1.0 / 240.0);
        }
        world
    }

    fn labels_for(world: &World<EuclideanR3>, keys: &[PairKey]) -> Vec<BodyId> {
        let mut parent = Vec::new();
        let mut labels = Vec::new();
        World::fill_islands(
            &world.bodies,
            keys.iter().copied(),
            &mut parent,
            &mut labels,
        );
        labels
    }

    /// Bodies for [`SYNTHETIC_EDGES`], with two of them static and two
    /// despawned so handle order and storage order disagree. Never stepped:
    /// the partition reads handles and `inv_mass` only, so positions are free
    /// and the graph can be shaped rather than waited for.
    fn synthetic_island_bodies() -> (World<EuclideanR3>, Vec<BodyId>) {
        const SPAWNS: usize = 14;
        const DESPAWNS: usize = 2;

        let mut world = World::new(EuclideanR3);
        let spawned: Vec<BodyId> = (0..SPAWNS)
            .map(|i| world.push_body(island_sphere(i as f32 * ISLAND_COLUMN_PITCH)))
            .collect();
        let survivors = spawned[DESPAWNS..].to_vec();
        for &position in &SYNTHETIC_STATICS {
            let id = survivors[position];
            world.bodies[id].mass = 0.0;
            world.bodies[id].inv_mass = 0.0;
        }
        // Taken from the low handles, so `swap_remove` moves the two highest
        // handles into the two lowest storage positions.
        for &doomed in &spawned[..DESPAWNS] {
            assert!(world.despawn_body(doomed));
        }
        (world, survivors)
    }

    /// Survivor positions that are static, by position in the survivor list.
    const SYNTHETIC_STATICS: [usize; 2] = [1, 7];

    /// Edges over `synthetic_island_bodies`' survivors, by position in that
    /// list. A hub with a cycle hanging off it, a four-body chain, and four
    /// edges that meet only at a static body.
    const SYNTHETIC_EDGES: [(usize, usize); 11] = [
        (0, 2),
        (0, 4),
        (0, 5),
        (2, 5),
        (6, 8),
        (8, 10),
        (10, 11),
        (1, 3),
        (1, 6),
        (7, 9),
        (7, 11),
    ];

    /// The components [`SYNTHETIC_EDGES`] defines. Everything unlisted is a
    /// singleton, including both statics and the two bodies that meet only
    /// through one.
    const SYNTHETIC_COMPONENTS: [&[usize]; 2] = [&[0, 2, 4, 5], &[6, 8, 10, 11]];

    /// The union-find's input is a set, so its output owes nothing to the order
    /// that set is presented in, and its labels owe nothing to storage. Both on
    /// a shaped graph: the physics fixtures build paths of at most three
    /// bodies, and on those a label taken from whichever root the unions left
    /// agrees with the component minimum by coincidence.
    #[test]
    fn island_labels_are_the_component_minimum_whatever_order_pairs_arrive_in_determinism() {
        let (world, ids) = synthetic_island_bodies();
        let canonical_keys: Vec<PairKey> = SYNTHETIC_EDGES
            .iter()
            .map(|&(a, b)| canonical_pair(ids[a], ids[b]))
            .collect();
        let canonical = labels_for(&world, &canonical_keys);

        for members in SYNTHETIC_COMPONENTS {
            let expected = members
                .iter()
                .map(|&i| ids[i])
                .min()
                .expect("a component with no members");
            for &member in members {
                let dense = world.bodies.dense_index(ids[member]).unwrap();
                assert_eq!(
                    canonical[dense], expected,
                    "body {member} is labelled {:?}, not its component's lowest handle",
                    canonical[dense]
                );
            }
        }
        let grouped: BTreeSet<usize> = SYNTHETIC_COMPONENTS
            .iter()
            .flat_map(|m| *m)
            .copied()
            .collect();
        for (position, &id) in ids.iter().enumerate() {
            if grouped.contains(&position) {
                continue;
            }
            let dense = world.bodies.dense_index(id).unwrap();
            assert_eq!(
                canonical[dense], id,
                "body {position} joined an island it has no edge into"
            );
        }

        for order in order_variants(SchedulePhase::Constraint) {
            let mut keys = canonical_keys.clone();
            order.apply(SchedulePhase::Constraint, &mut keys);
            assert_ne!(keys, canonical_keys, "{order:?} is the identity");
            assert_eq!(
                labels_for(&world, &keys),
                canonical,
                "{order:?} produced a different island assignment"
            );
        }
    }

    /// The label is the lowest handle in the component, which is what the
    /// invariance harness needs to name an island. Storage position is the
    /// tempting alternative and is wrong: a despawn compacts the arena and
    /// would rename an island that did not change.
    #[test]
    fn island_ids_are_the_lowest_body_id_not_the_lowest_storage_position_determinism() {
        let mut orders_disagreed = 0usize;
        for seed in PERMUTATION_SEEDS {
            let mut world = settled_columns(seed);
            for step in 0..8 {
                world.step(1.0 / 240.0);
                let islands = world.islands();
                for island in &islands {
                    let lowest_handle = island.bodies.iter().copied().min();
                    assert_eq!(
                        Some(island.id),
                        lowest_handle,
                        "seed {seed} step {step}: island {:?} is not named by its \
                         lowest handle",
                        island.id
                    );
                    let lowest_stored =
                        island.bodies.iter().copied().min_by_key(|&id| {
                            world.bodies.dense_index(id).expect(STALE_MANIFOLD_BODY)
                        });
                    if lowest_stored != lowest_handle {
                        orders_disagreed += 1;
                    }
                }
                assert!(
                    islands.windows(2).all(|w| w[0].id < w[1].id),
                    "seed {seed} step {step}: islands are not strictly ascending in id"
                );
            }
        }
        assert!(
            orders_disagreed > 0,
            "no island ever held a body whose handle order disagreed with its \
             storage order, so the two labelling rules were never told apart"
        );
    }

    /// A static body absorbs no impulse, so the two bodies resting on either
    /// side of one never reach each other and must not share an island. The
    /// shared floor is the case that matters in practice; the fixture's wedged
    /// static sphere is the same rule where a merge would be least visible.
    #[test]
    fn a_static_body_joins_no_island_and_merges_none_determinism() {
        for seed in PERMUTATION_SEEDS {
            let world = settled_columns(seed);
            let pinned = world
                .bodies
                .iter()
                .position(|body| {
                    body.inv_mass == 0.0 && matches!(body.collider, Collider::Sphere { .. })
                })
                .map(|dense| world.bodies.id_at(dense))
                .expect("the fixture lost its static sphere");

            let neighbours: Vec<BodyId> = world
                .manifolds
                .keys()
                .filter_map(|&(a, b)| match (a == pinned, b == pinned) {
                    (true, false) => Some(b),
                    (false, true) => Some(a),
                    _ => None,
                })
                .collect();
            assert_eq!(
                neighbours.len(),
                2,
                "seed {seed}: the static sphere touches {} bodies, so it is \
                 not wedged between two",
                neighbours.len()
            );

            let islands = world.islands();
            let island_of = |id: BodyId| {
                islands
                    .iter()
                    .find(|island| island.bodies.contains(&id))
                    .map(|island| island.id)
            };
            assert!(
                island_of(pinned).is_none(),
                "seed {seed}: a static body joined an island"
            );
            assert_ne!(
                island_of(neighbours[0]),
                island_of(neighbours[1]),
                "seed {seed}: two bodies that meet only through a static one \
                 were merged into one island"
            );
        }
    }

    /// Connected components by flood fill over an adjacency list built from the
    /// manifold keys: the definition the union-find is an acceleration of, so
    /// it owes exact agreement rather than self-consistency.
    ///
    /// Static bodies are absent from the adjacency: they carry no island and
    /// join none, which is the rule that keeps three groups on one floor apart.
    fn flood_fill_islands(world: &World<EuclideanR3>) -> Vec<Island> {
        let dynamic = |id: BodyId| world.bodies[id].inv_mass != 0.0;
        let mut adjacency: BTreeMap<BodyId, Vec<BodyId>> = BTreeMap::new();
        for &(a, b) in world.manifolds.keys() {
            for id in [a, b].into_iter().filter(|&id| dynamic(id)) {
                adjacency.entry(id).or_default();
            }
            if dynamic(a) && dynamic(b) {
                adjacency.entry(a).or_default().push(b);
                adjacency.entry(b).or_default().push(a);
            }
        }

        let mut seen: BTreeSet<BodyId> = BTreeSet::new();
        let mut islands = Vec::new();
        for &seed in adjacency.keys() {
            if !seen.insert(seed) {
                continue;
            }
            let mut bodies = vec![seed];
            let mut frontier = vec![seed];
            while let Some(body) = frontier.pop() {
                for &next in &adjacency[&body] {
                    if seen.insert(next) {
                        bodies.push(next);
                        frontier.push(next);
                    }
                }
            }
            bodies.sort_unstable();
            let constraints = world
                .manifolds
                .keys()
                .copied()
                .filter(|&(a, b)| {
                    bodies.binary_search(&a).is_ok() || bodies.binary_search(&b).is_ok()
                })
                .collect();
            islands.push(Island {
                id: bodies[0],
                bodies,
                constraints,
            });
        }
        islands.sort_unstable_by_key(|island| island.id);
        islands
    }

    /// The partition itself, against the independent oracle. Equality of the
    /// whole island list also pins that no body lands in two islands and that
    /// every touched pair lands in exactly one.
    #[test]
    fn islands_match_a_flood_fill_of_the_contact_graph_determinism() {
        let mut ever_multi_body = false;
        for seed in PERMUTATION_SEEDS {
            let mut world = settled_columns(seed);
            for step in 0..8 {
                world.step(1.0 / 240.0);
                let islands = world.islands();
                ever_multi_body |= islands.iter().any(|island| island.bodies.len() > 1);
                assert_eq!(
                    islands,
                    flood_fill_islands(&world),
                    "seed {seed} step {step}: union-find disagreed with the flood fill"
                );
            }
        }
        assert!(
            ever_multi_body,
            "no island ever held two bodies, so the comparison never covered a union"
        );
    }

    /// Criterion for a world whose contact graph is connected: grouping has
    /// nothing to move, so the constraint order the solver sees is the one it
    /// saw before islands existed, and the solve is unchanged bit for bit
    /// rather than merely close.
    #[test]
    fn a_single_island_solves_in_the_global_ascending_key_order_determinism() {
        let world = settled_sphere_stack(1.0 / 240.0, 200);
        let islands = world.islands();
        assert_eq!(
            islands.len(),
            1,
            "the stack is not one island, so this fixture cannot state the \
             single-island case"
        );

        let ascending: Vec<PairKey> = world.manifolds.keys().copied().collect();
        assert!(ascending.len() > 1, "too few constraints to be ordered");
        assert_eq!(
            world.constraint_keys, ascending,
            "grouping moved a constraint in a world with a single island"
        );
        assert_eq!(islands[0].constraints, ascending);
        assert_eq!(
            islands[0].bodies.len(),
            3,
            "the island should hold the three spheres and not the static floor"
        );
    }

    /// The grouped buffer is the islands laid end to end, and on this fixture
    /// that is genuinely a different sequence from ascending key order: the
    /// chain's contacts and the pair's interleave when sorted by key alone.
    /// `multi_island_scenario_matches_golden_determinism_hash` still holds
    /// against a constant recorded before the grouping existed, which is what
    /// makes the reordering bit-neutral rather than merely untested.
    #[test]
    fn constraint_buffer_runs_island_by_island_determinism() {
        let mut world = multi_island_world(Schedule::default());
        for _ in 0..MULTI_ISLAND_STEPS {
            world.step(MULTI_ISLAND_DT);
        }

        let islands = world.islands();
        assert_eq!(
            islands.len(),
            3,
            "the groups share only the static floor, so they are three islands"
        );
        let grouped: Vec<PairKey> = islands
            .iter()
            .flat_map(|island| island.constraints.iter().copied())
            .collect();
        assert_eq!(
            world.constraint_keys, grouped,
            "the solved buffer is not the islands in order"
        );

        let ascending: Vec<PairKey> = world.manifolds.keys().copied().collect();
        assert_ne!(
            grouped, ascending,
            "the fixture's islands happen to be contiguous in ascending key \
             order, so it cannot show that grouping reorders anything"
        );
    }

    /// The step's island work runs out of the buffers the world retains, on
    /// the same terms as the sweep above. Measured on `collect_constraints` so
    /// the in-place sort is inside the probe, not only the union-find.
    #[test]
    fn island_grouping_allocates_nothing_after_the_first_pass() {
        let mut world = settled_columns(PERMUTATION_SEEDS[0]);
        for _ in 0..2 {
            world.collect_constraints();
        }
        assert!(world.constraint_keys.len() > 1);

        let bytes = alloc_probe::bytes_allocated_by(|| {
            for _ in 0..16 {
                world.collect_constraints();
            }
        });
        assert_eq!(
            bytes, 0,
            "16 island passes over a steady contact set asked the allocator for \
             {bytes} bytes"
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
