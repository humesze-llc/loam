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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::euclidean_r3::{halfspace_body_r3, register_default_narrowphase, sphere_body_r3};
    use crate::field::Gravity;
    use glam::Vec3;
    use loam_math::EuclideanR3;

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
