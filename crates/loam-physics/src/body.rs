//! [`RigidBody<S>`], the physical object a [`crate::World`] simulates, and
//! [`BodyArena<S>`], the generational storage that gives it an identity
//! independent of where it currently sits in memory.

use std::ops::{Add, Deref, Index, IndexMut, Mul};

use loam_math::Bivector;

use crate::collider::Collider;
use crate::integrator::PhysicsSpace;

/// A rigid body in some [`PhysicsSpace`]. Public-fields struct so the solver and user code can
/// read and write components directly, this crate doesn't hide state, it just provides the rules
/// for advancing it.
///
/// `inv_mass == 0.0` means a static body: gravity and impulses have no effect on its velocity,
/// and [`crate::integrate_body`] skips it.
pub struct RigidBody<S: PhysicsSpace> {
    pub position: S::Point,
    pub velocity: S::Vector,
    pub orientation: S::Iso,
    pub angular_velocity: S::AngVel,

    pub mass: f32,
    pub inv_mass: f32,
    pub inertia: S::Inertia,

    pub collider: Collider,

    /// Coefficient of restitution for elastic bounces. 0 = perfectly inelastic, 1 = perfectly
    /// elastic.
    pub restitution: f32,
}

impl<S: PhysicsSpace> RigidBody<S> {
    /// Build a dynamic body at `position` with the given mass and collider. `space` is passed so
    /// the caller can source an identity isometry without naming the space's [`crate::Collider`]
    /// types directly.
    pub fn new(
        position: S::Point,
        velocity: S::Vector,
        collider: Collider,
        mass: f32,
        inertia: S::Inertia,
        space: &S,
    ) -> Self {
        // Half-spaces are infinite planes; a finite mass with infinite extent breaks the
        // integrator's assumptions (no centre of mass, no bounded inertia). Static-only is the
        // only sensible mode; catch the misuse in debug builds before it produces silently wrong
        // physics in release.
        debug_assert!(
            !matches!(
                collider,
                Collider::HalfSpace { .. } | Collider::HalfSpace4D { .. }
            ) || mass <= 0.0,
            "half-space colliders must be static (mass <= 0); got mass = {mass}"
        );
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        Self {
            position,
            velocity,
            orientation: space.iso_identity(),
            angular_velocity: <S::AngVel as Bivector>::zero(),
            mass,
            inv_mass,
            inertia,
            collider,
            restitution: 0.2,
        }
    }

    /// Static body: infinite mass, zero velocity, immovable.
    pub fn fixed(position: S::Point, collider: Collider, inertia: S::Inertia, space: &S) -> Self
    where
        S::Vector: Default,
    {
        Self {
            position,
            velocity: S::Vector::default(),
            orientation: space.iso_identity(),
            angular_velocity: <S::AngVel as Bivector>::zero(),
            mass: 0.0,
            inv_mass: 0.0,
            inertia,
            collider,
            restitution: 0.2,
        }
    }

    /// Apply an impulse whose line of action passes through the centre of
    /// mass: `v += J/m`, no angular response. Static bodies ignore it.
    ///
    /// Impulse-momentum relation, Baraff 1997, "Physically Based Modeling:
    /// Rigid Body Simulation", colliding-contact section.
    pub fn apply_impulse(&mut self, impulse: S::Vector)
    where
        S::Vector: Add<Output = S::Vector> + Mul<f32, Output = S::Vector>,
    {
        if self.inv_mass == 0.0 {
            return;
        }
        self.velocity = self.velocity + impulse * self.inv_mass;
    }

    /// Apply an impulse at world point `point`: `v += J/m` and
    /// `ω += I⁻¹(r ∧ J)` for `r` the offset from the centre of mass to
    /// `point`. Static bodies ignore it.
    ///
    /// Same reference as [`Self::apply_impulse`]; the angular half is the
    /// wedge form of `Δω = I⁻¹(r × J)`.
    pub fn apply_impulse_at_point(&mut self, space: &S, impulse: S::Vector, point: S::Point)
    where
        S::Vector: Add<Output = S::Vector> + Mul<f32, Output = S::Vector>,
    {
        if self.inv_mass == 0.0 {
            return;
        }
        self.velocity = self.velocity + impulse * self.inv_mass;
        // The lever arm is a tangent vector at the body, so it is `log`, not a
        // chart-coordinate subtraction; the two agree in the flat spaces that
        // implement PhysicsSpace today.
        let lever = space.log(self.position, point);
        let torque = space.wedge(lever, impulse);
        self.angular_velocity =
            self.angular_velocity + space.apply_inv_inertia(self.inertia, torque);
    }
}

/// Stable handle to a body in a [`BodyArena`]. `slot` names a table entry;
/// `generation` counts how often that entry has been reused, so a handle to a
/// despawned body fails to resolve rather than aliasing whatever body later
/// takes its place.
///
/// Ordering is lexicographic in `(slot, generation)`, which is what lets a
/// pair of handles serve as a canonical contact-manifold key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BodyId {
    slot: u32,
    generation: u32,
}

impl BodyId {
    pub fn slot(self) -> u32 {
        self.slot
    }

    pub fn generation(self) -> u32 {
        self.generation
    }

    /// Handles are minted by [`BodyArena::spawn`]; a test exercising a
    /// consumer of a handle rather than the arena itself has to forge one.
    #[cfg(test)]
    pub(crate) fn forge(slot: u32, generation: u32) -> Self {
        Self { slot, generation }
    }
}

const STALE_HANDLE: &str = "BodyId refers to a despawned body";

/// Slot-table entry. `dense` is `None` while the slot is vacant.
struct Slot {
    generation: u32,
    dense: Option<u32>,
}

/// Bodies in dense storage behind generational handles.
///
/// Storage is contiguous and hole-free so every phase loop walks a slice:
/// despawn swaps the last body down into the vacated position. That move is
/// invisible to a caller holding a [`BodyId`], which is the whole point of the
/// indirection; manifold keys, island membership, and warm-start caches key on
/// the handle and survive it.
///
/// [`Deref`] exposes the shared slice. There is deliberately no [`DerefMut`]:
/// it would hand out every reordering method on `[RigidBody<S>]`, and a
/// permutation the slot table does not follow leaves `slots[ids[d].slot].dense
/// != d`, silently rebinding each manifold's accumulated impulses to the wrong
/// pair. Mutation runs through [`Self::get_mut`], the [`IndexMut`] impls, and
/// [`Self::iter_mut`], none of which can move a body between positions.
///
/// The three `compile_fail` blocks below share the hidden setup of the passing
/// one, so each fails on its visible line and not on its fixture.
///
/// ```
/// # use glam::Vec3;
/// # use loam_physics::euclidean_r3::sphere_body_r3;
/// # use loam_physics::BodyArena;
/// # let mut arena = BodyArena::new();
/// # let first = arena.spawn(sphere_body_r3(Vec3::ZERO, Vec3::ZERO, 0.5, 1.0));
/// # arena.spawn(sphere_body_r3(Vec3::X, Vec3::ZERO, 0.5, 1.0));
/// for body in arena.iter_mut() {
///     body.restitution = 0.0;
/// }
/// # assert_eq!(arena.id_at(0), first);
/// ```
///
/// ```compile_fail
/// # use glam::Vec3;
/// # use loam_physics::euclidean_r3::sphere_body_r3;
/// # use loam_physics::BodyArena;
/// # let mut arena = BodyArena::new();
/// # arena.spawn(sphere_body_r3(Vec3::ZERO, Vec3::ZERO, 0.5, 1.0));
/// # arena.spawn(sphere_body_r3(Vec3::X, Vec3::ZERO, 0.5, 1.0));
/// arena.swap(0, 1);
/// ```
///
/// ```compile_fail
/// # use glam::Vec3;
/// # use loam_physics::euclidean_r3::sphere_body_r3;
/// # use loam_physics::BodyArena;
/// # let mut arena = BodyArena::new();
/// # arena.spawn(sphere_body_r3(Vec3::ZERO, Vec3::ZERO, 0.5, 1.0));
/// # arena.spawn(sphere_body_r3(Vec3::X, Vec3::ZERO, 0.5, 1.0));
/// arena.reverse();
/// ```
///
/// ```compile_fail
/// # use glam::Vec3;
/// # use loam_physics::euclidean_r3::sphere_body_r3;
/// # use loam_physics::BodyArena;
/// # let mut arena = BodyArena::new();
/// # arena.spawn(sphere_body_r3(Vec3::ZERO, Vec3::ZERO, 0.5, 1.0));
/// # arena.spawn(sphere_body_r3(Vec3::X, Vec3::ZERO, 0.5, 1.0));
/// arena.sort_unstable_by_key(|body| body.position.y.to_bits());
/// ```
pub struct BodyArena<S: PhysicsSpace> {
    dense: Vec<RigidBody<S>>,
    /// Dense position -> handle, parallel to `dense`.
    ids: Vec<BodyId>,
    slots: Vec<Slot>,
    /// Vacant slots, reused LIFO. Reuse order is part of the determinism
    /// contract: one spawn/despawn sequence must mint one handle sequence.
    free: Vec<u32>,
}

impl<S: PhysicsSpace> Default for BodyArena<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: PhysicsSpace> BodyArena<S> {
    pub fn new() -> Self {
        Self {
            dense: Vec::new(),
            ids: Vec::new(),
            slots: Vec::new(),
            free: Vec::new(),
        }
    }

    pub fn spawn(&mut self, body: RigidBody<S>) -> BodyId {
        let dense = self.dense.len() as u32;
        let slot = match self.free.pop() {
            Some(slot) => {
                self.slots[slot as usize].dense = Some(dense);
                slot
            }
            None => {
                self.slots.push(Slot {
                    generation: 0,
                    dense: Some(dense),
                });
                (self.slots.len() - 1) as u32
            }
        };
        let id = BodyId {
            slot,
            generation: self.slots[slot as usize].generation,
        };
        self.dense.push(body);
        self.ids.push(id);
        id
    }

    /// Remove the body `id` names, or `None` if the handle is stale.
    pub fn despawn(&mut self, id: BodyId) -> Option<RigidBody<S>> {
        let dense = self.dense_index(id)?;
        let slot = &mut self.slots[id.slot as usize];
        slot.dense = None;
        // A slot whose generation would wrap is retired instead of recycled:
        // wrapping is the one way a stale handle could come back to life and
        // alias a later body.
        if let Some(next) = slot.generation.checked_add(1) {
            slot.generation = next;
            self.free.push(id.slot);
        }

        let removed = self.dense.swap_remove(dense);
        self.ids.swap_remove(dense);
        if let Some(&moved) = self.ids.get(dense) {
            self.slots[moved.slot as usize].dense = Some(dense as u32);
        }
        Some(removed)
    }

    /// Position of `id` in the dense slice, or `None` if the handle is stale.
    /// Positions move under [`Self::despawn`] and are valid only until the
    /// next one.
    pub fn dense_index(&self, id: BodyId) -> Option<usize> {
        let slot = self.slots.get(id.slot as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        slot.dense.map(|dense| dense as usize)
    }

    /// Handle of the body at dense position `dense`.
    pub fn id_at(&self, dense: usize) -> BodyId {
        self.ids[dense]
    }

    pub fn get(&self, id: BodyId) -> Option<&RigidBody<S>> {
        self.dense_index(id).map(|dense| &self.dense[dense])
    }

    pub fn get_mut(&mut self, id: BodyId) -> Option<&mut RigidBody<S>> {
        match self.dense_index(id) {
            Some(dense) => Some(&mut self.dense[dense]),
            None => None,
        }
    }

    /// Mutable iteration in dense order: per-body edits without a handle on the
    /// slice the ordering invariant lives in.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, RigidBody<S>> {
        self.dense.iter_mut()
    }

    /// The dense slice, mutable. Split-borrowing the two bodies of a contact
    /// needs it and cannot go through [`Self::get_mut`] twice; it does not
    /// leave the crate, where the phase loops are the only callers and none of
    /// them permutes.
    pub(crate) fn dense_mut(&mut self) -> &mut [RigidBody<S>] {
        &mut self.dense
    }
}

impl<S: PhysicsSpace> Deref for BodyArena<S> {
    type Target = [RigidBody<S>];

    fn deref(&self) -> &Self::Target {
        &self.dense
    }
}

impl<S: PhysicsSpace> Index<BodyId> for BodyArena<S> {
    type Output = RigidBody<S>;

    fn index(&self, id: BodyId) -> &RigidBody<S> {
        self.get(id).expect(STALE_HANDLE)
    }
}

impl<S: PhysicsSpace> IndexMut<BodyId> for BodyArena<S> {
    fn index_mut(&mut self, id: BodyId) -> &mut RigidBody<S> {
        self.get_mut(id).expect(STALE_HANDLE)
    }
}

/// Dense-position indexing, the iteration-order view. [`Deref`] already
/// exposes the slice; this is the operator form of `arena[..][dense]`.
impl<S: PhysicsSpace> Index<usize> for BodyArena<S> {
    type Output = RigidBody<S>;

    fn index(&self, dense: usize) -> &RigidBody<S> {
        &self.dense[dense]
    }
}

impl<S: PhysicsSpace> IndexMut<usize> for BodyArena<S> {
    fn index_mut(&mut self, dense: usize) -> &mut RigidBody<S> {
        &mut self.dense[dense]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec3, Vec4};
    use loam_math::{Bivector3, Bivector4, EuclideanR3, EuclideanR4};

    #[test]
    fn dynamic_halfspace_3d_panics_in_debug() {
        let result = std::panic::catch_unwind(|| {
            RigidBody::<EuclideanR3>::new(
                Vec3::ZERO,
                Vec3::ZERO,
                Collider::HalfSpace {
                    normal: Vec3::Y,
                    offset: 0.0,
                },
                1.0,
                1.0,
                &EuclideanR3,
            )
        });
        assert!(
            result.is_err(),
            "expected debug_assert to fire on dynamic 3D half-space"
        );
    }

    #[test]
    fn dynamic_halfspace_4d_panics_in_debug() {
        let result = std::panic::catch_unwind(|| {
            RigidBody::<EuclideanR4>::new(
                Vec4::ZERO,
                Vec4::ZERO,
                Collider::HalfSpace4D {
                    normal: Vec4::Y,
                    offset: 0.0,
                },
                1.0,
                1.0,
                &EuclideanR4,
            )
        });
        assert!(
            result.is_err(),
            "expected debug_assert to fire on dynamic 4D half-space"
        );
    }

    // The impulse tests pass powers of two for mass and inertia so every
    // expected value below is exact in f32 and the asserts pin the formula
    // rather than a rounding tolerance.
    fn body_r3(position: Vec3, mass: f32, inertia: f32) -> RigidBody<EuclideanR3> {
        RigidBody::new(
            position,
            Vec3::ZERO,
            Collider::sphere_at_origin(0.5),
            mass,
            inertia,
            &EuclideanR3,
        )
    }

    fn body_r4(position: Vec4, mass: f32, inertia: f32) -> RigidBody<EuclideanR4> {
        RigidBody::new(
            position,
            Vec4::ZERO,
            Collider::sphere_at_origin(0.5),
            mass,
            inertia,
            &EuclideanR4,
        )
    }

    #[test]
    fn linear_impulse_changes_velocity_by_impulse_over_mass() {
        let mut body = body_r3(Vec3::ZERO, 4.0, 0.5);
        body.velocity = Vec3::new(1.0, 0.0, 0.0);
        body.apply_impulse(Vec3::new(8.0, -4.0, 2.0));
        assert_eq!(body.velocity, Vec3::new(3.0, -1.0, 0.5));
        assert_eq!(body.angular_velocity, Bivector3::ZERO);
    }

    /// An impulse through the centre of mass is pure translation, whichever
    /// entry point applies it.
    #[test]
    fn central_impulse_produces_no_spin_and_matches_linear_form() {
        let position = Vec3::new(2.0, -1.0, 3.0);
        let impulse = Vec3::new(8.0, -4.0, 2.0);

        let mut at_point = body_r3(position, 4.0, 0.5);
        at_point.apply_impulse_at_point(&EuclideanR3, impulse, position);

        let mut linear = body_r3(position, 4.0, 0.5);
        linear.apply_impulse(impulse);

        assert_eq!(at_point.velocity, linear.velocity);
        assert_eq!(at_point.angular_velocity, Bivector3::ZERO);
    }

    /// Off-centre impulse in R³: `Δω = I⁻¹(r ∧ J)`, sign included. A lever
    /// along +y with an impulse along +x spins negatively in the xy plane.
    #[test]
    fn off_center_impulse_spins_body_by_inverse_inertia_times_lever_wedge_impulse_r3() {
        let mut body = body_r3(Vec3::ZERO, 2.0, 0.5);
        body.apply_impulse_at_point(
            &EuclideanR3,
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        );

        assert_eq!(body.velocity, Vec3::new(1.5, 0.0, 0.0));
        // r ∧ J = (0,2,0) ∧ (3,0,0) = -6 e_xy; I⁻¹ = 2.
        assert_eq!(body.angular_velocity, Bivector3::new(-12.0, 0.0, 0.0));
    }

    /// The R⁴ twin, with the lever along +w so the response lands in the xw
    /// plane: a rotation plane with no R³ analogue.
    #[test]
    fn off_center_impulse_spins_body_by_inverse_inertia_times_lever_wedge_impulse_r4() {
        let mut body = body_r4(Vec4::ZERO, 2.0, 0.5);
        body.apply_impulse_at_point(
            &EuclideanR4,
            Vec4::new(3.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 2.0),
        );

        assert_eq!(body.velocity, Vec4::new(1.5, 0.0, 0.0, 0.0));
        // r ∧ J = (0,0,0,2) ∧ (3,0,0,0) = -6 e_xw; I⁻¹ = 2.
        let expected = Bivector4 {
            xw: -12.0,
            ..Bivector4::ZERO
        };
        assert_eq!(body.angular_velocity, expected);
    }

    /// Equal and opposite impulses at one shared world point are an internal
    /// interaction: total linear momentum and total angular momentum about a
    /// fixed origin are both unchanged. Pins the lever arm and the inverse
    /// inertia against the linear term; the absolute sign of the wedge is
    /// pinned by the two off-centre tests above.
    #[test]
    fn equal_and_opposite_impulses_conserve_linear_and_angular_momentum() {
        let space = EuclideanR3;

        let a_velocity_before = Vec3::new(0.5, -0.25, 1.0);
        let a_spin_before = Bivector3::new(0.125, -0.5, 0.25);
        let mut a = body_r3(Vec3::new(-1.0, 0.0, 0.0), 2.0, 0.5);
        a.velocity = a_velocity_before;
        a.angular_velocity = a_spin_before;

        let b_velocity_before = Vec3::new(-0.75, 0.5, 0.125);
        let b_spin_before = Bivector3::new(-0.25, 0.0, 0.5);
        let mut b = body_r3(Vec3::new(1.0, 0.5, 0.0), 4.0, 0.25);
        b.velocity = b_velocity_before;
        b.angular_velocity = b_spin_before;

        // L about the origin: Σ (x ∧ m·v) + I·ω, with scalar isotropic I.
        let momenta = |a: &RigidBody<EuclideanR3>, b: &RigidBody<EuclideanR3>| {
            let linear = a.velocity * a.mass + b.velocity * b.mass;
            let angular = space.wedge(a.position, a.velocity * a.mass)
                + a.angular_velocity * a.inertia
                + space.wedge(b.position, b.velocity * b.mass)
                + b.angular_velocity * b.inertia;
            (linear, angular)
        };

        let (linear_before, angular_before) = momenta(&a, &b);

        let point = Vec3::new(0.0, 0.25, 0.0);
        let impulse = Vec3::new(1.5, -0.5, 0.25);
        a.apply_impulse_at_point(&space, -impulse, point);
        b.apply_impulse_at_point(&space, impulse, point);

        let (linear_after, angular_after) = momenta(&a, &b);

        // Conservation is also satisfied by doing nothing, so pin that both
        // channels of both bodies actually responded.
        assert!((a.velocity - a_velocity_before).length() > 0.1);
        assert!((b.velocity - b_velocity_before).length() > 0.1);
        assert!((a.angular_velocity + a_spin_before * -1.0).magnitude() > 0.1);
        assert!((b.angular_velocity + b_spin_before * -1.0).magnitude() > 0.1);

        let linear_drift = (linear_after - linear_before).length();
        assert!(
            linear_drift < 1e-5,
            "linear momentum drifted by {linear_drift}"
        );
        let angular_drift = (angular_after + angular_before * -1.0).magnitude();
        assert!(
            angular_drift < 1e-5,
            "angular momentum drifted by {angular_drift}"
        );
    }

    /// `inv_mass == 0` is the documented static contract: impulses are inert
    /// on both channels, including the angular one that reads `inertia`
    /// directly.
    #[test]
    fn static_body_ignores_both_impulse_forms() {
        let mut body = RigidBody::<EuclideanR3>::fixed(
            Vec3::ZERO,
            Collider::sphere_at_origin(0.5),
            1.0,
            &EuclideanR3,
        );
        body.apply_impulse(Vec3::new(5.0, 5.0, 5.0));
        body.apply_impulse_at_point(
            &EuclideanR3,
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        assert_eq!(body.velocity, Vec3::ZERO);
        assert_eq!(body.angular_velocity, Bivector3::ZERO);
    }

    /// The property the generation exists for: after a slot is recycled the
    /// old handle must fail to resolve. Positional indices alias here, which
    /// is what silently rebinds a manifold key to a different body.
    #[test]
    fn stale_handle_never_resolves_after_its_slot_is_recycled() {
        let mut arena = BodyArena::new();
        let doomed = arena.spawn(body_r3(Vec3::X, 1.0, 1.0));
        assert!(arena.despawn(doomed).is_some());

        let recycled = arena.spawn(body_r3(Vec3::Y, 1.0, 1.0));
        assert_eq!(
            recycled.slot(),
            doomed.slot(),
            "the free slot was not reused, so this test is not exercising aliasing"
        );
        assert_ne!(recycled.generation(), doomed.generation());

        assert!(arena.get(doomed).is_none());
        assert!(arena.dense_index(doomed).is_none());
        assert_eq!(arena[recycled].position, Vec3::Y);
        assert!(
            arena.despawn(doomed).is_none(),
            "stale despawn must be inert"
        );
        assert!(
            arena.get(recycled).is_some(),
            "a stale despawn removed the live body sharing its slot"
        );
    }

    /// Despawn compacts storage, so the surviving handles must follow their
    /// bodies rather than their old positions.
    #[test]
    fn despawn_keeps_storage_dense_and_survivor_handles_valid() {
        let mut arena = BodyArena::new();
        let first = arena.spawn(body_r3(Vec3::X, 1.0, 1.0));
        let middle = arena.spawn(body_r3(Vec3::Y, 1.0, 1.0));
        let last = arena.spawn(body_r3(Vec3::Z, 1.0, 1.0));

        assert_eq!(arena.despawn(middle).map(|b| b.position), Some(Vec3::Y));

        assert_eq!(arena.len(), 2);
        assert_eq!(arena[first].position, Vec3::X);
        assert_eq!(
            arena[last].position,
            Vec3::Z,
            "the moved body's handle broke"
        );
        assert_eq!(arena.id_at(arena.dense_index(last).unwrap()), last);
        let positions: Vec<Vec3> = arena.iter().map(|b| b.position).collect();
        assert_eq!(positions, vec![Vec3::X, Vec3::Z]);
    }

    /// The correspondence the whole indirection rests on, stated over the
    /// public surface: every dense position resolves back through the handle
    /// stored at it. A permutation of the dense slice that the slot table did
    /// not follow breaks exactly this, and breaks it silently, which is why the
    /// mutable slice does not leave the crate.
    #[test]
    fn every_dense_position_resolves_back_through_its_own_handle() {
        let mut arena = BodyArena::new();
        let assert_consistent = |arena: &BodyArena<EuclideanR3>| {
            for dense in 0..arena.len() {
                assert_eq!(
                    arena.dense_index(arena.id_at(dense)),
                    Some(dense),
                    "slot table disagrees with dense position {dense}"
                );
            }
        };

        let mut live = Vec::new();
        for i in 0..5 {
            live.push(arena.spawn(body_r3(Vec3::splat(i as f32), 1.0, 1.0)));
            assert_consistent(&arena);
        }
        // The three despawns land at dense 0, then mid-slice, then the tail,
        // where swap_remove moves nothing and the slot write must be skipped.
        for victim in [live[0], live[2], live[3]] {
            assert!(arena.despawn(victim).is_some());
            assert_consistent(&arena);
        }
        arena.spawn(body_r3(Vec3::ZERO, 1.0, 1.0));
        assert_consistent(&arena);

        // Per-body mutation is the supported write path and must leave the
        // correspondence alone.
        for (dense, body) in arena.iter_mut().enumerate() {
            body.restitution = dense as f32;
        }
        assert_consistent(&arena);
        let restitutions: Vec<f32> = arena.iter().map(|b| b.restitution).collect();
        assert_eq!(restitutions, vec![0.0, 1.0, 2.0]);
    }

    /// Handle allocation is part of the determinism contract: two arenas
    /// driven by the same spawn/despawn sequence must mint the same handles,
    /// or a replay keyed on handles diverges.
    #[test]
    fn handle_sequence_is_reproducible_across_identical_operation_sequences() {
        let run = || {
            let mut arena = BodyArena::new();
            let mut minted = Vec::new();
            for i in 0..4 {
                minted.push(arena.spawn(body_r3(Vec3::splat(i as f32), 1.0, 1.0)));
            }
            assert!(arena.despawn(minted[1]).is_some());
            assert!(arena.despawn(minted[3]).is_some());
            for i in 0..3 {
                minted.push(arena.spawn(body_r3(Vec3::splat(-(i as f32)), 1.0, 1.0)));
            }
            minted
        };
        let first = run();
        assert_eq!(first, run());
        // Two recycled slots and one fresh one, or the sequence never
        // exercised the free list.
        assert_eq!(first[4].slot(), first[3].slot());
        assert_eq!(first[5].slot(), first[1].slot());
        assert_eq!(first[6].slot(), 4);
    }

    #[test]
    fn static_halfspace_4d_is_allowed() {
        // Mass = 0 means static; the guard must not fire.
        let _ = RigidBody::<EuclideanR4>::new(
            Vec4::ZERO,
            Vec4::ZERO,
            Collider::HalfSpace4D {
                normal: Vec4::Y,
                offset: 0.0,
            },
            0.0,
            1.0,
            &EuclideanR4,
        );
    }
}
