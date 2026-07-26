//! The playground's rigid-body layer: a [`World<EuclideanR4>`] holding one
//! dynamic body per rendered row slot. Every render path sources its pose
//! here, so the SDF, the section caps, and the wireframe cannot disagree
//! about where a body is.
//!
//! The chamber is zero-g and empty of static geometry: no [`ForceField`] is
//! registered, so a body only moves once something throws it.
//!
//! [`ForceField`]: loam_physics::ForceField

use glam::{Vec3, Vec4};
use loam_math::{EuclideanR4, Rotor, Rotor4};
use loam_physics::euclidean_r4::{ball4_inertia, register_default_narrowphase, sphere_body_r4};
use loam_physics::{Collider, World};

use crate::state::body_position;

/// Physics tick, matching the app's fixed 60 Hz sim tick. The world advances
/// `FrameCtx::n_ticks` steps of this rather than one step of the frame's wall
/// time, so a trajectory is reproducible across frame rates.
const PHYSICS_DT: f32 = 1.0 / 60.0;

/// Uniform body mass. Row members differ in vertex count, which is not a
/// quantity the bounding-sphere collider prices, so a per-shape mass would be
/// a number with nothing behind it.
const BODY_MASS: f32 = 1.0;

/// Rendered orientation for a body: the UI spin applied first, then the
/// body's physics orientation. [`Rotor4`] multiplies left-first
/// (`apply(a * b, v) == apply(b, apply(a, v))`), so the world-frame physics
/// rotor is the right factor.
///
/// An identity physics orientation returns `spin` component-for-component:
/// every product against the identity's zeros vanishes exactly, which is what
/// leaves the rotation UI untouched while nothing has been thrown.
pub(crate) fn composed_rotor(spin: Rotor4, orientation: Rotor4) -> Rotor4 {
    spin * orientation
}

/// A rendered body's world pose.
#[derive(Copy, Clone, Debug)]
pub(crate) struct BodyPose {
    pub(crate) position: Vec4,
    pub(crate) rotor: Rotor4,
}

impl BodyPose {
    /// The R³ translation the raster paths apply AFTER projection, so a
    /// Perspective4D divide never scales the body's x-position.
    pub(crate) fn position_r3(&self) -> Vec3 {
        self.position.truncate()
    }

    /// A canonical vertex in the body's own 4D frame: oriented, scaled by
    /// `size`, then offset by the body's `w`. The `w` offset is what keeps the
    /// world `w_slice` cutting the body where physics put it instead of always
    /// through its centre; it is exactly zero for a body on the layout.
    pub(crate) fn body_local(&self, canonical: Vec4, size: f32) -> Vec4 {
        size * self.rotor.apply(canonical) + Vec4::W * self.position.w
    }
}

pub(crate) struct PlaygroundPhysics {
    pub(crate) world: World<EuclideanR4>,
}

impl PlaygroundPhysics {
    pub(crate) fn new(slots: usize, radius: f32) -> Self {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        let mut physics = Self { world };
        physics.respawn(slots, radius);
        physics
    }

    /// Drop every body back onto the static layout at rest. Manifolds go with
    /// them: warm-start impulses are keyed by slot, so a surviving entry would
    /// be inherited by whichever body lands in that slot next.
    pub(crate) fn respawn(&mut self, slots: usize, radius: f32) {
        self.world.bodies.clear();
        self.world.manifolds.clear();
        for slot in 0..slots {
            let position = Vec4::from_array(body_position(slot, slots));
            self.world
                .push_body(sphere_body_r4(position, Vec4::ZERO, radius, BODY_MASS));
        }
    }

    /// Reconcile with a row of `slots` bodies. A slot-count change respawns
    /// the row, because the layout position is a function of the count and so
    /// every body moves. A same-count call only refreshes the collider, which
    /// is what makes this safe to run every frame: a throw in flight survives.
    pub(crate) fn sync(&mut self, slots: usize, radius: f32) {
        if self.world.bodies.len() != slots {
            self.respawn(slots, radius);
            return;
        }
        for body in &mut self.world.bodies {
            body.collider = Collider::sphere_at_origin(radius);
            body.inertia = ball4_inertia(body.mass, radius);
        }
    }

    /// True while no body carries motion. Exact zero rather than a sleep
    /// threshold: with no force field and no damping the resting row is an
    /// exact fixpoint of the integrator, so this reads as "nothing has been
    /// thrown yet".
    pub(crate) fn at_rest(&self) -> bool {
        self.world
            .bodies
            .iter()
            .all(|b| b.velocity == Vec4::ZERO && b.angular_velocity.magnitude_squared() == 0.0)
    }

    /// Advance `ticks` fixed steps, skipped entirely while at rest. The skip
    /// is load-bearing rather than an optimization: `surface scale` past
    /// `BODY_X_SPACING / (2 · BODY_SIZE)` overlaps neighbouring bounding
    /// spheres, and solving that overlap would push a row nobody threw off its
    /// layout.
    pub(crate) fn step(&mut self, ticks: usize) {
        if self.at_rest() {
            return;
        }
        for _ in 0..ticks {
            self.world.step(PHYSICS_DT);
        }
    }

    /// Pose of `slot`'s body under the UI spin rotor `spin`.
    pub(crate) fn pose(&self, slot: usize, spin: Rotor4) -> BodyPose {
        let body = &self.world.bodies[slot];
        BodyPose {
            position: body.position,
            rotor: composed_rotor(spin, body.orientation.rotation),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_math::{Bivector, Plane4};

    const RADIUS: f32 = crate::consts::BODY_SIZE;

    fn rotor_at(plane: Plane4, angle: f32) -> Rotor4 {
        (plane.unit_bivector() * angle).exp().normalize()
    }

    /// A world nothing has thrown holds the static layout exactly, however
    /// long it runs: the demo's boot state is a fixpoint, not a slow drift.
    #[test]
    fn at_rest_world_holds_the_static_layout() {
        let slots = 4;
        let mut physics = PlaygroundPhysics::new(slots, RADIUS);
        assert!(physics.at_rest());
        physics.step(600);
        for slot in 0..slots {
            let pose = physics.pose(slot, Rotor4::IDENTITY);
            assert_eq!(pose.position.to_array(), body_position(slot, slots));
            assert_eq!(pose.rotor, Rotor4::IDENTITY);
        }
    }

    /// Overlapping bounding spheres, which `surface scale` past
    /// `BODY_X_SPACING / (2 · BODY_SIZE)` produces, must not push a row nobody
    /// threw off its layout. Pins the at-rest skip in [`PlaygroundPhysics::step`]:
    /// without it the contact solver separates the row on its own.
    #[test]
    fn overlapping_layout_at_rest_is_never_pushed_apart() {
        let slots = 4;
        let mut physics = PlaygroundPhysics::new(slots, crate::consts::BODY_X_SPACING);
        physics.step(120);
        for slot in 0..slots {
            assert_eq!(
                physics.pose(slot, Rotor4::IDENTITY).position.to_array(),
                body_position(slot, slots)
            );
        }
    }

    /// The rendered rotor equals the UI spin component-for-component while the
    /// physics orientation is identity. This is the rotation-UI half of "idle
    /// physics changes nothing": a tolerance here would let a real drift hide.
    #[test]
    fn idle_orientation_leaves_the_spin_rotor_exact() {
        let physics = PlaygroundPhysics::new(3, RADIUS);
        for plane in Plane4::ALL {
            for &angle in &[0.3_f32, 1.7, -2.4] {
                let spin = rotor_at(plane, angle);
                for slot in 0..3 {
                    assert_eq!(
                        physics.pose(slot, spin).rotor,
                        spin,
                        "{plane:?} at {angle} rad perturbed the spin rotor"
                    );
                }
            }
        }
    }

    /// A body at the layout `w = 0` maps canonical vertices exactly as the
    /// pre-physics `size * rotor.apply(v)` did; a body physics moved off the
    /// slice carries its `w` into the frame the cut is taken in.
    #[test]
    fn body_local_carries_the_body_w_into_the_slice_frame() {
        let v = Vec4::new(0.5, -0.25, 0.125, 0.75);
        let flat = BodyPose {
            position: Vec4::new(1.0, 0.9, 0.0, 0.0),
            rotor: Rotor4::IDENTITY,
        };
        assert_eq!(flat.body_local(v, RADIUS), RADIUS * v);
        assert_eq!(flat.position_r3(), Vec3::new(1.0, 0.9, 0.0));

        let lifted = BodyPose {
            position: Vec4::new(1.0, 0.9, 0.0, 0.25),
            rotor: Rotor4::IDENTITY,
        };
        assert_eq!(
            lifted.body_local(v, RADIUS),
            RADIUS * v + Vec4::new(0.0, 0.0, 0.0, 0.25)
        );
    }

    /// An impulse moves the thrown body's rendered pose by `J/m · t` and
    /// leaves every other slot on the layout: poses follow the bodies, and
    /// only the bodies that were thrown.
    #[test]
    fn impulse_drives_the_thrown_slot_and_only_that_slot() {
        let slots = 3;
        let ticks = 30;
        let mut physics = PlaygroundPhysics::new(slots, RADIUS);
        // Thrown along +w: the one axis with no R³ analogue, and one that
        // cannot bring the body into contact with its neighbours.
        let impulse = Vec4::new(0.0, 0.0, 0.0, 2.0);
        physics.world.bodies[1].apply_impulse(impulse);
        assert!(!physics.at_rest());
        physics.step(ticks);

        let expected =
            Vec4::from_array(body_position(1, slots)) + impulse * (ticks as f32 * PHYSICS_DT);
        let moved = physics.pose(1, Rotor4::IDENTITY).position;
        assert!(
            (moved - expected).length() < 1e-5,
            "thrown pose {moved} away from {expected}"
        );
        for slot in [0, 2] {
            assert_eq!(
                physics.pose(slot, Rotor4::IDENTITY).position.to_array(),
                body_position(slot, slots),
                "untouched slot {slot} moved"
            );
        }
    }

    /// An off-centre impulse spins the body, and the rendered rotor applies
    /// the UI spin FIRST and the physics orientation second. Reversing the
    /// factors would rotate the body's own animation into the world frame.
    #[test]
    fn angular_impulse_composes_after_the_ui_spin() {
        let mut physics = PlaygroundPhysics::new(1, RADIUS);
        let layout = Vec4::from_array(body_position(0, 1));
        physics.world.bodies[0].apply_impulse_at_point(
            &EuclideanR4,
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            layout + Vec4::W * 0.5,
        );
        physics.step(10);

        let orientation = physics.world.bodies[0].orientation.rotation;
        assert_ne!(
            orientation,
            Rotor4::IDENTITY,
            "off-centre impulse produced no rotation"
        );

        // The impulse's lever `∧` force lands in the xw plane, so the spin
        // must share an index with it: absolutely orthogonal rotors commute
        // and could not tell the two composition orders apart.
        let spin = rotor_at(Plane4::Xy, 0.9);
        let composed = physics.pose(0, spin).rotor;
        let v = Vec4::new(0.3, -0.2, 0.9, 0.1);
        let staged = orientation.apply(spin.apply(v));
        assert!(
            (composed.apply(v) - staged).length() < 1e-5,
            "composition order is not spin-then-physics"
        );
    }

    /// `sync` respawns on a slot-count change (the layout is a function of the
    /// count) and leaves poses alone otherwise, which is what lets the render
    /// path call it every frame without cancelling a throw.
    #[test]
    fn sync_respawns_only_when_the_slot_count_changes() {
        let mut physics = PlaygroundPhysics::new(3, RADIUS);
        physics.world.bodies[0].apply_impulse(Vec4::new(0.0, 0.0, 0.0, 1.0));
        physics.step(10);
        let in_flight = physics.pose(0, Rotor4::IDENTITY).position;

        physics.sync(3, RADIUS);
        assert_eq!(
            physics.pose(0, Rotor4::IDENTITY).position,
            in_flight,
            "same-count sync cancelled a throw"
        );

        physics.sync(4, RADIUS);
        assert!(physics.at_rest(), "respawn left motion behind");
        for slot in 0..4 {
            assert_eq!(
                physics.pose(slot, Rotor4::IDENTITY).position.to_array(),
                body_position(slot, 4)
            );
        }
    }
}
