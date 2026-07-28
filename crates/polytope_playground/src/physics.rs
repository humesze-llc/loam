//! The playground's rigid-body layer: a [`World<EuclideanR4>`] holding one
//! dynamic body per rendered row slot. The four Shapes-view render paths (SDF
//! upload, section caps, wireframe overlay, point sprites) source their pose
//! here, so they cannot disagree about where a body is.
//!
//! Filmstrip is outside the seam: its cells are a w/t sweep of a single
//! subject drawn at a fixed centre from the UI spin rotor alone, with no body
//! behind them.
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
    ///
    /// Precondition of the wireframe's S³ arc path (`blend > 0` in
    /// [`crate::wireframe_geom::push_blended_edge`]): `position.w == 0`. That
    /// path reads each endpoint's `length()` as its circumradius, which holds
    /// only while the frame is origin-centred; the `w` offset moves the body
    /// off the origin, so the endpoints stop sharing a radius and the interior
    /// bows onto a sphere the body is not on. Dormant until something throws a
    /// body off the slice, and the fix is to arc in the body's own centred
    /// frame rather than to drop the offset (the section cut needs it).
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
    /// them: their keys name the despawned handles, so every surviving entry
    /// is unreachable warm-start state the next step would walk for nothing.
    pub(crate) fn respawn(&mut self, slots: usize, radius: f32) {
        // Despawn rather than replace the arena: a fresh arena restarts
        // generations at 0, so a handle held across a respawn would alias
        // whichever body lands in its slot next, which is the exact aliasing
        // the generation counter exists to prevent.
        while let Some(last) = self.world.bodies.len().checked_sub(1) {
            let id = self.world.bodies.id_at(last);
            self.world.bodies.despawn(id);
        }
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
    /// is what makes this safe to run on any frame: a throw in flight survives.
    pub(crate) fn sync(&mut self, slots: usize, radius: f32) {
        if self.world.bodies.len() != slots {
            self.respawn(slots, radius);
            return;
        }
        for body in self.world.bodies.iter_mut() {
            body.collider = Collider::sphere_at_origin(radius);
            body.inertia = ball4_inertia(body.mass, radius);
        }
    }

    /// True while no body carries motion. Exact zero rather than a sleep
    /// threshold: with no force field and no damping the resting row is an
    /// exact fixpoint of the integrator, so this reads as "nothing is moving
    /// right now". It is not a record of whether anything was ever thrown; a
    /// throw the contact solver has fully cancelled reads at rest again.
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

    /// Pose of `slot` in a rendered row of `slots` bodies, under the UI spin
    /// rotor `spin`.
    ///
    /// `slots` is the caller's own row length and is checked, not trusted: the
    /// layout is frozen into each body at [`Self::respawn`] time, so a world
    /// that missed a row edit would draw every body at another slot's layout
    /// position and index past the end on the tail. [`Self::sync`] is the
    /// reconciliation point, and the body upload runs it at every row and size
    /// edit, before any render path reads a pose.
    pub(crate) fn pose(&self, slot: usize, slots: usize, spin: Rotor4) -> BodyPose {
        assert_eq!(
            self.world.bodies.len(),
            slots,
            "physics world not synced to the rendered row"
        );
        let body = &self.world.bodies[slot];
        BodyPose {
            position: body.position,
            rotor: composed_rotor(spin, body.orientation.rotation),
        }
    }

    /// Carry `canonical` into `slot`'s live body frame (writing `out`) and
    /// return the R³ translate the raster paths apply AFTER projection. `out`
    /// is cleared and refilled so a caller's scratch keeps its capacity.
    ///
    /// The single seam between the world and the raster passes: points,
    /// section caps, and the wireframe take all of their per-body geometry
    /// from here, which is what stops a pass from quietly falling back to the
    /// authored spin rotor over the static layout.
    pub(crate) fn body_frame(
        &self,
        slot: usize,
        slots: usize,
        spin: Rotor4,
        canonical: &[Vec4],
        size: f32,
        out: &mut Vec<Vec4>,
    ) -> Vec3 {
        let pose = self.pose(slot, slots, spin);
        out.clear();
        out.extend(canonical.iter().map(|v| pose.body_local(*v, size)));
        pose.position_r3()
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
            let pose = physics.pose(slot, slots, Rotor4::IDENTITY);
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
                physics
                    .pose(slot, slots, Rotor4::IDENTITY)
                    .position
                    .to_array(),
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
                        physics.pose(slot, 3, spin).rotor,
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
        let moved = physics.pose(1, slots, Rotor4::IDENTITY).position;
        assert!(
            (moved - expected).length() < 1e-5,
            "thrown pose {moved} away from {expected}"
        );
        for slot in [0, 2] {
            assert_eq!(
                physics
                    .pose(slot, slots, Rotor4::IDENTITY)
                    .position
                    .to_array(),
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
        let composed = physics.pose(0, 1, spin).rotor;
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
        let in_flight = physics.pose(0, 3, Rotor4::IDENTITY).position;

        physics.sync(3, RADIUS);
        assert_eq!(
            physics.pose(0, 3, Rotor4::IDENTITY).position,
            in_flight,
            "same-count sync cancelled a throw"
        );

        physics.sync(4, RADIUS);
        assert!(physics.at_rest(), "respawn left motion behind");
        for slot in 0..4 {
            assert_eq!(
                physics.pose(slot, 4, Rotor4::IDENTITY).position.to_array(),
                body_position(slot, 4)
            );
        }
    }

    /// Throw slot 1 off-centre so it carries BOTH a linear and an angular
    /// velocity, then run it far enough that its pose cannot be confused with
    /// the layout.
    fn tumbling(slots: usize) -> PlaygroundPhysics {
        let mut physics = PlaygroundPhysics::new(slots, RADIUS);
        let layout = Vec4::from_array(body_position(1, slots));
        // The +w lever puts the torque in the xw plane. The push is mostly +w,
        // the axis on which the body cannot reach a neighbour, with enough +x
        // to move its R³ translate off the layout as well; 0.16 of travel over
        // these ticks against a 0.4 surface gap keeps the row a clean control
        // group.
        physics.world.bodies[1].apply_impulse_at_point(
            &EuclideanR4,
            Vec4::new(0.4, 0.0, 0.0, 1.2),
            layout + Vec4::W * 0.5,
        );
        physics.step(24);
        physics
    }

    /// [`PlaygroundPhysics::body_frame`] is the seam every raster pass reads
    /// its per-body geometry through, so it must report the LIVE pose: the
    /// composed rotor and the body's own `w` in the frame vertices, the body's
    /// live centre in the R³ translate. Reverting it to the authored
    /// `size * spin.apply(v)` over the static layout, which is what unwiring a
    /// raster pass from physics means, fails here.
    #[test]
    fn body_frame_reports_the_live_pose_not_the_authored_spin() {
        let slots = 3;
        let physics = tumbling(slots);
        let spin = rotor_at(Plane4::Xy, 0.7);
        let size = 0.4;
        let canonical = [
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 0.6, -0.3, 0.2),
        ];

        let mut out = Vec::new();
        let origin = physics.body_frame(1, slots, spin, &canonical, size, &mut out);

        let body = &physics.world.bodies[1];
        let composed = composed_rotor(spin, body.orientation.rotation);
        assert_ne!(
            body.orientation.rotation,
            Rotor4::IDENTITY,
            "throw produced no rotation, so the pin below is vacuous"
        );
        assert_eq!(origin, body.position.truncate());
        assert_ne!(
            origin,
            Vec4::from_array(body_position(1, slots)).truncate(),
            "R³ translate still reads the static layout"
        );
        for (i, v) in canonical.iter().enumerate() {
            assert_eq!(
                out[i],
                size * composed.apply(*v) + Vec4::W * body.position.w
            );
            assert_ne!(
                out[i],
                size * spin.apply(*v),
                "frame vertex {i} still reads the authored spin alone"
            );
        }
    }

    /// An untouched slot's frame is byte-identical to the pre-physics
    /// `size * spin.apply(v)`: the seam adds no drift to a body nobody threw,
    /// which is what lets the pin above use exact equality.
    #[test]
    fn body_frame_of_an_untouched_slot_is_the_authored_spin_exactly() {
        let slots = 3;
        let physics = tumbling(slots);
        let spin = rotor_at(Plane4::Zw, -1.1);
        let size = 0.4;
        let canonical = [Vec4::new(0.2, -0.7, 0.5, 0.1)];

        let mut out = Vec::new();
        let origin = physics.body_frame(2, slots, spin, &canonical, size, &mut out);
        assert_eq!(out[0], size * spin.apply(canonical[0]));
        assert_eq!(origin, Vec4::from_array(body_position(2, slots)).truncate());
    }

    /// `out` is refilled, not appended to, so a caller passing a per-frame
    /// scratch buffer gets exactly one body's vertices.
    #[test]
    fn body_frame_refills_the_scratch_buffer() {
        let physics = PlaygroundPhysics::new(2, RADIUS);
        let canonical = [Vec4::X, Vec4::Y, Vec4::Z];
        let mut out = vec![Vec4::ONE; 7];
        physics.body_frame(0, 2, Rotor4::IDENTITY, &canonical, 1.0, &mut out);
        assert_eq!(out.len(), canonical.len());
    }

    /// A render path reading a world the row edit never reached is a bug, not
    /// a rendering: the slot count is checked at the seam rather than left to
    /// index out of bounds on the tail or, worse, silently draw a body at
    /// another slot's layout position.
    #[test]
    #[should_panic(expected = "physics world not synced to the rendered row")]
    fn pose_rejects_a_row_the_world_was_not_synced_to() {
        let physics = PlaygroundPhysics::new(3, RADIUS);
        physics.pose(0, 4, Rotor4::IDENTITY);
    }
}
