//! The determinism scenarios, one hash mixer, and one committed golden
//! constant per scenario, shared by every determinism gate in the crate.
//! Callers drive [`World::step`] through this module rather than
//! re-implementing the phase loop, so a schedule variant is always compared
//! against the simulation the golden hash pins.

use std::ops::Range;

use glam::{Vec3, Vec4};
use loam_math::{EuclideanR3, EuclideanR4};

use crate::body::RigidBody;
use crate::collision::VectorOps;
use crate::euclidean_r3::{
    halfspace_body_r3, register_default_narrowphase as register_narrowphase_r3, sphere_body_r3,
};
use crate::euclidean_r4::{
    halfspace4_body_r4, register_default_narrowphase as register_narrowphase_r4, sphere_body_r4,
};
use crate::field::Gravity;
use crate::integrator::PhysicsSpace;
use crate::world::{Schedule, World};

/// FNV-1a 64-bit (Fowler/Noll/Vo 1991; reference offset basis and prime,
/// <http://www.isthe.com/chongo/tech/comp/fnv/>). `std`'s `DefaultHasher` is
/// documented as unstable across releases, so a hash committed as a constant
/// needs its own mixer.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a64_update(mut hash: u64, words: &[u32]) -> u64 {
    for word in words {
        // Fixed little-endian byte order so the hash does not depend on host
        // endianness.
        for byte in word.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

pub fn fnv1a64(words: &[u32]) -> u64 {
    fnv1a64_update(FNV_OFFSET_BASIS, words)
}

/// FNV-1a of [`ScenarioRun::trajectory`] under the default schedule, recorded
/// on x86_64. This pins behavior rather than self-consistency: a
/// deterministic-but-changed integrator, solver order, contact constant, or
/// narrowphase moves it. Scoped to one architecture family, since glam's SIMD
/// dot reduces in a different order than its scalar fallback. When a
/// simulation change is intended, replace this with the value the assertion
/// prints.
pub const GOLDEN_TRAJECTORY_HASH: u64 = 0xfcfa_9165_cc85_e57b;

pub struct ScenarioRun {
    /// Every step, every body, linear and angular state as raw f32 bits, so
    /// the pins see the path and not just the endpoint.
    pub trajectory: Vec<u32>,
    /// Running hash after each step over a superset of `trajectory`: the same
    /// body words plus the manifold key list, each manifold's point count, and
    /// each point's accumulated normal impulse. Warm-start impulses are
    /// carried state, so a schedule change can leave body state identical for
    /// one step and diverge several steps later; hashing bodies alone would
    /// find that late or not at all. A vector rather than a scalar because the
    /// first differing index is the first divergent step, which is the whole
    /// triage story for a hash mismatch.
    pub step_hashes: Vec<u64>,
}

const STEPS: usize = 240;
const WORDS_PER_BODY_R4: usize = 14;
const WORDS_PER_BODY_R3: usize = 9;

/// Drive `world` and sample it. Every fixture routes through here so all of
/// them hash the same quantities in the same order, and so none of them can
/// drift into re-implementing the phase loop instead of driving
/// [`World::step`].
fn run_scenario<S, F>(
    world: &mut World<S>,
    dt: f32,
    steps: usize,
    words_per_body: usize,
    sample: F,
) -> ScenarioRun
where
    S: PhysicsSpace,
    S::Vector: VectorOps,
    S::Point: Copy + std::ops::Sub<Output = S::Vector>,
    F: Fn(&RigidBody<S>, &mut Vec<u32>),
{
    let mut run = ScenarioRun {
        trajectory: Vec::with_capacity(steps * world.bodies.len() * words_per_body),
        step_hashes: Vec::with_capacity(steps),
    };
    let mut hash = FNV_OFFSET_BASIS;
    let mut contact_words = Vec::new();
    for _ in 0..steps {
        world.step(dt);
        let step_start = run.trajectory.len();
        for body in world.bodies.iter() {
            sample(body, &mut run.trajectory);
        }

        // Sampled in `BTreeMap` key order under every schedule, so a moved
        // hash means the simulation diverged and never that the instrument
        // read the same state in a different order.
        contact_words.clear();
        for (key, manifold) in &world.manifolds {
            // Slot only: no fixture despawns, so the generation is a constant
            // zero and would add a word without adding a distinction.
            contact_words.push(key.0.slot());
            contact_words.push(key.1.slot());
            contact_words.push(manifold.points.len() as u32);
            for cp in &manifold.points {
                contact_words.push(cp.normal_impulse.to_bits());
            }
        }

        hash = fnv1a64_update(hash, &run.trajectory[step_start..]);
        hash = fnv1a64_update(hash, &contact_words);
        run.step_hashes.push(hash);
    }
    run
}

/// Orientation is deliberately not sampled by either sampler. `Bivector::exp`
/// routes through libm `sin`/`cos`, whose last-ULP results differ between
/// platform libms, and for sphere colliders orientation never feeds back into
/// the dynamics. Every sampled quantity comes from +, -, *, / and sqrt, which
/// IEEE-754 rounds exactly, so a trajectory is reproducible wherever glam takes
/// the same reduction path.
fn sample_body_r4(body: &RigidBody<EuclideanR4>, words: &mut Vec<u32>) {
    let p = body.position;
    let v = body.velocity;
    let w = body.angular_velocity;
    words.extend_from_slice(&[
        p.x.to_bits(),
        p.y.to_bits(),
        p.z.to_bits(),
        p.w.to_bits(),
        v.x.to_bits(),
        v.y.to_bits(),
        v.z.to_bits(),
        v.w.to_bits(),
        w.xy.to_bits(),
        w.xz.to_bits(),
        w.xw.to_bits(),
        w.yz.to_bits(),
        w.yw.to_bits(),
        w.zw.to_bits(),
    ]);
}

fn sample_body_r3(body: &RigidBody<EuclideanR3>, words: &mut Vec<u32>) {
    let p = body.position;
    let v = body.velocity;
    let w = body.angular_velocity;
    words.extend_from_slice(&[
        p.x.to_bits(),
        p.y.to_bits(),
        p.z.to_bits(),
        v.x.to_bits(),
        v.y.to_bits(),
        v.z.to_bits(),
        w.xy.to_bits(),
        w.yz.to_bits(),
        w.zx.to_bits(),
    ]);
}

/// The determinism fixture: 4D gravity, a static floor, and a six-sphere stack
/// at fixed offsets landing on it. No RNG, so any run-to-run difference is
/// genuine nondeterminism rather than seed noise.
pub fn determinism_scenario_run(schedule: Schedule) -> ScenarioRun {
    let mut world = World::new(EuclideanR4);
    // Contacts are what make the trajectory worth pinning: without a
    // narrowphase the scenario is free fall and exercises no solver, manifold,
    // or iteration-order behavior.
    register_narrowphase_r4(&mut world.narrowphase);
    world.schedule = schedule;
    world.push_field(Box::new(Gravity::new(Vec4::new(0.0, -9.8, 0.0, 0.0))));
    world.push_body(halfspace4_body_r4(Vec4::Y, 0.0));
    for i in 0..6u32 {
        let y = 1.0 + i as f32 * 0.45;
        let x = ((i % 3) as f32 - 1.0) * 0.05;
        world.push_body(sphere_body_r4(
            Vec4::new(x, y, 0.0, 0.0),
            Vec4::ZERO,
            0.2,
            1.0,
        ));
    }

    run_scenario(
        &mut world,
        1.0 / 60.0,
        STEPS,
        WORDS_PER_BODY_R4,
        sample_body_r4,
    )
}

pub fn determinism_scenario_trajectory() -> Vec<u32> {
    determinism_scenario_run(Schedule::default()).trajectory
}

/// Index of the first step whose hash differs, for an assertion message that
/// names where a schedule diverged instead of only that it did.
pub fn first_divergent_step(a: &ScenarioRun, b: &ScenarioRun) -> Option<usize> {
    a.step_hashes
        .iter()
        .zip(&b.step_hashes)
        .position(|(x, y)| x != y)
}

// ---------------------------------------------------------------------------
// Multi-island R3 fixture.
// ---------------------------------------------------------------------------

const ISLAND_RADIUS: f32 = 0.5;
/// Group centres on the x axis, separated by several diameters so a group
/// would have to travel to reach its neighbour.
const ISLAND_X: [f32; 3] = [-4.0, 0.0, 4.0];
/// Bodies per group. The four-body chain is why this fixture exists: island
/// order and colour order are both vacuous on a single stack, so the R4
/// scenario cannot serve the axes that arrive with union-find islands and a
/// coloured solve.
const ISLAND_SIZES: [usize; 3] = [4, 2, 1];
/// Initial surface-to-surface separation, so the run opens with an impact
/// rather than with bodies already resting in contact.
const ISLAND_GAP: f32 = 0.05;
pub const MULTI_ISLAND_DT: f32 = 1.0 / 60.0;
pub const MULTI_ISLAND_STEPS: usize = 240;

/// Dynamic body slot ranges, one per group. Slot 0 is the static floor and
/// belongs to no group. Slots rather than handles because the fixture never
/// despawns, so slot allocation is dense and contiguous by group.
pub fn multi_island_groups() -> [Range<usize>; 3] {
    std::array::from_fn(|group| {
        let start = 1 + ISLAND_SIZES[..group].iter().sum::<usize>();
        start..start + ISLAND_SIZES[group]
    })
}

/// Three spatially disjoint sphere groups over one static floor: a four-body
/// chain, a pair, and a singleton.
///
/// Every body starts on its group's vertical axis at rest, every contact normal
/// in the scenario is therefore vertical, and the friction solve returns early
/// below its 1e-8 tangential-speed floor, so no group ever moves laterally.
/// That is what makes the island partition constant for the whole run rather
/// than a property that happens to hold for the first few hundred steps.
///
/// The floor is static, so it transmits no impulse between groups and does not
/// merge islands under the usual union-find rule.
pub fn multi_island_world(schedule: Schedule) -> World<EuclideanR3> {
    let mut world = World::new(EuclideanR3);
    register_narrowphase_r3(&mut world.narrowphase);
    world.schedule = schedule;
    world.push_field(Box::new(Gravity::new(Vec3::new(0.0, -9.8, 0.0))));
    world.push_body(halfspace_body_r3(Vec3::Y, 0.0));
    for (group, &size) in ISLAND_SIZES.iter().enumerate() {
        for level in 0..size {
            let y = ISLAND_RADIUS + ISLAND_GAP + level as f32 * (2.0 * ISLAND_RADIUS + ISLAND_GAP);
            world.push_body(sphere_body_r3(
                Vec3::new(ISLAND_X[group], y, 0.0),
                Vec3::ZERO,
                ISLAND_RADIUS,
                1.0,
            ));
        }
    }
    world
}

pub fn multi_island_scenario_run(schedule: Schedule) -> ScenarioRun {
    let mut world = multi_island_world(schedule);
    run_scenario(
        &mut world,
        MULTI_ISLAND_DT,
        MULTI_ISLAND_STEPS,
        WORDS_PER_BODY_R3,
        sample_body_r3,
    )
}

/// FNV-1a of [`multi_island_scenario_run`]'s trajectory under the default
/// schedule, recorded on x86_64 on the same terms as
/// [`GOLDEN_TRAJECTORY_HASH`].
pub const GOLDEN_MULTI_ISLAND_HASH: u64 = 0x56fd_21a0_2e4f_76e2;
