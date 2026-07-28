//! What the sort-and-sweep broadphase costs and what it prunes, at three body
//! counts. `cargo bench -p loam-physics`.
//!
//! Three timings per size, because "the sweep is Nx faster" means two different
//! things depending on the denominator.
//!
//! `ap_ns` is the broadphase the sweep replaced: every pair that is not two
//! static bodies, with no distance test and no sort. A speedup claim for the
//! sweep is a claim over this, and it is also what the narrowphase used to be
//! handed, so `ap/emitted` is the factor by which the narrowphase's input
//! shrank. That is where the win lives, since the narrowphase runs GJK per
//! emitted pair.
//!
//! `scan_ns` is the harder denominator: the sweep's own candidate predicate
//! with no acceleration structure, radii hoisted out of the quadratic loop so
//! the comparison is structure against no structure and not one loop forgetting
//! to hoist. The scan already does the culling the sweep is credited for above,
//! so `scan/sweep` isolates what the sort buys over identical output.
//!
//! Both denominators fill caller-owned buffers, because the code the sweep
//! replaced filled the world's retained `pair_order` rather than a fresh `Vec`.
//! Timing them against an owned allocation inflates them in proportion to their
//! output, and at 131x the emitted count that is most of the measurement. The
//! sweep is deliberately left on `World::broadphase`, which allocates its two
//! scratch buffers and its output per call where a step reuses all three, so
//! `sweep_ns` is an upper bound and every ratio below is a floor.
//!
//! Measured on a 13th Gen Intel Core i9-13980HX, Windows 11 Pro 10.0.26200,
//! rustc 1.95.0, `cargo bench` (opt-level 3, no debug assertions). Each cell is
//! the median of five process runs, each itself the median of nine batches.
//! Run-to-run spread is within 1.1x on every cell.
//!
//! ```text
//! bodies  emitted  ap_pairs  sweep_ns  scan_ns   ap_ns  scan/sweep  ap/sweep  ap/emitted
//!    101      160      5050      9585     6852    7107        0.7x      0.7x         32x
//!    201      312     20100     24724    24290   28296        1.0x      1.1x         64x
//!    401      611     80200     63976    91855  113825        1.4x      1.8x        131x
//! ```
//!
//! The sweep does not pay for itself in wall-clock time at these sizes. It is
//! slower than the broadphase it replaced at 101 bodies, break-even at 201, and
//! only 1.8x ahead at 401; against the identical-predicate scan it crosses over
//! later still. The ratios do grow with body count, which is the acceleration
//! structure doing what it is for, but 400 bodies is well short of where that
//! growth pays the sort back.
//!
//! What the sweep actually buys is `ap/emitted`: 32x to 131x fewer pairs handed
//! to the narrowphase, which runs GJK on each one. That is the number the
//! change was worth making for, and it is the one that is not close.
//!
//! Whether the interval sort or the active-list churn dominates the sweep's own
//! overhead is unmeasured.

use std::hint::black_box;
use std::time::Instant;

use glam::Vec3;
use loam_math::{EuclideanR3, Space};
use loam_physics::body::BodyId;
use loam_physics::collider::Collider;
use loam_physics::euclidean_r3::{
    box_body, halfspace_body_r3, register_default_narrowphase, sphere_body_r3,
};
use loam_physics::field::Gravity;
use loam_physics::world::PairKey;
use loam_physics::World;

const BODY_COUNTS: [usize; 3] = [100, 200, 400];
/// Half-width of the spawn box at 100 bodies, scaled as the cube root of the
/// count so the scene's density does not drift across the three sizes.
const SPREAD_AT_100: f32 = 6.0;
/// Long enough for the scene to fall and rest, so the timings describe the
/// contact-rich configuration a running world spends its time in.
const SETTLE_STEPS: usize = 240;
const DT: f32 = 1.0 / 240.0;
const GRAVITY_Y: f32 = -9.8;
/// Arbitrary but fixed, so a run reproduces.
const SEED: u64 = 0x9e37_79b9_7f4a_7c15;
const REPS: u32 = 200;
/// Odd, so the reported median is a batch that was actually observed.
const BATCHES: usize = 9;

/// xorshift64 (Marsaglia 2003, "Xorshift RNGs", J. Stat. Soft. 8(14), the
/// 13/7/17 triple).
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

/// `count` seeded spheres and boxes over one static floor, settled. The mix
/// matters: a polytope's bounding radius is a fold over its vertices, so a
/// scene of spheres alone would understate what the quadratic scan pays per
/// pair and overstate what it pays per body.
fn settled_scene(count: usize) -> World<EuclideanR3> {
    let spread = SPREAD_AT_100 * (count as f32 / 100.0).cbrt();
    let mut rng = Xorshift::new(SEED);
    let mut world = World::new(EuclideanR3);
    register_default_narrowphase(&mut world.narrowphase);
    world.push_field(Box::new(Gravity::new(Vec3::new(0.0, GRAVITY_Y, 0.0))));
    world.push_body(halfspace_body_r3(Vec3::Y, 0.0));

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
        world.bodies[id].restitution = 0.0;
    }
    for _ in 0..SETTLE_STEPS {
        world.step(DT);
    }
    world
}

fn bounding_radius(collider: &Collider) -> f32 {
    match collider {
        Collider::Sphere { radius, .. } => *radius,
        Collider::ConvexPolytope3D { vertices } => vertices
            .iter()
            .map(|v| v.length_squared())
            .fold(0.0_f32, f32::max)
            .sqrt(),
        Collider::HalfSpace { .. } => f32::INFINITY,
        other => unreachable!("the scene builds spheres, boxes and one half-space, not {other:?}"),
    }
}

/// The candidate definition without an acceleration structure: every pair that
/// is not two static bodies and whose bounding balls overlap.
fn scan(world: &World<EuclideanR3>, radii: &mut Vec<f32>, pairs: &mut Vec<PairKey>) {
    radii.clear();
    radii.extend(
        world
            .bodies
            .iter()
            .map(|body| bounding_radius(&body.collider)),
    );
    pairs.clear();
    let n = world.bodies.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let (a, b) = (&world.bodies[i], &world.bodies[j]);
            if a.inv_mass == 0.0 && b.inv_mass == 0.0 {
                continue;
            }
            if world.space.distance(a.position, b.position) <= radii[i] + radii[j] {
                pairs.push(canonical(world.bodies.id_at(i), world.bodies.id_at(j)));
            }
        }
    }
    pairs.sort_unstable();
}

/// The broadphase the sweep replaced: every pair that is not two static bodies,
/// with no bounding test and no sort. The scene never despawns, so handles
/// ascend with storage order and this emission is already sorted; dropping the
/// sort is faithful to the original rather than a head start.
fn all_pairs(world: &World<EuclideanR3>, pairs: &mut Vec<PairKey>) {
    pairs.clear();
    let n = world.bodies.len();
    for i in 0..n {
        for j in (i + 1)..n {
            if world.bodies[i].inv_mass == 0.0 && world.bodies[j].inv_mass == 0.0 {
                continue;
            }
            pairs.push(canonical(world.bodies.id_at(i), world.bodies.id_at(j)));
        }
    }
}

fn canonical(a: BodyId, b: BodyId) -> PairKey {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Median batch mean. A single batch mean is not reportable here: the all-pairs
/// emitter grows its output `Vec` from empty on every call, and the mean of one
/// batch of that moved by 4x between process runs. Taking the median across
/// batches holds 201 and 401 bodies to 1.7x and 1.2x; 101 stays bimodal, so the
/// spread there is stated with the number rather than averaged away.
fn median_nanos(mut body: impl FnMut()) -> f64 {
    let mut batches = [0.0_f64; BATCHES];
    for batch in &mut batches {
        let start = Instant::now();
        for _ in 0..REPS {
            body();
        }
        *batch = start.elapsed().as_nanos() as f64 / f64::from(REPS);
    }
    batches.sort_unstable_by(f64::total_cmp);
    batches[BATCHES / 2]
}

fn main() {
    println!(
        "bodies emitted ap_pairs sweep_ns scan_ns ap_ns          scan/sweep ap/sweep ap/emitted"
    );
    let mut radii = Vec::new();
    let mut scan_out = Vec::new();
    let mut ap_out = Vec::new();
    for count in BODY_COUNTS {
        let world = settled_scene(count);
        let emitted = world.broadphase();
        scan(&world, &mut radii, &mut scan_out);
        assert_eq!(
            emitted, scan_out,
            "{count}: the sweep and the scan disagree, so the timings below              compare different work"
        );

        all_pairs(&world, &mut ap_out);
        let n = world.bodies.len();
        assert_eq!(
            ap_out.len(),
            n * (n - 1) / 2,
            "{count}: the baseline dropped a pair, so it is culling and is no              longer the unaccelerated denominator"
        );
        let mut baseline = ap_out.clone();
        baseline.sort_unstable();
        assert!(
            emitted
                .iter()
                .all(|key| baseline.binary_search(key).is_ok()),
            "{count}: the sweep emits a pair the baseline never did, so the              ratio below is not a pruning factor"
        );

        // One untimed pass each, so the timed loop measures a warm cache and
        // not the scene's first-touch faults, and so both denominators enter
        // the timed region with their buffers already at steady size.
        black_box(world.broadphase());
        scan(&world, &mut radii, &mut scan_out);
        all_pairs(&world, &mut ap_out);

        let sweep_ns = median_nanos(|| {
            black_box(world.broadphase());
        });
        let scan_ns = median_nanos(|| {
            scan(&world, &mut radii, &mut scan_out);
            black_box(&scan_out);
        });
        let ap_ns = median_nanos(|| {
            all_pairs(&world, &mut ap_out);
            black_box(&ap_out);
        });

        println!(
            "{n:6} {:7} {:8} {sweep_ns:8.0} {scan_ns:7.0} {ap_ns:11.0}              {:9.1}x {:13.1}x {:15.0}x",
            emitted.len(),
            baseline.len(),
            scan_ns / sweep_ns,
            ap_ns / sweep_ns,
            baseline.len() as f64 / emitted.len() as f64,
        );
    }
}
