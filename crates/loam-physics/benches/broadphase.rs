//! What the sort-and-sweep broadphase costs and what it prunes, at three body
//! counts. `cargo bench -p loam-physics`.
//!
//! Three timings per size, because "the sweep is Nx faster" means two different
//! things depending on the denominator.
//!
//! `allpairs_ns` is the broadphase the sweep replaced: every pair that is not
//! two static bodies, with no distance test and no sort. A speedup claim for
//! the sweep is a claim over this, and it is also what the narrowphase used to
//! be handed, so `allpairs/emitted` is the factor by which the narrowphase's
//! input shrank. That is where the larger win lives, since the narrowphase runs
//! GJK per emitted pair.
//!
//! `scan_ns` is the harder denominator: the sweep's own candidate predicate
//! with no acceleration structure, radii hoisted out of the quadratic loop so
//! the comparison is structure against no structure and not one loop forgetting
//! to hoist. The scan already does the culling the sweep is credited for above,
//! so `scan/sweep` isolates what the sort buys over identical output.
//!
//! All three grow an owned `Vec<PairKey>` inside the timed region. The sweep is
//! not otherwise handed the same deal: `World::broadphase` allocates the two
//! sweep scratch buffers per call where the step reuses buffers the world
//! retains, so `sweep_ns` is an upper bound on what a step pays. The sweep's
//! output is asserted equal to the scan's and contained in the all-pairs set,
//! the only relation that holds against a denominator with no distance test.
//!
//! Measured on a 13th Gen Intel Core i9-13980HX, Windows 11 Pro 10.0.26200,
//! rustc 1.95.0, `cargo bench` (opt-level 3, no debug assertions). Each cell is
//! the median over seven process runs: the within-run median still leaves
//! `allpairs_ns` at 101 bodies bimodal, observed anywhere from 12 to 55
//! microseconds, where the 201 and 401 rows hold to within 1.7x and 1.2x. Read
//! the 101 ratios as an order of magnitude, not a figure.
//!
//! ```text
//! bodies emitted allpairs sweep_ns scan_ns allpairs_ns scan/sweep allpairs/sweep allpairs/emitted
//!    101     160     5050     9946    6907       31698       0.7x           3.2x              32x
//!    201     312    20100    26848   24770       95717       0.9x           3.6x              64x
//!    401     611    80200    74506   89880      563359       1.2x           7.6x             131x
//! ```
//!
//! The sweep's win over the broadphase it replaced grows with the body count,
//! 3.2x to 7.6x over a 4x range, so what it buys is the acceleration structure
//! and not a fixed overhead difference. Against the identical-predicate scan it
//! is behind at 101 bodies and only crosses over near 401, so at these sizes
//! the sweep's lead is the pairs it never emits and not the pairs it never
//! tests. Whether the interval sort or the active-list churn dominates that
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
fn scan(world: &World<EuclideanR3>) -> Vec<PairKey> {
    let radii: Vec<f32> = world
        .bodies
        .iter()
        .map(|body| bounding_radius(&body.collider))
        .collect();
    let mut pairs = Vec::new();
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
    pairs
}

/// The broadphase the sweep replaced: every pair that is not two static bodies,
/// with no bounding test and no sort. The scene never despawns, so handles
/// ascend with storage order and this emission is already sorted; dropping the
/// sort is faithful to the original rather than a head start.
fn all_pairs(world: &World<EuclideanR3>) -> Vec<PairKey> {
    let mut pairs = Vec::new();
    let n = world.bodies.len();
    for i in 0..n {
        for j in (i + 1)..n {
            if world.bodies[i].inv_mass == 0.0 && world.bodies[j].inv_mass == 0.0 {
                continue;
            }
            pairs.push(canonical(world.bodies.id_at(i), world.bodies.id_at(j)));
        }
    }
    pairs
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
        "bodies emitted allpairs sweep_ns scan_ns allpairs_ns \
         scan/sweep allpairs/sweep allpairs/emitted"
    );
    for count in BODY_COUNTS {
        let world = settled_scene(count);
        let emitted = world.broadphase();
        assert_eq!(
            emitted,
            scan(&world),
            "{count}: the sweep and the scan disagree, so the timings below \
             compare different work"
        );

        let mut baseline = all_pairs(&world);
        let n = world.bodies.len();
        assert_eq!(
            baseline.len(),
            n * (n - 1) / 2,
            "{count}: the baseline dropped a pair, so it is culling and is no \
             longer the unaccelerated denominator"
        );
        baseline.sort_unstable();
        assert!(
            emitted
                .iter()
                .all(|key| baseline.binary_search(key).is_ok()),
            "{count}: the sweep emits a pair the baseline never did, so the \
             ratio below is not a pruning factor"
        );

        // One untimed pass each, so the timed loop measures a warm cache and
        // not the scene's first-touch faults.
        black_box(world.broadphase());
        black_box(scan(&world));
        black_box(all_pairs(&world));

        let sweep_ns = median_nanos(|| {
            black_box(world.broadphase());
        });
        let scan_ns = median_nanos(|| {
            black_box(scan(&world));
        });
        let allpairs_ns = median_nanos(|| {
            black_box(all_pairs(&world));
        });

        println!(
            "{n:6} {:7} {:8} {sweep_ns:8.0} {scan_ns:7.0} {allpairs_ns:11.0} \
             {:9.1}x {:13.1}x {:15.0}x",
            emitted.len(),
            baseline.len(),
            scan_ns / sweep_ns,
            allpairs_ns / sweep_ns,
            baseline.len() as f64 / emitted.len() as f64,
        );
    }
}
