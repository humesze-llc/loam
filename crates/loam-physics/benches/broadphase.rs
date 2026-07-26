//! What the sort-and-sweep broadphase costs and what it prunes, at three body
//! counts. `cargo bench -p loam-physics`.
//!
//! Two numbers per size, because the sweep pays off in two different places.
//! `scan/sweep` is the phase's own speedup against a brute-force scan of the
//! same candidate predicate, with bounding radii hoisted so the comparison is
//! acceleration structure against no acceleration structure and not one loop
//! forgetting to hoist. `quadratic/emitted` is the factor by which the
//! narrowphase's input shrinks, which is where the larger win lives: the
//! narrowphase runs GJK per emitted pair.
//!
//! Both sides return an owned `Vec<PairKey>` and both compute their radii
//! inside the timed region, so neither is handed work the other pays for.
//! Emission is asserted equal before timing, so a run that reports a speedup
//! is reporting one over identical output.

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

fn canonical(a: BodyId, b: BodyId) -> PairKey {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

fn mean_nanos(mut body: impl FnMut()) -> f64 {
    let start = Instant::now();
    for _ in 0..REPS {
        body();
    }
    start.elapsed().as_nanos() as f64 / f64::from(REPS)
}

fn main() {
    println!("bodies emitted quadratic sweep_ns scan_ns scan/sweep quadratic/emitted");
    for count in BODY_COUNTS {
        let world = settled_scene(count);
        let emitted = world.broadphase();
        assert_eq!(
            emitted,
            scan(&world),
            "{count}: the sweep and the scan disagree, so the timings below \
             compare different work"
        );

        // One untimed pass each, so the timed loop measures a warm cache and
        // not the scene's first-touch faults.
        black_box(world.broadphase());
        black_box(scan(&world));

        let sweep_ns = mean_nanos(|| {
            black_box(world.broadphase());
        });
        let scan_ns = mean_nanos(|| {
            black_box(scan(&world));
        });

        let n = world.bodies.len();
        let quadratic = n * (n - 1) / 2;
        println!(
            "{n:6} {:7} {quadratic:9} {sweep_ns:8.0} {scan_ns:7.0} {:10.1}x {:16.0}x",
            emitted.len(),
            scan_ns / sweep_ns,
            quadratic as f64 / emitted.len() as f64,
        );
    }
}
