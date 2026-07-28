//! A perf tripwire CI can run: one fixed CPU sphere-trace workload, timed
//! against a stated budget, failing the build when it is exceeded.
//!
//! `#[ignore]`d so the ordinary `cargo test` gate neither pays for it nor
//! reports a timing taken from a debug build. CI runs it in its own release
//! job.
//!
//! # What is timed, and what is not
//!
//! [`march_grid`] sphere-traces a fixed grid of primary rays through a fixed
//! scene, stepping along geodesics with [`Space::exp`] and
//! [`Space::parallel_transport`] and sampling with [`Scene::eval`]. That is the
//! entire denominator. The timed region allocates nothing: the scene is built
//! before it, and `Scene::eval` walks the tree by recursion with no heap
//! traffic. It takes no lock, opens no file, emits and validates no WGSL, and
//! never requests a wgpu adapter.
//!
//! Naming the denominator is the point. The broadphase bench under
//! `crates/loam-physics/benches/` documents its first version timing an
//! allocation-dominated baseline, which reported a speedup roughly 4x larger
//! than the real one; a budget measured against a denominator that includes
//! allocation or I/O is a budget on those, not on the march.
//!
//! So this gate does not cover: shader assembly and validation, pipeline
//! creation, GPU execution, the egui pair, presentation, or the frame loop as a
//! whole. A regression in any of those passes here.
//!
//! # Why a median of a CPU workload and not a frame percentile
//!
//! Measured on the maintainer's machine the playground's `frame` section runs
//! p50 3.96ms against p95 13.18ms (`docs/PERF.md` records the trace method).
//! That 3.3x spread lives in scheduler, compositor and present-path tails, not
//! in arithmetic, and a shared CI runner has more of it and bounds none of it.
//! A gate on a p95 would be a gate on the runner's other tenants and would
//! flake by construction.
//!
//! The statistic here is the median of [`BATCHES`] batches, each batch one full
//! grid pass. `BATCHES` is odd, so the reported figure is a pass that actually
//! ran rather than an interpolation, and the median survives up to
//! `BATCHES / 2` preempted passes. What remains after preemption is removed is
//! scalar f32 work on a warm working set, which is the most reproducible thing
//! available on a machine this code does not own.
//!
//! # The budget
//!
//! Measured in release on a 13th Gen Intel Core i9-13980HX, Windows 11 Pro
//! 10.0.26200, rustc 1.95.0, over eighteen process runs in three conditions:
//!
//! ```text
//! condition                     E³ ns/ray        S³ ns/ray
//! quiet machine  (5 runs)      288.7 - 302.9   1376.4 - 1407.5
//! ordinary load (10 runs)      302.2 - 357.4   1407.9 - 1755.6
//! 24 spinners    (3 runs)      392.3 - 412.3   1770.6 - 1874.1
//! ```
//!
//! The quiet-machine medians round to the 300 and 1390 below. The three-way
//! split is the flake evidence: saturating 24 of 32 hardware threads with spin
//! loops moves the median by 1.43x and 1.36x, not by the 3.3x the frame
//! percentile spans on an idle machine. That is the property being bought by
//! gating a median of pure arithmetic instead of a frame statistic, and it is
//! the closest available stand-in for a shared runner.
//!
//! The budgets carry [`RUNNER_MARGIN`] over the quiet-machine numbers, which
//! leaves roughly 3x headroom over the worst measurement above. GitHub's
//! `ubuntu-latest` standard runner is a shared 4-vCPU VM whose single-core
//! throughput is not measured here; the rest of the margin covers that
//! throughput gap and a codegen difference between the two toolchains. It is an
//! assumption about relative single-core speed, not a measurement on the
//! runner, which is why the test prints its measured median unconditionally:
//! the number that would replace the assumption is in the log of every run,
//! green or red.
//!
//! What a margin this coarse buys is a gate on gross regressions, which is the
//! failure mode worth failing a build over: an allocation introduced into the
//! march inner loop, a `Space::exp` that stops being closed form, an SDF walk
//! that acquires a rescan. It does not catch ten-percent drift, and it is not
//! meant to.

use std::hint::black_box;
use std::time::Instant;

use loam_math::{EuclideanR3, Space, SphericalS3};
use loam_scene::{Scene, SceneNode};

/// `glam::Vec3` under the one name the facade crate can reach: it has no direct
/// glam dependency, and both Spaces below agree on this associated type.
type Point = <EuclideanR3 as Space>::Point;

/// Rays per pass is `GRID * GRID`. 64 keeps a pass in the low milliseconds, so
/// nine of them plus warmup stay well inside a test's patience while still
/// averaging over a mix of hit, miss and escape paths.
const GRID: usize = 64;
/// Odd, so the reported median is a pass that was actually observed.
const BATCHES: usize = 9;
/// Enough to fault in the scene, the grid loop and both Space impls, and to let
/// the branch predictors settle. Untimed.
const WARMUP_PASSES: usize = 2;

/// Ratio of the CI budget to the maintainer-machine median. See the module doc
/// for what it is covering and what it is assuming.
const RUNNER_MARGIN: f64 = 4.0;
const EUCLIDEAN_MEDIAN_NS: f64 = 300.0;
const SPHERICAL_MEDIAN_NS: f64 = 1390.0;

/// Hit fraction each pass must land inside. A scene that has gone invisible
/// marches one step and escapes, which is fast enough to pass any budget, and a
/// scene that has swallowed the eye hits on step one, which is faster still.
/// The band exists to reject both, so it is deliberately loose: the measured
/// fractions are 0.13 in E³ and 0.27 in S³, and pinning them exactly would
/// stake the gate on `asin` and `sin` agreeing bit for bit between the
/// maintainer's libm and the runner's, which no standard requires.
const MIN_HIT_FRACTION: f64 = 0.05;
const MAX_HIT_FRACTION: f64 = 0.60;

/// March parameters shaped after `loam_shader::GEODESIC_MARCH_KERNEL`, so the
/// timed loop does the kind of work the shipped kernel does. They are not
/// pinned to it: this is a stopwatch, not a parity check, and the kernel is
/// free to move without invalidating the budget.
const MAX_STEPS: usize = 128;
const STEP_SAFETY: f32 = 0.85;
const PROBE_EPS: f32 = 1e-4;

struct MarchConfig {
    /// Distance from the origin along +Z at which the grid's rays start.
    eye_distance: f32,
    /// Total geodesic arc a ray may accumulate before it is a miss.
    arc_budget: f32,
    /// Origin distance past which a ray leaves the chart and is a miss. The
    /// kernel needs this in S³, where the chart saturates at unit radius; in E³
    /// it is what bounds an escaping ray's step count.
    escape_radius: f32,
    hit_eps: f32,
    min_step: f32,
}

/// Two spheres joined by a smooth union plus a box, positioned so the grid
/// splits into hits, misses and chart escapes rather than resolving to one
/// branch. Half-spaces are excluded deliberately: `Primitive::eval` sentinels
/// them in curved Spaces, so the S³ pass would be timing a constant.
fn workload_scene(scale: f32) -> Scene {
    Scene::new(
        SceneNode::sphere(Point::new(-0.35 * scale, 0.0, 0.0), 0.30 * scale)
            .smooth_union(
                SceneNode::sphere(Point::new(0.35 * scale, 0.0, 0.0), 0.22 * scale),
                0.15 * scale,
            )
            .union(SceneNode::box_(Point::new(
                0.18 * scale,
                0.10 * scale,
                0.18 * scale,
            ))),
    )
}

/// One pass: `GRID * GRID` primary rays sphere-traced through `scene`, returning
/// the hit count and the sum of the hit distances. The sum is returned so the
/// caller can `black_box` a value the whole loop feeds, and the count so a scene
/// that has gone invisible cannot pass the budget by measuring nothing.
fn march_grid<S: Space<Point = Point, Vector = Point>>(
    space: &S,
    scene: &Scene,
    config: &MarchConfig,
) -> (u32, f32) {
    let eye = Point::new(0.0, 0.0, config.eye_distance);
    let mut hits = 0u32;
    let mut distance_sum = 0.0f32;

    for row in 0..GRID {
        for column in 0..GRID {
            // Half-pixel centers over [-0.5, 0.5]² against unit depth, so
            // 2·atan(0.5) ≈ 53 degrees across each axis.
            let sx = (column as f32 + 0.5) / GRID as f32 - 0.5;
            let sy = (row as f32 + 0.5) / GRID as f32 - 0.5;
            let direction = Point::new(sx, sy, -1.0).normalize();

            if let Some(t) = march(space, scene, config, eye, direction) {
                hits += 1;
                distance_sum += t;
            }
        }
    }
    (hits, distance_sum)
}

fn march<S: Space<Point = Point, Vector = Point>>(
    space: &S,
    scene: &Scene,
    config: &MarchConfig,
    eye: Point,
    direction: Point,
) -> Option<f32> {
    let mut p = eye;

    // The chart direction is not a unit tangent under the Riemannian metric, so
    // rescale it by the metric's local stretch before stepping. Probing with
    // exp/distance rather than reading a metric tensor keeps this to the `Space`
    // surface, which is the same trade the shipped kernel makes.
    let probed = space.exp(p, direction * PROBE_EPS);
    let stretch = space.distance(p, probed) / PROBE_EPS;
    let mut v = direction / stretch.max(1e-7);

    let mut arc = 0.0f32;
    for _ in 0..MAX_STEPS {
        if p.length() > config.escape_radius {
            return None;
        }
        let d = scene.eval(space, p);
        if d < config.hit_eps {
            return Some(arc);
        }
        if arc > config.arc_budget {
            return None;
        }
        let step = (d * STEP_SAFETY).max(config.min_step);
        let next = space.exp(p, v * step);
        let transported = space.parallel_transport(p, next, v);
        p = next;
        if transported.length_squared() > 1e-12 {
            v = transported;
        }
        arc += step;
    }
    None
}

/// Median nanoseconds per ray over [`BATCHES`] passes, plus the final pass's
/// hit count.
fn median_nanos_per_ray(mut pass: impl FnMut() -> (u32, f32)) -> (f64, u32) {
    for _ in 0..WARMUP_PASSES {
        black_box(pass());
    }

    let rays = (GRID * GRID) as f64;
    let mut batches = [0.0f64; BATCHES];
    let mut hits = 0u32;
    for batch in &mut batches {
        let start = Instant::now();
        let (pass_hits, distance_sum) = pass();
        *batch = start.elapsed().as_nanos() as f64 / rays;
        black_box(distance_sum);
        hits = pass_hits;
    }
    batches.sort_unstable_by(f64::total_cmp);
    (batches[BATCHES / 2], hits)
}

/// Fails when a fixed CPU march costs more than its stated budget.
///
/// Two Spaces, because they fail differently: the E³ pass is dominated by
/// `Scene::eval`, since flat `exp` is an add and flat transport is the
/// identity, while the S³ pass adds the closed-form geodesic and transport on
/// top of the same scene walk. A regression in the SDF walk moves both; a
/// regression in the curved metric moves only the second.
#[test]
#[ignore = "perf budget; needs --release, run in CI's own job"]
fn cpu_march_stays_inside_its_budget() {
    // Failing loudly beats reporting a debug-build timing against a release
    // budget, which would be a finding about the profile and not the code.
    if cfg!(debug_assertions) {
        panic!("the budget is a release-build number; re-run with --release");
    }

    // E³: no chart boundary, so the escape radius is what bounds an escaping
    // ray, and the eye sits outside a unit-scale scene.
    let euclidean_scene = workload_scene(1.0);
    let euclidean_config = MarchConfig {
        eye_distance: 2.0,
        arc_budget: 20.0,
        escape_radius: 8.0,
        hit_eps: 1e-3,
        min_step: 1e-4,
    };

    // S³: the chart saturates at unit radius, so the whole configuration lives
    // well inside it and the escape radius stays below the saturation shell.
    let spherical_scene = workload_scene(0.35);
    let spherical_config = MarchConfig {
        eye_distance: 0.80,
        arc_budget: 2.0,
        escape_radius: 0.92,
        hit_eps: 3.5e-4,
        min_step: 3.5e-5,
    };

    let (euclidean_ns, euclidean_hits) =
        median_nanos_per_ray(|| march_grid(&EuclideanR3, &euclidean_scene, &euclidean_config));
    let (spherical_ns, spherical_hits) =
        median_nanos_per_ray(|| march_grid(&SphericalS3, &spherical_scene, &spherical_config));

    let rays = (GRID * GRID) as u32;
    println!(
        "[perf_budget] rays/pass {rays}, batches {BATCHES}\n\
         [perf_budget] EuclideanR3 {euclidean_ns:7.1} ns/ray  budget {:7.1}  hits {euclidean_hits}\n\
         [perf_budget] SphericalS3 {spherical_ns:7.1} ns/ray  budget {:7.1}  hits {spherical_hits}",
        EUCLIDEAN_MEDIAN_NS * RUNNER_MARGIN,
        SPHERICAL_MEDIAN_NS * RUNNER_MARGIN,
    );

    assert_measured_the_work("EuclideanR3", euclidean_hits);
    assert_measured_the_work("SphericalS3", spherical_hits);

    assert!(
        euclidean_ns <= EUCLIDEAN_MEDIAN_NS * RUNNER_MARGIN,
        "EuclideanR3 march {euclidean_ns:.1} ns/ray is past its \
         {:.1} ns/ray budget",
        EUCLIDEAN_MEDIAN_NS * RUNNER_MARGIN,
    );
    assert!(
        spherical_ns <= SPHERICAL_MEDIAN_NS * RUNNER_MARGIN,
        "SphericalS3 march {spherical_ns:.1} ns/ray is past its \
         {:.1} ns/ray budget",
        SPHERICAL_MEDIAN_NS * RUNNER_MARGIN,
    );
}

fn assert_measured_the_work(label: &str, hits: u32) {
    let fraction = f64::from(hits) / (GRID * GRID) as f64;
    assert!(
        (MIN_HIT_FRACTION..=MAX_HIT_FRACTION).contains(&fraction),
        "{label} hit fraction {fraction:.3} is outside \
         [{MIN_HIT_FRACTION}, {MAX_HIT_FRACTION}]: the pass is no longer \
         marching the scene, so its timing is not a march timing",
    );
}
