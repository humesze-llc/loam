//! 4D physics demo, drop one or two pentatopes (4-simplices) under
//! gravity and dump per-tick state to stdout.
//!
//! There's no rendering: `loam-render` is 3D-only, and a real 4D viewer
//! needs a hyperslice + slider (queued as a follow-up demo). For now
//! the example is the harness that drives [`loam_physics::EuclideanR4`]
//! through the integrator + collision pipeline so we can see numbers.
//!
//! Modes:
//!
//! - **Default**: one pentatope at `y = 5`, gravity along `−y`. Falls
//!   forever (no floor, no other body), exercising integrator + 4D
//!   orientation transport on a 4D rigid body.
//! - **`--floor`**: adds a static 4D half-space at `y = 0` (normal
//!   along `+y`). The pentatope falls onto this 4D ground and
//!   settles, exercising the polytope-vs-halfspace 4D narrowphase.
//! - **`--collide`**: same plus a second pentatope held static at the
//!   origin. The falling one drops onto it; their first contact (and
//!   any subsequent ones) prints a line. This exercises the full 4D
//!   collision pipeline: GJK in 4D, EPA penetration depth, the new
//!   pentatope-pentatope robustness fix.
//!
//! Other CLI flags:
//!
//! - `--steps N`: total sim ticks (default 600 ≈ 10 s at 60 Hz).
//! - `--print-every N`: print state every N ticks (default 60).

use glam::Vec4;
use loam_math::EuclideanR4;
use loam_physics::{
    euclidean_r4::{
        halfspace4_body_r4, pentatope_vertices, polytope_body_r4, register_default_narrowphase,
    },
    field::Gravity,
    World,
};

const DT: f32 = 1.0 / 60.0;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// One pentatope, no floor, no second body, pure free-fall.
    #[default]
    Drop,
    /// Pentatope plus a 4D `y ≥ 0` half-space floor.
    Floor,
    /// Pentatope plus a static second pentatope at the origin.
    Collide,
}

#[derive(Debug)]
struct Args {
    mode: Mode,
    steps: usize,
    print_every: usize,
}

impl Args {
    fn parse() -> Self {
        let mut args = Args {
            mode: Mode::Drop,
            steps: 600,
            print_every: 60,
        };
        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--floor" => args.mode = Mode::Floor,
                "--collide" => args.mode = Mode::Collide,
                "--steps" => {
                    args.steps = iter
                        .next()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(args.steps);
                }
                "--print-every" => {
                    args.print_every = iter
                        .next()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(args.print_every);
                }
                "-h" | "--help" => {
                    println!("loam 4D physics demo");
                    println!();
                    println!("Usage: cargo run --example physics4d [options]");
                    println!();
                    println!("Options:");
                    println!("  --floor                spawn a 4D half-space ground at y = 0");
                    println!(
                        "  --collide              spawn a second static pentatope at the origin"
                    );
                    println!("  --steps N              total sim ticks (default 600)");
                    println!("  --print-every N        print state every N ticks (default 60)");
                    std::process::exit(0);
                }
                other => {
                    eprintln!("warning: unknown flag {other:?}; pass --help to see options");
                }
            }
        }
        args
    }
}

fn fmt_v4(v: Vec4) -> String {
    format!(
        "({: >7.3}, {: >7.3}, {: >7.3}, {: >7.3})",
        v.x, v.y, v.z, v.w
    )
}

fn main() {
    let args = Args::parse();

    let mut world = World::new(EuclideanR4);
    register_default_narrowphase(&mut world.narrowphase);
    world.push_field(Box::new(Gravity::new(Vec4::new(0.0, -9.8, 0.0, 0.0))));

    // Mass 1, unit-circumradius pentatope: a 4D simplex with 5
    // tetrahedral cells. The falling body's index is always last, so
    // it's id 1 in --floor / --collide modes (anchor is id 0) and
    // id 0 in plain drop mode.
    match args.mode {
        Mode::Floor => {
            // 4D ground: half-space `{ p : p.y ≥ 0 }`. Outward normal
            // points along +y so the pentatope (above the ground)
            // stays out of the half-space.
            let _floor_id = world.push_body(halfspace4_body_r4(Vec4::Y, 0.0));
        }
        Mode::Collide => {
            // Static second pentatope at origin so the falling one
            // collides with it.
            let mut anchor = polytope_body_r4(Vec4::ZERO, Vec4::ZERO, pentatope_vertices(1.0), 1.0);
            anchor.inv_mass = 0.0;
            anchor.mass = 0.0;
            let _anchor_id = world.push_body(anchor);
        }
        Mode::Drop => {}
    }

    let falling_id = world.push_body(polytope_body_r4(
        Vec4::new(0.0, 5.0, 0.0, 0.0),
        Vec4::ZERO,
        pentatope_vertices(1.0),
        1.0,
    ));

    let mode_label = match args.mode {
        Mode::Drop => "drop",
        Mode::Floor => "floor",
        Mode::Collide => "collide",
    };
    println!("4D physics demo - {mode_label} mode");
    println!("  gravity:  {}", fmt_v4(Vec4::new(0.0, -9.8, 0.0, 0.0)));
    println!(
        "  bodies:   {} (falling pentatope is id {falling_id:?})",
        world.bodies.len()
    );
    println!("  dt:       1/60 s");
    println!(
        "  duration: {} ticks ({:.2} s)",
        args.steps,
        args.steps as f32 * DT
    );
    println!();
    println!(
        "{:>6} {:>8} {:<35} {:<35} {:>8}",
        "tick", "t (s)", "position", "velocity", "contacts"
    );
    println!("{}", "-".repeat(96));

    let mut last_contacts = 0;
    for tick in 0..args.steps {
        world.step(DT);

        let manifold_count = world.manifolds.len();

        // Print on the configured cadence, plus on the first frame
        // any contact appears so the user sees the impact moment.
        let new_contact = manifold_count > last_contacts;
        let on_cadence = tick % args.print_every == 0;
        if on_cadence || new_contact {
            let body = &world.bodies[falling_id];
            let marker = if new_contact { " *" } else { "" };
            println!(
                "{:>6} {:>8.3} {:<35} {:<35} {:>8}{marker}",
                tick,
                tick as f32 * DT,
                fmt_v4(body.position),
                fmt_v4(body.velocity),
                manifold_count,
            );
        }
        last_contacts = manifold_count;
    }

    println!();
    let body = &world.bodies[falling_id];
    println!("final state:");
    println!("  position: {}", fmt_v4(body.position));
    println!("  velocity: {}", fmt_v4(body.velocity));
    println!(
        "  |omega|²: {:.4} (bivector magnitude squared)",
        omega_mag_sq(body.angular_velocity)
    );
    println!("  manifolds at end: {}", world.manifolds.len());
}

fn omega_mag_sq(b: loam_math::Bivector4) -> f32 {
    b.xy * b.xy + b.xz * b.xz + b.xw * b.xw + b.yz * b.yz + b.yw * b.yw + b.zw * b.zw
}
