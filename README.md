# Rye

**Rye** is a game engine where non-Euclidean geometry is a primitive, not a camera trick.

It's a personal project I started because I wanted to build games where the geometry itself is the gameplay rather than the backdrop, and you can't really do that in Bevy or Godot without faking it. A sphere physically rolling along a hyperbolic geodesic. A 4D rigid body colliding via its 3D cross-section. A manifold whose curvature varies smoothly between two source geometries so a player flies between them instead of teleporting.

Where I'd like it to go is a research platform for computational physics over Riemannian geometric structure, where every new abstract primitive ships alongside the engineering pipeline that makes it run on consumer hardware. Right now the engine is in proof-of-concept territory, and the first real game built on it is in active development in parallel.

**Why open source.** As far as I've been able to find, there are no open-source resources for computational physics in 4D or curved geometry, in Rust or any other language. The math is there in the textbooks; it's mostly just been waiting to be lifted into algorithms. Rye is meant to do that lift in the open, with an asset format (signed distance fields) that's space-agnostic so contributors can add their own `Space` implementations or `Primitive` implementations without having to touch engine internals. I don't have a PhD in differential geometry or computational physics, and I'll need help from people who do - particularly on the harder primitives I haven't built yet, and on optimization once the surface area gets bigger.

Named for Riemann. Also, I like whiskey.

## What the project commits to

Two halves that have to stay in lockstep:

- **Math.** Every new geometric primitive (`Space` trait, `BlendedSpace`, `Bivector4` rotors, future pseudo-Riemannian and fractal-dimensional spaces) ships with mathematical-invariant tests that catch wrongness no game would notice: Gauss-Bonnet on small triangles, isometry preservation of distance, parallel-transport length preservation, holonomy round-trip closure, curvature continuity across transition zones.
- **Engineering.** Performance is treated as a research deliverable, not a postscript. Every primitive that lands ships with a real-time WGSL implementation, designed and tuned for consumer GPU hardware: the math has to survive the trip from textbook to GPU shader without losing its rigor or its frame budget, and the rigid-body physics has to settle into stable stacks the way you'd expect. Benchmark coverage grows alongside the primitives; today's evidence is the example demos and the visual-correctness tests.

Pure research without performance won't ship games, and performance without rigor isn't a research platform; both halves matter equally and the project doesn't move forward without both.

If you'd like to follow along, collaborate, or hear about the game ahead of release, get in touch at [humesze@proton.me](mailto:humesze@proton.me).

## What's shipping

The engine is pre-1.0. The `Space` and `WgslSpace` trait surfaces are stable and additive-only; everything else can still move. Recent landings, in order of how proud I am of them:

- `BlendedSpace<A, B, F>`, the first concrete second-thesis primitive. Not portals between spaces, but a single Riemannian manifold whose curvature interpolates continuously between two source geometries. RK4 geodesic integration, Gauss-Newton `log` shooting, and RK4 parallel transport, on both CPU and GPU.
- 4D physics. `Bivector4` and `Rotor4` with invariant-decomposition exponential, `EuclideanR4` `PhysicsSpace`, 4D GJK and EPA, persistent contact manifolds, and a projected Gauss-Seidel solver for tall stacks.
- 4D polytope rendering. `Hyperslice4DNode` with face-hyperplane SDFs for the 5-cell, tesseract, 16-cell, and 24-cell, plus a Rotor4 inverse-sandwich path so bodies can rotate arbitrarily in 4D and you watch the cross-section morph.
- `rye-app`, a thin App-trait framework with `Camera<S>` and `OrbitController<S>` / `FirstPersonController<S>` generic over `Space`.
- `rye-text`, an `ab_glyph`-backed HUD overlay (rasterization, no text shaping) sufficient for game UIs and HUDs while a richer `glyphon`-based path waits on a workspace `wgpu` bump.

## Examples

`cargo run --release --example <name>` from the workspace root. They double as demos and as integration tests for the relevant subsystems.

| Example | Demonstrates |
|---|---|
| `blended` | First-class `BlendedSpace<E³, H³>` demo. Checkerboard floor and spheres under a smooth metric transition. Visibly distinct E³ side, transition zone, and Poincaré-chart H³ side in one view. WASD fly-through, drag to look. |
| `rotate_polytopes` | Interactive 4D-rotation demo over `Hyperslice4DNode`. All six convex regular polychora (pentatope through 600-cell) rotating under user-composed `Bivector4` velocity (toggle planes 1..6, watch slice signatures morph). Drag-and-drop shape-row reordering, a composer for sequenced rotor terms, and a live `exp(...)` formula popup. |
| `pentatope_slice` | Live-physics 4D viewer. A pentatope falls onto a 4D floor and the 3D viewport renders its cross-section at user-controlled `w₀` (↑/↓ to scrub). |
| `hypersphere` | Drop up to 32 4D balls onto a 4D floor under gravity. Two modes: **slice** (3D cross-section at `w₀`) and **ghost** (`G` toggle; full 4D extent as translucent volume). |
| `fractal_app` | Mandelbulb scene on the `rye-app` framework with `Camera<S>` and `OrbitController<S>`. Reference for what an app looks like on the framework. |
| `fractal` | Original Mandelbulb ray-marcher with WGSL hot reload and orbit camera; the legacy raw-winit version. APNG and GIF capture via `--capture-apng` / `--capture-gif`. |
| `geodesic_spheres` | Three spheres rendered in E³, H³, and S³ via the `WgslSpace` prelude swap. Same scene, different metric. |
| `lattice` | The same typed `Scene` rendered in all three Spaces side-by-side; no per-Space WGSL written by the caller. |
| `corridor` | Extruded corridor stress-testing the typed primitive emit path under booleans and repeats. |
| `physics3d` | Mixed spheres and oriented boxes falling onto a floor. Exercises the full GJK + EPA + manifold + PGS path in 3D. |
| `physics4d` | Headless 4D demo (`--floor` / `--collide` / free-fall). 4D GJK + EPA + orientation transport. Prints per-tick state to stdout. |
| `physics2d` | 2D rigid bodies under gravity with polygon SAT, friction, and the manifold + PGS solver. The reference for stack settling. |
| `text_smoke` | Smoke test for `rye-text` (ab_glyph-backed screen-space HUD overlay). Renders ASCII labels at multiple sizes and colors over a clear-color background. |
| `sysinfo` | Enumerates every wgpu adapter the host can see, with backend, device type, driver, vendor/device IDs, and selected limits. Useful for bug reports and first-run sanity checks. |

`examples/capture.rs` is a frame-capture utility module (APNG / GIF), pulled in by demos that opt into recording rather than run as a standalone example.

## Why not just use Bevy or Godot?

This is the most common question I receive from friends I show the project to. The reality is, Bevy is the best ergonomic Rust game engine going, and Godot is the best general-purpose 2D/3D engine for indie work. For 99% of projects, you should reach for one of those.

Equally true: you can't make geodesic-faithful hyperbolic, 4D, or variable-metric games in Bevy or Godot, without camera or postprocessing tricks: both assume Euclidean ℝ³ at the transform layer, and both treat curved geometry as something to fake with shader effects rather than something the simulation natively understands. Rye exists to handle these as first-class primitives, and the design choices below all flow from that one constraint:

- **Transform is Euclidean elsewhere; here it's a `Space` trait.** Bevy's `Transform` and camera math assume ℝ³, which makes a hyperbolic metric as a first-class citizen invasive to retrofit. In Rye the `Space` trait at the math layer is the substrate, and Euclidean ℝ³ is one implementation among several (E², E³, E⁴, H³, S³, `BlendedSpace<A, B, F>` so far).
- **No ECS.** Bevy is built on an entity-component-system substrate where everything composes through component queries, which is the right model when entity composition is the primary axis of complexity. For Rye the geometry primitive is the substrate instead, and the thesis lives in the `Space` trait and what every other crate routes through it; an ECS layer over that wouldn't pay rent. Library-style composition (each engine concern is a normal callable Rust crate, and the app constructs and calls them explicitly) handles the scope I care about while keeping the math primitives as the visible spine.
- **Determinism, scoped honestly.** Bevy's parallel scheduler makes lockstep, replay, and rollback netcode hard because task execution order is non-deterministic across runs. Rye's simulation runs on a fixed timestep with a deterministic task graph: tasks that don't share state run in parallel via Rayon, but outputs are reduced in a fixed order each tick. IO (asset loading, networking) is async via Tokio and feeds into per-tick input queues at tick boundaries, never touching sim state directly. The scope of the determinism claim, in three layers: same-binary same-architecture replay (bit-identical state given the same seed and inputs) is the design target and the layer that seeded reproducibility and single-machine debugging need. Same-architecture cross-machine determinism (lockstep multiplayer between players on the same arch) is the engineering target as netcode demands it. Cross-architecture (x86 ↔ ARM) determinism is out of scope for v0; transcendental f32 routines vary per platform-libm and matching them would mean shipping a soft-float layer no current consumer needs.
- **Shader hot reload from day one.** For shader-heavy work like ray marching, editing a WGSL file should reflect on screen immediately, so it's baked into `rye-shader` rather than reached for later.
- **Ergonomics is the open problem, not the win.** Bevy and Godot have years of editor and tooling investment Rye can't match, and that's fine, because Rye isn't competing with them on developer experience for Euclidean games. What it does need is ergonomic tooling for visualizing and authoring scenes in curved and higher-dimensional geometry, which is genuinely new territory; there's no inherited best practice for "scene editor for 4D" or "asset pipeline for variable-curvature manifolds." Figuring that out is part of the research, not a stable destination this README can promise yet.

## Building and running

Hardware-agnostic via wgpu. Vulkan, DX12, Metal, WebGPU.

Stable Rust toolchain (pinned via `rust-toolchain.toml`). Standard `cargo build --workspace --all-targets` to compile everything; `cargo test --workspace` to run the test suite (~200 tests including math invariants and naga shader-validation probes).

## Getting involved

I use AI coding tools (primarily Claude Code) heavily across this project. The mathematical correctness of every primitive is enforced by the invariant test suite regardless of whether code was AI-drafted or hand-written; the engineering decisions and the responsibility for what ships are mine. PRs from contributors who use AI are welcome on the same terms.

This is a single-maintainer project right now, so speculative PRs aren't the right way to help; I can't manage code review on unannounced work. There are tagged GitHub issues marked as up-for-grabs that I'd be glad to have collaborators on. Pick one and reach out before starting.

For longer conversations (engine direction, the game in development, research collaboration, anything that doesn't fit in an issue thread) get in touch at [humesze@proton.me](mailto:humesze@proton.me).

## License

Rye is dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option. Both licenses require attribution; please retain the copyright notice and license text in substantial portions of the work.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual-licensed as above, without any additional terms or conditions.
