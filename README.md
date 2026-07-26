# Loam

**A geometry and physics substrate for spaces that are not flat 3D: higher dimensions and curved manifolds, real-time, in Rust on wgpu.**

[![CI](https://github.com/humesze-llc/loam/actions/workflows/ci.yml/badge.svg)](https://github.com/humesze-llc/loam/actions/workflows/ci.yml)

<!-- HERO SLOT: assets/readme/hero.webp
     v1 capture: the playground's default polytope row under xw rotation,
     exact cross-sections shaded, ~1200x675, loop-clean (one full rotation
     period), dark background.
     Planned replacement: volumetric LOAM letters falling to the ground,
     polychora knocking them into 4D, then a scrub along w. Lands with the
     playground's interactive-4D-physics milestone. -->

What works today:

- **4D rigid-body physics.** First-party geometric algebra (`Bivector4`/`Rotor4` with invariant-decomposition exponential), GJK and EPA lifted to R⁴ (EPA validated against analytical penetration depths), persistent contact manifolds, a warm-started projected Gauss-Seidel solver.
- **Exact polychoral cross-sections.** All six regular convex 4-polytopes with exact topology at unit circumradius; sections are computed as geometry, not mesh approximations, alongside raymarched-SDF and wireframe render modes.
- **Geometry as a type parameter.** The same scene renders in Euclidean, hyperbolic, and spherical 3-space through one `Space` trait, and both render paths (SDF raymarch, rasterizer) run on the same implementations.
- **Determinism.** Simulation is bit-reproducible: same binary, same inputs, same bits. Pinned by tests that run in CI.

## Run it

```
cargo run --release -p polytope_playground               # native
cd crates/polytope_playground && trunk serve --release   # browser (local)
```

Stable Rust 1.95 or newer; any wgpu backend (Vulkan, DX12, Metal, WebGPU). The browser build needs [trunk](https://trunkrs.dev/). Controls live in the on-screen panels; backtick opens the debug console (`trace summary` prints per-section frame timings).

<!-- 600-CELL SLOT: assets/readme/600-cell.webp
     The 600-cell rotating through the exact cross-section path, wireframe
     overlay with signed-w depth colors. Same capture settings as hero. -->

## How it's built

- A geometry is anything implementing `Space` (exp, log, distance, parallel transport, isometries). Everything downstream is an opt-in capability trait (`WgslSpace`, `RasterizableSpace`, `SectionableSpace`, `PhysicsSpace`), so a new geometry is wired in by implementing what it actually supports.
- Scenes are typed SDF trees that emit WGSL on demand; the emit chain combines scene code with the selected space's shader prelude.
- Correctness is enforced by an invariant test suite, not by inspecting rendered output: Gauss-Bonnet on geodesic triangles, isometry invariance of distance, transport length preservation, loop holonomy.
- Apart from `glam` and `bytemuck`, the math layer is first-party. The crate-by-crate map lives in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- A variable-curvature `BlendedSpace` (a single Riemannian metric interpolating E³ to H³) exists as a validated reference implementation; its numerical geodesics do not yet meet a gameplay frame budget.

## Lineage

Marc ten Bosch's *4D Toys* is the direct conceptual ancestor of the polytope playground: four-dimensional objects as inhabitants of a 4D space whose 3D slices the viewer inspects. Zeno Rogue's *HyperRogue*, CodeParade's HyperEngine/*Hyperbolica*, and Michael Walczyk's `polychora` shaped the non-Euclidean and 4D rendering choices. Textbook citations (Coxeter, Hestenes and Sobczyk, do Carmo, Foley et al., Knuth) live in the rustdocs at each use site.

## Getting involved

I use AI coding tools, primarily Claude Code, heavily; invariant tests gate every primitive regardless of authorship, and responsibility for what ships is mine. Single-maintainer project: reach out before starting anything at [humesze@proton.me](mailto:humesze@proton.me).

## License

Dual-licensed under MIT OR Apache-2.0. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
