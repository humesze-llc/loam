# Loam

**Loam is a deterministic geometry and physics substrate for games set in spaces that are not flat 3D: higher dimensions and curved manifolds, closed-form where the mathematics allows it, measured on consumer hardware.** The `Space` trait models smooth Riemannian manifolds. The implemented geometries are Euclidean space in two, three, and four dimensions; hyperbolic and spherical 3-space; and `BlendedSpace`, a variable-curvature Riemannian manifold kept as a validated reference implementation. Downstream systems are organized around capability traits rather than hard-coded geometry cases, so a new geometry is wired through rendering, interaction, and simulation by implementing the capabilities it actually supports.

Loam exists to build games that exploit exotic geometry for artistic purposes. A game engine (Rye) will grow on top of it as those games demand; each capability ships with the game that needed it. Named for the soil: this is the ground the games grow from.

## What you can run today

**`polytope_playground`** is the flagship demonstration and a spiritual successor to Marc ten Bosch's *4D Toys*. The six regular convex 4-polytopes rotate side-by-side under user-composed bivector velocities, and you inspect the three-dimensional cross-section through the available rendering modes: a raymarched signed-distance-field surface, the exact polychoral cross-section, and a wireframe overlay, with per-edge color cues for slice activity and signed-w depth. (The 120- and 600-cell render through the exact cross-section path; their SDF is a not-yet-correct approximation, so the raymarch surface is disabled for them.) The demo will grow over time, hosting additional 4D experiments inside the same harness.

```
cargo run --release -p polytope_playground               # native
cd crates/polytope_playground && trunk serve --release   # browser (local)
```

A hosted, click-to-run browser build is the next step.

`BlendedSpace<A, B, F>` (for example, a continuous Euclidean-to-hyperbolic manifold) is the variable-curvature primitive: a single Riemannian manifold whose metric interpolates continuously between two source geometries, rather than a portal or a camera trick. The mathematics is real and tested (RK4 geodesic integration, Gauss-Newton `log` shooting, and RK4 parallel transport, on CPU and GPU). It is kept as the oracle rather than a shipping path: numerical geodesics are the one tier of the geometry that does not run at gameplay frame rates on consumer hardware, and any future fast approximation will be validated against it with a measured error bound.

## What Loam provides

A geometry in Loam is anything that implements `Space`, the trait that surfaces smooth Riemannian structure at the math layer: `exp`, `log`, `distance`, parallel transport, and the isometry group. Capability traits (`WgslSpace`, `RasterizableSpace`, `SectionableSpace`, `PhysicsSpace`) carry everything a geometry can do downstream, and they are split because they change at different rates: the math trait is the most stable surface in the workspace, the shader ABI the least.

The four-dimensional physics is built on a first-party geometric algebra. `Bivector4` and `Rotor4` with invariant-decomposition exponential give the right rotation primitive for a space where the standard quaternion shortcut does not apply. The polychoral content covers the six regular convex 4-polytopes with exact topology (vertices, edges, and cells at unit circumradius) and an exact cross-section algorithm that renders cells as Lambert-shaded triangles. Collision detection extends the GJK and EPA algorithms into four dimensions; the contact manifold representation and the projected Gauss-Seidel solver follow the structure of the 3D literature, lifted carefully.

Geometric content can be authored as a typed `Scene` graph backed by signed distance fields, which emits WGSL on demand. The authoring model is deliberately space-aware without being space-hard-coded: a primitive declares its shape mathematically, and the emit chain combines that scene code with the selected space prelude. The same `Scene` can be rendered in Euclidean, hyperbolic, spherical, or `BlendedSpace` contexts when the relevant capability implementations exist, which is what makes the cross-geometry comparison demos possible. Apart from `glam` for SIMD vectors and matrices and `bytemuck` for zero-copy GPU upload, the math layer is first-party: the `Space` trait and its implementations, the 4D geometric algebra, the polytope topology, the cross-section algorithm, the SDFs, the WGSL emit chains, and the rigid-body solver are all written here.

Hardware-agnostic GPU access is through wgpu, which gives Loam Vulkan, DirectX 12, Metal, and WebGPU backends from a single codebase. The WebGPU path is not a port-later afterthought; the playground compiles to WebAssembly via `trunk` and runs on OffscreenCanvas in a dedicated worker, with the renderer's interop budget monitored per-frame. The browser path is what makes the demos shareable: something anyone can click and run, not software someone has to compile.

## Workspace

| Crate | Role |
|---|---|
| `loam-math` | `Space` traits and metrics, bivector/rotor geometric algebra, projections |
| `loam-shape` | geometry and topology data: the `Shape` enum, polytope topology, vertex/face generators |
| `loam-scene` | declarative SDF scene graph; emits WGSL on demand |
| `loam-render` | render nodes and GPU helpers (raymarch, rasterizer, line) |
| `loam-physics` | 4D rigid-body physics: geometric algebra, 4D GJK/EPA, contact solver |
| `loam-app` | native and wasm application shell, cameras and controllers |
| `polytope_playground`, `tesseract_demo` | demos, not API |

Supporting crates: `loam-egui` (UI), `loam-text` (HUD overlay), `loam-camera`, `loam-input`, `loam-time`, `loam-shader`, `loam-asset`, `loam-player`. The stable surfaces (`loam-math`, `loam-shape`) do not depend on the volatile ones (`loam-render`, the app shell); the dependency graph stays a tree.

## Correctness and reproducibility

Mathematical correctness is enforced by an invariant test suite, not by inspection of rendered output. Core primitives ship with tests that check properties no visualization would catch: Gauss-Bonnet on small geodesic triangles (angle defect / excess scaling with area), isometry preservation of distance, parallel-transport length preservation, nonzero-and-bounded holonomy from transport around a closed loop in curved space (the connection carries real curvature), curvature continuity across `BlendedSpace` transition zones, and exact graph-coloring invariants for polytope line-graphs. A primitive whose visualization looks correct but whose invariants fail does not ship.

Determinism is a property, not a mechanism: same binary, same inputs, same bits. The simulation path is built on fixed timestep, seeded randomness, deterministic iteration order in simulation-critical containers, and input routed through frame/tick boundaries rather than applied arbitrarily from event callbacks. GPU compute and presentation code sit outside that boundary and do not pretend otherwise.

## Intellectual lineage

Loam builds on prior work that has shaped both its mathematical content and its visualization choices.

Marc ten Bosch's *4D Toys*, and the body of research that produced it, is the direct conceptual ancestor of the polytope playground. The framing of four-dimensional objects as inhabitants of a four-dimensional space whose three-dimensional slices the viewer inspects, rather than as projections rendered onto a two-dimensional screen, is his. The polytope playground is a spiritual successor: regular convex 4-polytopes rather than arbitrary 4D solids, browser-deployable rather than installed, exact polychoral cross-sections rather than approximations, but the same fundamental commitment to letting a viewer move through a four-dimensional space and see what is there.

HackerPoet (CodeParade)'s HyperEngine and the Hyperbolica project (the commercial non-Euclidean game built on HyperEngine) demonstrated that non-Euclidean rendering is a viable interactive medium and made the architectural patterns inspectable. Zeno Rogue's HyperRogue and RogueViz cover breadth across hyperbolic geometries and projection models that any non-Euclidean engine has to engage with. Michael Walczyk's `polychora` showed what 4-polytope generation and slicing looks like in Rust specifically.

The mathematical content draws on the standard references in differential geometry, geometric algebra, and computer graphics:

- Coxeter, H. S. M. *Regular Polytopes*. The f-vector tables, edge lengths at unit circumradius, and the pentatope midpoint section.
- Foley, J. D., van Dam, A., et al. *Computer Graphics: Principles and Practice*, 2nd ed. §13.3.4 for the HSV-to-RGB conversion behind the wireframe color modes.
- Hestenes, D. *New Foundations for Classical Mechanics*, 2nd ed. §2.5 for the rotor-as-rotation-operator construction.
- Hestenes, D. & Sobczyk, G. *Clifford Algebra to Geometric Calculus*. The foundations behind `Bivector4` / `Rotor4` and the invariant-decomposition exponential.
- Knuth, D. *The Art of Computer Programming*, vol. 3, §6.4 for the golden-ratio hue spacing in the unique-edge palette.

The mature open-source 3D physics literature (Bullet, Jolt, ODE, Rapier in Rust) established the GJK/EPA/PGS pipeline that Loam's four-dimensional physics extends. The novelty in Loam is not in any individual algorithm but in the integration: a single substrate in which variable-curvature Riemannian manifolds, four-dimensional rigid-body physics, and polychoral cross-section rendering live together, deploy to a browser, and are exercised by a math-invariant test suite that catches violations no visualization would notice.

## Where this is going

Loam is built in the open for a specific ladder of games and experiments. The polytope playground grows toward interactive 4D physics. A set of 4D experiments (game of life, boids) will exercise GPU compute over the same machinery. A suika-style game about managing multiple dimensions is the first full game. The long-horizon target is a horror game set in multiply-connected, non-orientable spaces: quotients and gluings of the homogeneous geometries, which deliver impossible architecture as exact mathematics at closed-form cost. Which spaces admit real-time realization on consumer hardware gets mapped by measurement as the ladder climbs.

## Getting involved

I use AI coding tools, primarily Claude Code, heavily across this project. The mathematical correctness of every primitive is enforced by the invariant test suite regardless of whether code was AI-drafted or hand-written; the engineering decisions and the responsibility for what ships are mine. Contributions from collaborators who use AI are welcome on the same terms.

This is a single-maintainer project, so speculative pull requests are not the right way to help; reach out before starting anything. For longer conversations (engine direction, or the games and visualizations you would like to see built on this) get in touch at [humesze@proton.me](mailto:humesze@proton.me).

## License

Dual-licensed under MIT OR Apache-2.0. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
