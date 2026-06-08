# Rye

**Rye** is a geometry engine for spaces where curvature and dimension are structural, not visual. Today's `Space` trait models smooth Riemannian manifolds: the implemented geometries include Euclidean space in two, three, and four dimensions; the three constant-curvature Thurston-family geometries; and `BlendedSpace`, a variable-curvature Riemannian manifold. The longer-term design goal is broader than Riemannian geometry: a capability hierarchy that can eventually admit Lorentzian manifolds, pseudo-Riemannian structure, fractal-dimensional spaces, and other exotic geometries without pretending they all share one identical interface.

The engine exists because the spaces, manifolds, and dimensions that show up in geometry textbooks deserve a computational realization that lets you simulate in them and see the results, on consumer hardware, in a browser. The frontier of which classes of spaces admit that realization is itself an open research question.

Named for Riemann.

## What you can run today

**`polytope_playground`** is the flagship demonstration. The six regular convex 4-polytopes rotate side-by-side under user-composed bivector velocities, and you inspect the three-dimensional cross-section through the available rendering modes: a raymarched signed-distance-field surface, the exact polychoral cross-section, and a wireframe overlay, with per-edge color cues for slice activity and signed-w depth. (The 120- and 600-cell render through the exact cross-section path; their SDF is a not-yet-correct approximation, so the raymarch surface is disabled for them.) The demo will grow over time, hosting additional 4D experiments inside the same harness.

```
cargo run --release -p polytope_playground               # native
cd crates/polytope_playground && trunk serve --release   # browser (local)
```

A hosted, click-to-run browser build is the next step.

`BlendedSpace<A, B, F>` (for example, a continuous Euclidean-to-hyperbolic manifold) is the first concrete variable-curvature primitive: a single Riemannian manifold whose metric interpolates continuously between two source geometries, rather than a portal or a camera trick. The framework and the math are real and tested (RK4 geodesic integration, Gauss-Newton `log` shooting, and RK4 parallel transport, on CPU and GPU), but it is a research direction more than a polished output, and is performance-limited at present; a standalone demo will land once it is solid enough to record.

## What Rye provides

A geometry in Rye today is anything that implements `Space`, the trait that surfaces smooth Riemannian structure at the math layer. The current implementations are the two-, three-, and four-dimensional Euclidean spaces, three-dimensional hyperbolic space, three-dimensional spherical space, and `BlendedSpace<A, B, F>` — a single Riemannian manifold whose metric interpolates continuously between two source geometries, supporting RK4 geodesic integration, Gauss-Newton shooting for the logarithm map, and RK4 parallel transport along the discovered connection. Downstream systems are organized around capability traits (`WgslSpace`, `RasterizableSpace`, `PhysicsSpace`) rather than hard-coded geometry cases, so a new geometry can be wired through rendering, interaction, and simulation by implementing the capabilities it actually supports instead of rewriting every subsystem.

The four-dimensional physics is built on a first-party geometric algebra. `Bivector4` and `Rotor4` with invariant-decomposition exponential give the right rotation primitive for a space where the standard quaternion shortcut does not apply. The polychoral content covers the six regular convex 4-polytopes with exact topology (vertices, edges, and cells at unit circumradius) and an exact cross-section algorithm that renders cells as Lambert-shaded triangles. Collision detection extends the GJK and EPA algorithms into four dimensions; the contact manifold representation and the projected Gauss-Seidel solver follow the structure of the 3D literature, lifted carefully.

Geometric content can be authored as a typed `Scene` graph backed by signed distance fields, which emits WGSL on demand. The authoring model is deliberately space-aware without being space-hard-coded: a primitive declares its shape mathematically, and the emit chain combines that scene code with the selected space prelude. The same `Scene` can be rendered in Euclidean, hyperbolic, spherical, or `BlendedSpace` contexts when the relevant capability implementations exist, which is what makes the cross-geometry comparison demos possible. Apart from `glam` for SIMD vectors and matrices and `bytemuck` for zero-copy GPU upload, the math layer is first-party: the `Space` trait and its implementations, the 4D geometric algebra, the polytope topology, the cross-section algorithm, the SDFs, the WGSL emit chains, and the rigid-body solver are all written here.

Hardware-agnostic GPU access is through wgpu, which gives Rye Vulkan, DirectX 12, Metal, and WebGPU backends from a single codebase. The WebGPU path is not a port-later afterthought; the playground compiles to WebAssembly via `trunk` and runs on OffscreenCanvas in a dedicated worker, with the renderer's interop budget monitored per-frame. That browser path is what makes the engine's goal possible: visualization work that functions as shareable research output rather than as software someone has to compile.

## Workspace

| Crate | Role |
|---|---|
| `rye-math` | `Space` traits and metrics, bivector/rotor geometric algebra, projections |
| `rye-shape` | geometry and topology data: the `Shape` enum, polytope topology, vertex/face generators |
| `rye-scene` | declarative SDF scene graph; emits WGSL on demand |
| `rye-render` | render nodes and GPU helpers (raymarch, rasterizer, line) |
| `rye-physics` | 4D rigid-body physics: geometric algebra, 4D GJK/EPA, contact solver |
| `rye-app` | native and wasm application shell, cameras and controllers |
| `polytope_playground`, `tesseract_demo` | demos, not API |

Supporting crates: `rye-egui` (UI), `rye-text` (HUD overlay), `rye-camera`, `rye-input`, `rye-time`, `rye-shader`, `rye-asset`, `rye-player`. The stable surfaces (`rye-math`, `rye-shape`) do not depend on the volatile ones (`rye-render`, the app shell); the dependency graph stays a tree.

## Correctness and reproducibility

Mathematical correctness is enforced by an invariant test suite, not by inspection of rendered output. Core primitives ship with tests that check properties no visualization would catch: Gauss-Bonnet on small geodesic triangles (angle defect / excess scaling with area), isometry preservation of distance, parallel-transport length preservation, nonzero-and-bounded holonomy from transport around a closed loop in curved space (the connection carries real curvature), curvature continuity across `BlendedSpace` transition zones, and exact graph-coloring invariants for polytope line-graphs. A primitive whose visualization looks correct but whose invariants fail does not ship.

Reproducibility is treated as a research commitment. The simulation code is structured toward replayable execution: fixed timestep, deterministic iteration order in simulation-critical containers, and input routed through frame/tick boundaries rather than applied arbitrarily from event callbacks. Same-binary same-architecture bit-identical replay is the target for the runtime, because reproducible numerical experiments and shareable computational results eventually need more than "it looked the same when I ran it twice."

## Intellectual lineage

Rye builds on prior work that has shaped both its mathematical content and its visualization choices.

Marc ten Bosch's *4D Toys*, and the body of research that produced it, is the direct conceptual ancestor of the polytope playground. The framing of four-dimensional objects as inhabitants of a four-dimensional space whose three-dimensional slices the viewer inspects, rather than as projections rendered onto a two-dimensional screen, is his. The polytope playground is a spiritual successor: regular convex 4-polytopes rather than arbitrary 4D solids, browser-deployable rather than installed, exact polychoral cross-sections rather than approximations, but the same fundamental commitment to letting a viewer move through a four-dimensional space and see what is there.

HackerPoet (CodeParade)'s HyperEngine and the Hyperbolica project (the commercial non-Euclidean game built on HyperEngine) demonstrated that non-Euclidean rendering is a viable interactive medium and made the architectural patterns inspectable. Zeno Rogue's HyperRogue and RogueViz cover breadth across hyperbolic geometries and projection models that any non-Euclidean engine has to engage with. Michael Walczyk's `polychora` showed what 4-polytope generation and slicing looks like in Rust specifically.

The mathematical content draws on the standard references in differential geometry, geometric algebra, and computer graphics:

- Coxeter, H. S. M. *Regular Polytopes*. The f-vector tables, edge lengths at unit circumradius, and the pentatope midpoint section.
- Foley, J. D., van Dam, A., et al. *Computer Graphics: Principles and Practice*, 2nd ed. §13.3.4 for the HSV-to-RGB conversion behind the wireframe color modes.
- Hestenes, D. *New Foundations for Classical Mechanics*, 2nd ed. §2.5 for the rotor-as-rotation-operator construction.
- Hestenes, D. & Sobczyk, G. *Clifford Algebra to Geometric Calculus*. The foundations behind `Bivector4` / `Rotor4` and the invariant-decomposition exponential.
- Knuth, D. *The Art of Computer Programming*, vol. 3, §6.4 for the golden-ratio hue spacing in the unique-edge palette.

The mature open-source 3D physics literature (Bullet, Jolt, ODE, Rapier in Rust) established the GJK/EPA/PGS pipeline that Rye's four-dimensional physics extends. The novelty in Rye is not in any individual algorithm but in the integration: a single engine in which variable-curvature Riemannian manifolds, four-dimensional rigid-body physics, and polychoral cross-section rendering live together, deploy to a browser, and are exercised by a math-invariant test suite that catches violations no visualization would notice.

## Research directions

Rye's long-term target is a hierarchy of computational spaces. The current trait surface assumes smooth Riemannian structure, which is correct for the geometries implemented today but inadequate for the directions the engine is moving toward. The longer-term thesis is that the architecture should refine as new classes of spaces are added: generalization of the metric-tensor assumption to admit pseudo-Riemannian and Lorentzian structure, relaxation of the smoothness assumption to admit fractal Hausdorff-dimensional spaces, and refinements within the metric-space family that the engine has not yet been asked to handle. The refactor happens when the mathematics demands it, not preemptively.

The engine exists to make certain kinds of work tractable. Computational physics in geometries where the standard frameworks do not apply: fluid and continuum dynamics on curved manifolds, rigid-body dynamics in higher dimensions, transport phenomena on spaces whose curvature varies. Visualization of mathematical objects that have historically been the domain of static figures and offline-rendered animation: polychora, hyperbolic tilings, manifolds of constant curvature, the boundary structure of fractal sets. Numerical experiments that benefit from interactive parameter exploration, where the cost of changing a coefficient and seeing the result needs to be a frame rather than a recompile. Pedagogy of geometric concepts that resist textbook illustration: an upcoming Flatland demo will let a viewer occupy a two-dimensional cross-section while a four-dimensional object passes through, on the same principle that lets the polytope playground work but turned toward the question of how a 2D observer would perceive a 3D world, the analogy for how a 3D observer must perceive a 4D one.

Where exactly the frontier of consumer-hardware feasibility lies for each of these is itself open. Some geometries that are mathematically natural turn out to be computationally tractable in ways that surprise; others resist real-time visualization for reasons that take serious effort to characterize. Mapping that frontier, finding the classes of spaces where a capability layer stops being well-defined and the classes where it is well-defined but no known algorithm is fast enough, is part of what the engine is for.

This section will be replaced, over time, by the work that has been done with Rye. Until then it is an invitation: if a direction here, or a direction not here, is one you would want to explore, get in touch.

## Getting involved

I use AI coding tools, primarily Claude Code, heavily across this project. The mathematical correctness of every primitive is enforced by the invariant test suite regardless of whether code was AI-drafted or hand-written; the engineering decisions and the responsibility for what ships are mine. Contributions from collaborators who use AI are welcome on the same terms.

This is a single-maintainer project, so speculative pull requests are not the right way to help; code review on unannounced work is not something I can manage at the volume that would invite. As the project opens to outside work I'll tag issues as up-for-grabs; until then, the best move is to reach out before starting anything. Particularly welcome are collaborators with formal background in differential geometry, pseudo-Riemannian geometry, geometric algebra, numerical methods on manifolds, or fractal geometry: there are primitives I have not built yet where domain expertise would land more cleanly than my engineering instincts alone.

For longer conversations (engine direction, research collaboration, the visualizations or simulations you would like the engine to support) get in touch at [humesze@proton.me](mailto:humesze@proton.me).

## License

Dual-licensed under MIT OR Apache-2.0. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
