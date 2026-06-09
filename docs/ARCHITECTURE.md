# Architecture

How the crates fit together and which decisions ripple. For one-line per-crate
roles see the Workspace table in the [README](../README.md); this document is the
dependency structure and the trait boundaries behind it.

## Dependency tiers

The workspace is a DAG, layered so the stable surfaces never depend on the
volatile ones. Each crate depends only on tiers below it.

```
tier 0  rye-math   rye-input   rye-time   rye-text   rye-asset      (no rye deps)
tier 1  rye-shape    rye-camera    rye-player    rye-shader
tier 2  rye-shape ── rye-scene    rye-physics    rye-egui
tier 3  rye-render
tier 4  rye-app
tier 5  polytope_playground   tesseract_demo                        (demos, not API)
```

- `rye-math` is the root: the `Space` trait, metrics, the bivector/rotor
  geometric algebra, projections. Nothing in the workspace is below it.
- `rye-shape` (math + the geometry/topology data: `Shape`, polytope topology,
  the vertex/face generators) is the other stable surface. The two together are
  the foundation; promoting anything into them is a deliberate decision.
- `rye-render` depends on `rye-math`, `rye-shape`, `rye-time`, `rye-scene` and
  NOT on `rye-physics`: rendering must not pull in the simulation layer. Polytope
  topology lives in `rye-shape` precisely so the renderer can use it without the
  physics dependency.
- `rye-physics` is consumed by the demos, not by the engine shell (`rye-app`).
  4D rigid-body simulation is an application capability, not a render-path
  prerequisite.

The rule: a change in a low tier ripples upward, so the low tiers carry the
strictest review. A change isolated to `rye-render` or a demo is cheap.

## The capability-trait split

A geometry is anything that implements `Space` (the smooth-Riemannian core:
`exp`, `log`, `distance`, parallel transport). Everything a geometry can *do*
downstream is an opt-in capability trait, not a hard-coded geometry case:

- `WgslSpace`: emit the WGSL prelude to raymarch this space on the GPU.
- `RasterizableSpace`: project and tessellate edges/sections for the rasterizer.
- `SectionableSpace`: cross-section algorithm support.
- `PhysicsSpace`: rigid-body simulation in this space.

A new geometry is wired through the engine by implementing the capabilities it
actually supports, each with its own tests, rather than editing every renderer
and demo. The traits are split because they change at different rates: `Space`
is the most stable surface; the WGSL ABI is the least; they are not one trait
because they do not move together.

## One geometry, two rendering paths

The SDF raymarch path and the rasterizer path run on the *same* `Space` impls.
Code that presumes one rendering path is a defect: a geometry that implements
`WgslSpace` raymarches, one that implements `RasterizableSpace` rasterizes, and
several do both. This is what lets the playground show a polytope as a
raymarched surface, an exact cross-section, and a wireframe from one source of
truth.

## The determinism boundary

The math and simulation layers (`rye-math`, `rye-physics`, the sim-critical
parts of the demos) are held to a reproducibility contract: f32, single-threaded
fixed-step, deterministic iteration order, no fast-math. Same-binary
same-architecture replay is bit-identical. Code that runs only locally for
presentation (UI, render node setup, camera framing) is free of that constraint
and must not pretend to honor it. The boundary is explicit so a contributor
knows which side they are editing.

## Public surface vs internal

`rye-math` and `rye-shape` are the surfaces an external consumer would build on;
their `pub` items are contracts. The render/app crates are usable but still
moving pre-1.0. `polytope_playground` and `tesseract_demo` are demonstrations,
not API: depend on the engine crates, not on a demo.
