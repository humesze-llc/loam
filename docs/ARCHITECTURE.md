# Architecture

How the crates fit together and which decisions ripple. For one-line per-crate
roles see the Workspace table in the [README](../README.md); this document is the
dependency structure and the trait boundaries behind it.

## Dependency tiers

The workspace is a DAG, layered so the stable surfaces never depend on the
volatile ones. Each crate depends only on tiers below it.

```
tier 0  loam-math   loam-input   loam-time   loam-text   loam-asset      (no loam deps)
tier 1  loam-shape    loam-camera    loam-player    loam-shader
tier 2  loam-shape ── loam-scene    loam-physics    loam-egui
tier 3  loam-render
tier 4  loam-app
tier 5  polytope_playground   tesseract_demo                        (demos, not API)
```

- `loam-math` is the root: the `Space` trait, metrics, the bivector/rotor
  geometric algebra, projections. Nothing in the workspace is below it.
- `loam-shape` (math + the geometry/topology data: `Shape`, polytope topology,
  the vertex/face generators) is the other stable surface. The two together are
  the foundation; promoting anything into them is a deliberate decision.
- `loam-render` depends on `loam-math`, `loam-shape`, `loam-time`, `loam-scene` and
  NOT on `loam-physics`: rendering must not pull in the simulation layer. Polytope
  topology lives in `loam-shape` precisely so the renderer can use it without the
  physics dependency.
- `loam-physics` is consumed by the demos, not by the engine shell (`loam-app`).
  4D rigid-body simulation is an application capability, not a render-path
  prerequisite.

The rule: a change in a low tier ripples upward, so the low tiers carry the
strictest review. A change isolated to `loam-render` or a demo is cheap.

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

The math and simulation layers (`loam-math`, `loam-physics`, the sim-critical
parts of the demos) are held to a reproducibility contract stated as a property,
not a mechanism: same binary, same inputs, same bits. Concretely: f32,
fixed-step, deterministic iteration and reduction order, seeded randomness, no
fast-math. The sim is single-threaded today; parallelism is admissible only with
a fixed schedule and fixed reduction order, and only backed by a measurement.
Code that runs only for presentation (UI, render node setup, camera framing) and
GPU compute are outside the boundary and must not pretend to honor it. The
boundary is explicit so a contributor knows which side they are editing.

## Public surface vs internal

`loam-math` and `loam-shape` are the surfaces an external consumer would build on;
their `pub` items are contracts. The render/app crates are usable but still
moving pre-1.0. `polytope_playground` and `tesseract_demo` are demonstrations,
not API: depend on the engine crates, not on a demo.
