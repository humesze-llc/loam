# Performance

Performance is a research deliverable, so frame budgets are backed by
measurement, not estimate. The engine already records per-section frame timing;
this page is how to turn that into a table, and where the current numbers live.

## How to measure

The runtime keeps a rolling-window trace of every frame, scoped by section
(`frame`, `sim-ticks`, `app-update`, `app-ui`, `app-record`, `ui-paint`,
`composite`, `present`). To read it:

1. `cargo run --release -p polytope_playground`
2. Exercise the scene you want to characterize (let it reach steady state).
3. Open the console (backtick) and run `trace summary` for the aggregate
   (p50 / p95 / p99 / max per section, p95-descending), or `trace` for the
   last frame.

Numbers are hardware-specific, so record the GPU, CPU, OS, and backend
alongside them. This is a measured artifact; do not fill it with estimates.

## Results

> Fill from a `trace summary` run on the maintainer's hardware. Until then this
> table is the shape, not the data.

**Machine:** _(GPU / CPU / OS / wgpu backend)_

Frame-time p50 / p95 in milliseconds, steady state.

| Scene | frame | sim-ticks | app-update | wireframe rebuild | render passes | present |
|---|---|---|---|---|---|---|
| 6-polytope row, rotating, wireframe | | | | | | |
| 600-cell, stereographic, single | | | | | | |
| 24-cell, SDF surface | | | | | | |
| 600-cell, exact cross-section | | | | | | |

A change that pushes a steady-state scene past its frame budget is a
regression even if every test passes; capture a `trace summary` before and
after any change to the hot path (the wireframe rebuild is the dominant
per-frame cost in the rotating-row scene).
