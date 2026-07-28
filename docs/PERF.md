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

**Machine:** 13th Gen Intel Core i9-13980HX, Windows 11 Pro 10.0.26200, Vulkan.
**Scene:** all valid SDF objects including smooth solids, xy rotation, 120
frames. Excludes the 120-cell and 600-cell, which are the expensive cases and
are not represented here.

| section | p50 | p95 | max |
|---|---|---|---|
| between-frames | 4.13ms | 13.35ms | 16.69ms |
| frame | 3.96ms | 13.18ms | 16.58ms |
| idle | 110.5us | 2.18ms | 12.09ms |
| gpu-total | 802.8us | 1.59ms | 2.01ms |
| app-ui | 224.1us | 373.1us | 644.4us |
| ui-paint | 146.7us | 220.5us | 802.9us |
| app-record | 131.6us | 202.6us | 313.9us |
| pp-sdf | 94.3us | 149.3us | 232.1us |
| present | 42.5us | 61.3us | 118.1us |
| app-update | 15.9us | 42.0us | 164.7us |
| hot-reload | 700ns | 1.1us | 1.5us |
| sim-ticks | 200ns | 400ns | 900ns |

Three things this says, all of which contradict where the roadmap was pointing:

**The SDF is not the bottleneck.** `pp-sdf` is 94us of a 3.96ms median frame,
2.4%. Making it "blazing fast" would buy at most that. `gpu-total` is 803us
(20%) and the egui pair, `app-ui` plus `ui-paint`, is 371us (9.4%): the UI costs
four times what the SDF does.

**The tail is the cost, not the median.** p50 is 3.96ms (~250fps) while p95 is
13.18ms, a 3.3x spread, and `idle` moves with it (110us p50, 12.09ms max). Nine
milliseconds on the worst 5% of frames dwarfs every section in the table. Find
that before optimizing anything in it.

**`sim-ticks` at 200ns is the physics layer confirming it is inert.** Nothing
applies an impulse, so `World::step` returns at the at-rest check every frame.
Any physics performance number taken today measures an early return.

A change that pushes a steady-state scene past its frame budget is a
regression even if every test passes; capture a `trace summary` before and
after any change to the hot path.
