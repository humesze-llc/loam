# Quality bar

The gate a change clears before it ships. Items 1-7 are mechanical and run in
CI (`.github/workflows/ci.yml`); 8-10 are review judgment; 11 is release-only.
A red required item blocks the merge.

## Mechanical (CI-enforced)

1. **Format**: `cargo fmt --all --check` clean.
2. **Lints**: `cargo clippy --workspace --all-targets -- -D warnings` clean.
3. **Tests**: `cargo test --workspace --all-targets` green. Math primitives
   ship with invariant tests (not output-pinning); new `WgslSpace` methods ship
   with CPU/GPU parity (probe) tests; boundary cases are explicit.
4. **Docs**: `cargo doc --no-deps --workspace` under `RUSTDOCFLAGS=-D warnings`
   (broken intra-doc links and missing code-block languages fail).
5. **WebAssembly**: the browser demo builds on wasm32:
   `cargo build --target wasm32-unknown-unknown -p polytope_playground --no-default-features`.
   `--no-default-features` drops the native-only `capture` feature, matching
   what `index.html` tells Trunk to run.
6. **GPU probes**: `cargo test --workspace gpu_probe -- --include-ignored`
   green on a software Vulkan adapter. The selector means "needs a real
   adapter"; CPU/GPU parity for every WGSL `Space`/SDF kernel with a CPU
   counterpart is the largest member, alongside device-bound smoke tests that
   only check a chain still assembles.
7. **Determinism**: `cargo test -p loam-physics determinism` green: a
   fixed-seed simulation replays bit-identically on the same architecture (the
   reproducibility contract).

## Style (review)

8. **Comment discipline**: comments only when load-bearing (a non-obvious
   WHY, a named invariant, a citation). No narration, no over-explained math,
   no `TODO`/`FIXME`/stub in committed code. No em-dashes; no decorative ASCII
   arrows.
9. **Error policy**: `thiserror` at library boundaries where matching has a
   real callsite, `anyhow` at app boundaries; no `unwrap`/`expect`/`todo!` in
   library code.
10. **No magic numbers**: every constant has a named binding or an inline note
    tying it to a formula or a measured bound. No defensive abstraction (a
    trait with one impl, a flag with one consumer).

## Release (manual)

11. **Public surface current**: `README.md` describes what actually ships (no
    stale run commands, no aspirational claims stated as present tense); the
    rustdoc and any hosted demo reflect the current branch; a representative
    screenshot/gallery exists for visual features.

## Conventions

Code is ground truth: read it before recommending against it. Determinism is
Tier 0 in the math and simulation layers: same binary, same inputs, same bits
(f32, fixed-step, deterministic order, no fast-math). Stable surfaces
(`loam-math`, `loam-shape`) do not depend on volatile
ones (`loam-render`, app shell). Cite the public reference for any non-obvious
formula or named algorithm at the use site.
