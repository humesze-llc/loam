# Quality bar

The gate a change clears before it ships. Items 1-6 are mechanical and should
be enforced in CI; 7-9 are review judgment; 10-11 are release-only. A red
required item blocks the merge.

## Mechanical (CI-enforced)

1. **Format**: `cargo fmt --all --check` clean.
2. **Lints**: `cargo clippy --workspace --all-targets -- -D warnings` clean.
3. **Tests**: `cargo test --workspace` green. Math primitives ship with
   invariant tests (not output-pinning); new `WgslSpace` methods ship with
   CPU/GPU parity (probe) tests; boundary cases are explicit.
4. **Docs**: `cargo doc --workspace --no-deps` with no warnings (broken
   intra-doc links and private-item links fail).
5. **WebAssembly**: `cargo build -p loam-app --target wasm32-unknown-unknown`
   builds; the browser path is a shipping target, not an afterthought.
6. **GPU probes**: the shader-probe tests (`loam-shader`) pass: every WGSL
   `Space`/SDF kernel that has a CPU counterpart is checked for parity.

## Style (review)

7. **Comment discipline**: comments only when load-bearing (a non-obvious
   WHY, a named invariant, a citation). No narration, no over-explained math,
   no `TODO`/`FIXME`/stub in committed code. No em-dashes; no decorative ASCII
   arrows.
8. **Error policy**: `thiserror` at library boundaries where matching has a
   real callsite, `anyhow` at app boundaries; no `unwrap`/`expect`/`todo!` in
   library code.
9. **No magic numbers**: every constant has a named binding or an inline note
   tying it to a formula or a measured bound. No defensive abstraction (a trait
   with one impl, a flag with one consumer).

## Release (manual)

10. **Determinism**: a fixed-seed simulation replays bit-identically on the
    same architecture (the reproducibility contract); the determinism check is
    green.
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
