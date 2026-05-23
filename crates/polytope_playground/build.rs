//! Bake a short git hash + working-tree-dirty flag into the binary as
//! `BUILD_HASH` and `BUILD_DIRTY` env vars, surfaced at runtime via `env!`.
//!
//! Purpose: visible build-identifier label in the demo's HUD so a tester
//! reloading the page can confirm at-a-glance that they're looking at the
//! latest build (the wasm filename includes a content hash, but the browser
//! cache can still serve a stale page+script combination). Falls back to
//! `nogit` when not building from a git checkout (e.g. crates.io download
//! someday), so packaging in non-git contexts still works.

use std::process::Command;

fn main() {
    // Short 8-char hash. Enough to disambiguate any two commits we'd ever
    // realistically compare side-by-side. Falls back to "nogit" if git isn't
    // on PATH or this isn't a git checkout.
    let hash = Command::new("git")
        .args(["rev-parse", "--short=8", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "nogit".to_string());
    println!("cargo:rustc-env=BUILD_HASH={hash}");

    // Marker for uncommitted changes. Empty when working tree is clean;
    // "+dirty" otherwise. Easier to read than a pure bool in the HUD.
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);
    let dirty_marker = if dirty { "+dirty" } else { "" };
    println!("cargo:rustc-env=BUILD_DIRTY={dirty_marker}");

    // Re-run this build script when HEAD changes or files get staged/unstaged.
    // `.git/HEAD` covers branch + commit moves; `.git/index` covers staging.
    // Note: editing a tracked file without staging it won't re-trigger this
    // script via cargo's usual src-watch (cargo only watches .rs / Cargo.toml
    // by default), so a dirty-but-not-committed change may show stale.
    // Acceptable for now; for stricter freshness, also list specific files
    // or `cargo:rerun-if-changed=src` (which would trigger every src edit).
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/index");
}
