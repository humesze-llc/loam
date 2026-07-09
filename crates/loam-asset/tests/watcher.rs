//! Integration tests for [`AssetWatcher`].
//!
//! These tests touch the real filesystem and notify backend. Event delivery has
//! platform-dependent latency, so each assertion uses a retry loop rather than a
//! fixed sleep.

use std::fs;
use std::path::Path;
use std::thread::sleep;
use std::time::{Duration, Instant};

use loam_asset::{AssetEvent, AssetEventKind, AssetWatcher};

/// Poll until `pred` matches one of the drained events or the timeout
/// elapses. Returns the matching event, or panics with a diagnostic.
fn wait_for<F>(watcher: &AssetWatcher, timeout: Duration, mut pred: F) -> AssetEvent
where
    F: FnMut(&AssetEvent) -> bool,
{
    let deadline = Instant::now() + timeout;
    let mut seen = Vec::new();
    while Instant::now() < deadline {
        for ev in watcher.poll() {
            if pred(&ev) {
                return ev;
            }
            seen.push(ev);
        }
        sleep(Duration::from_millis(25));
    }
    panic!("timeout waiting for event; saw: {seen:?}");
}

fn has_path(kind: AssetEventKind, target: &Path) -> impl Fn(&AssetEvent) -> bool + '_ {
    move |ev: &AssetEvent| {
        ev.kind == kind
            && ev
                .path
                .canonicalize()
                .ok()
                .zip(target.canonicalize().ok())
                .map(|(a, b)| a == b)
                .unwrap_or(false)
    }
}

#[test]
fn reports_created_file() {
    let dir = tempfile::tempdir().unwrap();
    let mut watcher = AssetWatcher::new().unwrap();
    watcher.watch(dir.path()).unwrap();

    let file = dir.path().join("hello.txt");
    fs::write(&file, b"hi").unwrap();

    wait_for(
        &watcher,
        Duration::from_secs(3),
        has_path(AssetEventKind::Created, &file),
    );
}

#[test]
fn reports_modified_file() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("shader.wgsl");
    fs::write(&file, b"v1").unwrap();

    let mut watcher = AssetWatcher::new().unwrap();
    watcher.watch(dir.path()).unwrap();
    // Drain any stray events from the initial setup.
    let _ = watcher.poll();

    fs::write(&file, b"v2").unwrap();

    wait_for(
        &watcher,
        Duration::from_secs(3),
        has_path(AssetEventKind::Modified, &file),
    );
}

/// `Removed` event fires end-to-end when a watched file is deleted. Pinned because the
/// merge logic in `watcher.rs::merge_kinds` special-cases Created/Modified bursts but a
/// delete should pass through cleanly.
///
/// Compares by file name + parent-directory containment (not `canonicalize`-equality)
/// because `canonicalize` fails on a removed path, and on Windows the notify-reported path
/// lacks the `\\?\` verbatim prefix that `canonicalize` would add.
#[test]
fn reports_removed_file() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("doomed.txt");
    fs::write(&file, b"goodbye").unwrap();
    let target_name = file.file_name().unwrap().to_owned();
    let dir_canonical = dir.path().canonicalize().unwrap();

    let mut watcher = AssetWatcher::new().unwrap();
    watcher.watch(dir.path()).unwrap();
    // Drain the Created event from the initial setup.
    let _ = watcher.poll();

    fs::remove_file(&file).unwrap();

    wait_for(&watcher, Duration::from_secs(3), |ev| {
        if ev.kind != AssetEventKind::Removed {
            return false;
        }
        // Match by file name; verify the parent is the watched dir
        // (canonicalise the *parent*; that path still exists).
        ev.path.file_name() == Some(&target_name)
            && ev
                .path
                .parent()
                .and_then(|p| p.canonicalize().ok())
                .map(|p| p == dir_canonical)
                .unwrap_or(false)
    });
}

#[test]
fn poll_is_empty_when_no_changes() {
    let dir = tempfile::tempdir().unwrap();
    let mut watcher = AssetWatcher::new().unwrap();
    watcher.watch(dir.path()).unwrap();
    sleep(Duration::from_millis(100));
    assert!(watcher.poll().is_empty());
}
