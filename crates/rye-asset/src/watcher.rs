//! Filesystem watcher built on [`notify`] (native targets only).
//!
//! Hot-reload is a desktop-only convenience: a `notify::RecommendedWatcher` tracks paths and
//! the app polls a channel each frame to drain events. On `wasm32` the browser has no
//! filesystem to watch, so this module ships a no-op stub that keeps the public API shape
//! (so consumers compile against both targets without `cfg`-littering their call sites) but
//! never emits events and never errors out on `watch` / `unwatch`.
//!
//! Both impls share the [`AssetEvent`] + [`AssetEventKind`] types and the per-poll
//! deduplication merge rule ([`merge_kinds`]); only the [`AssetWatcher`] struct differs.

use std::path::PathBuf;

/// A filesystem change observed by [`AssetWatcher`].
#[derive(Clone, Debug)]
pub struct AssetEvent {
    pub path: PathBuf,
    pub kind: AssetEventKind,
}

/// The nature of a filesystem change.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AssetEventKind {
    Created,
    Modified,
    Removed,
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::{merge_kinds, AssetEvent, AssetEventKind};
    use anyhow::{Context, Result};
    use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::mpsc::{channel, Receiver};

    /// Watches one or more filesystem paths and yields coalesced [`AssetEvent`]s on
    /// demand. Native impl backed by `notify::RecommendedWatcher`.
    ///
    /// Events arrive on a background thread managed by `notify`;
    /// [`poll`](Self::poll) drains the channel non-blockingly and deduplicates events
    /// per path within one poll cycle. Editor saves that produce a burst of raw events
    /// (remove temp -> create target -> modify) collapse to a single `Modified` or
    /// `Created` event per file. That's the usual shape a shader cache wants.
    ///
    /// Not `Sync`: own one per app. `Send` is fine.
    pub struct AssetWatcher {
        watcher: RecommendedWatcher,
        rx: Receiver<notify::Result<notify::Event>>,
    }

    impl AssetWatcher {
        /// Start a new watcher. No paths are watched until [`watch`](Self::watch) is called.
        pub fn new() -> Result<Self> {
            let (tx, rx) = channel();
            let watcher = notify::recommended_watcher(move |res| {
                // If the receiver has been dropped the app is shutting down;
                // silently drop the event.
                let _ = tx.send(res);
            })
            .context("creating notify watcher")?;
            Ok(Self { watcher, rx })
        }

        /// Begin watching `path` recursively.
        pub fn watch(&mut self, path: impl AsRef<Path>) -> Result<()> {
            let path = path.as_ref();
            self.watcher
                .watch(path, RecursiveMode::Recursive)
                .with_context(|| format!("watching {}", path.display()))?;
            Ok(())
        }

        /// Stop watching `path`.
        pub fn unwatch(&mut self, path: impl AsRef<Path>) -> Result<()> {
            let path = path.as_ref();
            self.watcher
                .unwatch(path)
                .with_context(|| format!("unwatching {}", path.display()))?;
            Ok(())
        }

        /// Drain all pending events, deduplicating per path.
        pub fn poll(&self) -> Vec<AssetEvent> {
            let mut latest: HashMap<PathBuf, AssetEventKind> = HashMap::new();

            while let Ok(res) = self.rx.try_recv() {
                let Ok(event) = res else {
                    // `warn` (not `debug`) because notify errors are usually platform-
                    // watcher failures (handle exhaustion on Windows, permission denied,
                    // dropped events) that silently degrade hot-reload. A user not
                    // seeing reloads should at least see something in stderr.
                    tracing::warn!("notify error: {:?}", res.err());
                    continue;
                };
                let kind = match event.kind {
                    EventKind::Create(_) => AssetEventKind::Created,
                    EventKind::Modify(_) => AssetEventKind::Modified,
                    EventKind::Remove(_) => AssetEventKind::Removed,
                    _ => continue,
                };
                for path in event.paths {
                    let merged = match latest.get(&path) {
                        Some(&old) => merge_kinds(old, kind),
                        None => kind,
                    };
                    latest.insert(path, merged);
                }
            }

            latest
                .into_iter()
                .map(|(path, kind)| AssetEvent { path, kind })
                .collect()
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use super::AssetEvent;
    use anyhow::Result;
    use std::path::Path;

    /// No-op stub for the wasm32 target. The browser has no filesystem to watch, so
    /// `AssetWatcher::new` succeeds, `watch` / `unwatch` succeed, and `poll` always
    /// returns an empty vector. Consumers compile against the same API as native and
    /// silently skip the hot-reload path.
    pub struct AssetWatcher {
        // Zero-sized field keeps the `Send` / non-`Sync` characteristics consistent
        // with the native impl (notify's watcher is also `Send + !Sync`).
        _private: (),
    }

    impl AssetWatcher {
        pub fn new() -> Result<Self> {
            Ok(Self { _private: () })
        }

        pub fn watch(&mut self, _path: impl AsRef<Path>) -> Result<()> {
            Ok(())
        }

        pub fn unwatch(&mut self, _path: impl AsRef<Path>) -> Result<()> {
            Ok(())
        }

        pub fn poll(&self) -> Vec<AssetEvent> {
            Vec::new()
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::AssetWatcher;
#[cfg(target_arch = "wasm32")]
pub use web::AssetWatcher;

/// Merge two events for the same path within a single poll cycle.
///
/// `Created` is preserved across a subsequent `Modified`, on Windows, `fs::write` on a fresh
/// file emits Create+Modify, and downstream consumers expect "new file" to look different
/// from "existing file changed." Otherwise the later event wins, which correctly handles
/// save-by-atomic-replace (Remove->Create->target exists).
///
/// Only used by the native watcher (the wasm stub never emits events), so we gate the
/// definition to silence dead-code warnings on wasm32.
#[cfg(not(target_arch = "wasm32"))]
fn merge_kinds(old: AssetEventKind, new: AssetEventKind) -> AssetEventKind {
    use AssetEventKind::*;
    match (old, new) {
        (Created, Modified) | (Modified, Created) => Created,
        (_, new) => new,
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn merge_created_modified_stays_created() {
        assert_eq!(
            merge_kinds(AssetEventKind::Created, AssetEventKind::Modified),
            AssetEventKind::Created
        );
        assert_eq!(
            merge_kinds(AssetEventKind::Modified, AssetEventKind::Created),
            AssetEventKind::Created
        );
    }

    #[test]
    fn merge_removed_wins_over_earlier_events() {
        assert_eq!(
            merge_kinds(AssetEventKind::Created, AssetEventKind::Removed),
            AssetEventKind::Removed
        );
        assert_eq!(
            merge_kinds(AssetEventKind::Modified, AssetEventKind::Removed),
            AssetEventKind::Removed
        );
    }

    #[test]
    fn merge_create_after_remove_wins() {
        assert_eq!(
            merge_kinds(AssetEventKind::Removed, AssetEventKind::Created),
            AssetEventKind::Created
        );
    }
}
