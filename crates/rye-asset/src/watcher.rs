//! Filesystem watcher built on [`notify`] (native targets only).
//!
//! Native backs [`AssetWatcher`] with `notify::RecommendedWatcher`; the
//! `wasm32` stub keeps the same API shape but never emits events. Both
//! impls share [`AssetEvent`], [`AssetEventKind`], and the per-poll
//! deduplication rule ([`merge_kinds`]).

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

    /// Watches filesystem paths and yields coalesced [`AssetEvent`]s on
    /// demand. [`poll`](Self::poll) drains non-blockingly and deduplicates
    /// per path per cycle, so an editor save burst collapses to one event
    /// per file. Not `Sync`: own one per app.
    pub struct AssetWatcher {
        watcher: RecommendedWatcher,
        rx: Receiver<notify::Result<notify::Event>>,
    }

    impl AssetWatcher {
        /// Start a new watcher. No paths are watched until [`watch`](Self::watch) is called.
        pub fn new() -> Result<Self> {
            let (tx, rx) = channel();
            let watcher = notify::recommended_watcher(move |res| {
                // A dropped receiver means the app is shutting down.
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
                    // `warn`, not `debug`: notify errors are platform-watcher
                    // failures that silently degrade hot-reload, so surface
                    // them when reloads stop working.
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

    /// No-op stub for wasm32: every call succeeds and `poll` returns empty,
    /// so consumers compile against the native API and skip hot-reload.
    pub struct AssetWatcher {
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
/// `Created` survives a later `Modified` because Windows `fs::write` on a
/// fresh file emits Create+Modify and consumers want "new file" distinct
/// from "changed file." Otherwise the later event wins, handling
/// save-by-atomic-replace correctly.
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
