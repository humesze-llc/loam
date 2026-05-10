//! Filesystem watcher built on [`notify`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};

use anyhow::{Context, Result};
use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};

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

/// Watches one or more filesystem paths and yields coalesced [`AssetEvent`]s on demand.
///
/// Events arrive on a background thread managed by `notify`; [`poll`](Self::poll) drains
/// the channel non-blockingly and deduplicates events per path within one poll cycle. That
/// means editor saves that produce a burst of raw events (remove temp -> create target ->
/// modify) collapse to a single `Modified` or `Created` event per file. That's the usual
/// shape a shader cache wants.
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
    ///
    /// When the same path produces multiple events since the last poll, they are merged:
    /// `Created` beats `Modified` (a new file should look new, not merely modified),
    /// otherwise the later event wins. Events that aren't create/modify/remove (access,
    /// metadata, other) are dropped.
    pub fn poll(&self) -> Vec<AssetEvent> {
        let mut latest: HashMap<PathBuf, AssetEventKind> = HashMap::new();

        while let Ok(res) = self.rx.try_recv() {
            let Ok(event) = res else {
                // `warn` (not `debug`) because notify errors are usually platform-watcher
                // failures (handle exhaustion on Windows, permission denied, dropped events)
                // that silently degrade hot-reload. A user not seeing reloads should at
                // least see something in stderr.
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

/// Merge two events for the same path within a single poll cycle.
///
/// `Created` is preserved across a subsequent `Modified`, on Windows, `fs::write` on a
/// fresh file emits Create+Modify, and downstream consumers expect "new file" to look
/// different from "existing file changed." Otherwise the later event wins, which correctly
/// handles save-by-atomic-replace (Remove->Create->target exists).
fn merge_kinds(old: AssetEventKind, new: AssetEventKind) -> AssetEventKind {
    use AssetEventKind::*;
    match (old, new) {
        (Created, Modified) | (Modified, Created) => Created,
        (_, new) => new,
    }
}

#[cfg(test)]
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
