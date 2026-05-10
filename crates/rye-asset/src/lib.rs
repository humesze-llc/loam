//! `rye-asset`: filesystem watching and hot-reload events for Rye.
//!
//! The core type is [`AssetWatcher`]. Downstream crates (currently `rye-shader`,
//! eventually meshes / textures / scripts) poll it each frame and invalidate their
//! caches on the returned events.
//!
//! The watcher is deliberately minimal: no asset loading, no type-specific handling,
//! no ID indirection. It just tells you which paths changed. Consumers decide what
//! "changed" means for them.

mod watcher;

pub use watcher::{AssetEvent, AssetEventKind, AssetWatcher};
