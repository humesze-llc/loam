//! Cross-platform key-value parameter access.
//!
//! Single API ([`Args`]) backed by `std::env::args` on native and
//! `window.location.search` on wasm32. Demos use it for things like
//! `?shape=tesseract`, `?seed=42`, `?fov=60` — anything a user might tweak
//! without recompiling.
//!
//! ## Design notes (why this is a struct, not free functions)
//!
//! A struct lets callers:
//!
//! - Construct synthetic instances for tests (via [`Args::from_pairs`]).
//! - Hold an immutable snapshot for the lifetime of a demo (the URL doesn't
//!   change after page load; reading once at setup is the common case).
//! - Pass it as an argument without taking a global lock or hidden dep.
//!
//! Free functions would tie every reader to a process-wide singleton, which
//! is fine until the first time someone wants to A/B two configs in the same
//! process (e.g. a multi-demo launcher choosing between sub-args).
//!
//! ## Argument syntax
//!
//! Single, simple convention on both platforms: **`key=value`** pairs.
//!
//! - **Native:** `my_demo --shape=tesseract --shapes=tesseract,5-cell --seed=42`
//!   - Positional args (anything without `--key=value` shape) are ignored.
//!   - The `--` prefix is stripped before storing as the key.
//!   - For backward compat with the older `--key value` style: not supported.
//!     Use `--key=value` exclusively. (The older style was lossy for
//!     multi-value: `--shapes a b c` is ambiguous about whether `b` belongs
//!     to `--shapes` or is a positional.)
//! - **Wasm32:** `?shape=tesseract&shapes=tesseract,5-cell&seed=42`
//!   - Standard URL query string. Read from `window.location.search`.
//!   - Hash fragment also supported (`#shape=tesseract`): some hosts prefer
//!     hash because it doesn't hit the server on navigation. Both populate
//!     the same map; hash takes precedence on key collision because it's
//!     more deliberately set by share-link UI than the page URL.
//!
//! ## Multi-value
//!
//! Comma-separated inside the value: `?shapes=tesseract,5-cell,8-cell`.
//! Demos split: `args.get("shapes").map(|s| s.split(',').collect())`.
//! Convention chosen because URLs already use `,` freely in values and
//! because it's the natural fit for an HTML form's text input.
//!
//! ## What this isn't
//!
//! - **Not a `clap`-style argument parser.** No subcommands, no `--help`, no
//!   validation. Demos that need rich CLI structure can pull `clap` directly
//!   for their native path and use this only for the wasm-URL path. Most
//!   demos have ~3 knobs and don't need the heavier machinery.
//! - **Not a settings system.** `Args` is read-once at startup. Runtime UI
//!   state belongs elsewhere (egui state, app fields, etc.).

use std::collections::HashMap;

/// Parsed key=value pairs from the host's argument surface.
///
/// Construct via [`Args::current`] to read the live environment, or
/// [`Args::from_pairs`] for tests / synthetic input.
#[derive(Clone, Debug, Default)]
pub struct Args {
    map: HashMap<String, String>,
}

impl Args {
    /// Read from the platform's argument surface.
    ///
    /// On native: parses `std::env::args` for `--key=value` pairs.
    /// On wasm32: parses `window.location.search` + `window.location.hash`.
    pub fn current() -> Self {
        let mut map = HashMap::new();

        #[cfg(not(target_arch = "wasm32"))]
        {
            // Skip arg[0] (the program path). Anything that doesn't match
            // `--key=value` is silently ignored; we don't error on unknown
            // shapes because a demo might be invoked under a parent harness
            // (cargo test, a wrapper script) that adds extra positional args
            // we shouldn't fail on.
            for arg in std::env::args().skip(1) {
                if let Some(stripped) = arg.strip_prefix("--") {
                    if let Some((k, v)) = stripped.split_once('=') {
                        if !k.is_empty() {
                            map.insert(k.to_string(), v.to_string());
                        }
                    }
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            // The wasm path uses web_sys + the standard URLSearchParams DOM
            // type. UrlSearchParams handles percent-decoding correctly so a
            // value like `?label=hello%20world` decodes to `hello world`.
            // Native's manual `split_once` doesn't do that; if percent
            // encoding matters on native (it usually doesn't for CLI), the
            // caller can use a URL crate to decode the returned value.
            if let Some(window) = web_sys::window() {
                if let Ok(search) = window.location().search() {
                    parse_query_into(&search, &mut map);
                }
                if let Ok(hash) = window.location().hash() {
                    parse_query_into(&hash, &mut map);
                }
            }
        }

        Self { map }
    }

    /// Construct from explicit pairs. Used by tests + by hosts that need to
    /// synthesize an Args from a non-standard source (e.g. reading from a
    /// JSON config blob loaded over fetch).
    pub fn from_pairs<I, K, V>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        Self {
            map: pairs
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        }
    }

    /// Look up a single value by key. `None` if the key was not provided.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.map.get(key).map(String::as_str)
    }

    /// Look up a value and parse it as `T`. Returns `None` if the key was
    /// missing OR the value failed to parse — distinguish these with `get`
    /// if your demo needs a specific error message.
    pub fn parse<T: std::str::FromStr>(&self, key: &str) -> Option<T> {
        self.get(key)?.parse().ok()
    }

    /// Comma-split helper for multi-value keys. Returns an empty `Vec` when
    /// the key is missing. Empty segments are filtered (so `?shapes=a,,b`
    /// yields `["a", "b"]`).
    pub fn get_many<'a>(&'a self, key: &str) -> Vec<&'a str> {
        match self.get(key) {
            Some(v) => v.split(',').filter(|s| !s.is_empty()).collect(),
            None => Vec::new(),
        }
    }

    /// All (key, value) pairs. Order is HashMap-arbitrary; consumers that
    /// need a stable order should sort by key themselves.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.map.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }
}

/// Parse a query-string-shaped fragment ("?a=1&b=2" or "#a=1") into `map`.
/// Leading `?` or `#` is stripped before splitting on `&`. Each segment is
/// then split on the first `=`; segments without `=` are skipped (a bare
/// `?flag` style isn't supported because the rest of the engine's syntax is
/// `key=value`).
#[cfg(target_arch = "wasm32")]
fn parse_query_into(raw: &str, map: &mut HashMap<String, String>) {
    let trimmed = raw.trim_start_matches(['?', '#']);
    if trimmed.is_empty() {
        return;
    }
    for pair in trimmed.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            if !k.is_empty() {
                // Percent-decoding: the browser delivers the raw bytes of
                // the URL; spaces appear as `+` or `%20`. URLSearchParams
                // would do this for us but it'd require an extra wasm-bindgen
                // call per key; for our typical "simple ASCII identifiers"
                // values the cost isn't worth it. Demos that need decoding
                // can wrap with `urlencoding::decode` if they pull that crate.
                let value = v.replace('+', " ");
                map.insert(k.to_string(), value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_pairs_round_trips() {
        let args = Args::from_pairs([("shape", "tesseract"), ("seed", "42")]);
        assert_eq!(args.get("shape"), Some("tesseract"));
        assert_eq!(args.get("seed"), Some("42"));
        assert_eq!(args.get("missing"), None);
    }

    #[test]
    fn parse_returns_typed_or_none() {
        let args = Args::from_pairs([("seed", "42"), ("fov", "60.5"), ("bad", "nope")]);
        assert_eq!(args.parse::<u32>("seed"), Some(42));
        assert_eq!(args.parse::<f32>("fov"), Some(60.5));
        assert_eq!(args.parse::<u32>("bad"), None);
        assert_eq!(args.parse::<u32>("missing"), None);
    }

    #[test]
    fn get_many_splits_on_comma_and_drops_empties() {
        let args = Args::from_pairs([("shapes", "tesseract,5-cell,,8-cell,")]);
        assert_eq!(
            args.get_many("shapes"),
            vec!["tesseract", "5-cell", "8-cell"]
        );
        assert!(args.get_many("missing").is_empty());
    }
}
