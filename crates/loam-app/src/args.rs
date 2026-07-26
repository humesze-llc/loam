//! Cross-platform `key=value` parameter access.
//!
//! Single API ([`Args`]) backed by `std::env::args` on native and
//! `window.location.search` on wasm32, for recompile-free knobs like
//! `?shape=tesseract`, `?seed=42`.
//!
//! A struct (not free functions) so callers can synthesize test instances
//! ([`Args::from_pairs`]) and A/B two configs in one process.
//!
//! ## Syntax
//!
//! - **Native:** `--key=value` pairs; positional args ignored, `--` stripped.
//!   The older `--key value` style is unsupported (ambiguous for multi-value).
//! - **Wasm32:** `?key=value&...` query string plus `#key=value` hash; both
//!   populate the same map, hash winning on collision (share-link UI sets it
//!   more deliberately than the page URL).
//! - **Multi-value:** comma-separated, split by [`Args::get_many`].
//!
//! Not a `clap`-style parser (no subcommands/help/validation) and not a
//! settings system (read-once at startup).

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
            // Skip arg[0]. Non-`--key=value` args are ignored rather than
            // errored: a parent harness (cargo test, a wrapper) may add
            // positionals we shouldn't fail on.
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
            if let Some(window) = web_sys::window() {
                if let Ok(search) = window.location().search() {
                    parse_query_into(&search, &mut map);
                }
                if let Ok(hash) = window.location().hash() {
                    parse_query_into(&hash, &mut map);
                }
            }
            // Workers have no `window`; the init message forwards the page's
            // query (see set_query_override).
            QUERY_OVERRIDE.with(|q| {
                if let Some((search, hash)) = q.borrow().as_ref() {
                    parse_query_into(search, &mut map);
                    parse_query_into(hash, &mut map);
                }
            });
        }

        Self { map }
    }

    /// Construct from explicit pairs. For tests and hosts synthesizing an
    /// `Args` from a non-standard source (e.g. a fetched JSON config).
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
    /// missing OR the value failed to parse; distinguish these with `get`
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

#[cfg(target_arch = "wasm32")]
thread_local! {
    /// Page query forwarded into a worker via the init message; workers have
    /// no `window.location`.
    static QUERY_OVERRIDE: std::cell::RefCell<Option<(String, String)>> =
        const { std::cell::RefCell::new(None) };
}

/// Stash the page's `location.search` and `location.hash` inside a worker.
/// Called by the worker entry before `App::setup`; hash wins on collision,
/// matching the main-thread parse order.
#[cfg(target_arch = "wasm32")]
pub fn set_query_override(search: String, hash: String) {
    QUERY_OVERRIDE.with(|q| *q.borrow_mut() = Some((search, hash)));
}

/// Parse a query/hash fragment ("?a=1&b=2" or "#a=1") into `map`. Leading
/// `?`/`#` stripped; segments without `=` skipped (bare `?flag` unsupported).
/// Repeated keys resolve last-write-wins, which is what makes the caller's
/// search-then-hash order mean "hash wins".
///
/// Pure `str` parsing with no `web_sys` dependency, so the `test` arm keeps
/// it compiled and exercised on the host even though only wasm32 calls it.
#[cfg(any(target_arch = "wasm32", test))]
fn parse_query_into(raw: &str, map: &mut HashMap<String, String>) {
    let trimmed = raw.trim_start_matches(['?', '#']);
    if trimmed.is_empty() {
        return;
    }
    for pair in trimmed.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            if !k.is_empty() {
                // Only `+` -> space; skipping full percent-decode (an extra
                // wasm-bindgen call per key) since values are simple ASCII
                // identifiers. Demos needing more can use `urlencoding`.
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

    /// Fold fragments into one map in caller order (search, then hash) and
    /// sort, so the assertions below can be exact rather than key-by-key.
    fn parse_all(fragments: &[&str]) -> Vec<(String, String)> {
        let mut map = HashMap::new();
        for fragment in fragments {
            parse_query_into(fragment, &mut map);
        }
        let mut pairs: Vec<(String, String)> = map.into_iter().collect();
        pairs.sort();
        pairs
    }

    fn pairs(expected: &[(&str, &str)]) -> Vec<(String, String)> {
        expected
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn query_leading_markers_are_stripped_regardless_of_count_or_kind() {
        assert_eq!(parse_all(&["?a=1&b=2"]), pairs(&[("a", "1"), ("b", "2")]));
        assert_eq!(parse_all(&["#a=1"]), pairs(&[("a", "1")]));
        assert_eq!(parse_all(&["??a=1"]), pairs(&[("a", "1")]));
        assert_eq!(parse_all(&["#?a=1"]), pairs(&[("a", "1")]));
    }

    #[test]
    fn query_segments_without_a_nonempty_key_and_separator_are_dropped() {
        assert_eq!(parse_all(&["?flag"]), pairs(&[]));
        assert_eq!(parse_all(&["?=1"]), pairs(&[]));
        assert_eq!(parse_all(&["?"]), pairs(&[]));
        assert_eq!(parse_all(&[""]), pairs(&[]));

        // A malformed segment must not swallow its well-formed neighbours.
        assert_eq!(parse_all(&["?flag&=1&a=2"]), pairs(&[("a", "2")]));
    }

    #[test]
    fn query_values_decode_plus_as_space_and_keep_empty_values() {
        assert_eq!(
            parse_all(&["?title=hello+world&plus=a+b+c&empty="]),
            pairs(&[("empty", ""), ("plus", "a b c"), ("title", "hello world")])
        );
    }

    #[test]
    fn query_repeated_keys_resolve_last_write_wins_within_and_across_fragments() {
        assert_eq!(parse_all(&["?a=1&a=2&a=3"]), pairs(&[("a", "3")]));

        // Caller order is search then hash, so the hash value must survive.
        assert_eq!(
            parse_all(&["?a=search&b=only-search", "#a=hash"]),
            pairs(&[("a", "hash"), ("b", "only-search")])
        );
    }

    #[test]
    fn query_pairs_split_on_the_first_equals_so_values_keep_the_rest() {
        assert_eq!(parse_all(&["?state=a=b"]), pairs(&[("state", "a=b")]));
        assert_eq!(parse_all(&["?a=1=2=3"]), pairs(&[("a", "1=2=3")]));
        assert_eq!(parse_all(&["?eq=="]), pairs(&[("eq", "=")]));

        // Base64 padding is the value shape that reaches a share link with a
        // trailing '=' without any percent-encoding to hide it.
        assert_eq!(parse_all(&["?t=abc=="]), pairs(&[("t", "abc==")]));
        assert_eq!(parse_all(&["?t=YQ="]), pairs(&[("t", "YQ=")]));

        assert_eq!(
            parse_all(&["?a=x=y&b=2"]),
            pairs(&[("a", "x=y"), ("b", "2")])
        );
    }

    #[test]
    fn query_empty_fragment_leaves_prior_entries_intact() {
        assert_eq!(parse_all(&["?a=1", "#", ""]), pairs(&[("a", "1")]));
    }
}
