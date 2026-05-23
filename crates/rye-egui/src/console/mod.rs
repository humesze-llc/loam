//! Quake-style developer console: half-screen drop-down overlay with a command
//! registry, scrollback, hotkey binding, and tab autocomplete.
//!
//! The interaction model follows the idTech console (Quake, 1996): a drop-down
//! activated by `` ` ``, monospace scrollback above an input line, history navigated
//! with Up/Down, completion via Tab, hotkey binds for arbitrary command lines.
//!
//! ## What lives here vs what doesn't
//!
//! - **Here**: `Console` (the main type), [`Command`] trait + [`cmd`] closure shim,
//!   [`ConsoleWriter`] (output collector), key handling for the input line, the parser.
//!   `Console` is generic over a `Ctx` type so consuming crates choose what state
//!   commands operate on.
//! - **Not here**: built-in commands (`screenshot`, `capture.start`, `bind`, `quit`).
//!   Those depend on app/runtime state and live in `rye-app::builtins`. This module
//!   ships only `help` and `clear`, which need only `Console` itself.
//!
//! ## Why egui consumes the keys before TextEdit sees them
//!
//! Egui's `TextEdit::singleline` swallows printable characters into the buffer and uses
//! `Tab` to move focus. Without explicit interception, the toggle key (`` ` ``) types a
//! backtick into the input the moment the console opens, and `Tab` shifts focus off the
//! input box. The key handler at the top of [`Console::ui`] uses
//! `egui::InputState::consume_key` to claim the keystrokes before TextEdit's per-frame
//! processing runs. Up/Down/Tab/Esc/Ctrl+L/Ctrl+C are intercepted the same way; Enter
//! is detected via the TextEdit response in the panel module so we still get
//! `lost_focus`-on-submit semantics.
//!
//! ## Constants
//!
//! Tunables are module-level `const`s ([`MAX_HISTORY_LINES`], [`MAX_INPUT_HISTORY`],
//! [`ANIM_DURATION_SECS`], [`PANEL_HEIGHT_FRACTION`]) rather than runtime config; the
//! values are UX choices, not deployment knobs.

use std::collections::{BTreeMap, HashMap, VecDeque};

mod panel;

/// Scrollback line cap. Older lines drop when the buffer exceeds this. Sized for a
/// session's worth of debugging without unbounded memory.
pub const MAX_HISTORY_LINES: usize = 2000;

/// Input-history cap (Up/Down nav). 100 covers a typical session; larger and old
/// entries become noise during cycling.
pub const MAX_INPUT_HISTORY: usize = 100;

/// Slide-down animation duration. 0.15s is fast enough to feel responsive, slow enough
/// to read as motion.
pub const ANIM_DURATION_SECS: f32 = 0.15;

/// Fraction of the viewport height the open console occupies. 0.5 is the Quake
/// convention: enough scrollback visible, scene visible below.
pub const PANEL_HEIGHT_FRACTION: f32 = 0.5;

/// Runtime flag controlling whether new scrollback lines also echo to the
/// browser DevTools console (via direct `console.log`, NOT through `tracing`).
///
/// Off by default. On wasm32 the path is `Console::push_history` ->
/// `web_sys::console::log_1`; on native the flag has no effect because the
/// native log subscriber already prints to stderr. Toggled via the `log echo`
/// console subcommand registered by `rye_app::log`. Process-global because the
/// typical demo has one Console; multi-Console demos would share the toggle.
#[cfg(target_arch = "wasm32")]
static ECHO_TO_BROWSER: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Enable / disable scrollback echo to the browser DevTools console (wasm32
/// only). See `ECHO_TO_BROWSER` for the architectural rationale. On native
/// this is a no-op so demos can call it unconditionally during command setup.
pub fn set_console_echo(enabled: bool) {
    #[cfg(target_arch = "wasm32")]
    ECHO_TO_BROWSER.store(enabled, std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(target_arch = "wasm32"))]
    let _ = enabled;
}

/// Returns the current scrollback-echo state. Always `false` on native (the
/// native log path is fundamentally different and doesn't go through this
/// flag).
pub fn console_echo_enabled() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        ECHO_TO_BROWSER.load(std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        false
    }
}

// ---------------------------------------------------------------------------
// History line types
// ---------------------------------------------------------------------------

/// A single line in the scrollback buffer.
#[derive(Clone, Debug)]
pub struct HistoryLine {
    pub kind: LineKind,
    pub text: String,
}

impl HistoryLine {
    pub fn input(text: impl Into<String>) -> Self {
        Self {
            kind: LineKind::Input,
            text: text.into(),
        }
    }
    pub fn output(text: impl Into<String>) -> Self {
        Self {
            kind: LineKind::Output,
            text: text.into(),
        }
    }
    pub fn error(text: impl Into<String>) -> Self {
        Self {
            kind: LineKind::Error,
            text: text.into(),
        }
    }
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            kind: LineKind::System,
            text: text.into(),
        }
    }
}

/// Classifies a scrollback line so the panel can color it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineKind {
    /// User-typed input echoed back. Rendered prominent.
    Input,
    /// Command-produced output. Rendered standard.
    Output,
    /// Error from command execution or unknown-command lookup.
    Error,
    /// Console-produced status (e.g., bind set, history cleared).
    System,
}

// ---------------------------------------------------------------------------
// Output collector handed to commands
// ---------------------------------------------------------------------------

/// Per-invocation output sink. Commands push lines via [`ConsoleWriter::line`] /
/// [`ConsoleWriter::error`]; the console drains the collected lines into the scrollback
/// after the command returns.
///
/// The two-phase design (command writes to local Vec, console drains) avoids the borrow
/// conflict between the command's mutable access to the registry slot and the console's
/// mutable access to its own scrollback during the same `execute` call.
pub struct ConsoleWriter {
    lines: Vec<HistoryLine>,
}

impl ConsoleWriter {
    fn new() -> Self {
        Self { lines: Vec::new() }
    }

    /// Append a regular output line.
    pub fn line(&mut self, text: impl Into<String>) {
        self.lines.push(HistoryLine::output(text));
    }

    /// Append an error line. Use for command-level failures the user should see;
    /// bubble unrecoverable errors via `Result` instead.
    pub fn error(&mut self, text: impl Into<String>) {
        self.lines.push(HistoryLine::error(text));
    }
}

// ---------------------------------------------------------------------------
// Command trait + closure shim
// ---------------------------------------------------------------------------

/// Console command implementation. Generic over a `Ctx` type so the consuming crate
/// decides what state commands can mutate. For Rye this is typically a struct holding
/// `&mut dyn App`, `&mut Capture`, and an exit signal.
pub trait Command<Ctx>: 'static {
    /// The name typed at the prompt. Conventionally lowercase, dotted for namespacing
    /// (`capture.start`).
    fn name(&self) -> &str;

    /// One-line description shown by `help` (no argument) when listing every command.
    /// Conventionally <= ~60 chars so the listing fits in a single console row.
    fn help(&self) -> &str;

    /// Multi-line help shown by `help <name>` for a specific command. Default returns
    /// just the one-line [`Self::help`] string -- override when the command's surface
    /// is richer than fits one line (multiple subcommands, multi-step usage examples,
    /// arg-by-arg explanations).
    ///
    /// `\n` line breaks are honored; the panel paints each line into the scrollback as
    /// its own entry so word-wrap and scroll behavior stay consistent with the rest of
    /// the console. Returns owned `String` so subcommand-dispatching commands can build
    /// the listing dynamically from their registered children without storing a static.
    fn long_help(&self) -> String {
        self.help().to_string()
    }

    /// Tab-completion choices for the `arg_index`-th positional argument, without
    /// awareness of values typed in prior slots. Default is empty (no completion /
    /// free-form arg like a path or number). Override via [`FnCommand::with_args`]
    /// when an arg is a fixed enum like `pre|post|both`, or include a `key=` entry
    /// to declare a key-value arg whose values are supplied separately by
    /// [`Command::arg_value_choices`].
    ///
    /// Most commands should override this. Subcommand-style commands whose value
    /// slot depends on what subcommand was picked should override
    /// [`Command::arg_choices_ctx`] instead (this method's default returns `&[]`,
    /// and `arg_choices_ctx`'s default delegates back here).
    fn arg_choices(&self, arg_index: usize) -> &[&'static str] {
        let _ = arg_index;
        &[]
    }

    /// Context-aware variant of [`Command::arg_choices`]. Receives the arg tokens
    /// parsed BEFORE the current completion position (`prior.len() == arg_index`),
    /// so completion can branch on prior choices.
    ///
    /// Default delegates to [`Command::arg_choices`], so commands that don't need
    /// context don't have to override this. Subcommand dispatch (e.g.
    /// [`SubcommandSet`]) overrides this to gate the value-slot choices on the
    /// selected subcommand.
    ///
    /// The explicit `'a` lifetime ties the returned slice to `&self`; the nested
    /// `&[&str]` in `prior` would otherwise confuse lifetime elision.
    fn arg_choices_ctx<'a>(&'a self, arg_index: usize, prior: &[&str]) -> &'a [&'static str] {
        let _ = prior;
        self.arg_choices(arg_index)
    }

    /// Enumerable values for a `key=value` arg at `arg_index` whose key is `key`
    /// (no trailing `=`). Enables two-step tab completion: the user first Tabs
    /// onto the `key=` prefix, then a second Tab cycles through these values. An
    /// empty return means free-form (the user types whatever after the `=`).
    fn arg_value_choices(&self, arg_index: usize, key: &str) -> &[&'static str] {
        let _ = (arg_index, key);
        &[]
    }

    /// Context-aware variant of [`Command::arg_value_choices`]. Receives the arg
    /// tokens parsed BEFORE the current completion position; subcommand-dispatching
    /// commands route kv-value lookups to the active subcommand's value table
    /// using this. Default delegates to [`Command::arg_value_choices`].
    fn arg_value_choices_ctx<'a>(
        &'a self,
        arg_index: usize,
        key: &str,
        prior: &[&str],
    ) -> &'a [&'static str] {
        let _ = prior;
        self.arg_value_choices(arg_index, key)
    }

    /// Run the command. `args` are whitespace-split tokens after the command name.
    /// Output goes to `out`; recoverable issues get `out.error(..)`; unrecoverable
    /// ones return `Err`.
    fn run(&mut self, args: &[&str], ctx: &mut Ctx, out: &mut ConsoleWriter) -> anyhow::Result<()>;
}

/// Closure-backed [`Command`] implementation. Use [`cmd`] to construct.
pub struct FnCommand<F> {
    name: &'static str,
    help: &'static str,
    /// Optional multi-line text returned by [`Command::long_help`] when set. When `None`,
    /// `long_help` falls back to repeating `help`. Set via [`FnCommand::with_long_help`].
    long_help: Option<&'static str>,
    arg_choices: Vec<Vec<&'static str>>,
    /// Per-key value choices for `key=value` args, applied across every arg
    /// position the key appears at. Keyed by the bare key name (no `=`).
    value_choices: HashMap<&'static str, Vec<&'static str>>,
    f: F,
}

impl<F> FnCommand<F> {
    /// Declare positional-argument choices for tab-completion. Each inner slice lists
    /// the valid values for that positional position. Trailing free-form args (paths,
    /// numbers, expressions) can be omitted; the console returns no completions for
    /// positions beyond the declared list.
    ///
    /// ```ignore
    /// cmd("capture", "...", |args, ctx, out| { ... }).with_args(&[
    ///     &["png", "frames", "toggle", "stop"],
    ///     &["pre", "post", "both"],
    /// ])
    /// ```
    pub fn with_args(mut self, choices: &[&[&'static str]]) -> Self {
        self.arg_choices = choices.iter().map(|s| s.to_vec()).collect();
        self
    }

    /// Declare enumerable values for a `key=value` arg. The first Tab completes
    /// the user's key prefix to `key=`; once `=` is in the input, subsequent
    /// completions cycle through these values. Free-form numeric args (`fps=N`)
    /// simply don't call this and only the bare `key=` shows up in tab cycling.
    ///
    /// ```ignore
    /// cmd("capture", "...", |a, c, o| ...)
    ///     .with_args(&[&["png", "gif"], &["fps=", "palette="]])
    ///     .with_value_choices("palette", &["local", "global"])
    /// ```
    pub fn with_value_choices(mut self, key: &'static str, values: &[&'static str]) -> Self {
        self.value_choices.insert(key, values.to_vec());
        self
    }

    /// Attach a multi-line help block returned by [`Command::long_help`]. Newlines are
    /// honored; the console paints each line as its own scrollback entry so wrapping +
    /// scroll behavior stay consistent. Use for commands whose surface (multiple
    /// subcommand-style args, usage examples) doesn't fit one line.
    ///
    /// ```ignore
    /// cmd("wireframe", "wireframe overlay (see help)", handler)
    ///     .with_args(&[&["on", "off", "nearest-active"], &["on", "off"]])
    ///     .with_long_help(
    ///         "Cross-section + parent-wireframe overlay.\n\
    ///          \n\
    ///          subcommands:\n  \
    ///          on              enable the overlay\n  \
    ///          off             disable\n  \
    ///          nearest-active  toggle the per-cell brightness gradient",
    ///     )
    /// ```
    pub fn with_long_help(mut self, long: &'static str) -> Self {
        self.long_help = Some(long);
        self
    }
}

/// Build a [`Command`] from a closure. The closure mutates a `Ctx` and writes lines
/// into the [`ConsoleWriter`]. Idiomatic for inline per-demo registrations.
///
/// ```ignore
/// console.register(cmd("teleport", "teleport <x> <y> <z>", |args, ctx, out| {
///     let [x, y, z] = parse3(args)?;
///     ctx.player.position = vec3(x, y, z);
///     out.line(format!("ok, now at ({x}, {y}, {z})"));
///     Ok(())
/// }));
/// ```
pub fn cmd<Ctx, F>(name: &'static str, help: &'static str, f: F) -> FnCommand<F>
where
    F: FnMut(&[&str], &mut Ctx, &mut ConsoleWriter) -> anyhow::Result<()> + 'static,
{
    FnCommand {
        name,
        help,
        long_help: None,
        arg_choices: Vec::new(),
        value_choices: HashMap::new(),
        f,
    }
}

impl<Ctx, F> Command<Ctx> for FnCommand<F>
where
    F: FnMut(&[&str], &mut Ctx, &mut ConsoleWriter) -> anyhow::Result<()> + 'static,
{
    fn name(&self) -> &str {
        self.name
    }
    fn help(&self) -> &str {
        self.help
    }
    fn long_help(&self) -> String {
        self.long_help
            .map(str::to_string)
            .unwrap_or_else(|| self.help.to_string())
    }
    fn arg_choices(&self, arg_index: usize) -> &[&'static str] {
        self.arg_choices
            .get(arg_index)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
    fn arg_value_choices(&self, _arg_index: usize, key: &str) -> &[&'static str] {
        self.value_choices
            .get(key)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
    fn run(&mut self, args: &[&str], ctx: &mut Ctx, out: &mut ConsoleWriter) -> anyhow::Result<()> {
        (self.f)(args, ctx, out)
    }
}

// ---------------------------------------------------------------------------
// Subcommand dispatch
// ---------------------------------------------------------------------------

// Tab-completion for toggle subcommands intentionally returns an empty list at the value
// slot. The primary UX for a toggle is *bare* invocation (e.g. `wireframe nearest-active`
// flips), and surfacing `on|off` in completion implied that one of them was required when it
// isn't. The parser still accepts `on|off|true|false|1|0` for callers that want explicit
// set; it's just no longer promoted in the cycle.

/// Boxed handler for an on/off toggle subcommand. The framework passes `Some(bool)`
/// when the user supplied `on|off|true|false|1|0` and `None` when the user invoked the
/// subcommand bare (no value). The handler is responsible for flipping the field in
/// the `None` case; the framework can't because it doesn't own the field.
///
/// Idiomatic shape for a `bool` field on `Ctx`:
///
/// ```ignore
/// .toggle("axes", "toggle world-axes", |ctx, v| {
///     ctx.show_axes = v.unwrap_or(!ctx.show_axes);
///     Ok(())
/// })
/// ```
type ToggleHandler<Ctx> = Box<dyn FnMut(&mut Ctx, Option<bool>) -> anyhow::Result<()>>;

/// Boxed handler for a fixed-choice subcommand. The framework passes `Some(value)` when the
/// user supplied a value and `None` when the subcommand was invoked bare (no value),
/// mirroring [`ToggleHandler`]'s shape. On `None`, the handler is expected to cycle to the
/// next choice (or whatever's contextually meaningful); the framework can't help because it
/// doesn't know the current state.
///
/// Idiomatic handler for a `Mode` enum field with a `cycle` method:
///
/// ```ignore
/// .choice("color", "...", &["unique", "active"], |ctx, name| {
///     ctx.color_mode = match name {
///         Some(n) => parse_mode(n)?,
///         None => ctx.color_mode.cycle(),
///     };
///     Ok(())
/// })
/// ```
type ChoiceHandler<Ctx> = Box<dyn FnMut(&mut Ctx, Option<&str>) -> anyhow::Result<()>>;

/// Boxed handler for a `SubcommandSet`'s bare invocation (no subcommand supplied).
/// When set via [`SubcommandSet::on_bare`], replaces the default usage-block error
/// with a caller-defined action, typically flipping a primary toggle field so
/// `wireframe` alone reads as "flip the overlay's main on/off."
type BareHandler<Ctx> = Box<dyn FnMut(&mut Ctx) -> anyhow::Result<()>>;

/// Boxed handler for a custom-grammar subcommand. Receives the user's context, the
/// raw args slice AFTER the subcommand name (positional + key-value tokens, framework
/// does not parse them), and the writer. Used for subcommands whose grammar doesn't
/// fit the simpler `.toggle` / `.choice` shapes (e.g. `capture gif post fps=30
/// scale=720 palette=global`).
type CustomHandler<Ctx> =
    Box<dyn FnMut(&mut Ctx, &[&str], &mut ConsoleWriter) -> anyhow::Result<()>>;

/// One entry in a [`SubcommandSet`]. The dispatch kind decides how the framework
/// parses the value slot and what's offered for tab completion.
enum SubcommandKind<Ctx> {
    /// On/off subcommand. The framework parses `args[1]` as `on|off|true|false|1|0` and
    /// passes `Some(bool)`; bare invocation (no value) passes `None` and the handler is
    /// expected to flip its field. No value-slot tab completion is offered (bare-flip is the
    /// canonical UX; explicit set is supported but not promoted).
    Toggle { handler: ToggleHandler<Ctx> },
    /// Fixed-choice subcommand. The framework completes the value slot from `choices`
    /// and passes `Some(value)` when present, `None` on bare invocation (handler is
    /// expected to cycle).
    Choice {
        choices: Vec<&'static str>,
        handler: ChoiceHandler<Ctx>,
    },
    /// Custom-grammar subcommand. Per-slot positional choices drive tab completion
    /// (slot 0 is the first arg AFTER the subcommand name); per-key value enumerables
    /// drive two-step kv completion. The framework dispatches by subcommand name and
    /// then hands the raw args + writer to the handler; arg parsing is the
    /// handler's responsibility.
    Custom {
        arg_choices: Vec<Vec<&'static str>>,
        value_choices: HashMap<&'static str, Vec<&'static str>>,
        handler: CustomHandler<Ctx>,
    },
}

struct SubcommandEntry<Ctx> {
    help: &'static str,
    kind: SubcommandKind<Ctx>,
}

/// A command that dispatches to one of several named subcommands based on the first
/// positional arg. Provides typed dispatch (no `match arg.to_lowercase()` boilerplate
/// per command) and context-aware tab completion (the value-slot list narrows to the
/// chosen subcommand's allowed values).
///
/// Build with [`subcommands`] and chain [`SubcommandSet::toggle`] /
/// [`SubcommandSet::choice`] to register subcommands. Register the whole set as a
/// single command via `console.register(set)`.
///
/// ```ignore
/// let tests = subcommands::<MyCtx>("tests", "select what renders")
///     .toggle("axes", "toggle world-axes", |ctx, on| {
///         ctx.show_axes = on;
///         Ok(())
///     })
///     .choice(
///         "polytope",
///         "set R⁴ polytope overlay",
///         &["5cell", "tesseract", "16cell", "off"],
///         |ctx, name| { ctx.polytope = parse_polytope(name)?; Ok(()) },
///     );
/// console.register(tests);
/// ```
pub struct SubcommandSet<Ctx> {
    name: &'static str,
    help: &'static str,
    /// Insertion-ordered subcommands. BTreeMap so iteration is deterministic and Tab
    /// cycling order is alphabetical, matching the rest of the console.
    subs: BTreeMap<&'static str, SubcommandEntry<Ctx>>,
    /// Cached sorted slice of subcommand names. Populated lazily on first
    /// [`Command::arg_choices`] / [`Command::arg_choices_ctx`] call so [`Self::toggle`]
    /// and [`Self::choice`] can stay infallible chainable builders.
    name_cache: std::cell::OnceCell<Vec<&'static str>>,
    /// Optional bare-invocation handler. When set via [`Self::on_bare`], the
    /// command's `run` calls it instead of returning the usage-block error when the
    /// user types just the command name with no subcommand. Use for "primary toggle"
    /// commands where bare invocation should flip a main field.
    bare: Option<BareHandler<Ctx>>,
}

impl<Ctx: 'static> SubcommandSet<Ctx> {
    /// Register an on/off subcommand. The framework parses the value slot as
    /// `on | off | true | false | 1 | 0` when present and passes `Some(bool)`; when
    /// the user types just the subcommand name with no value, the handler is called
    /// with `None` so it can flip the field in place.
    ///
    /// Idiomatic handler shape for a `bool` field:
    ///
    /// ```ignore
    /// .toggle("axes", "toggle world-axes", |ctx, v| {
    ///     ctx.show_axes = v.unwrap_or(!ctx.show_axes);
    ///     Ok(())
    /// })
    /// ```
    pub fn toggle<F>(mut self, name: &'static str, help: &'static str, handler: F) -> Self
    where
        F: FnMut(&mut Ctx, Option<bool>) -> anyhow::Result<()> + 'static,
    {
        self.subs.insert(
            name,
            SubcommandEntry {
                help,
                kind: SubcommandKind::Toggle {
                    handler: Box::new(handler),
                },
            },
        );
        self
    }

    /// Register a fixed-choice subcommand. The framework completes the value slot from
    /// `choices`; the handler receives the raw value string (which is one of `choices`
    /// only after Tab-completion or exact match, since the framework does not validate
    /// the value against `choices` before dispatch).
    pub fn choice<F>(
        mut self,
        name: &'static str,
        help: &'static str,
        choices: &[&'static str],
        handler: F,
    ) -> Self
    where
        F: FnMut(&mut Ctx, Option<&str>) -> anyhow::Result<()> + 'static,
    {
        self.subs.insert(
            name,
            SubcommandEntry {
                help,
                kind: SubcommandKind::Choice {
                    choices: choices.to_vec(),
                    handler: Box::new(handler),
                },
            },
        );
        self
    }

    /// Register a custom-grammar subcommand. Use when `.toggle` / `.choice` are too
    /// rigid: subcommands with multiple positional args, key-value pairs, or both.
    ///
    /// - `arg_choices[i]` lists tab-completion choices for the i-th positional arg
    ///   AFTER the subcommand name. Include `key=` entries for kv-pair prefixes.
    /// - `value_choices[k]` lists enumerable values for the `key=value` arg whose
    ///   bare key is `k` (the framework looks this up when the user types `k=` and
    ///   hits Tab).
    /// - `handler` receives the raw args after the subcommand name plus the writer;
    ///   it owns the parsing of positionals and kv tokens.
    ///
    /// ```ignore
    /// subcommands::<Ctx>("capture", "...")
    ///     .custom(
    ///         "gif",
    ///         "gif sequence (with fps/scale/palette knobs)",
    ///         &[
    ///             &["pre", "post", "both"],
    ///             &["fps=", "palette=", "scale="],
    ///             &["fps=", "palette=", "scale="],
    ///         ],
    ///         &[("palette", &["local", "global"])],
    ///         |ctx, args, out| { /* parse and act */ Ok(()) },
    ///     )
    /// ```
    pub fn custom<F>(
        mut self,
        name: &'static str,
        help: &'static str,
        arg_choices: &[&[&'static str]],
        value_choices: &[(&'static str, &[&'static str])],
        handler: F,
    ) -> Self
    where
        F: FnMut(&mut Ctx, &[&str], &mut ConsoleWriter) -> anyhow::Result<()> + 'static,
    {
        let mut vc = HashMap::new();
        for (k, vs) in value_choices {
            vc.insert(*k, vs.to_vec());
        }
        self.subs.insert(
            name,
            SubcommandEntry {
                help,
                kind: SubcommandKind::Custom {
                    arg_choices: arg_choices.iter().map(|slot| slot.to_vec()).collect(),
                    value_choices: vc,
                    handler: Box::new(handler),
                },
            },
        );
        self
    }

    /// Attach a bare-invocation handler. When set, typing just the command name
    /// (no subcommand, no args) runs `handler` instead of returning a usage error.
    /// Use for "primary toggle" commands where bare invocation should flip a main
    /// field (e.g. `wireframe` toggles the overlay's main on/off, then
    /// `wireframe nearest-active` and `wireframe color` modulate behavior).
    ///
    /// ```ignore
    /// subcommands::<Ctx>("wireframe", "...")
    ///     .on_bare(|ctx| {
    ///         ctx.wireframe_enabled = !ctx.wireframe_enabled;
    ///         Ok(())
    ///     })
    ///     .toggle("nearest-active", "...", |ctx, v| { ... })
    ///     .choice("color", "...", &["position", "active"], |ctx, v| { ... })
    /// ```
    pub fn on_bare<F>(mut self, handler: F) -> Self
    where
        F: FnMut(&mut Ctx) -> anyhow::Result<()> + 'static,
    {
        self.bare = Some(Box::new(handler));
        self
    }

    fn cached_names(&self) -> &[&'static str] {
        self.name_cache
            .get_or_init(|| self.subs.keys().copied().collect())
    }
}

/// Build a [`SubcommandSet`] for a multi-subcommand console command. See the
/// [`SubcommandSet`] docs for the full builder pattern.
pub fn subcommands<Ctx: 'static>(name: &'static str, help: &'static str) -> SubcommandSet<Ctx> {
    SubcommandSet {
        name,
        help,
        subs: BTreeMap::new(),
        name_cache: std::cell::OnceCell::new(),
        bare: None,
    }
}

impl<Ctx: 'static> Command<Ctx> for SubcommandSet<Ctx> {
    fn name(&self) -> &str {
        self.name
    }
    fn help(&self) -> &str {
        self.help
    }

    fn long_help(&self) -> String {
        // First line is the umbrella's one-liner; subsequent lines list every
        // registered subcommand with its description. `help <set-name>` then reads as a
        // mini-manual page for the whole subcommand family.
        let mut out = String::with_capacity(128 + self.subs.len() * 64);
        out.push_str(self.help);
        if !self.subs.is_empty() {
            out.push_str("\nsubcommands:");
            for (name, entry) in &self.subs {
                let kind = match entry.kind {
                    SubcommandKind::Toggle { .. } => "<on|off>",
                    SubcommandKind::Choice { .. } => "<choice>",
                    SubcommandKind::Custom { .. } => "<args...>",
                };
                out.push_str(&format!("\n  {name:14} {kind:9}  {}", entry.help));
            }
        }
        out
    }

    fn arg_choices(&self, arg_index: usize) -> &[&'static str] {
        if arg_index == 0 {
            self.cached_names()
        } else {
            &[]
        }
    }

    fn arg_choices_ctx<'a>(&'a self, arg_index: usize, prior: &[&str]) -> &'a [&'static str] {
        if arg_index == 0 {
            return self.cached_names();
        }
        let Some(&sub_name) = prior.first() else {
            return &[];
        };
        let Some(entry) = self.subs.get(sub_name) else {
            return &[];
        };
        // Within-subcommand slot index: arg_index 1 is the first arg AFTER the
        // subcommand name, which is slot 0 of the subcommand's own grammar.
        let sub_slot = arg_index - 1;
        match &entry.kind {
            SubcommandKind::Toggle { .. } => {
                // Empty by design (see the comment on `ON_OFF_CHOICES`'s old slot
                // above). Bare invocation is the canonical UX; `on|off` is still
                // accepted as input but not surfaced as a suggestion.
                &[]
            }
            SubcommandKind::Choice { choices, .. } => {
                if sub_slot == 0 {
                    choices.as_slice()
                } else {
                    &[]
                }
            }
            SubcommandKind::Custom { arg_choices, .. } => arg_choices
                .get(sub_slot)
                .map(|v| v.as_slice())
                .unwrap_or(&[]),
        }
    }

    fn arg_value_choices_ctx<'a>(
        &'a self,
        _arg_index: usize,
        key: &str,
        prior: &[&str],
    ) -> &'a [&'static str] {
        let Some(&sub_name) = prior.first() else {
            return &[];
        };
        let Some(entry) = self.subs.get(sub_name) else {
            return &[];
        };
        match &entry.kind {
            SubcommandKind::Custom { value_choices, .. } => {
                value_choices.get(key).map(|v| v.as_slice()).unwrap_or(&[])
            }
            _ => &[],
        }
    }

    fn run(&mut self, args: &[&str], ctx: &mut Ctx, out: &mut ConsoleWriter) -> anyhow::Result<()> {
        let Some((sub_name, rest)) = args.split_first() else {
            // Bare invocation. If `on_bare` is registered, call it ("primary toggle"
            // pattern); otherwise emit a usage block listing subcommands.
            if let Some(handler) = self.bare.as_mut() {
                return handler(ctx);
            }
            let mut msg = format!("usage: {} <subcommand> <value>; subcommands:", self.name);
            for (name, entry) in &self.subs {
                msg.push_str(&format!("\n  {name:12} {}", entry.help));
            }
            return Err(anyhow::anyhow!(msg));
        };
        let Some(entry) = self.subs.get_mut(*sub_name) else {
            let names: Vec<&str> = self.subs.keys().copied().collect();
            return Err(anyhow::anyhow!(
                "unknown subcommand `{sub_name}` for `{}` (try {})",
                self.name,
                names.join(", ")
            ));
        };
        match &mut entry.kind {
            SubcommandKind::Toggle { handler } => {
                // Bare subcommand invocation (no value): hand `None` to the handler so
                // it can flip its field. With a value: parse `on|off|true|false|1|0`
                // and pass `Some(bool)`.
                let v: Option<bool> = match rest.first() {
                    None => None,
                    Some(value) => match value.to_ascii_lowercase().as_str() {
                        "on" | "true" | "1" => Some(true),
                        "off" | "false" | "0" => Some(false),
                        other => {
                            return Err(anyhow::anyhow!(
                                "unknown value `{other}` for `{} {sub_name}` (try on|off)",
                                self.name
                            ))
                        }
                    },
                };
                let _ = out;
                handler(ctx, v)
            }
            SubcommandKind::Choice { handler, .. } => {
                // `Some(value)` with an explicit arg, `None` on bare invocation. The
                // handler decides what to do on `None` (cycle to next choice is the
                // canonical pattern; alternatives include "error" or "no-op").
                let value: Option<&str> = rest.first().copied();
                let _ = out;
                handler(ctx, value)
            }
            SubcommandKind::Custom { handler, .. } => handler(ctx, rest, out),
        }
    }
}

// ---------------------------------------------------------------------------
// Console
// ---------------------------------------------------------------------------

/// The dev console. Owns the command registry, scrollback, input line, hotkey binds,
/// and open/close state.
///
/// Construct, register commands and binds during app setup, then call [`Console::ui`]
/// once per frame from inside the host's egui pass:
///
/// ```ignore
/// let mut console = Console::<MyCtx>::new();
/// console.register(cmd("hello", "say hi", |_, _, out| {
///     out.line("hi");
///     Ok(())
/// }));
/// console.bind(egui::Key::F9, "capture.toggle");
///
/// // per frame:
/// console.ui(&egui_ctx, &mut my_ctx);
/// ```
pub struct Console<Ctx> {
    commands: BTreeMap<String, Box<dyn Command<Ctx>>>,
    binds: HashMap<egui::Key, String>,
    toggle_key: egui::Key,
    history: VecDeque<HistoryLine>,
    input: String,
    input_history: VecDeque<String>,
    /// `Some(i)` while cycling history with Up/Down; `None` after `Enter` or after
    /// typing into a fresh input.
    input_history_pos: Option<usize>,
    /// Active tab-completion cycle, if any. Cleared on any input edit that isn't a
    /// tab-complete itself.
    tab: Option<TabState>,
    open: bool,
    /// True for the frame after `open` becomes true so the panel can request focus
    /// once.
    pending_focus: bool,
    /// Right-aligned text in the title row. Host fills with anything useful (fps,
    /// recording state, current scene); empty by default.
    status: String,
    /// `false` for the half-screen drop-down (default Quake-style), `true` for a
    /// draggable / resizable egui Window. Detached mode lets the user click outside
    /// the console to give keyboard focus back to the app, since the docked console
    /// permanently captures keyboard while open.
    detached: bool,
    /// Set in docked mode when the user clicks outside the panel rect; suppresses the
    /// input row's per-frame focus re-request so mouse + keyboard go back to the app
    /// while the console stays visible. Cleared by clicking back inside the panel or
    /// by reopening the console.
    user_defocused: bool,
    /// One-frame flag set whenever code outside the TextEdit mutates [`Self::input`]
    /// (tab-complete cycles, history nav). The panel snaps the TextEdit's internal
    /// cursor to the end of the new input and clears the flag, so typing continues
    /// from the tail rather than wherever the cursor previously sat.
    pending_cursor_to_end: bool,
}

struct TabState {
    matches: Vec<String>,
    index: usize,
    ctx: CompletionContext,
}

/// What the user is currently typing, partitioned for completion. `prefix` is the
/// partially-typed token under the cursor; an empty `prefix` with trailing whitespace
/// means the user has finished the previous token and is starting the next one.
#[derive(Clone, Debug)]
enum CompletionContext {
    /// Completing the command name (no whitespace yet, or whitespace-leading input).
    Command { prefix: String },
    /// Completing positional argument `arg_index` of `cmd_name`. `prior` carries the
    /// fully-typed arg tokens BEFORE the cursor (`prior.len() == arg_index`); commands
    /// that branch their value-slot completion on prior choices (subcommand dispatch)
    /// read it via [`Command::arg_choices_ctx`].
    Arg {
        cmd_name: String,
        arg_index: usize,
        prior: Vec<String>,
        prefix: String,
    },
}

impl CompletionContext {
    fn prefix(&self) -> &str {
        match self {
            CompletionContext::Command { prefix } => prefix,
            CompletionContext::Arg { prefix, .. } => prefix,
        }
    }
}

impl<Ctx: 'static> Default for Console<Ctx> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Ctx: 'static> Console<Ctx> {
    /// Empty console with no commands, no binds, default `` ` `` toggle key.
    pub fn new() -> Self {
        Self {
            commands: BTreeMap::new(),
            binds: HashMap::new(),
            toggle_key: egui::Key::Backtick,
            history: VecDeque::new(),
            input: String::new(),
            input_history: VecDeque::new(),
            input_history_pos: None,
            tab: None,
            open: false,
            pending_focus: false,
            status: String::new(),
            detached: false,
            user_defocused: false,
            pending_cursor_to_end: false,
        }
    }

    /// Override the toggle key. Default is [`egui::Key::Backtick`].
    pub fn with_toggle_key(mut self, key: egui::Key) -> Self {
        self.toggle_key = key;
        self
    }

    /// Register a command. Replaces any existing command of the same name without
    /// warning; consuming crates can pre-check via [`Console::has_command`] if they
    /// care.
    pub fn register<C: Command<Ctx> + 'static>(&mut self, command: C) {
        let name = command.name().to_string();
        self.commands.insert(name, Box::new(command));
    }

    /// True if a command with this name is registered. `help` and `clear` are built in
    /// and always return true.
    pub fn has_command(&self, name: &str) -> bool {
        name == "help" || name == "clear" || self.commands.contains_key(name)
    }

    /// Bind `key` (no modifiers) to execute `command_line` when the console is closed.
    /// Re-binding overwrites the previous binding.
    pub fn bind(&mut self, key: egui::Key, command_line: impl Into<String>) {
        self.binds.insert(key, command_line.into());
    }

    /// Remove a bind. No-op if the key wasn't bound.
    pub fn unbind(&mut self, key: egui::Key) {
        self.binds.remove(&key);
    }

    /// Open the console. Idempotent.
    pub fn open(&mut self) {
        if !self.open {
            self.open = true;
            self.pending_focus = true;
            // Reopening clears any prior click-outside defocus so typing lands in the
            // input again.
            self.user_defocused = false;
        }
    }

    /// Close the console. Idempotent.
    pub fn close(&mut self) {
        self.open = false;
    }

    /// Toggle open/closed.
    pub fn toggle(&mut self) {
        if self.open {
            self.close()
        } else {
            self.open()
        }
    }

    /// Currently open?
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Switch to detached mode: console renders as a draggable / resizable
    /// [`egui::Window`] instead of the half-screen drop-down. Idempotent.
    pub fn detach(&mut self) {
        self.detached = true;
    }

    /// Switch to docked mode: half-screen drop-down (the default). Idempotent.
    pub fn dock(&mut self) {
        self.detached = false;
    }

    /// Detached or docked?
    pub fn is_detached(&self) -> bool {
        self.detached
    }

    /// Append a line to the scrollback. Useful for system messages generated outside
    /// command execution (e.g., a background recording finishing).
    pub fn write(&mut self, line: HistoryLine) {
        self.push_history(line);
    }

    /// Set the title-row status text (right-aligned). Host calls this each frame with
    /// whatever readout it wants visible in the console: fps, recording elapsed, scene
    /// name, etc.
    pub fn set_status(&mut self, text: impl Into<String>) {
        self.status = text.into();
    }

    /// Per-frame entry point. Handles toggle key, hotkey binds (when closed), in-panel
    /// keys (when open), animation, and panel rendering. Call once per frame from the
    /// host's egui pass.
    pub fn ui(&mut self, egui_ctx: &egui::Context, ctx: &mut Ctx) {
        // 1. Toggle key always active. `consume_key` strips the Key event, but
        // printable keys (Backtick, etc.) also produce a Text event that TextEdit reads
        // independently; strip that too or it leaks into the input box on the second
        // press.
        let toggle_text = key_text(self.toggle_key);
        let toggle_pressed = egui_ctx.input_mut(|i| {
            let pressed = i.consume_key(egui::Modifiers::NONE, self.toggle_key);
            if pressed {
                if let Some(t) = toggle_text {
                    i.events
                        .retain(|e| !matches!(e, egui::Event::Text(s) if s == t));
                }
            }
            pressed
        });
        if toggle_pressed {
            self.toggle();
        }

        // 2. Bound keys fire only when closed (avoid hijacking typing).
        if !self.open && !self.binds.is_empty() {
            let pressed: Vec<String> = {
                let mut hit = Vec::new();
                egui_ctx.input_mut(|i| {
                    for (key, line) in &self.binds {
                        if i.consume_key(egui::Modifiers::NONE, *key) {
                            hit.push(line.clone());
                        }
                    }
                });
                hit
            };
            for line in pressed {
                self.execute(&line, ctx);
            }
        }

        // 3. In-panel keys (consume before TextEdit sees them).
        if self.open {
            self.handle_panel_keys(egui_ctx);
        }

        // 4. Animate panel height (docked only). Detached mode shows/hides instantly:
        // the egui Window has its own appearance, so animating a slide value would only
        // produce dead frames where `open=false` is still being drawn.
        let target = if self.open { 1.0 } else { 0.0 };
        let progress = egui_ctx.animate_value_with_time(
            egui::Id::new("rye_console_open_progress"),
            target,
            ANIM_DURATION_SECS,
        );

        let visible = if self.detached {
            self.open
        } else {
            progress > 0.0
        };
        if visible {
            panel::draw(self, egui_ctx, ctx, progress);
        }
    }

    // -----------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------

    fn handle_panel_keys(&mut self, ctx: &egui::Context) {
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
            self.close();
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp)) {
            self.history_prev();
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown)) {
            self.history_next();
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Tab)) {
            self.tab_complete();
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::L)) {
            self.history.clear();
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::C)) {
            self.input.clear();
            self.input_history_pos = None;
            self.tab = None;
        }
    }

    fn push_history(&mut self, line: HistoryLine) {
        // Optional echo to the browser DevTools console. Off by default; demos
        // toggle it via `log echo on` for debugging the in-canvas console
        // text from outside (the canvas's pixels aren't selectable, so without
        // this the user can't copy command output for paste-back to a chat /
        // bug report).
        //
        // Architectural note: this deliberately bypasses `tracing` and calls
        // `web_sys::console::log_1` directly. `tracing::info!` would conflict
        // with the existing `log on` feature (tracing -> scrollback via
        // `rye_app::log::ConsoleLayer`); running both directions through
        // tracing creates a feedback loop where each emitted event lands in
        // the scrollback, gets re-echoed, lands again, ad infinitum. The
        // direct console.log path is feedback-free because no Rust subscriber
        // consumes it.
        #[cfg(target_arch = "wasm32")]
        if ECHO_TO_BROWSER.load(std::sync::atomic::Ordering::Relaxed) {
            web_sys::console::log_1(&line.text.as_str().into());
        }
        self.history.push_back(line);
        while self.history.len() > MAX_HISTORY_LINES {
            self.history.pop_front();
        }
    }

    fn history_prev(&mut self) {
        if self.input_history.is_empty() {
            return;
        }
        let pos = match self.input_history_pos {
            None => self.input_history.len() - 1,
            Some(0) => 0,
            Some(p) => p - 1,
        };
        self.input.clone_from(&self.input_history[pos]);
        self.input_history_pos = Some(pos);
        self.tab = None;
        self.pending_cursor_to_end = true;
    }

    fn history_next(&mut self) {
        let Some(pos) = self.input_history_pos else {
            return;
        };
        if pos + 1 >= self.input_history.len() {
            self.input.clear();
            self.input_history_pos = None;
        } else {
            self.input_history_pos = Some(pos + 1);
            self.input.clone_from(&self.input_history[pos + 1]);
        }
        self.tab = None;
        self.pending_cursor_to_end = true;
    }

    fn tab_complete(&mut self) {
        // Continue an existing cycle if still applicable.
        if let Some(tab) = self.tab.as_mut() {
            if !tab.matches.is_empty() {
                tab.index = (tab.index + 1) % tab.matches.len();
                let new_input = apply_completion(&self.input, &tab.ctx, &tab.matches[tab.index]);
                self.input = new_input;
                self.pending_cursor_to_end = true;
                return;
            }
        }
        // Fresh completion: build context from current input, falling back to the
        // empty-prefix command list so Tab on a blank prompt cycles through commands
        // (matches what the ghost preview shows).
        let ctx = self
            .completion_context()
            .unwrap_or(CompletionContext::Command {
                prefix: String::new(),
            });
        let matches = self.completion_matches(&ctx);
        if matches.is_empty() {
            return;
        }
        self.input = apply_completion(&self.input, &ctx, &matches[0]);
        self.pending_cursor_to_end = true;
        if matches.len() > 1 {
            self.tab = Some(TabState {
                matches,
                index: 0,
                ctx,
            });
        } else {
            self.tab = None;
        }
    }

    fn all_command_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.commands.keys().cloned().collect();
        names.extend(Builtin::ALL.iter().map(|b| b.name().to_string()));
        names.sort();
        names
    }

    /// Inspect [`Console::input`] to decide what the user is currently completing: the
    /// command name, or the n-th positional argument of a known command. Returns `None`
    /// for empty input. Uses the quote-aware [`tokenize`] so `tests "5 cell" o<Tab>`
    /// completes on arg 1 with prefix `o`, not on a garbage `cell"` token.
    fn completion_context(&self) -> Option<CompletionContext> {
        if self.input.is_empty() {
            return None;
        }
        let parsed = tokenize(&self.input);
        if parsed.is_empty() {
            return None;
        }
        let trailing_ws = self.input.ends_with(char::is_whitespace);

        // No whitespace yet: still typing the command name.
        if parsed.len() == 1 && !trailing_ws {
            return Some(CompletionContext::Command {
                prefix: parsed.into_iter().next().unwrap(),
            });
        }

        // After whitespace: we're on an argument. `arg_index` is 0-based positional.
        // The `else` arm is reached only when `parsed.len() >= 2` (the `len() == 1 &&
        // !trailing_ws` case returned above), so the partial-token pop is safe. `prior`
        // captures the fully-typed arg tokens before the cursor so subcommand-dispatching
        // commands can gate their value-slot completion on what came earlier.
        let mut parts = parsed;
        let cmd_name = parts.remove(0);
        let (arg_index, prefix, prior) = if trailing_ws {
            let idx = parts.len();
            (idx, String::new(), parts)
        } else {
            let partial = parts.pop().unwrap_or_default();
            (parts.len(), partial, parts)
        };
        Some(CompletionContext::Arg {
            cmd_name,
            arg_index,
            prior,
            prefix,
        })
    }

    fn completion_matches(&self, ctx: &CompletionContext) -> Vec<String> {
        match ctx {
            CompletionContext::Command { prefix } => self
                .all_command_names()
                .into_iter()
                .filter(|name| name.starts_with(prefix.as_str()))
                .collect(),
            CompletionContext::Arg {
                cmd_name,
                arg_index,
                prior,
                prefix,
            } => {
                let Some(cmd) = self.commands.get(cmd_name) else {
                    return Vec::new();
                };
                let prior_refs: Vec<&str> = prior.iter().map(String::as_str).collect();

                // Mid-`key=` value completion: if the partial token already
                // contains an `=`, we're past the key and completing its value.
                // Look up the enumerable values declared via `with_value_choices`
                // (or the context-aware variant for subcommand-dispatching commands)
                // and return them prefixed with `key=`.
                if let Some(eq) = prefix.find('=') {
                    let key = &prefix[..eq];
                    let value_prefix = &prefix[eq + 1..];
                    let mut matches: Vec<String> = cmd
                        .arg_value_choices_ctx(*arg_index, key, &prior_refs)
                        .iter()
                        .filter(|v| v.starts_with(value_prefix))
                        .map(|v| format!("{key}={v}"))
                        .collect();
                    matches.sort();
                    return matches;
                }

                // Identify key=value args already provided in earlier positions
                // (e.g. `fps=30`, `palette=global`) so we don't suggest a choice
                // that shares the same `key=` prefix. Skips the partial last
                // token (that's what we're completing). Stage keywords and plain
                // positionals aren't filtered: re-typing a positional may be
                // intentional, and the parser would just overwrite an earlier
                // value.
                let parsed: Vec<&str> = self.input.split_whitespace().collect();
                let trailing_ws = self.input.ends_with(char::is_whitespace);
                let consumed = if trailing_ws {
                    parsed.as_slice()
                } else {
                    &parsed[..parsed.len().saturating_sub(1)]
                };
                let used_kv_prefixes: Vec<&str> = consumed
                    .iter()
                    .filter_map(|t| t.find('=').map(|i| &t[..=i]))
                    .collect();

                // Sort matches alphabetically so command authors can declare choices
                // in any order (workflow, frequency, narrative) without affecting Tab
                // cycling order. Matches the command-name path, which is also sorted.
                // Uses the context-aware variant so subcommand-dispatching commands
                // can gate value-slot choices on the prior subcommand pick.
                let mut matches: Vec<String> = cmd
                    .arg_choices_ctx(*arg_index, &prior_refs)
                    .iter()
                    .filter(|choice| choice.starts_with(prefix.as_str()))
                    .filter(|choice| {
                        // Suppress any choice whose `key=` prefix was already used.
                        // Choices without `=` (bare keywords) are never filtered.
                        match choice.find('=') {
                            None => true,
                            Some(eq) => {
                                let key = &choice[..=eq];
                                !used_kv_prefixes.contains(&key)
                            }
                        }
                    })
                    .map(|choice| (*choice).to_string())
                    .collect();
                matches.sort();
                matches
            }
        }
    }

    /// Suffix of the *first* (sort-order) completion that matches the current input.
    /// The panel paints this as dim ghost text after the cursor so the user sees
    /// exactly what `Tab` will insert on the next press.
    ///
    /// Works for command names and positional argument choices, with one carve-out:
    /// empty input returns `None` so the bare prompt doesn't visually default to
    /// the first registered command. Tab on empty still cycles through commands,
    /// it just doesn't surface a hint until the user types a character.
    pub fn tab_preview(&self) -> Option<String> {
        let ctx = self.completion_context()?;
        let matches = self.completion_matches(&ctx);
        let first = matches.first()?;
        let prefix_len = ctx.prefix().len();
        if first.len() > prefix_len {
            Some(first[prefix_len..].to_string())
        } else {
            None
        }
    }

    /// Execute a command line. Echoes input, looks up the command, runs it, drains
    /// output. Built-in `help` and `clear` short-circuit before registry lookup.
    pub fn execute(&mut self, line: &str, ctx: &mut Ctx) {
        let line = line.trim();
        if line.is_empty() {
            return;
        }
        self.push_history(HistoryLine::input(format!("> {line}")));

        // Input history (skip consecutive duplicates).
        if self.input_history.back().map(String::as_str) != Some(line) {
            self.input_history.push_back(line.to_string());
            while self.input_history.len() > MAX_INPUT_HISTORY {
                self.input_history.pop_front();
            }
        }
        self.input_history_pos = None;
        self.tab = None;

        let Some((name, args)) = parse_line(line) else {
            return;
        };

        // Built-ins first. Framework-owned, dispatched off the `Builtin` enum so the
        // name + help info isn't duplicated across `execute`, `builtin_help`, and
        // `all_command_names` (single source of truth for the four primitives).
        if let Some(builtin) = Builtin::from_name(&name) {
            self.run_builtin(builtin, args.first().map(String::as_str));
            return;
        }

        if !self.commands.contains_key(&name) {
            self.push_history(HistoryLine::error(format!(
                "no command '{name}'. try: help"
            )));
            return;
        }

        let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
        let mut writer = ConsoleWriter::new();
        let result = {
            let cmd = self.commands.get_mut(&name).expect("checked above");
            cmd.run(&arg_refs, ctx, &mut writer)
        };
        for hl in writer.lines {
            self.push_history(hl);
        }
        if let Err(e) = result {
            self.push_history(HistoryLine::error(format!("error: {e:#}")));
        }
    }

    /// Dispatch one of the framework built-ins. `target` is the optional first-arg
    /// token (only used by `help` to look up a specific command).
    fn run_builtin(&mut self, builtin: Builtin, target: Option<&str>) {
        match builtin {
            Builtin::Help => self.builtin_help(target),
            Builtin::Clear => self.history.clear(),
            Builtin::Detach => {
                self.detached = true;
                self.push_history(HistoryLine::system("console detached"));
            }
            Builtin::Dock => {
                self.detached = false;
                self.push_history(HistoryLine::system("console docked"));
            }
        }
    }

    fn builtin_help(&mut self, target: Option<&str>) {
        match target {
            Some(name) => {
                if let Some(b) = Builtin::from_name(name) {
                    // Built-ins only have one-line descriptions; no multi-line variant.
                    self.push_history(HistoryLine::output(format!("{}: {}", b.name(), b.help())));
                } else {
                    // Materialize the help lines BEFORE pushing to history: `c` borrows
                    // `self.commands` immutably and `push_history` borrows `self` mutably,
                    // so the two can't coexist.
                    let prepared: Option<(String, Vec<String>)> =
                        self.commands.get(name).map(|c| {
                            let header_prefix = format!("{}: ", c.name());
                            let body = c.long_help();
                            let indent = " ".repeat(c.name().len() + 2);
                            let mut lines = body.lines();
                            let first = lines.next().unwrap_or("");
                            let mut rendered = vec![format!("{header_prefix}{first}")];
                            for line in lines {
                                rendered.push(format!("{indent}{line}"));
                            }
                            (c.name().to_string(), rendered)
                        });
                    if let Some((_name, lines)) = prepared {
                        for line in lines {
                            self.push_history(HistoryLine::output(line));
                        }
                    } else {
                        self.push_history(HistoryLine::error(format!("no command '{name}'")));
                    }
                }
            }
            None => {
                self.push_history(HistoryLine::output("commands:"));
                let mut entries: Vec<(String, String)> = self
                    .commands
                    .values()
                    .map(|c| (c.name().to_string(), c.help().to_string()))
                    .collect();
                for b in Builtin::ALL {
                    entries.push((b.name().to_string(), b.help().to_string()));
                }
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                for (name, help) in entries {
                    self.push_history(HistoryLine::output(format!("  {name:16} {help}")));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in commands
// ---------------------------------------------------------------------------

/// Framework-owned commands that mutate `Console` internal state directly: history,
/// detached flag, etc. They can't go through [`Command<Ctx>`] cleanly because that
/// trait only sees `&mut Ctx` (the user's context), not `&mut Console<Ctx>`. Storing
/// their name + help in one enum centralizes what was previously duplicated across
/// [`Console::execute`], [`Console::builtin_help`], and [`Console::all_command_names`].
///
/// User crates cannot add new built-ins; framework primitives only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Builtin {
    Help,
    Clear,
    Detach,
    Dock,
}

impl Builtin {
    /// Iteration order is alphabetical, matching the rest of the console's sort
    /// conventions (Tab cycling, help listing). Keep this slice sorted by `name()`.
    const ALL: &'static [Builtin] = &[
        Builtin::Clear,
        Builtin::Detach,
        Builtin::Dock,
        Builtin::Help,
    ];

    fn from_name(name: &str) -> Option<Builtin> {
        match name {
            "help" => Some(Builtin::Help),
            "clear" => Some(Builtin::Clear),
            "detach" => Some(Builtin::Detach),
            "dock" => Some(Builtin::Dock),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Builtin::Help => "help",
            Builtin::Clear => "clear",
            Builtin::Detach => "detach",
            Builtin::Dock => "dock",
        }
    }

    fn help(self) -> &'static str {
        match self {
            Builtin::Help => "list commands or describe one",
            Builtin::Clear => "clear the scrollback buffer",
            Builtin::Detach => "render as a draggable window",
            Builtin::Dock => "render as a half-screen drop-down (default)",
        }
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Printable text the OS produces for a key, when one exists. Used to strip the
/// corresponding `egui::Event::Text` after consuming the key event for the toggle:
/// without this, pressing Backtick to toggle the console open and pressing it again to
/// toggle closed also types `` ` `` into the input box (the Key event is consumed but
/// the Text event isn't).
///
/// Only the keys that plausibly serve as a console-toggle are covered (Backtick is the
/// default; Tilde is the natural alternate on US layouts). Other keys return `None`;
/// their Text events stay untouched.
fn key_text(key: egui::Key) -> Option<&'static str> {
    match key {
        egui::Key::Backtick => Some("`"),
        _ => None,
    }
}

/// Quote-aware tokenizer. Returns `(command_name, args)` or `None` for an
/// empty/whitespace-only line. Tokens are whitespace-separated; double-quoted
/// (`"..."`) and single-quoted (`'...'`) strings preserve internal whitespace.
/// Inside double quotes, `\"` and `\\` are escapes; inside single quotes, the
/// content is literal (no escapes, matching shell convention).
///
/// Unterminated quotes are tolerated for interactive ergonomics: trailing content
/// after an opening quote with no matching close becomes one token through end of
/// line. This lets mid-typing tab completion work without erroring on the partial
/// quote state.
fn parse_line(line: &str) -> Option<(String, Vec<String>)> {
    let mut tokens = tokenize(line);
    if tokens.is_empty() {
        return None;
    }
    let name = tokens.remove(0);
    Some((name, tokens))
}

/// Quote-aware token splitter. See [`parse_line`] for the grammar.
fn tokenize(line: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut in_token = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' => {
                in_token = true;
                while let Some(&next) = chars.peek() {
                    chars.next();
                    if next == '"' {
                        break;
                    }
                    if next == '\\' {
                        if let Some(&escaped) = chars.peek() {
                            if matches!(escaped, '"' | '\\') {
                                cur.push(escaped);
                                chars.next();
                                continue;
                            }
                        }
                    }
                    cur.push(next);
                }
            }
            '\'' => {
                in_token = true;
                while let Some(&next) = chars.peek() {
                    chars.next();
                    if next == '\'' {
                        break;
                    }
                    cur.push(next);
                }
            }
            c if c.is_whitespace() => {
                if in_token {
                    out.push(std::mem::take(&mut cur));
                    in_token = false;
                }
            }
            c => {
                in_token = true;
                cur.push(c);
            }
        }
    }
    if in_token {
        out.push(cur);
    }
    out
}

/// Splice `choice` into `input` at the position the user is completing, preserving
/// everything before. For command-name completion this replaces the whole input; for
/// argument completion it replaces only the last partial token (or appends after a
/// trailing space when starting a fresh argument).
fn apply_completion(input: &str, ctx: &CompletionContext, choice: &str) -> String {
    match ctx {
        CompletionContext::Command { .. } => choice.to_string(),
        CompletionContext::Arg { .. } => {
            // Preserve the input string verbatim up to the start of the partial token
            // under the cursor, then append the completion choice. Verbatim
            // preservation is important for quoted args -- a re-tokenize-and-rejoin
            // would mangle `tests "5 cell"` into `tests 5 cell` on the rejoin step.
            if input.ends_with(char::is_whitespace) {
                return format!("{input}{choice}");
            }
            let prefix_end = input.rfind(char::is_whitespace).map_or(0, |i| i + 1);
            format!("{}{choice}", &input[..prefix_end])
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    type Ctx = u32;

    fn echo_cmd() -> impl Command<Ctx> {
        cmd("echo", "echo args back", |args, _ctx, out| {
            out.line(args.join(" "));
            Ok(())
        })
    }

    fn add_cmd() -> impl Command<Ctx> {
        cmd("add", "add args to ctx", |args, ctx, out| {
            for a in args {
                let n: u32 = a.parse()?;
                *ctx += n;
            }
            out.line(format!("ctx={ctx}"));
            Ok(())
        })
    }

    #[test]
    fn tab_preview_shows_first_match_suffix() {
        // Use the four built-in commands: clear, detach, dock, help. No additional
        // registrations needed. Sort order: clear < detach < dock < help.
        let mut c = Console::<Ctx>::new();

        // Empty input -> no ghost (don't visually default the bare prompt to the
        // first command; the user hasn't typed anything yet). Tab still works.
        c.input = String::new();
        assert_eq!(c.tab_preview(), None);

        // Single match -> preview is the rest of that command.
        c.input = "de".into();
        assert_eq!(c.tab_preview().as_deref(), Some("tach"));

        c.input = "do".into();
        assert_eq!(c.tab_preview().as_deref(), Some("ck"));

        c.input = "cl".into();
        assert_eq!(c.tab_preview().as_deref(), Some("ear"));

        c.input = "h".into();
        assert_eq!(c.tab_preview().as_deref(), Some("elp"));

        // Multiple matches sharing only a prefix -> previews the first by sort order.
        // `d` matches `detach` and `dock`; alphabetically `detach` is first.
        c.input = "d".into();
        assert_eq!(c.tab_preview().as_deref(), Some("etach"));

        // No match -> no preview.
        c.input = "zzz".into();
        assert_eq!(c.tab_preview(), None);
    }

    #[test]
    fn tab_preview_completes_declared_arg_choices() {
        let mut c = Console::<Ctx>::new();
        c.register(cmd("capture", "", |_, _, _| Ok(())).with_args(&[
            &["png", "frames", "toggle", "stop"],
            &["pre", "post", "both"],
        ]));

        // Mid-arg-0 prefix narrows to a single choice; ghost = its suffix.
        c.input = "capture p".into();
        assert_eq!(c.tab_preview().as_deref(), Some("ng"));

        // Multiple matches with common prefix `to` -> ghost = `ggle` suffix beyond
        // the `t` the user typed (matches: `toggle`, no others starting with `t`).
        c.input = "capture t".into();
        assert_eq!(c.tab_preview().as_deref(), Some("oggle"));

        // Trailing whitespace = starting next arg. Previews the first arg-1 choice
        // alphabetically (`both` < `post` < `pre`).
        c.input = "capture png ".into();
        assert_eq!(c.tab_preview().as_deref(), Some("both"));

        // Mid-arg-1: `po` -> `post`.
        c.input = "capture png po".into();
        assert_eq!(c.tab_preview().as_deref(), Some("st"));

        // Past the declared arg list: no completion.
        c.input = "capture png post extra ".into();
        assert_eq!(c.tab_preview(), None);
    }

    #[test]
    fn two_step_kv_value_completion() {
        let mut c = Console::<Ctx>::new();
        c.register(
            cmd("capture", "", |_, _, _| Ok(()))
                .with_args(&[&["fps=", "palette="]])
                .with_value_choices("palette", &["local", "global"]),
        );

        // Step 1: typing the key prefix completes to `key=` (no value yet).
        c.input = "capture pal".into();
        assert_eq!(c.tab_preview().as_deref(), Some("ette="));
        c.tab_complete();
        assert_eq!(c.input, "capture palette=");

        // Step 2: at the trailing `=`, completion now suggests values.
        // Alphabetical: global < local; first match wins.
        let ctx = c.completion_context().unwrap();
        let matches = c.completion_matches(&ctx);
        assert_eq!(matches, vec!["palette=global", "palette=local"]);

        // Free-form key (no value choices) stops at the bare prefix.
        c.input = "capture fps=".into();
        let ctx = c.completion_context().unwrap();
        let matches = c.completion_matches(&ctx);
        assert!(
            matches.is_empty(),
            "fps= should suggest no values; got {matches:?}"
        );
        assert_eq!(c.tab_preview(), None);
    }

    #[test]
    fn arg_completion_filters_already_used_kv_prefixes() {
        let mut c = Console::<Ctx>::new();
        c.register(cmd("rec", "", |_, _, _| Ok(())).with_args(&[
            &["both", "fps=", "post", "scale="],
            &["fps=", "scale="],
            &["fps=", "scale="],
        ]));

        // Fresh prompt: every choice surfaces.
        c.input = "rec ".into();
        let ctx = c.completion_context().unwrap();
        let m = c.completion_matches(&ctx);
        assert!(m.contains(&"fps=".into()));
        assert!(m.contains(&"scale=".into()));

        // After picking `fps=30`, the next completion shouldn't suggest fps= again.
        c.input = "rec fps=30 ".into();
        let ctx = c.completion_context().unwrap();
        let m = c.completion_matches(&ctx);
        assert!(!m.contains(&"fps=".into()), "got matches: {m:?}");
        assert!(m.contains(&"scale=".into()));

        // Both kv prefixes used -> no kv suggestions remain.
        c.input = "rec fps=30 scale=720 ".into();
        let ctx = c.completion_context().unwrap();
        let m = c.completion_matches(&ctx);
        assert!(m.is_empty(), "got matches: {m:?}");
    }

    #[test]
    fn tab_complete_applies_arg_choice() {
        let mut c = Console::<Ctx>::new();
        c.register(cmd("capture", "", |_, _, _| Ok(())).with_args(&[
            &["png", "frames", "toggle", "stop"],
            &["pre", "post", "both"],
        ]));

        c.input = "capture p".into();
        c.tab_complete();
        assert_eq!(c.input, "capture png");

        c.input = "capture png p".into();
        c.tab_complete();
        // Matches are sorted: `post` < `pre` (o < r at second char). First Tab lands on
        // `post`; pressing Tab again cycles to `pre`.
        assert_eq!(c.input, "capture png post");
        c.tab_complete();
        assert_eq!(c.input, "capture png pre");
    }

    #[test]
    fn parse_line_handles_basic_cases() {
        assert_eq!(parse_line("foo"), Some(("foo".into(), vec![])));
        assert_eq!(
            parse_line("foo bar baz"),
            Some(("foo".into(), vec!["bar".into(), "baz".into()]))
        );
        assert_eq!(
            parse_line("  foo   bar  "),
            Some(("foo".into(), vec!["bar".into()]))
        );
        assert_eq!(parse_line(""), None);
        assert_eq!(parse_line("   "), None);
    }

    #[test]
    fn execute_echoes_input_and_collects_output() {
        let mut c = Console::<Ctx>::new();
        c.register(echo_cmd());
        let mut ctx: Ctx = 0;
        c.execute("echo hello world", &mut ctx);

        let lines: Vec<&HistoryLine> = c.history.iter().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].kind, LineKind::Input);
        assert_eq!(lines[0].text, "> echo hello world");
        assert_eq!(lines[1].kind, LineKind::Output);
        assert_eq!(lines[1].text, "hello world");
    }

    #[test]
    fn execute_mutates_ctx() {
        let mut c = Console::<Ctx>::new();
        c.register(add_cmd());
        let mut ctx: Ctx = 10;
        c.execute("add 5 3", &mut ctx);
        assert_eq!(ctx, 18);
    }

    #[test]
    fn unknown_command_produces_error_line() {
        let mut c = Console::<Ctx>::new();
        let mut ctx: Ctx = 0;
        c.execute("nope", &mut ctx);
        let last = c.history.back().unwrap();
        assert_eq!(last.kind, LineKind::Error);
        assert!(last.text.contains("nope"));
    }

    #[test]
    fn builtin_help_lists_registered_and_builtin() {
        let mut c = Console::<Ctx>::new();
        c.register(echo_cmd());
        let mut ctx: Ctx = 0;
        c.execute("help", &mut ctx);
        let texts: Vec<&str> = c.history.iter().map(|l| l.text.as_str()).collect();
        assert!(texts.iter().any(|t| t.contains("commands:")));
        assert!(texts.iter().any(|t| t.contains("echo")));
        assert!(texts.iter().any(|t| t.contains("help")));
        assert!(texts.iter().any(|t| t.contains("clear")));
    }

    #[test]
    fn builtin_help_describes_one_command() {
        let mut c = Console::<Ctx>::new();
        c.register(echo_cmd());
        let mut ctx: Ctx = 0;
        c.execute("help echo", &mut ctx);
        let last = c.history.back().unwrap();
        assert_eq!(last.kind, LineKind::Output);
        assert!(last.text.contains("echo"));
        assert!(last.text.contains("echo args back"));
    }

    #[test]
    fn builtin_clear_empties_scrollback() {
        let mut c = Console::<Ctx>::new();
        c.register(echo_cmd());
        let mut ctx: Ctx = 0;
        c.execute("echo a", &mut ctx);
        c.execute("echo b", &mut ctx);
        assert!(!c.history.is_empty());
        c.execute("clear", &mut ctx);
        assert!(c.history.is_empty());
    }

    #[test]
    fn input_history_appends_and_dedupes_consecutive() {
        let mut c = Console::<Ctx>::new();
        c.register(echo_cmd());
        let mut ctx: Ctx = 0;
        c.execute("echo a", &mut ctx);
        c.execute("echo a", &mut ctx);
        c.execute("echo b", &mut ctx);
        let h: Vec<&str> = c.input_history.iter().map(String::as_str).collect();
        assert_eq!(h, vec!["echo a", "echo b"]);
    }

    #[test]
    fn input_history_caps_at_max() {
        let mut c = Console::<Ctx>::new();
        c.register(echo_cmd());
        let mut ctx: Ctx = 0;
        for i in 0..(MAX_INPUT_HISTORY + 50) {
            c.execute(&format!("echo n{i}"), &mut ctx);
        }
        assert_eq!(c.input_history.len(), MAX_INPUT_HISTORY);
        assert!(c.input_history.front().unwrap().starts_with("echo n50"));
    }

    #[test]
    fn history_caps_at_max() {
        let mut c = Console::<Ctx>::new();
        c.register(echo_cmd());
        let mut ctx: Ctx = 0;
        // Each execute pushes 2 lines (input + output); push enough to overflow.
        for i in 0..(MAX_HISTORY_LINES + 100) {
            c.execute(&format!("echo {i}"), &mut ctx);
        }
        assert_eq!(c.history.len(), MAX_HISTORY_LINES);
    }

    #[test]
    fn history_prev_walks_backwards_then_history_next_returns_to_blank() {
        let mut c = Console::<Ctx>::new();
        c.register(echo_cmd());
        let mut ctx: Ctx = 0;
        c.execute("echo first", &mut ctx);
        c.execute("echo second", &mut ctx);

        c.history_prev();
        assert_eq!(c.input, "echo second");
        c.history_prev();
        assert_eq!(c.input, "echo first");
        c.history_prev();
        assert_eq!(c.input, "echo first"); // clamped at oldest
        c.history_next();
        assert_eq!(c.input, "echo second");
        c.history_next();
        assert_eq!(c.input, ""); // back to blank input
    }

    #[test]
    fn tab_complete_unique_prefix_completes_immediately() {
        let mut c = Console::<Ctx>::new();
        c.register(echo_cmd());
        c.input.clone_from(&"ec".to_string());
        c.tab_complete();
        assert_eq!(c.input, "echo");
        assert!(c.tab.is_none());
    }

    #[test]
    fn tab_complete_ambiguous_prefix_cycles() {
        let mut c = Console::<Ctx>::new();
        c.register(cmd::<Ctx, _>("capture.start", "x", |_, _, _| Ok(())));
        c.register(cmd::<Ctx, _>("capture.stop", "x", |_, _, _| Ok(())));
        c.register(cmd::<Ctx, _>("capture.toggle", "x", |_, _, _| Ok(())));
        c.input.clone_from(&"capture.s".to_string());

        c.tab_complete();
        assert_eq!(c.input, "capture.start");
        c.tab_complete();
        assert_eq!(c.input, "capture.stop");
        c.tab_complete();
        // capture.toggle starts with "capture.t", not "capture.s", so it isn't in the
        // match set; cycling wraps back to start.
        assert_eq!(c.input, "capture.start");
    }

    #[test]
    fn tab_complete_no_match_is_noop() {
        let mut c = Console::<Ctx>::new();
        c.register(echo_cmd());
        c.input.clone_from(&"xyz".to_string());
        c.tab_complete();
        assert_eq!(c.input, "xyz");
    }

    #[test]
    fn bind_and_unbind() {
        let mut c = Console::<Ctx>::new();
        c.bind(egui::Key::F9, "echo hello");
        assert_eq!(
            c.binds.get(&egui::Key::F9).map(String::as_str),
            Some("echo hello")
        );
        c.unbind(egui::Key::F9);
        assert!(!c.binds.contains_key(&egui::Key::F9));
    }

    #[test]
    fn detach_and_dock_methods_flip_state() {
        let mut c = Console::<Ctx>::new();
        assert!(!c.is_detached());
        c.detach();
        assert!(c.is_detached());
        c.detach(); // idempotent
        assert!(c.is_detached());
        c.dock();
        assert!(!c.is_detached());
        c.dock(); // idempotent
        assert!(!c.is_detached());
    }

    #[test]
    fn builtin_detach_command_flips_state_and_emits_system_line() {
        let mut c = Console::<Ctx>::new();
        let mut ctx: Ctx = 0;
        c.execute("detach", &mut ctx);
        assert!(c.is_detached());
        let last = c.history.back().unwrap();
        assert_eq!(last.kind, LineKind::System);
        assert!(last.text.contains("detached"));
        c.execute("dock", &mut ctx);
        assert!(!c.is_detached());
        let last = c.history.back().unwrap();
        assert_eq!(last.kind, LineKind::System);
        assert!(last.text.contains("docked"));
    }

    #[test]
    fn builtin_help_lists_detach_and_dock() {
        let mut c = Console::<Ctx>::new();
        let mut ctx: Ctx = 0;
        c.execute("help", &mut ctx);
        let texts: Vec<&str> = c.history.iter().map(|l| l.text.as_str()).collect();
        assert!(texts.iter().any(|t| t.contains("detach")));
        assert!(texts.iter().any(|t| t.contains("dock")));
    }

    #[test]
    fn tab_complete_includes_detach_and_dock() {
        let mut c = Console::<Ctx>::new();
        c.input.clone_from(&"de".to_string());
        c.tab_complete();
        assert_eq!(c.input, "detach");
    }

    #[test]
    fn command_returning_err_pushes_error_line() {
        let mut c = Console::<Ctx>::new();
        c.register(cmd("fail", "always fails", |_, _, _| anyhow::bail!("nope")));
        let mut ctx: Ctx = 0;
        c.execute("fail", &mut ctx);
        let last = c.history.back().unwrap();
        assert_eq!(last.kind, LineKind::Error);
        assert!(last.text.contains("nope"));
    }

    // ----------------- SubcommandSet -----------------

    /// Holds a single `Ctx: u32` slot plus a `last_choice` string so tests can verify
    /// which branch ran.
    type SubCtx = (u32, String);

    fn sample_subset() -> SubcommandSet<SubCtx> {
        subcommands::<SubCtx>("tests", "umbrella")
            .toggle("axes", "toggle axes", |c, v| {
                // Bare invocation flips between 1 and 0; explicit on|off sets directly.
                let on = v.unwrap_or(c.0 != 1);
                c.0 = if on { 1 } else { 0 };
                c.1 = format!("axes={on}");
                Ok(())
            })
            .toggle("cube", "toggle cube", |c, v| {
                let on = v.unwrap_or(c.0 != 2);
                c.0 = if on { 2 } else { 0 };
                c.1 = format!("cube={on}");
                Ok(())
            })
            .choice(
                "polytope",
                "set polytope",
                &["5cell", "tesseract", "off"],
                |c, name| {
                    c.1 = format!("polytope={}", name.unwrap_or("<bare>"));
                    Ok(())
                },
            )
    }

    #[test]
    fn subcommand_dispatch_runs_correct_handler() {
        let mut con = Console::<SubCtx>::new();
        con.register(sample_subset());
        let mut ctx: SubCtx = (0, String::new());

        con.execute("tests axes on", &mut ctx);
        assert_eq!(ctx, (1, "axes=true".into()));

        con.execute("tests cube off", &mut ctx);
        assert_eq!(ctx, (0, "cube=false".into()));

        con.execute("tests polytope tesseract", &mut ctx);
        assert_eq!(ctx.1, "polytope=tesseract");
    }

    #[test]
    fn subcommand_toggle_accepts_aliases() {
        let mut con = Console::<SubCtx>::new();
        con.register(sample_subset());
        let mut ctx: SubCtx = (0, String::new());
        for alias in &["on", "true", "1"] {
            con.execute(&format!("tests axes {alias}"), &mut ctx);
            assert_eq!(ctx.1, "axes=true", "alias `{alias}`");
        }
        for alias in &["off", "false", "0"] {
            con.execute(&format!("tests axes {alias}"), &mut ctx);
            assert_eq!(ctx.1, "axes=false", "alias `{alias}`");
        }
    }

    #[test]
    fn subcommand_unknown_subcommand_errors() {
        let mut con = Console::<SubCtx>::new();
        con.register(sample_subset());
        let mut ctx: SubCtx = (0, String::new());
        con.execute("tests xyzzy on", &mut ctx);
        let last = con.history.back().unwrap();
        assert_eq!(last.kind, LineKind::Error);
        assert!(
            last.text.contains("unknown subcommand"),
            "got: {}",
            last.text
        );
    }

    /// Bare toggle invocation (no value) is a flip. The handler receives `None` and
    /// is expected to invert the current field state. Verifies the
    /// "`wireframe nearest-active` flips without explicit on|off" UX path the demos
    /// rely on.
    #[test]
    fn subcommand_toggle_bare_invocation_flips() {
        let mut con = Console::<SubCtx>::new();
        con.register(sample_subset());
        let mut ctx: SubCtx = (0, String::new());
        // First bare invocation: 0 != 1 -> on.
        con.execute("tests axes", &mut ctx);
        assert_eq!(ctx, (1, "axes=true".into()));
        // Second bare invocation: 1 == 1 -> off.
        con.execute("tests axes", &mut ctx);
        assert_eq!(ctx, (0, "axes=false".into()));
    }

    /// Bare choice invocation passes `None` to the handler; the handler decides what
    /// "no value" means (cycle, no-op, error). Pins the framework-level contract that
    /// the handler is invoked at all.
    #[test]
    fn subcommand_choice_bare_invocation_passes_none() {
        let mut con = Console::<SubCtx>::new();
        con.register(sample_subset());
        let mut ctx: SubCtx = (0, String::new());
        con.execute("tests polytope", &mut ctx);
        assert_eq!(ctx.1, "polytope=<bare>");
    }

    /// Bare `SubcommandSet` invocation (no subcommand) calls the registered
    /// `on_bare` handler instead of returning a usage-block error. Verifies the
    /// "`wireframe` flips main on/off" UX path.
    #[test]
    fn subcommand_bare_runs_on_bare_handler() {
        let mut con = Console::<SubCtx>::new();
        con.register(sample_subset().on_bare(|c| {
            c.1 = "bare!".into();
            Ok(())
        }));
        let mut ctx: SubCtx = (0, String::new());
        con.execute("tests", &mut ctx);
        assert_eq!(ctx.1, "bare!");
    }

    /// Without `on_bare`, bare `SubcommandSet` invocation falls back to the usage
    /// block (the historical behavior).
    #[test]
    fn subcommand_bare_without_handler_emits_usage() {
        let mut con = Console::<SubCtx>::new();
        con.register(sample_subset());
        let mut ctx: SubCtx = (0, String::new());
        con.execute("tests", &mut ctx);
        let last = con.history.back().unwrap();
        assert_eq!(last.kind, LineKind::Error);
        assert!(last.text.contains("subcommands"), "got: {}", last.text);
    }

    /// Tab completion at the value slot narrows to ONLY the chosen subcommand's choices.
    /// This is the load-bearing context-aware-completion test: the value slot
    /// completion is narrow to the chosen subcommand's allowed values, not the
    /// union over all subcommands.
    ///
    /// Toggle subcommands deliberately surface NO value-slot suggestions: bare
    /// invocation flips, so `on|off` is supported as input but not promoted in
    /// the cycle (avoids implying that one of them is required when it isn't).
    /// Choice subcommands surface their declared choice list.
    #[test]
    fn subcommand_value_completion_is_context_aware() {
        let mut con = Console::<SubCtx>::new();
        con.register(sample_subset());

        // `tests axes ` -> toggle, no suggestions (bare invocation is the UX;
        // typing `on|off` still works but isn't promoted).
        con.input = "tests axes ".into();
        let ctx = con.completion_context().unwrap();
        let m = con.completion_matches(&ctx);
        assert!(
            m.is_empty(),
            "toggle value slot should suggest nothing, got {m:?}"
        );

        // `tests polytope ` -> only polytope names in the cycle, no on/off.
        con.input = "tests polytope ".into();
        let ctx = con.completion_context().unwrap();
        let m = con.completion_matches(&ctx);
        assert_eq!(
            m,
            vec![
                "5cell".to_string(),
                "off".to_string(),
                "tesseract".to_string()
            ]
        );
        assert!(!m.contains(&"on".into()));
    }

    /// Tab completion at the subcommand slot lists every registered subcommand,
    /// sorted alphabetically (matches the rest of the console's completion convention).
    #[test]
    fn subcommand_first_slot_completion_lists_subcommands() {
        let mut con = Console::<SubCtx>::new();
        con.register(sample_subset());
        con.input = "tests ".into();
        let ctx = con.completion_context().unwrap();
        let m = con.completion_matches(&ctx);
        assert_eq!(
            m,
            vec![
                "axes".to_string(),
                "cube".to_string(),
                "polytope".to_string()
            ]
        );
    }

    // ----------------- SubcommandSet::Custom -----------------

    type CustomCtx = Vec<String>;

    fn custom_subset() -> SubcommandSet<CustomCtx> {
        subcommands::<CustomCtx>("capture", "umbrella")
            // No-arg subcommand.
            .custom("stop", "stop running capture", &[], &[], |c, rest, _out| {
                c.push(format!("stop;rest={}", rest.join(",")));
                Ok(())
            })
            // Single-slot positional subcommand.
            .custom(
                "png",
                "one-shot png",
                &[&["pre", "post", "both"]],
                &[],
                |c, rest, _out| {
                    c.push(format!("png;rest={}", rest.join(",")));
                    Ok(())
                },
            )
            // Multi-slot with kv pairs + enumerable value for one of them.
            .custom(
                "gif",
                "gif sequence",
                &[
                    &["pre", "post", "both"],
                    &["fps=", "palette=", "scale="],
                    &["fps=", "palette=", "scale="],
                ],
                &[("palette", &["local", "global"])],
                |c, rest, _out| {
                    c.push(format!("gif;rest={}", rest.join(",")));
                    Ok(())
                },
            )
    }

    #[test]
    fn custom_subcommand_dispatch_receives_full_rest() {
        let mut con = Console::<CustomCtx>::new();
        con.register(custom_subset());
        let mut ctx: CustomCtx = Vec::new();

        con.execute("capture png post", &mut ctx);
        con.execute("capture gif both fps=30 palette=global", &mut ctx);
        con.execute("capture stop", &mut ctx);

        assert_eq!(
            ctx,
            vec![
                "png;rest=post".to_string(),
                "gif;rest=both,fps=30,palette=global".to_string(),
                "stop;rest=".to_string(),
            ]
        );
    }

    /// Multi-slot tab completion: each positional slot AFTER the subcommand name
    /// returns the slot-specific arg_choices. Slot 0 of `gif` is the stage; slot 1
    /// is the first kv key.
    #[test]
    fn custom_multi_slot_completion_per_slot() {
        let mut con = Console::<CustomCtx>::new();
        con.register(custom_subset());

        // `capture gif ` -> slot 0 of `gif`: stages.
        con.input = "capture gif ".into();
        let ctx = con.completion_context().unwrap();
        let m = con.completion_matches(&ctx);
        assert_eq!(
            m,
            vec!["both".to_string(), "post".to_string(), "pre".to_string()]
        );

        // `capture gif post ` -> slot 1 of `gif`: kv prefixes.
        con.input = "capture gif post ".into();
        let ctx = con.completion_context().unwrap();
        let m = con.completion_matches(&ctx);
        assert!(m.contains(&"fps=".into()));
        assert!(m.contains(&"palette=".into()));
        assert!(m.contains(&"scale=".into()));

        // `capture png ` -> slot 0 of `png`: stages, NOT kv prefixes (those belong
        // to gif).
        con.input = "capture png ".into();
        let ctx = con.completion_context().unwrap();
        let m = con.completion_matches(&ctx);
        assert!(m.contains(&"post".into()));
        assert!(!m.contains(&"fps=".into()), "got: {m:?}");

        // `capture stop ` -> no choices (zero-slot subcommand).
        con.input = "capture stop ".into();
        let ctx = con.completion_context().unwrap();
        let m = con.completion_matches(&ctx);
        assert!(m.is_empty(), "got: {m:?}");
    }

    /// Two-step kv-value completion: after the user types `palette=`, ghost/Tab
    /// should cycle the declared values. This is the context-aware kv path -- the
    /// same `palette=` prefix in a hypothetical non-gif subcommand wouldn't produce
    /// these (gif is the only subcommand that declares `palette` value-choices).
    #[test]
    fn custom_subcommand_kv_value_completion_is_context_aware() {
        let mut con = Console::<CustomCtx>::new();
        con.register(custom_subset());

        con.input = "capture gif post palette=".into();
        let ctx = con.completion_context().unwrap();
        let m = con.completion_matches(&ctx);
        assert_eq!(
            m,
            vec!["palette=global".to_string(), "palette=local".to_string()]
        );
    }

    // ----------------- Quoted-string tokenizer -----------------

    #[test]
    fn tokenize_handles_bare_words() {
        assert_eq!(tokenize("foo bar baz"), vec!["foo", "bar", "baz"]);
        assert_eq!(tokenize("   foo    bar  "), vec!["foo", "bar"]);
        assert_eq!(tokenize(""), Vec::<String>::new());
    }

    #[test]
    fn tokenize_preserves_spaces_in_double_quotes() {
        assert_eq!(
            tokenize(r#"foo "bar baz" qux"#),
            vec!["foo", "bar baz", "qux"]
        );
    }

    #[test]
    fn tokenize_preserves_spaces_in_single_quotes() {
        assert_eq!(tokenize("foo 'bar baz' qux"), vec!["foo", "bar baz", "qux"]);
    }

    #[test]
    fn tokenize_handles_double_quote_escapes() {
        // `\"` -> literal `"`; `\\` -> literal `\`; other `\x` keeps the backslash.
        assert_eq!(
            tokenize(r#"a "he said \"hi\"" b"#),
            vec!["a", r#"he said "hi""#, "b"]
        );
        assert_eq!(tokenize(r#""back\\slash""#), vec![r"back\slash"]);
    }

    #[test]
    fn tokenize_single_quotes_are_literal() {
        // Backslashes inside single quotes are literal (matches shell convention).
        assert_eq!(tokenize(r"'a \n b'"), vec![r"a \n b"]);
    }

    #[test]
    fn tokenize_unterminated_quote_consumes_to_end() {
        // For interactive ergonomics: don't error on unterminated quotes; treat
        // trailing content as one token.
        assert_eq!(
            tokenize(r#"foo "unterminated"#),
            vec!["foo", "unterminated"]
        );
    }

    #[test]
    fn parse_line_routes_quoted_args_to_handler() {
        type Ctx = Vec<String>;
        let mut con = Console::<Ctx>::new();
        con.register(cmd("echoargs", "record args", |args, c: &mut Ctx, _out| {
            for a in args {
                c.push((*a).to_string());
            }
            Ok(())
        }));
        let mut ctx: Ctx = Vec::new();
        con.execute(r#"echoargs "5 cell" off"#, &mut ctx);
        assert_eq!(ctx, vec!["5 cell".to_string(), "off".to_string()]);
    }

    // ----------------- Unified built-ins -----------------

    #[test]
    fn builtin_from_name_round_trips() {
        for b in Builtin::ALL {
            assert_eq!(Builtin::from_name(b.name()), Some(*b));
        }
        assert_eq!(Builtin::from_name("nope"), None);
    }

    #[test]
    fn help_lists_user_commands_and_builtins_sorted() {
        type Ctx = u32;
        let mut con = Console::<Ctx>::new();
        con.register(cmd("zebra", "fast horse", |_, _, _| Ok(())));
        con.register(cmd("alpha", "first letter", |_, _, _| Ok(())));
        let mut ctx: Ctx = 0;
        con.execute("help", &mut ctx);
        let texts: Vec<&str> = con.history.iter().map(|h| h.text.as_str()).collect();
        let i_alpha = texts.iter().position(|t| t.contains("alpha")).unwrap();
        let i_clear = texts.iter().position(|t| t.contains("clear")).unwrap();
        let i_zebra = texts.iter().position(|t| t.contains("zebra")).unwrap();
        // Alphabetical: alpha < clear < zebra.
        assert!(i_alpha < i_clear);
        assert!(i_clear < i_zebra);
    }

    #[test]
    fn clear_builtin_empties_history() {
        type Ctx = u32;
        let mut con = Console::<Ctx>::new();
        let mut ctx: Ctx = 0;
        con.push_history(HistoryLine::output("first"));
        con.push_history(HistoryLine::output("second"));
        con.execute("clear", &mut ctx);
        assert!(con.history.is_empty());
    }

    #[test]
    fn detach_dock_builtins_flip_flag() {
        type Ctx = u32;
        let mut con = Console::<Ctx>::new();
        let mut ctx: Ctx = 0;
        assert!(!con.detached);
        con.execute("detach", &mut ctx);
        assert!(con.detached);
        con.execute("dock", &mut ctx);
        assert!(!con.detached);
    }
}
