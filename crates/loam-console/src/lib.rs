//! Quake-style developer console: command registry, scrollback, hotkey binding
//! and tab autocomplete, with no UI-framework dependency. Interaction model
//! follows the idTech console (Quake, 1996).
//!
//! [`Console`] is generic over a `Ctx` type so consuming crates choose what state
//! commands operate on. Built-in commands that depend on app/runtime state live in
//! `loam-app`; this crate ships only `help`, `clear`, `detach`, `dock`.
//!
//! A frontend owns presentation and input plumbing: it translates its key events
//! into [`Key`], calls the [`Console`] frontend hooks to move the console's
//! state, and paints [`Console::history`] and the input line. `loam-egui` is the
//! frontend in this workspace.

use std::collections::{BTreeMap, HashMap, VecDeque};

mod key;

pub use key::Key;

/// Scrollback line cap. Older lines drop when the buffer exceeds this.
pub const MAX_HISTORY_LINES: usize = 2000;

/// Input-history cap (Up/Down nav).
pub const MAX_INPUT_HISTORY: usize = 100;

/// Echo new scrollback lines to the browser DevTools console via direct
/// `console.log` (not `tracing`; see `push_history` for the feedback-loop
/// rationale). Off by default; toggled via `loam_app::log`'s `log echo` subcommand.
#[cfg(target_arch = "wasm32")]
static ECHO_TO_BROWSER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Enable / disable scrollback echo to the browser DevTools console (wasm32
/// only); no-op on native so demos can call it unconditionally.
pub fn set_console_echo(enabled: bool) {
    #[cfg(target_arch = "wasm32")]
    ECHO_TO_BROWSER.store(enabled, std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(target_arch = "wasm32"))]
    let _ = enabled;
}

/// Current scrollback-echo state. Always `false` on native.
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

/// Classifies a scrollback line so the frontend can color it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineKind {
    /// User-typed input echoed back.
    Input,
    /// Command-produced output.
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
/// [`ConsoleWriter::error`]; the console drains them into scrollback after the command
/// returns. The two-phase design sidesteps the borrow conflict between the command's
/// mutable registry slot and the console's mutable scrollback during one `execute`.
pub struct ConsoleWriter {
    lines: Vec<HistoryLine>,
}

impl ConsoleWriter {
    /// An empty writer. `pub` so command handlers can be unit-tested against a
    /// fresh writer without standing up a full `Console` and `Ctx`.
    pub fn new() -> Self {
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

impl Default for ConsoleWriter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Command trait + closure shim
// ---------------------------------------------------------------------------

/// Console command implementation. Generic over a `Ctx` type so the consuming crate
/// decides what state commands can mutate.
pub trait Command<Ctx>: 'static {
    /// The name typed at the prompt. Conventionally lowercase, dotted for namespacing
    /// (`capture.start`).
    fn name(&self) -> &str;

    /// One-line description shown by `help` when listing every command.
    fn help(&self) -> &str;

    /// Multi-line help shown by `help <name>`. Default returns the one-line
    /// [`Self::help`]; override for commands with a richer surface. `\n` breaks paint
    /// as separate scrollback entries. Owned `String` so subcommand-dispatching
    /// commands can build the listing from their children.
    fn long_help(&self) -> String {
        self.help().to_string()
    }

    /// Tab-completion choices for the `arg_index`-th positional argument. Default is
    /// empty (free-form arg). Override via [`FnCommand::with_args`] for a fixed enum,
    /// or include a `key=` entry for a key-value arg whose values come from
    /// [`Command::arg_value_choices`]. Subcommand-style commands override
    /// [`Command::arg_choices_ctx`] instead.
    fn arg_choices(&self, arg_index: usize) -> &[&'static str] {
        let _ = arg_index;
        &[]
    }

    /// Context-aware variant of [`Command::arg_choices`]. `prior` carries the arg
    /// tokens parsed before the cursor (`prior.len() == arg_index`). Default delegates
    /// to [`Command::arg_choices`]; [`SubcommandSet`] overrides it to gate value-slot
    /// choices on the selected subcommand. The explicit `'a` ties the returned slice
    /// to `&self` past the nested-reference elision.
    fn arg_choices_ctx<'a>(&'a self, arg_index: usize, prior: &[&str]) -> &'a [&'static str] {
        let _ = prior;
        self.arg_choices(arg_index)
    }

    /// Enumerable values for a `key=value` arg at `arg_index` with key `key` (no
    /// trailing `=`). Drives two-step tab completion; empty means free-form.
    fn arg_value_choices(&self, arg_index: usize, key: &str) -> &[&'static str] {
        let _ = (arg_index, key);
        &[]
    }

    /// Context-aware variant of [`Command::arg_value_choices`]; routes kv-value
    /// lookups to the active subcommand's table. Default delegates to
    /// [`Command::arg_value_choices`].
    fn arg_value_choices_ctx<'a>(
        &'a self,
        arg_index: usize,
        key: &str,
        prior: &[&str],
    ) -> &'a [&'static str] {
        let _ = prior;
        self.arg_value_choices(arg_index, key)
    }

    /// Run the command. `args` are tokens after the command name. Output goes to
    /// `out`; recoverable issues get `out.error(..)`, unrecoverable ones return `Err`.
    fn run(&mut self, args: &[&str], ctx: &mut Ctx, out: &mut ConsoleWriter) -> anyhow::Result<()>;
}

/// Closure-backed [`Command`] implementation. Use [`cmd`] to construct.
pub struct FnCommand<F> {
    name: &'static str,
    help: &'static str,
    /// Multi-line [`Command::long_help`] text; `None` falls back to `help`.
    long_help: Option<&'static str>,
    arg_choices: Vec<Vec<&'static str>>,
    /// Per-key value choices for `key=value` args, keyed by the bare key (no `=`).
    value_choices: HashMap<&'static str, Vec<&'static str>>,
    f: F,
}

impl<F> FnCommand<F> {
    /// Declare positional-argument choices for tab-completion, one inner slice per
    /// position. Trailing free-form args can be omitted.
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

    /// Declare enumerable values for a `key=value` arg: first Tab completes to
    /// `key=`, subsequent Tabs cycle these values. Free-form args (`fps=N`) skip
    /// this and only surface the bare `key=`.
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

    /// Attach a multi-line help block returned by [`Command::long_help`]. Newlines
    /// paint as separate scrollback entries.
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

/// Build a [`Command`] from a closure that mutates `Ctx` and writes to the
/// [`ConsoleWriter`]. Idiomatic for inline per-demo registrations.
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

/// Boxed handler for an on/off toggle subcommand. `Some(bool)` when the user supplied
/// `on|off|true|false|1|0`, `None` on bare invocation (the handler flips the field).
type ToggleHandler<Ctx> = Box<dyn FnMut(&mut Ctx, Option<bool>) -> anyhow::Result<()>>;

/// Boxed handler for a fixed-choice subcommand. `Some(value)` when supplied, `None` on
/// bare invocation (the handler typically cycles to the next choice).
type ChoiceHandler<Ctx> = Box<dyn FnMut(&mut Ctx, Option<&str>) -> anyhow::Result<()>>;

/// Boxed handler for a `SubcommandSet`'s bare invocation, set via
/// [`SubcommandSet::on_bare`]; replaces the default usage-block error.
type BareHandler<Ctx> = Box<dyn FnMut(&mut Ctx) -> anyhow::Result<()>>;

/// Boxed handler for a custom-grammar subcommand. Receives the raw args after the
/// subcommand name (unparsed) plus the writer. For grammars that don't fit
/// `.toggle` / `.choice` (e.g. `capture gif post fps=30 scale=720`).
type CustomHandler<Ctx> =
    Box<dyn FnMut(&mut Ctx, &[&str], &mut ConsoleWriter) -> anyhow::Result<()>>;

/// One entry in a [`SubcommandSet`]. The kind decides value-slot parsing and tab
/// completion.
enum SubcommandKind<Ctx> {
    /// On/off subcommand; no value-slot completion (bare-flip is the canonical UX,
    /// explicit `on|off` accepted but not promoted).
    Toggle { handler: ToggleHandler<Ctx> },
    /// Fixed-choice subcommand; completes the value slot from `choices`.
    Choice {
        choices: Vec<&'static str>,
        handler: ChoiceHandler<Ctx>,
    },
    /// Custom-grammar subcommand. Per-slot positional choices drive completion (slot 0
    /// is the first arg after the subcommand name); per-key enumerables drive two-step
    /// kv completion. The handler owns arg parsing.
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

/// A command that dispatches to named subcommands by the first positional arg, with
/// typed dispatch and context-aware tab completion (the value slot narrows to the
/// chosen subcommand). Build with [`subcommands`] and chain
/// [`SubcommandSet::toggle`] / [`SubcommandSet::choice`].
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
    /// BTreeMap so iteration and Tab cycling are alphabetical.
    subs: BTreeMap<&'static str, SubcommandEntry<Ctx>>,
    /// Sorted subcommand names, populated lazily so the builders stay infallible.
    name_cache: std::cell::OnceCell<Vec<&'static str>>,
    /// Bare-invocation handler set via [`Self::on_bare`]; replaces the usage-block
    /// error so the command name alone can flip a primary field.
    bare: Option<BareHandler<Ctx>>,
}

impl<Ctx: 'static> SubcommandSet<Ctx> {
    /// Register an on/off subcommand. The value slot parses
    /// `on|off|true|false|1|0` to `Some(bool)`; bare invocation passes `None` so the
    /// handler flips the field in place.
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

    /// Register a fixed-choice subcommand. The value slot completes from `choices`;
    /// the handler receives the raw value string (not validated against `choices`).
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

    /// Register a custom-grammar subcommand for multiple positionals, kv pairs, or
    /// both. `arg_choices[i]` lists completion choices for the i-th arg after the
    /// subcommand name (include `key=` entries for kv prefixes); `value_choices[k]`
    /// lists enumerable values for key `k`; `handler` owns parsing of the raw args.
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

    /// Attach a bare-invocation handler: typing just the command name runs `handler`
    /// instead of returning a usage error (e.g. `wireframe` flips the overlay on/off).
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

/// Build a [`SubcommandSet`]; see its docs for the builder pattern.
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
        // arg_index 1 is the first arg after the subcommand name = slot 0 of the
        // subcommand's own grammar.
        let sub_slot = arg_index - 1;
        match &entry.kind {
            // Empty by design: bare-flip is the canonical UX, `on|off` accepted but
            // not surfaced.
            SubcommandKind::Toggle { .. } => &[],
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
            // Bare invocation: run `on_bare` if set, else emit a usage block.
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
/// and open/close state. Register commands and binds at setup, then let a frontend
/// drive it once per frame.
///
/// ```ignore
/// let mut console = Console::<MyCtx>::new();
/// console.register(cmd("hello", "say hi", |_, _, out| {
///     out.line("hi");
///     Ok(())
/// }));
/// console.bind(Key::F9, "capture.toggle");
///
/// // per frame, through the frontend's entry point (loam_egui::ConsoleUi):
/// console.ui(&egui_ctx, &mut my_ctx);
/// ```
///
/// The methods grouped under `Frontend hooks` below are the surface a frontend
/// drives; command authors need none of them.
pub struct Console<Ctx> {
    commands: BTreeMap<String, Box<dyn Command<Ctx>>>,
    /// BTreeMap, not HashMap: several bound keys can land in one frame and the
    /// commands they run must fire in a fixed order.
    binds: BTreeMap<Key, String>,
    toggle_key: Key,
    history: VecDeque<HistoryLine>,
    input: String,
    input_history: VecDeque<String>,
    /// `Some(i)` while cycling history with Up/Down; `None` otherwise.
    input_history_pos: Option<usize>,
    /// Active tab-completion cycle; cleared on any non-tab input edit.
    tab: Option<TabState>,
    open: bool,
    /// True for the frame after `open` becomes true so the frontend requests focus once.
    pending_focus: bool,
    /// Right-aligned title-row text; host-supplied, empty by default.
    status: String,
    /// `false` for the docked drop-down, `true` for a draggable window.
    /// Detached mode lets the user click out to return keyboard focus to the app,
    /// which the docked console captures while open.
    detached: bool,
    /// Set in docked mode on a click outside the panel; suppresses the focus
    /// re-request so input goes to the app while the console stays visible. Cleared
    /// by clicking back inside or reopening.
    user_defocused: bool,
    /// One-frame flag set when code outside the text field mutates [`Self::input`]
    /// (tab-complete, history nav); the frontend then snaps the cursor to the tail.
    pending_cursor_to_end: bool,
}

struct TabState {
    matches: Vec<String>,
    index: usize,
    ctx: CompletionContext,
}

/// What the user is currently typing, partitioned for completion. `prefix` is the
/// partial token under the cursor; empty `prefix` with trailing whitespace means a
/// fresh token is starting.
#[derive(Clone, Debug)]
enum CompletionContext {
    /// Completing the command name.
    Command { prefix: String },
    /// Completing positional arg `arg_index` of `cmd_name`. `prior` carries the
    /// arg tokens before the cursor (`prior.len() == arg_index`), read via
    /// [`Command::arg_choices_ctx`] for subcommand-aware completion.
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
    /// Empty console with the default `` ` `` toggle key.
    pub fn new() -> Self {
        Self {
            commands: BTreeMap::new(),
            binds: BTreeMap::new(),
            toggle_key: Key::Backtick,
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

    /// Override the toggle key. Default is [`Key::Backtick`].
    pub fn with_toggle_key(mut self, key: Key) -> Self {
        self.toggle_key = key;
        self
    }

    /// Register a command. Silently replaces any existing command of the same name;
    /// pre-check via [`Console::has_command`] if needed.
    pub fn register<C: Command<Ctx> + 'static>(&mut self, command: C) {
        let name = command.name().to_string();
        self.commands.insert(name, Box::new(command));
    }

    /// True if a command with this name is registered. `help` and `clear` are built in
    /// and always return true.
    pub fn has_command(&self, name: &str) -> bool {
        name == "help" || name == "clear" || self.commands.contains_key(name)
    }

    /// Bind `key` (no modifiers) to run `command_line` when the console is closed.
    /// Re-binding overwrites.
    pub fn bind(&mut self, key: Key, command_line: impl Into<String>) {
        self.binds.insert(key, command_line.into());
    }

    /// Remove a bind. No-op if the key wasn't bound.
    pub fn unbind(&mut self, key: Key) {
        self.binds.remove(&key);
    }

    /// Open the console. Idempotent.
    pub fn open(&mut self) {
        if !self.open {
            self.open = true;
            self.pending_focus = true;
            // Clear any prior click-outside defocus so typing lands in the input.
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

    /// Render as a draggable window instead of the drop-down. Idempotent.
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

    /// Append a line to the scrollback, for messages generated outside command
    /// execution (e.g. a background recording finishing).
    pub fn write(&mut self, line: HistoryLine) {
        self.push_history(line);
    }

    /// Set the right-aligned title-row status text.
    pub fn set_status(&mut self, text: impl Into<String>) {
        self.status = text.into();
    }

    // -----------------------------------------------------------------
    // Frontend hooks
    // -----------------------------------------------------------------

    /// Key that opens and closes the console.
    pub fn toggle_key(&self) -> Key {
        self.toggle_key
    }

    /// Bound keys and the command lines they run, in [`Key`] order. Binds fire
    /// only while the console is closed, so they never hijack typing.
    pub fn binds(&self) -> impl Iterator<Item = (Key, &str)> {
        self.binds.iter().map(|(key, line)| (*key, line.as_str()))
    }

    /// Scrollback, oldest first.
    pub fn history(&self) -> &VecDeque<HistoryLine> {
        &self.history
    }

    /// The live input line.
    pub fn input(&self) -> &str {
        &self.input
    }

    /// The live input line, for a frontend text field to edit in place. Any edit
    /// made through this handle must be followed by [`Console::cancel_tab_cycle`].
    pub fn input_mut(&mut self) -> &mut String {
        &mut self.input
    }

    /// Right-aligned title-row status text.
    pub fn status(&self) -> &str {
        &self.status
    }

    /// Run the current input line and clear it.
    pub fn submit(&mut self, ctx: &mut Ctx) {
        let line = std::mem::take(&mut self.input);
        self.execute(&line, ctx);
    }

    /// Empty the scrollback.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Discard the input line and any history / completion cycle in progress.
    pub fn clear_input(&mut self) {
        self.input.clear();
        self.input_history_pos = None;
        self.tab = None;
    }

    /// Invalidate an in-progress Tab cycle. Call after any input edit that did not
    /// come from [`Console::tab_complete`].
    pub fn cancel_tab_cycle(&mut self) {
        self.tab = None;
    }

    /// Take the one-shot focus request raised by [`Console::open`].
    pub fn take_pending_focus(&mut self) -> bool {
        std::mem::take(&mut self.pending_focus)
    }

    /// True while the console should hold keyboard focus every frame: the docked
    /// panel is modal by design, until the user clicks out to the app.
    pub fn wants_persistent_focus(&self) -> bool {
        !self.detached && !self.user_defocused
    }

    /// Record whether the user clicked outside the docked panel; `true` releases
    /// focus to the app while the console stays visible.
    pub fn set_user_defocused(&mut self, defocused: bool) {
        self.user_defocused = defocused;
    }

    /// Take the one-shot request to snap the text cursor to the end of the input,
    /// raised when tab-complete or history nav replaced the buffer.
    pub fn take_pending_cursor_to_end(&mut self) -> bool {
        std::mem::take(&mut self.pending_cursor_to_end)
    }

    /// Walk one step back through the input history (Up).
    pub fn history_prev(&mut self) {
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

    /// Walk one step forward through the input history (Down); past the newest
    /// entry the input returns to blank.
    pub fn history_next(&mut self) {
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

    /// Apply the next tab completion, starting or advancing the match cycle.
    pub fn tab_complete(&mut self) {
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
        // Fresh completion; fall back to the empty-prefix command list so Tab on a
        // blank prompt cycles commands.
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

    // -----------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------

    fn push_history(&mut self, line: HistoryLine) {
        // Optional browser-console echo, off by default. Bypasses `tracing` on
        // purpose: routing through it would feed back via `loam_app::log::ConsoleLayer`
        // (tracing -> scrollback -> re-echo -> ...). Direct console.log has no Rust
        // subscriber, so no loop.
        #[cfg(target_arch = "wasm32")]
        if ECHO_TO_BROWSER.load(std::sync::atomic::Ordering::Relaxed) {
            web_sys::console::log_1(&line.text.as_str().into());
        }
        self.history.push_back(line);
        while self.history.len() > MAX_HISTORY_LINES {
            self.history.pop_front();
        }
    }

    fn all_command_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.commands.keys().cloned().collect();
        names.extend(Builtin::ALL.iter().map(|b| b.name().to_string()));
        names.sort();
        names
    }

    /// Decide what the input is completing: command name or n-th positional arg.
    /// `None` for empty input. Quote-aware via [`tokenize`] so `tests "5 cell" o<Tab>`
    /// completes arg 1 with prefix `o`.
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

        // On an argument now (0-based `arg_index`). `prior` captures the arg tokens
        // before the cursor for subcommand-aware value-slot completion.
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

                // Mid-`key=` value completion: an `=` in the partial token means we're
                // completing the value; return enumerables prefixed with `key=`.
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

                // Collect `key=` prefixes already used in earlier positions so we
                // don't re-suggest them. Skips the partial last token; bare
                // positionals aren't filtered (re-typing may be intentional).
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

                // Sorted so authors can declare choices in any order; context-aware
                // so subcommand dispatch gates value-slot choices on the prior pick.
                let mut matches: Vec<String> = cmd
                    .arg_choices_ctx(*arg_index, &prior_refs)
                    .iter()
                    .filter(|choice| choice.starts_with(prefix.as_str()))
                    .filter(|choice| {
                        // Suppress any choice whose `key=` prefix was already used;
                        // bare keywords (no `=`) are never filtered.
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

    /// Suffix of the first (sort-order) completion matching the input, painted as
    /// dim ghost text showing what the next `Tab` inserts. Empty input returns `None`
    /// so the bare prompt doesn't default to the first command.
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

    /// Execute a command line: echo, look up, run, drain output. Built-ins
    /// short-circuit before registry lookup.
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

        // Built-ins first, off the `Builtin` enum (single source of truth for name +
        // help across `execute`, `builtin_help`, `all_command_names`).
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

    /// Dispatch a framework built-in. `target` is the optional first-arg token (only
    /// `help` uses it).
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
                    self.push_history(HistoryLine::output(format!("{}: {}", b.name(), b.help())));
                } else {
                    // Materialize before pushing: `c` borrows `self.commands` and
                    // `push_history` borrows `self` mutably.
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

/// Framework-owned commands that mutate `Console` state directly (history, detached
/// flag). They can't go through [`Command<Ctx>`], which only sees `&mut Ctx`. One enum
/// centralizes their name + help; user crates can't add new built-ins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Builtin {
    Help,
    Clear,
    Detach,
    Dock,
}

impl Builtin {
    /// Keep sorted by `name()`; iteration order is alphabetical to match Tab cycling
    /// and the help listing.
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

/// Split a line into `(command_name, args)` or `None` if empty. Quote-aware:
/// double quotes honor `\"` / `\\` escapes, single quotes are literal (shell
/// convention). Unterminated quotes consume to end-of-line for interactive typing.
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

/// Splice `choice` into `input` at the completion position. Command-name completion
/// replaces the whole input; argument completion replaces the last partial token (or
/// appends after a trailing space).
fn apply_completion(input: &str, ctx: &CompletionContext, choice: &str) -> String {
    match ctx {
        CompletionContext::Command { .. } => choice.to_string(),
        CompletionContext::Arg { .. } => {
            // Preserve the input verbatim up to the partial token; a
            // tokenize-and-rejoin would mangle quoted args like `tests "5 cell"`.
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
        // Built-ins, sorted: clear < detach < dock < help.
        let mut c = Console::<Ctx>::new();

        // Empty input -> no ghost; the bare prompt shouldn't default to a command.
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

        // `d` matches `detach` and `dock`; first by sort order is `detach`.
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

        // `t` -> only `toggle`; ghost = `oggle`.
        c.input = "capture t".into();
        assert_eq!(c.tab_preview().as_deref(), Some("oggle"));

        // Trailing whitespace = next arg; first arg-1 choice (`both` < `post` < `pre`).
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

        // Invariant: one Input then one Output line; Input echoes the typed text,
        // Output carries the joined args. Exact prompt prefix is not pinned.
        let lines: Vec<&HistoryLine> = c.history.iter().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].kind, LineKind::Input);
        assert!(
            lines[0].text.contains("echo hello world"),
            "Input line should include the user's typed text, got: {:?}",
            lines[0].text
        );
        assert_eq!(lines[1].kind, LineKind::Output);
        assert!(
            lines[1].text.contains("hello world"),
            "Output line should contain echo's joined args, got: {:?}",
            lines[1].text
        );
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
        c.bind(Key::F9, "echo hello");
        assert_eq!(
            c.binds.get(&Key::F9).map(String::as_str),
            Some("echo hello")
        );
        c.unbind(Key::F9);
        assert!(!c.binds.contains_key(&Key::F9));
    }

    /// Binds are handed to the frontend in `Key` order so several keys pressed in
    /// one frame always run their commands in the same sequence.
    #[test]
    fn binds_are_enumerated_in_key_order() {
        let mut c = Console::<Ctx>::new();
        c.bind(Key::F12, "third");
        c.bind(Key::Backtick, "first");
        c.bind(Key::F1, "second");
        let seen: Vec<(Key, &str)> = c.binds().collect();
        assert_eq!(
            seen,
            vec![
                (Key::Backtick, "first"),
                (Key::F1, "second"),
                (Key::F12, "third"),
            ]
        );
    }

    /// Rebinding a key replaces the command line rather than accumulating.
    #[test]
    fn rebinding_a_key_replaces_the_command_line() {
        let mut c = Console::<Ctx>::new();
        c.bind(Key::F9, "one");
        c.bind(Key::F9, "two");
        let seen: Vec<(Key, &str)> = c.binds().collect();
        assert_eq!(seen, vec![(Key::F9, "two")]);
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

    // ----------------- Frontend hooks -----------------

    /// `submit` runs the input line and leaves the prompt empty, which is what the
    /// frontend's Enter handler relies on.
    #[test]
    fn submit_runs_the_input_line_and_clears_it() {
        let mut c = Console::<Ctx>::new();
        c.register(add_cmd());
        let mut ctx: Ctx = 1;
        *c.input_mut() = "add 4".into();
        c.submit(&mut ctx);
        assert_eq!(ctx, 5);
        assert!(c.input.is_empty());
    }

    /// Submitting a blank line is inert: no scrollback, no input-history entry.
    #[test]
    fn submit_of_blank_input_is_inert() {
        let mut c = Console::<Ctx>::new();
        let mut ctx: Ctx = 0;
        *c.input_mut() = "   ".into();
        c.submit(&mut ctx);
        assert!(c.history.is_empty());
        assert!(c.input_history.is_empty());
    }

    /// Both one-shot flags are consumed on read, so a frontend that polls every
    /// frame acts on them exactly once.
    #[test]
    fn one_shot_frontend_flags_are_consumed_on_take() {
        let mut c = Console::<Ctx>::new();
        c.open();
        assert!(c.take_pending_focus());
        assert!(!c.take_pending_focus());

        c.register(echo_cmd());
        let mut ctx: Ctx = 0;
        c.execute("echo a", &mut ctx);
        c.history_prev();
        assert!(c.take_pending_cursor_to_end());
        assert!(!c.take_pending_cursor_to_end());
    }

    /// The docked console holds focus every frame until the user clicks out;
    /// detaching hands focus back unconditionally.
    #[test]
    fn persistent_focus_is_docked_and_not_user_defocused() {
        let mut c = Console::<Ctx>::new();
        assert!(c.wants_persistent_focus());
        c.set_user_defocused(true);
        assert!(!c.wants_persistent_focus());
        c.set_user_defocused(false);
        c.detach();
        assert!(!c.wants_persistent_focus());
        // Reopening clears a stale click-outside so typing lands in the input.
        c.dock();
        c.set_user_defocused(true);
        c.open();
        assert!(c.wants_persistent_focus());
    }

    /// `clear_input` drops the prompt, the history cursor and the Tab cycle
    /// together; a leftover cycle would splice into the next keystroke.
    #[test]
    fn clear_input_drops_prompt_history_cursor_and_tab_cycle() {
        let mut c = Console::<Ctx>::new();
        c.register(cmd::<Ctx, _>("capture.start", "x", |_, _, _| Ok(())));
        c.register(cmd::<Ctx, _>("capture.stop", "x", |_, _, _| Ok(())));
        let mut ctx: Ctx = 0;
        c.execute("capture.stop", &mut ctx);
        c.history_prev();
        c.input = "capture.s".into();
        c.tab_complete();
        assert!(c.tab.is_some());

        c.clear_input();
        assert!(c.input.is_empty());
        assert!(c.tab.is_none());
        assert!(c.input_history_pos.is_none());
    }

    /// `clear_history` empties the scrollback without touching input history.
    #[test]
    fn clear_history_leaves_input_history_intact() {
        let mut c = Console::<Ctx>::new();
        c.register(echo_cmd());
        let mut ctx: Ctx = 0;
        c.execute("echo a", &mut ctx);
        c.clear_history();
        assert!(c.history.is_empty());
        assert_eq!(c.input_history.len(), 1);
    }

    /// The default toggle key is the Quake backtick, and `with_toggle_key`
    /// overrides it.
    #[test]
    fn toggle_key_defaults_to_backtick_and_is_overridable() {
        let c = Console::<Ctx>::new();
        assert_eq!(c.toggle_key(), Key::Backtick);
        let c = Console::<Ctx>::new().with_toggle_key(Key::F1);
        assert_eq!(c.toggle_key(), Key::F1);
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

    /// Bare toggle invocation passes `None`; the handler inverts the field.
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

    /// Bare choice invocation passes `None` to the handler (it's still invoked).
    #[test]
    fn subcommand_choice_bare_invocation_passes_none() {
        let mut con = Console::<SubCtx>::new();
        con.register(sample_subset());
        let mut ctx: SubCtx = (0, String::new());
        con.execute("tests polytope", &mut ctx);
        assert_eq!(ctx.1, "polytope=<bare>");
    }

    /// Bare `SubcommandSet` invocation calls the `on_bare` handler, not the usage block.
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

    /// Without `on_bare`, bare invocation falls back to the usage block.
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

    /// Value-slot completion narrows to the chosen subcommand's choices, not the
    /// union. Toggles surface nothing (bare-flip UX); choices surface their list.
    #[test]
    fn subcommand_value_completion_is_context_aware() {
        let mut con = Console::<SubCtx>::new();
        con.register(sample_subset());

        // `tests axes ` -> toggle, no suggestions.
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

    /// The subcommand slot lists every registered subcommand, sorted.
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

    /// Each positional slot after the subcommand name returns its own arg_choices.
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

    /// After `palette=`, completion cycles the declared values; the kv lookup is
    /// scoped to the subcommand that declared them.
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
