//! Quake-style developer console: half-screen drop-down overlay with a
//! command registry, scrollback, hotkey binding, and tab autocomplete.
//!
//! The interaction model follows the idTech console (Quake, 1996): a
//! drop-down activated by `` ` ``, monospace scrollback above an input
//! line, history navigated with Up/Down, completion via Tab, hotkey
//! binds for arbitrary command lines.
//!
//! ## What lives here vs what doesn't
//!
//! - **Here**: [`Console`] (the main type), [`Command`] trait + [`cmd`]
//!   closure shim, [`ConsoleWriter`] (output collector), key handling
//!   for the input line, the parser. `Console` is generic over a `Ctx`
//!   type so consuming crates choose what state commands operate on.
//! - **Not here**: built-in commands (`screenshot`, `capture.start`,
//!   `bind`, `quit`). Those depend on app/runtime state and live in
//!   `rye-app::builtins`. This module ships only `help` and `clear`,
//!   which need only `Console` itself.
//!
//! ## Why egui consumes the keys before TextEdit sees them
//!
//! Egui's `TextEdit::singleline` swallows printable characters into
//! the buffer and uses `Tab` to move focus. Without explicit
//! interception, the toggle key (`` ` ``) types a backtick into the
//! input the moment the console opens, and `Tab` shifts focus off the
//! input box. The key handler at the top of [`Console::ui`] uses
//! `egui::InputState::consume_key` to claim the keystrokes before
//! TextEdit's per-frame processing runs. Up/Down/Tab/Esc/Ctrl+L/Ctrl+C
//! are intercepted the same way; Enter is detected via the TextEdit
//! response in the panel module so we still get `lost_focus`-on-submit
//! semantics.
//!
//! ## Constants
//!
//! Tunables are module-level `const`s ([`MAX_HISTORY_LINES`],
//! [`MAX_INPUT_HISTORY`], [`ANIM_DURATION_SECS`],
//! [`PANEL_HEIGHT_FRACTION`]) rather than runtime config; the values are
//! UX choices, not deployment knobs.

use std::collections::{BTreeMap, HashMap, VecDeque};

mod panel;

/// Scrollback line cap. Older lines drop when the buffer exceeds this.
/// Sized for a session's worth of debugging without unbounded memory.
pub const MAX_HISTORY_LINES: usize = 2000;

/// Input-history cap (Up/Down nav). 100 covers a typical session;
/// larger and old entries become noise during cycling.
pub const MAX_INPUT_HISTORY: usize = 100;

/// Slide-down animation duration. 0.15s is fast enough to feel
/// responsive, slow enough to read as motion.
pub const ANIM_DURATION_SECS: f32 = 0.15;

/// Fraction of the viewport height the open console occupies. 0.5 is
/// the Quake convention: enough scrollback visible, scene visible
/// below.
pub const PANEL_HEIGHT_FRACTION: f32 = 0.5;

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

/// Per-invocation output sink. Commands push lines via
/// [`ConsoleWriter::line`] / [`ConsoleWriter::error`]; the console
/// drains the collected lines into the scrollback after the command
/// returns.
///
/// The two-phase design (command writes to local Vec, console drains)
/// avoids the borrow conflict between the command's mutable access to
/// the registry slot and the console's mutable access to its own
/// scrollback during the same `execute` call.
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

    /// Append an error line. Use for command-level failures the user
    /// should see; bubble unrecoverable errors via `Result` instead.
    pub fn error(&mut self, text: impl Into<String>) {
        self.lines.push(HistoryLine::error(text));
    }
}

// ---------------------------------------------------------------------------
// Command trait + closure shim
// ---------------------------------------------------------------------------

/// Console command implementation. Generic over a `Ctx` type so the
/// consuming crate decides what state commands can mutate. For Rye
/// this is typically a struct holding `&mut dyn App`, `&mut Capture`,
/// and an exit signal.
pub trait Command<Ctx>: 'static {
    /// The name typed at the prompt. Conventionally lowercase, dotted
    /// for namespacing (`capture.start`).
    fn name(&self) -> &str;

    /// One-line description shown by `help`.
    fn help(&self) -> &str;

    /// Run the command. `args` are whitespace-split tokens after the
    /// command name. Output goes to `out`; recoverable issues get
    /// `out.error(..)`; unrecoverable ones return `Err`.
    fn run(&mut self, args: &[&str], ctx: &mut Ctx, out: &mut ConsoleWriter) -> anyhow::Result<()>;
}

/// Closure-backed [`Command`] implementation. Use [`cmd`] to construct.
pub struct FnCommand<F> {
    name: &'static str,
    help: &'static str,
    f: F,
}

/// Build a [`Command`] from a closure. The closure mutates a `Ctx` and
/// writes lines into the [`ConsoleWriter`]. Idiomatic for inline
/// per-demo registrations.
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
    FnCommand { name, help, f }
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
    fn run(&mut self, args: &[&str], ctx: &mut Ctx, out: &mut ConsoleWriter) -> anyhow::Result<()> {
        (self.f)(args, ctx, out)
    }
}

// ---------------------------------------------------------------------------
// Console
// ---------------------------------------------------------------------------

/// The dev console. Owns the command registry, scrollback, input line,
/// hotkey binds, and open/close state.
///
/// Construct, register commands and binds during app setup, then call
/// [`Console::ui`] once per frame from inside the host's egui pass:
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
    /// `Some(i)` while cycling history with Up/Down; `None` after
    /// `Enter` or after typing into a fresh input.
    input_history_pos: Option<usize>,
    /// Active tab-completion cycle, if any. Cleared on any input edit
    /// that isn't a tab-complete itself.
    tab: Option<TabState>,
    open: bool,
    /// True for the frame after `open` becomes true so the panel can
    /// request focus once.
    pending_focus: bool,
    /// Right-aligned text in the title row. Host fills with anything
    /// useful (fps, recording state, current scene); empty by default.
    status: String,
}

struct TabState {
    matches: Vec<String>,
    index: usize,
}

impl<Ctx: 'static> Default for Console<Ctx> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Ctx: 'static> Console<Ctx> {
    /// Empty console with no commands, no binds, default `` ` `` toggle
    /// key.
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
        }
    }

    /// Override the toggle key. Default is [`egui::Key::Backtick`].
    pub fn with_toggle_key(mut self, key: egui::Key) -> Self {
        self.toggle_key = key;
        self
    }

    /// Register a command. Replaces any existing command of the same
    /// name without warning; consuming crates can pre-check via
    /// [`Console::has_command`] if they care.
    pub fn register<C: Command<Ctx> + 'static>(&mut self, command: C) {
        let name = command.name().to_string();
        self.commands.insert(name, Box::new(command));
    }

    /// True if a command with this name is registered. `help` and
    /// `clear` are built in and always return true.
    pub fn has_command(&self, name: &str) -> bool {
        name == "help" || name == "clear" || self.commands.contains_key(name)
    }

    /// Bind `key` (no modifiers) to execute `command_line` when the
    /// console is closed. Re-binding overwrites the previous binding.
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

    /// Append a line to the scrollback. Useful for system messages
    /// generated outside command execution (e.g., a background
    /// recording finishing).
    pub fn write(&mut self, line: HistoryLine) {
        self.push_history(line);
    }

    /// Set the title-row status text (right-aligned). Host calls this
    /// each frame with whatever readout it wants visible in the
    /// console: fps, recording elapsed, scene name, etc.
    pub fn set_status(&mut self, text: impl Into<String>) {
        self.status = text.into();
    }

    /// Per-frame entry point. Handles toggle key, hotkey binds (when
    /// closed), in-panel keys (when open), animation, and panel
    /// rendering. Call once per frame from the host's egui pass.
    pub fn ui(&mut self, egui_ctx: &egui::Context, ctx: &mut Ctx) {
        // 1. Toggle key always active.
        let toggle_pressed =
            egui_ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, self.toggle_key));
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

        // 4. Animate panel height.
        let target = if self.open { 1.0 } else { 0.0 };
        let progress = egui_ctx.animate_value_with_time(
            egui::Id::new("rye_console_open_progress"),
            target,
            ANIM_DURATION_SECS,
        );

        if progress > 0.0 {
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
    }

    fn tab_complete(&mut self) {
        // Continue an existing cycle if still applicable.
        if let Some(tab) = self.tab.as_mut() {
            if !tab.matches.is_empty() {
                tab.index = (tab.index + 1) % tab.matches.len();
                self.input.clone_from(&tab.matches[tab.index]);
                return;
            }
        }
        // Fresh completion: collect all command names with the input as prefix.
        let prefix = self.input.clone();
        let matches: Vec<String> = self
            .all_command_names()
            .into_iter()
            .filter(|name| name.starts_with(&prefix))
            .collect();
        if matches.is_empty() {
            return;
        }
        self.input.clone_from(&matches[0]);
        if matches.len() > 1 {
            self.tab = Some(TabState { matches, index: 0 });
        } else {
            self.tab = None;
        }
    }

    fn all_command_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.commands.keys().cloned().collect();
        names.push("help".into());
        names.push("clear".into());
        names.sort();
        names
    }

    /// Execute a command line. Echoes input, looks up the command,
    /// runs it, drains output. Built-in `help` and `clear` short-
    /// circuit before registry lookup.
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

        // Built-ins first.
        if name == "help" {
            self.builtin_help(args.first().map(String::as_str));
            return;
        }
        if name == "clear" {
            self.history.clear();
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

    fn builtin_help(&mut self, target: Option<&str>) {
        match target {
            Some("help") => {
                self.push_history(HistoryLine::output(
                    "help: list commands, or 'help <name>' for one",
                ));
            }
            Some("clear") => {
                self.push_history(HistoryLine::output("clear: clear the scrollback buffer"));
            }
            Some(name) => match self.commands.get(name) {
                Some(c) => {
                    let line = format!("{}: {}", c.name(), c.help());
                    self.push_history(HistoryLine::output(line));
                }
                None => self.push_history(HistoryLine::error(format!("no command '{name}'"))),
            },
            None => {
                self.push_history(HistoryLine::output("commands:"));
                let mut entries: Vec<(String, String)> = self
                    .commands
                    .values()
                    .map(|c| (c.name().to_string(), c.help().to_string()))
                    .collect();
                entries.push(("help".into(), "list commands or describe one".into()));
                entries.push(("clear".into(), "clear the scrollback buffer".into()));
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                for (name, help) in entries {
                    self.push_history(HistoryLine::output(format!("  {name:16} {help}")));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Whitespace-split parser. Returns `(command_name, args)` or `None`
/// for an empty/whitespace-only line. No quoting in v0; commands that
/// need spaces in args should split on something else or wait for the
/// quoted-arg upgrade.
fn parse_line(line: &str) -> Option<(String, Vec<String>)> {
    let mut parts = line.split_whitespace();
    let name = parts.next()?.to_string();
    let args = parts.map(String::from).collect();
    Some((name, args))
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
        // capture.toggle starts with "capture.t", not "capture.s" — so it isn't
        // in the match set; cycling wraps back to start.
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
    fn command_returning_err_pushes_error_line() {
        let mut c = Console::<Ctx>::new();
        c.register(cmd("fail", "always fails", |_, _, _| anyhow::bail!("nope")));
        let mut ctx: Ctx = 0;
        c.execute("fail", &mut ctx);
        let last = c.history.back().unwrap();
        assert_eq!(last.kind, LineKind::Error);
        assert!(last.text.contains("nope"));
    }
}
