//! Engine-level wrappers for free-floating UI containers (windows, callouts).
//!
//! Both wrap raw `egui` containers (`Window`, `Painter`) with the conventions Rye demos
//! share: collapsible-but-not-resizable by default, standardised id naming, consistent
//! anchor / leader-line styling. Demos use these instead of touching `egui::Window`
//! directly so that the look + feel stays consistent across the family of demos in the
//! engine.
//!
//! Two primitives ship here:
//!
//! - [`floating_panel`]: the "settings modal" pattern. A draggable, collapsible window
//!   that opens via an external toggle (gear button, menu entry) and closes via the
//!   standard title-bar X. Replaces ad-hoc `egui::Window::new(...).open(...)` calls
//!   scattered across demo code.
//! - [`callout`]: the "tutorial annotation" pattern. A 3D world-anchor point plus a
//!   leader line plus a draggable explanatory panel. The 3D anchor is reprojected by
//!   the caller each frame; this primitive draws the leader line and hosts the panel
//!   on the egui side, all in screen-space overlays so no GPU pipeline changes are
//!   needed.
//!
//! See `docs/devlog/context/CALLOUTS_AND_BILLBOARDS.md` for the design rationale and
//! the bigger picture (projection-billboard sibling lands in M5+M6).

use egui::{Context, Id, Painter, Pos2, Rect, Stroke, Ui, Window};

/// Builder for [`floating_panel`]. Most callers won't need this directly: the
/// `floating_panel(ctx, id, title, open, content)` free function covers the
/// settings-modal default shape. Reach for `FloatingPanelBuilder` when the panel needs
/// to be resizable, larger than the default settings width, or positioned away from
/// centre (e.g. a help / about modal anchored to the screen corner).
#[must_use = "FloatingPanelBuilder does nothing until `.show()` is called"]
pub struct FloatingPanelBuilder<'a> {
    ctx: &'a Context,
    id: &'a str,
    title: &'a str,
    open: &'a mut bool,
    resizable: bool,
    collapsible: bool,
    default_size: Option<(f32, f32)>,
    default_width: f32,
    default_pos: Option<Pos2>,
}

impl<'a> FloatingPanelBuilder<'a> {
    /// Allow the user to resize the panel via the standard egui resize grip. Default
    /// `false`; settings modals typically don't benefit from resize, but information /
    /// help modals do.
    pub fn resizable(mut self, on: bool) -> Self {
        self.resizable = on;
        self
    }

    /// Allow the panel to be collapsed via the chevron in its title bar. Default
    /// `true`. Turn off for modals whose collapsed state would be confusing (e.g. a
    /// help dialog that should always show its content while open).
    pub fn collapsible(mut self, on: bool) -> Self {
        self.collapsible = on;
        self
    }

    /// Set both default width and height. Overrides the settings-modal default of
    /// "width 260, height auto". Use for larger info panels.
    pub fn default_size(mut self, width: f32, height: f32) -> Self {
        self.default_size = Some((width, height));
        self
    }

    /// Set just the default width. The height stays content-sized. Default `260.0`.
    pub fn default_width(mut self, width: f32) -> Self {
        self.default_width = width;
        self
    }

    /// Initial position on first display. Subsequent frames respect any user drag.
    /// Default: egui's automatic centre-of-screen placement.
    pub fn default_pos(mut self, pos: Pos2) -> Self {
        self.default_pos = Some(pos);
        self
    }

    /// Render the panel. Same semantics as the free function: closure runs only when
    /// `*open == true`; clicking the title-bar X clears `*open`.
    pub fn show<R>(self, content: impl FnOnce(&mut Ui) -> R) -> Option<R> {
        if !*self.open {
            return None;
        }
        let mut local_open = *self.open;
        let mut window = Window::new(self.title)
            .id(Id::new(self.id))
            .open(&mut local_open)
            .collapsible(self.collapsible)
            .resizable(self.resizable);
        if let Some((w, h)) = self.default_size {
            window = window.default_size(egui::vec2(w, h));
        } else {
            window = window.default_width(self.default_width);
        }
        if let Some(pos) = self.default_pos {
            window = window.default_pos(pos);
        }
        let result = window.show(self.ctx, content).and_then(|r| r.inner);
        *self.open = local_open;
        result
    }
}

/// Open a floating, draggable, collapsible panel hosting demo-specific settings or
/// inspection content. Standard engine convention: title-bar with a close X, default
/// width hint, opens in the centre of the screen on first display. Non-resizable;
/// non-collapsible disabled by default (the chevron lets the user fold it away). For
/// shapes outside this default (resizable info modals, larger help dialogs) use
/// [`FloatingPanelBuilder`] (via [`floating_panel_builder`]) instead.
///
/// The `open` reference doubles as the toggle state: pass `&mut self.show_render_panel`
/// (or similar) and the helper clears it when the user clicks the close X. The closure
/// is invoked only while `*open == true`, so callers don't need to wrap the call.
///
/// ```ignore
/// rye_egui::floating_panel(ctx, "polytope-playground-render", "Render", &mut self.show_render_panel, |ui| {
///     ui.label("Surface");
///     ui.radio_value(&mut self.surface_mode, SurfaceMode::Raster, "Raster");
///     // ...
/// });
/// ```
///
/// The returned value is `None` when the panel is closed (closure not invoked) and
/// `Some(R)` with the closure's result when open. Most callers ignore it.
pub fn floating_panel<R>(
    ctx: &Context,
    id: &str,
    title: &str,
    open: &mut bool,
    content: impl FnOnce(&mut Ui) -> R,
) -> Option<R> {
    floating_panel_builder(ctx, id, title, open).show(content)
}

/// Builder-flavored entry point for floating panels that need non-default config (a
/// larger default size, resizable behavior, an initial position). Returns a
/// [`FloatingPanelBuilder`]; chain `.resizable(true).default_size(w, h)...` then
/// `.show(|ui| { ... })`.
///
/// ```ignore
/// rye_egui::floating_panel_builder(ctx, "playground-about", "About", &mut self.show_help)
///     .resizable(true)
///     .default_size(560.0, 460.0)
///     .default_pos(egui::pos2(80.0, 80.0))
///     .show(|ui| {
///         egui::ScrollArea::vertical().show(ui, |ui| {
///             // help text
///         });
///     });
/// ```
pub fn floating_panel_builder<'a>(
    ctx: &'a Context,
    id: &'a str,
    title: &'a str,
    open: &'a mut bool,
) -> FloatingPanelBuilder<'a> {
    FloatingPanelBuilder {
        ctx,
        id,
        title,
        open,
        resizable: false,
        collapsible: true,
        default_size: None,
        default_width: 260.0,
        default_pos: None,
    }
}

/// "Sticky" menu button: a dropdown that stays open while the user clicks checkboxes,
/// radio buttons, sliders, or anything else inside it. Closes only on a click outside
/// or `Esc`. Replacement for `egui::menu_button` in the case where the menu hosts
/// sticky toggles rather than one-shot actions.
///
/// `egui::menu_button` closes on every interactive click via its `MenuRoot` cascade,
/// which makes "View > Show Foo / Show Bar / Show Baz" menus unusable: the user has
/// to re-open the menu after every checkbox flick. `sticky_menu` uses
/// `popup_below_widget` with `PopupCloseBehavior::CloseOnClickOutside` instead, which
/// is what the user expects for a settings-style dropdown.
///
/// ```ignore
/// rye_egui::sticky_menu(ui, "View", |ui| {
///     ui.checkbox(&mut self.show_controls, "Rotation controls");
///     ui.checkbox(&mut self.show_formula, "Formula popup");
///     ui.checkbox(&mut self.example_callout.open, "Example callout");
/// });
/// ```
///
/// One-shot menu entries (buttons that *should* close the menu when clicked) call
/// `ui.memory_mut(|m| m.close_popup())` from inside the content closure to opt back
/// into close-on-click behavior. The `?` button below the gear is a good example.
pub fn sticky_menu<R>(
    ui: &mut Ui,
    label: &str,
    add_contents: impl FnOnce(&mut Ui) -> R,
) -> Option<R> {
    let response = ui.button(label);
    // `Popup::menu` wires up the standard toggle-on-click behavior + menu styling +
    // below-the-button positioning. The override is `CloseOnClickOutside`; without
    // it, the default is `CloseOnClick` and clicking a checkbox inside collapses the
    // dropdown, which is the bug we're fixing.
    egui::Popup::menu(&response)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(add_contents)
        .map(|r| r.inner)
}

/// Persistent state for a [`callout`]. Stored on the caller's side so the panel
/// position survives frames and a future toggle can hide / show without resetting the
/// drag position.
#[derive(Clone, Debug)]
pub struct CalloutState {
    /// Top-left of the callout window in screen pixels. Updated each frame after the
    /// user drags the window.
    pub window_pos: Pos2,
    /// `true` when the callout is open. The title-bar X clears this; flip it back to
    /// `true` from a parent UI to reopen.
    pub open: bool,
}

impl CalloutState {
    /// Convenience constructor for a callout that starts open at a given screen
    /// position.
    pub fn open_at(window_pos: Pos2) -> Self {
        Self {
            window_pos,
            open: true,
        }
    }
}

/// Draw an annotation callout: a small anchor disc at `anchor_screen_pos`, a leader
/// line from the anchor to the nearest edge of a draggable panel, and the panel
/// itself hosting `content`.
///
/// The caller is responsible for projecting the 3D world anchor to screen space each
/// frame and handing the result here (`anchor_screen_pos`). This primitive doesn't
/// know about cameras; it composes onto egui's screen-space layer above the scene.
///
/// `state` survives across frames: panel position persists through dragging, and the
/// `open` flag closes / reopens the callout. No-op when `state.open == false`.
///
/// All drawing happens in egui's foreground area, ordered above the central scene.
/// No GPU pipeline changes; the cost is one Vec3-to-Vec2 projection per anchor per
/// frame (on the caller's side) and a single egui line + circle on this side.
pub fn callout(
    ctx: &Context,
    id: &str,
    anchor_screen_pos: Pos2,
    state: &mut CalloutState,
    title: &str,
    content: impl FnOnce(&mut Ui),
) {
    if !state.open {
        return;
    }

    // Leader-line + anchor disc styling. The leader line uses the window background
    // color so the line reads as a continuation of the window panel, not as a
    // foreign UI element competing with scene content. Anchor disc has a small dark
    // outline so the dot stays visible against bright scene backgrounds.
    const ANCHOR_RADIUS: f32 = 4.0;
    const LEADER_STROKE: f32 = 1.5;
    const PANEL_DEFAULT_WIDTH: f32 = 220.0;
    let leader_color = ctx.style().visuals.window_fill;
    let anchor_outline = ctx.style().visuals.window_stroke.color;

    // Window first: position from `state`, capture the actual frame rect via the
    // Window's response so the leader line can attach to the nearest edge.
    let mut local_open = state.open;
    let window_response = Window::new(title)
        .id(Id::new(id))
        .open(&mut local_open)
        .collapsible(true)
        .resizable(false)
        .default_width(PANEL_DEFAULT_WIDTH)
        .current_pos(state.window_pos)
        .show(ctx, content);
    state.open = local_open;

    // Capture the (possibly user-dragged) window rect for both leader-line attachment
    // and next-frame position persistence.
    let window_rect: Option<Rect> = window_response.as_ref().map(|r| r.response.rect);
    if let Some(rect) = window_rect {
        state.window_pos = rect.min;
    }

    // Leader line + anchor disc draw on `Order::Background`. egui's Window defaults to
    // `Order::Middle`, and within the same Order later draws win, so a Middle-layer
    // line queued after the window would paint over the window. Background sits
    // strictly below every egui surface but still above the wgpu scene render, which
    // is the layer order we want: line under window, both over scene. Non-interactive
    // (no hit-testing); pure visual overlay.
    let painter_layer = egui::LayerId::new(
        egui::Order::Background,
        Id::new(format!("{id}-callout-overlay")),
    );
    let painter = Painter::new(ctx.clone(), painter_layer, ctx.content_rect());
    if let Some(rect) = window_rect {
        // Attach the leader to the window's CENTER rather than the nearest edge.
        // Center attachment reads as "this thing is part of the window" because the
        // line visually emerges from under the window content; edge attachment looks
        // more like a separate UI element pointing at the window.
        painter.line_segment(
            [rect.center(), anchor_screen_pos],
            Stroke::new(LEADER_STROKE, leader_color),
        );
    }
    painter.circle_filled(anchor_screen_pos, ANCHOR_RADIUS, leader_color);
    painter.circle_stroke(
        anchor_screen_pos,
        ANCHOR_RADIUS + 1.0,
        Stroke::new(1.0, anchor_outline),
    );
}

// (Earlier revision had a `nearest_edge_point` helper that attached the leader line
// to the rect perimeter; switched to centre-attachment per UX feedback so the line
// reads as part of the window rather than a foreign element pointing at it.)
