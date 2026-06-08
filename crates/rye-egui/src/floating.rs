//! Engine-level wrappers for free-floating UI containers (windows, callouts),
//! wrapping raw `egui` containers with the conventions Rye demos share so look +
//! feel stays consistent.
//!
//! - [`floating_panel`]: settings-modal pattern; a draggable, collapsible window
//!   opened by an external toggle and closed via the title-bar X.
//! - [`sticky_menu`]: a dropdown that stays open while toggles inside it are clicked.
//! - [`callout`]: tutorial-annotation pattern; a caller-projected 3D world anchor,
//!   a leader line, and a draggable panel, all in screen-space overlays.

use egui::{Context, Id, Painter, Pos2, Rect, Stroke, Ui, Window};

/// Builder for [`floating_panel`]. Use it when the panel needs to be resizable,
/// wider than the settings default, or positioned away from centre; otherwise the
/// [`floating_panel`] free function covers the default shape.
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
    /// Allow resizing via the standard egui grip. Default `false`.
    pub fn resizable(mut self, on: bool) -> Self {
        self.resizable = on;
        self
    }

    /// Allow collapsing via the title-bar chevron. Default `true`.
    pub fn collapsible(mut self, on: bool) -> Self {
        self.collapsible = on;
        self
    }

    /// Set default width and height, overriding the "width 260, height auto" default.
    pub fn default_size(mut self, width: f32, height: f32) -> Self {
        self.default_size = Some((width, height));
        self
    }

    /// Set the default width; height stays content-sized. Default `260.0`.
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

    /// Render the panel: the closure runs only when `*open == true`, and the
    /// title-bar X clears `*open`.
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

/// Floating, draggable, collapsible settings panel: title-bar X, default width hint,
/// centre placement on first display, non-resizable. For other shapes use
/// [`floating_panel_builder`].
///
/// `open` doubles as toggle state: the helper clears it on close-X, and the closure
/// runs only while `*open == true`.
///
/// ```ignore
/// rye_egui::floating_panel(
///     ctx, "polytope-playground-render", "Render",
///     &mut self.show_render_panel, |ui| {
///     ui.label("Surface");
///     ui.radio_value(&mut self.surface_mode, SurfaceMode::Raster, "Raster");
///     // ...
/// });
/// ```
///
/// Returns `None` when closed (closure not invoked), `Some(R)` when open.
pub fn floating_panel<R>(
    ctx: &Context,
    id: &str,
    title: &str,
    open: &mut bool,
    content: impl FnOnce(&mut Ui) -> R,
) -> Option<R> {
    floating_panel_builder(ctx, id, title, open).show(content)
}

/// Builder entry point for floating panels needing non-default config. Returns a
/// [`FloatingPanelBuilder`]; chain config setters then `.show(|ui| { ... })`.
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

/// "Sticky" menu button: a dropdown that stays open while toggles inside it are
/// clicked, closing only on click-outside or `Esc`. Use instead of
/// `egui::menu_button`, which closes on every interactive click and makes
/// multi-checkbox menus unusable.
///
/// ```ignore
/// rye_egui::sticky_menu(ui, "View", |ui| {
///     ui.checkbox(&mut self.show_controls, "Rotation controls");
///     ui.checkbox(&mut self.show_formula, "Formula popup");
///     ui.checkbox(&mut self.example_callout.open, "Example callout");
/// });
/// ```
///
/// One-shot entries that should close on click call
/// `ui.memory_mut(|m| m.close_popup())` from inside the content closure.
pub fn sticky_menu<R>(
    ui: &mut Ui,
    label: &str,
    add_contents: impl FnOnce(&mut Ui) -> R,
) -> Option<R> {
    let response = ui.button(label);
    // The `CloseOnClickOutside` override is the point; the default `CloseOnClick`
    // collapses the dropdown when a checkbox inside is clicked.
    egui::Popup::menu(&response)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(add_contents)
        .map(|r| r.inner)
}

/// Caller-owned persistent state for a [`callout`] so panel position and open state
/// survive across frames.
#[derive(Clone, Debug)]
pub struct CalloutState {
    /// Top-left of the callout window in screen pixels; updated after each drag.
    pub window_pos: Pos2,
    /// `true` when open. The title-bar X clears it; set it back to reopen.
    pub open: bool,
}

impl CalloutState {
    /// A callout that starts open at the given screen position.
    pub fn open_at(window_pos: Pos2) -> Self {
        Self {
            window_pos,
            open: true,
        }
    }
}

/// Draw an annotation callout: an anchor disc at `anchor_screen_pos`, a leader line
/// to a draggable panel, and the panel hosting `content`. The caller projects the 3D
/// world anchor to screen space each frame. No-op when `state.open == false`.
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

    // Leader line uses the window fill so it reads as part of the panel; anchor disc
    // gets a dark outline to stay visible against bright scenes.
    const ANCHOR_RADIUS: f32 = 4.0;
    const LEADER_STROKE: f32 = 1.5;
    const PANEL_DEFAULT_WIDTH: f32 = 220.0;
    let leader_color = ctx.style().visuals.window_fill;
    let anchor_outline = ctx.style().visuals.window_stroke.color;

    // Window first so the leader line can attach to its captured frame rect.
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

    // Persist the (possibly dragged) window rect for next frame's position.
    let window_rect: Option<Rect> = window_response.as_ref().map(|r| r.response.rect);
    if let Some(rect) = window_rect {
        state.window_pos = rect.min;
    }

    // Draw on `Order::Background` so the line sits under the Window (default
    // `Order::Middle`) but still over the wgpu scene; non-interactive overlay.
    let painter_layer = egui::LayerId::new(
        egui::Order::Background,
        Id::new(format!("{id}-callout-overlay")),
    );
    let painter = Painter::new(ctx.clone(), painter_layer, ctx.content_rect());
    if let Some(rect) = window_rect {
        // Attach to the window CENTER so the line emerges from under the panel;
        // edge attachment reads as a separate element pointing at it.
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
