//! Worker-side egui integration.
//!
//! Parallel to [`rye_egui::UiIntegration`] but without `egui_winit`
//! (winit's web backend assumes a `web_sys::Window`, which panics in
//! `WorkerGlobalScope`). Translates the worker's [`super::messages::InputMessage`]
//! events directly into `egui::RawInput::events`.
//!
//! Mirrors `UiIntegration`'s `begin_frame` + `paint` lifecycle so the
//! `App::ui` method works unchanged. The differences are internal:
//!
//! - No cursor / clipboard / IME platform-output handling (those go
//!   through `egui_winit::State::handle_platform_output` in the
//!   windowed path; not load-bearing for the demos we ship).
//! - No `winit_state.take_egui_input` step (we build `RawInput` ourselves
//!   from accumulated events).
//! - Bypassed entirely on first paint: no warmup yet (Phase B+ adds
//!   N3-style worker-side egui pipeline warming).

use rye_egui::egui;

use super::messages::InputMessage;

/// Owns the egui-wgpu `Renderer`, an `egui::Context`, and a per-frame
/// `RawInput` accumulator. Constructed once per worker session;
/// reused frame-to-frame.
pub struct WorkerUi {
    ctx: egui::Context,
    renderer: egui_wgpu::Renderer,
    /// Accumulates events between frames. Drained into `begin_pass` at
    /// the start of each frame; populated by [`Self::record_input`]
    /// whenever an `InputMessage` arrives from main.
    raw_events: Vec<egui::Event>,
    /// Current modifier state. Updated on each key event so subsequent
    /// pointer/key events carry the right `Modifiers`.
    modifiers: egui::Modifiers,
    /// Canvas pixel dimensions + DPR. Egui works in "points" (CSS-pixel
    /// equivalents), wgpu in pixels — `pixels_per_point` is the conversion.
    width_px: u32,
    height_px: u32,
    pixels_per_point: f32,
    /// `wants_pointer_input || wants_keyboard_input` from the last
    /// completed frame. Fed back to the App via `FrameCtx::ui_has_focus`
    /// so gameplay code (camera, hotkeys) can avoid double-handling
    /// events egui consumed.
    pub wants_input: bool,
}

impl WorkerUi {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        sample_count: u32,
        width_px: u32,
        height_px: u32,
        pixels_per_point: f32,
    ) -> Self {
        let ctx = egui::Context::default();
        let renderer = egui_wgpu::Renderer::new(
            device,
            target_format,
            egui_wgpu::RendererOptions {
                msaa_samples: sample_count,
                ..Default::default()
            },
        );
        Self {
            ctx,
            renderer,
            raw_events: Vec::new(),
            modifiers: egui::Modifiers::default(),
            width_px,
            height_px,
            pixels_per_point,
            wants_input: false,
        }
    }

    /// Translate one InputMessage into zero or more egui events.
    /// Updates `modifiers` as a side effect on key events.
    pub fn record_input(&mut self, msg: &InputMessage) {
        match msg {
            InputMessage::MouseMove { x, y, .. } => {
                let pos = self.point(*x, *y);
                self.raw_events.push(egui::Event::PointerMoved(pos));
            }
            InputMessage::MouseButton {
                x,
                y,
                button,
                pressed,
            } => {
                let pos = self.point(*x, *y);
                if let Some(b) = crate::keymap::mouse_button_egui(*button) {
                    self.raw_events.push(egui::Event::PointerButton {
                        pos,
                        button: b,
                        pressed: *pressed,
                        modifiers: self.modifiers,
                    });
                }
            }
            InputMessage::MouseWheel { dx, dy } => {
                // egui's MouseWheel unit is points (its "lines"-equivalent).
                // The DOM->lines conversion already happened on main thread.
                self.raw_events.push(egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Line,
                    delta: egui::vec2(-*dx, -*dy), // egui convention: up = +y
                    modifiers: self.modifiers,
                });
            }
            InputMessage::Key {
                code,
                key,
                pressed,
                repeat,
                ctrl,
                shift,
                alt,
                meta,
            } => {
                // Update modifier state. Egui's Modifiers carries each
                // boolean independently; we keep them current here.
                self.modifiers = egui::Modifiers {
                    alt: *alt,
                    ctrl: *ctrl,
                    shift: *shift,
                    mac_cmd: *meta,
                    command: *ctrl || *meta,
                };
                // Emit a Key event when the physical code maps to an
                // egui::Key. Unknown codes are silently dropped (the
                // App's hotkey routing covers them via InputState).
                if let Some(egui_key) = crate::keymap::keycode_egui(code) {
                    self.raw_events.push(egui::Event::Key {
                        key: egui_key,
                        physical_key: Some(egui_key),
                        pressed: *pressed,
                        repeat: *repeat,
                        modifiers: self.modifiers,
                    });
                }
                // Emit a Text event for printable keys on press. Filters
                // modifier-only chords (Ctrl+C etc.) so they don't
                // accidentally insert text. `key.len() == 1` is a rough
                // "is this a single printable character" check that
                // works for ASCII + common Latin codepoints; multi-codepoint
                // logical keys ("ArrowUp", "Shift") are filtered out.
                if *pressed
                    && !*ctrl
                    && !*alt
                    && !*meta
                    && key.chars().count() == 1
                    && !key.starts_with(char::is_control)
                {
                    self.raw_events.push(egui::Event::Text(key.clone()));
                }
            }
            InputMessage::Focus(focused) => {
                self.raw_events.push(egui::Event::WindowFocused(*focused));
            }
            // Non-egui-relevant variants: Resize is handled by the
            // runner; Visibility hasn't been wired up; Start fires
            // outside the frame loop entirely (handled in handle_message).
            InputMessage::Resize { .. }
            | InputMessage::Visibility(_)
            | InputMessage::Start => {}
        }
    }

    /// Convert canvas-relative CSS pixels to egui points. Egui works in
    /// logical (DPI-independent) coordinates, so callers downstream
    /// divide by `pixels_per_point` where needed. The InputMessage
    /// carries CSS pixels already (DOM `MouseEvent.offsetX/Y` are in
    /// CSS pixels), so we pass them straight through.
    fn point(&self, x: f32, y: f32) -> egui::Pos2 {
        egui::pos2(x, y)
    }

    /// Begin a frame: take the accumulated raw input, set screen rect +
    /// pixels-per-point, return the Context for the App's `ui` method.
    pub fn begin_frame(&mut self) -> &egui::Context {
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(
                    self.width_px as f32 / self.pixels_per_point,
                    self.height_px as f32 / self.pixels_per_point,
                ),
            )),
            events: std::mem::take(&mut self.raw_events),
            modifiers: self.modifiers,
            viewport_id: egui::ViewportId::ROOT,
            time: None,
            ..egui::RawInput::default()
        };
        self.ctx.begin_pass(raw_input);
        &self.ctx
    }

    /// Finish the frame + paint into `view` via the supplied encoder.
    /// Mirrors `UiIntegration::paint` but without the winit
    /// `handle_platform_output` step (no cursor / clipboard routing yet).
    pub fn paint(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        resolve_target: Option<&wgpu::TextureView>,
    ) {
        let full_output = self.ctx.end_pass();
        self.wants_input =
            self.ctx.wants_pointer_input() || self.ctx.wants_keyboard_input();

        let primitives = self
            .ctx
            .tessellate(full_output.shapes, self.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.width_px, self.height_px],
            pixels_per_point: self.pixels_per_point,
        };

        for (id, image_delta) in &full_output.textures_delta.set {
            self.renderer
                .update_texture(device, queue, *id, image_delta);
        }
        self.renderer
            .update_buffers(device, queue, encoder, &primitives, &screen);

        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rye_app::wasm::worker::egui-paint"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            // egui-wgpu requires a `'static` lifetime on the pass; this
            // is the standard wasm-bindgen / egui-wgpu idiom.
            self.renderer
                .render(&mut pass.forget_lifetime(), &primitives, &screen);
        }

        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }

    pub fn resize(&mut self, width: u32, height: u32, dpr: f32) {
        self.width_px = width;
        self.height_px = height;
        self.pixels_per_point = dpr;
    }
}
