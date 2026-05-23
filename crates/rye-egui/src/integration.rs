//! egui + wgpu + winit integration owned by `rye-app::Runner`.
//!
//! The lifecycle the framework runs each frame:
//!
//! 1. Window events feed `egui_winit::State::on_window_event` so egui sees clicks, key
//!    presses, IME, scroll, focus.
//! 2. Before the user's `App::ui` runs, the framework drains accumulated input via
//!    [`UiIntegration::begin_frame`].
//! 3. The user's `App::ui` runs; egui builds a frame's worth of paint commands.
//! 4. After `App::render`, the framework calls [`UiIntegration::paint`] to overlay the
//!    egui output on the surface.

use std::sync::Arc;

use egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor};
use winit::window::Window;

/// Per-window egui state. One instance lives inside the `rye-app` `Runner`; user apps
/// don't construct this themselves.
pub struct UiIntegration {
    ctx: egui::Context,
    winit_state: egui_winit::State,
    renderer: Renderer,
    /// Pixels-per-point at the time of the most recent `begin_frame`. Cached so
    /// `paint` doesn't have to re-query the window.
    pixels_per_point: f32,
    /// `wants_pointer_input || wants_keyboard_input` from the last frame. Fed back to
    /// apps via `FrameCtx::ui_has_focus`.
    last_focus: bool,
}

impl UiIntegration {
    /// Construct. Called from `Runner::resumed` once the device and window are
    /// available.
    ///
    /// **Note on sRGB framebuffers.** `RenderDevice` deliberately picks an sRGB surface
    /// format because Rye's scene rendering wants gamma-correct output. egui-wgpu logs a
    /// warning that it "prefers Rgba8Unorm or Bgra8Unorm" because its shader's blend
    /// math assumes linear space; on an sRGB surface egui internally gamma-corrects with
    /// a small color shift that's invisible at HUD/widget scale. Accepting the warning
    /// is the right tradeoff for v0; rendering egui to a non-sRGB intermediate texture
    /// and compositing onto the sRGB surface would be the correct fix when widget color
    /// fidelity becomes load-bearing.
    pub fn new(
        device: &wgpu::Device,
        window: &Arc<Window>,
        surface_format: wgpu::TextureFormat,
        msaa_samples: u32,
    ) -> Self {
        let ctx = egui::Context::default();
        // egui-winit translates winit events into egui events; the viewport id is
        // `ROOT` for single-window apps.
        let winit_state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            window.theme(),
            None,
        );
        let renderer = Renderer::new(
            device,
            surface_format,
            RendererOptions {
                msaa_samples,
                ..Default::default()
            },
        );
        let pixels_per_point = window.scale_factor() as f32;
        Self {
            ctx,
            winit_state,
            renderer,
            pixels_per_point,
            last_focus: false,
        }
    }

    /// Forward a winit event to egui. Returns `true` if egui consumed the event
    /// (caller can still see it; the bool just signals "egui acted on this," useful
    /// for debug logging).
    pub fn handle_event(
        &mut self,
        window: &Window,
        event: &winit::event::WindowEvent,
    ) -> egui_winit::EventResponse {
        self.winit_state.on_window_event(window, event)
    }

    /// Force egui-wgpu's lazy pipeline compilation by running one minimal
    /// frame into a 1×1 dummy attachment. Eliminates the first-paint stall
    /// the user would otherwise see when the first real frame triggers
    /// shader-module + pipeline-state-object compilation for text + rect +
    /// line shape variants.
    ///
    /// Called once from the runner's `setup_after_device` so the cost is
    /// paid during click-to-start / window-create, where a brief delay is
    /// acceptable. Skipped on `--no-default-features` lean builds (egui
    /// itself is then a stub).
    ///
    /// `target_format` and `sample_count` MUST match the values
    /// `UiIntegration::new` was constructed with; otherwise the warm-up
    /// compiles a pipeline variant that doesn't match what the real frame
    /// will request and no warming actually happens.
    ///
    /// Architectural note: lives on `UiIntegration` (not the runner)
    /// because only this struct knows the egui-wgpu internals (the
    /// `Renderer`, the context lifecycle). The runner just orchestrates.
    pub fn warm_pipelines(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        window: &Window,
        target_format: wgpu::TextureFormat,
        sample_count: u32,
    ) {
        // Dummy 1×1 attachment matching the pipeline's expected format +
        // sample count. Anything else and egui-wgpu would compile the wrong
        // pipeline variant (different format / different MSAA), defeating
        // the warm.
        let dummy_color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rye-egui::warm color"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let dummy_view = dummy_color.create_view(&wgpu::TextureViewDescriptor::default());

        // MSAA path needs a separate single-sample resolve target. Skip
        // when sample_count==1 (no resolve happens).
        let resolve_tex = if sample_count > 1 {
            Some(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("rye-egui::warm resolve"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: target_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            }))
        } else {
            None
        };
        let resolve_view = resolve_tex
            .as_ref()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rye-egui::warm encoder"),
        });

        // Run one minimal frame. Three widget kinds cover the pipeline
        // variants egui-wgpu compiles lazily: text (glyph rendering), a
        // filled rect (background panel), and a line stroke (separator /
        // painter line). Adding a fourth widget kind here is cheap; the
        // pipelines that DON'T get touched here will still lazy-compile
        // when first used, so the trade-off is "what does the demo set
        // need" vs the warm-up time.
        let ctx = self.begin_frame(window);
        let ctx = ctx.clone();
        egui::Area::new(egui::Id::new("rye-egui::warm")).show(&ctx, |ui| {
            ui.label("warm");
            ui.separator();
            ui.painter().line_segment(
                [egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)],
                egui::Stroke::new(1.0, egui::Color32::WHITE),
            );
        });
        self.paint(
            device,
            queue,
            &mut encoder,
            &dummy_view,
            resolve_view.as_ref(),
            window,
            (1, 1),
        );

        queue.submit(Some(encoder.finish()));
    }

    /// Drain accumulated input + start a fresh egui frame. Returns the `egui::Context`
    /// for the user's `App::ui` to build against.
    pub fn begin_frame(&mut self, window: &Window) -> &egui::Context {
        let raw_input = self.winit_state.take_egui_input(window);
        self.pixels_per_point = window.scale_factor() as f32;
        self.ctx.begin_pass(raw_input);
        &self.ctx
    }

    /// Finish the egui frame, capture output, and paint onto `view` using the supplied
    /// encoder. Pairs with `begin_frame`.
    ///
    /// `viewport` is `(width_px, height_px)`. Caller is responsible for having already
    /// cleared and rendered the main scene; this paint overlays with `LoadOp::Load`.
    ///
    /// `resolve_target`, when `Some`, is the single-sample target the MSAA render
    /// attachment is resolved into at the end of this pass. Pass
    /// `Some(swapchain_view)` when MSAA is enabled (`view` being the multisampled color
    /// attachment) so the deferred MSAA resolve happens here rather than requiring a
    /// separate resolve pass. Pass `None` when MSAA is disabled (`view` is the
    /// swapchain view).
    #[allow(clippy::too_many_arguments)]
    pub fn paint(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        resolve_target: Option<&wgpu::TextureView>,
        window: &Window,
        viewport: (u32, u32),
    ) {
        let full_output = self.ctx.end_pass();
        self.last_focus = self.ctx.wants_pointer_input() || self.ctx.wants_keyboard_input();

        // Forward platform output (cursor changes, clipboard writes, open-link
        // requests). Without this, hovering a button never changes the cursor and
        // clicking a `Hyperlink` does nothing.
        self.winit_state
            .handle_platform_output(window, full_output.platform_output);

        let primitives = self
            .ctx
            .tessellate(full_output.shapes, self.pixels_per_point);

        let screen = ScreenDescriptor {
            size_in_pixels: [viewport.0, viewport.1],
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
                label: Some("rye-egui::paint"),
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
            // egui-wgpu wants a `RenderPass<'static>`; forward_render is the helper
            // that does that lifetime dance for us.
            self.renderer
                .render(&mut pass.forget_lifetime(), &primitives, &screen);
        }

        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }

    /// `true` if egui currently wants pointer or keyboard input (a widget is hovered,
    /// focused, or accepting text). Apps should gate gameplay input on
    /// `!ui_has_focus()` so a player typing in a settings field doesn't also fire WASD
    /// movement.
    pub fn ui_has_focus(&self) -> bool {
        self.last_focus
    }
}
