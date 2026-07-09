//! Smoke test for the `loam-egui` egui integration.
//!
//! **What this is.** A tiny test that the framework wiring works:
//! widgets render, input flows in, focus gating works, world-anchored
//! labels project correctly. The `Space` type is a stub
//! (`EuclideanR3`) only because `App::Space` is a required associated
//! type; this app does not use a Space, does not have a player, does
//! not render a scene, and does not have a floor.
//!
//! **What you should see.** A coloured background, an egui window
//! titled "loam-egui smoke" with a few widgets, and (when enabled) a
//! small floating label that orbits the centre of the screen.
//!
//! **What the widgets do:**
//!
//! - **`Window title`** (text input): updates `App::title` so the
//!   OS window-bar text changes as you type. While focus is on this
//!   field, `ctx.ui_has_focus` is true and the "gated forward axis"
//!   readout below stays at zero even if you hold W. That's the
//!   pattern apps use to keep WASD movement from firing while a
//!   user types into a settings panel.
//! - **`Hue`** (slider): drives the clear color via HSV.
//! - **`Toggle anchor label`** (button): shows/hides the orbiting
//!   label.
//! - **`Anchor radius` / `Anchor speed`** (sliders): drive a fictional
//!   point orbiting the origin in a fixed-camera coordinate system.
//!   Its world-space position is projected to screen via
//!   `world_to_screen` each frame; an egui Area is anchored at that
//!   pixel. There is no 3D rendering of the point, only its label.
//!
//! **What this is not.** Not a Space-aware demo. Not a movement
//! demo. Not an SDF render. The WASD reading is purely for showing
//! the focus gate; nothing in the world responds to it.

use std::borrow::Cow;

use anyhow::Result;
use glam::Vec3;
use loam_app::{egui, run_with_config, world_to_screen, App, FrameCtx, RunConfig, SetupCtx};
use loam_camera::CameraView;
use loam_render::device::RenderDevice;
use winit::window::WindowAttributes;

struct UiSmokeApp {
    space: loam_math::EuclideanR3,

    // Driven by the UI.
    title: String,
    hue: f32,
    anchor_radius: f32,
    anchor_speed: f32,
    show_anchor_label: bool,

    // Driven by the sim tick.
    anchor_angle: f32,
    /// Cumulative WASD-forward axis seen this frame, displayed in the
    /// UI to make the focus-gate behavior visible: while a text edit
    /// is focused, this should stay at zero even if the user is
    /// physically holding W.
    last_forward_axis: f32,
}

impl App for UiSmokeApp {
    type Space = loam_math::EuclideanR3;

    fn setup(_ctx: &mut SetupCtx<'_>) -> Result<Self> {
        Ok(Self {
            space: loam_math::EuclideanR3,
            title: "loam-egui smoke".to_string(),
            hue: 0.55,
            anchor_radius: 2.0,
            anchor_speed: 1.0,
            show_anchor_label: true,
            anchor_angle: 0.0,
            last_forward_axis: 0.0,
        })
    }

    fn space(&self) -> &Self::Space {
        &self.space
    }

    fn tick(&mut self, dt: f32, _ctx: &mut loam_app::TickCtx) {
        self.anchor_angle += self.anchor_speed * dt;
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        // Demonstrate the focus gate: only count WASD forward when
        // the UI isn't claiming keyboard input.
        if !ctx.ui_has_focus {
            self.last_forward_axis = ctx.input.move_forward;
        } else {
            self.last_forward_axis = 0.0;
        }
    }

    fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
        egui::Window::new("loam-egui smoke")
            .default_pos([16.0, 16.0])
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Window title:");
                    ui.text_edit_singleline(&mut self.title);
                });

                ui.add(egui::Slider::new(&mut self.hue, 0.0..=1.0).text("Hue"));

                if ui.button("Toggle anchor label").clicked() {
                    self.show_anchor_label = !self.show_anchor_label;
                }

                ui.separator();
                ui.add(egui::Slider::new(&mut self.anchor_radius, 0.5..=4.0).text("Anchor radius"));
                ui.add(egui::Slider::new(&mut self.anchor_speed, 0.0..=4.0).text("Anchor speed"));

                ui.separator();
                ui.label(format!(
                    "ui_has_focus: {} | gated forward axis: {:.2}",
                    frame.ui_has_focus, self.last_forward_axis,
                ));
                ui.label(
                    "Press / hold W to make the axis above read +1.0. \
                     Click into 'Window title' and type, the axis will \
                     read 0.00 while focus is on the text edit. That's \
                     the focus gate; nothing in the scene moves either way.",
                );
            });

        // World-anchored label. Project the anchor's world position
        // into screen-space via the framework's `world_to_screen`
        // helper, then drop an egui Area at that pixel.
        if self.show_anchor_label {
            let anchor_world = Vec3::new(
                self.anchor_radius * self.anchor_angle.cos(),
                0.0,
                self.anchor_radius * self.anchor_angle.sin(),
            );
            let camera = anchor_demo_camera();
            let viewport = (
                frame.rd.surface_bundle.size.width,
                frame.rd.surface_bundle.size.height,
            );
            if let Some(pos) = world_to_screen(
                anchor_world,
                &camera,
                60_f32.to_radians(),
                viewport,
                0.05,
                100.0,
            ) {
                egui::Area::new(egui::Id::new("anchor-label"))
                    .fixed_pos(pos)
                    .show(ctx, |ui| {
                        let frame_style = egui::Frame::popup(ui.style());
                        frame_style.show(ui, |ui| {
                            ui.label(format!(
                                "Anchor ({:.2}, {:.2}, {:.2})",
                                anchor_world.x, anchor_world.y, anchor_world.z,
                            ));
                        });
                    });
            }
        }
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        // Single clear pass, hue-controlled. The framework paints the
        // egui overlay on top after this returns.
        let (r, g, b) = hsv_to_rgb(self.hue, 0.4, 0.18);
        let mut encoder = rd
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ui_smoke::clear"),
            });
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ui_smoke::clear-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: r as f64,
                            g: g as f64,
                            b: b as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
        rd.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn title(&self, fps: f32) -> Cow<'static, str> {
        Cow::Owned(format!("{} | {fps:.0} fps", self.title))
    }
}

/// Fixed camera for the anchor-label demo. At z = 5, looking at the
/// origin, world-up = +Y. The anchor orbits in the XZ plane at y = 0.
fn anchor_demo_camera() -> CameraView {
    CameraView {
        position: Vec3::new(0.0, 0.0, 5.0),
        forward: Vec3::new(0.0, 0.0, -1.0),
        right: Vec3::new(1.0, 0.0, 0.0),
        up: Vec3::new(0.0, 1.0, 0.0),
    }
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match (i as i32) % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

fn main() -> Result<()> {
    run_with_config::<UiSmokeApp>(RunConfig {
        window: WindowAttributes::default()
            .with_title("loam-egui smoke")
            .with_visible(false),
        // Drop wgpu_hal to `error` so the Windows Vulkan loader's noisy
        // "validation layer not found" + "D3D12 mapping JSON parse"
        // warnings stay silent. Those are platform-loader chatter, not
        // anything we can act on. Keep wgpu_core at `warn` so genuine
        // wgpu issues still surface.
        log_filter: Some("info,wgpu_core=warn,wgpu_hal=error".to_string()),
        ..Default::default()
    })
}
