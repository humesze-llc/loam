//! Smoke test for `loam-text`. Opens a window with a colored
//! background and renders a few text labels at different sizes and
//! colors. Visual verification only, no assertion.
//!
//! Font discovery: tries a small set of well-known system font
//! locations. On Windows, Arial is reliably present.
//!
//! Esc to exit.

use std::path::Path;

use anyhow::{anyhow, Result};
use loam_app::{run_with_config, App, FrameCtx, RunConfig, SetupCtx};
use loam_math::EuclideanR3;
use loam_render::device::RenderDevice;
use loam_text::TextRenderer;
use winit::window::WindowAttributes;

fn load_system_font() -> Result<Vec<u8>> {
    const CANDIDATES: &[&str] = &[
        // Windows
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        // macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        // Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ];
    for &p in CANDIDATES {
        if Path::new(p).exists() {
            return Ok(std::fs::read(p)?);
        }
    }
    Err(anyhow!(
        "no candidate system font found; tried {CANDIDATES:?}"
    ))
}

struct TextSmokeApp {
    space: EuclideanR3,
    text: TextRenderer,
}

impl App for TextSmokeApp {
    type Space = EuclideanR3;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let font = load_system_font()?;
        let text = TextRenderer::new(
            &ctx.rd.device,
            &ctx.rd.queue,
            ctx.rd.surface_bundle.config.format,
            &font,
            48.0,
        )?;
        Ok(Self {
            space: EuclideanR3,
            text,
        })
    }

    fn space(&self) -> &Self::Space {
        &self.space
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        let cfg = &ctx.rd.surface_bundle.config;
        let w = cfg.width as f32;
        let h = cfg.height as f32;

        // Title-ish label, large.
        self.text.queue(
            "loam-text smoke test",
            [16.0, 16.0],
            48.0,
            [1.0, 1.0, 1.0, 1.0],
        );

        // Demonstrate sizes and colors.
        self.text.queue(
            &format!("frame {}, fps {:.0}", ctx.tick, ctx.fps),
            [16.0, 80.0],
            24.0,
            [0.85, 0.85, 0.95, 1.0],
        );

        self.text.queue(
            "Score: 0123456789",
            [16.0, 120.0],
            32.0,
            [0.4, 1.0, 0.6, 1.0],
        );

        self.text.queue(
            "the quick brown fox jumps over the lazy dog",
            [16.0, 170.0],
            18.0,
            [1.0, 0.8, 0.2, 1.0],
        );

        // Multi-line.
        self.text.queue(
            "ABCDEF GHIJKL\nMNOPQR STUVWX\nYZ !@#$%^&*()",
            [16.0, 220.0],
            22.0,
            [0.9, 0.5, 0.9, 1.0],
        );

        // Bottom-right anchor demo (manual placement, no measurement helper yet).
        self.text.queue(
            "bottom-right",
            [w - 200.0, h - 40.0],
            20.0,
            [1.0, 1.0, 1.0, 0.7],
        );
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        // Clear the surface to a dark color, then overlay text.
        let mut encoder = rd
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("text_smoke clear"),
            });
        {
            let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("text_smoke clear pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.07,
                            b: 0.12,
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

        let cfg = &rd.surface_bundle.config;
        self.text.render(
            &rd.device,
            &rd.queue,
            view,
            [cfg.width as f32, cfg.height as f32],
        )?;
        Ok(())
    }

    fn title(&self, fps: f32) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Owned(format!("loam-text smoke | {fps:.0} fps"))
    }
}

fn main() -> Result<()> {
    let config = RunConfig {
        window: WindowAttributes::default()
            .with_title("loam-text smoke")
            .with_visible(false),
        ..RunConfig::default()
    };
    run_with_config::<TextSmokeApp>(config)
}
