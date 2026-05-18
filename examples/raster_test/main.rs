//! Rasterization-pipeline development playground.
//!
//! Renders a curated test scene of line-segment primitives via
//! [`rye_render::LineRasterNode`]: world-axes for depth perception, a unit cube wireframe,
//! a width sweep (1 / 2 / 4 / 8 px) to visually check AA at different widths, a color-gradient
//! line, and a fan of tilted lines to validate AA at all screen-space orientations.
//!
//! Pairs with the M1 first-ship work in [`docs/devlog/RASTERIZATION_TIER.md`]. Used to validate
//! the rasterizer in isolation, before wiring it into a real demo (Polytope Playground, M2+).
//!
//! ## Controls
//!
//! - **Mouse left-drag**: orbit camera.
//! - **Scroll**: zoom.
//! - **Esc**: exit.
//! - **\` (backtick)**: toggle console (default `rye-egui` bind).
//!
//! ## Console commands
//!
//! - `axes on|off`: toggle the world-space colored axes.
//! - `cube on|off`: toggle the unit cube wireframe.
//! - `widths on|off`: toggle the four horizontal width-sweep lines.
//! - `gradient on|off`: toggle the rainbow gradient line.
//! - `tilted on|off`: toggle the fan of tilted lines.
//! - `samples N`: set per-segment tessellation density (1 for flat space; higher matters
//!   only when curved-space impls land post-blog).
//! - `reset`: restore all toggles to default (everything on, samples = 1).

use std::borrow::Cow;

use anyhow::Result;
use glam::{Mat4, Vec2, Vec3};
use rye_app::{egui, run_with_config, App, Camera, FrameCtx, OrbitController, RunConfig, SetupCtx};
use rye_egui::{Console, ConsoleWriter};
use rye_math::{EuclideanR3, Projection};
use rye_render::{device::RenderDevice, LineRasterNode};
use rye_shape::LineMesh;
use winit::window::WindowAttributes;

/// Test-scene toggles. Each maps to a category in [`build_mesh`].
#[derive(Clone, Copy, Debug)]
struct Toggles {
    axes: bool,
    cube: bool,
    widths: bool,
    gradient: bool,
    tilted: bool,
}

impl Default for Toggles {
    fn default() -> Self {
        Self {
            axes: true,
            cube: true,
            widths: true,
            gradient: true,
            tilted: true,
        }
    }
}

/// Build the full test-scene [`LineMesh<3>`] from the current toggles. Re-runs when the
/// console mutates toggles; the rasterizer re-uploads the result via [`LineRasterNode::upload`].
fn build_mesh(t: Toggles) -> LineMesh<3> {
    let mut mesh: LineMesh<3> = LineMesh::default();

    // --- World-axes: classic R/G/B basis vectors anchored at origin. Length 3.0, 2 px width.
    if t.axes {
        let axes: [([f32; 3], [f32; 3], [f32; 4]); 3] = [
            ([0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.95, 0.20, 0.20, 1.0]),
            ([0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.25, 0.90, 0.30, 1.0]),
            ([0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.30, 0.55, 0.95, 1.0]),
        ];
        for (a, b, color) in axes {
            mesh.segments.push((a, b));
            mesh.colors.push((color, color));
            mesh.widths.push(2.0);
        }
    }

    // --- Unit cube wireframe, white, centered at the origin, ±1 on each axis. 12 edges.
    if t.cube {
        let s = 1.5_f32;
        let corners: [[f32; 3]; 8] = [
            [-s, -s, -s],
            [s, -s, -s],
            [-s, s, -s],
            [s, s, -s],
            [-s, -s, s],
            [s, -s, s],
            [-s, s, s],
            [s, s, s],
        ];
        let edges: [(usize, usize); 12] = [
            (0, 1),
            (1, 3),
            (3, 2),
            (2, 0), // back face
            (4, 5),
            (5, 7),
            (7, 6),
            (6, 4), // front face
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7), // connectors
        ];
        let white = [0.9_f32, 0.9, 0.9, 1.0];
        for (a, b) in edges {
            mesh.segments.push((corners[a], corners[b]));
            mesh.colors.push((white, white));
            mesh.widths.push(2.0);
        }
    }

    // --- Width sweep: 4 horizontal lines at increasing widths, separated vertically.
    //     Validates that the AA fragment shader's coverage smoothstep produces clean edges
    //     at 1, 2, 4, 8 px and that thicker lines stay opaque in the center.
    if t.widths {
        let widths = [1.0, 2.0, 4.0, 8.0];
        let y_start = 3.5;
        let y_step = 0.4;
        let color = [0.95_f32, 0.85, 0.30, 1.0]; // warm yellow
        for (i, &w) in widths.iter().enumerate() {
            let y = y_start + y_step * (i as f32);
            mesh.segments.push(([-2.5, y, 0.0], [2.5, y, 0.0]));
            mesh.colors.push((color, color));
            mesh.widths.push(w);
        }
    }

    // --- Gradient line: single segment with start color != end color. Tests per-endpoint
    //     color interpolation in the rasterizer.
    if t.gradient {
        mesh.segments.push(([-3.0, -2.5, 0.0], [3.0, -2.5, 0.0]));
        mesh.colors.push((
            [0.95, 0.20, 0.20, 1.0], // red
            [0.30, 0.55, 0.95, 1.0], // blue
        ));
        mesh.widths.push(3.0);
    }

    // --- Tilted-line fan: 8 lines radiating from (-4, 0, 4), each at a different screen-
    //     space angle. AA quality should be uniform across angles; if some look smooth and
    //     others jagged, the quad-expansion math is wrong.
    if t.tilted {
        let center = [-4.0_f32, 0.0, 4.0];
        let r = 1.5_f32;
        let color = [0.85_f32, 0.45, 0.95, 1.0]; // magenta
        for i in 0..8 {
            let phi = (i as f32) * std::f32::consts::TAU / 8.0;
            let end = [
                center[0] + r * phi.cos(),
                center[1] + r * phi.sin(),
                center[2],
            ];
            mesh.segments.push((center, end));
            mesh.colors.push((color, color));
            mesh.widths.push(2.0);
        }
    }

    mesh
}

struct Demo {
    space: EuclideanR3,
    camera: Camera<EuclideanR3>,
    orbit: OrbitController<EuclideanR3>,
    line_raster: LineRasterNode,
    toggles: Toggles,
    /// Tessellation samples-per-segment. 1 for flat space (no interior subdivision); higher
    /// values exercise the writer-pattern path that geodesic-space impls will use later.
    samples: usize,
    /// Set by console toggle / `samples` commands to re-upload the mesh on the next frame.
    /// Persists across frames so multiple console mutations within one frame coalesce.
    dirty: bool,
}

impl Demo {
    fn new(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let line_raster = LineRasterNode::new(
            &ctx.rd.device,
            ctx.rd.surface_bundle.config.format,
            ctx.rd.sample_count(),
        );

        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 0.0, 8.0);
        let mut orbit: OrbitController<EuclideanR3> = OrbitController::default();
        orbit.set_orbit(8.0, -0.15);

        let mut demo = Self {
            space: EuclideanR3,
            camera,
            orbit,
            line_raster,
            toggles: Toggles::default(),
            samples: 1,
            dirty: true,
        };
        demo.reupload(ctx.rd);
        Ok(demo)
    }

    /// Re-tessellate and re-upload the mesh. Called whenever toggles or `samples` change.
    fn reupload(&mut self, rd: &RenderDevice) {
        let mesh = build_mesh(self.toggles);
        self.line_raster.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &mesh,
            &Projection::Identity,
            self.samples,
        );
        self.dirty = false;
    }
}

impl Demo {
    fn space(&self) -> &EuclideanR3 {
        &self.space
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        use rye_camera::CameraController;
        if !ctx.ui_has_focus {
            self.orbit
                .advance(ctx.input, &mut self.camera, &EuclideanR3, 0.0);
        }
        if self.dirty {
            self.reupload(ctx.rd);
        }
    }

    fn on_event(&mut self, _ev: &winit::event::WindowEvent, _ctx: &mut FrameCtx<'_>) {
        // No demo-specific key bindings yet; navigation is mouse-only via the orbit
        // controller. Console + ` key handle text input independently.
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        let cfg = &rd.surface_bundle.config;
        let view_dir = self.camera.view();
        let aspect = cfg.width as f32 / cfg.height as f32;
        let view_mat = Mat4::look_to_rh(view_dir.position, view_dir.forward, view_dir.up);
        let proj_mat = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.1, 100.0);
        let view_proj = proj_mat * view_mat;
        let vp_size = Vec2::new(cfg.width as f32, cfg.height as f32);
        self.line_raster.set_camera(&rd.queue, view_proj, vp_size);

        // Clear the framebuffer to a dark slate so the lines are visible. The rasterizer's
        // pass uses LoadOp::Load, so we need a prior pass to clear; do it inline here via a
        // bare-bones encoder. (In a real demo, this clear would come from whatever scene
        // pass runs before the rasterizer; in this test example the rasterizer IS the
        // scene.)
        let mut clear_encoder = rd
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("raster_test clear"),
            });
        {
            let _ = clear_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("raster_test clear pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.06,
                            g: 0.07,
                            b: 0.10,
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
        rd.queue.submit(Some(clear_encoder.finish()));

        self.line_raster.execute(rd, view)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// App wrapper: Demo + Console<Demo>
// ---------------------------------------------------------------------------

struct RasterTestApp {
    demo: Demo,
    console: Console<Demo>,
    last_egui_keyboard: bool,
}

impl RasterTestApp {
    fn build_console() -> Console<Demo> {
        let mut c = Console::<Demo>::new();
        c.register(rye_egui::cmd(
            "axes",
            "toggle world-axes (on|off)",
            toggle_cmd("axes", |t| &mut t.axes),
        ));
        c.register(rye_egui::cmd(
            "cube",
            "toggle unit-cube wireframe (on|off)",
            toggle_cmd("cube", |t| &mut t.cube),
        ));
        c.register(rye_egui::cmd(
            "widths",
            "toggle width-sweep lines (on|off)",
            toggle_cmd("widths", |t| &mut t.widths),
        ));
        c.register(rye_egui::cmd(
            "gradient",
            "toggle gradient line (on|off)",
            toggle_cmd("gradient", |t| &mut t.gradient),
        ));
        c.register(rye_egui::cmd(
            "tilted",
            "toggle tilted-line fan (on|off)",
            toggle_cmd("tilted", |t| &mut t.tilted),
        ));
        c.register(rye_egui::cmd(
            "samples",
            "set tessellation samples-per-segment (default 1)",
            |args, demo: &mut Demo, _out: &mut ConsoleWriter| {
                let n: usize = args
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("usage: samples <N>"))?
                    .parse()?;
                if n == 0 {
                    return Err(anyhow::anyhow!("samples must be >= 1"));
                }
                demo.samples = n;
                demo.dirty = true;
                Ok(())
            },
        ));
        c.register(rye_egui::cmd(
            "reset",
            "restore all toggles to default (everything on, samples = 1)",
            |_args, demo: &mut Demo, _out: &mut ConsoleWriter| {
                demo.toggles = Toggles::default();
                demo.samples = 1;
                demo.dirty = true;
                Ok(())
            },
        ));

        // Standard framework log mirror so tracing events show up in the console scrollback.
        rye_app::log::register_command(&mut c);
        c
    }
}

/// Helper: build a console command that flips a boolean inside [`Toggles`].
fn toggle_cmd<F>(
    name: &'static str,
    field: F,
) -> impl FnMut(&[&str], &mut Demo, &mut ConsoleWriter) -> Result<()> + 'static
where
    F: Fn(&mut Toggles) -> &mut bool + 'static,
{
    move |args, demo: &mut Demo, _out: &mut ConsoleWriter| {
        let new = match args.first().copied() {
            Some("on") => true,
            Some("off") => false,
            Some(_) => return Err(anyhow::anyhow!("usage: {name} on|off")),
            None => !*field(&mut demo.toggles), // toggle if no arg
        };
        *field(&mut demo.toggles) = new;
        demo.dirty = true;
        Ok(())
    }
}

impl App for RasterTestApp {
    type Space = EuclideanR3;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let demo = Demo::new(ctx)?;
        let console = Self::build_console();
        Ok(Self {
            demo,
            console,
            last_egui_keyboard: false,
        })
    }

    fn space(&self) -> &EuclideanR3 {
        self.demo.space()
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        self.demo.update(ctx);
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        self.demo.render(rd, view)
    }

    fn title(&self, fps: f32) -> Cow<'static, str> {
        Cow::Owned(format!("rye: raster_test  {fps:.0} fps"))
    }

    fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
        // Settings panel: shows the current toggle state. Read-only; mutations go through
        // the console for a single source of truth.
        egui::Window::new("raster_test")
            .default_pos([8.0, 8.0])
            .default_width(220.0)
            .show(ctx, |ui| {
                ui.label(format!(
                    "axes:     {}",
                    if self.demo.toggles.axes { "on" } else { "off" }
                ));
                ui.label(format!(
                    "cube:     {}",
                    if self.demo.toggles.cube { "on" } else { "off" }
                ));
                ui.label(format!(
                    "widths:   {}",
                    if self.demo.toggles.widths {
                        "on"
                    } else {
                        "off"
                    }
                ));
                ui.label(format!(
                    "gradient: {}",
                    if self.demo.toggles.gradient {
                        "on"
                    } else {
                        "off"
                    }
                ));
                ui.label(format!(
                    "tilted:   {}",
                    if self.demo.toggles.tilted {
                        "on"
                    } else {
                        "off"
                    }
                ));
                ui.separator();
                ui.label(format!("samples: {}", self.demo.samples));
                ui.label(format!("camera fps: {:.0}", frame.fps));
                ui.separator();
                ui.label("press ` to open console");
                ui.label("orbit: drag, zoom: scroll, exit: Esc");
            });
        rye_app::log::pump_into(&mut self.console);
        self.console.ui(ctx, &mut self.demo);
        self.last_egui_keyboard = ctx.wants_keyboard_input();
    }

    fn on_event(&mut self, ev: &winit::event::WindowEvent, ctx: &mut FrameCtx<'_>) {
        if !self.last_egui_keyboard {
            self.demo.on_event(ev, ctx);
        }
    }
}

fn main() -> Result<()> {
    let cfg = RunConfig {
        window: WindowAttributes::default()
            .with_title("rye: raster_test")
            .with_visible(false),
        ..RunConfig::default()
    };
    run_with_config::<RasterTestApp>(cfg)
}
