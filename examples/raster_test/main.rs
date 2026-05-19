//! Rasterization-pipeline development playground.
//!
//! Renders a curated test scene of line-segment primitives via [`rye_render::LineRasterNode`]:
//! world-axes for depth perception, a unit cube wireframe, a width sweep (1 / 2 / 4 / 8 px) to
//! visually check AA at different widths, a color-gradient line, and a fan of tilted lines to
//! validate AA at all screen-space orientations. Used to validate the rasterizer pipeline in
//! isolation before wiring it into a scene-level demo.
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
//! - `tests <target> <value>`: select what renders. `<target>` is one of:
//!   - `all`, `axes`, `cube`, `widths`, `gradient`, `tilted`: toggle that R³ category;
//!     `<value>` is `on` or `off`. Use `tests all off` to clear the R³ scene when you only
//!     want to see the R⁴ polytope.
//!   - `polytope`: set the R⁴ polytope overlay. `<value>` is one of `5cell`, `tesseract`,
//!     `16cell`, `24cell`, `120cell`, `600cell` (hyphenated aliases accepted) or `off`. The
//!     wireframe projects to R³ via `Orthographic { drop_axis: 3 }`, scaled 2.5x, with an
//!     animated xw + yz rotation so axis-aligned polytopes don't collapse on projection.
//! - `samples N`: set per-segment tessellation density. `1` is the default and is correct for
//!   flat-Euclidean impls; higher values exercise the writer-pattern path that curved-space
//!   impls will use.
//! - `reset`: restore all toggles to default and samples = 1 and polytope off.

use std::borrow::Cow;

use anyhow::Result;
use glam::{Mat4, Vec2, Vec3};
use rye_app::{egui, run_with_config, App, Camera, FrameCtx, OrbitController, RunConfig, SetupCtx};
use rye_egui::{Console, ConsoleWriter};
use rye_math::{EuclideanR3, EuclideanR4, Projection};
use rye_physics::polytope::Polytope4;
use rye_render::{device::RenderDevice, LineRasterNode};
use rye_shape::LineMesh;
use winit::window::WindowAttributes;

/// R⁴ scale factor for the polytope wireframe. Polytope vertices live on the unit
/// 3-sphere; scaling up to 2.5 makes the wireframe comfortably visible alongside the R³
/// test scene without overflowing the cube's ±1.5 extent.
const POLYTOPE_SCALE: f32 = 2.5;

/// Angular speeds (rad/s) of the animated 4D rotation applied to polytope vertices before
/// drop-w projection. Two independent simple rotations: `XW_RATE` rotates in the xw plane
/// (separates the tesseract's inner / outer cubes when projected); `YZ_RATE` rotates inside
/// the projected R³ for visual depth. Non-commensurate rates so the orbit doesn't repeat.
const XW_RATE: f32 = 0.40;
const YZ_RATE: f32 = 0.30;

/// Apply an animated 4D rotation to a single point. Rotates in the xw plane by `t * XW_RATE`
/// and in the yz plane by `t * YZ_RATE`. Used to make w-axis-aligned polytopes (the
/// tesseract is the worst offender) render as visibly 4D under `Orthographic { drop_axis: 3 }`
/// rather than collapsing onto a degenerate R³ silhouette.
fn rotate_4d(p: [f32; 4], t: f32) -> [f32; 4] {
    let (sxw, cxw) = (t * XW_RATE).sin_cos();
    let (syz, cyz) = (t * YZ_RATE).sin_cos();
    let [x, y, z, w] = p;
    [
        x * cxw - w * sxw,
        y * cyz - z * syz,
        y * syz + z * cyz,
        x * sxw + w * cxw,
    ]
}

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
/// console mutates toggles; the rasterizer re-uploads the result via the upload method.
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
    /// R³ rasterizer for the curated test scene (axes, cube, width sweep, etc.).
    line_raster_r3: LineRasterNode,
    /// R⁴ rasterizer for the optional polytope wireframe overlay. Separate from the R³ node
    /// so both pipelines can render in sequence against the same color attachment.
    line_raster_r4: LineRasterNode,
    toggles: Toggles,
    /// Active R⁴ polytope. `None` disables the R⁴ overlay entirely; `Some(p)` uploads `p`'s
    /// `Visualizable<4>` line mesh through the `Orthographic { drop_axis: 3 }` (drop-w)
    /// projection.
    polytope: Option<Polytope4>,
    /// Tessellation samples-per-segment. 1 for flat space (no interior subdivision); higher
    /// values exercise the writer-pattern path that geodesic-space impls will use later.
    samples: usize,
    /// Wall-clock time threaded in from [`FrameCtx::time`] each frame. Drives the animated
    /// 4D rotation in [`rotate_4d`]; only read when `polytope.is_some()`.
    time: f32,
    /// Set by console toggle / `samples` commands to re-upload meshes on the next frame.
    /// Persists across frames so multiple console mutations within one frame coalesce.
    /// When `polytope.is_some()` the per-frame animation flips this every frame so the
    /// rotated mesh re-uploads continuously.
    dirty: bool,
}

impl Demo {
    fn new(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let line_raster_r3 = LineRasterNode::new(
            &ctx.rd.device,
            ctx.rd.surface_bundle.config.format,
            ctx.rd.sample_count(),
        );
        let line_raster_r4 = LineRasterNode::new(
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
            line_raster_r3,
            line_raster_r4,
            toggles: Toggles::default(),
            polytope: None,
            samples: 1,
            time: 0.0,
            dirty: true,
        };
        demo.reupload(ctx.rd);
        Ok(demo)
    }

    /// Re-tessellate and re-upload both R³ and R⁴ meshes. Called whenever toggles, the
    /// active polytope, or `samples` change.
    fn reupload(&mut self, rd: &RenderDevice) {
        let r3_mesh = build_mesh(self.toggles);
        self.line_raster_r3.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &r3_mesh,
            &Projection::Identity,
            self.samples,
        );

        // R⁴ polytope wireframe (if enabled). Vertices are first rotated in R⁴ (so
        // axis-aligned polytopes don't collapse under drop-w; see [`rotate_4d`]) and then
        // scaled up so the wireframe ends up at ~2.5 unit world-space radius. Baking both
        // into the mesh keeps the rasterizer's instance data simple: no per-frame model
        // matrix threaded through the pipeline.
        if let Some(p) = self.polytope {
            // Position-derived per-vertex color: dense wireframes like the 600-cell read as
            // a coherent color field rather than a uniform tangle. See
            // [`Polytope4::lines_colored_by_position`].
            let mut mesh = p.lines_colored_by_position();
            for (a, b) in mesh.segments.iter_mut() {
                *a = rotate_4d(*a, self.time);
                *b = rotate_4d(*b, self.time);
                for k in 0..4 {
                    a[k] *= POLYTOPE_SCALE;
                    b[k] *= POLYTOPE_SCALE;
                }
            }
            self.line_raster_r4.upload::<EuclideanR4, 4>(
                &rd.device,
                &rd.queue,
                &mesh,
                &Projection::Orthographic { drop_axis: 3 },
                self.samples,
            );
        } else {
            // Upload an empty mesh so the R⁴ pass becomes a no-op (zero instances).
            let empty: LineMesh<4> = LineMesh::default();
            self.line_raster_r4.upload::<EuclideanR4, 4>(
                &rd.device,
                &rd.queue,
                &empty,
                &Projection::Orthographic { drop_axis: 3 },
                self.samples,
            );
        }

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
        self.time = ctx.time;
        // Animated 4D rotation: re-upload the polytope mesh every frame while it's
        // visible. ~32-720 edges per polytope so the per-frame upload is cheap.
        if self.polytope.is_some() {
            self.dirty = true;
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
        self.line_raster_r3
            .set_camera(&rd.queue, view_proj, vp_size);
        self.line_raster_r4
            .set_camera(&rd.queue, view_proj, vp_size);

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

        self.line_raster_r3.execute(rd, view)?;
        self.line_raster_r4.execute(rd, view)?;
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
    /// Capture parameters panel (output dir, format, fps, scale, start/stop). Toggled via the
    /// `capture panel` console command or the F11 default bind. The framework runner reads the
    /// post-UI swapchain after every frame, so png / gif / apng / frame-sequence output works
    /// without any per-app render code.
    capture_panel: rye_app::capture::CapturePanel,
}

impl RasterTestApp {
    fn build_console() -> Console<Demo> {
        let mut c = Console::<Demo>::new();
        // `tests` umbrella: typed subcommands instead of one ad-hoc string-dispatch body.
        // The framework parses on/off for toggle subs, completes polytope names at the value
        // slot only when the user has typed `tests polytope ` (context-aware completion).
        c.register(
            rye_egui::subcommands::<Demo>("tests", "toggle what renders in raster_test")
                .toggle(
                    "all",
                    "toggle every R³ raster-test category at once",
                    |d, v| {
                        d.toggles.axes = v;
                        d.toggles.cube = v;
                        d.toggles.widths = v;
                        d.toggles.gradient = v;
                        d.toggles.tilted = v;
                        d.dirty = true;
                        Ok(())
                    },
                )
                .toggle("axes", "toggle world-axes (R/G/B basis vectors)", |d, v| {
                    d.toggles.axes = v;
                    d.dirty = true;
                    Ok(())
                })
                .toggle("cube", "toggle unit-cube wireframe", |d, v| {
                    d.toggles.cube = v;
                    d.dirty = true;
                    Ok(())
                })
                .toggle("widths", "toggle width-sweep horizontal lines", |d, v| {
                    d.toggles.widths = v;
                    d.dirty = true;
                    Ok(())
                })
                .toggle("gradient", "toggle red-to-blue gradient line", |d, v| {
                    d.toggles.gradient = v;
                    d.dirty = true;
                    Ok(())
                })
                .toggle("tilted", "toggle tilted-line fan", |d, v| {
                    d.toggles.tilted = v;
                    d.dirty = true;
                    Ok(())
                })
                .choice(
                    "polytope",
                    "set R⁴ polytope overlay (or `off` to clear it)",
                    &[
                        "off",
                        "5cell",
                        "tesseract",
                        "16cell",
                        "24cell",
                        "120cell",
                        "600cell",
                    ],
                    |d, name| {
                        d.polytope = match name.to_ascii_lowercase().as_str() {
                            "off" | "none" => None,
                            "5cell" | "5-cell" | "pentatope" => Some(Polytope4::Pentatope),
                            "8cell" | "8-cell" | "tesseract" => Some(Polytope4::Tesseract),
                            "16cell" | "16-cell" => Some(Polytope4::Cell16),
                            "24cell" | "24-cell" => Some(Polytope4::Cell24),
                            "120cell" | "120-cell" => Some(Polytope4::Cell120),
                            "600cell" | "600-cell" => Some(Polytope4::Cell600),
                            other => {
                                return Err(anyhow::anyhow!(
                                    "unknown polytope `{other}` (try 5cell, tesseract, \
                                     16cell, 24cell, 120cell, 600cell, or off)"
                                ))
                            }
                        };
                        d.dirty = true;
                        Ok(())
                    },
                ),
        );
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
            "restore all toggles to default (everything on, samples = 1, no polytope)",
            |_args, demo: &mut Demo, _out: &mut ConsoleWriter| {
                demo.toggles = Toggles::default();
                demo.samples = 1;
                demo.polytope = None;
                demo.dirty = true;
                Ok(())
            },
        ));

        // Framework-provided capture: `capture png [pre|post|both] [dir]`,
        // `capture frames|gif|apng [pre|post|both] [dir]`, `capture stop`, `capture panel`.
        // Bound to F12 (one-shot png), F9 (toggle gif sequence), F11 (panel).
        rye_app::capture::register_commands(&mut c);
        rye_app::capture::bind_default_hotkeys(&mut c);

        // Standard framework log mirror so tracing events show up in the console scrollback.
        rye_app::log::register_command(&mut c);
        c
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
            capture_panel: rye_app::capture::CapturePanel::new(),
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
                let polytope_label = match self.demo.polytope {
                    None => Cow::Borrowed("off"),
                    Some(Polytope4::Pentatope) => Cow::Borrowed("5-cell"),
                    Some(Polytope4::Tesseract) => Cow::Borrowed("tesseract"),
                    Some(Polytope4::Cell16) => Cow::Borrowed("16-cell"),
                    Some(Polytope4::Cell24) => Cow::Borrowed("24-cell"),
                    Some(Polytope4::Cell120) => Cow::Borrowed("120-cell"),
                    Some(Polytope4::Cell600) => Cow::Borrowed("600-cell"),
                };
                ui.label(format!("polytope (R⁴): {polytope_label}"));
                ui.label(format!("samples: {}", self.demo.samples));
                ui.label(format!("camera fps: {:.0}", frame.fps));
                ui.separator();
                ui.label("press ` to open console");
                ui.label("orbit: drag, zoom: scroll, exit: Esc");
            });
        self.capture_panel.show(ctx);
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
