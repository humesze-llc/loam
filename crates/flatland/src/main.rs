//! Flatland: a 3D solid (A Sphere, a cube, a tetrahedron) passes through the
//! plane `z = 0`, and the highlighted loop is the 2D cross-section a Flatland
//! inhabitant at that plane would perceive. Orbit to watch the slice change as
//! the solid descends.

use anyhow::Result;
use glam::{Mat4, Vec2, Vec3};
use rye_app::{egui, run, App, FrameCtx, RenderCtx, RunConfig, SetupCtx};
use rye_camera::{Camera, CameraController, OrbitController};
use rye_math::{EuclideanR3, Projection, ZPlane};
use rye_render::{DepthMode, LineRasterNode, Viewport};
use rye_shape::{convex_section_polygon, LineMesh, Solid3};
use winit::window::WindowAttributes;

const SOLID_SCALE: f32 = 1.5;
const FLATLAND_HALF_EXTENT: f32 = 3.0;
const FLATLAND_GRID_LINES: usize = 8;

const COLOR_WIRE: [f32; 4] = [0.42, 0.46, 0.62, 1.0];
const COLOR_FLATLAND: [f32; 4] = [0.16, 0.40, 0.46, 0.7];
const COLOR_SECTION: [f32; 4] = [1.0, 0.70, 0.16, 1.0];

#[derive(Copy, Clone, PartialEq, Eq)]
enum Which {
    Sphere,
    Cube,
    Tetra,
}

impl Which {
    fn solid(self) -> Solid3 {
        match self {
            Which::Sphere => Solid3::Icosahedron,
            Which::Cube => Solid3::Cube,
            Which::Tetra => Solid3::Tetrahedron,
        }
    }
}

struct FlatlandApp {
    lines: LineRasterNode,
    camera: Camera<EuclideanR3>,
    orbit: OrbitController<EuclideanR3>,
    which: Which,
    center: Vec3,
    section_sides: usize,
}

fn push_segment(mesh: &mut LineMesh<3>, a: Vec3, b: Vec3, color: [f32; 4], width: f32) {
    mesh.segments.push((a.to_array(), b.to_array()));
    mesh.colors.push((color, color));
    mesh.widths.push(width);
}

impl FlatlandApp {
    fn build_mesh(&self) -> (LineMesh<3>, usize) {
        let solid = self.which.solid();
        let edges = solid.edges();
        let world: Vec<Vec3> = solid
            .vertices()
            .iter()
            .map(|v| *v * SOLID_SCALE + self.center)
            .collect();

        let mut mesh = LineMesh::<3>::default();

        for e in &edges {
            push_segment(
                &mut mesh,
                world[e[0] as usize],
                world[e[1] as usize],
                COLOR_WIRE,
                1.5,
            );
        }

        let r = FLATLAND_HALF_EXTENT;
        for i in 0..=FLATLAND_GRID_LINES {
            let t = -r + 2.0 * r * (i as f32) / (FLATLAND_GRID_LINES as f32);
            push_segment(
                &mut mesh,
                Vec3::new(t, -r, 0.0),
                Vec3::new(t, r, 0.0),
                COLOR_FLATLAND,
                1.0,
            );
            push_segment(
                &mut mesh,
                Vec3::new(-r, t, 0.0),
                Vec3::new(r, t, 0.0),
                COLOR_FLATLAND,
                1.0,
            );
        }

        let poly = convex_section_polygon(&world, &edges, ZPlane::new(0.0));
        for i in 0..poly.len() {
            let a = poly[i];
            let b = poly[(i + 1) % poly.len()];
            push_segment(
                &mut mesh,
                Vec3::new(a.x, a.y, 0.0),
                Vec3::new(b.x, b.y, 0.0),
                COLOR_SECTION,
                3.0,
            );
        }

        (mesh, poly.len())
    }
}

impl App for FlatlandApp {
    type Space = EuclideanR3;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let lines = LineRasterNode::new(
            &ctx.rd.device,
            ctx.rd.target_format(),
            DepthMode::Off,
            ctx.rd.sample_count(),
        );

        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 2.5, 6.0);
        let mut orbit = OrbitController::<EuclideanR3>::default();
        orbit.set_orbit(7.0, -0.35);

        Ok(Self {
            lines,
            camera,
            orbit,
            which: Which::Sphere,
            center: Vec3::new(0.0, 0.0, 1.2),
            section_sides: 0,
        })
    }

    fn space(&self) -> &EuclideanR3 {
        &EuclideanR3
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        if !ctx.ui_has_focus {
            self.orbit
                .advance(ctx.input, &mut self.camera, &EuclideanR3, ctx.dt.min(0.1));
        }
    }

    fn record(&mut self, ctx: &mut RenderCtx<'_>) -> Result<()> {
        let (mesh, sides) = self.build_mesh();
        self.section_sides = sides;

        let cfg = &ctx.rd.surface_bundle.config;
        self.lines.upload::<EuclideanR3, 3>(
            &ctx.rd.device,
            &ctx.rd.queue,
            &mesh,
            &Projection::Identity,
            1,
        );

        let aspect = cfg.width as f32 / cfg.height.max(1) as f32;
        let view = self.camera.view();
        let view_m = Mat4::look_to_rh(view.position, view.forward, view.up);
        let proj = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.05, 100.0);
        let vp_size = Vec2::new(cfg.width as f32, cfg.height as f32);
        self.lines.set_camera(&ctx.rd.queue, proj * view_m, vp_size);

        {
            let _clear = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("flatland::clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ctx.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.027,
                            g: 0.027,
                            b: 0.035,
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

        let viewport = Viewport::full([cfg.width, cfg.height]);
        self.lines
            .record(ctx.encoder, ctx.view, None, Some(&viewport));
        Ok(())
    }

    fn ui(&mut self, ctx: &egui::Context, _frame: &mut FrameCtx<'_>) {
        egui::Window::new("Flatland").show(ctx, |ui| {
            ui.label("A Sphere passes through Flatland (the z = 0 plane).");
            ui.label("The amber loop is what a 2D inhabitant perceives.");
            ui.separator();
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.which, Which::Sphere, "Sphere");
                ui.selectable_value(&mut self.which, Which::Cube, "Cube");
                ui.selectable_value(&mut self.which, Which::Tetra, "Tetra");
            });
            ui.add(egui::Slider::new(&mut self.center.z, -3.0..=3.0).text("height (z)"));
            ui.add(egui::Slider::new(&mut self.center.x, -3.0..=3.0).text("x"));
            ui.add(egui::Slider::new(&mut self.center.y, -3.0..=3.0).text("y"));
            ui.separator();
            let msg = match self.section_sides {
                0 => "A Square sees: nothing".to_string(),
                n => format!("A Square sees: a {n}-sided shape"),
            };
            ui.label(msg);
        });
    }

    fn title(&self, fps: f32) -> std::borrow::Cow<'static, str> {
        format!("flatland  -  {fps:.0} fps").into()
    }
}

fn main() -> Result<()> {
    run::<FlatlandApp>(RunConfig {
        window: WindowAttributes::default()
            .with_title("flatland")
            .with_visible(false),
        ..RunConfig::default()
    })
}
