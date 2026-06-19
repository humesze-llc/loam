//! Flatland, Act 1. Left pane: the 3D truth, A Sphere passing through the plane
//! `z = 0`. Right pane: Flatland itself, seen from above, where A Square lives on
//! the plane and perceives the sphere only as the amber loop where it crosses,
//! and ultimately only as the 1D bright segment on his retina line.

use anyhow::Result;
use glam::{Mat4, Vec2, Vec3};
use rye_app::{egui, run, App, FrameCtx, RenderCtx, RunConfig, SetupCtx};
use rye_camera::{Camera, CameraController, OrbitController};
use rye_math::{EuclideanR3, Projection, ZPlane};
use rye_render::{DepthMode, LineRasterNode, Viewport};
use rye_shape::{convex_section_polygon, icosphere, LineMesh, Solid3};
use std::collections::BTreeSet;
use winit::window::WindowAttributes;

const SPHERE_RADIUS: f32 = 1.3;
const SPHERE_SUBDIVISIONS: u32 = 3;
const FLATLAND_HALF_EXTENT: f32 = 3.5;
const FLATLAND_GRID_LINES: usize = 10;

const SQUARE_X: f32 = -3.0;
const SQUARE_Y: f32 = 0.0;
const SQUARE_HALF: f32 = 0.22;
const FOV_HALF_ANGLE: f32 = 0.55;
const RETINA_DISTANCE: f32 = 0.8;

const COLOR_SPHERE: [f32; 4] = [0.45, 0.50, 0.70, 1.0];
const COLOR_GRID: [f32; 4] = [0.14, 0.34, 0.40, 0.6];
const COLOR_SECTION: [f32; 4] = [1.0, 0.70, 0.16, 1.0];
const COLOR_SQUARE: [f32; 4] = [0.45, 0.90, 0.55, 1.0];
const COLOR_FOV: [f32; 4] = [0.35, 0.50, 0.85, 0.45];
const COLOR_RETINA: [f32; 4] = [0.60, 0.60, 0.66, 0.7];
const COLOR_SEES: [f32; 4] = [1.0, 0.55, 0.12, 1.0];
const COLOR_TANGENT: [f32; 4] = [0.80, 0.55, 0.25, 0.4];

#[derive(Copy, Clone, PartialEq, Eq)]
enum Which {
    Sphere,
    Cube,
    Tetra,
}

struct FlatlandApp {
    scene_3d: LineRasterNode,
    flatland_2d: LineRasterNode,
    camera: Camera<EuclideanR3>,
    orbit: OrbitController<EuclideanR3>,
    which: Which,
    center: Vec3,
    section_sides: usize,
}

fn v3(x: f32, y: f32) -> Vec3 {
    Vec3::new(x, y, 0.0)
}

fn push(mesh: &mut LineMesh<3>, a: Vec3, b: Vec3, color: [f32; 4], width: f32) {
    mesh.segments.push((a.to_array(), b.to_array()));
    mesh.colors.push((color, color));
    mesh.widths.push(width);
}

fn push_both(
    m3: &mut LineMesh<3>,
    m2: &mut LineMesh<3>,
    a: Vec3,
    b: Vec3,
    color: [f32; 4],
    w: f32,
) {
    push(m3, a, b, color, w);
    push(m2, a, b, color, w);
}

impl FlatlandApp {
    fn geometry(&self) -> (Vec<Vec3>, Vec<[u32; 2]>) {
        match self.which {
            Which::Sphere => {
                let (verts, faces) = icosphere(SPHERE_SUBDIVISIONS);
                let mut set = BTreeSet::new();
                for t in &faces {
                    for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                        set.insert(if a < b { [a, b] } else { [b, a] });
                    }
                }
                (verts, set.into_iter().collect())
            }
            Which::Cube => (Solid3::Cube.vertices(), Solid3::Cube.edges()),
            Which::Tetra => (Solid3::Tetrahedron.vertices(), Solid3::Tetrahedron.edges()),
        }
    }

    fn build(&self) -> (LineMesh<3>, LineMesh<3>, usize) {
        let mut m3 = LineMesh::<3>::default();
        let mut m2 = LineMesh::<3>::default();

        let (local, edges) = self.geometry();
        let world: Vec<Vec3> = local
            .iter()
            .map(|v| *v * SPHERE_RADIUS + self.center)
            .collect();
        for e in &edges {
            push(
                &mut m3,
                world[e[0] as usize],
                world[e[1] as usize],
                COLOR_SPHERE,
                1.2,
            );
        }

        let r = FLATLAND_HALF_EXTENT;
        for i in 0..=FLATLAND_GRID_LINES {
            let t = -r + 2.0 * r * (i as f32) / (FLATLAND_GRID_LINES as f32);
            push(&mut m3, v3(t, -r), v3(t, r), COLOR_GRID, 1.0);
            push(&mut m3, v3(-r, t), v3(r, t), COLOR_GRID, 1.0);
        }

        let poly = convex_section_polygon(&world, &edges, ZPlane::new(0.0));
        let n = poly.len();
        for i in 0..n {
            let a = poly[i];
            let b = poly[(i + 1) % n];
            push_both(
                &mut m3,
                &mut m2,
                v3(a.x, a.y),
                v3(b.x, b.y),
                COLOR_SECTION,
                3.0,
            );
        }

        let s = Vec2::new(SQUARE_X, SQUARE_Y);
        let facing = Vec2::new(1.0, 0.0);
        let perp = Vec2::new(0.0, 1.0);
        let h = SQUARE_HALF;
        let corners = [
            s + Vec2::new(-h, -h),
            s + Vec2::new(h, -h),
            s + Vec2::new(h, h),
            s + Vec2::new(-h, h),
        ];
        for i in 0..4 {
            let a = corners[i];
            let b = corners[(i + 1) % 4];
            push_both(
                &mut m3,
                &mut m2,
                v3(a.x, a.y),
                v3(b.x, b.y),
                COLOR_SQUARE,
                2.0,
            );
        }
        let tick = s + facing * 0.4;
        push_both(
            &mut m3,
            &mut m2,
            v3(s.x, s.y),
            v3(tick.x, tick.y),
            COLOR_SQUARE,
            2.0,
        );

        let facing_angle = facing.y.atan2(facing.x);
        for sign in [-1.0_f32, 1.0] {
            let a = facing_angle + sign * FOV_HALF_ANGLE;
            let dir = Vec2::new(a.cos(), a.sin());
            let end = s + dir * (2.0 * r);
            push_both(
                &mut m3,
                &mut m2,
                v3(s.x, s.y),
                v3(end.x, end.y),
                COLOR_FOV,
                1.0,
            );
        }

        push_retina(&mut m2, &poly, s, facing, perp);

        (m3, m2, n)
    }
}

/// A Square's 1D retina with the interval A Sphere's section subtends on it: the
/// 2D circle collapses to a single bright segment, the proof that he sees a line.
fn push_retina(mesh: &mut LineMesh<3>, poly: &[Vec2], s: Vec2, facing: Vec2, perp: Vec2) {
    let halfwidth = RETINA_DISTANCE * FOV_HALF_ANGLE.tan();
    let center = s + facing * RETINA_DISTANCE;
    let r0 = center - perp * halfwidth;
    let r1 = center + perp * halfwidth;
    push(mesh, v3(r0.x, r0.y), v3(r1.x, r1.y), COLOR_RETINA, 2.0);

    if poly.len() < 3 {
        return;
    }
    let centroid = poly.iter().copied().fold(Vec2::ZERO, |a, b| a + b) / poly.len() as f32;
    let radius = poly.iter().map(|q| q.distance(centroid)).sum::<f32>() / poly.len() as f32;
    let to_c = centroid - s;
    let dist = to_c.length();
    if dist <= radius {
        return;
    }

    let base = to_c.y.atan2(to_c.x);
    let half = (radius / dist).asin();
    let mut coords = Vec::new();
    for sign in [-1.0_f32, 1.0] {
        let dir = Vec2::from_angle(base + sign * half);
        let denom = dir.dot(facing);
        if denom > 1e-4 {
            let t = (center - s).dot(facing) / denom;
            let hit = s + dir * t;
            coords.push((hit - center).dot(perp));
            push(mesh, v3(s.x, s.y), v3(hit.x, hit.y), COLOR_TANGENT, 1.0);
        }
    }
    if coords.len() == 2 {
        let lo = coords[0].min(coords[1]).clamp(-halfwidth, halfwidth);
        let hi = coords[0].max(coords[1]).clamp(-halfwidth, halfwidth);
        let b0 = center + perp * lo;
        let b1 = center + perp * hi;
        push(mesh, v3(b0.x, b0.y), v3(b1.x, b1.y), COLOR_SEES, 4.0);
    }
}

impl App for FlatlandApp {
    type Space = EuclideanR3;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let make = || {
            LineRasterNode::new(
                &ctx.rd.device,
                ctx.rd.target_format(),
                DepthMode::Off,
                ctx.rd.sample_count(),
            )
        };
        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 2.5, 6.0);
        let mut orbit = OrbitController::<EuclideanR3>::default();
        orbit.set_orbit(7.5, -0.35);

        Ok(Self {
            scene_3d: make(),
            flatland_2d: make(),
            camera,
            orbit,
            which: Which::Sphere,
            center: Vec3::ZERO,
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
        let (m3, m2, sides) = self.build();
        self.section_sides = sides;

        let cfg = &ctx.rd.surface_bundle.config;
        let w = cfg.width;
        let h = cfg.height.max(1);
        let half = (w / 2).max(1);
        let right_w = (w - half).max(1);

        self.scene_3d.upload::<EuclideanR3, 3>(
            &ctx.rd.device,
            &ctx.rd.queue,
            &m3,
            &Projection::Identity,
            1,
        );
        let view = self.camera.view();
        let view_m = Mat4::look_to_rh(view.position, view.forward, view.up);
        let proj = Mat4::perspective_rh(60.0_f32.to_radians(), half as f32 / h as f32, 0.05, 100.0);
        self.scene_3d.set_camera(
            &ctx.rd.queue,
            proj * view_m,
            Vec2::new(half as f32, h as f32),
        );

        self.flatland_2d.upload::<EuclideanR3, 3>(
            &ctx.rd.device,
            &ctx.rd.queue,
            &m2,
            &Projection::Identity,
            1,
        );
        let ext_x = FLATLAND_HALF_EXTENT + 1.0;
        let ext_y = ext_x * h as f32 / right_w as f32;
        let ortho = Mat4::orthographic_rh(-ext_x, ext_x, -ext_y, ext_y, 0.1, 100.0);
        let look = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO, Vec3::Y);
        self.flatland_2d.set_camera(
            &ctx.rd.queue,
            ortho * look,
            Vec2::new(right_w as f32, h as f32),
        );

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

        let left = Viewport {
            x: 0,
            y: 0,
            width: half,
            height: h,
        };
        let right = Viewport {
            x: half,
            y: 0,
            width: right_w,
            height: h,
        };
        self.scene_3d
            .record(ctx.encoder, ctx.view, None, Some(&left));
        self.flatland_2d
            .record(ctx.encoder, ctx.view, None, Some(&right));
        Ok(())
    }

    fn ui(&mut self, ctx: &egui::Context, _frame: &mut FrameCtx<'_>) {
        egui::Area::new(egui::Id::new("label-3d"))
            .anchor(egui::Align2::LEFT_TOP, [14.0, 12.0])
            .show(ctx, |ui| {
                ui.colored_label(egui::Color32::from_rgb(150, 160, 190), "3D truth");
            });
        egui::Area::new(egui::Id::new("label-2d"))
            .anchor(egui::Align2::RIGHT_TOP, [-14.0, 12.0])
            .show(ctx, |ui| {
                ui.colored_label(
                    egui::Color32::from_rgb(210, 170, 90),
                    "Flatland: what A Square sees",
                );
            });

        egui::Window::new("Controls").show(ctx, |ui| {
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
                n => format!("A Square's slice: {n} sides; on his retina, one segment"),
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
