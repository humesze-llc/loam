//! Flatland, an explorable story of the dimensional ladder. The viewer is a 3D
//! (then 4D) observer watching A Square, a 2D being, who can perceive A Sphere
//! only as a circle that appears, swells, and vanishes. A guided sequence of
//! beats drives a tweened reveal: Flatland alone, then Flatland shown as a slice
//! of the 3D space it lives in.
//!
//! Phase A skeleton: beat-spine + tweened layout + bottom story bar + daylight
//! palette. A Square's face, filled sections, depth, the 1D vision strip, the
//! dialogue, and Act 2 land in later phases.

use anyhow::Result;
use glam::{Mat4, Vec2, Vec3};
use rye_app::{egui, run, App, FrameCtx, RenderCtx, RunConfig, SetupCtx};
use rye_camera::{Camera, CameraController, OrbitController};
use rye_egui::{ease_in_out_cubic, ease_out_cubic, Animated};
use rye_math::{EuclideanR3, Projection, ZPlane};
use rye_render::{DepthMode, LineRasterNode, Viewport};
use rye_shape::{convex_section_polygon, icosphere, LineMesh};
use std::collections::BTreeSet;
use winit::window::WindowAttributes;

const SPHERE_RADIUS: f32 = 1.3;
const SPHERE_SUBDIVISIONS: u32 = 3;
const SPHERE_TOP: f32 = 2.6;
const SPHERE_HIDDEN: f32 = 9.0;
const PASSAGE_SECONDS: f32 = 6.0;
const REVEAL_SECONDS: f32 = 0.9;

const FLATLAND_HALF_EXTENT: f32 = 3.5;
const FLATLAND_GRID_LINES: usize = 10;
const SQUARE_X: f32 = -3.0;
const SQUARE_Y: f32 = 0.0;
const SQUARE_HALF: f32 = 0.22;
const FOV_HALF_ANGLE: f32 = 0.55;
const RETINA_DISTANCE: f32 = 0.8;

const CLEAR: wgpu::Color = wgpu::Color {
    r: 0.93,
    g: 0.93,
    b: 0.90,
    a: 1.0,
};
const COLOR_SPHERE: [f32; 4] = [0.24, 0.30, 0.46, 1.0];
const COLOR_GRID: [f32; 4] = [0.62, 0.66, 0.66, 0.8];
const COLOR_SECTION: [f32; 4] = [0.93, 0.52, 0.10, 1.0];
const COLOR_SQUARE: [f32; 4] = [0.14, 0.44, 0.40, 1.0];
const COLOR_FOV: [f32; 4] = [0.40, 0.50, 0.78, 0.5];
const COLOR_RETINA: [f32; 4] = [0.40, 0.40, 0.46, 0.85];
const COLOR_SEES: [f32; 4] = [0.93, 0.42, 0.06, 1.0];
const COLOR_TANGENT: [f32; 4] = [0.82, 0.55, 0.30, 0.5];

struct Beat {
    title: &'static str,
    caption: &'static str,
}

const BEATS: &[Beat] = &[
    Beat {
        title: "Welcome",
        caption: "Welcome to Flatland. This is A Square, a being of two dimensions.",
    },
    Beat {
        title: "His sight",
        caption: "A Square cannot look down on his world. He perceives it only as a line.",
    },
    Beat {
        title: "A slice",
        caption: "His whole world is a single flat slice of a 3D space he cannot point to.",
    },
    Beat {
        title: "A Sphere",
        caption:
            "Something arrives from that hidden direction: A Sphere, hovering above the plane.",
    },
    Beat {
        title: "Passage",
        caption:
            "A Sphere descends through Flatland. A Square sees a circle appear, swell, and vanish.",
    },
    Beat {
        title: "You",
        caption: "A Square could not picture the sphere. Now imagine 4D passing through us.",
    },
];

const BEAT_REVEAL: usize = 2;
const BEAT_SPHERE: usize = 3;
const BEAT_PASSAGE: usize = 4;

struct FlatlandApp {
    scene_3d: LineRasterNode,
    flatland_2d: LineRasterNode,
    camera: Camera<EuclideanR3>,
    orbit: OrbitController<EuclideanR3>,
    sphere_verts: Vec<Vec3>,
    sphere_edges: Vec<[u32; 2]>,
    beat: usize,
    split: Animated,
    sphere_z: Animated,
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
    fn enter_beat(&mut self, beat: usize) {
        self.beat = beat.min(BEATS.len() - 1);
        let split_target = if self.beat >= BEAT_REVEAL { 1.0 } else { 0.0 };
        self.split
            .animate_to(split_target, REVEAL_SECONDS, ease_in_out_cubic);
        match self.beat {
            b if b < BEAT_SPHERE => self.sphere_z.snap(SPHERE_HIDDEN),
            BEAT_SPHERE => self.sphere_z.animate_to(SPHERE_TOP, 0.8, ease_out_cubic),
            BEAT_PASSAGE => {
                self.sphere_z
                    .animate_to(-SPHERE_TOP, PASSAGE_SECONDS, ease_in_out_cubic)
            }
            _ => self.sphere_z.snap(-SPHERE_TOP),
        }
    }

    fn build(&self, sphere_z: f32) -> (LineMesh<3>, LineMesh<3>, usize) {
        let mut m3 = LineMesh::<3>::default();
        let mut m2 = LineMesh::<3>::default();

        let center = Vec3::new(0.0, 0.0, sphere_z);
        let world: Vec<Vec3> = self.sphere_verts.iter().map(|v| *v + center).collect();
        if sphere_z < SPHERE_TOP + 1.0 {
            for e in &self.sphere_edges {
                push(
                    &mut m3,
                    world[e[0] as usize],
                    world[e[1] as usize],
                    COLOR_SPHERE,
                    1.2,
                );
            }
        }

        let r = FLATLAND_HALF_EXTENT;
        for i in 0..=FLATLAND_GRID_LINES {
            let t = -r + 2.0 * r * (i as f32) / (FLATLAND_GRID_LINES as f32);
            push(&mut m3, v3(t, -r), v3(t, r), COLOR_GRID, 1.0);
            push(&mut m3, v3(-r, t), v3(r, t), COLOR_GRID, 1.0);
        }

        let poly = convex_section_polygon(&world, &self.sphere_edges, ZPlane::new(0.0));
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

/// A Square's 1D retina with the interval A Sphere's section subtends on it.
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

        let (raw, faces) = icosphere(SPHERE_SUBDIVISIONS);
        let sphere_verts = raw.iter().map(|v| *v * SPHERE_RADIUS).collect();
        let mut set = BTreeSet::new();
        for t in &faces {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                set.insert(if a < b { [a, b] } else { [b, a] });
            }
        }

        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 2.5, 6.0);
        let mut orbit = OrbitController::<EuclideanR3>::default();
        orbit.set_orbit(7.5, -0.35);

        let mut app = Self {
            scene_3d: make(),
            flatland_2d: make(),
            camera,
            orbit,
            sphere_verts,
            sphere_edges: set.into_iter().collect(),
            beat: 0,
            split: Animated::new(0.0),
            sphere_z: Animated::new(SPHERE_HIDDEN),
            section_sides: 0,
        };
        app.enter_beat(0);
        Ok(app)
    }

    fn space(&self) -> &EuclideanR3 {
        &EuclideanR3
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        let dt = ctx.dt.min(0.1);
        self.split.advance(dt);
        self.sphere_z.advance(dt);
        if self.split.value() > 0.5 && !ctx.ui_has_focus {
            self.orbit
                .advance(ctx.input, &mut self.camera, &EuclideanR3, dt);
        }
    }

    fn record(&mut self, ctx: &mut RenderCtx<'_>) -> Result<()> {
        let (m3, m2, sides) = self.build(self.sphere_z.value());
        self.section_sides = sides;

        let cfg = &ctx.rd.surface_bundle.config;
        let w = cfg.width;
        let h = cfg.height.max(1);
        let split = self.split.value();
        let twod_w = (w as f32 * (1.0 - 0.5 * split))
            .round()
            .clamp(1.0, w as f32) as u32;
        let threed_w = w.saturating_sub(twod_w);

        self.flatland_2d.upload::<EuclideanR3, 3>(
            &ctx.rd.device,
            &ctx.rd.queue,
            &m2,
            &Projection::Identity,
            1,
        );
        let ext_x = FLATLAND_HALF_EXTENT + 1.0;
        let ext_y = ext_x * h as f32 / twod_w as f32;
        let ortho = Mat4::orthographic_rh(-ext_x, ext_x, -ext_y, ext_y, 0.1, 100.0);
        let look = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO, Vec3::Y);
        self.flatland_2d.set_camera(
            &ctx.rd.queue,
            ortho * look,
            Vec2::new(twod_w as f32, h as f32),
        );

        if threed_w > 0 {
            self.scene_3d.upload::<EuclideanR3, 3>(
                &ctx.rd.device,
                &ctx.rd.queue,
                &m3,
                &Projection::Identity,
                1,
            );
            let view = self.camera.view();
            let view_m = Mat4::look_to_rh(view.position, view.forward, view.up);
            let proj = Mat4::perspective_rh(
                60.0_f32.to_radians(),
                threed_w as f32 / h as f32,
                0.05,
                100.0,
            );
            self.scene_3d.set_camera(
                &ctx.rd.queue,
                proj * view_m,
                Vec2::new(threed_w as f32, h as f32),
            );
        }

        {
            let _clear = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("flatland::clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ctx.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(CLEAR),
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
            width: twod_w,
            height: h,
        };
        self.flatland_2d
            .record(ctx.encoder, ctx.view, None, Some(&left));
        if threed_w > 0 {
            let right = Viewport {
                x: twod_w,
                y: 0,
                width: threed_w,
                height: h,
            };
            self.scene_3d
                .record(ctx.encoder, ctx.view, None, Some(&right));
        }
        Ok(())
    }

    fn ui(&mut self, ctx: &egui::Context, _frame: &mut FrameCtx<'_>) {
        let mut jump_to: Option<usize> = None;
        let mut scrub: Option<f32> = None;

        egui::TopBottomPanel::bottom("story-bar").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.label(egui::RichText::new(BEATS[self.beat].caption).size(15.0));
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(self.beat > 0, egui::Button::new("\u{25c0} Back"))
                    .clicked()
                {
                    jump_to = Some(self.beat - 1);
                }
                for (i, b) in BEATS.iter().enumerate() {
                    if ui.selectable_label(i == self.beat, b.title).clicked() {
                        jump_to = Some(i);
                    }
                }
                if ui
                    .add_enabled(
                        self.beat < BEATS.len() - 1,
                        egui::Button::new("Next \u{25b6}"),
                    )
                    .clicked()
                {
                    jump_to = Some(self.beat + 1);
                }
            });
            if self.beat == BEAT_PASSAGE {
                let mut z = self.sphere_z.value();
                if ui
                    .add(
                        egui::Slider::new(&mut z, SPHERE_TOP..=-SPHERE_TOP)
                            .text("A Sphere's height"),
                    )
                    .changed()
                {
                    scrub = Some(z);
                }
            }
            ui.add_space(4.0);
        });

        if let Some(z) = scrub {
            self.sphere_z.snap(z);
        }
        if let Some(i) = jump_to {
            self.enter_beat(i);
        }
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
