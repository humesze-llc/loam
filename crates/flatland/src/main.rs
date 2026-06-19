//! Flatland, an explorable story of the dimensional ladder. One 3D scene: the
//! plane `z = 0` is Flatland, literally a cross-section of the surrounding 3D
//! Spaceland. The camera dolly-zooms between a straight-down view (which reads as
//! pure 2D) and an orbited view that reveals the third dimension. A Square, a 2D
//! being, perceives A Sphere only as a circle that appears, swells, and vanishes.

use anyhow::Result;
use glam::{Mat4, Vec2, Vec3};
use rye_app::{egui, run, App, FrameCtx, RunConfig, SetupCtx};
use rye_egui::{ease_in_out_cubic, ease_out_cubic, Animated};
use rye_math::{EuclideanR3, Projection, ZPlane};
use rye_render::device::RenderDevice;
use rye_render::{DepthMode, FragmentShading, LineRasterNode, TriangleRasterNode};
use rye_shape::{convex_section_polygon, fill_convex_polygon, icosphere, LineMesh, TriangleMesh};
use std::collections::BTreeSet;
use std::f32::consts::TAU;
use winit::window::WindowAttributes;

mod character;

const SPHERE_RADIUS: f32 = 1.3;
const SPHERE_SUBDIVISIONS: u32 = 3;
const SPHERE_TOP: f32 = 2.6;
const SPHERE_HIDDEN: f32 = 9.0;
const PASSAGE_SECONDS: f32 = 6.0;
const REVEAL_SECONDS: f32 = 1.6;

const FLATLAND_HALF_EXTENT: f32 = 3.5;
const FLATLAND_GRID_LINES: usize = 10;
const GROUND_Y: f32 = -2.0;
const SQUARE_X: f32 = -2.6;
const SQUARE_HALF: f32 = 0.30;
const FOV_HALF_ANGLE: f32 = 0.55;

// Camera poses interpolated by `reveal`: a far telephoto top-down (reads as 2D)
// easing to an orbited 3/4 view of Spaceland.
const CAM_DIST: (f32, f32) = (34.0, 9.0);
const CAM_FOV_DEG: (f32, f32) = (13.0, 52.0);
const CAM_ELEV_DEG: (f32, f32) = (89.5, 52.0);
const CAM_AZIM_DEG: (f32, f32) = (0.0, -28.0);

const CLEAR: wgpu::Color = wgpu::Color {
    r: 0.93,
    g: 0.93,
    b: 0.90,
    a: 1.0,
};
const COLOR_SPHERE: [f32; 4] = [0.24, 0.30, 0.46, 1.0];
const COLOR_GRID: [f32; 4] = [0.66, 0.70, 0.70, 0.7];
const COLOR_GROUND: [f32; 4] = [0.30, 0.42, 0.44, 1.0];
const COLOR_PRIMITIVE: [f32; 4] = [0.36, 0.52, 0.52, 1.0];
const COLOR_SHEET: [f32; 4] = [0.52, 0.60, 0.80, 0.16];
const COLOR_SECTION: [f32; 4] = [0.90, 0.46, 0.07, 1.0];
const COLOR_SECTION_FILL: [f32; 4] = [0.95, 0.62, 0.18, 0.50];

struct Beat {
    title: &'static str,
    caption: &'static str,
    square: Option<&'static str>,
    sphere: Option<&'static str>,
}

const BEATS: &[Beat] = &[
    Beat {
        title: "Welcome",
        caption: "Welcome to Flatland, a two-dimensional world. This is A Square, one of its residents.",
        square: None,
        sphere: None,
    },
    Beat {
        title: "His sight",
        caption: "A Square cannot look down on his world. He perceives it only as a line.",
        square: Some("All I see is a line. It is my whole world."),
        sphere: None,
    },
    Beat {
        title: "A slice",
        caption: "Flatland is a single flat slice of a 3D space he cannot point to.",
        square: None,
        sphere: None,
    },
    Beat {
        title: "A Sphere",
        caption: "Something arrives from that hidden direction: A Sphere, hovering above the plane.",
        square: Some("I see only a circle. Reveal yourself!"),
        sphere: Some("Greetings, Square. I am A Sphere, from a direction you cannot point to."),
    },
    Beat {
        title: "Passage",
        caption: "A Sphere descends through Flatland. A Square sees a circle appear, swell, and vanish.",
        square: Some("It grows... it shrinks... it is gone! Sorcery!"),
        sphere: None,
    },
    Beat {
        title: "You",
        caption: "From outside his plane we even see A Square's insides, as 4D would see ours. You are A Square.",
        square: None,
        sphere: None,
    },
];

const BEAT_REVEAL: usize = 2;
const BEAT_SPHERE: usize = 3;
const BEAT_PASSAGE: usize = 4;

struct Frame {
    lines: LineMesh<3>,
    tris: TriangleMesh<3>,
    sides: usize,
}

struct FlatlandApp {
    lines: LineRasterNode,
    tris: TriangleRasterNode,
    sphere_verts: Vec<Vec3>,
    sphere_edges: Vec<[u32; 2]>,
    beat: usize,
    reveal: Animated,
    sphere_z: Animated,
    time: f32,
    section_sides: usize,
    gaze_target: Vec2,
    surprise: Animated,
    surprised: bool,
    vision_band: Option<(f32, f32)>,
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn v3(x: f32, y: f32) -> Vec3 {
    Vec3::new(x, y, 0.0)
}

fn push(mesh: &mut LineMesh<3>, a: Vec3, b: Vec3, color: [f32; 4], width: f32) {
    mesh.segments.push((a.to_array(), b.to_array()));
    mesh.colors.push((color, color));
    mesh.widths.push(width);
}

fn append_tris(dst: &mut TriangleMesh<3>, src: &TriangleMesh<3>) {
    let base = dst.vertices.len() as u32;
    dst.vertices.extend_from_slice(&src.vertices);
    dst.colors.extend_from_slice(&src.colors);
    for t in &src.indices {
        dst.indices.push([t[0] + base, t[1] + base, t[2] + base]);
    }
}

fn square_pos() -> Vec2 {
    Vec2::new(SQUARE_X, GROUND_Y + SQUARE_HALF)
}

impl FlatlandApp {
    fn enter_beat(&mut self, beat: usize) {
        self.beat = beat.min(BEATS.len() - 1);
        let reveal_target = if self.beat >= BEAT_REVEAL { 1.0 } else { 0.0 };
        self.reveal
            .animate_to(reveal_target, REVEAL_SECONDS, ease_in_out_cubic);
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

    fn camera_view_proj(&self, aspect: f32) -> Mat4 {
        let rv = self.reveal.value();
        let el = lerp(CAM_ELEV_DEG.0, CAM_ELEV_DEG.1, rv).to_radians();
        let az = lerp(CAM_AZIM_DEG.0, CAM_AZIM_DEG.1, rv).to_radians();
        let dist = lerp(CAM_DIST.0, CAM_DIST.1, rv);
        let fov = lerp(CAM_FOV_DEG.0, CAM_FOV_DEG.1, rv).to_radians();
        let (se, ce) = el.sin_cos();
        let (sa, ca) = az.sin_cos();
        let eye = Vec3::new(ce * sa, ce * ca, se) * dist;
        let up = Vec3::Y.lerp(Vec3::Z, rv).normalize();
        Mat4::perspective_rh(fov, aspect, 0.1, 80.0) * Mat4::look_at_rh(eye, Vec3::ZERO, up)
    }

    fn build(&self, sphere_z: f32) -> Frame {
        let mut lines = LineMesh::<3>::default();
        let mut tris = TriangleMesh::<3>::default();
        let r = FLATLAND_HALF_EXTENT;

        append_tris(
            &mut tris,
            &fill_convex_polygon(
                &[
                    Vec2::new(-r, -r),
                    Vec2::new(r, -r),
                    Vec2::new(r, r),
                    Vec2::new(-r, r),
                ],
                COLOR_SHEET,
            ),
        );

        for i in 0..=FLATLAND_GRID_LINES {
            let t = -r + 2.0 * r * (i as f32) / (FLATLAND_GRID_LINES as f32);
            push(&mut lines, v3(t, -r), v3(t, r), COLOR_GRID, 1.0);
            push(&mut lines, v3(-r, t), v3(r, t), COLOR_GRID, 1.0);
        }
        push(
            &mut lines,
            v3(-r, GROUND_Y),
            v3(r, GROUND_Y),
            COLOR_GROUND,
            2.5,
        );

        // Slowly rotating 2D primitives populating Flatland.
        for (c, n, rad, spin) in [
            (Vec2::new(1.6, 1.3), 3usize, 0.34, 0.5),
            (Vec2::new(-1.1, 1.7), 5, 0.30, -0.4),
            (Vec2::new(2.3, -0.4), 4, 0.26, 0.7),
        ] {
            let phase = self.time * spin;
            for k in 0..n {
                let a0 = phase + TAU * k as f32 / n as f32;
                let a1 = phase + TAU * (k + 1) as f32 / n as f32;
                let p0 = c + Vec2::from_angle(a0) * rad;
                let p1 = c + Vec2::from_angle(a1) * rad;
                push(
                    &mut lines,
                    v3(p0.x, p0.y),
                    v3(p1.x, p1.y),
                    COLOR_PRIMITIVE,
                    1.5,
                );
            }
        }

        let center = Vec3::new(0.0, 0.0, sphere_z);
        let world: Vec<Vec3> = self.sphere_verts.iter().map(|v| *v + center).collect();
        if sphere_z < SPHERE_TOP + 1.0 {
            for e in &self.sphere_edges {
                push(
                    &mut lines,
                    world[e[0] as usize],
                    world[e[1] as usize],
                    COLOR_SPHERE,
                    1.2,
                );
            }
        }

        let poly = convex_section_polygon(&world, &self.sphere_edges, ZPlane::new(0.0));
        let sides = poly.len();
        if sides >= 3 {
            append_tris(&mut tris, &fill_convex_polygon(&poly, COLOR_SECTION_FILL));
            for i in 0..sides {
                let a = poly[i];
                let b = poly[(i + 1) % sides];
                push(&mut lines, v3(a.x, a.y), v3(b.x, b.y), COLOR_SECTION, 3.0);
            }
        }

        let gaze = if self.beat >= BEAT_SPHERE && sphere_z < SPHERE_TOP + 1.0 {
            Vec2::ZERO
        } else {
            self.gaze_target
        };
        character::push_face(
            &mut lines,
            &mut tris,
            &character::Face {
                pos: square_pos(),
                half: SQUARE_HALF,
                gaze,
                surprise: self.surprise.value(),
            },
        );

        Frame { lines, tris, sides }
    }
}

impl App for FlatlandApp {
    type Space = EuclideanR3;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let line_node = LineRasterNode::new(
            &ctx.rd.device,
            ctx.rd.target_format(),
            DepthMode::Off,
            ctx.rd.sample_count(),
        );
        let tri_node = TriangleRasterNode::new(
            &ctx.rd.device,
            ctx.rd.target_format(),
            DepthMode::Off,
            FragmentShading::Flat,
            ctx.rd.sample_count(),
        );

        let (raw, faces) = icosphere(SPHERE_SUBDIVISIONS);
        let sphere_verts = raw.iter().map(|v| *v * SPHERE_RADIUS).collect();
        let mut set = BTreeSet::new();
        for t in &faces {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                set.insert(if a < b { [a, b] } else { [b, a] });
            }
        }

        let mut app = Self {
            lines: line_node,
            tris: tri_node,
            sphere_verts,
            sphere_edges: set.into_iter().collect(),
            beat: 0,
            reveal: Animated::new(0.0),
            sphere_z: Animated::new(SPHERE_HIDDEN),
            time: 0.0,
            section_sides: 0,
            gaze_target: Vec2::new(2.0, 1.0),
            surprise: Animated::new(0.0),
            surprised: false,
            vision_band: None,
        };
        app.enter_beat(0);
        Ok(app)
    }

    fn space(&self) -> &EuclideanR3 {
        &EuclideanR3
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        let dt = ctx.dt.min(0.1);
        self.time += dt;
        self.reveal.advance(dt);
        self.sphere_z.advance(dt);

        let crossing = self.sphere_z.value().abs() < SPHERE_RADIUS;
        if crossing != self.surprised {
            self.surprised = crossing;
            self.surprise
                .animate_to(if crossing { 1.0 } else { 0.0 }, 0.3, ease_out_cubic);
        }
        self.surprise.advance(dt);

        let z = self.sphere_z.value();
        self.vision_band = if z.abs() < SPHERE_RADIUS {
            let rc = (SPHERE_RADIUS * SPHERE_RADIUS - z * z).sqrt();
            let to_c = Vec2::ZERO - square_pos();
            let dist = to_c.length();
            (dist > rc).then(|| {
                let base = to_c.y.atan2(to_c.x);
                let half = (rc / dist).asin();
                let lo = ((base - half) / FOV_HALF_ANGLE).clamp(-1.0, 1.0);
                let hi = ((base + half) / FOV_HALF_ANGLE).clamp(-1.0, 1.0);
                (lo, hi)
            })
        } else {
            None
        };
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        let frame = self.build(self.sphere_z.value());
        self.section_sides = frame.sides;

        let cfg = &rd.surface_bundle.config;
        let w = cfg.width;
        let h = cfg.height.max(1);
        let view_proj = self.camera_view_proj(w as f32 / h as f32);

        {
            let mut enc = rd
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("flatland::clear"),
                });
            let _clear = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("flatland::clear pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
            drop(_clear);
            rd.queue.submit(Some(enc.finish()));
        }

        self.tris.set_camera(&rd.queue, view_proj);
        self.tris.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &frame.tris,
            &Projection::Identity,
        );
        self.tris.execute(rd, view, None, None)?;
        self.lines
            .set_camera(&rd.queue, view_proj, Vec2::new(w as f32, h as f32));
        self.lines.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &frame.lines,
            &Projection::Identity,
            1,
        );
        self.lines.execute(rd, view, None, None)?;
        Ok(())
    }

    fn ui(&mut self, ctx: &egui::Context, _frame: &mut FrameCtx<'_>) {
        // Cursor gaze, only meaningful while the camera reads as top-down 2D.
        if self.reveal.value() < 0.4 {
            if let Some(p) = ctx.pointer_latest_pos() {
                let screen = ctx.content_rect();
                let rv = self.reveal.value();
                let dist = lerp(CAM_DIST.0, CAM_DIST.1, rv);
                let fov = lerp(CAM_FOV_DEG.0, CAM_FOV_DEG.1, rv).to_radians();
                let ext_y = dist * (fov * 0.5).tan();
                let ext_x = ext_y * screen.width() / screen.height();
                let wx = (((p.x - screen.left()) / screen.width()) * 2.0 - 1.0) * ext_x;
                let wy = (1.0 - ((p.y - screen.top()) / screen.height()) * 2.0) * ext_y;
                self.gaze_target = Vec2::new(wx, wy);
            }
        }

        egui::Area::new(egui::Id::new("vision-strip"))
            .anchor(egui::Align2::LEFT_BOTTOM, [18.0, -92.0])
            .show(ctx, |ui| {
                ui.label(
                    egui::RichText::new("A Square's eye (1D)")
                        .size(12.0)
                        .color(egui::Color32::from_gray(90)),
                );
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(280.0, 26.0), egui::Sense::hover());
                let painter = ui.painter();
                painter.rect_filled(rect, 4.0, egui::Color32::from_gray(70));
                if let Some((lo, hi)) = self.vision_band {
                    let x = |n: f32| rect.left() + (n + 1.0) * 0.5 * rect.width();
                    let band = egui::Rect::from_min_max(
                        egui::pos2(x(lo), rect.top()),
                        egui::pos2(x(hi), rect.bottom()),
                    );
                    painter.rect_filled(band, 4.0, egui::Color32::from_rgb(235, 150, 40));
                }
            });

        let screen = ctx.content_rect();
        let top = screen.top() + 64.0;
        if let Some(t) = BEATS[self.beat].square {
            speech_bubble(
                ctx,
                "square-bubble",
                egui::pos2(screen.left() + screen.width() * 0.3, top),
                t,
                egui::Color32::from_rgb(40, 130, 110),
            );
        }
        if let Some(t) = BEATS[self.beat].sphere {
            if self.reveal.value() > 0.6 {
                speech_bubble(
                    ctx,
                    "sphere-bubble",
                    egui::pos2(screen.left() + screen.width() * 0.7, top),
                    t,
                    egui::Color32::from_rgb(60, 90, 170),
                );
            }
        }

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

fn speech_bubble(
    ctx: &egui::Context,
    id: &str,
    pos: egui::Pos2,
    text: &str,
    accent: egui::Color32,
) {
    egui::Area::new(egui::Id::new(id))
        .fixed_pos(pos)
        .pivot(egui::Align2::CENTER_TOP)
        .show(ctx, |ui| {
            egui::Frame::popup(&ctx.style())
                .fill(egui::Color32::from_rgb(252, 252, 250))
                .stroke(egui::Stroke::new(1.5, accent))
                .show(ui, |ui| {
                    ui.set_max_width(250.0);
                    ui.label(
                        egui::RichText::new(text)
                            .size(14.0)
                            .color(egui::Color32::from_gray(30)),
                    );
                });
        });
}

fn main() -> Result<()> {
    run::<FlatlandApp>(RunConfig {
        window: WindowAttributes::default()
            .with_title("flatland")
            .with_visible(false),
        ..RunConfig::default()
    })
}
