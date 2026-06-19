//! Flatland, an explorable story of the dimensional ladder. One 3D scene: the
//! plane `z = 0` is Flatland, literally a cross-section of the surrounding 3D
//! Spaceland. A single dolly-zoom camera eases between a straight-down view (which
//! reads as pure 2D) and an orbited view that reveals the third dimension. The
//! story plays itself through a sequence of timed beats; A Square, a 2D being,
//! perceives A Sphere only as a circle that appears, swells, and vanishes.

use anyhow::Result;
use glam::{Mat4, Vec2, Vec3};
use rye_app::{egui, run, App, FrameCtx, RunConfig, SetupCtx};
use rye_egui::{ease_in_out_cubic, ease_out_cubic, Animated};
use rye_math::{EuclideanR3, Projection, ZPlane};
use rye_render::device::RenderDevice;
use rye_render::{DepthMode, FragmentShading, LineRasterNode, TriangleRasterNode};
use rye_shape::{convex_section_polygon, fill_convex_polygon, icosphere, LineMesh, TriangleMesh};
use std::collections::BTreeSet;
use std::f32::consts::{PI, TAU};
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
const VISION_HALF_ANGLE: f32 = 0.5;

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
const COLOR_CONE: [f32; 4] = [0.95, 0.78, 0.30, 0.18];
const COLOR_CONE_EDGE: [f32; 4] = [0.80, 0.62, 0.20, 0.7];

#[derive(Clone, Copy, PartialEq)]
enum Cue {
    Hidden,
    Hover,
    Descend,
    Below,
}

struct Beat {
    caption: &'static str,
    secs: f32,
    reveal: f32,
    sleeping: bool,
    vision: bool,
    cue: Cue,
    square: Option<&'static str>,
    sphere: Option<&'static str>,
}

const BEATS: &[Beat] = &[
    Beat {
        caption: "Welcome to Flatland.",
        secs: 3.0,
        reveal: 0.0,
        sleeping: true,
        vision: false,
        cue: Cue::Hidden,
        square: None,
        sphere: None,
    },
    Beat {
        caption: "A flat, two-dimensional world.",
        secs: 3.2,
        reveal: 0.0,
        sleeping: true,
        vision: false,
        cue: Cue::Hidden,
        square: None,
        sphere: None,
    },
    Beat {
        caption: "This is A Square, one of its residents.",
        secs: 3.4,
        reveal: 0.0,
        sleeping: true,
        vision: false,
        cue: Cue::Hidden,
        square: None,
        sphere: None,
    },
    Beat {
        caption: "He wakes. He has never once seen his world from outside it.",
        secs: 4.2,
        reveal: 0.0,
        sleeping: false,
        vision: false,
        cue: Cue::Hidden,
        square: Some("Another ordinary day in Flatland."),
        sphere: None,
    },
    Beat {
        caption: "Yet Flatland is only a slice of a larger 3D world.",
        secs: 5.5,
        reveal: 1.0,
        sleeping: false,
        vision: false,
        cue: Cue::Hidden,
        square: None,
        sphere: None,
    },
    Beat {
        caption: "In 3D, our vision is a 2D projection of that world.",
        secs: 4.5,
        reveal: 1.0,
        sleeping: false,
        vision: false,
        cue: Cue::Hidden,
        square: None,
        sphere: None,
    },
    Beat {
        caption: "In 2D, A Square's vision is just a 1D projection of his.",
        secs: 7.0,
        reveal: 0.0,
        sleeping: false,
        vision: true,
        cue: Cue::Hidden,
        square: None,
        sphere: None,
    },
    Beat {
        caption: "Then a visitor arrives, from the direction he cannot point to.",
        secs: 5.5,
        reveal: 1.0,
        sleeping: false,
        vision: false,
        cue: Cue::Hover,
        square: Some("I see only a circle. Reveal yourself!"),
        sphere: Some("Greetings, Square. I am A Sphere, from a direction beyond your plane."),
    },
    Beat {
        caption: "A Sphere descends through Flatland.",
        secs: PASSAGE_SECONDS + 2.5,
        reveal: 1.0,
        sleeping: false,
        vision: false,
        cue: Cue::Descend,
        square: Some("It grows... it shrinks... it is gone! Sorcery!"),
        sphere: None,
    },
    Beat {
        caption: "From outside his plane we see even his insides, as 4D would see ours. You are A Square.",
        secs: 9.0,
        reveal: 1.0,
        sleeping: false,
        vision: false,
        cue: Cue::Below,
        square: None,
        sphere: None,
    },
];

/// Slowly rotating 2D primitives populating Flatland: (center, sides, radius, spin).
fn primitives() -> [(Vec2, usize, f32, f32); 3] {
    [
        (Vec2::new(1.6, 1.3), 3, 0.34, 0.5),
        (Vec2::new(-1.1, 1.7), 5, 0.30, -0.4),
        (Vec2::new(2.3, -0.4), 4, 0.26, 0.7),
    ]
}

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
    beat_elapsed: f32,
    paused: bool,
    intro: Animated,
    reveal: Animated,
    sphere_z: Animated,
    surprise: Animated,
    surprised: bool,
    wake_shake: f32,
    time: f32,
    gaze_target: Vec2,
    section_sides: usize,
    vision_bands: Vec<(f32, f32)>,
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

/// The angular interval, normalized across A Square's field of view, that a disc
/// of `radius` at `center` subtends relative to his look direction. `None` if he
/// is inside it or it falls outside his field of view.
fn angular_band(s: Vec2, look_ang: f32, center: Vec2, radius: f32) -> Option<(f32, f32)> {
    let to_c = center - s;
    let dist = to_c.length();
    if dist <= radius {
        return None;
    }
    let mut rel = to_c.y.atan2(to_c.x) - look_ang;
    while rel > PI {
        rel -= TAU;
    }
    while rel < -PI {
        rel += TAU;
    }
    let half = (radius / dist).asin();
    let lo = (rel - half) / VISION_HALF_ANGLE;
    let hi = (rel + half) / VISION_HALF_ANGLE;
    if hi < -1.0 || lo > 1.0 {
        return None;
    }
    Some((lo.clamp(-1.0, 1.0), hi.clamp(-1.0, 1.0)))
}

impl FlatlandApp {
    fn enter_beat(&mut self, beat: usize) {
        let was_sleeping = BEATS[self.beat].sleeping;
        self.beat = beat.min(BEATS.len() - 1);
        self.beat_elapsed = 0.0;
        let b = &BEATS[self.beat];
        self.reveal
            .animate_to(b.reveal, REVEAL_SECONDS, ease_in_out_cubic);
        match b.cue {
            Cue::Hidden => self.sphere_z.snap(SPHERE_HIDDEN),
            Cue::Hover => self.sphere_z.animate_to(SPHERE_TOP, 0.9, ease_out_cubic),
            Cue::Descend => {
                self.sphere_z
                    .animate_to(-SPHERE_TOP, PASSAGE_SECONDS, ease_in_out_cubic)
            }
            Cue::Below => self.sphere_z.snap(-SPHERE_TOP),
        }
        if was_sleeping && !b.sleeping {
            self.wake_shake = 0.6;
        }
    }

    fn look_angle(&self) -> f32 {
        let d = self.gaze_target - square_pos();
        if d.length() > 1e-4 {
            d.y.atan2(d.x)
        } else {
            0.0
        }
    }

    fn camera_view_proj(&self, aspect: f32) -> Mat4 {
        let rv = self.reveal.value();
        let el = lerp(CAM_ELEV_DEG.0, CAM_ELEV_DEG.1, rv).to_radians();
        let az = lerp(CAM_AZIM_DEG.0, CAM_AZIM_DEG.1, rv).to_radians();
        let dist = lerp(CAM_DIST.0, CAM_DIST.1, rv) + (1.0 - self.intro.value()) * 9.0;
        let fov = lerp(CAM_FOV_DEG.0, CAM_FOV_DEG.1, rv).to_radians();
        let (se, ce) = el.sin_cos();
        let (sa, ca) = az.sin_cos();
        let eye = Vec3::new(ce * sa, ce * ca, se) * dist;
        let up = Vec3::Y.lerp(Vec3::Z, rv).normalize();
        Mat4::perspective_rh(fov, aspect, 0.1, 80.0) * Mat4::look_at_rh(eye, Vec3::ZERO, up)
    }

    fn face(&self) -> character::Face {
        let pos = square_pos();
        let awake = !BEATS[self.beat].sleeping;
        let breathe = if awake {
            (self.time * 1.3).sin() * 0.012
        } else {
            0.03 + (self.time * 0.9).sin() * 0.05
        };
        let bp = self.time % 3.4;
        let blink = if awake && bp < 0.16 {
            (bp / 0.08 - 1.0).abs().min(1.0)
        } else {
            1.0
        };
        let gaze = self.gaze_target;
        let surprise = self.surprise.value();
        let mut lean = if awake {
            ((gaze.x - pos.x) * 0.10).clamp(-0.28, 0.28)
        } else {
            0.0
        };
        if self.wake_shake > 0.0 {
            lean += (self.time * 32.0).sin() * 0.22 * (self.wake_shake / 0.6);
        }
        let base = if awake { lerp(0.5, 1.0, surprise) } else { 0.0 };
        let left_more = if awake && gaze.x < pos.x {
            0.5 * (1.0 - surprise)
        } else {
            0.0
        };
        let right_more = if awake && gaze.x > pos.x {
            0.5 * (1.0 - surprise)
        } else {
            0.0
        };
        // Eyes ease open as he wakes (alongside the head-shake).
        let wake_open = (1.0 - self.wake_shake / 0.6).clamp(0.0, 1.0);
        character::Face {
            pos,
            half: SQUARE_HALF,
            lean,
            breathe,
            eye_open: [
                ((base + left_more) * blink * wake_open).min(1.0),
                ((base + right_more) * blink * wake_open).min(1.0),
            ],
            look: gaze - pos,
        }
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

        for (c, n, rad, spin) in primitives() {
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

        // A Square's sweeping vision cone (the "his sight" beat).
        if BEATS[self.beat].vision {
            let s = square_pos();
            let la = self.look_angle();
            let reach = 6.0;
            let mut wedge = vec![s];
            for k in 0..=14 {
                let a = la - VISION_HALF_ANGLE + 2.0 * VISION_HALF_ANGLE * (k as f32 / 14.0);
                wedge.push(s + Vec2::from_angle(a) * reach);
            }
            append_tris(&mut tris, &fill_convex_polygon(&wedge, COLOR_CONE));
            for sign in [-1.0_f32, 1.0] {
                let a = la + sign * VISION_HALF_ANGLE;
                let e = s + Vec2::from_angle(a) * reach;
                push(&mut lines, v3(s.x, s.y), v3(e.x, e.y), COLOR_CONE_EDGE, 1.5);
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

        character::push_face(&mut lines, &mut tris, &self.face());

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
            beat_elapsed: 0.0,
            paused: false,
            intro: Animated::new(0.0),
            reveal: Animated::new(0.0),
            sphere_z: Animated::new(SPHERE_HIDDEN),
            surprise: Animated::new(0.0),
            surprised: false,
            wake_shake: 0.0,
            time: 0.0,
            gaze_target: Vec2::new(2.0, 1.0),
            section_sides: 0,
            vision_bands: Vec::new(),
        };
        app.enter_beat(0);
        app.intro.animate_to(1.0, 1.8, ease_out_cubic);
        Ok(app)
    }

    fn space(&self) -> &EuclideanR3 {
        &EuclideanR3
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        let dt = ctx.dt.min(0.1);
        self.time += dt;
        self.intro.advance(dt);
        self.reveal.advance(dt);
        self.sphere_z.advance(dt);
        self.wake_shake = (self.wake_shake - dt).max(0.0);

        if !self.paused && self.beat + 1 < BEATS.len() {
            self.beat_elapsed += dt;
            if self.beat_elapsed >= BEATS[self.beat].secs {
                self.enter_beat(self.beat + 1);
            }
        }

        let beat = &BEATS[self.beat];
        let s = square_pos();

        // Scripted gaze: sweep during the vision beat, watch A Sphere when present.
        if beat.vision {
            self.gaze_target = s + Vec2::from_angle((self.time * 0.7).sin() * 1.1 + 0.4) * 2.0;
        } else if matches!(beat.cue, Cue::Hover | Cue::Descend) {
            self.gaze_target = Vec2::ZERO;
        }

        let crossing = self.sphere_z.value().abs() < SPHERE_RADIUS;
        if crossing != self.surprised {
            self.surprised = crossing;
            self.surprise
                .animate_to(if crossing { 1.0 } else { 0.0 }, 0.3, ease_out_cubic);
        }
        self.surprise.advance(dt);

        // A Square's 1D percept: bands subtended on his retina by A Sphere's
        // section and, during the vision beat, by the 2D primitives his cone sweeps.
        let look = self.look_angle();
        let mut bands = Vec::new();
        let z = self.sphere_z.value();
        if z.abs() < SPHERE_RADIUS {
            let rc = (SPHERE_RADIUS * SPHERE_RADIUS - z * z).sqrt();
            if let Some(b) = angular_band(s, look, Vec2::ZERO, rc) {
                bands.push(b);
            }
        }
        if beat.vision {
            for (c, _, rad, _) in primitives() {
                if let Some(b) = angular_band(s, look, c, rad) {
                    bands.push(b);
                }
            }
        }
        self.vision_bands = bands;
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        let frame = self.build(self.sphere_z.value());
        self.section_sides = frame.sides;

        let cfg = &rd.surface_bundle.config;
        let w = cfg.width;
        let h = cfg.height.max(1);
        let view_proj = self.camera_view_proj(w as f32 / h as f32);

        // Opening fade from dark to the daylight clear colour.
        let iv = self.intro.value() as f64;
        let dark = 0.06;
        let clear = wgpu::Color {
            r: dark + (CLEAR.r - dark) * iv,
            g: dark + (CLEAR.g - dark) * iv,
            b: (dark + 0.02) + (CLEAR.b - dark - 0.02) * iv,
            a: 1.0,
        };

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
                        load: wgpu::LoadOp::Clear(clear),
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
        let beat = &BEATS[self.beat];

        // Cursor gaze only during free, top-down, non-scripted beats.
        let free_look = self.reveal.value() < 0.4 && !beat.vision && beat.cue == Cue::Hidden;
        if free_look {
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

        self.vision_window(ctx);
        self.dialogue(ctx);

        // Caption fades in and out across each beat.
        let fade = {
            let e = self.beat_elapsed;
            let s = beat.secs;
            (e / 0.5)
                .clamp(0.0, 1.0)
                .min(((s - e) / 0.7).clamp(0.0, 1.0))
        };
        let a = (fade * 255.0) as u8;
        egui::Area::new(egui::Id::new("caption"))
            .anchor(egui::Align2::CENTER_BOTTOM, [0.0, -58.0])
            .show(ctx, |ui| {
                egui::Frame::popup(&ctx.style())
                    .fill(egui::Color32::from_rgba_unmultiplied(250, 250, 247, a / 2))
                    .stroke(egui::Stroke::NONE)
                    .show(ui, |ui| {
                        ui.set_max_width(640.0);
                        ui.vertical_centered(|ui| {
                            ui.label(
                                egui::RichText::new(beat.caption)
                                    .size(18.0)
                                    .color(egui::Color32::from_rgba_unmultiplied(30, 34, 40, a)),
                            );
                        });
                    });
            });

        self.controls(ctx);
    }

    fn title(&self, fps: f32) -> std::borrow::Cow<'static, str> {
        format!("flatland  -  {fps:.0} fps").into()
    }
}

impl FlatlandApp {
    fn vision_window(&self, ctx: &egui::Context) {
        egui::Area::new(egui::Id::new("vision-strip"))
            .anchor(egui::Align2::LEFT_TOP, [18.0, 18.0])
            .show(ctx, |ui| {
                ui.label(
                    egui::RichText::new("A Square's vision (1D)")
                        .size(12.0)
                        .color(egui::Color32::from_gray(90)),
                );
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(280.0, 24.0), egui::Sense::hover());
                let painter = ui.painter();
                painter.rect_filled(rect, 4.0, egui::Color32::from_gray(70));
                let x = |n: f32| rect.left() + (n + 1.0) * 0.5 * rect.width();
                for (lo, hi) in &self.vision_bands {
                    let band = egui::Rect::from_min_max(
                        egui::pos2(x(*lo), rect.top()),
                        egui::pos2(x(*hi), rect.bottom()),
                    );
                    painter.rect_filled(band, 3.0, egui::Color32::from_rgb(235, 150, 40));
                }
            });
    }

    fn dialogue(&self, ctx: &egui::Context) {
        let beat = &BEATS[self.beat];
        let screen = ctx.content_rect();
        let top = screen.top() + 70.0;
        if let Some(t) = beat.square {
            speech_bubble(
                ctx,
                "square-bubble",
                egui::pos2(screen.left() + screen.width() * 0.3, top),
                t,
                egui::Color32::from_rgb(40, 130, 110),
            );
        }
        if let Some(t) = beat.sphere {
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
    }

    fn controls(&mut self, ctx: &egui::Context) {
        let mut jump: Option<usize> = None;
        let mut scrub: Option<f32> = None;
        egui::Area::new(egui::Id::new("controls"))
            .anchor(egui::Align2::CENTER_BOTTOM, [0.0, -16.0])
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui
                        .button(if self.paused { "Play" } else { "Pause" })
                        .clicked()
                    {
                        self.paused = !self.paused;
                    }
                    if ui
                        .add_enabled(self.beat > 0, egui::Button::new("Back"))
                        .clicked()
                    {
                        jump = Some(self.beat - 1);
                    }
                    if ui.button("Replay").clicked() {
                        jump = Some(0);
                        self.paused = false;
                    }
                    if BEATS[self.beat].cue == Cue::Descend {
                        let mut z = self.sphere_z.value();
                        if ui
                            .add(
                                egui::Slider::new(&mut z, SPHERE_TOP..=-SPHERE_TOP)
                                    .show_value(false)
                                    .text("scrub"),
                            )
                            .changed()
                        {
                            scrub = Some(z);
                            self.paused = true;
                        }
                    }
                });
            });
        if let Some(z) = scrub {
            self.sphere_z.snap(z);
        }
        if let Some(i) = jump {
            self.enter_beat(i);
        }
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
