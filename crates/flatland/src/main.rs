//! Flatland, an explorable story of the dimensional ladder. One 3D scene: the
//! plane `z = 0` is Flatland, an upright pane standing in a larger 3D Spaceland
//! (a checkerboard floor, a sky, distant tumbling solids). A single camera orbits
//! from a head-on view (which reads as pure 2D) around to reveal the third
//! dimension, while Flatland stays upright. A Sphere arrives along the hidden
//! depth axis; A Square, a 2D being, perceives it only as a circle.

use anyhow::Result;
use glam::{Mat3, Mat4, Vec2, Vec3};
use rye_app::{egui, run, App, FrameCtx, RunConfig, SetupCtx};
use rye_egui::{ease_in_out_cubic, ease_out_cubic, Animated};
use rye_math::{EuclideanR3, Projection, ZPlane};
use rye_render::device::RenderDevice;
use rye_render::{DepthBuffer, DepthMode, FragmentShading, LineRasterNode, TriangleRasterNode};
use rye_shape::{convex_section_polygon, icosphere, LineMesh, Solid3, TriangleMesh};
use std::collections::BTreeSet;
use std::f32::consts::{PI, TAU};
use winit::window::WindowAttributes;

mod character;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

const SPHERE_RADIUS: f32 = 1.3;
const SPHERE_SUBDIVISIONS: u32 = 3;
const SPHERE_TOP: f32 = 2.6;
const SPHERE_HIDDEN: f32 = 9.0;
const SPHERE_XY: (f32, f32) = (1.5, 0.6);
const PASSAGE_SECONDS: f32 = 6.0;
const REVEAL_SECONDS: f32 = 1.6;

const FLATLAND_HALF: f32 = 2.8;
const FLATLAND_CELLS: usize = 10;
const FLOOR_Y: f32 = -FLATLAND_HALF;
const ON_PANE: f32 = 0.0;
const PANE_Z: f32 = -0.02;
const VISION_HALF_ANGLE: f32 = 0.5;

const CAM_DIST: (f32, f32) = (24.0, 11.0);
const CAM_FOV_DEG: (f32, f32) = (15.0, 50.0);
const CAM_AZIM_DEG: (f32, f32) = (0.0, 34.0);
const CAM_ELEV_DEG: (f32, f32) = (0.0, 16.0);
fn cam_target() -> Vec3 {
    Vec3::new(0.6, 0.3, 0.0)
}

const SKY: wgpu::Color = wgpu::Color {
    r: 0.74,
    g: 0.82,
    b: 0.90,
    a: 1.0,
};
const COLOR_SPHERE: [f32; 4] = [0.86, 0.45, 0.30, 1.0];
const FLATLAND_A: [f32; 4] = [0.92, 0.92, 0.88, 1.0];
const FLATLAND_B: [f32; 4] = [0.85, 0.87, 0.84, 1.0];
const FLOOR_A: [f32; 4] = [0.56, 0.62, 0.58, 1.0];
const FLOOR_B: [f32; 4] = [0.49, 0.56, 0.53, 1.0];
const COLOR_PRIMITIVE: [f32; 4] = [0.30, 0.46, 0.48, 1.0];
const COLOR_DISTANT: [f32; 4] = [0.52, 0.60, 0.66, 1.0];
const COLOR_SECTION: [f32; 4] = [0.90, 0.40, 0.06, 1.0];
const COLOR_SECTION_FILL: [f32; 4] = [0.95, 0.55, 0.15, 0.6];
const COLOR_CONE: [f32; 4] = [0.95, 0.78, 0.30, 0.20];
const COLOR_CONE_EDGE: [f32; 4] = [0.80, 0.60, 0.18, 0.8];

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
        secs: 3.4,
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
        secs: 6.0,
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
        sphere: Some("Greetings, Square. I am A Sphere, from beyond your plane."),
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

/// Slowly rotating 2D primitives on Flatland: (center, sides, radius, spin).
fn primitives() -> [(Vec2, usize, f32, f32); 3] {
    [
        (Vec2::new(-1.7, 1.3), 3, 0.34, 0.5),
        (Vec2::new(-1.9, -1.2), 5, 0.30, -0.4),
        (Vec2::new(2.1, -1.6), 4, 0.26, 0.7),
    ]
}

/// Distant tumbling solids in Spaceland: (center, solid, scale, spin).
fn distants() -> [(Vec3, Solid3, f32, f32); 4] {
    [
        (Vec3::new(5.5, 1.0, -8.0), Solid3::Cube, 1.0, 0.5),
        (Vec3::new(-6.5, 2.6, -11.0), Solid3::Icosahedron, 1.3, -0.35),
        (Vec3::new(4.0, -0.6, -14.0), Solid3::Tetrahedron, 1.5, 0.4),
        (Vec3::new(-3.5, 3.4, -17.0), Solid3::Cube, 1.7, -0.25),
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
    depth: Option<DepthBuffer>,
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
    Vec3::new(x, y, ON_PANE)
}

fn push(mesh: &mut LineMesh<3>, a: Vec3, b: Vec3, color: [f32; 4], width: f32) {
    mesh.segments.push((a.to_array(), b.to_array()));
    mesh.colors.push((color, color));
    mesh.widths.push(width);
}

fn quad3(tris: &mut TriangleMesh<3>, p: [Vec3; 4], color: [f32; 4]) {
    let base = tris.vertices.len() as u32;
    for v in p {
        tris.vertices.push(v.to_array());
        tris.colors.push(color);
    }
    tris.indices.push([base, base + 1, base + 2]);
    tris.indices.push([base, base + 2, base + 3]);
}

fn square_pos() -> Vec2 {
    Vec2::new(0.0, 0.0)
}

fn sphere_xy() -> Vec2 {
    Vec2::new(SPHERE_XY.0, SPHERE_XY.1)
}

/// The angular interval, normalized across A Square's field of view, that a disc
/// of `radius` at `center` subtends relative to his look direction.
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
        let az = lerp(CAM_AZIM_DEG.0, CAM_AZIM_DEG.1, rv).to_radians();
        let el = lerp(CAM_ELEV_DEG.0, CAM_ELEV_DEG.1, rv).to_radians();
        let dist = lerp(CAM_DIST.0, CAM_DIST.1, rv) + (1.0 - self.intro.value()) * 8.0;
        let fov = lerp(CAM_FOV_DEG.0, CAM_FOV_DEG.1, rv).to_radians();
        let dir = Vec3::new(az.sin() * el.cos(), el.sin(), az.cos() * el.cos());
        let eye = cam_target() + dir * dist;
        Mat4::perspective_rh(fov, aspect, 0.1, 120.0) * Mat4::look_at_rh(eye, cam_target(), Vec3::Y)
    }

    fn face(&self) -> character::Face {
        let pos = square_pos();
        let awake = !BEATS[self.beat].sleeping;
        let look = self.gaze_target - pos;
        let idle = if awake {
            (self.time * 1.3).sin() * 0.012
        } else {
            0.03 + (self.time * 0.9).sin() * 0.05
        };
        let reach = if awake {
            (look.y * 0.05).clamp(-0.10, 0.16)
        } else {
            0.0
        };
        let bp = self.time % 3.4;
        let blink = if awake && bp < 0.16 {
            (bp / 0.08 - 1.0).abs().min(1.0)
        } else {
            1.0
        };
        let surprise = self.surprise.value();
        let mut lean = if awake {
            (look.x * 0.09).clamp(-0.28, 0.28)
        } else {
            0.0
        };
        if self.wake_shake > 0.0 {
            lean += (self.time * 32.0).sin() * 0.22 * (self.wake_shake / 0.6);
        }
        let base = if awake { lerp(0.5, 1.0, surprise) } else { 0.0 };
        let left_more = if awake && look.x < 0.0 {
            0.5 * (1.0 - surprise)
        } else {
            0.0
        };
        let right_more = if awake && look.x > 0.0 {
            0.5 * (1.0 - surprise)
        } else {
            0.0
        };
        let wake_open = (1.0 - self.wake_shake / 0.6).clamp(0.0, 1.0);
        character::Face {
            pos,
            half: 0.34,
            lean,
            stretch: idle + reach,
            eye_open: [
                ((base + left_more) * blink * wake_open).min(1.0),
                ((base + right_more) * blink * wake_open).min(1.0),
            ],
            look,
        }
    }

    fn build(&self, sphere_z: f32) -> Frame {
        let mut lines = LineMesh::<3>::default();
        let mut tris = TriangleMesh::<3>::default();

        // Spaceland floor, a checkerboard receding into the distance.
        for ix in -10..10 {
            for iz in -16..3 {
                let (x0, z0) = (ix as f32 * 1.2, iz as f32 * 1.2);
                let c = if (ix + iz) & 1 == 0 { FLOOR_A } else { FLOOR_B };
                quad3(
                    &mut tris,
                    [
                        Vec3::new(x0, FLOOR_Y, z0),
                        Vec3::new(x0 + 1.2, FLOOR_Y, z0),
                        Vec3::new(x0 + 1.2, FLOOR_Y, z0 + 1.2),
                        Vec3::new(x0, FLOOR_Y, z0 + 1.2),
                    ],
                    c,
                );
            }
        }

        // Distant tumbling solids that make Spaceland's depth legible.
        for (center, solid, scale, spin) in distants() {
            let rot = Mat3::from_rotation_y(self.time * spin)
                * Mat3::from_rotation_x(self.time * spin * 0.5);
            let vs: Vec<Vec3> = solid
                .vertices()
                .iter()
                .map(|v| center + rot * (*v * scale))
                .collect();
            for e in solid.edges() {
                push(
                    &mut lines,
                    vs[e[0] as usize],
                    vs[e[1] as usize],
                    COLOR_DISTANT,
                    1.2,
                );
            }
        }

        // Flatland: the upright pane at z = 0, its own checkerboard.
        let h = FLATLAND_HALF;
        let step = 2.0 * h / FLATLAND_CELLS as f32;
        for i in 0..FLATLAND_CELLS {
            for j in 0..FLATLAND_CELLS {
                let (x0, y0) = (-h + i as f32 * step, -h + j as f32 * step);
                let c = if (i + j) & 1 == 0 {
                    FLATLAND_A
                } else {
                    FLATLAND_B
                };
                quad3(
                    &mut tris,
                    [
                        Vec3::new(x0, y0, PANE_Z),
                        Vec3::new(x0 + step, y0, PANE_Z),
                        Vec3::new(x0 + step, y0 + step, PANE_Z),
                        Vec3::new(x0, y0 + step, PANE_Z),
                    ],
                    c,
                );
            }
        }

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

        if BEATS[self.beat].vision {
            let s = square_pos();
            let la = self.look_angle();
            let mut wedge = vec![Vec3::new(s.x, s.y, ON_PANE)];
            for k in 0..=14 {
                let a = la - VISION_HALF_ANGLE + 2.0 * VISION_HALF_ANGLE * (k as f32 / 14.0);
                let e = s + Vec2::from_angle(a) * 6.0;
                wedge.push(Vec3::new(e.x, e.y, ON_PANE));
            }
            // Fan-fill the cone manually (it is already centered at s).
            let base = tris.vertices.len() as u32;
            for w in &wedge {
                tris.vertices.push(w.to_array());
                tris.colors.push(COLOR_CONE);
            }
            for k in 1..wedge.len() as u32 - 1 {
                tris.indices.push([base, base + k, base + k + 1]);
            }
            for sign in [-1.0_f32, 1.0] {
                let e = s + Vec2::from_angle(la + sign * VISION_HALF_ANGLE) * 6.0;
                push(&mut lines, v3(s.x, s.y), v3(e.x, e.y), COLOR_CONE_EDGE, 1.5);
            }
        }

        let center = Vec3::new(SPHERE_XY.0, SPHERE_XY.1, sphere_z);
        let world: Vec<Vec3> = self.sphere_verts.iter().map(|v| *v + center).collect();
        if sphere_z < SPHERE_TOP + 1.0 {
            for e in &self.sphere_edges {
                push(
                    &mut lines,
                    world[e[0] as usize],
                    world[e[1] as usize],
                    COLOR_SPHERE,
                    1.4,
                );
            }
        }

        let poly = convex_section_polygon(&world, &self.sphere_edges, ZPlane::new(0.0));
        let sides = poly.len();
        if sides >= 3 {
            let lifted: Vec<Vec2> = poly.clone();
            let base = tris.vertices.len() as u32;
            for q in &lifted {
                tris.vertices.push([q.x, q.y, ON_PANE]);
                tris.colors.push(COLOR_SECTION_FILL);
            }
            for k in 1..sides as u32 - 1 {
                tris.indices.push([base, base + k, base + k + 1]);
            }
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
            DepthMode::ReadOnly {
                format: DEPTH_FORMAT,
            },
            ctx.rd.sample_count(),
        );
        let tri_node = TriangleRasterNode::new(
            &ctx.rd.device,
            ctx.rd.target_format(),
            DepthMode::ReadWrite {
                format: DEPTH_FORMAT,
            },
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
            depth: None,
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
            gaze_target: sphere_xy(),
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
        if beat.vision {
            self.gaze_target = s + Vec2::from_angle((self.time * 0.7).sin() * 1.1 + 0.4) * 2.0;
        } else if matches!(beat.cue, Cue::Hover | Cue::Descend) {
            self.gaze_target = sphere_xy();
        }

        // Surprise when A Sphere is present (arrives or crosses), not only mid-cut.
        let reacting = matches!(beat.cue, Cue::Hover | Cue::Descend)
            && self.sphere_z.value() < SPHERE_TOP + 0.5;
        if reacting != self.surprised {
            self.surprised = reacting;
            self.surprise
                .animate_to(if reacting { 1.0 } else { 0.0 }, 0.35, ease_out_cubic);
        }
        self.surprise.advance(dt);

        let look = self.look_angle();
        let mut bands = Vec::new();
        let z = self.sphere_z.value();
        if z.abs() < SPHERE_RADIUS {
            let rc = (SPHERE_RADIUS * SPHERE_RADIUS - z * z).sqrt();
            if let Some(b) = angular_band(s, look, sphere_xy(), rc) {
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

        let iv = self.intro.value() as f64;
        let dark = 0.05;
        let clear = wgpu::Color {
            r: dark + (SKY.r - dark) * iv,
            g: dark + (SKY.g - dark) * iv,
            b: (dark + 0.04) + (SKY.b - dark - 0.04) * iv,
            a: 1.0,
        };

        DepthBuffer::ensure(
            &mut self.depth,
            &rd.device,
            DEPTH_FORMAT,
            (w, h),
            rd.sample_count(),
        );
        let dview = self.depth.as_ref().expect("ensured").view.clone();

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
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &dview,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
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
        self.tris.execute(rd, view, Some(&dview), None)?;
        self.lines
            .set_camera(&rd.queue, view_proj, Vec2::new(w as f32, h as f32));
        self.lines.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &frame.lines,
            &Projection::Identity,
            1,
        );
        self.lines.execute(rd, view, Some(&dview), None)?;
        Ok(())
    }

    fn ui(&mut self, ctx: &egui::Context, _frame: &mut FrameCtx<'_>) {
        let beat = &BEATS[self.beat];

        let free_look = self.reveal.value() < 0.4 && !beat.vision && beat.cue == Cue::Hidden;
        if free_look {
            if let Some(p) = ctx.pointer_latest_pos() {
                let screen = ctx.content_rect();
                let dist = lerp(CAM_DIST.0, CAM_DIST.1, self.reveal.value());
                let fov = lerp(CAM_FOV_DEG.0, CAM_FOV_DEG.1, self.reveal.value()).to_radians();
                let ext_y = dist * (fov * 0.5).tan();
                let ext_x = ext_y * screen.width() / screen.height();
                let t = cam_target();
                let wx = t.x + (((p.x - screen.left()) / screen.width()) * 2.0 - 1.0) * ext_x;
                let wy = t.y + (1.0 - ((p.y - screen.top()) / screen.height()) * 2.0) * ext_y;
                self.gaze_target = Vec2::new(wx, wy);
            }
        }

        self.vision_window(ctx);
        self.dialogue(ctx);

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
                        .color(egui::Color32::from_gray(70)),
                );
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(280.0, 24.0), egui::Sense::hover());
                let painter = ui.painter();
                painter.rect_filled(rect, 4.0, egui::Color32::from_gray(60));
                let x = |n: f32| rect.left() + (n + 1.0) * 0.5 * rect.width();
                for (lo, hi) in &self.vision_bands {
                    let band = egui::Rect::from_min_max(
                        egui::pos2(x(*lo), rect.top()),
                        egui::pos2(x(*hi), rect.bottom()),
                    );
                    painter.rect_filled(band, 3.0, egui::Color32::from_rgb(235, 140, 40));
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
                egui::Color32::from_rgb(30, 110, 120),
            );
        }
        if let Some(t) = beat.sphere {
            if self.reveal.value() > 0.6 {
                speech_bubble(
                    ctx,
                    "sphere-bubble",
                    egui::pos2(screen.left() + screen.width() * 0.7, top),
                    t,
                    egui::Color32::from_rgb(170, 80, 55),
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
