//! Flatland, an explorable story of the dimensional ladder. One 3D scene: the
//! plane z=0 is Flatland, an upright pane standing in a larger 3D Spaceland
//! (checkerboard floor, gradient sky, distant tumbling solids). A single camera
//! orbits from a head-on view (reads as pure 2D) around to reveal the third
//! dimension while Flatland stays upright. A Sphere arrives along the hidden
//! depth axis; A Square, a 2D being, perceives it only as a circle.
//!
//! The narrative is a pure function of the timeline position `t` (Playhead +
//! Tracks): play advances `t`, but every visual is `f(t)`, so scrub, replay, and
//! frame-exact capture all reproduce identical frames.

use anyhow::Result;
use glam::{Mat3, Vec2, Vec3};
use rye_anim::{ease_in_out_cubic, ease_out_cubic, linear, Playhead, Track};
use rye_app::{egui, run_scene, FrameCtx, RunConfig, Scene, SetupCtx};
use rye_camera::OrbitPose;
use rye_math::{EuclideanR3, Projection, ZPlane};
use rye_render::device::RenderDevice;
use rye_render::{
    DepthBuffer, DepthMode, FragmentShading, LineRasterNode, ShaderEffect, TriangleRasterNode,
    Viewport,
};
use rye_shape::{convex_section_polygon, icosphere, LineMesh, Solid3, TriangleMesh};
use std::collections::BTreeSet;
use std::f32::consts::TAU;
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
const INTRO_SECONDS: f32 = 1.8;

const FLATLAND_HALF: f32 = 2.8;
const FLATLAND_CELLS: usize = 10;
const FLOOR_Y: f32 = -FLATLAND_HALF;
const PANE_Z: f32 = -0.02;
const VISION_HALF_ANGLE: f32 = 0.5;

const SKY_TOP: [f32; 3] = [0.42, 0.58, 0.82];
const SKY_HORIZON: [f32; 3] = [0.82, 0.88, 0.93];
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

const SKY_WGSL: &str = r#"
struct Sky { top: vec4<f32>, horizon: vec4<f32>, fade: vec4<f32> };
@group(0) @binding(0) var<uniform> sky: Sky;
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let g = clamp(in.uv.y, 0.0, 1.0);
    let col = mix(sky.top.rgb, sky.horizon.rgb, g);
    let dark = vec3<f32>(0.05, 0.05, 0.09);
    return vec4<f32>(mix(dark, col, sky.fade.x), 1.0);
}
"#;

const WASH_WGSL: &str = r#"
struct Wash { p: vec4<f32> };
@group(0) @binding(0) var<uniform> w: Wash;
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let d = in.uv.x - w.p.x;
    let band = exp(-d * d / 0.01);
    return vec4<f32>(1.0, 1.0, 0.93, band * w.p.y);
}
"#;

const VISION_WGSL: &str = r#"
struct Vision { meta: vec4<f32>, bands: array<vec4<f32>, 6> };
@group(0) @binding(0) var<uniform> v: Vision;
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var col = vec3<f32>(0.12, 0.13, 0.16);
    let count = i32(v.meta.x);
    for (var i = 0; i < count; i = i + 1) {
        if (in.uv.x >= v.bands[i].x && in.uv.x <= v.bands[i].y) {
            col = vec3<f32>(0.95, 0.58, 0.16);
        }
    }
    let scan = fract(v.meta.y * 0.25);
    let g = exp(-pow((in.uv.x - scan) * 26.0, 2.0));
    col = col + vec3<f32>(0.16, 0.16, 0.20) * g;
    return vec4<f32>(col, 0.92);
}
"#;

const WASH_BEAT: usize = 1;

fn f32_le(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

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
    Beat { caption: "Welcome to Flatland.", secs: 3.0, reveal: 0.0, sleeping: true, vision: false, cue: Cue::Hidden, square: None, sphere: None },
    Beat { caption: "A flat, two-dimensional world.", secs: 3.4, reveal: 0.0, sleeping: true, vision: false, cue: Cue::Hidden, square: None, sphere: None },
    Beat { caption: "This is A Square, one of its residents.", secs: 3.4, reveal: 0.0, sleeping: true, vision: false, cue: Cue::Hidden, square: None, sphere: None },
    Beat { caption: "He wakes. He has never once seen his world from outside it.", secs: 4.2, reveal: 0.0, sleeping: false, vision: false, cue: Cue::Hidden, square: Some("Another ordinary day in Flatland."), sphere: None },
    Beat { caption: "Yet Flatland is only a slice of a larger 3D world.", secs: 6.0, reveal: 1.0, sleeping: false, vision: false, cue: Cue::Hidden, square: None, sphere: None },
    Beat { caption: "In 3D, our vision is a 2D projection of that world.", secs: 4.5, reveal: 1.0, sleeping: false, vision: false, cue: Cue::Hidden, square: None, sphere: None },
    Beat { caption: "In 2D, A Square's vision is just a 1D projection of his.", secs: 7.0, reveal: 0.0, sleeping: false, vision: true, cue: Cue::Hidden, square: None, sphere: None },
    Beat { caption: "Then a visitor arrives, from the direction he cannot point to.", secs: 5.5, reveal: 1.0, sleeping: false, vision: false, cue: Cue::Hover, square: Some("I see only a circle. Reveal yourself!"), sphere: Some("Greetings, Square. I am A Sphere, from beyond your plane.") },
    Beat { caption: "A Sphere descends through Flatland.", secs: PASSAGE_SECONDS + 2.5, reveal: 1.0, sleeping: false, vision: false, cue: Cue::Descend, square: Some("It grows... it shrinks... it is gone! Sorcery!"), sphere: None },
    Beat { caption: "From outside his plane we see even his insides, as 4D would see ours. You are A Square.", secs: 9.0, reveal: 1.0, sleeping: false, vision: false, cue: Cue::Below, square: None, sphere: None },
];

fn primitives() -> [(Vec2, usize, f32, f32); 3] {
    [
        (Vec2::new(-1.7, 1.3), 3, 0.34, 0.5),
        (Vec2::new(-1.9, -1.2), 5, 0.30, -0.4),
        (Vec2::new(2.1, -1.6), 4, 0.26, 0.7),
    ]
}

fn distants() -> [(Vec3, Solid3, f32, f32); 4] {
    [
        (Vec3::new(5.5, 1.0, -8.0), Solid3::Cube, 1.0, 0.5),
        (Vec3::new(-6.5, 2.6, -11.0), Solid3::Icosahedron, 1.3, -0.35),
        (Vec3::new(4.0, -0.6, -14.0), Solid3::Tetrahedron, 1.5, 0.4),
        (Vec3::new(-3.5, 3.4, -17.0), Solid3::Cube, 1.7, -0.25),
    ]
}

fn cam_target() -> Vec3 {
    Vec3::new(0.6, 0.3, 0.0)
}

fn flat_pose() -> OrbitPose {
    OrbitPose {
        target: cam_target(),
        azimuth: 0.0,
        elevation: 0.0,
        distance: 24.0,
        fov_y: 15.0_f32.to_radians(),
    }
}

fn space_pose() -> OrbitPose {
    OrbitPose {
        target: cam_target(),
        azimuth: 34.0_f32.to_radians(),
        elevation: 16.0_f32.to_radians(),
        distance: 11.0,
        fov_y: 50.0_f32.to_radians(),
    }
}

fn square_pos() -> Vec2 {
    Vec2::new(0.0, 0.0)
}

fn sphere_xy() -> Vec2 {
    Vec2::new(SPHERE_XY.0, SPHERE_XY.1)
}

fn wake_beat() -> usize {
    BEATS.iter().position(|b| !b.sleeping).unwrap()
}

/// Build the timeline tracks (everything is `f(t)`): camera reveal, A Sphere's
/// height, and the intro fade, plus each beat's start time and the total length.
fn build_tracks() -> (Track, Track, Track, Vec<f32>, f32) {
    let mut starts = Vec::with_capacity(BEATS.len());
    let mut t = 0.0;
    for b in BEATS {
        starts.push(t);
        t += b.secs;
    }
    let total = t;

    let mut reveal = Track::new().key(0.0, BEATS[0].reveal, linear);
    let mut cur = BEATS[0].reveal;
    for (i, b) in BEATS.iter().enumerate() {
        if b.reveal != cur {
            let s = starts[i];
            reveal = reveal.key(s, cur, linear).key(
                s + REVEAL_SECONDS.min(b.secs),
                b.reveal,
                ease_in_out_cubic,
            );
            cur = b.reveal;
        }
    }

    let mut sz = Track::new().key(0.0, SPHERE_HIDDEN, linear);
    let mut zcur = SPHERE_HIDDEN;
    for (i, b) in BEATS.iter().enumerate() {
        let s = starts[i];
        match b.cue {
            Cue::Hidden => {
                if zcur != SPHERE_HIDDEN {
                    sz = sz.key(s, SPHERE_HIDDEN, linear);
                    zcur = SPHERE_HIDDEN;
                }
            }
            Cue::Hover => {
                sz = sz
                    .key(s, zcur, linear)
                    .key(s + 0.9, SPHERE_TOP, ease_out_cubic);
                zcur = SPHERE_TOP;
            }
            Cue::Descend => {
                sz = sz.key(s, zcur, linear).key(
                    s + PASSAGE_SECONDS.min(b.secs),
                    -SPHERE_TOP,
                    ease_in_out_cubic,
                );
                zcur = -SPHERE_TOP;
            }
            Cue::Below => {
                if zcur != -SPHERE_TOP {
                    sz = sz.key(s, -SPHERE_TOP, linear);
                    zcur = -SPHERE_TOP;
                }
            }
        }
    }

    let intro = Track::new()
        .key(0.0, 0.0, linear)
        .key(INTRO_SECONDS, 1.0, ease_out_cubic);

    (reveal, sz, intro, starts, total)
}

struct FlatlandApp {
    lines: LineRasterNode,
    tris: TriangleRasterNode,
    sky: ShaderEffect,
    wash: ShaderEffect,
    depth: Option<DepthBuffer>,
    vision_slide: f32,
    sphere_verts: Vec<Vec3>,
    sphere_edges: Vec<[u32; 2]>,
    playhead: Playhead,
    reveal: Track,
    sphere_z: Track,
    intro: Track,
    beat_starts: Vec<f32>,
    gaze_target: Vec2,
    vision_bands: Vec<(f32, f32)>,
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

fn fan3(tris: &mut TriangleMesh<3>, pts: &[Vec3], color: [f32; 4]) {
    if pts.len() < 3 {
        return;
    }
    let base = tris.vertices.len() as u32;
    for v in pts {
        tris.vertices.push(v.to_array());
        tris.colors.push(color);
    }
    for k in 1..pts.len() as u32 - 1 {
        tris.indices.push([base, base + k, base + k + 1]);
    }
}

fn v3(x: f32, y: f32) -> Vec3 {
    Vec3::new(x, y, 0.0)
}

/// A filled disc in the plane parallel to xy at `center.z` (a cheap billboard for
/// A Sphere's eyes, which face the head-on camera).
fn disc3(tris: &mut TriangleMesh<3>, center: Vec3, radius: f32, color: [f32; 4], n: usize) {
    let pts: Vec<Vec3> = (0..n)
        .map(|i| {
            let a = TAU * i as f32 / n as f32;
            center + Vec3::new(a.cos() * radius, a.sin() * radius, 0.0)
        })
        .collect();
    fan3(tris, &pts, color);
}

fn angular_band(s: Vec2, look_ang: f32, center: Vec2, radius: f32) -> Option<(f32, f32)> {
    let to_c = center - s;
    let dist = to_c.length();
    if dist <= radius {
        return None;
    }
    let mut rel = to_c.y.atan2(to_c.x) - look_ang;
    while rel > std::f32::consts::PI {
        rel -= TAU;
    }
    while rel < -std::f32::consts::PI {
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
    fn beat_index(&self) -> usize {
        self.beat_starts
            .partition_point(|&s| s <= self.playhead.t)
            .saturating_sub(1)
            .min(BEATS.len() - 1)
    }

    fn beat_elapsed(&self) -> f32 {
        self.playhead.t - self.beat_starts[self.beat_index()]
    }

    fn seek_beat(&mut self, beat: usize) {
        let b = beat.min(BEATS.len() - 1);
        self.playhead.seek(self.beat_starts[b]);
    }

    fn project_to_screen(&self, world: Vec3, screen: egui::Rect) -> Option<egui::Pos2> {
        let clip = self.camera_view_proj(screen.width() / screen.height()) * world.extend(1.0);
        if clip.w <= 0.001 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        if ndc.x.abs() > 1.3 || ndc.y.abs() > 1.3 {
            return None;
        }
        Some(egui::pos2(
            screen.left() + (ndc.x * 0.5 + 0.5) * screen.width(),
            screen.top() + (1.0 - (ndc.y * 0.5 + 0.5)) * screen.height(),
        ))
    }

    fn camera_view_proj(&self, aspect: f32) -> glam::Mat4 {
        let rv = self.reveal.sample(self.playhead.t);
        let intro = self.intro.sample(self.playhead.t);
        let pose = OrbitPose::lerp(&flat_pose(), &space_pose(), rv);
        let pose = pose.with_distance(pose.distance + (1.0 - intro) * 8.0);
        pose.view_proj(aspect, 0.1, 120.0)
    }

    fn face(&self) -> character::Face {
        let t = self.playhead.t;
        let beat = self.beat_index();
        let awake = !BEATS[beat].sleeping;
        let mut pos = square_pos();
        let look = self.gaze_target - pos;

        let z = self.sphere_z.sample(t);
        let surprise = (1.0 - ((z.abs() - SPHERE_RADIUS).max(0.0) / 1.3)).clamp(0.0, 1.0);

        let wake_t = if beat >= wake_beat() {
            (t - self.beat_starts[wake_beat()]).max(0.0)
        } else {
            -1.0
        };
        let wake_open = if wake_t < 0.0 {
            0.0
        } else {
            (wake_t / 0.6).clamp(0.0, 1.0)
        };
        // Head-shake shimmy on wake (a horizontal wobble; no shear since he floats).
        if (0.0..0.6).contains(&wake_t) {
            pos.x += (t * 30.0).sin() * 0.05 * (1.0 - wake_t / 0.6);
        }

        // Squash/stretch: breathing + a vertical reach toward what he watches +
        // a surprise pop, with a complementary horizontal squash for volume.
        let breathe = if awake {
            (t * 1.3).sin() * 0.015
        } else {
            0.04 + (t * 0.9).sin() * 0.05
        };
        let reach = if awake {
            (look.y * 0.04).clamp(-0.10, 0.18)
        } else {
            0.0
        };
        let pop = 0.14 * surprise;
        let sy = 1.0 + breathe + reach + pop;
        let sx = 1.0 - reach * 0.4 + pop * 0.5;

        let blink_phase = t % 3.4;
        let blink = if awake && blink_phase < 0.16 {
            (blink_phase / 0.08 - 1.0).abs().min(1.0)
        } else {
            1.0
        };
        let base = if awake { 0.5 + 0.5 * surprise } else { 0.0 };
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

        character::Face {
            pos,
            half: 0.34,
            scale: Vec2::new(sx, sy),
            eye_open: [
                ((base + left_more) * blink * wake_open).min(1.0),
                ((base + right_more) * blink * wake_open).min(1.0),
            ],
            look,
        }
    }

    fn build(&self) -> (LineMesh<3>, TriangleMesh<3>, usize) {
        let t = self.playhead.t;
        let beat = self.beat_index();
        let sphere_z = self.sphere_z.sample(t);
        let mut lines = LineMesh::<3>::default();
        let mut tris = TriangleMesh::<3>::default();

        // Spaceland floor, receding into the distance.
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

        for (center, solid, scale, spin) in distants() {
            let rot = Mat3::from_rotation_y(t * spin) * Mat3::from_rotation_x(t * spin * 0.5);
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

        // Flatland pane checkerboard (just behind the on-plane content).
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
            let phase = t * spin;
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

        if BEATS[beat].vision {
            let s = square_pos();
            let la = self.look_angle();
            let mut wedge = vec![v3(s.x, s.y)];
            for k in 0..=14 {
                let a = la - VISION_HALF_ANGLE + 2.0 * VISION_HALF_ANGLE * (k as f32 / 14.0);
                let e = s + Vec2::from_angle(a) * 6.0;
                wedge.push(v3(e.x, e.y));
            }
            fan3(&mut tris, &wedge, COLOR_CONE);
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
            // A Sphere is a character too: white eyes on its front face.
            let ez = center.z + SPHERE_RADIUS * 0.85;
            for sx in [-1.0_f32, 1.0] {
                let ec = Vec3::new(
                    center.x + sx * 0.42 * SPHERE_RADIUS,
                    center.y + 0.18 * SPHERE_RADIUS,
                    ez,
                );
                disc3(
                    &mut tris,
                    ec,
                    0.22 * SPHERE_RADIUS,
                    [0.98, 0.99, 0.97, 1.0],
                    14,
                );
                disc3(
                    &mut tris,
                    ec + Vec3::new(-0.06 * SPHERE_RADIUS, -0.06 * SPHERE_RADIUS, 0.01),
                    0.11 * SPHERE_RADIUS,
                    [0.10, 0.16, 0.20, 1.0],
                    10,
                );
            }
        }

        let poly = convex_section_polygon(&world, &self.sphere_edges, ZPlane::new(0.0));
        let sides = poly.len();
        if sides >= 3 {
            let disc: Vec<Vec3> = poly.iter().map(|q| v3(q.x, q.y)).collect();
            fan3(&mut tris, &disc, COLOR_SECTION_FILL);
            for i in 0..sides {
                let a = poly[i];
                let b = poly[(i + 1) % sides];
                push(&mut lines, v3(a.x, a.y), v3(b.x, b.y), COLOR_SECTION, 3.0);
            }
        }

        character::push_face(&mut lines, &mut tris, &self.face());

        (lines, tris, sides)
    }

    fn look_angle(&self) -> f32 {
        let d = self.gaze_target - square_pos();
        if d.length() > 1e-4 {
            d.y.atan2(d.x)
        } else {
            0.0
        }
    }
}

impl Scene for FlatlandApp {
    fn new(ctx: &mut SetupCtx<'_>) -> Result<Self> {
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
        let sky = ShaderEffect::new(
            &ctx.rd.device,
            ctx.rd.target_format(),
            SKY_WGSL,
            48,
            wgpu::BlendState::REPLACE,
            ctx.rd.sample_count(),
        );
        let wash = ShaderEffect::new(
            &ctx.rd.device,
            ctx.rd.target_format(),
            WASH_WGSL,
            16,
            wgpu::BlendState::ALPHA_BLENDING,
            ctx.rd.sample_count(),
        );

        let (raw, faces) = icosphere(SPHERE_SUBDIVISIONS);
        let sphere_verts = raw.iter().map(|v| *v * SPHERE_RADIUS).collect();
        let mut set = BTreeSet::new();
        for f in &faces {
            for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                set.insert(if a < b { [a, b] } else { [b, a] });
            }
        }

        let (reveal, sphere_z, intro, beat_starts, total) = build_tracks();

        Ok(Self {
            lines: line_node,
            tris: tri_node,
            sky,
            wash,
            depth: None,
            vision_slide: 0.0,
            sphere_verts,
            sphere_edges: set.into_iter().collect(),
            playhead: Playhead::new(total),
            reveal,
            sphere_z,
            intro,
            beat_starts,
            gaze_target: sphere_xy(),
            vision_bands: Vec::new(),
        })
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        self.playhead.advance(ctx.dt.min(0.1));
        let t = self.playhead.t;
        let beat = &BEATS[self.beat_index()];
        let s = square_pos();

        if beat.vision {
            self.gaze_target = s + Vec2::from_angle((t * 0.7).sin() * 1.1 + 0.4) * 2.0;
        } else if matches!(beat.cue, Cue::Hover | Cue::Descend) {
            self.gaze_target = sphere_xy();
        }

        let look = self.look_angle();
        let mut bands = Vec::new();
        let z = self.sphere_z.sample(t);
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

        let active = beat.vision || z.abs() < SPHERE_RADIUS;
        let target = if active { 1.0 } else { 0.0 };
        let dt = ctx.dt.min(0.1);
        self.vision_slide += (target - self.vision_slide) * (1.0 - (-dt * 8.0).exp());
    }

    fn render(
        &mut self,
        rd: &RenderDevice,
        view: &wgpu::TextureView,
        viewport: Viewport,
    ) -> Result<()> {
        let (line_mesh, tri_mesh, _sides) = self.build();

        let cfg = &rd.surface_bundle.config;
        let vw = viewport.width.max(1);
        let vh = viewport.height.max(1);
        let view_proj = self.camera_view_proj(vw as f32 / vh as f32);

        DepthBuffer::ensure(
            &mut self.depth,
            &rd.device,
            DEPTH_FORMAT,
            (cfg.width, cfg.height.max(1)),
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
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
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

        let intro = self.intro.sample(self.playhead.t);
        self.sky.set_uniforms(
            &rd.queue,
            &f32_le(&[
                SKY_TOP[0],
                SKY_TOP[1],
                SKY_TOP[2],
                1.0, //
                SKY_HORIZON[0],
                SKY_HORIZON[1],
                SKY_HORIZON[2],
                1.0, //
                intro,
                0.0,
                0.0,
                0.0,
            ]),
        );
        self.sky.execute(rd, view, Some(&viewport))?;

        self.tris.set_camera(&rd.queue, view_proj);
        self.tris
            .upload::<EuclideanR3, 3>(&rd.device, &rd.queue, &tri_mesh, &Projection::Identity);
        self.tris.execute(rd, view, Some(&dview), Some(&viewport))?;
        self.lines
            .set_camera(&rd.queue, view_proj, Vec2::new(vw as f32, vh as f32));
        self.lines.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &line_mesh,
            &Projection::Identity,
            1,
        );
        self.lines
            .execute(rd, view, Some(&dview), Some(&viewport))?;

        // The "two-dimensional world" beat gets a light wash sweeping across.
        if self.beat_index() == WASH_BEAT {
            let p = (self.beat_elapsed() / BEATS[WASH_BEAT].secs).clamp(0.0, 1.0);
            let strength = (p * std::f32::consts::PI).sin() * 0.5;
            self.wash
                .set_uniforms(&rd.queue, &f32_le(&[p, strength, 0.0, 0.0]));
            self.wash.execute(rd, view, Some(&viewport))?;
        }
        Ok(())
    }

    fn ui(&mut self, ctx: &egui::Context, _frame: &mut FrameCtx<'_>) {
        let beat = self.beat_index();
        let b = &BEATS[beat];

        if self.reveal.sample(self.playhead.t) < 0.4 && !b.vision && b.cue == Cue::Hidden {
            if let Some(p) = ctx.pointer_latest_pos() {
                let screen = ctx.content_rect();
                let pose = OrbitPose::lerp(
                    &flat_pose(),
                    &space_pose(),
                    self.reveal.sample(self.playhead.t),
                );
                let ext_y = pose.distance * (pose.fov_y * 0.5).tan();
                let ext_x = ext_y * screen.width() / screen.height();
                let tgt = cam_target();
                let wx = tgt.x + (((p.x - screen.left()) / screen.width()) * 2.0 - 1.0) * ext_x;
                let wy = tgt.y + (1.0 - ((p.y - screen.top()) / screen.height()) * 2.0) * ext_y;
                self.gaze_target = Vec2::new(wx, wy);
            }
        }

        self.vision_window(ctx);
        self.dialogue(ctx, beat);

        let fade = {
            let e = self.beat_elapsed();
            (e / 0.5)
                .clamp(0.0, 1.0)
                .min(((b.secs - e) / 0.7).clamp(0.0, 1.0))
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
                                egui::RichText::new(b.caption)
                                    .size(18.0)
                                    .color(egui::Color32::from_rgba_unmultiplied(30, 34, 40, a)),
                            );
                        });
                    });
            });

        self.controls(ctx, beat);
    }

    fn title(&self) -> std::borrow::Cow<'static, str> {
        "flatland".into()
    }
}

impl FlatlandApp {
    fn vision_window(&self, ctx: &egui::Context) {
        // Slides in from the left while his vision is the subject, out otherwise.
        let x = -300.0 + 318.0 * self.vision_slide;
        if self.vision_slide < 0.01 {
            return;
        }
        let mut u = vec![
            self.vision_bands.len().min(6) as f32,
            self.playhead.t,
            0.0,
            0.0,
        ];
        for i in 0..6 {
            if let Some((lo, hi)) = self.vision_bands.get(i) {
                u.extend([(lo + 1.0) * 0.5, (hi + 1.0) * 0.5, 0.0, 0.0]);
            } else {
                u.extend([0.0; 4]);
            }
        }
        egui::Area::new(egui::Id::new("vision-strip"))
            .anchor(egui::Align2::LEFT_TOP, [x, 18.0])
            .show(ctx, |ui| {
                ui.label(
                    egui::RichText::new("A Square's vision (1D)")
                        .size(12.0)
                        .color(egui::Color32::from_gray(70)),
                );
                rye_egui::shader_widget(ui, egui::vec2(280.0, 26.0), VISION_WGSL, 112, f32_le(&u));
            });
    }

    fn dialogue(&self, ctx: &egui::Context, beat: usize) {
        let b = &BEATS[beat];
        let screen = ctx.content_rect();

        if let Some(t) = b.square {
            if let Some(p) = self.project_to_screen(Vec3::new(0.0, 0.4, 0.0), screen) {
                speech_bubble(
                    ctx,
                    "square-bubble",
                    p - egui::vec2(0.0, 18.0),
                    t,
                    egui::Color32::from_rgb(30, 110, 120),
                );
            }
        }

        let z = self.sphere_z.sample(self.playhead.t);
        if self.reveal.sample(self.playhead.t) > 0.6 && z < SPHERE_TOP + 1.0 {
            let head = Vec3::new(SPHERE_XY.0, SPHERE_XY.1 + SPHERE_RADIUS, z);
            if let Some(p) = self.project_to_screen(head, screen) {
                if matches!(b.cue, Cue::Hover) {
                    callout(
                        ctx,
                        "sphere-callout",
                        p,
                        "A Sphere",
                        egui::Color32::from_rgb(170, 80, 55),
                    );
                }
                if let Some(t) = b.sphere {
                    speech_bubble(
                        ctx,
                        "sphere-bubble",
                        p - egui::vec2(0.0, 18.0),
                        t,
                        egui::Color32::from_rgb(170, 80, 55),
                    );
                }
            }
        }
    }

    fn controls(&mut self, ctx: &egui::Context, beat: usize) {
        let mut seek_to_beat: Option<usize> = None;
        let mut seek_t: Option<f32> = None;
        let mut set_playing: Option<bool> = None;
        let total = self.playhead.duration;
        let now = self.playhead.t;
        let playing = self.playhead.playing;

        egui::Area::new(egui::Id::new("controls"))
            .anchor(egui::Align2::CENTER_BOTTOM, [0.0, -14.0])
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    // Thin progress bar with a tick per beat; click/drag to scrub.
                    let (rect, resp) = ui.allocate_exact_size(
                        egui::vec2(520.0, 10.0),
                        egui::Sense::click_and_drag(),
                    );
                    let painter = ui.painter();
                    painter.rect_filled(rect, 4.0, egui::Color32::from_gray(60));
                    for &s in &self.beat_starts {
                        let x = rect.left() + (s / total) * rect.width();
                        painter.line_segment(
                            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                            egui::Stroke::new(1.0, egui::Color32::from_gray(120)),
                        );
                    }
                    let fx = rect.left() + (now / total).clamp(0.0, 1.0) * rect.width();
                    painter.rect_filled(
                        egui::Rect::from_min_max(rect.left_top(), egui::pos2(fx, rect.bottom())),
                        4.0,
                        egui::Color32::from_rgb(230, 150, 40),
                    );
                    painter.circle_filled(
                        egui::pos2(fx, rect.center().y),
                        5.0,
                        egui::Color32::from_rgb(250, 185, 75),
                    );
                    if let Some(p) = resp.interact_pointer_pos() {
                        let u = ((p.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                        seek_t = Some(u * total);
                        set_playing = Some(false);
                    }
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if ui.button(if playing { "Pause" } else { "Play" }).clicked() {
                            set_playing = Some(!playing);
                        }
                        if ui
                            .add_enabled(beat > 0, egui::Button::new("Back"))
                            .clicked()
                        {
                            seek_to_beat = Some(beat.saturating_sub(1));
                        }
                        if ui.button("Replay").clicked() {
                            seek_t = Some(0.0);
                            set_playing = Some(true);
                        }
                    });
                });
            });

        if let Some(p) = set_playing {
            self.playhead.playing = p;
        }
        if let Some(t) = seek_t {
            self.playhead.seek(t);
        }
        if let Some(bi) = seek_to_beat {
            self.seek_beat(bi);
        }
    }
}

fn callout(ctx: &egui::Context, id: &str, target: egui::Pos2, text: &str, accent: egui::Color32) {
    let anchor = target + egui::vec2(72.0, -64.0);
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new(id),
    ));
    painter.line_segment([anchor, target], egui::Stroke::new(2.0, accent));
    let dir = (target - anchor).normalized();
    let perp = egui::vec2(-dir.y, dir.x);
    let base = target - dir * 12.0;
    painter.add(egui::Shape::convex_polygon(
        vec![target, base + perp * 6.0, base - perp * 6.0],
        accent,
        egui::Stroke::NONE,
    ));
    egui::Area::new(egui::Id::new(format!("{id}-label")))
        .fixed_pos(anchor)
        .pivot(egui::Align2::CENTER_BOTTOM)
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            egui::Frame::popup(&ctx.style())
                .fill(egui::Color32::from_rgb(252, 252, 250))
                .stroke(egui::Stroke::new(1.5, accent))
                .show(ui, |ui| {
                    ui.label(egui::RichText::new(text).size(14.0).strong());
                });
        });
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
    run_scene::<FlatlandApp>(RunConfig {
        window: WindowAttributes::default()
            .with_title("flatland")
            .with_visible(false),
        ..RunConfig::default()
    })
}
