//! Flatland, an explorable story of the dimensional ladder. One 3D scene: the
//! ground plane y=0 IS Flatland, an infinite 2D cross-section of a surrounding 3D
//! Spaceland. At the start the camera looks straight down (orthographic), so only
//! the ground is visible and it reads as a pure 2D world. The reveal tilts the
//! camera up to expose the hidden vertical axis, along which A Sphere descends; A
//! Square, a 2D being, perceives it only as a growing-then-shrinking circle.
//!
//! The world genuinely lives in 3D: the ground, the spinning polygons, the
//! section, and A Sphere are all real 3D geometry under one camera. The section
//! circle is the closed form of sphere-meets-plane. The only flat things are A
//! Square's body and eyes, which is correct: a 2D being has no thickness.
//!
//! Sections are stepped manually (no auto-advance) so each beat can be tuned in
//! isolation; idle and interactive animation (breathing, sleep, wake, gaze) runs
//! off a free clock independent of which section is showing.

use anyhow::Result;
use glam::{Mat3, Mat4, Vec2, Vec3};
use rye_anim::{ease_in_out_cubic, ease_out_cubic, Animated};
use rye_app::{egui, run_scene, FrameCtx, RunConfig, Scene, SetupCtx};
use rye_camera::OrbitPose;
use rye_math::{EuclideanR3, Projection};
use rye_render::device::RenderDevice;
use rye_render::{
    DepthMode, FragmentShading, LineRasterNode, ShaderEffect, TriangleRasterNode, Viewport,
};
use rye_shape::{icosphere, LineMesh, Solid3, TriangleMesh};
use std::collections::BTreeSet;
use std::f32::consts::TAU;
use winit::window::WindowAttributes;

mod character;

const SPHERE_RADIUS: f32 = 1.3;
const SPHERE_SUBDIVISIONS: u32 = 3;
const SPHERE_TOP: f32 = 2.6;
const SPHERE_HIDDEN: f32 = 9.0;
const SPHERE_XY: (f32, f32) = (1.6, 0.7);
const VISION_HALF_ANGLE: f32 = 0.5;

const INTRO_SECONDS: f32 = 1.6;
const REVEAL_SECONDS: f32 = 1.4;
const PASSAGE_SECONDS: f32 = 4.0;

// Ground (infinite Flatland cross-section). A fixed large grid whose far cells
// fade into the horizon colour so no hard edge shows when the camera tilts.
const GROUND_HALF: f32 = 26.0;
const GROUND_STEP: f32 = 0.8;
const GROUND_FADE_NEAR: f32 = 10.0;
const GROUND_FADE_FAR: f32 = 24.0;

// Deep-slate palette with bright character/section accents.
const SKY_TOP: [f32; 3] = [0.05, 0.07, 0.12];
const SKY_HORIZON: [f32; 3] = [0.13, 0.17, 0.24];
const GROUND_A: [f32; 4] = [0.13, 0.16, 0.22, 1.0];
const GROUND_B: [f32; 4] = [0.10, 0.13, 0.18, 1.0];
const COLOR_SECTION: [f32; 4] = [1.0, 0.62, 0.22, 1.0];
const COLOR_SECTION_FILL: [f32; 4] = [0.98, 0.55, 0.18, 0.45];
const COLOR_CONE: [f32; 4] = [0.96, 0.82, 0.35, 0.14];
const COLOR_SPHERE: [f32; 4] = [0.96, 0.52, 0.36, 1.0];
const COLOR_DISTANT: [f32; 4] = [0.30, 0.36, 0.46, 1.0];

const SKY_WGSL: &str = r#"
struct Sky { top: vec4<f32>, horizon: vec4<f32> };
@group(0) @binding(0) var<uniform> sky: Sky;
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let g = clamp(in.uv.y, 0.0, 1.0);
    return vec4<f32>(mix(sky.top.rgb, sky.horizon.rgb, g), 1.0);
}
"#;

// params: x=band count, y=clock, z=cursor position (0..1, <0 = none).
// bands[i]: x=lo, y=hi (both 0..1), z=brightness (distance shading).
// The bands arrive ordered far -> near, so the last one covering a column wins
// (nearer shapes occlude farther ones).
const VISION_WGSL: &str = r#"
struct Vision { params: vec4<f32>, bands: array<vec4<f32>, 6>, cols: array<vec4<f32>, 6> };
@group(0) @binding(0) var<uniform> v: Vision;
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var col = vec3<f32>(0.07, 0.08, 0.11);
    let n = i32(v.params.x);
    for (var i = 0; i < n; i = i + 1) {
        if (in.uv.x >= v.bands[i].x && in.uv.x <= v.bands[i].y) {
            // Shade across the band (bright center, darker edges) so each shape
            // reads as a rounded form, then dim by distance.
            let w = max(v.bands[i].y - v.bands[i].x, 1e-4);
            let tt = (in.uv.x - v.bands[i].x) / w * 2.0 - 1.0;
            let shade = 1.0 - 0.5 * tt * tt;
            col = v.cols[i].rgb * v.bands[i].z * shade;
        }
    }
    let cu = v.params.z;
    if (cu >= 0.0) {
        let m = smoothstep(0.014, 0.0, abs(in.uv.x - cu));
        col = mix(col, vec3<f32>(1.0, 1.0, 1.0), m * 0.85);
    }
    return vec4<f32>(col, 0.92);
}
"#;

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
    reveal: f32,
    sleeping: bool,
    vision: bool,
    cue: Cue,
}

const BEATS: &[Beat] = &[
    Beat {
        caption: "Welcome to Flatland, a flat two-dimensional world.",
        reveal: 0.0,
        sleeping: true,
        vision: false,
        cue: Cue::Hidden,
    },
    Beat {
        caption: "This is A Square, one of its residents.",
        reveal: 0.0,
        sleeping: true,
        vision: false,
        cue: Cue::Hidden,
    },
    Beat {
        caption: "He has never once seen his world from outside it.",
        reveal: 0.0,
        sleeping: false,
        vision: false,
        cue: Cue::Hidden,
    },
    Beat {
        caption: "Yet Flatland is only a slice of a larger 3D world.",
        reveal: 1.0,
        sleeping: false,
        vision: false,
        cue: Cue::Hidden,
    },
    Beat {
        caption: "In 3D, our vision is a 2D projection of that world.",
        reveal: 1.0,
        sleeping: false,
        vision: false,
        cue: Cue::Hidden,
    },
    Beat {
        caption: "In 2D, A Square's vision is just a 1D projection of his.",
        reveal: 0.0,
        sleeping: false,
        vision: true,
        cue: Cue::Hidden,
    },
    Beat {
        caption: "Then a visitor arrives, from the direction he cannot point to.",
        reveal: 1.0,
        sleeping: false,
        vision: false,
        cue: Cue::Hover,
    },
    Beat {
        caption: "A Sphere descends through Flatland.",
        reveal: 1.0,
        sleeping: false,
        vision: false,
        cue: Cue::Descend,
    },
    Beat {
        caption: "From outside his plane we see even his insides, as 4D would see ours.",
        reveal: 1.0,
        sleeping: false,
        vision: false,
        cue: Cue::Below,
    },
    Beat {
        caption: "Now picture us. In our 3D world, we are the Square.",
        reveal: 1.0,
        sleeping: false,
        vision: false,
        cue: Cue::Hidden,
    },
    Beat {
        caption: "A 4D being sees us whole, and all our insides at once. You are A Square.",
        reveal: 1.0,
        sleeping: false,
        vision: false,
        cue: Cue::Below,
    },
];

/// Spinning polygons that will become characters: red triangle, yellow pentagon,
/// smaller green square. (center, sides, radius, spin, color). The pentagon and
/// square are placed nearly collinear from A Square (origin) so the near pentagon
/// occludes the far square in his 1D vision.
fn primitives() -> [(Vec2, usize, f32, f32, [f32; 4]); 5] {
    [
        (Vec2::new(-1.3, 1.5), 3, 0.30, 0.6, [0.90, 0.32, 0.30, 1.0]),
        (Vec2::new(1.0, 1.4), 5, 0.28, -0.5, [0.95, 0.80, 0.25, 1.0]),
        (Vec2::new(1.8, 2.5), 4, 0.24, 0.8, [0.40, 0.80, 0.42, 1.0]),
        (Vec2::new(-2.1, -0.2), 6, 0.26, 0.4, [0.58, 0.46, 0.86, 1.0]),
        (Vec2::new(0.3, 2.7), 3, 0.20, -0.9, [0.32, 0.72, 0.86, 1.0]),
    ]
}

/// The angular interval a rotating convex polygon's silhouette subtends in A
/// Square's 1D vision (so the band pulses as it spins), with its centre distance
/// for depth shading. Normalized to his half field of view; `None` if outside it.
fn silhouette_band(
    s: Vec2,
    look_ang: f32,
    center: Vec2,
    sides: usize,
    radius: f32,
    phase: f32,
) -> Option<(f32, f32, f32)> {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for k in 0..sides {
        let v = center + Vec2::from_angle(phase + TAU * k as f32 / sides as f32) * radius;
        let to = v - s;
        let mut rel = to.y.atan2(to.x) - look_ang;
        while rel > std::f32::consts::PI {
            rel -= TAU;
        }
        while rel < -std::f32::consts::PI {
            rel += TAU;
        }
        lo = lo.min(rel);
        hi = hi.max(rel);
    }
    let (lo, hi) = (lo / VISION_HALF_ANGLE, hi / VISION_HALF_ANGLE);
    if hi < -1.0 || lo > 1.0 {
        return None;
    }
    Some((
        lo.clamp(-1.0, 1.0),
        hi.clamp(-1.0, 1.0),
        (center - s).length(),
    ))
}

/// Ambient Spaceland solids, floating above the ground; faded in with the reveal.
/// (center, solid, scale, spin)
fn distants() -> [(Vec3, Solid3, f32, f32); 4] {
    [
        (Vec3::new(5.5, 3.0, -4.0), Solid3::Cube, 1.0, 0.5),
        (Vec3::new(-6.5, 4.2, -1.0), Solid3::Icosahedron, 1.3, -0.35),
        (Vec3::new(4.0, 2.4, 4.0), Solid3::Tetrahedron, 1.5, 0.4),
        (Vec3::new(-3.5, 5.0, 6.0), Solid3::Cube, 1.7, -0.25),
    ]
}

fn cam_target() -> Vec3 {
    Vec3::ZERO
}

fn square_pos() -> Vec2 {
    Vec2::ZERO
}

fn sphere_xy() -> Vec2 {
    Vec2::new(SPHERE_XY.0, SPHERE_XY.1)
}

/// Map a 2D Flatland point onto the 3D ground plane (y=0). The hidden third
/// dimension is +y; the 2D vertical axis maps to world -z so that, viewed
/// top-down with up=-z, it reads right-side up.
fn flat(p: Vec2) -> Vec3 {
    Vec3::new(p.x, 0.0, -p.y)
}

/// Half-height of the orthographic top-down view for a section: tight on A Square
/// while he is introduced, wider once Flatland and the vision are the subject.
fn section_zoom(section: usize) -> f32 {
    if BEATS[section].vision {
        3.8
    } else {
        match section {
            0 => 3.2,
            1 => 1.9,
            2 => 2.6,
            _ => 3.4,
        }
    }
}

/// The angled perspective pose the reveal settles into. Pulled back and rotated
/// off-axis so the third dimension reads clearly.
fn reveal_pose() -> OrbitPose {
    OrbitPose {
        target: cam_target(),
        azimuth: 26.0_f32.to_radians(),
        elevation: 44.0_f32.to_radians(),
        distance: 13.5,
        fov_y: 42.0_f32.to_radians(),
    }
}

/// A Sphere's height target (along +y) for a section's cue.
fn sphere_target(cue: Cue) -> f32 {
    match cue {
        Cue::Hidden => SPHERE_HIDDEN,
        Cue::Hover => SPHERE_TOP,
        Cue::Descend | Cue::Below => -SPHERE_TOP,
    }
}

/// The angular interval an off-plane disc subtends in A Square's 1D vision,
/// normalized to his half field of view; `None` if outside it.
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

struct FlatlandApp {
    lines: LineRasterNode,
    tris: TriangleRasterNode,
    sky: ShaderEffect,
    sphere_verts: Vec<Vec3>,
    sphere_edges: Vec<[u32; 2]>,
    scratch_lines: LineMesh<3>,
    scratch_tris: TriangleMesh<3>,
    clock: f32,
    section: usize,
    section_clock: f32,
    reveal: Animated,
    sphere_h: Animated,
    zoom: Animated,
    gaze: Vec2,
    gaze_target: Vec2,
    cursor_present: bool,
    woken: bool,
    wake_clock: f32,
    /// Clock time the cursor was last (re)found (absent -> present), for the
    /// happy reaction.
    found_clock: f32,
    /// Ordered far -> near: (lo, hi, brightness, rgb).
    vision_bands: Vec<(f32, f32, f32, [f32; 3])>,
    /// Cursor position in the 1D vision strip (0..1), or -1 when absent.
    cursor_u: f32,
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

impl FlatlandApp {
    fn goto(&mut self, section: usize) {
        self.section = section.min(BEATS.len() - 1);
        self.section_clock = self.clock;
        let b = &BEATS[self.section];
        self.reveal
            .animate_to(b.reveal, REVEAL_SECONDS, ease_in_out_cubic);
        let dur = if b.cue == Cue::Descend {
            PASSAGE_SECONDS
        } else {
            1.2
        };
        self.sphere_h
            .animate_to(sphere_target(b.cue), dur, ease_in_out_cubic);
        self.zoom.animate_to(
            section_zoom(self.section),
            REVEAL_SECONDS,
            ease_in_out_cubic,
        );
        // Returning to the start re-arms the one-shot wake.
        if self.section == 0 {
            self.woken = false;
            self.wake_clock = -1.0;
        }
    }

    fn look_angle(&self) -> f32 {
        let d = self.gaze - square_pos();
        if d.length() > 1e-4 {
            d.y.atan2(d.x)
        } else {
            0.0
        }
    }

    fn intro(&self) -> f32 {
        (self.clock / INTRO_SECONDS).clamp(0.0, 1.0)
    }

    fn camera_view_proj(&self, aspect: f32) -> Mat4 {
        let rv = self.reveal.value();
        let intro = self.intro();
        // Top-down orthographic view (pure 2D) at rv=0; the up vector is -z so the
        // 2D vertical axis (mapped to world -z) reads upward. Blends to the angled
        // perspective pose as the third dimension is revealed.
        let half = self.zoom.value();
        let eye = cam_target() + Vec3::new(0.0, 30.0, 0.0);
        let ortho = Mat4::orthographic_rh(-half * aspect, half * aspect, -half, half, 0.1, 200.0)
            * Mat4::look_at_rh(eye, cam_target(), Vec3::new(0.0, 0.0, -1.0));
        let space = reveal_pose()
            .with_distance(reveal_pose().distance + (1.0 - intro) * 3.0)
            .view_proj(aspect, 0.1, 200.0);
        ortho * (1.0 - rv) + space * rv
    }

    fn face(&self) -> character::Face {
        let t = self.clock;
        let sleeping = BEATS[self.section].sleeping;
        let look = self.gaze - square_pos();

        let h = self.sphere_h.value();
        let surprise = if h.abs() < SPHERE_RADIUS {
            (1.0 - h.abs() / SPHERE_RADIUS).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // None = asleep; Some(wp) = seconds since wake began (>=1.9 fully awake).
        let waking = if sleeping {
            if self.wake_clock >= 0.0 {
                Some(t - self.wake_clock)
            } else {
                None
            }
        } else {
            Some(10.0)
        };

        let pi = std::f32::consts::PI;
        let tracking = self.cursor_present && self.reveal.value() < 0.5;
        let searching = !self.cursor_present && self.reveal.value() < 0.5;

        let mut pos = square_pos();
        let sx;
        let mut sy;
        let eye_open;
        let mut eye_squash = [1.0_f32; 2];
        let mut eye_scale = 1.0_f32;
        let mut lookv = Vec2::ZERO;

        match waking {
            None => {
                // Asleep: slow bob and breathe; eyes closed (rectangle).
                let breath = (t * 1.1).sin();
                pos.y += breath * 0.05;
                sy = 1.0 + breath * 0.03;
                sx = 1.0 - breath * 0.03;
                eye_open = [0.0, 0.0];
            }
            Some(wp) if wp < 2.3 => {
                // Wake (plays once): panic, frantically hunt for the cursor, then
                // find it and beam directly AT it.
                //   A startle (0..0.45): eyes snap wide + oversized, body jolt then
                //     recoil with a tremble, eyes darting.
                //   B search (0.45..1.35): squint, dart through directions hunting.
                //   C find + happy (1.35..2.3): gaze locks to the cursor, eyes go
                //     cheery then open, with a happy bounce, looking at it.
                let e;
                let escale;
                if wp < 0.10 {
                    let a = wp / 0.10;
                    e = a;
                    escale = 1.0 + 0.7 * a;
                    sy = 1.0 + 0.34 * a;
                    sx = 1.0 - 0.24 * a;
                    lookv = Vec2::from_angle(wp * 40.0);
                } else if wp < 0.45 {
                    let a = (wp - 0.10) / 0.35;
                    e = 1.0;
                    escale = 1.7 - 0.2 * a;
                    sy = 1.34 - 0.44 * a;
                    sx = 0.76 + 0.32 * a;
                    pos.x += (wp * 55.0).sin() * 0.05 * (1.0 - a);
                    lookv = Vec2::from_angle(wp * 40.0);
                } else if wp < 1.35 {
                    let a = (wp - 0.45) / 0.9;
                    e = 0.75;
                    escale = 1.5 - 0.3 * a;
                    eye_squash = [0.7, 0.7];
                    sy = 0.95 + 0.05 * a;
                    sx = 1.05 - 0.05 * a;
                    let dirs = [2.2_f32, 0.5, 1.7, 0.9, 1.3];
                    lookv = Vec2::from_angle(dirs[((wp - 0.45) / 0.18) as usize % dirs.len()]);
                    pos.x += (wp * 30.0).sin() * 0.01;
                } else {
                    let a = ((wp - 1.35) / 0.95).clamp(0.0, 1.0);
                    e = if a < 0.5 {
                        0.5
                    } else {
                        0.5 + 0.5 * ease_in_out_cubic((a - 0.5) / 0.5)
                    };
                    escale = 1.0 + 0.2 * (1.0 - a);
                    sy = 1.0 + (a * pi).sin() * 0.12 * (1.0 - a);
                    sx = 1.0;
                    lookv = look;
                }
                eye_open = [e, e];
                eye_scale = escale;
            }
            Some(_) => {
                // Awake: eyes follow where his gaze points (cursor or his search
                // scan). Breathing + a surprise pop/widen at the section circle.
                // When the cursor is gone he squints and hunts; when it reappears
                // he beams at it briefly before settling to engaged.
                let breathe = (t * 1.2).sin() * 0.02;
                let pop = 0.16 * surprise;
                sy = 1.0 + breathe + pop;
                sx = 1.0 - pop * 0.3;
                lookv = look;

                let phase = t % 4.2;
                let blink = if phase < 0.13 {
                    (phase / 0.065 - 1.0).abs().min(1.0)
                } else {
                    1.0
                };
                let ft = t - self.found_clock;
                if surprise > 0.05 {
                    eye_open = [blink, blink];
                    eye_scale = 1.0 + 0.25 * surprise;
                } else if tracking && ft < 0.8 {
                    let e = if ft < 0.45 {
                        0.5
                    } else {
                        0.5 + 0.5 * ease_out_cubic((ft - 0.45) / 0.35)
                    };
                    eye_open = [e * blink, e * blink];
                    sy += (ft / 0.8 * pi).sin() * 0.06;
                } else if searching {
                    eye_open = [0.6 * blink, 0.6 * blink];
                    eye_squash = [0.6, 0.6];
                } else {
                    eye_open = [blink, blink];
                }
            }
        }

        character::Face {
            pos,
            half: 0.34,
            scale: Vec2::new(sx, sy),
            eye_open,
            eye_squash,
            eye_scale,
            look: lookv,
        }
    }

    fn build(&mut self) {
        let section = self.section;
        let rv = self.reveal.value();
        let h = self.sphere_h.value();
        let vision = BEATS[section].vision;
        let face = self.face();
        let look = self.look_angle();

        let mut lines = std::mem::take(&mut self.scratch_lines);
        let mut tris = std::mem::take(&mut self.scratch_tris);
        lines.segments.clear();
        lines.colors.clear();
        lines.widths.clear();
        tris.vertices.clear();
        tris.indices.clear();
        tris.colors.clear();

        // Flatland: the ground checkerboard (the 2D cross-section). Far cells fade
        // to the horizon colour so the grid reads as infinite with no visible edge.
        let cells = (2.0 * GROUND_HALF / GROUND_STEP).round() as i32;
        let horizon = [SKY_HORIZON[0], SKY_HORIZON[1], SKY_HORIZON[2], 1.0];
        for i in 0..cells {
            for j in 0..cells {
                let x0 = -GROUND_HALF + i as f32 * GROUND_STEP;
                let z0 = -GROUND_HALF + j as f32 * GROUND_STEP;
                let base = if (i + j) & 1 == 0 { GROUND_A } else { GROUND_B };
                let d = (x0 + GROUND_STEP * 0.5).hypot(z0 + GROUND_STEP * 0.5);
                let f =
                    ((d - GROUND_FADE_NEAR) / (GROUND_FADE_FAR - GROUND_FADE_NEAR)).clamp(0.0, 1.0);
                let c = [
                    base[0] + (horizon[0] - base[0]) * f,
                    base[1] + (horizon[1] - base[1]) * f,
                    base[2] + (horizon[2] - base[2]) * f,
                    1.0,
                ];
                quad3(
                    &mut tris,
                    [
                        Vec3::new(x0, 0.0, z0),
                        Vec3::new(x0 + GROUND_STEP, 0.0, z0),
                        Vec3::new(x0 + GROUND_STEP, 0.0, z0 + GROUND_STEP),
                        Vec3::new(x0, 0.0, z0 + GROUND_STEP),
                    ],
                    c,
                );
            }
        }

        // Ambient Spaceland solids, faded in with the reveal.
        let space_fade = ((rv - 0.05) / 0.45).clamp(0.0, 1.0);
        if space_fade > 0.001 {
            let mut dc = COLOR_DISTANT;
            dc[3] *= space_fade;
            for (center, solid, scale, spin) in distants() {
                let rot = Mat3::from_rotation_y(self.clock * spin)
                    * Mat3::from_rotation_x(self.clock * spin * 0.5);
                let vs: Vec<Vec3> = solid
                    .vertices()
                    .iter()
                    .map(|v| center + rot * (*v * scale))
                    .collect();
                for e in solid.edges() {
                    push(&mut lines, vs[e[0] as usize], vs[e[1] as usize], dc, 1.2);
                }
            }
        }

        // The spinning polygons (future characters) living in the plane.
        for (c, n, rad, spin, color) in primitives() {
            let phase = self.clock * spin;
            for k in 0..n {
                let a0 = phase + TAU * k as f32 / n as f32;
                let a1 = phase + TAU * (k + 1) as f32 / n as f32;
                let p0 = c + Vec2::from_angle(a0) * rad;
                let p1 = c + Vec2::from_angle(a1) * rad;
                push(&mut lines, flat(p0), flat(p1), color, 2.0);
            }
        }

        // A Sphere: a 3D visitor descending along +y. Its section with the ground
        // is the closed form circle of radius sqrt(R^2 - h^2).
        let center = Vec3::new(SPHERE_XY.0, h, -SPHERE_XY.1);
        if h < SPHERE_TOP + 1.5 {
            let spin = Mat3::from_rotation_y(self.clock * 0.5);
            for e in &self.sphere_edges {
                let a = center + spin * self.sphere_verts[e[0] as usize];
                let b = center + spin * self.sphere_verts[e[1] as usize];
                push(&mut lines, a, b, COLOR_SPHERE, 1.4);
            }
        }
        if h.abs() < SPHERE_RADIUS {
            let rc = (SPHERE_RADIUS * SPHERE_RADIUS - h * h).sqrt();
            let n = 48;
            let ring: Vec<Vec3> = (0..n)
                .map(|i| flat(sphere_xy() + Vec2::from_angle(TAU * i as f32 / n as f32) * rc))
                .collect();
            fan3(&mut tris, &ring, COLOR_SECTION_FILL);
            for i in 0..n {
                push(&mut lines, ring[i], ring[(i + 1) % n], COLOR_SECTION, 3.0);
            }
        }

        // A Square's 1D field of view, drawn under him (a fill, no edges).
        if vision {
            let s = square_pos();
            let mut wedge = vec![flat(s)];
            for k in 0..=14 {
                let a = look - VISION_HALF_ANGLE + 2.0 * VISION_HALF_ANGLE * (k as f32 / 14.0);
                wedge.push(flat(s + Vec2::from_angle(a) * 6.0));
            }
            fan3(&mut tris, &wedge, COLOR_CONE);
        }

        // A Square last, so he sits over the section circle and the vision cone.
        let map = |p: Vec2| flat(p).to_array();
        character::push_face(&mut tris, &face, &map);

        self.scratch_lines = lines;
        self.scratch_tris = tris;
    }
}

impl Scene for FlatlandApp {
    fn new(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        // The world is the coplanar 2D ground plus flat decals; depth-testing it
        // against itself only causes z-fighting. Both nodes draw in painter order.
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
        let sky = ShaderEffect::new(
            &ctx.rd.device,
            ctx.rd.target_format(),
            SKY_WGSL,
            32,
            wgpu::BlendState::REPLACE,
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

        let mut app = Self {
            lines: line_node,
            tris: tri_node,
            sky,
            sphere_verts,
            sphere_edges: set.into_iter().collect(),
            scratch_lines: LineMesh::default(),
            scratch_tris: TriangleMesh::default(),
            clock: 0.0,
            section: 0,
            section_clock: 0.0,
            reveal: Animated::new(BEATS[0].reveal),
            sphere_h: Animated::new(sphere_target(BEATS[0].cue)),
            zoom: Animated::new(section_zoom(0)),
            gaze: sphere_xy(),
            gaze_target: sphere_xy(),
            cursor_present: false,
            woken: false,
            wake_clock: -1.0,
            found_clock: -10.0,
            vision_bands: Vec::new(),
            cursor_u: -1.0,
        };
        app.goto(0);
        Ok(app)
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        let dt = ctx.dt.min(0.1);
        self.clock += dt;
        self.reveal.advance(dt);
        self.sphere_h.advance(dt);
        self.zoom.advance(dt);

        // Wake once, the first time the cursor enters during a sleeping section.
        // It is a one-shot (re-armed only by returning to section 0).
        if BEATS[self.section].sleeping && self.cursor_present && !self.woken {
            self.woken = true;
            self.wake_clock = self.clock;
        }

        // He tracks the cursor only in the 2D (top-down) view, where the screen
        // maps cleanly onto the plane. Otherwise his gaze pans automatically (a
        // sweep across the shapes during the vision beat, the visitor otherwise).
        let cursor_drives = self.cursor_present && self.reveal.value() < 0.5;
        if !cursor_drives {
            if BEATS[self.section].vision {
                // Purposeful search: hold each of a few discrete look directions
                // (the gaze easing turns the steps into look-hold-look), rather
                // than aimlessly sweeping a circle.
                let dirs = [2.1_f32, 0.7, 1.6, 1.1, 0.4];
                let idx = (self.clock / 0.9) as usize % dirs.len();
                self.gaze_target = square_pos() + Vec2::from_angle(dirs[idx]) * 3.0;
            } else {
                self.gaze_target = sphere_xy();
            }
        }
        self.gaze += (self.gaze_target - self.gaze) * (1.0 - (-dt * 10.0).exp());

        // A Square's 1D vision: shapes in his field of view, with distance shading
        // and ordered far -> near so a near shape occludes a far one.
        let s = square_pos();
        let look = self.look_angle();
        let bright = |d: f32| (1.0 - (d - 2.0) * 0.08).clamp(0.45, 1.0);
        let mut bands: Vec<(f32, f32, f32, [f32; 3])> = Vec::new();
        let h = self.sphere_h.value();
        if h.abs() < SPHERE_RADIUS {
            let rc = (SPHERE_RADIUS * SPHERE_RADIUS - h * h).sqrt();
            if let Some((lo, hi)) = angular_band(s, look, sphere_xy(), rc) {
                let d = (sphere_xy() - s).length();
                bands.push((
                    lo,
                    hi,
                    bright(d),
                    [COLOR_SPHERE[0], COLOR_SPHERE[1], COLOR_SPHERE[2]],
                ));
            }
        }
        if BEATS[self.section].vision {
            for (c, n, rad, spin, color) in primitives() {
                if let Some((lo, hi, d)) = silhouette_band(s, look, c, n, rad, self.clock * spin) {
                    bands.push((lo, hi, bright(d), [color[0], color[1], color[2]]));
                }
            }
        }
        bands.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        self.vision_bands = bands;

        // The cursor marker sits where the cursor actually is in his field of
        // view. His gaze lags, so while he catches up the cursor rides off-center.
        // It is hidden when a nearer shape stands between him and the cursor.
        self.cursor_u = if cursor_drives {
            let to = self.gaze_target - s;
            let mut rel = to.y.atan2(to.x) - look;
            while rel > std::f32::consts::PI {
                rel -= TAU;
            }
            while rel < -std::f32::consts::PI {
                rel += TAU;
            }
            let cur_n = rel / VISION_HALF_ANGLE;
            let cursor_bright = bright(to.length());
            let occluded = self
                .vision_bands
                .iter()
                .any(|(lo, hi, b, _)| *b > cursor_bright + 1e-3 && cur_n >= *lo && cur_n <= *hi);
            let u = (cur_n + 1.0) * 0.5;
            if occluded || !(-0.1..=1.1).contains(&u) {
                -1.0
            } else {
                u.clamp(0.0, 1.0)
            }
        } else {
            -1.0
        };
    }

    fn render(
        &mut self,
        rd: &RenderDevice,
        view: &wgpu::TextureView,
        viewport: Viewport,
    ) -> Result<()> {
        self.build();

        let vw = viewport.width.max(1);
        let vh = viewport.height.max(1);
        let view_proj = self.camera_view_proj(vw as f32 / vh as f32);

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
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            drop(_clear);
            rd.queue.submit(Some(enc.finish()));
        }

        self.sky.set_uniforms(
            &rd.queue,
            &f32_le(&[
                SKY_TOP[0],
                SKY_TOP[1],
                SKY_TOP[2],
                1.0,
                SKY_HORIZON[0],
                SKY_HORIZON[1],
                SKY_HORIZON[2],
                1.0,
            ]),
        );
        self.sky.execute(rd, view, Some(&viewport))?;

        self.tris.set_camera(&rd.queue, view_proj);
        self.tris.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &self.scratch_tris,
            &Projection::Identity,
        );
        self.tris.execute(rd, view, None, Some(&viewport))?;
        self.lines
            .set_camera(&rd.queue, view_proj, Vec2::new(vw as f32, vh as f32));
        self.lines.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &self.scratch_lines,
            &Projection::Identity,
            1,
        );
        self.lines.execute(rd, view, None, Some(&viewport))?;
        Ok(())
    }

    fn ui(&mut self, ctx: &egui::Context, _frame: &mut FrameCtx<'_>) {
        let screen = ctx.content_rect();
        let hover = ctx.input(|i| i.pointer.hover_pos());
        let now_present = hover.is_some();
        if now_present && !self.cursor_present {
            self.found_clock = self.clock;
        }
        self.cursor_present = now_present;
        // The cursor maps cleanly onto the plane only in the top-down view; once
        // the camera tilts into 3D, stop steering his gaze by it.
        if let Some(p) = hover.filter(|_| self.reveal.value() < 0.5) {
            let ext_y = self.zoom.value();
            let ext_x = ext_y * screen.width().max(1.0) / screen.height().max(1.0);
            let ndc_x = ((p.x - screen.left()) / screen.width()) * 2.0 - 1.0;
            let ndc_up = 1.0 - ((p.y - screen.top()) / screen.height()) * 2.0;
            self.gaze_target = square_pos() + Vec2::new(ndc_x * ext_x, ndc_up * ext_y);
        }

        // Dark-to-light intro: a full-screen black veil that fades out.
        let veil = 1.0 - self.intro();
        if veil > 0.001 {
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Background,
                egui::Id::new("intro-veil"),
            ));
            painter.rect_filled(
                screen,
                0.0,
                egui::Color32::from_black_alpha((veil * 255.0) as u8),
            );
        }

        self.vision_window(ctx);

        // Caption, fading in on section change.
        let fade = ((self.clock - self.section_clock) / 0.4).clamp(0.0, 1.0);
        let a = (fade * 255.0) as u8;
        egui::Area::new(egui::Id::new("caption"))
            .anchor(egui::Align2::CENTER_BOTTOM, [0.0, -64.0])
            .show(ctx, |ui| {
                egui::Frame::popup(&ctx.style())
                    .fill(egui::Color32::from_rgba_unmultiplied(18, 22, 30, a / 2))
                    .stroke(egui::Stroke::NONE)
                    .show(ui, |ui| {
                        ui.set_max_width(640.0);
                        ui.vertical_centered(|ui| {
                            ui.label(
                                egui::RichText::new(BEATS[self.section].caption)
                                    .size(18.0)
                                    .color(egui::Color32::from_rgba_unmultiplied(225, 230, 238, a)),
                            );
                        });
                    });
            });

        self.controls(ctx);
    }

    fn title(&self) -> std::borrow::Cow<'static, str> {
        "flatland".into()
    }
}

impl FlatlandApp {
    fn vision_window(&self, ctx: &egui::Context) {
        if !BEATS[self.section].vision {
            return;
        }
        // Slide in from the left when the vision beat opens.
        let slide = ((self.clock - self.section_clock) / 0.4).clamp(0.0, 1.0);
        let x = -300.0 + 318.0 * ease_out_cubic(slide);
        let mut u = vec![
            self.vision_bands.len().min(6) as f32,
            self.clock,
            self.cursor_u,
            0.0,
        ];
        for i in 0..6 {
            if let Some((lo, hi, b, _)) = self.vision_bands.get(i) {
                u.extend([(lo + 1.0) * 0.5, (hi + 1.0) * 0.5, *b, 0.0]);
            } else {
                u.extend([0.0; 4]);
            }
        }
        for i in 0..6 {
            if let Some((_, _, _, c)) = self.vision_bands.get(i) {
                u.extend([c[0], c[1], c[2], 1.0]);
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
                        .color(egui::Color32::from_gray(150)),
                );
                rye_egui::shader_widget(ui, egui::vec2(280.0, 26.0), VISION_WGSL, 208, f32_le(&u));
            });
    }

    fn controls(&mut self, ctx: &egui::Context) {
        let mut goto: Option<usize> = None;
        let section = self.section;
        let last = BEATS.len() - 1;

        egui::Area::new(egui::Id::new("controls"))
            .anchor(egui::Align2::CENTER_BOTTOM, [0.0, -16.0])
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(section > 0, egui::Button::new("\u{25C0} Prev"))
                        .clicked()
                    {
                        goto = Some(section - 1);
                    }
                    ui.label(
                        egui::RichText::new(format!("Section {} / {}", section + 1, last + 1))
                            .size(14.0)
                            .color(egui::Color32::from_gray(210)),
                    );
                    if ui
                        .add_enabled(section < last, egui::Button::new("Next \u{25B6}"))
                        .clicked()
                    {
                        goto = Some(section + 1);
                    }
                });
            });

        if let Some(s) = goto {
            self.goto(s);
        }
    }
}

fn main() -> Result<()> {
    // `--offline` renders deterministically to disk with no window (the harness
    // build). Detected from raw argv because `Args` only parses `--key=value`.
    #[cfg(feature = "harness")]
    if std::env::args().any(|a| a == "--offline") {
        return run_offline();
    }
    run_scene::<FlatlandApp>(RunConfig {
        window: WindowAttributes::default()
            .with_title("flatland")
            .with_visible(false),
        msaa_samples: 4,
        ..RunConfig::default()
    })
}

/// Render flatland headless over a fixed-dt timeline window. `--scenario=NAME`
/// selects a preset (window + scripted cursor); otherwise `--from`/`--to`
/// seconds, `--fps`, and an optional bare run with no cursor. `--out` is a
/// `.png` file for a single frame, else a directory for a `frame_NNNN.png`
/// sequence.
#[cfg(feature = "harness")]
fn run_offline() -> Result<()> {
    use rye_app::harness::OfflineRender;

    let args = rye_app::args::Args::current();
    let out = args.get("out").unwrap_or("captures/offline").to_string();

    // (from, to, fps, cursor) by scenario; bare runs read --from/--to/--fps.
    let (from, to, fps, cursor) = match args.get("scenario") {
        Some("wake") => (0.0, 2.4, 12, Some(wake_cursor())),
        Some(other) => anyhow::bail!("unknown --scenario={other} (known: wake)"),
        None => (
            args.parse("from").unwrap_or(0.0),
            args.parse("to").unwrap_or(0.0),
            args.parse("fps").unwrap_or(1),
            None,
        ),
    };

    let cfg = OfflineRender {
        width: 1280,
        height: 720,
        from,
        to,
        fps,
        cursor,
        out: std::path::Path::new(&out),
    };
    let frames = rye_app::harness::render_scene::<FlatlandApp>(&cfg)?;
    println!(
        "offline: wrote {} frame(s) to {out} ({}x{}, {from}..{to}s @ {fps}fps)",
        frames.len(),
        cfg.width,
        cfg.height,
    );
    Ok(())
}

/// The "wake" scenario cursor: A Square sleeps alone, the visitor's cursor
/// arrives upper-left (~0.55s) to trigger the startle-search-find wake, then
/// drifts right (~1.9s) so his gaze visibly tracks it. Screen points in a
/// 1280x720 frame; center (640,360) maps to him looking at himself.
#[cfg(feature = "harness")]
fn wake_cursor() -> rye_app::harness::CursorTrack {
    rye_app::harness::CursorTrack::new(vec![
        (0.0, None),
        (0.55, Some((400.0, 235.0))),
        (1.9, Some((840.0, 300.0))),
    ])
}

#[cfg(test)]
mod shader_tests {
    use super::*;

    fn check(fragment: &str) {
        let src = format!("{}\n{}", rye_render::FULLSCREEN_VERTEX_WGSL, fragment);
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("WGSL parse error:\n{}", e.emit_to_string(&src)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("WGSL validation error: {e:?}"));
    }

    #[test]
    fn sky_wgsl_is_valid() {
        check(SKY_WGSL);
    }

    #[test]
    fn vision_wgsl_is_valid() {
        check(VISION_WGSL);
    }
}
