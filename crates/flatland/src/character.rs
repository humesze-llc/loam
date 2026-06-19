//! A Square as a solid-coloured, mouthless character with white eyes. Expression
//! comes entirely from the body's lean (horizontal shear) and stretch (vertical
//! scale) and from the eyes, which go closed (asleep) -> resting semicircle ->
//! full interested circle and track a gaze point. Drawn on the z=0 plane.

use glam::Vec2;
use rye_shape::{fill_convex_polygon, LineMesh, TriangleMesh};
use std::f32::consts::TAU;

const BODY: [f32; 4] = [0.20, 0.52, 0.60, 1.0];
const OUTLINE: [f32; 4] = [0.10, 0.30, 0.36, 1.0];
const EYE_WHITE: [f32; 4] = [0.98, 0.99, 0.97, 1.0];
const PUPIL: [f32; 4] = [0.10, 0.16, 0.20, 1.0];

pub struct Face {
    pub pos: Vec2,
    pub half: f32,
    /// Horizontal shear; positive leans toward +x.
    pub lean: f32,
    /// Vertical scale delta (breathing + reaching up): 0 is rest.
    pub stretch: f32,
    /// Per-eye openness: 0 closed, 0.5 resting semicircle, 1 full circle.
    pub eye_open: [f32; 2],
    /// Gaze direction the pupils shift toward.
    pub look: Vec2,
}

/// Lean (shear by height) and stretch (vertical scale) applied to every point, so
/// body and eyes deform together.
fn xf(face: &Face, p: Vec2) -> Vec2 {
    let rel = p - face.pos;
    let rel = Vec2::new(rel.x, rel.y * (1.0 + face.stretch));
    face.pos + Vec2::new(rel.x + face.lean * rel.y, rel.y)
}

fn disc(face: &Face, center: Vec2, radius: f32, n: usize) -> Vec<Vec2> {
    (0..n)
        .map(|i| {
            xf(
                face,
                center + Vec2::from_angle(TAU * i as f32 / n as f32) * radius,
            )
        })
        .collect()
}

pub fn push_face(lines: &mut LineMesh<3>, tris: &mut TriangleMesh<3>, face: &Face) {
    let h = face.half;
    let p = face.pos;

    let body: Vec<Vec2> = [
        Vec2::new(-h, -h),
        Vec2::new(h, -h),
        Vec2::new(h, h),
        Vec2::new(-h, h),
    ]
    .iter()
    .map(|&c| xf(face, p + c))
    .collect();
    append(tris, &fill_convex_polygon(&body, BODY));
    for i in 0..4 {
        seg(lines, body[i], body[(i + 1) % 4], OUTLINE, 2.0);
    }

    let eye_r = h * 0.30;
    let look = if face.look.length() > 1e-4 {
        face.look.normalize()
    } else {
        Vec2::new(0.0, 1.0)
    };
    for (idx, sx) in [(0usize, -1.0_f32), (1, 1.0)] {
        let socket = p + Vec2::new(sx * h * 0.40, h * 0.16);
        let open = face.eye_open[idx].clamp(0.0, 1.0);
        if open < 0.12 {
            // Asleep: a small closed curve, no disc.
            let a = xf(face, socket + Vec2::new(-eye_r * 0.85, eye_r * 0.1));
            let m = xf(face, socket + Vec2::new(0.0, -eye_r * 0.22));
            let b = xf(face, socket + Vec2::new(eye_r * 0.85, eye_r * 0.1));
            seg(lines, a, m, PUPIL, 2.5);
            seg(lines, m, b, PUPIL, 2.5);
            continue;
        }
        append(
            tris,
            &fill_convex_polygon(&disc(face, socket, eye_r, 18), EYE_WHITE),
        );
        let pupil = socket + look * (eye_r * 0.42);
        append(
            tris,
            &fill_convex_polygon(&disc(face, pupil, eye_r * 0.5, 12), PUPIL),
        );
        // Body-coloured lid over the top `1 - open` of the eye: semicircle (rest)
        // to circle (interested).
        let lid = 1.0 - open;
        if lid > 0.01 {
            let y_top = socket.y + eye_r * 1.1;
            let y_cut = y_top - 2.2 * eye_r * lid;
            let cap = [
                xf(face, Vec2::new(socket.x - eye_r * 1.2, y_cut)),
                xf(face, Vec2::new(socket.x + eye_r * 1.2, y_cut)),
                xf(face, Vec2::new(socket.x + eye_r * 1.2, y_top)),
                xf(face, Vec2::new(socket.x - eye_r * 1.2, y_top)),
            ];
            append(tris, &fill_convex_polygon(&cap, BODY));
        }
    }
}

fn seg(mesh: &mut LineMesh<3>, a: Vec2, b: Vec2, color: [f32; 4], width: f32) {
    mesh.segments.push(([a.x, a.y, 0.0], [b.x, b.y, 0.0]));
    mesh.colors.push((color, color));
    mesh.widths.push(width);
}

fn append(dst: &mut TriangleMesh<3>, src: &TriangleMesh<3>) {
    let base = dst.vertices.len() as u32;
    dst.vertices.extend_from_slice(&src.vertices);
    dst.colors.extend_from_slice(&src.colors);
    for t in &src.indices {
        dst.indices.push([t[0] + base, t[1] + base, t[2] + base]);
    }
}
