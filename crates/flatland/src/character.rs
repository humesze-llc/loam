//! A Square as a little character, drawn in the 2D top-down pane where his face
//! is visible: a filled body, two eyes whose pupils track a gaze point, and a
//! mouth that opens from rest to surprise.

use glam::Vec2;
use rye_shape::{fill_convex_polygon, LineMesh, TriangleMesh};

const BODY_FILL: [f32; 4] = [0.82, 0.91, 0.87, 1.0];
const OUTLINE: [f32; 4] = [0.12, 0.40, 0.36, 1.0];
const EYE_WHITE: [f32; 4] = [0.98, 0.99, 0.97, 1.0];
const PUPIL: [f32; 4] = [0.10, 0.20, 0.22, 1.0];

pub struct Face {
    pub pos: Vec2,
    pub half: f32,
    pub gaze: Vec2,
    pub surprise: f32,
}

pub fn push_face(lines: &mut LineMesh<3>, tris: &mut TriangleMesh<3>, face: &Face) {
    let p = face.pos;
    let h = face.half;
    let corners = [
        p + Vec2::new(-h, -h),
        p + Vec2::new(h, -h),
        p + Vec2::new(h, h),
        p + Vec2::new(-h, h),
    ];

    append(tris, &fill_convex_polygon(&corners, BODY_FILL));
    for i in 0..4 {
        seg(lines, corners[i], corners[(i + 1) % 4], OUTLINE, 2.0);
    }

    // Eyes sit toward the +x front, the direction A Square watches from.
    let eye_r = h * 0.26;
    for sy in [-1.0_f32, 1.0] {
        let eye = p + Vec2::new(h * 0.38, sy * h * 0.38);
        append(
            tris,
            &fill_convex_polygon(&circle(eye, eye_r, 16), EYE_WHITE),
        );
        let dir = face.gaze - eye;
        let dir = if dir.length() > 1e-4 {
            dir.normalize()
        } else {
            Vec2::new(1.0, 0.0)
        };
        let pupil = eye + dir * (eye_r * 0.45);
        append(
            tris,
            &fill_convex_polygon(&circle(pupil, eye_r * 0.5, 12), PUPIL),
        );
    }

    // Mouth: a small mark that grows into an O with surprise.
    let mouth = p + Vec2::new(h * 0.62, -h * 0.42);
    let radius = h * (0.06 + 0.22 * face.surprise.clamp(0.0, 1.0));
    append(
        tris,
        &fill_convex_polygon(&circle(mouth, radius, 14), PUPIL),
    );
}

fn circle(center: Vec2, radius: f32, n: usize) -> Vec<Vec2> {
    (0..n)
        .map(|i| {
            let a = std::f32::consts::TAU * i as f32 / n as f32;
            center + Vec2::from_angle(a) * radius
        })
        .collect()
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
