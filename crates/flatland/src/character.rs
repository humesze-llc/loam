//! A Square: a solid-coloured, mouthless character with white eyes, drawn through
//! [`rye_shape::Canvas`]. Expression comes from body squash/stretch (the Canvas
//! scale) and the eyes (gaze + a lid from closed -> resting semicircle -> full
//! circle). No shear/lean: he floats centered in Flatland.

use glam::Vec2;
use rye_shape::{Canvas, LineMesh, TriangleMesh};

const BODY: [f32; 4] = [0.20, 0.52, 0.60, 1.0];
const OUTLINE: [f32; 4] = [0.10, 0.30, 0.36, 1.0];
const EYE_WHITE: [f32; 4] = [0.98, 0.99, 0.97, 1.0];
const PUPIL: [f32; 4] = [0.10, 0.16, 0.20, 1.0];

pub struct Face {
    pub pos: Vec2,
    pub half: f32,
    /// Non-uniform scale about the center (squash/stretch); `(1, 1)` is rest.
    pub scale: Vec2,
    /// Per-eye openness: 0 closed, 0.5 resting semicircle, 1 full circle.
    pub eye_open: [f32; 2],
    /// Gaze direction the pupils shift toward.
    pub look: Vec2,
}

pub fn push_face(lines: &mut LineMesh<3>, tris: &mut TriangleMesh<3>, face: &Face) {
    let mut c = Canvas::new();
    c.translate(face.pos);
    c.scale(face.scale);

    let h = face.half;
    let body = [
        Vec2::new(-h, -h),
        Vec2::new(h, -h),
        Vec2::new(h, h),
        Vec2::new(-h, h),
    ];
    c.fill_poly(&body, BODY);
    c.stroke_poly(&body, true, OUTLINE, 2.0);

    let eye_r = h * 0.30;
    let look = if face.look.length() > 1e-4 {
        face.look.normalize()
    } else {
        Vec2::new(0.0, 1.0)
    };
    for (idx, sx) in [(0usize, -1.0_f32), (1, 1.0)] {
        let socket = Vec2::new(sx * h * 0.40, h * 0.16);
        let open = face.eye_open[idx].clamp(0.0, 1.0);
        if open < 0.12 {
            c.line(
                socket + Vec2::new(-eye_r * 0.85, eye_r * 0.1),
                socket + Vec2::new(0.0, -eye_r * 0.22),
                PUPIL,
                2.5,
            );
            c.line(
                socket + Vec2::new(0.0, -eye_r * 0.22),
                socket + Vec2::new(eye_r * 0.85, eye_r * 0.1),
                PUPIL,
                2.5,
            );
            continue;
        }
        c.disc(socket, eye_r, EYE_WHITE, 18);
        c.disc(socket + look * (eye_r * 0.42), eye_r * 0.5, PUPIL, 12);
        let lid = 1.0 - open;
        if lid > 0.01 {
            let y_top = socket.y + eye_r * 1.1;
            let y_cut = y_top - 2.2 * eye_r * lid;
            c.fill_poly(
                &[
                    Vec2::new(socket.x - eye_r * 1.2, y_cut),
                    Vec2::new(socket.x + eye_r * 1.2, y_cut),
                    Vec2::new(socket.x + eye_r * 1.2, y_top),
                    Vec2::new(socket.x - eye_r * 1.2, y_top),
                ],
                BODY,
            );
        }
    }

    lines.segments.extend_from_slice(&c.lines.segments);
    lines.colors.extend_from_slice(&c.lines.colors);
    lines.widths.extend_from_slice(&c.lines.widths);
    let base = tris.vertices.len() as u32;
    tris.vertices.extend_from_slice(&c.tris.vertices);
    tris.colors.extend_from_slice(&c.tris.colors);
    for t in &c.tris.indices {
        tris.indices.push([t[0] + base, t[1] + base, t[2] + base]);
    }
}
