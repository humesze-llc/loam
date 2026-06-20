//! A Square: a single flat color, no border, with off-white eyes and no pupils.
//! Expressiveness comes from the eye shape morphing rectangle (closed) ->
//! semicircle (resting/happy) -> circle (open/alert), the body squash/stretch
//! (which also ovals the eyes), and the gaze shift. Geometry is built in 2D and
//! placed into the world by a caller-supplied `map`, so the character does not
//! need to know the world orientation.

use glam::Vec2;
use rye_shape::TriangleMesh;
use std::f32::consts::{PI, TAU};

const BODY: [f32; 4] = [0.22, 0.72, 0.80, 1.0];
const EYE: [f32; 4] = [0.93, 0.96, 0.97, 1.0];
const EYE_N: usize = 24;

pub struct Face {
    pub pos: Vec2,
    pub half: f32,
    /// Non-uniform scale about the center (squash/stretch); `(1, 1)` is rest.
    pub scale: Vec2,
    /// Per-eye shape: 0 closed (rectangle), 0.5 resting (semicircle), 1 open
    /// (circle).
    pub eye_open: [f32; 2],
    /// Gaze direction the eyes shift toward.
    pub look: Vec2,
}

pub fn push_face(tris: &mut TriangleMesh<3>, face: &Face, map: &dyn Fn(Vec2) -> [f32; 3]) {
    let h = face.half;
    let s = face.scale;
    let c = face.pos;

    let body = [
        c + Vec2::new(-h * s.x, -h * s.y),
        c + Vec2::new(h * s.x, -h * s.y),
        c + Vec2::new(h * s.x, h * s.y),
        c + Vec2::new(-h * s.x, h * s.y),
    ];
    fill(tris, &body, BODY, map);

    let look = if face.look.length() > 1e-4 {
        face.look.normalize()
    } else {
        Vec2::ZERO
    };
    let eye_r = 0.24 * h;
    for (idx, sgn) in [(0usize, -1.0_f32), (1, 1.0)] {
        let socket = c + Vec2::new(sgn * 0.40 * h * s.x, 0.12 * h * s.y) + look * (0.12 * h);
        let poly: Vec<Vec2> = eye_polygon(face.eye_open[idx], eye_r)
            .into_iter()
            .map(|p| socket + Vec2::new(p.x * s.x, p.y * s.y))
            .collect();
        fill(tris, &poly, EYE, map);
    }
}

fn fill(
    tris: &mut TriangleMesh<3>,
    poly: &[Vec2],
    color: [f32; 4],
    map: &dyn Fn(Vec2) -> [f32; 3],
) {
    if poly.len() < 3 {
        return;
    }
    let base = tris.vertices.len() as u32;
    for p in poly {
        tris.vertices.push(map(*p));
        tris.colors.push(color);
    }
    for k in 1..poly.len() as u32 - 1 {
        tris.indices.push([base, base + k, base + k + 1]);
    }
}

/// The eye boundary as `EYE_N` points, morphing rectangle (closed, o=0) ->
/// semicircle (resting, o=0.5) -> circle (open, o=1). All three keys share the
/// same perimeter parametrisation (CCW from +x), so the per-point lerp is well
/// defined.
fn eye_polygon(o: f32, r: f32) -> Vec<Vec2> {
    let o = o.clamp(0.0, 1.0);
    (0..EYE_N)
        .map(|i| {
            let u = i as f32 / EYE_N as f32;
            let a = TAU * u;
            let circle = Vec2::new(a.cos() * r, a.sin() * r);
            let semi = semicircle_point(u, r);
            if o >= 0.5 {
                semi.lerp(circle, (o - 0.5) * 2.0)
            } else {
                rect_point(u, r).lerp(semi, o * 2.0)
            }
        })
        .collect()
}

/// Resting eye: flat top, round bottom (a smiling "u" shape). Upper half is the
/// flat top edge; lower half is the circle's lower arc (so it matches the circle
/// there and only the top flattens as o drops from 1 to 0.5).
fn semicircle_point(u: f32, r: f32) -> Vec2 {
    if u < 0.5 {
        Vec2::new(r - 2.0 * r * (u / 0.5), 0.0)
    } else {
        let a = PI + PI * ((u - 0.5) / 0.5);
        Vec2::new(a.cos() * r, a.sin() * r)
    }
}

/// Closed eye: a thin horizontal bar, same perimeter direction as the arc.
fn rect_point(u: f32, r: f32) -> Vec2 {
    let rh = 0.16 * r;
    if u < 0.5 {
        Vec2::new(r - 2.0 * r * (u / 0.5), rh)
    } else {
        Vec2::new(-r + 2.0 * r * ((u - 0.5) / 0.5), -rh)
    }
}
