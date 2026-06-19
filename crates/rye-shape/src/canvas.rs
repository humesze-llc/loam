//! Immediate-mode 2D drawing onto the engine's [`LineMesh<3>`] / [`TriangleMesh<3>`]
//! at a chosen `z`, with an affine transform stack. Replaces the per-demo hand
//! building of circles, fans, quads, and character transforms: a demo draws
//! shapes through a [`Canvas`], then uploads `canvas.tris` / `canvas.lines`.
//!
//! Polygons are assumed convex (fan-triangulated); every current consumer
//! (discs, rects, regular polygons, character parts) is convex.

use glam::{Affine2, Mat2, Vec2};
use std::f32::consts::TAU;

use crate::{LineMesh, TriangleMesh};

/// Accumulates 2D geometry under a transform stack. `z` places everything on a
/// chosen depth plane (e.g. just in front of a backdrop).
pub struct Canvas {
    pub lines: LineMesh<3>,
    pub tris: TriangleMesh<3>,
    stack: Vec<Affine2>,
    z: f32,
}

impl Default for Canvas {
    fn default() -> Self {
        Self::new()
    }
}

impl Canvas {
    pub fn new() -> Self {
        Self {
            lines: LineMesh::default(),
            tris: TriangleMesh::default(),
            stack: vec![Affine2::IDENTITY],
            z: 0.0,
        }
    }

    /// Depth plane for subsequent draws.
    pub fn set_z(&mut self, z: f32) {
        self.z = z;
    }

    /// Push a copy of the current transform; pair with [`Self::restore`].
    pub fn save(&mut self) {
        let top = *self.stack.last().unwrap();
        self.stack.push(top);
    }

    pub fn restore(&mut self) {
        if self.stack.len() > 1 {
            self.stack.pop();
        }
    }

    /// Compose `m` onto the current transform (applied to subsequent points).
    pub fn transform(&mut self, m: Affine2) {
        let top = self.stack.last_mut().unwrap();
        *top *= m;
    }

    pub fn translate(&mut self, v: Vec2) {
        self.transform(Affine2::from_translation(v));
    }

    pub fn rotate(&mut self, radians: f32) {
        self.transform(Affine2::from_angle(radians));
    }

    pub fn scale(&mut self, s: Vec2) {
        self.transform(Affine2::from_scale(s));
    }

    /// Shear: `x += kx*y`, `y += ky*x`.
    pub fn shear(&mut self, kx: f32, ky: f32) {
        self.transform(Affine2::from_mat2(Mat2::from_cols_array(&[
            1.0, ky, kx, 1.0,
        ])));
    }

    fn pt(&self, p: Vec2) -> [f32; 3] {
        let q = self.stack.last().unwrap().transform_point2(p);
        [q.x, q.y, self.z]
    }

    pub fn line(&mut self, a: Vec2, b: Vec2, color: [f32; 4], width: f32) {
        self.lines.segments.push((self.pt(a), self.pt(b)));
        self.lines.colors.push((color, color));
        self.lines.widths.push(width);
    }

    pub fn stroke_poly(&mut self, pts: &[Vec2], closed: bool, color: [f32; 4], width: f32) {
        if pts.len() < 2 {
            return;
        }
        for i in 0..pts.len() - 1 {
            self.line(pts[i], pts[i + 1], color, width);
        }
        if closed && pts.len() > 2 {
            self.line(pts[pts.len() - 1], pts[0], color, width);
        }
    }

    /// Fill a convex polygon (fan-triangulated).
    pub fn fill_poly(&mut self, pts: &[Vec2], color: [f32; 4]) {
        if pts.len() < 3 {
            return;
        }
        let base = self.tris.vertices.len() as u32;
        for p in pts {
            self.tris.vertices.push(self.pt(*p));
            self.tris.colors.push(color);
        }
        for k in 1..pts.len() as u32 - 1 {
            self.tris.indices.push([base, base + k, base + k + 1]);
        }
    }

    pub fn fill_rect(&mut self, min: Vec2, max: Vec2, color: [f32; 4]) {
        self.fill_poly(
            &[min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)],
            color,
        );
    }

    pub fn disc(&mut self, center: Vec2, radius: f32, color: [f32; 4], segments: usize) {
        self.fill_poly(&ring_points(center, radius, segments), color);
    }

    pub fn circle(
        &mut self,
        center: Vec2,
        radius: f32,
        color: [f32; 4],
        width: f32,
        segments: usize,
    ) {
        self.stroke_poly(&ring_points(center, radius, segments), true, color, width);
    }

    pub fn fill_regular(
        &mut self,
        center: Vec2,
        sides: usize,
        radius: f32,
        rotation: f32,
        color: [f32; 4],
    ) {
        self.fill_poly(&regular_points(center, sides, radius, rotation), color);
    }

    pub fn stroke_regular(
        &mut self,
        center: Vec2,
        sides: usize,
        radius: f32,
        rotation: f32,
        color: [f32; 4],
        width: f32,
    ) {
        self.stroke_poly(
            &regular_points(center, sides, radius, rotation),
            true,
            color,
            width,
        );
    }
}

fn ring_points(center: Vec2, radius: f32, segments: usize) -> Vec<Vec2> {
    (0..segments.max(3))
        .map(|i| center + Vec2::from_angle(TAU * i as f32 / segments.max(3) as f32) * radius)
        .collect()
}

fn regular_points(center: Vec2, sides: usize, radius: f32, rotation: f32) -> Vec<Vec2> {
    (0..sides.max(3))
        .map(|i| {
            center + Vec2::from_angle(rotation + TAU * i as f32 / sides.max(3) as f32) * radius
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_rect_is_two_triangles() {
        let mut c = Canvas::new();
        c.fill_rect(Vec2::new(-1.0, -1.0), Vec2::new(1.0, 1.0), [1.0; 4]);
        assert_eq!(c.tris.indices.len(), 2);
        assert_eq!(c.tris.vertices.len(), 4);
    }

    #[test]
    fn disc_area_approaches_pi_r_squared() {
        let mut c = Canvas::new();
        c.disc(Vec2::ZERO, 2.0, [1.0; 4], 128);
        let area: f32 = c
            .tris
            .indices
            .iter()
            .map(|t| {
                let a = c.tris.vertices[t[0] as usize];
                let b = c.tris.vertices[t[1] as usize];
                let d = c.tris.vertices[t[2] as usize];
                0.5 * ((b[0] - a[0]) * (d[1] - a[1]) - (d[0] - a[0]) * (b[1] - a[1])).abs()
            })
            .sum();
        assert!((area - std::f32::consts::PI * 4.0).abs() < 0.05);
    }

    #[test]
    fn transform_and_z_apply_to_points() {
        let mut c = Canvas::new();
        c.set_z(0.5);
        c.translate(Vec2::new(3.0, 0.0));
        c.fill_poly(
            &[Vec2::ZERO, Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)],
            [1.0; 4],
        );
        assert_eq!(c.tris.vertices[0], [3.0, 0.0, 0.5]);
        assert_eq!(c.tris.vertices[1], [4.0, 0.0, 0.5]);
    }

    #[test]
    fn save_restore_isolates_transform() {
        let mut c = Canvas::new();
        c.save();
        c.translate(Vec2::new(5.0, 5.0));
        c.restore();
        c.fill_poly(
            &[Vec2::ZERO, Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)],
            [1.0; 4],
        );
        assert_eq!(c.tris.vertices[0], [0.0, 0.0, 0.0]);
    }
}
