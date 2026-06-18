//! Cross-section of a 3D mesh edge set by an axis-aligned z-slice, returned as
//! an ordered 2D polygon: the shape a Flatland inhabitant at `z = z_slice`
//! perceives. Convex case only (regular solids), so angular ordering about the
//! centroid recovers the boundary; a concave section would need edge-adjacency
//! walking, which no current caller wants.

use glam::{Vec2, Vec3};
use rye_math::{EuclideanR3, SectionableSpace, ZPlane};

/// Two section points closer than this (in R²) are the same crossing: a slice
/// through a shared vertex yields an identical point on every incident edge.
const SECTION_POINT_MERGE: f32 = 1e-6;

/// Convex cross-section of `(vertices, edges)` by `slice`, as a CCW-ordered 2D
/// polygon. Empty if the slice yields fewer than 3 distinct crossings (a miss,
/// or a vertex/edge graze, which is not a polygon a Flatlander would see as an
/// area).
pub fn convex_section_polygon(vertices: &[Vec3], edges: &[[u32; 2]], slice: ZPlane) -> Vec<Vec2> {
    let mut pts: Vec<Vec2> = Vec::new();
    for e in edges {
        let p0 = vertices[e[0] as usize];
        let p1 = vertices[e[1] as usize];
        if let Some((_, p)) = <EuclideanR3 as SectionableSpace<3>>::edge_section(&slice, p0, p1) {
            let merge2 = SECTION_POINT_MERGE * SECTION_POINT_MERGE;
            if !pts.iter().any(|q| q.distance_squared(p) < merge2) {
                pts.push(p);
            }
        }
    }
    if pts.len() < 3 {
        return Vec::new();
    }

    let mut centroid = Vec2::ZERO;
    for p in &pts {
        centroid += *p;
    }
    centroid /= pts.len() as f32;

    // The section of a convex solid is convex, so angular order about the
    // centroid is the boundary order. total_cmp keeps it deterministic.
    pts.sort_by(|a, b| {
        let aa = (a.y - centroid.y).atan2(a.x - centroid.x);
        let bb = (b.y - centroid.y).atan2(b.x - centroid.x);
        aa.total_cmp(&bb)
    });
    pts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Solid3;

    fn signed_area(poly: &[Vec2]) -> f32 {
        let mut a = 0.0;
        for i in 0..poly.len() {
            let p = poly[i];
            let q = poly[(i + 1) % poly.len()];
            a += p.x * q.y - q.x * p.y;
        }
        a * 0.5
    }

    /// Cube through its center: a square at the four vertical-edge midpoints, the
    /// eight horizontal edges being parallel to the slice. CCW, side 2c.
    #[test]
    fn cube_midslice_is_ccw_square() {
        let v = Solid3::Cube.vertices();
        let e = Solid3::Cube.edges();
        let poly = convex_section_polygon(&v, &e, ZPlane::new(0.0));
        assert_eq!(poly.len(), 4);
        let c = 1.0 / 3.0_f32.sqrt();
        for p in &poly {
            assert!((p.x.abs() - c).abs() < 1e-5 && (p.y.abs() - c).abs() < 1e-5);
        }
        assert!(signed_area(&poly) > 0.0, "expected CCW winding");
        assert!((signed_area(&poly) - 4.0 * c * c).abs() < 1e-4);
    }

    /// A slice past the solid produces no crossings.
    #[test]
    fn slice_beyond_solid_is_empty() {
        let v = Solid3::Cube.vertices();
        let e = Solid3::Cube.edges();
        assert!(convex_section_polygon(&v, &e, ZPlane::new(5.0)).is_empty());
    }

    /// The tetrahedron yields a non-degenerate polygon mid-span.
    #[test]
    fn tetra_midspan_section_has_area() {
        let v = Solid3::Tetrahedron.vertices();
        let e = Solid3::Tetrahedron.edges();
        let s = 1.0 / 3.0_f32.sqrt();
        let poly = convex_section_polygon(&v, &e, ZPlane::new(0.5 * s));
        assert!(poly.len() >= 3);
        assert!(signed_area(&poly).abs() > 1e-4);
    }
}
