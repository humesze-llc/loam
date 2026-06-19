//! Subdivided icosphere: a near-uniform triangulated unit sphere, for a smooth
//! rasterized sphere and a circular cross-section. Each subdivision splits every
//! triangle into four and projects the new midpoints back to the unit sphere, so
//! the count is `10 * 4^n + 2` vertices and `20 * 4^n` faces.

use glam::Vec3;
use std::collections::BTreeMap;

use crate::Solid3;

/// Vertices (on the unit sphere) and CCW-outward triangular faces of an icosphere
/// with `subdivisions` refinement steps. `subdivisions = 0` is the icosahedron.
pub fn icosphere(subdivisions: u32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let mut vertices = Solid3::Icosahedron.vertices();
    let mut faces = Solid3::Icosahedron.faces();

    for _ in 0..subdivisions {
        let mut cache: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        let mut next = Vec::with_capacity(faces.len() * 4);
        for f in &faces {
            let ab = midpoint(f[0], f[1], &mut vertices, &mut cache);
            let bc = midpoint(f[1], f[2], &mut vertices, &mut cache);
            let ca = midpoint(f[2], f[0], &mut vertices, &mut cache);
            next.push([f[0], ab, ca]);
            next.push([f[1], bc, ab]);
            next.push([f[2], ca, bc]);
            next.push([ab, bc, ca]);
        }
        faces = next;
    }

    (vertices, faces)
}

fn midpoint(
    a: u32,
    b: u32,
    vertices: &mut Vec<Vec3>,
    cache: &mut BTreeMap<(u32, u32), u32>,
) -> u32 {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(&idx) = cache.get(&key) {
        return idx;
    }
    let m = ((vertices[a as usize] + vertices[b as usize]) * 0.5).normalize();
    let idx = vertices.len() as u32;
    vertices.push(m);
    cache.insert(key, idx);
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_follow_subdivision_formula() {
        for n in 0..=3 {
            let (v, f) = icosphere(n);
            assert_eq!(v.len(), 10 * 4usize.pow(n) + 2, "vertices at n={n}");
            assert_eq!(f.len(), 20 * 4usize.pow(n), "faces at n={n}");
        }
    }

    #[test]
    fn vertices_on_unit_sphere() {
        let (v, _) = icosphere(2);
        for p in v {
            assert!((p.length() - 1.0).abs() < 1e-5);
        }
    }

    /// V - E + F = 2 with edges deduped from the faces (every face is a triangle,
    /// so face edges are the real edges).
    #[test]
    fn euler_characteristic() {
        let (v, f) = icosphere(2);
        let mut edges = std::collections::BTreeSet::new();
        for t in &f {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                edges.insert(if a < b { (a, b) } else { (b, a) });
            }
        }
        assert_eq!(v.len() as i32 - edges.len() as i32 + f.len() as i32, 2);
    }
}
