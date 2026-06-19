//! Regular convex solids in R3 as rasterizable + sectionable mesh data:
//! vertices on the unit circumradius sphere, CCW-outward triangular faces, and
//! edges derived by minimal edge length. The R3 counterpart to the `Polytope4`
//! mesh data, used by the Flatland demo's cross-section views and any 3D
//! rasterized scene. `Visualizable<3>` is impl'd here, not in a downstream role
//! crate, because these are standalone primitive data with no SDF/physics role.

use glam::Vec3;

use crate::visualizable::{LineMesh, NotVisualizable, PointMesh, TriangleMesh, Visualizable};

/// A regular convex solid in R3, vertices on the unit circumradius sphere.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Solid3 {
    Tetrahedron,
    Cube,
    Icosahedron,
}

impl Solid3 {
    /// Vertices on the unit circumradius sphere (`|v| = 1`).
    pub fn vertices(self) -> Vec<Vec3> {
        match self {
            Solid3::Tetrahedron => {
                let s = 1.0 / 3.0_f32.sqrt();
                vec![
                    Vec3::new(s, s, s),
                    Vec3::new(s, -s, -s),
                    Vec3::new(-s, s, -s),
                    Vec3::new(-s, -s, s),
                ]
            }
            Solid3::Cube => {
                let c = 1.0 / 3.0_f32.sqrt();
                let mut v = Vec::with_capacity(8);
                for z in [-c, c] {
                    for y in [-c, c] {
                        for x in [-c, c] {
                            v.push(Vec3::new(x, y, z));
                        }
                    }
                }
                v
            }
            Solid3::Icosahedron => {
                let p = (1.0 + 5.0_f32.sqrt()) / 2.0;
                let n = 1.0 / (1.0 + p * p).sqrt();
                let v = |a: f32, b: f32, c: f32| Vec3::new(a, b, c) * n;
                vec![
                    v(0.0, 1.0, p),
                    v(0.0, 1.0, -p),
                    v(0.0, -1.0, p),
                    v(0.0, -1.0, -p),
                    v(1.0, p, 0.0),
                    v(1.0, -p, 0.0),
                    v(-1.0, p, 0.0),
                    v(-1.0, -p, 0.0),
                    v(p, 0.0, 1.0),
                    v(p, 0.0, -1.0),
                    v(-p, 0.0, 1.0),
                    v(-p, 0.0, -1.0),
                ]
            }
        }
    }

    /// Triangular faces wound CCW seen from outside (normal `(b-a) x (c-a)` points
    /// away from the centroid). Pinned by `faces_wind_outward`.
    pub fn faces(self) -> Vec<[u32; 3]> {
        match self {
            Solid3::Tetrahedron | Solid3::Icosahedron => deltahedron_faces(&self.vertices()),
            Solid3::Cube => vec![
                [0, 3, 1],
                [0, 2, 3],
                [4, 7, 6],
                [4, 5, 7],
                [0, 5, 4],
                [0, 1, 5],
                [2, 7, 3],
                [2, 6, 7],
                [0, 6, 2],
                [0, 4, 6],
                [1, 7, 5],
                [1, 3, 7],
            ],
        }
    }

    /// Real geometric edges: vertex pairs at the minimal pairwise distance,
    /// excluding the triangulation's face diagonals (a cube has 12 edges, not 18),
    /// which is what the cross-section and wireframe need.
    pub fn edges(self) -> Vec<[u32; 2]> {
        let v = self.vertices();
        let mut min_d2 = f32::INFINITY;
        for i in 0..v.len() {
            for j in (i + 1)..v.len() {
                min_d2 = min_d2.min(v[i].distance_squared(v[j]));
            }
        }
        let tol = min_d2 * 1e-3;
        let mut edges = Vec::new();
        for i in 0..v.len() {
            for j in (i + 1)..v.len() {
                if (v[i].distance_squared(v[j]) - min_d2).abs() <= tol {
                    edges.push([i as u32, j as u32]);
                }
            }
        }
        edges
    }
}

/// Faces of a convex deltahedron: each triple of mutually-edge-adjacent vertices,
/// wound outward. Valid only where every face is an equilateral triangle.
fn deltahedron_faces(v: &[Vec3]) -> Vec<[u32; 3]> {
    let mut min_d2 = f32::INFINITY;
    for i in 0..v.len() {
        for j in (i + 1)..v.len() {
            min_d2 = min_d2.min(v[i].distance_squared(v[j]));
        }
    }
    let tol = min_d2 * 1e-3;
    let is_edge = |a: usize, b: usize| (v[a].distance_squared(v[b]) - min_d2).abs() <= tol;
    let mut faces = Vec::new();
    for i in 0..v.len() {
        for j in (i + 1)..v.len() {
            if !is_edge(i, j) {
                continue;
            }
            for k in (j + 1)..v.len() {
                if is_edge(i, k) && is_edge(j, k) {
                    let (a, b, c) = (v[i], v[j], v[k]);
                    let outward = (b - a).cross(c - a).dot((a + b + c) / 3.0) >= 0.0;
                    faces.push(if outward {
                        [i as u32, j as u32, k as u32]
                    } else {
                        [i as u32, k as u32, j as u32]
                    });
                }
            }
        }
    }
    faces
}

impl Visualizable<3> for Solid3 {
    fn to_lines(&self) -> Result<LineMesh<3>, NotVisualizable> {
        let v = self.vertices();
        let segments: Vec<_> = self
            .edges()
            .iter()
            .map(|e| (v[e[0] as usize].to_array(), v[e[1] as usize].to_array()))
            .collect();
        let white = [1.0, 1.0, 1.0, 1.0];
        let n = segments.len();
        Ok(LineMesh {
            segments,
            colors: vec![(white, white); n],
            widths: vec![1.5; n],
        })
    }

    fn to_triangles(&self) -> Result<TriangleMesh<3>, NotVisualizable> {
        let vertices: Vec<_> = self.vertices().iter().map(|p| p.to_array()).collect();
        let colors = vec![[0.70, 0.72, 0.78, 1.0]; vertices.len()];
        Ok(TriangleMesh {
            vertices,
            indices: self.faces(),
            colors,
        })
    }

    fn to_points(&self) -> Result<PointMesh<3>, NotVisualizable> {
        let positions: Vec<_> = self.vertices().iter().map(|p| p.to_array()).collect();
        let n = positions.len();
        Ok(PointMesh {
            positions,
            colors: vec![[1.0, 1.0, 1.0, 1.0]; n],
            sizes: vec![6.0; n],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Solid3; 3] = [Solid3::Tetrahedron, Solid3::Cube, Solid3::Icosahedron];

    #[test]
    fn vertices_on_unit_circumradius() {
        for s in ALL {
            for v in s.vertices() {
                assert!(
                    (v.length() - 1.0).abs() < 1e-6,
                    "{s:?} vertex off unit sphere"
                );
            }
        }
    }

    #[test]
    fn vertex_and_edge_counts() {
        assert_eq!(
            (
                Solid3::Tetrahedron.vertices().len(),
                Solid3::Tetrahedron.edges().len()
            ),
            (4, 6)
        );
        assert_eq!(
            (Solid3::Cube.vertices().len(), Solid3::Cube.edges().len()),
            (8, 12)
        );
        assert_eq!(
            (
                Solid3::Icosahedron.vertices().len(),
                Solid3::Icosahedron.edges().len()
            ),
            (12, 30)
        );
    }

    #[test]
    fn triangle_face_counts() {
        assert_eq!(Solid3::Tetrahedron.faces().len(), 4);
        assert_eq!(Solid3::Cube.faces().len(), 12);
        assert_eq!(Solid3::Icosahedron.faces().len(), 20);
    }

    /// V - E + F = 2 with geometric faces (tetra 4, cube 6, icosa 20).
    #[test]
    fn euler_characteristic() {
        for (s, geom_faces) in [
            (Solid3::Tetrahedron, 4i32),
            (Solid3::Cube, 6),
            (Solid3::Icosahedron, 20),
        ] {
            let v = s.vertices().len() as i32;
            let e = s.edges().len() as i32;
            assert_eq!(v - e + geom_faces, 2, "{s:?}");
        }
    }

    /// Every face winds CCW seen from outside: its normal points away from the
    /// centroid (the origin).
    #[test]
    fn faces_wind_outward() {
        for s in ALL {
            let v = s.vertices();
            for f in s.faces() {
                let a = v[f[0] as usize];
                let b = v[f[1] as usize];
                let c = v[f[2] as usize];
                let normal = (b - a).cross(c - a);
                let centroid = (a + b + c) / 3.0;
                assert!(
                    normal.dot(centroid) > 0.0,
                    "face {f:?} of {s:?} winds inward (normal points at centroid)"
                );
            }
        }
    }
}
