//! Regular convex solids in R3 as rasterizable + sectionable mesh data:
//! vertices on the unit circumradius sphere, CCW-outward triangular faces, and
//! edges derived from the faces. The R3 counterpart to the `Polytope4` mesh
//! data, used by the Flatland demo's cross-section views and any 3D rasterized
//! scene. (`Visualizable<3>` is impl'd here, not in a downstream role crate,
//! because these are standalone primitive data with no SDF/physics role.)

use glam::Vec3;

use crate::visualizable::{LineMesh, NotVisualizable, PointMesh, TriangleMesh, Visualizable};

/// A regular convex solid in R3, vertices on the unit circumradius sphere.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Solid3 {
    Tetrahedron,
    Cube,
}

impl Solid3 {
    /// Vertices on the unit circumradius sphere (`|v| = 1`).
    pub fn vertices(self) -> Vec<Vec3> {
        match self {
            // Two opposite tetrahedra of the cube; circumradius sqrt(3), so /sqrt(3).
            Solid3::Tetrahedron => {
                let s = 1.0 / 3.0_f32.sqrt();
                vec![
                    Vec3::new(s, s, s),
                    Vec3::new(s, -s, -s),
                    Vec3::new(-s, s, -s),
                    Vec3::new(-s, -s, s),
                ]
            }
            // index = (x>0) + 2(y>0) + 4(z>0); circumradius sqrt(3), so /sqrt(3).
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
        }
    }

    /// Triangular faces, vertices wound counter-clockwise seen from outside
    /// (so the face normal `(b-a) x (c-a)` points away from the centroid). Pinned
    /// by `faces_wind_outward`.
    pub fn faces(self) -> Vec<[u32; 3]> {
        match self {
            Solid3::Tetrahedron => vec![[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]],
            Solid3::Cube => vec![
                [0, 3, 1],
                [0, 2, 3], // -z
                [4, 7, 6],
                [4, 5, 7], // +z
                [0, 5, 4],
                [0, 1, 5], // -y
                [2, 7, 3],
                [2, 6, 7], // +y
                [0, 6, 2],
                [0, 4, 6], // -x
                [1, 7, 5],
                [1, 3, 7], // +x
            ],
        }
    }

    /// Real geometric edges: vertex pairs at the minimal pairwise distance. This
    /// excludes the triangulation's face diagonals (a cube has 12 edges, not the
    /// 18 a face-derived set would give), which is what the cross-section and the
    /// wireframe need. Regular solids have a single edge length.
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

    const ALL: [Solid3; 2] = [Solid3::Tetrahedron, Solid3::Cube];

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
    }

    /// V - E + F = 2 with geometric faces (tetra 4, cube 6). Catches a missing
    /// edge or a stray diagonal in the edge derivation.
    #[test]
    fn euler_characteristic() {
        for (s, geom_faces) in [(Solid3::Tetrahedron, 4i32), (Solid3::Cube, 6)] {
            let v = s.vertices().len() as i32;
            let e = s.edges().len() as i32;
            assert_eq!(v - e + geom_faces, 2, "{s:?}");
        }
    }

    /// Every face winds CCW seen from outside: its normal points away from the
    /// centroid (the origin). Catches an authoring sign flip.
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
