//! [`Visualizable`] trait + mesh data types for the rasterization tier.
//!
//! The rasterization-role counterpart to `Primitive` (SDF) in `loam-scene` and
//! `Collider` (physics) in `loam-physics`. Impls live downstream (`loam-scene` for
//! [`crate::Shape`], `loam-physics` for `Polytope4`).
//!
//! The trait + mesh types live here despite `loam-shape`'s data-only charter: a
//! trait *definition* is a data-shape interface, not behavior, and the mesh types
//! ([`LineMesh<N>`], [`TriangleMesh<N>`], [`PointMesh<N>`]) are pure data both
//! impl sites must see.
//!
//! `N` is the ambient dimension (2/3/4...). Const-generic so dimension mismatches
//! are compile errors and vertex storage is stack-friendly (`[f32; N]`). The viral
//! parameter is contained by `loam-scene`'s `RasterMesh` enum and `loam-render`'s
//! generic upload path.
//!
//! Colors are RGBA linear `[f32; 4]`: linear because the fragment shader
//! interpolates in linear space (sRGB conversion is at the output attachment), and
//! alpha because it carries AA coverage at silhouettes.

use serde::{Deserialize, Serialize};

/// Why a shape cannot produce a particular mesh. Callers that filter can `.ok()`
/// to an [`Option`]; callers that want diagnostics pattern-match the variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotVisualizable {
    /// Shape extends to infinity (e.g. [`crate::Shape::HalfSpace`]); no bounded
    /// mesh exists.
    Unbounded,

    /// Shape's natural dimension doesn't match the requested `N` (e.g.
    /// `Visualizable<3>` on a [`crate::Shape::HyperSphere4D`]). Pick a different
    /// `N` or project first.
    WrongDimension,

    /// Degenerate parameters: zero radius, empty/collinear vertices. Nothing to
    /// draw, not a bug.
    Degenerate,
}

/// Anything that can emit rasterizable geometry in RN. Three orthogonal output
/// flavors: [`to_lines`](Self::to_lines) (wireframe edges),
/// [`to_triangles`](Self::to_triangles) (filled surfaces; often
/// [`NotVisualizable::Unbounded`] for smooth shapes), and
/// [`to_points`](Self::to_points) (vertex markers). Impls live in the role crates
/// that own the data (`loam-scene`, `loam-physics`).
pub trait Visualizable<const N: usize> {
    /// Emit the shape as line segments in RN.
    fn to_lines(&self) -> Result<LineMesh<N>, NotVisualizable>;

    /// Emit the shape as indexed triangles in RN.
    fn to_triangles(&self) -> Result<TriangleMesh<N>, NotVisualizable>;

    /// Emit the shape as point markers in RN.
    fn to_points(&self) -> Result<PointMesh<N>, NotVisualizable>;
}

/// Line segments in RN. [`segments`](Self::segments) / [`colors`](Self::colors) /
/// [`widths`](Self::widths) lengths must match. Width is per-segment scalar
/// (keeps the GPU instance layout simple); color is per-endpoint (gradient edges).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(bound(
    serialize = "[f32; N]: Serialize",
    deserialize = "[f32; N]: Deserialize<'de>"
))]
pub struct LineMesh<const N: usize> {
    /// `(start, end)` endpoint pairs, each an `[f32; N]`.
    pub segments: Vec<([f32; N], [f32; N])>,
    /// `(start_color, end_color)` per segment, RGBA linear; `colors.len() ==
    /// segments.len()`.
    pub colors: Vec<([f32; 4], [f32; 4])>,
    /// Width per segment in pixels. `widths.len() == segments.len()`.
    pub widths: Vec<f32>,
}

/// Filled triangles in RN, indexed by [`indices`](Self::indices), per-vertex
/// color. No normals: lighting an R⁴ triangle has no standard convention and the
/// v1 consumers (Schlegel fills, slice sections) only need color. Add `normals`
/// when a consumer asks.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(bound(
    serialize = "[f32; N]: Serialize",
    deserialize = "[f32; N]: Deserialize<'de>"
))]
pub struct TriangleMesh<const N: usize> {
    /// All vertices, in RN.
    pub vertices: Vec<[f32; N]>,
    /// Three vertex indices per triangle. Counter-clockwise winding (looking down the normal).
    pub indices: Vec<[u32; 3]>,
    /// Per-vertex color, RGBA linear, `colors.len() == vertices.len()`.
    pub colors: Vec<[f32; 4]>,
}

/// Point markers in RN, instanced as sprite quads. [`sizes`](Self::sizes) are
/// screen-space pixel radii; the fragment shader radial-smoothsteps them into
/// antialiased discs.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(bound(
    serialize = "[f32; N]: Serialize",
    deserialize = "[f32; N]: Deserialize<'de>"
))]
pub struct PointMesh<const N: usize> {
    /// Marker centers, in RN.
    pub positions: Vec<[f32; N]>,
    /// Per-point color, RGBA linear, `colors.len() == positions.len()`.
    pub colors: Vec<[f32; 4]>,
    /// Per-point screen-space radius in pixels.
    pub sizes: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`LineMesh<3>`] round-trips through RON; pins the const-generic-with-serde
    /// behavior scene persistence relies on.
    #[test]
    fn line_mesh_3d_ron_round_trip() {
        let original: LineMesh<3> = LineMesh {
            segments: vec![
                ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
                ([0.5, 1.0, 0.0], [0.5, 1.0, 1.0]),
            ],
            colors: vec![
                ([1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]),
                ([0.0, 0.0, 1.0, 0.7], [0.0, 0.0, 1.0, 0.7]),
            ],
            widths: vec![1.5, 2.0],
        };
        let s = ron::ser::to_string(&original).expect("serialize");
        let parsed: LineMesh<3> = ron::de::from_str(&s).expect("deserialize");
        assert_eq!(parsed.segments, original.segments);
        assert_eq!(parsed.colors, original.colors);
        assert_eq!(parsed.widths, original.widths);
    }

    /// Same for [`LineMesh<4>`]. Pins that the const generic doesn't break for higher dims.
    #[test]
    fn line_mesh_4d_ron_round_trip() {
        let original: LineMesh<4> = LineMesh {
            segments: vec![([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0])],
            colors: vec![([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0])],
            widths: vec![1.0],
        };
        let s = ron::ser::to_string(&original).expect("serialize");
        let parsed: LineMesh<4> = ron::de::from_str(&s).expect("deserialize");
        assert_eq!(parsed.segments, original.segments);
    }

    /// [`TriangleMesh<3>`] round-trips and the index buffer is preserved exactly.
    #[test]
    fn triangle_mesh_3d_ron_round_trip() {
        let original: TriangleMesh<3> = TriangleMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            indices: vec![[0, 1, 2], [1, 3, 2]],
            colors: vec![
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
            ],
        };
        let s = ron::ser::to_string(&original).expect("serialize");
        let parsed: TriangleMesh<3> = ron::de::from_str(&s).expect("deserialize");
        assert_eq!(parsed.vertices, original.vertices);
        assert_eq!(parsed.indices, original.indices);
        assert_eq!(parsed.colors, original.colors);
    }

    /// [`PointMesh<4>`] round-trips, sizes preserved.
    #[test]
    fn point_mesh_4d_ron_round_trip() {
        let original: PointMesh<4> = PointMesh {
            positions: vec![[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            colors: vec![[1.0, 1.0, 1.0, 1.0], [1.0, 0.5, 0.5, 1.0]],
            sizes: vec![3.0, 5.0],
        };
        let s = ron::ser::to_string(&original).expect("serialize");
        let parsed: PointMesh<4> = ron::de::from_str(&s).expect("deserialize");
        assert_eq!(parsed.positions, original.positions);
        assert_eq!(parsed.sizes, original.sizes);
    }

    /// Default-constructed meshes have empty buffers, for callers that build them
    /// incrementally rather than via a struct literal.
    #[test]
    fn default_meshes_are_empty() {
        let lm: LineMesh<3> = LineMesh::default();
        assert!(lm.segments.is_empty());
        assert!(lm.colors.is_empty());
        assert!(lm.widths.is_empty());

        let tm: TriangleMesh<4> = TriangleMesh::default();
        assert!(tm.vertices.is_empty());
        assert!(tm.indices.is_empty());
        assert!(tm.colors.is_empty());

        let pm: PointMesh<3> = PointMesh::default();
        assert!(pm.positions.is_empty());
        assert!(pm.sizes.is_empty());
    }
}
