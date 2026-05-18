//! [`Visualizable`] trait + mesh data types for the rasterization tier.
//!
//! Symmetric to the [`Primitive`](../../rye_scene/index.html) (SDF role) and
//! [`Collider`](../../rye_physics/index.html) (physics role) traits in their respective
//! consumer crates: this trait carries the rasterization role. Implementations live downstream
//! (`rye-scene` impls it for [`crate::Shape`] variants; `rye-physics` impls it for
//! [`rye_physics::polytope::Polytope4`]).
//!
//! ## Why this trait + the mesh types live in `rye-shape`
//!
//! `rye-shape`'s charter is "data only, no behavior, dep-graph leaf." Defining a trait here
//! would seem to violate that. The resolution: trait *definitions* are data-shape interfaces,
//! not behavior. The trait says "if you can produce a `LineMesh<N>`, here is the function
//! signature to do so" without any associated logic. Impls live in role crates (`rye-scene`,
//! `rye-physics`) where they belong, alongside their respective behavior traits.
//!
//! The mesh types ([`LineMesh<N>`], [`TriangleMesh<N>`], [`PointMesh<N>`]) are pure data:
//! arrays of points + colors + sizes. They're returned from the trait method, so they have to
//! live in a crate both impl sites can see, which means here.
//!
//! ## Const-generic dim
//!
//! `N` is the ambient dimension: 2 for R², 3 for R³, 4 for R⁴, etc. Const-generic so dimension
//! mismatches are compile-time errors and vertex storage is stack-friendly (`[f32; N]`, not
//! `Vec<f32>`). The viral type parameter is contained by:
//!
//! - Scene-level wrappers in `rye-scene` use an enum (`SceneNode::Lines3 | Lines4 | ...`) so
//!   downstream code only sees `RasterMesh` (enum), not generic types.
//! - The rasterizer pipelines in `rye-render` take the mesh type as a generic argument; only
//!   the upload path has to spell out `N`.
//!
//! ## Color convention
//!
//! All colors are RGBA linear-space `[f32; 4]`. Linear (not sRGB) because the fragment shader
//! interpolates and combines colors in linear space; sRGB conversion happens at the output
//! attachment via wgpu's swapchain format. RGBA (not RGB) because alpha is essential for AA
//! coverage at silhouettes.

use serde::{Deserialize, Serialize};

/// Why a shape cannot produce a particular mesh representation. Returned by [`Visualizable`]
/// methods so callers can distinguish "skip this primitive" from "your input is wrong."
///
/// Callers that just want to filter can `.ok()` to drop the reason and treat the result as an
/// [`Option`]. Callers that want diagnostics get a concrete variant they can pattern-match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotVisualizable {
    /// Shape extends to infinity in some direction (e.g., [`Shape::HalfSpace`], an infinite
    /// plane). No bounded mesh representation exists; the editor / debug renderer may still
    /// draw a "ghost" via a different mechanism (clipped at the view frustum, say).
    Unbounded,

    /// Shape's natural dimension doesn't match the requested `N`. For example, asking for
    /// `Visualizable<3>` output on a [`Shape::HyperSphere4D`] (which is intrinsically 4D)
    /// returns this variant. The caller should either pick a different `N` or project first.
    WrongDimension,

    /// Shape's parameters are degenerate: zero radius, empty vertex list, collinear polytope
    /// vertices. Not a bug, just nothing to draw.
    Degenerate,
}

/// Anything that can emit rasterizable geometry in N-dimensional space.
///
/// Three orthogonal output flavors:
/// - [`to_lines`](Self::to_lines): wireframe edges. Most common; works for polytopes,
///   parametric grids on smooth shapes, debug gizmos.
/// - [`to_triangles`](Self::to_triangles): filled surfaces. For polytopes this means 2-face
///   triangulation; for smooth shapes it's parametric sampling. Optional; many shapes return
///   [`NotVisualizable::Unbounded`] or skip it.
/// - [`to_points`](Self::to_points): vertex markers. For polytopes the literal vertex set; for
///   smooth shapes a sampled set (e.g., sphere poles, torus parameter origin).
///
/// Implementations are in the role crates that own the shape data (`rye-scene` for
/// [`crate::Shape`], `rye-physics` for `Polytope4`).
pub trait Visualizable<const N: usize> {
    /// Emit the shape as line segments in RN.
    fn to_lines(&self) -> Result<LineMesh<N>, NotVisualizable>;

    /// Emit the shape as indexed triangles in RN.
    fn to_triangles(&self) -> Result<TriangleMesh<N>, NotVisualizable>;

    /// Emit the shape as point markers in RN.
    fn to_points(&self) -> Result<PointMesh<N>, NotVisualizable>;
}

/// Line segments in RN. One entry per segment in [`segments`](Self::segments) /
/// [`colors`](Self::colors) / [`widths`](Self::widths); array lengths must match.
///
/// Per-segment width is scalar (constant along the segment). Varying-width lines are
/// expressible as multiple segments at the change points; per-segment scalar keeps the GPU
/// instance-buffer layout simple. Per-endpoint color allows gradient edges (depth-fade,
/// cell-membership coloring, hover-highlight transitions).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(bound(
    serialize = "[f32; N]: Serialize",
    deserialize = "[f32; N]: Deserialize<'de>"
))]
pub struct LineMesh<const N: usize> {
    /// `(start, end)` pairs. Each endpoint is a fixed-size array of `N` floats.
    pub segments: Vec<([f32; N], [f32; N])>,
    /// `(start_color, end_color)` per segment, RGBA in linear space. `colors.len() ==
    /// segments.len()`. Gradient interpolation is done in linear space by the fragment shader.
    pub colors: Vec<([f32; 4], [f32; 4])>,
    /// Width per segment in pixels. `widths.len() == segments.len()`.
    pub widths: Vec<f32>,
}

/// Filled triangles in RN. Vertices indexed by [`indices`](Self::indices); per-vertex color.
///
/// Per-vertex normals are deliberately omitted at v1. Lighting an R⁴ triangle has no standard
/// convention; v1 deliverables (Schlegel face fills, slice cross-sections) only need
/// flat-shaded or color-only triangles. Add a `normals` field if a real consumer asks for it.
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

/// Point markers in RN, instanced as sprite quads by the rasterizer.
///
/// Sizes are screen-space pixel radii. The GPU sprite is centered at the projected screen
/// position and expanded perpendicular to the view direction; the fragment shader applies a
/// radial smoothstep for antialiased disc rendering.
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

    /// [`LineMesh<3>`] round-trips through RON serialization. Pins the const-generic-with-serde
    /// behavior we rely on for scene file persistence.
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

    /// Default-constructed meshes have empty buffers, matching the `Vec::default()` semantics
    /// the `Default` derive expands to. Used by callers that build meshes incrementally
    /// (push segments / triangles in a loop) instead of constructing the struct literal.
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
