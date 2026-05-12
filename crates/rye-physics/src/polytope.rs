//! Topology of the six convex regular 4-polytopes: vertex / edge / cell incidence
//! data, addressable by a [`Polytope4`] enum that mirrors the renderer's `SHAPE_*`
//! constants.
//!
//! ## Why this is a separate module from `euclidean_r4`
//!
//! `euclidean_r4` already exposes vertex generators
//! ([`crate::euclidean_r4::pentatope_vertices`] etc.) that take a circumradius
//! parameter and return a heap-allocated `Vec<Vec4>`. Those are useful for
//! constructing physics bodies at arbitrary scale. This module wraps them into
//! cached, unit-circumradius `&'static` slices, alongside derived edge and cell
//! incidence data the visualization layer needs.
//!
//! ## What lives here vs not here
//!
//! - **Here**: combinatorial topology (which vertices are connected to which
//!   edges, which vertices form each cell), in canonical (unit-circumradius)
//!   coordinates. Read-only after first access. Pure CPU data; no GPU.
//! - **Not here**: the SDF / WGSL kernel data in
//!   `rye_render::raymarch::polytope_data` (face normals + vertex tables in the
//!   shader). That stays where the raymarcher needs it. The data here is for
//!   downstream renderers (line rasterizer for wireframes, future triangle
//!   rasterizer for hyperslice meshes) and CPU-side analysis.
//!
//! ## Phase status
//!
//! Phase 1 (this commit): vertex tables for all six polytopes. Edges and cells
//! land in follow-on commits; the corresponding fields are currently empty
//! slices and the f-vector accessors return 0 for them.
//!
//! See `docs/devlog/POLYTOPE_TOPOLOGY.md` (gitignored) for the full spec.
//!
//! ## Example
//!
//! ```
//! use rye_physics::polytope::Polytope4;
//!
//! let topo = Polytope4::Tesseract.topology();
//! assert_eq!(topo.vertices.len(), 16);
//! // Each vertex is on the unit 3-sphere.
//! for v in topo.vertices {
//!     assert!((v.length() - 1.0).abs() < 1e-5);
//! }
//! ```
use std::sync::LazyLock;

use glam::Vec4;

use crate::euclidean_r4::{
    cell120_vertices, cell16_vertices, cell24_vertices, cell600_vertices, pentatope_vertices,
    tesseract_vertices,
};

/// One of the six convex regular 4-polytopes. Discriminants match the
/// `rye_render::raymarch::SHAPE_*` constants used by the kernel so the same
/// `u32` can drive both the renderer and the topology lookup.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Polytope4 {
    /// 5-cell / pentatope / 4-simplex.
    Pentatope = 0,
    /// 8-cell / tesseract / hypercube.
    Tesseract = 1,
    /// 16-cell / hexadecachoron / 4-orthoplex.
    Cell16 = 2,
    /// 24-cell / icositetrachoron. Unique to 4D.
    Cell24 = 3,
    /// 120-cell / hecatonicosachoron. The 4D analogue of the dodecahedron.
    Cell120 = 4,
    /// 600-cell / hexacosichoron. The 4D analogue of the icosahedron.
    Cell600 = 5,
}

/// Full topology of a 4-polytope in canonical (unit-circumradius) coordinates.
///
/// Static lifetimes throughout: the data is allocated once per polytope on first
/// access via [`std::sync::LazyLock`] and never mutated. Cheap to copy the
/// reference around.
pub struct Polytope4Topology {
    /// All vertices, in canonical (unit-circumradius) coordinates. Vertex
    /// indices used by `edges` and `cells` are positions into this slice.
    pub vertices: &'static [Vec4],
    /// Edges as pairs of vertex indices. Vertex order within a pair is
    /// arbitrary (the edge is undirected). Empty until Phase 2 lands.
    pub edges: &'static [[u32; 2]],
    /// Cells as variable-length vertex-index lists. Each inner slice is one
    /// 3-cell's vertices (e.g., 4 indices for a tetrahedral cell, 8 for a
    /// cubic cell, 20 for a dodecahedral cell). Vertex order within a cell
    /// is arbitrary. Empty until Phase 3 lands.
    pub cells: &'static [&'static [u32]],
}

impl Polytope4 {
    /// Borrow this polytope's full topology. First access lazily computes the
    /// vertex / edge / cell tables and caches them for the rest of process
    /// lifetime; subsequent calls are a pointer dereference.
    pub fn topology(self) -> &'static Polytope4Topology {
        match self {
            Polytope4::Pentatope => &PENTATOPE_TOPOLOGY,
            Polytope4::Tesseract => &TESSERACT_TOPOLOGY,
            Polytope4::Cell16 => &CELL16_TOPOLOGY,
            Polytope4::Cell24 => &CELL24_TOPOLOGY,
            Polytope4::Cell120 => &CELL120_TOPOLOGY,
            Polytope4::Cell600 => &CELL600_TOPOLOGY,
        }
    }

    pub fn vertex_count(self) -> usize {
        self.topology().vertices.len()
    }

    pub fn edge_count(self) -> usize {
        self.topology().edges.len()
    }

    pub fn cell_count(self) -> usize {
        self.topology().cells.len()
    }
}

// ---------------------------------------------------------------------------
// Per-polytope vertex caches
// ---------------------------------------------------------------------------
//
// Each `LazyLock<Vec<Vec4>>` holds the unit-circumradius vertex set, computed
// once via `euclidean_r4`'s existing generators. The Topology struct borrows
// these via `as_slice()` (leaked through `Box::leak` to get a `'static` slice;
// they're never freed, which matches the LazyLock's semantics of "alive for
// the whole process").

static PENTATOPE_VERTICES: LazyLock<&'static [Vec4]> =
    LazyLock::new(|| Box::leak(pentatope_vertices(1.0).into_boxed_slice()));
static TESSERACT_VERTICES: LazyLock<&'static [Vec4]> =
    LazyLock::new(|| Box::leak(tesseract_vertices(1.0).into_boxed_slice()));
static CELL16_VERTICES: LazyLock<&'static [Vec4]> =
    LazyLock::new(|| Box::leak(cell16_vertices(1.0).into_boxed_slice()));
static CELL24_VERTICES: LazyLock<&'static [Vec4]> =
    LazyLock::new(|| Box::leak(cell24_vertices(1.0).into_boxed_slice()));
static CELL120_VERTICES: LazyLock<&'static [Vec4]> =
    LazyLock::new(|| Box::leak(cell120_vertices(1.0).into_boxed_slice()));
static CELL600_VERTICES: LazyLock<&'static [Vec4]> =
    LazyLock::new(|| Box::leak(cell600_vertices(1.0).into_boxed_slice()));

// ---------------------------------------------------------------------------
// Per-polytope topology assemblies
// ---------------------------------------------------------------------------
//
// LazyLock-of-struct so the topology itself is constructed once. Inner fields
// borrow from the per-polytope vertex / edge / cell caches above; both layers
// of `LazyLock` ensure the data outlives any caller.

const EMPTY_EDGES: &[[u32; 2]] = &[];
const EMPTY_CELLS: &[&[u32]] = &[];

static PENTATOPE_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *PENTATOPE_VERTICES,
    edges: EMPTY_EDGES,
    cells: EMPTY_CELLS,
});
static TESSERACT_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *TESSERACT_VERTICES,
    edges: EMPTY_EDGES,
    cells: EMPTY_CELLS,
});
static CELL16_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *CELL16_VERTICES,
    edges: EMPTY_EDGES,
    cells: EMPTY_CELLS,
});
static CELL24_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *CELL24_VERTICES,
    edges: EMPTY_EDGES,
    cells: EMPTY_CELLS,
});
static CELL120_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *CELL120_VERTICES,
    edges: EMPTY_EDGES,
    cells: EMPTY_CELLS,
});
static CELL600_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *CELL600_VERTICES,
    edges: EMPTY_EDGES,
    cells: EMPTY_CELLS,
});

#[cfg(test)]
mod tests {
    use super::*;

    /// All six polytopes have the f-vector vertex counts listed in any
    /// reference (e.g., Coxeter's *Regular Polytopes*, table I).
    #[test]
    fn vertex_counts_match_f_vector() {
        assert_eq!(Polytope4::Pentatope.vertex_count(), 5);
        assert_eq!(Polytope4::Tesseract.vertex_count(), 16);
        assert_eq!(Polytope4::Cell16.vertex_count(), 8);
        assert_eq!(Polytope4::Cell24.vertex_count(), 24);
        assert_eq!(Polytope4::Cell120.vertex_count(), 600);
        assert_eq!(Polytope4::Cell600.vertex_count(), 120);
    }

    /// Every vertex sits on the unit 3-sphere (circumradius = 1). Tolerance is
    /// loose enough to absorb f32 accumulation from sqrt + division in the
    /// `*_vertices` generators; tight enough to catch a unit-scale bug
    /// (factor-of-2 errors would blow past).
    #[test]
    fn vertices_on_unit_circumradius() {
        for p in [
            Polytope4::Pentatope,
            Polytope4::Tesseract,
            Polytope4::Cell16,
            Polytope4::Cell24,
            Polytope4::Cell120,
            Polytope4::Cell600,
        ] {
            for v in p.topology().vertices {
                let r = v.length();
                assert!(
                    (r - 1.0).abs() < 1e-5,
                    "{p:?} vertex {v:?} has |v| = {r}, expected 1.0"
                );
            }
        }
    }

    /// Discriminants match the `rye_render::raymarch::SHAPE_*` constants the
    /// kernel uses, so the same `u32` is interchangeable between renderer and
    /// topology lookup. (Hard-coded here since `rye_render` isn't a dep of
    /// `rye-physics`; if the renderer's table changes, this assertion needs
    /// updating to match.)
    #[test]
    fn discriminants_match_renderer_shape_constants() {
        assert_eq!(Polytope4::Pentatope as u32, 0);
        assert_eq!(Polytope4::Tesseract as u32, 1);
        assert_eq!(Polytope4::Cell16 as u32, 2);
        assert_eq!(Polytope4::Cell24 as u32, 3);
        assert_eq!(Polytope4::Cell120 as u32, 4);
        assert_eq!(Polytope4::Cell600 as u32, 5);
    }

    /// Edge / cell tables are empty until Phase 2 / Phase 3 land.
    #[test]
    fn phase_1_edges_and_cells_are_empty_placeholders() {
        for p in [
            Polytope4::Pentatope,
            Polytope4::Tesseract,
            Polytope4::Cell16,
            Polytope4::Cell24,
            Polytope4::Cell120,
            Polytope4::Cell600,
        ] {
            assert_eq!(p.edge_count(), 0);
            assert_eq!(p.cell_count(), 0);
        }
    }
}
