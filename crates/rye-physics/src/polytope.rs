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
//! Phase 1: vertex tables for all six polytopes.
//! Phase 2 (this commit): edge tables, derived from all-pairs vertex distance
//! against the canonical edge length per polytope.
//! Phase 3 (later): cell tables.
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
    /// arbitrary (the edge is undirected). The pairs themselves are listed in
    /// `(min, max)` index order and sorted lexicographically so the iteration
    /// order is deterministic.
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
// Per-polytope edge caches
// ---------------------------------------------------------------------------

/// Canonical edge length at unit circumradius. Sourced from Coxeter,
/// *Regular Polytopes*, Table I, and cross-checked against Wikipedia plus the
/// empirical minimum pairwise distance over each polytope's vertex set.
///
/// The 120-cell value warrants explanation: Wikipedia gives the edge length as
/// `3 − √5` at circumradius `2√2` (the "natural" coordinate convention also
/// used by [`crate::euclidean_r4::cell120_vertices`]). Rescaling to unit
/// circumradius divides by `2√2`, giving `(3 − √5)/(2√2) = 1/(φ²·√2)`. A
/// common mistake (and a stale note in early drafts of the topology spec) is
/// to drop the `√2`, which is the cell-edge-length identity for the 600-cell
/// dual but not the 120-cell itself.
fn canonical_edge_length(p: Polytope4) -> f32 {
    let phi = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let sqrt2 = 2.0_f32.sqrt();
    match p {
        Polytope4::Pentatope => (5.0_f32 / 2.0).sqrt(),
        Polytope4::Tesseract => 1.0,
        Polytope4::Cell16 => sqrt2,
        Polytope4::Cell24 => 1.0,
        Polytope4::Cell120 => 1.0 / (phi * phi * sqrt2),
        Polytope4::Cell600 => 1.0 / phi,
    }
}

/// All-pairs distance check. A pair `(i, j)` with `i < j` forms an edge iff
/// `|v_i − v_j|` matches the canonical edge length within `EDGE_TOLERANCE`.
///
/// Tolerance set empirically: needs to absorb f32 accumulation across `Vec4`
/// subtraction + dot + sqrt (~1e-5 at unit scale), and be much smaller than the
/// gap to the next-shortest inter-vertex chord. The tightest gap is on the
/// 120-cell (edge 0.382, next-shortest chord ~0.627), so 1e-4 leaves a 4× gap
/// either side.
const EDGE_TOLERANCE: f32 = 1e-4;

fn derive_edges(vertices: &[Vec4], edge_length: f32) -> Vec<[u32; 2]> {
    let mut edges = Vec::new();
    for i in 0..vertices.len() {
        for j in (i + 1)..vertices.len() {
            let d = (vertices[i] - vertices[j]).length();
            if (d - edge_length).abs() < EDGE_TOLERANCE {
                edges.push([i as u32, j as u32]);
            }
        }
    }
    edges
}

/// `cache_edges` is parameterised on the vertex slice rather than the
/// `Polytope4` enum to avoid a deadlock: the `EDGES` LazyLock would otherwise
/// recursively read `TOPOLOGY` (which it transitively initialises), or vice
/// versa. Taking `vertices` directly breaks the cycle.
fn cache_edges(vertices: &'static [Vec4], edge_length: f32) -> &'static [[u32; 2]] {
    Box::leak(derive_edges(vertices, edge_length).into_boxed_slice())
}

static PENTATOPE_EDGES: LazyLock<&'static [[u32; 2]]> = LazyLock::new(|| {
    cache_edges(
        *PENTATOPE_VERTICES,
        canonical_edge_length(Polytope4::Pentatope),
    )
});
static TESSERACT_EDGES: LazyLock<&'static [[u32; 2]]> = LazyLock::new(|| {
    cache_edges(
        *TESSERACT_VERTICES,
        canonical_edge_length(Polytope4::Tesseract),
    )
});
static CELL16_EDGES: LazyLock<&'static [[u32; 2]]> =
    LazyLock::new(|| cache_edges(*CELL16_VERTICES, canonical_edge_length(Polytope4::Cell16)));
static CELL24_EDGES: LazyLock<&'static [[u32; 2]]> =
    LazyLock::new(|| cache_edges(*CELL24_VERTICES, canonical_edge_length(Polytope4::Cell24)));
static CELL120_EDGES: LazyLock<&'static [[u32; 2]]> =
    LazyLock::new(|| cache_edges(*CELL120_VERTICES, canonical_edge_length(Polytope4::Cell120)));
static CELL600_EDGES: LazyLock<&'static [[u32; 2]]> =
    LazyLock::new(|| cache_edges(*CELL600_VERTICES, canonical_edge_length(Polytope4::Cell600)));

// ---------------------------------------------------------------------------
// Per-polytope topology assemblies
// ---------------------------------------------------------------------------
//
// LazyLock-of-struct so the topology itself is constructed once. Inner fields
// borrow from the per-polytope vertex / edge / cell caches above; both layers
// of `LazyLock` ensure the data outlives any caller.

const EMPTY_CELLS: &[&[u32]] = &[];

static PENTATOPE_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *PENTATOPE_VERTICES,
    edges: *PENTATOPE_EDGES,
    cells: EMPTY_CELLS,
});
static TESSERACT_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *TESSERACT_VERTICES,
    edges: *TESSERACT_EDGES,
    cells: EMPTY_CELLS,
});
static CELL16_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *CELL16_VERTICES,
    edges: *CELL16_EDGES,
    cells: EMPTY_CELLS,
});
static CELL24_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *CELL24_VERTICES,
    edges: *CELL24_EDGES,
    cells: EMPTY_CELLS,
});
static CELL120_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *CELL120_VERTICES,
    edges: *CELL120_EDGES,
    cells: EMPTY_CELLS,
});
static CELL600_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *CELL600_VERTICES,
    edges: *CELL600_EDGES,
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

    /// Edge counts per f-vector (Coxeter Table I).
    #[test]
    fn edge_counts_match_f_vector() {
        assert_eq!(Polytope4::Pentatope.edge_count(), 10);
        assert_eq!(Polytope4::Tesseract.edge_count(), 32);
        assert_eq!(Polytope4::Cell16.edge_count(), 24);
        assert_eq!(Polytope4::Cell24.edge_count(), 96);
        assert_eq!(Polytope4::Cell120.edge_count(), 1200);
        assert_eq!(Polytope4::Cell600.edge_count(), 720);
    }

    /// Every edge has length matching the canonical edge length within the same
    /// tolerance used during derivation. (Trivially true by construction, but
    /// the test catches a misuse where someone hand-edits an edge list and
    /// forgets to check.)
    #[test]
    fn edge_lengths_match_canonical() {
        for p in [
            Polytope4::Pentatope,
            Polytope4::Tesseract,
            Polytope4::Cell16,
            Polytope4::Cell24,
            Polytope4::Cell120,
            Polytope4::Cell600,
        ] {
            let expected = canonical_edge_length(p);
            let t = p.topology();
            for &[i, j] in t.edges {
                let d = (t.vertices[i as usize] - t.vertices[j as usize]).length();
                assert!(
                    (d - expected).abs() < EDGE_TOLERANCE,
                    "{p:?} edge ({i}, {j}) length = {d}, expected {expected}"
                );
            }
        }
    }

    /// Each edge index pair is in `(min, max)` order (the construction
    /// guarantees this since the inner loop runs `j > i`).
    #[test]
    fn edge_pairs_in_min_max_order() {
        for p in [
            Polytope4::Pentatope,
            Polytope4::Tesseract,
            Polytope4::Cell16,
            Polytope4::Cell24,
            Polytope4::Cell120,
            Polytope4::Cell600,
        ] {
            for &[i, j] in p.topology().edges {
                assert!(i < j, "{p:?} edge ({i}, {j}) not in (min, max) order");
            }
        }
    }

    /// No duplicate edges, and `(j, i)` never appears as a separate entry from
    /// `(i, j)`. Catches accidental double-insertion in the derivation.
    #[test]
    fn edges_are_unique() {
        for p in [
            Polytope4::Pentatope,
            Polytope4::Tesseract,
            Polytope4::Cell16,
            Polytope4::Cell24,
            Polytope4::Cell120,
            Polytope4::Cell600,
        ] {
            let edges = p.topology().edges;
            let mut seen = std::collections::HashSet::new();
            for &[i, j] in edges {
                let key = (i.min(j), i.max(j));
                assert!(seen.insert(key), "{p:?} edge ({i}, {j}) duplicated");
            }
        }
    }

    /// The canonical edge length equals the actual minimum pairwise distance
    /// among the vertices, within tolerance. Catches drift between the
    /// theoretical value (used to derive edges) and the vertex set produced by
    /// `euclidean_r4`. If `cell120_vertices` is rescaled or the literature
    /// convention shifts, this test fires before any silent edge-set breakage.
    #[test]
    fn canonical_edge_length_matches_empirical_min() {
        for p in [
            Polytope4::Pentatope,
            Polytope4::Tesseract,
            Polytope4::Cell16,
            Polytope4::Cell24,
            Polytope4::Cell120,
            Polytope4::Cell600,
        ] {
            let vs = p.topology().vertices;
            let mut min_d = f32::INFINITY;
            for i in 0..vs.len() {
                for j in (i + 1)..vs.len() {
                    let d = (vs[i] - vs[j]).length();
                    if d < min_d {
                        min_d = d;
                    }
                }
            }
            let expected = canonical_edge_length(p);
            assert!(
                (min_d - expected).abs() < EDGE_TOLERANCE,
                "{p:?}: empirical min pairwise distance {min_d} != canonical {expected}"
            );
        }
    }

    /// Cell tables stay empty until Phase 3 lands.
    #[test]
    fn phase_3_cells_are_empty_placeholders() {
        for p in [
            Polytope4::Pentatope,
            Polytope4::Tesseract,
            Polytope4::Cell16,
            Polytope4::Cell24,
            Polytope4::Cell120,
            Polytope4::Cell600,
        ] {
            assert_eq!(p.cell_count(), 0);
        }
    }
}
