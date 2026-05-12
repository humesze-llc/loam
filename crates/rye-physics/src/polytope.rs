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
//! ## How the tables are derived
//!
//! - Vertices come from the [`crate::euclidean_r4`] generators at unit
//!   circumradius, leaked into `&'static` slices on first access.
//! - Edges are derived by all-pairs distance: a vertex pair forms an edge iff
//!   its Euclidean distance matches the canonical edge length within
//!   tolerance.
//! - Cells are derived by local 3-flat fitting against the polytope's own
//!   edge graph. For each vertex and each triple of its edge-neighbors we
//!   compute the 3-flat through `{v_0, n_a, n_b, n_c}` and count polytope
//!   vertices on it; only triples whose 3-flat picks up exactly the cell's
//!   expected vertex count survive. This avoids dependence on an external
//!   "dual" polytope, whose vertex generator may not be rotation-aligned
//!   with this one (see the cell-cache section for details).
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

use glam::{Vec3, Vec4};

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
#[derive(Debug)]
pub struct Polytope4Topology {
    /// All vertices, in canonical (unit-circumradius) coordinates. Vertex
    /// indices used by `edges` and `cells` are positions into this slice.
    pub vertices: &'static [Vec4],
    /// Edges as pairs of vertex indices. Within each pair the lower index
    /// comes first (`pair[0] < pair[1]`). The pairs themselves are sorted
    /// lexicographically by `(pair[0], pair[1])`, so iteration order is
    /// deterministic across runs.
    pub edges: &'static [[u32; 2]],
    /// Cells as variable-length vertex-index lists. Each inner slice is one
    /// 3-cell's vertices: 4 for tetrahedral cells (pentatope, 16-cell,
    /// 600-cell), 8 for cubical (tesseract), 6 for octahedral (24-cell), 20
    /// for dodecahedral (120-cell). Within each cell the vertex indices are
    /// in ascending order, and the cells themselves are sorted
    /// lexicographically by their vertex list, so iteration order is
    /// deterministic across runs.
    pub cells: &'static [&'static [u32]],
}

impl Polytope4 {
    /// All six variants, in `repr(u32)` discriminant order. Useful for
    /// iteration without spelling each variant out at the call site.
    pub const ALL: [Polytope4; 6] = [
        Polytope4::Pentatope,
        Polytope4::Tesseract,
        Polytope4::Cell16,
        Polytope4::Cell24,
        Polytope4::Cell120,
        Polytope4::Cell600,
    ];

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
// Per-polytope cell caches
// ---------------------------------------------------------------------------
//
// Cells are derived by hyperplane fitting against the polytope's *own* edge
// graph, with no reference to an external "dual" polytope. The 600-cell and
// 120-cell vertex generators in [`crate::euclidean_r4`] are not in mutually-
// dual orientation: they share the 24-cell sub-orbit (axes + tesseract
// corners) but the 96 golden-ratio vertices are oriented differently, so the
// 600-cell's golden-ratio vertices are *not* face normals of the 120-cell.
// Fitting from local vertex figures avoids that misalignment entirely.

/// Tolerance for membership of a vertex in a cell's hyperplane (a 3-flat in
/// 4D). f32 noise on `Vec4::dot` is bounded at ~5e-7 absolute, so 1e-4
/// leaves ~100× margin while still rejecting vertices on an adjacent cell's
/// hyperplane (the spread of `n · v` across non-cell vertices is order 0.1
/// even for the 120-cell, where it is tightest).
const CELL_TOLERANCE: f32 = 1e-4;

/// Vertex-count expected per cell, by polytope. Used as a sanity filter when
/// we trial a candidate 3-flat fit: a non-cell 3-flat through 4 of the
/// polytope's points generically contains only those 4 points, while a true
/// cell's 3-flat contains the full cell.
const fn cell_vertex_count(p: Polytope4) -> usize {
    match p {
        Polytope4::Pentatope => 4,
        Polytope4::Tesseract => 8,
        Polytope4::Cell16 => 4,
        Polytope4::Cell24 => 6,
        Polytope4::Cell120 => 20,
        Polytope4::Cell600 => 4,
    }
}

/// 4D cross product: given three vectors `a`, `b`, `c` in 4D, returns a
/// vector orthogonal to all three. By cofactor expansion the `i`-th
/// component is `(-1)^i · det(M_i)`, where `M_i` is the 3×3 minor of the
/// 3×4 matrix `[a; b; c]` with column `i` dropped. Each 3×3 determinant is
/// computed as the standard triple product `row_a · (row_b × row_c)` on
/// the three surviving columns.
fn cross4(a: Vec4, b: Vec4, c: Vec4) -> Vec4 {
    let drop_x = |v: Vec4| Vec3::new(v.y, v.z, v.w);
    let drop_y = |v: Vec4| Vec3::new(v.x, v.z, v.w);
    let drop_z = |v: Vec4| Vec3::new(v.x, v.y, v.w);
    let drop_w = |v: Vec4| Vec3::new(v.x, v.y, v.z);
    Vec4::new(
        drop_x(a).dot(drop_x(b).cross(drop_x(c))),
        -drop_y(a).dot(drop_y(b).cross(drop_y(c))),
        drop_z(a).dot(drop_z(b).cross(drop_z(c))),
        -drop_w(a).dot(drop_w(b).cross(drop_w(c))),
    )
}

/// Adjacency list from the edge table. `adj[i]` lists the indices of every
/// vertex sharing an edge with vertex `i`. Order within each entry is
/// insertion order, which matches the lexicographic edge order: it is
/// deterministic but otherwise meaningless.
fn adjacency(num_vertices: usize, edges: &[[u32; 2]]) -> Vec<Vec<u32>> {
    let mut adj = vec![Vec::new(); num_vertices];
    for &[i, j] in edges {
        adj[i as usize].push(j);
        adj[j as usize].push(i);
    }
    adj
}

/// Minimum length of `cross4(n_a - v_0, n_b - v_0, n_c - v_0)` for which we
/// trust the 3-flat normal. Below this, the three difference vectors are
/// (nearly) linearly dependent and there is no well-defined 3-flat. Distinct
/// from [`CELL_TOLERANCE`]: that one bounds *on-plane membership* in dot-
/// product units, this one bounds *normal magnitude* in 4D vector-length
/// units.
const MIN_CROSS4_LENGTH: f32 = 1e-4;

/// Derive the cell list by local 3-flat fitting.
///
/// Strategy: each cell of a regular 4-polytope is a regular 3-polyhedron
/// whose vertex set lies on a unique 3-flat in 4D. The cells incident to a
/// vertex `v_0` are in bijection with the faces of `v_0`'s vertex figure;
/// each such face corresponds to a subset of `v_0`'s edge-neighbors that
/// span the cell's 3-flat together with `v_0`.
///
/// We don't bother enumerating vertex-figure faces explicitly. Instead, for
/// every vertex `v_0` and every 3-subset of its edge-neighbors, we fit the
/// 3-flat through `{v_0, n_a, n_b, n_c}` via the 4D cross product, count
/// polytope vertices on that flat, and accept the result only if the count
/// matches `cell_vertex_count(p)`. Non-cell triples produce 3-flats that
/// either pick up the wrong number of vertices or coincide with a separate
/// cell already found.
///
/// Returns one `Vec<u32>` per cell with vertex indices in ascending order.
/// The outer `Vec` is ordered lexicographically by cell vertex list, which
/// is deterministic across runs.
///
/// Cost is `O(V · D^3 · V)` for vertex count `V` and vertex-figure valence
/// `D`. The dominant case is the 600-cell at `120 · 220 · 120 ≈ 3.2 M` plane
/// scans. Cells are cached behind a [`LazyLock`], so the cost is paid once
/// per polytope per process; do not call this in a hot loop.
fn derive_cells(vertices: &[Vec4], edges: &[[u32; 2]], cell_size: usize) -> Vec<Vec<u32>> {
    use std::collections::BTreeSet;

    let adj = adjacency(vertices.len(), edges);
    let mut cells_set: BTreeSet<Vec<u32>> = BTreeSet::new();
    for v_idx in 0..vertices.len() {
        let v_0 = vertices[v_idx];
        let neighbors = &adj[v_idx];
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                for k in (j + 1)..neighbors.len() {
                    let n_a = vertices[neighbors[i] as usize];
                    let n_b = vertices[neighbors[j] as usize];
                    let n_c = vertices[neighbors[k] as usize];
                    let normal = cross4(n_a - v_0, n_b - v_0, n_c - v_0);
                    let mag = normal.length();
                    if mag < MIN_CROSS4_LENGTH {
                        continue;
                    }
                    let n = normal / mag;
                    let offset = v_0.dot(n);
                    let on_plane: Vec<u32> = (0..vertices.len() as u32)
                        .filter(|&p| (vertices[p as usize].dot(n) - offset).abs() < CELL_TOLERANCE)
                        .collect();
                    if on_plane.len() == cell_size {
                        cells_set.insert(on_plane);
                    }
                }
            }
        }
    }
    cells_set.into_iter().collect()
}

/// Materialise the cell incidence list as a leaked, two-level `&'static`
/// slice. The outer slice has one entry per cell; the inner slices hold the
/// vertex indices that belong to that cell.
///
/// Like [`cache_edges`], this takes the data slices directly rather than a
/// [`Polytope4`] so it cannot recursively re-enter the same [`LazyLock`]
/// during init.
fn cache_cells(
    vertices: &'static [Vec4],
    edges: &'static [[u32; 2]],
    cell_size: usize,
) -> &'static [&'static [u32]] {
    let cells: Vec<&'static [u32]> = derive_cells(vertices, edges, cell_size)
        .into_iter()
        .map(|c| &*Box::leak(c.into_boxed_slice()))
        .collect();
    Box::leak(cells.into_boxed_slice())
}

static PENTATOPE_CELLS: LazyLock<&'static [&'static [u32]]> = LazyLock::new(|| {
    cache_cells(
        *PENTATOPE_VERTICES,
        *PENTATOPE_EDGES,
        cell_vertex_count(Polytope4::Pentatope),
    )
});
static TESSERACT_CELLS: LazyLock<&'static [&'static [u32]]> = LazyLock::new(|| {
    cache_cells(
        *TESSERACT_VERTICES,
        *TESSERACT_EDGES,
        cell_vertex_count(Polytope4::Tesseract),
    )
});
static CELL16_CELLS: LazyLock<&'static [&'static [u32]]> = LazyLock::new(|| {
    cache_cells(
        *CELL16_VERTICES,
        *CELL16_EDGES,
        cell_vertex_count(Polytope4::Cell16),
    )
});
static CELL24_CELLS: LazyLock<&'static [&'static [u32]]> = LazyLock::new(|| {
    cache_cells(
        *CELL24_VERTICES,
        *CELL24_EDGES,
        cell_vertex_count(Polytope4::Cell24),
    )
});
static CELL120_CELLS: LazyLock<&'static [&'static [u32]]> = LazyLock::new(|| {
    cache_cells(
        *CELL120_VERTICES,
        *CELL120_EDGES,
        cell_vertex_count(Polytope4::Cell120),
    )
});
static CELL600_CELLS: LazyLock<&'static [&'static [u32]]> = LazyLock::new(|| {
    cache_cells(
        *CELL600_VERTICES,
        *CELL600_EDGES,
        cell_vertex_count(Polytope4::Cell600),
    )
});

// ---------------------------------------------------------------------------
// Per-polytope topology assemblies
// ---------------------------------------------------------------------------
//
// LazyLock-of-struct so the topology itself is constructed once. Inner fields
// borrow from the per-polytope vertex / edge / cell caches above; both layers
// of `LazyLock` ensure the data outlives any caller.

static PENTATOPE_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *PENTATOPE_VERTICES,
    edges: *PENTATOPE_EDGES,
    cells: *PENTATOPE_CELLS,
});
static TESSERACT_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *TESSERACT_VERTICES,
    edges: *TESSERACT_EDGES,
    cells: *TESSERACT_CELLS,
});
static CELL16_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *CELL16_VERTICES,
    edges: *CELL16_EDGES,
    cells: *CELL16_CELLS,
});
static CELL24_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *CELL24_VERTICES,
    edges: *CELL24_EDGES,
    cells: *CELL24_CELLS,
});
static CELL120_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *CELL120_VERTICES,
    edges: *CELL120_EDGES,
    cells: *CELL120_CELLS,
});
static CELL600_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *CELL600_VERTICES,
    edges: *CELL600_EDGES,
    cells: *CELL600_CELLS,
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
        for p in Polytope4::ALL {
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
        for p in Polytope4::ALL {
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
        for p in Polytope4::ALL {
            for &[i, j] in p.topology().edges {
                assert!(i < j, "{p:?} edge ({i}, {j}) not in (min, max) order");
            }
        }
    }

    /// No duplicate edges, and `(j, i)` never appears as a separate entry from
    /// `(i, j)`. Catches accidental double-insertion in the derivation.
    #[test]
    fn edges_are_unique() {
        for p in Polytope4::ALL {
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
        for p in Polytope4::ALL {
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

    /// Cell counts per f-vector (Coxeter Table I).
    #[test]
    fn cell_counts_match_f_vector() {
        assert_eq!(Polytope4::Pentatope.cell_count(), 5);
        assert_eq!(Polytope4::Tesseract.cell_count(), 8);
        assert_eq!(Polytope4::Cell16.cell_count(), 16);
        assert_eq!(Polytope4::Cell24.cell_count(), 24);
        assert_eq!(Polytope4::Cell120.cell_count(), 120);
        assert_eq!(Polytope4::Cell600.cell_count(), 600);
    }

    /// Each cell has the vertex count of its 3D shape: pentatope cells are
    /// tetrahedral (4), tesseract cubical (8), 16-cell tetrahedral (4),
    /// 24-cell octahedral (6), 120-cell dodecahedral (20), 600-cell
    /// tetrahedral (4).
    #[test]
    fn cell_vertex_counts_match_shape() {
        let cases: &[(Polytope4, usize)] = &[
            (Polytope4::Pentatope, 4),
            (Polytope4::Tesseract, 8),
            (Polytope4::Cell16, 4),
            (Polytope4::Cell24, 6),
            (Polytope4::Cell120, 20),
            (Polytope4::Cell600, 4),
        ];
        for &(p, expected) in cases {
            for (i, cell) in p.topology().cells.iter().enumerate() {
                assert_eq!(
                    cell.len(),
                    expected,
                    "{p:?} cell {i} has {} vertices, expected {expected}",
                    cell.len()
                );
            }
        }
    }

    /// All vertices of a cell lie on a common 3-flat. Operationally: their
    /// projections onto the cell's centroid direction agree within
    /// tolerance. True by construction (the derivation explicitly fits a
    /// 3-flat), but the test guards against future refactors swapping in a
    /// derivation that loses this property.
    #[test]
    fn cells_lie_on_common_hyperplane() {
        for p in Polytope4::ALL {
            let topo = p.topology();
            for (idx, cell) in topo.cells.iter().enumerate() {
                let centroid: Vec4 = cell
                    .iter()
                    .map(|&i| topo.vertices[i as usize])
                    .fold(Vec4::ZERO, |acc, v| acc + v)
                    / cell.len() as f32;
                let dots: Vec<f32> = cell
                    .iter()
                    .map(|&i| topo.vertices[i as usize].dot(centroid))
                    .collect();
                let lo = dots.iter().copied().fold(f32::INFINITY, f32::min);
                let hi = dots.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                assert!(
                    hi - lo < CELL_TOLERANCE,
                    "{p:?} cell {idx} centroid-projection spread {} > {CELL_TOLERANCE}",
                    hi - lo
                );
            }
        }
    }

    /// Within each cell, the number of vertex pairs at canonical edge length
    /// equals the edge count of the cell's 3D shape: tetrahedron 6, cube 12,
    /// octahedron 12, dodecahedron 30. Catches a cell whose vertices are
    /// merely coplanar but don't form the expected regular polytope.
    #[test]
    fn cell_internal_edge_counts_match_shape() {
        let cases: &[(Polytope4, usize)] = &[
            (Polytope4::Pentatope, 6),
            (Polytope4::Tesseract, 12),
            (Polytope4::Cell16, 6),
            (Polytope4::Cell24, 12),
            (Polytope4::Cell120, 30),
            (Polytope4::Cell600, 6),
        ];
        for &(p, expected) in cases {
            let topo = p.topology();
            let edge_len = canonical_edge_length(p);
            for (idx, cell) in topo.cells.iter().enumerate() {
                let mut count = 0;
                for i in 0..cell.len() {
                    for j in (i + 1)..cell.len() {
                        let a = topo.vertices[cell[i] as usize];
                        let b = topo.vertices[cell[j] as usize];
                        if ((a - b).length() - edge_len).abs() < EDGE_TOLERANCE {
                            count += 1;
                        }
                    }
                }
                assert_eq!(
                    count, expected,
                    "{p:?} cell {idx} has {count} internal edges, expected {expected}"
                );
            }
        }
    }

    /// Every edge of the polytope is shared by at least two cells. This is
    /// the closed-polytope invariant: every (n-2)-face of a closed convex
    /// n-polytope lies on the boundary between adjacent cells. The exact
    /// per-polytope share count is 3 for pentatope/tesseract/24-cell/120-
    /// cell, 4 for 16-cell, 5 for 600-cell, but the weak `>= 2` form is
    /// enough to catch a derivation that drops an edge or fits the wrong
    /// cell to it.
    #[test]
    fn every_edge_in_at_least_two_cells() {
        for p in Polytope4::ALL {
            let topo = p.topology();
            for &[i, j] in topo.edges {
                let count = topo
                    .cells
                    .iter()
                    .filter(|cell| cell.contains(&i) && cell.contains(&j))
                    .count();
                assert!(
                    count >= 2,
                    "{p:?} edge ({i}, {j}) is in only {count} cell(s), expected >= 2"
                );
            }
        }
    }

    /// `cross4(a, b, c)` returns a non-zero vector orthogonal to each input
    /// when the inputs are linearly independent.
    #[test]
    fn cross4_is_orthogonal_to_inputs() {
        let a = Vec4::new(1.0, 0.5, -0.3, 0.7);
        let b = Vec4::new(-0.2, 1.0, 0.4, -0.1);
        let c = Vec4::new(0.6, -0.8, 1.0, 0.3);
        let n = cross4(a, b, c);
        assert!(
            n.length() > 0.1,
            "cross4 of linearly-independent inputs is near-zero (|n| = {})",
            n.length()
        );
        for (label, v) in [("a", a), ("b", b), ("c", c)] {
            assert!(
                n.dot(v).abs() < 1e-5,
                "cross4 result not orthogonal to {label}: n·{label} = {}",
                n.dot(v)
            );
        }
    }

    /// `cross4` returns ~zero when its inputs span a 2-flat (one input is a
    /// linear combination of the others), since no unique normal exists.
    /// The derivation loop in [`derive_cells`] relies on this to skip
    /// degenerate triples.
    #[test]
    fn cross4_zero_for_linearly_dependent_inputs() {
        let a = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let b = Vec4::new(0.0, 1.0, 0.0, 0.0);
        let c = a * 2.0 + b * 3.0;
        let n = cross4(a, b, c);
        assert!(
            n.length() < 1e-5,
            "cross4 of linearly-dependent inputs is not zero: {n:?} (|n| = {})",
            n.length()
        );
    }

    /// Euler-Poincaré relation for closed convex 4-polytopes:
    /// `V - E + F - C = 0`.
    ///
    /// `V`, `E`, `C` are checked against the topology tables. `F` (the count
    /// of 2-faces) is sourced from Coxeter, *Regular Polytopes*, Table I;
    /// we don't expose 2-faces in the API since no planned visualization
    /// needs them, but the count itself is a useful invariant test.
    #[test]
    fn euler_poincare_relation_holds() {
        // (Polytope, F = number of 2-faces). V/E/C come from `.topology()`.
        let face_counts: &[(Polytope4, i64)] = &[
            (Polytope4::Pentatope, 10),
            (Polytope4::Tesseract, 24),
            (Polytope4::Cell16, 32),
            (Polytope4::Cell24, 96),
            (Polytope4::Cell120, 720),
            (Polytope4::Cell600, 1200),
        ];
        for &(p, f) in face_counts {
            let v = p.vertex_count() as i64;
            let e = p.edge_count() as i64;
            let c = p.cell_count() as i64;
            assert_eq!(
                v - e + f - c,
                0,
                "{p:?} Euler-Poincaré: V({v}) - E({e}) + F({f}) - C({c}) != 0"
            );
        }
    }
}
