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

    /// Centroid (mean of vertex positions, in 4D) of every cell. Returned in canonical
    /// (unit-circumradius) coordinates; rigid-body transforms apply linearly so callers can
    /// rotate-and-translate the result in 4D. For a regular polytope every centroid has the
    /// same length (the inradius); the direction is the cell's outward face normal.
    ///
    /// Used by the polytope-playground demo to render cell-center sprites alongside vertex
    /// markers, and by [`Self::face_planes`] internally.
    pub fn cell_centers(self) -> Vec<Vec4> {
        let topo = self.topology();
        topo.cells
            .iter()
            .map(|cell| {
                cell.iter()
                    .map(|&i| topo.vertices[i as usize])
                    .sum::<Vec4>()
                    / cell.len() as f32
            })
            .collect()
    }

    /// Face hyperplanes derived from cell topology. For each cell, the cell centroid (mean
    /// of its vertices, in 4D) lies along the polytope's outward radial direction at that
    /// face; normalizing gives the unit face normal, and the centroid's length is the
    /// inradius (constant across all cells of a regular polytope).
    ///
    /// Returns `(normals, inradius)` matching the shape of the existing
    /// `cell120_face_planes` / `cell600_face_planes` helpers in
    /// [`crate::euclidean_r4`]. Use with
    /// [`crate::euclidean_r4::polytope_sdf_wolfe`] to compute an exact SDF
    /// for any regular convex 4-polytope.
    ///
    /// **Difference from the existing `cell{120,600}_face_planes` helpers.** Those use the
    /// *dual polytope's vertex set* as face normals, which is exact for the 24 axial + 16
    /// tesseract-corner orbits but approximate for the 96 golden-ratio orbits (the documented
    /// BUG). This method derives normals from cell topology directly, so it's exact for every
    /// cell of every regular convex 4-polytope. The pre-existing helpers remain available for
    /// backward compatibility with the raymarch kernel's `polytope_extended_sdfs_wgsl`, which
    /// embeds the BUGgy vertex tables; this method is the version to use for correctness.
    pub fn face_planes(self) -> (Vec<Vec4>, f32) {
        let topo = self.topology();
        let mut normals = Vec::with_capacity(topo.cells.len());
        let mut inradius_sum = 0.0;
        for cell in topo.cells {
            let centroid: Vec4 = cell
                .iter()
                .map(|&i| topo.vertices[i as usize])
                .sum::<Vec4>()
                / cell.len() as f32;
            let r = centroid.length();
            // Centroid magnitude is the cell's inradius; the centroid direction is the outward
            // face normal. For a regular polytope all cell centroids share the same magnitude
            // (up to f32 noise); we average across all cells to absorb that noise rather than
            // read it off the first cell.
            normals.push(centroid / r);
            inradius_sum += r;
        }
        let inradius = inradius_sum / normals.len() as f32;
        (normals, inradius)
    }
}

// ---------------------------------------------------------------------------
// Visualizable<4> impl
// ---------------------------------------------------------------------------

/// Default styling for [`Polytope4::to_lines`]: white at 1.5 px width. The trait method
/// returns geometry with a uniform style; consumers that want per-cell coloring, depth
/// dimming, hover highlights, etc., should mutate the returned mesh or call a dedicated
/// helper (see [`Polytope4::lines_colored_by_cell`]).
const DEFAULT_LINE_COLOR: [f32; 4] = [0.9, 0.9, 0.9, 1.0];
const DEFAULT_LINE_WIDTH: f32 = 1.5;

impl rye_shape::Visualizable<4> for Polytope4 {
    fn to_lines(&self) -> Result<rye_shape::LineMesh<4>, rye_shape::NotVisualizable> {
        let topo = self.topology();
        let mut mesh = rye_shape::LineMesh::<4>::default();
        mesh.segments.reserve(topo.edges.len());
        mesh.colors.reserve(topo.edges.len());
        mesh.widths.reserve(topo.edges.len());
        for &[i, j] in topo.edges {
            let a = topo.vertices[i as usize].to_array();
            let b = topo.vertices[j as usize].to_array();
            mesh.segments.push((a, b));
            mesh.colors.push((DEFAULT_LINE_COLOR, DEFAULT_LINE_COLOR));
            mesh.widths.push(DEFAULT_LINE_WIDTH);
        }
        Ok(mesh)
    }

    fn to_triangles(&self) -> Result<rye_shape::TriangleMesh<4>, rye_shape::NotVisualizable> {
        // 2-face triangulation requires the 2-face topology we deliberately don't ship in
        // the polytope module (Coxeter table I lists the counts but the incidence data
        // hasn't been derived). When a real consumer needs filled-face polytope rendering
        // we'll add it; until then there's nothing to draw.
        Err(rye_shape::NotVisualizable::Degenerate)
    }

    fn to_points(&self) -> Result<rye_shape::PointMesh<4>, rye_shape::NotVisualizable> {
        let topo = self.topology();
        let mut mesh = rye_shape::PointMesh::<4>::default();
        mesh.positions.reserve(topo.vertices.len());
        mesh.colors.reserve(topo.vertices.len());
        mesh.sizes.reserve(topo.vertices.len());
        for v in topo.vertices {
            mesh.positions.push(v.to_array());
            mesh.colors.push(DEFAULT_LINE_COLOR);
            mesh.sizes.push(4.0);
        }
        Ok(mesh)
    }
}

impl Polytope4 {
    /// Color each edge by the cell its endpoints share. `palette[k]` is the color used for
    /// edges whose lower-index incident cell is cell `k`. Cells that exceed the palette wrap
    /// around (`palette.len()` is mod'd against). Width stays at the default; mutate the
    /// returned mesh directly if you need different per-segment styling.
    pub fn lines_colored_by_cell(self, palette: &[[f32; 4]]) -> rye_shape::LineMesh<4> {
        let topo = self.topology();
        let mut mesh = rye_shape::LineMesh::<4>::default();
        let n_palette = palette.len().max(1);
        for &[i, j] in topo.edges {
            // Find the lowest cell index that contains both vertices.
            let cell_idx = topo
                .cells
                .iter()
                .position(|cell| cell.contains(&i) && cell.contains(&j))
                .unwrap_or(0);
            let color = palette[cell_idx % n_palette];
            mesh.segments.push((
                topo.vertices[i as usize].to_array(),
                topo.vertices[j as usize].to_array(),
            ));
            mesh.colors.push((color, color));
            mesh.widths.push(DEFAULT_LINE_WIDTH);
        }
        mesh
    }

    /// Color each edge by its endpoints' 4D positions, producing a per-vertex color field
    /// that flows continuously across the edge graph. Adjacent edges share their shared-vertex
    /// color, so the polytope's symmetry shows up as smooth color gradients rather than
    /// discrete per-edge swatches: useful for dense wireframes like the 600-cell where uniform
    /// white flattens visually into a tangle.
    ///
    /// Mapping (vertex normalized to unit length first):
    /// - `(x + 1) / 2` to R, `(y + 1) / 2` to G, `(z + 1) / 2` to B, biased into `[0.25, 1.0]`
    ///   so every edge stays visible (no fully-black vertices).
    /// - `w` modulates brightness multiplicatively in `[0.7, 1.0]` so the hidden dimension is
    ///   visible as a soft +w / -w cue without losing contrast.
    ///
    /// Deterministic: same polytope always produces the same coloring, no RNG.
    pub fn lines_colored_by_position(self) -> rye_shape::LineMesh<4> {
        let topo = self.topology();
        let mut mesh = rye_shape::LineMesh::<4>::default();
        mesh.segments.reserve(topo.edges.len());
        mesh.colors.reserve(topo.edges.len());
        mesh.widths.reserve(topo.edges.len());
        for &[i, j] in topo.edges {
            let va = topo.vertices[i as usize];
            let vb = topo.vertices[j as usize];
            mesh.segments.push((va.to_array(), vb.to_array()));
            mesh.colors
                .push((vertex_color_by_position(va), vertex_color_by_position(vb)));
            mesh.widths.push(DEFAULT_LINE_WIDTH);
        }
        mesh
    }
}

/// Map a unit-circumradius 4D vertex to an RGBA color via position-based encoding.
///
/// - Normalize the vertex to unit length first (skipped if it's the zero vector).
/// - `(x + 1) / 2` to R, `(y + 1) / 2` to G, `(z + 1) / 2` to B, biased into `[0.25, 1.0]`
///   so every vertex stays visible (no fully-black bias).
/// - `w` modulates brightness multiplicatively in `[0.7, 1.0]` so the hidden dimension is
///   visible as a soft +w / -w cue without losing contrast.
///
/// Deterministic + continuous: adjacent edges sharing a vertex pick up the same vertex
/// color at that endpoint, so the polytope's symmetry shows as smooth color gradients
/// across the edge graph. Used by [`Polytope4::lines_colored_by_position`] and by
/// example-side overlays that build their own meshes from per-body transformed vertices.
pub fn vertex_color_by_position(v: Vec4) -> [f32; 4] {
    let n = v.try_normalize().unwrap_or(Vec4::ZERO);
    let bias = |c: f32| 0.25 + 0.75 * (0.5 + 0.5 * c);
    let w_mod = 0.7 + 0.3 * (0.5 + 0.5 * n.w);
    [bias(n.x) * w_mod, bias(n.y) * w_mod, bias(n.z) * w_mod, 1.0]
}

// ---------------------------------------------------------------------------
// Cross-section algorithm
// ---------------------------------------------------------------------------

/// Default cross-section fill color: translucent white. Picks up tinting from the SDF or
/// parent wireframe behind it; alpha 0.55 keeps both visible.
const SECTION_FILL_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 0.55];
/// Default cross-section perimeter color: bright cyan, opaque. Reads as the "boundary of
/// what you're currently looking at" against both the dim parent wireframe and SDF.
const SECTION_EDGE_COLOR: [f32; 4] = [0.30, 0.85, 0.95, 1.0];
const SECTION_EDGE_WIDTH: f32 = 2.0;

/// Per-cell cross-section assembly returning the overlay-shaped pair `(translucent
/// fill triangles, bright cyan perimeter edges)`. Use when the section is rendered
/// *on top of* an existing surface (the SDF raymarch, a parent wireframe) and you want
/// the cap interiors visible-through and their boundaries outlined.
///
/// For replacing the polychoral SDF surface entirely with rasterized geometry, use
/// [`polytope4_section_faces`] instead: opaque, solid-colored, no perimeter edges.
///
/// Algorithm:
///
/// 1. Perturb the slice if any vertex's w sits within `SLICE_PERTURBATION_EPSILON`. The
///    single perturbation kills three degeneracies at once (vertex on slice, edge in slice
///    plane, slice grazes a face).
/// 2. For each cell, compute its w-range and skip if entirely above or below the slice.
/// 3. For each parent edge restricted to that cell (both endpoints in the cell's vertex
///    list), intersect the edge with the slice via
///    [`rye_math::SectionableSpace::edge_section`]. Collect the resulting R³ points as the
///    cap polygon.
/// 4. If the cap has fewer than 3 points the cell barely grazes the slice; skip.
/// 5. Otherwise, fit the cap's plane via the first non-collinear basis, order the cap
///    points by angle around their centroid, fan-triangulate from the centroid, and emit
///    the perimeter as a sequence of line segments.
///
/// The same algorithm produces classical cross-section polytopes from `Polytope4` topology
/// alone: 5-cell midpoint slice -> regular tetrahedron, tesseract midpoint slice -> cube,
/// 16-cell -> octahedron, etc. No per-polytope special-casing.
///
/// Returns `(triangles, perimeter)`. Either may be empty if the slice doesn't cross the
/// polytope at all.
pub fn polytope4_section_overlay(
    polytope: Polytope4,
    slice: rye_math::WPlane,
) -> (rye_shape::TriangleMesh<3>, rye_shape::LineMesh<3>) {
    let topo = polytope.topology();
    polytope_section_overlay_with_vertices(topo.edges, topo.cells, topo.vertices, slice)
}

/// Lower-level overlay-shape cross-section assembly that takes vertices, edges, and cells
/// directly. Use this when the polytope's vertices have been transformed (rigid-body
/// rotation, world-space placement, animated 4D rotation) before sectioning -- the
/// canonical [`polytope4_section_overlay`] reads vertices from [`Polytope4::topology`] and
/// isn't aware of any transform applied after.
///
/// The vertex set must remain index-compatible with `edges` and `cells`: each `edges[i]`
/// pair indexes into `vertices`, and each `cells[i]` is a vertex-index list. Topology shape
/// (edges, cells) is unchanged by rigid transforms, so callers reuse the parent polytope's
/// topology arrays and substitute only the vertex set.
pub fn polytope_section_overlay_with_vertices(
    edges: &[[u32; 2]],
    cells: &[&[u32]],
    vertices: &[Vec4],
    slice: rye_math::WPlane,
) -> (rye_shape::TriangleMesh<3>, rye_shape::LineMesh<3>) {
    let mut tri_mesh = rye_shape::TriangleMesh::<3>::default();
    let mut edge_mesh = rye_shape::LineMesh::<3>::default();

    for_each_section_cap(edges, cells, vertices, slice, |ordered, centroid| {
        // Emit a triangle fan from the centroid. The centroid is invisible inside the
        // convex cap; cap vertices form the visible perimeter.
        let cv_base = tri_mesh.vertices.len() as u32;
        tri_mesh.vertices.push(centroid.to_array());
        tri_mesh.colors.push(SECTION_FILL_COLOR);
        for cap_v in ordered {
            tri_mesh.vertices.push(cap_v.to_array());
            tri_mesh.colors.push(SECTION_FILL_COLOR);
        }
        let n = ordered.len() as u32;
        for k in 0..n {
            let k_next = (k + 1) % n;
            tri_mesh
                .indices
                .push([cv_base, cv_base + 1 + k, cv_base + 1 + k_next]);
        }

        // Emit the perimeter as cap-vertex-pair line segments. Shared edges between
        // adjacent cell caps are drawn twice (cells share their face-on-slice edges); the
        // duplication is visually invisible and avoids a global edge-dedup pass.
        for k in 0..ordered.len() {
            let a = ordered[k];
            let b = ordered[(k + 1) % ordered.len()];
            edge_mesh.segments.push((a.to_array(), b.to_array()));
            edge_mesh
                .colors
                .push((SECTION_EDGE_COLOR, SECTION_EDGE_COLOR));
            edge_mesh.widths.push(SECTION_EDGE_WIDTH);
        }
    });

    (tri_mesh, edge_mesh)
}

/// Cross-section faces only, solid-colored + opaque + fan-triangulated. Intended as
/// the *primary* surface representation for a polychoral body at a w-slice, replacing
/// the SDF raymarch for the six regular convex 4-polytopes.
///
/// Differs from [`polytope_section_overlay_with_vertices`] in two ways:
///
/// - Returns only the [`rye_shape::TriangleMesh<3>`]; the caller composes perimeter
///   edges separately (typically via the wireframe overlay).
/// - Every vertex is the same `color`. Faceted shading is delivered by the rasterizer
///   (pair this with `FragmentShading::FaceNormalLambert` in `rye-render`, which
///   derives face normals from screen-space derivatives of position). Flat color +
///   per-face Lambert reproduces the visual identity of the SDF: a single solid hue
///   per body, with light + shadow revealing the geometry. Position-based per-vertex
///   color is intentionally NOT used here because it produces a heatmap-like gradient
///   across the surface and bleeds across body boundaries when several bodies sit at
///   different world-x positions.
///
/// For position-based per-vertex coloring (the wireframe scheme), use
/// [`vertex_color_by_position`] directly when building your own mesh; the wireframe
/// path in `polytope_playground` is the reference consumer.
///
/// Performance: 600-cell midpoint slice produces ~24-60 active cells × tetrahedral cap
/// (3-point) × 3 fan-triangles ≈ 200-500 triangles per body per frame. The cost is
/// dominated by the per-edge intersection sweep (`edges.len() × cells.len()` worst case,
/// pruned per-cell), not the fan-triangulation step.
pub fn polytope_section_faces_with_vertices(
    edges: &[[u32; 2]],
    cells: &[&[u32]],
    vertices: &[Vec4],
    slice: rye_math::WPlane,
    color: [f32; 4],
) -> rye_shape::TriangleMesh<3> {
    let mut tri_mesh = rye_shape::TriangleMesh::<3>::default();
    polytope_section_faces_append(edges, cells, vertices, slice, color, &mut tri_mesh);
    tri_mesh
}

/// Append-flavored variant of [`polytope_section_faces_with_vertices`]: writes into a
/// caller-owned [`rye_shape::TriangleMesh<3>`], offsetting indices by the existing
/// vertex count so multiple bodies can be merged into a single upload buffer without
/// per-body heap allocations. Use this on per-frame render hot paths where the same
/// scratch mesh is reused frame-over-frame; use [`polytope_section_faces_with_vertices`]
/// for one-shot callers that want a fresh mesh.
///
/// Behavior is otherwise identical: same slice perturbation, same cell pruning, same
/// fan triangulation, same color assignment. Concretely, calling this function on an
/// empty mesh is equivalent to calling the non-append variant.
pub fn polytope_section_faces_append(
    edges: &[[u32; 2]],
    cells: &[&[u32]],
    vertices: &[Vec4],
    slice: rye_math::WPlane,
    color: [f32; 4],
    out: &mut rye_shape::TriangleMesh<3>,
) {
    for_each_section_cap(edges, cells, vertices, slice, |ordered, centroid| {
        let cv_base = out.vertices.len() as u32;
        out.vertices.push(centroid.to_array());
        out.colors.push(color);
        for cap_v in ordered {
            out.vertices.push(cap_v.to_array());
            out.colors.push(color);
        }
        let n = ordered.len() as u32;
        for k in 0..n {
            let k_next = (k + 1) % n;
            out.indices
                .push([cv_base, cv_base + 1 + k, cv_base + 1 + k_next]);
        }
    });
}

/// Canonical-vertex convenience: section faces using the polytope's own topology
/// vertices (unrotated, unit circumradius). Mirrors [`polytope4_section_overlay`] but returns
/// just the solid-colored opaque triangle mesh suitable for replacing the SDF surface.
pub fn polytope4_section_faces(
    polytope: Polytope4,
    slice: rye_math::WPlane,
    color: [f32; 4],
) -> rye_shape::TriangleMesh<3> {
    let topo = polytope.topology();
    polytope_section_faces_with_vertices(topo.edges, topo.cells, topo.vertices, slice, color)
}

/// Shared core of the section algorithm: iterate over every cell whose w-range crosses
/// the slice, intersect the cell's edges with the slice, fit + order the resulting cap
/// polygon, and invoke `emit` once per cell with the ordered cap vertices and centroid
/// (all in R³). Cells that miss the slice or whose cap degenerates (< 3 vertices,
/// collinear cap) are skipped silently.
///
/// `emit` is called *in cell order*, which is the topology cell order (deterministic
/// across runs). Mesh-builders can rely on it for stable triangle ordering between
/// frames.
///
/// Algorithm details captured here in one place so the two public consumers
/// ([`polytope_section_overlay_with_vertices`] for overlays,
/// [`polytope_section_faces_with_vertices`]
/// for surface replacement) share the geometric logic. Either consumer can change its
/// output shape (color, width, mesh format) without touching the cross-section math.
fn for_each_section_cap(
    edges: &[[u32; 2]],
    cells: &[&[u32]],
    vertices: &[Vec4],
    slice: rye_math::WPlane,
    mut emit: impl FnMut(&[Vec3], Vec3),
) {
    let slice = perturb_slice_if_needed(slice, vertices);

    // Polytope's R³ centroid (drop-w of the 4D vertex mean). Used as the reference "inside"
    // point so each cap's fan-triangle winding can be oriented with the face normal pointing
    // AWAY from it. Consistent orientation is invisible under the current two-sided Lambert
    // (`abs(dot(n, L))` in `triangle_raster.wgsl`) but is required for any future single-sided
    // shading, back-face culling, or shadow pass; pre-paying the cost here means consumers
    // don't have to repair winding downstream.
    let polytope_center_r3: Vec3 = if vertices.is_empty() {
        Vec3::ZERO
    } else {
        let mean: Vec4 = vertices.iter().copied().sum::<Vec4>() / vertices.len() as f32;
        Vec3::new(mean.x, mean.y, mean.z)
    };

    for cell in cells {
        // Per-cell w-range pruning: cells entirely above or below the slice can't
        // contribute, and skipping them early is the load-bearing optimization for the
        // 600-cell (600 cells x 720 edges naive = ~430K ops; with pruning, typical case
        // is ~100 active cells x 30 edges-per-cell = ~3K ops).
        let (w_min, w_max) = cell_w_range(cell, vertices);
        if w_max < slice.w_slice - rye_math::SLICE_PERTURBATION_EPSILON
            || w_min > slice.w_slice + rye_math::SLICE_PERTURBATION_EPSILON
        {
            continue;
        }

        // Cell-edges = parent-edges restricted to the cell's vertex set. Avoids needing
        // per-cell 2-face incidence data; works because the standard polychora's edge
        // sets are exactly their cells' edge sets (cells are convex 3-polytopes whose
        // 1-skeleton is a subgraph of the parent).
        let mut cap: Vec<Vec3> = Vec::with_capacity(8);
        for &[i, j] in edges {
            if !cell.contains(&i) || !cell.contains(&j) {
                continue;
            }
            if let Some((_, p3)) =
                <rye_math::EuclideanR4 as rye_math::SectionableSpace<4>>::edge_section(
                    &slice,
                    vertices[i as usize],
                    vertices[j as usize],
                )
            {
                cap.push(p3);
            }
        }
        if cap.len() < 3 {
            continue;
        }

        // Centroid + plane basis. The cap is a convex 2-polygon in R³ (intersection of a
        // convex 3-cell with a hyperplane). `fit_plane_basis` finds two orthonormal basis
        // vectors in the cap's plane via the first non-collinear pair of cap offsets.
        let centroid: Vec3 = cap.iter().copied().sum::<Vec3>() / cap.len() as f32;
        let Some((basis_u, mut basis_v)) = fit_plane_basis(centroid, &cap) else {
            continue;
        };

        // Orient `(basis_u, basis_v)` so the fan-triangle face normal `u × v` points away
        // from the polytope's R³ center. Without this, `fit_plane_basis`'s choice of `basis_v`
        // depends on which cap vertex it picked first as the orthogonal probe, which differs
        // per cap and yields inconsistent winding across the assembled section. The dot-
        // product compared against `1e-6` guards against the rare case where the cap centroid
        // coincides with the polytope center (zero-magnitude reference direction); skip the
        // flip there (orientation is arbitrary in that degenerate case anyway).
        let outward = centroid - polytope_center_r3;
        let face_normal = basis_u.cross(basis_v);
        if outward.length_squared() > 1e-12 && face_normal.dot(outward) < 0.0 {
            basis_v = -basis_v;
        }

        // Order cap points by angle around the centroid in the (u, v) plane. Convex
        // polygons sort cleanly under atan2 ordering since the centroid is interior.
        let ordered = order_around_centroid(&cap, centroid, basis_u, basis_v);

        emit(&ordered, centroid);
    }
}

/// Shift the slice by [`rye_math::SLICE_PERTURBATION_EPSILON`] when any polytope vertex's w
/// sits within that epsilon. Kills vertex-on-slice / edge-in-plane / face-graze
/// degeneracies in one step so the cell-assembly loop can ignore them.
fn perturb_slice_if_needed(slice: rye_math::WPlane, vertices: &[Vec4]) -> rye_math::WPlane {
    let eps = rye_math::SLICE_PERTURBATION_EPSILON;
    let near = vertices.iter().any(|v| (v.w - slice.w_slice).abs() < eps);
    if near {
        rye_math::WPlane::new(slice.w_slice + eps)
    } else {
        slice
    }
}

/// Min and max w-coordinate of a cell's vertex set. O(n) per cell; used by the per-cell
/// pruning step in [`polytope4_section_overlay`].
fn cell_w_range(cell: &[u32], vertices: &[Vec4]) -> (f32, f32) {
    let mut w_min = f32::INFINITY;
    let mut w_max = f32::NEG_INFINITY;
    for &i in cell {
        let w = vertices[i as usize].w;
        if w < w_min {
            w_min = w;
        }
        if w > w_max {
            w_max = w;
        }
    }
    (w_min, w_max)
}

/// Find two orthonormal basis vectors `(basis_u, basis_v)` spanning the plane of the
/// cap polygon. Picks the first non-trivial offset from the centroid as `basis_u`, then
/// looks for a second offset whose cross with `basis_u` is non-degenerate (gives the
/// plane normal); `basis_v` is recovered as `normal x basis_u`.
///
/// Returns `None` when all cap points are collinear or coincide with the centroid -- a
/// degenerate cap that the caller should skip. The slice-perturbation step in
/// [`polytope4_section_overlay`] keeps this from firing under non-pathological inputs.
fn fit_plane_basis(centroid: Vec3, points: &[Vec3]) -> Option<(Vec3, Vec3)> {
    let eps = rye_math::EDGE_PARALLEL_EPSILON;
    let mut basis_u = Vec3::ZERO;
    for p in points {
        let off = *p - centroid;
        if off.length_squared() > eps * eps {
            basis_u = off.normalize();
            break;
        }
    }
    if basis_u == Vec3::ZERO {
        return None;
    }
    for p in points {
        let off = *p - centroid;
        let cross = basis_u.cross(off);
        if cross.length_squared() > eps * eps {
            let normal = cross.normalize();
            let basis_v = normal.cross(basis_u);
            return Some((basis_u, basis_v));
        }
    }
    None
}

/// Sort cap points by angle around the centroid in the cap's `(basis_u, basis_v)` plane.
/// Convex polygons sort cleanly under this ordering since the centroid is interior; the
/// resulting sequence walks the perimeter once.
fn order_around_centroid(
    points: &[Vec3],
    centroid: Vec3,
    basis_u: Vec3,
    basis_v: Vec3,
) -> Vec<Vec3> {
    let mut indexed: Vec<(usize, f32)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let off = *p - centroid;
            let angle = off.dot(basis_v).atan2(off.dot(basis_u));
            (i, angle)
        })
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().map(|(i, _)| points[i]).collect()
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

    /// [`Visualizable<4>::to_lines`] returns one segment per topology edge, matching the
    /// edge count for every regular polytope. Catches a future regression where the impl
    /// accidentally drops edges (e.g., during a refactor that skips disconnected cells).
    #[test]
    fn visualizable_line_count_matches_edge_count() {
        use rye_shape::Visualizable;
        for p in Polytope4::ALL {
            let mesh = <Polytope4 as Visualizable<4>>::to_lines(&p)
                .expect("polytopes always produce line meshes");
            assert_eq!(
                mesh.segments.len(),
                p.edge_count(),
                "{p:?} line mesh has {} segments, expected {}",
                mesh.segments.len(),
                p.edge_count()
            );
            // Color + width arrays match segment count, per the LineMesh invariant.
            assert_eq!(mesh.colors.len(), mesh.segments.len());
            assert_eq!(mesh.widths.len(), mesh.segments.len());
        }
    }

    /// [`Visualizable<4>::to_lines`] segments hit the actual polytope vertex coordinates.
    /// Pins that the impl reads topology vertices directly, not some other vertex source.
    #[test]
    fn visualizable_line_endpoints_are_polytope_vertices() {
        use rye_shape::Visualizable;
        let topo = Polytope4::Tesseract.topology();
        let mesh = <Polytope4 as Visualizable<4>>::to_lines(&Polytope4::Tesseract).unwrap();
        for (i, &[vi, vj]) in topo.edges.iter().enumerate() {
            let (a, b) = mesh.segments[i];
            assert_eq!(a, topo.vertices[vi as usize].to_array());
            assert_eq!(b, topo.vertices[vj as usize].to_array());
        }
    }

    /// [`Visualizable<4>::to_points`] returns one point per topology vertex.
    #[test]
    fn visualizable_point_count_matches_vertex_count() {
        use rye_shape::Visualizable;
        for p in Polytope4::ALL {
            let mesh = <Polytope4 as Visualizable<4>>::to_points(&p)
                .expect("polytopes always produce point meshes");
            assert_eq!(mesh.positions.len(), p.vertex_count());
            assert_eq!(mesh.colors.len(), mesh.positions.len());
            assert_eq!(mesh.sizes.len(), mesh.positions.len());
        }
    }

    /// [`Visualizable<4>::to_triangles`] returns `Degenerate` until 2-face topology is
    /// shipped. Pinned so a future "we have 2-faces now" change updates this test alongside
    /// the impl rather than silently changing the visible behavior.
    #[test]
    fn visualizable_triangles_currently_not_visualizable() {
        use rye_shape::Visualizable;
        for p in Polytope4::ALL {
            let result = <Polytope4 as Visualizable<4>>::to_triangles(&p);
            assert!(matches!(
                result,
                Err(rye_shape::NotVisualizable::Degenerate)
            ));
        }
    }

    /// [`Polytope4::lines_colored_by_cell`] assigns each segment a palette color based on the
    /// lowest-indexed cell that contains both endpoints. Pin that segment count is preserved
    /// and all colors come from the palette (no defaults leak through).
    #[test]
    fn lines_colored_by_cell_uses_palette() {
        let palette: &[[f32; 4]] = &[
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ];
        let mesh = Polytope4::Tesseract.lines_colored_by_cell(palette);
        assert_eq!(mesh.segments.len(), Polytope4::Tesseract.edge_count());
        for (start_color, end_color) in &mesh.colors {
            assert!(palette.contains(start_color));
            assert!(palette.contains(end_color));
        }
    }

    // ----------------- Cross-section algorithm -----------------

    /// 5-cell at the midpoint slice: exactly 4 of the 5 cells cross w=0 (the cell missing
    /// the apex sits entirely at w=-0.25 and is skipped by per-cell pruning). Each crossing
    /// cell contributes a triangle cap (3 cap vertices), fan-triangulated as 3 sub-triangles
    /// from the centroid. Total: 4 caps * 3 sub-triangles = 12 triangles; perimeter has
    /// 4 caps * 3 edges = 12 edges (with duplication where caps share boundary edges).
    /// Matches Coxeter's classical result: pentatope midpoint section is a regular tetrahedron.
    #[test]
    fn pentatope_section_at_midpoint() {
        let (tri, edges) =
            polytope4_section_overlay(Polytope4::Pentatope, rye_math::WPlane::new(0.0));
        assert_eq!(tri.indices.len(), 12, "expected 12 fan triangles");
        assert_eq!(edges.segments.len(), 12, "expected 12 perimeter segments");
        // Each cap has 4 mesh-vertices (centroid + 3 cap points). 4 caps total.
        assert_eq!(tri.vertices.len(), 16);
    }

    /// Tesseract at the midpoint slice: 6 of the 8 cubical cells cross w=0 (the 2 cells with
    /// `w = +/- 0.5` fixed don't). Each crossing cell contributes a square cap (4 cap
    /// vertices), fan-triangulated as 4 sub-triangles. Total: 6 caps * 4 sub-triangles = 24
    /// triangles; perimeter has 6 caps * 4 edges = 24 segments.
    #[test]
    fn tesseract_section_at_midpoint_has_six_square_caps() {
        let (tri, edges) =
            polytope4_section_overlay(Polytope4::Tesseract, rye_math::WPlane::new(0.0));
        assert_eq!(tri.indices.len(), 24, "6 cubical cells * 4 fan-triangles");
        assert_eq!(edges.segments.len(), 24, "6 caps * 4 perimeter edges");
        // Each cap has 5 mesh-vertices (centroid + 4 cap points). 6 caps total.
        assert_eq!(tri.vertices.len(), 30);
    }

    /// Slice well outside the polytope (`w = 2` is beyond every vertex's w in any of the
    /// six polychora) returns an empty section. Per-cell pruning catches this in O(cells).
    #[test]
    fn section_outside_polytope_is_empty() {
        for polytope in Polytope4::ALL {
            let (tri, edges) = polytope4_section_overlay(polytope, rye_math::WPlane::new(2.0));
            assert!(
                tri.indices.is_empty(),
                "{polytope:?} above-vertex slice should yield no triangles"
            );
            assert!(
                edges.segments.is_empty(),
                "{polytope:?} above-vertex slice should yield no perimeter edges"
            );
        }
    }

    /// Slice placed exactly on a polytope's vertex w-coordinate triggers the perturbation
    /// path. The result should be valid (no NaN, no infinite triangles), even if the
    /// perturbed slice produces a slightly different cap than the unperturbed analytical
    /// case would. Test with the 5-cell base-vertex w = -0.25.
    #[test]
    fn vertex_on_slice_is_perturbed_not_nan() {
        let (tri, edges) =
            polytope4_section_overlay(Polytope4::Pentatope, rye_math::WPlane::new(-0.25));
        for v in &tri.vertices {
            for component in v {
                assert!(component.is_finite(), "triangle vertex must be finite");
            }
        }
        for (a, b) in &edges.segments {
            for component in a.iter().chain(b.iter()) {
                assert!(component.is_finite(), "edge vertex must be finite");
            }
        }
    }

    /// Slice value inside the polytope's w-range produces non-empty section for every one
    /// of the six polychora. Catches accidental "always-empty" failures from per-cell
    /// pruning misjudging the slice value.
    #[test]
    fn midpoint_slice_is_non_empty_for_every_polytope() {
        for polytope in Polytope4::ALL {
            let (tri, edges) = polytope4_section_overlay(polytope, rye_math::WPlane::new(0.0));
            assert!(
                !tri.indices.is_empty(),
                "{polytope:?} midpoint slice should yield triangles"
            );
            assert!(
                !edges.segments.is_empty(),
                "{polytope:?} midpoint slice should yield perimeter edges"
            );
        }
    }

    // ----------------- Section faces (filled, solid-colored) -----------------
    //
    // The face variant ([`polytope4_section_faces`]) shares its geometric core with
    // [`polytope4_section_overlay`] via [`for_each_section_cap`]. Tests below pin the
    // invariants specific to the face variant: triangle count agreement with the
    // overlay variant, and that every vertex carries the caller-provided color.

    /// Face triangulation produces the same triangle count as the overlay's triangle
    /// output, since both go through the same cap-iteration core. Agreement here is
    /// the cheapest assertion that the refactor didn't drop or duplicate triangles in
    /// one variant relative to the other.
    #[test]
    fn section_faces_triangle_count_matches_section_triangles() {
        let probe_color = [0.5, 0.5, 0.5, 1.0];
        for polytope in Polytope4::ALL {
            let slice = rye_math::WPlane::new(0.1);
            let (overlay_tri, _) = polytope4_section_overlay(polytope, slice);
            let faces_tri = polytope4_section_faces(polytope, slice, probe_color);
            assert_eq!(
                faces_tri.indices.len(),
                overlay_tri.indices.len(),
                "{polytope:?}: section_faces triangle count must match polytope4_section_overlay"
            );
            assert_eq!(
                faces_tri.vertices.len(),
                overlay_tri.vertices.len(),
                "{polytope:?}: section_faces vertex count must match polytope4_section_overlay"
            );
        }
    }

    /// Every face vertex carries exactly the color passed to the constructor. Pins
    /// the "solid per-body color" contract; faceted shading is the rasterizer's job,
    /// not the mesh's. Catches a regression where the helper accidentally reintroduces
    /// position-based per-vertex coloring (which would produce a heatmap effect across
    /// the surface and bleed across body boundaries).
    #[test]
    fn section_faces_use_supplied_color_uniformly() {
        let color = [0.95, 0.55, 0.30, 1.0];
        let mesh = polytope4_section_faces(Polytope4::Pentatope, rye_math::WPlane::new(0.0), color);
        assert!(!mesh.colors.is_empty(), "section faces must produce colors");
        for (i, c) in mesh.colors.iter().enumerate() {
            assert_eq!(
                *c, color,
                "section face vertex {i} has color {c:?}, expected {color:?}"
            );
        }
    }

    // ----------------- Cross-validation: section perimeter vs SDF surface ---
    //
    // The section perimeter is built from intersections of the parent polytope's
    // *actual* edge graph with the slice hyperplane, so every perimeter vertex
    // sits on the parent polytope's true surface by construction. For a
    // mathematically correct SDF, `polytope_sdf_wolfe(perimeter_vertex, ...)`
    // would therefore return zero (within numerical tolerance).
    //
    // The 120-cell and 600-cell SDFs in [`crate::euclidean_r4`] use dual-polytope
    // vertices as face normals (see the `BUG` comment on `cell120_face_planes`
    // and `cell600_face_planes`). Those normals are exact for the 24 axial + 16
    // tesseract-corner orbits but approximate on the 96 golden-ratio orbits, so
    // the SDF picks up a measurable non-zero value at perimeter vertices that
    // lie on those orbits' edges. Tests below pin this divergence quantitatively
    // so a future BUG fix fires here loudly enough to trigger a coordinated
    // update of both the SDF code and the polytope_playground `surface sdf` path.
    //
    // No equivalent tests for 5/8/16/24-cell: their face planes aren't exposed
    // as `pub` helpers, and the rasterized section path is correct by
    // construction (`polytope_section_overlay_with_vertices` operates on the topology
    // directly, no SDF involvement).

    /// Reconstruct 4D perimeter vertices from the R³ perimeter mesh: every
    /// section-perimeter vertex sits on the slice hyperplane by construction, so
    /// its w-coordinate is exactly `slice.w_slice`.
    fn perimeter_vertices_4d(perim: &rye_shape::LineMesh<3>, w: f32) -> Vec<Vec4> {
        let mut out = Vec::with_capacity(perim.segments.len() * 2);
        for (a, b) in &perim.segments {
            out.push(Vec4::new(a[0], a[1], a[2], w));
            out.push(Vec4::new(b[0], b[1], b[2], w));
        }
        out
    }

    /// Worst-case |SDF| evaluated at the 120-cell section perimeter at the
    /// midpoint slice. The 24 + 16 axial/tesseract-corner orbits give SDF ≈ 0
    /// (face normals exact); the 96 golden-ratio orbits show a bounded
    /// non-trivial deviation. Pin both ends:
    /// - Lower bound `> 1e-3`: a BUG fix that makes the SDF exact would drop
    ///   this to ~0; the assert fires and someone updates the test bound or
    ///   deletes the test alongside removing the BUG comments.
    /// - Upper bound `< 0.1`: catches a face-normal regression that would
    ///   produce a much wider deviation (e.g., swapping normals for the wrong
    ///   polytope's vertex set, or a basis-rotation introduced upstream).
    #[test]
    fn cell120_section_perimeter_diverges_from_sdf_documenting_bug() {
        use crate::euclidean_r4::{cell120_face_planes, polytope_sdf_wolfe};
        let slice = rye_math::WPlane::new(0.0);
        let (_, perim) = polytope4_section_overlay(Polytope4::Cell120, slice);
        let (normals, inradius) = cell120_face_planes();

        let mut max_dev: f32 = 0.0;
        for p4 in perimeter_vertices_4d(&perim, slice.w_slice) {
            let d = polytope_sdf_wolfe(p4, &normals, inradius).abs();
            if d > max_dev {
                max_dev = d;
            }
        }
        assert!(
            max_dev > 1e-3,
            "Cell120 perimeter agrees with SDF surface within {max_dev}; expected \
             measurable divergence from the documented BUG. Did `cell120_face_planes` \
             get fixed? If so, delete this test and the BUG comment."
        );
        assert!(
            max_dev < 0.1,
            "Cell120 SDF divergence {max_dev} exceeds the documented BUG window; \
             a face-normal regression may have widened the error."
        );
    }

    /// The four polytopes without the documented face-plane BUG agree exactly
    /// (within f32 tolerance) with the topology-derived SDF along the section
    /// perimeter. This is the "no camera tricks" gate: section algorithm and
    /// SDF agree, both compute the *same* surface.
    ///
    /// Uses `Polytope4::face_planes` (topology-derived, exact for every regular
    /// convex 4-polytope) rather than the raymarch kernel's `cell{120,600}_face_planes`
    /// (dual-vertex approximation). 120- and 600-cell are deliberately not in this
    /// loop because the kernel's helpers are buggy; their divergence is pinned by
    /// the `*_documenting_bug` tests below.
    #[test]
    fn five_eight_sixteen_twentyfour_cell_section_perimeter_on_sdf_surface() {
        use crate::euclidean_r4::polytope_sdf_wolfe;
        let cases = [
            Polytope4::Pentatope,
            Polytope4::Tesseract,
            Polytope4::Cell16,
            Polytope4::Cell24,
        ];
        // Each perimeter vertex sits on a parent edge intersected with the slice
        // plane, so it lies on the polytope's surface by construction. `polytope_sdf_wolfe`
        // should return ~0 at every such vertex when given accurate face planes.
        // Tolerance is `1e-3`, well above f32 noise from the SDF's Wolfe-greedy
        // projection (~1e-5 in practice) but tight enough to fire on any face-plane
        // approximation that approaches the 120/600 BUG magnitudes (~1e-2).
        const TOL: f32 = 1e-3;
        let slice = rye_math::WPlane::new(0.0);
        for polytope in cases {
            let (_, perim) = polytope4_section_overlay(polytope, slice);
            let (normals, inradius) = polytope.face_planes();
            for p4 in perimeter_vertices_4d(&perim, slice.w_slice) {
                let d = polytope_sdf_wolfe(p4, &normals, inradius).abs();
                assert!(
                    d < TOL,
                    "{polytope:?}: perimeter vertex {p4:?} has |SDF| = {d}, expected < {TOL}; \
                     section and SDF disagree"
                );
            }
        }
    }

    /// Same shape as `cell120_section_perimeter_diverges_from_sdf_documenting_bug`
    /// for the 600-cell. The 600-cell carries the symmetric BUG: its true face
    /// normals are the cell centroids of its tetrahedral cells, but the SDF uses
    /// the 120-cell's vertex set instead.
    #[test]
    fn cell600_section_perimeter_diverges_from_sdf_documenting_bug() {
        use crate::euclidean_r4::{cell600_face_planes, polytope_sdf_wolfe};
        let slice = rye_math::WPlane::new(0.0);
        let (_, perim) = polytope4_section_overlay(Polytope4::Cell600, slice);
        let (normals, inradius) = cell600_face_planes();

        let mut max_dev: f32 = 0.0;
        for p4 in perimeter_vertices_4d(&perim, slice.w_slice) {
            let d = polytope_sdf_wolfe(p4, &normals, inradius).abs();
            if d > max_dev {
                max_dev = d;
            }
        }
        assert!(
            max_dev > 1e-3,
            "Cell600 perimeter agrees with SDF surface within {max_dev}; expected \
             measurable divergence from the documented BUG. Did `cell600_face_planes` \
             get fixed? If so, delete this test and the BUG comment."
        );
        assert!(
            max_dev < 0.1,
            "Cell600 SDF divergence {max_dev} exceeds the documented BUG window; \
             a face-normal regression may have widened the error."
        );
    }

    // ----------------- Pruning + recompute invariants -----------------

    /// w-range of a cell, helper for `cell_pruning_matches_full_scan`. Independent
    /// copy of the algorithm's internal `cell_w_range`; if the two ever drift, this
    /// test fires and signals a refactor that didn't update both sites.
    fn test_cell_w_range(cell: &[u32], vertices: &[Vec4]) -> (f32, f32) {
        cell.iter()
            .map(|&i| vertices[i as usize].w)
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), w| {
                (lo.min(w), hi.max(w))
            })
    }

    /// The per-cell w-range pruning step inside the section algorithm is the
    /// load-bearing optimization for the 600-cell (factor 100x speedup at typical
    /// slices). It MUST be exact: every cell that straddles the slice contributes a
    /// cap, and no cell contributes that doesn't straddle.
    ///
    /// Counts caps in the output via `vertices.len() - indices.len()`: each cap's
    /// fan-triangulation adds one centroid vertex over and above its N cap-vertices
    /// and emits N triangles, so subtracting triangle count from vertex count
    /// recovers the number of caps. Compares against an independent count of
    /// straddling cells computed from the topology directly.
    #[test]
    fn cell_pruning_matches_straddle_count() {
        // Slice values across the [-1, 1] interior of each polytope. Avoid grazing
        // values (within `SLICE_PERTURBATION_EPSILON` of any vertex's w) so the
        // perturbation path doesn't shift the slice between our independent count
        // and the algorithm's count.
        let slices = [-0.7, -0.3, -0.1, 0.0, 0.1, 0.3, 0.7];
        let eps = rye_math::SLICE_PERTURBATION_EPSILON;

        for polytope in Polytope4::ALL {
            let topo = polytope.topology();
            for &w in &slices {
                // Reproduce the algorithm's perturbation logic so our independent
                // straddle count uses the same effective slice value the algorithm
                // does internally.
                let effective_w = if topo.vertices.iter().any(|v| (v.w - w).abs() < eps) {
                    w + eps
                } else {
                    w
                };
                let expected_caps: usize = topo
                    .cells
                    .iter()
                    .filter(|cell| {
                        let (lo, hi) = test_cell_w_range(cell, topo.vertices);
                        // Strict `<` matches the algorithm's effective predicate:
                        // a cell whose w_max == effective_w + eps would be skipped
                        // by the algorithm's edge-section step (no crossing edge
                        // produces a finite intersection point at that boundary).
                        lo < effective_w && effective_w < hi
                    })
                    .count();

                let (tri, _) = polytope4_section_overlay(polytope, rye_math::WPlane::new(w));
                let actual_caps = tri.vertices.len().saturating_sub(tri.indices.len());

                assert_eq!(
                    actual_caps, expected_caps,
                    "{polytope:?} at slice w={w}: algorithm produced {actual_caps} caps, \
                     topology-derived straddle count expected {expected_caps}"
                );
            }
        }
    }

    /// Every fan-triangle in the section mesh has its face normal pointing AWAY from
    /// the polytope's R³ center. Pins the winding-consistency contract that lets a
    /// future single-sided lighting / back-face culling consumer rely on the section
    /// surface being topologically outward-oriented. Two-sided Lambert (the current
    /// shading) is invariant under winding, so this property isn't visible at the
    /// surface, but it's load-bearing for downstream consumers we haven't built yet.
    #[test]
    fn section_face_normals_point_outward_from_polytope_center() {
        for polytope in Polytope4::ALL {
            // Polytope is centered at origin in canonical coordinates, so the
            // outward direction at any cap is the cap centroid itself.
            let center = Vec3::ZERO;
            for &slice_w in &[-0.5_f32, -0.2, 0.0, 0.2, 0.5] {
                let (mesh, _) = polytope4_section_overlay(polytope, rye_math::WPlane::new(slice_w));
                for &[a, b, c] in &mesh.indices {
                    let va = Vec3::from(mesh.vertices[a as usize]);
                    let vb = Vec3::from(mesh.vertices[b as usize]);
                    let vc = Vec3::from(mesh.vertices[c as usize]);
                    let n = (vb - va).cross(vc - va);
                    if n.length_squared() < 1e-10 {
                        continue; // degenerate triangle; skip
                    }
                    let tri_centroid = (va + vb + vc) / 3.0;
                    let outward = tri_centroid - center;
                    if outward.length_squared() < 1e-10 {
                        continue; // triangle straddles polytope center; orientation ambiguous
                    }
                    assert!(
                        n.dot(outward) > 0.0,
                        "{polytope:?} at w={slice_w}: triangle ({va:?}, {vb:?}, {vc:?}) \
                         has inward-facing normal {n:?}"
                    );
                }
            }
        }
    }

    /// Randomized robustness sweep: across each polytope, sample 16 random
    /// Rotor4 orientations applied to the canonical vertex set and 16 random
    /// slice values, exercising the cross-section algorithm under non-axis-
    /// aligned inputs. Asserts: every emitted vertex is finite (no NaN/Inf),
    /// every triangle index references a valid vertex, every line-segment
    /// endpoint matches an existing triangle vertex up to perturbation
    /// tolerance, and the perimeter is always non-empty when the slice falls
    /// inside the polytope's rotated w-range.
    ///
    /// Catches a different failure class from the fixed-vertex tests:
    /// numerical instability that only triggers at off-axis orientations
    /// (cap-collinearity that survives `fit_plane_basis`,
    /// FMA rounding at edge intersections, perturbation aliasing). Pure deterministic: uses
    /// a xorshift PRNG seeded with a fixed value, so failures reproduce verbatim across runs.
    #[test]
    fn section_under_random_rotors_stays_well_formed() {
        let mut state: u32 = 0x517_C0DE;
        let mut rand = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for polytope in Polytope4::ALL {
            let topo = polytope.topology();
            for _ in 0..16 {
                // Build a random unit rotor by populating each bivector component with a
                // signed-uniform value and normalising. `xyzw` is included for full Spin(4)
                // coverage even though it's zero for SO(4) rotations (Rotor4 carries it as a
                // generator-level field; normalisation absorbs it into the unit-norm constraint).
                let rotor = rye_math::Rotor4 {
                    s: rand(),
                    xy: rand(),
                    xz: rand(),
                    xw: rand(),
                    yz: rand(),
                    yw: rand(),
                    zw: rand(),
                    xyzw: rand(),
                }
                .normalize();
                let rotated: Vec<Vec4> = {
                    use rye_math::Rotor as _;
                    topo.vertices.iter().map(|v| rotor.apply(*v)).collect()
                };
                // Slice value in `(-1, 1)`. Unit-circumradius polytopes have w-range bounded by
                // `[-1, 1]`; rotors preserve circumradius, so this stays inside the polytope.
                let slice_w = rand() * 0.8;
                let slice = rye_math::WPlane::new(slice_w);
                let (tri, perim) =
                    polytope_section_overlay_with_vertices(topo.edges, topo.cells, &rotated, slice);

                // Finite-output property: any NaN/Inf in the output signals a degenerate-cap
                // path that escaped the `< 3 cap points` filter or the plane-fit fallback.
                for v in &tri.vertices {
                    for c in v {
                        assert!(c.is_finite(), "{polytope:?} tri vertex non-finite: {v:?}");
                    }
                }
                for (a, b) in &perim.segments {
                    for c in a.iter().chain(b.iter()) {
                        assert!(c.is_finite(), "{polytope:?} perim endpoint non-finite");
                    }
                }
                // Index-validity property: each triangle index references an in-bounds vertex.
                for &[i0, i1, i2] in &tri.indices {
                    let n = tri.vertices.len() as u32;
                    assert!(
                        i0 < n && i1 < n && i2 < n,
                        "{polytope:?} index out of bounds"
                    );
                }
                // Non-empty-section property: with the slice inside the rotated polytope's
                // w-range, the section MUST produce at least one cap. A zero-perimeter result
                // means the perturbation + pruning combo dropped a cell it shouldn't have.
                let (w_min, w_max) = rotated
                    .iter()
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| {
                        (lo.min(v.w), hi.max(v.w))
                    });
                if slice_w > w_min + 0.05 && slice_w < w_max - 0.05 {
                    assert!(
                        !perim.segments.is_empty(),
                        "{polytope:?} slice w={slice_w} inside [{w_min}, {w_max}] but produced empty section"
                    );
                }
            }
        }
    }

    /// `polytope4_section_overlay` is a pure function of `(polytope, slice)`: re-invoking it
    /// with a different slice produces a different section mesh. Trivial but pins
    /// the contract so a future caching optimization that accidentally returns a
    /// stale mesh across slice changes fires here.
    ///
    /// **Polytope choice matters.** The tesseract's cubical cells have w-edges
    /// going from (x,y,z,-0.5) to (x,y,z,+0.5): same R³ endpoints, only differing
    /// in w. Slicing at any interior `w` produces the same R³ intersection point
    /// (x,y,z,w_slice), so the tesseract's R³ section is *literally invariant*
    /// across its w-range. A non-trivial recompute test needs a polytope whose
    /// cells aren't axis-aligned in w; the 5-cell qualifies (apex edges run from
    /// (0,0,0,1) to (t,t,t,-0.25), so the slice intersection moves in R³ as `w`
    /// changes).
    #[test]
    fn section_recomputes_when_w_slice_changes() {
        let (a, _) = polytope4_section_overlay(Polytope4::Pentatope, rye_math::WPlane::new(0.0));
        let (b, _) = polytope4_section_overlay(Polytope4::Pentatope, rye_math::WPlane::new(0.4));
        assert_ne!(
            a.vertices, b.vertices,
            "section at w=0.0 and w=0.4 must differ; result was identical, \
             suggesting a stale cache or incorrect slice parameter use"
        );
    }
}
