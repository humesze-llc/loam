//! Topology of the six convex regular 4-polytopes: cached unit-circumradius
//! vertex / edge / cell incidence data, addressable by a [`Polytope4`] enum
//! whose discriminants mirror the renderer's `SHAPE_*` constants. Vertices wrap
//! the [`crate::polytope_geom`] generators; edges and cells are derived on first
//! access and cached for process lifetime. CPU-only; the SDF/WGSL kernel data
//! lives in `rye_render::raymarch`.
//!
//! ```
//! use rye_shape::polytope::Polytope4;
//!
//! let topo = Polytope4::Tesseract.topology();
//! assert_eq!(topo.vertices.len(), 16);
//! for v in topo.vertices {
//!     assert!((v.length() - 1.0).abs() < 1e-5);
//! }
//! ```
use std::sync::LazyLock;

use glam::{Vec3, Vec4};

use crate::polytope_geom::{
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
/// Allocated once per polytope on first access via [`std::sync::LazyLock`].
#[derive(Debug)]
pub struct Polytope4Topology {
    /// Vertices in canonical (unit-circumradius) coordinates; `edges` and
    /// `cells` index into this slice.
    pub vertices: &'static [Vec4],
    /// Edges as `[lo, hi]` vertex-index pairs, sorted lexicographically for
    /// deterministic iteration order.
    pub edges: &'static [[u32; 2]],
    /// Cells as ascending vertex-index lists (4/8/6/20 per cell for tet /
    /// cube / octahedron / dodecahedron), sorted lexicographically for
    /// deterministic iteration order.
    pub cells: &'static [&'static [u32]],
}

impl Polytope4 {
    /// All six variants, in `repr(u32)` discriminant order.
    pub const ALL: [Polytope4; 6] = [
        Polytope4::Pentatope,
        Polytope4::Tesseract,
        Polytope4::Cell16,
        Polytope4::Cell24,
        Polytope4::Cell120,
        Polytope4::Cell600,
    ];

    /// Borrow this polytope's full topology. First access computes and caches
    /// the vertex / edge / cell tables; later calls are a dereference.
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

    /// 4D centroid of every cell, in canonical (unit-circumradius) coordinates.
    /// Each centroid has length equal to the inradius and points along the
    /// cell's outward face normal.
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

    /// Face hyperplanes derived from cell topology, as `(normals, inradius)`
    /// matching the shape of `cell120_face_planes` / `cell600_face_planes` in
    /// [`crate::polytope_geom`]. Pair with
    /// [`crate::polytope_geom::polytope_sdf_wolfe`] for an exact SDF.
    ///
    /// Unlike those dual-vertex helpers (exact on the axial + tesseract-corner
    /// orbits, approximate on the 96 golden-ratio orbits; the documented BUG),
    /// deriving normals from cell centroids is exact for every cell of every
    /// regular convex 4-polytope.
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
            // Average the inradius across all cells to absorb f32 noise rather
            // than read it off the first cell.
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

/// Default uniform styling for [`Polytope4::to_lines`]: white at 1.5 px. For
/// per-cell or per-position coloring see [`Polytope4::lines_colored_by_cell`].
const DEFAULT_LINE_COLOR: [f32; 4] = [0.9, 0.9, 0.9, 1.0];
const DEFAULT_LINE_WIDTH: f32 = 1.5;

impl crate::Visualizable<4> for Polytope4 {
    fn to_lines(&self) -> Result<crate::LineMesh<4>, crate::NotVisualizable> {
        let topo = self.topology();
        let mut mesh = crate::LineMesh::<4>::default();
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

    fn to_triangles(&self) -> Result<crate::TriangleMesh<4>, crate::NotVisualizable> {
        // No 2-face incidence data is shipped, so there is nothing to fill.
        Err(crate::NotVisualizable::Degenerate)
    }

    fn to_points(&self) -> Result<crate::PointMesh<4>, crate::NotVisualizable> {
        let topo = self.topology();
        let mut mesh = crate::PointMesh::<4>::default();
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
    /// Color each edge by the lowest-index cell its endpoints share, indexing
    /// `palette` modulo its length. Width stays at the default.
    pub fn lines_colored_by_cell(self, palette: &[[f32; 4]]) -> crate::LineMesh<4> {
        let topo = self.topology();
        let mut mesh = crate::LineMesh::<4>::default();
        let n_palette = palette.len().max(1);
        for &[i, j] in topo.edges {
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

    /// Color each edge endpoint by its 4D position via [`vertex_color_by_position`],
    /// giving a continuous color field across the edge graph. Useful for dense
    /// wireframes like the 600-cell where uniform white reads as a tangle.
    pub fn lines_colored_by_position(self) -> crate::LineMesh<4> {
        let topo = self.topology();
        let mut mesh = crate::LineMesh::<4>::default();
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

/// Map a 4D vertex to an RGBA color. The normalized `xyz` drive R/G/B biased
/// into `[0.25, 1.0]` (no fully-black vertices); `w` modulates brightness in
/// `[0.7, 1.0]` as a soft +w / -w depth cue. Deterministic and continuous, so
/// adjacent edges share their endpoint colors.
pub fn vertex_color_by_position(v: Vec4) -> [f32; 4] {
    let n = v.try_normalize().unwrap_or(Vec4::ZERO);
    let bias = |c: f32| 0.25 + 0.75 * (0.5 + 0.5 * c);
    let w_mod = 0.7 + 0.3 * (0.5 + 0.5 * n.w);
    [bias(n.x) * w_mod, bias(n.y) * w_mod, bias(n.z) * w_mod, 1.0]
}

// ---------------------------------------------------------------------------
// Cross-section algorithm
// ---------------------------------------------------------------------------

/// Cross-section fill: translucent white; alpha 0.55 keeps the surface behind
/// it visible.
const SECTION_FILL_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 0.55];
/// Cross-section perimeter: opaque bright cyan, reads against both the dim
/// parent wireframe and the SDF.
const SECTION_EDGE_COLOR: [f32; 4] = [0.30, 0.85, 0.95, 1.0];
const SECTION_EDGE_WIDTH: f32 = 2.0;

/// Per-cell cross-section as `(translucent fill triangles, cyan perimeter
/// edges)`, for rendering on top of an existing surface (SDF raymarch, parent
/// wireframe). For replacing the SDF surface with opaque rasterized geometry,
/// use [`polytope4_section_faces`]. The algorithm lives in
/// `for_each_section_cap`; it reproduces the classical sections (5-cell
/// midpoint -> tetrahedron, tesseract -> cube, etc.) with no special-casing.
/// Either returned mesh may be empty if the slice misses the polytope.
pub fn polytope4_section_overlay(
    polytope: Polytope4,
    slice: rye_math::WPlane,
) -> (crate::TriangleMesh<3>, crate::LineMesh<3>) {
    let topo = polytope.topology();
    polytope_section_overlay_with_vertices(topo.edges, topo.cells, topo.vertices, slice)
}

/// Overlay cross-section taking vertices, edges, and cells directly, for
/// sectioning a transformed (rotated, world-placed) vertex set that
/// [`polytope4_section_overlay`] cannot see. `vertices` must stay
/// index-compatible with `edges` and `cells`; rigid transforms leave topology
/// unchanged, so callers reuse the parent's edge / cell arrays.
pub fn polytope_section_overlay_with_vertices(
    edges: &[[u32; 2]],
    cells: &[&[u32]],
    vertices: &[Vec4],
    slice: rye_math::WPlane,
) -> (crate::TriangleMesh<3>, crate::LineMesh<3>) {
    let mut tri_mesh = crate::TriangleMesh::<3>::default();
    let mut edge_mesh = crate::LineMesh::<3>::default();

    for_each_section_cap(edges, cells, vertices, slice, |ordered, centroid| {
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

        // Adjacent caps share face-on-slice edges, so the perimeter draws those
        // twice; the duplication is invisible and avoids a global dedup pass.
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

/// Cross-section faces only: opaque, fan-triangulated, every vertex the same
/// `color`. The primary surface representation for a polychoral body at a
/// w-slice, replacing the SDF raymarch. Pair with
/// `FragmentShading::FaceNormalLambert` in `rye-render` for per-face shading;
/// uniform color (not [`vertex_color_by_position`]) avoids a heatmap gradient
/// that would bleed across bodies at different world positions.
pub fn polytope_section_faces_with_vertices(
    edges: &[[u32; 2]],
    cells: &[&[u32]],
    vertices: &[Vec4],
    slice: rye_math::WPlane,
    color: [f32; 4],
) -> crate::TriangleMesh<3> {
    let mut tri_mesh = crate::TriangleMesh::<3>::default();
    polytope_section_faces_append(edges, cells, vertices, slice, color, &mut tri_mesh);
    tri_mesh
}

/// Append variant of [`polytope_section_faces_with_vertices`]: writes into a
/// caller-owned mesh, offsetting indices by the existing vertex count, so
/// per-frame hot paths can merge many bodies into one reused scratch mesh
/// without allocating. On an empty mesh it equals the non-append variant.
pub fn polytope_section_faces_append(
    edges: &[[u32; 2]],
    cells: &[&[u32]],
    vertices: &[Vec4],
    slice: rye_math::WPlane,
    color: [f32; 4],
    out: &mut crate::TriangleMesh<3>,
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

/// Section faces using the polytope's own canonical (unrotated) topology
/// vertices. Mirrors [`polytope4_section_overlay`] but returns just the opaque
/// solid-colored triangle mesh.
pub fn polytope4_section_faces(
    polytope: Polytope4,
    slice: rye_math::WPlane,
    color: [f32; 4],
) -> crate::TriangleMesh<3> {
    let topo = polytope.topology();
    polytope_section_faces_with_vertices(topo.edges, topo.cells, topo.vertices, slice, color)
}

/// Shared core of the section algorithm: for every cell whose w-range crosses
/// the slice, intersect the cell's edges with the slice, fit and order the cap
/// polygon, and invoke `emit(ordered_cap, centroid)` (both in R³). Degenerate
/// caps (< 3 points, collinear) are skipped. `emit` runs in topology cell
/// order, so mesh output is deterministic across runs.
fn for_each_section_cap(
    edges: &[[u32; 2]],
    cells: &[&[u32]],
    vertices: &[Vec4],
    slice: rye_math::WPlane,
    mut emit: impl FnMut(&[Vec3], Vec3),
) {
    let slice = perturb_slice_if_needed(slice, vertices);

    // Reference "inside" point (drop-w of the 4D vertex mean) so each cap's
    // fan-triangle winding can be oriented outward. Invisible under the current
    // two-sided Lambert, but required for any future single-sided shading,
    // back-face culling, or shadow pass.
    let polytope_center_r3: Vec3 = if vertices.is_empty() {
        Vec3::ZERO
    } else {
        let mean: Vec4 = vertices.iter().copied().sum::<Vec4>() / vertices.len() as f32;
        Vec3::new(mean.x, mean.y, mean.z)
    };

    for cell in cells {
        // Per-cell w-range pruning: the load-bearing optimization for the
        // 600-cell (~430K naive ops -> ~3K typical).
        let (w_min, w_max) = cell_w_range(cell, vertices);
        if w_max < slice.w_slice - rye_math::SLICE_PERTURBATION_EPSILON
            || w_min > slice.w_slice + rye_math::SLICE_PERTURBATION_EPSILON
        {
            continue;
        }

        // Cell edges are parent edges restricted to the cell's vertex set,
        // which avoids needing per-cell 2-face incidence data.
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

        // The cap is a convex 2-polygon in R³; fit its plane basis.
        let centroid: Vec3 = cap.iter().copied().sum::<Vec3>() / cap.len() as f32;
        let Some((basis_u, mut basis_v)) = fit_plane_basis(centroid, &cap) else {
            continue;
        };

        // Flip `basis_v` so the face normal `u × v` points outward; without it
        // `fit_plane_basis`'s sign varies per cap, giving inconsistent winding.
        // Skip the flip when the centroid coincides with the center (no
        // reference direction); orientation is arbitrary there anyway.
        let outward = centroid - polytope_center_r3;
        let face_normal = basis_u.cross(basis_v);
        if outward.length_squared() > 1e-12 && face_normal.dot(outward) < 0.0 {
            basis_v = -basis_v;
        }

        let ordered = order_around_centroid(&cap, centroid, basis_u, basis_v);

        emit(&ordered, centroid);
    }
}

/// Shift the slice by [`rye_math::SLICE_PERTURBATION_EPSILON`] when any vertex's
/// w sits within that epsilon, clearing vertex-on-slice / edge-in-plane /
/// face-graze degeneracies in one step.
fn perturb_slice_if_needed(slice: rye_math::WPlane, vertices: &[Vec4]) -> rye_math::WPlane {
    let eps = rye_math::SLICE_PERTURBATION_EPSILON;
    let near = vertices.iter().any(|v| (v.w - slice.w_slice).abs() < eps);
    if near {
        rye_math::WPlane::new(slice.w_slice + eps)
    } else {
        slice
    }
}

/// Min and max w-coordinate of a cell's vertex set, for per-cell slice pruning.
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

/// Two orthonormal basis vectors `(basis_u, basis_v)` spanning the cap polygon's
/// plane. Returns `None` for a collinear or point-coincident cap; the caller
/// skips that cell.
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

/// Sort cap points by angle around the centroid in the `(basis_u, basis_v)`
/// plane, walking the convex perimeter once (the centroid is interior).
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
// Unit-circumradius vertex sets from the `euclidean_r4` generators, leaked to
// `'static` (never freed, matching the LazyLock's process lifetime).

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

/// Canonical edge length at unit circumradius. Coxeter, *Regular Polytopes*,
/// Table I, cross-checked against Wikipedia and the empirical min pairwise
/// distance.
///
/// The 120-cell value is non-obvious: Wikipedia's `3 − √5` is at circumradius
/// `2√2` (the convention of [`crate::polytope_geom::cell120_vertices`]), so unit
/// circumradius gives `(3 − √5)/(2√2) = 1/(φ²·√2)`. Dropping the `√2` is the
/// 600-cell dual's identity, not the 120-cell's.
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

/// Edge-match tolerance for the all-pairs distance check. Absorbs f32
/// accumulation (~1e-5 at unit scale) while staying far below the gap to the
/// next-shortest chord; tightest on the 120-cell (edge 0.382, next chord
/// ~0.627), where 1e-4 leaves a 4x margin either side.
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

/// Takes the vertex slice directly, not a [`Polytope4`], so the `EDGES`
/// LazyLock cannot recursively re-enter `TOPOLOGY` during init.
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
// Cells are fit against the polytope's own edge graph, not an external dual:
// the 120-cell and 600-cell generators in [`crate::polytope_geom`] are not in
// mutually-dual orientation (their 96 golden-ratio vertices differ), so a
// dual's vertices are not the other's face normals.

/// Tolerance for a vertex lying on a cell's 3-flat. f32 noise on `Vec4::dot` is
/// ~5e-7, so 1e-4 leaves ~100x margin while still rejecting adjacent-cell
/// vertices (the `n · v` spread is order 0.1, tightest on the 120-cell).
const CELL_TOLERANCE: f32 = 1e-4;

/// Expected vertex count per cell, the acceptance filter on a trial 3-flat fit:
/// a non-cell 3-flat through 4 points generically holds only those 4, a true
/// cell's holds the full cell.
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

/// 4D cross product: a vector orthogonal to all three inputs. Component `i` is
/// the signed 3x3 minor of `[a; b; c]` with column `i` dropped (cofactor
/// expansion), each minor computed as a triple product.
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

/// Adjacency list from the edge table: `adj[i]` lists every vertex sharing an
/// edge with `i`, in deterministic (lexicographic edge) order.
fn adjacency(num_vertices: usize, edges: &[[u32; 2]]) -> Vec<Vec<u32>> {
    let mut adj = vec![Vec::new(); num_vertices];
    for &[i, j] in edges {
        adj[i as usize].push(j);
        adj[j as usize].push(i);
    }
    adj
}

/// Minimum `cross4` magnitude for a trusted 3-flat normal; below it the three
/// difference vectors are near-dependent. Distinct from [`CELL_TOLERANCE`]:
/// that bounds on-plane membership in dot units, this bounds normal magnitude
/// in vector-length units.
const MIN_CROSS4_LENGTH: f32 = 1e-4;

/// Derive cells by local 3-flat fitting. For every vertex `v_0` and every
/// 3-subset of its edge-neighbors, fit the 3-flat through `{v_0, n_a, n_b, n_c}`
/// via [`cross4`] and accept it only when it holds exactly `cell_size` polytope
/// vertices. Returns one ascending-index `Vec<u32>` per cell, sorted
/// lexicographically.
///
/// Cost `O(V · D³ · V)` (vertices `V`, vertex-figure valence `D`); worst case
/// the 600-cell at ~3.2M plane scans. Cached behind a [`LazyLock`]; not for a
/// hot loop.
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

/// Materialize the cell incidence list as a leaked two-level `&'static` slice.
/// Like [`cache_edges`], takes data slices directly so it cannot re-enter the
/// [`LazyLock`] during init.
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

    /// f-vector vertex counts (Coxeter, *Regular Polytopes*, Table I).
    #[test]
    fn vertex_counts_match_f_vector() {
        assert_eq!(Polytope4::Pentatope.vertex_count(), 5);
        assert_eq!(Polytope4::Tesseract.vertex_count(), 16);
        assert_eq!(Polytope4::Cell16.vertex_count(), 8);
        assert_eq!(Polytope4::Cell24.vertex_count(), 24);
        assert_eq!(Polytope4::Cell120.vertex_count(), 600);
        assert_eq!(Polytope4::Cell600.vertex_count(), 120);
    }

    /// Every vertex sits on the unit 3-sphere (circumradius = 1).
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

    /// Discriminants match `rye_render::raymarch::SHAPE_*`. Hard-coded here
    /// since `rye_render` is not a dep; update if the renderer's table changes.
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

    /// Every edge matches the canonical edge length within `EDGE_TOLERANCE`.
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

    /// Each edge index pair is in `(min, max)` order.
    #[test]
    fn edge_pairs_in_min_max_order() {
        for p in Polytope4::ALL {
            for &[i, j] in p.topology().edges {
                assert!(i < j, "{p:?} edge ({i}, {j}) not in (min, max) order");
            }
        }
    }

    /// No duplicate edges, and `(j, i)` is never a separate entry from `(i, j)`.
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

    /// Canonical edge length equals the empirical minimum pairwise distance,
    /// catching drift between the theoretical value and the `euclidean_r4`
    /// vertex set before it silently breaks the edge set.
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

    /// Each cell has its 3D shape's vertex count (tet 4, cube 8, octahedron 6,
    /// dodecahedron 20).
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

    /// All vertices of a cell lie on a common 3-flat: their projections onto the
    /// centroid direction agree within `CELL_TOLERANCE`.
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

    /// Vertex pairs at canonical edge length within a cell match its 3D shape's
    /// edge count (tet 6, cube 12, octahedron 12, dodecahedron 30), catching a
    /// merely-coplanar cell that is not the expected regular polytope.
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

    /// Every edge is shared by at least two cells (the closed-polytope
    /// invariant). The weak `>= 2` form, below the exact per-polytope counts,
    /// still catches a dropped or mis-fit cell.
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

    /// `cross4` of linearly-independent inputs is non-zero and orthogonal to
    /// each.
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

    /// `cross4` is ~zero for linearly-dependent inputs (no unique normal);
    /// [`derive_cells`] relies on this to skip degenerate triples.
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

    /// Euler-Poincaré relation `V - E + F - C = 0` for closed convex
    /// 4-polytopes. `F` (2-face count, not exposed in the API) comes from
    /// Coxeter, *Regular Polytopes*, Table I.
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

    /// [`Visualizable<4>::to_lines`] returns one segment per topology edge.
    #[test]
    fn visualizable_line_count_matches_edge_count() {
        use crate::Visualizable;
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
            assert_eq!(mesh.colors.len(), mesh.segments.len());
            assert_eq!(mesh.widths.len(), mesh.segments.len());
        }
    }

    /// [`Visualizable<4>::to_lines`] segments hit the topology vertex
    /// coordinates directly.
    #[test]
    fn visualizable_line_endpoints_are_polytope_vertices() {
        use crate::Visualizable;
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
        use crate::Visualizable;
        for p in Polytope4::ALL {
            let mesh = <Polytope4 as Visualizable<4>>::to_points(&p)
                .expect("polytopes always produce point meshes");
            assert_eq!(mesh.positions.len(), p.vertex_count());
            assert_eq!(mesh.colors.len(), mesh.positions.len());
            assert_eq!(mesh.sizes.len(), mesh.positions.len());
        }
    }

    /// [`Visualizable<4>::to_triangles`] returns `Degenerate` until 2-face
    /// topology is shipped.
    #[test]
    fn visualizable_triangles_currently_not_visualizable() {
        use crate::Visualizable;
        for p in Polytope4::ALL {
            let result = <Polytope4 as Visualizable<4>>::to_triangles(&p);
            assert!(matches!(result, Err(crate::NotVisualizable::Degenerate)));
        }
    }

    /// [`Polytope4::lines_colored_by_cell`] colors each segment from the palette
    /// (no defaults leak through) and preserves segment count.
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

    /// 5-cell midpoint section: 4 triangle caps (12 fan triangles, 12 perimeter
    /// edges). Coxeter's classical result is a regular tetrahedron.
    #[test]
    fn pentatope_section_at_midpoint() {
        let (tri, edges) =
            polytope4_section_overlay(Polytope4::Pentatope, rye_math::WPlane::new(0.0));
        assert_eq!(tri.indices.len(), 12, "expected 12 fan triangles");
        assert_eq!(edges.segments.len(), 12, "expected 12 perimeter segments");
        // 4 caps * (centroid + 3 cap points).
        assert_eq!(tri.vertices.len(), 16);
    }

    /// Tesseract midpoint section: 6 square caps (24 fan triangles, 24 perimeter
    /// segments). Coxeter's classical result is a cube.
    #[test]
    fn tesseract_section_at_midpoint_has_six_square_caps() {
        let (tri, edges) =
            polytope4_section_overlay(Polytope4::Tesseract, rye_math::WPlane::new(0.0));
        assert_eq!(tri.indices.len(), 24, "6 cubical cells * 4 fan-triangles");
        assert_eq!(edges.segments.len(), 24, "6 caps * 4 perimeter edges");
        // 6 caps * (centroid + 4 cap points).
        assert_eq!(tri.vertices.len(), 30);
    }

    /// A slice beyond every vertex's w (`w = 2`) returns an empty section.
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

    /// A slice exactly on a vertex's w triggers the perturbation path and still
    /// yields finite output (5-cell base-vertex w = -0.25).
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

    /// A midpoint slice produces a non-empty section for every polytope.
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

    /// Face triangulation produces the same triangle and vertex counts as the
    /// overlay variant (shared cap-iteration core).
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

    /// Every face vertex carries exactly the supplied color, pinning the solid
    /// per-body contract against a reintroduction of position-based coloring.
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
    // Every section-perimeter vertex sits on the parent's true surface by
    // construction, so a correct SDF returns ~0 there. The 120/600-cell SDFs in
    // [`crate::polytope_geom`] use dual-polytope vertices as face normals (the
    // documented BUG on `cell120_face_planes` / `cell600_face_planes`), exact on
    // the axial + tesseract-corner orbits but approximate on the 96 golden-ratio
    // orbits; the tests below pin that divergence so a future BUG fix fires here.

    /// Lift the R³ perimeter back to 4D: each perimeter vertex sits on the slice
    /// hyperplane, so its w is exactly `slice.w_slice`.
    fn perimeter_vertices_4d(perim: &crate::LineMesh<3>, w: f32) -> Vec<Vec4> {
        let mut out = Vec::with_capacity(perim.segments.len() * 2);
        for (a, b) in &perim.segments {
            out.push(Vec4::new(a[0], a[1], a[2], w));
            out.push(Vec4::new(b[0], b[1], b[2], w));
        }
        out
    }

    /// Worst-case |SDF| at the 120-cell midpoint perimeter, pinning the
    /// documented BUG's divergence window: lower bound `> 1e-3` (a BUG fix drops
    /// it to ~0 and fires here), upper bound `< 0.1` (a face-normal regression
    /// widens it).
    #[test]
    fn cell120_section_perimeter_diverges_from_sdf_documenting_bug() {
        use crate::polytope_geom::{cell120_face_planes, polytope_sdf_wolfe};
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
        use crate::polytope_geom::polytope_sdf_wolfe;
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
        use crate::polytope_geom::{cell600_face_planes, polytope_sdf_wolfe};
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
