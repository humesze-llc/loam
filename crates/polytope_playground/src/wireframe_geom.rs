//! Wireframe edge geometry: stereographic clip and near-pole reconstruction,
//! chord and great-circle-arc edge building, and per-cell w-slice helpers.

use glam::{Vec3, Vec4};
use loam_physics::polytope::Polytope4;
use loam_shape::LineMesh;

use crate::consts::{HYPERSLICE_MIN_THICKNESS, SPACE_TESSELLATION_SAMPLES};

/// Denominator floor for the affine Perspective4D scale; matches the
/// `PROJECTION_DENOM_EPSILON` the `EuclideanR4` projection uses internally so
/// the shim and the per-vertex path clamp identically.
pub(crate) const PERSPECTIVE_SCALE_DENOM_EPSILON: f32 = 1e-4;

/// Stereographic arc clip radius as a fraction of the live camera-to-body
/// distance (see [`stereographic_view_radius`]). A fraction `< 1` keeps every
/// arc endpoint in front of the eye on zoom-in; an endpoint reaching past the
/// eye plane is hyper-sensitive to rotation and produces the long/short
/// rubberband the near-plane line clip cannot remove. Applies to the 16-cell
/// only. `0.75` leaves a `0.25 ×` nearest-approach margin.
pub(crate) const STEREOGRAPHIC_VIEW_RADIUS_FRACTION: f32 = 0.75;

/// Floor on the stereographic clip radius so a close zoom never clips the
/// figure itself: a unit-circumradius polytope's legitimate image reaches
/// radius `~1.7` (a `w = 0.5` vertex), so the clip stays at least this far out.
pub(crate) const STEREOGRAPHIC_VIEW_RADIUS_FLOOR: f32 = 2.5;

/// Ceiling on the 16-cell stereographic clip radius. Beyond this the arc runs
/// into the steep near-pole region where the fixed-count
/// [`SPACE_TESSELLATION_SAMPLES`] sampling is too coarse: consecutive samples
/// jump several-fold in magnitude, faceting the arc (jagged) and twitching as
/// rotation shifts which sample brackets the boundary (the bounce). Extending
/// further smoothly needs a higher sample count or adaptive subdivision.
pub(crate) const STEREOGRAPHIC_CELL16_RADIUS_MAX: f32 = 10.0;

/// Reference clip radius for tests (no live camera): the value
/// [`stereographic_view_radius`] yields for the 16-cell at an 8-unit camera
/// distance (`0.75 × 8`).
#[cfg(test)]
pub(crate) const STEREOGRAPHIC_VIEW_RADIUS: f32 = 6.0;

/// Per-shape, camera-adaptive stereographic clip radius.
///
/// The 16-cell's `+w` pole sits exactly on a vertex (`±e_w`), so its near-pole
/// edges blow up to infinity and must be bounded: a fraction of the camera
/// distance ([`STEREOGRAPHIC_VIEW_RADIUS_FRACTION`]), floored
/// ([`STEREOGRAPHIC_VIEW_RADIUS_FLOOR`]) and capped
/// ([`STEREOGRAPHIC_CELL16_RADIUS_MAX`]). Every other polytope has its vertices
/// off the pole (tesseract `dot = ½`, 24-cell `1/√2`), so its image is bounded
/// and drawn with no clip (`f32::INFINITY`); a vertex rotated onto the pole is a
/// transient the near-plane line clip already keeps finite.
pub(crate) fn stereographic_view_radius(polytope: Polytope4, camera_distance: f32) -> f32 {
    match polytope {
        Polytope4::Cell16 => (camera_distance * STEREOGRAPHIC_VIEW_RADIUS_FRACTION).clamp(
            STEREOGRAPHIC_VIEW_RADIUS_FLOOR,
            STEREOGRAPHIC_CELL16_RADIUS_MAX,
        ),
        _ => f32::INFINITY,
    }
}

/// Cap on the reconstructed near-pole image magnitude (see
/// `stereographic_view_point`). The true magnitude `sqrt((1 + dot) / (1 - dot))`
/// diverges at the pole; `1e4` is far above any view radius (so the sample
/// clips out) yet `1e4^2 = 1e8` stays inside f32's exact-integer range for the
/// segment/sphere boundary solve.
pub(crate) const STEREOGRAPHIC_POLE_FAR_CAP: f32 = 1.0e4;

/// Body-local projected-radius clip for a `projection` at the given
/// `view_radius`, or `None` when the projection needs no clip. Only
/// [`loam_math::Projection::Stereographic`] has a genuine point-at-infinity in
/// its image; the affine and Schlegel projections keep every sample. The radius
/// is body-local, so the same 4D edge clips identically at every row slot.
pub(crate) fn stereographic_clip_radius(
    projection: &loam_math::Projection<4>,
    view_radius: f32,
) -> Option<f32> {
    match *projection {
        loam_math::Projection::Stereographic { .. } => Some(view_radius),
        loam_math::Projection::Identity
        | loam_math::Projection::Orthographic { .. }
        | loam_math::Projection::Perspective4D { .. }
        | loam_math::Projection::Schlegel { .. } => None,
    }
}

/// Uniform R³ scale an affine `projection` applies to a 4D point at
/// `w = w_slice`. `Some` where a single scalar is exact: `Identity` /
/// `Orthographic` (`1.0`) and `Perspective4D`
/// (`focal_distance / (focal_distance - w_slice)`, clamped against
/// [`PERSPECTIVE_SCALE_DENOM_EPSILON`]). `None` for the non-affine
/// `Schlegel` / `Stereographic`, whose image depends on all four coordinates;
/// those callers project per-vertex via [`cap_vertex_projected_and_world`].
///
/// `Orthographic` is `1.0` because the only one the wireframe selects is
/// `drop_axis: 3` (drop-w), which matches the section algorithm's internal
/// drop-w; spatial-axis drops are unreachable from
/// [`crate::WireframeProjection`].
pub(crate) fn perspective_scale_at_w(
    w_slice: f32,
    projection: &loam_math::Projection<4>,
) -> Option<f32> {
    match *projection {
        loam_math::Projection::Identity | loam_math::Projection::Orthographic { .. } => Some(1.0),
        loam_math::Projection::Perspective4D { focal_distance } => {
            Some(focal_distance / (focal_distance - w_slice).max(PERSPECTIVE_SCALE_DENOM_EPSILON))
        }
        // Non-affine: no single-scalar shortcut; caller projects per-vertex.
        loam_math::Projection::Schlegel { .. } | loam_math::Projection::Stereographic { .. } => {
            None
        }
    }
}

/// Whether `projection` maps a straight R4 chord to a straight R3 segment, so an
/// edge can render as a single line between projected endpoints. `Identity` /
/// `Orthographic` are linear, `Perspective4D` and `Schlegel` are central
/// projections (line-preserving). Stereographic curves a sampled chord in R3.
pub(crate) fn projection_maps_chords_to_lines(projection: &loam_math::Projection<4>) -> bool {
    match *projection {
        loam_math::Projection::Identity
        | loam_math::Projection::Orthographic { .. }
        | loam_math::Projection::Perspective4D { .. }
        | loam_math::Projection::Schlegel { .. } => true,
        loam_math::Projection::Stereographic { .. } => false,
    }
}

/// Whether a flat wireframe edge (`blend == 0`) should render as the R3 chord
/// between projected endpoints. Stereographic edges are always drawn as S3 arcs
/// (`blend == 1`), so this chord path is the affine projections' geometry.
pub(crate) fn flat_edge_uses_endpoint_chord(projection: &loam_math::Projection<4>) -> bool {
    match *projection {
        loam_math::Projection::Stereographic { .. } => true,
        _ => projection_maps_chords_to_lines(projection),
    }
}

/// Map a body-local R³ point to world R³: scale by `section_scale` (the
/// perspective scale at the cap's w) then translate by the body's R³ position.
pub(crate) fn local_r3_to_world(p: [f32; 3], section_scale: f32, body_pos_r3: Vec3) -> [f32; 3] {
    let scaled = Vec3::from_array(p) * section_scale;
    (scaled + body_pos_r3).to_array()
}

/// Map one body-local section-cap vertex to world R³ under the active wireframe
/// projection, returning both its body-local projected point (what the
/// stereographic clip tests) and its world R³ point. Returning the projected
/// point lets the perimeter outline and cap fill drop the same near-pole samples
/// the edges do, against the radius [`stereographic_clip_radius`] defines.
///
/// `section_scale` is the affine fast path: `Some(scale)` just scales and
/// translates (as [`local_r3_to_world`]). `None` (non-affine Schlegel /
/// Stereographic) reconstructs the cap's 4D vertex and projects it per-vertex
/// through the same `EuclideanR4::project_point` the wireframe uses, so the cap
/// outline lands on the projected wireframe rather than a w-only-scaled ghost.
///
/// Reconstruction is exact: the section algorithm intersects every cell edge
/// with the w-slice, so each cap vertex has `w = w_slice`; appending it inverts
/// the internal drop-w. (The algorithm's `SLICE_PERTURBATION_EPSILON` nudge
/// moves true w by at most `1e-5`, below render roundoff.)
pub(crate) fn cap_vertex_projected_and_world(
    p_r3: [f32; 3],
    w_slice: f32,
    section_scale: Option<f32>,
    projection: &loam_math::Projection<4>,
    body_pos_r3: Vec3,
) -> (Vec3, [f32; 3]) {
    match section_scale {
        Some(scale) => {
            let projected = Vec3::from_array(p_r3) * scale;
            (projected, local_r3_to_world(p_r3, scale, body_pos_r3))
        }
        None => {
            let p4 = Vec4::new(p_r3[0], p_r3[1], p_r3[2], w_slice);
            let projected =
                <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(
                    p4, projection,
                );
            (projected, (projected + body_pos_r3).to_array())
        }
    }
}

/// Project a body-local 4D point through the wireframe's `projection`, then
/// translate by the body's R³ position. No extra scale (unlike
/// [`local_r3_to_world`]); Perspective4D folds the w-scale into the projection.
///
/// Test-only oracle: the render paths inline this because they also need the
/// pre-translate projected point for the stereographic clip, which this discards.
#[cfg(test)]
pub(crate) fn project_to_world(
    p: Vec4,
    projection: &loam_math::Projection<4>,
    body_pos_r3: Vec3,
) -> Vec3 {
    <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(p, projection)
        + body_pos_r3
}

/// Smallest endpoint radius with a well-defined direction on the circumsphere;
/// below this the slerp has no axis and falls back to the flat chord. No real
/// vertex is this close to center; the guard only blocks a divide-by-zero.
pub(crate) const MIN_EDGE_RADIUS: f32 = 1e-6;

/// Wireframe Hyperslice band test: does `[interval_min, interval_max]` intersect
/// the slab `[w_slice - half, w_slice + half]` (`half = thickness / 2`,
/// `thickness` floored at [`HYPERSLICE_MIN_THICKNESS`])? Standard 1D
/// interval-overlap with closed `<=` bounds, so an endpoint exactly on the slab
/// edge is kept (the tesseract's `w = +/- 0.5` exact-boundary case).
///
/// The cull feeds this each CELL's w-range (see [`cell_w_range`]), not the
/// edge's own endpoints, so the kept-edge decision matches the cell-level
/// active coloring and the cross-section.
pub(crate) fn slab_overlaps(
    interval_min: f32,
    interval_max: f32,
    w_slice: f32,
    thickness: f32,
) -> bool {
    let half = thickness.max(HYPERSLICE_MIN_THICKNESS) * 0.5;
    let slab_min = w_slice - half;
    let slab_max = w_slice + half;
    interval_min <= slab_max && interval_max >= slab_min
}

/// The body-local w-range `[w_min, w_max]` of a cell over its vertices in
/// `local_vertices` (rotor-rotated, `body_size`-scaled). Single source of a
/// cell's w-extent for both [`crate::compute_cell_strengths`] and the Hyperslice cull,
/// so the two cannot drift. Fold order `(lo.min, hi.max)` for bit-reproducibility.
pub(crate) fn cell_w_range(cell: &[u32], local_vertices: &[Vec4]) -> (f32, f32) {
    cell.iter()
        .map(|&i| local_vertices[i as usize].w)
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), w| {
            (lo.min(w), hi.max(w))
        })
}

/// Append a flat R⁴ chord `a` -> `b` to `mesh`, subdivided into
/// [`SPACE_TESSELLATION_SAMPLES`] sub-segments and projected per-sample so a
/// non-affine `projection` renders the edge as the curve it actually is. The
/// chord geometry is unchanged (each sample `a.lerp(b, s)`); only the polyline
/// is refined. Colors lerp linearly, matching [`push_blended_edge`].
///
/// Under [`loam_math::Projection::Stereographic`] a sub-segment is emitted only
/// when both endpoints are within the clip radius; near-pole samples are dropped
/// rather than rescaled (rescaling keeps the pole-crossing direction flip), and
/// the polyline resumes from the next in-bounds sample without bridging the gap.
///
/// Future escape hatch for a flat-to-curved projection; no built-in mode uses it
/// for `blend == 0` (those use endpoint chords).
#[allow(clippy::too_many_arguments)]
pub(crate) fn push_projected_chord(
    mesh: &mut LineMesh<3>,
    a: Vec4,
    b: Vec4,
    color_a: [f32; 4],
    color_b: [f32; 4],
    width: f32,
    projection: &loam_math::Projection<4>,
    body_pos_r3: Vec3,
    view_radius: f32,
) {
    let samples = SPACE_TESSELLATION_SAMPLES;
    let clip_radius = stereographic_clip_radius(projection, view_radius);
    // Returns the pre-translate projected point (clip test) and world point (mesh).
    let sample_at = |p4: Vec4| {
        let projected = <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(
            p4, projection,
        );
        (projected, (projected + body_pos_r3).to_array())
    };
    let (proj0, world0) = sample_at(a);
    let mut prev_world = world0;
    let mut prev_c = color_a;
    let mut prev_in = sample_in_radius(proj0, clip_radius);
    for k in 1..=samples {
        let s = k as f32 / samples as f32;
        let (proj, world) = sample_at(a.lerp(b, s));
        let c = [
            color_a[0] + (color_b[0] - color_a[0]) * s,
            color_a[1] + (color_b[1] - color_a[1]) * s,
            color_a[2] + (color_b[2] - color_a[2]) * s,
            color_a[3] + (color_b[3] - color_a[3]) * s,
        ];
        let cur_in = sample_in_radius(proj, clip_radius);
        // Both endpoints in-radius: emit. A drop breaks the polyline rather than
        // bridging the gap through the pole region.
        if prev_in && cur_in {
            mesh.segments.push((prev_world, world));
            mesh.colors.push((prev_c, c));
            mesh.widths.push(width);
        }
        prev_world = world;
        prev_c = c;
        prev_in = cur_in;
    }
}

/// Whether a body-local projected sample lies within the clip radius.
/// `None` keeps every sample; `Some(r)` drops magnitude `> r`. Closed `<=`
/// against `r * r` (no `sqrt`).
#[inline]
pub(crate) fn sample_in_radius(projected: Vec3, radius: Option<f32>) -> bool {
    match radius {
        None => true,
        Some(r) => projected.length_squared() <= r * r,
    }
}

/// Parameter `t in [0, 1]` where the segment `p_in` (inside) -> `p_out`
/// (outside) crosses the clip sphere of radius `r`, in the body-local projected
/// frame. Standard segment/sphere intersection (do Carmo, *Differential
/// Geometry of Curves and Surfaces*, §1.5): with `d = p_out - p_in`,
///   `|d|^2 t^2 + 2(p_in·d) t + (|p_in|^2 - r^2) = 0`.
/// Since `|p_in| <= r < |p_out|` the roots straddle zero, so the crossing is the
/// larger root `(-b + sqrt(b^2 - a*c)) / a`. Discriminant floored at 0 and the
/// result clamped to `[0, 1]` against boundary-sample roundoff.
///
/// Smooth-clip primitive: cutting the sub-segment AT the boundary (rather than
/// dropping it) lets the arc end ride the clip sphere continuously as a vertex
/// sweeps the pole, killing the 16-cell "bounce". The cut lies on the real
/// chord, so it preserves the pole-crossing inversion; not a radial rescale.
pub(crate) fn radius_crossing_t(p_in: Vec3, p_out: Vec3, r: f32) -> f32 {
    let d = p_out - p_in;
    let a = d.length_squared();
    // Coincident samples have no crossing direction; clip at the far end.
    if a <= f32::MIN_POSITIVE {
        return 1.0;
    }
    let b = p_in.dot(d);
    let c = p_in.length_squared() - r * r;
    let disc = (b * b - a * c).max(0.0);
    ((-b + disc.sqrt()) / a).clamp(0.0, 1.0)
}

/// The clip-sphere boundary point at `t` (from [`radius_crossing_t`]) along
/// `p_in -> p_out`, as `(world_point, color)`: the projected crossing translated
/// by `body_pos`, with colors lerped by the same `t`.
pub(crate) fn clip_point(
    p_in: Vec3,
    p_out: Vec3,
    c_in: [f32; 4],
    c_out: [f32; 4],
    t: f32,
    body_pos: Vec3,
) -> ([f32; 3], [f32; 4]) {
    let boundary = (p_in.lerp(p_out, t) + body_pos).to_array();
    let color = [
        c_in[0] + (c_out[0] - c_in[0]) * t,
        c_in[1] + (c_out[1] - c_in[1]) * t,
        c_in[2] + (c_out[2] - c_in[2]) * t,
        c_in[3] + (c_out[3] - c_in[3]) * t,
    ];
    (boundary, color)
}

/// Emit one tessellation sub-segment into `mesh` under the stereographic radius
/// clip. Each end is `(projected, world, color, in_radius)`. With no clip both
/// ends are in-radius (bit-identical to the unclipped path). Under the clip: a
/// fully-inside sub-segment is emitted whole; a straddling one is cut at the
/// boundary via [`radius_crossing_t`] / [`clip_point`] (so the arc end rides the
/// sphere smoothly, killing the "bounce"); a fully-outside one is dropped.
pub(crate) fn push_clipped_subsegment(
    mesh: &mut LineMesh<3>,
    clip_radius: Option<f32>,
    width: f32,
    body_pos_r3: Vec3,
    prev: (Vec3, [f32; 3], [f32; 4], bool),
    cur: (Vec3, [f32; 3], [f32; 4], bool),
) {
    let (prev_proj, prev_world, prev_c, prev_in) = prev;
    let (cur_proj, cur_world, cur_c, cur_in) = cur;
    let mut push = |a_world, b_world, a_c, b_c| {
        mesh.segments.push((a_world, b_world));
        mesh.colors.push((a_c, b_c));
        mesh.widths.push(width);
    };
    match (clip_radius, prev_in, cur_in) {
        // No clip, or both samples inside: emit the whole sub-segment.
        (None, _, _) | (Some(_), true, true) => push(prev_world, cur_world, prev_c, cur_c),
        // Both outside: drop, so the polyline breaks across the pole region.
        (Some(_), false, false) => {}
        // Inside -> outside: cut the far end to the clip sphere.
        (Some(r), true, false) => {
            let t = radius_crossing_t(prev_proj, cur_proj, r);
            let (bw, bc) = clip_point(prev_proj, cur_proj, prev_c, cur_c, t, body_pos_r3);
            push(prev_world, bw, prev_c, bc);
        }
        // Outside -> inside: cut the near end (measured from the inside sample).
        (Some(r), false, true) => {
            let t = radius_crossing_t(cur_proj, prev_proj, r);
            let (bw, bc) = clip_point(cur_proj, prev_proj, cur_c, prev_c, t, body_pos_r3);
            push(bw, cur_world, bc, cur_c);
        }
    }
}

/// Body-local projected point of a sample `p` for the stereographic CLIP,
/// correcting the conformal map's near-pole denominator clamp.
///
/// The map floors `1 - dot(p, pole)` at `STEREOGRAPHIC_POLE_EPSILON`, but the
/// numerator `p - dot*pole` also vanishes at the pole, so within the clamp band
/// the rendered magnitude `|perp| / eps` DEFLATES toward the origin instead of
/// diverging; a pole-sweeping vertex would drag its edges through screen center.
///
/// Inside the band this substitutes the true conformal magnitude
/// `sqrt((1 + dot) / (1 - dot))` (capped by [`STEREOGRAPHIC_POLE_FAR_CAP`]) in
/// the deflated point's direction (the clamp scales `perp` uniformly, so only
/// the magnitude is wrong), so the clip treats it as the point-at-infinity it
/// is. Outside the band, and for non-stereographic projections, this is exactly
/// `project_point`. The exact pole has no outward direction, so it is left at
/// the origin (the map's pole-to-origin value).
pub(crate) fn stereographic_view_point(p: Vec4, projection: &loam_math::Projection<4>) -> Vec3 {
    let proj =
        <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(p, projection);
    let loam_math::Projection::Stereographic { pole } = projection else {
        return proj;
    };
    let dot = p.normalize().dot(*pole).clamp(-1.0, 1.0);
    let raw = 1.0 - dot;
    if raw < loam_math::STEREOGRAPHIC_POLE_EPSILON && proj.length() > MIN_EDGE_RADIUS {
        let true_mag = ((1.0 + dot) / raw.max(f32::MIN_POSITIVE))
            .sqrt()
            .min(STEREOGRAPHIC_POLE_FAR_CAP);
        proj.normalize() * true_mag
    } else {
        proj
    }
}

/// In-place near-pole clip for the section-cap FILL at TRIANGLE granularity: a
/// just-appended triangle in `indices[start_i..]` survives only when all three
/// vertices are within `radius`, mirroring the per-segment perimeter rule so
/// fill and outline cull in lockstep (a mixed triangle would tear into a gap the
/// perimeter already drops).
///
/// `projected[i - start_v]` is mesh vertex `i`'s projected point
/// ([`cap_vertex_projected_and_world`]'s first element); `start_v` rebases the
/// absolute index into the per-append slice. `radius == None` keeps every
/// triangle (affine layers), bit-identical to the unclipped path. Streaming
/// two-pointer retain, no allocation; truncates to the kept-triangle count.
pub(crate) fn retain_in_radius_triangles(
    indices: &mut Vec<[u32; 3]>,
    start_i: usize,
    start_v: usize,
    projected: &[Vec3],
    radius: Option<f32>,
) {
    if radius.is_none() {
        return;
    }
    let appended = &mut indices[start_i..];
    let mut write = 0usize;
    for read in 0..appended.len() {
        let tri = appended[read];
        let in_radius = tri
            .iter()
            .all(|&i| sample_in_radius(projected[i as usize - start_v], radius));
        if in_radius {
            appended[write] = tri;
            write += 1;
        }
    }
    indices.truncate(start_i + write);
}

/// Append one polytope edge to `mesh`, morphed between a flat R⁴ chord and an
/// S³ great-circle arc by `blend` (0 = chord, 1 = arc; derived from the
/// projection via [`crate::state::default_edge_blend`]).
///
/// `a` / `b` are the body-local 4D endpoints. Both curves share them (the
/// vertices sit on the circumsphere), so the morph only bows the interior onto
/// the sphere. `blend == 0` renders the R3 chord between projected endpoints
/// (under stereographic, a comparison overlay; the faithful arc is `blend == 1`);
/// `blend > 0` subdivides into [`SPACE_TESSELLATION_SAMPLES`] with a per-sample
/// chord/arc blend and a linear color gradient. `slerp_scratch` is a
/// caller-owned reused buffer, cleared on entry.
///
/// Each sample is a direct `flat.lerp(sphere, blend)` in ambient R⁴, not a
/// metric (`BlendedSpace::exp_target`) geodesic. The wireframe only needs the
/// visual chord-to-arc bow, which the lerp delivers bit-deterministically and
/// far cheaper than per-sample RK4 integration; this runs over every edge of
/// the 600-cell each frame (the dominant per-frame cost). Shared endpoints mean
/// no visible fidelity is lost.
#[allow(clippy::too_many_arguments)]
pub(crate) fn push_blended_edge(
    mesh: &mut LineMesh<3>,
    a: Vec4,
    b: Vec4,
    color_a: [f32; 4],
    color_b: [f32; 4],
    width: f32,
    blend: f32,
    projection: &loam_math::Projection<4>,
    body_pos_r3: Vec3,
    slerp_scratch: &mut Vec<Vec4>,
    view_radius: f32,
) {
    // Flat edge: one R3 chord per edge (a comparison overlay under stereographic).
    if blend <= 0.0 {
        if flat_edge_uses_endpoint_chord(projection) {
            let clip_radius = stereographic_clip_radius(projection, view_radius);
            let a3_local =
                <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(
                    a, projection,
                );
            let b3_local =
                <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(
                    b, projection,
                );
            if sample_in_radius(a3_local, clip_radius) && sample_in_radius(b3_local, clip_radius) {
                mesh.segments.push((
                    (a3_local + body_pos_r3).to_array(),
                    (b3_local + body_pos_r3).to_array(),
                ));
                mesh.colors.push((color_a, color_b));
                mesh.widths.push(width);
            }
            return;
        }
        // Future non-line-preserving flat projection: sample the chord.
        push_projected_chord(
            mesh,
            a,
            b,
            color_a,
            color_b,
            width,
            projection,
            body_pos_r3,
            view_radius,
        );
        return;
    }

    let radius_a = a.length();
    let radius_b = b.length();
    if radius_a < MIN_EDGE_RADIUS || radius_b < MIN_EDGE_RADIUS {
        // Vertex at the body center: no slerp axis, so degrade to the flat chord
        // (never reached for regular polytopes; guards degenerate input).
        if flat_edge_uses_endpoint_chord(projection) {
            let clip_radius = stereographic_clip_radius(projection, view_radius);
            let a3_local =
                <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(
                    a, projection,
                );
            let b3_local =
                <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(
                    b, projection,
                );
            if sample_in_radius(a3_local, clip_radius) && sample_in_radius(b3_local, clip_radius) {
                mesh.segments.push((
                    (a3_local + body_pos_r3).to_array(),
                    (b3_local + body_pos_r3).to_array(),
                ));
                mesh.colors.push((color_a, color_b));
                mesh.widths.push(width);
            }
            return;
        }
        push_projected_chord(
            mesh,
            a,
            b,
            color_a,
            color_b,
            width,
            projection,
            body_pos_r3,
            view_radius,
        );
        return;
    }

    let samples = SPACE_TESSELLATION_SAMPLES;
    let clip_radius = stereographic_clip_radius(projection, view_radius);
    // Unit endpoints on S³; the per-sample radius lerp below restores body scale.
    let p0u = a / radius_a;
    let p1u = b / radius_b;
    slerp_scratch.clear();
    <loam_math::SphericalS3Embedded as loam_math::RasterizableSpace<4>>::tessellate_segment(
        p0u,
        p1u,
        samples,
        slerp_scratch,
    );

    // Sample 0 is `a` exactly; seed `prev` from it and walk indices 1..=samples
    // (`slerp_scratch` holds `samples + 1` points). Per-sub-segment clipping
    // lives in `push_clipped_subsegment`; `stereographic_view_point` corrects the
    // near-pole denominator-clamp deflation (see its docs).
    let proj0 = stereographic_view_point(a, projection);
    let mut prev_proj = proj0;
    let mut prev_world = (proj0 + body_pos_r3).to_array();
    let mut prev_c = color_a;
    let mut prev_in = sample_in_radius(proj0, clip_radius);
    for (k, &arc_pt) in slerp_scratch.iter().enumerate().skip(1) {
        let s = k as f32 / samples as f32;
        let flat = a.lerp(b, s);
        let radius = radius_a + (radius_b - radius_a) * s;
        let sphere = radius * arc_pt;
        let proj = stereographic_view_point(flat.lerp(sphere, blend), projection);
        let world = (proj + body_pos_r3).to_array();
        let c = [
            color_a[0] + (color_b[0] - color_a[0]) * s,
            color_a[1] + (color_b[1] - color_a[1]) * s,
            color_a[2] + (color_b[2] - color_a[2]) * s,
            color_a[3] + (color_b[3] - color_a[3]) * s,
        ];
        let cur_in = sample_in_radius(proj, clip_radius);
        push_clipped_subsegment(
            mesh,
            clip_radius,
            width,
            body_pos_r3,
            (prev_proj, prev_world, prev_c, prev_in),
            (proj, world, c, cur_in),
        );
        prev_proj = proj;
        prev_world = world;
        prev_c = c;
        prev_in = cur_in;
    }
}
