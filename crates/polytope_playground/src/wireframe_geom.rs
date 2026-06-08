//! Wireframe edge geometry: stereographic clip and near-pole reconstruction,
//! chord and great-circle-arc edge building, and per-cell w-slice helpers.

use glam::{Vec3, Vec4};
use rye_physics::polytope::Polytope4;
use rye_shape::LineMesh;

use crate::consts::{HYPERSLICE_MIN_THICKNESS, SPACE_TESSELLATION_SAMPLES};

/// Denominator floor for the affine Perspective4D scale, matching the
/// `PROJECTION_DENOM_EPSILON` the `EuclideanR4` projection uses internally so the
/// shim and the per-vertex path agree at the clamp. A vertex with
/// `w == focal_distance` sits on the viewer's 3-flat; flooring keeps the scale
/// large-but-finite rather than dividing by zero.
pub(crate) const PERSPECTIVE_SCALE_DENOM_EPSILON: f32 = 1e-4;

/// Fraction of the live camera-to-body distance used as the stereographic arc
/// clip radius (see [`stereographic_view_radius`]).
///
/// **The clip must sit below the camera distance.** A stereographic near-pole
/// edge images to an arc that runs off toward infinity; the clip bounds it at a
/// world radius. If that radius exceeds the camera's distance to the figure, an
/// arc endpoint in the camera's direction lands AT or BEHIND the eye, and the
/// perspective projection of a point grazing the eye plane is hyper-sensitive: a
/// small rotation step swings its screen image from the far edge (a "long" arc)
/// to a finite on-screen point (a "short" arc), the long/short rubberband. The
/// engine's near-plane line clip (`line_raster.wgsl`) removes the sign-flip
/// *width* artifact of a behind-eye endpoint, but it cannot remove this *length*
/// discontinuity, which is inherent to an extended arc reaching past a close
/// camera. Tying the radius to a fraction `< 1` of the live camera distance keeps
/// every arc endpoint in front of the eye when zooming in (the zoom-robust
/// property). On zoom-OUT the radius is held at
/// [`STEREOGRAPHIC_CELL16_RADIUS_MAX`] rather than growing without bound, which
/// also brings the arcs on-screen (their NDC shrinks) so a pulled-back camera
/// frames the figure. This fraction applies to the 16-cell (the only shape the
/// clip bounds; see [`stereographic_view_radius`]).
///
/// `0.75` leaves a nearest-approach margin of `0.25 ×` the camera distance (a
/// gentle off-axis sweep, never a graze); at an 8-unit camera distance it yields
/// ~6.0, the eyes-on-confirmed extent (the test reference `STEREOGRAPHIC_VIEW_RADIUS`).
pub(crate) const STEREOGRAPHIC_VIEW_RADIUS_FRACTION: f32 = 0.75;

/// Floor on the stereographic clip radius, so a close zoom never clips the figure
/// itself: a unit-circumradius polytope's legitimate image reaches radius `~1.7`
/// (a `w = 0.5` vertex), so the clip is held at least this far out. When the
/// camera is nearer than `~3.3` units the floor can exceed the camera distance
/// (the user is essentially inside the figure); the near-plane line clip is the
/// backstop there.
pub(crate) const STEREOGRAPHIC_VIEW_RADIUS_FLOOR: f32 = 2.5;

/// Ceiling on the stereographic clip radius FOR THE 16-CELL. Beyond this the arc
/// would be drawn deep into the steep near-pole region, where the fixed-count
/// [`SPACE_TESSELLATION_SAMPLES`] arc sampling (uniform in 4D arc-angle) is far
/// too coarse for the nonlinear projection: consecutive samples jump several-fold
/// in image magnitude, so the rendered arc facets into long straight chords
/// (jagged) and those chords twitch as rotation shifts which sample brackets the
/// boundary (the bounce). `10.0` keeps the 16-cell's arcs extended but bounded
/// within the (mostly) well-sampled region. To extend further AND stay smooth,
/// raise this together with the arc sample count (or subdivide adaptively by
/// projected segment length); a future arc-extent slider would drive both.
pub(crate) const STEREOGRAPHIC_CELL16_RADIUS_MAX: f32 = 10.0;

/// Reference clip radius for tests (which have no live camera): the value
/// [`stereographic_view_radius`] yields for the 16-cell at an 8-unit camera
/// distance (`0.75 × 8`). A representative mid-range radius the clip-mechanics
/// tests exercise; the live render path uses the camera- and shape-adaptive
/// [`stereographic_view_radius`], so this is compiled only for tests.
#[cfg(test)]
pub(crate) const STEREOGRAPHIC_VIEW_RADIUS: f32 = 6.0;

/// Per-shape, camera-adaptive stereographic clip radius.
///
/// **The 16-cell is special.** The default `+w` pole sits exactly on a 16-cell
/// vertex (`±e_w`), so its near-pole edges genuinely blow up to infinity and must
/// be bounded. Its radius is a fraction of the live camera distance (zoom-robust:
/// shrinks on zoom-in so an endpoint never reaches the camera plane, no
/// rubberband), floored so a close zoom never clips the figure, and capped at
/// [`STEREOGRAPHIC_CELL16_RADIUS_MAX`] so a far zoom never drives the arc into the
/// under-tessellated steep region (no jaggedness/bounce).
///
/// **Every other polytope has its vertices OFF the `+w` pole** (the tesseract's
/// reach `dot = ½`, the 24-cell's `1/√2`, etc.), so its stereographic image is
/// naturally bounded and is drawn with NO clip (`f32::INFINITY`, so the clip never
/// engages), giving the full undistorted conformal extent. A vertex only reaches
/// the pole if rotated exactly onto it, a transient the near-plane line clip
/// already keeps finite.
pub(crate) fn stereographic_view_radius(polytope: Polytope4, camera_distance: f32) -> f32 {
    match polytope {
        Polytope4::Cell16 => (camera_distance * STEREOGRAPHIC_VIEW_RADIUS_FRACTION).clamp(
            STEREOGRAPHIC_VIEW_RADIUS_FLOOR,
            STEREOGRAPHIC_CELL16_RADIUS_MAX,
        ),
        _ => f32::INFINITY,
    }
}

/// Cap on the reconstructed near-pole image magnitude (see `near_pole_view_point`).
/// A point within the pole-denominator clamp band is a point-at-infinity; the
/// conformal map's denominator clamp deflates its rendered magnitude toward the
/// origin, so the wireframe builder substitutes the true magnitude
/// `sqrt((1 + dot) / (1 - dot))` in the same projected direction. That true
/// magnitude diverges at the pole, so it is capped here purely for f32 safety:
/// `1e4` is far above any camera-adaptive view radius (so the sample reliably
/// clips out) and far below any value that would lose precision in the
/// segment/sphere boundary solve (`1e4^2 = 1e8`, inside f32's exact-integer range).
pub(crate) const STEREOGRAPHIC_POLE_FAR_CAP: f32 = 1.0e4;

/// Body-local projected-radius clip for a `projection` at the given per-frame
/// `view_radius`, or `None` when the projection needs no clip. Only
/// [`rye_math::Projection::Stereographic`] has a genuine point-at-infinity in its
/// image (a vertex on the pole), so it is the only projection whose
/// near-singularity samples are clipped; the affine projections and Schlegel's
/// bounded-finite clamp keep every sample. `view_radius` is the camera-adaptive
/// [`stereographic_view_radius`] in the live render path (tests pass the fixed
/// `STEREOGRAPHIC_VIEW_RADIUS`); it is a body-local radius, so the same 4D edge
/// clips identically at every row slot regardless of the body's R³ position.
pub(crate) fn stereographic_clip_radius(
    projection: &rye_math::Projection<4>,
    view_radius: f32,
) -> Option<f32> {
    match *projection {
        rye_math::Projection::Stereographic { .. } => Some(view_radius),
        rye_math::Projection::Identity
        | rye_math::Projection::Orthographic { .. }
        | rye_math::Projection::Perspective4D { .. }
        | rye_math::Projection::Schlegel { .. } => None,
    }
}

/// Uniform R³ scale factor that an *affine* `projection` applies to a 4D point with
/// `w = w_slice`. `Some(scale)` for the projections where a single scalar at the slice's
/// w is exact: `Identity`/`Orthographic` (`1.0`) and `Perspective4D`
/// (`focal_distance / (focal_distance - w_slice)`, clamped against the same epsilon the
/// projection impl uses). `None` for `Schlegel`/`Stereographic`, which are non-affine: their
/// R³ image of a point depends on all four coordinates, not just `w`, so no single scalar
/// rescales the section cap correctly. Callers that get `None` must project the cap's 4D
/// vertices per-vertex through `EuclideanR4::project_point` (see
/// [`cap_vertex_projected_and_world`]), matching the wireframe path so cap outline
/// and wireframe coincide.
///
/// `Orthographic` returns `1.0` because the only orthographic projection the demo's
/// wireframe ever selects is `drop_axis: 3` (drop-w), which agrees exactly with the
/// section algorithm's own internal drop-w at unit scale. Orthographic drops of a spatial
/// axis are unreachable from the demo's [`crate::WireframeProjection`]; they would need the
/// per-vertex path too, but no caller produces them.
pub(crate) fn perspective_scale_at_w(
    w_slice: f32,
    projection: &rye_math::Projection<4>,
) -> Option<f32> {
    match *projection {
        rye_math::Projection::Identity | rye_math::Projection::Orthographic { .. } => Some(1.0),
        rye_math::Projection::Perspective4D { focal_distance } => {
            Some(focal_distance / (focal_distance - w_slice).max(PERSPECTIVE_SCALE_DENOM_EPSILON))
        }
        // Non-affine: the single-scalar cap shortcut does not exist. The caller falls back to
        // per-vertex projection of the recovered 4D cap vertices.
        rye_math::Projection::Schlegel { .. } | rye_math::Projection::Stereographic { .. } => None,
    }
}

/// Whether `projection` maps a straight R4 chord to a straight R3 segment, so a
/// polytope edge can be rendered as a single line between its projected endpoints.
///
/// `Identity` / `Orthographic` are linear and `Perspective4D` is a central
/// projection (the bodies sit at `w = 0`, so the perspective divide is a
/// line-preserving map onto R3). Schlegel is central projection onto the chosen
/// cell's hyperplane, so it is line-preserving too even though it is not affine
/// and cannot use the section-cap scalar shim. Stereographic is the odd one out:
/// sampling a chord through its S3-normalizing projection generally curves in
/// R3. The flat stereographic wireframe path is a separate endpoint-chord
/// overlay, not this projected chord interior.
pub(crate) fn projection_maps_chords_to_lines(projection: &rye_math::Projection<4>) -> bool {
    match *projection {
        rye_math::Projection::Identity
        | rye_math::Projection::Orthographic { .. }
        | rye_math::Projection::Perspective4D { .. }
        | rye_math::Projection::Schlegel { .. } => true,
        rye_math::Projection::Stereographic { .. } => false,
    }
}

/// Whether a flat wireframe edge (`blend == 0`) should render as the R3 chord
/// between projected endpoints. Stereographic edges are always drawn as S3 arcs
/// (`blend == 1`), so this chord path is the affine projections' geometry.
pub(crate) fn flat_edge_uses_endpoint_chord(projection: &rye_math::Projection<4>) -> bool {
    match *projection {
        rye_math::Projection::Stereographic { .. } => true,
        _ => projection_maps_chords_to_lines(projection),
    }
}

/// Map a body-local R³ point to world R³: scale by `section_scale` (the perspective scale at
/// the cap's w-coordinate) then translate by the body's R³ position. Cap rendering uses this
/// because the cross-section algorithm internally drops w and emits body-local R³; the world
/// transform happens here.
pub(crate) fn local_r3_to_world(p: [f32; 3], section_scale: f32, body_pos_r3: Vec3) -> [f32; 3] {
    let scaled = Vec3::from_array(p) * section_scale;
    (scaled + body_pos_r3).to_array()
}

/// Map one body-local section-cap vertex to world R³ under the active wireframe
/// projection, returning BOTH the cap vertex's *body-local projected* point (the
/// first tuple element, the point whose magnitude the stereographic clip tests)
/// and its world R³ point (the second). Returning the projected point keeps the
/// clip honest without re-projecting: the perimeter outline and the cap fill both
/// drop a sample whose projected magnitude exceeds `STEREOGRAPHIC_VIEW_RADIUS`,
/// using the same pre-translate point [`stereographic_clip_radius`] is defined
/// against.
///
/// `section_scale` is the affine fast path: `Some(scale)` carries the uniform R³
/// scale at the slice's w (Identity/Orthographic/Perspective4D), and the cap
/// vertex is just scaled and translated, identical to [`local_r3_to_world`]; the
/// body-local point is then `cap * scale`, and affine projections carry no clip
/// ([`stereographic_clip_radius`] is `None`) so its magnitude is never tested.
/// `None` means the projection is non-affine (Schlegel/Stereographic), so the
/// cap's 4D vertex is reconstructed and projected per-vertex through the SAME
/// `EuclideanR4::project_point` the parent wireframe uses, then translated; this
/// is what makes the flat cap outline land on the projected wireframe instead of
/// a w-only-scaled ghost, and the body-local point is exactly that per-vertex
/// projection.
///
/// The cap vertex's 4D coordinate is recoverable because the section algorithm
/// intersects every cell edge with the w-slice, so each cap vertex shares
/// `w = w_slice`; the algorithm drops w internally and returns only `(x, y, z)`,
/// and appending `w_slice` is the exact inverse for the conformal/central maps
/// that only read all four coordinates. (The algorithm's internal
/// `SLICE_PERTURBATION_EPSILON` nudge can move the true w by at most `1e-5`; that
/// is far below the visible threshold and below the per-vertex projection's own
/// roundoff, so reconstructing at the un-nudged `w_slice` is exact for rendering.)
pub(crate) fn cap_vertex_projected_and_world(
    p_r3: [f32; 3],
    w_slice: f32,
    section_scale: Option<f32>,
    projection: &rye_math::Projection<4>,
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
                <rye_math::EuclideanR4 as rye_math::RasterizableSpace<4>>::project_point(
                    p4, projection,
                );
            (projected, (projected + body_pos_r3).to_array())
        }
    }
}

/// Project a body-local 4D point to world R³: through the wireframe's 4D->R³
/// `projection`, then translated by the body's R³ position. Perspective4D
/// already folds the w-dependent scale into the projection, so no extra scale
/// factor is applied here (unlike [`local_r3_to_world`], which the cross-section
/// path needs because it drops w before this stage).
///
/// Test-only: the render paths inline this (they also need the pre-translate
/// projected point for the stereographic clip, which this discards), so it
/// survives solely as a tessellation/projection oracle for the wireframe tests.
#[cfg(test)]
pub(crate) fn project_to_world(
    p: Vec4,
    projection: &rye_math::Projection<4>,
    body_pos_r3: Vec3,
) -> Vec3 {
    <rye_math::EuclideanR4 as rye_math::RasterizableSpace<4>>::project_point(p, projection)
        + body_pos_r3
}

/// Smallest endpoint radius (distance from body center) that still has a
/// well-defined direction on the body's circumsphere. Below this, the slerp has
/// no axis to interpolate and we fall back to the flat chord. No polytope vertex
/// is this close to the center in practice; the guard only exists so a
/// degenerate input can never divide by zero.
pub(crate) const MIN_EDGE_RADIUS: f32 = 1e-6;

/// Wireframe Hyperslice band test: does a w-interval `[interval_min,
/// interval_max]` intersect the slab centered on `w_slice`?
///
/// The slab is `[w_slice - half, w_slice + half]` where `half = thickness / 2`,
/// with `thickness` floored at [`HYPERSLICE_MIN_THICKNESS`] so a user-set 0
/// still admits straddling intervals instead of demanding f32 exact equality
/// (see the constant's docstring). The two intervals intersect iff
/// `interval_min <= slab_max && interval_max >= slab_min` (closed-band `<=`,
/// so an interval endpoint sitting exactly on `w_slice +/- half` is kept).
/// This is the standard 1D interval-overlap predicate; the closed bound is
/// what makes the tesseract's `w = +/- 0.5` vertices a deterministic
/// exact-boundary case.
///
/// The Hyperslice cull feeds this the w-range of each CELL the edge belongs to
/// (not the edge's own endpoints), so the kept-edge decision agrees with the
/// cell-level active-edge coloring and the cross-section, which are also
/// cell-level. See [`cell_w_range`] and the cull closure in
/// `render_wireframe_overlay`.
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

/// The body-local w-range `[w_min, w_max]` of a cell, folded over its vertex
/// indices into `local_vertices` (rotor-rotated, `body_size`-scaled). This is
/// the single source of a cell's w-extent: [`compute_cell_strengths`] folds it
/// for the crossing-strength gradient and the Hyperslice cull folds it for the
/// slab test, so the cull can never drift from the activity coloring. The fold
/// order is `(lo.min, hi.max)` exactly, preserved for bit-reproducibility.
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
/// straight-chord geometry is unchanged (each sample is `a.lerp(b, s)`, never
/// bowed toward the sphere); only the screen polyline is refined. Colors lerp
/// linearly between the endpoints, matching [`push_blended_edge`]'s sampling
/// convention so the two paths are visually seamless.
///
/// Under [`rye_math::Projection::Stereographic`] the polyline is clipped at
/// `STEREOGRAPHIC_VIEW_RADIUS`: a sub-segment is emitted only when BOTH its
/// endpoints' body-local projected magnitudes are within the radius, so a
/// near-pole sample (which the pole-denominator clamp maps to a huge finite
/// point) is dropped rather than drawn. The clip is a sample-granularity drop in
/// the same streaming `continue` idiom as the rasterizer's non-finite cull, not
/// a magnitude rescale: rescaling a near-pole sample to the radius would keep the
/// 180-degree direction flip across a pole crossing (see
/// `STEREOGRAPHIC_VIEW_RADIUS`). When the projected point re-enters the radius
/// the polyline resumes from the new in-bounds sample, never bridging across the
/// dropped gap, so the edge runs out toward the view boundary and the offscreen
/// blow-up is culled. No clip is applied for other projections
/// ([`stereographic_clip_radius`] returns `None`).
///
/// Kept as the future escape hatch for a projection that genuinely maps flat R4
/// chords to curved R3 images. Current built-in modes do not use it for
/// `blend == 0`: line-preserving projections and the stereographic comparison
/// overlay both use endpoint chords.
#[allow(clippy::too_many_arguments)]
pub(crate) fn push_projected_chord(
    mesh: &mut LineMesh<3>,
    a: Vec4,
    b: Vec4,
    color_a: [f32; 4],
    color_b: [f32; 4],
    width: f32,
    projection: &rye_math::Projection<4>,
    body_pos_r3: Vec3,
    view_radius: f32,
) {
    let samples = SPACE_TESSELLATION_SAMPLES;
    let clip_radius = stereographic_clip_radius(projection, view_radius);
    // Seed from sample 0 (`a` exactly). `sample_at` returns the pre-translate
    // projected point (for the clip test) and the world point (for the mesh).
    let sample_at = |p4: Vec4| {
        let projected = <rye_math::EuclideanR4 as rye_math::RasterizableSpace<4>>::project_point(
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
        // Sample the straight 4D chord; `s == 1` recovers `b` exactly.
        let (proj, world) = sample_at(a.lerp(b, s));
        let c = [
            color_a[0] + (color_b[0] - color_a[0]) * s,
            color_a[1] + (color_b[1] - color_a[1]) * s,
            color_a[2] + (color_b[2] - color_a[2]) * s,
            color_a[3] + (color_b[3] - color_a[3]) * s,
        ];
        let cur_in = sample_in_radius(proj, clip_radius);
        // Emit only when both endpoints are within the clip radius. A dropped
        // sample breaks the polyline; the next in-bounds pair starts a fresh
        // sub-segment rather than bridging the gap through the pole region.
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
/// `radius == None` (no clip for this projection) keeps every sample;
/// `Some(r)` drops samples whose magnitude exceeds `r`. Uses `length_squared`
/// against `r * r` to avoid the `sqrt`, and the `<=` keeps a sample sitting
/// exactly on the boundary (matching the closed-band discipline elsewhere).
#[inline]
pub(crate) fn sample_in_radius(projected: Vec3, radius: Option<f32>) -> bool {
    match radius {
        None => true,
        Some(r) => projected.length_squared() <= r * r,
    }
}

/// Parameter `t in [0, 1]` at which the straight segment `inside -> outside`
/// (`p_in` within the clip sphere, `p_out` beyond it) crosses the clip sphere of
/// radius `r`, both points in the body-local projected frame. Solving
/// `|p_in + t*(p_out - p_in)|^2 = r^2` for `t` is the standard segment/sphere
/// intersection (do Carmo, *Differential Geometry of Curves and Surfaces*, §1.5,
/// the line/quadric form): with `d = p_out - p_in`,
///   `|d|^2 t^2 + 2(p_in·d) t + (|p_in|^2 - r^2) = 0`.
/// Because `|p_in| <= r < |p_out|` the constant term is non-positive and the
/// leading term positive, so the two roots straddle zero and the unique crossing
/// in `[0, 1]` is the larger root `(-b + sqrt(b^2 - a*c)) / a`. The discriminant
/// is non-negative for a genuine straddle; it is floored at 0 against f32
/// roundoff on a sample sitting essentially on the boundary, and the result is
/// clamped to `[0, 1]` so a hair-past-unit root from the same roundoff cannot
/// push the clip point off the segment.
///
/// This is the smooth-clip primitive: a near-pole arc sub-segment that leaves
/// the view radius is cut AT the boundary rather than dropped whole, so the
/// visible arc end rides the clip sphere continuously as a vertex sweeps the
/// pole under rotation, instead of popping between discrete tessellation samples
/// (the 16-cell "bounce"). The cut point lies on the real polyline chord, so it
/// preserves the arc's path and the genuine pole-crossing inversion; it is NOT a
/// radial rescale of an off-arc sample.
pub(crate) fn radius_crossing_t(p_in: Vec3, p_out: Vec3, r: f32) -> f32 {
    let d = p_out - p_in;
    let a = d.length_squared();
    // Coincident projected samples have no crossing direction; clip at the far
    // end (degenerate, effectively never hit for a real straddle).
    if a <= f32::MIN_POSITIVE {
        return 1.0;
    }
    let b = p_in.dot(d);
    let c = p_in.length_squared() - r * r;
    let disc = (b * b - a * c).max(0.0);
    ((-b + disc.sqrt()) / a).clamp(0.0, 1.0)
}

/// The clip-sphere boundary point at parameter `t` along the projected segment
/// `p_in -> p_out`, returned as `(world_point, color)`: the body-local projected
/// crossing `lerp(p_in, p_out, t)` translated by `body_pos`, paired with the
/// endpoint colors linearly interpolated by the same `t` so the cut inherits the
/// edge's gradient. `t` comes from [`radius_crossing_t`].
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
/// clip. Each end is `(projected, world, color, in_radius)`: the body-local
/// projected point (what the clip tests), its world point (what the mesh draws),
/// the endpoint color, and whether the projected point is within the clip radius.
///
/// With no clip (`clip_radius == None`) both ends are always in-radius and the
/// whole sub-segment is emitted, bit-identical to the unclipped path. Under the
/// clip a fully-inside sub-segment is emitted whole; a sub-segment that STRADDLES
/// the boundary is cut AT the clip sphere via [`radius_crossing_t`] /
/// [`clip_point`], so the visible arc end rides the boundary smoothly as a vertex
/// sweeps the pole (this is what removes the near-pole "bounce": the tip tracks
/// the boundary continuously instead of popping between discrete tessellation
/// samples); a fully-outside sub-segment is dropped, breaking the polyline so the
/// next in-bounds pair starts fresh rather than bridging the gap through the pole
/// region.
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

/// Body-local projected point of a wireframe sample `p` for the stereographic
/// CLIP, correcting the conformal map's near-pole denominator clamp.
///
/// The map floors the denominator `1 - dot(p, pole)` at `STEREOGRAPHIC_POLE_EPSILON`
/// so a vertex at the pole stays finite, but the numerator `p - dot*pole` vanishes
/// at the pole too. So WITHIN the clamp band the rendered magnitude
/// `|perp| / eps` DEFLATES toward the origin as the sample nears the pole, instead
/// of diverging. A vertex sweeping the pole under rotation would then drag its
/// incident edges in through the screen center and back out (the gradient visibly
/// sliding along a line that should read as a static axis), rather than the edges
/// running off to the clip boundary and the figure inverting cleanly.
///
/// For a sample genuinely inside the clamp band this returns the TRUE conformal
/// magnitude `sqrt((1 + dot) / (1 - dot))` (capped by [`STEREOGRAPHIC_POLE_FAR_CAP`]
/// for f32 safety) in the SAME projected direction, so the magnitude clip treats
/// it as the point-at-infinity it is and the boundary cut runs the edge out toward
/// it. The projected direction is taken from the deflated point, which is correct:
/// the clamp scales `perp` uniformly, so only the magnitude is wrong, not the
/// direction. Outside the band, and for every non-stereographic projection, this
/// is exactly `project_point` (no perturbation, bit-identical to the unclamped
/// path). The exact pole (`perp ~ 0`, projected point at the origin) has no
/// direction to send outward, so it is left at the origin (the map's documented
/// pole-to-origin value); a rotating vertex effectively never lands on it exactly.
pub(crate) fn stereographic_view_point(p: Vec4, projection: &rye_math::Projection<4>) -> Vec3 {
    let proj =
        <rye_math::EuclideanR4 as rye_math::RasterizableSpace<4>>::project_point(p, projection);
    let rye_math::Projection::Stereographic { pole } = projection else {
        return proj;
    };
    let dot = p.normalize().dot(*pole).clamp(-1.0, 1.0);
    let raw = 1.0 - dot;
    if raw < rye_math::STEREOGRAPHIC_POLE_EPSILON && proj.length() > MIN_EDGE_RADIUS {
        let true_mag = ((1.0 + dot) / raw.max(f32::MIN_POSITIVE))
            .sqrt()
            .min(STEREOGRAPHIC_POLE_FAR_CAP);
        proj.normalize() * true_mag
    } else {
        proj
    }
}

/// In-place near-pole clip for the section-cap FILL, at TRIANGLE granularity: a
/// just-appended triangle in `indices[start_i..]` survives only when all three of
/// its vertices' body-local projected points are within `radius`, mirroring the
/// per-segment perimeter rule so fill and outline cull in lockstep. A triangle
/// with a kept and a dropped vertex would otherwise tear into a gap the perimeter
/// already drops, reintroducing the fill/outline mismatch.
///
/// `projected[i - start_v]` is the body-local projected point of mesh vertex `i`
/// ([`cap_vertex_projected_and_world`]'s first element); indices are absolute into
/// `mesh.vertices` (see `polytope_section_faces_append`), so `start_v` rebases an
/// index into the per-append `projected` slice. Dropped triangles leave orphan
/// vertices no kept triangle references, exactly as a dropped perimeter segment
/// leaves its endpoints unreferenced. `radius == None` keeps every triangle
/// (affine layers), so the appended range is untouched and bit-identical to the
/// unclipped path. Compaction is a streaming two-pointer retain with no
/// allocation. Returns the kept-triangle count for the caller to truncate to.
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
/// S³ great-circle arc by `blend` (0 = chord, 1 = arc). The caller derives
/// `blend` from the projection ([`state::default_edge_blend`]): Stereographic
/// passes 1, the affine projections pass 0.
///
/// `a` / `b` are the body-local 4D endpoints (rotor-rotated and `body_size`-scaled).
/// Both interpolation curves share these endpoints because the polytope's
/// vertices sit on the body's circumsphere, so the morph only bows the edge
/// interior outward onto the sphere. Flat edges (`blend == 0`) render as the R3
/// chord between projected endpoints. Under stereographic that flat chord is a
/// comparison overlay; the faithful S3 great-circle edge is the `blend == 1`
/// path. The blended (`blend > 0`) path subdivides into
/// [`SPACE_TESSELLATION_SAMPLES`] sub-segments with a per-sample chord/arc blend
/// and a linear color gradient between the endpoint colors.
///
/// `slerp_scratch` is a caller-owned buffer reused across edges to keep the
/// great-circle sampling off the per-edge allocation path; it is cleared on
/// entry.
///
/// Divergence from a metric-geodesic morph: a `BlendedSpace::exp_target`
/// geodesic between the flat and spherical metrics, RK4-integrated, is the
/// textbook approach; this deliberately bypasses it. Each sample here is a
/// direct `flat.lerp(sphere, blend)` (below): the flat point is the R⁴ chord
/// sample, the spherical point is the scaled great-circle arc sample, and the
/// two are linearly blended in the ambient R⁴. This is not a metric geodesic;
/// it is a straight-line interpolation between two precomputed curves. The
/// wireframe only needs the chord-to-arc *visual* morph (bow the edge interior
/// out onto the sphere), and the direct lerp delivers exactly that while being
/// (a) bit-deterministic with no RK4 step-size or accumulation error and (b)
/// far cheaper than per-sample geodesic integration, which matters because this
/// runs over every edge of the 600-cell wireframe each frame (the documented
/// dominant per-frame cost). The endpoints are shared by both curves, so no
/// metric fidelity is lost where it would be visible; only the interior path
/// differs, and the arc sample already carries the S3 bow the morph is
/// meant to show.
#[allow(clippy::too_many_arguments)]
pub(crate) fn push_blended_edge(
    mesh: &mut LineMesh<3>,
    a: Vec4,
    b: Vec4,
    color_a: [f32; 4],
    color_b: [f32; 4],
    width: f32,
    blend: f32,
    projection: &rye_math::Projection<4>,
    body_pos_r3: Vec3,
    slerp_scratch: &mut Vec<Vec4>,
    view_radius: f32,
) {
    // Flat edge: one straight R3 chord per polytope edge. For stereographic this
    // is a comparison overlay, not the S3 great-circle image.
    if blend <= 0.0 {
        if flat_edge_uses_endpoint_chord(projection) {
            let clip_radius = stereographic_clip_radius(projection, view_radius);
            let a3_local = <rye_math::EuclideanR4 as rye_math::RasterizableSpace<4>>::project_point(
                a, projection,
            );
            let b3_local = <rye_math::EuclideanR4 as rye_math::RasterizableSpace<4>>::project_point(
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
        // Future non-line-preserving flat projection: sample the R⁴ chord.
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
        // Vertex effectively at the body center: no radial direction for slerp, so
        // the sphere arc is undefined and the edge degrades to the flat chord.
        // Never reached in practice (regular polytope vertices are at
        // circumradius `body_size`), but route through the same flat-edge chord
        // policy so degenerate inputs don't invent spherical samples.
        if flat_edge_uses_endpoint_chord(projection) {
            let clip_radius = stereographic_clip_radius(projection, view_radius);
            let a3_local = <rye_math::EuclideanR4 as rye_math::RasterizableSpace<4>>::project_point(
                a, projection,
            );
            let b3_local = <rye_math::EuclideanR4 as rye_math::RasterizableSpace<4>>::project_point(
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
    // Unit endpoints on S³ for the great-circle arc; the per-sample radius lerp
    // below restores the body's scale and keeps the endpoints exactly on `a`/`b`.
    let p0u = a / radius_a;
    let p1u = b / radius_b;
    slerp_scratch.clear();
    <rye_math::SphericalS3Embedded as rye_math::RasterizableSpace<4>>::tessellate_segment(
        p0u,
        p1u,
        samples,
        slerp_scratch,
    );

    // Sample 0 is `a` exactly (flat == sphere == a), so seed `prev` from it and
    // emit consecutive sub-segments from sample 1. `slerp_scratch` holds exactly
    // `samples + 1` points, so skipping the first walks indices 1..=samples.
    // Stereographic clip: a sub-segment straddling the view radius is cut AT the
    // boundary (`push_clipped_subsegment`), so a near-pole arc end rides the
    // clip sphere smoothly under rotation instead of popping between samples; a
    // fully-outside sub-segment is dropped, breaking the polyline across the pole
    // region. `clip_radius == None` keeps every sample whole.
    // `stereographic_view_point` corrects the near-pole denominator-clamp
    // deflation so a sample inside the pole band reads as the point-at-infinity it
    // is (large magnitude, correct direction) rather than collapsing toward the
    // origin; outside the band, and for every other projection, it is exactly
    // `project_point`.
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
