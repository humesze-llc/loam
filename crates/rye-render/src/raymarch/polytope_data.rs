//! WGSL emission for the 120-cell and 600-cell SDFs.
//!
//! These two polytopes have face-hyperplane sets large enough (120 and 600 respectively) that
//! hand-writing them as inline WGSL literals is impractical. Instead we generate the WGSL at
//! runtime from the Rust face-plane / vertex generators in `rye_physics::euclidean_r4`.
//!
//! The emitted WGSL fragment defines:
//!
//! - `CELL_INRADIUS_UNIT: f32`: shared inradius constant for both
//!   polytopes at unit circumradius (`φ²/(2√2)`).
//! - `CELL120_FACE_NORMALS: array<vec4<f32>, 120>`: 120-cell face
//!   directions (= 600-cell vertex set).
//! - `CELL600_FACE_NORMALS: array<vec4<f32>, 600>`: 600-cell face
//!   directions (= 120-cell vertex set).
//! - `CELL120_VERTICES: array<vec4<f32>, 600>`: 120-cell vertex set,
//!   used by the |S|=4 vertex-lookup branch of the 120-cell Wolfe SDF.
//! - `CELL600_VERTICES: array<vec4<f32>, 120>`: analogous for
//!   600-cell.
//! - `cell120_sdf_local(p: vec4<f32>) -> f32`: true-Euclidean SDF.
//! - `cell600_sdf_local(p: vec4<f32>) -> f32`: true-Euclidean SDF.
//!
//! Both SDFs use Wolfe's greedy hyperplane projection (matching the CPU port in
//! `rye_physics::euclidean_r4::polytope_sdf_wolfe`):
//!   - |S|=1: project onto closest face plane.
//!   - |S|=2: 2x2 Lagrange-multiplier solve.
//!   - |S|=3: 3x3 Lagrange-multiplier solve via cofactor expansion.
//!   - |S|=4: closest polytope vertex (the 4-plane intersection IS a
//!     vertex; brute-force search the vertex array).
//!
//! Total emitted size: ~24 KB of WGSL (mostly the const-array literals).

use std::fmt::Write;

use glam::Vec4;
use rye_physics::euclidean_r4::{
    cell120_face_planes, cell120_vertices, cell600_face_planes, cell600_vertices,
    icosian_inradius_unit,
};

/// No-op stub WGSL satisfying the kernel's `cell120_sdf_local` / `cell600_sdf_local` symbol
/// references when the scene doesn't use either polytope. Returns `+1e9` (invisible far-away
/// surface) so the dispatch branches are inert at runtime.
///
/// The kernel's `body_polytope_sdf_4d` always references both function names: naga rejects the
/// WGSL otherwise: so callers must include either this stub or [`polytope_extended_sdfs_wgsl`].
/// The stub is tiny (~150 bytes) and adds zero register pressure; prefer it when the scene
/// contains no 120-cell or 600-cell bodies.
pub fn polytope_stub_sdfs_wgsl() -> &'static str {
    "// ---- Polytope stub SDFs (no 120-cell/600-cell bodies in scene) ----\n\
     fn cell120_sdf_local(p: vec4<f32>) -> f32 { return 1.0e9; }\n\
     fn cell600_sdf_local(p: vec4<f32>) -> f32 { return 1.0e9; }\n"
}

/// Emit the full WGSL fragment for the 120-cell and 600-cell SDFs. Append this to the
/// hyperslice4d kernel before naga validation.
///
/// Includes ~24 KB of `const` array data (face normals + vertex sets for both polytopes). On
/// some GPU drivers this constant data competes with scalar registers and slows ALL
/// pixel-shader work, even when the cell120/cell600 dispatch branches are never reached. If
/// your scene has no 120-cell or 600-cell bodies, prefer [`polytope_stub_sdfs_wgsl`] instead.
pub fn polytope_extended_sdfs_wgsl() -> String {
    let mut s = String::with_capacity(32 * 1024);
    s.push_str("// ---- Extended polytope SDFs (120-cell, 600-cell) ----\n");
    writeln!(
        s,
        "const CELL_INRADIUS_UNIT: f32 = {:.10};",
        icosian_inradius_unit()
    )
    .unwrap();

    let (n120, _) = cell120_face_planes();
    let (n600, _) = cell600_face_planes();
    let v120 = cell120_vertices(1.0);
    let v600 = cell600_vertices(1.0);

    emit_vec4_array(&mut s, "CELL120_FACE_NORMALS", &n120);
    emit_vec4_array(&mut s, "CELL600_FACE_NORMALS", &n600);
    emit_vec4_array(&mut s, "CELL120_VERTICES", &v120);
    emit_vec4_array(&mut s, "CELL600_VERTICES", &v600);

    s.push_str(WOLFE_PROJECTION_HELPER_WGSL);
    s.push_str(&wolfe_sdf_function(
        "cell120_sdf_local",
        "CELL120_FACE_NORMALS",
        120,
        "CELL120_VERTICES",
        600,
    ));
    s.push_str(&wolfe_sdf_function(
        "cell600_sdf_local",
        "CELL600_FACE_NORMALS",
        600,
        "CELL600_VERTICES",
        120,
    ));

    s
}

fn emit_vec4_array(out: &mut String, name: &str, data: &[Vec4]) {
    writeln!(
        out,
        "const {name}: array<vec4<f32>, {len}> = array(",
        len = data.len()
    )
    .unwrap();
    for v in data {
        writeln!(
            out,
            "    vec4<f32>({:.10}, {:.10}, {:.10}, {:.10}),",
            v.x, v.y, v.z, v.w
        )
        .unwrap();
    }
    out.push_str(");\n");
}

/// Project `p` onto the intersection of `count` (1..=3) active hyperplanes
/// (`dot(active[i], q) = inradius`) via Lagrange multipliers. Mirrors
/// `rye_physics::euclidean_r4::project_onto_active_planes` 1:1 for the |S|=1, 2, 3 cases;
/// |S|=4 is handled by the per-polytope SDF via vertex lookup.
const WOLFE_PROJECTION_HELPER_WGSL: &str = r#"
fn polytope_project_active(
    p: vec4<f32>,
    a0: vec4<f32>, a1: vec4<f32>, a2: vec4<f32>,
    count: u32,
    inradius: f32,
) -> vec4<f32> {
    let b0 = dot(a0, p) - inradius;
    let b1 = dot(a1, p) - inradius;
    let b2 = dot(a2, p) - inradius;
    if (count == 1u) {
        return p - b0 * a0;
    }
    if (count == 2u) {
        let g01 = dot(a0, a1);
        let det = 1.0 - g01 * g01;
        if (abs(det) < 1.0e-9) { return p; }
        let inv_det = 1.0 / det;
        let l0 = (b0 - g01 * b1) * inv_det;
        let l1 = (b1 - g01 * b0) * inv_det;
        return p - l0 * a0 - l1 * a1;
    }
    // count == 3: 3x3 solve via cofactor expansion of the symmetric
    // Gram matrix (diagonals = 1 for unit normals).
    let g01 = dot(a0, a1);
    let g02 = dot(a0, a2);
    let g12 = dot(a1, a2);
    let det = 1.0 + 2.0 * g01 * g02 * g12 - g01 * g01 - g02 * g02 - g12 * g12;
    if (abs(det) < 1.0e-9) { return p; }
    let inv_det = 1.0 / det;
    let c00 = 1.0 - g12 * g12;
    let c01 = g02 * g12 - g01;
    let c02 = g01 * g12 - g02;
    let c11 = 1.0 - g02 * g02;
    let c12 = g01 * g02 - g12;
    let c22 = 1.0 - g01 * g01;
    let l0 = (c00 * b0 + c01 * b1 + c02 * b2) * inv_det;
    let l1 = (c01 * b0 + c11 * b1 + c12 * b2) * inv_det;
    let l2 = (c02 * b0 + c12 * b1 + c22 * b2) * inv_det;
    return p - l0 * a0 - l1 * a1 - l2 * a2;
}
"#;

/// Generate a Wolfe-SDF function for a specific polytope. The function reads from
/// `face_normals_name` (size `face_count`) for plane queries and `vertices_name` (size
/// `vertex_count`) for the |S|=4 fallback.
fn wolfe_sdf_function(
    fn_name: &str,
    face_normals_name: &str,
    face_count: u32,
    vertices_name: &str,
    vertex_count: u32,
) -> String {
    format!(
        r#"
fn {fn_name}(p: vec4<f32>) -> f32 {{
    let inradius = CELL_INRADIUS_UNIT;
    // Phase 1: max plane distance + initial active plane.
    var max_d: f32 = -1.0e9;
    var max_i: u32 = 0u;
    for (var i: u32 = 0u; i < {face_count}u; i = i + 1u) {{
        let d = dot({face_normals_name}[i], p) - inradius;
        if (d > max_d) {{
            max_d = d;
            max_i = i;
        }}
    }}
    if (max_d <= 0.0) {{ return max_d; }}

    // Outside-circumsphere fast-path: max-plane distance is a
    // Lipschitz-1 lower bound on the true SDF; tight in face-Voronoi
    // regions, slightly loose in corner regions but still safe for
    // sphere-tracing convergence. Skip Wolfe iteration here so distant
    // grazing rays don't pay the full 4-level projection cost.
    if (dot(p, p) > 1.0) {{ return max_d; }}

    var active_idx_0: u32 = max_i;
    var active_idx_1: u32 = 0u;
    var active_idx_2: u32 = 0u;
    var active_count: u32 = 1u;
    var a0: vec4<f32> = {face_normals_name}[max_i];
    var a1: vec4<f32> = vec4<f32>(0.0);
    var a2: vec4<f32> = vec4<f32>(0.0);
    var q: vec4<f32> = p - max_d * a0;
    let tol: f32 = 1.0e-6;

    // Iteratively add the next-most-violated plane (max 3 more times).
    for (var iter: u32 = 0u; iter < 3u; iter = iter + 1u) {{
        var next_d: f32 = tol;
        var next_i: u32 = 0xffffffffu;
        for (var i: u32 = 0u; i < {face_count}u; i = i + 1u) {{
            if (i == active_idx_0) {{ continue; }}
            if (active_count >= 2u && i == active_idx_1) {{ continue; }}
            if (active_count >= 3u && i == active_idx_2) {{ continue; }}
            let d = dot({face_normals_name}[i], q) - inradius;
            if (d > next_d) {{
                next_d = d;
                next_i = i;
            }}
        }}
        if (next_i == 0xffffffffu) {{
            return length(p - q);
        }}
        if (active_count == 1u) {{
            active_idx_1 = next_i;
            a1 = {face_normals_name}[next_i];
            active_count = 2u;
            q = polytope_project_active(p, a0, a1, a2, 2u, inradius);
        }} else if (active_count == 2u) {{
            active_idx_2 = next_i;
            a2 = {face_normals_name}[next_i];
            active_count = 3u;
            q = polytope_project_active(p, a0, a1, a2, 3u, inradius);
        }} else {{
            // |S|=4: closest point is a polytope vertex. Brute-force
            // search the vertex array. (Avoids a 4x4 matrix inverse in
            // WGSL.)
            var best_d2: f32 = 1.0e30;
            for (var j: u32 = 0u; j < {vertex_count}u; j = j + 1u) {{
                let dv = p - {vertices_name}[j];
                let d2 = dot(dv, dv);
                if (d2 < best_d2) {{ best_d2 = d2; }}
            }}
            return sqrt(best_d2);
        }}
    }}
    return length(p - q);
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The emitted WGSL fragment parses and validates against naga. Catches any drift between
    /// the emit shape and WGSL syntax.
    #[test]
    fn polytope_extended_sdfs_wgsl_validates() {
        let wgsl = polytope_extended_sdfs_wgsl();
        // Wrap with a minimal compute shader that calls each SDF, so naga
        // has an entry point to anchor validation against.
        let probe = format!(
            "{wgsl}\n\
             @compute @workgroup_size(1) fn main() {{\n\
             let p = vec4<f32>(0.5, 0.0, 0.0, 0.0);\n\
             _ = cell120_sdf_local(p);\n\
             _ = cell600_sdf_local(p);\n\
             }}\n"
        );
        let module = naga::front::wgsl::parse_str(&probe)
            .unwrap_or_else(|e| panic!("WGSL parse failed: {e}\n--- source ---\n{probe}"));
        let flags = naga::valid::ValidationFlags::all();
        let caps = naga::valid::Capabilities::empty();
        naga::valid::Validator::new(flags, caps)
            .validate(&module)
            .expect("polytope WGSL should validate");
    }

    /// Sanity-check the array sizes match the expected polytope counts.
    #[test]
    fn emitted_wgsl_has_expected_array_sizes() {
        let wgsl = polytope_extended_sdfs_wgsl();
        assert!(wgsl.contains("array<vec4<f32>, 120>"));
        assert!(wgsl.contains("array<vec4<f32>, 600>"));
        assert!(wgsl.contains("CELL120_FACE_NORMALS"));
        assert!(wgsl.contains("CELL600_FACE_NORMALS"));
        assert!(wgsl.contains("CELL120_VERTICES"));
        assert!(wgsl.contains("CELL600_VERTICES"));
        assert!(wgsl.contains("fn cell120_sdf_local"));
        assert!(wgsl.contains("fn cell600_sdf_local"));
    }

    /// The stub WGSL also parses + validates against naga, satisfying the kernel's symbol
    /// references with no const-array overhead.
    #[test]
    fn polytope_stub_sdfs_wgsl_validates() {
        let wgsl = polytope_stub_sdfs_wgsl();
        let probe = format!(
            "{wgsl}\n\
             @compute @workgroup_size(1) fn main() {{\n\
             let p = vec4<f32>(0.5, 0.0, 0.0, 0.0);\n\
             _ = cell120_sdf_local(p);\n\
             _ = cell600_sdf_local(p);\n\
             }}\n"
        );
        let module = naga::front::wgsl::parse_str(&probe).expect("stub must parse");
        let flags = naga::valid::ValidationFlags::all();
        let caps = naga::valid::Capabilities::empty();
        naga::valid::Validator::new(flags, caps)
            .validate(&module)
            .expect("stub must validate");
    }
}
