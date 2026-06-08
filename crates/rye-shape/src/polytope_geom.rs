//! Convex regular 4-polytope vertex / face-plane generators and the Wolfe SDF.
//! Factored out of the physics layer so the rasterizer and topology can build
//! shapes without depending on rye-physics. Origin-centered, circumradius `r`.

use glam::Vec4;

// 4D regular polytopes. Every generator returns origin-centered vertices scaled
// so the circumradius equals `r`; the caller translates.

/// **5-cell / pentatope** (4D simplex): 5 vertices, 10 edges, 10 faces, 5
/// tetrahedral cells. The 4D analogue of the tetrahedron.
pub fn pentatope_vertices(r: f32) -> Vec<Vec4> {
    // Regular tetrahedron in the `w = -r/4` hyperplane plus an apex at
    // `(0, 0, 0, r)`, base circumradius `r·sqrt(15)/4` chosen for equal edges.
    let k = r;
    let base_w = -r * 0.25;
    let base_r = r * (15.0_f32).sqrt() / 4.0;
    // Use a regular tetrahedron's vertex set for the base, scaled.
    let t = base_r / 3.0_f32.sqrt();
    vec![
        Vec4::new(0.0, 0.0, 0.0, k),
        Vec4::new(t, t, t, base_w),
        Vec4::new(t, -t, -t, base_w),
        Vec4::new(-t, t, -t, base_w),
        Vec4::new(-t, -t, t, base_w),
    ]
}

/// **Tesseract / 8-cell** (hypercube): 16 vertices, 32 edges, 24 square faces,
/// 8 cubic cells. Vertices `(±r/2, ±r/2, ±r/2, ±r/2)` give circumradius `r`.
pub fn tesseract_vertices(r: f32) -> Vec<Vec4> {
    let a = r * 0.5;
    let mut v = Vec::with_capacity(16);
    for &w in &[-a, a] {
        for &z in &[-a, a] {
            for &y in &[-a, a] {
                for &x in &[-a, a] {
                    v.push(Vec4::new(x, y, z, w));
                }
            }
        }
    }
    v
}

/// **16-cell / hexadecachoron** (cross-polytope): 8 vertices, 24 edges, 32
/// triangular faces, 16 tetrahedral cells. Vertices are `±r` on each axis.
pub fn cell16_vertices(r: f32) -> Vec<Vec4> {
    vec![
        Vec4::new(r, 0.0, 0.0, 0.0),
        Vec4::new(-r, 0.0, 0.0, 0.0),
        Vec4::new(0.0, r, 0.0, 0.0),
        Vec4::new(0.0, -r, 0.0, 0.0),
        Vec4::new(0.0, 0.0, r, 0.0),
        Vec4::new(0.0, 0.0, -r, 0.0),
        Vec4::new(0.0, 0.0, 0.0, r),
        Vec4::new(0.0, 0.0, 0.0, -r),
    ]
}

/// **24-cell / icositetrachoron**: 24 vertices, 96 edges, 96 triangle faces, 24
/// octahedral cells. Unique to 4D (no 3D analogue). Vertices are all 24
/// permutations of `(±r/√2, ±r/√2, 0, 0)`.
pub fn cell24_vertices(r: f32) -> Vec<Vec4> {
    let k = r / 2.0_f32.sqrt();
    let mut v = Vec::with_capacity(24);
    for i in 0..4 {
        for j in (i + 1)..4 {
            for &si in &[-k, k] {
                for &sj in &[-k, k] {
                    let mut c = [0.0_f32; 4];
                    c[i] = si;
                    c[j] = sj;
                    v.push(Vec4::new(c[0], c[1], c[2], c[3]));
                }
            }
        }
    }
    v
}

/// The 12 even permutations (alternating group A₄) of a 4-tuple, used by the
/// 600-cell and 120-cell vertex orbits. Result `[i]` is `arr[σ(i)]`.
fn even_permutations_4<T: Copy>(arr: [T; 4]) -> [[T; 4]; 12] {
    [
        // Identity
        [arr[0], arr[1], arr[2], arr[3]],
        // 3-cycles (8 of them)
        [arr[1], arr[2], arr[0], arr[3]], // (012)
        [arr[2], arr[0], arr[1], arr[3]], // (021)
        [arr[1], arr[3], arr[2], arr[0]], // (013)
        [arr[3], arr[0], arr[2], arr[1]], // (031)
        [arr[2], arr[1], arr[3], arr[0]], // (023)
        [arr[3], arr[1], arr[0], arr[2]], // (032)
        [arr[0], arr[2], arr[3], arr[1]], // (123)
        [arr[0], arr[3], arr[1], arr[2]], // (132)
        // (2,2)-cycles (3 of them)
        [arr[1], arr[0], arr[3], arr[2]], // (01)(23)
        [arr[2], arr[3], arr[0], arr[1]], // (02)(13)
        [arr[3], arr[2], arr[1], arr[0]], // (03)(12)
    ]
}

/// **600-cell / hexacosichoron**: 120 vertices, 720 edges, 1200 triangle faces,
/// 600 tetrahedral cells. H₄ symmetry; dual of the 120-cell.
///
/// Vertex set at circumradius 1 (Wikipedia "600-cell"): 8 axial `(±1, 0, 0, 0)`,
/// 16 `(±1/2, ±1/2, ±1/2, ±1/2)`, and 96 even permutations of
/// `(0, ±1/2, ±φ/2, ±1/(2φ))`. Total 120.
pub fn cell600_vertices(r: f32) -> Vec<Vec4> {
    let phi = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let mut v = Vec::with_capacity(120);

    // 8 axial.
    for axis in 0..4 {
        for sign in [r, -r] {
            let mut c = [0.0_f32; 4];
            c[axis] = sign;
            v.push(Vec4::from_array(c));
        }
    }

    // 16 half-tesseract corners.
    let h = r * 0.5;
    for s in 0..16u32 {
        let x = if s & 1 == 1 { -h } else { h };
        let y = if (s >> 1) & 1 == 1 { -h } else { h };
        let z = if (s >> 2) & 1 == 1 { -h } else { h };
        let w = if (s >> 3) & 1 == 1 { -h } else { h };
        v.push(Vec4::new(x, y, z, w));
    }

    // 96 even permutations of (0, ±r/2, ±rφ/2, ±r/(2φ)).
    let base = [0.0_f32, r * 0.5, r * phi * 0.5, r / (2.0 * phi)];
    for perm in even_permutations_4(base) {
        for sign_mask in 0..8u32 {
            let mut x = perm;
            let mut k = 0usize;
            for xi in x.iter_mut() {
                if *xi != 0.0 {
                    if (sign_mask >> k) & 1 == 1 {
                        *xi = -*xi;
                    }
                    k += 1;
                }
            }
            v.push(Vec4::from_array(x));
        }
    }

    v
}

/// **120-cell / hecatonicosachoron**: 600 vertices, 1200 edges, 720 pentagonal
/// faces, 120 dodecahedral cells. H₄ symmetry; dual of the 600-cell.
///
/// Vertex set at circumradius `2√2` before rescaling (Wikipedia "120-cell"):
/// - 24: permutations of `(0, 0, ±2, ±2)`.
/// - 64 each: `(±1, ±1, ±1, ±√5)`, `(±1/φ, ±1/φ, ±1/φ, ±φ²)`,
///   `(±1/φ², ±φ, ±φ, ±φ)`, with the odd-one-out in any of 4 positions.
/// - 96 each: even perms of `(0, ±1/φ², ±1, ±φ²)` and `(0, ±1/φ, ±φ, ±√5)`.
/// - 192: even permutations of `(±1/φ, ±1, ±φ, ±2)`.
pub fn cell120_vertices(r: f32) -> Vec<Vec4> {
    let phi = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let phi2 = phi * phi;
    let inv_phi = 1.0 / phi;
    let inv_phi2 = inv_phi * inv_phi;
    let sqrt5 = 5.0_f32.sqrt();
    let scale = r / (2.0 * 2.0_f32.sqrt());
    let mut v = Vec::with_capacity(600);

    // 24 permutations of (0, 0, ±2, ±2).
    for i in 0..4 {
        for j in (i + 1)..4 {
            for si in [2.0_f32, -2.0] {
                for sj in [2.0_f32, -2.0] {
                    let mut c = [0.0_f32; 4];
                    c[i] = si * scale;
                    c[j] = sj * scale;
                    v.push(Vec4::from_array(c));
                }
            }
        }
    }

    // `special` at one of 4 positions, `common` at the other 3, all signs
    // independent: 4·16 = 64 vertices.
    let mut emit_one_special = |special: f32, common: f32| {
        for special_pos in 0..4 {
            for sm in 0..16u32 {
                let mut c = [0.0_f32; 4];
                for (i, ci) in c.iter_mut().enumerate() {
                    let val = if i == special_pos { special } else { common };
                    let sign = if (sm >> i) & 1 == 1 { -1.0 } else { 1.0 };
                    *ci = val * sign * scale;
                }
                v.push(Vec4::from_array(c));
            }
        }
    };

    emit_one_special(sqrt5, 1.0);
    emit_one_special(phi2, inv_phi);
    emit_one_special(inv_phi2, phi);

    // Even permutations of (0, ±a, ±b, ±c), independent signs: 12·8 = 96.
    let mut emit_even_zero = |a: f32, b: f32, c: f32| {
        let base = [0.0_f32, a, b, c];
        for perm in even_permutations_4(base) {
            for sign_mask in 0..8u32 {
                let mut x = perm;
                let mut k = 0usize;
                for xi in x.iter_mut() {
                    if *xi != 0.0 {
                        if (sign_mask >> k) & 1 == 1 {
                            *xi = -*xi;
                        }
                        k += 1;
                    }
                }
                for ci in &mut x {
                    *ci *= scale;
                }
                v.push(Vec4::from_array(x));
            }
        }
    };

    emit_even_zero(inv_phi2, 1.0, phi2);
    emit_even_zero(inv_phi, phi, sqrt5);

    // 192 even permutations of (±1/φ, ±1, ±φ, ±2): 12·16 = 192.
    let base7 = [inv_phi, 1.0, phi, 2.0_f32];
    for perm in even_permutations_4(base7) {
        for sm in 0..16u32 {
            let mut x = perm;
            for (i, xi) in x.iter_mut().enumerate() {
                if (sm >> i) & 1 == 1 {
                    *xi = -*xi;
                }
            }
            for ci in &mut x {
                *ci *= scale;
            }
            v.push(Vec4::from_array(x));
        }
    }

    v
}

/// Inradius of the 120-cell and 600-cell at unit circumradius. Equal by polar
/// duality: `φ² / (2√2) = (3 + √5) / (4√2) ≈ 0.92562`.
pub fn icosian_inradius_unit() -> f32 {
    let phi = (1.0 + 5.0_f32.sqrt()) * 0.5;
    phi * phi / (2.0 * 2.0_f32.sqrt())
}

/// Face hyperplanes of the 120-cell at unit circumradius. Returns `(normals,
/// offset)`: 120 unit normals (the 600-cell's vertices, the dual) and the
/// inradius. Inside iff `dot(n_i, p) <= offset` for all i.
// BUG: dual-vertex normals are exact for the 24 axial + 16 tesseract-corner
// orbits but only approximate for the 96 golden-ratio orbits, so the SDF
// surface is a slightly-truncated 120-cell. Forward path: rasterized
// cross-section faces replace the SDF for this polytope's surface.
pub fn cell120_face_planes() -> (Vec<Vec4>, f32) {
    (cell600_vertices(1.0), icosian_inradius_unit())
}

/// Face hyperplanes of the 600-cell at unit circumradius. Returns `(normals,
/// offset)`: the 600 unit normals are the 120-cell's vertices.
// BUG: same approximation as [`cell120_face_planes`]. The true normals are the
// 600 tetrahedral-cell centroids; the dual vertices diverge on the 96
// golden-ratio orbits. Same rasterized-section forward path.
pub fn cell600_face_planes() -> (Vec<Vec4>, f32) {
    (cell120_vertices(1.0), icosian_inradius_unit())
}

/// Exact signed Euclidean distance from `p` to a convex polytope of
/// uniform-distance face hyperplanes (`dot(n_i, x) = inradius`, `n_i` unit).
///
/// Wolfe's greedy hyperplane projection: add the most-violated plane to the
/// active set, project onto the intersection via Lagrange multipliers, repeat
/// until no violations remain or |S|=4 (a vertex).
pub fn polytope_sdf_wolfe(p: Vec4, face_normals: &[Vec4], inradius: f32) -> f32 {
    let mut max_d = f32::NEG_INFINITY;
    let mut active_idx = [0usize; 4];
    for (i, n) in face_normals.iter().enumerate() {
        let d = n.dot(p) - inradius;
        if d > max_d {
            max_d = d;
            active_idx[0] = i;
        }
    }
    if max_d <= 0.0 {
        return max_d; // inside
    }
    let mut active_count = 1usize;

    // Cache active normals so the 4-level projection doesn't re-index.
    let mut active = [Vec4::ZERO; 4];
    active[0] = face_normals[active_idx[0]];

    let tol = 1e-6_f32;
    let mut q = p - max_d * active[0]; // |S|=1 projection

    while active_count < 4 {
        let mut next_d = tol;
        let mut next_i = usize::MAX;
        for (i, n) in face_normals.iter().enumerate() {
            if active_idx[..active_count].contains(&i) {
                continue;
            }
            let d = n.dot(q) - inradius;
            if d > next_d {
                next_d = d;
                next_i = i;
            }
        }
        if next_i == usize::MAX {
            return (p - q).length();
        }
        active_idx[active_count] = next_i;
        active[active_count] = face_normals[next_i];
        active_count += 1;
        q = project_onto_active_planes(p, &active, active_count, inradius);
    }
    (p - q).length()
}

/// Project `p` onto the intersection of `count` active hyperplanes (`dot(active[i], x) =
/// inradius` for `i in 0..count`) via Lagrange multipliers. Solves `G λ = b` where `G` is the
/// Gram matrix of the active normals; closed-form for each `count` in `1..=4`.
fn project_onto_active_planes(p: Vec4, active: &[Vec4; 4], count: usize, inradius: f32) -> Vec4 {
    let b = [
        active[0].dot(p) - inradius,
        active[1].dot(p) - inradius,
        active[2].dot(p) - inradius,
        active[3].dot(p) - inradius,
    ];
    match count {
        1 => p - b[0] * active[0],
        2 => {
            // 2x2 Gram matrix; unit normals so diagonals = 1.
            let g01 = active[0].dot(active[1]);
            let det = 1.0 - g01 * g01;
            if det.abs() < 1e-9 {
                return p;
            }
            let inv_det = 1.0 / det;
            let l0 = (b[0] - g01 * b[1]) * inv_det;
            let l1 = (b[1] - g01 * b[0]) * inv_det;
            p - l0 * active[0] - l1 * active[1]
        }
        3 => {
            // 3x3 Gram matrix. Symmetric, with unit-normal diagonals.
            let g01 = active[0].dot(active[1]);
            let g02 = active[0].dot(active[2]);
            let g12 = active[1].dot(active[2]);
            let det = 1.0 + 2.0 * g01 * g02 * g12 - g01 * g01 - g02 * g02 - g12 * g12;
            if det.abs() < 1e-9 {
                return p;
            }
            let inv_det = 1.0 / det;
            // Cofactors of the 3x3 symmetric matrix.
            let c00 = 1.0 - g12 * g12;
            let c01 = g02 * g12 - g01;
            let c02 = g01 * g12 - g02;
            let c11 = 1.0 - g02 * g02;
            let c12 = g01 * g02 - g12;
            let c22 = 1.0 - g01 * g01;
            let l0 = (c00 * b[0] + c01 * b[1] + c02 * b[2]) * inv_det;
            let l1 = (c01 * b[0] + c11 * b[1] + c12 * b[2]) * inv_det;
            let l2 = (c02 * b[0] + c12 * b[1] + c22 * b[2]) * inv_det;
            p - l0 * active[0] - l1 * active[1] - l2 * active[2]
        }
        4 => {
            // The 4-plane intersection is a single vertex q = M⁻¹·inradius·1 with
            // M.row(i) = active[i]. glam::Mat4 is column-major, so transpose at
            // build: column j holds component j of each active normal.
            let m = glam::Mat4::from_cols(
                Vec4::new(active[0].x, active[1].x, active[2].x, active[3].x),
                Vec4::new(active[0].y, active[1].y, active[2].y, active[3].y),
                Vec4::new(active[0].z, active[1].z, active[2].z, active[3].z),
                Vec4::new(active[0].w, active[1].w, active[2].w, active[3].w),
            );
            if m.determinant().abs() < 1e-9 {
                return p;
            }
            m.inverse() * Vec4::splat(inradius)
        }
        _ => p,
    }
}
