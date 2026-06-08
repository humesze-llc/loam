//! Wireframe color helpers: deterministic per-edge palettes and the w-depth
//! diverging gradient.

// Cool/warm diverging palette for the w-depth color mode, keyed on SIGNED w so
// a +w and a -w vertex read as different colors. Matches the depth cue in the
// LineRasterStaticR4 shader.
const W_DEPTH_BACK: [f32; 3] = [0.30, 0.42, 0.58];
const W_DEPTH_FRONT: [f32; 3] = [1.00, 0.78, 0.45];

/// HSV to RGB; h, s, v and output channels in [0, 1].
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h6 = h.fract() * 6.0;
    let c = v * s;
    let x = c * (1.0 - (h6 % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match h6.floor() as i32 % 6 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [r + m, g + m, b + m]
}

/// Deterministic per-index color: golden-ratio hue spacing plus an S/V cycle so
/// adjacent indices land far apart in HSV.
fn unique_edge_palette_color(idx: usize) -> [f32; 4] {
    const PHI_INV: f32 = 0.618_034;
    let h = ((idx as f32) * PHI_INV).fract();
    let s = 0.78 + 0.18 * ((idx % 3) as f32 / 2.0);
    let v = 0.92 - 0.18 * (((idx / 3) % 2) as f32);
    let [r, g, b] = hsv_to_rgb(h, s, v);
    [r, g, b, 1.0]
}

/// Per-edge RGBA via greedy first-fit coloring of the edge line-graph: edges
/// sharing a vertex get different colors. Deterministic in `edges` order.
pub(crate) fn unique_edge_palette(edges: &[[u32; 2]]) -> Vec<[f32; 4]> {
    let n = edges.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            let [a0, a1] = edges[i];
            let [b0, b1] = edges[j];
            if a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1 {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    let mut color_idx = vec![usize::MAX; n];
    let mut used = std::collections::HashSet::<usize>::new();
    for i in 0..n {
        used.clear();
        for &nbr in &adj[i] {
            if color_idx[nbr] != usize::MAX {
                used.insert(color_idx[nbr]);
            }
        }
        let mut c = 0;
        while used.contains(&c) {
            c += 1;
        }
        color_idx[i] = c;
    }
    color_idx
        .into_iter()
        .map(unique_edge_palette_color)
        .collect()
}

/// Per-vertex w-depth color: cool at -w, warm at +w, neutral on the slice
/// plane. `w_extent_local` is the fixed post-scale |w| bound, so the gradient
/// is orientation-stable.
pub(crate) fn w_depth_color(w: f32, w_extent_local: f32) -> [f32; 4] {
    let denom = w_extent_local.max(1e-6);
    let t = ((w / denom) * 0.5 + 0.5).clamp(0.0, 1.0);
    [
        W_DEPTH_BACK[0] + (W_DEPTH_FRONT[0] - W_DEPTH_BACK[0]) * t,
        W_DEPTH_BACK[1] + (W_DEPTH_FRONT[1] - W_DEPTH_BACK[1]) * t,
        W_DEPTH_BACK[2] + (W_DEPTH_FRONT[2] - W_DEPTH_BACK[2]) * t,
        1.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hsv_to_rgb_primaries() {
        let red = hsv_to_rgb(0.0, 1.0, 1.0);
        assert!((red[0] - 1.0).abs() < 1e-5);
        assert!(red[1].abs() < 1e-5);
        assert!(red[2].abs() < 1e-5);

        let green = hsv_to_rgb(1.0 / 3.0, 1.0, 1.0);
        assert!(green[0].abs() < 1e-5);
        assert!((green[1] - 1.0).abs() < 1e-5);
        assert!(green[2].abs() < 1e-5);

        let blue = hsv_to_rgb(2.0 / 3.0, 1.0, 1.0);
        assert!(blue[0].abs() < 1e-5);
        assert!(blue[1].abs() < 1e-5);
        assert!((blue[2] - 1.0).abs() < 1e-5);
    }

    /// Zero saturation is gray regardless of hue.
    #[test]
    fn hsv_to_rgb_zero_saturation_is_gray() {
        for h in [0.0, 0.25, 0.5, 0.75, 0.999_f32] {
            let rgb = hsv_to_rgb(h, 0.0, 0.7);
            assert!((rgb[0] - 0.7).abs() < 1e-5, "h={h}: r should be 0.7");
            assert!((rgb[1] - 0.7).abs() < 1e-5, "h={h}: g should be 0.7");
            assert!((rgb[2] - 0.7).abs() < 1e-5, "h={h}: b should be 0.7");
        }
    }

    /// Zero value is black regardless of hue/saturation.
    #[test]
    fn hsv_to_rgb_zero_value_is_black() {
        let rgb = hsv_to_rgb(0.5, 0.8, 0.0);
        assert!(rgb.iter().all(|c| c.abs() < 1e-5));
    }

    /// Edges sharing a vertex must get distinct colors (the mode's invariant).
    #[test]
    fn unique_edge_palette_separates_adjacent_edges() {
        let edges: &[[u32; 2]] = &[[0, 1], [0, 2], [0, 3]];
        let palette = unique_edge_palette(edges);
        assert_eq!(palette.len(), 3, "one color per edge");
        for i in 0..palette.len() {
            for j in (i + 1)..palette.len() {
                assert_ne!(
                    palette[i], palette[j],
                    "edges {i} and {j} share vertex 0; palette must differ"
                );
            }
        }
    }

    /// Non-adjacent edges may reuse a color (greedy first-fit).
    #[test]
    fn unique_edge_palette_non_adjacent_edges_may_share_color() {
        let edges: &[[u32; 2]] = &[[0, 1], [2, 3]];
        let palette = unique_edge_palette(edges);
        assert_eq!(palette.len(), 2);
        assert_eq!(palette[0], palette[1]);
    }

    /// Deterministic in edge order (the palette is cached per `Polytope4`).
    #[test]
    fn unique_edge_palette_is_deterministic() {
        let edges: &[[u32; 2]] = &[[0, 1], [1, 2], [2, 3], [0, 3]];
        let a = unique_edge_palette(edges);
        let b = unique_edge_palette(edges);
        assert_eq!(a, b);
    }

    #[test]
    fn w_depth_color_zero_w_is_midpoint() {
        let c = w_depth_color(0.0, 1.0);
        for ch in 0..3 {
            let expected = (W_DEPTH_BACK[ch] + W_DEPTH_FRONT[ch]) * 0.5;
            assert!(
                (c[ch] - expected).abs() < 1e-5,
                "channel {ch}: expected {expected}, got {}",
                c[ch],
            );
        }
        assert!((c[3] - 1.0).abs() < 1e-5, "alpha is 1.0");
    }

    #[test]
    fn w_depth_color_neg_extent_is_back() {
        let c = w_depth_color(-1.0, 1.0);
        for ch in 0..3 {
            assert!(
                (c[ch] - W_DEPTH_BACK[ch]).abs() < 1e-5,
                "channel {ch}: expected back tint",
            );
        }
    }

    #[test]
    fn w_depth_color_pos_extent_is_front() {
        let c = w_depth_color(1.0, 1.0);
        for ch in 0..3 {
            assert!(
                (c[ch] - W_DEPTH_FRONT[ch]).abs() < 1e-5,
                "channel {ch}: expected front tint",
            );
        }
    }

    /// Past the extent, clamp to the endpoint color instead of over-saturating.
    #[test]
    fn w_depth_color_clamps_past_extent() {
        let c_far = w_depth_color(5.0, 1.0);
        let c_at_extent = w_depth_color(1.0, 1.0);
        assert_eq!(c_far, c_at_extent, "+w past extent should clamp to front");

        let c_back = w_depth_color(-5.0, 1.0);
        let c_at_neg_extent = w_depth_color(-1.0, 1.0);
        assert_eq!(
            c_back, c_at_neg_extent,
            "-w past extent should clamp to back"
        );
    }
}
