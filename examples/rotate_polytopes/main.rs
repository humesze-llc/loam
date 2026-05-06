//! Interactive demo of 4D rotation over `Hyperslice4DNode`. Renders
//! a row of convex regular polychora (5-cell, tesseract, 16-cell,
//! 24-cell by default; 120-cell and 600-cell selectable via
//! `--shapes` or the in-app `+` button) on a 4D `y = 0` floor,
//! with `w`-slice scrubbing and two UIs for composing arbitrary
//! 4D rotations.
//!
//! In **Active set** mode the user toggles individual rotation
//! planes (1..6 -> xy, xz, xw, yz, yw, zw); active planes'
//! bivectors sum into the per-frame angular velocity, which
//! integrates into a rotor via `(ω · dt).exp()`. Sum-of-bivectors
//! composition is commutative, so toggle order doesn't matter and
//! the result is always predictable from the visible active set.
//!
//! In **Composer** mode the user builds a sequence of `RotorTerm`s
//! (each a sum of planes with an optional scalar magnitude),
//! reorders them with drag-and-drop, and either applies them as a
//! one-shot rotor multiplication or feeds the seq into the
//! continuous-spin angular velocity.
//!
//! All six convex regular 4-polytopes ship; the 120-cell and
//! 600-cell use a Rust-side face-hyperplane generator (their orbit
//! sets are too large to inline as WGSL literals). Their SDFs run
//! a true-Euclidean Wolfe greedy hyperplane projection, not a
//! max-plane lower bound.
//!
//! All live state and controls help are drawn as a `rye-egui`
//! overlay via the `App::ui` hook.
//!
//! ## Controls
//!
//! - **Mouse left-drag**: orbit camera.
//! - **Up / Down arrows**: scrub `w`-slice (0.5 u/s).
//! - **Space / T**: toggle 4D rotation (pause/resume freezes
//!   orientation in place, does NOT snap back to identity).
//! - **1..6**: toggle the corresponding rotation plane on/off.
//!   The mapping is `1=xy, 2=xz, 3=xw, 4=yz, 5=yw, 6=zw`. Active
//!   planes' bivectors sum into the angular velocity. Famous
//!   compositions: `3` alone = single xw stretch; `3+4` =
//!   isoclinic xw+yz; `3+5+6` = three w-planes drift through
//!   SO(4). Pure-3D combinations (`1+2+4`) just rotate the
//!   cross-section as a rigid 3D shape.
//! - **R**: full reset, slice, rate, all toggles off, AND
//!   orientation back to canonical pose.
//! - **H**: toggle the bottom-overlay expanded section.
//! - **Esc**: exit.
//!
//! ## CLI
//!
//! - `--shapes name1 name2 ...`: choose the polytopes to render
//!   in left-to-right order. Names accepted include the math form
//!   (`5-cell`, `tesseract`, `16-cell`, `24-cell`, `120-cell`,
//!   `600-cell`) and Platonic-slice aliases (`tetrahedron`, `cube`,
//!   `octahedron`, `cuboctahedron`, `dodecahedron`, `icosahedron`).

use anyhow::{anyhow, Result};
use glam::{Vec3, Vec4};
use rye_app::{egui, run_with_config, App, Camera, FrameCtx, OrbitController, RunConfig, SetupCtx};
use rye_egui::{
    dnd::{
        apply_drop_pre_pass as dnd_apply_drop_pre_pass,
        drag_source_collapsing as dnd_drag_source_collapsing, drop_target_idx, force_opaque_active,
        make_room_gap, pickup_t as drag_pickup_t,
    },
    media::{add_button, chevron_button, play_pause_button, rate_toggle, refresh_button},
    slider_with_edit,
};
use rye_math::{Bivector, Bivector4, EuclideanR3, Plane4, Rotor, Rotor4};
use rye_render::{
    device::RenderDevice,
    raymarch::{
        polytope_extended_sdfs_wgsl, BodyUniform, Hyperslice4DNode, HYPERSLICE_KERNEL_WGSL,
        SHAPE_120CELL, SHAPE_16CELL, SHAPE_24CELL, SHAPE_3SPHERE, SHAPE_600CELL,
        SHAPE_CLIFFORD_TORUS, SHAPE_DUOCYLINDER, SHAPE_PENTATOPE, SHAPE_SPHERINDER,
        SHAPE_TESSERACT,
    },
    Viewport,
};
use rye_sdf::{Scene4, SceneNode4};
use winit::window::WindowAttributes;

/// Cap on shapes per row from the runtime "Add" buttons. Keeps the
/// scene visible without scroll-zoom and bounds the per-frame body
/// loop. The CLI `--shapes` argument can still spawn up to
/// `MAX_BODIES` (32) at startup.
const MAX_ROW_LEN: usize = 8;

/// Uniform width for shape cards in the row. Wide enough to fit
/// the longest label ("120-cell" / "600-cell") in bold without the
/// label wrapping; wrapping would make those cards taller than
/// the others, which egui's horizontal cross-alignment then turns
/// into a descending staircase as the row's running max-height
/// grows past earlier (now lower-aligned) cards. Labels also use
/// `WrapMode::Extend` as a belt-and-suspenders check.
const SHAPE_CARD_WIDTH: f32 = 64.0;

/// Unified height for every interactive widget in the bottom
/// overlay: rate, play, chevron, plus, and refresh buttons,
/// make-room drag gaps, and shape cards (via their Frame's
/// inner_margin). Sized to match the cards' natural rendered
/// height; strong-styled body text in egui's default font measures
/// ~17 pt, plus the cards' 6-pt vertical inner_margin = 29 pt.
/// Keeping all controls at this same height removes the height
/// mismatch that would otherwise make the + button appear higher
/// than the cards.
const CONTROL_H: f32 = 29.0;
/// Standard width for square control buttons in the overlay's
/// rate row and shape row (`<<`, `<`, `>`, `>>`, refresh, the per-
/// shape `×`). Matches the visual cadence of the row without each
/// callsite hardcoding the same `28.0`. The play/pause button is
/// deliberately wider (see [`PLAY_PAUSE_W`]) and the smaller help
/// / close glyphs use [`MINI_BUTTON_W`].
const CONTROL_W: f32 = 28.0;
/// Wider central play/pause control. Asymmetry signals the primary
/// action in the rate cluster.
const PLAY_PAUSE_W: f32 = 36.0;
/// Compact close / help glyphs (`×`, `?`). Smaller than the rate-
/// cluster controls so they read as utility chrome, not primary
/// actions.
const MINI_BUTTON_W: f32 = 22.0;
/// Horizontal spacing between adjacent cards in the term and shape
/// rows. The make-room gap animates open to a card's width *plus*
/// this gap, so the value is shared and can't desync.
const CARD_ITEM_SPACING_X: f32 = 4.0;

const W_SCRUB_RATE: f32 = 0.5;
const W_RANGE: f32 = 1.5;

/// Initial maximum value for the t slider's range. Chosen so the
/// per-pixel scrub precision matches the w slider's: w spans
/// `2 × W_RANGE = 3.0` over the same slider track, so starting t
/// at 3.0 means dragging t feels just as smooth as dragging w.
/// The runaway guard in `update()` doubles this as the spin
/// pushes `rot_time` past it; precision halves with each
/// doubling but the user keeps the high precision early on when
/// fine scrubbing matters most.
const T_SLIDER_INITIAL: f32 = 3.0;

/// Base rotation angular rate (rad/s). Scaled by `rate_scale` per
/// frame so the rate buttons can speed it up or slow it down.
const BASE_ROTATION_RATE: f32 = std::f32::consts::TAU * 0.3;

/// Spacing between body centers along x. Slightly larger than
/// `BODY_SIZE * 2` so rotated bodies can stretch into a neighbor's
/// column without overlap during animation.
const BODY_X_SPACING: f32 = 1.8;
/// Per-body circumradius. Smaller than the `[-2, +2]` first row of
/// shapes was at, letting four shapes fit in view at once.
const BODY_SIZE: f32 = 0.7;
/// Center-y for all bodies; floor is at y=0.
const BODY_Y: f32 = 0.9;

/// One polytope's metadata: shape index in the kernel's table,
/// per-body fragment color (driven into `BodyUniform.color` on the
/// GPU side, NOT the panel's card color; those are uniformly grey
/// in the redesigned UI), short display label, and long
/// mathematical name shown in card tooltips. The long name uses
/// the `pentachoron` / `tesseract` / `hexadecachoron` family; the
/// `*-plex` aliases (pentaplex, dodecaplex, ...) are deliberately
/// avoided since "plex" is dimension-generalized rather than
/// being the actual 4D name.
#[derive(Copy, Clone, PartialEq)]
struct ShapeEntry {
    shape: u32,
    body_color: [f32; 3],
    label: &'static str,
    long_name: &'static str,
}

/// Default row when no `--shapes` argument is given. Ordered to put
/// the 24-cell first (most "4D-distinct" cross-section), then the
/// pentachoron / 16-cell / tesseract triple; visually contrasting
/// shapes left-to-right.
const DEFAULT_ROW: &[ShapeEntry] = &[
    ShapeEntry {
        shape: SHAPE_24CELL,
        body_color: [0.95, 0.45, 0.85],
        label: "24-cell",
        long_name: "icositetrachoron",
    },
    ShapeEntry {
        shape: SHAPE_PENTATOPE,
        body_color: [0.95, 0.55, 0.30],
        label: "5-cell",
        long_name: "pentachoron",
    },
    ShapeEntry {
        shape: SHAPE_16CELL,
        body_color: [0.55, 0.95, 0.40],
        label: "16-cell",
        long_name: "hexadecachoron",
    },
    ShapeEntry {
        shape: SHAPE_TESSERACT,
        body_color: [0.30, 0.55, 0.95],
        label: "8-cell",
        long_name: "tesseract",
    },
];

/// Catalog of every shipped 4D shape: the six convex regular
/// polychora plus four non-polychoral SDF-trivial shapes
/// (3-sphere, duocylinder, Clifford torus, spherinder). Used by
/// the filmstrip subject picker and the `+` shape menu. Colours
/// are RGB float channels passed straight to the WGSL kernel
/// (engine doesn't constrain the colour space).
const SHAPE_CATALOG: &[ShapeEntry] = &[
    ShapeEntry {
        shape: SHAPE_PENTATOPE,
        body_color: [0.95, 0.55, 0.30],
        label: "5-cell",
        long_name: "pentachoron",
    },
    ShapeEntry {
        shape: SHAPE_TESSERACT,
        body_color: [0.30, 0.55, 0.95],
        label: "8-cell",
        long_name: "tesseract",
    },
    ShapeEntry {
        shape: SHAPE_16CELL,
        body_color: [0.55, 0.95, 0.40],
        label: "16-cell",
        long_name: "hexadecachoron",
    },
    ShapeEntry {
        shape: SHAPE_24CELL,
        body_color: [0.95, 0.45, 0.85],
        label: "24-cell",
        long_name: "icositetrachoron",
    },
    ShapeEntry {
        shape: SHAPE_120CELL,
        body_color: [0.40, 0.85, 0.85],
        label: "120-cell",
        long_name: "hecatonicosachoron",
    },
    ShapeEntry {
        shape: SHAPE_600CELL,
        body_color: [0.95, 0.85, 0.40],
        label: "600-cell",
        long_name: "hexacosichoron",
    },
    ShapeEntry {
        shape: SHAPE_3SPHERE,
        body_color: [0.85, 0.40, 0.40],
        label: "3-sphere",
        long_name: "hypersphere (4-ball)",
    },
    ShapeEntry {
        shape: SHAPE_DUOCYLINDER,
        body_color: [0.60, 0.45, 0.90],
        label: "duocyl",
        long_name: "duocylinder (D² × D²)",
    },
    ShapeEntry {
        shape: SHAPE_CLIFFORD_TORUS,
        body_color: [0.70, 0.85, 0.35],
        label: "clifford",
        long_name: "Clifford torus tube",
    },
    ShapeEntry {
        shape: SHAPE_SPHERINDER,
        body_color: [0.85, 0.55, 0.75],
        label: "spherinder",
        long_name: "spherinder (B³ × interval)",
    },
];

/// Render a category-grouped shape menu into the current ui.
/// Both call sites (the `+` shape menu and the filmstrip
/// subject combo) use this so the layout stays consistent: top
/// level lists the [`SHAPE_CATEGORIES`] entries, each opens a
/// nested submenu of the shapes in that category, every entry
/// carries a `long_name` hover tooltip. `on_select` fires when
/// the user clicks an entry; the helper closes the menu.
fn render_shape_catalog_menu(ui: &mut egui::Ui, mut on_select: impl FnMut(ShapeEntry)) {
    for cat in SHAPE_CATEGORIES {
        ui.menu_button(cat.name, |ui| {
            for entry in &SHAPE_CATALOG[cat.start..cat.end] {
                if ui
                    .button(entry.label)
                    .on_hover_text(entry.long_name)
                    .clicked()
                {
                    on_select(*entry);
                    ui.close_kind(egui::UiKind::Menu);
                }
            }
        });
    }
}

/// Subcategories of [`SHAPE_CATALOG`], expressed as half-open
/// index ranges into the catalog. Used by the shape menus
/// (`+` button and filmstrip subject combo) to group entries
/// with a header label and separator. Keeping the categories as
/// ranges (rather than nested slices) lets `parse_shape_name`
/// and direct `SHAPE_CATALOG[i]` lookups stay flat.
struct ShapeCategory {
    name: &'static str,
    start: usize,
    end: usize,
}

const SHAPE_CATEGORIES: &[ShapeCategory] = &[
    ShapeCategory {
        name: "Regular polychora",
        start: 0,
        end: 6,
    },
    ShapeCategory {
        name: "Smooth solids",
        start: 6,
        end: 10,
    },
];

/// Catalog of named shapes. Both common math-name aliases (the
/// `n-cell` form) and Platonic-slice aliases (the `tetrahedron` /
/// `cube` / etc. form) resolve to the same shape index.
fn parse_shape_name(name: &str) -> Result<ShapeEntry> {
    let n = name.to_lowercase();
    let needle: &str = n.as_str();
    for entry in SHAPE_CATALOG {
        if needle == entry.label.to_lowercase() || needle == entry.long_name.to_lowercase() {
            return Ok(*entry);
        }
    }
    // Common aliases not in the catalog's `label` / `long_name`.
    Ok(match needle {
        "5cell" | "pentatope" | "tetrahedron" => SHAPE_CATALOG[0],
        "8cell" | "hypercube" | "cube" => SHAPE_CATALOG[1],
        "16cell" | "octahedron" => SHAPE_CATALOG[2],
        "24cell" | "cuboctahedron" => SHAPE_CATALOG[3],
        "120cell" | "dodecahedron" => SHAPE_CATALOG[4],
        "600cell" | "icosahedron" => SHAPE_CATALOG[5],
        "hypersphere" | "3sphere" | "s3" | "4-ball" => SHAPE_CATALOG[6],
        "duocylinder" => SHAPE_CATALOG[7],
        "clifford" | "clifford-torus" | "torus" => SHAPE_CATALOG[8],
        "spherinder" => SHAPE_CATALOG[9],
        _ => {
            return Err(anyhow!(
                "unknown shape name {name:?}; valid: 5-cell, 8-cell, \
                 16-cell, 24-cell, 120-cell, 600-cell, 3-sphere, \
                 duocyl, clifford, spherinder (plus Platonic aliases: \
                 tetrahedron, cube, octahedron, cuboctahedron, \
                 dodecahedron, icosahedron)"
            ));
        }
    })
}

/// Parse the row from CLI arguments. Looks for `--shapes name1 name2 ...`
/// (consumes everything after the flag). Returns `DEFAULT_ROW` if
/// the flag isn't present.
fn parse_row_from_args() -> Result<Vec<ShapeEntry>> {
    let args: Vec<String> = std::env::args().collect();
    let Some(idx) = args.iter().position(|a| a == "--shapes") else {
        return Ok(DEFAULT_ROW.to_vec());
    };
    let names = &args[idx + 1..];
    if names.is_empty() {
        return Err(anyhow!("--shapes flag passed but no shape names followed"));
    }
    names.iter().map(|n| parse_shape_name(n)).collect()
}

fn body_position(slot: usize, n: usize) -> [f32; 4] {
    let x = (slot as f32 - (n as f32 - 1.0) * 0.5) * BODY_X_SPACING;
    [x, BODY_Y, 0.0, 0.0]
}

// ---------------------------------------------------------------------------
// Rotation planes
// ---------------------------------------------------------------------------
//
// `Plane4` (rye-math) is the basis-bivector enumeration; this demo
// drives angular velocity by summing `Plane4::unit_bivector()`
// values for the toggled planes. Sum-of-bivectors composition is
// **commutative** (vector-space addition), so toggle order doesn't
// matter; only the active *set* does. The three w-involving
// planes pull visible axes into the hidden dimension and drive the
// slice-shape changes the viewer was built to show; the three
// pure-3D planes act as ordinary 3D rotations on the cross-section.

/// Angular velocity from a composed seq: sum over terms of
/// `scalar * sum_of_unit_bivectors_in_term`, scaled by rate_scale.
/// Bivector addition is commutative, so term order is irrelevant
/// in this continuous mode (it matters for the multiplicative
/// `Apply` action, but that's a separate one-shot path).
///
/// The Active-mode angular velocity is structurally a special case:
/// each active plane is one unit term with `scalar = None`. The
/// app-level `omega_per_sec` dispatcher inlines that walk over the
/// `[bool; 6]` directly to avoid allocating a transient seq each
/// frame.
fn angular_velocity_from_seq(seq: &[RotorTerm], rate_scale: f32) -> Bivector4 {
    let mut omega = Bivector4::ZERO;
    for term in seq {
        let phi = term.scalar.unwrap_or(1.0);
        for plane in &term.planes {
            omega = omega + plane.unit_bivector() * phi;
        }
    }
    omega * (BASE_ROTATION_RATE * rate_scale)
}

/// Render the 4x4 antisymmetric bivector matrix view: rows
/// and columns labeled `x y z w`, the upper triangle filled
/// with the bivector's component for that pair (in degrees),
/// the lower triangle the negation, the diagonal zero. Pure
/// presentation; reads `b` once and writes a Grid of labels.
///
/// Useful in the formula popup as a more structured view of
/// the rotor's decomposition than the inline `exp(B · t)`
/// summary, which only lists non-zero terms. The matrix shows
/// the full 6-component bivector at a glance.
fn render_bivector_matrix(ui: &mut egui::Ui, b: &Bivector4) {
    const AXIS: [&str; 4] = ["x", "y", "z", "w"];
    // Upper-triangle entries indexed by (row, col) with row < col,
    // mapped to bivector components. e_i ∧ e_j convention: xy at
    // (0, 1), xz at (0, 2), xw at (0, 3), yz at (1, 2), yw at
    // (1, 3), zw at (2, 3).
    let pair = |row: usize, col: usize| -> f32 {
        match (row, col) {
            (0, 1) => b.xy,
            (0, 2) => b.xz,
            (0, 3) => b.xw,
            (1, 2) => b.yz,
            (1, 3) => b.yw,
            (2, 3) => b.zw,
            _ => unreachable!(),
        }
    };
    egui::Grid::new("bivec-matrix")
        .num_columns(5)
        .spacing([8.0, 2.0])
        .show(ui, |ui| {
            ui.label("");
            for axis in AXIS {
                ui.add(egui::Label::new(
                    egui::RichText::new(axis).monospace().weak(),
                ));
            }
            ui.end_row();
            for (row, row_axis) in AXIS.iter().enumerate() {
                ui.add(egui::Label::new(
                    egui::RichText::new(*row_axis).monospace().weak(),
                ));
                for col in 0..4 {
                    let text = if row == col {
                        "0".to_string()
                    } else if row < col {
                        format!("{:>+5.1}", pair(row, col).to_degrees())
                    } else {
                        format!("{:>+5.1}", -pair(col, row).to_degrees())
                    };
                    ui.add(egui::Label::new(egui::RichText::new(text).monospace()));
                }
                ui.end_row();
            }
        });
}

/// Name a recognizable combination of active planes. Indices match
/// `Plane4::ALL`: `0=xy 1=xz 2=xw 3=yz 4=yw 5=zw`. Order-independent,
/// only the active *set* matters.
///
/// Curated entries cover common 4D-geometry classics: single
/// stretches, the three perpendicular-pair isoclinics (the only
/// commuting bivector pairs in 4D, related to left/right Hopf
/// maps), pure-3D rotations, and the famous "all w-planes"
/// composition that drives the cross-section through its main-
/// diagonal extreme.
fn combo_name(active: &[bool; 6]) -> Option<&'static str> {
    // Build a 6-bit mask for compact pattern matching.
    let mut mask = 0u8;
    for (i, &on) in active.iter().enumerate() {
        if on {
            mask |= 1 << i;
        }
    }
    // Bit positions: 0=xy 1=xz 2=xw 3=yz 4=yw 5=zw
    let xy = 1 << 0;
    let xz = 1 << 1;
    let xw = 1 << 2;
    let yz = 1 << 3;
    let yw = 1 << 4;
    let zw = 1 << 5;
    let m = mask;
    Some(match m {
        0 => return None,
        // Single planes, three w-stretchers and three pure-3D rotations.
        x if x == xw => "x-into-w stretch",
        x if x == yw => "y-into-w stretch",
        x if x == zw => "z-into-w stretch",
        x if x == xy => "xy spin (3D only)",
        x if x == xz => "xz spin (3D only)",
        x if x == yz => "yz spin (3D only)",
        // Perpendicular-pair isoclinics, the only commuting
        // bivector pairs in 4D.
        x if x == xw | yz => "isoclinic xw+yz",
        x if x == xz | yw => "isoclinic xz+yw",
        x if x == xy | zw => "isoclinic xy+zw",
        // Pure-3D combos, equivalent to standard 3D rotations.
        x if x == xy | xz | yz => "full 3D spin",
        // The famous "all w-planes", pulls every visible axis
        // into w simultaneously, drives the tesseract through its
        // main-diagonal cross-section (max-volume octahedron).
        x if x == xw | yw | zw => "main-diagonal spin (all-w)",
        // Maximally compound, every plane active.
        x if x == xy | xz | xw | yz | yw | zw => "chaotic SO(4) drift",
        _ => "compound",
    })
}

// ---------------------------------------------------------------------------
// RotatePolytopesApp
// ---------------------------------------------------------------------------

struct RotatePolytopesApp {
    space: EuclideanR3,
    camera: Camera<EuclideanR3>,
    orbit: OrbitController<EuclideanR3>,
    node: Hyperslice4DNode,
    /// Polytope row built at startup from `--shapes` CLI args (or
    /// `DEFAULT_ROW`); drives both the body uniforms and per-body
    /// label lookups in the overlay.
    row: Vec<ShapeEntry>,

    w_slice: f32,
    slider_up_held: bool,
    slider_down_held: bool,

    rotate: bool,
    rot_state: Rotor4,
    /// Toggle bitmap for the six rotation planes; sum of active
    /// planes' unit bivectors becomes the per-frame angular
    /// velocity. See [`Plane4::ALL`] for the index -> plane mapping.
    active: [bool; 6],
    rate_scale: f32,
    /// Accumulated time spent rotating (advances only while
    /// `rotate == true`; resets on **R**). Useful for spotting
    /// periodicities in compound-bivector animations.
    rot_time: f32,
    /// Upper bound on the `t` slider's range. Doubles every time
    /// the spin's accumulated `rot_time` exceeds the current
    /// bound, so the slider's handle stays meaningful at long
    /// elapsed times instead of pinning at the right edge.
    /// Reset to the initial bound on `R`.
    t_slider_max: f32,

    /// Whether the bottom controls overlay is expanded. When
    /// `false` only the always-on slider strip + rate row is shown
    /// at the bottom; when `true` the strip extends upward to also
    /// show the rotation-mode tabs, mode-specific UI, and shape
    /// row. Toggle via the `^` / `v` chevron button or the **H**
    /// key. There is no longer a side panel: the scene renders to
    /// the full window and the overlay floats over it.
    expanded: bool,

    /// Whether the modal "About / help" window is open. Triggered
    /// by clicking the `?` button; closes via the window's title-
    /// bar X (egui's `Window::open(&mut bool)` flips it).
    show_help: bool,

    /// Cached natural overlay width on first frame. Used as the
    /// fixed width of the overlay regardless of the current
    /// window size, so resizing the demo window doesn't stretch
    /// the controls. Set lazily on first render.
    overlay_pinned_width: Option<f32>,

    /// Whether the top-right rotation-formula popup is rendered.
    /// Off by default; the formula is dense for newcomers; the
    /// expanded section has a checkbox to turn it on for users who
    /// want to see exactly which bivectors and scalars compose into
    /// the current orientation.
    show_formula: bool,

    /// Whether the bottom controls overlay is rendered. On by
    /// default so first-time users see all the demo's state at
    /// once; toggle off via `View > Rotation controls` or the
    /// `H` key for an unobstructed scene (e.g., for screenshots
    /// or focused viewing).
    show_controls: bool,

    /// Top-level visualisation mode. `Shapes` shows `self.row`
    /// side-by-side at one `w_slice`; `Filmstrip` shows one
    /// polytope (`self.strip_subject`) sampled across an axis
    /// of w, an axis of t, or both at once (a 2D grid).
    view_mode: ViewMode,
    /// Filmstrip-axis toggles. At least one MUST be active when
    /// `view_mode == Filmstrip` (UI prevents both being off);
    /// when only `strip_w` is on the panel renders a horizontal
    /// row of cells across the w slider's value, when only
    /// `strip_t` is on it renders a vertical column across the
    /// rotation animation's `rot_time`, and when both are on it
    /// renders a 2D grid (w on one axis, t on the other; default
    /// orientation has w on columns and t on rows, swappable via
    /// `strip_swap_axes`).
    strip_w: bool,
    strip_t: bool,
    /// When both `strip_w` and `strip_t` are active, swap the
    /// default axis assignment (w-on-columns / t-on-rows becomes
    /// t-on-columns / w-on-rows).
    strip_swap_axes: bool,
    /// Cell counts along each filmstrip axis. Range 3..=21.
    strip_count_w: usize,
    strip_count_t: usize,
    /// Forward extent of the t-axis fan in animation seconds.
    /// The first cell is at the current `rot_time` (offset 0)
    /// and the last is at `rot_time + strip_t_extent`. Cells
    /// are evenly spaced in between; you read each cell as
    /// "the rotor at this absolute t in the future." Negative
    /// offsets aren't shown (no looking back). Default is
    /// roughly one rotation period at the base rate.
    strip_t_extent: f32,
    /// Polytope rendered in each filmstrip cell. Independent of
    /// `self.row`: filmstrip's single-shape view is decoupled
    /// from the multi-shape row so the user can pick any of the
    /// shipped polytopes regardless of what's been added to the
    /// row.
    strip_subject: ShapeEntry,

    /// Which rotation source drives the continuous spin: the
    /// six-checkbox active set (`Active`), or the composed
    /// sequence's bivector sum (`Composer`). Both share the rate
    /// scale and time slider; switching mode re-points the omega
    /// derivation. The composer mode collapses the seq's terms into
    /// a single bivector velocity (sum of `scalar * planes` per
    /// term); order doesn't matter for continuous-mode since
    /// bivector addition is commutative.
    rotation_mode: RotationMode,

    /// Mode change requested this frame by the mode tabs. Applied
    /// after the overlay finishes rendering so that the body that
    /// renders this frame still sees `rotation_mode` (the OLD
    /// value), and only the next frame swaps to the new mode.
    /// Without this deferral, clicking a tab would change
    /// `rotation_mode` mid-pass; the visible pass would render
    /// the new mode's body while `BottomOverlay`'s measure pass
    /// already captured the old mode's natural height, producing
    /// a one-frame layout mismatch the user perceives as flicker.
    pending_mode: Option<RotationMode>,

    /// View change requested this frame by the view tab row.
    /// Same deferred-write rationale as `pending_mode`: switching
    /// Shapes <-> Filmstrip changes the body's natural height
    /// significantly (shape row vs subject combo), so the
    /// `BottomOverlay` two-pass shape would mismatch on the
    /// transition frame and flicker.
    pending_view_mode: Option<ViewMode>,

    /// Composer-mode actions deferred to end-of-frame for the same
    /// reason as `pending_mode`: any mutation that grows or shrinks
    /// the overlay's body height (adding a draft plane, committing
    /// a term, clearing the draft) must apply *after* both
    /// `BottomOverlay` passes have rendered, otherwise pass 1 sees
    /// the OLD body height and pass 2 the NEW one; flicker.
    pending_actions: Vec<DeferredAction>,

    /// Sequence of [`RotorTerm`]s the user is building in the panel.
    /// Apply composes them onto `rot_state` left-to-right via rotor
    /// multiplication.
    seq: Vec<RotorTerm>,
    /// In-progress draft for the next term. Plane buttons append
    /// here; "Add" commits this list as a new term in `seq` and
    /// clears the draft. Bivector planes only; the optional
    /// scalar attaches to a committed term, not to the draft.
    draft: Vec<Plane4>,

    /// Typed-formula input for the Composer's text bar. Single
    /// expression per submit (Enter); pushes a RotorTerm into seq
    /// and clears. The chip row remains the fast path for single-
    /// plane terms.
    formula_input: String,
    /// Last parse error from the formula bar, rendered under the
    /// input until cleared by a successful submit.
    formula_error: Option<String>,
}

/// One term in the rotor-composition sequence: a sum of unit
/// bivectors with an optional leading scalar (angle in radians).
///
/// Without a scalar the term is `exp(sum_of_unit_bivectors)`,
/// which is the natural unit-magnitude rotation along the term's
/// bivector direction. With a scalar `phi` it becomes
/// `exp(phi * sum_of_unit_bivectors)`. The scalar is optional by
/// design: most uses ("rotate 90° in xy") want a scalar, but the
/// "raw direction" form (just the bivector itself) is useful for
/// composing isoclinics where the magnitude is implicit.
///
/// Bivector addition within a term is commutative, so plane order
/// inside a term doesn't matter. Rotor multiplication between
/// terms is non-commutative, so the seq's term order does.
#[derive(Clone, Debug, Default)]
struct RotorTerm {
    /// Unit-bivector planes summed inside `exp(...)`. Non-empty
    /// for a term to display; an empty term is dropped.
    planes: Vec<Plane4>,
    /// Optional scalar prefix `phi` in radians. `None` means the
    /// raw bivector sum (unit magnitude); `Some(phi)` scales the
    /// whole sum before `exp()`. The panel's "Add scalar" action
    /// initialises this to `FRAC_PI_2`; `Default::default()` is
    /// `None` so an empty draft commits as a unit-magnitude term.
    scalar: Option<f32>,
}

/// Render `(p_0 + p_1 + ...)` (with parens iff multi-plane) into
/// the current ui. Each plane goes through `render_plane`, which
/// decides whether it's an interactive drag pill (term card),
/// plain monospace (draft card), or anything else. The paren
/// logic and `+` separators are shared so the visual reading of
/// a bivector sum stays identical across all callsites.
fn render_plane_sum(
    ui: &mut egui::Ui,
    planes: &[Plane4],
    mut render_plane: impl FnMut(&mut egui::Ui, usize, Plane4),
) {
    let multi = planes.len() > 1;
    if multi {
        ui.monospace("(");
    }
    for (i, plane) in planes.iter().enumerate() {
        if i > 0 {
            ui.monospace("+");
        }
        render_plane(ui, i, *plane);
    }
    if multi {
        ui.monospace(")");
    }
}

/// Render a single [`RotorTerm`] as the `scalar · bivec` form
/// that appears inside `exp(...)`. Multi-plane terms get inner
/// parens; the lone scalar prefix is dropped when absent. Pure
/// presentation, no math.
fn render_term(term: &RotorTerm) -> String {
    let plane_str = term
        .planes
        .iter()
        .map(|p| p.label())
        .collect::<Vec<_>>()
        .join(" + ");
    let bivec = if term.planes.len() > 1 {
        format!("({plane_str})")
    } else {
        plane_str
    };
    match term.scalar {
        Some(phi) => format!("{:.0}° · {}", phi.to_degrees(), bivec),
        None => bivec,
    }
}

/// Wrap a list of bivector-expression parts into a single bivector
/// expression (paren-grouped when there's more than one part). None
/// when the list is empty so the caller can return early.
fn render_bivector_sum(parts: &[String]) -> Option<String> {
    match parts {
        [] => None,
        [only] => Some(only.clone()),
        many => Some(format!("({})", many.join(" + "))),
    }
}

/// Parse a single term written like `90° (xy + zw)`, `xy + xz`,
/// `90 xy`, `0.5 rad xy`, into a [`RotorTerm`]. Degrees are the
/// default unit for the scalar; `rad` suffix overrides. The `*`
/// or `·` between scalar and bivector is optional. Outer parens
/// around the bivector sum are optional.
///
/// Single expression per call: chained terms (`exp(A) * exp(B)`)
/// are not parsed here, since the user submits each term separately
/// via the input bar; rotor multiplication lives in the seq.
fn parse_formula_term(input: &str) -> Result<RotorTerm, String> {
    let normalized = input.trim().replace('·', "*").replace('°', "deg ");
    let s = normalized.trim();
    if s.is_empty() {
        return Err("empty input".into());
    }
    let (scalar, rest) = peel_scalar(s)?;
    let bivec_str = rest.trim();
    let inner = if bivec_str.starts_with('(') && bivec_str.ends_with(')') {
        bivec_str[1..bivec_str.len() - 1].trim()
    } else {
        bivec_str
    };
    if inner.is_empty() {
        return Err("missing bivector after scalar".into());
    }
    let mut planes = Vec::new();
    for part in inner.split('+') {
        let p = part.trim();
        if p.is_empty() {
            return Err("empty plane between '+'".into());
        }
        planes.push(parse_plane(p)?);
    }
    Ok(RotorTerm { planes, scalar })
}

fn peel_scalar(s: &str) -> Result<(Option<f32>, &str), String> {
    let bytes = s.as_bytes();
    let mut i = 0;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        i += 1;
    }
    let digits_start = i;
    while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
        i += 1;
    }
    if i == digits_start {
        return Ok((None, s));
    }
    let num_str = &s[..i];
    let value: f32 = num_str
        .parse()
        .map_err(|_| format!("not a number: `{num_str}`"))?;
    let mut tail = s[i..].trim_start();
    let radians = if let Some(rest) = tail.strip_prefix("rad") {
        tail = rest.trim_start();
        value
    } else if let Some(rest) = tail.strip_prefix("deg") {
        tail = rest.trim_start();
        value.to_radians()
    } else {
        value.to_radians()
    };
    if let Some(rest) = tail.strip_prefix('*') {
        tail = rest.trim_start();
    }
    Ok((Some(radians), tail))
}

fn parse_plane(s: &str) -> Result<Plane4, String> {
    match s {
        "xy" => Ok(Plane4::Xy),
        "xz" => Ok(Plane4::Xz),
        "xw" => Ok(Plane4::Xw),
        "yz" => Ok(Plane4::Yz),
        "yw" => Ok(Plane4::Yw),
        "zw" => Ok(Plane4::Zw),
        _ => Err(format!("unknown plane `{s}` (expected xy/xz/xw/yz/yw/zw)")),
    }
}

/// Continuous-rotation source. Two distinct UIs (active-set
/// checkboxes vs composed sequence) populate the angular velocity
/// independently; the user picks which one drives `omega` for the
/// spin animation via a tab in the rotation tab row.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum RotationMode {
    /// Sum of unit bivectors of planes whose checkboxes are on.
    /// The classic toggleable mode: 1..6 keys / panel checkboxes.
    Active,
    /// Sum of bivectors derived from the composed seq: each term
    /// contributes `scalar.unwrap_or(1.0) * sum_of_unit_bivectors`.
    /// Apply (one-shot rotor multiplication) is still available in
    /// this mode and is independent of the spin animation.
    Composer,
}

/// Visualisation mode. Orthogonal to [`RotationMode`]: rotation
/// configures *how* the rotor evolves, view configures *what* the
/// scene shows. Two distinct visual demos live here, picked by a
/// top-level tab row above the rotation tabs.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ViewMode {
    /// Multi-shape comparison: `self.row` of [`ShapeEntry`]s
    /// rendered side-by-side at one common `w_slice`. Shape order
    /// in the row is meaningful; drag-and-drop rearranges the
    /// scene's left-to-right layout.
    Shapes,
    /// Single-shape filmstrip: one [`ShapeEntry`] (independent of
    /// the row) rendered N times across evenly-spaced `w_slice`
    /// values around the slider's current `w`. Order of the
    /// scene's row is irrelevant in this mode; the row UI is
    /// hidden entirely.
    Filmstrip,
}

/// State mutations queued during overlay rendering and applied
/// AFTER the overlay's measure + visible passes finish. Any
/// mutation that changes the overlay's natural content height
/// must go through this; applying mid-frame would make the two
/// `BottomOverlay` passes disagree on body height and the user
/// would see a one-frame layout mismatch as flicker.
#[derive(Clone, Debug)]
enum DeferredAction {
    /// `+xy` etc. button on the plane row: append to draft.
    DraftPush(Plane4),
    /// `Add` button on the draft preview: commit current draft as a
    /// new RotorTerm in seq, clear draft.
    SeqCommitDraft,
    /// `×` button on the draft preview: discard the draft.
    DraftClear,
    /// Typed-formula bar: push a fully-formed term to seq.
    SeqPushTerm(RotorTerm),
}

/// Drag-and-drop payload for the rotor sequence UI. Terms (whole
/// cards) and plane entries (pills inside cards) both ride this
/// single enum so a term card can be a single drop zone that
/// branches on the variant: a `Term` payload reorders the seq, an
/// `Entry` payload migrates a plane into this term.
#[derive(Clone, Copy, Debug)]
enum DragPayload {
    /// The whole term at this seq index is being dragged.
    Term(usize),
    /// `Entry(term_idx, plane_idx)`: a single plane pill from the
    /// given term is being dragged.
    Entry(usize, usize),
}

impl RotatePolytopesApp {
    /// Per-animation-second angular velocity (the bivector
    /// that, integrated over animation time, produces
    /// `rot_state`). Independent of `rate_scale`. Active mode
    /// sums the toggled basis bivectors; Composer mode delegates
    /// to the seq walker. The rate buttons advance animation
    /// time faster or slower (see [`Self::dt_animation`]); they
    /// don't change this velocity.
    ///
    /// This factoring lets `rot_time` be displayed as
    /// "animation time" and still give consistent
    /// `rot_state = exp(omega_animation * rot_time)` semantics
    /// across rate changes.
    /// The composer seq's net bivector direction (no rate or
    /// base-rate scaling). This is the "function" the seq
    /// describes: sum over terms of `scalar * sum_planes`. The
    /// scrub slider uses this as its rotation axis-bivector;
    /// the projection of `log(rot_state)` onto this direction is
    /// the slider's value.
    fn compose_omega(&self) -> Bivector4 {
        let mut omega = Bivector4::ZERO;
        for term in &self.seq {
            let phi = term.scalar.unwrap_or(1.0);
            for plane in &term.planes {
                omega = omega + plane.unit_bivector() * phi;
            }
        }
        omega
    }

    fn omega_animation(&self) -> Bivector4 {
        match self.rotation_mode {
            RotationMode::Active => {
                let mut omega = Bivector4::ZERO;
                for (i, &on) in self.active.iter().enumerate() {
                    if on {
                        omega = omega + Plane4::ALL[i].unit_bivector();
                    }
                }
                omega * BASE_ROTATION_RATE
            }
            RotationMode::Composer => angular_velocity_from_seq(&self.seq, 1.0),
        }
    }

    /// Drive every body in the row with the same rotor, lets the
    /// user directly compare slice signatures under identical 4D motion.
    fn write_all(&mut self, rotor: Rotor4) {
        let n = self.row.len();
        for (slot, entry) in self.row.iter().enumerate() {
            let body = BodyUniform::polytope_with_rotor(
                body_position(slot, n),
                entry.shape,
                BODY_SIZE,
                rotor,
                entry.body_color,
            );
            self.node.set_body(slot, body);
        }
    }

    /// Expanded section of the bottom overlay. Two tab rows
    /// stacked vertically:
    ///
    /// 1. **View tabs** (Shapes / Filmstrip): top-level visual
    ///    demo. Shapes shows the multi-shape row; Filmstrip
    ///    shows one shape across N w-slices.
    /// 2. **Rotation tabs** (Active set / Composer): how the
    ///    rotor evolves. Independent of view mode.
    ///
    /// Always-visible controls (Spin/Pause, rate buttons,
    /// sliders) live below this in `render_overlay`.
    fn render_expanded_body(&mut self, ui: &mut egui::Ui) {
        self.render_view_tab_row(ui);
        match self.view_mode {
            ViewMode::Shapes => self.render_shapes_section(ui),
            ViewMode::Filmstrip => self.render_filmstrip_body(ui),
        }
        ui.separator();
        self.render_rotation_tab_row(ui);
        if self.rotation_mode == RotationMode::Active {
            self.render_active_mode(ui);
        } else {
            self.render_composer_mode(ui);
        }
    }

    /// Top tab row of the expanded body: visual demo selector.
    /// Shapes (multi-shape side-by-side row) vs Filmstrip (one
    /// shape across multiple w-slices). Tab change is staged
    /// into `pending_view_mode` for the same `BottomOverlay`
    /// two-pass reason as `pending_mode`: the two body shapes
    /// have very different natural heights and an immediate
    /// swap mid-frame would flicker.
    fn render_view_tab_row(&mut self, ui: &mut egui::Ui) {
        let mut staged = self.view_mode;
        ui.horizontal(|ui| {
            ui.selectable_value(&mut staged, ViewMode::Shapes, "Shapes")
                .on_hover_text("Side-by-side row of shapes at one w-slice");
            ui.selectable_value(&mut staged, ViewMode::Filmstrip, "Filmstrip")
                .on_hover_text(
                    "One shape rendered N times across w-slices fanning out by \
                     ±BODY_SIZE around the w slider's value",
                );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.checkbox(&mut self.show_formula, "Show formula")
                    .on_hover_text("Top-right popup with the live exp(...) form of the rotor");
            });
        });
        if staged != self.view_mode {
            self.pending_view_mode = Some(staged);
        }
    }

    /// Render axis labels around the filmstrip grid. Top edge
    /// gets w-value tags above each column (whichever axis
    /// carries w); left edge gets t-offset tags beside each row
    /// (whichever axis carries t). The cell whose offset along
    /// each axis is closest to zero is highlighted in active-set
    /// warning gold. For 1D cases the orthogonal-axis labels
    /// are omitted (just one row or one column).
    fn render_filmstrip_cell_labels(&mut self, ctx: &egui::Context) {
        let (cols, rows, w_on_cols) = match (self.strip_w, self.strip_t) {
            (true, true) => {
                if self.strip_swap_axes {
                    (self.strip_count_t, self.strip_count_w, false)
                } else {
                    (self.strip_count_w, self.strip_count_t, true)
                }
            }
            (true, false) => (self.strip_count_w, 1, true),
            (false, true) => (1, self.strip_count_t, false),
            (false, false) => return,
        };
        if cols == 0 || rows == 0 {
            return;
        }
        let screen = ctx.content_rect();
        let cell_w_px = screen.width() / cols as f32;
        let cell_h_px = screen.height() / rows as f32;
        let strip_w_extent = BODY_SIZE;

        let label_color = |is_center: bool| {
            if is_center {
                egui::Color32::from_rgb(255, 217, 140)
            } else {
                egui::Color32::from_gray(220)
            }
        };
        let label_frame = egui::Frame::default()
            .fill(egui::Color32::from_black_alpha(160))
            .inner_margin(egui::Margin::symmetric(6, 2))
            .corner_radius(3);

        // Per-axis cell label + center-cell flag. `axis_label`
        // computes the (text, is_current) pair: w cells fan
        // symmetrically around the slider so the center index
        // is "current"; t cells fan FORWARD from the current
        // `rot_time`, so index 0 is "current" and the rest are
        // future predictions.
        let w_axis_label = |i: usize, n: usize| -> (String, bool) {
            let off = if n <= 1 {
                0.0
            } else {
                let t = i as f32 / (n - 1) as f32;
                -strip_w_extent + t * (2.0 * strip_w_extent)
            };
            let mid = if n == 0 { 0 } else { n / 2 };
            (format!("w={:>+.3}", self.w_slice + off), i == mid)
        };
        let t_axis_label = |i: usize, n: usize| -> (String, bool) {
            let off = if n <= 1 {
                0.0
            } else {
                let t = i as f32 / (n - 1) as f32;
                t * self.strip_t_extent
            };
            (format!("t={:.2}s", self.rot_time + off), i == 0)
        };

        // Top edge: column labels.
        for i in 0..cols {
            let center_x = screen.left() + (i as f32 + 0.5) * cell_w_px;
            let (text, is_center) = if w_on_cols {
                w_axis_label(i, cols)
            } else {
                t_axis_label(i, cols)
            };
            let pos = egui::pos2(center_x, screen.top() + 96.0);
            egui::Area::new(egui::Id::new(("strip-col-label", i)))
                .fixed_pos(pos)
                .pivot(egui::Align2::CENTER_TOP)
                .order(egui::Order::Foreground)
                .show(ctx, |ui| {
                    label_frame.show(ui, |ui| {
                        ui.add(egui::Label::new(
                            egui::RichText::new(text)
                                .color(label_color(is_center))
                                .monospace()
                                .size(12.0),
                        ));
                    });
                });
        }
        // Left edge: row labels (only when > 1 row).
        if rows > 1 {
            for j in 0..rows {
                let center_y = screen.top() + (j as f32 + 0.5) * cell_h_px;
                let (text, is_center) = if w_on_cols {
                    t_axis_label(j, rows)
                } else {
                    w_axis_label(j, rows)
                };
                let pos = egui::pos2(screen.left() + 16.0, center_y);
                egui::Area::new(egui::Id::new(("strip-row-label", j)))
                    .fixed_pos(pos)
                    .pivot(egui::Align2::LEFT_CENTER)
                    .order(egui::Order::Foreground)
                    .show(ctx, |ui| {
                        label_frame.show(ui, |ui| {
                            ui.add(egui::Label::new(
                                egui::RichText::new(text)
                                    .color(label_color(is_center))
                                    .monospace()
                                    .size(12.0),
                            ));
                        });
                    });
            }
        }
    }

    /// Filmstrip body: subject combo (over [`SHAPE_CATALOG`], so
    /// the user can pick any of the six known polytopes
    /// independent of `self.row`) plus a cells DragValue. Heavy-
    /// shape warning surfaces here when the subject is 120/600-
    /// cell since `render_shapes_section` (where the warning
    /// otherwise lives) is hidden in this view.
    fn render_filmstrip_body(&mut self, ui: &mut egui::Ui) {
        let heavy =
            self.strip_subject.shape == SHAPE_120CELL || self.strip_subject.shape == SHAPE_600CELL;
        if heavy {
            ui.colored_label(
                egui::Color32::from_rgb(242, 130, 70),
                "120/600-cell SDFs are heavy; expect <60 fps.",
            );
        }
        // Row 1: axis toggles + (when both are on) the swap.
        // Invariant: at least one of `strip_w` / `strip_t` must
        // be on. Clicking the on-toggle while the other is off
        // is a no-op (visual checkbox stays checked).
        ui.horizontal(|ui| {
            let mut w_on = self.strip_w;
            let mut t_on = self.strip_t;
            if ui
                .checkbox(&mut w_on, "w cells")
                .on_hover_text("Sample across w around the slider's value")
                .changed()
                && (w_on || self.strip_t)
            {
                self.strip_w = w_on;
            }
            if ui
                .checkbox(&mut t_on, "t cells")
                .on_hover_text(
                    "Sample across animation time around the t slider; \
                     fans by ±strip_t_extent seconds",
                )
                .changed()
                && (t_on || self.strip_w)
            {
                self.strip_t = t_on;
            }
            if self.strip_w && self.strip_t {
                ui.checkbox(&mut self.strip_swap_axes, "swap axes")
                    .on_hover_text(
                        "Default puts w on columns, t on rows. \
                         Swap to put t on columns, w on rows.",
                    );
            }
        });
        // Row 2: counts + t-extent + subject combo.
        ui.horizontal(|ui| {
            if self.strip_w {
                ui.add(
                    egui::DragValue::new(&mut self.strip_count_w)
                        .range(3..=21)
                        .speed(0.2)
                        .prefix("w: "),
                );
            }
            if self.strip_t {
                ui.add(
                    egui::DragValue::new(&mut self.strip_count_t)
                        .range(3..=21)
                        .speed(0.2)
                        .prefix("t: "),
                );
                ui.add(
                    egui::DragValue::new(&mut self.strip_t_extent)
                        .range(0.1..=10.0)
                        .speed(0.02)
                        .fixed_decimals(2)
                        .suffix("s")
                        .prefix("Δt: "),
                )
                .on_hover_text(
                    "Forward extent of the t fan; cells span \
                     [t, t+Δt] seconds of animation time",
                );
            }
            // Same Popup::menu pattern as the `+` shape menu in
            // the shape row, so the subject picker has identical
            // visuals (nested category submenus) instead of
            // egui's ComboBox styling, which renders the menu
            // entries with combo-dropdown chrome that doesn't
            // match the rest of the demo's menus.
            let subject_button = ui
                .button(format!("subject: {}", self.strip_subject.label))
                .on_hover_text("Pick the polytope rendered in each filmstrip cell");
            egui::Popup::menu(&subject_button).show(|ui| {
                ui.set_min_width(140.0);
                render_shape_catalog_menu(ui, |entry| {
                    self.strip_subject = entry;
                });
            });
        });
    }

    /// Rotation-mode tabs: which source drives `omega`. The tab
    /// change is staged into `self.pending_mode` rather than
    /// applied directly so `BottomOverlay`'s two-pass measure-
    /// then-render captures the same body height in both passes;
    /// clicking a tab swaps modes on the *next* frame, with the
    /// height animation, but no mid-frame mismatch flicker.
    fn render_rotation_tab_row(&mut self, ui: &mut egui::Ui) {
        let mut staged = self.rotation_mode;
        ui.horizontal(|ui| {
            ui.selectable_value(&mut staged, RotationMode::Active, "Active set")
                .on_hover_text("Six checkbox-toggled bivectors (xy, xz, ...)");
            ui.selectable_value(&mut staged, RotationMode::Composer, "Composer")
                .on_hover_text("Sum of bivectors from the composed sequence");
        });
        if staged != self.rotation_mode {
            self.pending_mode = Some(staged);
        }
    }

    /// Active-set body: six plane cells laid out as a 3x2 grid.
    /// Pure-3D planes (xy, xz, yz) on the top row, w-involving
    /// planes (xw, yw, zw) on the bottom row. Each cell is
    /// `[checkbox][label][slider][value]`:
    ///
    /// - Checkbox = "include this plane in continuous spin omega"
    ///   (the classic Active mode toggle).
    /// - Label = plane name (xy / xz / ...).
    /// - Slider = the log decomposition of `rot_state` in that
    ///   basis bivector, in degrees, range -180..=180. Dragging
    ///   sets that component of `log(rot_state)` and rebuilds
    ///   `rot_state` via exp. No separate manual-angle state, so
    ///   the slider is a true window onto the current rotor.
    /// - Value = right-click-editable label.
    ///
    /// All sub-component widths are pinned via constants (not
    /// derived from `available_width` per row) so the columns
    /// align EXACTLY across rows; the previous diagonal drift
    /// came from per-cell `slider_width = available - n` reading
    /// slightly different `available` values each row as cell
    /// content + spacing accumulated.
    ///
    /// Combo name ("isoclinic xw+yz" etc.) is dropped from this
    /// body; it lives only in the formula popup now.
    /// Active body: 3-per-row 2-row grid of
    /// `[checkbox][label][slider][value]`. Pinned widths so
    /// columns align across rows (see issue-history note about
    /// the previous staircase). Each value cell is right-click
    /// editable via the shared `slider_with_edit` helper.
    fn render_active_mode(&mut self, ui: &mut egui::Ui) {
        const TOP_ROW: [usize; 3] = [0, 1, 3]; // xy, xz, yz
        const BOTTOM_ROW: [usize; 3] = [2, 4, 5]; // xw, yw, zw

        const CELL_INNER_SPACING: f32 = 4.0;
        const CHECKBOX_W: f32 = 18.0;
        const LABEL_W: f32 = 22.0;
        const VALUE_W: f32 = 56.0;
        const ROW_GAP: f32 = 6.0;

        let total_w = ui.available_width();
        let cell_w = ((total_w - 2.0 * ROW_GAP) / 3.0).floor();
        let slider_w =
            (cell_w - CHECKBOX_W - LABEL_W - VALUE_W - 3.0 * CELL_INNER_SPACING).max(40.0);

        for plane_indices in [TOP_ROW, BOTTOM_ROW] {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = ROW_GAP;
                for &i in &plane_indices {
                    ui.allocate_ui_with_layout(
                        egui::vec2(cell_w, CONTROL_H),
                        egui::Layout::left_to_right(egui::Align::Center),
                        |ui| {
                            ui.spacing_mut().item_spacing.x = CELL_INNER_SPACING;
                            ui.spacing_mut().slider_width = slider_w;
                            self.render_plane_slider_cell(
                                ui, i, CHECKBOX_W, LABEL_W, slider_w, VALUE_W,
                            );
                        },
                    );
                }
            });
        }
    }

    /// One plane cell. All component widths pinned by the caller
    /// so the cell renders identically regardless of which row
    /// or column it's in.
    fn render_plane_slider_cell(
        &mut self,
        ui: &mut egui::Ui,
        plane_idx: usize,
        checkbox_w: f32,
        label_w: f32,
        slider_w: f32,
        value_w: f32,
    ) {
        let plane = Plane4::ALL[plane_idx];
        let bivec = self.rot_state.log();
        let current_rad = bivec.component(plane);
        // Slider range matches the rotor's actual period.
        // `Rotor4` lives in Spin(4), the double cover of SO(4):
        // a 360° physical rotation maps to the rotor `-1`, and
        // 720° brings the rotor back to `+1`. So the natural
        // period of any single-plane rotor parameter is 720°,
        // and `Rotor4::log` returns values across [-360, 360].
        // Showing the full ±360° range exposes the double-cover
        // structure honestly; a previous [-180, 180] wrap hid
        // it but pinned the slider during the rotor's "second
        // half" or jumped twice per cycle.
        let mut deg = current_rad.to_degrees();
        ui.add_sized(
            [checkbox_w, 18.0],
            egui::Checkbox::new(&mut self.active[plane_idx], ""),
        );
        ui.add_sized(
            [label_w, 18.0],
            egui::Label::new(egui::RichText::new(plane.label()).monospace()),
        );
        let slider = egui::Slider::new(&mut deg, -360.0..=360.0)
            .show_value(false)
            .smart_aim(false)
            .clamping(egui::SliderClamping::Always);
        let slider_resp = ui.add_sized([slider_w, 18.0], slider);
        let formatted = format!("{deg:>+6.1}°");
        let mut popup_changed = false;
        ui.allocate_ui_with_layout(
            egui::vec2(value_w, 18.0),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                let label_resp = ui.add(
                    egui::Button::new(egui::RichText::new(formatted).monospace())
                        .frame(false)
                        .small(),
                );
                label_resp
                    .on_hover_cursor(egui::CursorIcon::ContextMenu)
                    .on_hover_text("Right-click to edit value")
                    .context_menu(|ui| {
                        let drag_resp = ui.add(
                            egui::DragValue::new(&mut deg)
                                .range(-360.0..=360.0)
                                .suffix("°")
                                .fixed_decimals(1),
                        );
                        if drag_resp.changed() {
                            popup_changed = true;
                        }
                    });
            },
        );
        if slider_resp.changed() || popup_changed {
            let mut new_bivec = bivec;
            new_bivec.set_component(plane, deg.to_radians());
            self.rot_state = new_bivec.exp();
            self.write_all(self.rot_state);
        }
    }

    /// Composer-mode body: typed-formula bar, single-plane chip
    /// row, draft preview card, drag-and-drop term sequence,
    /// Apply / Clear actions. The state-mutation collectors
    /// (`term_moves`, `entry_moves`, `remove_term`, etc.) are
    /// gathered during card rendering and applied at the end of
    /// this function so that the `BottomOverlay`'s measure-then-
    /// render two-pass shape sees the same seq in both passes.
    fn render_composer_mode(&mut self, ui: &mut egui::Ui) {
        ui.separator();

        // Typed-formula bar + single-plane chip row on the same
        // line. Layout: `f: [text input] [Add] [+xy] ... [+zw]`.
        // The chips append to the draft (fast path for single-
        // plane terms); the text input takes a full expression.
        ui.horizontal_wrapped(|ui| {
            ui.label("f:");
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.formula_input)
                    .hint_text("e.g. 90° (xy + zw)")
                    .desired_width(180.0),
            );
            let submitted = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            let add_clicked = ui.small_button("Add").clicked();
            if submitted || add_clicked {
                match parse_formula_term(&self.formula_input) {
                    Ok(term) => {
                        self.pending_actions.push(DeferredAction::SeqPushTerm(term));
                        self.formula_input.clear();
                        self.formula_error = None;
                        if submitted {
                            resp.request_focus();
                        }
                    }
                    Err(e) => self.formula_error = Some(e),
                }
            } else if self.formula_input.is_empty() {
                self.formula_error = None;
            }
            ui.separator();
            for plane in Plane4::ALL.iter() {
                if ui
                    .small_button(format!("+{}", plane.label()))
                    .on_hover_text("Add to the current draft term")
                    .clicked()
                {
                    self.pending_actions.push(DeferredAction::DraftPush(*plane));
                }
            }
            // Clear at the right end of the chips row.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add_enabled(!self.seq.is_empty(), egui::Button::new("Clear"))
                    .on_hover_text("Remove all terms from the sequence")
                    .clicked()
                {
                    self.seq.clear();
                }
            });
        });
        if let Some(err) = &self.formula_error {
            ui.colored_label(
                egui::Color32::from_rgb(255, 120, 120),
                format!("parse error: {err}"),
            );
        }

        // Draft preview rendered as a card matching the committed-
        // term style. Add commits to seq; Discard scraps the draft.
        // No "make continuous" action here; the mode tab governs
        // continuous rotation now, the seq drives it directly.
        if !self.draft.is_empty() {
            egui::Frame::group(ui.style())
                .inner_margin(4.0)
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(egui::RichText::new("draft").small().weak());
                        render_plane_sum(ui, &self.draft, |ui, _, plane| {
                            ui.monospace(plane.label());
                        });
                        ui.add_space(8.0);
                        if ui
                            .small_button("Add")
                            .on_hover_text("Commit as one-shot term in sequence")
                            .clicked()
                        {
                            self.pending_actions.push(DeferredAction::SeqCommitDraft);
                        }
                        if ui
                            .add(
                                egui::Button::new(egui::RichText::new("×").size(14.0))
                                    .min_size(egui::vec2(MINI_BUTTON_W, MINI_BUTTON_W)),
                            )
                            .on_hover_text("Discard draft")
                            .clicked()
                        {
                            self.pending_actions.push(DeferredAction::DraftClear);
                        }
                    });
                });
        }

        self.render_composer_seq_cards(ui);
        self.render_composer_scrub_slider(ui);
    }

    /// "Slide-to-rotate" slider for the composer: a full-width
    /// row sized like the w/t sliders. Drives a rotation along
    /// the seq's net bivector direction.
    ///
    /// Math: let `D = compose_omega() / |compose_omega()|` (the
    /// seq's unit-bivector direction). The slider value is the
    /// projection of `log(rot_state)` onto `D`, in degrees. On
    /// drag, the projection is updated; the perpendicular
    /// component of `log(rot_state)` is preserved, so adjusting
    /// the slider rotates ALONG the seq's direction without
    /// disturbing other rotations. Hidden when the seq is empty
    /// or its net bivector is zero (terms cancel out).
    fn render_composer_scrub_slider(&mut self, ui: &mut egui::Ui) {
        let omega = self.compose_omega();
        let mag_sq = omega.magnitude_squared();
        if mag_sq < 1e-12 {
            return;
        }
        let unit = omega * (1.0 / mag_sq.sqrt());
        let bivec = self.rot_state.log();
        let proj_rad = bivec.dot(unit);
        let mut proj_deg = proj_rad.to_degrees();

        const VALUE_CELL_W: f32 = 86.0;
        let avail = ui.available_width();
        let spacing = ui.spacing().item_spacing.x;
        let slider_w = (avail - VALUE_CELL_W - spacing).max(140.0);
        ui.spacing_mut().slider_width = slider_w;
        let row_size = egui::vec2(avail, CONTROL_H);
        let row_layout = egui::Layout::left_to_right(egui::Align::Center);

        // Same -360..360 range as the per-plane sliders, for the
        // same Spin(4) double-cover reason: a 360° projection
        // along the seq's direction lands at the negative-rotor
        // -1, and 720° returns to identity. Showing the full
        // range honestly exposes the rotor's period.
        let formatted = format!("f {proj_deg:>+6.1}°");
        ui.allocate_ui_with_layout(row_size, row_layout, |ui| {
            let changed = slider_with_edit(
                ui,
                &mut proj_deg,
                -360.0..=360.0,
                &formatted,
                "°",
                1,
                VALUE_CELL_W,
            );
            if changed {
                let new_proj = proj_deg.to_radians();
                let old_proj = bivec.dot(unit);
                let new_b = bivec + unit * (new_proj - old_proj);
                self.rot_state = new_b.exp();
                self.write_all(self.rot_state);
            }
        });
    }

    /// Composer's seq-card row: each [`RotorTerm`] renders as a
    /// single-row card. The whole card is its own drag source
    /// (Term payload, reorders the seq) and also a drop zone for
    /// `Entry` payloads (cross-term plane migration). Insertion-
    /// pipe gaps between cards give precise drop indication for
    /// the Term-reorder path. State mutations gathered during
    /// card rendering (`term_moves`, `entry_moves`,
    /// `remove_term`, `remove_scalar`, `add_scalar`,
    /// `edit_scalar`) are all applied at the end of this
    /// function so the card-rendering loop can borrow `self.seq`
    /// immutably while in flight.
    fn render_composer_seq_cards(&mut self, ui: &mut egui::Ui) {
        let mut entry_moves: Vec<(usize, usize, usize)> = Vec::new();
        let mut remove_term: Option<usize> = None;
        let mut remove_scalar: Option<usize> = None;
        let mut add_scalar: Option<usize> = None;
        let mut edit_scalar: Option<(usize, f32)> = None;

        if !self.seq.is_empty() {
            ui.label("Sequence:");
            let term_h = CONTROL_H;
            // Drop target slot index for the term row, computed
            // from cursor position vs last frame's row geometry.
            // Active only when a Term-variant payload is in
            // flight; Entry-variant payloads (cross-term plane
            // migration) drop on cards, not gaps.
            let dragging_term = matches!(
                egui::DragAndDrop::payload::<DragPayload>(ui.ctx()).as_deref(),
                Some(DragPayload::Term(_))
            );
            let term_row_rect_id = ui.make_persistent_id("term-row-rect");
            let last_term_row_rect: Option<egui::Rect> =
                ui.ctx().memory(|m| m.data.get_temp(term_row_rect_id));
            let term_drop_idx = last_term_row_rect
                .and_then(|rect| drop_target_idx(ui.ctx(), dragging_term, rect, self.seq.len()));
            // Width of the currently-dragged term card, captured
            // last frame. Used as the gap's open width so the
            // gap matches the slot the card will eventually
            // occupy; without this, gap-vs-card width mismatch
            // produces a one-frame horizontal layout shift on
            // drop. Falls back to a sensible default.
            let dragged_term_idx =
                match egui::DragAndDrop::payload::<DragPayload>(ui.ctx()).as_deref() {
                    Some(DragPayload::Term(i)) => Some(*i),
                    _ => None,
                };
            let dragged_term_width = dragged_term_idx
                .map(|i| ui.make_persistent_id(("term-card", i)).with("width"))
                .and_then(|key| ui.ctx().memory(|m| m.data.get_temp::<f32>(key)))
                .unwrap_or(72.0);
            let term_row_resp = ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = CARD_ITEM_SPACING_X;
                // Term-reorder pre-pass: apply the reorder NOW
                // so the render loop sees the new order and the
                // gap is closed. Issue-#54 fix; shared via
                // `dnd_apply_drop_pre_pass`. Filters to the
                // `Term(_)` payload variant only; `Entry(_, _)`
                // payloads (cross-term plane migration) drop on
                // cards, not gaps.
                //
                // Critical: the pre-pass MUST run inside this
                // `horizontal_wrapped` closure, not on the outer
                // `ui`. `make_persistent_id` resolves through
                // the parent chain, so snapping animations on
                // the outer scope targets DIFFERENT ids than the
                // card render loop reads inside the inner
                // scope; the snap silently misses, and the
                // dragged card's pickup glow leaks onto the
                // card now sitting at the dragged card's old
                // index for one frame.
                let _ = dnd_apply_drop_pre_pass::<RotorTerm, DragPayload>(
                    ui,
                    &mut self.seq,
                    term_drop_idx,
                    |p| match p {
                        DragPayload::Term(i) => Some(*i),
                        _ => None,
                    },
                    "term-gap",
                    "term-card",
                    32,
                );
                let still_dragging_term = matches!(
                    egui::DragAndDrop::payload::<DragPayload>(ui.ctx()).as_deref(),
                    Some(DragPayload::Term(_))
                );
                let render_term_drop_idx = if still_dragging_term {
                    term_drop_idx
                } else {
                    None
                };
                for term_idx in 0..self.seq.len() {
                    let gap_id = ui.make_persistent_id(("term-gap", term_idx));
                    let _ = make_room_gap(
                        ui,
                        render_term_drop_idx == Some(term_idx),
                        gap_id,
                        term_h,
                        dragged_term_width,
                    );
                    if term_idx > 0 {
                        ui.label(egui::RichText::new("·").size(16.0).strong());
                    }
                    let card_id = ui.make_persistent_id(("term-card", term_idx));
                    let pickup_t = drag_pickup_t(ui.ctx(), card_id);
                    let stroke = if pickup_t > 0.0 {
                        egui::Stroke::new(
                            1.0 + pickup_t * 1.5,
                            egui::Color32::from_rgb(255, 200, 60),
                        )
                    } else {
                        egui::Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color)
                    };
                    // Term card: Frame is INSIDE the drag source so
                    // the entire card (background + stroke + math
                    // expression) follows the cursor as the tooltip
                    // when dragged. Drop detection for cross-term
                    // plane migration (`DragPayload::Entry`) is done
                    // manually on the card's rect; `dnd_drop_zone`
                    // would have to wrap the source, which forces
                    // the frame outside the source body.
                    let card_resp = dnd_drag_source_collapsing(
                        ui,
                        card_id,
                        DragPayload::Term(term_idx),
                        |ui| {
                            if ui.ctx().is_being_dragged(card_id) {
                                force_opaque_active(ui);
                            }
                            egui::Frame::default()
                                .fill(ui.visuals().widgets.noninteractive.bg_fill)
                                .stroke(stroke)
                                .inner_margin(3.0)
                                .corner_radius(egui::CornerRadius::same(3))
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        let term = &self.seq[term_idx];
                                        if let Some(phi) = term.scalar {
                                            let phi_color = egui::Color32::from_rgb(255, 150, 150);
                                            // Read-only label; the term card's drag source
                                            // captures pointer-down events at this rect, so
                                            // the previous editable DragValue couldn't reach
                                            // text-edit mode reliably. Editing now happens
                                            // through the right-click context menu, which
                                            // lives in its own popup layer.
                                            ui.add(
                                                egui::Label::new(
                                                    egui::RichText::new(format!(
                                                        "{:.2}°",
                                                        phi.to_degrees()
                                                    ))
                                                    .monospace()
                                                    .color(phi_color),
                                                )
                                                .selectable(false),
                                            )
                                            .on_hover_text(
                                                "Right-click the term to edit or remove",
                                            );
                                            ui.monospace("·");
                                        }
                                        let planes = self.seq[term_idx].planes.clone();
                                        render_plane_sum(ui, &planes, |ui, plane_idx, plane| {
                                            let pill_id = ui.make_persistent_id((
                                                "plane-pill",
                                                term_idx,
                                                plane_idx,
                                            ));
                                            ui.dnd_drag_source(
                                                pill_id,
                                                DragPayload::Entry(term_idx, plane_idx),
                                                |ui| {
                                                    ui.monospace(plane.label());
                                                },
                                            )
                                            .response
                                            .on_hover_cursor(egui::CursorIcon::Grab);
                                        });
                                    });
                                });
                        },
                    );
                    // Manual Entry drop detection on the card's rect.
                    // Skipped for the dragged card itself (its
                    // response rect is the placeholder, not the
                    // card body; and dropping a plane onto your own
                    // term is a no-op anyway).
                    let is_self_dragged = ui.ctx().is_being_dragged(card_id);
                    if !is_self_dragged {
                        let card_rect = card_resp.rect;
                        let dragging_entry = matches!(
                            egui::DragAndDrop::payload::<DragPayload>(ui.ctx()).as_deref(),
                            Some(DragPayload::Entry(_, _))
                        );
                        let cursor = ui.ctx().input(|i| i.pointer.hover_pos());
                        let hovered =
                            dragging_entry && cursor.is_some_and(|p| card_rect.contains(p));
                        if hovered && ui.ctx().input(|i| i.pointer.any_released()) {
                            if let Some(arc) =
                                egui::DragAndDrop::take_payload::<DragPayload>(ui.ctx())
                            {
                                if let DragPayload::Entry(from_t, idx) = *arc {
                                    if from_t != term_idx {
                                        entry_moves.push((from_t, idx, term_idx));
                                    }
                                }
                            }
                        }
                    }
                    // Capture this term's outer width so the make-
                    // room gap can match it on the frame this term
                    // is dragged. Skipped when this is the dragged
                    // card (its `card_resp.rect` is the collapsing
                    // placeholder, not the term's natural width).
                    if !ui.ctx().is_being_dragged(card_id) {
                        let width_key = card_id.with("width");
                        let w = card_resp.rect.width();
                        ui.ctx().memory_mut(|m| m.data.insert_temp(width_key, w));
                    }
                    let scalar_now = self.seq[term_idx].scalar;
                    let menu_resp = card_resp.interact(egui::Sense::click());
                    menu_resp.context_menu(|ui| {
                        if let Some(phi) = scalar_now {
                            let current_deg = phi.to_degrees();
                            ui.menu_button(format!("Edit scalar ({current_deg:.2}°)"), |ui| {
                                let mut deg = current_deg;
                                // DragValue lives inside this menu's
                                // popup layer, separate from the term
                                // card's drag source, so click-to-type
                                // and drag-to-scrub both work without
                                // the outer drag intercepting.
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut deg)
                                            .suffix("°")
                                            .speed(1.0)
                                            .fixed_decimals(2)
                                            .range(-720.0..=720.0),
                                    )
                                    .changed()
                                {
                                    edit_scalar = Some((term_idx, deg.to_radians()));
                                }
                            });
                            if ui.button("Remove scalar (φ)").clicked() {
                                remove_scalar = Some(term_idx);
                                ui.close_kind(egui::UiKind::Menu);
                            }
                        } else if ui.button("Add scalar (φ = 90°)").clicked() {
                            add_scalar = Some(term_idx);
                            ui.close_kind(egui::UiKind::Menu);
                        }
                        ui.separator();
                        if ui.button("Delete term").clicked() {
                            remove_term = Some(term_idx);
                            ui.close_kind(egui::UiKind::Menu);
                        }
                    });
                }
                // Trailing insertion gap: pre-pass already handled
                // any drop here; render with `is_target=false` so
                // the gap stays closed.
                let trailing_id = ui.make_persistent_id(("term-gap", self.seq.len()));
                let _ = make_room_gap(
                    ui,
                    render_term_drop_idx == Some(self.seq.len()),
                    trailing_id,
                    term_h,
                    dragged_term_width,
                );
                // Reset per-index term animation state when a
                // mutation will fire; same reasoning as the
                // shape-row reset: ids resolve correctly only
                // inside this ui scope.
                if !entry_moves.is_empty() || remove_term.is_some() {
                    let ctx = ui.ctx();
                    for i in 0..32 {
                        let card_id = ui.make_persistent_id(("term-card", i));
                        let _ = ctx.animate_value_with_time(card_id.with("pickup"), 0.0, 0.0);
                    }
                }
            });
            ui.ctx().memory_mut(|m| {
                m.data
                    .insert_temp(term_row_rect_id, term_row_resp.response.rect)
            });
        }

        // Apply deferred mutations in an order that keeps indices valid.
        if let Some(i) = add_scalar {
            if let Some(t) = self.seq.get_mut(i) {
                t.scalar = Some(std::f32::consts::FRAC_PI_2);
            }
        }
        if let Some(i) = remove_scalar {
            if let Some(t) = self.seq.get_mut(i) {
                t.scalar = None;
            }
        }
        if let Some((i, new_phi)) = edit_scalar {
            if let Some(t) = self.seq.get_mut(i) {
                if t.scalar.is_some() {
                    t.scalar = Some(new_phi);
                }
            }
        }
        // Cross-term entry migrations. Sort by (source term, plane idx
        // descending) so removals don't shift earlier indices.
        entry_moves.sort_by_key(|(from, idx, _)| (*from, std::cmp::Reverse(*idx)));
        for (from_t, idx, to_t) in entry_moves {
            if let Some(src) = self.seq.get_mut(from_t) {
                if idx < src.planes.len() {
                    let plane = src.planes.remove(idx);
                    if let Some(dest) = self.seq.get_mut(to_t) {
                        dest.planes.push(plane);
                    }
                }
            }
        }
        // Drop emptied terms (after entry moves).
        self.seq.retain(|t| !t.planes.is_empty());
        if let Some(i) = remove_term {
            if i < self.seq.len() {
                self.seq.remove(i);
            }
        }
    }

    /// Shape row + add-menu + drag-and-drop reorder. Extracted as a
    /// method so it can be called from both rotation modes (after
    /// the active-set checkboxes in `Active`, after the seq +
    /// Apply/Clear in `Composer`).
    fn render_shapes_section(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        let has_heavy = self
            .row
            .iter()
            .any(|e| e.shape == SHAPE_120CELL || e.shape == SHAPE_600CELL);
        if has_heavy {
            ui.colored_label(
                egui::Color32::from_rgb(242, 130, 70),
                "120/600-cell SDFs are heavy; expect <60 fps.",
            );
        }
        let mut row_changed = false;

        // Cards in a horizontally scrolling area: never wraps, so
        // resizing the panel doesn't reflow the row. Drop is via
        // an animated "make room" gap that opens at the cursor's
        // insertion slot during drag; no separate marker line
        // needed; the gap itself is the indicator.
        let mut remove_idx: Option<usize> = None;
        let row_len = self.row.len();
        let row_h = CONTROL_H;
        // Slot index where the drop should land. Computed once
        // from cursor position and last-frame's row geometry, so
        // every slot agrees on which one is "the target."
        let row_rect_id = ui.make_persistent_id("shape-row-rect");
        let last_row_rect: Option<egui::Rect> = ui.ctx().memory(|m| m.data.get_temp(row_rect_id));
        let dragging_shape = egui::DragAndDrop::payload::<usize>(ui.ctx()).is_some();
        let drop_idx =
            last_row_rect.and_then(|rect| drop_target_idx(ui.ctx(), dragging_shape, rect, row_len));
        let row_rect = egui::ScrollArea::horizontal()
            .auto_shrink([false, true])
            .id_salt("rotate-polytopes-shapes-scroll")
            .show(ui, |ui| {
                let row_response =
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Min), |ui| {
                        // Tighter inter-card spacing; the make-
                        // room gap takes over from item_spacing as
                        // the visual room-maker.
                        ui.spacing_mut().item_spacing.x = CARD_ITEM_SPACING_X;

                        // Drop pre-pass: apply the reorder NOW so the
                        // render loop sees the new order and gaps
                        // are closed. See `dnd_apply_drop_pre_pass`
                        // for the issue-#54 rationale.
                        if dnd_apply_drop_pre_pass::<ShapeEntry, usize>(
                            ui,
                            &mut self.row,
                            drop_idx,
                            |p| Some(*p),
                            "shape-gap",
                            "shape-card",
                            MAX_ROW_LEN,
                        ) {
                            row_changed = true;
                        }
                        // After the pre-pass the payload is gone, so
                        // `is_target` evaluates false in every gap
                        // and the render loop reflects the new
                        // ordering with all gaps closed.
                        let still_dragging =
                            egui::DragAndDrop::payload::<usize>(ui.ctx()).is_some();
                        let render_drop_idx = if still_dragging { drop_idx } else { None };
                        let row_len = self.row.len();
                        for (i, entry) in self.row.iter().enumerate() {
                            let gap_id = ui.make_persistent_id(("shape-gap", i));
                            let _ = make_room_gap(
                                ui,
                                render_drop_idx == Some(i),
                                gap_id,
                                row_h,
                                SHAPE_CARD_WIDTH + 8.0,
                            );
                            if Self::render_shape_card(ui, i, entry, row_len) {
                                remove_idx = Some(i);
                            }
                        }
                        let trailing_id = ui.make_persistent_id(("shape-gap", row_len));
                        let _ = make_room_gap(
                            ui,
                            render_drop_idx == Some(row_len),
                            trailing_id,
                            row_h,
                            SHAPE_CARD_WIDTH + 16.0,
                        );
                        // "+" trigger inline with the shape cards.
                        // Custom-painted plus on a 28×24 button rect so
                        // the height matches the cards exactly and the
                        // visual vocabulary matches the play / rate /
                        // chevron buttons (no font-glyph dependency).
                        if self.row.len() < MAX_ROW_LEN {
                            let plus_resp = add_button(ui, egui::vec2(CONTROL_W, CONTROL_H - 2.0))
                                .on_hover_text("Add a shape to the row");
                            egui::Popup::menu(&plus_resp).show(|ui| {
                                ui.set_min_width(140.0);
                                render_shape_catalog_menu(ui, |entry| {
                                    self.row.push(entry);
                                    row_changed = true;
                                });
                            });
                        }
                        // Per-index animation state is keyed by ids
                        // resolved against THIS ui's scope. After a
                        // reorder, the cards now sitting at the old
                        // indices would otherwise inherit the
                        // previous occupants' `pickup_t = 1.0` and
                        // ghost-fade. Snap defaults here, while the
                        // ui scope still resolves to the same ids
                        // we used during rendering; outside this
                        // closure, `ui.make_persistent_id(...)`
                        // would resolve to *different* ids.
                        if remove_idx.is_some() {
                            let ctx = ui.ctx();
                            for i in 0..=MAX_ROW_LEN {
                                let card_id = ui.make_persistent_id(("shape-card", i));
                                let _ =
                                    ctx.animate_value_with_time(card_id.with("pickup"), 0.0, 0.0);
                            }
                        }
                    });
                row_response.response.rect
            })
            .inner;
        ui.ctx()
            .memory_mut(|m| m.data.insert_temp(row_rect_id, row_rect));
        if let Some(i) = remove_idx {
            self.row.remove(i);
            row_changed = true;
        }
        if row_changed {
            self.rebuild_bodies();
        }
    }

    /// One shape card: drag source for reorder, hover-name
    /// tooltip, right-click "Remove from row" context menu.
    /// Returns `true` when the user clicked Remove this frame so
    /// the caller can record the index for end-of-frame removal
    /// (in-flight removal would invalidate the row's iteration).
    /// Card chrome (stroke colour, drag pickup glow, opaque-while-
    /// dragged frame) all lives here so the row-rendering loop
    /// reads as `gap, card, gap, card, ...` without each card
    /// inlining ~50 LOC of frame setup.
    fn render_shape_card(ui: &mut egui::Ui, i: usize, entry: &ShapeEntry, row_len: usize) -> bool {
        let card_id = ui.make_persistent_id(("shape-card", i));
        let pickup_t = drag_pickup_t(ui.ctx(), card_id);
        let card_fill = ui.visuals().widgets.noninteractive.bg_fill;
        let stroke_color = if pickup_t > 0.0 {
            egui::Color32::from_rgb(255, 200, 60)
        } else {
            ui.visuals().widgets.noninteractive.bg_stroke.color
        };
        let stroke = egui::Stroke::new(1.0 + pickup_t * 1.5, stroke_color);
        let drag_resp = dnd_drag_source_collapsing(ui, card_id, i, |ui| {
            if ui.ctx().is_being_dragged(card_id) {
                force_opaque_active(ui);
            }
            egui::Frame::default()
                .fill(card_fill)
                .stroke(stroke)
                .inner_margin(egui::Margin::symmetric(4, 6))
                .corner_radius(egui::CornerRadius::same(3))
                .show(ui, |ui| {
                    ui.allocate_ui_with_layout(
                        egui::vec2(SHAPE_CARD_WIDTH, 0.0),
                        egui::Layout::top_down(egui::Align::Center),
                        |ui| {
                            ui.add(
                                egui::Label::new(egui::RichText::new(entry.label).strong())
                                    .selectable(false)
                                    .wrap_mode(egui::TextWrapMode::Extend),
                            );
                        },
                    );
                });
        });
        let mut removed = false;
        drag_resp
            .on_hover_cursor(egui::CursorIcon::Grab)
            .on_hover_text(entry.long_name)
            .interact(egui::Sense::click())
            .context_menu(|ui| {
                if row_len > 1 && ui.button("Remove from row").clicked() {
                    removed = true;
                    ui.close_kind(egui::UiKind::Menu);
                }
            });
        removed
    }

    /// Top menu bar: Edit / View. Always visible.
    ///
    /// The File menu (New / Open / Save / Quit) is intentionally
    /// absent until those items are wired: project-settings
    /// persistence is a follow-up, and `Quit` via
    /// `ViewportCommand::Close` doesn't reliably close the root
    /// viewport in the current `rye-app` runner. Adding the
    /// menu back is a one-block edit once items are functional.
    fn render_menu_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("rotate-polytopes-menu-bar").show(ctx, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("Edit", |ui| {
                    if ui.button("Reset orientation").clicked() {
                        self.rot_state = Rotor4::IDENTITY;
                        self.write_all(self.rot_state);
                        ui.close_kind(egui::UiKind::Menu);
                    }
                    if ui
                        .add(egui::Button::new("Reset all").shortcut_text("R"))
                        .clicked()
                    {
                        self.reset();
                        ui.close_kind(egui::UiKind::Menu);
                    }
                });
                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.show_controls, "Rotation controls (H)");
                    ui.checkbox(&mut self.show_formula, "Formula popup");
                    ui.separator();
                    if ui.button("About this program").clicked() {
                        self.show_help = true;
                        ui.close_kind(egui::UiKind::Menu);
                    }
                });
            });
        });
    }

    fn render_help_window(&mut self, ctx: &egui::Context) {
        if !self.show_help {
            return;
        }
        let mut open = self.show_help;
        egui::Window::new("About 4D Polytope Rotation")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .movable(true)
            .default_size(egui::vec2(560.0, 460.0))
            .default_pos(egui::pos2(80.0, 80.0))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("What this program shows");
                    ui.label(
                        "You're looking at 3D cross-sections of four-dimensional \
                         polytopes. As they rotate through 4D space their cross-\
                         sections morph in characteristic ways; the point of the \
                         demo is to make 4D shape intuition reachable from 3D.",
                    );
                    ui.add_space(8.0);

                    ui.heading("3D cross-sections, briefly");
                    ui.label(
                        "A cross-section is what you get when a higher-\
                         dimensional object passes through a lower-dimensional \
                         space. A 3D apple intersecting a 2D table gives a 2D \
                         shape (a circle, an oval) that changes as the apple \
                         moves. One dimension up: a 4D polytope passing through \
                         3D gives a 3D shape that changes with the slicing w. \
                         That's what the w slider scrubs.",
                    );
                    ui.add_space(8.0);

                    ui.heading("The shapes");
                    ui.label("All six convex regular 4-polytopes (\"polychora\") ship:");
                    ui.label("• 5-cell (pentachoron); 5 tetrahedra; the 4D simplex.");
                    ui.label("• 8-cell (tesseract); 8 cubes; the 4D cube.");
                    ui.label(
                        "• 16-cell (hexadecachoron); 16 tetrahedra; the 4D analog \
                         of the octahedron.",
                    );
                    ui.label(
                        "• 24-cell (icositetrachoron); 24 octahedra; uniquely 4D, \
                         no 3D analog.",
                    );
                    ui.label("• 120-cell (hecatonicosachoron); 120 dodecahedra.");
                    ui.label(
                        "• 600-cell (hexacosichoron); 600 tetrahedra; the 4D \
                         analog of the icosahedron.",
                    );
                    ui.add_space(8.0);

                    ui.heading("Rotation");
                    ui.label(
                        "4D rotations are generated by bivectors (2-planes), not \
                         axes. There are six independent planes: xy, xz, xw, yz, \
                         yw, zw. The three w-involving planes pull a visible \
                         axis through the hidden 4th dimension and produce the \
                         interesting cross-section morphs; the three pure-3D \
                         planes rotate the cross-section as a rigid 3D shape.",
                    );
                    ui.label(
                        "Active-set mode: each plane has a checkbox (include in \
                         spin) and a -180..=180° slider (the rotor's component \
                         in that plane). Composer mode: build a sequence of \
                         exp(scalar · planes) terms via chips or the typed \
                         formula bar.",
                    );
                    ui.add_space(8.0);

                    ui.heading("Views");
                    ui.label(
                        "Shapes view: a row of polytopes side-by-side at one \
                         w-slice. Drag-and-drop to reorder. Filmstrip view: one \
                         polytope rendered N times across w-slices fanning out \
                         by ±BODY_SIZE around the slider's value, so the centre \
                         cell tracks w.",
                    );
                    ui.add_space(8.0);

                    ui.heading("Keyboard");
                    ui.label("• Space / T: toggle continuous spin.");
                    ui.label("• Up / Down arrows: scrub w with the keyboard.");
                    ui.label("• 1..6: toggle a plane in the Active set.");
                    ui.label("• H: expand / collapse the controls panel.");
                    ui.label("• R: full reset.");
                    ui.label("• Esc: exit.");
                    ui.add_space(8.0);

                    ui.heading("Mouse");
                    ui.label("• Drag in the viewport: orbit camera.");
                    ui.label(
                        "• Right-click on any value label (w, t, plane angle, \
                         scalar): typed-edit popup.",
                    );
                    ui.label(
                        "• Drag the controls panel by its frame to move it; \
                         drag the formula popup the same way.",
                    );
                });
            });
        self.show_help = open;
    }

    /// Unified controls overlay. `egui::Window` with
    /// `pivot(CENTER_BOTTOM)` so the bottom edge is the anchor
    /// and the panel grows upward when the expanded body is
    /// shown. Always draggable. On first frame the default
    /// position is bottom-centre of the viewport; subsequent
    /// frames remember whatever position the user dragged it to.
    fn render_overlay(&mut self, ctx: &egui::Context) {
        let screen = ctx.content_rect();
        let pad = 16.0;
        let natural_w = (screen.width() - 2.0 * pad).max(280.0);
        let pinned = *self.overlay_pinned_width.get_or_insert(natural_w);
        let area_w = pinned.min(natural_w).max(280.0);

        let visuals = &ctx.style().visuals;
        let frame = egui::Frame::default()
            .fill(visuals.window_fill)
            .stroke(visuals.window_stroke)
            .corner_radius(visuals.window_corner_radius)
            .inner_margin(10.0);

        let default_bottom_centre = egui::pos2(screen.center().x, screen.bottom() - pad);

        egui::Window::new("rotate-polytopes-overlay")
            .id(egui::Id::new("rotate-polytopes-overlay"))
            .title_bar(false)
            .resizable(false)
            .collapsible(false)
            .movable(true)
            // `auto_sized()` forces the Window to recompute its
            // outer rect from current content every frame.
            // Without it the saved rect from the previous frame
            // is reused, which on an expand-toggle one-frame
            // glitches: the new content is shorter/taller than
            // the saved rect, the Window briefly clips or
            // mispositions before egui catches up. The user-
            // visible effect was a one-frame disappear of the
            // panel after toggling `^` / `v`.
            .auto_sized()
            .pivot(egui::Align2::CENTER_BOTTOM)
            .default_pos(default_bottom_centre)
            .default_width(area_w)
            .frame(frame)
            .show(ctx, |ui| {
                ui.set_width(area_w);
                if self.expanded {
                    self.render_expanded_body(ui);
                    ui.separator();
                }
                self.render_slider_strip(ui, area_w);
                self.render_rate_row(ui);
            });

        // Apply any deferred state changes AFTER the overlay
        // finishes rendering, so both BottomOverlay passes saw
        // the same content this frame. Effective on the next
        // frame.
        if let Some(new_mode) = self.pending_mode.take() {
            self.rotation_mode = new_mode;
        }
        if let Some(new_view) = self.pending_view_mode.take() {
            self.view_mode = new_view;
        }
        for action in std::mem::take(&mut self.pending_actions) {
            match action {
                DeferredAction::DraftPush(plane) => self.draft.push(plane),
                DeferredAction::SeqCommitDraft => {
                    if !self.draft.is_empty() {
                        self.seq.push(RotorTerm {
                            planes: self.draft.clone(),
                            scalar: None,
                        });
                        self.draft.clear();
                    }
                }
                DeferredAction::DraftClear => self.draft.clear(),
                DeferredAction::SeqPushTerm(term) => self.seq.push(term),
            }
        }
    }

    /// Two big sliders (w, t) with fixed-width monospace value
    /// labels.
    fn render_slider_strip(&mut self, ui: &mut egui::Ui, _area_w: f32) {
        // Sliders use the shared `slider_with_edit` helper, which
        // hides the in-slider value display (so click-on-value
        // can't accidentally drag) and renders the value in a
        // fixed-width side-Label cell. The fixed cell width keeps
        // the slider's right edge stable as the value's char
        // count varies (`0.5` -> `12.34`); without it the entire
        // overlay frame would oscillate each frame as the spin
        // advances `rot_time`. Right-click on the value cell
        // opens an Edit popup with a real DragValue.
        //
        // Slider width is computed from the row's actual
        // `available_width()` (post inner-margin), not the
        // outer `area_w`, so there's no dead space at the right
        // edge when the Window's frame margin shrinks the
        // usable width below `area_w`.
        const VALUE_CELL_W: f32 = 86.0;
        let avail = ui.available_width();
        let spacing = ui.spacing().item_spacing.x;
        let slider_w = (avail - VALUE_CELL_W - spacing).max(140.0);
        ui.spacing_mut().slider_width = slider_w;

        // w / t rows allocate `CONTROL_H` tall so the vertical
        // pitch matches the rotor-plane sliders above (which
        // pin each cell to `CONTROL_H` via
        // `allocate_ui_with_layout`). Without this the w/t
        // rows are ~22 px tall and the rotor rows are 29 px,
        // giving the strip a noticeably looser feel mid-panel.
        let row_size = egui::vec2(avail, CONTROL_H);
        let row_layout = egui::Layout::left_to_right(egui::Align::Center);
        ui.allocate_ui_with_layout(row_size, row_layout, |ui| {
            let formatted = format!("w {:>+.3}", self.w_slice);
            slider_with_edit(
                ui,
                &mut self.w_slice,
                -W_RANGE..=W_RANGE,
                &formatted,
                "",
                3,
                VALUE_CELL_W,
            );
        });
        // Gate the scrub-from-zero recomputation on the slider
        // *being dragged* so it ONLY fires while the user is
        // actively scrubbing. Using `.changed()` would misfire
        // every frame the spin's `rot_time += dt_secs`
        // accumulator advanced the value, producing a snap when
        // toggling active checkboxes (omega would shift, and
        // re-deriving `exp(omega_new * t)` from an accumulated
        // `t` is a discontinuous jump rather than the smooth
        // integrated path). The Edit-popup path doesn't share
        // this hazard since its DragValue only fires on user
        // input; the `t_dragged` gate is intentionally
        // slider-only.
        // `t_slider_max` is grown by `update()` when the spin
        // pushes `rot_time` past the current bound; we don't
        // grow it here because (a) the slider clamps user
        // input to its current range so dragging can't push
        // `rot_time` past it without the spin's help, and
        // (b) growing here while the user is actively
        // dragging at the right edge causes a feedback loop
        // (drag clamps to t_max, spin adds dt, t_max doubles,
        // drag's screen-x re-maps to the new t_max value,
        // repeat).
        let t_max = self.t_slider_max;
        let mut t_dragged = false;
        ui.allocate_ui_with_layout(row_size, row_layout, |ui| {
            let formatted = format!("t {:>5.2}s", self.rot_time);
            let slider_resp = ui.add(
                egui::Slider::new(&mut self.rot_time, 0.0..=t_max)
                    .show_value(false)
                    .smart_aim(false)
                    .clamping(egui::SliderClamping::Always),
            );
            t_dragged = slider_resp.dragged();
            ui.allocate_ui_with_layout(
                egui::vec2(VALUE_CELL_W, 14.0),
                egui::Layout::left_to_right(egui::Align::Center),
                |ui| {
                    let label_resp = ui.add(
                        egui::Button::new(egui::RichText::new(formatted).monospace())
                            .frame(false)
                            .small(),
                    );
                    label_resp
                        .on_hover_cursor(egui::CursorIcon::ContextMenu)
                        .on_hover_text("Right-click to edit value")
                        .context_menu(|ui| {
                            ui.add(
                                egui::DragValue::new(&mut self.rot_time)
                                    .range(0.0..=f32::INFINITY)
                                    .suffix("s")
                                    .fixed_decimals(2),
                            );
                        });
                },
            );
        });
        if t_dragged {
            // Scrub uses the rate-independent
            // `omega_animation`; `rot_time` is animation time
            // (already rate-scaled at integration), so
            // `exp(omega_animation * rot_time)` equals what the
            // continuous-spin path would have integrated.
            let omega = self.omega_animation();
            self.rot_state = (omega * self.rot_time).exp().normalize();
            self.write_all(self.rot_state);
        }
    }

    /// Always-visible single row directly under the sliders.
    /// Center-justified play / rate / refresh cluster with the
    /// right-aligned utility cluster on the same line:
    ///
    /// ```text
    ///                  [<<] [<] [play/pause] [>] [>>] [refresh]    [?] [^]
    /// ```
    ///
    /// Rate buttons toggle: clicking a highlighted preset clears it
    /// back to ×1.00 (the default) so the user can switch out of a
    /// non-default rate without resetting everything.
    fn render_rate_row(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            // True row centering: leading pad = (total_w - group_w) /
            // 2 so the play group's center sits exactly at the row's
            // midpoint. The right cluster is anchored to the right
            // edge separately via Layout::right_to_left, so it
            // doesn't enter into the centering math at all.
            //
            // `PLAY_GROUP_W` is empirically the natural width of the
            // 6-widget cluster (`<<` `<` play/pause `>` `>>` refresh plus default
            // item spacing). If button labels or padding change,
            // re-tune here.
            const PLAY_GROUP_W: f32 = 215.0;
            let total_w = ui.available_width();
            let leading = ((total_w - PLAY_GROUP_W) / 2.0).max(8.0);

            ui.add_space(leading);
            let ctrl_size = egui::vec2(CONTROL_W, CONTROL_H);
            let play_size = egui::vec2(PLAY_PAUSE_W, CONTROL_H);
            rate_toggle(ui, ctrl_size, &mut self.rate_scale, 0.25, true, false);
            rate_toggle(ui, ctrl_size, &mut self.rate_scale, 0.5, false, false);
            if play_pause_button(ui, play_size, self.rotate)
                .on_hover_text("Toggle continuous rotation (Space)")
                .clicked()
            {
                self.rotate = !self.rotate;
            }
            rate_toggle(ui, ctrl_size, &mut self.rate_scale, 2.0, false, true);
            rate_toggle(ui, ctrl_size, &mut self.rate_scale, 4.0, true, true);
            if refresh_button(ui, ctrl_size)
                .on_hover_text("Reset slice, rate, active set, orientation, time (R)")
                .clicked()
            {
                self.reset();
            }

            // Right cluster: claims the rest of the row with a
            // right-to-left sub-layout so widgets stack at the right
            // edge regardless of where the cursor is. Reset moved to
            // the play group as refresh; this cluster is now just the
            // help and expand toggles.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if chevron_button(
                    ui,
                    egui::vec2(CONTROL_W, CONTROL_H),
                    !self.expanded,
                    if self.expanded {
                        "Collapse (H)"
                    } else {
                        "Expand controls (H)"
                    },
                )
                .clicked()
                {
                    self.expanded = !self.expanded;
                }
                if ui
                    .add(
                        egui::Button::new(egui::RichText::new("?").strong())
                            .min_size(egui::vec2(MINI_BUTTON_W, MINI_BUTTON_W)),
                    )
                    .on_hover_text("About this program")
                    .clicked()
                {
                    self.show_help = true;
                }
            });
        });
    }

    /// Rebuild the full body uniform array from the current row +
    /// rotor. Use this when the row's length or order changes; the
    /// per-body position depends on the row's `n` and the body's slot
    /// index, so a single body update is not enough.
    fn rebuild_bodies(&mut self) {
        let n = self.row.len();
        let rotor = self.rot_state;
        let bodies: Vec<BodyUniform> = self
            .row
            .iter()
            .enumerate()
            .map(|(slot, entry)| {
                BodyUniform::polytope_with_rotor(
                    body_position(slot, n),
                    entry.shape,
                    BODY_SIZE,
                    rotor,
                    entry.body_color,
                )
            })
            .collect();
        self.node.set_bodies(&bodies);
    }

    /// Render a compact `exp(B · 0.30·t)` form for whichever mode
    /// drives the spin. `B` is the bivector velocity expression: a
    /// sum of plane terms (Active mode: each enabled plane is one
    /// unit-bivector term; Composer mode: each seq entry is its
    /// scalar-weighted bivector). Empty string when nothing is
    /// contributing.
    fn formula_string(&self) -> String {
        let parts: Vec<String> = match self.rotation_mode {
            RotationMode::Active => Plane4::ALL
                .iter()
                .zip(self.active.iter())
                .filter(|(_, on)| **on)
                .map(|(p, _)| p.label().to_string())
                .collect(),
            RotationMode::Composer => self.seq.iter().map(render_term).collect(),
        };
        match render_bivector_sum(&parts) {
            Some(bivec) => format!(
                "exp({} · {:.2}·t)",
                bivec,
                BASE_ROTATION_RATE / std::f32::consts::TAU
            ),
            None => String::new(),
        }
    }

    /// Full reset: pause spin, slice, rate, active set, orientation,
    /// time, draft. Reset implies "stop", so `rotate` flips off too;
    /// otherwise the next frame's `update()` would immediately start
    /// spinning the freshly-reset state, which the user almost never
    /// wants.
    fn reset(&mut self) {
        self.rotate = false;
        self.w_slice = 0.0;
        self.rate_scale = 1.0;
        // Restore the xw-only default active set so a first-time
        // user resetting and then hitting Spin sees motion.
        self.active = [false, false, true, false, false, false];
        self.rot_state = Rotor4::IDENTITY;
        self.rot_time = 0.0;
        self.t_slider_max = T_SLIDER_INITIAL;
        self.draft.clear();
        self.write_all(Rotor4::IDENTITY);
    }
}

impl App for RotatePolytopesApp {
    type Space = EuclideanR3;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let row = parse_row_from_args()?;
        if row.is_empty() {
            return Err(anyhow!("--shapes produced an empty row"));
        }

        let scene = Scene4::new(SceneNode4::halfspace(Vec4::Y, 0.0));
        // Always include the extended polytope WGSL so any of the six
        // shapes can be added to the row at runtime via the panel.
        // The ~24 KB const-array cost is fixed per app and acceptable
        // for a viz/demo target.
        let shader_source = format!(
            "{kernel}\n{polytope}\n{scene}\n",
            kernel = HYPERSLICE_KERNEL_WGSL,
            polytope = polytope_extended_sdfs_wgsl(),
            scene = scene.to_hyperslice_wgsl("u.w_slice"),
        );
        let module = ctx
            .rd
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rotate_polytopes shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });
        let mut node =
            Hyperslice4DNode::new(&ctx.rd.device, ctx.rd.surface_bundle.config.format, &module);

        let n = row.len();
        let bodies: Vec<BodyUniform> = row
            .iter()
            .enumerate()
            .map(|(slot, entry)| {
                BodyUniform::polytope_with_rotor(
                    body_position(slot, n),
                    entry.shape,
                    BODY_SIZE,
                    Rotor4::IDENTITY,
                    entry.body_color,
                )
            })
            .collect();
        node.set_bodies(&bodies);

        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 3.0, 9.0);
        let mut orbit: OrbitController<EuclideanR3> = OrbitController::default();
        // Wider orbit so all four bodies in the row are visible at
        // default zoom; user can scroll-zoom in.
        orbit.set_orbit(9.5, -0.25);

        // Always start at w=0 regardless of row contents. Auto-shifting
        // to the 120/600-cell's "Platonic-named" cross-section was
        // confusing in mixed rows: the other shapes' slices got pulled
        // off-centre. Users who want the dodecahedral / icosahedral
        // view scrub there with the slider.
        let initial_w = 0.0;

        Ok(Self {
            space: EuclideanR3,
            camera,
            orbit,
            node,
            row,
            w_slice: initial_w,
            slider_up_held: false,
            slider_down_held: false,
            rotate: false,
            rot_state: Rotor4::IDENTITY,
            // Default: xw spin enabled (active[2] = Plane4::Xw). A
            // first-time user who hits "Spin" before toggling any
            // checkbox now sees motion immediately; the most
            // characteristic 4D rotation, pulling the visible x-axis
            // through the hidden w-axis.
            active: [false, false, true, false, false, false],
            rate_scale: 1.0,
            rot_time: 0.0,
            t_slider_max: T_SLIDER_INITIAL,
            expanded: false,
            show_help: false,
            overlay_pinned_width: None,
            show_formula: false,
            show_controls: true,
            view_mode: ViewMode::Shapes,
            strip_w: true,
            strip_t: false,
            strip_swap_axes: false,
            strip_count_w: 11,
            strip_count_t: 5,
            // Match the t slider's initial range
            // (`T_SLIDER_INITIAL`) so a row of t cells covers the
            // same animation interval the t slider can scrub
            // through at high precision.
            strip_t_extent: T_SLIDER_INITIAL,
            strip_subject: SHAPE_CATALOG[3],
            rotation_mode: RotationMode::Active,
            pending_mode: None,
            pending_view_mode: None,
            pending_actions: Vec::new(),
            seq: Vec::new(),
            draft: Vec::new(),
            formula_input: String::new(),
            formula_error: None,
        })
    }

    fn space(&self) -> &EuclideanR3 {
        &self.space
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        let dt_secs = ctx.n_ticks as f32 / 60.0;

        // Slice scrub.
        let dir = (self.slider_up_held as i32 - self.slider_down_held as i32) as f32;
        if dir != 0.0 {
            self.w_slice = (self.w_slice + dir * W_SCRUB_RATE * dt_secs).clamp(-W_RANGE, W_RANGE);
        }

        // 4D rotation animation. Both bodies share the same rotor
        // so the user can directly compare their slice signatures
        // under identical 4D motion. `rot_state` is the spin
        // baseline; the manual-rotation window's sliders ride on
        // top as a transient display offset (composed at write_all
        // time), so the user can scrub orientation while the spin
        // is running without disturbing the spin itself.
        if self.rotate {
            // Animation time advances by `dt_real * rate_scale`
            // so the rate buttons make `t` count faster/slower
            // (per-real-second). The integrated rotation is
            // `exp(omega_animation * dt_animation)` per frame,
            // which = `exp(omega_animation * rate_scale * dt_real)`.
            // This way `rot_state` and `rot_time` stay in sync:
            // dragging `t` to N reproduces what the spin would
            // have integrated to at animation time N, regardless
            // of how the rate varied along the way.
            let dt_animation = dt_secs * self.rate_scale;
            self.rot_time += dt_animation;
            // Grow the t-slider's max range when the spin has
            // pushed `rot_time` past it, capped so the value
            // can't run away if (e.g.) `rate_scale` is huge or
            // the demo is left running for days. 1e6 seconds
            // (~12 days at ×1) is past any realistic use; if
            // we hit it, `rot_time` clamps to the cap.
            const T_SLIDER_CAP: f32 = 1.0e6;
            if self.rot_time > self.t_slider_max {
                let new_max = (self.rot_time * 2.0).min(T_SLIDER_CAP);
                self.t_slider_max = new_max;
                if self.rot_time > T_SLIDER_CAP {
                    self.rot_time = T_SLIDER_CAP;
                }
            }
            let omega = self.omega_animation() * dt_animation;
            if omega.magnitude_squared() > 0.0 {
                let delta = omega.exp();
                self.rot_state = (delta * self.rot_state).normalize();
            }
        }
        self.write_all(self.rot_state);

        // Camera. Gate the orbit on `!ui_has_focus` so dragging the
        // egui w-slice slider doesn't also rotate the camera.
        //
        // In 2D grid filmstrip mode the body sits low in each
        // cell because the orbit target is at y = 0 (origin)
        // while the body is at y = BODY_Y; that puts the body
        // near the horizon and crowds the polytope at the
        // bottom of every cell. Lifting the orbit target up to
        // body height re-centres the polytope vertically in
        // each cell so the grid reads as a tidy matrix instead
        // of a row of horizon shots.
        let lift_orbit = self.view_mode == ViewMode::Filmstrip && self.strip_w && self.strip_t;
        self.orbit.target.y = if lift_orbit { BODY_Y } else { 0.0 };
        use rye_camera::CameraController;
        if !ctx.ui_has_focus {
            self.orbit
                .advance(ctx.input, &mut self.camera, &EuclideanR3, 0.0);
        }
        let view = self.camera.view();

        // Hyperslice uniforms.
        let cfg = &ctx.rd.surface_bundle.config;
        {
            let u = self.node.uniforms_mut();
            u.camera_pos = view.position.to_array();
            u.camera_forward = view.forward.to_array();
            u.camera_right = view.right.to_array();
            u.camera_up = view.up.to_array();
            u.fov_y_tan = (60.0_f32.to_radians() * 0.5).tan();
            u.resolution = [cfg.width as f32, cfg.height as f32];
            u.time = ctx.time;
            u.tick = ctx.tick as f32;
            u.w_slice = self.w_slice;
        }
        self.node.flush_uniforms(&ctx.rd.queue);
    }

    fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
        // Disable Ctrl+/Ctrl- keyboard-zoom. egui's built-in zoom
        // changes pixels_per_point but the wgpu surface stays at the
        // native resolution, so the scene ends up letter-boxed
        // (black bars) and the tessellator complains about clipped
        // geometry. UI scale stays at native PPP; the scene already
        // supports mouse-wheel orbit-zoom.
        ctx.options_mut(|o| o.zoom_with_keyboard = false);

        // Menu bar always visible at the top. Renders before
        // every other UI so its docked space is reserved (and
        // `ctx.content_rect()` reflects the area below it for
        // subsequent positioning calculations).
        self.render_menu_bar(ctx);

        // Top-left: title + fps + framebuffer size. Replaces the old
        // panel header now that the side panel is gone. Larger
        // typography so the title reads as the program's nameplate
        // rather than just another label.
        let cfg = &frame.rd.surface_bundle.config;
        let (fb_w, fb_h) = (cfg.width, cfg.height);
        // y offset clears the menu bar (~24-28px depending on
        // font) plus a small visual margin. egui::Area's anchor
        // is screen-relative, not content-rect-relative, so the
        // offset must include the menu bar height manually.
        egui::Area::new(egui::Id::new("rotate-polytopes-title"))
            .anchor(egui::Align2::LEFT_TOP, [20.0, 50.0])
            .show(ctx, |ui| {
                ui.add(egui::Label::new(
                    egui::RichText::new("4D Polytope Rotation")
                        .size(22.0)
                        .strong()
                        .color(egui::Color32::WHITE),
                ));
                ui.add(egui::Label::new(
                    egui::RichText::new(format!("{:.0} fps   {}×{}", frame.fps, fb_w, fb_h))
                        .size(13.0)
                        .color(egui::Color32::from_gray(190)),
                ));
            });

        // Live rotation formula popup, plus combo name (Active
        // mode) and the rotor's bivector decomposition matrix.
        // Defaults to top-right; freely draggable. Off by
        // default; toggled by the "Show formula" checkbox.
        if self.show_formula {
            let formula = self.formula_string();
            let name = if self.rotation_mode == RotationMode::Active {
                combo_name(&self.active)
            } else {
                None
            };
            let bivec = self.rot_state.log();
            let screen = ctx.content_rect();
            let default_pos = egui::pos2(screen.right() - 280.0, screen.top() + 16.0);
            let popup_frame = egui::Frame::popup(&ctx.style()).inner_margin(8.0);
            // Cap width so a long formula or term sum doesn't
            // make the popup expand off-screen. The matrix's
            // intrinsic width sets the lower bound (~280 px);
            // formula and combo-name labels wrap inside.
            const FORMULA_POPUP_W: f32 = 320.0;
            egui::Window::new("formula")
                .id(egui::Id::new("rotate-polytopes-formula"))
                .title_bar(false)
                .resizable(false)
                .collapsible(false)
                .movable(true)
                .default_pos(default_pos)
                .default_width(FORMULA_POPUP_W)
                .max_width(FORMULA_POPUP_W)
                .frame(popup_frame)
                .show(ctx, |ui| {
                    ui.set_max_width(FORMULA_POPUP_W);
                    if !formula.is_empty() {
                        ui.add(egui::Label::new(egui::RichText::new(&formula).monospace()).wrap());
                    }
                    if let Some(n) = name {
                        ui.add(egui::Label::new(
                            egui::RichText::new(n).color(egui::Color32::from_rgb(255, 217, 140)),
                        ));
                    }
                    ui.separator();
                    ui.label(egui::RichText::new("log(R) bivector").small().weak());
                    render_bivector_matrix(ui, &bivec);
                });
        }

        // Filmstrip cell labels: per-cell `w` annotation overlaid
        // on top of the rendered scene so users can see which cell
        // tracks the slider and read the cell-by-cell w sweep.
        if self.view_mode == ViewMode::Filmstrip {
            self.render_filmstrip_cell_labels(ctx);
        }

        // Bottom-anchored unified controls overlay. Hidden by
        // default; toggle via `View > Rotation controls` or `H`.
        if self.show_controls {
            self.render_overlay(ctx);
        }

        // Modal help window (opened by the `?` button).
        self.render_help_window(ctx);
    }

    fn on_event(&mut self, ev: &winit::event::WindowEvent, _ctx: &mut FrameCtx<'_>) {
        use winit::event::{ElementState, WindowEvent};
        use winit::keyboard::{KeyCode, PhysicalKey};
        let WindowEvent::KeyboardInput { event, .. } = ev else {
            return;
        };
        let PhysicalKey::Code(kc) = event.physical_key else {
            return;
        };
        let pressed = event.state == ElementState::Pressed;
        match kc {
            KeyCode::ArrowUp => self.slider_up_held = pressed,
            KeyCode::ArrowDown => self.slider_down_held = pressed,
            KeyCode::KeyR if pressed => self.reset(),
            KeyCode::KeyH if pressed => self.show_controls = !self.show_controls,
            KeyCode::KeyT | KeyCode::Space if pressed => {
                // Pause / resume only, DO NOT touch rot_state. The
                // bodies keep their current orientation when paused
                // and resume from there when toggled back on. Both
                // T (legacy) and Space (media-player convention)
                // bind to the same toggle.
                self.rotate = !self.rotate;
            }
            // Plane toggles. Sum-of-bivectors composition is
            // commutative, so the order in which planes are toggled
            // doesn't affect the resulting motion, only the active
            // set matters.
            KeyCode::Digit1 | KeyCode::Numpad1 if pressed => self.active[0] = !self.active[0],
            KeyCode::Digit2 | KeyCode::Numpad2 if pressed => self.active[1] = !self.active[1],
            KeyCode::Digit3 | KeyCode::Numpad3 if pressed => self.active[2] = !self.active[2],
            KeyCode::Digit4 | KeyCode::Numpad4 if pressed => self.active[3] = !self.active[3],
            KeyCode::Digit5 | KeyCode::Numpad5 if pressed => self.active[4] = !self.active[4],
            KeyCode::Digit6 | KeyCode::Numpad6 if pressed => self.active[5] = !self.active[5],
            _ => {}
        }
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        // Scene renders to the full window. The bottom controls
        // overlay floats on top; `BottomOverlay` is an Area, not
        // a docked panel, so the scene viewport doesn't need to
        // skip a bottom strip.
        let cfg = &rd.surface_bundle.config;
        let viewport = Viewport::full([cfg.width, cfg.height]);
        if self.view_mode == ViewMode::Filmstrip {
            // Filmstrip: each cell shows the `strip_subject`
            // polytope (independent of `self.row`) at a different
            // `w_slice`. We swap the GPU body list to just the
            // subject for the duration of this render, then
            // restore via `rebuild_bodies` so the Shapes view and
            // any subsequent state read sees the full row.
            let entry = self.strip_subject;
            // 2D filmstrip rendering. cols is the column count
            // (horizontal axis), rows is the row count (vertical).
            // Default axis assignment: w on columns, t on rows;
            // `strip_swap_axes` flips it.
            //
            // Per-cell rendering: viewport (cell rect), w_slice
            // (cell's w), and body (cell's rotor for that t).
            // The base rotor `self.rot_state` is offset along
            // omega_animation by `(t_offset)` to give the cell's
            // rotor: `exp(omega_animation * t_offset) * rot_state`.
            // For the w-only and t-only 1D cases, the second
            // axis collapses to a single cell with offset=0.
            let strip_w_extent = BODY_SIZE;
            let (cols, rows, w_on_cols) = match (self.strip_w, self.strip_t) {
                (true, true) => {
                    if self.strip_swap_axes {
                        (self.strip_count_t, self.strip_count_w, false)
                    } else {
                        (self.strip_count_w, self.strip_count_t, true)
                    }
                }
                (true, false) => (self.strip_count_w, 1, true),
                (false, true) => (1, self.strip_count_t, false),
                // UI invariant prevents both being off; defensive.
                (false, false) => (1, 1, true),
            };
            let col_vps = viewport.split_horizontal(cols as u32);
            let omega = self.omega_animation();
            let mut grid_cells: Vec<(Viewport, f32, BodyUniform)> = Vec::with_capacity(cols * rows);
            for (col_idx, col_vp) in col_vps.into_iter().enumerate() {
                let row_vps = col_vp.split_vertical(rows as u32);
                for (row_idx, cell_vp) in row_vps.into_iter().enumerate() {
                    // Decide what (w_offset, t_offset) this cell
                    // corresponds to based on which axis carries
                    // which dimension.
                    let (w_idx, w_n, t_idx, t_n) = if w_on_cols {
                        (col_idx, cols, row_idx, rows)
                    } else {
                        (row_idx, rows, col_idx, cols)
                    };
                    let w_t = if w_n <= 1 {
                        0.5
                    } else {
                        w_idx as f32 / (w_n - 1) as f32
                    };
                    let w_offset = -strip_w_extent + w_t * (2.0 * strip_w_extent);
                    let cell_w_slice = self.w_slice + w_offset;
                    let t_offset = if !self.strip_t || t_n <= 1 {
                        0.0
                    } else {
                        // Fan FORWARD only: cell 0 = now, cell
                        // last = rot_time + strip_t_extent. Reads
                        // as "what the rotor will look like at
                        // this future time."
                        let t_norm = t_idx as f32 / (t_n - 1) as f32;
                        t_norm * self.strip_t_extent
                    };
                    // Cell's rotor: spin from `rot_state` by
                    // `omega * t_offset` (animation-time offset).
                    let cell_rotor = if t_offset == 0.0 {
                        self.rot_state
                    } else {
                        ((omega * t_offset).exp() * self.rot_state).normalize()
                    };
                    let body = BodyUniform::polytope_with_rotor(
                        [0.0, BODY_Y, 0.0, 0.0],
                        entry.shape,
                        BODY_SIZE,
                        cell_rotor,
                        entry.body_color,
                    );
                    grid_cells.push((cell_vp, cell_w_slice, body));
                }
            }
            let result = self.node.execute_strip(rd, view, &grid_cells);
            // Restore the full row of bodies for any non-strip
            // consumer (state save, mode switch, etc.).
            self.rebuild_bodies();
            result
        } else {
            {
                let u = self.node.uniforms_mut();
                u.resolution = viewport.resolution_f32();
                u.viewport_origin = [viewport.x as f32, viewport.y as f32];
            }
            self.node.flush_uniforms(&rd.queue);
            self.node.execute_in_viewport(rd, view, viewport)
        }
    }

    fn title(&self, _fps: f32) -> std::borrow::Cow<'static, str> {
        // Window title is now decorative, all live state is in the
        // overlay. Keep the title static so OS task switchers show
        // a stable label.
        std::borrow::Cow::Borrowed("rotate polytopes")
    }
}

fn main() -> Result<()> {
    let config = RunConfig {
        window: WindowAttributes::default()
            .with_title("rotate polytopes")
            .with_visible(false),
        ..RunConfig::default()
    };
    run_with_config::<RotatePolytopesApp>(config)
}

// ---------------------------------------------------------------------------
// Layout regression tests
// ---------------------------------------------------------------------------
//
// `cargo test --example rotate_polytopes` to run.
//
// These tests headless-render the shape row through `egui::Context::run`
// and inspect the actual placed-rect positions of every card and the
// trailing `+` button. They guard against the "descending staircase"
// regression where adding a long-label shape (120/600-cell) caused
// label-wrapping to grow that card's frame, which in turn pushed
// egui's horizontal Center cross-alignment to recompute against a
// new max-height; leaving earlier cards aligned to the old (lower)
// center while the new card centered higher.
//
// `egui::Context` works fine without a renderer for layout-only
// tests; nothing here touches the GPU.

#[cfg(test)]
mod alignment_tests {
    use super::*;

    /// Headless-render the same widget layout as `render_shapes_section`
    /// (minus the surrounding ScrollArea + Frame::popup, which don't
    /// affect intra-row alignment) and capture each card's response
    /// rect plus the trailing `+` button's rect.
    fn capture_row_rects(row: &[ShapeEntry]) -> Vec<egui::Rect> {
        let ctx = egui::Context::default();
        let mut rects = Vec::new();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                // Top-align cross-axis: with `Align::Min` egui places
                // each widget at the row's top edge, skipping the
                // `frame_size.y = max(child, avail)` recursion that
                // Center alignment uses (and that recursion is what
                // produced the converging staircase tops 14 -> 18.5
                // -> 20.75 -> 21.88; each card pulled halfway toward
                // the avail.center as `avail` grew with placed
                // widgets).
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Min), |ui| {
                    for (i, entry) in row.iter().enumerate() {
                        let drag_id = ui.make_persistent_id(("shape-card", i));
                        let frame = egui::Frame::default()
                            .fill(egui::Color32::from_rgb(80, 80, 80))
                            .stroke(egui::Stroke::new(1.0, egui::Color32::GRAY))
                            .inner_margin(egui::Margin::symmetric(4, 6))
                            .corner_radius(egui::CornerRadius::same(3));
                        let (inner_resp, _) = ui.dnd_drop_zone::<usize, _>(frame, |ui| {
                            let _ = ui.dnd_drag_source(drag_id, i, |ui| {
                                ui.allocate_ui_with_layout(
                                    egui::vec2(SHAPE_CARD_WIDTH, 0.0),
                                    egui::Layout::top_down(egui::Align::Center),
                                    |ui| {
                                        ui.add(
                                            egui::Label::new(
                                                egui::RichText::new(entry.label)
                                                    .strong()
                                                    .color(egui::Color32::WHITE),
                                            )
                                            .selectable(false)
                                            .wrap_mode(egui::TextWrapMode::Extend),
                                        );
                                    },
                                );
                            });
                        });
                        rects.push(inner_resp.response.rect);
                    }
                    let plus = add_button(ui, egui::vec2(CONTROL_W, CONTROL_H - 2.0));
                    rects.push(plus.rect);
                });
            });
        });
        rects
    }

    fn rect_table(rects: &[egui::Rect]) -> String {
        rects
            .iter()
            .enumerate()
            .map(|(i, r)| {
                format!(
                    "[{i}] top={:.2} bottom={:.2} center.y={:.2} h={:.2}",
                    r.top(),
                    r.bottom(),
                    r.center().y,
                    r.height()
                )
            })
            .collect::<Vec<_>>()
            .join("\n        ")
    }

    /// All widgets must share a top y. With Top-align cross-axis,
    /// this is the meaningful invariant; heights may vary (the +
    /// button is intentionally 2pt shorter than the cards) but
    /// tops align.
    fn assert_top_aligned(rects: &[egui::Rect], context: &str) {
        if rects.is_empty() {
            return;
        }
        let first_top = rects[0].top();
        for (i, rect) in rects.iter().enumerate() {
            let top = rect.top();
            assert!(
                (top - first_top).abs() < 0.5,
                "{context}: widget {i} top={top:.2} differs from widget 0's \
                 top={first_top:.2}\n        {table}",
                table = rect_table(rects),
            );
        }
    }

    /// Cards (everything except the trailing + button) must have
    /// uniform height. The + is excluded because it's intentionally
    /// 2pt shorter for visual balance.
    fn assert_cards_h_uniform(rects: &[egui::Rect], context: &str) {
        if rects.len() < 2 {
            return;
        }
        let cards = &rects[..rects.len() - 1];
        let first_h = cards[0].height();
        for (i, rect) in cards.iter().enumerate() {
            let h = rect.height();
            assert!(
                (h - first_h).abs() < 0.5,
                "{context}: card {i} height={h:.2} differs from card 0's \
                 height={first_h:.2}\n        {table}",
                table = rect_table(rects),
            );
        }
    }

    #[test]
    fn default_row_4_shapes_aligned() {
        let row = DEFAULT_ROW.to_vec();
        let rects = capture_row_rects(&row);
        assert_cards_h_uniform(&rects, "default 4-shape row");
        assert_top_aligned(&rects, "default 4-shape row");
    }

    #[test]
    fn row_with_120cell_aligned() {
        let mut row = DEFAULT_ROW.to_vec();
        row.push(parse_shape_name("120-cell").unwrap());
        let rects = capture_row_rects(&row);
        assert_cards_h_uniform(&rects, "default + 120-cell");
        assert_top_aligned(&rects, "default + 120-cell");
    }

    #[test]
    fn row_with_120cell_and_600cell_aligned() {
        let mut row = DEFAULT_ROW.to_vec();
        row.push(parse_shape_name("120-cell").unwrap());
        row.push(parse_shape_name("600-cell").unwrap());
        let rects = capture_row_rects(&row);
        assert_cards_h_uniform(&rects, "default + 120-cell + 600-cell");
        assert_top_aligned(&rects, "default + 120-cell + 600-cell");
    }
}

/// Drag-and-drop regression tests for `dnd_drag_source_collapsing`.
/// The headless `egui::Context::run` driver lets us simulate a
/// pointer press + drag-past-threshold and assert the helper's
/// drag detection still wakes up. Two prior regressions this guards
/// against:
///   1. Switching the drag id from `ui.make_persistent_id` to
///      `egui::Id::new` accidentally broke detection (this exists
///      to verify the helper works with both kinds of id).
///   2. Wrapping the body in a `Frame` (so the whole card follows
///      the cursor) must not eat the drag's hit-test rect; the
///      drag rect is the body's rect, which equals the Frame's
///      outer rect after `Frame::show`.
#[cfg(test)]
mod drag_tests {
    use super::*;

    fn screen() -> egui::Rect {
        egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 600.0))
    }

    /// Egui's drag detection uses `time - press_start_time` against
    /// `Options::max_click_duration`. Without advancing `time`
    /// between frames, every press is "still within click window"
    /// and `is_decidedly_dragging` returns false, even with
    /// movement. We thread a monotonic clock so each frame's input
    /// has `time = N * 50ms`; well past the default click duration.
    fn pointer_press(time: f64, pos: egui::Pos2) -> egui::RawInput {
        let mut input = egui::RawInput::default();
        input.screen_rect = Some(screen());
        input.time = Some(time);
        input.events.push(egui::Event::PointerMoved(pos));
        input.events.push(egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: Default::default(),
        });
        input
    }

    fn pointer_move(time: f64, pos: egui::Pos2) -> egui::RawInput {
        let mut input = egui::RawInput::default();
        input.screen_rect = Some(screen());
        input.time = Some(time);
        input.events.push(egui::Event::PointerMoved(pos));
        input
    }

    fn warmup_input(time: f64) -> egui::RawInput {
        let mut input = egui::RawInput::default();
        input.screen_rect = Some(screen());
        input.time = Some(time);
        input
    }

    /// Simulate "click on card, then drag past the drag threshold"
    /// against `dnd_drag_source_collapsing` and assert that
    /// `ctx.is_being_dragged(id)` becomes true. Press alone is not
    /// enough; egui requires movement past `start_drag_threshold`
    /// (~6 px) before flipping the drag flag.
    fn drive_drag(id: egui::Id) -> egui::Context {
        let ctx = egui::Context::default();
        let card_pos = egui::pos2(60.0, 30.0);
        let render = |ctx: &egui::Context| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let _ = dnd_drag_source_collapsing(ui, id, 42_usize, |ui| {
                    egui::Frame::default()
                        .fill(egui::Color32::DARK_GRAY)
                        .inner_margin(egui::Margin::symmetric(4, 6))
                        .show(ui, |ui| {
                            ui.allocate_exact_size(egui::vec2(80.0, 18.0), egui::Sense::hover());
                        });
                });
            });
        };
        let _ = ctx.run(warmup_input(0.0), render);
        let _ = ctx.run(pointer_press(0.05, card_pos), render);
        let _ = ctx.run(pointer_move(0.10, card_pos + egui::vec2(20.0, 0.0)), render);
        let _ = ctx.run(pointer_move(0.15, card_pos + egui::vec2(40.0, 0.0)), render);
        ctx
    }

    /// Baseline: stock `Ui::dnd_drag_source` must start a drag with
    /// our test driver. If THIS fails, the test driver is wrong (not
    /// the helper); the helper-specific tests below are then
    /// meaningless until the driver is fixed.
    #[test]
    fn baseline_stock_dnd_drag_source_starts_drag() {
        let ctx = egui::Context::default();
        let id = egui::Id::new("baseline-test");
        let mut last_rect = egui::Rect::NOTHING;
        let render = |ctx: &egui::Context, last_rect: &mut egui::Rect| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let resp = ui.dnd_drag_source(id, 1_usize, |ui| {
                    egui::Frame::default()
                        .fill(egui::Color32::DARK_GRAY)
                        .inner_margin(egui::Margin::symmetric(4, 6))
                        .show(ui, |ui| {
                            ui.allocate_exact_size(egui::vec2(80.0, 18.0), egui::Sense::hover());
                        });
                });
                *last_rect = resp.response.rect;
            });
        };
        let card_pos = egui::pos2(60.0, 30.0);
        let _ = ctx.run(warmup_input(0.0), |c| render(c, &mut last_rect));
        let _ = ctx.run(pointer_press(0.05, card_pos), |c| render(c, &mut last_rect));
        let _ = ctx.run(pointer_move(0.10, card_pos + egui::vec2(20.0, 0.0)), |c| {
            render(c, &mut last_rect)
        });
        let _ = ctx.run(pointer_move(0.15, card_pos + egui::vec2(40.0, 0.0)), |c| {
            render(c, &mut last_rect)
        });
        assert!(
            ctx.is_being_dragged(id),
            "stock dnd_drag_source should detect drag with this driver"
        );
    }

    /// `egui::Id::new(...)` keys must drive `dnd_drag_source_collapsing`
    /// just as well as `ui.make_persistent_id`. The regression that
    /// motivated this test: shape and term cards stopped responding
    /// to drags after a refactor that switched their drag ids to
    /// `Id::new` for stable per-row-index keys.
    #[test]
    fn id_new_starts_drag() {
        let id = egui::Id::new(("rotate-polytopes-shape-card-test", 0_usize));
        let ctx = drive_drag(id);
        assert!(
            ctx.is_being_dragged(id),
            "drag should be active after press + move past threshold; \
             dnd_drag_source_collapsing failed to wire up the drag rect"
        );
        assert!(
            egui::DragAndDrop::has_payload_of_type::<usize>(&ctx),
            "drag payload should be set after drag starts"
        );
    }

    /// Regression test for the bug the user hit: drag-source ids
    /// keyed by `egui::Id::new(...)` (i.e., NOT scoped to the
    /// rendering ui) collide across `BottomOverlay`'s two passes
    /// and silently break drag detection in release / panic the
    /// `debug_assert!` in debug. The production fix is to derive
    /// the drag id from the per-pass ui scope via
    /// `ui.make_persistent_id(...)` so the two passes see
    /// distinct ids.
    ///
    /// We can't directly test drag detection inside Areas in
    /// headless `Context::run` (Area-routed input doesn't seem to
    /// reach the interaction step the same way it does in a real
    /// winit-driven loop). Instead we verify that:
    /// 1. Rendering the same source closure in two `Area`s with
    ///    different layers does NOT trigger the debug-assert when
    ///    ids are scoped per-ui (`make_persistent_id`).
    /// 2. The IDs actually ARE distinct between the two passes.
    /// The first part; running this test without panic in debug
    ///; is what catches a regression to globally-stable ids.
    #[test]
    fn make_persistent_id_per_pass_avoids_layer_collision() {
        let ctx = egui::Context::default();
        let mut measure_id: Option<egui::Id> = None;
        let mut visible_id: Option<egui::Id> = None;
        let render = |ctx: &egui::Context,
                      measure_id: &mut Option<egui::Id>,
                      visible_id: &mut Option<egui::Id>| {
            let _ = egui::Area::new(egui::Id::new("measure"))
                .order(egui::Order::Background)
                .interactable(false)
                .fixed_pos(egui::pos2(-99_999.0, -99_999.0))
                .show(ctx, |ui| {
                    ui.set_invisible();
                    let id = ui.make_persistent_id("test-card");
                    *measure_id = Some(id);
                    let _ = ui.dnd_drag_source(id, 7_usize, |ui| {
                        ui.allocate_exact_size(egui::vec2(80.0, 18.0), egui::Sense::hover());
                    });
                });
            let _ = egui::Area::new(egui::Id::new("visible"))
                .fixed_pos(egui::pos2(0.0, 0.0))
                .movable(false)
                .show(ctx, |ui| {
                    let id = ui.make_persistent_id("test-card");
                    *visible_id = Some(id);
                    let _ = ui.dnd_drag_source(id, 7_usize, |ui| {
                        ui.allocate_exact_size(egui::vec2(80.0, 18.0), egui::Sense::hover());
                    });
                });
        };
        // Render without panicking. If a future change reverts to
        // `egui::Id::new(...)` for the drag id, both passes resolve
        // to the same id, the same id ends up in two layers, and
        // egui's `debug_assert!` panics here.
        let _ = ctx.run(warmup_input(0.0), |c| {
            render(c, &mut measure_id, &mut visible_id)
        });
        let _ = ctx.run(warmup_input(0.05), |c| {
            render(c, &mut measure_id, &mut visible_id)
        });
        let measure_id = measure_id.expect("measure ran");
        let visible_id = visible_id.expect("visible ran");
        assert_ne!(
            measure_id, visible_id,
            "ui.make_persistent_id resolves through per-ui scope, so the same \
             source must produce different ids in measure vs visible passes; \
             if these ids ever match, the next regression is the debug_assert \
             in egui's WidgetRects::insert"
        );
    }

    /// Regression test for the "card snaps to the right for a frame"
    /// bug: the make-room gap's `open_width` must match the rendered
    /// card slot's outer width (the Frame's outer rect, not the
    /// inner content), otherwise dropping a card causes a one-frame
    /// horizontal layout shift as the gap closes and the card
    /// occupies a slightly-different-sized slot.
    #[test]
    fn shape_gap_open_width_matches_card_slot_width() {
        let ctx = egui::Context::default();
        let mut card_outer_w = 0.0_f32;
        let _ = ctx.run(warmup_input(0.0), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let resp = egui::Frame::default()
                    .fill(egui::Color32::DARK_GRAY)
                    .inner_margin(egui::Margin::symmetric(4, 6))
                    .corner_radius(egui::CornerRadius::same(3))
                    .show(ui, |ui| {
                        ui.allocate_ui_with_layout(
                            egui::vec2(SHAPE_CARD_WIDTH, 0.0),
                            egui::Layout::top_down(egui::Align::Center),
                            |ui| {
                                ui.add(
                                    egui::Label::new(egui::RichText::new("test").strong())
                                        .selectable(false)
                                        .wrap_mode(egui::TextWrapMode::Extend),
                                );
                            },
                        );
                    });
                card_outer_w = resp.response.rect.width();
            });
        });
        let gap_open_width = SHAPE_CARD_WIDTH + 8.0;
        let drift = (gap_open_width - card_outer_w).abs();
        assert!(
            drift < 1.0,
            "make-room gap open width ({gap_open_width:.1}) must match the \
             rendered shape card outer width ({card_outer_w:.1}); a mismatch \
             produces a one-frame horizontal rubberband when the gap closes \
             and the card takes its slot. drift = {drift:.1} pt"
        );
    }

    /// Simulates dragging a card from one slot to another and
    /// verifies that the row's total width is INVARIANT through
    /// the drag -> drop transition. If the dragged card takes some
    /// space during drag and a different amount after drop, OR if
    /// the make-room gap's width doesn't match the dropped card's
    /// slot width, the OTHER cards shift horizontally on drop.
    /// That's the rubberband the user sees.
    ///
    /// Render N "cards" with stable widths via the same helper
    /// (`dnd_drag_source_collapsing` + `make_room_gap`) the live
    /// shape row uses, simulate a press + drag-past-threshold +
    /// hover-over-target + release, and capture neighbouring card
    /// positions on the last drag frame and on the post-drop
    /// frame.
    #[test]
    fn shape_row_total_width_invariant_through_drop() {
        const N: usize = 4;
        const CARD_W: f32 = SHAPE_CARD_WIDTH + 8.0;
        const SPACING: f32 = 4.0;
        let ctx = egui::Context::default();
        // We measure widths under a "card 0 is being dragged"
        // scenario (drop at trailing slot N) and compare with the
        // post-drop scenario (no drag in flight, all cards rendered
        // normally).
        let target_slot = N;

        let mut total_during_drag = 0.0_f32;
        let render_during_drag = |ctx: &egui::Context, total: &mut f32| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Min), |ui| {
                    ui.spacing_mut().item_spacing.x = SPACING;
                    let drop_idx = Some(target_slot);
                    for i in 0..N {
                        // Make-room gap before card i.
                        let gap_id = ui.make_persistent_id(("gap", i));
                        let _ = make_room_gap(ui, drop_idx == Some(i), gap_id, 18.0, CARD_W);
                        let card_id = ui.make_persistent_id(("card", i));
                        let _ = dnd_drag_source_collapsing(ui, card_id, i, |ui| {
                            egui::Frame::default()
                                .inner_margin(egui::Margin::symmetric(4, 6))
                                .show(ui, |ui| {
                                    ui.allocate_exact_size(
                                        egui::vec2(SHAPE_CARD_WIDTH, 0.0),
                                        egui::Sense::hover(),
                                    );
                                });
                        });
                    }
                    // Trailing gap.
                    let trail_id = ui.make_persistent_id(("gap", N));
                    let _ = make_room_gap(ui, drop_idx == Some(N), trail_id, 18.0, CARD_W);
                    *total = ui.min_rect().width();
                });
            });
        };

        // Drive a real drag on card `dragged_idx`. Card centers
        // are predictable: card 0 center = CARD_W/2 = 36.
        let card0_center = egui::pos2(CARD_W / 2.0, 9.0);
        let _ = ctx.run(warmup_input(0.0), |c| {
            render_during_drag(c, &mut total_during_drag)
        });
        let _ = ctx.run(pointer_press(0.05, card0_center), |c| {
            render_during_drag(c, &mut total_during_drag)
        });
        // Move past drag threshold AND past the row to land at
        // the trailing slot. card0_center is at x=36, drag to x=400.
        let target_pos = egui::pos2(400.0, 9.0);
        let _ = ctx.run(pointer_move(0.10, target_pos), |c| {
            render_during_drag(c, &mut total_during_drag)
        });
        // Several frames at the same target so the gap can settle
        // open at full width.
        for k in 0..15 {
            let t = 0.15 + (k as f64) * 0.02;
            let _ = ctx.run(pointer_move(t, target_pos), |c| {
                render_during_drag(c, &mut total_during_drag)
            });
        }
        let drag_total = total_during_drag;
        let dragged_id = ctx.dragged_id();
        // Release.
        let mut release_input = egui::RawInput::default();
        release_input.screen_rect = Some(screen());
        release_input.time = Some(0.6);
        release_input
            .events
            .push(egui::Event::PointerMoved(target_pos));
        release_input.events.push(egui::Event::PointerButton {
            pos: target_pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: Default::default(),
        });
        let _ = ctx.run(release_input, |c| {
            render_during_drag(c, &mut total_during_drag)
        });
        // Frame after release: drag is over, no make-room gap, no
        // dragged card collapse. Re-render to measure post-drop.
        let _ = ctx.run(warmup_input(0.65), |c| {
            render_during_drag(c, &mut total_during_drag)
        });
        let post_drop_total = total_during_drag;

        eprintln!(
            "drag_total = {drag_total:.1}, post_drop_total = {post_drop_total:.1}, \
             dragged_id = {dragged_id:?}"
        );
        let drift = (drag_total - post_drop_total).abs();
        assert!(
            drift < 1.0,
            "row total width must stay constant from drag -> drop, otherwise \
             cards rubberband horizontally on release. drag={drag_total:.1}, \
             post_drop={post_drop_total:.1}, drift={drift:.1}"
        );
    }

    /// Same regression check applied to `dnd_drag_source_collapsing`:
    /// the helper must round-trip through a content closure that
    /// runs in two egui layers without producing a same-id-in-two-
    /// layers panic.
    #[test]
    fn collapsing_helper_in_two_pass_no_layer_collision() {
        let ctx = egui::Context::default();
        let render = |ctx: &egui::Context| {
            let _ = egui::Area::new(egui::Id::new("measure"))
                .order(egui::Order::Background)
                .interactable(false)
                .fixed_pos(egui::pos2(-99_999.0, -99_999.0))
                .show(ctx, |ui| {
                    ui.set_invisible();
                    let id = ui.make_persistent_id("test-card");
                    let _ = dnd_drag_source_collapsing(ui, id, 7_usize, |ui| {
                        ui.allocate_exact_size(egui::vec2(80.0, 18.0), egui::Sense::hover());
                    });
                });
            let _ = egui::Area::new(egui::Id::new("visible"))
                .fixed_pos(egui::pos2(0.0, 0.0))
                .movable(false)
                .show(ctx, |ui| {
                    let id = ui.make_persistent_id("test-card");
                    let _ = dnd_drag_source_collapsing(ui, id, 7_usize, |ui| {
                        ui.allocate_exact_size(egui::vec2(80.0, 18.0), egui::Sense::hover());
                    });
                });
        };
        let _ = ctx.run(warmup_input(0.0), render);
        let _ = ctx.run(warmup_input(0.05), render);
    }

    /// `ui.make_persistent_id(...)` keys must also work; protect
    /// against a future regression that hard-codes one id flavour.
    #[test]
    fn make_persistent_id_starts_drag() {
        let ctx = egui::Context::default();
        let render = |ctx: &egui::Context, captured_id: &mut Option<egui::Id>| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let id = ui.make_persistent_id(("test-card", 0_usize));
                *captured_id = Some(id);
                let _ = dnd_drag_source_collapsing(ui, id, 99_usize, |ui| {
                    egui::Frame::default()
                        .fill(egui::Color32::DARK_GRAY)
                        .inner_margin(egui::Margin::symmetric(4, 6))
                        .show(ui, |ui| {
                            ui.allocate_exact_size(egui::vec2(80.0, 18.0), egui::Sense::hover());
                        });
                });
            });
        };
        let card_pos = egui::pos2(60.0, 30.0);
        let mut id = None;
        let _ = ctx.run(warmup_input(0.0), |ctx| render(ctx, &mut id));
        let _ = ctx.run(pointer_press(0.05, card_pos), |ctx| render(ctx, &mut id));
        let _ = ctx.run(
            pointer_move(0.10, card_pos + egui::vec2(20.0, 0.0)),
            |ctx| render(ctx, &mut id),
        );
        let _ = ctx.run(
            pointer_move(0.15, card_pos + egui::vec2(40.0, 0.0)),
            |ctx| render(ctx, &mut id),
        );
        let id = id.expect("captured id");
        assert!(
            ctx.is_being_dragged(id),
            "drag should be active for make_persistent_id keys too"
        );
    }
}
