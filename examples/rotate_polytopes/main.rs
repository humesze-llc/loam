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
//! - **T**: toggle 4D rotation (pause/resume freezes orientation
//!   in place, does NOT snap back to identity).
//! - **1..6**: toggle the corresponding rotation plane on/off.
//!   The mapping is `1=xy, 2=xz, 3=xw, 4=yz, 5=yw, 6=zw`. Active
//!   planes' bivectors sum into the angular velocity. Famous
//!   compositions: `3` alone = single xw stretch; `3+4` =
//!   isoclinic xw+yz; `3+5+6` = three w-planes drift through
//!   SO(4). Pure-3D combinations (`1+2+4`) just rotate the
//!   cross-section as a rigid 3D shape.
//! - **+ / -**: adjust the global rotation rate.
//! - **R**: full reset, slice, rate, all toggles off, AND
//!   orientation back to canonical pose.
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
use rye_app::{
    egui, run_with_config, App, BottomOverlay, Camera, FrameCtx, LinearIndicator, OrbitController,
    RotorVisualizer, RunConfig, SetupCtx,
};
use rye_math::{Bivector, Bivector4, EuclideanR3, Plane4, Rotor4};
use rye_render::{
    device::RenderDevice,
    raymarch::{
        polytope_extended_sdfs_wgsl, BodyUniform, Hyperslice4DNode, HYPERSLICE_KERNEL_WGSL,
        SHAPE_120CELL, SHAPE_16CELL, SHAPE_24CELL, SHAPE_600CELL, SHAPE_PENTATOPE, SHAPE_TESSERACT,
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

const W_SCRUB_RATE: f32 = 0.5;
const W_RANGE: f32 = 1.5;

/// Base rotation angular rate (rad/s). Scaled by `rate_scale` per
/// frame so +/- can speed it up or slow it down.
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
/// the `pentatope` / `tesseract` / `hexadecachoron` family; the
/// `*-plex` aliases (pentaplex, dodecaplex, ...) are deliberately
/// avoided since "plex" is dimension-generalized rather than
/// being the actual 4D name.
#[derive(Copy, Clone)]
struct ShapeEntry {
    shape: u32,
    body_color: [f32; 3],
    label: &'static str,
    long_name: &'static str,
}

/// Default row when no `--shapes` argument is given. Ordered to put
/// the 24-cell first (most "4D-distinct" cross-section), then the
/// pentatope / 16-cell / tesseract triple; visually contrasting
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
        long_name: "pentatope",
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

/// Catalog of named shapes. Both common math-name aliases (the
/// `n-cell` form) and Platonic-slice aliases (the `tetrahedron` /
/// `cube` / etc. form) resolve to the same shape index.
fn parse_shape_name(name: &str) -> Result<ShapeEntry> {
    let n = name.to_lowercase();
    Ok(match n.as_str() {
        "5-cell" | "5cell" | "pentatope" | "pentachoron" | "tetrahedron" => ShapeEntry {
            shape: SHAPE_PENTATOPE,
            body_color: [0.95, 0.55, 0.30],
            label: "5-cell",
            long_name: "pentatope",
        },
        "8-cell" | "8cell" | "tesseract" | "hypercube" | "cube" => ShapeEntry {
            shape: SHAPE_TESSERACT,
            body_color: [0.30, 0.55, 0.95],
            label: "8-cell",
            long_name: "tesseract",
        },
        "16-cell" | "16cell" | "hexadecachoron" | "octahedron" => ShapeEntry {
            shape: SHAPE_16CELL,
            body_color: [0.55, 0.95, 0.40],
            label: "16-cell",
            long_name: "hexadecachoron",
        },
        "24-cell" | "24cell" | "icositetrachoron" | "cuboctahedron" => ShapeEntry {
            shape: SHAPE_24CELL,
            body_color: [0.95, 0.45, 0.85],
            label: "24-cell",
            long_name: "icositetrachoron",
        },
        "120-cell" | "120cell" | "hecatonicosachoron" | "dodecahedron" => ShapeEntry {
            shape: SHAPE_120CELL,
            body_color: [0.40, 0.85, 0.85],
            label: "120-cell",
            long_name: "hecatonicosachoron",
        },
        "600-cell" | "600cell" | "hexacosichoron" | "icosahedron" => ShapeEntry {
            shape: SHAPE_600CELL,
            body_color: [0.95, 0.85, 0.40],
            label: "600-cell",
            long_name: "hexacosichoron",
        },
        _ => {
            return Err(anyhow!(
                "unknown shape name {name:?}; valid names: 5-cell, \
                 tesseract, 16-cell, 24-cell, 120-cell, 600-cell \
                 (or Platonic aliases: tetrahedron, cube, octahedron, \
                 cuboctahedron, dodecahedron, icosahedron)"
            ))
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

/// Angular velocity from the active set: sum of unit bivectors of
/// active planes, scaled by base rate × rate_scale.
fn angular_velocity(active: &[bool; 6], rate_scale: f32) -> Bivector4 {
    let mut omega = Bivector4::ZERO;
    for (i, &on) in active.iter().enumerate() {
        if on {
            omega = omega + Plane4::ALL[i].unit_bivector();
        }
    }
    omega * (BASE_ROTATION_RATE * rate_scale)
}

/// Angular velocity from a composed seq: sum over terms of
/// `scalar * sum_of_unit_bivectors_in_term`, scaled by rate_scale.
/// Bivector addition is commutative, so term order is irrelevant
/// in this continuous mode (it matters for the multiplicative
/// `Apply` action, but that's a separate one-shot path).
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
// Font discovery (portable system-font fallback)
// ---------------------------------------------------------------------------

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

    /// Whether the top-right rotation-formula popup is rendered.
    /// Off by default; the formula is dense for newcomers; the
    /// expanded section has a checkbox to turn it on for users who
    /// want to see exactly which bivectors and scalars compose into
    /// the current orientation.
    show_formula: bool,

    /// Filmstrip mode: render `strip_count` thumbnails across the
    /// scene area, each at an evenly-spaced `w_slice` value across
    /// `[-W_RANGE, W_RANGE]`. Lets the user see the full 4D extent
    /// of every polytope at once instead of scrubbing one slice.
    strip_view: bool,
    /// Number of cells in the multi-slice strip. Range 3..=21.
    strip_count: usize,
    /// Show the read-only `rye_egui::LinearIndicator` for the
    /// current `w_slice` plane. Top-left fixed area when on.
    show_w_indicator: bool,
    /// Show the `rye_egui::RotorVisualizer` for the current
    /// angular-velocity bivector. Surfaces the SO(4) double-
    /// rotation structure as one or two labeled arcs.
    show_rotor_viz: bool,

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
    /// whole sum before `exp()`. Default `Some(FRAC_PI_2)` when
    /// the user adds a scalar via the panel.
    scalar: Option<f32>,
}

impl RotorTerm {
    /// Compose this term as a delta rotor.
    fn delta(&self) -> Rotor4 {
        let mut sum = Bivector4::ZERO;
        for plane in &self.planes {
            sum = sum + plane.unit_bivector();
        }
        let phi = self.scalar.unwrap_or(1.0);
        (sum * phi).exp()
    }
}

/// Continuous-rotation source. Two distinct UIs (active-set
/// checkboxes vs composed sequence) populate the angular velocity
/// independently; the user picks which one drives `omega` for the
/// spin animation via a tab at the top of the rotation section.
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

/// Drop-in replacement for `Ui::dnd_drag_source` that takes the
/// dragged item out of the parent layout entirely while the drag
/// is in flight. The body paints into a Tooltip layer that follows
/// the cursor (egui's standard drag preview); the parent layout
/// allocates **zero space** for the dragged item; neighbouring
/// widgets fill its old slot instantly, and the make-room gap at
/// the drop target is the slot the item will eventually occupy on
/// drop.
///
/// This matches `egui_dnd`'s shape and keeps the row's total width
/// constant from drag through drop, so dropping doesn't trigger a
/// horizontal layout shift on the frame the item slot is replaced
/// with the actual card.
///
/// `egui::Ui::dnd_drag_source`'s dragged path uses `scope_builder`,
/// which advances the parent cursor by the body's natural width.
/// That's why egui's stock helper leaves the original slot
/// allocated. We bypass `scope_builder` via `Ui::new_child`, which
/// does NOT advance the parent cursor.
fn dnd_drag_source_collapsing<P>(
    ui: &mut egui::Ui,
    id: egui::Id,
    payload: P,
    body: impl FnOnce(&mut egui::Ui),
) -> egui::Response
where
    P: 'static + Send + Sync,
{
    let ctx = ui.ctx().clone();
    let is_dragged = ctx.is_being_dragged(id);
    if !is_dragged {
        return ui.dnd_drag_source(id, payload, body).response;
    }
    egui::DragAndDrop::set_payload(&ctx, payload);
    let layer_id = egui::LayerId::new(egui::Order::Tooltip, id);
    let mut child = ui.new_child(egui::UiBuilder::new().layer_id(layer_id));
    body(&mut child);
    let body_rect = child.min_rect();
    if let Some(pos) = ctx.pointer_interact_pos() {
        let delta = pos - body_rect.center();
        ctx.transform_layer_shapes(layer_id, egui::emath::TSTransform::from_translation(delta));
    }
    // Register a hit-rect at the body's natural position so callers
    // (context menus, hover text) still get a usable response, but
    // do NOT allocate space in the parent layout. This is the
    // whole point: the dragged card occupies zero width in the
    // row while the drag is in flight.
    ui.interact(body_rect, id, egui::Sense::hover())
}

/// Animated "make room" insertion gap at one slot of a horizontal
/// row. The slot whose `is_target` is `true` expands to `open_width`
/// over ~120 ms; others stay at zero width. Cards on either side
/// slide outward as the gap opens, giving a clear drop preview
/// without a separate marker line. The gap collapses back to zero
/// when the drag ends. Returns `true` if a pointer release occurred
/// on the targeted gap this frame; the caller takes whatever
/// payload it expects from `DragAndDrop` and applies the move.
fn make_room_gap(
    ui: &mut egui::Ui,
    is_target: bool,
    slot_id: egui::Id,
    height: f32,
    open_width: f32,
) -> bool {
    let target_w = if is_target { open_width } else { 0.0 };
    let smooth_w = ui.ctx().animate_value_with_time(slot_id, target_w, 0.12);
    if smooth_w >= 0.5 {
        let _ = ui.allocate_exact_size(egui::vec2(smooth_w, height), egui::Sense::hover());
    }
    let dropped = is_target && ui.ctx().input(|i| i.pointer.any_released());
    if dropped {
        // Snap the gap closed instantly on drop. Without this, the
        // gap animates from `open_width` -> 0 over the next ~120 ms
        // while the row's right side rubberbands leftward as the
        // gap closes; a visible "settle" the user reads as jank.
        let _ = ui.ctx().animate_value_with_time(slot_id, 0.0, 0.0);
    }
    dropped
}

/// Map cursor x-position over a row's bounding `row_rect` to a
/// 0-based insertion slot index in `0..=item_count`. Returns
/// `None` when no drag is active (`is_dragging` is `false`) or
/// the cursor isn't over the row band. Hit band extends ±40 pt
/// vertically so a card dragged a bit above or below the row
/// still snaps to a slot.
fn drop_target_idx(
    ctx: &egui::Context,
    is_dragging: bool,
    row_rect: egui::Rect,
    item_count: usize,
) -> Option<usize> {
    if !is_dragging {
        return None;
    }
    let cursor = ctx.input(|i| i.pointer.hover_pos())?;
    let band = row_rect.expand2(egui::vec2(0.0, 40.0));
    if !band.x_range().contains(cursor.x) || !band.y_range().contains(cursor.y) {
        return None;
    }
    let n_slots = item_count + 1;
    let slot_w = (row_rect.width() / n_slots as f32).max(1.0);
    let rel = (cursor.x - row_rect.left()).max(0.0);
    Some(((rel / slot_w) as usize).min(item_count))
}

/// Inside a `dnd_drag_source` body, force fully-opaque widget
/// visuals on the current ui when the source is being dragged.
/// egui paints the body to a Tooltip layer when dragged, where
/// widgets never register hover and therefore default to the
/// dimmed `inactive` style; this override lifts inactive and
/// noninteractive fills/strokes to match `active` so the floating
/// ghost reads as a solid card.
fn force_opaque_active(ui: &mut egui::Ui) {
    let active = ui.visuals().widgets.active;
    let v = ui.visuals_mut();
    v.widgets.inactive.bg_fill = active.bg_fill;
    v.widgets.inactive.weak_bg_fill = active.weak_bg_fill;
    v.widgets.inactive.fg_stroke = active.fg_stroke;
    v.widgets.inactive.bg_stroke = active.bg_stroke;
    v.widgets.noninteractive.bg_fill = active.bg_fill;
    v.widgets.noninteractive.weak_bg_fill = active.weak_bg_fill;
}

/// "Pickup" pulse intensity in `[0.0, 1.0]` for the card identified
/// by `drag_id`. Animates from 0 to 1 in 120 ms when the source
/// starts being dragged, and back to 0 over the same time when the
/// drag ends. Use it to interpolate stroke width / color / scale on
/// the dragged frame so the card visibly "lifts" on pickup.
fn drag_pickup_t(ctx: &egui::Context, drag_id: egui::Id) -> f32 {
    let target = if ctx.is_being_dragged(drag_id) {
        1.0
    } else {
        0.0
    };
    ctx.animate_value_with_time(drag_id.with("pickup"), target, 0.12)
}

/// Rate "skip" button drawn as one or two solid triangles.
/// Matches the play/pause button's media-player vocabulary so the
/// whole row reads as a single set of media controls. Highlights
/// when `*rate == value`; clicking when already selected resets
/// `rate = 1.0` (lets the user step out of a non-default rate
/// without the global Reset).
///
/// `double = true` paints two adjacent triangles (`<<` / `>>`),
/// `false` paints one (`<` / `>`). `forward = true` points right.
fn rate_toggle(
    ui: &mut egui::Ui,
    rate: &mut f32,
    value: f32,
    double: bool,
    forward: bool,
) -> egui::Response {
    let selected = (*rate - value).abs() < 1e-6;
    let size = egui::vec2(28.0, CONTROL_H);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    let style = ui.style().interact_selectable(&response, selected);
    ui.painter().rect(
        rect,
        egui::CornerRadius::same(2),
        style.bg_fill,
        style.bg_stroke,
        egui::StrokeKind::Inside,
    );
    let color = style.fg_stroke.color;
    let cx = rect.center().x;
    let cy = rect.center().y;
    let r_w = 4.5_f32;
    let r_h = 5.5_f32;
    let triangle_at = |tip_x: f32| -> Vec<egui::Pos2> {
        if forward {
            vec![
                egui::pos2(tip_x - r_w * 0.5, cy - r_h),
                egui::pos2(tip_x - r_w * 0.5, cy + r_h),
                egui::pos2(tip_x + r_w * 0.7, cy),
            ]
        } else {
            vec![
                egui::pos2(tip_x + r_w * 0.5, cy - r_h),
                egui::pos2(tip_x + r_w * 0.5, cy + r_h),
                egui::pos2(tip_x - r_w * 0.7, cy),
            ]
        }
    };
    if double {
        let offset = 4.0;
        ui.painter().add(egui::Shape::convex_polygon(
            triangle_at(cx - offset),
            color,
            egui::Stroke::NONE,
        ));
        ui.painter().add(egui::Shape::convex_polygon(
            triangle_at(cx + offset),
            color,
            egui::Stroke::NONE,
        ));
    } else {
        ui.painter().add(egui::Shape::convex_polygon(
            triangle_at(cx),
            color,
            egui::Stroke::NONE,
        ));
    }
    if response.clicked() {
        *rate = if selected { 1.0 } else { value };
    }
    response.on_hover_text(format!("Set rate to ×{value} (click again to reset to ×1)"))
}

/// `+` button painted as two crossed bars on a button-styled rect.
/// Same primitive-shape vocabulary as the play / rate / chevron
/// buttons so the row reads as one consistent set of custom-painted
/// controls (avoids the `menu_button`'s default-padding height
/// mismatch with the shape cards).
fn add_button(ui: &mut egui::Ui) -> egui::Response {
    // Slightly shorter than `CONTROL_H` because the cards' tinted
    // backgrounds carry more visual weight than this neutral
    // button's outline; equal heights made the + read as
    // visually taller than the cards next to it.
    let size = egui::vec2(28.0, CONTROL_H - 2.0);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    let style = ui.style().interact(&response);
    ui.painter().rect(
        rect,
        egui::CornerRadius::same(2),
        style.bg_fill,
        style.bg_stroke,
        egui::StrokeKind::Inside,
    );
    let cx = rect.center().x;
    let cy = rect.center().y;
    let arm = 5.5_f32;
    let thick = 2.0_f32;
    let color = style.fg_stroke.color;
    ui.painter().rect_filled(
        egui::Rect::from_center_size(egui::pos2(cx, cy), egui::vec2(arm * 2.0, thick)),
        egui::CornerRadius::ZERO,
        color,
    );
    ui.painter().rect_filled(
        egui::Rect::from_center_size(egui::pos2(cx, cy), egui::vec2(thick, arm * 2.0)),
        egui::CornerRadius::ZERO,
        color,
    );
    response
}

/// `R` retry button: a clockwise arc with an arrowhead, painted
/// from primitives. Replaces a font glyph (egui's default font has
/// patchy coverage of the Mathematical Operators block where
/// circular-arrow code points live).
fn refresh_button(ui: &mut egui::Ui) -> egui::Response {
    let size = egui::vec2(28.0, CONTROL_H);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    let style = ui.style().interact(&response);
    ui.painter().rect(
        rect,
        egui::CornerRadius::same(2),
        style.bg_fill,
        style.bg_stroke,
        egui::StrokeKind::Inside,
    );
    let cx = rect.center().x;
    let cy = rect.center().y;
    let radius = 6.5_f32;
    let stroke = egui::Stroke::new(1.6, style.fg_stroke.color);
    use std::f32::consts::PI;
    // Clockwise arc starting just past the top, sweeping ~280°.
    // egui's y-axis points down, so positive angles sweep clockwise
    // visually.
    let start_angle: f32 = -PI / 2.0 + 0.45;
    let sweep: f32 = PI * 1.55;
    let n = 16;
    let points: Vec<egui::Pos2> = (0..=n)
        .map(|i| {
            let t = i as f32 / n as f32;
            let angle = start_angle + t * sweep;
            egui::pos2(cx + radius * angle.cos(), cy + radius * angle.sin())
        })
        .collect();
    ui.painter().add(egui::Shape::line(points, stroke));
    // Arrowhead at the START of the arc (the top-right gap), pointing
    // in the direction the arc would continue if it kept going CW.
    let arrow_size = 3.5_f32;
    let anchor = egui::pos2(
        cx + radius * start_angle.cos(),
        cy + radius * start_angle.sin(),
    );
    let tan = start_angle - PI / 2.0;
    let perp = tan + PI / 2.0;
    let tip = egui::pos2(
        anchor.x + arrow_size * tan.cos(),
        anchor.y + arrow_size * tan.sin(),
    );
    let base_l = egui::pos2(
        anchor.x + arrow_size * 0.8 * perp.cos(),
        anchor.y + arrow_size * 0.8 * perp.sin(),
    );
    let base_r = egui::pos2(
        anchor.x - arrow_size * 0.8 * perp.cos(),
        anchor.y - arrow_size * 0.8 * perp.sin(),
    );
    ui.painter().add(egui::Shape::convex_polygon(
        vec![tip, base_l, base_r],
        style.fg_stroke.color,
        egui::Stroke::NONE,
    ));
    response
}

/// Single button that toggles between a play triangle (when
/// `playing == false`) and a pause symbol (two bars, when
/// `playing == true`). Painted as primitive shapes so it's
/// font-independent and reads as a media-player control on every
/// platform. Returns the response so the caller reads `.clicked()`.
fn play_pause_button(ui: &mut egui::Ui, playing: bool) -> egui::Response {
    let size = egui::vec2(36.0, CONTROL_H);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    let style = ui.style().interact(&response);
    ui.painter().rect(
        rect,
        egui::CornerRadius::same(3),
        style.bg_fill,
        style.bg_stroke,
        egui::StrokeKind::Inside,
    );
    let color = style.fg_stroke.color;
    let cx = rect.center().x;
    let cy = rect.center().y;
    if playing {
        // Pause: two vertical bars.
        let bar_w = 4.0;
        let bar_h = 12.0;
        let gap = 3.0;
        let half_gap = gap / 2.0;
        ui.painter().rect_filled(
            egui::Rect::from_min_size(
                egui::pos2(cx - half_gap - bar_w, cy - bar_h / 2.0),
                egui::vec2(bar_w, bar_h),
            ),
            egui::CornerRadius::ZERO,
            color,
        );
        ui.painter().rect_filled(
            egui::Rect::from_min_size(
                egui::pos2(cx + half_gap, cy - bar_h / 2.0),
                egui::vec2(bar_w, bar_h),
            ),
            egui::CornerRadius::ZERO,
            color,
        );
    } else {
        // Play: filled rightward triangle.
        let r_h = 7.0;
        let r_w = 8.0;
        let p1 = egui::pos2(cx - r_w * 0.4, cy - r_h);
        let p2 = egui::pos2(cx - r_w * 0.4, cy + r_h);
        let p3 = egui::pos2(cx + r_w * 0.7, cy);
        ui.painter().add(egui::Shape::convex_polygon(
            vec![p1, p2, p3],
            color,
            egui::Stroke::NONE,
        ));
    }
    response
}

/// Allocate a clickable button with a custom-painted up- or down-
/// chevron (`^` / `v`, drawn as two stroked line segments). Used
/// instead of a font glyph so it doesn't depend on the egui font
/// having Mathematical Operators (∧/∨) coverage. Returns the
/// response so the caller can read `.clicked()`.
fn chevron_button(ui: &mut egui::Ui, up: bool, hover: &str) -> egui::Response {
    let size = egui::vec2(28.0, CONTROL_H);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    let style = ui.style().interact(&response);
    ui.painter().rect(
        rect,
        egui::CornerRadius::same(2),
        style.bg_fill,
        style.bg_stroke,
        egui::StrokeKind::Inside,
    );
    let cx = rect.center().x;
    let cy = rect.center().y;
    let dx = 6.0;
    let dy = 4.0;
    let stroke = egui::Stroke::new(2.0, style.fg_stroke.color);
    if up {
        ui.painter().line_segment(
            [egui::pos2(cx - dx, cy + dy), egui::pos2(cx, cy - dy)],
            stroke,
        );
        ui.painter().line_segment(
            [egui::pos2(cx + dx, cy + dy), egui::pos2(cx, cy - dy)],
            stroke,
        );
    } else {
        ui.painter().line_segment(
            [egui::pos2(cx - dx, cy - dy), egui::pos2(cx, cy + dy)],
            stroke,
        );
        ui.painter().line_segment(
            [egui::pos2(cx + dx, cy - dy), egui::pos2(cx, cy + dy)],
            stroke,
        );
    }
    response.on_hover_text(hover)
}

impl RotatePolytopesApp {
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

    /// Expanded section of the bottom overlay: rotation-mode tabs,
    /// the active-set checkboxes (Active mode) or the composer
    /// (Composer mode), and the shape row. Always-visible controls
    /// (Spin/Pause, rate buttons, sliders) live below this in
    /// `render_overlay` and are rendered separately.
    fn render_expanded_body(&mut self, ui: &mut egui::Ui) {
        // Mode tab: which source drives the continuous spin. Two
        // sub-panels swap below. The formula-display toggle sits at
        // the right of the same row; it's a viewport-level option
        // (independent of mode), not a mode setting itself.
        // Mode change deferred via `self.pending_mode`: the body
        // below this row reads `self.rotation_mode` (still the
        // OLD value this frame), and `render_overlay` swaps in
        // the new mode after `BottomOverlay::show` returns. This
        // keeps `BottomOverlay`'s measure pass and visible pass
        // rendering the same body height; clicking a tab shows
        // the new mode on the *next* frame, with the height
        // animation, but no mid-frame mismatch flicker.
        let mut staged = self.rotation_mode;
        ui.horizontal(|ui| {
            ui.selectable_value(&mut staged, RotationMode::Active, "Active set")
                .on_hover_text("Six checkbox-toggled bivectors (xy, xz, ...)");
            ui.selectable_value(&mut staged, RotationMode::Composer, "Composer")
                .on_hover_text("Sum of bivectors from the composed sequence");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.checkbox(&mut self.show_rotor_viz, "Rotor viz")
                    .on_hover_text("Top-right SO(4) plane decomposition of the angular velocity");
                ui.checkbox(&mut self.show_w_indicator, "w indicator")
                    .on_hover_text("Top-left scrub bar showing where the slice plane sits");
                ui.checkbox(&mut self.show_formula, "Show formula")
                    .on_hover_text("Top-right popup with the live exp(...) form of the rotor");
            });
        });
        if staged != self.rotation_mode {
            self.pending_mode = Some(staged);
        }
        // Filmstrip toggle row. Sliders' w_slice is ignored when
        // strip is on (cells span the whole range), so a separate
        // row is the clearest place for this view-mode switch.
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.strip_view, "Filmstrip")
                .on_hover_text(
                    "Render N cells across the scene at evenly-spaced w_slice; \
                     replaces the single-slice view",
                );
            if self.strip_view {
                ui.add(
                    egui::DragValue::new(&mut self.strip_count)
                        .range(3..=21)
                        .speed(0.2)
                        .prefix("cells: "),
                );
            }
        });

        // Mode-specific UI.
        if self.rotation_mode == RotationMode::Active {
            ui.horizontal(|ui| {
                for (active, plane) in self.active.iter_mut().zip(Plane4::ALL.iter()) {
                    ui.checkbox(active, plane.label());
                }
                // Combo name (e.g., "isoclinic xw+yz") inline on the
                // same row as the checkboxes; saves a row of
                // vertical space and the name reads as a label
                // applied to the active set right next to it.
                if let Some(name) = combo_name(&self.active) {
                    ui.add_space(8.0);
                    ui.colored_label(egui::Color32::from_rgb(255, 217, 140), name);
                }
            });
            return self.render_shapes_section(ui);
        }

        // Composer mode below.
        ui.separator();

        // Plane buttons append into the draft.
        ui.horizontal_wrapped(|ui| {
            for plane in Plane4::ALL.iter() {
                if ui
                    .small_button(format!("+{}", plane.label()))
                    .on_hover_text("Add to the current draft term")
                    .clicked()
                {
                    self.pending_actions.push(DeferredAction::DraftPush(*plane));
                }
            }
        });

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
                        let multi = self.draft.len() > 1;
                        if multi {
                            ui.monospace("(");
                        }
                        for (k, plane) in self.draft.iter().enumerate() {
                            if k > 0 {
                                ui.monospace("+");
                            }
                            ui.monospace(plane.label());
                        }
                        if multi {
                            ui.monospace(")");
                        }
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
                                    .min_size(egui::vec2(22.0, 22.0)),
                            )
                            .on_hover_text("Discard draft")
                            .clicked()
                        {
                            self.pending_actions.push(DeferredAction::DraftClear);
                        }
                    });
                });
        }

        // Sequence: each term is a single-row card. Whole card is
        // its own drag source (no separate handle); the card body
        // is also a drop zone that branches on payload variant.
        // Term payloads reorder, Entry payloads migrate a plane in.
        // Insertion pipes between cards give precise drop indication
        // for the Term-reorder path.
        let mut term_moves: Vec<(usize, usize)> = Vec::new();
        let mut entry_moves: Vec<(usize, usize, usize)> = Vec::new();
        let mut remove_term: Option<usize> = None;
        let mut remove_scalar: Option<usize> = None;
        let mut add_scalar: Option<usize> = None;

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
                ui.spacing_mut().item_spacing.x = 4.0;
                for term_idx in 0..self.seq.len() {
                    let gap_id = ui.make_persistent_id(("term-gap", term_idx));
                    if make_room_gap(
                        ui,
                        term_drop_idx == Some(term_idx),
                        gap_id,
                        term_h,
                        dragged_term_width,
                    ) {
                        if let Some(arc) = egui::DragAndDrop::take_payload::<DragPayload>(ui.ctx())
                        {
                            if let DragPayload::Term(from) = *arc {
                                if from != term_idx {
                                    term_moves.push((from, term_idx));
                                }
                            }
                        }
                    }
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
                                        let term = &mut self.seq[term_idx];
                                        if let Some(phi) = term.scalar.as_mut() {
                                            let mut deg = phi.to_degrees();
                                            let phi_color = egui::Color32::from_rgb(255, 150, 150);
                                            ui.scope(|ui| {
                                                let v = ui.visuals_mut();
                                                v.widgets.inactive.fg_stroke.color = phi_color;
                                                v.widgets.hovered.fg_stroke.color = phi_color;
                                                v.widgets.active.fg_stroke.color = phi_color;
                                                v.override_text_color = Some(phi_color);
                                                if ui
                                                    .add(
                                                        egui::DragValue::new(&mut deg)
                                                            .speed(0.0)
                                                            .suffix("°")
                                                            .range(-720.0..=720.0),
                                                    )
                                                    .on_hover_text("Click to type a new angle")
                                                    .changed()
                                                {
                                                    *phi = deg.to_radians();
                                                }
                                            });
                                            ui.monospace("·");
                                        }
                                        let n_planes = self.seq[term_idx].planes.len();
                                        let need_parens = n_planes > 1;
                                        if need_parens {
                                            ui.monospace("(");
                                        }
                                        for plane_idx in 0..n_planes {
                                            if plane_idx > 0 {
                                                ui.monospace("+");
                                            }
                                            let pill_id = ui.make_persistent_id((
                                                "plane-pill",
                                                term_idx,
                                                plane_idx,
                                            ));
                                            let plane_label =
                                                self.seq[term_idx].planes[plane_idx].label();
                                            ui.dnd_drag_source(
                                                pill_id,
                                                DragPayload::Entry(term_idx, plane_idx),
                                                |ui| {
                                                    ui.monospace(plane_label);
                                                },
                                            )
                                            .response
                                            .on_hover_cursor(egui::CursorIcon::Grab);
                                        }
                                        if need_parens {
                                            ui.monospace(")");
                                        }
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
                    let has_scalar = self.seq[term_idx].scalar.is_some();
                    let menu_resp = card_resp.interact(egui::Sense::click());
                    menu_resp.context_menu(|ui| {
                        if ui
                            .button(if has_scalar {
                                "Remove scalar (φ)"
                            } else {
                                "Add scalar (φ = 90°)"
                            })
                            .clicked()
                        {
                            if has_scalar {
                                remove_scalar = Some(term_idx);
                            } else {
                                add_scalar = Some(term_idx);
                            }
                            ui.close_kind(egui::UiKind::Menu);
                        }
                        ui.separator();
                        if ui.button("Delete term").clicked() {
                            remove_term = Some(term_idx);
                            ui.close_kind(egui::UiKind::Menu);
                        }
                    });
                }
                // Trailing insertion gap: drop after the last term.
                let trailing_id = ui.make_persistent_id(("term-gap", self.seq.len()));
                if make_room_gap(
                    ui,
                    term_drop_idx == Some(self.seq.len()),
                    trailing_id,
                    term_h,
                    dragged_term_width,
                ) {
                    if let Some(arc) = egui::DragAndDrop::take_payload::<DragPayload>(ui.ctx()) {
                        if let DragPayload::Term(from) = *arc {
                            term_moves.push((from, self.seq.len()));
                        }
                    }
                }
                // Reset per-index term animation state when a
                // mutation will fire; same reasoning as the
                // shape-row reset: ids resolve correctly only
                // inside this ui scope.
                if !term_moves.is_empty() || !entry_moves.is_empty() || remove_term.is_some() {
                    let ctx = ui.ctx();
                    for i in 0..32 {
                        let card_id = ui.make_persistent_id(("term-card", i));
                        let _ = ctx.animate_value_with_time(card_id.with("pickup"), 0.0, 0.0);
                        let _ = ctx.animate_value_with_time(card_id.with("collapse"), 1.0, 0.0);
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
        for (from, to) in term_moves {
            if from < self.seq.len() {
                let item = self.seq.remove(from);
                let dest = if to > from { to - 1 } else { to };
                self.seq.insert(dest.min(self.seq.len()), item);
            }
        }
        if let Some(i) = remove_term {
            if i < self.seq.len() {
                self.seq.remove(i);
            }
        }

        ui.horizontal(|ui| {
            let apply = ui
                .add_enabled(!self.seq.is_empty(), egui::Button::new("Apply"))
                .on_hover_text("Compose seq onto rot_state (one-shot)")
                .clicked();
            if apply {
                let terms = self.seq.clone();
                for term in &terms {
                    self.rot_state = (term.delta() * self.rot_state).normalize();
                }
                self.write_all(self.rot_state);
            }
            if ui.button("Clear").clicked() {
                self.seq.clear();
            }
        });

        self.render_shapes_section(ui);
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
        let mut shape_moves: Vec<(usize, usize)> = Vec::new();
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
                        ui.spacing_mut().item_spacing.x = 4.0;
                        for (i, entry) in self.row.iter().enumerate() {
                            // Animated insertion gap before card i.
                            // `ui.make_persistent_id` (NOT `Id::new`)
                            // is load-bearing: `BottomOverlay` runs
                            // its content closure twice per frame
                            // (measure pass off-screen + visible
                            // pass), and same-id-in-different-layer
                            // breaks egui's hit-testing. Per-pass
                            // ui scope makes the same source resolve
                            // to different ids between passes.
                            let gap_id = ui.make_persistent_id(("shape-gap", i));
                            if make_room_gap(
                                ui,
                                drop_idx == Some(i),
                                gap_id,
                                row_h,
                                SHAPE_CARD_WIDTH + 8.0,
                            ) {
                                if let Some(arc) =
                                    egui::DragAndDrop::take_payload::<usize>(ui.ctx())
                                {
                                    let from = *arc;
                                    if from != i {
                                        shape_moves.push((from, i));
                                    }
                                }
                            }
                            let drag_id = ui.make_persistent_id(("shape-card", i));
                            let pickup_t = drag_pickup_t(ui.ctx(), drag_id);
                            // Uniform gray cards. egui's noninteractive
                            // bg_fill matches surrounding panel chrome so
                            // the cards read as a "list of equally-
                            // weighted items" rather than a categorical
                            // color legend.
                            let card_fill = ui.visuals().widgets.noninteractive.bg_fill;
                            let stroke_color = if pickup_t > 0.0 {
                                egui::Color32::from_rgb(255, 200, 60)
                            } else {
                                ui.visuals().widgets.noninteractive.bg_stroke.color
                            };
                            let stroke = egui::Stroke::new(1.0 + pickup_t * 1.5, stroke_color);
                            let card_id = drag_id;
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
                                                    egui::Label::new(
                                                        egui::RichText::new(entry.label).strong(),
                                                    )
                                                    .selectable(false)
                                                    .wrap_mode(egui::TextWrapMode::Extend),
                                                );
                                            },
                                        );
                                    });
                            });
                            drag_resp
                                .on_hover_cursor(egui::CursorIcon::Grab)
                                .on_hover_text(entry.long_name)
                                .interact(egui::Sense::click())
                                .context_menu(|ui| {
                                    if row_len > 1 && ui.button("Remove from row").clicked() {
                                        remove_idx = Some(i);
                                        ui.close_kind(egui::UiKind::Menu);
                                    }
                                });
                        }
                        // Trailing insertion gap; drop after the
                        // last card.
                        let trailing_id = ui.make_persistent_id(("shape-gap", row_len));
                        if make_room_gap(
                            ui,
                            drop_idx == Some(row_len),
                            trailing_id,
                            row_h,
                            SHAPE_CARD_WIDTH + 16.0,
                        ) {
                            if let Some(arc) = egui::DragAndDrop::take_payload::<usize>(ui.ctx()) {
                                shape_moves.push((*arc, row_len));
                            }
                        }
                        // "+" trigger inline with the shape cards.
                        // Custom-painted plus on a 28×24 button rect so
                        // the height matches the cards exactly and the
                        // visual vocabulary matches the play / rate /
                        // chevron buttons (no font-glyph dependency).
                        if self.row.len() < MAX_ROW_LEN {
                            let plus_resp = add_button(ui).on_hover_text("Add a shape to the row");
                            egui::Popup::menu(&plus_resp).show(|ui| {
                                ui.set_min_width(80.0);
                                for shape_name in [
                                    "5-cell", "8-cell", "16-cell", "24-cell", "120-cell",
                                    "600-cell",
                                ] {
                                    if ui.button(shape_name).clicked() {
                                        if let Ok(entry) = parse_shape_name(shape_name) {
                                            self.row.push(entry);
                                            row_changed = true;
                                        }
                                        ui.close_kind(egui::UiKind::Menu);
                                    }
                                }
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
                        if !shape_moves.is_empty() || remove_idx.is_some() {
                            let ctx = ui.ctx();
                            for i in 0..=MAX_ROW_LEN {
                                let card_id = ui.make_persistent_id(("shape-card", i));
                                let _ =
                                    ctx.animate_value_with_time(card_id.with("pickup"), 0.0, 0.0);
                                let _ =
                                    ctx.animate_value_with_time(card_id.with("collapse"), 1.0, 0.0);
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
        for (from, to) in shape_moves {
            if from < self.row.len() {
                let item = self.row.remove(from);
                let dest = if to > from { to - 1 } else { to };
                self.row.insert(dest.min(self.row.len()), item);
                row_changed = true;
            }
        }
        if row_changed {
            self.rebuild_bodies();
        }
    }

    /// Modal help window; shown when `self.show_help` is `true`.
    /// Closes via the window's title-bar X (egui's
    /// `Window::open(&mut bool)` flips the bool).
    fn render_help_window(&mut self, ctx: &egui::Context) {
        if !self.show_help {
            return;
        }
        let mut open = self.show_help;
        egui::Window::new("About 4D Polytope Rotation")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_size(egui::vec2(560.0, 460.0))
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("What this program shows");
                    ui.label(
                        "You're looking at the 3D cross-sections of a row of \
                         four-dimensional polytopes. As they rotate through 4D \
                         space, their cross-sections morph in characteristic ways \
                        ; that's the whole point of the demo: to make 4D shape \
                         intuition reachable for someone in 3D.",
                    );
                    ui.add_space(8.0);

                    ui.heading("3D cross-sections, briefly");
                    ui.label(
                        "A cross-section is what you get when a higher-\
                         dimensional object passes through a lower-dimensional \
                         space. A 3D apple intersecting a 2D table at any moment \
                         gives a 2D shape (a circle, an oval, a curve), and the \
                         shape changes as the apple moves.",
                    );
                    ui.label(
                        "The same idea works one dimension up: a 4D polytope \
                         passing through 3D space gives a 3D shape at every \
                         instant. The hidden 4th axis is conventionally called \
                         w. As w changes, the polytope's 3D cross-section morphs \
                        ; that is what the w slider scrubs through.",
                    );
                    ui.add_space(8.0);

                    ui.heading("The shapes");
                    ui.label("All six convex regular 4-polytopes (\"polychora\") ship:");
                    ui.label("• 5-cell (pentatope); 5 tetrahedra; the 4D simplex.");
                    ui.label("• 8-cell (tesseract); 8 cubes; the 4D cube.");
                    ui.label(
                        "• 16-cell (hexadecachoron); 16 tetrahedra; \
                         the 4D analog of the octahedron.",
                    );
                    ui.label(
                        "• 24-cell (icositetrachoron); 24 octahedra; \
                         uniquely 4-dimensional, no 3D analog.",
                    );
                    ui.label("• 120-cell (hecatonicosachoron); 120 dodecahedra.");
                    ui.label(
                        "• 600-cell (hexacosichoron); 600 tetrahedra; \
                         the 4D analog of the icosahedron.",
                    );
                    ui.add_space(8.0);

                    ui.heading("Rotation");
                    ui.label(
                        "4D space has six independent rotation planes, not three: \
                         xy, xz, xw, yz, yw, zw. In 3D you spin around an axis; \
                         in 4D you spin in a plane (the bivector picture). The \
                         three planes that include w (xw, yw, zw) pull a visible \
                         axis through the hidden 4th dimension and produce the \
                         interesting cross-section morphs. The three pure-3D \
                         planes (xy, xz, yz) just rotate the cross-section as a \
                         rigid 3D shape.",
                    );
                    ui.label(
                        "Active set mode: toggle which planes contribute. The \
                         angular velocity is the sum of the active unit \
                         bivectors. Composer mode: build a sequence of terms \
                         (each term is a sum of planes, optionally scaled by a \
                         scalar φ); the seq sums into the angular velocity.",
                    );
                    ui.add_space(8.0);

                    ui.heading("Controls");
                    ui.label("• w slider: scrub the slicing hyperplane along the 4th axis.");
                    ui.label("• t slider: scrub the rotation animation by absolute time.");
                    ui.label("• Play / Pause: start or pause the spin.");
                    ui.label("• << < > >>: set the rate to ×0.25 / ×0.5 / ×2 / ×4.");
                    ui.label("• Reset (R): zero everything.");
                    ui.label("• ^ / v (H): expand or collapse the controls overlay.");
                    ui.label("• 1..6: toggle a plane in the active set.");
                    ui.label("• T: toggle spin.");
                    ui.label("• Up / Down arrows: scrub the w-slice with the keyboard.");
                    ui.label("• Drag in the viewport: orbit the camera.");
                });
            });
        self.show_help = open;
    }

    /// Unified bottom-of-window controls overlay. The expanded
    /// section (mode tabs, mode-specific UI, shape row) appears
    /// when `self.expanded`; the slider strip + rate row are always
    /// visible. The whole overlay is a single translucent popup
    /// painted over the full-window scene; there's no side panel
    /// anymore, and the scene fills the entire viewport.
    fn render_overlay(&mut self, ctx: &egui::Context) {
        let screen = ctx.content_rect();
        let pad = 16.0;
        let area_w = (screen.width() - 2.0 * pad).max(280.0);

        // `BottomOverlay` auto-sizes to its content and animates
        // height transitions smoothly, so when the user toggles
        // expand or switches rotation modes the panel grows /
        // shrinks to exactly fit the new content with no dead
        // space and no flicker.
        let visuals = &ctx.style().visuals;
        let frame = egui::Frame::default()
            .fill(visuals.window_fill)
            .stroke(visuals.window_stroke)
            .corner_radius(visuals.window_corner_radius)
            .inner_margin(10.0);

        BottomOverlay::new("rotate-polytopes-overlay")
            .width(area_w)
            .margin_y(pad)
            .frame(frame)
            .show(ctx, |ui| {
                // Render top-down: body at top, sliders, rate row at
                // bottom. `BottomOverlay`'s internal ScrollArea
                // anchors the bottom, so the rate row stays visible
                // throughout collapse animations.
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
            }
        }
    }

    /// Two big sliders (w, t) with fixed-width monospace value
    /// labels. `area_w` is the parent's content width in points.
    fn render_slider_strip(&mut self, ui: &mut egui::Ui, area_w: f32) {
        // Sliders use `show_value(false)` + a separately-allocated
        // fixed-width monospace label per row, so the slider's
        // bounding rect width never changes as the value's char
        // count does (e.g., "0.5" -> "0.50" -> "12.34"). Without this
        // stabilization, the popup Frame's painted rect oscillates
        // each frame as the spin advances `rot_time`, and the
        // entire overlay visibly jitters.
        // No leading axis-name cell; the axis is folded into the
        // trailing value (e.g. "w +0.000", "t  8.70s"), so the
        // slider hugs the frame's left edge with no dead space.
        // Trailing cell is fixed-width monospace so the value's
        // char count never makes the slider rect oscillate as the
        // spin advances `rot_time`.
        const VALUE_CELL_W: f32 = 86.0;
        let slider_w = (area_w - VALUE_CELL_W - 16.0).max(140.0);
        ui.spacing_mut().slider_width = slider_w;
        let value_layout = egui::Layout::left_to_right(egui::Align::Center);

        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut self.w_slice, -W_RANGE..=W_RANGE).show_value(false));
            ui.allocate_ui_with_layout(egui::vec2(VALUE_CELL_W, 14.0), value_layout, |ui| {
                ui.add(egui::Label::new(
                    egui::RichText::new(format!("w {:>+.3}", self.w_slice)).monospace(),
                ));
            });
        });
        let mut t_dragged = false;
        ui.horizontal(|ui| {
            let resp = ui.add(egui::Slider::new(&mut self.rot_time, 0.0..=30.0).show_value(false));
            ui.allocate_ui_with_layout(egui::vec2(VALUE_CELL_W, 14.0), value_layout, |ui| {
                ui.add(egui::Label::new(
                    egui::RichText::new(format!("t {:>5.2}s", self.rot_time)).monospace(),
                ));
            });
            t_dragged = resp.dragged();
        });
        // Gate the scrub-from-zero recomputation on `.dragged()` so
        // it ONLY fires while the user is actively scrubbing the
        // slider. Using `.changed()` here misfired every frame the
        // spin's `rot_time += dt_secs` accumulator advanced the
        // value, producing the snap the user reported when toggling
        // active checkboxes (omega would shift, and re-deriving
        // `exp(omega_new * t)` from an accumulated `t` is a
        // discontinuous jump rather than the smooth integrated path).
        if t_dragged {
            let omega = match self.rotation_mode {
                RotationMode::Active => angular_velocity(&self.active, self.rate_scale),
                RotationMode::Composer => angular_velocity_from_seq(&self.seq, self.rate_scale),
            };
            self.rot_state = (omega * self.rot_time).exp().normalize();
            self.write_all(self.rot_state);
        }
    }

    /// Always-visible single row directly under the sliders.
    /// Center-justified play/rate cluster with the right-aligned
    /// utility cluster on the same line:
    ///
    /// ```text
    ///                  [<<] [<] [play/pause] [>] [>>]      [Reset] [?] [^]
    ///                            ×1.00
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
            rate_toggle(ui, &mut self.rate_scale, 0.25, true, false);
            rate_toggle(ui, &mut self.rate_scale, 0.5, false, false);
            if play_pause_button(ui, self.rotate)
                .on_hover_text("Toggle continuous rotation (Space)")
                .clicked()
            {
                self.rotate = !self.rotate;
            }
            rate_toggle(ui, &mut self.rate_scale, 2.0, false, true);
            rate_toggle(ui, &mut self.rate_scale, 4.0, true, true);
            if refresh_button(ui)
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
                            .min_size(egui::vec2(22.0, 22.0)),
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

    /// Render a compact formula for what's currently driving the
    /// rotor: the continuous active-set bivector (multiplied by the
    /// rate and time, when nonzero) followed by the multiplicative
    /// composed sequence (each term parenthesized when it's a sum).
    /// Empty string when nothing is contributing.
    fn formula_string(&self) -> String {
        // The rotation source is exclusive; only one mode drives
        // the spin at a time. The formula popup must reflect THAT
        // mode's expression, not concatenate both, otherwise the
        // user reads it as "we're applying both" when in fact the
        // off-mode's terms aren't contributing to omega.
        match self.rotation_mode {
            RotationMode::Active => {
                let active_planes: Vec<&'static str> = Plane4::ALL
                    .iter()
                    .zip(self.active.iter())
                    .filter(|(_, on)| **on)
                    .map(|(p, _)| p.label())
                    .collect();
                if active_planes.is_empty() {
                    return String::new();
                }
                let bivec = if active_planes.len() == 1 {
                    active_planes[0].to_string()
                } else {
                    format!("({})", active_planes.join(" + "))
                };
                format!(
                    "exp({} · {:.2}·t)",
                    bivec,
                    self.rate_scale * BASE_ROTATION_RATE / std::f32::consts::TAU
                )
            }
            RotationMode::Composer => {
                let parts: Vec<String> = self
                    .seq
                    .iter()
                    .filter(|t| !t.planes.is_empty())
                    .map(|term| {
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
                        let body = match term.scalar {
                            Some(phi) => format!("{:.0}° · {}", phi.to_degrees(), bivec),
                            None => bivec,
                        };
                        format!("exp({body})")
                    })
                    .collect();
                parts.join(" · ")
            }
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
            expanded: false,
            show_help: false,
            show_formula: false,
            strip_view: false,
            strip_count: 11,
            show_w_indicator: true,
            show_rotor_viz: true,
            rotation_mode: RotationMode::Active,
            pending_mode: None,
            pending_actions: Vec::new(),
            seq: Vec::new(),
            draft: Vec::new(),
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
        // under identical 4D motion. Rotor accumulates per-frame
        // (delta = exp(ω · dt)) so pause naturally freezes
        // orientation in place, see KeyT handler.
        if self.rotate {
            self.rot_time += dt_secs;
            let omega_per_sec = match self.rotation_mode {
                RotationMode::Active => angular_velocity(&self.active, self.rate_scale),
                RotationMode::Composer => angular_velocity_from_seq(&self.seq, self.rate_scale),
            };
            let omega = omega_per_sec * dt_secs;
            // No-op when no planes are active; skip the exp+mul.
            if omega.magnitude_squared() > 0.0 {
                let delta = omega.exp();
                self.rot_state = (delta * self.rot_state).normalize();
                self.write_all(self.rot_state);
            }
        }

        // Camera. Gate the orbit on `!ui_has_focus` so dragging the
        // egui w-slice slider doesn't also rotate the camera.
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

        // Top-left: title + fps + framebuffer size. Replaces the old
        // panel header now that the side panel is gone. Larger
        // typography so the title reads as the program's nameplate
        // rather than just another label.
        let cfg = &frame.rd.surface_bundle.config;
        let (fb_w, fb_h) = (cfg.width, cfg.height);
        egui::Area::new(egui::Id::new("rotate-polytopes-title"))
            .anchor(egui::Align2::LEFT_TOP, [20.0, 18.0])
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

        // Top-right: live rotation formula and the combo name. Off
        // by default (the math notation is dense for newcomers);
        // toggled by the "Show formula" checkbox in the expanded
        // section.
        if self.show_formula {
            let formula = self.formula_string();
            // Combo name ("isoclinic xw+yz" etc.) is an Active-mode
            // label; it describes the active-set bivector, not the
            // composer's seq. Suppress it in Composer mode so the
            // popup reads as the seq's expression alone.
            let name = if self.rotation_mode == RotationMode::Active {
                combo_name(&self.active)
            } else {
                None
            };
            if !formula.is_empty() || name.is_some() {
                egui::Area::new(egui::Id::new("rotate-polytopes-formula"))
                    .anchor(egui::Align2::RIGHT_TOP, [-16.0, 16.0])
                    .show(ctx, |ui| {
                        egui::Frame::popup(&ctx.style())
                            .inner_margin(8.0)
                            .show(ui, |ui| {
                                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                                if !formula.is_empty() {
                                    ui.add(egui::Label::new(
                                        egui::RichText::new(&formula).monospace(),
                                    ));
                                }
                                if let Some(n) = name {
                                    ui.add(egui::Label::new(
                                        egui::RichText::new(n)
                                            .color(egui::Color32::from_rgb(255, 217, 140)),
                                    ));
                                }
                            });
                    });
            }
        }

        // Top-left under the title: read-only `w` slice indicator.
        // Hidden in filmstrip mode (the strip itself answers
        // "where in the w-extent are we").
        if self.show_w_indicator && !self.strip_view {
            egui::Area::new(egui::Id::new("rotate-polytopes-w-indicator"))
                .anchor(egui::Align2::LEFT_TOP, [20.0, 92.0])
                .show(ctx, |ui| {
                    egui::Frame::popup(&ctx.style())
                        .inner_margin(6.0)
                        .show(ui, |ui| {
                            LinearIndicator::new("w", self.w_slice, -W_RANGE..=W_RANGE)
                                .desired_width(180.0)
                                .show(ui);
                        });
                });
        }

        // Top-right under the formula popup: SO(4) rotation
        // visualizer. The omega bivector for the active rotation
        // source decomposes into one or two simple rotation planes.
        if self.show_rotor_viz {
            let omega = match self.rotation_mode {
                RotationMode::Active => angular_velocity(&self.active, self.rate_scale),
                RotationMode::Composer => angular_velocity_from_seq(&self.seq, self.rate_scale),
            };
            // Stack below the formula popup if it's open. The
            // formula's typical height is ~50pt; the rotor viz is
            // ~52pt tall, so a 80pt stagger keeps them from
            // overlapping when both are shown.
            let y_offset = if self.show_formula { 80.0 } else { 16.0 };
            egui::Area::new(egui::Id::new("rotate-polytopes-rotor-viz"))
                .anchor(egui::Align2::RIGHT_TOP, [-16.0, y_offset])
                .show(ctx, |ui| {
                    egui::Frame::popup(&ctx.style())
                        .inner_margin(6.0)
                        .show(ui, |ui| {
                            RotorVisualizer::new(omega, "omega").show(ui);
                        });
                });
        }

        // Bottom-anchored unified controls overlay. Sliders + rate
        // row always visible; the rest expands above on chevron/H.
        self.render_overlay(ctx);

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
            KeyCode::KeyH if pressed => self.expanded = !self.expanded,
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
        if self.strip_view {
            // Filmstrip: tile the framebuffer with N cells, each at
            // an evenly-spaced `w_slice` from `-W_RANGE` to
            // `+W_RANGE`. The freeze-frame view of the 4D extent.
            let cells = viewport.split_horizontal(self.strip_count as u32);
            let n = cells.len().max(1);
            let strip: Vec<(Viewport, f32)> = cells
                .into_iter()
                .enumerate()
                .map(|(i, vp)| {
                    let t = if n == 1 {
                        0.0
                    } else {
                        i as f32 / (n - 1) as f32
                    };
                    let w = -W_RANGE + t * (2.0 * W_RANGE);
                    (vp, w)
                })
                .collect();
            self.node.execute_strip(rd, view, &strip)
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
                    let plus = add_button(ui);
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
