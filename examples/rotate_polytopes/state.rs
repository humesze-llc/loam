//! Demo state: the [`RotatePolytopesApp`] struct, the mode/view/
//! deferred-action enums, the [`RotorTerm`] data type and its display
//! helpers, the angular-velocity derivation, body layout, and full
//! reset.
//!
//! This module owns the data model. Per-mode UI rendering lives in
//! `modes/{active,composer,filmstrip,shapes}.rs` as additional `impl
//! RotatePolytopesApp` blocks; cross-cutting overlay UI lives in
//! `ui.rs`. All struct fields are `pub(crate)` so those sibling impls
//! can access them directly without per-field accessors.

use rye_app::{Camera, OrbitController};
use rye_math::{Bivector4, EuclideanR3, Plane4, Rotor4};
use rye_render::raymarch::{BodyUniform, Hyperslice4DNode};

use crate::catalog::ShapeEntry;
use crate::consts::{BASE_ROTATION_RATE, BODY_SIZE, BODY_X_SPACING, BODY_Y, T_SLIDER_INITIAL};

// ---------------------------------------------------------------------------
// Mode + view enums
// ---------------------------------------------------------------------------

/// Continuous-rotation source. Two distinct UIs (active-set
/// checkboxes vs composed sequence) populate the angular velocity
/// independently; the user picks which one drives `omega` for the
/// spin animation via a tab in the rotation tab row.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum RotationMode {
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
pub(crate) enum ViewMode {
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

// ---------------------------------------------------------------------------
// RotorTerm + display helpers
// ---------------------------------------------------------------------------

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
pub(crate) struct RotorTerm {
    /// Unit-bivector planes summed inside `exp(...)`. Non-empty
    /// for a term to display; an empty term is dropped.
    pub(crate) planes: Vec<Plane4>,
    /// Optional scalar prefix `phi` in radians. `None` means the
    /// raw bivector sum (unit magnitude); `Some(phi)` scales the
    /// whole sum before `exp()`. The panel's "Add scalar" action
    /// initialises this to `FRAC_PI_2`; `Default::default()` is
    /// `None` so an empty draft commits as a unit-magnitude term.
    pub(crate) scalar: Option<f32>,
}

/// Render `(p_0 + p_1 + ...)` (with parens iff multi-plane) into
/// the current ui. Each plane goes through `render_plane`, which
/// decides whether it's an interactive drag pill (term card),
/// plain monospace (draft card), or anything else. The paren
/// logic and `+` separators are shared so the visual reading of
/// a bivector sum stays identical across all callsites.
pub(crate) fn render_plane_sum(
    ui: &mut rye_app::egui::Ui,
    planes: &[Plane4],
    mut render_plane: impl FnMut(&mut rye_app::egui::Ui, usize, Plane4),
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
pub(crate) fn render_term(term: &RotorTerm) -> String {
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
pub(crate) fn render_bivector_sum(parts: &[String]) -> Option<String> {
    match parts {
        [] => None,
        [only] => Some(only.clone()),
        many => Some(format!("({})", many.join(" + "))),
    }
}

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
pub(crate) fn angular_velocity_from_seq(seq: &[RotorTerm], rate_scale: f32) -> Bivector4 {
    let mut omega = Bivector4::ZERO;
    for term in seq {
        let phi = term.scalar.unwrap_or(1.0);
        for plane in &term.planes {
            omega = omega + plane.unit_bivector() * phi;
        }
    }
    omega * (BASE_ROTATION_RATE * rate_scale)
}

// ---------------------------------------------------------------------------
// Deferred action queue
// ---------------------------------------------------------------------------

/// State mutations queued during overlay rendering and applied
/// AFTER the overlay's measure + visible passes finish. Any
/// mutation that changes the overlay's natural content height
/// must go through this; applying mid-frame would make the two
/// `BottomOverlay` passes disagree on body height and the user
/// would see a one-frame layout mismatch as flicker.
#[derive(Clone, Debug)]
pub(crate) enum DeferredAction {
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
pub(crate) enum DragPayload {
    /// The whole term at this seq index is being dragged.
    Term(usize),
    /// `Entry(term_idx, plane_idx)`: a single plane pill from the
    /// given term is being dragged.
    Entry(usize, usize),
}

// ---------------------------------------------------------------------------
// Body layout helper
// ---------------------------------------------------------------------------

/// Position of the `slot`-th body in a row of `n` bodies, centred
/// on the world origin and spaced by [`BODY_X_SPACING`]. Used by
/// both initial body layout and per-frame body uniforms.
pub(crate) fn body_position(slot: usize, n: usize) -> [f32; 4] {
    let x = (slot as f32 - (n as f32 - 1.0) * 0.5) * BODY_X_SPACING;
    [x, BODY_Y, 0.0, 0.0]
}

// ---------------------------------------------------------------------------
// The App struct
// ---------------------------------------------------------------------------

pub(crate) struct RotatePolytopesApp {
    pub(crate) space: EuclideanR3,
    pub(crate) camera: Camera<EuclideanR3>,
    pub(crate) orbit: OrbitController<EuclideanR3>,
    pub(crate) node: Hyperslice4DNode,
    /// Polytope row built at startup from `--shapes` CLI args (or
    /// `DEFAULT_ROW`); drives both the body uniforms and per-body
    /// label lookups in the overlay.
    pub(crate) row: Vec<ShapeEntry>,

    pub(crate) w_slice: f32,
    pub(crate) slider_up_held: bool,
    pub(crate) slider_down_held: bool,

    pub(crate) rotate: bool,
    pub(crate) rot_state: Rotor4,
    /// Toggle bitmap for the six rotation planes; sum of active
    /// planes' unit bivectors becomes the per-frame angular
    /// velocity. See [`Plane4::ALL`] for the index -> plane mapping.
    pub(crate) active: [bool; 6],
    pub(crate) rate_scale: f32,
    /// Accumulated time spent rotating (advances only while
    /// `rotate == true`; resets on **R**). Useful for spotting
    /// periodicities in compound-bivector animations.
    pub(crate) rot_time: f32,
    /// Upper bound on the `t` slider's range. Doubles every time
    /// the spin's accumulated `rot_time` exceeds the current
    /// bound, so the slider's handle stays meaningful at long
    /// elapsed times instead of pinning at the right edge.
    /// Reset to the initial bound on `R`.
    pub(crate) t_slider_max: f32,

    /// Whether the bottom controls overlay is expanded. When
    /// `false` only the always-on slider strip + rate row is shown
    /// at the bottom; when `true` the strip extends upward to also
    /// show the rotation-mode tabs, mode-specific UI, and shape
    /// row. Toggle via the `^` / `v` chevron button or the **H**
    /// key. There is no longer a side panel: the scene renders to
    /// the full window and the overlay floats over it.
    pub(crate) expanded: bool,

    /// Whether the modal "About / help" window is open. Triggered
    /// by clicking the `?` button; closes via the window's title-
    /// bar X (egui's `Window::open(&mut bool)` flips it).
    pub(crate) show_help: bool,

    /// Cached natural overlay width on first frame. Used as the
    /// fixed width of the overlay regardless of the current
    /// window size, so resizing the demo window doesn't stretch
    /// the controls. Set lazily on first render.
    pub(crate) overlay_pinned_width: Option<f32>,

    /// Whether the top-right rotation-formula popup is rendered.
    /// Off by default; the formula is dense for newcomers; the
    /// expanded section has a checkbox to turn it on for users who
    /// want to see exactly which bivectors and scalars compose into
    /// the current orientation.
    pub(crate) show_formula: bool,

    /// Whether the bottom controls overlay is rendered. On by
    /// default so first-time users see all the demo's state at
    /// once; toggle off via `View > Rotation controls` or the
    /// `H` key for an unobstructed scene (e.g., for screenshots
    /// or focused viewing).
    pub(crate) show_controls: bool,

    /// Top-level visualisation mode. `Shapes` shows `self.row`
    /// side-by-side at one `w_slice`; `Filmstrip` shows one
    /// polytope (`self.strip_subject`) sampled across an axis
    /// of w, an axis of t, or both at once (a 2D grid).
    pub(crate) view_mode: ViewMode,
    /// Filmstrip-axis toggles. At least one MUST be active when
    /// `view_mode == Filmstrip` (UI prevents both being off);
    /// when only `strip_w` is on the panel renders a horizontal
    /// row of cells across the w slider's value, when only
    /// `strip_t` is on it renders a vertical column across the
    /// rotation animation's `rot_time`, and when both are on it
    /// renders a 2D grid (w on one axis, t on the other; default
    /// orientation has w on columns and t on rows, swappable via
    /// `strip_swap_axes`).
    pub(crate) strip_w: bool,
    pub(crate) strip_t: bool,
    /// When both `strip_w` and `strip_t` are active, swap the
    /// default axis assignment (w-on-columns / t-on-rows becomes
    /// t-on-columns / w-on-rows).
    pub(crate) strip_swap_axes: bool,
    /// Cell counts along each filmstrip axis. Range 3..=21.
    pub(crate) strip_count_w: usize,
    pub(crate) strip_count_t: usize,
    /// Forward extent of the t-axis fan in animation seconds.
    pub(crate) strip_t_extent: f32,
    /// Polytope rendered in each filmstrip cell. Independent of
    /// `self.row`.
    pub(crate) strip_subject: ShapeEntry,

    /// Which rotation source drives the continuous spin.
    pub(crate) rotation_mode: RotationMode,

    /// Mode change requested this frame by the mode tabs. Applied
    /// after the overlay finishes rendering so that the body that
    /// renders this frame still sees `rotation_mode` (the OLD
    /// value), and only the next frame swaps to the new mode.
    pub(crate) pending_mode: Option<RotationMode>,

    /// View change requested this frame by the view tab row.
    pub(crate) pending_view_mode: Option<ViewMode>,

    /// Composer-mode actions deferred to end-of-frame for the same
    /// reason as `pending_mode`.
    pub(crate) pending_actions: Vec<DeferredAction>,

    /// Sequence of [`RotorTerm`]s the user is building in the panel.
    pub(crate) seq: Vec<RotorTerm>,
    /// In-progress draft for the next term. Plane buttons append
    /// here; "Add" commits this list as a new term in `seq` and
    /// clears the draft.
    pub(crate) draft: Vec<Plane4>,

    /// Typed-formula input for the Composer's text bar.
    pub(crate) formula_input: String,
    /// Last parse error from the formula bar.
    pub(crate) formula_error: Option<String>,
}

// ---------------------------------------------------------------------------
// State methods
// ---------------------------------------------------------------------------

impl RotatePolytopesApp {
    /// The composer seq's net bivector direction (no rate or
    /// base-rate scaling). This is the "function" the seq
    /// describes: sum over terms of `scalar * sum_planes`. The
    /// scrub slider uses this as its rotation axis-bivector;
    /// the projection of `log(rot_state)` onto this direction is
    /// the slider's value.
    pub(crate) fn compose_omega(&self) -> Bivector4 {
        let mut omega = Bivector4::ZERO;
        for term in &self.seq {
            let phi = term.scalar.unwrap_or(1.0);
            for plane in &term.planes {
                omega = omega + plane.unit_bivector() * phi;
            }
        }
        omega
    }

    /// Per-animation-second angular velocity (the bivector that,
    /// integrated over animation time, produces `rot_state`).
    /// Independent of `rate_scale`. Active mode sums the toggled
    /// basis bivectors; Composer mode delegates to the seq walker.
    pub(crate) fn omega_animation(&self) -> Bivector4 {
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
    pub(crate) fn write_all(&mut self, rotor: Rotor4) {
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

    /// Re-emit every body's uniform from the current row + rotor
    /// state. Called after row mutations (add/remove/reorder) to
    /// resync the GPU side to what the panel shows.
    pub(crate) fn rebuild_bodies(&mut self) {
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
    pub(crate) fn formula_string(&self) -> String {
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
    pub(crate) fn reset(&mut self) {
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
