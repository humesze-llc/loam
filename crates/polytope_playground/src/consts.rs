//! Demo-wide layout, sizing, and animation constants.

/// Max shapes per row via the runtime "Add" button; CLI `--shapes`
/// can still spawn up to `MAX_BODIES` at startup.
pub(crate) const MAX_ROW_LEN: usize = 8;

/// Uniform shape-card width. Wide enough to fit the longest label
/// ("120-cell") in bold unwrapped; wrapping would stagger card
/// heights into a staircase via egui's cross-alignment.
pub(crate) const SHAPE_CARD_WIDTH: f32 = 64.0;

/// Unified height for every interactive widget in the bottom
/// overlay. ~17 pt strong body text + 6 pt vertical inner_margin.
pub(crate) const CONTROL_H: f32 = 29.0;

/// Square control-button width (rate row, shape row). Play/pause is
/// wider (see [`PLAY_PAUSE_W`]); help/close use [`MINI_BUTTON_W`].
pub(crate) const CONTROL_W: f32 = 28.0;

/// Wider central play/pause control; asymmetry marks the primary
/// action in the rate cluster.
pub(crate) const PLAY_PAUSE_W: f32 = 36.0;

/// Compact close / help glyphs (`×`, `?`), read as utility chrome.
pub(crate) const MINI_BUTTON_W: f32 = 22.0;

/// Horizontal card spacing; the make-room gap reuses this value so
/// it can't desync.
pub(crate) const CARD_ITEM_SPACING_X: f32 = 4.0;

pub(crate) const W_SCRUB_RATE: f32 = 0.5;
pub(crate) const W_RANGE: f32 = 1.5;

/// Floor on the Hyperslice slab full-width. Thickness 0 would demand
/// exact f32 equality `w_min == w_slice == w_max`, which never fires
/// for a generic rotor and hides the wireframe; this keeps a razor-
/// thin band that a straddling w-range can still cross.
pub(crate) const HYPERSLICE_MIN_THICKNESS: f32 = 1e-4;

/// Default Hyperslice slab full-width: keeps edges within `+/- 0.1`
/// of the cut. Tuned against `BODY_SIZE`-scaled polytopes (w-extent
/// ~0.7).
pub(crate) const HYPERSLICE_DEFAULT_THICKNESS: f32 = 0.2;

/// Arrow-key t scrub rate (seconds of rot_time per real second).
/// Faster than the w scrub since t ranges further than w's bounded
/// slice axis.
pub(crate) const T_SCRUB_RATE: f32 = 1.0;

/// Initial t-slider max. Set to `2 × W_RANGE = 3.0` so per-pixel
/// scrub precision matches the w slider; the `update()` runaway
/// guard doubles it as rot_time grows.
pub(crate) const T_SLIDER_INITIAL: f32 = 3.0;

/// Base rotation angular rate (rad/s), scaled by `rate_scale`.
pub(crate) const BASE_ROTATION_RATE: f32 = std::f32::consts::TAU * 0.3;

/// Body-center x spacing; slightly over `BODY_SIZE * 2` so rotated
/// bodies can reach into a neighbor's column without overlap.
pub(crate) const BODY_X_SPACING: f32 = 1.8;

/// Per-body circumradius; sized so four shapes fit in view at once.
pub(crate) const BODY_SIZE: f32 = 0.7;

/// Center-y for all bodies; floor is at y=0.
pub(crate) const BODY_Y: f32 = 0.9;

/// Sub-segments per wireframe edge when a curved `space` blend is
/// active, so each great-circle arc reads smooth. Pure-flat mode
/// bypasses tessellation and emits one segment per edge.
pub(crate) const SPACE_TESSELLATION_SAMPLES: usize = 16;
