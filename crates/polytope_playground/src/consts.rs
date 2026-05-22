//! Demo-wide layout, sizing, and animation constants.
//!
//! All values are UX choices specific to this demo, lifted out of
//! `main.rs` only for readability. Keep them here rather than at use
//! sites so a future tweak to e.g. `CONTROL_H` ripples consistently
//! across the rate row, shape row, and term cards.

/// Cap on shapes per row from the runtime "Add" buttons. Keeps the
/// scene visible without scroll-zoom and bounds the per-frame body
/// loop. The CLI `--shapes` argument can still spawn up to
/// `MAX_BODIES` (32) at startup.
pub(crate) const MAX_ROW_LEN: usize = 8;

/// Uniform width for shape cards in the row. Wide enough to fit
/// the longest label ("120-cell" / "600-cell") in bold without the
/// label wrapping; wrapping would make those cards taller than
/// the others, which egui's horizontal cross-alignment then turns
/// into a descending staircase as the row's running max-height
/// grows past earlier (now lower-aligned) cards. Labels also use
/// `WrapMode::Extend` as a belt-and-suspenders check.
pub(crate) const SHAPE_CARD_WIDTH: f32 = 64.0;

/// Unified height for every interactive widget in the bottom
/// overlay: rate, play, chevron, plus, and refresh buttons,
/// make-room drag gaps, and shape cards (via their Frame's
/// inner_margin). Sized to match the cards' natural rendered
/// height; strong-styled body text in egui's default font measures
/// ~17 pt, plus the cards' 6-pt vertical inner_margin = 29 pt.
/// Keeping all controls at this same height removes the height
/// mismatch that would otherwise make the + button appear higher
/// than the cards.
pub(crate) const CONTROL_H: f32 = 29.0;

/// Standard width for square control buttons in the overlay's
/// rate row and shape row (`<<`, `<`, `>`, `>>`, refresh, the per-
/// shape `×`). Matches the visual cadence of the row without each
/// callsite hardcoding the same `28.0`. The play/pause button is
/// deliberately wider (see [`PLAY_PAUSE_W`]) and the smaller help
/// / close glyphs use [`MINI_BUTTON_W`].
pub(crate) const CONTROL_W: f32 = 28.0;

/// Wider central play/pause control. Asymmetry signals the primary
/// action in the rate cluster.
pub(crate) const PLAY_PAUSE_W: f32 = 36.0;

/// Compact close / help glyphs (`×`, `?`). Smaller than the rate-
/// cluster controls so they read as utility chrome, not primary
/// actions.
pub(crate) const MINI_BUTTON_W: f32 = 22.0;

/// Horizontal spacing between adjacent cards in the term and shape
/// rows. The make-room gap animates open to a card's width *plus*
/// this gap, so the value is shared and can't desync.
pub(crate) const CARD_ITEM_SPACING_X: f32 = 4.0;

pub(crate) const W_SCRUB_RATE: f32 = 0.5;
pub(crate) const W_RANGE: f32 = 1.5;

/// Animation-time scrub rate for the left / right arrow keys, in
/// seconds-of-rot_time per real second held. 1.0 means a one-second
/// real-time hold advances `rot_time` by one second of animation.
/// Faster than the w-axis scrub because t scrubs "into the future"
/// over a longer range than w's bounded slice axis.
pub(crate) const T_SCRUB_RATE: f32 = 1.0;

/// Initial maximum value for the t slider's range. Chosen so the
/// per-pixel scrub precision matches the w slider's: w spans
/// `2 × W_RANGE = 3.0` over the same slider track, so starting t
/// at 3.0 means dragging t feels just as smooth as dragging w.
/// The runaway guard in `update()` doubles this as the spin
/// pushes `rot_time` past it; precision halves with each
/// doubling but the user keeps the high precision early on when
/// fine scrubbing matters most.
pub(crate) const T_SLIDER_INITIAL: f32 = 3.0;

/// Base rotation angular rate (rad/s). Scaled by `rate_scale` per
/// frame so the rate buttons can speed it up or slow it down.
pub(crate) const BASE_ROTATION_RATE: f32 = std::f32::consts::TAU * 0.3;

/// Spacing between body centers along x. Slightly larger than
/// `BODY_SIZE * 2` so rotated bodies can stretch into a neighbor's
/// column without overlap during animation.
pub(crate) const BODY_X_SPACING: f32 = 1.8;

/// Per-body circumradius. Smaller than the `[-2, +2]` first row of
/// shapes was at, letting four shapes fit in view at once.
pub(crate) const BODY_SIZE: f32 = 0.7;

/// Center-y for all bodies; floor is at y=0.
pub(crate) const BODY_Y: f32 = 0.9;
