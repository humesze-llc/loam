//! Floating bottom-anchored overlay panel with flicker-free size transitions.
//!
//! ## The problem this solves
//!
//! egui's [`Area`](egui::Area) recomputes its position from content size each frame. When
//! content size changes drastically between frames (an app state toggle adding or removing
//! a big chunk of UI), the area's pivot recomputes in a single frame and the in-between
//! rendering reads as the overlay "flickering" or "disappearing" briefly.
//!
//! For floating bottom HUDs in games this happens whenever the HUD expands a panel, switches
//! modes, opens an inventory, etc. Single-frame jumps are unacceptable polish-wise.
//!
//! ## How this widget fixes it
//!
//! The panel's BOTTOM stays anchored at a fixed screen position (the conventional spot,
//! [`margin_y`](Self::margin_y) above `screen.bottom()`). The TOP edge animates over a
//! configurable duration via [`Context::animate_value_with_time`] toward the panel's
//! target height. By default that target is the natural content size, so the panel hugs
//! its content with no dead space and grows or shrinks smoothly when content changes between
//! frames. A caller can override with [`target_h`](Self::target_h) to pin a fixed height
//! (HUDs that scroll internally rather than resize).
//!
//! The natural content size is measured by rendering the user's content closure **twice per
//! frame**: once invisibly via [`Ui::set_invisible`] to capture this frame's height, then
//! again for real at the correctly-anchored position. Two passes are what let the bottom
//! edge stay rock-solid on the very frame content size changes; a single-pass design with
//! stale-measurement positioning always lags by one frame at transitions and the user
//! perceives the lag as flicker. The measure pass disables widgets, so interaction-gated
//! side effects (clicks, drags, slider edits) only fire in the visible pass.
//!
//! Content is rendered inside an internal [`ScrollArea`](egui::ScrollArea) so that
//! mid-transition, when the animated height is transiently smaller than the natural content,
//! the TOP scrolls out of view while the bottom (always-visible footer) stays on screen.
//!
//! Render content in normal top-down order. The ScrollArea's bottom-anchored offset handles
//! clip-from-top, no layout reversal needed.
//!
//! ## Why not [`TopBottomPanel`](egui::TopBottomPanel)?
//!
//! `TopBottomPanel::bottom` is docked: it carves a strip out of the central area, which
//! forces the scene viewport to skip that strip. For games that render the scene full-window
//! with a floating HUD on top, that's the wrong shape; the HUD should float, not dock.
//!
//! ## Example
//!
//! ```ignore
//! rye_egui::BottomOverlay::new("game-hud")
//!     .width(area_w)
//!     .frame(my_frame)
//!     .show(ctx, |ui| {
//!         // Render in normal top-down order. The panel
//!         // auto-sizes to this content; if `self.expanded`
//!         // toggles, the panel smoothly resizes to match.
//!         if self.expanded {
//!             self.render_expanded_body(ui);
//!             ui.separator();
//!         }
//!         self.render_slider_strip(ui);
//!         self.render_status_bar(ui);
//!     });
//! ```

use egui::{Area, Context, Frame, Id, InnerResponse, Pos2, Ui};

/// A floating overlay anchored at the bottom-center of the screen, with smoothly-animated
/// size transitions to eliminate the single-frame jumps that egui's plain `Area` produces
/// when content size changes.
pub struct BottomOverlay {
    id: Id,
    margin_y: f32,
    target_h: Option<f32>,
    width: f32,
    transition_secs: f32,
    frame: Option<Frame>,
}

impl BottomOverlay {
    /// Construct an overlay with sensible defaults. `id_source` must be unique per-overlay
    /// across the app.
    pub fn new(id_source: impl std::hash::Hash) -> Self {
        Self {
            id: Id::new(id_source),
            margin_y: 16.0,
            target_h: None,
            width: 600.0,
            transition_secs: 0.18,
            frame: None,
        }
    }

    /// Pin the overlay to a fixed height instead of auto-sizing to content. Use for HUDs that
    /// should keep a constant size and scroll their content internally; omit for the typical
    /// "panel grows with its body" pattern.
    pub fn target_h(mut self, h: f32) -> Self {
        self.target_h = Some(h);
        self
    }

    /// Overlay width, in points.
    pub fn width(mut self, w: f32) -> Self {
        self.width = w;
        self
    }

    /// Pixel margin between the overlay's bottom edge and the screen's bottom edge.
    pub fn margin_y(mut self, m: f32) -> Self {
        self.margin_y = m;
        self
    }

    /// Animation duration for height transitions, in seconds.
    /// 0.18 reads as "snappy but visible" on a 60+ fps display.
    pub fn transition_secs(mut self, t: f32) -> Self {
        self.transition_secs = t;
        self
    }

    /// Optional [`Frame`](egui::Frame) for the overlay's visual styling (fill, stroke, corner
    /// radius, inner margin).
    pub fn frame(mut self, frame: Frame) -> Self {
        self.frame = Some(frame);
        self
    }

    /// Render the overlay. Returns the underlying `Area`'s [`InnerResponse`].
    ///
    /// `content` should render in normal top-down order (mode header / body / footer style).
    /// The overlay sizes to its content (or to a caller-pinned `target_h`); height transitions
    /// animate smoothly, and during shrinks the TOP is clipped via an internal `ScrollArea` so
    /// widgets rendered late (the footer) stay in view.
    ///
    /// `content` is called *twice per frame*: once invisibly to measure this frame's natural
    /// content height, then again for the actual paint at the correctly-anchored position.
    /// The measure pass uses [`Ui::set_invisible`], which disables widgets and skips painting,
    /// so interaction-gated side effects (button clicks, drag drops, slider edits) only fire
    /// in the visible pass. The two-pass shape is what lets the bottom edge stay anchored on
    /// the very frame content size changes; a single-pass design with stale-measurement
    /// positioning always lags by one frame at transitions and the user perceives the lag as
    /// flicker.
    pub fn show<R>(self, ctx: &Context, mut content: impl FnMut(&mut Ui) -> R) -> InnerResponse<R> {
        let screen = ctx.content_rect();
        let frame = self.frame.unwrap_or_default();

        // Pass 1: measure pass. Render content invisibly off-screen to capture this frame's
        // natural content height. `Ui::set_invisible` disables widgets (no interaction) and
        // skips painting; widgets still allocate space, so `min_rect` reflects the natural
        // laid-out size.
        let measure_id = self.id.with("measure-area");
        let measure_resp = Area::new(measure_id)
            .order(egui::Order::Background)
            .interactable(false)
            .fixed_pos(MEASURE_PASS_POS)
            .show(ctx, |ui| {
                ui.set_invisible();
                ui.set_min_width(self.width);
                ui.set_max_width(self.width);
                frame.show(ui, |ui| {
                    content(ui);
                });
            });
        let natural_h = measure_resp.response.rect.height();

        // Animate the displayed height toward the target. Auto-size mode uses this frame's
        // natural height; pinned mode uses the caller-supplied target_h.
        let target = self.target_h.unwrap_or(natural_h);
        let smooth_h =
            ctx.animate_value_with_time(self.id.with("smooth_h"), target, self.transition_secs);

        // Position so the overlay's BOTTOM edge is at `screen.bottom() - margin_y` regardless
        // of `smooth_h`. Top moves with the animation; bottom stays fixed.
        let area_x = screen.center().x - self.width / 2.0;
        let area_y = screen.bottom() - self.margin_y - smooth_h;

        // Pass 2: visible paint at the correct anchored position. Inner ScrollArea's offset is
        // computed from THIS frame's natural_h so the content's bottom always sits at the
        // viewport's bottom, even on the very frame content size changes.
        Area::new(self.id)
            .fixed_pos(Pos2::new(area_x, area_y))
            .constrain(false)
            // Don't let the user drag the panel by clicking its background; its position is
            // fully derived from `screen.bottom() - margin_y - smooth_h` each frame and any
            // drag offset would be overwritten next frame anyway, producing a
            // "pickup-then-snap-back" jitter.
            .movable(false)
            .show(ctx, |ui| {
                ui.set_min_width(self.width);
                ui.set_max_width(self.width);
                // Pin outer ui height to the animated value. Without this, egui's `Area` keeps
                // `state.size` sticky and the inner ScrollArea's available space round-trips
                // through itself, never growing past the initial value.
                ui.set_min_height(smooth_h);
                ui.set_max_height(smooth_h);
                frame
                    .show(ui, |ui| {
                        // Scroll offset = how far past the visible viewport the natural
                        // content is. Frame inner_margin contributes to both `natural_h`
                        // and the outer pinned height equally, so it cancels out of
                        // `(natural_h - smooth_h)`.
                        let scroll_offset = (natural_h - smooth_h).max(0.0);
                        egui::ScrollArea::vertical()
                            .auto_shrink([false, false])
                            .vertical_scroll_offset(scroll_offset)
                            .scroll_bar_visibility(
                                egui::scroll_area::ScrollBarVisibility::AlwaysHidden,
                            )
                            .id_salt(self.id.with("scroll"))
                            .show(ui, |ui| content(ui))
                            .inner
                    })
                    .inner
            })
    }
}

/// Fixed-position anchor for the measure-pass `Area`. We place it far enough off-screen that
/// egui's clipping treats it as fully outside the viewport and skips painting; the value just
/// needs to be safely below any plausible `screen_rect.min`. The measure pass's
/// `set_invisible` already prevents painting, but the off-screen position also prevents the
/// area from ever participating in cursor hit-tests.
const MEASURE_PASS_POS: Pos2 = Pos2::new(-99_999.0, -99_999.0);

#[cfg(test)]
mod tests {
    use super::*;

    /// Run the overlay through `n` headless egui frames so animation can settle, returning
    /// the final response rect. egui's `Context::run` advances input + animation state;
    /// 30 frames at the default tick is plenty for a 0.18s transition to converge.
    fn measure(target_h: f32, content_lines: usize) -> egui::Rect {
        let ctx = egui::Context::default();
        let mut rect = egui::Rect::NOTHING;
        for _ in 0..30 {
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                let resp = BottomOverlay::new("test-overlay")
                    .target_h(target_h)
                    .width(400.0)
                    .show(ctx, |ui| {
                        for i in 0..content_lines {
                            ui.label(format!("line {i}"));
                        }
                    });
                rect = resp.response.rect;
            });
        }
        rect
    }

    /// `show()`'s response rect should be at least `target_h` tall once animation has settled.
    /// If it's not, animation isn't driving the panel size at all; the regression where
    /// expanding produces no visible change.
    #[test]
    fn response_height_reaches_target_h() {
        let rect = measure(200.0, 3);
        assert!(
            rect.height() >= 200.0,
            "expected response height ≥ 200 after settling, got {}",
            rect.height()
        );
    }

    /// Different `target_h` values must produce visibly different panel heights. This guards
    /// against a "stuck at default height" bug where `target_h` gets ignored.
    #[test]
    fn target_h_drives_height() {
        let small = measure(80.0, 3);
        let big = measure(240.0, 3);
        assert!(
            big.height() > small.height() + 100.0,
            "big target (240) should produce a much taller panel than small (80): \
             {} vs {}",
            big.height(),
            small.height()
        );
    }

    /// The content closure runs twice per frame (measure pass + visible pass), so over N
    /// frames it should be invoked exactly 2 * N times. The measure pass is what lets the
    /// overlay anchor its bottom on a growth frame, so this double-invocation is by design;
    /// callers should keep non-interaction side effects idempotent / cheap.
    #[test]
    fn content_closure_runs_twice_per_frame() {
        let ctx = egui::Context::default();
        let mut count = 0;
        for _ in 0..7 {
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                BottomOverlay::new("test-overlay")
                    .target_h(150.0)
                    .width(400.0)
                    .show(ctx, |_ui| {
                        count += 1;
                    });
            });
        }
        assert_eq!(
            count, 14,
            "content closure runs twice per frame (measure + visible passes); \
             expected 2*7=14 calls over 7 frames, got {count}"
        );
    }

    /// When the overlay is at a large `target_h`, the rate-row-style widgets (rendered LAST,
    /// conventionally at the bottom) must have a y-position that's actually inside the
    /// response rect. Catches a regression where ScrollArea + stick_to_bottom would scroll
    /// widgets off-screen entirely.
    #[test]
    fn last_widget_visible_when_large_target_h() {
        let ctx = egui::Context::default();
        let mut last_widget_rect = egui::Rect::NOTHING;
        let mut overlay_rect = egui::Rect::NOTHING;
        for _ in 0..30 {
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                let resp = BottomOverlay::new("test-overlay")
                    .target_h(300.0)
                    .width(400.0)
                    .show(ctx, |ui| {
                        ui.label("first");
                        ui.label("middle");
                        last_widget_rect = ui.label("last").rect;
                    });
                overlay_rect = resp.response.rect;
            });
        }
        assert!(
            overlay_rect.intersects(last_widget_rect),
            "the last widget's rect ({last_widget_rect:?}) should intersect the \
             overlay's rect ({overlay_rect:?}); i.e., the widget is visible \
             inside the panel"
        );
    }

    /// Sanity check: egui's `animate_value_with_time` should progress toward a new target
    /// across frames in a headless `Context::run` loop. If this fails, the test harness isn't
    /// driving time forward correctly and the higher-level `BottomOverlay` tests can't be
    /// trusted.
    #[test]
    fn animate_value_progresses_across_frames() {
        let ctx = egui::Context::default();
        let mut last = 0.0;
        // Phase 1: target = 80, 30 frames. Should converge to 80.
        for _ in 0..30 {
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                last = ctx.animate_value_with_time(egui::Id::new("v"), 80.0, 0.18);
            });
        }
        let phase1 = last;
        // Phase 2: target = 220, 30 frames. Should converge to ~220.
        for _ in 0..30 {
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                last = ctx.animate_value_with_time(egui::Id::new("v"), 220.0, 0.18);
            });
        }
        let phase2 = last;
        assert!(
            phase2 > phase1 + 100.0,
            "animate_value_with_time should progress from {phase1:.2} to ~220 \
             across phase 2's frames, got {phase2:.2}"
        );
    }

    /// The "chevron toggles expand/collapse" pattern: `target_h` is small + body conditionally
    /// hidden (collapsed), then `target_h` grows + body conditionally rendered (expanded).
    /// After settling in the expanded state, the body's widget rects MUST intersect the
    /// overlay's rect; i.e., the body actually shows up to the user, not just renders into some
    /// clipped void.
    ///
    /// This is the test for the regression where expanding a `BottomOverlay` produces a
    /// visibly larger panel but with the same content as the collapsed state; body widgets get
    /// rendered (the closure runs) but are clipped out of view.
    #[test]
    fn expand_toggle_makes_body_visible() {
        let ctx = egui::Context::default();

        // Phase 1: 30 frames in collapsed state.
        let mut collapsed_overlay = egui::Rect::NOTHING;
        let mut collapsed_body_rendered = false;
        for _ in 0..30 {
            collapsed_body_rendered = false;
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                let resp = BottomOverlay::new("toggle-test")
                    .target_h(80.0)
                    .width(400.0)
                    .show(ctx, |ui| {
                        let expanded = false;
                        if expanded {
                            ui.label("body content");
                            collapsed_body_rendered = true;
                        }
                        ui.label("footer 1");
                        ui.label("footer 2");
                    });
                collapsed_overlay = resp.response.rect;
            });
        }

        // Phase 2: 30 frames in expanded state.
        let mut expanded_overlay = egui::Rect::NOTHING;
        let mut expanded_body_rect = egui::Rect::NOTHING;
        let mut expanded_body_rendered = false;
        for _ in 0..30 {
            expanded_body_rendered = false;
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                let resp = BottomOverlay::new("toggle-test")
                    .target_h(220.0)
                    .width(400.0)
                    .show(ctx, |ui| {
                        let expanded = true;
                        if expanded {
                            let r = ui.label("body content");
                            expanded_body_rect = r.rect;
                            expanded_body_rendered = true;
                        }
                        ui.label("footer 1");
                        ui.label("footer 2");
                    });
                expanded_overlay = resp.response.rect;
            });
        }

        assert!(
            !collapsed_body_rendered,
            "body must not render in collapsed state"
        );
        assert!(expanded_body_rendered, "body must render in expanded state");
        assert!(
            expanded_overlay.height() > collapsed_overlay.height() + 50.0,
            "expanded overlay height ({}) should be much larger than collapsed ({})",
            expanded_overlay.height(),
            collapsed_overlay.height()
        );
        assert!(
            expanded_overlay.intersects(expanded_body_rect),
            "expanded body rect ({expanded_body_rect:?}) should intersect overlay \
             rect ({expanded_overlay:?}); i.e., the body is visible inside the panel"
        );
    }

    /// Auto-size mode (no `target_h` set): the panel must hug its content with no dead space
    /// after settling. Render N labels, settle the animation, then assert overlay height
    /// matches natural content height to within a few points (frame margin + scroll-area
    /// padding).
    #[test]
    fn auto_size_hugs_content() {
        let ctx = egui::Context::default();
        let mut overlay_rect = egui::Rect::NOTHING;
        let mut content_natural_h = 0.0;
        for _ in 0..30 {
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                let resp = BottomOverlay::new("auto-size-test")
                    .width(400.0)
                    .show(ctx, |ui| {
                        let r = ui.scope(|ui| {
                            for i in 0..5 {
                                ui.label(format!("line {i}"));
                            }
                        });
                        content_natural_h = r.response.rect.height();
                    });
                overlay_rect = resp.response.rect;
            });
        }
        let drift = (overlay_rect.height() - content_natural_h).abs();
        assert!(
            drift < 24.0,
            "auto-sized panel ({:.1}) should match content height ({:.1}) within \
             frame padding; drift={:.1}",
            overlay_rect.height(),
            content_natural_h,
            drift
        );
    }

    /// Auto-size with state-driven content change: collapse to expand should grow the panel;
    /// expand to collapse should shrink it. Both transitions converge to a panel that hugs the
    /// new content (no dead space at rest).
    #[test]
    fn auto_size_responds_to_content_change() {
        let ctx = egui::Context::default();
        let mut h_collapsed = 0.0;
        for _ in 0..30 {
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                let resp = BottomOverlay::new("dynamic-test")
                    .width(400.0)
                    .show(ctx, |ui| {
                        ui.label("footer 1");
                        ui.label("footer 2");
                    });
                h_collapsed = resp.response.rect.height();
            });
        }
        let mut h_expanded = 0.0;
        for _ in 0..30 {
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                let resp = BottomOverlay::new("dynamic-test")
                    .width(400.0)
                    .show(ctx, |ui| {
                        for i in 0..6 {
                            ui.label(format!("body line {i}"));
                        }
                        ui.separator();
                        ui.label("footer 1");
                        ui.label("footer 2");
                    });
                h_expanded = resp.response.rect.height();
            });
        }
        assert!(
            h_expanded > h_collapsed + 50.0,
            "expanded auto-sized panel ({h_expanded:.1}) should be much taller \
             than collapsed ({h_collapsed:.1})"
        );
    }

    /// Regression test for the "panel flickers low for one frame when content grows" bug. The
    /// setup mimics the user-reported scenario: settle the overlay in collapsed state
    /// (footer-only content), then on the next frame suddenly render expanded content (body +
    /// footer). On THAT first frame, the footer's rendered rect must still be inside the
    /// overlay's visible rect; not scrolled off the bottom because the inner ScrollArea was
    /// still using last frame's offset.
    #[test]
    fn footer_stays_visible_on_first_growth_frame() {
        let ctx = egui::Context::default();
        // Phase 1: settle in collapsed state (footer only).
        for _ in 0..30 {
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                BottomOverlay::new("growth-test")
                    .width(400.0)
                    .show(ctx, |ui| {
                        ui.label("footer 1");
                        ui.label("footer 2");
                    });
            });
        }
        // Phase 2: ONE frame with sudden growth; body added on top.
        let mut footer_rect = egui::Rect::NOTHING;
        let mut overlay_rect = egui::Rect::NOTHING;
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            let resp = BottomOverlay::new("growth-test")
                .width(400.0)
                .show(ctx, |ui| {
                    for i in 0..6 {
                        ui.label(format!("body line {i}"));
                    }
                    ui.separator();
                    ui.label("footer 1");
                    footer_rect = ui.label("footer 2").rect;
                });
            overlay_rect = resp.response.rect;
        });
        assert!(
            overlay_rect.intersects(footer_rect),
            "on the first frame after content grew, footer rect ({footer_rect:?}) \
             must still intersect overlay rect ({overlay_rect:?}); otherwise the \
             ScrollArea's stale offset is showing the top of the new content where \
             the footer should be"
        );
    }

    /// Mimics polytope_playground's pattern: an outer `BottomOverlay` containing a body section
    /// (with its own inner horizontal `ScrollArea`), then sliders, then a footer. Verifies
    /// that when the overlay is at a `target_h` smaller than the natural content height, the
    /// FOOTER widgets (rendered last) stay inside the overlay rect; which is what the
    /// stick_to_bottom-anchored ScrollArea is supposed to guarantee.
    #[test]
    fn polytope_like_pattern_keeps_footer_visible_under_overflow() {
        let ctx = egui::Context::default();
        let mut footer_rect = egui::Rect::NOTHING;
        let mut overlay_rect = egui::Rect::NOTHING;
        // target_h (130) < natural content height (the body alone pushes past 130 once the
        // inner row's allocated). Tests the stick_to_bottom path.
        for _ in 0..30 {
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                let resp = BottomOverlay::new("nested-test")
                    .target_h(130.0)
                    .width(800.0)
                    .show(ctx, |ui| {
                        // Body: mode tabs + checkboxes + nested
                        // horizontal scroll area + footer label.
                        ui.horizontal(|ui| {
                            ui.label("Active set");
                            ui.label("Composer");
                        });
                        ui.horizontal(|ui| {
                            for plane in ["xy", "xz", "xw", "yz", "yw", "zw"] {
                                ui.label(plane);
                            }
                        });
                        ui.separator();
                        ui.label("Shapes (drop a card onto another to insert there)");
                        egui::ScrollArea::horizontal()
                            .auto_shrink([false, true])
                            .id_salt("inner-shapes")
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    for shape in ["24-cell", "5-cell", "16-cell", "8-cell"] {
                                        ui.label(shape);
                                    }
                                });
                            });
                        ui.separator();
                        // Sliders (mid)
                        ui.label("w slider");
                        ui.label("t slider");
                        // Footer (LAST, must stay visible)
                        footer_rect = ui.label("rate row").rect;
                    });
                overlay_rect = resp.response.rect;
            });
        }
        assert!(
            overlay_rect.intersects(footer_rect),
            "footer rect ({footer_rect:?}) should intersect overlay rect \
             ({overlay_rect:?}); stick_to_bottom should keep the last widget \
             on screen even when content overflows the viewport"
        );
    }
}
