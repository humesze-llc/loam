//! Floating bottom-anchored overlay panel with flicker-free size
//! transitions.
//!
//! ## The problem this solves
//!
//! egui's [`Area`](egui::Area) recomputes its position from
//! content size each frame. When content size changes drastically
//! between frames — typically because some app state toggled and a
//! big chunk of UI appeared or disappeared — the area's pivot
//! recomputes in a single frame, and the in-between rendering
//! reads as the overlay "flickering" or "disappearing" briefly.
//!
//! For floating bottom HUDs in games, this happens whenever the
//! HUD expands a panel, switches modes, opens an inventory, etc.
//! Single-frame jumps are unacceptable polish-wise.
//!
//! ## How this widget fixes it
//!
//! The panel's BOTTOM stays anchored at a fixed screen position
//! (the conventional spot, [`margin_y`](Self::margin_y) above
//! `screen.bottom()`). The TOP edge animates over a configurable
//! duration via [`Context::animate_value_with_time`] toward the
//! panel's target height — by default, the natural content size
//! captured from the previous frame's render (so the panel hugs
//! its content with no dead space, and shrinks/grows smoothly
//! when content changes between frames). A caller can override
//! with [`target_h`](Self::target_h) to pin a fixed height
//! (HUDs that scroll internally rather than resize).
//!
//! Content is rendered inside an internal
//! [`ScrollArea`](egui::ScrollArea) configured with
//! `stick_to_bottom(true)`, so when the animated height is
//! transiently smaller than the content (mid-transition), the
//! TOP scrolls out of view while the bottom (always-visible
//! footer) stays on screen. After settling, content fits
//! exactly inside the panel.
//!
//! Render content in normal top-down order — the ScrollArea's
//! bottom-stick handles the clip-from-top behavior, no layout
//! reversal needed.
//!
//! ## Why not [`TopBottomPanel`](egui::TopBottomPanel)?
//!
//! `TopBottomPanel::bottom` is docked — it carves a strip out of
//! the central area, which forces the scene viewport to skip that
//! strip. For games that render the scene full-window with a
//! floating HUD on top, that's the wrong shape; the HUD should
//! float, not dock.
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

/// A floating overlay anchored at the bottom-center of the screen,
/// with smoothly-animated size transitions to eliminate the
/// single-frame jumps that egui's plain `Area` produces when
/// content size changes.
pub struct BottomOverlay {
    id: Id,
    margin_y: f32,
    target_h: Option<f32>,
    width: f32,
    transition_secs: f32,
    frame: Option<Frame>,
}

impl BottomOverlay {
    /// Construct an overlay with sensible defaults. `id_source`
    /// must be unique per-overlay across the app.
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

    /// Pin the overlay to a fixed height instead of auto-sizing
    /// to content. Use for HUDs that should keep a constant size
    /// and scroll their content internally; omit for the typical
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

    /// Pixel margin between the overlay's bottom edge and the
    /// screen's bottom edge.
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

    /// Optional [`Frame`](egui::Frame) for the overlay's visual
    /// styling (fill, stroke, corner radius, inner margin).
    pub fn frame(mut self, frame: Frame) -> Self {
        self.frame = Some(frame);
        self
    }

    /// Render the overlay. Returns the underlying `Area`'s
    /// [`InnerResponse`].
    ///
    /// `content` should render in normal top-down order (mode
    /// header / body / footer style); the overlay clips from the
    /// TOP during shrinks via an internal `ScrollArea` with
    /// `stick_to_bottom`, so widgets rendered late (the footer)
    /// stay in view throughout collapse animations. After the
    /// transition settles, the panel hugs its content exactly.
    pub fn show<R>(self, ctx: &Context, content: impl FnOnce(&mut Ui) -> R) -> InnerResponse<R> {
        let screen = ctx.content_rect();

        // Target height for this frame: caller's pinned value if
        // set, otherwise last frame's measured natural content
        // height (the panel hugs its content with no dead space).
        // Default 60.0 covers the very first frame before any
        // measurement exists; subsequent frames overwrite this.
        let measured_id = self.id.with("measured_h");
        let target = self.target_h.unwrap_or_else(|| {
            ctx.memory(|m| m.data.get_temp::<f32>(measured_id))
                .unwrap_or(60.0)
        });

        // Animate the displayed height toward the target. egui's
        // `animate_value_with_time` smoothly interpolates frame-to-
        // frame; on the frame where target changes, this returns
        // the previous frame's value, then lerps over
        // `transition_secs`.
        let smooth_h =
            ctx.animate_value_with_time(self.id.with("smooth_h"), target, self.transition_secs);

        // Position so the overlay's BOTTOM edge is at
        // `screen.bottom() - margin_y` regardless of `smooth_h`.
        // Top moves with the animation; bottom stays fixed.
        let area_x = screen.center().x - self.width / 2.0;
        let area_y = screen.bottom() - self.margin_y - smooth_h;

        let frame = self.frame.unwrap_or_default();
        let frame_margin = frame.inner_margin.top as f32 + frame.inner_margin.bottom as f32;

        Area::new(self.id)
            .fixed_pos(Pos2::new(area_x, area_y))
            .constrain(false)
            .show(ctx, |ui| {
                ui.set_min_width(self.width);
                ui.set_max_width(self.width);
                // Pin the outer ui's height to the animated value.
                // This is load-bearing: egui's `Area` keeps its
                // `state.size` sticky from the previous frame, and
                // a child `ScrollArea`'s outer size clamps to the
                // *available* space (which equals last frame's
                // state.size). Without explicit `set_*_height`, the
                // size round-trips through itself and never grows
                // past the initial value when target increases.
                // Locking via `set_min_height`/`set_max_height`
                // makes the animated value authoritative for the
                // ui's height, so `state.size` correctly tracks the
                // animation each frame.
                ui.set_min_height(smooth_h);
                ui.set_max_height(smooth_h);
                frame
                    .show(ui, |ui| {
                        // ScrollArea with `stick_to_bottom(true)`
                        // anchors the bottom of the content to the
                        // bottom of the viewport. The ScrollArea
                        // fills the locked-height ui; when content
                        // exceeds that, the TOP scrolls out of view
                        // rather than the bottom — always-visible
                        // controls (footer) stay on screen during
                        // collapse animations.
                        let so = egui::ScrollArea::vertical()
                            .auto_shrink([false, false])
                            .stick_to_bottom(true)
                            .scroll_bar_visibility(
                                egui::scroll_area::ScrollBarVisibility::AlwaysHidden,
                            )
                            .id_salt(self.id.with("scroll"))
                            .show(ui, content);
                        // Capture the natural content size for next
                        // frame's auto-size target. ScrollArea's
                        // `content_size` is the unclamped natural
                        // size of the inner content; add the frame's
                        // inner_margin to get the panel's total
                        // height. Skip when caller pinned target_h
                        // (no point feeding the auto-size loop).
                        if self.target_h.is_none() {
                            let natural = so.content_size.y + frame_margin;
                            ctx.memory_mut(|m| m.data.insert_temp(measured_id, natural));
                        }
                        so.inner
                    })
                    .inner
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run the overlay through `n` headless egui frames so animation
    /// can settle, returning the final response rect. egui's
    /// `Context::run` advances input + animation state; 30 frames at
    /// the default tick is plenty for a 0.18s transition to converge.
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

    /// `show()`'s response rect should be at least `target_h` tall
    /// once animation has settled. If it's not, animation isn't
    /// driving the panel size at all — the regression where
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

    /// Different `target_h` values must produce visibly different
    /// panel heights. This guards against a "stuck at default
    /// height" bug where `target_h` gets ignored.
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

    /// The content closure must be called every frame `show` is
    /// invoked (egui is immediate-mode; if the closure isn't run,
    /// the content widgets never exist).
    #[test]
    fn content_closure_runs_each_frame() {
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
        assert_eq!(count, 7, "content closure should run once per frame");
    }

    /// When the overlay is at a large `target_h`, the rate-row-style
    /// widgets (rendered LAST, conventionally at the bottom) must
    /// have a y-position that's actually inside the response rect.
    /// Catches a regression where ScrollArea + stick_to_bottom would
    /// scroll widgets off-screen entirely.
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
             overlay's rect ({overlay_rect:?}) — i.e., the widget is visible \
             inside the panel"
        );
    }

    /// Sanity check: egui's `animate_value_with_time` should
    /// progress toward a new target across frames in a headless
    /// `Context::run` loop. If this fails, the test harness isn't
    /// driving time forward correctly and the higher-level
    /// `BottomOverlay` tests can't be trusted.
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

    /// The "chevron toggles expand/collapse" pattern:
    /// `target_h` is small + body conditionally hidden (collapsed),
    /// then `target_h` grows + body conditionally rendered
    /// (expanded). After settling in the expanded state, the body's
    /// widget rects MUST intersect the overlay's rect — i.e., the
    /// body actually shows up to the user, not just renders into
    /// some clipped void.
    ///
    /// This is the test for the regression where expanding a
    /// `BottomOverlay` produces a visibly larger panel but with
    /// the same content as the collapsed state — body widgets get
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
             rect ({expanded_overlay:?}) — i.e., the body is visible inside the panel"
        );
    }

    /// Auto-size mode (no `target_h` set): the panel must hug its
    /// content with no dead space after settling. Render N labels,
    /// settle the animation, then assert overlay height matches
    /// natural content height to within a few points (frame margin
    /// + scroll-area padding).
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

    /// Auto-size with state-driven content change: collapse →
    /// expand should grow the panel; expand → collapse should
    /// shrink it. Both transitions converge to a panel that hugs
    /// the new content (no dead space at rest).
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

    /// Mimics polytope_smoke's pattern: an outer `BottomOverlay`
    /// containing a body section (with its own inner horizontal
    /// `ScrollArea`), then sliders, then a footer. Verifies that
    /// when the overlay is at a `target_h` smaller than the natural
    /// content height, the FOOTER widgets (rendered last) stay
    /// inside the overlay rect — which is what the
    /// stick_to_bottom-anchored ScrollArea is supposed to guarantee.
    #[test]
    fn polytope_like_pattern_keeps_footer_visible_under_overflow() {
        let ctx = egui::Context::default();
        let mut footer_rect = egui::Rect::NOTHING;
        let mut overlay_rect = egui::Rect::NOTHING;
        // target_h (130) < natural content height (the body alone
        // pushes past 130 once the inner row's allocated). Tests
        // the stick_to_bottom path.
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
                        // Footer (LAST → must stay visible)
                        footer_rect = ui.label("rate row").rect;
                    });
                overlay_rect = resp.response.rect;
            });
        }
        assert!(
            overlay_rect.intersects(footer_rect),
            "footer rect ({footer_rect:?}) should intersect overlay rect \
             ({overlay_rect:?}) — stick_to_bottom should keep the last widget \
             on screen even when content overflows the viewport"
        );
    }
}
