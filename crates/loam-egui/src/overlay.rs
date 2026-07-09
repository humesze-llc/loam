//! Floating bottom-anchored overlay panel with flicker-free size transitions.
//!
//! egui's [`Area`](egui::Area) recomputes position from content size each frame,
//! so a large single-frame content change reads as a flicker. [`BottomOverlay`]
//! pins its BOTTOM edge at [`margin_y`](BottomOverlay::margin_y) above
//! `screen.bottom()` and animates the TOP toward the target height via
//! [`Context::animate_value_with_time`]. Default target is natural content size;
//! [`target_h`](BottomOverlay::target_h) pins a fixed height instead.
//!
//! The content closure runs twice per frame: once invisibly via
//! [`Ui::set_invisible`] to measure this frame's natural height, then for real at
//! the anchored position. The measure pass is what keeps the bottom edge solid on
//! the frame content size changes; a single pass would lag one frame and read as
//! flicker. `set_invisible` also gates interaction side effects to the visible pass.
//!
//! An internal [`ScrollArea`](egui::ScrollArea) clips the TOP during shrinks so the
//! always-visible footer stays on screen; render content in normal top-down order.
//!
//! [`TopBottomPanel`](egui::TopBottomPanel) is rejected because it docks (carves a
//! strip out of the central area); a game HUD over a full-window scene must float.
//!
//! ```ignore
//! loam_egui::BottomOverlay::new("game-hud")
//!     .width(area_w)
//!     .frame(my_frame)
//!     .show(ctx, |ui| {
//!         if self.expanded {
//!             self.render_expanded_body(ui);
//!             ui.separator();
//!         }
//!         self.render_slider_strip(ui);
//!         self.render_status_bar(ui);
//!     });
//! ```

use egui::{Area, Context, Frame, Id, InnerResponse, Pos2, Ui};

/// Floating overlay anchored at the bottom-center of the screen with animated
/// size transitions; see the module docs for the two-pass measure design.
pub struct BottomOverlay {
    id: Id,
    margin_y: f32,
    target_h: Option<f32>,
    width: f32,
    transition_secs: f32,
    frame: Option<Frame>,
}

impl BottomOverlay {
    /// Construct an overlay with defaults. `id_source` must be unique per-overlay.
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

    /// Pin a fixed height instead of auto-sizing to content; for HUDs that scroll
    /// their content internally rather than resize.
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

    /// Animation duration for height transitions, in seconds (default 0.18).
    pub fn transition_secs(mut self, t: f32) -> Self {
        self.transition_secs = t;
        self
    }

    /// Optional [`Frame`](egui::Frame) for the overlay's visual styling.
    pub fn frame(mut self, frame: Frame) -> Self {
        self.frame = Some(frame);
        self
    }

    /// Render the overlay; returns the underlying `Area`'s [`InnerResponse`].
    ///
    /// `content` renders in normal top-down order and is called twice per frame
    /// (measure pass + visible pass); keep non-interaction side effects idempotent.
    /// See the module docs for the two-pass design and `set_invisible` gating.
    pub fn show<R>(self, ctx: &Context, mut content: impl FnMut(&mut Ui) -> R) -> InnerResponse<R> {
        let screen = ctx.content_rect();
        let frame = self.frame.unwrap_or_default();

        // Measure pass: render invisibly off-screen so `min_rect` reflects the
        // natural laid-out size without painting or interaction.
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

        let target = self.target_h.unwrap_or(natural_h);
        let smooth_h =
            ctx.animate_value_with_time(self.id.with("smooth_h"), target, self.transition_secs);

        // Bottom edge fixed at `screen.bottom() - margin_y`; top moves with smooth_h.
        let area_x = screen.center().x - self.width / 2.0;
        let area_y = screen.bottom() - self.margin_y - smooth_h;

        // Visible pass at the anchored position; the inner ScrollArea offset uses
        // this frame's natural_h so the content bottom stays pinned.
        Area::new(self.id)
            .fixed_pos(Pos2::new(area_x, area_y))
            .constrain(false)
            // Position is derived every frame, so a drag offset would snap back
            // next frame as jitter; disable dragging.
            .movable(false)
            .show(ctx, |ui| {
                ui.set_min_width(self.width);
                ui.set_max_width(self.width);
                // Pin the outer height; otherwise `Area`'s sticky `state.size`
                // round-trips through the inner ScrollArea and never grows.
                ui.set_min_height(smooth_h);
                ui.set_max_height(smooth_h);
                frame
                    .show(ui, |ui| {
                        // How far the natural content overflows the viewport; frame
                        // inner_margin cancels out of `(natural_h - smooth_h)`.
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

/// Off-screen anchor for the measure-pass `Area`, safely below any plausible
/// `screen_rect.min` so it never participates in cursor hit-tests.
const MEASURE_PASS_POS: Pos2 = Pos2::new(-99_999.0, -99_999.0);

#[cfg(test)]
mod tests {
    use super::*;

    /// Run the overlay through 30 headless frames so the 0.18s animation settles,
    /// returning the final response rect.
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

    /// Settled response rect is at least `target_h` tall; guards the regression
    /// where animation never drives the panel size.
    #[test]
    fn response_height_reaches_target_h() {
        let rect = measure(200.0, 3);
        assert!(
            rect.height() >= 200.0,
            "expected response height ≥ 200 after settling, got {}",
            rect.height()
        );
    }

    /// Different `target_h` values produce visibly different heights; guards a
    /// "stuck at default height" bug where `target_h` is ignored.
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

    /// The content closure runs exactly twice per frame (measure + visible passes).
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

    /// At a large `target_h`, the last-rendered widget's rect stays inside the
    /// response rect; catches ScrollArea scrolling it off-screen.
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

    /// Harness sanity check: `animate_value_with_time` progresses toward a new
    /// target across headless frames, so the higher-level tests can be trusted.
    #[test]
    fn animate_value_progresses_across_frames() {
        let ctx = egui::Context::default();
        let mut last = 0.0;
        for _ in 0..30 {
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                last = ctx.animate_value_with_time(egui::Id::new("v"), 80.0, 0.18);
            });
        }
        let phase1 = last;
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

    /// Expand/collapse toggle: after settling expanded, the body's widget rects
    /// intersect the overlay rect. Guards the regression where expanding grows the
    /// panel but the body renders into a clipped void.
    #[test]
    fn expand_toggle_makes_body_visible() {
        let ctx = egui::Context::default();

        // Collapsed state.
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

        // Expanded state.
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

    /// Auto-size mode (no `target_h`): the settled panel height matches natural
    /// content height to within frame + scroll-area padding.
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

    /// Auto-size with a content change: more content settles to a taller panel
    /// that still hugs the new content.
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

    /// Regression: on the first frame content suddenly grows, the footer rect stays
    /// inside the overlay rect rather than scrolling off via a stale ScrollArea offset.
    #[test]
    fn footer_stays_visible_on_first_growth_frame() {
        let ctx = egui::Context::default();
        // Settle in collapsed state (footer only).
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
        // One frame with sudden growth; body added on top.
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

    /// Nested-content pattern (body with inner horizontal ScrollArea, sliders,
    /// footer) at a `target_h` smaller than natural height: the footer stays inside
    /// the overlay rect.
    #[test]
    fn polytope_like_pattern_keeps_footer_visible_under_overflow() {
        let ctx = egui::Context::default();
        let mut footer_rect = egui::Rect::NOTHING;
        let mut overlay_rect = egui::Rect::NOTHING;
        // target_h (130) < natural content height; tests the overflow path.
        for _ in 0..30 {
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                let resp = BottomOverlay::new("nested-test")
                    .target_h(130.0)
                    .width(800.0)
                    .show(ctx, |ui| {
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
                        ui.label("w slider");
                        ui.label("t slider");
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
