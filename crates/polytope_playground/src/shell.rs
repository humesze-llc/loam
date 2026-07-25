//! Scene shell: one shared menu bar with a Demo switcher, boot selection
//! via `--scene=<slug>` / `?scene=<slug>`, and an embed mode
//! (`--embed=1` / `?embed=1`) that hides the bar for page embeds.

use anyhow::Result;
use loam_app::{args::Args, egui, App, FrameCtx, SetupCtx};
use loam_math::EuclideanR3;
use loam_render::device::RenderDevice;

pub(crate) trait Scene {
    fn space(&self) -> &EuclideanR3;
    /// Contributions to the shared menu bar, rendered after the Demo menu.
    fn menus(&mut self, ui: &mut egui::Ui);
    fn update(&mut self, ctx: &mut FrameCtx<'_>);
    fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>);
    fn on_key(
        &mut self,
        code: winit::keyboard::KeyCode,
        state: winit::event::ElementState,
        ctx: &mut FrameCtx<'_>,
    );
    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()>;
    fn title(&self, fps: f32) -> std::borrow::Cow<'static, str>;
}

pub(crate) struct SceneEntry {
    pub slug: &'static str,
    pub label: &'static str,
    pub build: fn(&mut SetupCtx<'_>) -> Result<Box<dyn Scene>>,
}

pub(crate) const SCENES: &[SceneEntry] = &[SceneEntry {
    slug: "rotate",
    label: "Rotate polytopes",
    build: |ctx| Ok(Box::new(crate::RotateScene::new(ctx)?)),
}];

pub(crate) struct ShellApp {
    /// All scenes are built at setup: `SetupCtx` (shader db, watcher) is not
    /// reachable after `App::setup`, so switching selects among live
    /// instances rather than constructing on demand.
    scenes: Vec<Box<dyn Scene>>,
    active: usize,
    /// Embed mode: no menu bar; the page chrome owns navigation.
    embed: bool,
    capture_panel: loam_app::capture::CapturePanel,
    perf: loam_app::trace::PerfOverlay,
}

/// (boot scene index, embed). Unknown slugs fall back to scene 0.
fn resolve_boot(args: &Args) -> (usize, bool) {
    let active = match args.get("scene") {
        None => 0,
        Some(slug) => SCENES
            .iter()
            .position(|s| s.slug == slug)
            .unwrap_or_else(|| {
                tracing::warn!("unknown scene '{slug}'; defaulting to '{}'", SCENES[0].slug);
                0
            }),
    };
    let embed = args.get("embed").is_some_and(|v| v != "0" && v != "false");
    (active, embed)
}

impl App for ShellApp {
    type Space = EuclideanR3;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let (active, embed) = resolve_boot(&Args::current());
        let scenes = SCENES
            .iter()
            .map(|entry| (entry.build)(ctx))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            scenes,
            active,
            embed,
            capture_panel: loam_app::capture::CapturePanel::new(),
            perf: loam_app::trace::PerfOverlay::new(),
        })
    }

    fn space(&self) -> &EuclideanR3 {
        self.scenes[self.active].space()
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        self.scenes[self.active].update(ctx);
    }

    fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
        if !self.embed {
            // Bar renders first so its docked space is reserved and
            // `content_rect()` reflects the area below it.
            let Self { scenes, active, .. } = self;
            egui::TopBottomPanel::top("shell-menu-bar").show(ctx, |ui| {
                egui::MenuBar::new().ui(ui, |ui| {
                    ui.menu_button("Demo", |ui| {
                        for (i, entry) in SCENES.iter().enumerate() {
                            if ui.selectable_label(*active == i, entry.label).clicked() {
                                *active = i;
                                ui.close_kind(egui::UiKind::Menu);
                            }
                        }
                    });
                    scenes[*active].menus(ui);
                });
            });
        }
        self.scenes[self.active].ui(ctx, frame);
        self.capture_panel.show(ctx);
        self.perf.show(ctx);
    }

    fn on_key(
        &mut self,
        code: winit::keyboard::KeyCode,
        state: winit::event::ElementState,
        ctx: &mut FrameCtx<'_>,
    ) {
        self.scenes[self.active].on_key(code, state, ctx);
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        self.scenes[self.active].render(rd, view)
    }

    fn title(&self, fps: f32) -> std::borrow::Cow<'static, str> {
        self.scenes[self.active].title(fps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boot_defaults_to_first_scene_without_params() {
        assert_eq!(
            resolve_boot(&Args::from_pairs::<[(&str, &str); 0], _, _>([])),
            (0, false)
        );
    }

    #[test]
    fn boot_resolves_known_slug_and_falls_back_on_unknown() {
        assert_eq!(resolve_boot(&Args::from_pairs([("scene", "rotate")])).0, 0);
        assert_eq!(resolve_boot(&Args::from_pairs([("scene", "nope")])).0, 0);
    }

    #[test]
    fn embed_is_truthy_except_zero_and_false() {
        assert!(resolve_boot(&Args::from_pairs([("embed", "1")])).1);
        assert!(resolve_boot(&Args::from_pairs([("embed", "true")])).1);
        assert!(!resolve_boot(&Args::from_pairs([("embed", "0")])).1);
        assert!(!resolve_boot(&Args::from_pairs([("embed", "false")])).1);
    }
}
