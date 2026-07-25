//! Scene shell: one shared menu bar with a Demo switcher, boot selection
//! via `--scene=<slug>` / `?scene=<slug>`, and an embed mode
//! (`--embed=1` / `?embed=1`) that hides the bar for page embeds. Embed
//! mode leaves the `scene` console command as the only in-app switcher.

use anyhow::{anyhow, Result};
use loam_app::{args::Args, egui, App, FrameCtx, SetupCtx};
use loam_egui::Console;
use loam_math::EuclideanR3;
use loam_render::device::RenderDevice;
use std::sync::Mutex;

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

/// Shell state a scene's console can reach. A console command's context is
/// the scene's own state, so it cannot see `ShellApp`; the shell publishes
/// `active` and drains `pending` around the scene's `ui`. Same publish/drain
/// shape as `loam_app::capture`'s request queue.
struct Switcher {
    active: usize,
    pending: Option<usize>,
}

static SWITCHER: Mutex<Switcher> = Mutex::new(Switcher {
    active: 0,
    pending: None,
});

fn with_switcher<R>(f: impl FnOnce(&mut Switcher) -> R) -> R {
    f(&mut SWITCHER.lock().expect("scene switcher poisoned"))
}

/// Registry index for `slug`, or `None` when no scene claims it.
fn scene_index(scenes: &[SceneEntry], slug: &str) -> Option<usize> {
    scenes.iter().position(|entry| entry.slug == slug)
}

/// Queue a switch to the scene named `slug`, applied after the current
/// frame's scene `ui` returns.
fn request_scene(slug: &str) -> Result<()> {
    let index = scene_index(SCENES, slug).ok_or_else(|| {
        let known = SCENES
            .iter()
            .map(|entry| entry.slug)
            .collect::<Vec<_>>()
            .join("|");
        anyhow!("unknown scene `{slug}` (try {known})")
    })?;
    with_switcher(|s| s.pending = Some(index));
    Ok(())
}

/// Register the `scene` command: bare lists the registry and marks the
/// active entry, `scene <slug>` switches. Free of the console's `Ctx`
/// because the shell, not the scene, owns the selection.
pub(crate) fn register_scene_command<Ctx: 'static>(console: &mut Console<Ctx>) {
    let slugs = SCENES.iter().map(|entry| entry.slug).collect::<Vec<_>>();
    console.register(
        loam_egui::cmd::<Ctx, _>(
            "scene",
            "list scenes (active marked `*`); `scene <slug>` switches, same slugs as --scene= / ?scene=",
            |args, _ctx: &mut Ctx, out| match args.first().copied() {
                None => {
                    let active = with_switcher(|s| s.active);
                    for (i, entry) in SCENES.iter().enumerate() {
                        let mark = if i == active { '*' } else { ' ' };
                        out.line(format!("{mark} {} - {}", entry.slug, entry.label));
                    }
                    Ok(())
                }
                Some(slug) => {
                    request_scene(slug)?;
                    out.line(format!("scene: switching to `{slug}`"));
                    Ok(())
                }
            },
        )
        .with_args(&[&slugs])
        .with_long_help(
            "Selects which scene the shell renders. The same slugs boot the demo\n\
             directly: `--scene=<slug>` natively, `?scene=<slug>` in the browser.\n\
             \n\
             `--embed=1` / `?embed=1` hides the shell menu bar for page embeds,\n\
             which leaves this command as the only in-app switcher.\n\
             \n\
             Bare `scene` lists every registered slug and marks the active one.",
        ),
    );
}

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

/// (boot scene index, embed). Unknown slugs fall back to scene 0. Takes the
/// registry rather than reading [`SCENES`] so the lookup is exercisable
/// against a multi-entry table.
fn resolve_boot(scenes: &[SceneEntry], args: &Args) -> (usize, bool) {
    let active = match args.get("scene") {
        None => 0,
        Some(slug) => scene_index(scenes, slug).unwrap_or_else(|| {
            tracing::warn!("unknown scene '{slug}'; defaulting to '{}'", scenes[0].slug);
            0
        }),
    };
    let embed = args.get("embed").is_some_and(|v| v != "0" && v != "false");
    (active, embed)
}

impl App for ShellApp {
    type Space = EuclideanR3;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let (active, embed) = resolve_boot(SCENES, &Args::current());
        let scenes = SCENES
            .iter()
            .map(|entry| (entry.build)(ctx))
            .collect::<Result<Vec<_>>>()?;
        with_switcher(|s| s.active = active);
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
            // Bar renders first so the scene's own windows see it in
            // `available_rect()`. `content_rect()` is the viewport minus OS
            // safe-area insets and never shrinks for a panel.
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
        // Drained after the scene's `ui` returns: the `scene` command runs
        // inside it, holding the borrow a switch would invalidate. The
        // republish also carries a menu-bar click back to the console.
        let active = self.active;
        self.active = with_switcher(|s| {
            s.active = s.pending.take().unwrap_or(active);
            s.active
        });
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

    /// Stand-in registry. [`SCENES`] holds a single entry, so every slug in it
    /// resolves to 0 and a lookup asserted against it is indistinguishable
    /// from no lookup at all.
    const REGISTRY: &[SceneEntry] = &[
        SceneEntry {
            slug: "first",
            label: "First",
            build: |_| unreachable!("registry fixture is never built"),
        },
        SceneEntry {
            slug: "second",
            label: "Second",
            build: |_| unreachable!("registry fixture is never built"),
        },
        SceneEntry {
            slug: "third",
            label: "Third",
            build: |_| unreachable!("registry fixture is never built"),
        },
    ];

    #[test]
    fn boot_defaults_to_first_scene_without_params() {
        assert_eq!(
            resolve_boot(REGISTRY, &Args::from_pairs::<[(&str, &str); 0], _, _>([])),
            (0, false)
        );
    }

    #[test]
    fn boot_slug_selects_its_own_registry_index() {
        for (index, entry) in REGISTRY.iter().enumerate() {
            assert_eq!(
                resolve_boot(REGISTRY, &Args::from_pairs([("scene", entry.slug)])).0,
                index,
                "slug '{}' should boot registry index {index}",
                entry.slug
            );
        }
    }

    #[test]
    fn boot_falls_back_to_first_scene_on_unknown_slug() {
        assert_eq!(
            resolve_boot(REGISTRY, &Args::from_pairs([("scene", "nope")])).0,
            0
        );
    }

    #[test]
    fn embed_is_truthy_except_zero_and_false() {
        assert!(resolve_boot(REGISTRY, &Args::from_pairs([("embed", "1")])).1);
        assert!(resolve_boot(REGISTRY, &Args::from_pairs([("embed", "true")])).1);
        assert!(!resolve_boot(REGISTRY, &Args::from_pairs([("embed", "0")])).1);
        assert!(!resolve_boot(REGISTRY, &Args::from_pairs([("embed", "false")])).1);
    }

    /// The `scene` command must reject a slug no scene claims instead of
    /// queueing a switch the shell would index with.
    #[test]
    fn scene_request_queues_known_slugs_and_rejects_unknown() {
        assert!(request_scene("nope").is_err());
        assert!(with_switcher(|s| s.pending).is_none());
        assert!(request_scene(SCENES[0].slug).is_ok());
        assert_eq!(with_switcher(|s| s.pending.take()), Some(0));
    }
}
