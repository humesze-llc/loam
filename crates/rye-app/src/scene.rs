//! A host-agnostic scene: the unit a future Polytope-Playground host can switch
//! between, and that [`run_scene`] also runs standalone (incl. wasm). Unlike
//! [`crate::App`], a `Scene` does not own the window or run loop and renders into
//! a caller-provided `view` + [`Viewport`], so the same impl serves both a
//! standalone window and an embedded panel.

use std::borrow::Cow;

use rye_math::EuclideanR3;
use rye_render::device::RenderDevice;
use rye_render::Viewport;

use crate::{egui, run, App, FrameCtx, RunConfig, SetupCtx};

pub trait Scene: 'static {
    fn new(ctx: &mut SetupCtx<'_>) -> anyhow::Result<Self>
    where
        Self: Sized;

    fn update(&mut self, _ctx: &mut FrameCtx<'_>) {}

    fn render(
        &mut self,
        _rd: &RenderDevice,
        _view: &wgpu::TextureView,
        _viewport: Viewport,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn ui(&mut self, _ctx: &egui::Context, _frame: &mut FrameCtx<'_>) {}

    fn title(&self) -> Cow<'static, str> {
        Cow::Borrowed("rye scene")
    }

    /// Named animated scalars for the offline harness CSV curve dump
    /// ([`crate::harness`]). Default empty; a demo overrides it to expose the
    /// values driving its animation so "the curve is wrong" can be told apart
    /// from "the drawing is wrong" without eyeballing every frame. Names must be
    /// stable across frames (they become CSV columns).
    fn debug_scalars(&self) -> Vec<(&'static str, f32)> {
        Vec::new()
    }
}

/// Run a single scene as a standalone app (its own window + run loop).
pub fn run_scene<S: Scene>(config: RunConfig) -> anyhow::Result<()> {
    run::<SceneHost<S>>(config)
}

struct SceneHost<S: Scene> {
    scene: S,
}

impl<S: Scene> App for SceneHost<S> {
    type Space = EuclideanR3;

    fn setup(ctx: &mut SetupCtx<'_>) -> anyhow::Result<Self> {
        Ok(Self {
            scene: S::new(ctx)?,
        })
    }

    fn space(&self) -> &EuclideanR3 {
        &EuclideanR3
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        self.scene.update(ctx);
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> anyhow::Result<()> {
        let cfg = &rd.surface_bundle.config;
        self.scene
            .render(rd, view, Viewport::full([cfg.width, cfg.height]))
    }

    fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
        self.scene.ui(ctx, frame);
    }

    fn title(&self, _fps: f32) -> Cow<'static, str> {
        self.scene.title()
    }
}
