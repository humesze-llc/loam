//! Window surface + wgpu adapter/device acquisition.
//!
//! [`RenderDevice::new`] picks a high-performance adapter and an sRGB surface format when
//! available, then optionally allocates a multisampled color attachment matching the surface's size
//! and format. Resize is handled by [`RenderDevice::resize`].
//! [`RenderDevice::begin_frame`] returns the per-frame `(SurfaceTexture, TextureView)` pair the
//! render graph draws into; when MSAA is enabled, [`RenderDevice::msaa_view`] is the actual
//! render target and the swapchain view is used as the resolve target at the end of the egui paint
//! pass.

use anyhow::Result;
use std::sync::Arc;
use wgpu::*;
use winit::window::Window;

/// Surface + per-frame configuration. Owned by [`RenderDevice`]; held out as a struct so
/// resize-aware code can read the current size and format without poking at private fields.
pub struct SurfaceBundle {
    pub surface: Surface<'static>,
    pub config: SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
}

/// Multisampled color attachment paired with the surface. Allocated when the configured sample
/// count is > 1; [`MsaaTarget::view`] is the render target every scene + UI pass writes into,
/// with the swapchain view used as the resolve target.
pub struct MsaaTarget {
    // Held to keep the GPU allocation alive for the lifetime of `view` (which is a borrow into
    // this texture). Never read through the field itself.
    #[allow(dead_code)]
    texture: Texture,
    pub view: TextureView,
}

/// Offscreen render target paired with the surface; the texture + a default view of
/// it. Allocated when the surface format is non-sRGB (the browser-WebGPU case): the
/// scene + UI render into this texture in sRGB encoding (because the texture format
/// is `*UnormSrgb`), then the [`crate::composite::CompositeNode`] reads it +
/// gamma-encodes for write to the linear swapchain. Same shape as [`MsaaTarget`];
/// kept as a separate struct for the readability of code that branches on it.
pub struct OffscreenTarget {
    // Keeps the GPU allocation alive for the lifetime of `view`. Not read directly.
    #[allow(dead_code)]
    texture: Texture,
    pub view: TextureView,
}

/// All wgpu state the engine carries: shared `Instance`, the chosen `Adapter`, the logical
/// `Device`, the submission `Queue`, the current surface bundle, and an optional multisampled
/// color attachment matching the surface. One per app; cloning this is not supported.
pub struct RenderDevice {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub surface_bundle: SurfaceBundle,
    sample_count: u32,
    msaa_target: Option<MsaaTarget>,
    /// GPU timestamp query infrastructure. `Some` when the adapter advertised
    /// `Features::TIMESTAMP_QUERY` and we asked for it at device creation; `None`
    /// otherwise (no-op for code paths that opt in via `if let Some(t)`). The runner
    /// owns the per-frame write_start / write_end_and_resolve / tick lifecycle; this
    /// is also reachable to apps that want sub-pass instrumentation.
    pub gpu_timer: Option<crate::gpu_timer::GpuTimer>,
    /// Offscreen sRGB scene texture + composite node, allocated when the swapchain
    /// surface format is non-sRGB (browser-WebGPU on Chrome circa 2026-05). Scene +
    /// UI render into [`OffscreenTarget::view`]; the composite pass samples it +
    /// applies the sRGB transfer function + writes to the linear swapchain.
    /// `None` on native (where the swapchain itself is sRGB and the GPU handles
    /// gamma encoding on write).
    scene_target: Option<OffscreenTarget>,
    composite: Option<crate::composite::CompositeNode>,
    /// sRGB sibling of `surface_bundle.config.format`. Cached at construction so
    /// resize can recreate the scene target with the same format. `None` when
    /// scene_target is `None`.
    scene_format: Option<TextureFormat>,
    /// Present modes the adapter advertised for this surface. Cached at
    /// construction so the runtime `vsync` command can validate without
    /// re-querying `get_surface_capabilities` (which would force a roundtrip on
    /// every console command). Wasm browsers typically advertise only
    /// `PresentMode::Fifo`; native usually has all four.
    present_modes: Vec<PresentMode>,
}

impl RenderDevice {
    /// Acquire a surface for `window`, request a high-performance adapter, and configure the
    /// surface for sRGB rendering when the platform supports it. `requested_msaa_samples` of 1
    /// means no MSAA; values > 1 (typically 4) request a multisampled color attachment. The
    /// returned [`RenderDevice`] reports its effective sample count via
    /// [`RenderDevice::sample_count`], which may fall back to 1 if the requested count isn't
    /// supported by the adapter for the chosen surface format.
    pub async fn new(window: Arc<Window>, requested_msaa_samples: u32) -> Result<Self> {
        let instance = Instance::default();
        let surface = instance.create_surface(window.clone())?;
        let size = window.inner_size();
        Self::from_surface(instance, surface, size, requested_msaa_samples).await
    }

    /// Variant of [`Self::new`] that takes a wgpu [`Surface`] directly. Used by code
    /// paths that don't have a winit [`Window`] (Web Worker mode constructs the
    /// surface from an `OffscreenCanvas` in `rye-app::wasm::worker`, then hands it
    /// here).
    ///
    /// Keeps `rye-render` decoupled from `web-sys` and the worker-mode plumbing:
    /// the caller owns the surface-creation specifics; this function owns the
    /// adapter / device / configuration / scene-target / composite / msaa setup
    /// that's identical regardless of how the surface was obtained.
    ///
    /// `size` is the surface's pixel dimensions. The caller decides how to derive
    /// it (`window.inner_size()` in the windowed path, the OffscreenCanvas
    /// dimensions in the worker path).
    pub async fn from_surface(
        instance: Instance,
        surface: Surface<'static>,
        size: winit::dpi::PhysicalSize<u32>,
        requested_msaa_samples: u32,
    ) -> Result<Self> {
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
            })
            .await?;

        // GPU timer queries. wgpu 27 splits the capability across two features:
        //   - TIMESTAMP_QUERY: enables `RenderPassDescriptor::timestamp_writes`
        //     (timestamps attached to a render pass)
        //   - TIMESTAMP_QUERY_INSIDE_ENCODERS: enables
        //     `CommandEncoder::write_timestamp` (free-floating in the encoder, our
        //     current path because App::render owns its own render passes that
        //     rye-app can't reach into)
        //
        // Chrome's WebGPU on the current build (2026-05-22) advertises TIMESTAMP_QUERY
        // but NOT TIMESTAMP_QUERY_INSIDE_ENCODERS, so requesting the latter would fail
        // the adapter check; requesting only the former and then calling
        // write_timestamp panics (validation error -> wgpu panic on wasm). We require
        // BOTH for the GPU timer to be enabled; otherwise we silently skip it and the
        // gpu-total section just doesn't appear in `trace summary`.
        let needed = Features::TIMESTAMP_QUERY | Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        let timestamps_ok = adapter.features().contains(needed);
        let required_features = if timestamps_ok {
            needed
        } else {
            Features::empty()
        };
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("Rye Device"),
                required_features,
                required_limits: Limits::default(),
                memory_hints: MemoryHints::default(),
                trace: Trace::Off,
                // wgpu v27 requires opting in to experimental features explicitly; we don't use
                // any.
                experimental_features: Default::default(),
            })
            .await?;
        if timestamps_ok {
            tracing::info!("GPU timestamp queries enabled (TIMESTAMP_QUERY + INSIDE_ENCODERS)");
        } else {
            tracing::info!(
                "GPU timestamp queries unavailable (adapter features: {:?})",
                adapter.features()
            );
        }

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        tracing::info!(
            "surface picked format={format:?} (advertised={:?})",
            caps.formats
        );

        // Prefer an opaque alpha mode over `PreMultiplied`. On the WebGPU/WebGL surface in
        // the browser, the adapter advertises `PreMultiplied` first, which tells the
        // compositor to interpret the shader output as already-multiplied-by-alpha and
        // composite the canvas over the page underneath. When the page background is dark
        // (or anything non-white), shader output with an alpha less than 1 ends up
        // darkened against it. Picking `Opaque` when supported sidesteps the issue
        // entirely; if the adapter doesn't offer it (rare native edge cases) we accept
        // whatever it advertises first.
        let alpha_mode = caps
            .alpha_modes
            .iter()
            .copied()
            .find(|m| *m == CompositeAlphaMode::Opaque)
            .unwrap_or(caps.alpha_modes[0]);

        let config = SurfaceConfiguration {
            // COPY_SRC keeps texture readback open for headless screenshot tools and any future
            // capture path; cost is negligible vs. the headache of re-creating the surface to
            // enable it later.
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            format,
            width: size.width,
            height: size.height,
            present_mode: PresentMode::Fifo,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        // Offscreen sRGB scene target + composite. When the swapchain format is
        // already sRGB (native + the rare browser that advertises sRGB), we render
        // straight into the swapchain and skip the composite — that's the standard
        // path. When the swapchain is linear (Chrome WebGPU canvas on 2026-05), we
        // allocate an sRGB scene texture, redirect rendering into it, and add a
        // final pass that samples + gamma-encodes for write to the linear swapchain.
        // The MSAA path doesn't compose with the offscreen scene target in v1
        // (would need to retarget the MSAA resolve_target from swapchain to scene
        // texture, and recompute the composite to sample from the resolved scene
        // texture); we force sample_count = 1 in the offscreen-composite case and
        // log the override.
        let needs_composite = !format.is_srgb();
        let effective_msaa = if needs_composite {
            if requested_msaa_samples > 1 {
                tracing::warn!(
                    "MSAA={requested_msaa_samples}x ignored: composite pass for sRGB \
                     gamma encoding (browser-WebGPU linear surface) is incompatible \
                     with MSAA in v1; falling back to sample_count=1",
                );
            }
            1
        } else {
            requested_msaa_samples
        };

        let sample_count = negotiate_sample_count(&adapter, format, effective_msaa);
        let msaa_target = (sample_count > 1)
            .then(|| create_msaa_target(&device, format, size.width, size.height, sample_count));

        let (scene_target, composite, scene_format) = if needs_composite {
            let scene_fmt = format.add_srgb_suffix();
            tracing::info!(
                "non-sRGB surface; rendering through offscreen scene target {scene_fmt:?} \
                 with composite pass to {format:?} swapchain"
            );
            let scene = create_scene_target(&device, scene_fmt, size.width, size.height);
            let mut comp = crate::composite::CompositeNode::new(&device, format);
            comp.set_scene_view(&device, &scene.view);
            (Some(scene), Some(comp), Some(scene_fmt))
        } else {
            (None, None, None)
        };

        let gpu_timer = crate::gpu_timer::GpuTimer::new(&device, &queue);

        let present_modes = caps.present_modes.clone();
        tracing::info!("surface present modes advertised: {present_modes:?}");

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            surface_bundle: SurfaceBundle {
                surface,
                config,
                size,
            },
            sample_count,
            msaa_target,
            gpu_timer,
            scene_target,
            composite,
            scene_format,
            present_modes,
        })
    }

    /// Reconfigure the surface for the new window size. No-ops on width or height of zero (the
    /// minimized-window case wgpu rejects outright). Recreates the MSAA texture to match the new
    /// dimensions when MSAA is enabled, and the offscreen scene texture (rewiring the composite
    /// pass's bind group) when the wasm-style sRGB-composite path is active.
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.surface_bundle.size = new_size;
        self.surface_bundle.config.width = new_size.width;
        self.surface_bundle.config.height = new_size.height;
        self.surface_bundle
            .surface
            .configure(&self.device, &self.surface_bundle.config);
        if self.sample_count > 1 {
            self.msaa_target = Some(create_msaa_target(
                &self.device,
                self.surface_bundle.config.format,
                new_size.width,
                new_size.height,
                self.sample_count,
            ));
        }
        if let (Some(scene_fmt), Some(composite)) = (self.scene_format, self.composite.as_mut()) {
            let scene = create_scene_target(&self.device, scene_fmt, new_size.width, new_size.height);
            composite.set_scene_view(&self.device, &scene.view);
            self.scene_target = Some(scene);
        }
    }

    /// Acquire the next swapchain texture and its default view, ready for a render pass. Returns
    /// the wgpu surface error directly so callers can branch on `Lost` / `Outdated` / `Timeout`
    /// without extra wrapping.
    ///
    /// When [`RenderDevice::sample_count`] is > 1, the swapchain view is the *resolve target*,
    /// not the direct render target; pass [`RenderDevice::msaa_view`]'s result to render passes
    /// and use the swapchain view as the resolve target on the final (egui-paint) pass.
    pub fn begin_frame(
        &self,
    ) -> std::result::Result<(SurfaceTexture, TextureView), wgpu::SurfaceError> {
        let frame = self.surface_bundle.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        Ok((frame, view))
    }

    /// Effective MSAA sample count. 1 = MSAA off; 4 (or other power of two) = MSAA on. May
    /// differ from the value requested at construction if the adapter doesn't support the
    /// requested count for the chosen surface format.
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Currently configured present mode. `PresentMode::Fifo` (vsync) is the
    /// default the surface is constructed with.
    pub fn present_mode(&self) -> PresentMode {
        self.surface_bundle.config.present_mode
    }

    /// Present modes the adapter advertised for this surface. Use to validate
    /// before calling [`Self::set_present_mode`]; modes outside this list
    /// trigger a wgpu validation error at `surface.configure`.
    pub fn supported_present_modes(&self) -> &[PresentMode] {
        &self.present_modes
    }

    /// Switch the surface to a new present mode at runtime. Returns `Ok(())` if
    /// the mode was applied, `Err(mode)` if the adapter does not advertise it
    /// (no surface change occurs). Reconfigures the surface in place; the next
    /// `begin_frame` will use the new mode.
    ///
    /// Common modes:
    /// - `Fifo`: vsync; blocks at `present` until the next display refresh.
    ///   Lowest power, no tearing, max framerate = display refresh rate. The
    ///   default and the only mode advertised by browser WebGPU surfaces.
    /// - `Mailbox`: triple-buffered; latest frame replaces the queued one. No
    ///   tearing AND no `present` block, so framerate is uncapped by vsync.
    ///   Preferred for "vsync off" when supported.
    /// - `Immediate`: single-buffered; `present` returns immediately, tearing
    ///   visible. Use only if `Mailbox` isn't available.
    /// - `FifoRelaxed`: adaptive vsync; tears under the rate, vsyncs above.
    pub fn set_present_mode(&mut self, mode: PresentMode) -> std::result::Result<(), PresentMode> {
        if !self.present_modes.contains(&mode) {
            return Err(mode);
        }
        if self.surface_bundle.config.present_mode == mode {
            return Ok(());
        }
        self.surface_bundle.config.present_mode = mode;
        self.surface_bundle
            .surface
            .configure(&self.device, &self.surface_bundle.config);
        tracing::info!("surface present_mode -> {mode:?}");
        Ok(())
    }

    /// View into the multisampled color attachment, when MSAA is enabled. `None` when
    /// [`RenderDevice::sample_count`] is 1. Render passes should use this as the color attachment
    /// view when present, with the swapchain view as the resolve target on the final pass.
    pub fn msaa_view(&self) -> Option<&TextureView> {
        self.msaa_target.as_ref().map(|t| &t.view)
    }

    /// View into the offscreen sRGB scene texture, when the composite path is
    /// active. `None` on native (where the swapchain is sRGB and renders write
    /// directly into it). The render-target priority chain for the runner is:
    /// `msaa_view()` first (MSAA on, native path), then `scene_view()` (composite
    /// path), then the swapchain view directly (native, no MSAA).
    pub fn scene_view(&self) -> Option<&TextureView> {
        self.scene_target.as_ref().map(|t| &t.view)
    }

    /// Format that render pipelines should target. Differs from
    /// `surface_bundle.config.format` only when the composite path is active: the
    /// surface itself is the linear swap format (browser-WebGPU's only option) but
    /// scene + UI pipelines actually write into the sRGB offscreen texture, so
    /// their `ColorTargetState.format` needs to match THAT. The composite pass
    /// itself targets the linear swap format and is built with that directly in
    /// `CompositeNode::new`; downstream consumers shouldn't need to special-case
    /// it.
    ///
    /// Use this in pipeline constructors instead of reading
    /// `surface_bundle.config.format` directly.
    pub fn target_format(&self) -> TextureFormat {
        self.scene_format
            .unwrap_or(self.surface_bundle.config.format)
    }

    /// Run the final composite pass: sample the scene texture, gamma-encode in the
    /// fragment shader, and write to `swap_view`. Caller submits the encoder.
    /// No-op when `scene_view()` is `None` (native fast path).
    pub fn composite_to_swap(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        swap_view: &TextureView,
    ) {
        if let Some(composite) = self.composite.as_ref() {
            composite.run(encoder, swap_view);
        }
    }

    /// Force the composite pipeline through one dummy draw so the GPU driver
    /// compiles its PSO during setup instead of stalling the first real frame.
    /// No-op on the native fast path (no composite pipeline exists).
    ///
    /// The composite's scene-view binding from construction is reused as-is;
    /// the dummy target is a 1×1 texture in the swap format that the
    /// pipeline was built against. One `queue.submit` at warm time.
    ///
    /// Architectural note: warming lives on `RenderDevice` (not the runner)
    /// because only this struct has the composite handle + the matching
    /// swap format. The runner just calls this once during setup_after_device
    /// alongside `UiIntegration::warm_pipelines` + `App::warm_pipelines`.
    pub fn warm_composite(&self) {
        if self.composite.is_none() {
            return;
        }
        let format = self.surface_bundle.config.format;
        let dummy = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rye-render::composite::warm dummy"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let dummy_view = dummy.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rye-render::composite::warm encoder"),
            });
        self.composite_to_swap(&mut encoder, &dummy_view);
        self.queue.submit(Some(encoder.finish()));
    }
}

/// Allocate the offscreen scene-target texture used by the sRGB composite path. The
/// texture is the same dimensions as the surface; `RENDER_ATTACHMENT` so render passes
/// can target it, `TEXTURE_BINDING` so the composite shader can sample it.
fn create_scene_target(
    device: &Device,
    format: TextureFormat,
    width: u32,
    height: u32,
) -> OffscreenTarget {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rye-render::scene_target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    OffscreenTarget { texture, view }
}

/// Pick the highest sample count supported by the adapter for the given format that is
/// `<= requested`. Returns 1 if `requested == 1` or no multisampled count is supported.
fn negotiate_sample_count(adapter: &Adapter, format: TextureFormat, requested: u32) -> u32 {
    if requested <= 1 {
        return 1;
    }
    let features = adapter.get_texture_format_features(format);
    let flags = features.flags;
    // Walk requested -> 2 looking for a supported count. wgpu's sample-count flags expose 1, 2,
    // 4, 8, 16. Most consumer GPUs support 4 on sRGB surface formats.
    for count in [16u32, 8, 4, 2] {
        if count > requested {
            continue;
        }
        let supported = match count {
            2 => flags.contains(TextureFormatFeatureFlags::MULTISAMPLE_X2),
            4 => flags.contains(TextureFormatFeatureFlags::MULTISAMPLE_X4),
            8 => flags.contains(TextureFormatFeatureFlags::MULTISAMPLE_X8),
            16 => flags.contains(TextureFormatFeatureFlags::MULTISAMPLE_X16),
            _ => false,
        };
        if supported {
            if count != requested {
                tracing::warn!(
                    "requested MSAA {requested}x not supported on {format:?}; falling back to {count}x"
                );
            }
            return count;
        }
    }
    tracing::warn!("no multisampled count supported on {format:?}; MSAA disabled");
    1
}

fn create_msaa_target(
    device: &Device,
    format: TextureFormat,
    width: u32,
    height: u32,
    sample_count: u32,
) -> MsaaTarget {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("rye-render::msaa-color"),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    MsaaTarget { texture, view }
}
