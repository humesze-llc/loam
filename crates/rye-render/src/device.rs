//! Window surface + wgpu adapter/device acquisition.
//!
//! [`RenderDevice::new`] picks a high-performance adapter and an sRGB surface
//! format when available, optionally allocating a multisampled color
//! attachment. [`RenderDevice::begin_frame`] returns the per-frame
//! `(SurfaceTexture, TextureView)`; under MSAA, [`RenderDevice::msaa_view`] is
//! the render target and the swapchain view is the resolve target.

use anyhow::Result;
use std::sync::Arc;
use wgpu::*;
use winit::window::Window;

/// Surface + per-frame configuration. Owned by [`RenderDevice`]; exposed so
/// resize-aware code can read the current size and format. `surface` is `None`
/// on the headless path ([`RenderDevice::new_headless`]); `config` and `size`
/// stay populated there (synthesized from the requested dimensions) so every
/// caller reading `config.format` / `config.width` / `size` is path-agnostic.
pub struct SurfaceBundle {
    pub surface: Option<Surface<'static>>,
    pub config: SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
}

/// Owned color target for the headless path: both the render attachment and the
/// readback copy-source, standing in for the absent swapchain texture. `view` is
/// the color attachment; `texture` is the `copy_texture_to_buffer` source.
struct HeadlessColor {
    texture: Texture,
    view: TextureView,
}

/// Multisampled color attachment, allocated when the sample count is > 1.
/// [`MsaaTarget::view`] is the render target; the swapchain view resolves it.
pub struct MsaaTarget {
    // Keeps the GPU allocation alive for the lifetime of `view`.
    #[allow(dead_code)]
    texture: Texture,
    pub view: TextureView,
}

/// Offscreen sRGB render target for the non-sRGB-surface (browser-WebGPU) path:
/// scene + UI render here, then [`crate::composite::CompositeNode`]
/// gamma-encodes into the linear swapchain. Separate from [`MsaaTarget`] for
/// branch readability.
pub struct OffscreenTarget {
    // Keeps the GPU allocation alive for the lifetime of `view`.
    #[allow(dead_code)]
    texture: Texture,
    pub view: TextureView,
}

/// All wgpu state the engine carries. One per app; not cloneable.
pub struct RenderDevice {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub surface_bundle: SurfaceBundle,
    sample_count: u32,
    msaa_target: Option<MsaaTarget>,
    /// GPU timestamp query infrastructure. `Some` only when the adapter
    /// advertised the timestamp features and we requested them. The runner owns
    /// the per-frame lifecycle; apps can reach in for sub-pass instrumentation.
    pub gpu_timer: Option<crate::gpu_timer::GpuTimer>,
    /// Offscreen sRGB scene texture + composite, present only on the non-sRGB
    /// swapchain path. `None` on native (swapchain is sRGB).
    scene_target: Option<OffscreenTarget>,
    composite: Option<crate::composite::CompositeNode>,
    /// sRGB sibling of the surface format, so resize can recreate the scene
    /// target. `None` when `scene_target` is `None`.
    scene_format: Option<TextureFormat>,
    /// Advertised present modes, cached so the `vsync` command validates without
    /// re-querying `get_surface_capabilities`. Browsers typically advertise only
    /// `Fifo`; native usually all four.
    present_modes: Vec<PresentMode>,
    /// Owned offscreen color target on the headless path; `None` when a real
    /// surface is present. Mutually exclusive with `surface_bundle.surface`.
    headless_color: Option<HeadlessColor>,
}

impl RenderDevice {
    /// Acquire a surface for `window`, request a high-performance adapter, and
    /// configure for sRGB rendering when supported. `requested_msaa_samples` of
    /// 1 means no MSAA; the effective count (see
    /// [`RenderDevice::sample_count`]) may fall back to 1 if unsupported.
    pub async fn new(window: Arc<Window>, requested_msaa_samples: u32) -> Result<Self> {
        let instance = Instance::default();
        let surface = instance.create_surface(window.clone())?;
        let size = window.inner_size();
        Self::from_surface(instance, surface, size, requested_msaa_samples).await
    }

    /// [`Self::new`] variant taking a wgpu [`Surface`] directly, for callers
    /// without a winit [`Window`] (Web Worker mode builds the surface from an
    /// `OffscreenCanvas`). Keeps `rye-render` decoupled from `web-sys`: the
    /// caller owns surface creation, this owns the rest. `size` is the surface's
    /// pixel dimensions.
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

        // wgpu 27 splits timestamps into TIMESTAMP_QUERY (pass-attached) and
        // TIMESTAMP_QUERY_INSIDE_ENCODERS (free-floating write_timestamp, our
        // path since App::render owns its passes). Some browser builds advertise
        // only the former, where write_timestamp then panics; require both or
        // skip the timer entirely.
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

        // Prefer `Opaque` over the browser-advertised `PreMultiplied`, which
        // composites alpha < 1 shader output against the page and darkens it on
        // non-white backgrounds. Fall back to whatever is advertised first.
        let alpha_mode = caps
            .alpha_modes
            .iter()
            .copied()
            .find(|m| *m == CompositeAlphaMode::Opaque)
            .unwrap_or(caps.alpha_modes[0]);

        let config = SurfaceConfiguration {
            // COPY_SRC keeps headless screenshot readback open at negligible cost.
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

        // sRGB swapchains render directly; linear ones (browser-WebGPU) need an
        // offscreen sRGB scene texture plus a gamma-encoding composite pass.
        // MSAA doesn't compose with the offscreen target yet, so the composite
        // path forces sample_count = 1.
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
                surface: Some(surface),
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
            headless_color: None,
        })
    }

    /// Surfaceless constructor for offline/headless rendering: acquire a
    /// high-performance adapter with no `compatible_surface`, allocate an owned
    /// sRGB color target of `width`x`height`, and synthesize a `config`/`size`
    /// so size/format readers stay path-agnostic. Sample count is fixed at 1
    /// (no MSAA resolve) and the sRGB format sidesteps the composite path, so
    /// the demo's pipelines (built against [`target_format`] + [`sample_count`])
    /// match without change. Render into [`headless_view`], read back
    /// [`headless_texture`].
    ///
    /// [`target_format`]: RenderDevice::target_format
    /// [`sample_count`]: RenderDevice::sample_count
    /// [`headless_view`]: RenderDevice::headless_view
    /// [`headless_texture`]: RenderDevice::headless_texture
    pub async fn new_headless(width: u32, height: u32) -> Result<Self> {
        let instance = Instance::default();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                compatible_surface: None,
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
            })
            .await?;

        // Same timestamp-feature negotiation as the surface path (see
        // `from_surface`): request both query features or neither.
        let needed = Features::TIMESTAMP_QUERY | Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        let required_features = if adapter.features().contains(needed) {
            needed
        } else {
            Features::empty()
        };
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("Rye Device (headless)"),
                required_features,
                required_limits: Limits::default(),
                memory_hints: MemoryHints::default(),
                trace: Trace::Off,
                experimental_features: Default::default(),
            })
            .await?;

        // Rgba (not Bgra) sRGB: sRGB matches the native swapchain's gamma so
        // headless frames look identical to the window, and Rgba order needs no
        // channel swap on readback. sRGB also means `needs_composite` is false.
        let format = TextureFormat::Rgba8UnormSrgb;
        let size = winit::dpi::PhysicalSize::new(width, height);
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("rye-render::headless-color"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());

        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            format,
            width,
            height,
            present_mode: PresentMode::Fifo,
            alpha_mode: CompositeAlphaMode::Opaque,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let gpu_timer = crate::gpu_timer::GpuTimer::new(&device, &queue);

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            surface_bundle: SurfaceBundle {
                surface: None,
                config,
                size,
            },
            sample_count: 1,
            msaa_target: None,
            gpu_timer,
            scene_target: None,
            composite: None,
            scene_format: None,
            present_modes: Vec::new(),
            headless_color: Some(HeadlessColor { texture, view }),
        })
    }

    /// Color attachment view for the headless path, or `None` when a real
    /// surface is present. Render this frame's scene into it, then read it back
    /// with [`RenderDevice::headless_texture`].
    pub fn headless_view(&self) -> Option<&TextureView> {
        self.headless_color.as_ref().map(|h| &h.view)
    }

    /// The headless color texture, the `copy_texture_to_buffer` readback source.
    /// `None` when a real surface is present.
    pub fn headless_texture(&self) -> Option<&Texture> {
        self.headless_color.as_ref().map(|h| &h.texture)
    }

    /// Reconfigure the surface for `new_size`. No-ops on a zero dimension (the
    /// minimized case wgpu rejects). Recreates the MSAA and offscreen-scene
    /// textures (rewiring the composite bind group) when those paths are active.
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.surface_bundle.size = new_size;
        self.surface_bundle.config.width = new_size.width;
        self.surface_bundle.config.height = new_size.height;
        if let Some(surface) = &self.surface_bundle.surface {
            surface.configure(&self.device, &self.surface_bundle.config);
        }
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
            let scene =
                create_scene_target(&self.device, scene_fmt, new_size.width, new_size.height);
            composite.set_scene_view(&self.device, &scene.view);
            self.scene_target = Some(scene);
        }
    }

    /// Acquire the next swapchain texture and its default view. Returns the
    /// wgpu surface error directly so callers can branch on `Lost` / `Outdated`
    /// / `Timeout`. Under MSAA the swapchain view is the resolve target, not the
    /// render target; see [`RenderDevice::msaa_view`].
    pub fn begin_frame(
        &self,
    ) -> std::result::Result<(SurfaceTexture, TextureView), wgpu::SurfaceError> {
        let frame = self
            .surface_bundle
            .surface
            .as_ref()
            .expect("begin_frame on a headless RenderDevice (no surface)")
            .get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        Ok((frame, view))
    }

    /// Effective MSAA sample count (1 = off). May differ from the requested
    /// count if the adapter doesn't support it for the chosen format.
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Currently configured present mode (`Fifo`/vsync at construction).
    pub fn present_mode(&self) -> PresentMode {
        self.surface_bundle.config.present_mode
    }

    /// Advertised present modes. Modes outside this list trigger a wgpu
    /// validation error at `surface.configure`.
    pub fn supported_present_modes(&self) -> &[PresentMode] {
        &self.present_modes
    }

    /// Switch present mode at runtime. `Err(mode)` (no change) if the adapter
    /// doesn't advertise it; otherwise reconfigures in place for the next
    /// `begin_frame`.
    ///
    /// - `Fifo`: vsync; the default and the only browser-WebGPU mode.
    /// - `Mailbox`: triple-buffered, no tearing, uncapped; preferred "vsync off".
    /// - `Immediate`: tears; use only when `Mailbox` is unavailable.
    /// - `FifoRelaxed`: adaptive vsync; tears under the rate, vsyncs above.
    pub fn set_present_mode(&mut self, mode: PresentMode) -> std::result::Result<(), PresentMode> {
        if !self.present_modes.contains(&mode) {
            return Err(mode);
        }
        if self.surface_bundle.config.present_mode == mode {
            return Ok(());
        }
        self.surface_bundle.config.present_mode = mode;
        if let Some(surface) = &self.surface_bundle.surface {
            surface.configure(&self.device, &self.surface_bundle.config);
        }
        tracing::info!("surface present_mode -> {mode:?}");
        Ok(())
    }

    /// View into the multisampled color attachment, or `None` when MSAA is off.
    /// Use as the color attachment, resolving to the swapchain on the final pass.
    pub fn msaa_view(&self) -> Option<&TextureView> {
        self.msaa_target.as_ref().map(|t| &t.view)
    }

    /// View into the offscreen sRGB scene texture on the composite path, or
    /// `None` on native. Runner render-target priority: `msaa_view()`, then
    /// `scene_view()`, then the swapchain view directly.
    pub fn scene_view(&self) -> Option<&TextureView> {
        self.scene_target.as_ref().map(|t| &t.view)
    }

    /// Format scene + UI render pipelines should target. On the composite path
    /// this is the offscreen sRGB texture's format, not the linear swap format.
    /// Use this in pipeline constructors instead of reading the surface format
    /// directly.
    pub fn target_format(&self) -> TextureFormat {
        self.scene_format
            .unwrap_or(self.surface_bundle.config.format)
    }

    /// Run the final composite pass: sample the scene texture, gamma-encode in the
    /// fragment shader, and write to `swap_view`. Caller submits the encoder.
    /// No-op when `scene_view()` is `None` (native fast path).
    pub fn composite_to_swap(&self, encoder: &mut wgpu::CommandEncoder, swap_view: &TextureView) {
        if let Some(composite) = self.composite.as_ref() {
            composite.run(encoder, swap_view);
        }
    }

    /// Force one dummy composite draw so the driver compiles its PSO during
    /// setup, not on the first real frame. No-op on the native path. The dummy
    /// target is a 1x1 texture in the swap format the pipeline was built for.
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

/// Allocate the offscreen scene-target texture for the sRGB composite path.
/// `RENDER_ATTACHMENT` to draw into, `TEXTURE_BINDING` for the composite sample.
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

/// Highest adapter-supported sample count `<= requested` for `format`, or 1.
fn negotiate_sample_count(adapter: &Adapter, format: TextureFormat, requested: u32) -> u32 {
    if requested <= 1 {
        return 1;
    }
    let features = adapter.get_texture_format_features(format);
    let flags = features.flags;
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
