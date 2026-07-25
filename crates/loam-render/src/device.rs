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
/// resize-aware code can read the current size and format.
pub struct SurfaceBundle {
    pub surface: Surface<'static>,
    pub config: SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
}

/// Multisampled color attachment, allocated when the sample count is > 1.
/// [`MsaaTarget::view`] is the render target; the swapchain view resolves it.
pub struct MsaaTarget {
    // Keeps the GPU allocation alive for the lifetime of `view`.
    #[allow(dead_code)]
    texture: Texture,
    pub view: TextureView,
    /// Non-sRGB reinterpretation for the gamma-space UI pass, or a second
    /// handle on the same view as `view` where the adapter cannot reinterpret.
    pub ui_view: TextureView,
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

/// Formats the UI pass renders through, resolved once from the surface format
/// and the adapter's downlevel capabilities. See [`ui_target_formats`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct UiTargetFormats {
    /// Format the egui pipeline is built against.
    pub ui_format: TextureFormat,
    /// Registered as the swapchain's view format and used for the UI pass's
    /// swapchain view. `None` means the UI uses the swapchain's own format.
    pub swap_view_format: Option<TextureFormat>,
    /// Same, for the multisampled color attachment.
    pub msaa_view_format: Option<TextureFormat>,
}

/// Decide whether the UI pass may reinterpret the sRGB swapchain (and its MSAA
/// attachment) as the non-sRGB twin, which is how egui gets the gamma-space
/// blending its feathering assumes.
///
/// wgpu-core gates the two reinterpretations behind separate downlevel flags:
/// `SURFACE_VIEW_FORMATS` for a swapchain texture's view formats and
/// `VIEW_FORMATS` for an ordinary texture's. GL/WebGL advertises neither and
/// Vulkan advertises the surface one only with `VK_KHR_swapchain_mutable_format`,
/// so an unguarded entry is a validation failure at `surface.configure`, not a
/// degraded render. Both flags are required together because the UI pass draws
/// into the MSAA attachment and resolves into the swapchain, so one pipeline
/// format has to match both attachments whatever the sample count negotiates to.
fn ui_target_formats(surface_format: TextureFormat, downlevel: DownlevelFlags) -> UiTargetFormats {
    // Composite path: the UI paints into the offscreen sRGB scene texture and
    // `CompositeNode` encodes gamma afterwards, so nothing is reinterpreted.
    // `add_srgb_suffix` is the identity for formats without an sRGB sibling.
    if !surface_format.is_srgb() {
        return UiTargetFormats {
            ui_format: surface_format.add_srgb_suffix(),
            swap_view_format: None,
            msaa_view_format: None,
        };
    }
    let reinterpret = DownlevelFlags::SURFACE_VIEW_FORMATS | DownlevelFlags::VIEW_FORMATS;
    if !downlevel.contains(reinterpret) {
        return UiTargetFormats {
            ui_format: surface_format,
            swap_view_format: None,
            msaa_view_format: None,
        };
    }
    let gamma = surface_format.remove_srgb_suffix();
    UiTargetFormats {
        ui_format: gamma,
        swap_view_format: Some(gamma),
        msaa_view_format: Some(gamma),
    }
}

/// Swapchain configuration for a surface of `size`. Split out of
/// [`RenderDevice::from_surface`] so the `view_formats` registration, the field
/// wgpu validates against the downlevel flags, is checkable without a device.
fn surface_configuration(
    format: TextureFormat,
    size: winit::dpi::PhysicalSize<u32>,
    alpha_mode: CompositeAlphaMode,
    ui_targets: UiTargetFormats,
) -> SurfaceConfiguration {
    SurfaceConfiguration {
        // COPY_SRC keeps headless screenshot readback open at negligible cost.
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        format,
        width: size.width,
        height: size.height,
        present_mode: PresentMode::Fifo,
        alpha_mode,
        view_formats: ui_targets.swap_view_format.into_iter().collect(),
        desired_maximum_frame_latency: 2,
    }
}

/// UI-pass view of a target that registered `ui_view_format` as its
/// reinterpretation. `None` requests the target's own format, which is the only
/// legal request when nothing was registered.
fn ui_view_descriptor(ui_view_format: Option<TextureFormat>) -> TextureViewDescriptor<'static> {
    TextureViewDescriptor {
        format: ui_view_format,
        ..Default::default()
    }
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
    ui_targets: UiTargetFormats,
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
    /// `OffscreenCanvas`). Keeps `loam-render` decoupled from `web-sys`: the
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
                label: Some("Loam Device"),
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

        let ui_targets = ui_target_formats(format, adapter.get_downlevel_capabilities().flags);
        if format.is_srgb() && ui_targets.swap_view_format.is_none() {
            tracing::warn!(
                "adapter lacks SURFACE_VIEW_FORMATS/VIEW_FORMATS; UI blends in linear \
                 space and egui feathering will look thin on hairlines"
            );
        }
        let config = surface_configuration(format, size, alpha_mode, ui_targets);

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
        let msaa_target = (sample_count > 1).then(|| {
            create_msaa_target(
                &device,
                format,
                size.width,
                size.height,
                sample_count,
                ui_targets.msaa_view_format,
            )
        });

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
            ui_targets,
        })
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
                self.ui_targets.msaa_view_format,
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
        let frame = self.surface_bundle.surface.get_current_texture()?;
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
        self.surface_bundle
            .surface
            .configure(&self.device, &self.surface_bundle.config);
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

    /// Format the egui/UI pipeline should target: the swapchain format with the
    /// sRGB suffix stripped, so blending happens in gamma space as egui's
    /// feathering assumes. Falls back to the sRGB format where the adapter's
    /// downlevel capabilities forbid reinterpreting swapchain views. On the
    /// composite path the UI renders into the scene texture before the
    /// composite, so this stays `target_format`.
    pub fn ui_format(&self) -> TextureFormat {
        self.ui_targets.ui_format
    }

    /// View of the acquired swapchain texture for the UI pass: the non-sRGB
    /// reinterpretation where the adapter supports it, the texture's own format
    /// otherwise.
    pub fn create_ui_swap_view(&self, frame: &SurfaceTexture) -> TextureView {
        frame
            .texture
            .create_view(&ui_view_descriptor(self.ui_targets.swap_view_format))
    }

    /// UI-pass view of the MSAA attachment, `None` when MSAA is off. Non-sRGB
    /// where the adapter supports it, the attachment's own format otherwise.
    pub fn msaa_ui_view(&self) -> Option<&TextureView> {
        self.msaa_target.as_ref().map(|t| &t.ui_view)
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
            label: Some("loam-render::composite::warm dummy"),
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
                label: Some("loam-render::composite::warm encoder"),
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
        label: Some("loam-render::scene_target"),
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

/// `ui_view_format` is [`UiTargetFormats::msaa_view_format`]: `Some` registers
/// the reinterpretation the UI pass draws through, `None` leaves the attachment
/// single-format. Taken by reference so the descriptor's `view_formats` slice
/// can borrow it, which also keeps the descriptor checkable without a device.
fn msaa_texture_descriptor(
    format: TextureFormat,
    width: u32,
    height: u32,
    sample_count: u32,
    ui_view_format: &Option<TextureFormat>,
) -> TextureDescriptor<'_> {
    TextureDescriptor {
        label: Some("loam-render::msaa-color"),
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
        view_formats: ui_view_format.as_slice(),
    }
}

fn create_msaa_target(
    device: &Device,
    format: TextureFormat,
    width: u32,
    height: u32,
    sample_count: u32,
    ui_view_format: Option<TextureFormat>,
) -> MsaaTarget {
    let texture = device.create_texture(&msaa_texture_descriptor(
        format,
        width,
        height,
        sample_count,
        &ui_view_format,
    ));
    let view = texture.create_view(&TextureViewDescriptor::default());
    let ui_view = texture.create_view(&ui_view_descriptor(ui_view_format));
    MsaaTarget {
        texture,
        view,
        ui_view,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOTH: DownlevelFlags =
        DownlevelFlags::SURFACE_VIEW_FORMATS.union(DownlevelFlags::VIEW_FORMATS);

    /// Both swapchain paths (sRGB direct, linear composite) and a format with
    /// no sRGB sibling.
    const SURFACES: [TextureFormat; 4] = [
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rgba16Float,
    ];

    /// Neither flag, each alone, both, and everything.
    const DOWNLEVELS: [DownlevelFlags; 5] = [
        DownlevelFlags::empty(),
        DownlevelFlags::SURFACE_VIEW_FORMATS,
        DownlevelFlags::VIEW_FORMATS,
        BOTH,
        DownlevelFlags::all(),
    ];

    const SIZE: winit::dpi::PhysicalSize<u32> = winit::dpi::PhysicalSize {
        width: 800,
        height: 600,
    };

    #[test]
    fn srgb_surface_registers_view_formats_only_with_both_downlevel_flags() {
        let srgb = TextureFormat::Bgra8UnormSrgb;
        let gamma = TextureFormat::Bgra8Unorm;
        let table = [
            (DownlevelFlags::empty(), None),
            (DownlevelFlags::SURFACE_VIEW_FORMATS, None),
            (DownlevelFlags::VIEW_FORMATS, None),
            (BOTH, Some(gamma)),
            (DownlevelFlags::all(), Some(gamma)),
        ];
        for (downlevel, expected) in table {
            let targets = ui_target_formats(srgb, downlevel);
            assert_eq!(targets.swap_view_format, expected, "{downlevel:?}");
            assert_eq!(targets.msaa_view_format, expected, "{downlevel:?}");
            assert_eq!(targets.ui_format, expected.unwrap_or(srgb), "{downlevel:?}");
        }
    }

    #[test]
    fn composite_path_registers_no_view_formats_and_targets_the_srgb_scene_format() {
        let table = [
            (TextureFormat::Bgra8Unorm, TextureFormat::Bgra8UnormSrgb),
            (TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb),
            // No sRGB sibling: the offscreen scene target keeps this format.
            (TextureFormat::Rgba16Float, TextureFormat::Rgba16Float),
        ];
        for (surface, scene) in table {
            for downlevel in [DownlevelFlags::empty(), DownlevelFlags::all()] {
                let targets = ui_target_formats(surface, downlevel);
                assert_eq!(targets.swap_view_format, None, "{surface:?} {downlevel:?}");
                assert_eq!(targets.msaa_view_format, None, "{surface:?} {downlevel:?}");
                assert_eq!(targets.ui_format, scene, "{surface:?} {downlevel:?}");
            }
        }
    }

    /// The UI pipeline is built once against `ui_format` and draws into the
    /// swapchain view (or the MSAA attachment resolving into it), so a plan
    /// whose formats disagree fails pipeline/attachment validation at paint.
    #[test]
    fn ui_format_matches_every_view_the_ui_pass_renders_into() {
        for surface in SURFACES {
            for downlevel in DOWNLEVELS {
                let targets = ui_target_formats(surface, downlevel);
                let case = format!("{surface:?} {downlevel:?}");
                assert_eq!(
                    targets.swap_view_format.is_some(),
                    targets.msaa_view_format.is_some(),
                    "reinterpretations move together: {case}"
                );
                if surface.is_srgb() {
                    // Direct path: the UI writes to the swapchain itself.
                    let swap = targets.swap_view_format.unwrap_or(surface);
                    assert_eq!(targets.ui_format, swap, "{case}");
                    let msaa = targets.msaa_view_format.unwrap_or(surface);
                    assert_eq!(targets.ui_format, msaa, "{case}");
                } else {
                    // Composite path: the UI writes to the offscreen target.
                    assert_eq!(targets.ui_format, surface.add_srgb_suffix(), "{case}");
                }
            }
        }
    }

    /// The guard lives in `ui_target_formats`, but wgpu only ever sees the
    /// descriptors. A descriptor that recomputes the non-sRGB twin itself
    /// registers a view format the adapter may reject outright, which is the
    /// failure the guard exists to prevent.
    #[test]
    fn descriptors_register_only_the_sanctioned_reinterpretation() {
        for surface in SURFACES {
            for downlevel in DOWNLEVELS {
                let case = format!("{surface:?} {downlevel:?}");
                // Derived from the flags rather than from `ui_target_formats`,
                // so a descriptor cannot drift in step with the decision.
                let sanctioned: Vec<TextureFormat> =
                    if surface.is_srgb() && downlevel.contains(BOTH) {
                        vec![surface.remove_srgb_suffix()]
                    } else {
                        vec![]
                    };
                let targets = ui_target_formats(surface, downlevel);
                let config =
                    surface_configuration(surface, SIZE, CompositeAlphaMode::Opaque, targets);
                assert_eq!(config.view_formats, sanctioned, "swapchain: {case}");
                let msaa = msaa_texture_descriptor(
                    surface,
                    SIZE.width,
                    SIZE.height,
                    4,
                    &targets.msaa_view_format,
                );
                assert_eq!(msaa.view_formats, sanctioned, "msaa attachment: {case}");
            }
        }
    }

    /// `create_view` rejects a format absent from the target's `view_formats`,
    /// so the UI view descriptors must stay inside what the target descriptors
    /// registered even when the decision says no reinterpretation.
    #[test]
    fn ui_views_request_only_a_format_their_target_registered() {
        for surface in SURFACES {
            for downlevel in DOWNLEVELS {
                let case = format!("{surface:?} {downlevel:?}");
                let targets = ui_target_formats(surface, downlevel);
                let config =
                    surface_configuration(surface, SIZE, CompositeAlphaMode::Opaque, targets);
                if let Some(requested) = ui_view_descriptor(targets.swap_view_format).format {
                    assert!(
                        config.view_formats.contains(&requested),
                        "swapchain: {case}"
                    );
                }
                let msaa = msaa_texture_descriptor(
                    surface,
                    SIZE.width,
                    SIZE.height,
                    4,
                    &targets.msaa_view_format,
                );
                if let Some(requested) = ui_view_descriptor(targets.msaa_view_format).format {
                    assert!(
                        msaa.view_formats.contains(&requested),
                        "msaa attachment: {case}"
                    );
                }
            }
        }
    }
}
