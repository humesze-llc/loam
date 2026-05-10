//! Window surface + wgpu adapter/device acquisition.
//!
//! [`RenderDevice::new`] picks a high-performance adapter and an sRGB surface format when
//! available, then optionally allocates a multisampled color attachment matching the surface's
//! size and format. Resize is handled by [`RenderDevice::resize`]. [`RenderDevice::begin_frame`]
//! returns the per-frame `(SurfaceTexture, TextureView)` pair the render graph draws into;
//! when MSAA is enabled, [`RenderDevice::msaa_view`] is the actual render target and the
//! swapchain view is used as the resolve target at the end of the egui paint pass.

use anyhow::Result;
use std::sync::Arc;
use wgpu::*;
use winit::window::Window;

/// Surface + per-frame configuration. Owned by [`RenderDevice`]; held
/// out as a struct so resize-aware code can read the current size and
/// format without poking at private fields.
pub struct SurfaceBundle {
    pub surface: Surface<'static>,
    pub config: SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
}

/// Multisampled color attachment paired with the surface. Allocated
/// when the configured sample count is > 1; [`MsaaTarget::view`] is
/// the render target every scene + UI pass writes into, with the
/// swapchain view used as the resolve target.
pub struct MsaaTarget {
    // Held to keep the GPU allocation alive for the lifetime of
    // `view` (which is a borrow into this texture). Never read
    // through the field itself.
    #[allow(dead_code)]
    texture: Texture,
    pub view: TextureView,
}

/// All wgpu state the engine carries: shared `Instance`, the chosen
/// `Adapter`, the logical `Device`, the submission `Queue`, the
/// current surface bundle, and an optional multisampled color
/// attachment matching the surface. One per app; cloning this is not
/// supported.
pub struct RenderDevice {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub surface_bundle: SurfaceBundle,
    sample_count: u32,
    msaa_target: Option<MsaaTarget>,
}

impl RenderDevice {
    /// Acquire a surface for `window`, request a high-performance
    /// adapter, and configure the surface for sRGB rendering when
    /// the platform supports it. `requested_msaa_samples` of 1 means
    /// no MSAA; values > 1 (typically 4) request a multisampled
    /// color attachment. The returned [`RenderDevice`] reports its
    /// effective sample count via [`RenderDevice::sample_count`],
    /// which may fall back to 1 if the requested count isn't
    /// supported by the adapter for the chosen surface format.
    pub async fn new(window: Arc<Window>, requested_msaa_samples: u32) -> Result<Self> {
        let instance = Instance::default();

        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("Rye Device"),
                required_features: Features::empty(),
                required_limits: Limits::default(),
                memory_hints: MemoryHints::default(),
                trace: Trace::Off,
                // wgpu v27 requires opting in to experimental features explicitly;
                // we don't use any.
                experimental_features: Default::default(),
            })
            .await?;

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = SurfaceConfiguration {
            // COPY_SRC keeps texture readback open for headless screenshot tools
            // and any future capture path; cost is negligible vs. the headache of
            // re-creating the surface to enable it later.
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            format,
            width: size.width,
            height: size.height,
            present_mode: PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        let sample_count = negotiate_sample_count(&adapter, format, requested_msaa_samples);
        let msaa_target = (sample_count > 1)
            .then(|| create_msaa_target(&device, format, size.width, size.height, sample_count));

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
        })
    }

    /// Reconfigure the surface for the new window size. No-ops on
    /// width or height of zero (the minimized-window case wgpu rejects
    /// outright). Recreates the MSAA texture to match the new
    /// dimensions when MSAA is enabled.
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
    }

    /// Acquire the next swapchain texture and its default view, ready
    /// for a render pass. Returns the wgpu surface error directly so
    /// callers can branch on `Lost` / `Outdated` / `Timeout` without
    /// extra wrapping.
    ///
    /// When [`RenderDevice::sample_count`] is > 1, the swapchain view
    /// is the *resolve target*, not the direct render target; pass
    /// [`RenderDevice::msaa_view`]'s result to render passes and use
    /// the swapchain view as the resolve target on the final
    /// (egui-paint) pass.
    pub fn begin_frame(
        &self,
    ) -> std::result::Result<(SurfaceTexture, TextureView), wgpu::SurfaceError> {
        let frame = self.surface_bundle.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        Ok((frame, view))
    }

    /// Effective MSAA sample count. 1 = MSAA off; 4 (or other power
    /// of two) = MSAA on. May differ from the value requested at
    /// construction if the adapter doesn't support the requested
    /// count for the chosen surface format.
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// View into the multisampled color attachment, when MSAA is
    /// enabled. `None` when [`RenderDevice::sample_count`] is 1.
    /// Render passes should use this as the color attachment view
    /// when present, with the swapchain view as the resolve target
    /// on the final pass.
    pub fn msaa_view(&self) -> Option<&TextureView> {
        self.msaa_target.as_ref().map(|t| &t.view)
    }
}

/// Pick the highest sample count supported by the adapter for the
/// given format that is `<= requested`. Returns 1 if `requested == 1`
/// or no multisampled count is supported.
fn negotiate_sample_count(adapter: &Adapter, format: TextureFormat, requested: u32) -> u32 {
    if requested <= 1 {
        return 1;
    }
    let features = adapter.get_texture_format_features(format);
    let flags = features.flags;
    // Walk requested -> 2 looking for a supported count. wgpu's
    // sample-count flags expose 1, 2, 4, 8, 16. Most consumer GPUs
    // support 4 on sRGB surface formats.
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
