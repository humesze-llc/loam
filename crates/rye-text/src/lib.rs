//! `rye-text`: screen-space text rendering for game HUDs and overlays.
//!
//! Library-composition API with no global state and no `App`-trait coupling. Apps
//! construct a [`TextRenderer`] in `setup`, queue text strings each frame in `update`,
//! and call [`TextRenderer::render`] from `App::render` after the main scene.
//!
//! # Backend
//!
//! Uses [`ab_glyph`] for glyph rasterization and a hand-rolled wgpu pipeline for atlas +
//! textured-quad drawing. ASCII printable characters (`0x20..=0x7E`) are pre-baked at a
//! fixed atlas size at construction; per-call font sizes scale the resulting quads
//! bilinearly. Adequate for game HUD readouts (numbers, short Latin labels); not adequate
//! for typographic-quality text or non-Latin scripts.
//!
//! For richer text (mixed fonts, layout, multiline word wrap, complex UI), apps should
//! reach for `rye-egui` instead, which uses egui's own text system. `rye-text` exists
//! for the minimal-deps case where an app wants screen-space ASCII text without the full
//! egui stack.
//!
//! # Example
//!
//! ```ignore
//! let mut text = TextRenderer::new(device, queue, surface_format, font_bytes, 48.0)?;
//! text.queue("Score: 1234", [16.0, 16.0], 32.0, [1.0, 1.0, 1.0, 1.0]);
//! text.render(device, queue, &surface_view, [width, height])?;
//! ```

use std::collections::HashMap;

use ab_glyph::{Font, FontRef, Glyph, GlyphId, Point, ScaleFont};
use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use wgpu::*;

const ATLAS_SIZE: u32 = 1024;
const ATLAS_FORMAT: TextureFormat = TextureFormat::R8Unorm;

/// Per-glyph metadata in the atlas.
#[derive(Copy, Clone, Debug)]
struct GlyphEntry {
    /// Atlas UV rectangle (normalized 0..1).
    uv_min: [f32; 2],
    uv_max: [f32; 2],
    /// Pixel size at the atlas's bake size.
    px_width: f32,
    px_height: f32,
    /// Horizontal advance to the next glyph, at bake size.
    h_advance: f32,
    /// Pixel offset from baseline to glyph's top-left corner, at bake size.
    /// Positive `top` means above baseline (typical for letters);
    /// negative for descenders.
    bearing_x: f32,
    bearing_y: f32,
}

/// Vertex layout for textured-quad text drawing. One vertex per
/// quad corner; six vertices per glyph (two-triangle fan).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct TextVertex {
    pos: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct TextUniforms {
    viewport_size: [f32; 2],
    _pad: [f32; 2],
}

/// Screen-space text renderer.
///
/// Construct once. Each frame:
/// 1. Call [`TextRenderer::queue`] for each string to draw.
/// 2. Call [`TextRenderer::render`] to flush the queue to the surface.
///
/// The queue is reset every `render` call.
pub struct TextRenderer {
    pipeline: RenderPipeline,
    bind_group: BindGroup,
    uniform_buf: Buffer,
    // The atlas resources are referenced only through `bind_group`'s
    // resource bindings; we hold them on the struct so they stay
    // alive for the bind group's lifetime. Dropping them would free
    // the GPU resource the bind group still references and crash on
    // the next render.
    #[allow(dead_code)]
    atlas_tex: Texture,
    #[allow(dead_code)]
    atlas_view: TextureView,
    #[allow(dead_code)]
    atlas_sampler: Sampler,

    font_data: Vec<u8>,
    glyphs: HashMap<char, GlyphEntry>,
    bake_size_px: f32,
    line_height_px: f32,

    /// Vertex buffer; reallocated when capacity grows.
    vertex_buf: Buffer,
    vertex_capacity: u64,
    /// Per-frame queued vertices; cleared on render.
    queued: Vec<TextVertex>,
}

impl TextRenderer {
    /// Construct a renderer.
    ///
    /// `font_bytes` is the raw TTF/OTF font data (typically loaded via `include_bytes!`
    /// or read from disk). `bake_size_px` is the pixel size at which glyphs are
    /// rasterized into the atlas; smaller per-frame sizes look fine, larger sizes may
    /// show bilinear blurring. 48 is a reasonable default.
    pub fn new(
        device: &Device,
        queue: &Queue,
        surface_format: TextureFormat,
        font_bytes: &[u8],
        bake_size_px: f32,
    ) -> Result<Self> {
        let font_data = font_bytes.to_vec();
        let font = FontRef::try_from_slice(&font_data)
            .map_err(|e| anyhow!("rye-text: failed to parse font: {e}"))?;

        // Atlas texture (R8, alpha-only).
        let atlas_tex = device.create_texture(&TextureDescriptor {
            label: Some("rye-text atlas"),
            size: Extent3d {
                width: ATLAS_SIZE,
                height: ATLAS_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: ATLAS_FORMAT,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let atlas_view = atlas_tex.create_view(&TextureViewDescriptor::default());
        let atlas_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("rye-text atlas sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });

        // Pre-bake printable ASCII into the atlas.
        let (atlas_pixels, glyphs, line_height_px) = bake_ascii_atlas(&font, bake_size_px)?;
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &atlas_tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &atlas_pixels,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ATLAS_SIZE),
                rows_per_image: Some(ATLAS_SIZE),
            },
            Extent3d {
                width: ATLAS_SIZE,
                height: ATLAS_SIZE,
                depth_or_array_layers: 1,
            },
        );

        // Uniform buffer (viewport size).
        let uniform_buf = device.create_buffer(&BufferDescriptor {
            label: Some("rye-text uniforms"),
            size: std::mem::size_of::<TextUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group layout: uniforms + atlas + sampler.
        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("rye-text bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("rye-text bg"),
            layout: &bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&atlas_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&atlas_sampler),
                },
            ],
        });

        // Pipeline.
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("rye-text shader"),
            source: ShaderSource::Wgsl(WGSL_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("rye-text pipeline layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let vertex_attrs = wgpu::vertex_attr_array![
            0 => Float32x2,
            1 => Float32x2,
            2 => Float32x4,
        ];
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("rye-text pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[VertexBufferLayout {
                    array_stride: std::mem::size_of::<TextVertex>() as u64,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &vertex_attrs,
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: surface_format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Initial vertex buffer (grows on demand).
        let initial_capacity = 1024_u64;
        let zero_verts: Vec<TextVertex> =
            vec![bytemuck::Zeroable::zeroed(); initial_capacity as usize];
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rye-text vertices"),
            contents: bytemuck::cast_slice(&zero_verts),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });

        Ok(Self {
            pipeline,
            bind_group,
            uniform_buf,
            atlas_tex,
            atlas_view,
            atlas_sampler,
            font_data,
            glyphs,
            bake_size_px,
            line_height_px,
            vertex_buf,
            vertex_capacity: initial_capacity,
            queued: Vec::new(),
        })
    }

    /// Queue a string to be drawn this frame at `position` (top-left, pixel coordinates
    /// from top-left of viewport).
    ///
    /// `size_px` is the rendered glyph size; values close to the renderer's bake size
    /// give the cleanest result. Color is RGBA in 0..1 with straight (non-premultiplied)
    /// alpha.
    ///
    /// Newlines (`\n`) advance to the next line. Other control characters are skipped.
    pub fn queue(&mut self, text: &str, position: [f32; 2], size_px: f32, color: [f32; 4]) {
        layout_text(
            text,
            position,
            size_px,
            color,
            &self.glyphs,
            self.bake_size_px,
            self.line_height_px,
            &mut self.queued,
        );
    }

    /// Render queued text into `view` at `viewport_size` (pixels) and
    /// reset the queue.
    pub fn render(
        &mut self,
        device: &Device,
        queue: &Queue,
        view: &TextureView,
        viewport_size: [f32; 2],
    ) -> Result<()> {
        if self.queued.is_empty() {
            return Ok(());
        }

        // Upload uniforms.
        let uniforms = TextUniforms {
            viewport_size,
            _pad: [0.0; 2],
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        // Grow vertex buffer if needed (powers-of-two grow).
        let needed = self.queued.len() as u64;
        if needed > self.vertex_capacity {
            let mut new_cap = self.vertex_capacity.max(1);
            while new_cap < needed {
                new_cap *= 2;
            }
            self.vertex_buf = device.create_buffer(&BufferDescriptor {
                label: Some("rye-text vertices (grown)"),
                size: new_cap * std::mem::size_of::<TextVertex>() as u64,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.vertex_capacity = new_cap;
        }
        queue.write_buffer(&self.vertex_buf, 0, bytemuck::cast_slice(&self.queued));

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("rye-text encoder"),
        });
        {
            let mut rp = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("rye-text pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rp.set_pipeline(&self.pipeline);
            rp.set_bind_group(0, &self.bind_group, &[]);
            rp.set_vertex_buffer(0, self.vertex_buf.slice(..));
            rp.draw(0..self.queued.len() as u32, 0..1);
        }
        queue.submit(Some(encoder.finish()));

        self.queued.clear();
        Ok(())
    }

    /// Bake size that glyphs were rasterized at; per-frame sizes
    /// close to this value look cleanest.
    pub fn bake_size_px(&self) -> f32 {
        self.bake_size_px
    }

    /// Vertical advance between lines at the bake size.
    pub fn line_height_px(&self) -> f32 {
        self.line_height_px
    }

    /// Borrow the loaded font data (parsed lazily via [`FontRef`]
    /// each call). Useful for measurement helpers built on top.
    pub fn font_bytes(&self) -> &[u8] {
        &self.font_data
    }
}

/// Pure layout: append the vertices for `text` (six per glyph, two triangles) into
/// `out`. No GPU resources touched, so this can be tested with a hand-built glyph table.
///
/// `position` is the top-left of the first line in viewport coordinates; `size_px` is
/// the rendered glyph height; `scale = size_px / bake_size_px` rescales the baked atlas
/// geometry to the requested size. Newlines reset `cursor_x` to `position[0]` and
/// advance `cursor_y` by `line_height_px * scale`. Non-printable / out-of-ASCII chars
/// are skipped silently; chars with no glyph in the table (atlas didn't fit them) are
/// also skipped.
#[allow(clippy::too_many_arguments)] // pure layout helper, parameters are the layout state.
fn layout_text(
    text: &str,
    position: [f32; 2],
    size_px: f32,
    color: [f32; 4],
    glyphs: &HashMap<char, GlyphEntry>,
    bake_size_px: f32,
    line_height_px: f32,
    out: &mut Vec<TextVertex>,
) {
    let scale = size_px / bake_size_px;
    let line_h = line_height_px * scale;
    let mut cursor_x = position[0];
    let mut cursor_y = position[1];

    for c in text.chars() {
        if c == '\n' {
            cursor_x = position[0];
            cursor_y += line_h;
            continue;
        }
        if (c as u32) < 0x20 || (c as u32) > 0x7E {
            continue;
        }
        let Some(g) = glyphs.get(&c) else {
            // Missing glyph (atlas didn't fit it); silently skip.
            continue;
        };

        let x0 = cursor_x + g.bearing_x * scale;
        let y0 = cursor_y + (line_height_px + g.bearing_y) * scale;
        let x1 = x0 + g.px_width * scale;
        let y1 = y0 + g.px_height * scale;

        let (u0, v0) = (g.uv_min[0], g.uv_min[1]);
        let (u1, v1) = (g.uv_max[0], g.uv_max[1]);

        // Six vertices per glyph (two triangles).
        out.extend_from_slice(&[
            TextVertex {
                pos: [x0, y0],
                uv: [u0, v0],
                color,
            },
            TextVertex {
                pos: [x1, y0],
                uv: [u1, v0],
                color,
            },
            TextVertex {
                pos: [x0, y1],
                uv: [u0, v1],
                color,
            },
            TextVertex {
                pos: [x1, y0],
                uv: [u1, v0],
                color,
            },
            TextVertex {
                pos: [x1, y1],
                uv: [u1, v1],
                color,
            },
            TextVertex {
                pos: [x0, y1],
                uv: [u0, v1],
                color,
            },
        ]);

        cursor_x += g.h_advance * scale;
    }
}

// ---------------------------------------------------------------------------
// Atlas baking
// ---------------------------------------------------------------------------

fn bake_ascii_atlas(
    font: &FontRef<'_>,
    bake_size_px: f32,
) -> Result<(Vec<u8>, HashMap<char, GlyphEntry>, f32)> {
    let scaled = font.as_scaled(bake_size_px);
    let line_height = scaled.ascent() - scaled.descent() + scaled.line_gap();

    let mut atlas = vec![0u8; (ATLAS_SIZE * ATLAS_SIZE) as usize];
    let mut entries: HashMap<char, GlyphEntry> = HashMap::with_capacity(96);

    // Simple shelf packer: glyphs flow left-to-right; new shelf when
    // the current row is full.
    let pad = 1u32;
    let mut shelf_y: u32 = pad;
    let mut shelf_x: u32 = pad;
    let mut shelf_h: u32 = 0;

    for code in 0x20u32..=0x7E {
        let c = char::from_u32(code).unwrap();
        let gid: GlyphId = font.glyph_id(c);
        let h_adv = scaled.h_advance(gid);

        // Position glyph at origin so px_bounds is offset-from-origin
        // (we only need the size + offset, not absolute placement).
        let mut glyph: Glyph = scaled.scaled_glyph(c);
        glyph.position = Point { x: 0.0, y: 0.0 };

        let outlined = scaled.outline_glyph(glyph);
        match outlined {
            Some(o) => {
                let bounds = o.px_bounds();
                let gw = bounds.width().ceil() as u32;
                let gh = bounds.height().ceil() as u32;
                if gw == 0 || gh == 0 {
                    // Glyph has no rasterizable area (rare); record
                    // empty entry so cursor still advances correctly.
                    entries.insert(
                        c,
                        GlyphEntry {
                            uv_min: [0.0; 2],
                            uv_max: [0.0; 2],
                            px_width: 0.0,
                            px_height: 0.0,
                            h_advance: h_adv,
                            bearing_x: 0.0,
                            bearing_y: 0.0,
                        },
                    );
                    continue;
                }
                if shelf_x + gw + pad > ATLAS_SIZE {
                    shelf_y += shelf_h + pad;
                    shelf_x = pad;
                    shelf_h = 0;
                }
                if shelf_y + gh + pad > ATLAS_SIZE {
                    return Err(anyhow!(
                        "rye-text: ASCII atlas exceeded {ATLAS_SIZE}x{ATLAS_SIZE} at glyph {c:?}; \
                         reduce bake_size_px or extend the packer"
                    ));
                }

                // Rasterize into the atlas.
                let dst_x = shelf_x;
                let dst_y = shelf_y;
                o.draw(|gx, gy, cov| {
                    let px_x = dst_x + gx;
                    let px_y = dst_y + gy;
                    if px_x < ATLAS_SIZE && px_y < ATLAS_SIZE {
                        let idx = (px_y * ATLAS_SIZE + px_x) as usize;
                        // Saturate to u8.
                        let v = (cov * 255.0).round().clamp(0.0, 255.0) as u8;
                        atlas[idx] = atlas[idx].max(v);
                    }
                });

                let uv_min = [
                    dst_x as f32 / ATLAS_SIZE as f32,
                    dst_y as f32 / ATLAS_SIZE as f32,
                ];
                let uv_max = [
                    (dst_x + gw) as f32 / ATLAS_SIZE as f32,
                    (dst_y + gh) as f32 / ATLAS_SIZE as f32,
                ];

                entries.insert(
                    c,
                    GlyphEntry {
                        uv_min,
                        uv_max,
                        px_width: gw as f32,
                        px_height: gh as f32,
                        h_advance: h_adv,
                        bearing_x: bounds.min.x,
                        bearing_y: bounds.min.y,
                    },
                );

                shelf_x += gw + pad;
                shelf_h = shelf_h.max(gh);
            }
            None => {
                // Whitespace / no outline.
                entries.insert(
                    c,
                    GlyphEntry {
                        uv_min: [0.0; 2],
                        uv_max: [0.0; 2],
                        px_width: 0.0,
                        px_height: 0.0,
                        h_advance: h_adv,
                        bearing_x: 0.0,
                        bearing_y: 0.0,
                    },
                );
            }
        }
    }

    Ok((atlas, entries, line_height))
}

// ---------------------------------------------------------------------------
// WGSL
// ---------------------------------------------------------------------------

const WGSL_SHADER: &str = r#"
struct Uniforms {
    viewport_size: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var atlas_tex: texture_2d<f32>;
@group(0) @binding(2) var atlas_sam: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) in_pos: vec2<f32>,
    @location(1) in_uv: vec2<f32>,
    @location(2) in_color: vec4<f32>,
) -> VsOut {
    let ndc_x = (in_pos.x / u.viewport_size.x) * 2.0 - 1.0;
    // pixel y axis points down; NDC y axis points up; flip.
    let ndc_y = 1.0 - (in_pos.y / u.viewport_size.y) * 2.0;
    var out: VsOut;
    out.clip = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv = in_uv;
    out.color = in_color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let alpha = textureSample(atlas_tex, atlas_sam, in.uv).r;
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// Pure-CPU sanity check that atlas baking works on a real font.
    /// Doesn't require a wgpu device.
    #[test]
    fn baking_roundtrip_with_system_arial() {
        // Arial is reliably present on Windows. Skip cleanly if not.
        let path = std::path::Path::new(r"C:\Windows\Fonts\arial.ttf");
        if !path.exists() {
            eprintln!("skip: arial.ttf not present at {path:?}");
            return;
        }
        let bytes = std::fs::read(path).expect("read arial.ttf");
        let font = FontRef::try_from_slice(&bytes).expect("parse arial.ttf");
        let (atlas, glyphs, line_h) = bake_ascii_atlas(&font, 48.0).expect("bake");
        assert_eq!(atlas.len() as u32, ATLAS_SIZE * ATLAS_SIZE);
        // All printable ASCII should have an entry.
        assert_eq!(glyphs.len(), 0x7F - 0x20);
        // Line height should be positive and reasonable for 48px bake.
        assert!(line_h > 30.0 && line_h < 80.0, "line_h = {line_h}");
        // 'A' should have nonzero pixel size.
        let a = glyphs.get(&'A').expect("A in atlas");
        assert!(a.px_width > 0.0 && a.px_height > 0.0);
    }

    /// Build a minimal glyph table for layout tests: every printable ASCII char gets a
    /// unit-square glyph at the same UV slot. The math we want to pin (cursor advance,
    /// newline reset, vertex count) doesn't depend on the actual atlas geometry, only on
    /// per-glyph `h_advance`.
    fn mock_glyph_table(h_advance: f32) -> HashMap<char, GlyphEntry> {
        (0x20u8..=0x7Eu8)
            .map(|c| {
                (
                    c as char,
                    GlyphEntry {
                        uv_min: [0.0, 0.0],
                        uv_max: [1.0, 1.0],
                        bearing_x: 0.0,
                        bearing_y: 0.0,
                        px_width: 1.0,
                        px_height: 1.0,
                        h_advance,
                    },
                )
            })
            .collect()
    }

    /// Each printable glyph emits exactly 6 vertices (two triangles).
    /// "abc" produces 18 vertices.
    #[test]
    fn layout_emits_six_vertices_per_glyph() {
        let glyphs = mock_glyph_table(10.0);
        let mut out = Vec::new();
        layout_text(
            "abc",
            [0.0, 0.0],
            16.0,
            [1.0, 1.0, 1.0, 1.0],
            &glyphs,
            16.0,
            16.0,
            &mut out,
        );
        assert_eq!(out.len(), 18);
    }

    /// Newline resets `cursor_x` to `position[0]` and advances `cursor_y` by
    /// `line_height_px * scale`. Two-line text should produce vertices on two distinct
    /// y-bands.
    #[test]
    fn layout_newline_resets_x_and_advances_y() {
        let glyphs = mock_glyph_table(10.0);
        let mut out = Vec::new();
        // size_px = bake_size_px = 16.0, so scale = 1.0 and line_h = 16.0.
        layout_text(
            "a\nb",
            [5.0, 0.0],
            16.0,
            [1.0; 4],
            &glyphs,
            16.0,
            16.0,
            &mut out,
        );
        assert_eq!(out.len(), 12); // 6 verts × 2 glyphs

        // First glyph's top-left vertex sits at (cursor_x = 5, cursor_y + line_height).
        // After mock bearing_x = 0, bearing_y = 0: x0 = 5, y0 = 0 + (16 + 0)*1 = 16.
        let first = out[0];
        assert_eq!(first.pos[0], 5.0);
        assert!((first.pos[1] - 16.0).abs() < 1e-5);

        // Seventh vertex is the start of glyph 2 ('b'), after the newline.
        // cursor_x reset to 5 (position[0]); cursor_y advanced by line_h = 16.
        let second = out[6];
        assert_eq!(
            second.pos[0], 5.0,
            "newline must reset cursor_x to position[0]"
        );
        assert!(
            (second.pos[1] - 32.0).abs() < 1e-5,
            "newline must advance cursor_y by line_h ({})",
            16.0,
        );
    }

    /// Cursor advances by `h_advance * scale` per glyph, both horizontally on the
    /// baseline and through the resulting vertex positions.
    #[test]
    fn layout_cursor_advances_by_h_advance_scaled() {
        let glyphs = mock_glyph_table(10.0);
        let mut out = Vec::new();
        // Render at 32 px when bake is 16 px => scale = 2 => effective advance = 20
        // per glyph.
        layout_text(
            "ab",
            [0.0, 0.0],
            32.0,
            [1.0; 4],
            &glyphs,
            16.0,
            16.0,
            &mut out,
        );

        // Glyph 0 starts at x = 0; glyph 1 starts at x = 20 (one
        // scaled advance later). The top-left vertex of each glyph
        // is the first of its 6-vertex chunk.
        assert_eq!(out[0].pos[0], 0.0);
        assert_eq!(out[6].pos[0], 20.0);
    }

    /// Non-ASCII / control chars are skipped without crashing or emitting bogus vertices.
    /// Tabs, form-feeds, raw bytes 0x80+, emoji are all silently dropped.
    #[test]
    fn layout_skips_unprintable_and_out_of_range_chars() {
        let glyphs = mock_glyph_table(10.0);
        let mut out = Vec::new();
        layout_text(
            "a\tb\u{80}c😀d",
            [0.0, 0.0],
            16.0,
            [1.0; 4],
            &glyphs,
            16.0,
            16.0,
            &mut out,
        );
        // Only 'a', 'b', 'c', 'd' get glyphs. 4 × 6 = 24 vertices.
        assert_eq!(out.len(), 24);
    }

    /// Chars with no entry in the glyph table (because the atlas didn't fit them) are
    /// silently skipped, NOT rendered as missing-glyph fallback boxes. This keeps the
    /// layout deterministic when the atlas is partial.
    #[test]
    fn layout_skips_missing_glyphs() {
        let mut glyphs = mock_glyph_table(10.0);
        glyphs.remove(&'b');
        let mut out = Vec::new();
        layout_text(
            "ab",
            [0.0, 0.0],
            16.0,
            [1.0; 4],
            &glyphs,
            16.0,
            16.0,
            &mut out,
        );
        // Only 'a' produces 6 vertices.
        assert_eq!(out.len(), 6);
    }

    /// `WGSL_SHADER` is the shader module string the GPU pipeline loads. A naga-front
    /// parse + validate pass catches syntax / type / binding errors at test time rather
    /// than at first `TextRenderer::new` call (which needs a wgpu adapter).
    #[test]
    fn wgsl_shader_validates_via_naga() {
        let module = naga::front::wgsl::parse_str(WGSL_SHADER).expect("WGSL parse");
        let flags = naga::valid::ValidationFlags::all();
        let caps = naga::valid::Capabilities::empty();
        naga::valid::Validator::new(flags, caps)
            .validate(&module)
            .expect("WGSL validate");
    }
}
