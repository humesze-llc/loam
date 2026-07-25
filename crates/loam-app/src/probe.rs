//! Presented-frame probe for display-chain diagnosis. Every frame gets a
//! visible watermark encoding its index, and every post-UI swapchain frame
//! is read back to a PNG under `probe_frames/`. An OS-side screen grabber
//! captures the same watermarked frames; diffing buffer-vs-screen for equal
//! frame indices splits engine defects from composition/driver corruption.
//!
//! Enable with `--probe-secs=<f32>`; the process exits when the window
//! elapses. The sync readback stalls the pipeline, so probe timing is not
//! representative; that is acceptable because the comparison is per-frame
//! content, not cadence.

use std::path::PathBuf;
use std::time::Instant;

use crate::egui;
use loam_render::device::RenderDevice;

/// Watermark geometry, in physical pixels at `pixels_per_point == 1`.
/// 40 cells: 8 anchor bits (`ANCHOR`, LSB first) then a 32-bit frame index.
pub(crate) const MARK_ORIGIN: (f32, f32) = (8.0, 40.0);
pub(crate) const MARK_CELL: f32 = 8.0;
pub(crate) const MARK_H: f32 = 16.0;
pub(crate) const MARK_BITS: u64 = 40;
pub(crate) const ANCHOR: u64 = 0b1010_0101;

pub(crate) struct Probe {
    dir: PathBuf,
    started: Option<Instant>,
    secs: f32,
    frame: u32,
}

impl Probe {
    pub(crate) fn from_args() -> Option<Self> {
        let secs = crate::args::Args::current().parse::<f32>("probe-secs")?;
        let dir = PathBuf::from("probe_frames");
        if let Err(e) = std::fs::create_dir_all(&dir) {
            tracing::error!("probe: create {dir:?} failed: {e}");
            return None;
        }
        tracing::info!("probe: dumping every presented frame to {dir:?} for {secs}s");
        Some(Self {
            dir,
            started: None,
            secs,
            frame: 0,
        })
    }

    pub(crate) fn overlay(&self, ctx: &egui::Context) {
        if ctx.pixels_per_point() != 1.0 {
            // Decoder coordinates assume 1pt = 1px; a scaled watermark would
            // pair frames incorrectly rather than fail loudly.
            tracing::warn!(
                "probe: pixels_per_point = {}, watermark decode will be wrong",
                ctx.pixels_per_point()
            );
        }
        let origin = egui::pos2(MARK_ORIGIN.0, MARK_ORIGIN.1);
        let word = ANCHOR | ((self.frame as u64) << 8);
        egui::Area::new(egui::Id::new("loam-probe-mark"))
            .fixed_pos(origin)
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                let painter = ui.painter();
                painter.rect_filled(
                    egui::Rect::from_min_size(
                        origin,
                        egui::vec2(MARK_BITS as f32 * MARK_CELL, MARK_H),
                    ),
                    0.0,
                    egui::Color32::BLACK,
                );
                for i in 0..MARK_BITS {
                    if (word >> i) & 1 == 1 {
                        let x = origin.x + i as f32 * MARK_CELL;
                        painter.rect_filled(
                            egui::Rect::from_min_size(
                                egui::pos2(x + 1.0, origin.y + 1.0),
                                egui::vec2(MARK_CELL - 2.0, MARK_H - 2.0),
                            ),
                            0.0,
                            egui::Color32::WHITE,
                        );
                    }
                }
            });
    }

    /// Read back and write this frame; true once the probe window elapsed.
    pub(crate) fn consume(&mut self, rd: &RenderDevice, texture: &wgpu::Texture) -> bool {
        let started = *self.started.get_or_insert_with(Instant::now);
        match crate::capture::read_texture_rgba(
            &rd.device,
            &rd.queue,
            texture,
            rd.surface_bundle.size.width,
            rd.surface_bundle.size.height,
            rd.surface_bundle.config.format,
        ) {
            Ok(img) => {
                let path = self.dir.join(format!("f{:06}.png", self.frame));
                if let Err(e) = write_png(&path, img.width, img.height, &img.rgba) {
                    tracing::error!("probe: write {path:?} failed: {e:#}");
                }
            }
            Err(e) => tracing::error!("probe: readback failed: {e:#}"),
        }
        self.frame = self.frame.wrapping_add(1);
        let done = started.elapsed().as_secs_f32() >= self.secs;
        if done {
            tracing::info!("probe: complete, {} frames in {:?}", self.frame, self.dir);
        }
        done
    }
}

fn write_png(path: &std::path::Path, width: u32, height: u32, rgba: &[u8]) -> anyhow::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(rgba)?;
    Ok(())
}
