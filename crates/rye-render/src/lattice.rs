//! Pixel-space layout for render-pass viewports.
//!
//! Lets a render node draw to a sub-region of the framebuffer
//! instead of fullscreen, for cases like an egui side-panel
//! occluding the left strip of the window.
//!
//! The minimal abstraction: a [`Viewport`] is a rectangle in
//! pixel coordinates (origin top-left, +y down). Render nodes
//! that consume one apply `wgpu::RenderPass::set_viewport`
//! before drawing; the kernel reads the viewport's `width` /
//! `height` for aspect-correct projection.
//!
//! Future "lattice" extensions (named regions, named layout
//! presets, render-graph integration) layer on top.

/// A pixel rectangle within a framebuffer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Viewport {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Viewport {
    /// The whole framebuffer.
    pub fn full(framebuffer: [u32; 2]) -> Self {
        Self {
            x: 0,
            y: 0,
            width: framebuffer[0],
            height: framebuffer[1],
        }
    }

    /// The region to the right of a left-side panel of width
    /// `panel_width` pixels. Returns the empty viewport if the
    /// panel covers the framebuffer (degenerate, but well-defined).
    pub fn right_of_left_panel(panel_width: u32, framebuffer: [u32; 2]) -> Self {
        let panel = panel_width.min(framebuffer[0]);
        Self {
            x: panel,
            y: 0,
            width: framebuffer[0] - panel,
            height: framebuffer[1],
        }
    }

    /// Apply this viewport to a render pass. The depth range is set
    /// to the standard `[0.0, 1.0]`.
    pub fn apply(&self, rp: &mut wgpu::RenderPass<'_>) {
        rp.set_viewport(
            self.x as f32,
            self.y as f32,
            self.width as f32,
            self.height as f32,
            0.0,
            1.0,
        );
    }

    /// `[width, height]` as `[f32; 2]`, the format the hyperslice
    /// kernel's `u.resolution` uniform expects.
    pub fn resolution_f32(&self) -> [f32; 2] {
        [self.width as f32, self.height as f32]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_covers_framebuffer() {
        let v = Viewport::full([1280, 720]);
        assert_eq!(v.x, 0);
        assert_eq!(v.y, 0);
        assert_eq!(v.width, 1280);
        assert_eq!(v.height, 720);
    }

    #[test]
    fn right_of_left_panel_carves_correctly() {
        let v = Viewport::right_of_left_panel(300, [1280, 720]);
        assert_eq!(v.x, 300);
        assert_eq!(v.y, 0);
        assert_eq!(v.width, 980);
        assert_eq!(v.height, 720);
    }

    #[test]
    fn right_of_left_panel_clamps_when_panel_exceeds_framebuffer() {
        let v = Viewport::right_of_left_panel(2000, [1280, 720]);
        assert_eq!(v.x, 1280);
        assert_eq!(v.width, 0);
    }
}
