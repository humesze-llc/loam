//! GPU rendering for `Demo`: the wireframe overlay, point cloud, and the
//! rasterized section-face passes.
//!
//! Each pass splits in two: a CPU mesh build over the frame's
//! [`state::RowFrame`], written as a free function so it runs (and is pinned)
//! without a device, and the upload + execute half that needs one.

use crate::*;

impl Demo {
    pub(crate) fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        // Scene renders to the full window; the overlay and Render modal float on
        // top without reserving pixels, so the viewport is always the framebuffer.
        let cfg = &rd.surface_bundle.config;
        let viewport = Viewport::full([cfg.width, cfg.height]);
        if self.view_mode == ViewMode::Filmstrip {
            // Each cell shows the `strip_subject` at a different `w_slice`. Swap
            // the GPU body list to just the subject for this render, then restore
            // via `rebuild_bodies` so later state reads see the full row.
            let entry = self.strip_subject;
            // 2D grid: w on columns, t on rows by default (`strip_swap_axes`
            // flips it). A 1D case collapses the second axis to one cell.
            let strip_w_extent = self.effective_body_size();
            let (cols, rows, w_on_cols) = match (self.strip_w, self.strip_t) {
                (true, true) => {
                    if self.strip_swap_axes {
                        (self.strip_count_t, self.strip_count_w, false)
                    } else {
                        (self.strip_count_w, self.strip_count_t, true)
                    }
                }
                (true, false) => (self.strip_count_w, 1, true),
                (false, true) => (1, self.strip_count_t, false),
                // UI invariant prevents both being off.
                (false, false) => (1, 1, true),
            };
            let col_vps = viewport.split_horizontal(cols as u32);
            let mut grid_cells: Vec<(Viewport, f32, BodyUniform)> = Vec::with_capacity(cols * rows);
            for (col_idx, col_vp) in col_vps.into_iter().enumerate() {
                let row_vps = col_vp.split_vertical(rows as u32);
                for (row_idx, cell_vp) in row_vps.into_iter().enumerate() {
                    // (w_offset, t_offset) for this cell, by which axis carries
                    // which dimension.
                    let (w_idx, w_n, t_idx, t_n) = if w_on_cols {
                        (col_idx, cols, row_idx, rows)
                    } else {
                        (row_idx, rows, col_idx, cols)
                    };
                    let w_t = if w_n <= 1 {
                        0.5
                    } else {
                        w_idx as f32 / (w_n - 1) as f32
                    };
                    let w_offset = -strip_w_extent + w_t * (2.0 * strip_w_extent);
                    let cell_w_slice = self.w_slice + w_offset;
                    let t_offset = if !self.strip_t || t_n <= 1 {
                        0.0
                    } else {
                        // Fan forward only: cell 0 = now, last = rot_time +
                        // strip_t_extent ("the rotor at this future time").
                        let t_norm = t_idx as f32 / (t_n - 1) as f32;
                        t_norm * self.strip_t_extent
                    };
                    // Cell's rotor: the orientation at `rot_time + t_offset`, via
                    // the same `rotor_at_time` dispatch the spin + t-scrub use.
                    let cell_rotor = if t_offset == 0.0 {
                        self.rot_state
                    } else {
                        self.rotor_at_time(self.rot_time + t_offset)
                    };
                    let body = BodyUniform::polytope_with_rotor(
                        [0.0, BODY_Y, 0.0, 0.0],
                        entry.shape.shape_id(),
                        self.effective_body_size(),
                        cell_rotor,
                        entry.body_color,
                    );
                    grid_cells.push((cell_vp, cell_w_slice, body));
                }
            }
            let result = self.node.execute_strip(rd, view, &grid_cells);
            // Restore the full row for any non-strip consumer.
            self.rebuild_bodies();
            result
        } else {
            {
                let _scope = loam_time::frame_trace::scope("pp-sdf");
                {
                    let u = self.node.uniforms_mut();
                    u.resolution = viewport.resolution_f32();
                    u.viewport_origin = [viewport.x as f32, viewport.y as f32];
                }
                self.node.flush_uniforms(&rd.queue);
                self.node.execute_in_viewport(rd, view, viewport)?;
            }
            // Shared depth for the section pass + the wireframe's depth-test.
            // Order: SDF (color only) -> section_faces (writes depth in Raster) ->
            // wireframe (tests, no write). In SDF mode no pass writes depth, so the
            // cleared `1.0` lets every wireframe fragment pass.
            self.ensure_and_clear_shared_depth(rd)?;
            if matches!(self.surface_mode, SurfaceMode::Raster) {
                let _scope = loam_time::frame_trace::scope("pp-section-faces");
                self.render_section_faces(rd, view)?;
            }
            // Cross-section + wireframe overlay. Shapes view only: Filmstrip's
            // per-cell composition would need per-cell depth-clear + uploads not
            // worth the v1 plumbing.
            if self.wireframe_enabled {
                let _scope = loam_time::frame_trace::scope("pp-wireframe");
                self.render_wireframe_overlay(rd, view)?;
            }
            // Points overlay, drawn last so discs sit on top of edges and caps.
            if self.points_enabled {
                let _scope = loam_time::frame_trace::scope("pp-points");
                self.render_points(rd, view)?;
            }
            Ok(())
        }
    }

    /// Ensure the shared section-faces depth attachment exists at the current
    /// swapchain size + sample count, then clear it to `1.0`. Shared between
    /// `section_faces` (writes depth in Raster) and `parent_wireframe` (tests, no
    /// write), so one ensure + clear per frame covers both.
    fn ensure_and_clear_shared_depth(&mut self, rd: &RenderDevice) -> Result<()> {
        let cfg = &rd.surface_bundle.config;
        DepthBuffer::ensure(
            &mut self.section_faces_depth,
            &rd.device,
            SECTION_FACES_DEPTH_FORMAT,
            (cfg.width, cfg.height),
            rd.sample_count(),
        );
        let depth = self
            .section_faces_depth
            .as_ref()
            .expect("ensure() guarantees Some");
        let mut encoder = rd
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shared depth clear"),
            });
        let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shared depth clear pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rd.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Build the point sprites mesh ([`build_points_mesh`]), upload it, and
    /// execute the point-disc raster pass.
    fn render_points(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        let cfg = &rd.surface_bundle.config;
        let style = PointsStyle {
            color_mode: self.wireframe_color_mode,
            show_vertices: self.points_show_vertices,
            show_cell_centers: self.points_show_cell_centers,
            size_px: self.points_size_px,
        };
        // Mesh + scratch taken out of `self` for the build: the row frame
        // borrows the physics world for as long as the buffers it fills are
        // borrowed. Put back so their capacity persists across frames.
        let mut mesh = std::mem::take(&mut self.points_mesh_scratch);
        let mut centers_cache = std::mem::take(&mut self.cell_centers_cache);
        let mut local_vertices = std::mem::take(&mut self.overlay_local_vertices_scratch);
        let mut center_locals = std::mem::take(&mut self.overlay_center_locals_scratch);
        let mut cell_strengths = std::mem::take(&mut self.overlay_cell_strengths_scratch);
        build_points_mesh(
            &self.row_frame(),
            &style,
            &mut centers_cache,
            &mut local_vertices,
            &mut center_locals,
            &mut cell_strengths,
            &mut mesh,
        );
        self.cell_centers_cache = centers_cache;
        self.overlay_local_vertices_scratch = local_vertices;
        self.overlay_center_locals_scratch = center_locals;
        self.overlay_cell_strengths_scratch = cell_strengths;

        // Camera matches the wireframe overlay / section faces.
        let view_dir = self.camera.view();
        let aspect = cfg.width as f32 / cfg.height as f32;
        let view_mat = Mat4::look_to_rh(view_dir.position, view_dir.forward, view_dir.up);
        let proj_mat = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.1, 100.0);
        let view_proj = proj_mat * view_mat;
        let vp_size = Vec2::new(cfg.width as f32, cfg.height as f32);
        self.points_node.set_camera(&rd.queue, view_proj, vp_size);
        self.points_node.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &mesh,
            &loam_math::Projection::Identity,
        );
        self.points_mesh_scratch = mesh;
        // No depth attachment: see `PointRasterNode::new` (drop-w + ReadOnly
        // LessEqual occluded non-w=0 vertices behind their own caps).
        self.points_node.execute(rd, view, None, None)?;
        Ok(())
    }

    /// Render the rasterized section as TWO independent overlaid layers in one
    /// viewport:
    ///
    /// - the honest cross-section (the drop-w slice 3-flat, NEVER reprojected
    ///   through the active wireframe projection; the same geometry the SDF
    ///   raymarch shows), and
    /// - the projected cap (the same slice reprojected through the active
    ///   wireframe projection so it can sit on a Schlegel / stereographic
    ///   wireframe).
    ///
    /// Each layer's fill alpha is its own switch (`SectionLayer::fill_visible`):
    /// a layer with alpha 0 submits no triangles. The honest layer draws first so
    /// the opt-in projected cap composites over it when both are on. Defaults draw
    /// only the honest layer, so selecting a distorting projection never silently
    /// reshapes the slice the user reads as "the cross-section."
    fn render_section_faces(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        let cfg = &rd.surface_bundle.config;
        let cross = self.cross_section;
        let cap = self.projected_cap;
        // Both layers off: perimeter outlines belong to the wireframe overlay,
        // so an all-alpha-zero section skips the triangle passes entirely.
        if !cross.fill_visible() && !cap.fill_visible() {
            return Ok(());
        }

        // Scratch + layer meshes taken out of `self` for the build: the row
        // frame borrows the physics world for as long as they are borrowed. Put
        // back so their capacity persists across frames.
        let mut local_vertices = std::mem::take(&mut self.section_world_vertices_scratch);
        let mut proj_scratch = std::mem::take(&mut self.section_clip_projected_scratch);
        let mut cross_mesh = std::mem::take(&mut self.section_faces_mesh_scratch);
        let mut cap_mesh = std::mem::take(&mut self.section_faces_projected_scratch);
        let mut section_scratch = std::mem::take(&mut self.section_cap_scratch);
        build_section_layer_meshes(
            &self.row_frame(),
            cross,
            cap,
            SectionBuffers {
                local_vertices: &mut local_vertices,
                proj_scratch: &mut proj_scratch,
                cross_mesh: &mut cross_mesh,
                cap_mesh: &mut cap_mesh,
                section_scratch: &mut section_scratch,
            },
        );
        self.section_cap_scratch = section_scratch;
        self.section_world_vertices_scratch = local_vertices;
        self.section_clip_projected_scratch = proj_scratch;
        self.section_faces_mesh_scratch = cross_mesh;
        self.section_faces_projected_scratch = cap_mesh;

        // Camera matches the SDF raymarcher, for pixel-aligned composition.
        let view_dir = self.camera.view();
        let aspect = cfg.width as f32 / cfg.height as f32;
        let view_mat = Mat4::look_to_rh(view_dir.position, view_dir.forward, view_dir.up);
        let proj_mat = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.1, 100.0);
        let view_proj = proj_mat * view_mat;

        // Honest cross-section first, then the projected cap on top.
        if cross.fill_visible() {
            self.execute_section_layer(rd, view, view_proj, cross.surface_alpha, true)?;
        }
        if cap.fill_visible() {
            self.execute_section_layer(rd, view, view_proj, cap.surface_alpha, false)?;
        }
        Ok(())
    }

    /// Upload + execute one built section layer's mesh. Picks the opaque vs
    /// translucent node by `alpha`: opaque (>= 1.0) writes depth so caps occlude
    /// within a polytope; translucent (< 1.0) skips depth-write so layers behind
    /// show through. `is_cross_section` selects the scratch mesh.
    fn execute_section_layer(
        &mut self,
        rd: &RenderDevice,
        view: &wgpu::TextureView,
        view_proj: Mat4,
        alpha: f32,
        is_cross_section: bool,
    ) -> Result<()> {
        // Disjoint field borrows let the immutable depth + scratch reads coexist
        // with the `&mut` node. The shared depth is ensured + cleared per frame by
        // `ensure_and_clear_shared_depth`; here we consume its view.
        let depth_view = &self
            .section_faces_depth
            .as_ref()
            .expect("shared depth buffer must be ensured before section_faces")
            .view;
        let mesh = if is_cross_section {
            &self.section_faces_mesh_scratch
        } else {
            &self.section_faces_projected_scratch
        };
        // Empty-mesh short-circuit lives in `TriangleRasterNode::execute`.
        let node = if alpha >= 1.0 {
            &mut self.section_faces
        } else {
            &mut self.section_faces_translucent
        };
        node.set_camera(&rd.queue, view_proj);
        node.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            mesh,
            &loam_math::Projection::Identity,
        );
        node.execute(rd, view, Some(depth_view), None)?;
        Ok(())
    }

    /// Distance from the camera eye to the orbit target, used to scale the
    /// stereographic clip radius (see [`stereographic_view_radius`]). Reads the
    /// live eye position, not `orbit.distance`, so it is correct in FreeRoam too.
    pub(crate) fn camera_distance_to_focus(&self) -> f32 {
        (self.camera.position - self.orbit.target).length()
    }

    /// Build the section-perimeter and parent-wireframe overlay meshes
    /// ([`build_wireframe_meshes`]), upload them, and execute the raster passes
    /// over the SDF render.
    fn render_wireframe_overlay(
        &mut self,
        rd: &RenderDevice,
        view: &wgpu::TextureView,
    ) -> Result<()> {
        let cfg = &rd.surface_bundle.config;
        let style = WireframeStyle {
            color_mode: self.wireframe_color_mode,
            alpha: self.wireframe_alpha,
            width_px: self.wireframe_width_px,
            nearest_active: self.wireframe_nearest_active,
            // Edge geometry is projection-derived: Stereographic draws S³ arcs,
            // the affine projections draw flat R4 chords.
            space_blend: state::default_edge_blend(self.wireframe_projection),
            hyperslice: self
                .hyperslice_cull_active()
                .then_some(self.wireframe_hyperslice_thickness),
        };
        let cross = self.cross_section;
        let cap = self.projected_cap;
        // Cache, buffers and both meshes taken out of `self` for the build: the
        // row frame borrows the physics world for as long as they are borrowed.
        // Put back so their capacity persists across frames.
        let mut palette_cache = std::mem::take(&mut self.unique_edge_palette_cache);
        let mut slerp_scratch = std::mem::take(&mut self.slerp_scratch);
        let mut local_vertices = std::mem::take(&mut self.overlay_local_vertices_scratch);
        let mut cell_strengths = std::mem::take(&mut self.overlay_cell_strengths_scratch);
        let mut section_scratch = std::mem::take(&mut self.section_cap_scratch);
        let mut body_perimeter = std::mem::take(&mut self.body_perimeter_scratch);
        let mut section_edges = std::mem::take(&mut self.wireframe_section_edges_scratch);
        let mut parent_lines = std::mem::take(&mut self.wireframe_parent_lines_scratch);
        build_wireframe_meshes(
            &self.row_frame(),
            &style,
            cross,
            cap,
            &mut palette_cache,
            &mut slerp_scratch,
            &mut local_vertices,
            &mut cell_strengths,
            &mut section_scratch,
            &mut body_perimeter,
            &mut section_edges,
            &mut parent_lines,
        );
        self.unique_edge_palette_cache = palette_cache;
        self.slerp_scratch = slerp_scratch;
        self.overlay_local_vertices_scratch = local_vertices;
        self.overlay_cell_strengths_scratch = cell_strengths;
        self.section_cap_scratch = section_scratch;
        self.body_perimeter_scratch = body_perimeter;

        // Upload (no-op when a mesh is empty).
        self.section_edges.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &section_edges,
            &loam_math::Projection::Identity,
            1,
        );
        self.parent_wireframe.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &parent_lines,
            &loam_math::Projection::Identity,
            1,
        );
        self.wireframe_section_edges_scratch = section_edges;
        self.wireframe_parent_lines_scratch = parent_lines;

        // Same view+projection as the SDF raymarcher, so the overlay aligns
        // pixel-for-pixel.
        let view_dir = self.camera.view();
        let aspect = cfg.width as f32 / cfg.height as f32;
        let view_mat = Mat4::look_to_rh(view_dir.position, view_dir.forward, view_dir.up);
        let proj_mat = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.1, 100.0);
        let view_proj = proj_mat * view_mat;
        let vp_size = Vec2::new(cfg.width as f32, cfg.height as f32);
        self.section_edges.set_camera(&rd.queue, view_proj, vp_size);
        self.parent_wireframe
            .set_camera(&rd.queue, view_proj, vp_size);

        // Perimeter edges then parent wireframe, both depth-testing (no write)
        // against the shared section-faces depth so lines behind a cap occlude
        // correctly. In SDF mode the cleared `1.0` lets every fragment pass.
        let depth_view = self
            .section_faces_depth
            .as_ref()
            .map(|b| &b.view)
            .expect("shared depth buffer must be ensured before wireframe overlay");
        self.section_edges
            .execute(rd, view, Some(depth_view), None)?;
        self.parent_wireframe
            .execute(rd, view, Some(depth_view), None)?;
        Ok(())
    }
}

/// Style inputs for [`build_points_mesh`] that do not come from the bodies.
///
/// Sprites honor the active [`WireframeColorMode`] so the overlay reads as part
/// of the wireframe pass; `UniqueEdge` falls back to the position gradient (the
/// edge-indexed palette has no canonical vertex assignment).
#[derive(Copy, Clone)]
pub(crate) struct PointsStyle {
    pub(crate) color_mode: WireframeColorMode,
    pub(crate) show_vertices: bool,
    pub(crate) show_cell_centers: bool,
    pub(crate) size_px: f32,
}

/// Fill `mesh` with the vertex-marker and cell-center sprites of every
/// polychoral body in the row, in world R³. Cleared on entry, reusing the
/// caller's allocation. Same body-local project-then-translate pattern as the
/// wireframe and section-faces paths.
///
/// Every buffer is the caller's: `centers_cache` memoizes the topology-only
/// cell centroids, and the three scratch buffers are refilled per body. On the
/// 240 fps path nothing here reaches the allocator once they are warm.
///
/// Free function over [`RowFrame`] so "the sprites sit on the body physics put
/// there" is unit-testable without a GPU-backed [`Demo`];
/// [`Demo::render_points`] is the one production caller.
pub(crate) fn build_points_mesh(
    frame: &RowFrame<'_>,
    style: &PointsStyle,
    centers_cache: &mut std::collections::HashMap<loam_physics::polytope::Polytope4, Vec<Vec4>>,
    local_vertices: &mut Vec<Vec4>,
    center_locals: &mut Vec<Vec4>,
    cell_strengths: &mut Vec<f32>,
    mesh: &mut loam_shape::PointMesh<3>,
) {
    // Active-mode palette: green for vertices in an intersected cell, gray
    // otherwise. Same hues as the wireframe overlay.
    const ACTIVE_GREEN: [f32; 4] = [0.40, 1.00, 0.55, 1.0];
    const INACTIVE_GRAY: [f32; 4] = [0.55, 0.55, 0.58, 0.85];

    mesh.positions.clear();
    mesh.colors.clear();
    mesh.sizes.clear();

    for (slot, entry) in frame.row.iter().enumerate() {
        let Some(polytope) = entry.shape.polytope4() else {
            continue;
        };
        // Near-pole drop radius shared with the edges and cap outline: a point in
        // the pole band would draw as a giant clamp disc while its edges drop, so
        // the same predicate gates it. Resolved per shape (only 16-cell clipped).
        let points_clip_radius = stereographic_clip_radius(
            &frame.projection,
            stereographic_view_radius(polytope, frame.camera_distance),
        );
        let topo = polytope.topology();

        // Body-local 4D vertices (rotated + scaled), shared by the vertex and
        // cell-center loops (WDepth normalization, Active cell strengths).
        let body_pos_r3 = frame.body_local(slot, topo.vertices, frame.body_size, local_vertices);
        // Canonical-max-w normalization (see build_wireframe_meshes), so the
        // points' w-depth color stays in step with the edges.
        let w_extent_local: f32 = if matches!(style.color_mode, WireframeColorMode::WDepth) {
            let canonical_max_w = topo
                .vertices
                .iter()
                .map(|v| v.w.abs())
                .fold(0.0_f32, f32::max)
                .max(1e-6);
            canonical_max_w * frame.body_size
        } else {
            1.0
        };
        // Cell strengths only for Active mode (`compute_cell_strengths`, same as
        // the wireframe overlay).
        if matches!(style.color_mode, WireframeColorMode::Active) {
            compute_cell_strengths(topo.cells, local_vertices, frame.w_slice, cell_strengths);
        } else {
            cell_strengths.clear();
        }
        let vertex_is_active = |vi: usize| -> bool {
            topo.cells
                .iter()
                .zip(cell_strengths.iter())
                .any(|(cell, &s)| s > 0.0 && cell.contains(&(vi as u32)))
        };

        if style.show_vertices {
            for (vi, v) in topo.vertices.iter().enumerate() {
                let v_local = local_vertices[vi];
                let v3_local =
                    <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(
                        v_local,
                        &frame.projection,
                    );
                // Drop a near-pole vertex (clean blink) instead of a giant disc.
                if !sample_in_radius(v3_local, points_clip_radius) {
                    continue;
                }
                let v_world = v3_local + body_pos_r3;
                let color = match style.color_mode {
                    // Position covers UniqueEdge too (edge-indexed palette has no
                    // per-vertex assignment).
                    WireframeColorMode::VertexGradient | WireframeColorMode::UniqueEdge => {
                        vertex_color_by_position(*v)
                    }
                    WireframeColorMode::WDepth => w_depth_color(v_local.w, w_extent_local),
                    WireframeColorMode::Active => {
                        if vertex_is_active(vi) {
                            ACTIVE_GREEN
                        } else {
                            INACTIVE_GRAY
                        }
                    }
                };
                mesh.positions.push(v_world.to_array());
                mesh.colors.push(color);
                mesh.sizes.push(style.size_px);
            }
        }
        if style.show_cell_centers {
            // Pull centroids inward by `CELL_CENTER_INSET` so they read as
            // interior markers. `cell_centers()` returns centroids at the
            // inradius, which is the DUAL's vertex set; at full inradius the
            // sprites look like the wrong polytope, so inset them inside the cap.
            const CELL_CENTER_INSET: f32 = 0.5;
            let centers: &[Vec4] = centers_cache
                .entry(polytope)
                .or_insert_with(|| polytope.cell_centers());
            frame.body_local(
                slot,
                centers,
                frame.body_size * CELL_CENTER_INSET,
                center_locals,
            );
            for (ci, c) in centers.iter().enumerate() {
                let c_local = center_locals[ci];
                let c3_local =
                    <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(
                        c_local,
                        &frame.projection,
                    );
                // Same near-pole drop as the vertex loop above.
                if !sample_in_radius(c3_local, points_clip_radius) {
                    continue;
                }
                let c_world = c3_local + body_pos_r3;
                let color = match style.color_mode {
                    WireframeColorMode::VertexGradient | WireframeColorMode::UniqueEdge => {
                        vertex_color_by_position(*c)
                    }
                    WireframeColorMode::WDepth => w_depth_color(c_local.w, w_extent_local),
                    WireframeColorMode::Active => {
                        let s = cell_strengths.get(ci).copied().unwrap_or(0.0);
                        if s > 0.0 {
                            ACTIVE_GREEN
                        } else {
                            INACTIVE_GRAY
                        }
                    }
                };
                mesh.positions.push(c_world.to_array());
                mesh.colors.push(color);
                // Half-sized so they don't compete with the vertex discs.
                mesh.sizes.push(style.size_px * 0.5);
            }
        }
    }
}

/// The caller-owned working set [`build_section_layer_meshes`] fills. Bundled
/// rather than passed one by one because they are taken from `Demo` and
/// restored together, so a struct keeps them from drifting apart at a call
/// site.
pub(crate) struct SectionBuffers<'a> {
    pub(crate) local_vertices: &'a mut Vec<Vec4>,
    pub(crate) proj_scratch: &'a mut Vec<Vec3>,
    pub(crate) cross_mesh: &'a mut loam_shape::TriangleMesh<3>,
    pub(crate) cap_mesh: &'a mut loam_shape::TriangleMesh<3>,
    pub(crate) section_scratch: &'a mut SectionScratch,
}

/// Append every polychoral body's cross-section caps into the visible layer
/// meshes, each under its own projection ([`state::section_layer_projection`]:
/// drop-w for the honest cross-section, the active projection for the cap).
/// Both meshes are cleared on entry and reuse their allocations; an invisible
/// layer is skipped.
///
/// Single pass over the row: the body-local 4D section vertices are identical
/// for both layers, computed once per body before the per-layer transform.
///
/// Free function over [`RowFrame`] so "the caps are cut from the body physics
/// put there" is unit-testable without a GPU-backed [`Demo`];
/// [`Demo::render_section_faces`] is the one production caller.
pub(crate) fn build_section_layer_meshes(
    frame: &RowFrame<'_>,
    cross: state::SectionLayer,
    cap: state::SectionLayer,
    buffers: SectionBuffers<'_>,
) {
    let SectionBuffers {
        local_vertices,
        proj_scratch,
        cross_mesh,
        cap_mesh,
        section_scratch,
    } = buffers;
    let w_slice = frame.w_slice;
    // Honest layer is drop-w (Identity makes `perspective_scale_at_w` report
    // `Some(1.0)`, a scale-by-one + translate); the cap follows the active
    // projection.
    let cross_projection = state::section_layer_projection(true, frame.projection);
    let cap_projection = state::section_layer_projection(false, frame.projection);
    let cross_scale = perspective_scale_at_w(w_slice, &cross_projection);
    let cap_scale = perspective_scale_at_w(w_slice, &cap_projection);

    cross_mesh.vertices.clear();
    cross_mesh.colors.clear();
    cross_mesh.indices.clear();
    cap_mesh.vertices.clear();
    cap_mesh.colors.clear();
    cap_mesh.indices.clear();

    for (slot, entry) in frame.row.iter().enumerate() {
        let Some(polytope) = entry.shape.polytope4() else {
            continue;
        };
        // Near-pole drop radius, per shape: finite for the 16-cell, none for the
        // rest; the cross layer is drop-w so it never clips.
        let view_radius = stereographic_view_radius(polytope, frame.camera_distance);
        let cross_clip = stereographic_clip_radius(&cross_projection, view_radius);
        let cap_clip = stereographic_clip_radius(&cap_projection, view_radius);
        let topo = polytope.topology();

        // Body-local 4D section vertices (no world translate): keep the body's
        // R³ position out of the perspective divide. Shared by both layers.
        let body_pos_r3 = frame.body_local(slot, topo.vertices, frame.body_size, local_vertices);
        let cap_vertices: &[Vec4] = local_vertices;

        // Match the SDF's per-body coloring: catalog color, Lambert depth.
        // Alpha is the layer's `surface_alpha`; below 1.0 it renders through the
        // no-depth-write pipeline so layers behind composite through.
        let [r, g, b] = entry.body_color;

        // Append + world-transform one layer's caps.
        // `cap_vertex_projected_and_world` maps each body-local cap vertex
        // through the layer's projection and returns the projected point the
        // clip tests. Under Stereographic a fill triangle is dropped when any
        // vertex exceeds `clip_radius`, matching the perimeter rule so fill and
        // outline cull in lockstep.
        let append_layer = |mesh: &mut loam_shape::TriangleMesh<3>,
                            proj_scratch: &mut Vec<Vec3>,
                            scratch: &mut SectionScratch,
                            alpha: f32,
                            projection: &loam_math::Projection<4>,
                            scale: Option<f32>,
                            clip_radius: Option<f32>| {
            let start_v = mesh.vertices.len();
            let start_i = mesh.indices.len();
            polytope_section_faces_append(
                topo.edges,
                topo.cells,
                cap_vertices,
                WPlane::new(w_slice),
                [r, g, b, alpha],
                scratch,
                mesh,
            );
            proj_scratch.clear();
            for v in &mut mesh.vertices[start_v..] {
                let (projected, world) =
                    cap_vertex_projected_and_world(*v, w_slice, scale, projection, body_pos_r3);
                *v = world;
                proj_scratch.push(projected);
            }
            // Drop fill triangles touching a near-pole vertex (no-op for the
            // affine `None` layers).
            retain_in_radius_triangles(
                &mut mesh.indices,
                start_i,
                start_v,
                proj_scratch,
                clip_radius,
            );
        };

        if cross.fill_visible() {
            append_layer(
                cross_mesh,
                proj_scratch,
                section_scratch,
                cross.surface_alpha,
                &cross_projection,
                cross_scale,
                cross_clip,
            );
        }
        if cap.fill_visible() {
            append_layer(
                cap_mesh,
                proj_scratch,
                section_scratch,
                cap.surface_alpha,
                &cap_projection,
                cap_scale,
                cap_clip,
            );
        }
    }
}

/// Style inputs for [`build_wireframe_meshes`] that do not come from the bodies.
#[derive(Copy, Clone)]
pub(crate) struct WireframeStyle {
    pub(crate) color_mode: WireframeColorMode,
    /// Uniform edge alpha, ignored when `nearest_active` grades it per edge.
    pub(crate) alpha: f32,
    pub(crate) width_px: f32,
    /// Grade each edge's alpha by how close the slice is to the midpoint of the
    /// cells containing it, so brightness propagates as a wave under a scrub.
    pub(crate) nearest_active: bool,
    /// Chord-to-arc morph for edge geometry (0 = flat R⁴ chord, 1 = S³ arc).
    pub(crate) space_blend: f32,
    /// `Some(thickness)` runs the Hyperslice cull: keep only edges whose
    /// body-local w-interval intersects the slab around the slice, thinning the
    /// graph to what the slice passes through. A demo-side per-edge filter,
    /// deliberately not a `Projection` variant (the projection has discarded w,
    /// so it cannot carry a keep/drop signal).
    pub(crate) hyperslice: Option<f32>,
}

/// Build the cross-section perimeter mesh and the parent-wireframe edge mesh
/// over the row, in world R³. Non-polychoral shapes are skipped (no
/// [`loam_physics::polytope::Polytope4`] mapping).
///
/// Both meshes and every scratch buffer are the caller's, cleared on entry and
/// refilled. Under Stereographic the parent mesh reaches ~92k segments for a
/// full row of 600-cells, so owning them here would be megabytes of
/// grow-and-copy per frame.
///
/// Free function over [`RowFrame`] so "the edges wrap the body physics put
/// there" is unit-testable without a GPU-backed [`Demo`];
/// [`Demo::render_wireframe_overlay`] is the one production caller.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_wireframe_meshes(
    frame: &RowFrame<'_>,
    style: &WireframeStyle,
    cross: state::SectionLayer,
    cap: state::SectionLayer,
    palette_cache: &mut std::collections::HashMap<loam_physics::polytope::Polytope4, Vec<[f32; 4]>>,
    slerp_scratch: &mut Vec<Vec4>,
    local_vertices: &mut Vec<Vec4>,
    cell_strengths: &mut Vec<f32>,
    section_scratch: &mut SectionScratch,
    body_perimeter: &mut LineMesh<3>,
    section_edges: &mut LineMesh<3>,
    parent_lines: &mut LineMesh<3>,
) {
    section_edges.segments.clear();
    section_edges.colors.clear();
    section_edges.widths.clear();
    parent_lines.segments.clear();
    parent_lines.colors.clear();
    parent_lines.widths.clear();
    // `nearest-active` off: uniform `style.alpha`. On: per-edge interp between
    // DIM (slice misses the cell) and BRIGHT (slice at its midpoint). DIM/BRIGHT
    // are activity-gradient peaks, not a global opacity.
    const PARENT_ALPHA_DIM: f32 = 0.10;
    const PARENT_ALPHA_BRIGHT: f32 = 0.85;
    // Active-mode palette: green for edges in an intersected cell, gray
    // otherwise (binary contrast against the scene backdrop).
    const ACTIVE_GREEN: [f32; 4] = [0.40, 1.00, 0.55, 1.0];
    const INACTIVE_GRAY: [f32; 4] = [0.55, 0.55, 0.58, 1.0];
    let w_slice = frame.w_slice;
    // Honest layer's outline is forced to drop-w so it can never follow a
    // distorting projection.
    let cross_section_projection = state::section_layer_projection(true, frame.projection);

    for (slot, entry) in frame.row.iter().enumerate() {
        let Some(polytope) = entry.shape.polytope4() else {
            continue;
        };
        // Per-shape clip radius: finite for the 16-cell, `f32::INFINITY` else.
        let view_radius = stereographic_view_radius(polytope, frame.camera_distance);
        let topo = polytope.topology();
        // The body's `w` rides inside the body-local frame (see
        // `BodyPose::body_local`); projecting there and translating in R³ AFTER
        // keeps its apparent x-position stable when Perspective4D scales
        // (x, y, z) by `focal / (focal - w)`.
        let body_pos_r3 = frame.body_local(slot, topo.vertices, frame.body_size, local_vertices);

        // Cross-section perimeter outlines, one per enabled layer.
        // `polytope_section_perimeter_append` fills the body-local drop-w
        // perimeter once; the honest layer maps it through drop-w (so its
        // outline matches the SDF slice), the cap through the active projection.
        // Under Stereographic a segment is dropped when either endpoint exceeds
        // the clip radius (per-segment: a perimeter segment is a single cap edge).
        if cross.perimeter || cap.perimeter {
            body_perimeter.segments.clear();
            body_perimeter.colors.clear();
            body_perimeter.widths.clear();
            polytope_section_perimeter_append(
                topo.edges,
                topo.cells,
                local_vertices,
                WPlane::new(w_slice),
                section_scratch,
                body_perimeter,
            );
            let mut push_perimeter = |projection: &loam_math::Projection<4>| {
                let section_scale = perspective_scale_at_w(w_slice, projection);
                let clip_radius = stereographic_clip_radius(projection, view_radius);
                for ((a, b), (color, width)) in body_perimeter.segments.iter().zip(
                    body_perimeter
                        .colors
                        .iter()
                        .zip(body_perimeter.widths.iter()),
                ) {
                    let (pa, wa) = cap_vertex_projected_and_world(
                        *a,
                        w_slice,
                        section_scale,
                        projection,
                        body_pos_r3,
                    );
                    let (pb, wb) = cap_vertex_projected_and_world(
                        *b,
                        w_slice,
                        section_scale,
                        projection,
                        body_pos_r3,
                    );
                    if !sample_in_radius(pa, clip_radius) || !sample_in_radius(pb, clip_radius) {
                        continue;
                    }
                    section_edges.segments.push((wa, wb));
                    section_edges.colors.push(*color);
                    section_edges.widths.push(*width);
                }
            };
            // Honest cross-section first (drop-w), then the projected cap.
            if cross.perimeter {
                push_perimeter(&cross_section_projection);
            }
            if cap.perimeter {
                push_perimeter(&frame.projection);
            }
        }

        // Per-cell crossing strength in [0, 1], shared with build_points_mesh via
        // `compute_cell_strengths` so both passes agree on "active".
        compute_cell_strengths(topo.cells, local_vertices, w_slice, cell_strengths);

        // Per-edge brightness: max strength over cells containing both endpoints,
        // so an edge lights up when any containing cell is crossed.
        let edge_strength = |i: u32, j: u32| -> f32 {
            let mut best = 0.0_f32;
            for (cell, strength) in topo.cells.iter().zip(cell_strengths.iter()) {
                if cell.contains(&i) && cell.contains(&j) && *strength > best {
                    best = *strength;
                }
            }
            best
        };

        // Active-mode binary: an edge is active if some containing cell has the
        // slice strictly inside its w-range (`cell_strength > 0.0`).
        let edge_is_active = |i: u32, j: u32| -> bool {
            topo.cells
                .iter()
                .zip(cell_strengths.iter())
                .any(|(cell, &s)| s > 0.0 && cell.contains(&i) && cell.contains(&j))
        };

        // Hyperslice cull, CELL-level to match `edge_is_active`: an edge survives
        // iff some cell containing both endpoints has its w-range overlapping the
        // slab. The edge-level test would cull a far-side edge of an active cell
        // that the cell-level coloring paints green. The slab band is a superset
        // of `edge_is_active` (the zero-width plane), so active edges are never
        // culled while the band thins the graph.
        let edge_in_slab_cell = |i: u32, j: u32, thickness: f32| -> bool {
            topo.cells.iter().any(|cell| {
                if !(cell.contains(&i) && cell.contains(&j)) {
                    return false;
                }
                let (w_min, w_max) = cell_w_range(cell, local_vertices);
                slab_overlaps(w_min, w_max, w_slice, thickness)
            })
        };

        // Base RGB per `style.color_mode`: `VertexGradient` position-derived hue,
        // `UniqueEdge` greedy line-graph coloring, `WDepth` signed-w
        // cool-to-warm, `Active` binary green/gray. Alpha is then the
        // `nearest-active` strength or uniform.
        //
        // UniqueEdge palette is topology-only; memoize by `Polytope4` so the
        // 600-cell's ~520k pair-checks run once per launch, not per frame.
        let edge_palette: &[[f32; 4]] =
            if matches!(style.color_mode, WireframeColorMode::UniqueEdge) {
                palette_cache
                    .entry(polytope)
                    .or_insert_with(|| unique_edge_palette(topo.edges))
            } else {
                &[]
            };
        // WDepth normalizes against the CANONICAL max |w| (not the rotated
        // per-frame max), so the color stays temporally stable as the rotor
        // swings a vertex from -w to +w. Per-polytope because the band differs
        // (the tesseract's `[-0.5, +0.5]` does not fit the 16-cell, 5-cell).
        let w_extent_local: f32 = if matches!(style.color_mode, WireframeColorMode::WDepth) {
            let canonical_max_w = topo
                .vertices
                .iter()
                .map(|v| v.w.abs())
                .fold(0.0_f32, f32::max)
                .max(1e-6);
            canonical_max_w * frame.body_size
        } else {
            1.0
        };
        for (edge_idx, &[i, j]) in topo.edges.iter().enumerate() {
            let ia = i as usize;
            let ja = j as usize;
            let a = local_vertices[ia];
            let b = local_vertices[ja];
            // Cell-level Hyperslice cull before any color / projection work, so a
            // culled edge costs only the membership fold; `is_some_and` skips it
            // entirely when the affordance is off.
            if style
                .hyperslice
                .is_some_and(|thickness| !edge_in_slab_cell(i, j, thickness))
            {
                continue;
            }
            let (mut color_a, mut color_b) = match style.color_mode {
                WireframeColorMode::VertexGradient => (
                    vertex_color_by_position(topo.vertices[ia]),
                    vertex_color_by_position(topo.vertices[ja]),
                ),
                WireframeColorMode::UniqueEdge => {
                    let c = edge_palette[edge_idx];
                    (c, c)
                }
                WireframeColorMode::WDepth => (
                    w_depth_color(a.w, w_extent_local),
                    w_depth_color(b.w, w_extent_local),
                ),
                WireframeColorMode::Active => {
                    let c = if edge_is_active(i, j) {
                        ACTIVE_GREEN
                    } else {
                        INACTIVE_GRAY
                    };
                    (c, c)
                }
            };
            let alpha = if style.nearest_active {
                let s = edge_strength(i, j);
                PARENT_ALPHA_DIM + (PARENT_ALPHA_BRIGHT - PARENT_ALPHA_DIM) * s
            } else {
                style.alpha
            };
            color_a[3] = alpha;
            color_b[3] = alpha;
            // Emit the edge in body-local 4D, projected to world R³; `blend` is
            // projection-derived (Stereographic -> 1, affine -> 0).
            push_blended_edge(
                parent_lines,
                a,
                b,
                color_a,
                color_b,
                style.width_px,
                style.space_blend,
                &frame.projection,
                body_pos_r3,
                slerp_scratch,
                view_radius,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::ShapeEntry;
    use crate::physics::PlaygroundPhysics;
    use crate::state::{body_position, RowFrame, SectionLayer};
    use loam_math::{EuclideanR4, Plane4, Projection};
    use loam_physics::polytope::Polytope4;
    use loam_render::raymarch::RaymarchShape;

    /// A one-slot row, so every primitive a builder emits belongs to slot 0 and
    /// a pin can read the whole mesh instead of slicing per-slot offsets.
    const ROW: &[ShapeEntry] = &[ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Pentatope),
        body_color: [0.95, 0.55, 0.30],
        label: "5-cell",
        long_name: "pentachoron",
    }];

    /// The 16-cell is the fixture shape wherever a pin needs a known image:
    /// its vertices are `±e_i`, so twelve of its edges lie exactly in `w = 0`
    /// (drop-w loses nothing there) and its w-slices shrink with `|w|` (so a
    /// cut at the wrong slice is a different cap, not the same one). It is also
    /// the only polytope `stereographic_view_radius` clips.
    const CELL16: ShapeEntry = ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Cell16),
        body_color: [0.30, 0.70, 0.95],
        label: "16-cell",
        long_name: "hexadecachoron",
    };
    const ROW_16: &[ShapeEntry] = &[CELL16];
    /// Two slots of the same shape: each half of any mesh is the same geometry
    /// at the other slot's layout position, so a builder that read slot 0's
    /// pose for every slot collapses the halves onto each other.
    const ROW_16_PAIR: &[ShapeEntry] = &[CELL16, CELL16];
    /// Two slots of DIFFERENT shapes, which every other fixture here is not.
    /// The production row is heterogeneous (`catalog::DEFAULT_ROW` is four
    /// distinct polytopes), so a builder that serves one slot another slot's
    /// topology is invisible to a homogeneous fixture: the counts still agree
    /// with each other and nothing is length-inconsistent.
    const ROW_MIXED: &[ShapeEntry] = &[CELL16, ROW[0]];

    /// A slice off `w = 0`, inside the 16-cell's w-extent (`±BODY_SIZE`) so the
    /// cut is non-degenerate. A builder that substituted `0.0` for the frame's
    /// own `w_slice` cuts a visibly different cap here.
    const SLICE_W: f32 = 0.2;

    /// Eye-to-focus distance for the fixtures that do not vary it.
    const CAMERA_DISTANCE: f32 = 4.0;

    /// Tolerance for "the same mesh, translated": the pins compare
    /// `p + (centre + d)` against `(p + centre) + d`, and f32 addition does not
    /// associate. Four orders of magnitude below the throw's own displacement,
    /// so a pass that lost the throw cannot pass this.
    const TRANSLATE_TOL: f32 = 1e-5;

    /// Two worlds whose bodies must render identically up to ONE R³
    /// translation: `thrown` carries both a live centre and a live orientation,
    /// `at_rest` sits on the layout under the composed spin. The throw is +x
    /// from a +w lever, so the body's `w` never leaves the layout and both
    /// worlds cut the same body-local geometry at the same slice.
    struct TranslatedPair {
        thrown: PlaygroundPhysics,
        at_rest: PlaygroundPhysics,
        /// UI spin the thrown world renders under.
        spin: Rotor4,
        /// `spin · orientation`: the rotor the thrown body actually renders at,
        /// and the spin the at-rest world is given so the two match.
        composed: Rotor4,
        delta: Vec3,
    }

    fn translated_pair() -> TranslatedPair {
        let spin = rotor_at(Plane4::Xy, 0.7);
        let mut thrown = PlaygroundPhysics::new(1, BODY_SIZE);
        let layout = Vec4::from_array(body_position(0, 1));
        thrown.world.bodies[0].apply_impulse_at_point(
            &EuclideanR4,
            Vec4::X * 0.5,
            layout + Vec4::W * 0.5,
        );
        thrown.step(24);

        let pose = thrown.pose(0, 1, spin);
        assert_eq!(
            pose.position.w, layout.w,
            "throw left the layout w, so the two worlds cut different geometry"
        );
        let delta = pose.position_r3() - layout.truncate();
        assert!(
            delta.length() > 0.05,
            "throw did not move the body's R³ centre"
        );
        // A rotation the pins can see: an unwired pass draws `spin` where
        // physics says `spin · orientation`.
        let probe = Vec4::new(0.3, -0.2, 0.9, 0.1);
        assert!(
            (pose.rotor.apply(probe) - spin.apply(probe)).length() > 1e-2,
            "throw produced no visible rotation, so the rotor half of the pins is vacuous"
        );
        TranslatedPair {
            at_rest: PlaygroundPhysics::new(1, BODY_SIZE),
            composed: pose.rotor,
            thrown,
            spin,
            delta,
        }
    }

    /// Drop-w, on the `w = 0` slice, fixed camera: the builders' clip and scale
    /// paths then depend on the body pose alone.
    fn frame(physics: &PlaygroundPhysics, spin: Rotor4) -> RowFrame<'_> {
        frame_of(
            physics,
            ROW,
            spin,
            Projection::Identity,
            0.0,
            CAMERA_DISTANCE,
        )
    }

    /// Every field of the seam a fixture might vary, spelled out.
    fn frame_of<'a>(
        physics: &'a PlaygroundPhysics,
        row: &'a [ShapeEntry],
        spin: Rotor4,
        projection: Projection<4>,
        w_slice: f32,
        camera_distance: f32,
    ) -> RowFrame<'a> {
        RowFrame {
            physics,
            row,
            spin,
            body_size: BODY_SIZE,
            projection,
            w_slice,
            camera_distance,
        }
    }

    fn rotor_at(plane: Plane4, angle: f32) -> Rotor4 {
        (plane.unit_bivector() * angle).exp().normalize()
    }

    /// Flat chords, no Hyperslice cull, `Active` coloring. `Active` is the one
    /// color mode derived from `w_slice`, so a pin that compares colors covers
    /// the slice as well as the pose.
    fn slice_colored_style() -> WireframeStyle {
        WireframeStyle {
            color_mode: WireframeColorMode::Active,
            alpha: 1.0,
            width_px: 1.8,
            nearest_active: false,
            space_blend: 0.0,
            hyperslice: None,
        }
    }

    /// Everything the three mesh builders emit for one frame, as world-R³
    /// points in build order. Taking all of them at once is what keeps a
    /// fixture from pinning whichever path it happened to pick.
    struct BuiltRow {
        edges: Vec<[f32; 3]>,
        /// Cross-section perimeter then projected-cap perimeter, per slot.
        perimeter: Vec<[f32; 3]>,
        sprites: Vec<[f32; 3]>,
        sprite_colors: Vec<[f32; 4]>,
        cross_caps: Vec<[f32; 3]>,
        projected_caps: Vec<[f32; 3]>,
        /// The cap fill's surviving triangles. The near-pole clip drops these
        /// and leaves the vertices behind, so it is invisible in
        /// [`Self::projected_caps`].
        projected_cap_triangles: Vec<[u32; 3]>,
        cross_cap_triangles: Vec<[u32; 3]>,
    }

    /// Every caller-owned buffer the two overlay builders write through, in one
    /// value. Reusing a live `OverlayBuffers` across builds is what a frame does,
    /// so the alloc pins and the geometry pins exercise the same seam.
    #[derive(Default)]
    struct OverlayBuffers {
        palette_cache: std::collections::HashMap<Polytope4, Vec<[f32; 4]>>,
        centers_cache: std::collections::HashMap<Polytope4, Vec<Vec4>>,
        slerp: Vec<Vec4>,
        local_vertices: Vec<Vec4>,
        center_locals: Vec<Vec4>,
        cell_strengths: Vec<f32>,
        section_scratch: SectionScratch,
        cross_faces: loam_shape::TriangleMesh<3>,
        cap_faces: loam_shape::TriangleMesh<3>,
        proj: Vec<Vec3>,
        body_perimeter: LineMesh<3>,
        section_edges: LineMesh<3>,
        parent_lines: LineMesh<3>,
        sprites: loam_shape::PointMesh<3>,
    }

    impl OverlayBuffers {
        fn wireframe(
            &mut self,
            frame: &RowFrame<'_>,
            style: &WireframeStyle,
            cross: SectionLayer,
            cap: SectionLayer,
        ) {
            build_wireframe_meshes(
                frame,
                style,
                cross,
                cap,
                &mut self.palette_cache,
                &mut self.slerp,
                &mut self.local_vertices,
                &mut self.cell_strengths,
                &mut self.section_scratch,
                &mut self.body_perimeter,
                &mut self.section_edges,
                &mut self.parent_lines,
            );
        }

        fn sections(&mut self, frame: &RowFrame<'_>, cross: SectionLayer, cap: SectionLayer) {
            build_section_layer_meshes(
                frame,
                cross,
                cap,
                SectionBuffers {
                    local_vertices: &mut self.local_vertices,
                    proj_scratch: &mut self.proj,
                    cross_mesh: &mut self.cross_faces,
                    cap_mesh: &mut self.cap_faces,
                    section_scratch: &mut self.section_scratch,
                },
            );
        }

        fn points(&mut self, frame: &RowFrame<'_>, style: &PointsStyle) {
            build_points_mesh(
                frame,
                style,
                &mut self.centers_cache,
                &mut self.local_vertices,
                &mut self.center_locals,
                &mut self.cell_strengths,
                &mut self.sprites,
            );
        }
    }

    /// Both sprite kinds on, coloring taken from the wireframe style so the two
    /// builders are pinned under one mode.
    fn points_style_for(style: &WireframeStyle) -> PointsStyle {
        PointsStyle {
            color_mode: style.color_mode,
            show_vertices: true,
            show_cell_centers: true,
            size_px: 6.0,
        }
    }

    /// Both section layers visible and both sprite kinds on, so no branch of a
    /// builder sits outside the pins.
    fn build_row(frame: &RowFrame<'_>, style: &WireframeStyle) -> BuiltRow {
        let cross = SectionLayer::CROSS_SECTION_DEFAULT;
        let cap = SectionLayer {
            perimeter: true,
            surface_alpha: 0.5,
        };
        let mut buffers = OverlayBuffers::default();
        buffers.wireframe(frame, style, cross, cap);
        buffers.points(frame, &points_style_for(style));

        let mut local_vertices = Vec::new();
        let mut proj_scratch = Vec::new();
        let mut cross_mesh = loam_shape::TriangleMesh::<3>::default();
        let mut cap_mesh = loam_shape::TriangleMesh::<3>::default();
        build_section_layer_meshes(
            frame,
            cross,
            cap,
            SectionBuffers {
                local_vertices: &mut local_vertices,
                proj_scratch: &mut proj_scratch,
                cross_mesh: &mut cross_mesh,
                cap_mesh: &mut cap_mesh,
                section_scratch: &mut SectionScratch::default(),
            },
        );

        BuiltRow {
            edges: segment_points(&buffers.parent_lines),
            perimeter: segment_points(&buffers.section_edges),
            sprites: buffers.sprites.positions,
            sprite_colors: buffers.sprites.colors,
            cross_caps: cross_mesh.vertices,
            projected_caps: cap_mesh.vertices,
            projected_cap_triangles: cap_mesh.indices,
            cross_cap_triangles: cross_mesh.indices,
        }
    }

    impl BuiltRow {
        /// Each mesh in build order, so a pin covers every path at once.
        fn meshes(&self) -> [(&str, &[[f32; 3]]); 5] {
            [
                ("parent wireframe", &self.edges),
                ("section perimeter", &self.perimeter),
                ("point sprites", &self.sprites),
                ("cross-section caps", &self.cross_caps),
                ("projected caps", &self.projected_caps),
            ]
        }

        /// Every mesh is `rest`'s carried by `delta`, with the coloring
        /// untouched: the whole seam moved with the body and nothing else did.
        fn assert_carried_from(&self, rest: &BuiltRow, delta: Vec3) {
            for ((what, live), (_, at_rest)) in self.meshes().iter().zip(rest.meshes().iter()) {
                assert_translated(live, at_rest, delta, what);
            }
            assert_eq!(
                self.sprite_colors, rest.sprite_colors,
                "sprite coloring diverged between the two worlds"
            );
            assert_eq!(
                (&self.cross_cap_triangles, &self.projected_cap_triangles),
                (&rest.cross_cap_triangles, &rest.projected_cap_triangles),
                "cap triangulation diverged, so the vertex pins compare unrelated points"
            );
        }

        /// The row's two slots emitted the same geometry, separated by `delta`.
        fn assert_slots_separated_by(&self, delta: Vec3) {
            for (what, all) in self.meshes() {
                assert!(
                    !all.is_empty(),
                    "{what}: nothing was emitted, so the pin is vacuous"
                );
                let (first, second) = all.split_at(all.len() / 2);
                assert_translated(second, first, delta, what);
            }
        }
    }

    fn segment_points(mesh: &LineMesh<3>) -> Vec<[f32; 3]> {
        mesh.segments.iter().flat_map(|(a, b)| [*a, *b]).collect()
    }

    /// Each point of `live` is `rest`'s corresponding point carried by `delta`:
    /// same count, same order, same body-local geometry.
    fn assert_translated(live: &[[f32; 3]], rest: &[[f32; 3]], delta: Vec3, what: &str) {
        assert!(
            !live.is_empty(),
            "{what}: nothing was emitted, so the pin is vacuous"
        );
        assert_eq!(
            live.len(),
            rest.len(),
            "{what}: the two worlds emitted different geometry"
        );
        for (i, (l, r)) in live.iter().zip(rest).enumerate() {
            let expected = Vec3::from_array(*r) + delta;
            assert!(
                (Vec3::from_array(*l) - expected).length() < TRANSLATE_TOL,
                "{what} point {i}: {l:?} is not the at-rest body carried to the live centre {expected:?}"
            );
        }
    }

    /// The wireframe overlay wraps the body PHYSICS put there: a thrown,
    /// tumbling slot's edges and perimeter are the at-rest body under the
    /// composed rotor, carried to the live centre. Unwiring the pass back to
    /// the authored spin over the static layout loses both halves.
    #[test]
    fn wireframe_meshes_follow_the_physics_pose() {
        let pair = translated_pair();
        let style = WireframeStyle {
            color_mode: WireframeColorMode::VertexGradient,
            alpha: 1.0,
            width_px: 1.8,
            nearest_active: false,
            space_blend: 0.0,
            hyperslice: None,
        };
        let cross = SectionLayer::CROSS_SECTION_DEFAULT;
        let cap = SectionLayer::PROJECTED_CAP_DEFAULT;
        let mut live = OverlayBuffers::default();
        let mut rest = OverlayBuffers::default();

        live.wireframe(&frame(&pair.thrown, pair.spin), &style, cross, cap);
        rest.wireframe(&frame(&pair.at_rest, pair.composed), &style, cross, cap);

        assert_translated(
            &segment_points(&live.parent_lines),
            &segment_points(&rest.parent_lines),
            pair.delta,
            "parent wireframe",
        );
        assert_translated(
            &segment_points(&live.section_edges),
            &segment_points(&rest.section_edges),
            pair.delta,
            "section perimeter",
        );
    }

    /// Point sprites sit on the body PHYSICS put there, vertices and cell
    /// centres alike (the centres take a second, inset body frame, so they can
    /// be unwired independently of the vertices).
    #[test]
    fn point_sprites_follow_the_physics_pose() {
        let pair = translated_pair();
        let style = PointsStyle {
            color_mode: WireframeColorMode::VertexGradient,
            show_vertices: true,
            show_cell_centers: true,
            size_px: 6.0,
        };
        let mut live = OverlayBuffers::default();
        let mut rest = OverlayBuffers::default();

        live.points(&frame(&pair.thrown, pair.spin), &style);
        rest.points(&frame(&pair.at_rest, pair.composed), &style);

        assert_translated(
            &live.sprites.positions,
            &rest.sprites.positions,
            pair.delta,
            "point sprites",
        );
    }

    /// The section caps are cut from the body PHYSICS put there, on both
    /// layers: same triangles, carried to the live centre.
    #[test]
    fn section_caps_follow_the_physics_pose() {
        let pair = translated_pair();
        let cross = SectionLayer::CROSS_SECTION_DEFAULT;
        // Both layers visible so the projected-cap branch is pinned too.
        let cap = SectionLayer {
            perimeter: true,
            surface_alpha: 0.5,
        };
        let mut local_vertices = Vec::new();
        let mut proj_scratch = Vec::new();
        let mut live_cross = loam_shape::TriangleMesh::<3>::default();
        let mut live_cap = loam_shape::TriangleMesh::<3>::default();
        let mut rest_cross = loam_shape::TriangleMesh::<3>::default();
        let mut rest_cap = loam_shape::TriangleMesh::<3>::default();

        build_section_layer_meshes(
            &frame(&pair.thrown, pair.spin),
            cross,
            cap,
            SectionBuffers {
                local_vertices: &mut local_vertices,
                proj_scratch: &mut proj_scratch,
                cross_mesh: &mut live_cross,
                cap_mesh: &mut live_cap,
                section_scratch: &mut SectionScratch::default(),
            },
        );
        build_section_layer_meshes(
            &frame(&pair.at_rest, pair.composed),
            cross,
            cap,
            SectionBuffers {
                local_vertices: &mut local_vertices,
                proj_scratch: &mut proj_scratch,
                cross_mesh: &mut rest_cross,
                cap_mesh: &mut rest_cap,
                section_scratch: &mut SectionScratch::default(),
            },
        );

        assert_translated(
            &live_cross.vertices,
            &rest_cross.vertices,
            pair.delta,
            "cross-section caps",
        );
        assert_translated(
            &live_cap.vertices,
            &rest_cap.vertices,
            pair.delta,
            "projected caps",
        );
        assert_eq!(
            live_cross.indices, rest_cross.indices,
            "cap triangulation diverged, so the vertex pin above compares unrelated points"
        );
    }

    /// Every slot renders at its OWN body. A two-slot row of one shape emits
    /// the same geometry twice, separated by the layout spacing; a builder that
    /// read slot 0's pose for the whole row collapses the two halves onto each
    /// other.
    #[test]
    fn each_slot_renders_at_its_own_body() {
        let physics = PlaygroundPhysics::new(2, BODY_SIZE);
        let frame = frame_of(
            &physics,
            ROW_16_PAIR,
            rotor_at(Plane4::Xz, 0.5),
            Projection::Identity,
            SLICE_W,
            CAMERA_DISTANCE,
        );
        let built = build_row(&frame, &slice_colored_style());

        let layout = Vec4::from_array(body_position(1, 2)).truncate()
            - Vec4::from_array(body_position(0, 2)).truncate();
        assert!(layout.length() > 0.5, "the two slots share a position");
        built.assert_slots_separated_by(layout);
        let (first, second) = built.sprite_colors.split_at(built.sprite_colors.len() / 2);
        assert_eq!(
            first, second,
            "the two slots colored the same shape differently"
        );
    }

    /// A body physics threw off the slice is cut where it now IS. The same body
    /// lifted to `w = lift` and cut at `w_slice + lift` renders exactly what the
    /// unmoved body cut at `w_slice` does: `BodyPose::body_local`'s `w` term and
    /// the frame's own `w_slice` are the two halves of that, and dropping
    /// either one breaks the equality.
    #[test]
    fn a_body_lifted_off_the_slice_is_cut_where_physics_put_it() {
        let (lifted, lift) = thrown_along_w();
        let spin = rotor_at(Plane4::Xz, 0.5);
        let at_rest = PlaygroundPhysics::new(1, BODY_SIZE);
        let style = slice_colored_style();
        let build_at = |physics, w_slice| {
            build_row(
                &frame_of(
                    physics,
                    ROW_16,
                    spin,
                    Projection::Identity,
                    w_slice,
                    CAMERA_DISTANCE,
                ),
                &style,
            )
        };

        let live = build_at(&lifted, SLICE_W + lift);
        let rest = build_at(&at_rest, SLICE_W);
        // The lift is along +w alone, so the R³ centre never moved.
        live.assert_carried_from(&rest, Vec3::ZERO);

        // The same lifted body at the UNSHIFTED slice is a different cut;
        // without this the pin above would also hold for a build that ignored
        // the lift and the slice together.
        let off_slice = build_at(&lifted, SLICE_W);
        assert_ne!(
            off_slice.cross_caps, rest.cross_caps,
            "the cap did not move with the slice, so the pin above is vacuous"
        );
    }

    /// The honest cross-section ignores the active projection; the projected
    /// cap follows it. Under Perspective4D at a slice off `w = 0` the cap is
    /// the cross-section scaled about the body centre by
    /// `focal / (focal - w_slice)`, so swapping the two layers' projections
    /// swaps which mesh carries the scale.
    #[test]
    fn the_honest_layer_ignores_the_projection_the_cap_scales_by_it() {
        const FOCAL: f32 = 2.0;
        let physics = PlaygroundPhysics::new(1, BODY_SIZE);
        let frame = frame_of(
            &physics,
            ROW_16,
            rotor_at(Plane4::Xz, 0.5),
            Projection::Perspective4D {
                focal_distance: FOCAL,
            },
            SLICE_W,
            CAMERA_DISTANCE,
        );
        let built = build_row(&frame, &slice_colored_style());

        let scale = FOCAL / (FOCAL - SLICE_W);
        assert!(
            (scale - 1.0).abs() > 0.1,
            "the two layers would coincide at this focal distance"
        );
        let centre = Vec4::from_array(body_position(0, 1)).truncate();
        assert_scaled_about(
            &built.projected_caps,
            &built.cross_caps,
            centre,
            scale,
            "section cap fill",
        );
        // The perimeter mesh is the honest outline followed by the cap outline,
        // one slot, so its halves stand in the same relation.
        let (honest, cap) = built.perimeter.split_at(built.perimeter.len() / 2);
        assert_scaled_about(cap, honest, centre, scale, "section perimeter");
    }

    /// The Hyperslice cull keeps exactly the edges some containing cell carries
    /// across the slab, in edge order. Inverting the predicate keeps the
    /// complement: a different mesh, not a thinner one.
    ///
    /// The spin must tilt w into R³ (`Zw`, not the w-preserving `Xz` the other
    /// fixtures use): under a w-preserving spin drop-w sends the 16-cell's
    /// `+e_w` and `-e_w` to the same R³ point, the keep set and its complement
    /// are congruent, and an inverted predicate emits a mesh this pin cannot
    /// tell from the right one. The `assert_ne` below holds the fixture to it.
    #[test]
    fn hyperslice_keeps_the_edges_whose_cells_cross_the_slab() {
        const THICKNESS: f32 = 0.2;
        const TILT: f32 = 0.5;
        // Slab `[0.4, 0.6]`: above the tilted `e_z` w-extent
        // (`BODY_SIZE · sin TILT` = 0.34) and below the `e_w` one
        // (`BODY_SIZE · cos TILT` = 0.61), so a cell's w-range straddles the
        // near boundary exactly when the cell holds `+e_w`, and the edges
        // reaching only `-e_w` cells are the ones culled.
        const SLAB_CENTRE_W: f32 = 0.5;
        let physics = PlaygroundPhysics::new(1, BODY_SIZE);
        let frame = frame_of(
            &physics,
            ROW_16,
            rotor_at(Plane4::Zw, TILT),
            Projection::Identity,
            SLAB_CENTRE_W,
            CAMERA_DISTANCE,
        );
        let mut style = slice_colored_style();
        style.hyperslice = Some(THICKNESS);

        let topo = Polytope4::Cell16.topology();
        let mut local = Vec::new();
        let centre = frame.body_local(0, topo.vertices, BODY_SIZE, &mut local);
        // Drop-w and a flat blend, so a kept edge is exactly its two projected
        // endpoints and the whole mesh is an ordered function of the keep set.
        // `keep` reads the cell's slab overlap, so the mesh an inverted cull
        // would emit is one more call.
        let mesh_under = |keep: fn(bool) -> bool| -> Vec<([f32; 3], [f32; 3])> {
            topo.edges
                .iter()
                .filter(|[i, j]| {
                    topo.cells.iter().any(|cell| {
                        cell.contains(i) && cell.contains(j) && {
                            let (w_min, w_max) = cell_w_range(cell, &local);
                            keep(slab_overlaps(w_min, w_max, SLAB_CENTRE_W, THICKNESS))
                        }
                    })
                })
                .map(|&[i, j]| {
                    (
                        (local[i as usize].truncate() + centre).to_array(),
                        (local[j as usize].truncate() + centre).to_array(),
                    )
                })
                .collect()
        };
        let expected = mesh_under(|overlaps| overlaps);
        assert!(
            !expected.is_empty() && expected.len() < topo.edges.len(),
            "the slab kept {} of {} edges, so the cull is not under test",
            expected.len(),
            topo.edges.len()
        );
        assert_ne!(
            mesh_under(|overlaps| !overlaps),
            expected,
            "the inverted cull emits the same mesh here, so the pin below \
             cannot see the predicate's polarity"
        );

        let mut buffers = OverlayBuffers::default();
        buffers.wireframe(
            &frame,
            &style,
            SectionLayer::CROSS_SECTION_DEFAULT,
            SectionLayer::PROJECTED_CAP_DEFAULT,
        );
        assert_eq!(
            buffers.parent_lines.segments, expected,
            "the cull kept a different edge set than the slab crosses"
        );
    }

    /// `space_blend = 1` renders an edge as the S³ great-circle arc rather than
    /// the chord: one sub-segment per tessellation sample, and for an edge lying
    /// in `w = 0` (where drop-w loses nothing) every sample sits on the body's
    /// circumsphere instead of cutting across it.
    #[test]
    fn space_blend_one_bows_edges_onto_the_circumsphere() {
        let physics = PlaygroundPhysics::new(1, BODY_SIZE);
        // Identity spin, so the body-local vertices are the canonical `±e_i`
        // scaled by `BODY_SIZE` and the `w = 0` edges are exact.
        let frame = frame_of(
            &physics,
            ROW_16,
            Rotor4::IDENTITY,
            Projection::Identity,
            SLICE_W,
            CAMERA_DISTANCE,
        );
        let topo = Polytope4::Cell16.topology();
        let centre = Vec4::from_array(body_position(0, 1)).truncate();

        let mut chord_style = slice_colored_style();
        chord_style.space_blend = 0.0;
        let chords = build_row(&frame, &chord_style).edges;
        assert_eq!(chords.len(), topo.edges.len() * 2);

        let mut arc_style = slice_colored_style();
        arc_style.space_blend = 1.0;
        let arcs = build_row(&frame, &arc_style).edges;
        assert_eq!(
            arcs.len(),
            topo.edges.len() * SPACE_TESSELLATION_SAMPLES * 2,
            "blend 1 did not subdivide the edges"
        );

        let equatorial: Vec<usize> = topo
            .edges
            .iter()
            .enumerate()
            .filter(|(_, &[i, j])| {
                topo.vertices[i as usize].w == 0.0 && topo.vertices[j as usize].w == 0.0
            })
            .map(|(e, _)| e)
            .collect();
        assert!(!equatorial.is_empty(), "no edge lies in w = 0");
        // Endpoints per edge, edges emitted in topology order.
        let block = SPACE_TESSELLATION_SAMPLES * 2;
        for e in equatorial {
            for (k, p) in arcs[e * block..(e + 1) * block].iter().enumerate() {
                let radius = (Vec3::from_array(*p) - centre).length();
                assert!(
                    (radius - BODY_SIZE).abs() < 1e-4,
                    "edge {e} sample {k} sits at {radius}, off the circumsphere {BODY_SIZE}"
                );
            }
            // The chord through the same edge cuts inside: without this the pin
            // above would pass for a build that never left the endpoints.
            let mid = (Vec3::from_array(chords[e * 2]) + Vec3::from_array(chords[e * 2 + 1])) * 0.5;
            assert!(
                (mid - centre).length() < BODY_SIZE - 0.05,
                "edge {e}'s chord already lies on the circumsphere"
            );
        }
    }

    /// The near-pole clip radius is a fraction of the LIVE camera distance, so
    /// pulling the camera in clips the 16-cell harder: no sample survives past
    /// the radius that distance implies, and the far view keeps samples the
    /// near view drops. A builder reading a fixed distance clips both views
    /// identically.
    #[test]
    fn a_nearer_camera_clips_the_16cell_harder() {
        const NEAR: f32 = 4.0;
        const FAR: f32 = 12.0;
        let near_radius = stereographic_view_radius(Polytope4::Cell16, NEAR);
        let far_radius = stereographic_view_radius(Polytope4::Cell16, FAR);
        assert!(
            near_radius < far_radius,
            "both distances resolve to the same clip radius"
        );

        let physics = PlaygroundPhysics::new(1, BODY_SIZE);
        // A Zw spin swings a vertex toward the +w pole, where the conformal
        // image runs out past the near radius but stays inside the far one.
        let spin = rotor_at(Plane4::Zw, 1.1);
        let built_at = |camera_distance| {
            build_row(
                &frame_of(
                    &physics,
                    ROW_16,
                    spin,
                    Projection::Stereographic { pole: Vec4::W },
                    0.45,
                    camera_distance,
                ),
                &slice_colored_style(),
            )
        };
        let near = built_at(NEAR);
        let far = built_at(FAR);
        let centre = Vec4::from_array(body_position(0, 1)).truncate();

        for (what, near_mesh, far_mesh) in [
            ("parent wireframe", &near.edges, &far.edges),
            ("section perimeter", &near.perimeter, &far.perimeter),
            ("point sprites", &near.sprites, &far.sprites),
        ] {
            let near_max = max_local_radius(near_mesh, centre);
            let far_max = max_local_radius(far_mesh, centre);
            assert!(
                near_max <= near_radius,
                "{what}: kept a sample at {near_max}, past the near clip {near_radius}"
            );
            assert!(
                far_max > near_radius && far_max <= far_radius,
                "{what}: the far view's {far_max} is not between the two clip radii"
            );
        }
        // The cap fill clips at triangle granularity and leaves its dropped
        // vertices in the mesh, so its count is the only witness.
        assert!(
            near.projected_cap_triangles.len() < far.projected_cap_triangles.len(),
            "the near camera dropped no cap triangles"
        );
    }

    /// Largest body-local projected magnitude in `points`: what the near-pole
    /// clip radius bounds.
    fn max_local_radius(points: &[[f32; 3]], centre: Vec3) -> f32 {
        points
            .iter()
            .map(|p| (Vec3::from_array(*p) - centre).length())
            .fold(0.0_f32, f32::max)
    }

    /// One slot, at rest, thrown straight along `+w`: the body leaves the slice
    /// without rotating or moving in R³, so a world built from it differs from
    /// the layout in `w` alone. Returns the `w` it reached.
    fn thrown_along_w() -> (PlaygroundPhysics, f32) {
        let mut physics = PlaygroundPhysics::new(1, BODY_SIZE);
        physics.world.bodies[0].apply_impulse(Vec4::W);
        physics.step(15);
        let pose = physics.pose(0, 1, Rotor4::IDENTITY);
        let layout = Vec4::from_array(body_position(0, 1));
        assert_eq!(pose.rotor, Rotor4::IDENTITY, "the throw rotated the body");
        assert_eq!(pose.position.truncate(), layout.truncate());
        assert!(
            pose.position.w > 0.05 && pose.position.w < BODY_SIZE,
            "lift {} left the body's own w-extent",
            pose.position.w
        );
        let lift = pose.position.w;
        (physics, lift)
    }

    /// Counts what the thread running a probe asks of the allocator. The test
    /// runner gives each test its own thread, so a probe never sees a
    /// concurrent test's allocations; the counter is thread-local rather than
    /// global for exactly that reason. Const-initialised so reading it inside
    /// `alloc` cannot itself allocate.
    mod alloc_probe {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;

        thread_local! {
            static BYTES: Cell<usize> = const { Cell::new(0) };
        }

        pub struct Counting;

        unsafe impl GlobalAlloc for Counting {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                let _ = BYTES.try_with(|bytes| bytes.set(bytes.get() + layout.size()));
                System.alloc(layout)
            }

            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                System.dealloc(ptr, layout)
            }

            unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
                let _ = BYTES.try_with(|bytes| bytes.set(bytes.get() + new_size));
                System.realloc(ptr, layout, new_size)
            }
        }

        pub fn bytes_allocated_by(body: impl FnOnce()) -> usize {
            let before = BYTES.with(Cell::get);
            body();
            BYTES.with(Cell::get) - before
        }
    }

    // Crate-wide, not test-wide: `#[global_allocator]` is a per-binary
    // singleton, so every test in this crate allocates through `Counting` and a
    // second declaration anywhere in it is an E0152 hard error, not a merge
    // conflict. Only the three alloc pins read the counter.
    #[global_allocator]
    static COUNTING_ALLOCATOR: alloc_probe::Counting = alloc_probe::Counting;

    /// Steady state costs the allocator nothing: with both line meshes and
    /// every scratch buffer owned by the caller, a second build over the same
    /// row reaches the allocator zero times. Returning a fresh `LineMesh`, or
    /// hoisting a `Vec::new` back inside either builder, is tens of thousands
    /// of pushed elements under this fixture and shows up immediately.
    ///
    /// Perimeters are off here so a failure names which half regressed;
    /// [`the_perimeter_path_reaches_the_allocator_zero_times`] pins the section
    /// path with both layers on.
    #[test]
    fn a_warm_overlay_frame_reaches_the_allocator_zero_times() {
        // Stereographic at blend 1 is the expensive path: every edge
        // tessellates into `SPACE_TESSELLATION_SAMPLES` sub-segments through
        // the slerp buffer. `Active` coloring routes through the per-cell
        // strength buffer, and both sprite kinds route through the inset
        // cell-centre buffer and the centroid cache.
        let physics = PlaygroundPhysics::new(2, BODY_SIZE);
        let frame = frame_of(
            &physics,
            ROW_16_PAIR,
            rotor_at(Plane4::Zw, 0.4),
            Projection::Stereographic { pole: Vec4::W },
            SLICE_W,
            CAMERA_DISTANCE,
        );
        let mut style = slice_colored_style();
        style.space_blend = 1.0;
        let points_style = points_style_for(&style);
        let no_perimeter = SectionLayer {
            perimeter: false,
            surface_alpha: 0.0,
        };

        let mut buffers = OverlayBuffers::default();
        buffers.wireframe(&frame, &style, no_perimeter, no_perimeter);
        buffers.points(&frame, &points_style);
        assert!(
            !buffers.parent_lines.segments.is_empty() && !buffers.sprites.positions.is_empty(),
            "the fixture emitted no geometry, so the pin is vacuous"
        );

        let bytes = alloc_probe::bytes_allocated_by(|| {
            buffers.wireframe(&frame, &style, no_perimeter, no_perimeter);
            buffers.points(&frame, &points_style);
        });
        assert_eq!(
            bytes, 0,
            "a warm frame asked the allocator for {bytes} bytes through the overlay builders"
        );
    }

    /// With both section layers on, a warm frame still reaches the allocator
    /// zero times: the perimeter comes back through
    /// `polytope_section_perimeter_append` into caller-owned buffers, so
    /// nothing on this path owns a mesh or a per-cell `Vec` of its own.
    #[test]
    fn the_perimeter_path_reaches_the_allocator_zero_times() {
        let physics = PlaygroundPhysics::new(1, BODY_SIZE);
        let frame = frame_of(
            &physics,
            ROW_16,
            rotor_at(Plane4::Xz, 0.5),
            Projection::Identity,
            SLICE_W,
            CAMERA_DISTANCE,
        );
        let style = slice_colored_style();
        let cross = SectionLayer::CROSS_SECTION_DEFAULT;
        let cap = SectionLayer {
            perimeter: true,
            surface_alpha: 0.5,
        };

        let mut buffers = OverlayBuffers::default();
        buffers.wireframe(&frame, &style, cross, cap);
        assert!(
            !buffers.section_edges.segments.is_empty(),
            "no perimeter was emitted, so the pin is vacuous"
        );

        let warm =
            alloc_probe::bytes_allocated_by(|| buffers.wireframe(&frame, &style, cross, cap));
        assert_eq!(
            warm, 0,
            "a warm frame with both perimeters on asked the allocator for {warm} bytes"
        );
    }

    /// The fill half of the section layers, which `CROSS_SECTION_DEFAULT` ships
    /// on. Separate from the perimeter pin because the two run through
    /// different `loam_shape` entry points and each owned its own working set
    /// until the scratch was threaded through both.
    #[test]
    /// The cell-centre memo is keyed by [`Polytope4`]. Keyed by anything
    /// constant instead, slot 1 draws slot 0's cell table: wrong count, wrong
    /// positions, no panic, because the sprite buffers stay length-consistent
    /// with each other. Only a row of two different shapes can see it.
    #[test]
    fn cell_centre_sprites_come_from_each_slots_own_polytope() {
        let physics = PlaygroundPhysics::new(2, BODY_SIZE);
        let frame = frame_of(
            &physics,
            ROW_MIXED,
            rotor_at(Plane4::Xz, 0.5),
            Projection::Identity,
            SLICE_W,
            CAMERA_DISTANCE,
        );
        let mut style = points_style_for(&slice_colored_style());
        style.show_vertices = false;
        style.show_cell_centers = true;

        let mut buffers = OverlayBuffers::default();
        buffers.points(&frame, &style);

        let expected =
            Polytope4::Cell16.topology().cells.len() + Polytope4::Pentatope.topology().cells.len();
        assert_ne!(
            Polytope4::Cell16.topology().cells.len(),
            Polytope4::Pentatope.topology().cells.len(),
            "the two fixture shapes have the same cell count, so this pin              cannot see a memo that served one slot the other's table"
        );
        assert_eq!(
            buffers.sprites.positions.len(),
            expected,
            "the cell-centre memo served a slot the wrong polytope's cells"
        );
    }

    fn the_fill_path_reaches_the_allocator_zero_times() {
        let physics = PlaygroundPhysics::new(1, BODY_SIZE);
        let frame = frame_of(
            &physics,
            ROW_16,
            rotor_at(Plane4::Xz, 0.5),
            Projection::Identity,
            SLICE_W,
            CAMERA_DISTANCE,
        );
        let cross = SectionLayer::CROSS_SECTION_DEFAULT;
        let cap = SectionLayer {
            perimeter: true,
            surface_alpha: 0.5,
        };

        let mut buffers = OverlayBuffers::default();
        buffers.sections(&frame, cross, cap);
        assert!(
            !buffers.cross_faces.vertices.is_empty() && !buffers.cap_faces.vertices.is_empty(),
            "no fill was emitted, so the pin is vacuous"
        );

        let warm = alloc_probe::bytes_allocated_by(|| buffers.sections(&frame, cross, cap));
        assert_eq!(
            warm, 0,
            "a warm frame with both fills on asked the allocator for {warm} bytes"
        );
    }

    /// `scaled` is `base` scaled about `centre` by `scale`, point for point.
    fn assert_scaled_about(
        scaled: &[[f32; 3]],
        base: &[[f32; 3]],
        centre: Vec3,
        scale: f32,
        what: &str,
    ) {
        assert!(
            !base.is_empty(),
            "{what}: nothing was emitted, so the pin is vacuous"
        );
        assert_eq!(
            scaled.len(),
            base.len(),
            "{what}: the two layers emitted different geometry"
        );
        for (i, (s, b)) in scaled.iter().zip(base).enumerate() {
            let expected = (Vec3::from_array(*b) - centre) * scale + centre;
            assert!(
                (Vec3::from_array(*s) - expected).length() < TRANSLATE_TOL,
                "{what} point {i}: {s:?} is not {b:?} scaled by {scale} about {centre:?}"
            );
        }
    }
}
