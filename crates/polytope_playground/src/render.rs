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
        // Mesh taken out of `self` for the build: the row frame borrows the
        // physics world for as long as the mesh it fills is borrowed. Put back
        // so its capacity persists across frames.
        let mut mesh = std::mem::take(&mut self.points_mesh_scratch);
        build_points_mesh(&self.row_frame(), &style, &mut mesh);

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
        build_section_layer_meshes(
            &self.row_frame(),
            cross,
            cap,
            &mut local_vertices,
            &mut proj_scratch,
            &mut cross_mesh,
            &mut cap_mesh,
        );
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
        // Cache + great-circle buffer taken out of `self` for the build: the row
        // frame borrows the physics world for as long as they are borrowed. Put
        // back so their capacity persists across frames.
        let mut palette_cache = std::mem::take(&mut self.unique_edge_palette_cache);
        let mut slerp_scratch = std::mem::take(&mut self.slerp_scratch);
        let (section_edges, parent_lines) = build_wireframe_meshes(
            &self.row_frame(),
            &style,
            cross,
            cap,
            &mut palette_cache,
            &mut slerp_scratch,
        );
        self.unique_edge_palette_cache = palette_cache;
        self.slerp_scratch = slerp_scratch;

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
/// Free function over [`RowFrame`] so "the sprites sit on the body physics put
/// there" is unit-testable without a GPU-backed [`Demo`];
/// [`Demo::render_points`] is the one production caller.
pub(crate) fn build_points_mesh(
    frame: &RowFrame<'_>,
    style: &PointsStyle,
    mesh: &mut loam_shape::PointMesh<3>,
) {
    // Active-mode palette: green for vertices in an intersected cell, gray
    // otherwise. Same hues as the wireframe overlay.
    const ACTIVE_GREEN: [f32; 4] = [0.40, 1.00, 0.55, 1.0];
    const INACTIVE_GRAY: [f32; 4] = [0.55, 0.55, 0.58, 0.85];

    mesh.positions.clear();
    mesh.colors.clear();
    mesh.sizes.clear();
    // Body-frame buffers hoisted out of the row loop; `body_local` refills them
    // per body, so the row costs one growth, not one per shape.
    let mut local_vertices: Vec<Vec4> = Vec::new();
    let mut center_locals: Vec<Vec4> = Vec::new();

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
        let body_pos_r3 =
            frame.body_local(slot, topo.vertices, frame.body_size, &mut local_vertices);
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
        let cell_strengths: Vec<f32> = if matches!(style.color_mode, WireframeColorMode::Active) {
            compute_cell_strengths(topo.cells, &local_vertices, frame.w_slice)
        } else {
            Vec::new()
        };
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
            let centers = polytope.cell_centers();
            frame.body_local(
                slot,
                &centers,
                frame.body_size * CELL_CENTER_INSET,
                &mut center_locals,
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
    local_vertices: &mut Vec<Vec4>,
    proj_scratch: &mut Vec<Vec3>,
    cross_mesh: &mut loam_shape::TriangleMesh<3>,
    cap_mesh: &mut loam_shape::TriangleMesh<3>,
) {
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
/// Free function over [`RowFrame`] so "the edges wrap the body physics put
/// there" is unit-testable without a GPU-backed [`Demo`];
/// [`Demo::render_wireframe_overlay`] is the one production caller.
pub(crate) fn build_wireframe_meshes(
    frame: &RowFrame<'_>,
    style: &WireframeStyle,
    cross: state::SectionLayer,
    cap: state::SectionLayer,
    palette_cache: &mut std::collections::HashMap<loam_physics::polytope::Polytope4, Vec<[f32; 4]>>,
    slerp_scratch: &mut Vec<Vec4>,
) -> (LineMesh<3>, LineMesh<3>) {
    let mut section_edges = LineMesh::<3>::default();
    let mut parent_lines = LineMesh::<3>::default();
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
    // Body-frame buffer hoisted out of the row loop; `body_local` refills it per
    // body, so the row costs one growth, not one per shape.
    let mut local_vertices: Vec<Vec4> = Vec::new();

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
        let body_pos_r3 =
            frame.body_local(slot, topo.vertices, frame.body_size, &mut local_vertices);

        // Cross-section perimeter outlines, one per enabled layer.
        // `polytope_section_overlay_with_vertices` returns the body-local drop-w
        // perimeter once; the honest layer maps it through drop-w (so its
        // outline matches the SDF slice), the cap through the active projection.
        // Under Stereographic a segment is dropped when either endpoint exceeds
        // the clip radius (per-segment: a perimeter segment is a single cap edge).
        if cross.perimeter || cap.perimeter {
            let (_tri, perim) = polytope_section_overlay_with_vertices(
                topo.edges,
                topo.cells,
                &local_vertices,
                WPlane::new(w_slice),
            );
            let mut push_perimeter = |projection: &loam_math::Projection<4>| {
                let section_scale = perspective_scale_at_w(w_slice, projection);
                let clip_radius = stereographic_clip_radius(projection, view_radius);
                for ((a, b), (color, width)) in perim
                    .segments
                    .iter()
                    .zip(perim.colors.iter().zip(perim.widths.iter()))
                {
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
        let cell_strengths = compute_cell_strengths(topo.cells, &local_vertices, w_slice);

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
                let (w_min, w_max) = cell_w_range(cell, &local_vertices);
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
                &mut parent_lines,
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

    (section_edges, parent_lines)
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
        let spin = (Plane4::Xy.unit_bivector() * 0.7).exp().normalize();
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

    /// Drop-w and a fixed camera: the builders' clip and scale paths then
    /// depend on the body pose alone.
    fn frame(physics: &PlaygroundPhysics, spin: Rotor4) -> RowFrame<'_> {
        RowFrame {
            physics,
            row: ROW,
            spin,
            body_size: BODY_SIZE,
            projection: Projection::Identity,
            w_slice: 0.0,
            camera_distance: 4.0,
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
        let mut palette_cache = std::collections::HashMap::new();
        let mut slerp_scratch = Vec::new();

        let (live_perimeter, live_edges) = build_wireframe_meshes(
            &frame(&pair.thrown, pair.spin),
            &style,
            cross,
            cap,
            &mut palette_cache,
            &mut slerp_scratch,
        );
        let (rest_perimeter, rest_edges) = build_wireframe_meshes(
            &frame(&pair.at_rest, pair.composed),
            &style,
            cross,
            cap,
            &mut palette_cache,
            &mut slerp_scratch,
        );

        assert_translated(
            &segment_points(&live_edges),
            &segment_points(&rest_edges),
            pair.delta,
            "parent wireframe",
        );
        assert_translated(
            &segment_points(&live_perimeter),
            &segment_points(&rest_perimeter),
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
        let mut live = loam_shape::PointMesh::<3>::default();
        let mut rest = loam_shape::PointMesh::<3>::default();

        build_points_mesh(&frame(&pair.thrown, pair.spin), &style, &mut live);
        build_points_mesh(&frame(&pair.at_rest, pair.composed), &style, &mut rest);

        assert_translated(
            &live.positions,
            &rest.positions,
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
            &mut local_vertices,
            &mut proj_scratch,
            &mut live_cross,
            &mut live_cap,
        );
        build_section_layer_meshes(
            &frame(&pair.at_rest, pair.composed),
            cross,
            cap,
            &mut local_vertices,
            &mut proj_scratch,
            &mut rest_cross,
            &mut rest_cap,
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
}
