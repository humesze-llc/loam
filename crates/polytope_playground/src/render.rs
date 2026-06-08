//! GPU rendering for `Demo`: the wireframe overlay, point cloud, and the
//! rasterized section-face passes.

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
                let _scope = rye_time::frame_trace::scope("pp-sdf");
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
                let _scope = rye_time::frame_trace::scope("pp-section-faces");
                self.render_section_faces(rd, view)?;
            }
            // Cross-section + wireframe overlay. Shapes view only: Filmstrip's
            // per-cell composition would need per-cell depth-clear + uploads not
            // worth the v1 plumbing.
            if self.wireframe_enabled {
                let _scope = rye_time::frame_trace::scope("pp-wireframe");
                self.render_wireframe_overlay(rd, view)?;
            }
            // Points overlay, drawn last so discs sit on top of edges and caps.
            if self.points_enabled {
                let _scope = rye_time::frame_trace::scope("pp-points");
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

    /// Build the combined point sprites mesh (vertex markers + cell-center
    /// sprites) across every polychoral body in the row, upload it, and execute
    /// the point-disc raster pass. Same body-local project-then-translate pattern
    /// as the wireframe and section-faces paths.
    ///
    /// Sprites honor the active [`WireframeColorMode`] so the overlay reads as
    /// part of the wireframe pass; `UniqueEdge` falls back to the position
    /// gradient (the edge-indexed palette has no canonical vertex assignment).
    fn render_points(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        let cfg = &rd.surface_bundle.config;
        // Rendered row: full `row` in Shapes, the `strip_subject` in Single.
        let render_row = state::render_row_entries(self.view_mode, &self.row, &self.strip_subject);
        let n = render_row.len();
        let wireframe_projection = self.resolved_wireframe_projection();
        // Near-pole drop radius shared with the edges and cap outline: a point in
        // the pole band would draw as a giant clamp disc while its edges drop, so
        // the same predicate gates it. Resolved per shape (only 16-cell clipped).
        let cam_dist = self.camera_distance_to_focus();
        // Active-mode palette: green for vertices in an intersected cell, gray
        // otherwise. Same hues as the wireframe overlay.
        const ACTIVE_GREEN: [f32; 4] = [0.40, 1.00, 0.55, 1.0];
        const INACTIVE_GRAY: [f32; 4] = [0.55, 0.55, 0.58, 0.85];
        let color_mode = self.wireframe_color_mode;

        let body_size = self.effective_body_size();
        let w_slice = self.w_slice;
        let mesh = &mut self.points_mesh_scratch;
        mesh.positions.clear();
        mesh.colors.clear();
        mesh.sizes.clear();

        for (slot, entry) in render_row.iter().enumerate() {
            let Some(polytope) = entry.shape.polytope4() else {
                continue;
            };
            // Per-shape clip: finite for the 16-cell, none for the rest.
            let points_clip_radius = stereographic_clip_radius(
                &wireframe_projection,
                stereographic_view_radius(polytope, cam_dist),
            );
            let topo = polytope.topology();
            let body_pos = body_position(slot, n);
            let body_pos_r3 = Vec3::new(body_pos[0], body_pos[1], body_pos[2]);

            // Body-local 4D vertices (rotated + scaled), shared by the vertex and
            // cell-center loops (WDepth normalization, Active cell strengths).
            let local_vertices: Vec<Vec4> = topo
                .vertices
                .iter()
                .map(|v| body_size * self.rot_state.apply(*v))
                .collect();
            // Canonical-max-w normalization (see render_wireframe_overlay), so
            // the points' w-depth color stays in step with the edges.
            let w_extent_local: f32 = if matches!(color_mode, WireframeColorMode::WDepth) {
                let canonical_max_w = topo
                    .vertices
                    .iter()
                    .map(|v| v.w.abs())
                    .fold(0.0_f32, f32::max)
                    .max(1e-6);
                canonical_max_w * body_size
            } else {
                1.0
            };
            // Cell strengths only for Active mode (`compute_cell_strengths`, same
            // as the wireframe overlay).
            let cell_strengths: Vec<f32> = if matches!(color_mode, WireframeColorMode::Active) {
                compute_cell_strengths(topo.cells, &local_vertices, w_slice)
            } else {
                Vec::new()
            };
            let vertex_is_active = |vi: usize| -> bool {
                topo.cells
                    .iter()
                    .zip(cell_strengths.iter())
                    .any(|(cell, &s)| s > 0.0 && cell.contains(&(vi as u32)))
            };

            if self.points_show_vertices {
                for (vi, v) in topo.vertices.iter().enumerate() {
                    let v_local = local_vertices[vi];
                    let v3_local =
                        <rye_math::EuclideanR4 as rye_math::RasterizableSpace<4>>::project_point(
                            v_local,
                            &wireframe_projection,
                        );
                    // Drop a near-pole vertex (clean blink) instead of a giant disc.
                    if !sample_in_radius(v3_local, points_clip_radius) {
                        continue;
                    }
                    let v_world = v3_local + body_pos_r3;
                    let color = match color_mode {
                        // Position covers UniqueEdge too (edge-indexed palette has
                        // no per-vertex assignment).
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
                    mesh.sizes.push(self.points_size_px);
                }
            }
            if self.points_show_cell_centers {
                // Pull centroids inward by `CELL_CENTER_INSET` so they read as
                // interior markers. `cell_centers()` returns centroids at the
                // inradius, which is the DUAL's vertex set; at full inradius the
                // sprites look like the wrong polytope, so inset them inside the cap.
                const CELL_CENTER_INSET: f32 = 0.5;
                let centers = polytope.cell_centers();
                for (ci, c) in centers.iter().enumerate() {
                    let c_local = body_size * CELL_CENTER_INSET * self.rot_state.apply(*c);
                    let c3_local =
                        <rye_math::EuclideanR4 as rye_math::RasterizableSpace<4>>::project_point(
                            c_local,
                            &wireframe_projection,
                        );
                    // Same near-pole drop as the vertex loop above.
                    if !sample_in_radius(c3_local, points_clip_radius) {
                        continue;
                    }
                    let c_world = c3_local + body_pos_r3;
                    let color = match color_mode {
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
                    mesh.sizes.push(self.points_size_px * 0.5);
                }
            }
        }

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
            mesh,
            &rye_math::Projection::Identity,
        );
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

        // Resolve once before any `&mut` scratch borrow. The honest layer
        // overrides this to drop-w via `section_layer_projection`.
        let wireframe_projection = self.resolved_wireframe_projection();
        let w_slice = self.w_slice;

        // One pass over the row; the body-local 4D vertices are shared. Each
        // layer keeps its own scratch mesh and capacity across frames.
        self.build_section_layer_meshes(wireframe_projection, w_slice, cross, cap);

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

    /// Append every polychoral body's cross-section caps into the visible layer
    /// scratch meshes, each under its own projection
    /// ([`state::section_layer_projection`]: drop-w for the honest cross-section,
    /// the active projection for the cap). Both meshes are cleared on entry and
    /// reuse allocations; an invisible layer is skipped.
    ///
    /// Single pass over the row: the body-local 4D section vertices are identical
    /// for both layers, computed once per body before the per-layer transform.
    fn build_section_layer_meshes(
        &mut self,
        wireframe_projection: rye_math::Projection<4>,
        w_slice: f32,
        cross: state::SectionLayer,
        cap: state::SectionLayer,
    ) {
        let render_row = state::render_row_entries(self.view_mode, &self.row, &self.strip_subject);
        let n = render_row.len();
        let body_size = self.effective_body_size();

        // Honest layer is drop-w (Identity makes `perspective_scale_at_w` report
        // `Some(1.0)`, a scale-by-one + translate); the cap follows the active
        // projection.
        let cross_projection = state::section_layer_projection(true, wireframe_projection);
        let cap_projection = state::section_layer_projection(false, wireframe_projection);
        let cross_scale = perspective_scale_at_w(w_slice, &cross_projection);
        let cap_scale = perspective_scale_at_w(w_slice, &cap_projection);
        // Near-pole drop radius resolved per shape below (only the 16-cell is
        // clipped); the cross layer is drop-w so it never clips.
        let cam_dist = self.camera_distance_to_focus();

        // Reused per-vertex projected-point buffer for the fill clip, taken out so
        // `append_layer` can hold it `&mut` alongside the immutable
        // `section_world_vertices_scratch` borrow. Put back so capacity persists.
        let mut proj_scratch = std::mem::take(&mut self.section_clip_projected_scratch);

        let cross_mesh = &mut self.section_faces_mesh_scratch;
        cross_mesh.vertices.clear();
        cross_mesh.colors.clear();
        cross_mesh.indices.clear();
        let cap_mesh = &mut self.section_faces_projected_scratch;
        cap_mesh.vertices.clear();
        cap_mesh.colors.clear();
        cap_mesh.indices.clear();

        for (slot, entry) in render_row.iter().enumerate() {
            let Some(polytope) = entry.shape.polytope4() else {
                continue;
            };
            // Per-shape clip: finite for the 16-cell, none for the rest.
            let view_radius = stereographic_view_radius(polytope, cam_dist);
            let cross_clip = stereographic_clip_radius(&cross_projection, view_radius);
            let cap_clip = stereographic_clip_radius(&cap_projection, view_radius);
            let topo = polytope.topology();
            let body_pos = body_position(slot, n);
            let body_pos_r3 = Vec3::new(body_pos[0], body_pos[1], body_pos[2]);

            // Body-local 4D section vertices (no world translate): keep the body's
            // R³ position out of the perspective divide. Shared by both layers.
            self.section_world_vertices_scratch.clear();
            self.section_world_vertices_scratch.extend(
                topo.vertices
                    .iter()
                    .map(|v| body_size * self.rot_state.apply(*v)),
            );

            // Match the SDF's per-body coloring: catalog color, Lambert depth.
            // Alpha is the layer's `surface_alpha`; below 1.0 it renders through
            // the no-depth-write pipeline so layers behind composite through.
            let [r, g, b] = entry.body_color;

            // Append + world-transform one layer's caps.
            // `cap_vertex_projected_and_world` maps each body-local cap vertex
            // through the layer's projection and returns the projected point the
            // clip tests. Under Stereographic a fill triangle is dropped when any
            // vertex exceeds `clip_radius`, matching the perimeter rule so fill
            // and outline cull in lockstep.
            let append_layer = |mesh: &mut rye_shape::TriangleMesh<3>,
                                proj_scratch: &mut Vec<Vec3>,
                                alpha: f32,
                                projection: &rye_math::Projection<4>,
                                scale: Option<f32>,
                                clip_radius: Option<f32>| {
                let start_v = mesh.vertices.len();
                let start_i = mesh.indices.len();
                polytope_section_faces_append(
                    topo.edges,
                    topo.cells,
                    &self.section_world_vertices_scratch,
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
                    &mut proj_scratch,
                    cross.surface_alpha,
                    &cross_projection,
                    cross_scale,
                    cross_clip,
                );
            }
            if cap.fill_visible() {
                append_layer(
                    cap_mesh,
                    &mut proj_scratch,
                    cap.surface_alpha,
                    &cap_projection,
                    cap_scale,
                    cap_clip,
                );
            }
        }

        // Return the reused buffer so its capacity persists across frames.
        self.section_clip_projected_scratch = proj_scratch;
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
        node.upload::<EuclideanR3, 3>(&rd.device, &rd.queue, mesh, &rye_math::Projection::Identity);
        node.execute(rd, view, Some(depth_view), None)?;
        Ok(())
    }

    /// Distance from the camera eye to the orbit target, used to scale the
    /// stereographic clip radius (see [`stereographic_view_radius`]). Reads the
    /// live eye position, not `orbit.distance`, so it is correct in FreeRoam too.
    fn camera_distance_to_focus(&self) -> f32 {
        (self.camera.position - self.orbit.target).length()
    }

    /// Build the section-perimeter and parent-wireframe overlay meshes from the
    /// current row + rotor + w_slice, upload them, and execute the raster passes
    /// over the SDF render. Non-polychoral shapes in the row are skipped (no
    /// [`rye_physics::polytope::Polytope4`] mapping).
    fn render_wireframe_overlay(
        &mut self,
        rd: &RenderDevice,
        view: &wgpu::TextureView,
    ) -> Result<()> {
        let cfg = &rd.surface_bundle.config;
        // Rendered row: full `row` in Shapes, the `strip_subject` in Single.
        // Disjoint field borrows keep `&mut self.unique_edge_palette_cache`
        // accessible in the loop.
        let render_row = state::render_row_entries(self.view_mode, &self.row, &self.strip_subject);
        let n = render_row.len();

        let mut section_edges = LineMesh::<3>::default();
        let mut parent_lines = LineMesh::<3>::default();
        // `nearest-active` off: uniform `wireframe_alpha`. On: per-edge interp
        // between DIM (slice misses the cell) and BRIGHT (slice at its midpoint).
        // DIM/BRIGHT are activity-gradient peaks, not a global opacity.
        const PARENT_ALPHA_DIM: f32 = 0.10;
        const PARENT_ALPHA_BRIGHT: f32 = 0.85;
        let parent_alpha_uniform = self.wireframe_alpha;
        let parent_width = self.wireframe_width_px;
        // Active-mode palette: green for edges in an intersected cell, gray
        // otherwise (binary contrast against the scene backdrop).
        const ACTIVE_GREEN: [f32; 4] = [0.40, 1.00, 0.55, 1.0];
        const INACTIVE_GRAY: [f32; 4] = [0.55, 0.55, 0.58, 1.0];
        let nearest_active = self.wireframe_nearest_active;
        let color_mode = self.wireframe_color_mode;
        // Same projection for every body, so all share a consistent R³ embedding.
        let wireframe_projection = self.resolved_wireframe_projection();
        // Clip radius resolved per shape in the loop (only the 16-cell clips).
        let cam_dist = self.camera_distance_to_focus();
        // Honest layer's outline is forced to drop-w so it can never follow a
        // distorting projection.
        let cross_perimeter = self.cross_section.perimeter;
        let cap_perimeter = self.projected_cap.perimeter;
        let cross_section_projection = state::section_layer_projection(true, wireframe_projection);
        // Edge geometry is projection-derived: Stereographic draws S³ arcs, the
        // affine projections draw flat R4 chords (see `state::default_edge_blend`).
        let space_blend = state::default_edge_blend(self.wireframe_projection);
        // Wireframe Hyperslice cull: keep only edges whose body-local w-interval
        // intersects the slab around `w_slice`, thinning the graph to what the
        // slice passes through. A demo-side per-edge filter, deliberately not a
        // `Projection` variant (the projection has discarded w, so it cannot
        // carry a keep/drop signal).
        //
        // The `Hyperslice` projection mode IS this cull paired with drop-w
        // (resolves to `Identity`), so selecting it activates the filter even with
        // the standalone toggle off; the toggle composes with any other mode.
        let hyperslice_on = self.hyperslice_cull_active();
        let hyperslice_thickness = self.wireframe_hyperslice_thickness;
        let hyperslice_w_slice = self.w_slice;
        // Reused great-circle buffer; `push_blended_edge` clears it per edge,
        // put back after the loop so capacity persists.
        let mut slerp_scratch = std::mem::take(&mut self.slerp_scratch);

        for (slot, entry) in render_row.iter().enumerate() {
            let Some(polytope) = entry.shape.polytope4() else {
                continue;
            };
            // Per-shape clip radius: finite for the 16-cell, `f32::INFINITY` else.
            let view_radius = stereographic_view_radius(polytope, cam_dist);
            let topo = polytope.topology();
            let body_pos = body_position(slot, n);
            // Body's R³ position. The body sits at `w = 0`, so projecting in
            // body-local 4D and translating in R³ AFTER keeps its apparent
            // x-position stable when Perspective4D scales (x, y, z) by
            // `focal / (focal - w)`.
            let body_pos_r3 = Vec3::new(body_pos[0], body_pos[1], body_pos[2]);
            // Body-local 4D vertices (no world translate; that follows projection).
            let body_size = self.effective_body_size();
            let local_vertices: Vec<Vec4> = topo
                .vertices
                .iter()
                .map(|v| body_size * self.rot_state.apply(*v))
                .collect();

            // Cross-section perimeter outlines, one per enabled layer.
            // `polytope_section_overlay_with_vertices` returns the body-local
            // drop-w perimeter once; the honest layer maps it through drop-w (so
            // its outline matches the SDF slice), the cap through the active
            // projection. Under Stereographic a segment is dropped when either
            // endpoint exceeds the clip radius (per-segment: a perimeter segment
            // is a single cap edge).
            if cross_perimeter || cap_perimeter {
                let (_tri, perim) = polytope_section_overlay_with_vertices(
                    topo.edges,
                    topo.cells,
                    &local_vertices,
                    WPlane::new(self.w_slice),
                );
                let w_slice = self.w_slice;
                let mut push_perimeter = |projection: &rye_math::Projection<4>| {
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
                        if !sample_in_radius(pa, clip_radius) || !sample_in_radius(pb, clip_radius)
                        {
                            continue;
                        }
                        section_edges.segments.push((wa, wb));
                        section_edges.colors.push(*color);
                        section_edges.widths.push(*width);
                    }
                };
                // Honest cross-section first (drop-w), then the projected cap.
                if cross_perimeter {
                    push_perimeter(&cross_section_projection);
                }
                if cap_perimeter {
                    push_perimeter(&wireframe_projection);
                }
            }

            // Per-cell crossing strength in [0, 1], shared with render_points via
            // `compute_cell_strengths` so both passes agree on "active".
            let cell_strengths = compute_cell_strengths(topo.cells, &local_vertices, self.w_slice);

            // Per-edge brightness: max strength over cells containing both
            // endpoints, so an edge lights up when any containing cell is crossed.
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

            // Hyperslice cull, CELL-level to match `edge_is_active`: an edge
            // survives iff some cell containing both endpoints has its w-range
            // overlapping the slab. The edge-level test would cull a far-side edge
            // of an active cell that the cell-level coloring paints green. The
            // slab band is a superset of `edge_is_active` (the zero-width plane),
            // so active edges are never culled while the band thins the graph.
            let edge_in_slab_cell = |i: u32, j: u32| -> bool {
                topo.cells.iter().any(|cell| {
                    if !(cell.contains(&i) && cell.contains(&j)) {
                        return false;
                    }
                    let (w_min, w_max) = cell_w_range(cell, &local_vertices);
                    slab_overlaps(w_min, w_max, hyperslice_w_slice, hyperslice_thickness)
                })
            };

            // Base RGB per `wireframe_color_mode`: `VertexGradient` position-
            // derived hue, `UniqueEdge` greedy line-graph coloring, `WDepth`
            // signed-w cool-to-warm, `Active` binary green/gray. Alpha is then the
            // `nearest-active` strength or uniform.
            //
            // UniqueEdge palette is topology-only; memoize by `Polytope4` so the
            // 600-cell's ~520k pair-checks run once per launch, not per frame.
            let edge_palette: &[[f32; 4]] = if matches!(color_mode, WireframeColorMode::UniqueEdge)
            {
                self.unique_edge_palette_cache
                    .entry(polytope)
                    .or_insert_with(|| unique_edge_palette(topo.edges))
            } else {
                &[]
            };
            // WDepth normalizes against the CANONICAL max |w| (not the rotated
            // per-frame max), so the color stays temporally stable as the rotor
            // swings a vertex from -w to +w. Per-polytope because the band differs
            // (the tesseract's `[-0.5, +0.5]` does not fit the 16-cell, 5-cell).
            let w_extent_local: f32 = if matches!(color_mode, WireframeColorMode::WDepth) {
                let canonical_max_w = topo
                    .vertices
                    .iter()
                    .map(|v| v.w.abs())
                    .fold(0.0_f32, f32::max)
                    .max(1e-6);
                canonical_max_w * body_size
            } else {
                1.0
            };
            for (edge_idx, &[i, j]) in topo.edges.iter().enumerate() {
                let ia = i as usize;
                let ja = j as usize;
                let a = local_vertices[ia];
                let b = local_vertices[ja];
                // Cell-level Hyperslice cull before any color / projection work,
                // so a culled edge costs only the membership fold; `&&`
                // short-circuits it when the affordance is off.
                if hyperslice_on && !edge_in_slab_cell(i, j) {
                    continue;
                }
                let (mut color_a, mut color_b) = match color_mode {
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
                let alpha = if nearest_active {
                    let s = edge_strength(i, j);
                    PARENT_ALPHA_DIM + (PARENT_ALPHA_BRIGHT - PARENT_ALPHA_DIM) * s
                } else {
                    parent_alpha_uniform
                };
                color_a[3] = alpha;
                color_b[3] = alpha;
                // Emit the edge in body-local 4D, projected to world R³;
                // `blend` is projection-derived (Stereographic -> 1, affine -> 0).
                push_blended_edge(
                    &mut parent_lines,
                    a,
                    b,
                    color_a,
                    color_b,
                    parent_width,
                    space_blend,
                    &wireframe_projection,
                    body_pos_r3,
                    &mut slerp_scratch,
                    view_radius,
                );
            }
        }

        // Put the great-circle buffer back so its capacity is reused next frame.
        self.slerp_scratch = slerp_scratch;

        // Upload (no-op when a mesh is empty).
        self.section_edges.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &section_edges,
            &rye_math::Projection::Identity,
            1,
        );
        self.parent_wireframe.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &parent_lines,
            &rye_math::Projection::Identity,
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
