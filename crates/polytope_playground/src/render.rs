//! GPU rendering for `Demo`: the wireframe overlay, point cloud, and the
//! rasterized section-face passes.

use crate::*;

impl Demo {
    pub(crate) fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        // Scene renders to the full window. The bottom controls overlay floats on top
        // (Area, not a docked panel; doesn't reserve pixels); the Render settings modal
        // is also free-floating, so the scene viewport is always the framebuffer.
        let cfg = &rd.surface_bundle.config;
        let viewport = Viewport::full([cfg.width, cfg.height]);
        if self.view_mode == ViewMode::Filmstrip {
            // Filmstrip: each cell shows the `strip_subject`
            // polytope (independent of `self.row`) at a different
            // `w_slice`. We swap the GPU body list to just the
            // subject for the duration of this render, then
            // restore via `rebuild_bodies` so the Shapes view and
            // any subsequent state read sees the full row.
            let entry = self.strip_subject;
            // 2D filmstrip rendering. cols is the column count
            // (horizontal axis), rows is the row count (vertical).
            // Default axis assignment: w on columns, t on rows;
            // `strip_swap_axes` flips it.
            //
            // Per-cell rendering: viewport (cell rect), w_slice
            // (cell's w), and body (cell's rotor for that t).
            // The base rotor `self.rot_state` is offset along
            // omega_animation by `(t_offset)` to give the cell's
            // rotor: `exp(omega_animation * t_offset) * rot_state`.
            // For the w-only and t-only 1D cases, the second
            // axis collapses to a single cell with offset=0.
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
                // UI invariant prevents both being off; defensive.
                (false, false) => (1, 1, true),
            };
            let col_vps = viewport.split_horizontal(cols as u32);
            let mut grid_cells: Vec<(Viewport, f32, BodyUniform)> = Vec::with_capacity(cols * rows);
            for (col_idx, col_vp) in col_vps.into_iter().enumerate() {
                let row_vps = col_vp.split_vertical(rows as u32);
                for (row_idx, cell_vp) in row_vps.into_iter().enumerate() {
                    // Decide what (w_offset, t_offset) this cell
                    // corresponds to based on which axis carries
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
                        // Fan FORWARD only: cell 0 = now, cell
                        // last = rot_time + strip_t_extent. Reads
                        // as "what the rotor will look like at
                        // this future time."
                        let t_norm = t_idx as f32 / (t_n - 1) as f32;
                        t_norm * self.strip_t_extent
                    };
                    // Cell's rotor: the orientation at animation time
                    // `rot_time + t_offset`, via the same `rotor_at_time`
                    // dispatch the spin + t-scrub use. For Composer this
                    // equals the old `exp(omega * t_offset) * rot_state`
                    // (omega commutes with itself); for Active it's the
                    // product-of-exp sampled at the future time, which the
                    // old sum-based offset got wrong with 2+ active planes.
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
            // Restore the full row of bodies for any non-strip
            // consumer (state save, mode switch, etc.).
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
            // Shared depth attachment for the rasterized section pass + the parent
            // wireframe's depth-test. Ensured + cleared once per Shapes-view frame so
            // the order is: SDF (color only) -> section_faces (writes depth + color
            // when raster mode is on) -> wireframe (tests depth, no write). In SDF
            // mode no pass writes depth, so the cleared `1.0` buffer makes every
            // wireframe fragment pass the depth-test trivially -- visual unchanged.
            self.ensure_and_clear_shared_depth(rd)?;
            if matches!(self.surface_mode, SurfaceMode::Raster) {
                let _scope = rye_time::frame_trace::scope("pp-section-faces");
                self.render_section_faces(rd, view)?;
            }
            // Cross-section + parent-wireframe overlay (when toggled). Only in Shapes
            // view since Filmstrip's per-cell viewport composition would require
            // per-cell depth-clear + per-cell uploads that aren't worth the v1 plumbing.
            if self.wireframe_enabled {
                // pp-wireframe is the project-memory-flagged hot path
                // (`project_polychoral_raster_perf`). Want a per-frame number here so
                // we can confirm (or refute) that hypothesis with browser data.
                let _scope = rye_time::frame_trace::scope("pp-wireframe");
                self.render_wireframe_overlay(rd, view)?;
            }
            // Points overlay (vertex markers + cell-center sprites). Drawn last so the
            // discs sit on top of wireframe edges and section caps at the same depth.
            if self.points_enabled {
                let _scope = rye_time::frame_trace::scope("pp-points");
                self.render_points(rd, view)?;
            }
            Ok(())
        }
    }

    /// Ensure the shared section-faces depth attachment exists at the current
    /// swapchain size + sample count, then clear it to `1.0`. Called once per
    /// Shapes-view frame at the top of the rasterizer chain.
    ///
    /// The buffer is shared between `section_faces` (which writes depth when raster
    /// mode is on) and `parent_wireframe` (which depth-tests against it without
    /// writing). Sharing means we pay one ensure + one clear per frame regardless of
    /// surface mode.
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

    /// Build a combined section-faces mesh across every polychoral body in `self.row`,
    /// upload it, clear the section-faces depth attachment, and execute the triangle
    /// raster pass. No-op when no polychoral body is present in the row.
    ///
    /// Per-body world transform mirrors `render_wireframe_overlay`: canonical vertices
    /// `v` become `body_position + effective_body_size() * rot_state.apply(v)`. The section
    /// algorithm then runs on these world vertices against the demo's `w_slice`,
    /// producing R³ geometry that composes with the camera the SDF raymarcher uses.
    ///
    /// Depth: within a single cap, the fan triangles are coplanar (the cap is a 2D
    /// polygon in R³) and don't occlude one another. Different caps within one
    /// polytope, and caps across different polychoral bodies, all project to
    /// different camera depths after the view-projection step, so they DO require a
    /// depth attachment to resolve front-to-back occlusion correctly. The shared
    /// depth buffer is ensured + cleared at the top of the Shapes-view render path
    /// (see `Demo::ensure_and_clear_shared_depth`); this pass writes into it.
    /// Build the combined point sprites mesh (vertex markers + cell-center sprites) across
    /// every polychoral body in the row, upload it, and execute the point-disc raster pass.
    ///
    /// Same body-local + perspective + world-translate pattern as the wireframe and section-
    /// faces paths: each body's vertices and cell centers are computed in body-local 4D,
    /// projected through `wireframe_projection`, and translated by the body's R³ position so
    /// the perspective scale doesn't smear the body across its row x-offset.
    ///
    /// Coloring: sprites honor the active [`WireframeColorMode`] so the
    /// points overlay reads as part of the same wireframe rendering pass.
    /// Per-vertex sprites get the same color the matching wireframe vertex
    /// would carry; per-cell-center sprites get the dominant-cell color
    /// (cell strength for `Active`, cell-center w for `WDepth`, position-
    /// derived gradient otherwise). For `UniqueEdge` mode the points fall
    /// back to the position-gradient because the unique-edge palette is
    /// edge-indexed and has no canonical vertex assignment.
    fn render_points(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        let cfg = &rd.surface_bundle.config;
        // Rendered row: full `row` in Shapes, just the `strip_subject` in Single.
        // Disjoint field borrow so the `&mut self.points_mesh_scratch` below stays
        // accessible.
        let render_row = state::render_row_entries(self.view_mode, &self.row, &self.strip_subject);
        let n = render_row.len();
        let wireframe_projection = self.resolved_wireframe_projection();
        // Near-pole drop radius, shared with the wireframe edges and the cap
        // outline (`stereographic_clip_radius`): under Stereographic a vertex or
        // cell-center within the angular epsilon of the pole projects to the
        // large-but-finite clamp point, which would draw as a giant disc while
        // the touching edges are dropped. Gating the point push on the same
        // predicate keeps the points overlay consistent with the wireframe.
        // `None` for every other projection, so nothing is dropped there. The
        // radius is resolved per shape in the loop (only the 16-cell is clipped).
        let cam_dist = self.camera_distance_to_focus();
        // Active-mode palette: bright green for vertices that belong to a
        // currently-intersected cell, dim gray otherwise. Same hues the
        // wireframe overlay uses so the visual identity stays consistent.
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

            // Per-frame, per-polytope body-local 4D vertices (rotated + scaled).
            // Shared by the vertex AND cell-center loops: for WDepth we need
            // the maximum |w| across them to normalize the gradient, and for
            // Active we need them to derive cell strengths.
            let local_vertices: Vec<Vec4> = topo
                .vertices
                .iter()
                .map(|v| body_size * self.rot_state.apply(*v))
                .collect();
            // Same canonical-max-w normalization the wireframe overlay uses
            // (see render_wireframe_overlay), keeping the points' color in
            // step with the edges across the same w-depth scheme.
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
            // Cell strengths only computed for the Active mode; saves work in
            // the common case. Same definition the wireframe overlay uses
            // (`compute_cell_strengths`) so "vertex in an active cell" reads
            // consistently across edges + dots.
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
                    // Drop a near-pole vertex (clean blink) instead of a giant
                    // clamp disc; matches the wireframe/perimeter drop.
                    if !sample_in_radius(v3_local, points_clip_radius) {
                        continue;
                    }
                    let v_world = v3_local + body_pos_r3;
                    let color = match color_mode {
                        // Position-gradient also covers UniqueEdge, since the
                        // unique-edge palette is edge-indexed (no canonical
                        // assignment for a vertex shared by several edges).
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
                // Pull cell centroids radially inward by `CELL_CENTER_INSET` so
                // they read as "interior markers" instead of coinciding with the
                // DUAL polytope's vertices. Mathematically `polytope.cell_centers()`
                // returns each cell's centroid at the inradius, which IS the
                // dual's vertex set (16-cell's cell-centroids form a tesseract,
                // tesseract's form a 16-cell, etc). At full inradius the sprites
                // look like a smaller dual polytope sitting at the inradius; the
                // inset puts them visibly INSIDE the body's cap so a viewer reads
                // "one dot per cell, in the cell's direction" rather than
                // "vertices of the wrong polytope."
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
                    // Cell-center sprites half-sized so they don't compete visually with the
                    // brighter vertex discs.
                    mesh.sizes.push(self.points_size_px * 0.5);
                }
            }
        }

        // Camera matches the wireframe overlay / section faces (same view-projection).
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
        // No depth attachment: see `PointRasterNode::new` site for the rationale
        // (drop-w + ReadOnly LessEqual was occluding non-w=0 vertices behind their
        // own caps).
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
        // Nothing to fill: both layers off. The perimeter outlines are drawn by
        // the wireframe overlay, not here, so an all-alpha-zero section skips the
        // triangle passes entirely.
        if !cross.fill_visible() && !cap.fill_visible() {
            return Ok(());
        }

        // Resolve the active projection ONCE (the Schlegel arm reads
        // `schlegel_params` + `rot_state` immutably) before any `&mut` scratch
        // borrow. The honest layer overrides this to drop-w via
        // `section_layer_projection`; the projected cap keeps it.
        let wireframe_projection = self.resolved_wireframe_projection();
        let w_slice = self.w_slice;

        // Build whichever layers are visible, in one pass over the row so the
        // body-local 4D section vertices are computed once and shared. Each layer
        // gets its own scratch mesh; both keep their capacity across frames.
        self.build_section_layer_meshes(wireframe_projection, w_slice, cross, cap);

        // Camera matches the SDF raymarcher's effective view-projection (same as
        // the wireframe overlay uses), so pixel-aligned composition over the SDF.
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
    /// scratch meshes, mapping each layer's body-local R³ caps to world R³ under
    /// that layer's projection ([`state::section_layer_projection`]: drop-w for
    /// the honest cross-section, the active `wireframe_projection` for the
    /// projected cap). Both meshes are cleared on entry and reuse their
    /// allocations across frames; a layer that is not `SectionLayer::fill_visible`
    /// is skipped so its mesh stays empty.
    ///
    /// Single pass over the row: the body-local 4D section vertices
    /// (`rot_state`-rotated, `effective_body_size`-scaled) are identical for both
    /// layers, so they are computed once per body and both layers' caps are
    /// appended from the same `polytope_section_faces_append` source before the
    /// per-layer world transform.
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

        // The honest layer is always drop-w; the projected cap follows the active
        // projection. `Identity` makes `perspective_scale_at_w` report `Some(1.0)`,
        // so the honest cap is just scaled-by-one and translated (drop-w + world
        // translate), bit-identical to the inhabitant's view of the slice 3-flat.
        let cross_projection = state::section_layer_projection(true, wireframe_projection);
        let cap_projection = state::section_layer_projection(false, wireframe_projection);
        let cross_scale = perspective_scale_at_w(w_slice, &cross_projection);
        let cap_scale = perspective_scale_at_w(w_slice, &cap_projection);
        // Near-pole drop radius per layer, shared with the wireframe edges and the
        // cap outline. The cross layer is drop-w (Identity), so its radius is
        // `None` and every triangle is kept; the cap layer is `None` unless the
        // active projection is Stereographic. The radius is resolved PER SHAPE in
        // the loop (only the 16-cell is clipped), so capture the distance here.
        let cam_dist = self.camera_distance_to_focus();

        // Reused per-vertex projected-point buffer for the triangle-granularity
        // fill clip, taken out so the `append_layer` closure can hold it `&mut`
        // alongside the immutable `section_world_vertices_scratch` borrow without
        // a second `&mut self`. Put back at the end so its capacity persists.
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

            // Body-local 4D section vertices (rotor-rotated, scaled, NO world
            // translate): keep the body's R³ position out of the 4D perspective
            // math so it doesn't get scaled by `focal / (focal - w)`. Shared by
            // both layers below.
            self.section_world_vertices_scratch.clear();
            self.section_world_vertices_scratch.extend(
                topo.vertices
                    .iter()
                    .map(|v| body_size * self.rot_state.apply(*v)),
            );

            // Match the SDF's per-body solid coloring: every cap uses the body's
            // catalog color; per-face Lambert adds the geometric depth. Alpha is
            // the layer's own `surface_alpha`; below 1.0 the layer renders through
            // the no-depth-write pipeline so layers behind composite through.
            let [r, g, b] = entry.body_color;

            // Append + world-transform a single layer's caps for this body. The
            // section algorithm emits body-local drop-w R³;
            // `cap_vertex_projected_and_world` maps it through the layer's
            // projection (affine scale-and-translate, or per-vertex reconstruction
            // at `w_slice` for non-affine) and also returns the body-local
            // projected point the near-pole clip tests. Under a clipped projection
            // (Stereographic) a fill triangle is dropped when ANY of its three
            // projected vertices exceeds `clip_radius`, matching the per-segment
            // perimeter rule so fill and outline cull in lockstep. The triangle
            // drop is in-place over the just-appended index range; dropped
            // triangles leave orphan vertices that no kept triangle references.
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
                // affine `None` layers, which keep every triangle).
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

    /// Upload + execute one already-built section layer's triangle mesh. Picks the
    /// opaque vs translucent node by the layer's `alpha`: opaque (>= 1.0) writes
    /// depth so caps occlude one another within a polytope; translucent (< 1.0)
    /// skips depth-write so the parent wireframe (and any layer drawn behind) shows
    /// through. `is_cross_section` selects which scratch mesh to upload. Each call
    /// is a self-contained submit, so the two nodes are reused across both layers.
    fn execute_section_layer(
        &mut self,
        rd: &RenderDevice,
        view: &wgpu::TextureView,
        view_proj: Mat4,
        alpha: f32,
        is_cross_section: bool,
    ) -> Result<()> {
        // Disjoint field borrows: the depth attachment, the chosen scratch mesh,
        // and the chosen node are three distinct fields, so the borrow checker
        // accepts the immutable depth + scratch reads alongside the `&mut` node
        // within this one method body (the same pattern the pre-split path used).
        // The shared depth attachment is ensured + cleared once per frame by
        // `ensure_and_clear_shared_depth`; here we just consume its view.
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
        // Empty-mesh handling lives in `TriangleRasterNode::execute` (it short-
        // circuits when `index_count == 0`); no redundant early-return here.
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

    /// Build the three overlay meshes (section triangles, section perimeter edges, parent
    /// wireframe) from the current row + rotor + w_slice, upload them, clear the overlay
    /// depth buffer, and execute the three raster passes on top of the existing SDF render.
    ///
    /// Distance from the camera eye to the scene focus (the orbit target), used to
    /// scale the stereographic clip radius (see [`stereographic_view_radius`]).
    /// Reads the live eye position rather than `orbit.distance` so it is correct in
    /// FreeRoam too, where the orbit controller is not driving the camera.
    fn camera_distance_to_focus(&self) -> f32 {
        (self.camera.position - self.orbit.target).length()
    }

    /// Per-body transform: each canonical Polytope4 vertex `v` becomes the world Vec4
    /// `body.position + effective_body_size() * rot_state.apply(v)`. The section algorithm
    /// then runs on these world vertices against the demo's `w_slice`, producing geometry
    /// in world R³ that composes cleanly with the SDF camera frame.
    ///
    /// Non-polychoral shapes (Clifford torus, duocylinder, etc.) in the row are skipped:
    /// they have no [`rye_physics::polytope::Polytope4`] mapping and the cross-section
    /// algorithm doesn't apply to smooth surfaces.
    fn render_wireframe_overlay(
        &mut self,
        rd: &RenderDevice,
        view: &wgpu::TextureView,
    ) -> Result<()> {
        let cfg = &rd.surface_bundle.config;
        // Rendered row: full `row` in Shapes, just the `strip_subject` in Single.
        // Bound via disjoint field borrows so the `&mut self.unique_edge_palette_cache`
        // inside the loop stays accessible; no allocation on this hot path.
        let render_row = state::render_row_entries(self.view_mode, &self.row, &self.strip_subject);
        let n = render_row.len();

        // Build combined meshes across the rendered row.
        let mut section_edges = LineMesh::<3>::default();
        let mut parent_lines = LineMesh::<3>::default();
        // Uniform-alpha endpoints when `nearest-active` is off; the active-mode mapping
        // interpolates between DIM (cells the slice misses entirely) and BRIGHT (cells
        // the slice is at the midpoint of).
        // When `wireframe nearest-active` is OFF, every edge gets the user-
        // tuneable uniform alpha (`wireframe_alpha`, default 1.0). When ON,
        // edges interpolate between DIM (cells the slice misses entirely)
        // and BRIGHT (cells the slice is at the midpoint of). DIM/BRIGHT
        // stay hardcoded because they encode "very off" / "very on" peaks
        // of the activity gradient, not a global opacity setting.
        const PARENT_ALPHA_DIM: f32 = 0.10;
        const PARENT_ALPHA_BRIGHT: f32 = 0.85;
        let parent_alpha_uniform = self.wireframe_alpha;
        let parent_width = self.wireframe_width_px;
        // Active-mode palette. Green for edges in any currently-intersected cell;
        // neutral gray for the rest. Chosen for clear binary contrast against the
        // grayish-blue scene backdrop and the dim ground checkerboard.
        const ACTIVE_GREEN: [f32; 4] = [0.40, 1.00, 0.55, 1.0];
        const INACTIVE_GRAY: [f32; 4] = [0.55, 0.55, 0.58, 1.0];
        let nearest_active = self.wireframe_nearest_active;
        let color_mode = self.wireframe_color_mode;
        // Resolve once per frame; same projection applied to every body's wireframe so all
        // bodies share a consistent R³ embedding.
        let wireframe_projection = self.resolved_wireframe_projection();
        // Camera-to-focus distance captured once (Copy f32); the stereographic
        // clip radius is resolved PER SHAPE inside the loop below, since only the
        // 16-cell is clipped (see `stereographic_view_radius`).
        let cam_dist = self.camera_distance_to_focus();
        // Per-layer perimeter toggles + the honest layer's always-drop-w
        // projection (the active projection is forced to `Identity` for the
        // cross-section so its outline can never follow a distorting projection).
        let cross_perimeter = self.cross_section.perimeter;
        let cap_perimeter = self.projected_cap.perimeter;
        let cross_section_projection = state::section_layer_projection(true, wireframe_projection);
        // Flat-chord vs S³-arc morph for every edge (0 = chord). Captured once so
        // the per-edge helper stays free of `&self`.
        // Edge geometry is derived from the projection, not a control:
        // Stereographic draws S3 great-circle arcs, every affine projection draws
        // flat R4 chords (see `state::default_edge_blend`).
        let space_blend = state::default_edge_blend(self.wireframe_projection);
        // Wireframe Hyperslice cull: when on, only edges whose body-local
        // w-interval intersects the slab around `w_slice` survive. Captured
        // once per frame. This is a third, independent slicing affordance
        // alongside the SDF raymarch's w-slice (raymarch/hyperslice4d.rs) and
        // the cyan section perimeter; it culls the *parent wireframe edges* to
        // those near the current 4D cut so the graph thins to "what the slice
        // is passing through" instead of the whole polytope. A demo-side
        // per-edge FILTER, deliberately NOT a `Projection` variant: the
        // projection returns a Vec3 that has already discarded w, so it cannot
        // honestly carry a keep/drop signal, and a sentinel-NaN projection
        // would mis-clip boundary-crossing edges at vertex granularity.
        //
        // The `Hyperslice` projection mode IS this cull paired with drop-w: it
        // resolves to `Projection::Identity` (the slicing is the cull, not a
        // projection), so selecting it activates the filter even when the
        // independent `wireframe_hyperslice` toggle is off. The standalone toggle
        // still composes the cull with any other projection mode.
        let hyperslice_on = self.hyperslice_cull_active();
        let hyperslice_thickness = self.wireframe_hyperslice_thickness;
        let hyperslice_w_slice = self.w_slice;
        // Reused great-circle sampling buffer, taken from the demo so its
        // capacity persists across frames; `push_blended_edge` clears it per
        // edge, and it is put back after the loop.
        let mut slerp_scratch = std::mem::take(&mut self.slerp_scratch);

        for (slot, entry) in render_row.iter().enumerate() {
            let Some(polytope) = entry.shape.polytope4() else {
                continue;
            };
            // Per-shape stereographic clip radius (the perimeter closure and the
            // per-edge `push_blended_edge` below both consume it): finite + capped
            // for the 16-cell, `f32::INFINITY` (no clip) for every other shape.
            let view_radius = stereographic_view_radius(polytope, cam_dist);
            let topo = polytope.topology();
            let body_pos = body_position(slot, n);
            // Body's R³ position. The body sits at body_pos.w = 0 in 4D, so the perspective
            // projection at this body's location collapses to a pure R³ translation; doing
            // the projection in body-local 4D and translating in R³ AFTER is the only way
            // to keep the body's apparent x-position stable when Perspective4D scales the
            // (x, y, z) channel by `focal / (focal - w)`.
            let body_pos_r3 = Vec3::new(body_pos[0], body_pos[1], body_pos[2]);
            // Body-local 4D vertices: rotor-rotated, scaled, NO world translate yet. The
            // translate happens after the chosen `wireframe_projection` maps each vertex
            // from R⁴ to R³.
            let body_size = self.effective_body_size();
            let local_vertices: Vec<Vec4> = topo
                .vertices
                .iter()
                .map(|v| body_size * self.rot_state.apply(*v))
                .collect();

            // Cross-section perimeter outlines, one per enabled layer overlaid in
            // this one viewport. `polytope_section_overlay_with_vertices` returns
            // the slice's body-local drop-w R³ perimeter (computed once, shared by
            // both layers): the honest cross-section maps it through drop-w (NEVER
            // the active projection, so its outline matches the SDF slice), and the
            // projected cap maps it through the active `wireframe_projection` so its
            // outline sits on the projected wireframe. Each layer maps endpoints to
            // world R³ and, under a clipped projection (Stereographic), drops a
            // whole perimeter segment when either endpoint's body-local projected
            // magnitude exceeds the clip radius (per-segment because a perimeter
            // segment is a single cap edge, not a polyline).
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

            // Per-cell "crossing strength" in [0, 1] - shared with render_points
            // via `compute_cell_strengths` so both passes agree on what "active"
            // means for a given w_slice + rotated polytope.
            let cell_strengths = compute_cell_strengths(topo.cells, &local_vertices, self.w_slice);

            // Per-edge brightness: max strength across cells the edge belongs to. An edge
            // belongs to a cell when both endpoints sit in that cell's vertex list. This
            // lets a single edge "light up" as soon as ANY containing cell is being
            // crossed deep; doesn't require the slice to specifically hit the cell whose
            // boundary the edge sits on.
            let edge_strength = |i: u32, j: u32| -> f32 {
                let mut best = 0.0_f32;
                for (cell, strength) in topo.cells.iter().zip(cell_strengths.iter()) {
                    if cell.contains(&i) && cell.contains(&j) && *strength > best {
                        best = *strength;
                    }
                }
                best
            };

            // Active-mode binary classification: an edge is "active" if at least one of
            // its containing cells has the slice strictly between its w-range endpoints
            // (i.e., the slice is currently producing a cap from that cell). Uses the
            // same `edges-in-cell` membership rule as `edge_strength`; threshold is
            // `cell_strength > 0.0`, which corresponds exactly to `w_min < w_slice <
            // w_max` after the `(1 - dist / half_extent)` clamp.
            let edge_is_active = |i: u32, j: u32| -> bool {
                topo.cells
                    .iter()
                    .zip(cell_strengths.iter())
                    .any(|(cell, &s)| s > 0.0 && cell.contains(&i) && cell.contains(&j))
            };

            // Hyperslice cull, evaluated per edge before any color / projection
            // / tessellation work. The kept-edge decision is CELL-level, matching
            // `edge_is_active` and the cross-section: an edge survives iff some
            // cell containing BOTH endpoints has its w-range overlapping the slab.
            // The edge-level test (its own endpoints straddling the slab) would
            // cull a far-side edge of an active cell even though that edge is
            // colored active-green, since the coloring reads the whole cell's
            // w-range, not the edge's. Folding `cell_w_range` here keeps the two
            // in lockstep.
            //
            // This is the slab-with-thickness band, a SUPERSET of `edge_is_active`
            // (which is the zero-width `w_min < w_slice < w_max` plane): the cull
            // keeps every active-green edge plus the thickness margin the user
            // dials in, so active edges are never culled while the band still
            // thins the graph. Same `cells.iter()` membership cost as the two
            // coloring closures above, so the cull introduces no new asymptotic
            // work and no per-frame allocation.
            let edge_in_slab_cell = |i: u32, j: u32| -> bool {
                topo.cells.iter().any(|cell| {
                    if !(cell.contains(&i) && cell.contains(&j)) {
                        return false;
                    }
                    let (w_min, w_max) = cell_w_range(cell, &local_vertices);
                    slab_overlaps(w_min, w_max, hyperslice_w_slice, hyperslice_thickness)
                })
            };

            // Parent wireframe: every polytope edge as a world-R³ line. Base RGB is
            // picked by `wireframe_color_mode`:
            // - `VertexGradient`: per-vertex position-derived RGB from the canonical
            //   vertex set so each vertex gets a distinct hue from its 4D coordinates
            //   and the polytope's symmetry shows as smooth gradients (same scheme as
            //   `Polytope4::lines_colored_by_position`).
            // - `UniqueEdge`: each edge gets a distinct palette color via greedy
            //   graph-coloring on the line graph (see `unique_edge_palette`).
            // - `WDepth`: per-endpoint cool-blue-to-warm-orange by SIGNED w
            //   in body-local frame, normalized against the polytope's canonical
            //   max |w| (fixed-band, not per-frame), so a vertex paints the same
            //   color at the same w regardless of rotor orientation.
            // - `Active`: binary green/gray by cell-activity (see `edge_is_active`).
            // Alpha is then modulated per-edge by the `nearest-active` strength (when
            // that toggle is on) or held uniform.
            // UniqueEdge palette is topology-only (greedy graph-coloring on the line
            // graph). Memoize by `Polytope4` variant so the 600-cell's ~520k pair-checks
            // run once per launch instead of once per frame. Cache lives on Demo; an
            // empty cache simply triggers first-use population for the visited variants.
            let edge_palette: &[[f32; 4]] = if matches!(color_mode, WireframeColorMode::UniqueEdge)
            {
                self.unique_edge_palette_cache
                    .entry(polytope)
                    .or_insert_with(|| unique_edge_palette(topo.edges))
            } else {
                &[]
            };
            // For `WDepth` we normalize against this polytope's CANONICAL max
            // |w| (NOT the per-frame rotated max). The rotor preserves
            // magnitudes, so the maximum |w| any rotated vertex can reach is
            // bounded by the canonical max times `body_size`. Holding the
            // gradient endpoints fixed across frames is what makes the color
            // cue temporally stable: a vertex migrating from -w to +w as
            // the rotor swings paints a continuous cool-to-warm shift rather
            // than a per-frame normalized re-saturation. Mirrors the
            // `LineRasterStaticR4` shader's hardcoded `[-0.5, +0.5]` band
            // (which is correct for the tesseract); here it's per-polytope
            // because the 16-cell, 5-cell, etc. don't share that band.
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
                // Cell-level Hyperslice cull (see `edge_in_slab_cell`),
                // evaluated before any color / projection / tessellation work so
                // a culled edge costs only the membership fold. The local-`w`
                // frame the slab tests against is the same one the SDF marcher
                // and the section algorithm slice (the body sits at world
                // `w = 0`). The `&&` short-circuits the fold entirely when the
                // affordance is off.
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
                // Emit the edge in body-local 4D, projected to world R³. `blend`
                // is projection-derived (Stereographic -> 1 = S³ arc, affine -> 0
                // = one chord per edge). Projection is shared with the flat path:
                // Shadow is identity-on-(x, y, z), Perspective4D scales each
                // component by focal/(focal-w).
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

        // Put the great-circle sampling buffer back so its capacity is reused
        // next frame instead of reallocating.
        self.slerp_scratch = slerp_scratch;

        // Upload (each call is a no-op when its mesh is empty).
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

        // Camera. Build the same view+projection matrix the SDF raymarcher uses
        // implicitly via its ray basis, so the rasterized overlay aligns pixel-for-pixel
        // with the raymarched scene.
        let view_dir = self.camera.view();
        let aspect = cfg.width as f32 / cfg.height as f32;
        let view_mat = Mat4::look_to_rh(view_dir.position, view_dir.forward, view_dir.up);
        let proj_mat = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.1, 100.0);
        let view_proj = proj_mat * view_mat;
        let vp_size = Vec2::new(cfg.width as f32, cfg.height as f32);
        self.section_edges.set_camera(&rd.queue, view_proj, vp_size);
        self.parent_wireframe
            .set_camera(&rd.queue, view_proj, vp_size);

        // Section perimeter edges then dim parent wireframe. Both depth-test (no write)
        // against the shared section-faces depth attachment so lines behind a cap get
        // correctly occluded across polytopes in a row. The shared buffer is ensured +
        // cleared earlier in the frame; here we just borrow the view. In SDF mode no
        // pass writes depth, so the cleared `1.0` buffer makes every fragment pass the
        // test, preserving the historical visual.
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
