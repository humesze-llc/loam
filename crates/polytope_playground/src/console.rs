//! Console command registration (the `wireframe`, `pole`, `section`,
//! `surface`, `camera`, `floor` commands).

use crate::*;

impl RotatePolytopesApp {
    pub(crate) fn build_console() -> Console<Demo> {
        let mut c = Console::<Demo>::new();
        c.register(loam_egui::cmd(
            "reset",
            "full reset (R)",
            |_args, demo: &mut Demo, _out| {
                demo.reset();
                Ok(())
            },
        ));
        c.register(loam_egui::cmd(
            "spin",
            "toggle continuous rotation (Space / T)",
            |_args, demo: &mut Demo, _out| {
                demo.rotate = !demo.rotate;
                Ok(())
            },
        ));
        c.register(loam_egui::cmd(
            "controls",
            "toggle the bottom controls overlay (H)",
            |_args, demo: &mut Demo, _out| {
                demo.show_controls = !demo.show_controls;
                Ok(())
            },
        ));
        c.register(loam_egui::cmd(
            "formula",
            "toggle the top-right formula popup",
            |_args, demo: &mut Demo, _out| {
                demo.show_formula = !demo.show_formula;
                Ok(())
            },
        ));
        // Cross-section + parent-wireframe overlay. Bare `wireframe` flips
        // on/off; subcommands each carry their own value choices for context-
        // aware tab-completion via [`SubcommandSet`].
        c.register(
            loam_egui::subcommands::<Demo>("wireframe", "wireframe + cross-section overlay")
                .on_bare(|d| {
                    d.wireframe_enabled = !d.wireframe_enabled;
                    Ok(())
                })
                .toggle(
                    "nearest-active",
                    "per-edge alpha gradient by cell-crossing strength (bare flips)",
                    |d, v| {
                        d.wireframe_nearest_active = v.unwrap_or(!d.wireframe_nearest_active);
                        Ok(())
                    },
                )
                .custom(
                    "width",
                    "parent-wireframe edge thickness in pixels (default 1.8)",
                    &[&[]],
                    &[],
                    |d, args, out| {
                        match args.first().copied() {
                            None => {
                                out.line(format!(
                                    "wireframe width: {:.2} px",
                                    d.wireframe_width_px
                                ));
                            }
                            Some(s) => match s.parse::<f32>() {
                                Ok(w) if w > 0.0 && w <= 16.0 => {
                                    d.wireframe_width_px = w;
                                    out.line(format!(
                                        "wireframe width: set to {w:.2} px"
                                    ));
                                }
                                _ => {
                                    out.line(format!(
                                        "wireframe width: invalid `{s}` (need a float in (0, 16])"
                                    ));
                                }
                            },
                        }
                        Ok(())
                    },
                )
                .custom(
                    "alpha",
                    "uniform edge alpha when nearest-active is off (default 1.0)",
                    &[&[]],
                    &[],
                    |d, args, out| {
                        match args.first().copied() {
                            None => {
                                out.line(format!(
                                    "wireframe alpha: {:.3} ({})",
                                    d.wireframe_alpha,
                                    if d.wireframe_nearest_active {
                                        "overridden by nearest-active gradient; toggle off to apply"
                                    } else {
                                        "active"
                                    }
                                ));
                            }
                            Some(s) => match s.parse::<f32>() {
                                Ok(a) if a > 0.0 && a <= 1.0 => {
                                    d.wireframe_alpha = a;
                                    out.line(format!(
                                        "wireframe alpha: set to {a:.3}"
                                    ));
                                }
                                _ => {
                                    out.line(format!(
                                        "wireframe alpha: invalid `{s}` (need a float in (0, 1])"
                                    ));
                                }
                            },
                        }
                        Ok(())
                    },
                )
                .choice(
                    "color",
                    "parent-edge color mode (bare cycles): vertex-gradient|unique-edge|w-depth|active",
                    &["vertex-gradient", "unique-edge", "w-depth", "active"],
                    |d, name| {
                        d.wireframe_color_mode = match name {
                            Some(n) => WireframeColorMode::from_token(n).ok_or_else(|| {
                                anyhow!(
                                    "unknown color mode `{n}` (try vertex-gradient|unique-edge|w-depth|active)"
                                )
                            })?,
                            None => {
                                // Bare cycles through the canonical order.
                                let all = WireframeColorMode::ALL;
                                let i = all
                                    .iter()
                                    .position(|m| *m == d.wireframe_color_mode)
                                    .unwrap_or(0);
                                all[(i + 1) % all.len()]
                            }
                        };
                        Ok(())
                    },
                )
                .custom(
                    "perspective",
                    "wireframe 4D->R³ projection (bare cycles): shadow | w-pinhole | stereographic | hyperslice",
                    &[&["shadow", "w-pinhole", "stereographic", "hyperslice"]],
                    &[],
                    |d, args, out| {
                        // Schlegel is omitted here (it wants its own demo): both
                        // `from_token` and `ALL` exclude it.
                        let next = match args.first().copied() {
                            // Bare: cycle through ALL in variant order.
                            None => {
                                let all = WireframeProjection::ALL;
                                let i = all
                                    .iter()
                                    .position(|p| p.same_variant(d.wireframe_projection))
                                    .unwrap_or(0);
                                all[(i + 1) % all.len()]
                            }
                            Some(token) => WireframeProjection::from_token(token).ok_or_else(|| {
                                anyhow!(
                                    "unknown projection `{token}` (try shadow|w-pinhole|stereographic|hyperslice)"
                                )
                            })?,
                        };
                        d.wireframe_projection = next;
                        state::apply_projection_selection_defaults(
                            d.wireframe_projection,
                            &mut d.wireframe_enabled,
                        );
                        // No-op for these modes; kept for a future Schlegel re-wire.
                        d.resolve_schlegel_cache();
                        out.line(format!(
                            "wireframe perspective: {}",
                            d.wireframe_projection.label().to_lowercase()
                        ));
                        Ok(())
                    },
                )
                .custom(
                    "pole",
                    "stereographic projection pole (bare reports; sub: reset | +w | <x y z w>)",
                    &[&["reset", "+w"]],
                    &[],
                    |d, args, out| {
                        match args.first().copied() {
                            None => {
                                let p = d.stereographic_pole;
                                out.line(format!(
                                    "stereographic pole: ({:.3}, {:.3}, {:.3}, {:.3})",
                                    p.x, p.y, p.z, p.w
                                ));
                            }
                            // Cell-center default (see STEREOGRAPHIC_DEFAULT_POLE).
                            Some("reset") | Some("default") => {
                                d.stereographic_pole = state::STEREOGRAPHIC_DEFAULT_POLE;
                                out.line("stereographic pole: reset to the cell-center default");
                            }
                            // The textbook `(x, y, z) / (1 - w)` pole, as a named
                            // shortcut.
                            Some("+w") => {
                                d.stereographic_pole = Vec4::W;
                                out.line("stereographic pole: set to +w (textbook map)");
                            }
                            // Explicit pole: four floats normalized onto S³ (only
                            // the direction matters); reject a near-zero vector.
                            Some(_) => {
                                let coords: Result<Vec<f32>> = args
                                    .iter()
                                    .map(|t| {
                                        t.parse::<f32>()
                                            .map_err(|e| anyhow!("invalid pole component `{t}`: {e}"))
                                    })
                                    .collect();
                                let coords = coords?;
                                if coords.len() != 4 {
                                    return Err(anyhow!(
                                        "pole needs 4 components `<x y z w>`, got {}",
                                        coords.len()
                                    ));
                                }
                                let raw = Vec4::new(coords[0], coords[1], coords[2], coords[3]);
                                if raw.length() < MIN_EDGE_RADIUS {
                                    return Err(anyhow!(
                                        "pole vector is too close to zero to have a direction"
                                    ));
                                }
                                let pole = raw.normalize();
                                d.stereographic_pole = pole;
                                out.line(format!(
                                    "stereographic pole: set to ({:.3}, {:.3}, {:.3}, {:.3})",
                                    pole.x, pole.y, pole.z, pole.w
                                ));
                            }
                        }
                        Ok(())
                    },
                )
                .custom(
                    "hyperslice",
                    "cull parent edges to a w-slab around the slice (bare flips; sub: on|off|thickness <N>)",
                    &[&["on", "off", "thickness"]],
                    &[],
                    |d, args, out| {
                        match args.first().copied() {
                            None => {
                                d.wireframe_hyperslice = !d.wireframe_hyperslice;
                                out.line(format!(
                                    "wireframe hyperslice: {} (slab full-width {:.3})",
                                    if d.wireframe_hyperslice { "on" } else { "off" },
                                    d.wireframe_hyperslice_thickness
                                ));
                            }
                            Some("on") => {
                                d.wireframe_hyperslice = true;
                                out.line(format!(
                                    "wireframe hyperslice: on (slab full-width {:.3})",
                                    d.wireframe_hyperslice_thickness
                                ));
                            }
                            Some("off") => {
                                d.wireframe_hyperslice = false;
                                out.line("wireframe hyperslice: off (full edge graph)");
                            }
                            Some("thickness") => match args.get(1).copied() {
                                None => out.line(format!(
                                    "wireframe hyperslice thickness: {:.3}",
                                    d.wireframe_hyperslice_thickness
                                )),
                                Some(token) => {
                                    let t: f32 = token.parse().map_err(|e| {
                                        anyhow!("invalid thickness `{token}`: {e}")
                                    })?;
                                    // Floor is the predicate's razor band; the
                                    // `2 * W_RANGE` cap covers every reachable w, so
                                    // the filter no-ops there (same as "off").
                                    let max = 2.0 * consts::W_RANGE;
                                    if !(HYPERSLICE_MIN_THICKNESS..=max).contains(&t) {
                                        return Err(anyhow!(
                                            "hyperslice thickness {t} out of range; expected {HYPERSLICE_MIN_THICKNESS}..={max}"
                                        ));
                                    }
                                    d.wireframe_hyperslice_thickness = t;
                                    out.line(format!(
                                        "wireframe hyperslice thickness: set to {t:.3}"
                                    ));
                                }
                            },
                            Some(other) => {
                                return Err(anyhow!(
                                    "unknown hyperslice subcommand `{other}` (try on|off|thickness)"
                                ));
                            }
                        }
                        Ok(())
                    },
                )
                .custom(
                    "points",
                    "vertex + cell-center sprite overlay (bare flips; sub: vertices|cell-centers|size <N>)",
                    &[&["vertices", "cell-centers", "size"]],
                    &[],
                    |d, args, out| {
                        match args.first().copied() {
                            None => {
                                d.points_enabled = !d.points_enabled;
                                out.line(format!(
                                    "points: {}",
                                    if d.points_enabled { "on" } else { "off" }
                                ));
                            }
                            Some("vertices") => {
                                d.points_show_vertices = !d.points_show_vertices;
                                out.line(format!(
                                    "points vertices: {}",
                                    if d.points_show_vertices { "on" } else { "off" }
                                ));
                            }
                            Some("cell-centers") => {
                                d.points_show_cell_centers = !d.points_show_cell_centers;
                                out.line(format!(
                                    "points cell-centers: {}",
                                    if d.points_show_cell_centers { "on" } else { "off" }
                                ));
                            }
                            Some("size") => match args.get(1) {
                                None => out.line(format!(
                                    "points size: {:.1} px",
                                    d.points_size_px
                                )),
                                Some(token) => {
                                    let px: f32 = token.parse().map_err(|e| {
                                        anyhow!("invalid pixel value `{token}`: {e}")
                                    })?;
                                    if !(1.0..=64.0).contains(&px) {
                                        return Err(anyhow!(
                                            "points size {px} out of range; expected 1..=64"
                                        ));
                                    }
                                    d.points_size_px = px;
                                    out.line(format!("points size: set to {px:.1} px"));
                                }
                            },
                            Some(other) => {
                                return Err(anyhow!(
                                    "unknown points subcommand `{other}` (try vertices|cell-centers|size)"
                                ));
                            }
                        }
                        Ok(())
                    },
                ),
        );

        // Polychoral surface renderer: raster (default) / SDF / off. Bare
        // `surface` is shorthand for "off". `surface scale <N>` rescales the row
        // by multiplying `BODY_SIZE` (see [`Demo::effective_body_size`]).
        c.register(
            loam_egui::cmd(
                "surface",
                "polychoral surface mode: raster | sdf | off (bare = off); `scale <N>` to resize (per-layer cap alpha lives under `section`)",
                |args, demo: &mut Demo, out| {
                    if matches!(args.first().copied(), Some("scale")) {
                        match args.get(1).copied() {
                            None => {
                                out.line(format!(
                                    "surface scale: {:.3} (multiplies BODY_SIZE)",
                                    demo.surface_scale
                                ));
                            }
                            Some(token) => {
                                let parsed: f32 = token.parse().map_err(|e| {
                                    anyhow!("invalid scale `{token}`: {e}")
                                })?;
                                if !(0.05..=10.0).contains(&parsed) {
                                    return Err(anyhow!(
                                        "surface scale {parsed} out of range; expected 0.05..=10.0"
                                    ));
                                }
                                demo.surface_scale = parsed;
                                // Rebuild SDF body uniforms so the kernel sees the new
                                // radius immediately; raster paths read effective_body_size()
                                // every frame and don't need a rebuild.
                                demo.rebuild_bodies();
                                // Clamp the current w-slice into the new scaled range so
                                // a shrink doesn't leave the slider off the visible body.
                                let w_range = demo.effective_w_range();
                                demo.w_slice = demo.w_slice.clamp(-w_range, w_range);
                                out.line(format!(
                                    "surface scale: set to {parsed:.3}"
                                ));
                            }
                        }
                        return Ok(());
                    }
                    let next = match args.first().copied() {
                        Some(token) => SurfaceMode::from_token(token).ok_or_else(|| {
                            anyhow!("unknown arg `{token}` (try raster|sdf|off|scale; cap alpha lives under `section`)")
                        })?,
                        None => SurfaceMode::Off,
                    };
                    if next == SurfaceMode::Sdf && demo.sdf_blocked_by_heavy_polychora() {
                        return Err(anyhow!(
                            "surface sdf disabled while 120-cell or 600-cell is in the row \
                             (the SDF kernel crashes the browser tab on those); remove the \
                             heavy polychora first, or use `surface raster`"
                        ));
                    }
                    if next != demo.surface_mode {
                        demo.surface_mode = next;
                        // Re-emit the SDF body list: switching INTO Sdf mode makes the
                        // polychora live in the kernel, switching OUT marks them inert.
                        demo.rebuild_bodies();
                    }
                    Ok(())
                },
            )
            .with_args(&[&["raster", "sdf", "off", "scale"]])
            .with_long_help(
                "Selects how the six regular convex 4-polytopes (5-cell, tesseract, 16-cell,\n\
                 24-cell, 120-cell, 600-cell) are rendered, plus a runtime scale knob.\n\
                 \n\
                 subcommands:\n  \
                 raster      Rasterized cross-section cell-caps (the default). Face-normal\n                             Lambert lit, per-body solid color. Much faster for the\n                             120-cell + 600-cell and exact (no SDF approximation).\n  \
                 sdf         SDF raymarch. The historical pre-rasterizer path; smoother\n                             shading but the 120-cell and 600-cell carry a face-plane\n                             approximation BUG. Kept for visual comparison.\n  \
                 off         No surface rendered. Wireframe overlay + cross-section\n                             perimeter stay visible if enabled; the cap interiors are\n                             blank. Useful for inspecting the wireframe on its own.\n  \
                 scale <N>   Multiply the canonical body radius by N (default 1.0; range\n                             0.05..=10.0). Affects SDF kernel, raster cross-section caps,\n                             wireframe overlay, perimeter, and points sprites uniformly.\n\
                 \n\
                 Bare `surface` (no argument) is shorthand for `surface off`.\n\
                 \n\
                 The rasterized cross-section splits into two overlaid layers with\n\
                 independent perimeter + fill alpha: see the `section` command (the honest\n\
                 drop-w cross-section and the projection-following cap).\n\
                 \n\
                 Smooth-surface shapes (Clifford torus, duocylinder, spherinder, 3-sphere)\n\
                 ignore the mode and always render via the SDF; they have no rasterizer\n\
                 path. Surface scale still applies to their SDF body radius.",
            ),
        );

        // Section layers: the rasterized cross-section is two overlaid layers in
        // one viewport, each with its own perimeter outline + fill alpha.
        //   - `cross`: the honest drop-w slice (NEVER reprojected; the geometry the
        //     SDF raymarch shows). On by default so selecting Schlegel /
        //     stereographic never silently distorts the slice.
        //   - `cap`: the same slice reprojected through the active wireframe
        //     projection, so it can sit on a Schlegel / stereographic wireframe.
        //     Off by default.
        // Alpha `0` is the layer's off state; `(0, 1]` sets a visible fill (below 1
        // composites through the depth-write-disabled pipeline). Side-by-side /
        // multi-viewport comparison is deferred to the multi-viewport milestone.
        c.register(
            loam_egui::subcommands::<Demo>(
                "section",
                "rasterized cross-section layers: cross (honest drop-w) + cap (projection-following), each with perimeter + alpha",
            )
            .toggle(
                "cross-perimeter",
                "honest drop-w cross-section perimeter outline (bare flips)",
                |d, v| {
                    d.cross_section.perimeter = v.unwrap_or(!d.cross_section.perimeter);
                    Ok(())
                },
            )
            .toggle(
                "cap-perimeter",
                "projected-cap perimeter outline (bare flips)",
                |d, v| {
                    d.projected_cap.perimeter = v.unwrap_or(!d.projected_cap.perimeter);
                    Ok(())
                },
            )
            .custom(
                "cross-alpha",
                "honest cross-section fill alpha (0 = off; range (0, 1])",
                &[&[]],
                &[],
                |d, args, out| run_section_alpha("cross", &mut d.cross_section, args, out),
            )
            .custom(
                "cap-alpha",
                "projected-cap fill alpha (0 = off; range (0, 1])",
                &[&[]],
                &[],
                |d, args, out| run_section_alpha("cap", &mut d.projected_cap, args, out),
            ),
        );

        // Framework-provided capture: `capture png [pre|post|both] [dir]`,
        // `capture frames [pre|post|both] [dir]`, `capture stop`. Bound to F12 (one-shot)
        // and F9 (sequence start; use `capture stop` to end). Requests push to a global
        // queue; the runner drains and processes them at the render-loop's two taps.
        loam_app::capture::register_commands(&mut c);
        loam_app::capture::bind_default_hotkeys(&mut c);

        // Framework-provided log mirror: `log on|off|toggle` toggles whether
        // `tracing::*` events show up in the console scrollback.
        loam_app::log::register_command(&mut c);

        // Framework-provided frame-timing surface: `trace [summary|last|clear|cap N]`.
        // The runner is already recording per-section scopes on every redraw; this
        // command lets the user read them. Surfaces the slowest hot-path sections,
        // which is the data the pipeline-warming + wireframe-cache decisions read
        // from.
        loam_app::trace::register_command(&mut c);
        loam_app::fps::register_command(&mut c);
        loam_app::vsync::register_command(&mut c);
        loam_app::version::register_command(
            &mut c,
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION"),
            env!("BUILD_HASH"),
            env!("BUILD_DIRTY"),
        );

        // Demo-side camera mode toggle. Bare `camera` cycles between Orbit
        // (the default scroll-zoom/drag camera) and FreeRoam (WASD + mouse-
        // look). Explicit `camera orbit` resets the orbit controller to its
        // default distance + pitch so the camera returns to a known framing
        // around the world origin; `camera freecam` seeds the free-roam
        // position from the camera's current location.
        //
        // Freecam tuning subcommands (do NOT change mode):
        //   `camera freecam speed=<N>`        WASD/Space/Shift units/sec.
        //   `camera freecam speed`            Print the current speed.
        //   `camera freecam cursor_mode <m>`  `toggle` (default, FPS) or `hold` (MMO).
        //   `camera freecam cursor_mode`      Print the current mode.
        c.register(
            loam_egui::cmd::<Demo, _>(
                "camera",
                "camera mode: orbit | freecam; bare cycles. `camera freecam speed=<N>` / `cursor_mode hold|toggle` tune the preset",
                |args, demo, out| {
                    // Freecam-tuning forms have a second positional token.
                    // `speed=<N>` is parsed as one token (matches the user's
                    // `speed=<N>` spec); `speed` alone queries; `cursor_mode
                    // <m>` is two tokens.
                    if matches!(args.first().copied(), Some("freecam")) && args.len() >= 2 {
                        let second = args[1];
                        // `speed=<N>` and `speed <N>` and bare `speed`.
                        if let Some(value) = second.strip_prefix("speed=") {
                            let parsed: f32 = value
                                .parse()
                                .map_err(|e| anyhow!("invalid speed `{value}`: {e}"))?;
                            if !(0.1..=200.0).contains(&parsed) {
                                return Err(anyhow!(
                                    "camera freecam speed {parsed} out of range; expected 0.1..=200.0"
                                ));
                            }
                            demo.freecam.speed = parsed;
                            out.line(format!("camera freecam speed: set to {parsed:.2} u/sec"));
                            return Ok(());
                        }
                        if second == "speed" {
                            if let Some(value) = args.get(2) {
                                let parsed: f32 = value.parse().map_err(|e| {
                                    anyhow!("invalid speed `{value}`: {e}")
                                })?;
                                if !(0.1..=200.0).contains(&parsed) {
                                    return Err(anyhow!(
                                        "camera freecam speed {parsed} out of range; expected 0.1..=200.0"
                                    ));
                                }
                                demo.freecam.speed = parsed;
                                out.line(format!(
                                    "camera freecam speed: set to {parsed:.2} u/sec"
                                ));
                            } else {
                                out.line(format!(
                                    "camera freecam speed: {:.2} u/sec",
                                    demo.freecam.speed
                                ));
                            }
                            return Ok(());
                        }
                        if second == "cursor_mode" {
                            match args.get(2).copied() {
                                None => {
                                    out.line(format!(
                                        "camera freecam cursor_mode: {}",
                                        demo.freecam.cursor_mode().token()
                                    ));
                                }
                                Some(token) => {
                                    let mode = CursorMode::from_token(token).ok_or_else(|| {
                                        anyhow!(
                                            "unknown cursor_mode `{token}` (try hold|toggle)"
                                        )
                                    })?;
                                    demo.freecam.set_cursor_mode(mode);
                                    out.line(format!(
                                        "camera freecam cursor_mode: set to {}",
                                        mode.token()
                                    ));
                                }
                            }
                            return Ok(());
                        }
                        // Unknown second token under `camera freecam`: fall
                        // through to the mode-switch path which will yell
                        // about it.
                    }

                    let next = match args.first().copied() {
                        None => match demo.camera_mode {
                            CameraMode::Orbit => CameraMode::FreeRoam,
                            CameraMode::FreeRoam => CameraMode::Orbit,
                        },
                        Some("orbit") => CameraMode::Orbit,
                        Some("freecam") => CameraMode::FreeRoam,
                        Some(other) => {
                            out.line(format!(
                                "camera: unknown mode `{other}` (try orbit|freecam)"
                            ));
                            return Ok(());
                        }
                    };
                    demo.camera_mode = next;
                    match next {
                        CameraMode::Orbit => {
                            // Reset orbit so the camera returns to a known
                            // framing around (0, 0, 0) regardless of where
                            // freecam left it. Freecam's `set_active(false)`
                            // releases the cursor grab.
                            demo.orbit = OrbitController::default();
                            demo.orbit.set_orbit(8.0, -0.25);
                            demo.freecam.set_active(false, demo.camera.position);
                            out.line("camera: orbit (reset to world origin)");
                        }
                        CameraMode::FreeRoam => {
                            // Preset grabs cursor + seeds position from the
                            // camera's current pose so the toggle is
                            // continuous, not a teleport.
                            demo.freecam.set_active(true, demo.camera.position);
                            out.line(
                                "camera: freecam (WASD + Space/Shift; mouse-look; Alt to free cursor)",
                            );
                        }
                    }
                    Ok(())
                },
            )
            .with_args(&[
                &["orbit", "freecam"],
                &["speed=", "cursor_mode"],
                &["hold", "toggle"],
            ]),
        );

        // Floor toggle for the y=0 hyperplane ground. On by default. The
        // SDF kernel reads `u.params[0]` (set in `Demo::update`); when 0.0
        // the wrapper around `loam_scene_sdf` (injected into the shader at
        // setup time) short-circuits to a huge distance, so the marcher
        // never converges on the floor and the checkerboard never paints.
        // Bare `floor` flips the flag; `floor on|off` is the explicit form.
        c.register(
            loam_egui::cmd::<Demo, _>(
                "floor",
                "toggle the y=0 hyperplane ground (on | off; bare flips)",
                |args, demo, out| {
                    let next = match args.first().copied() {
                        None => !demo.floor_enabled,
                        Some("on") => true,
                        Some("off") => false,
                        Some(other) => {
                            return Err(anyhow!("floor: unknown arg `{other}` (try on|off)"));
                        }
                    };
                    demo.floor_enabled = next;
                    out.line(format!("floor: {}", if next { "on" } else { "off" }));
                    Ok(())
                },
            )
            .with_args(&[&["on", "off"]]),
        );

        c
    }
}
