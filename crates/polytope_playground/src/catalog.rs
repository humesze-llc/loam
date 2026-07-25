//! 4D shape catalog: the single source of truth for shape names,
//! colors, and tooltips. Holds per-polytope metadata, the default
//! startup row, the categorized catalog, the `+`-menu helper, and the
//! `shapes` argument parser.

use anyhow::{anyhow, Result};
use loam_app::args::Args;
use loam_app::egui;
use loam_physics::polytope::Polytope4;
use loam_render::raymarch::RaymarchShape;

/// One polytope's metadata. `body_color` drives `BodyUniform.color`
/// on the GPU, not the (uniformly grey) card color. `long_name` uses
/// the `pentachoron`/`tesseract`/`hexadecachoron` family, not the
/// dimension-generalized `*-plex` aliases.
#[derive(Copy, Clone, PartialEq, Debug)]
pub(crate) struct ShapeEntry {
    pub(crate) shape: RaymarchShape,
    pub(crate) body_color: [f32; 3],
    pub(crate) label: &'static str,
    pub(crate) long_name: &'static str,
}

/// Default row when no `shapes` argument is given. 24-cell first (most
/// 4D-distinct cross-section), then visually contrasting shapes.
pub(crate) const DEFAULT_ROW: &[ShapeEntry] = &[
    ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Cell24),
        body_color: [0.95, 0.45, 0.85],
        label: "24-cell",
        long_name: "icositetrachoron",
    },
    ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Pentatope),
        body_color: [0.95, 0.55, 0.30],
        label: "5-cell",
        long_name: "pentachoron",
    },
    ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Cell16),
        body_color: [0.55, 0.95, 0.40],
        label: "16-cell",
        long_name: "hexadecachoron",
    },
    ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Tesseract),
        body_color: [0.30, 0.55, 0.95],
        label: "8-cell",
        long_name: "tesseract",
    },
];

/// Every shipped 4D shape: the six convex regular polychora plus four
/// smooth solids (3-sphere, duocylinder, Clifford torus, spherinder).
/// `body_color` channels pass straight to the WGSL kernel.
pub(crate) const SHAPE_CATALOG: &[ShapeEntry] = &[
    ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Pentatope),
        body_color: [0.95, 0.55, 0.30],
        label: "5-cell",
        long_name: "pentachoron",
    },
    ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Tesseract),
        body_color: [0.30, 0.55, 0.95],
        label: "8-cell",
        long_name: "tesseract",
    },
    ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Cell16),
        body_color: [0.55, 0.95, 0.40],
        label: "16-cell",
        long_name: "hexadecachoron",
    },
    ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Cell24),
        body_color: [0.95, 0.45, 0.85],
        label: "24-cell",
        long_name: "icositetrachoron",
    },
    ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Cell120),
        body_color: [0.40, 0.85, 0.85],
        label: "120-cell",
        long_name: "hecatonicosachoron",
    },
    ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Cell600),
        body_color: [0.95, 0.85, 0.40],
        label: "600-cell",
        long_name: "hexacosichoron",
    },
    ShapeEntry {
        shape: RaymarchShape::ThreeSphere,
        body_color: [0.85, 0.40, 0.40],
        label: "3-sphere",
        long_name: "hypersphere (4-ball)",
    },
    ShapeEntry {
        shape: RaymarchShape::Duocylinder,
        body_color: [0.60, 0.45, 0.90],
        label: "duocyl",
        long_name: "duocylinder (D² × D²)",
    },
    ShapeEntry {
        shape: RaymarchShape::CliffordTorus,
        body_color: [0.70, 0.85, 0.35],
        label: "clifford",
        long_name: "Clifford torus tube",
    },
    ShapeEntry {
        shape: RaymarchShape::Spherinder,
        body_color: [0.85, 0.55, 0.75],
        label: "spherinder",
        long_name: "spherinder (B³ × interval)",
    },
];

/// Category-grouped shape menu: top level lists [`SHAPE_CATEGORIES`],
/// each a submenu of shapes with a `long_name` hover tooltip.
/// `on_select` fires on click; the helper closes the menu.
pub(crate) fn render_shape_catalog_menu(ui: &mut egui::Ui, mut on_select: impl FnMut(ShapeEntry)) {
    for cat in SHAPE_CATEGORIES {
        ui.menu_button(cat.name, |ui| {
            for entry in &SHAPE_CATALOG[cat.start..cat.end] {
                if ui
                    .button(entry.label)
                    .on_hover_text(entry.long_name)
                    .clicked()
                {
                    on_select(*entry);
                    ui.close_kind(egui::UiKind::Menu);
                }
            }
        });
    }
}

/// Half-open index ranges into [`SHAPE_CATALOG`] that group menu
/// entries under a header. Ranges (not nested slices) keep flat
/// `SHAPE_CATALOG[i]` lookups working.
struct ShapeCategory {
    name: &'static str,
    start: usize,
    end: usize,
}

const SHAPE_CATEGORIES: &[ShapeCategory] = &[
    ShapeCategory {
        name: "Regular polychora",
        start: 0,
        end: 6,
    },
    ShapeCategory {
        name: "Smooth solids",
        start: 6,
        end: 10,
    },
];

/// Resolve a shape name. Both `n-cell` math names and Platonic-slice
/// aliases (`tetrahedron`, `cube`, ...) map to the same shape.
pub(crate) fn parse_shape_name(name: &str) -> Result<ShapeEntry> {
    let n = name.to_lowercase();
    let needle: &str = n.as_str();
    for entry in SHAPE_CATALOG {
        if needle == entry.label.to_lowercase() || needle == entry.long_name.to_lowercase() {
            return Ok(*entry);
        }
    }
    // Common aliases not in the catalog's `label` / `long_name`.
    Ok(match needle {
        "5cell" | "pentatope" | "tetrahedron" => SHAPE_CATALOG[0],
        "8cell" | "hypercube" | "cube" => SHAPE_CATALOG[1],
        "16cell" | "octahedron" => SHAPE_CATALOG[2],
        "24cell" | "cuboctahedron" => SHAPE_CATALOG[3],
        "120cell" | "dodecahedron" => SHAPE_CATALOG[4],
        "600cell" | "icosahedron" => SHAPE_CATALOG[5],
        "hypersphere" | "3sphere" | "s3" | "4-ball" => SHAPE_CATALOG[6],
        "duocylinder" => SHAPE_CATALOG[7],
        "clifford" | "clifford-torus" | "torus" => SHAPE_CATALOG[8],
        "spherinder" => SHAPE_CATALOG[9],
        _ => {
            return Err(anyhow!(
                "unknown shape name {name:?}; valid: 5-cell, 8-cell, \
                 16-cell, 24-cell, 120-cell, 600-cell, 3-sphere, \
                 duocyl, clifford, spherinder (plus Platonic aliases: \
                 tetrahedron, cube, octahedron, cuboctahedron, \
                 dodecahedron, icosahedron)"
            ));
        }
    })
}

/// Reject the space-separated `--shapes a,b` form, given the process
/// arguments minus `argv[0]`.
///
/// [`Args`] keeps only `--key=value` tokens, so a bare `--shapes` and
/// its value disappear before [`parse_row`] can see them and the demo
/// starts on [`DEFAULT_ROW`] as if no row had been requested.
fn reject_space_separated_shapes(argv: impl IntoIterator<Item = String>) -> Result<()> {
    let mut argv = argv.into_iter();
    while let Some(arg) = argv.next() {
        if arg != "--shapes" {
            continue;
        }
        let corrected = match argv.next() {
            Some(value) => format!("--shapes={value}"),
            None => "--shapes=24-cell,8-cell".to_string(),
        };
        return Err(anyhow!(
            "`--shapes` needs its value attached with `=`; write \
             `{corrected}` instead (comma-separated for several shapes)"
        ));
    }
    Ok(())
}

/// Parse the comma-separated `shapes` key (`--shapes=a,b` natively,
/// `?shapes=a,b` in the browser). Returns [`DEFAULT_ROW`] when the key
/// is absent, and an error for the space-separated `--shapes a,b` form,
/// which reaches `args` as nothing at all.
pub(crate) fn parse_row(args: &Args) -> Result<Vec<ShapeEntry>> {
    reject_space_separated_shapes(std::env::args().skip(1))?;
    let Some(raw) = args.get("shapes") else {
        return Ok(DEFAULT_ROW.to_vec());
    };
    let names = args.get_many("shapes");
    if names.is_empty() {
        return Err(anyhow!("shapes={raw:?} listed no shape names"));
    }
    names.into_iter().map(parse_shape_name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_comes_from_the_args_value_not_the_process_environment() {
        // The wasm defect this pins: reading the environment yielded
        // DEFAULT_ROW in the browser no matter what the URL said.
        assert_eq!(parse_row(&Args::default()).unwrap(), DEFAULT_ROW);
        assert_eq!(
            parse_row(&Args::from_pairs([("seed", "42")])).unwrap(),
            DEFAULT_ROW
        );

        let row = parse_row(&Args::from_pairs([("shapes", "120-cell,tesseract")])).unwrap();
        assert_eq!(
            row.iter().map(|e| e.label).collect::<Vec<_>>(),
            ["120-cell", "8-cell"]
        );
    }

    fn argv(tokens: &[&str]) -> Vec<String> {
        tokens.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_bare_shapes_flag_is_diagnosed_and_names_the_equals_form() {
        let err = reject_space_separated_shapes(argv(&["--shapes", "nonesuch"]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("--shapes=nonesuch"), "{err}");

        let trailing = reject_space_separated_shapes(argv(&["--seed=42", "--shapes"]))
            .unwrap_err()
            .to_string();
        assert!(trailing.contains("--shapes="), "{trailing}");
    }

    #[test]
    fn the_equals_form_and_unrelated_arguments_pass_the_syntax_guard() {
        assert!(reject_space_separated_shapes(argv(&["--shapes=5-cell,8-cell"])).is_ok());
        assert!(
            reject_space_separated_shapes(argv(&["--seed=42", "shapes", "--x=--shapes"])).is_ok()
        );
        assert!(reject_space_separated_shapes(argv(&[])).is_ok());
    }

    #[test]
    fn a_present_but_nameless_or_unknown_shapes_value_is_an_error() {
        assert!(parse_row(&Args::from_pairs([("shapes", "")])).is_err());
        assert!(parse_row(&Args::from_pairs([("shapes", ",,")])).is_err());
        assert!(parse_row(&Args::from_pairs([("shapes", "5-cell,nonesuch")])).is_err());
    }
}
