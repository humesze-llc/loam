//! 4D shape catalog: per-polytope metadata, the default startup row,
//! the full catalog with categories, the `+`-button menu helper, and
//! the CLI `--shapes` parser.
//!
//! The catalog is the single source of truth for shape names, colors,
//! and tooltips used throughout the demo (`+` shape menu, filmstrip
//! subject combo, CLI parser, card labels). Keeping it in one file
//! means adding a new shape touches one place.

use anyhow::{anyhow, Result};
use rye_app::egui;
use rye_render::raymarch::{
    SHAPE_120CELL, SHAPE_16CELL, SHAPE_24CELL, SHAPE_3SPHERE, SHAPE_600CELL, SHAPE_CLIFFORD_TORUS,
    SHAPE_DUOCYLINDER, SHAPE_PENTATOPE, SHAPE_SPHERINDER, SHAPE_TESSERACT,
};

/// One polytope's metadata: shape index in the kernel's table,
/// per-body fragment color (driven into `BodyUniform.color` on the
/// GPU side, NOT the panel's card color; those are uniformly grey
/// in the redesigned UI), short display label, and long
/// mathematical name shown in card tooltips. The long name uses
/// the `pentachoron` / `tesseract` / `hexadecachoron` family; the
/// `*-plex` aliases (pentaplex, dodecaplex, ...) are deliberately
/// avoided since "plex" is dimension-generalized rather than
/// being the actual 4D name.
#[derive(Copy, Clone, PartialEq)]
pub(crate) struct ShapeEntry {
    pub(crate) shape: u32,
    pub(crate) body_color: [f32; 3],
    pub(crate) label: &'static str,
    pub(crate) long_name: &'static str,
}

/// Default row when no `--shapes` argument is given. Ordered to put
/// the 24-cell first (most "4D-distinct" cross-section), then the
/// pentachoron / 16-cell / tesseract triple; visually contrasting
/// shapes left-to-right.
pub(crate) const DEFAULT_ROW: &[ShapeEntry] = &[
    ShapeEntry {
        shape: SHAPE_24CELL,
        body_color: [0.95, 0.45, 0.85],
        label: "24-cell",
        long_name: "icositetrachoron",
    },
    ShapeEntry {
        shape: SHAPE_PENTATOPE,
        body_color: [0.95, 0.55, 0.30],
        label: "5-cell",
        long_name: "pentachoron",
    },
    ShapeEntry {
        shape: SHAPE_16CELL,
        body_color: [0.55, 0.95, 0.40],
        label: "16-cell",
        long_name: "hexadecachoron",
    },
    ShapeEntry {
        shape: SHAPE_TESSERACT,
        body_color: [0.30, 0.55, 0.95],
        label: "8-cell",
        long_name: "tesseract",
    },
];

/// Catalog of every shipped 4D shape: the six convex regular
/// polychora plus four non-polychoral SDF-trivial shapes
/// (3-sphere, duocylinder, Clifford torus, spherinder). Used by
/// the filmstrip subject picker and the `+` shape menu. Colours
/// are RGB float channels passed straight to the WGSL kernel
/// (engine doesn't constrain the colour space).
pub(crate) const SHAPE_CATALOG: &[ShapeEntry] = &[
    ShapeEntry {
        shape: SHAPE_PENTATOPE,
        body_color: [0.95, 0.55, 0.30],
        label: "5-cell",
        long_name: "pentachoron",
    },
    ShapeEntry {
        shape: SHAPE_TESSERACT,
        body_color: [0.30, 0.55, 0.95],
        label: "8-cell",
        long_name: "tesseract",
    },
    ShapeEntry {
        shape: SHAPE_16CELL,
        body_color: [0.55, 0.95, 0.40],
        label: "16-cell",
        long_name: "hexadecachoron",
    },
    ShapeEntry {
        shape: SHAPE_24CELL,
        body_color: [0.95, 0.45, 0.85],
        label: "24-cell",
        long_name: "icositetrachoron",
    },
    ShapeEntry {
        shape: SHAPE_120CELL,
        body_color: [0.40, 0.85, 0.85],
        label: "120-cell",
        long_name: "hecatonicosachoron",
    },
    ShapeEntry {
        shape: SHAPE_600CELL,
        body_color: [0.95, 0.85, 0.40],
        label: "600-cell",
        long_name: "hexacosichoron",
    },
    ShapeEntry {
        shape: SHAPE_3SPHERE,
        body_color: [0.85, 0.40, 0.40],
        label: "3-sphere",
        long_name: "hypersphere (4-ball)",
    },
    ShapeEntry {
        shape: SHAPE_DUOCYLINDER,
        body_color: [0.60, 0.45, 0.90],
        label: "duocyl",
        long_name: "duocylinder (D² × D²)",
    },
    ShapeEntry {
        shape: SHAPE_CLIFFORD_TORUS,
        body_color: [0.70, 0.85, 0.35],
        label: "clifford",
        long_name: "Clifford torus tube",
    },
    ShapeEntry {
        shape: SHAPE_SPHERINDER,
        body_color: [0.85, 0.55, 0.75],
        label: "spherinder",
        long_name: "spherinder (B³ × interval)",
    },
];

/// Render a category-grouped shape menu into the current ui.
/// Both call sites (the `+` shape menu and the filmstrip
/// subject combo) use this so the layout stays consistent: top
/// level lists the [`SHAPE_CATEGORIES`] entries, each opens a
/// nested submenu of the shapes in that category, every entry
/// carries a `long_name` hover tooltip. `on_select` fires when
/// the user clicks an entry; the helper closes the menu.
pub(crate) fn render_shape_catalog_menu(
    ui: &mut egui::Ui,
    mut on_select: impl FnMut(ShapeEntry),
) {
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

/// Subcategories of [`SHAPE_CATALOG`], expressed as half-open
/// index ranges into the catalog. Used by the shape menus
/// (`+` button and filmstrip subject combo) to group entries
/// with a header label and separator. Keeping the categories as
/// ranges (rather than nested slices) lets `parse_shape_name`
/// and direct `SHAPE_CATALOG[i]` lookups stay flat.
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

/// Catalog of named shapes. Both common math-name aliases (the
/// `n-cell` form) and Platonic-slice aliases (the `tetrahedron` /
/// `cube` / etc. form) resolve to the same shape index.
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

/// Parse the row from CLI arguments. Looks for `--shapes name1 name2 ...`
/// (consumes everything after the flag). Returns [`DEFAULT_ROW`] if
/// the flag isn't present.
pub(crate) fn parse_row_from_args() -> Result<Vec<ShapeEntry>> {
    let args: Vec<String> = std::env::args().collect();
    let Some(idx) = args.iter().position(|a| a == "--shapes") else {
        return Ok(DEFAULT_ROW.to_vec());
    };
    let names = &args[idx + 1..];
    if names.is_empty() {
        return Err(anyhow!("--shapes flag passed but no shape names followed"));
    }
    names.iter().map(|n| parse_shape_name(n)).collect()
}
