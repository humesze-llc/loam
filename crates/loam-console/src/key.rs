//! Key vocabulary for console binds and the toggle key.
//!
//! The console names its own keys so binds carry no UI-framework type. A
//! frontend translates its own key events into these at the input boundary
//! (`loam_egui::console` holds the egui table).

/// A physical key the console can bind or toggle on. Modifiers are not
/// representable: binds fire on unmodified presses only.
///
/// Declaration order is the [`Key::ALL`] order and the [`Ord`] order, which is
/// also the order bound keys fire in when several land in one frame.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Key {
    Escape,
    Tab,
    Backspace,
    Enter,
    Space,
    Delete,

    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,

    Home,
    End,
    PageUp,
    PageDown,

    Backtick,
    Minus,
    Equals,

    Num0,
    Num1,
    Num2,
    Num3,
    Num4,
    Num5,
    Num6,
    Num7,
    Num8,
    Num9,

    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,

    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
}

impl Key {
    /// Every variant, in declaration order, so a frontend can exhaustively
    /// exercise its own key table. `ALL[i] as usize == i` holds; adding a
    /// variant means adding it here.
    pub const ALL: &'static [Key] = &[
        Key::Escape,
        Key::Tab,
        Key::Backspace,
        Key::Enter,
        Key::Space,
        Key::Delete,
        Key::ArrowUp,
        Key::ArrowDown,
        Key::ArrowLeft,
        Key::ArrowRight,
        Key::Home,
        Key::End,
        Key::PageUp,
        Key::PageDown,
        Key::Backtick,
        Key::Minus,
        Key::Equals,
        Key::Num0,
        Key::Num1,
        Key::Num2,
        Key::Num3,
        Key::Num4,
        Key::Num5,
        Key::Num6,
        Key::Num7,
        Key::Num8,
        Key::Num9,
        Key::A,
        Key::B,
        Key::C,
        Key::D,
        Key::E,
        Key::F,
        Key::G,
        Key::H,
        Key::I,
        Key::J,
        Key::K,
        Key::L,
        Key::M,
        Key::N,
        Key::O,
        Key::P,
        Key::Q,
        Key::R,
        Key::S,
        Key::T,
        Key::U,
        Key::V,
        Key::W,
        Key::X,
        Key::Y,
        Key::Z,
        Key::F1,
        Key::F2,
        Key::F3,
        Key::F4,
        Key::F5,
        Key::F6,
        Key::F7,
        Key::F8,
        Key::F9,
        Key::F10,
        Key::F11,
        Key::F12,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ALL` is the discriminant sequence with no gaps, duplicates or
    /// reordering, so `ALL[i] as usize == i` can be relied on by frontends
    /// building index-keyed tables.
    #[test]
    fn all_is_the_dense_discriminant_sequence() {
        for (index, key) in Key::ALL.iter().enumerate() {
            assert_eq!(*key as usize, index, "ALL[{index}] is out of order");
        }
        let last = *Key::ALL.last().expect("ALL is non-empty");
        assert_eq!(
            Key::ALL.len(),
            last as usize + 1,
            "ALL skips a variant before {last:?}"
        );
    }

    /// Ord follows declaration order, which fixes the order bound keys fire in
    /// when several are pressed in the same frame.
    #[test]
    fn ord_follows_declaration_order() {
        assert!(Key::ALL.windows(2).all(|w| w[0] < w[1]));
    }
}
