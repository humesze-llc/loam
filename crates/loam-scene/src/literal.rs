//! Float-to-WGSL-literal printing, shared by the 3D and 4D emit paths.

/// Print `v` as a WGSL `f32` literal.
///
/// Shortest round-trip, so parsing the emitted token recovers the exact input
/// bits and the emitter contributes no floor to CPU/GPU parity.
///
/// The point or exponent is load-bearing, not cosmetic: WGSL types a bare
/// digit run as `AbstractInt`, whose range is `i64`, so a magnitude at or above
/// 2^63 is "not representable by target type" and the module fails to parse
/// (WGSL spec 15.2, numeric literals). `Debug` already carries one for every
/// finite `f32` where `Display` does not, but that is a formatting policy
/// rather than a stability guarantee, so the invariant is enforced here.
///
/// Non-finite input has no WGSL literal spelling; callers hold finite scene
/// data.
pub(crate) fn wgsl_f32(v: f32) -> String {
    let mut literal = format!("{v:?}");
    if !literal.contains(['.', 'e']) {
        literal.push_str(".0");
    }
    literal
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The emitted token must never be a bare digit run: WGSL would type it as
    /// `AbstractInt` and reject every magnitude at or above 2^63. Swept across
    /// the whole f32 exponent range, both signs, plus the integer-valued
    /// magnitudes where `Display` drops the point.
    #[test]
    fn every_finite_f32_prints_with_a_point_or_an_exponent() {
        for exponent in -45..=38 {
            for mantissa in [1.0_f32, 1.5, 9.99] {
                let magnitude = mantissa * 10.0_f32.powi(exponent);
                for v in [magnitude, -magnitude] {
                    if !v.is_finite() || v == 0.0 {
                        continue;
                    }
                    let literal = wgsl_f32(v);
                    assert!(
                        literal.contains('.') || literal.contains('e'),
                        "bare integer literal `{literal}` for {v:?}",
                    );
                }
            }
        }
        for v in [0.0_f32, -0.0, 1.0, -1.0, 1e9, 2.0_f32.powi(63), f32::MAX] {
            let literal = wgsl_f32(v);
            assert!(
                literal.contains('.') || literal.contains('e'),
                "bare integer literal `{literal}` for {v:?}",
            );
        }
    }

    /// Parsing the emitted literal must recover the exact input bits, so the
    /// GPU evaluates the same constant the CPU holds.
    #[test]
    fn emitted_literal_round_trips_to_the_input_bits() {
        for exponent in -45..=38 {
            for mantissa in [1.0_f32, 1.5, 9.99, 3.7] {
                let magnitude = mantissa * 10.0_f32.powi(exponent);
                for v in [magnitude, -magnitude] {
                    if !v.is_finite() {
                        continue;
                    }
                    let literal = wgsl_f32(v);
                    let parsed: f32 = literal.parse().expect("literal parses as f32");
                    assert_eq!(parsed.to_bits(), v.to_bits(), "`{literal}` from {v:?}");
                }
            }
        }
    }
}
