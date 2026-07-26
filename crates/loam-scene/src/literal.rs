//! Float-to-WGSL-literal printing, shared by the 3D and 4D emit paths.

/// Print `v` as a WGSL `f32` literal.
///
/// Shortest round-trip, so parsing the emitted token recovers the exact input
/// bits and the emitter contributes no floor to CPU/GPU parity.
///
/// `Debug` carries a decimal point or an exponent for every finite `f32` where
/// `Display` does not, and that is load-bearing rather than cosmetic: WGSL
/// types a bare digit run as `AbstractInt`, whose range is `i64`, so a
/// magnitude at or above 2^63 is "not representable by target type" and the
/// module fails to parse (WGSL spec 15.2, numeric literals).
/// `every_finite_f32_prints_with_a_point_or_an_exponent` pins that property, so
/// a std formatting change fails here rather than in naga.
///
/// # Panics
///
/// On non-finite `v`. WGSL 15.2 has no literal spelling for infinity or NaN,
/// and the spellings that do exist (`bitcast<f32>(0x7f800000u)`) would put a
/// constant on the GPU that poisons every distance derived from it to NaN,
/// where nothing can attribute the failure back to the scene. Scene data
/// reaching the emitter is finite; a violation is an upstream defect and fails
/// on the CPU naming the value.
pub(crate) fn wgsl_f32(v: f32) -> String {
    assert!(
        v.is_finite(),
        "non-finite scene constant {v:?} has no WGSL literal",
    );
    format!("{v:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The emitted token must never be a bare digit run: WGSL would type it as
    /// `AbstractInt` and reject every magnitude at or above 2^63. Swept across
    /// the whole f32 exponent range, both signs, plus the integer-valued
    /// magnitudes where `Display` drops the point. `wgsl_f32` leans on `Debug`
    /// for this rather than patching its output, so this is the pin on `Debug`
    /// itself.
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
        for v in [
            0.0_f32,
            -0.0,
            1.0,
            -1.0,
            1e9,
            2.0_f32.powi(63),
            f32::MAX,
            f32::MIN,
            f32::MIN_POSITIVE,
            f32::from_bits(1), // smallest subnormal
        ] {
            let literal = wgsl_f32(v);
            assert!(
                literal.contains('.') || literal.contains('e'),
                "bare integer literal `{literal}` for {v:?}",
            );
        }
    }

    /// Infinity and NaN have no WGSL literal spelling, and `Debug` prints them
    /// as `inf` / `NaN`, which parse as identifiers rather than numbers. The
    /// emitter must refuse them instead of shipping a token that either fails
    /// in naga or resolves to something the scene did not ask for.
    #[test]
    #[should_panic(expected = "non-finite")]
    fn positive_infinity_has_no_literal_and_is_rejected() {
        wgsl_f32(f32::INFINITY);
    }

    #[test]
    #[should_panic(expected = "non-finite")]
    fn negative_infinity_has_no_literal_and_is_rejected() {
        wgsl_f32(f32::NEG_INFINITY);
    }

    #[test]
    #[should_panic(expected = "non-finite")]
    fn nan_has_no_literal_and_is_rejected() {
        wgsl_f32(f32::NAN);
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
