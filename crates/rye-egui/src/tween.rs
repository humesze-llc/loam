//! Presentation-tier easing + a scalar that eases toward a target on wall-clock
//! dt. For camera moves, layout splits, expression blends, fades. This is NOT on
//! the deterministic sim/math path: animation reads real time and must never feed
//! simulation state.

/// Cubic ease-in-out on `[0, 1] -> [0, 1]`. Slow at both ends, fast in the
/// middle; the default for camera and layout moves.
pub fn ease_in_out_cubic(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    if t < 0.5 {
        4.0 * t * t * t
    } else {
        let u = -2.0 * t + 2.0;
        1.0 - u * u * u / 2.0
    }
}

/// Cubic ease-out on `[0, 1] -> [0, 1]`. Fast start, gentle settle; for things
/// that arrive (a shape entering, a panel opening).
pub fn ease_out_cubic(t: f32) -> f32 {
    let u = 1.0 - t.clamp(0.0, 1.0);
    1.0 - u * u * u
}

/// A scalar easing from a start value toward a target over a fixed duration,
/// advanced by wall-clock dt. A zero-duration target snaps.
#[derive(Clone, Copy, Debug)]
pub struct Animated {
    value: f32,
    from: f32,
    target: f32,
    elapsed: f32,
    duration: f32,
    ease: fn(f32) -> f32,
}

impl Animated {
    pub fn new(value: f32) -> Self {
        Self {
            value,
            from: value,
            target: value,
            elapsed: 0.0,
            duration: 0.0,
            ease: ease_in_out_cubic,
        }
    }

    /// Begin easing from the current value to `target` over `duration` seconds.
    pub fn animate_to(&mut self, target: f32, duration: f32, ease: fn(f32) -> f32) {
        self.from = self.value;
        self.target = target;
        self.elapsed = 0.0;
        self.duration = duration.max(0.0);
        self.ease = ease;
    }

    /// Jump to `value` immediately, cancelling any animation.
    pub fn snap(&mut self, value: f32) {
        self.value = value;
        self.from = value;
        self.target = value;
        self.elapsed = self.duration;
    }

    /// Advance by `dt` seconds and return the new value.
    pub fn advance(&mut self, dt: f32) -> f32 {
        self.elapsed += dt;
        let u = if self.duration <= 0.0 {
            1.0
        } else {
            (self.elapsed / self.duration).clamp(0.0, 1.0)
        };
        self.value = self.from + (self.target - self.from) * (self.ease)(u);
        self.value
    }

    pub fn value(&self) -> f32 {
        self.value
    }

    pub fn target(&self) -> f32 {
        self.target
    }

    pub fn is_done(&self) -> bool {
        self.elapsed >= self.duration
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn easings_fix_endpoints_and_midpoint() {
        for ease in [ease_in_out_cubic, ease_out_cubic] {
            assert!((ease(0.0) - 0.0).abs() < 1e-6);
            assert!((ease(1.0) - 1.0).abs() < 1e-6);
        }
        assert!((ease_in_out_cubic(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn easings_clamp_out_of_range() {
        assert_eq!(ease_in_out_cubic(-1.0), 0.0);
        assert_eq!(ease_out_cubic(2.0), 1.0);
    }

    #[test]
    fn animated_reaches_target_after_duration() {
        let mut a = Animated::new(0.0);
        a.animate_to(10.0, 1.0, ease_in_out_cubic);
        assert!(!a.is_done());
        a.advance(0.5);
        assert!(a.value() > 0.0 && a.value() < 10.0);
        a.advance(0.6);
        assert!(a.is_done());
        assert!((a.value() - 10.0).abs() < 1e-5);
    }

    #[test]
    fn snap_is_immediate() {
        let mut a = Animated::new(0.0);
        a.animate_to(5.0, 2.0, ease_out_cubic);
        a.snap(3.0);
        assert_eq!(a.value(), 3.0);
        assert!(a.is_done());
    }
}
