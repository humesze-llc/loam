//! Presentation-tier animation. Two complementary models:
//!
//! - **t-driven** ([`Track`], [`Playhead`]): the value is a pure function of a
//!   timeline position `t`. This gives *deterministic playback* -- exact
//!   scrubbing, exact replay, and frame-exact capture (step `t`, render,
//!   independent of frame rate). The narrative spine of a guided demo should be
//!   built this way.
//! - **dt-driven** ([`Animated`]): a value that eases toward a target on
//!   wall-clock dt. For live, non-recorded reactions (a cursor-follow, a hover)
//!   where there is no timeline position to evaluate against.
//!
//! This is NOT the engine's Tier-0 simulation determinism (no cross-machine
//! bit-reproducibility contract); it is presentation timing and must never feed
//! simulation state.

/// Cubic ease-in-out on `[0, 1] -> [0, 1]`. The default for camera/layout moves.
pub fn ease_in_out_cubic(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    if t < 0.5 {
        4.0 * t * t * t
    } else {
        let u = -2.0 * t + 2.0;
        1.0 - u * u * u / 2.0
    }
}

/// Cubic ease-out on `[0, 1] -> [0, 1]`. Fast start, gentle settle.
pub fn ease_out_cubic(t: f32) -> f32 {
    let u = 1.0 - t.clamp(0.0, 1.0);
    1.0 - u * u * u
}

/// Linear identity easing.
pub fn linear(t: f32) -> f32 {
    t.clamp(0.0, 1.0)
}

/// A scalar easing from a start value toward a target over a fixed duration,
/// advanced by wall-clock dt. For live reactions, not the recorded timeline.
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

    pub fn animate_to(&mut self, target: f32, duration: f32, ease: fn(f32) -> f32) {
        self.from = self.value;
        self.target = target;
        self.elapsed = 0.0;
        self.duration = duration.max(0.0);
        self.ease = ease;
    }

    pub fn snap(&mut self, value: f32) {
        self.value = value;
        self.from = value;
        self.target = value;
        self.elapsed = self.duration;
    }

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

/// One keyframe: the track holds `value` at time `t`, eased into from the
/// previous key by `ease`.
#[derive(Clone, Copy, Debug)]
pub struct Key {
    pub t: f32,
    pub value: f32,
    pub ease: fn(f32) -> f32,
}

/// A scalar as a pure function of timeline `t`, defined by sorted keyframes. The
/// value is held constant before the first key and after the last; between keys
/// it interpolates with the *later* key's easing. Sampling is stateless, so any
/// `t` reproduces the same value (deterministic playback).
#[derive(Clone, Debug, Default)]
pub struct Track {
    keys: Vec<Key>,
}

impl Track {
    pub fn new() -> Self {
        Self { keys: Vec::new() }
    }

    /// A track that is `value` everywhere.
    pub fn constant(value: f32) -> Self {
        Self {
            keys: vec![Key {
                t: 0.0,
                value,
                ease: linear,
            }],
        }
    }

    /// Add a keyframe (builder style). Keys should be added in ascending `t`.
    pub fn key(mut self, t: f32, value: f32, ease: fn(f32) -> f32) -> Self {
        self.keys.push(Key { t, value, ease });
        self
    }

    pub fn sample(&self, t: f32) -> f32 {
        match self.keys.first() {
            None => 0.0,
            Some(first) if t <= first.t => first.value,
            _ => {
                let last = self.keys.last().unwrap();
                if t >= last.t {
                    return last.value;
                }
                let i = self.keys.partition_point(|k| k.t <= t) - 1;
                let a = &self.keys[i];
                let b = &self.keys[i + 1];
                let span = b.t - a.t;
                let u = if span > 0.0 { (t - a.t) / span } else { 1.0 };
                a.value + (b.value - a.value) * (b.ease)(u)
            }
        }
    }
}

/// The timeline playhead: a position `t` in seconds within `[0, duration]`.
/// Playing advances `t`; seeking sets it. Everything downstream is sampled at
/// `t`, so play / pause / scrub / capture all reproduce identical frames.
#[derive(Clone, Copy, Debug)]
pub struct Playhead {
    pub t: f32,
    pub duration: f32,
    pub playing: bool,
}

impl Playhead {
    pub fn new(duration: f32) -> Self {
        Self {
            t: 0.0,
            duration,
            playing: true,
        }
    }

    pub fn advance(&mut self, dt: f32) {
        if self.playing {
            self.t = (self.t + dt).clamp(0.0, self.duration);
        }
    }

    pub fn seek(&mut self, t: f32) {
        self.t = t.clamp(0.0, self.duration);
    }

    pub fn progress(&self) -> f32 {
        if self.duration > 0.0 {
            (self.t / self.duration).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    pub fn finished(&self) -> bool {
        self.t >= self.duration
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn easings_fix_endpoints() {
        for ease in [ease_in_out_cubic, ease_out_cubic, linear] {
            assert!((ease(0.0)).abs() < 1e-6);
            assert!((ease(1.0) - 1.0).abs() < 1e-6);
        }
        assert!((ease_in_out_cubic(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn track_holds_then_interpolates() {
        let track = Track::new().key(1.0, 0.0, linear).key(2.0, 10.0, linear);
        assert_eq!(track.sample(0.0), 0.0); // before first
        assert_eq!(track.sample(1.0), 0.0);
        assert!((track.sample(1.5) - 5.0).abs() < 1e-5); // midpoint, linear
        assert_eq!(track.sample(2.0), 10.0);
        assert_eq!(track.sample(9.0), 10.0); // after last
    }

    #[test]
    fn track_sampling_is_stateless() {
        let track = Track::new().key(0.0, 0.0, linear).key(4.0, 8.0, linear);
        // Same t -> same value regardless of access order (deterministic playback).
        assert_eq!(track.sample(3.0), track.sample(3.0));
        assert!((track.sample(1.0) - 2.0).abs() < 1e-5);
        assert!((track.sample(3.0) - 6.0).abs() < 1e-5);
    }

    #[test]
    fn playhead_advances_seeks_and_clamps() {
        let mut p = Playhead::new(2.0);
        p.advance(0.5);
        assert!((p.t - 0.5).abs() < 1e-6);
        p.playing = false;
        p.advance(1.0);
        assert!((p.t - 0.5).abs() < 1e-6); // paused
        p.seek(5.0);
        assert_eq!(p.t, 2.0); // clamped
        assert!(p.finished());
    }

    #[test]
    fn animated_reaches_target() {
        let mut a = Animated::new(0.0);
        a.animate_to(10.0, 1.0, ease_in_out_cubic);
        a.advance(0.5);
        assert!(a.value() > 0.0 && a.value() < 10.0);
        a.advance(0.6);
        assert!(a.is_done() && (a.value() - 10.0).abs() < 1e-5);
    }
}
