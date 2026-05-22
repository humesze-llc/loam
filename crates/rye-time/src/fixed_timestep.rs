//! Fixed-timestep accumulator.

use std::ops::Range;
// `web_time` is a drop-in for `std::time` that works on both native and wasm32. On
// native it re-exports `std::time::Instant` / `Duration` verbatim; on wasm32 it backs
// `Instant` with `performance.now()`. `std::time::Instant::now` panics on wasm32, so
// the swap is mandatory for the browser runtime path.
use std::time::Duration;
use web_time::Instant;

/// Default cap on how many simulation ticks we'll run per frame when catching up to
/// wall-clock time. Excess is dropped to avoid the "spiral of death" where a slow sim
/// falls further behind each frame.
pub const DEFAULT_MAX_CATCH_UP: u32 = 10;

/// Tick-rate accumulator driving a deterministic sim from wall-clock time.
///
/// Construct with [`FixedTimestep::new`], then each frame call [`FixedTimestep::advance`]
/// with the current [`Instant`] and iterate the returned range to run sim ticks.
/// [`FixedTimestep::alpha`] gives the interpolation factor for rendering between the
/// last two tick states.
///
/// Tick duration is stored as nanoseconds computed from the target Hz, so a given
/// `FixedTimestep::new(hz)` produces the same tick duration on every machine.
#[derive(Debug, Clone)]
pub struct FixedTimestep {
    dt: Duration,
    accumulator: Duration,
    last_instant: Option<Instant>,
    tick: u64,
    max_catch_up: u32,
}

impl FixedTimestep {
    /// Construct with a target tick rate in hertz.
    ///
    /// Panics if `hz == 0`.
    pub fn new(hz: u32) -> Self {
        assert!(hz > 0, "tick rate must be positive");
        Self {
            dt: Duration::from_nanos(1_000_000_000 / u64::from(hz)),
            accumulator: Duration::ZERO,
            last_instant: None,
            tick: 0,
            max_catch_up: DEFAULT_MAX_CATCH_UP,
        }
    }

    /// Override the spiral-of-death cap. Default is [`DEFAULT_MAX_CATCH_UP`].
    pub fn with_max_catch_up(mut self, n: u32) -> Self {
        self.max_catch_up = n;
        self
    }

    /// Current tick number. Monotonic, starts at 0, advances by one for each tick
    /// yielded by [`FixedTimestep::advance`].
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Duration of one sim tick.
    pub fn dt(&self) -> Duration {
        self.dt
    }

    /// Duration of one sim tick as f32 seconds. Use this for physics integration in sim
    /// code.
    pub fn dt_seconds(&self) -> f32 {
        self.dt.as_secs_f32()
    }

    /// Interpolation alpha in `[0.0, 1.0)`: how far between the last completed tick and
    /// the next pending tick we are in wall-clock time. Use for render-side smoothing.
    pub fn alpha(&self) -> f32 {
        let a = self.accumulator.as_secs_f64() / self.dt.as_secs_f64();
        (a as f32).clamp(0.0, 1.0)
    }

    /// Advance wall-clock time to `now` and return the range of tick numbers the caller
    /// should execute this frame.
    ///
    /// The first call after construction primes the wall-clock reference and returns an
    /// empty range (no elapsed time to account for yet).
    ///
    /// If the sim has fallen further than `max_catch_up` ticks behind wall-clock time,
    /// the excess is dropped rather than queued, the render loop recovers to real-time
    /// at the cost of a visual jump.
    pub fn advance(&mut self, now: Instant) -> Range<u64> {
        let last = match self.last_instant.replace(now) {
            Some(t) => t,
            None => return self.tick..self.tick,
        };

        self.accumulator += now.saturating_duration_since(last);

        let start = self.tick;
        let mut catch_up = 0u32;
        while self.accumulator >= self.dt && catch_up < self.max_catch_up {
            self.accumulator -= self.dt;
            self.tick += 1;
            catch_up += 1;
        }

        // Spiral cap: discard any remaining whole-tick excess so the
        // accumulator stays bounded even under pathological stalls.
        while self.accumulator >= self.dt {
            self.accumulator -= self.dt;
        }

        start..self.tick
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Instant {
        Instant::now()
    }

    #[test]
    fn dt_matches_hz() {
        let t = FixedTimestep::new(60);
        assert_eq!(t.dt(), Duration::from_nanos(16_666_666));
    }

    #[test]
    fn first_advance_primes_and_yields_nothing() {
        let mut t = FixedTimestep::new(60);
        let range = t.advance(base());
        assert_eq!(range, 0..0);
        assert_eq!(t.tick(), 0);
    }

    #[test]
    fn exactly_one_dt_yields_one_tick() {
        let mut t = FixedTimestep::new(60);
        let b = base();
        t.advance(b);
        let range = t.advance(b + t.dt());
        assert_eq!(range, 0..1);
        assert_eq!(t.tick(), 1);
    }

    #[test]
    fn fractional_accumulator_drives_alpha() {
        let mut t = FixedTimestep::new(60);
        let b = base();
        t.advance(b);
        let dt = t.dt();
        let range = t.advance(b + dt * 3 + dt / 2);
        assert_eq!(range, 0..3);
        assert_eq!(t.tick(), 3);
        let a = t.alpha();
        assert!(
            (a - 0.5).abs() < 1e-3,
            "alpha should be ~0.5 after 3.5 dt, got {a}",
        );
    }

    #[test]
    fn alpha_is_zero_when_aligned() {
        let mut t = FixedTimestep::new(60);
        let b = base();
        t.advance(b);
        t.advance(b + t.dt() * 2);
        assert!(t.alpha() < 1e-6);
    }

    #[test]
    fn alpha_always_in_unit_range() {
        let mut t = FixedTimestep::new(60);
        let b = base();
        t.advance(b);
        for k in 1..100 {
            t.advance(b + Duration::from_millis(k * 7));
            let a = t.alpha();
            assert!((0.0..1.0).contains(&a), "alpha {a} out of [0,1)");
        }
    }

    #[test]
    fn spiral_cap_drops_excess_ticks() {
        let mut t = FixedTimestep::new(60).with_max_catch_up(5);
        let b = base();
        t.advance(b);
        // 100 ticks of wall-clock elapsed; cap should yield exactly 5.
        let range = t.advance(b + t.dt() * 100);
        assert_eq!(range.end - range.start, 5);
        assert_eq!(t.tick(), 5);
        // Accumulator should have been drained; alpha near zero.
        assert!(t.alpha() < 1e-3);
    }

    #[test]
    fn ticks_are_monotonic_across_many_frames() {
        let mut t = FixedTimestep::new(120);
        let b = base();
        t.advance(b);
        let mut last_end = 0;
        for frame in 1..=50 {
            let range = t.advance(b + Duration::from_millis(frame * 10));
            assert_eq!(range.start, last_end);
            assert!(range.end >= range.start);
            last_end = range.end;
        }
        assert_eq!(t.tick(), last_end);
    }

    #[test]
    fn dt_seconds_matches_hz() {
        let t = FixedTimestep::new(60);
        assert!((t.dt_seconds() - 1.0 / 60.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "tick rate must be positive")]
    fn zero_hz_panics() {
        let _ = FixedTimestep::new(0);
    }
}
