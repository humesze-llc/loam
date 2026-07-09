//! `loam-time`: deterministic fixed-timestep scheduling for Loam.
//!
//! Given that wall-clock time advanced by some amount, how many
//! fixed-duration simulation ticks should run?
//!
//! ## The deterministic-sim split
//!
//! The simulation is a pure, bit-reproducible function
//! `tick(state, input, tick_number) -> state'`; time enters only as the
//! tick number, never wall-clock. This crate is the
//! wall-clock-to-tick-count adapter the local render loop uses. Replays
//! and rollback netcode drive the sim by tick number directly and never
//! touch [`FixedTimestep`].
//!
//! ## Typical loop
//!
//! ```no_run
//! # use std::time::Instant;
//! # use loam_time::FixedTimestep;
//! # struct State; struct Input;
//! # fn collect_input() -> Input { Input }
//! # fn run_sim(_state: &mut State, _input: &Input, _tick: u64) {}
//! # fn render(_state: &State, _alpha: f32) {}
//! # let mut state = State;
//! let mut timestep = FixedTimestep::new(60);
//! loop {
//!     let input = collect_input();
//!     for tick in timestep.advance(Instant::now()) {
//!         run_sim(&mut state, &input, tick);
//!     }
//!     render(&state, timestep.alpha());
//! #   break;
//! }
//! ```

pub mod alloc;
mod fixed_timestep;
pub mod frame_trace;

pub use fixed_timestep::{FixedTimestep, DEFAULT_MAX_CATCH_UP};
