//! `rye-time`: deterministic fixed-timestep scheduling for Rye.
//!
//! This crate answers one question: given wall-clock time has advanced by
//! some amount, how many fixed-duration simulation ticks should run?
//!
//! ## The deterministic-sim split
//!
//! For lockstep multiplayer, the **simulation** is a pure function:
//!
//! ```text
//! fn tick(state, input, tick_number) -> state'
//! ```
//!
//! It must be bit-reproducible given the same inputs. It does NOT consult
//! wall-clock time, time only enters as the tick number.
//!
//! This crate is the wall-clock-to-tick-count *adapter* that the local
//! render loop uses. Replays and rollback netcode drive the sim directly
//! by tick number and never touch [`FixedTimestep`].
//!
//! ## Typical loop
//!
//! ```no_run
//! # use std::time::Instant;
//! # use rye_time::FixedTimestep;
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

mod fixed_timestep;
pub mod frame_trace;

pub use fixed_timestep::{FixedTimestep, DEFAULT_MAX_CATCH_UP};
