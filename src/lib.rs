//! platomon — Platonic-solid ASCII system monitor.
//!
//! The binary lives in `main.rs`; everything else is a library so the
//! renderer can be benchmarked and unit-tested.

pub mod ascii;
pub mod aspect;
pub mod clock;
pub mod collector;
pub mod config;
pub mod metrics;
pub mod overlay;
pub mod render;
pub mod scene;
pub mod sim;
pub mod ui;
