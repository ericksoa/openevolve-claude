//! Evolved acceptance heuristic for speculative decoding
//!
//! This file is modified by the evolution process.
//! The goal is to find a heuristic that:
//! 1. Has high acceptance rate (faster inference)
//! 2. Maintains quality (doesn't accept "wrong" tokens)
//!
//! Starting point: Standard rejection sampling

use crate::AcceptanceHeuristic;

pub struct Evolved;

impl AcceptanceHeuristic for Evolved {
    fn acceptance_threshold(
        &self,
        draft_prob: f64,
        target_prob: f64,
        _position: usize,
        _draft_entropy: f64,
        _target_entropy: f64,
        _top_token_match: bool,
    ) -> f64 {
        // Starting point: standard rejection sampling
        // Evolution will discover better formulations
        (target_prob / draft_prob).min(1.0)
    }
}
