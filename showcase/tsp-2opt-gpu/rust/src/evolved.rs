use crate::TwoOptPriority;

pub struct Evolved;

/// Starting point for evolution: Simple greedy approach.
/// This will be evolved to beat the baselines.
impl TwoOptPriority for Evolved {
    fn priority(
        &self,
        delta: f64,
        _edge1_len: f64,
        _edge2_len: f64,
        _new_edge1_len: f64,
        _new_edge2_len: f64,
        _tour_len: f64,
        _n: usize,
    ) -> f64 {
        // Simple greedy: just use negative delta as priority
        // Larger improvements (more negative delta) get higher priority
        -delta
    }

    fn name(&self) -> &'static str {
        "evolved"
    }
}
