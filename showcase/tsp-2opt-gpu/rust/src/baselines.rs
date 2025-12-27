use crate::TwoOptPriority;

/// Greedy Delta: Simply use the improvement amount as priority.
/// Standard first-improvement 2-opt when combined with break-on-first.
pub struct GreedyDelta;

impl TwoOptPriority for GreedyDelta {
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
        // Negative delta = improvement, we want high priority for improvements
        -delta
    }

    fn name(&self) -> &'static str {
        "greedy_delta"
    }
}

/// Best Improvement: Only consider improving moves, prioritize largest gain.
/// Classic best-improvement 2-opt.
pub struct BestImprovement;

impl TwoOptPriority for BestImprovement {
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
        if delta < 0.0 {
            -delta // Larger improvements get higher priority
        } else {
            f64::NEG_INFINITY // Non-improving moves get lowest priority
        }
    }

    fn name(&self) -> &'static str {
        "best_improvement"
    }
}

/// Relative Improvement: Prioritize by relative gain (delta / tour_len).
/// Favors moves that reduce tour by larger percentage.
pub struct RelativeGain;

impl TwoOptPriority for RelativeGain {
    fn priority(
        &self,
        delta: f64,
        _edge1_len: f64,
        _edge2_len: f64,
        _new_edge1_len: f64,
        _new_edge2_len: f64,
        tour_len: f64,
        _n: usize,
    ) -> f64 {
        if delta < 0.0 {
            -delta / tour_len
        } else {
            f64::NEG_INFINITY
        }
    }

    fn name(&self) -> &'static str {
        "relative_gain"
    }
}

/// Long Edge Removal: Prioritize removing the longest edges.
/// Based on intuition that long edges are likely suboptimal.
pub struct LongEdgeRemoval;

impl TwoOptPriority for LongEdgeRemoval {
    fn priority(
        &self,
        delta: f64,
        edge1_len: f64,
        edge2_len: f64,
        _new_edge1_len: f64,
        _new_edge2_len: f64,
        _tour_len: f64,
        _n: usize,
    ) -> f64 {
        if delta < 0.0 {
            // Combine improvement with edge lengths
            -delta + 0.1 * (edge1_len + edge2_len)
        } else {
            f64::NEG_INFINITY
        }
    }

    fn name(&self) -> &'static str {
        "long_edge_removal"
    }
}

/// Edge Ratio: Prioritize based on ratio of removed to added edge lengths.
/// Higher ratio = removing relatively longer edges.
pub struct EdgeRatio;

impl TwoOptPriority for EdgeRatio {
    fn priority(
        &self,
        delta: f64,
        edge1_len: f64,
        edge2_len: f64,
        new_edge1_len: f64,
        new_edge2_len: f64,
        _tour_len: f64,
        _n: usize,
    ) -> f64 {
        if delta < 0.0 {
            let removed = edge1_len + edge2_len;
            let added = new_edge1_len + new_edge2_len;
            // Ratio of removed to added (higher = better swap)
            removed / (added + 1e-10)
        } else {
            f64::NEG_INFINITY
        }
    }

    fn name(&self) -> &'static str {
        "edge_ratio"
    }
}

/// Lin-Kernighan Inspired: Combine multiple signals like LK heuristic.
/// Uses edge lengths and improvement together.
pub struct LKInspired;

impl TwoOptPriority for LKInspired {
    fn priority(
        &self,
        delta: f64,
        edge1_len: f64,
        edge2_len: f64,
        new_edge1_len: f64,
        new_edge2_len: f64,
        tour_len: f64,
        n: usize,
    ) -> f64 {
        if delta < 0.0 {
            let avg_edge = tour_len / n as f64;

            // Normalize edge lengths by average
            let e1_norm = edge1_len / avg_edge;
            let e2_norm = edge2_len / avg_edge;
            let n1_norm = new_edge1_len / avg_edge;
            let n2_norm = new_edge2_len / avg_edge;

            // Prioritize: large improvement + removing long edges + adding short edges
            let improvement_term = -delta / avg_edge;
            let removal_bonus = (e1_norm + e2_norm - 2.0).max(0.0);
            let addition_bonus = (2.0 - n1_norm - n2_norm).max(0.0);

            improvement_term + 0.5 * removal_bonus + 0.3 * addition_bonus
        } else {
            f64::NEG_INFINITY
        }
    }

    fn name(&self) -> &'static str {
        "lk_inspired"
    }
}

/// Balanced: Balance between improvement and edge structure.
pub struct Balanced;

impl TwoOptPriority for Balanced {
    fn priority(
        &self,
        delta: f64,
        edge1_len: f64,
        edge2_len: f64,
        new_edge1_len: f64,
        new_edge2_len: f64,
        tour_len: f64,
        _n: usize,
    ) -> f64 {
        if delta < 0.0 {
            let max_removed = edge1_len.max(edge2_len);
            let min_added = new_edge1_len.min(new_edge2_len);

            // Combine improvement with edge quality
            let improvement_score = -delta / tour_len;
            let structure_score = (max_removed - min_added) / tour_len;

            improvement_score + 0.2 * structure_score
        } else {
            f64::NEG_INFINITY
        }
    }

    fn name(&self) -> &'static str {
        "balanced"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_prefers_improvements() {
        let g = GreedyDelta;
        let p1 = g.priority(-10.0, 5.0, 5.0, 2.0, 3.0, 100.0, 10);
        let p2 = g.priority(10.0, 5.0, 5.0, 7.0, 8.0, 100.0, 10);
        assert!(p1 > p2);
    }

    #[test]
    fn test_best_improvement_ignores_worsening() {
        let b = BestImprovement;
        let p = b.priority(5.0, 5.0, 5.0, 7.0, 8.0, 100.0, 10);
        assert!(p == f64::NEG_INFINITY);
    }
}
