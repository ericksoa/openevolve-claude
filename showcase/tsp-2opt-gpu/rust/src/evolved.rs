use crate::TwoOptPriority;

pub struct Evolved;

impl TwoOptPriority for Evolved {
    fn priority(&self, delta: f64, edge1_len: f64, edge2_len: f64, new_edge1_len: f64, new_edge2_len: f64, _tour_len: f64, _n: usize) -> f64 {
        if delta < 0.0 {
            let removed_sum = edge1_len + edge2_len;
            let added_sum = new_edge1_len + new_edge2_len;
            let edge_ratio = removed_sum / (added_sum + 1e-10);
            let max_removed = edge1_len.max(edge2_len);
            let min_added = new_edge1_len.min(new_edge2_len);
            let max_edge_bonus = max_removed - min_added;
            -delta * edge_ratio + 0.3 * max_edge_bonus
        } else {
            f64::NEG_INFINITY
        }
    }
    fn name(&self) -> &'static str { "evolved" }
}
