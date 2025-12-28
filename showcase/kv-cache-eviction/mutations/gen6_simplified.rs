use crate::{EvictionScorer, TokenInfo};

/// Gen6 Mutation: Simplified approach
/// Hypothesis: Reduce complexity to match what works in hybrid_baseline
/// Key changes:
/// - Simpler weight scheme (less layer dependency)
/// - Smaller recency window matching hybrid baseline
/// - Remove position factor (not in hybrid baseline)

pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        if token.is_sink { return f64::MAX; }
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        // Simpler layer adaptation (milder)
        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;

        // Attention score: blend recent and cumulative (closer to hybrid)
        let attn_score = 0.6 * token.recent_attn + 0.4 * token.cumulative_attn;

        // Simpler recency bonus with smaller window (64 like hybrid)
        let recency_window = 64;
        let recency_score = if token.relative_pos < recency_window {
            0.35 * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        // Key norm penalty (same as hybrid)
        let norm_penalty = 0.1 * token.key_norm;

        0.45 * attn_score + recency_score - norm_penalty
    }

    fn name(&self) -> &'static str {
        "gen6_simplified"
    }
}
