use crate::{EvictionScorer, TokenInfo};

/// Gen6 Mutation: Heavy attention focus
/// Hypothesis: Attention signals are most predictive
/// Key changes:
/// - Higher weight on attention scores
/// - Use both recent and cumulative with dynamic blending
/// - Reduce influence of position/recency heuristics

pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        if token.is_sink { return f64::MAX; }
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;

        // Heavy attention weights (total 0.8 vs 0.4 in hybrid)
        let recent_weight = 0.5 - 0.15 * layer_ratio;
        let cumulative_weight = 0.3 + 0.15 * layer_ratio;

        // Normalize attention scores to prevent outliers from dominating
        let recent_normalized = token.recent_attn.min(5.0);
        let cumulative_normalized = token.cumulative_attn.min(10.0);

        // Minimal recency bonus (attention should handle this)
        let recency_bonus = if token.relative_pos < 32 {
            0.1 * (1.0 - token.relative_pos as f64 / 32.0)
        } else { 0.0 };

        // Stronger key norm penalty for outliers
        let key_norm_penalty = 0.15 * token.key_norm.min(2.5);

        recent_weight * recent_normalized
            + cumulative_weight * cumulative_normalized
            + recency_bonus
            - key_norm_penalty
    }

    fn name(&self) -> &'static str {
        "gen6_attention_focus"
    }
}
