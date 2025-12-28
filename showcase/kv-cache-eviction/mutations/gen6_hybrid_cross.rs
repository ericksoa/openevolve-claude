use crate::{EvictionScorer, TokenInfo};

/// Gen6 Mutation: Crossover of hybrid_baseline and gen4_champion
/// Hypothesis: Combine the best elements from both approaches
/// Key changes:
/// - Hybrid's attention blending (0.3 cum + 0.7 rec)
/// - Gen4's layer-aware position correction
/// - Gen4's key norm handling with hybrid's simpler recency

pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        if token.is_sink { return f64::MAX; }
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;

        // Hybrid's attention blend
        let attn_score = 0.3 * token.cumulative_attn + 0.7 * token.recent_attn;

        // Gen4's layer-adaptive position correction
        let position_power = 0.25 + 0.1 * layer_ratio;
        let position_factor = ((token.position as f64 + 1.0) / (token.sequence_len as f64)).powf(position_power);

        // Hybrid's recency window with gen4's layer adaptation
        let recency_window = 64;
        let recency_weight = 0.3 * (1.0 + 0.2 * layer_ratio);
        let recency_score = if token.relative_pos < recency_window {
            recency_weight * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        // Hybrid's norm penalty
        let norm_penalty = 0.1 * token.key_norm;

        0.4 * attn_score * position_factor + recency_score - norm_penalty
    }

    fn name(&self) -> &'static str {
        "gen6_hybrid_cross"
    }
}
