use crate::{EvictionScorer, TokenInfo};

/// Gen6 Mutation: Extreme layer differentiation
/// Hypothesis: Different layers need very different strategies
/// Key changes:
/// - Much stronger layer-dependent weight shifts
/// - Early layers: almost pure recency
/// - Late layers: almost pure cumulative attention

pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        if token.is_sink { return f64::MAX; }
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;

        // Extreme layer-dependent weights
        // Early layers (0-10): 0.9 recent, 0.1 cumulative
        // Late layers (22-32): 0.3 recent, 0.7 cumulative
        let recent_weight = 0.9 - 0.6 * layer_ratio;
        let cumulative_weight = 0.1 + 0.6 * layer_ratio;

        // Position factor - only apply in later layers
        let position_factor = if layer_ratio > 0.5 {
            let power = 0.2 * (layer_ratio - 0.5);
            ((token.position as f64 + 1.0) / (token.sequence_len as f64)).powf(power)
        } else { 1.0 };

        // Layer-adaptive recency window
        let recency_window = (64.0 + 128.0 * layer_ratio) as usize;
        let recency_bonus = if token.relative_pos < recency_window {
            0.2 * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        let key_norm_penalty = 0.1 * token.key_norm.min(2.0);

        recent_weight * token.recent_attn
            + cumulative_weight * token.cumulative_attn * position_factor
            + recency_bonus
            - key_norm_penalty
    }

    fn name(&self) -> &'static str {
        "gen6_layer_extreme"
    }
}
