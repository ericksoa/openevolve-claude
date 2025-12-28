use crate::{EvictionScorer, TokenInfo};

pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        // Always keep sink tokens (attention sinks)
        if token.is_sink { return f64::MAX; }

        // Protect very recent tokens with high scores
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        // Layer-dependent weighting: more recent emphasis in early layers
        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;
        let recent_weight = 0.75 - 0.25 * layer_ratio;  // 0.75 -> 0.5
        let cumulative_weight = 0.25 + 0.25 * layer_ratio;  // 0.25 -> 0.5

        // Position factor: slightly boost older tokens that have accumulated attention
        let position_power = 0.25 + 0.1 * layer_ratio;
        let position_factor = ((token.position as f64 + 1.0) / (token.sequence_len as f64)).powf(position_power);

        // Recency bonus for tokens within the sliding window
        let recency_window = 128;
        let base_recency = 0.15 + 0.1 * layer_ratio;
        let recency_bonus = if token.relative_pos < recency_window {
            base_recency * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        // Penalize tokens with high key norms (outliers)
        let key_norm_penalty = 0.1 * token.key_norm.min(2.0);

        // Final score: combine all factors
        recent_weight * token.recent_attn
            + cumulative_weight * token.cumulative_attn * position_factor
            + recency_bonus
            - key_norm_penalty
    }

    fn name(&self) -> &'static str {
        "gen5_weight_75"
    }
}
