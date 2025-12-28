use crate::{EvictionScorer, TokenInfo};

pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        // Always keep sink tokens (attention sinks)
        if token.is_sink { return f64::MAX; }

        // Strongly prefer keeping first few tokens after sink
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        // Layer-adaptive weighting: early layers favor recent, later layers favor cumulative
        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;
        let recent_weight = 0.65 - 0.15 * layer_ratio;     // 0.65 -> 0.5
        let cumulative_weight = 0.35 + 0.15 * layer_ratio; // 0.35 -> 0.5

        // Position factor: slightly favor tokens closer to end of sequence
        let position_power = 0.25 + 0.1 * layer_ratio;
        let position_factor = ((token.position as f64 + 1.0) / (token.sequence_len as f64)).powf(position_power);

        // Recency bonus: extra weight for very recent tokens
        let recency_window = 128;
        let base_recency = 0.15 + 0.1 * layer_ratio;
        let recency_bonus = if token.relative_pos < recency_window {
            base_recency * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        // Penalize tokens with unusually high key norms (potential outliers)
        let key_norm_penalty = 0.1 * token.key_norm.min(2.0);

        // Combine all factors
        recent_weight * token.recent_attn
            + cumulative_weight * token.cumulative_attn * position_factor
            + recency_bonus
            - key_norm_penalty
    }

    fn name(&self) -> &'static str {
        "gen5_weight_65"
    }
}
