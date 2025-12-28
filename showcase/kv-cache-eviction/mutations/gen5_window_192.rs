// Gen5: Larger recency window = 192
// Hypothesis: A larger window gives more gradual recency decay, may help with longer-range dependencies

use crate::{EvictionScorer, TokenInfo};

pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        // Always keep sink tokens
        if token.is_sink { return f64::MAX; }

        // Strongly prefer keeping the first few tokens
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        // Layer-adaptive weighting
        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;
        let recent_weight = 0.7 - 0.2 * layer_ratio;
        let cumulative_weight = 0.3 + 0.2 * layer_ratio;

        // Position factor with layer-adaptive power
        let position_power = 0.25 + 0.1 * layer_ratio;
        let position_factor = ((token.position as f64 + 1.0) / (token.sequence_len as f64)).powf(position_power);

        // MUTATION: Larger recency window = 192 (was 128)
        let recency_window = 192;
        let base_recency = 0.15 + 0.1 * layer_ratio;
        let recency_bonus = if token.relative_pos < recency_window {
            base_recency * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        // Key norm penalty
        let key_norm_penalty = 0.1 * token.key_norm.min(2.0);

        // Final score
        recent_weight * token.recent_attn
            + cumulative_weight * token.cumulative_attn * position_factor
            + recency_bonus
            - key_norm_penalty
    }

    fn name(&self) -> &'static str {
        "gen5_window_192"
    }
}
