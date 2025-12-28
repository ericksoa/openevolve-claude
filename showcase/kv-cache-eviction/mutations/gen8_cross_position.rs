use crate::{EvictionScorer, TokenInfo};

/// Gen8: Crossover of window_96 + position_strong
/// Base: gen7_window_96 (+1.84%) + gen7_position_strong (+1.64%)
/// Hypothesis: Combine the two best gen7 mutations

pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        if token.is_sink { return f64::MAX; }
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;

        let recent_weight = 0.25 - 0.05 * layer_ratio;
        let cumulative_weight = 0.15 + 0.05 * layer_ratio;
        let attn_component = recent_weight * token.recent_attn
            + cumulative_weight * token.cumulative_attn;

        // From window_96
        let recency_window = 96;
        let recency_component = if token.relative_pos < recency_window {
            0.30 * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        // From position_strong: power 0.3 vs 0.2
        let position_factor = (token.position as f64 / token.sequence_len as f64).powf(0.3);
        let position_component = 0.15 * position_factor;

        let norm_component = -0.15 * (token.key_norm - 1.0).max(0.0).min(1.5);

        attn_component + recency_component + position_component + norm_component
    }

    fn name(&self) -> &'static str {
        "gen8_cross_position"
    }
}
