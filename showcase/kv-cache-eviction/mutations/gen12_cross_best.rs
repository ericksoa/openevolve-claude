use crate::{EvictionScorer, TokenInfo};

/// Gen12: Cross window 140 with position 0.35
/// Base: gen11_window_140 (+4.46%) + gen11_position_35 (+4.40%)
/// Hypothesis: Combine the two best Gen11 elements

pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        if token.is_sink { return f64::MAX; }
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;

        let recent_weight = 0.23 - 0.05 * layer_ratio;
        let cumulative_weight = 0.14 + 0.05 * layer_ratio;
        let attn_component = recent_weight * token.recent_attn
            + cumulative_weight * token.cumulative_attn;

        // Window 140 from champion
        let recency_window = 140;
        let recency_component = if token.relative_pos < recency_window {
            0.35 * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        // Position 0.35 from runner-up
        let position_factor = (token.position as f64 / token.sequence_len as f64).powf(0.35);
        let position_component = 0.14 * position_factor;

        let norm_component = -0.14 * (token.key_norm - 1.0).max(0.0).min(1.5);

        attn_component + recency_component + position_component + norm_component
    }

    fn name(&self) -> &'static str {
        "gen12_cross_best"
    }
}
