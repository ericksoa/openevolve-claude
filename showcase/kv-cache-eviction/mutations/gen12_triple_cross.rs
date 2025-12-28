use crate::{EvictionScorer, TokenInfo};

/// Gen12: Triple cross - window 140 + position 0.35 + balanced_cross elements
/// Base: gen11_window_140 (+4.46%) + gen11_position_35 + gen11_balanced_cross
/// Hypothesis: Combine all best Gen11 elements

pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        if token.is_sink { return f64::MAX; }
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;

        // Layer-adaptive attention from balanced_cross
        let recent_weight = 0.24 - 0.06 * layer_ratio;
        let cumulative_weight = 0.13 + 0.06 * layer_ratio;
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
        "gen12_triple_cross"
    }
}
