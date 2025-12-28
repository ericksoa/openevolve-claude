use crate::{EvictionScorer, TokenInfo};

/// Gen8: Triple crossover of top 3 gen7 mutations
/// Base: window_96 + position_strong + layer_adaptive
/// Hypothesis: Combine all winning elements

pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        if token.is_sink { return f64::MAX; }
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;

        // Layer-adaptive attention
        let recent_weight = 0.25 - 0.05 * layer_ratio;
        let cumulative_weight = 0.15 + 0.05 * layer_ratio;
        let attn_component = recent_weight * token.recent_attn
            + cumulative_weight * token.cumulative_attn;

        // Window 96 with layer-adaptive weight
        let recency_window = 96;
        let recency_weight = 0.32 - 0.05 * layer_ratio;  // 0.32 early, 0.27 late
        let recency_component = if token.relative_pos < recency_window {
            recency_weight * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        // Strong position (0.3 power) with layer adaptation
        let position_power = 0.25 + 0.10 * layer_ratio;  // 0.25 early, 0.35 late
        let position_factor = (token.position as f64 / token.sequence_len as f64).powf(position_power);
        let position_component = 0.15 * position_factor;

        let norm_component = -0.15 * (token.key_norm - 1.0).max(0.0).min(1.5);

        attn_component + recency_component + position_component + norm_component
    }

    fn name(&self) -> &'static str {
        "gen8_triple_cross"
    }
}
