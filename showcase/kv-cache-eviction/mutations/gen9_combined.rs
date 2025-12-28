use crate::{EvictionScorer, TokenInfo};

/// Gen9: Combined best ideas
/// Base: gen8_window_128 + cross_position + layer_adaptive elements
/// Hypothesis: Synergy of multiple improvements

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

        // Large window 128 with slight layer adaptation on weight
        let recency_window = 128;
        let recency_weight = 0.32 - 0.04 * layer_ratio;  // 0.32 early, 0.28 late
        let recency_component = if token.relative_pos < recency_window {
            recency_weight * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        // Slightly stronger position with layer adaptation
        let position_power = 0.22 + 0.06 * layer_ratio;  // 0.22 early, 0.28 late
        let position_factor = (token.position as f64 / token.sequence_len as f64).powf(position_power);
        let position_component = 0.15 * position_factor;

        let norm_component = -0.15 * (token.key_norm - 1.0).max(0.0).min(1.5);

        attn_component + recency_component + position_component + norm_component
    }

    fn name(&self) -> &'static str {
        "gen9_combined"
    }
}
