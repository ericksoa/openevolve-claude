use crate::{EvictionScorer, TokenInfo};

/// Gen8: Crossover of window_96 + layer_adaptive
/// Base: gen7_window_96 (+1.84%) + gen7_layer_adaptive (+1.43%)
/// Hypothesis: Combine larger window with layer-adaptive components

pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        if token.is_sink { return f64::MAX; }
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;

        // Layer-adaptive attention (same)
        let recent_weight = 0.25 - 0.05 * layer_ratio;
        let cumulative_weight = 0.15 + 0.05 * layer_ratio;
        let attn_component = recent_weight * token.recent_attn
            + cumulative_weight * token.cumulative_attn;

        // From window_96 but with layer-adaptive weight
        let recency_window = 96;
        let recency_weight = 0.35 - 0.10 * layer_ratio;  // 0.35 early, 0.25 late
        let recency_component = if token.relative_pos < recency_window {
            recency_weight * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        // Layer-adaptive position
        let position_power = 0.15 + 0.10 * layer_ratio;
        let position_factor = (token.position as f64 / token.sequence_len as f64).powf(position_power);
        let position_component = 0.15 * position_factor;

        // Layer-adaptive norm
        let norm_weight = 0.12 + 0.06 * layer_ratio;
        let norm_component = -norm_weight * (token.key_norm - 1.0).max(0.0).min(1.5);

        attn_component + recency_component + position_component + norm_component
    }

    fn name(&self) -> &'static str {
        "gen8_cross_layer"
    }
}
