use crate::{EvictionScorer, TokenInfo};

/// Gen7: More layer-adaptive components
/// Base: gen6_balanced (+1.44% over hybrid)
/// Hypothesis: All components should adapt to layer depth

pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        if token.is_sink { return f64::MAX; }
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;

        // Layer-adaptive attention (same as gen6)
        let recent_weight = 0.25 - 0.05 * layer_ratio;
        let cumulative_weight = 0.15 + 0.05 * layer_ratio;
        let attn_component = recent_weight * token.recent_attn
            + cumulative_weight * token.cumulative_attn;

        // Layer-adaptive recency: early layers need more recency
        let recency_window = 80;
        let recency_weight = 0.35 - 0.10 * layer_ratio;  // 0.35 early, 0.25 late
        let recency_component = if token.relative_pos < recency_window {
            recency_weight * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        // Layer-adaptive position: late layers need more position correction
        let position_power = 0.15 + 0.10 * layer_ratio;  // 0.15 early, 0.25 late
        let position_factor = (token.position as f64 / token.sequence_len as f64).powf(position_power);
        let position_component = 0.15 * position_factor;

        // Layer-adaptive norm: late layers more sensitive to outliers
        let norm_weight = 0.12 + 0.06 * layer_ratio;  // 0.12 early, 0.18 late
        let norm_component = -norm_weight * (token.key_norm - 1.0).max(0.0).min(1.5);

        attn_component + recency_component + position_component + norm_component
    }

    fn name(&self) -> &'static str {
        "gen7_layer_adaptive"
    }
}
