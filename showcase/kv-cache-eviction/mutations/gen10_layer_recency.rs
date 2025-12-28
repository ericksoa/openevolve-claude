use crate::{EvictionScorer, TokenInfo};

/// Gen10: Layer-adaptive recency weight
/// Base: gen9_recency_35 (+2.65% over hybrid)
/// Hypothesis: Early layers may need different recency emphasis

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

        // Layer-adaptive recency: higher for early layers
        let recency_window = 128;
        let recency_weight = 0.38 - 0.06 * layer_ratio;  // 0.38 early, 0.32 late
        let recency_component = if token.relative_pos < recency_window {
            recency_weight * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        let position_factor = (token.position as f64 / token.sequence_len as f64).powf(0.2);
        let position_component = 0.14 * position_factor;

        let norm_component = -0.14 * (token.key_norm - 1.0).max(0.0).min(1.5);

        attn_component + recency_component + position_component + norm_component
    }

    fn name(&self) -> &'static str {
        "gen10_layer_recency"
    }
}
