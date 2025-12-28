use crate::{EvictionScorer, TokenInfo};

/// Gen11: Balanced cross of best Gen10 elements
/// Base: gen10_cross_position (+2.86% TRAIN) + gen10_combined insights
/// Hypothesis: Combining layer-adaptive recency with strong position

pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        if token.is_sink { return f64::MAX; }
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;

        // Layer-adaptive attention (from gen10_combined)
        let recent_weight = 0.24 - 0.06 * layer_ratio;
        let cumulative_weight = 0.13 + 0.06 * layer_ratio;
        let attn_component = recent_weight * token.recent_attn
            + cumulative_weight * token.cumulative_attn;

        // Layer-adaptive recency (36% avg, from gen10_combined)
        let recency_window = 128;
        let recency_weight = 0.37 - 0.04 * layer_ratio;  // 0.37 early, 0.33 late
        let recency_component = if token.relative_pos < recency_window {
            recency_weight * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        // Strong position from champion (0.3)
        let position_factor = (token.position as f64 / token.sequence_len as f64).powf(0.3);
        let position_component = 0.13 * position_factor;

        let norm_component = -0.13 * (token.key_norm - 1.0).max(0.0).min(1.5);

        attn_component + recency_component + position_component + norm_component
    }

    fn name(&self) -> &'static str {
        "gen11_balanced_cross"
    }
}
