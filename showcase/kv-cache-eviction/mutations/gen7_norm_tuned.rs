use crate::{EvictionScorer, TokenInfo};

/// Gen7: Tuned norm penalty
/// Base: gen6_balanced (+1.44% over hybrid)
/// Hypothesis: Different norm threshold and weight

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

        let recency_window = 80;
        let recency_component = if token.relative_pos < recency_window {
            0.30 * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        let position_factor = (token.position as f64 / token.sequence_len as f64).powf(0.2);
        let position_component = 0.15 * position_factor;

        // Tuned norm: lower threshold (0.8 vs 1.0), softer penalty
        // Penalize tokens slightly above average norm
        let norm_component = -0.12 * (token.key_norm - 0.8).max(0.0).min(2.0);

        attn_component + recency_component + position_component + norm_component
    }

    fn name(&self) -> &'static str {
        "gen7_norm_tuned"
    }
}
