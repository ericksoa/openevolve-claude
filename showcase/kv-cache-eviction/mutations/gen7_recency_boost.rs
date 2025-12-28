use crate::{EvictionScorer, TokenInfo};

/// Gen7: Boost recency component
/// Base: gen6_balanced (+1.44% over hybrid)
/// Hypothesis: Recency deserves more weight (35% vs 30%)

pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        if token.is_sink { return f64::MAX; }
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;

        // Same attention (40%)
        let recent_weight = 0.25 - 0.05 * layer_ratio;
        let cumulative_weight = 0.15 + 0.05 * layer_ratio;
        let attn_component = recent_weight * token.recent_attn
            + cumulative_weight * token.cumulative_attn;

        // Boosted recency (35% vs 30%)
        let recency_window = 80;
        let recency_component = if token.relative_pos < recency_window {
            0.35 * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        // Reduced position (12% vs 15%)
        let position_factor = (token.position as f64 / token.sequence_len as f64).powf(0.2);
        let position_component = 0.12 * position_factor;

        // Reduced norm penalty (13% vs 15%)
        let norm_component = -0.13 * (token.key_norm - 1.0).max(0.0).min(1.5);

        attn_component + recency_component + position_component + norm_component
    }

    fn name(&self) -> &'static str {
        "gen7_recency_boost"
    }
}
