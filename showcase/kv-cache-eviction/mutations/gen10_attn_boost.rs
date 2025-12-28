use crate::{EvictionScorer, TokenInfo};

/// Gen10: Boost attention back while keeping recency high
/// Base: gen9_recency_35 (+2.65% over hybrid)
/// Hypothesis: Better balance between attention and recency

pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        if token.is_sink { return f64::MAX; }
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;

        // Boosted attention back (40%)
        let recent_weight = 0.25 - 0.05 * layer_ratio;
        let cumulative_weight = 0.15 + 0.05 * layer_ratio;
        let attn_component = recent_weight * token.recent_attn
            + cumulative_weight * token.cumulative_attn;

        // Keep high recency (35%) but reduce position/norm
        let recency_window = 128;
        let recency_component = if token.relative_pos < recency_window {
            0.35 * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        // Reduced position (12%)
        let position_factor = (token.position as f64 / token.sequence_len as f64).powf(0.2);
        let position_component = 0.12 * position_factor;

        // Reduced norm (13%)
        let norm_component = -0.13 * (token.key_norm - 1.0).max(0.0).min(1.5);

        attn_component + recency_component + position_component + norm_component
    }

    fn name(&self) -> &'static str {
        "gen10_attn_boost"
    }
}
