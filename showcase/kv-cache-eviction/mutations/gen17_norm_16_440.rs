use crate::{EvictionScorer, TokenInfo};

/// Gen17: Stronger norm penalty 16% with window 440
/// Base: gen16_window_440 (+6.48% TEST)
/// Hypothesis: Increase norm penalty weight

pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        if token.is_sink { return f64::MAX; }
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;

        // Reduced attention for stronger norm
        let recent_weight = 0.22 - 0.05 * layer_ratio;
        let cumulative_weight = 0.13 + 0.05 * layer_ratio;
        let attn_component = recent_weight * token.recent_attn
            + cumulative_weight * token.cumulative_attn;

        let recency_window = 440;
        let recency_component = if token.relative_pos < recency_window {
            0.35 * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        let position_factor = (token.position as f64 / token.sequence_len as f64).powf(0.3);
        let position_component = 0.14 * position_factor;

        // Stronger norm penalty (16% vs 14%)
        let norm_component = -0.16 * (token.key_norm - 1.0).max(0.0).min(1.5);

        attn_component + recency_component + position_component + norm_component
    }

    fn name(&self) -> &'static str {
        "gen17_norm_16_440"
    }
}
