use crate::{EvictionScorer, TokenInfo};

/// Gen8: Optimized parameters around window 96
/// Base: gen7_window_96 (+1.84% over hybrid)
/// Hypothesis: Fine-tune all parameters together

pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        if token.is_sink { return f64::MAX; }
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;

        // Slightly boosted attention (42% vs 40%)
        let recent_weight = 0.26 - 0.05 * layer_ratio;
        let cumulative_weight = 0.16 + 0.05 * layer_ratio;
        let attn_component = recent_weight * token.recent_attn
            + cumulative_weight * token.cumulative_attn;

        // Window 96, slightly reduced weight (28% vs 30%)
        let recency_window = 96;
        let recency_component = if token.relative_pos < recency_window {
            0.28 * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        // Slightly stronger position (0.25 vs 0.2)
        let position_factor = (token.position as f64 / token.sequence_len as f64).powf(0.25);
        let position_component = 0.16 * position_factor;

        // Slightly reduced norm penalty
        let norm_component = -0.14 * (token.key_norm - 1.0).max(0.0).min(1.5);

        attn_component + recency_component + position_component + norm_component
    }

    fn name(&self) -> &'static str {
        "gen8_optimized"
    }
}
