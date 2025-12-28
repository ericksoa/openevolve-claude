use crate::{EvictionScorer, TokenInfo};

/// Gen6 Mutation: Balanced approach
/// Hypothesis: Current approaches over-weight some signals
/// Key changes:
/// - Equal-ish weights across all signals
/// - Moderate layer adaptation
/// - All factors contribute meaningfully

pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        if token.is_sink { return f64::MAX; }
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;

        // Component 1: Attention (40% of score)
        let recent_weight = 0.25 - 0.05 * layer_ratio;
        let cumulative_weight = 0.15 + 0.05 * layer_ratio;
        let attn_component = recent_weight * token.recent_attn
            + cumulative_weight * token.cumulative_attn;

        // Component 2: Recency position (30% of score)
        let recency_window = 80;
        let recency_component = if token.relative_pos < recency_window {
            0.3 * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        // Component 3: Position in sequence (15% of score)
        let position_factor = (token.position as f64 / token.sequence_len as f64).powf(0.2);
        let position_component = 0.15 * position_factor;

        // Component 4: Key norm penalty (15% impact)
        let norm_component = -0.15 * (token.key_norm - 1.0).max(0.0).min(1.5);

        attn_component + recency_component + position_component + norm_component
    }

    fn name(&self) -> &'static str {
        "gen6_balanced"
    }
}
