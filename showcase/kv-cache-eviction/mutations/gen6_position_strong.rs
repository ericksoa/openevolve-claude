use crate::{EvictionScorer, TokenInfo};

/// Gen6 Mutation: Stronger position correction
/// Hypothesis: Position bias is underweighted in current approaches
/// Key changes:
/// - Stronger position correction factor
/// - Position-aware attention weighting
/// - Compensate for early token advantage

pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        if token.is_sink { return f64::MAX; }
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;

        // Position-based boost for later tokens
        // Tokens at later positions had less time to accumulate attention
        let position_ratio = token.position as f64 / token.sequence_len as f64;
        let position_boost = 1.0 + 0.5 * position_ratio;  // 1.0 to 1.5 boost

        // Standard layer-adaptive weights
        let recent_weight = 0.65 - 0.2 * layer_ratio;
        let cumulative_weight = 0.35 + 0.2 * layer_ratio;

        // Apply position boost to cumulative (which has position bias)
        let attn_score = recent_weight * token.recent_attn
            + cumulative_weight * token.cumulative_attn * position_boost;

        // Recency bonus
        let recency_window = 96;
        let recency_bonus = if token.relative_pos < recency_window {
            0.2 * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        let key_norm_penalty = 0.1 * token.key_norm.min(2.0);

        attn_score + recency_bonus - key_norm_penalty
    }

    fn name(&self) -> &'static str {
        "gen6_position_strong"
    }
}
