use crate::{EvictionScorer, TokenInfo};

/// Gen6 Mutation: Optimized recency handling
/// Hypothesis: Recency window and weight need careful tuning
/// Key changes:
/// - Multiple recency tiers (immediate, recent, medium)
/// - Smoother decay function
/// - Layer-dependent recency importance

pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        if token.is_sink { return f64::MAX; }
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;

        // Tiered recency scoring
        let recency_score = if token.relative_pos < 16 {
            // Immediate: very high importance
            0.4 * (1.0 - token.relative_pos as f64 / 16.0)
        } else if token.relative_pos < 64 {
            // Recent: moderate importance
            0.2 * (1.0 - (token.relative_pos - 16) as f64 / 48.0)
        } else if token.relative_pos < 128 {
            // Medium: slight importance
            0.05 * (1.0 - (token.relative_pos - 64) as f64 / 64.0)
        } else {
            0.0
        };

        // Layer-dependent recency importance
        // Early layers: recency matters more
        let layer_recency_mult = 1.0 - 0.3 * layer_ratio;
        let adjusted_recency = recency_score * layer_recency_mult;

        // Attention scores
        let recent_weight = 0.55 - 0.15 * layer_ratio;
        let cumulative_weight = 0.25 + 0.15 * layer_ratio;

        let attn_score = recent_weight * token.recent_attn
            + cumulative_weight * token.cumulative_attn;

        let key_norm_penalty = 0.1 * token.key_norm.min(2.0);

        attn_score + adjusted_recency - key_norm_penalty
    }

    fn name(&self) -> &'static str {
        "gen6_recency_tuned"
    }
}
