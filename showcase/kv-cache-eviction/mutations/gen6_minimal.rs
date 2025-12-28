use crate::{EvictionScorer, TokenInfo};

/// Gen6 Mutation: Minimal complexity
/// Hypothesis: Simpler is better - hybrid_baseline is simple and it's winning
/// Key changes:
/// - Remove layer dependency entirely
/// - Just use core signals: attention + recency + norm
/// - Match hybrid_baseline structure exactly, then tweak one thing

pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        if token.is_sink { return f64::MAX; }

        // Hybrid uses relative_pos < recent_window implicitly
        // We use explicit protection for very recent
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        // Exactly hybrid's attention formula
        let attn_score = 0.3 * token.cumulative_attn + 0.7 * token.recent_attn;

        // Exactly hybrid's recency (64 window, 0.3 weight)
        let recent_window = 64;
        let recency_score = if token.relative_pos < recent_window {
            1.0 - (token.relative_pos as f64 / recent_window as f64)
        } else { 0.0 };

        // Exactly hybrid's norm penalty
        let norm_penalty = 0.1 * token.key_norm;

        // Hybrid formula: 0.4 * attn + 0.3 * recency + norm
        // Tweak: slightly favor recent attention more
        0.42 * attn_score + 0.32 * recency_score - norm_penalty
    }

    fn name(&self) -> &'static str {
        "gen6_minimal"
    }
}
