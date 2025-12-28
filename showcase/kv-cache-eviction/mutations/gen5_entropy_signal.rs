use crate::{EvictionScorer, TokenInfo};

pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        // Sink tokens must never be evicted
        if token.is_sink { return f64::MAX; }

        // Protect very recent tokens (attention sink region)
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        // Layer-dependent weighting: later layers rely more on cumulative patterns
        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;
        let recent_weight = 0.7 - 0.2 * layer_ratio;
        let cumulative_weight = 0.3 + 0.2 * layer_ratio;

        // Position factor: slight preference for tokens further in sequence
        let position_power = 0.25 + 0.1 * layer_ratio;
        let position_factor = ((token.position as f64 + 1.0) / (token.sequence_len as f64)).powf(position_power);

        // Recency bonus for tokens within a window
        let recency_window = 128;
        let base_recency = 0.15 + 0.1 * layer_ratio;
        let recency_bonus = if token.relative_pos < recency_window {
            base_recency * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        // Key norm penalty: very large keys can destabilize attention
        let key_norm_penalty = 0.1 * token.key_norm.min(2.0);

        // NEW: Entropy signal - higher entropy means attention is spread across many queries
        // This indicates the token is consistently useful across different contexts
        // Use a small weight to avoid disrupting the champion's balance
        // Normalize entropy (typically 0-4 range for attention) and apply small bonus
        let entropy_bonus = 0.03 * (token.attn_entropy / 4.0).min(1.0);

        recent_weight * token.recent_attn
            + cumulative_weight * token.cumulative_attn * position_factor
            + recency_bonus
            + entropy_bonus
            - key_norm_penalty
    }

    fn name(&self) -> &'static str {
        "gen5_entropy_signal"
    }
}
