use crate::{EvictionScorer, TokenInfo};

pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        // Sink tokens must never be evicted
        if token.is_sink { return f64::MAX; }

        // Protect initial tokens (attention sink pattern)
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;

        // Layer-adaptive attention weights
        let recent_weight = 0.7 - 0.2 * layer_ratio;
        let cumulative_weight = 0.3 + 0.2 * layer_ratio;

        // Position factor with layer-adaptive power
        let position_power = 0.25 + 0.1 * layer_ratio;
        let position_factor = ((token.position as f64 + 1.0) / (token.sequence_len as f64)).powf(position_power);

        // Recency bonus for recently accessed tokens
        let recency_window = 128;
        let base_recency = 0.15 + 0.1 * layer_ratio;
        let recency_bonus = if token.relative_pos < recency_window {
            base_recency * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        // Key norm penalty
        let key_norm_penalty = 0.1 * token.key_norm.min(2.0);

        // Token type awareness
        // token_type: 0=regular, 1=special (BOS, EOS, separator), 2=punctuation
        let token_type_bonus = match token.token_type {
            1 => {
                // Special tokens (BOS, EOS, separators) are structurally important
                // Give them a significant bonus, especially in early/middle layers
                // where structural information is more critical
                let layer_decay = 1.0 - 0.3 * layer_ratio;
                0.25 * layer_decay
            },
            2 => {
                // Punctuation serves as sentence boundaries and phrase delimiters
                // Moderate importance - helps with structural understanding
                // More important in earlier layers, less in later semantic layers
                let layer_decay = 1.0 - 0.4 * layer_ratio;
                0.12 * layer_decay
            },
            _ => 0.0,  // Regular tokens get no bonus
        };

        recent_weight * token.recent_attn
            + cumulative_weight * token.cumulative_attn * position_factor
            + recency_bonus
            + token_type_bonus
            - key_norm_penalty
    }

    fn name(&self) -> &'static str {
        "gen5_token_type"
    }
}
