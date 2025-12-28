//! Evolved KV-Cache Eviction Scorer
//!
//! Generation 11 Champion: gen11_window_140
//!
//! Improvements over hybrid baseline:
//! - TRAIN: +2.97% (0.0582 -> 0.0565)
//! - VALID: +3.25% (0.0566 -> 0.0547)
//! - TEST:  +4.46% (0.0662 -> 0.0633)
//!
//! Evolution history:
//! - Gen6: gen6_balanced (+1.44%) - balanced weights across signals
//! - Gen7: gen7_window_96 (+1.84%) - larger recency window (96 vs 80)
//! - Gen8: gen8_window_128 (+2.31%) - even larger window (128 vs 96)
//! - Gen9: gen9_recency_35 (+2.65%) - boosted recency weight (35% vs 30%)
//! - Gen10: gen10_cross_position (+2.86%) - stronger position correction (0.3 vs 0.2)
//! - Gen11: gen11_window_140 (+2.97%) - optimal window at 140 tokens
//!
//! Key insights:
//! - Window size optimal at 140 (128 was close, but 140 is better)
//! - Recency weight 35% with reduced other weights
//! - Position power 0.3 for stronger position bias correction

use crate::{EvictionScorer, TokenInfo};

/// Gen11 Champion: Recency window 140 + Strong position correction
///
/// Key parameters:
/// - Attention: 37% (0.23-0.05*layer + 0.14+0.05*layer)
/// - Recency: 35% with 140-token window
/// - Position: 14% with power 0.3 (stronger correction)
/// - Norm penalty: 14% for outliers
pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        // Sink tokens always kept
        if token.is_sink { return f64::MAX; }

        // Very recent tokens always kept
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;

        // Component 1: Attention (37%)
        let recent_weight = 0.23 - 0.05 * layer_ratio;
        let cumulative_weight = 0.14 + 0.05 * layer_ratio;
        let attn_component = recent_weight * token.recent_attn
            + cumulative_weight * token.cumulative_attn;

        // Component 2: Recency (35% with 140-token window)
        let recency_window = 140;
        let recency_component = if token.relative_pos < recency_window {
            0.35 * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        // Component 3: Position (14% with power 0.3)
        let position_factor = (token.position as f64 / token.sequence_len as f64).powf(0.3);
        let position_component = 0.14 * position_factor;

        // Component 4: Norm penalty (14%)
        let norm_component = -0.14 * (token.key_norm - 1.0).max(0.0).min(1.5);

        attn_component + recency_component + position_component + norm_component
    }

    fn name(&self) -> &'static str {
        "gen11_window_140"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sink_priority() {
        let scorer = Evolved;
        let mut sink = TokenInfo::new(2, 100);
        sink.is_sink = true;
        assert_eq!(scorer.score(&sink), f64::MAX);
    }

    #[test]
    fn test_attention_sensitivity() {
        let scorer = Evolved;

        let mut high_attn = TokenInfo::new(50, 200);
        high_attn.recent_attn = 5.0;
        high_attn.cumulative_attn = 10.0;

        let mut low_attn = TokenInfo::new(50, 200);
        low_attn.recent_attn = 0.5;
        low_attn.cumulative_attn = 1.0;

        assert!(scorer.score(&high_attn) > scorer.score(&low_attn));
    }

    #[test]
    fn test_recency_matters() {
        let scorer = Evolved;

        let recent = TokenInfo::new(95, 100);  // relative_pos = 5
        let old = TokenInfo::new(20, 100);     // relative_pos = 80

        assert!(scorer.score(&recent) > scorer.score(&old));
    }
}
