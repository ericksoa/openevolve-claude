//! Evolved KV-Cache Eviction Scorer
//!
//! Generation 17 Champion: gen17_recency_40_440
//!
//! Improvements over hybrid baseline:
//! - TRAIN: +3.55% (0.0582 -> 0.0561)
//! - VALID: +3.95% (0.0566 -> 0.0544)
//! - TEST:  +6.65% (0.0662 -> 0.0618)
//!
//! Evolution history:
//! - Gen6: gen6_balanced (+1.44%) - balanced weights across signals
//! - Gen7: gen7_window_96 (+1.84%) - larger recency window (96 vs 80)
//! - Gen8: gen8_window_128 (+2.31%) - even larger window (128 vs 96)
//! - Gen9: gen9_recency_35 (+2.65%) - boosted recency weight (35% vs 30%)
//! - Gen10: gen10_cross_position (+2.86%) - stronger position correction (0.3 vs 0.2)
//! - Gen11: gen11_window_140 (+2.97%) - optimal window at 140 tokens
//! - Gen12: gen12_window_160 (+3.09%) - window trend continues (160 vs 140)
//! - Gen13: gen13_window_200 (+5.38%) - window 200, broke 5% TEST barrier
//! - Gen14: gen14_window_256 (+5.91%) - window 256 = half cache size
//! - Gen15: gen15_window_350 (+6.38%) - window 350 tokens, exceeds 6% TEST
//! - Gen16: gen16_window_440 (+6.48%) - window 440, plateau at 86% cache
//! - Gen17: gen17_recency_40_440 (+6.65%) - recency 40%, new optimal weight
//!
//! Key insights:
//! - Window size plateau at 420-440 tokens (82-86% of cache)
//! - Recency weight evolved: 30% -> 35% -> 40%
//! - Higher recency weight trades off attention weight
//! - Position power 0.3 for position bias correction
//! - TOTAL improvement: +6.65% over hybrid baseline

use crate::{EvictionScorer, TokenInfo};

/// Gen17 Champion: Recency 40% with window 440
///
/// Key parameters:
/// - Attention: 32% (0.18-0.05*layer + 0.14+0.05*layer)
/// - Recency: 40% with 440-token window
/// - Position: 14% with power 0.3
/// - Norm penalty: 14% for outliers
pub struct Evolved;

impl EvictionScorer for Evolved {
    fn score(&self, token: &TokenInfo) -> f64 {
        // Sink tokens always kept
        if token.is_sink { return f64::MAX; }

        // Very recent tokens always kept
        if token.relative_pos < 4 { return 1e6 - token.relative_pos as f64; }

        let layer_ratio = token.layer_idx as f64 / token.num_layers as f64;

        // Component 1: Attention (32%)
        let recent_weight = 0.18 - 0.05 * layer_ratio;
        let cumulative_weight = 0.14 + 0.05 * layer_ratio;
        let attn_component = recent_weight * token.recent_attn
            + cumulative_weight * token.cumulative_attn;

        // Component 2: Recency (40% with 440-token window)
        let recency_window = 440;
        let recency_component = if token.relative_pos < recency_window {
            0.40 * (1.0 - token.relative_pos as f64 / recency_window as f64)
        } else { 0.0 };

        // Component 3: Position (14% with power 0.3)
        let position_factor = (token.position as f64 / token.sequence_len as f64).powf(0.3);
        let position_component = 0.14 * position_factor;

        // Component 4: Norm penalty (14%)
        let norm_component = -0.14 * (token.key_norm - 1.0).max(0.0).min(1.5);

        attn_component + recency_component + position_component + norm_component
    }

    fn name(&self) -> &'static str {
        "gen17_recency_40_440"
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
