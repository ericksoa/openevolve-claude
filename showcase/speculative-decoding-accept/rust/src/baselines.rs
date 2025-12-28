//! Baseline acceptance heuristics for speculative decoding
//!
//! These represent known algorithms to beat:
//! 1. Standard rejection sampling (theoretically optimal for lossless)
//! 2. Always accept (fast but low quality)
//! 3. Conservative (high quality but slow)
//! 4. Entropy-aware (adapts to uncertainty)

use crate::AcceptanceHeuristic;

/// Standard rejection sampling - the baseline to beat
///
/// This is theoretically optimal for lossless speculative decoding.
/// Accept threshold = min(1, p_target / p_draft)
///
/// Reference: "Accelerating Large Language Model Decoding with Speculative Sampling"
/// https://arxiv.org/abs/2302.01318
pub struct StandardRejectionSampling;

impl AcceptanceHeuristic for StandardRejectionSampling {
    fn acceptance_threshold(
        &self,
        draft_prob: f64,
        target_prob: f64,
        _position: usize,
        _draft_entropy: f64,
        _target_entropy: f64,
        _top_token_match: bool,
    ) -> f64 {
        // Standard rejection sampling: accept with probability min(1, p_target/p_draft)
        (target_prob / draft_prob).min(1.0)
    }
}

/// Always accept - maximum speed, lowest quality
///
/// This is the upper bound on acceptance rate but completely ignores
/// the target model's preferences. Useful as a speed benchmark.
pub struct AlwaysAccept;

impl AcceptanceHeuristic for AlwaysAccept {
    fn acceptance_threshold(
        &self,
        _draft_prob: f64,
        _target_prob: f64,
        _position: usize,
        _draft_entropy: f64,
        _target_entropy: f64,
        _top_token_match: bool,
    ) -> f64 {
        1.0 // Always accept
    }
}

/// Conservative - only accept when very confident
///
/// Requires target_prob >= draft_prob (target must "agree" with draft).
/// High quality but lower acceptance rate.
pub struct Conservative;

impl AcceptanceHeuristic for Conservative {
    fn acceptance_threshold(
        &self,
        draft_prob: f64,
        target_prob: f64,
        _position: usize,
        _draft_entropy: f64,
        _target_entropy: f64,
        _top_token_match: bool,
    ) -> f64 {
        // Only accept if target probability is at least as high as draft
        if target_prob >= draft_prob {
            1.0
        } else {
            0.0
        }
    }
}

/// Top-token match heuristic
///
/// Accept with high probability if both models agree on the top token,
/// use standard rejection sampling otherwise.
pub struct TopTokenMatch;

impl AcceptanceHeuristic for TopTokenMatch {
    fn acceptance_threshold(
        &self,
        draft_prob: f64,
        target_prob: f64,
        _position: usize,
        _draft_entropy: f64,
        _target_entropy: f64,
        top_token_match: bool,
    ) -> f64 {
        if top_token_match {
            // Models agree on top token - accept with high probability
            0.95
        } else {
            // Fall back to standard rejection sampling
            (target_prob / draft_prob).min(1.0)
        }
    }
}

/// Entropy-aware heuristic
///
/// More aggressive acceptance when both models are confident (low entropy),
/// more conservative when uncertain (high entropy).
pub struct EntropyAware;

impl AcceptanceHeuristic for EntropyAware {
    fn acceptance_threshold(
        &self,
        draft_prob: f64,
        target_prob: f64,
        _position: usize,
        draft_entropy: f64,
        target_entropy: f64,
        _top_token_match: bool,
    ) -> f64 {
        let base_threshold = (target_prob / draft_prob).min(1.0);

        // Average entropy (lower = more confident)
        let avg_entropy = (draft_entropy + target_entropy) / 2.0;

        // Entropy typically ranges from 0 to ~10 for vocab size 32k
        // Normalize to [0, 1] range assuming max entropy of 10
        let normalized_entropy = (avg_entropy / 10.0).min(1.0);

        // High confidence (low entropy): boost acceptance
        // Low confidence (high entropy): be more conservative
        let confidence_factor = 1.0 - normalized_entropy * 0.5;

        (base_threshold * confidence_factor).min(1.0)
    }
}

/// Position-aware heuristic
///
/// More conservative at the start of sequences (errors compound),
/// more aggressive later (less impact on overall quality).
pub struct PositionAware;

impl AcceptanceHeuristic for PositionAware {
    fn acceptance_threshold(
        &self,
        draft_prob: f64,
        target_prob: f64,
        position: usize,
        _draft_entropy: f64,
        _target_entropy: f64,
        _top_token_match: bool,
    ) -> f64 {
        let base_threshold = (target_prob / draft_prob).min(1.0);

        // Position boost: start conservative, become more aggressive
        // Caps at position 50
        let position_factor = 1.0 + (position.min(50) as f64 / 100.0);

        (base_threshold * position_factor).min(1.0)
    }
}

/// Probability ratio with floor
///
/// Like standard rejection sampling but with a minimum acceptance threshold.
/// Trades some quality for guaranteed minimum acceptance rate.
pub struct RatioWithFloor {
    pub floor: f64,
}

impl Default for RatioWithFloor {
    fn default() -> Self {
        Self { floor: 0.3 }
    }
}

impl AcceptanceHeuristic for RatioWithFloor {
    fn acceptance_threshold(
        &self,
        draft_prob: f64,
        target_prob: f64,
        _position: usize,
        _draft_entropy: f64,
        _target_entropy: f64,
        _top_token_match: bool,
    ) -> f64 {
        let base_threshold = (target_prob / draft_prob).min(1.0);
        base_threshold.max(self.floor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_rejection_sampling() {
        let heuristic = StandardRejectionSampling;

        // When target_prob > draft_prob, should accept with prob 1.0
        assert_eq!(heuristic.acceptance_threshold(0.5, 0.8, 0, 1.0, 1.0, true), 1.0);

        // When target_prob < draft_prob, should accept with prob target/draft
        assert!((heuristic.acceptance_threshold(0.8, 0.4, 0, 1.0, 1.0, false) - 0.5).abs() < 1e-10);

        // Equal probabilities
        assert_eq!(heuristic.acceptance_threshold(0.5, 0.5, 0, 1.0, 1.0, true), 1.0);
    }

    #[test]
    fn test_always_accept() {
        let heuristic = AlwaysAccept;
        assert_eq!(heuristic.acceptance_threshold(0.1, 0.9, 0, 5.0, 5.0, false), 1.0);
        assert_eq!(heuristic.acceptance_threshold(0.9, 0.1, 0, 1.0, 1.0, true), 1.0);
    }

    #[test]
    fn test_conservative() {
        let heuristic = Conservative;

        // Accept when target >= draft
        assert_eq!(heuristic.acceptance_threshold(0.5, 0.8, 0, 1.0, 1.0, true), 1.0);
        assert_eq!(heuristic.acceptance_threshold(0.5, 0.5, 0, 1.0, 1.0, true), 1.0);

        // Reject when target < draft
        assert_eq!(heuristic.acceptance_threshold(0.8, 0.5, 0, 1.0, 1.0, false), 0.0);
    }
}
