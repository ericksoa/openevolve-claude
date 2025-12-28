//! Speculative Decoding Acceptance Heuristics
//!
//! In speculative decoding, a small draft model proposes tokens that a larger
//! target model verifies. The acceptance decision traditionally uses rejection
//! sampling: accept if rand() < min(1, p_target/p_draft).
//!
//! This module evolves smarter acceptance heuristics that can:
//! - Accept more aggressively when confidence is high
//! - Reject early when acceptance is unlikely
//! - Trade off speed vs quality optimally

pub mod baselines;
pub mod evolved;

/// A single token verification instance from pre-computed data
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct TokenVerification {
    /// Probability assigned by draft model to the proposed token
    pub draft_prob: f64,
    /// Probability assigned by target model to the same token
    pub target_prob: f64,
    /// Position in the sequence (0-indexed)
    pub position: usize,
    /// Entropy of draft model's distribution at this position (uncertainty)
    pub draft_entropy: f64,
    /// Entropy of target model's distribution at this position
    pub target_entropy: f64,
    /// Whether draft and target agree on the top token
    pub top_token_match: bool,
    /// The uniform random value used for rejection sampling (pre-generated for reproducibility)
    pub rand_value: f64,
    /// Ground truth: would standard rejection sampling accept this?
    pub baseline_accepts: bool,
}

/// Result of evaluating an acceptance heuristic on a dataset
#[derive(Debug, Clone, Default)]
pub struct EvaluationResult {
    /// Total tokens evaluated
    pub total_tokens: usize,
    /// Tokens accepted by the heuristic
    pub accepted_tokens: usize,
    /// Tokens where heuristic matches baseline (correct decisions)
    pub correct_decisions: usize,
    /// False accepts: heuristic accepted but baseline would reject
    pub false_accepts: usize,
    /// False rejects: heuristic rejected but baseline would accept
    pub false_rejects: usize,
    /// Acceptance rate (higher = faster inference)
    pub acceptance_rate: f64,
    /// Accuracy vs baseline rejection sampling
    pub accuracy: f64,
    /// Quality score: penalizes false accepts more than false rejects
    /// (false accepts degrade output quality, false rejects just slow things down)
    pub quality_score: f64,
    /// Combined fitness: balances speed and quality
    pub fitness: f64,
}

/// Trait for speculative decoding acceptance heuristics
pub trait AcceptanceHeuristic {
    /// Returns the acceptance threshold for this token.
    /// Accept the token if rand_value < threshold.
    ///
    /// Standard rejection sampling returns: min(1.0, target_prob / draft_prob)
    ///
    /// # Arguments
    /// * `draft_prob` - P(token) from draft model, range (0, 1]
    /// * `target_prob` - P(token) from target model, range (0, 1]
    /// * `position` - Token position in sequence, 0-indexed
    /// * `draft_entropy` - Entropy of draft distribution (higher = more uncertain)
    /// * `target_entropy` - Entropy of target distribution
    /// * `top_token_match` - Whether draft and target agree on most likely token
    fn acceptance_threshold(
        &self,
        draft_prob: f64,
        target_prob: f64,
        position: usize,
        draft_entropy: f64,
        target_entropy: f64,
        top_token_match: bool,
    ) -> f64;

    /// Evaluate this heuristic on a dataset
    fn evaluate(&self, data: &[TokenVerification]) -> EvaluationResult {
        let mut result = EvaluationResult {
            total_tokens: data.len(),
            ..Default::default()
        };

        for token in data {
            let threshold = self.acceptance_threshold(
                token.draft_prob,
                token.target_prob,
                token.position,
                token.draft_entropy,
                token.target_entropy,
                token.top_token_match,
            );

            let accepts = token.rand_value < threshold;

            if accepts {
                result.accepted_tokens += 1;
            }

            // Compare to baseline
            if accepts == token.baseline_accepts {
                result.correct_decisions += 1;
            } else if accepts && !token.baseline_accepts {
                result.false_accepts += 1;
            } else {
                result.false_rejects += 1;
            }
        }

        result.acceptance_rate = result.accepted_tokens as f64 / result.total_tokens as f64;
        result.accuracy = result.correct_decisions as f64 / result.total_tokens as f64;

        // Quality score: heavily penalize false accepts (they degrade output)
        // False rejects just slow things down, which is less bad
        let false_accept_rate = result.false_accepts as f64 / result.total_tokens as f64;
        let false_reject_rate = result.false_rejects as f64 / result.total_tokens as f64;
        result.quality_score = 1.0 - (false_accept_rate * 10.0 + false_reject_rate * 0.1);
        result.quality_score = result.quality_score.max(0.0);

        // Fitness: reward higher acceptance rate, but only if quality is maintained
        // Quality must be >= 0.95 to get any benefit from higher acceptance rate
        if result.quality_score >= 0.95 {
            // Good quality: fitness is based on acceptance rate improvement
            result.fitness = result.acceptance_rate * result.quality_score;
        } else {
            // Poor quality: heavily penalize
            result.fitness = result.quality_score * 0.5;
        }

        result
    }
}

/// Load evaluation data from JSON file
pub fn load_data(path: &str) -> Result<Vec<TokenVerification>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let data: Vec<TokenVerification> = serde_json::from_reader(reader)?;
    Ok(data)
}
