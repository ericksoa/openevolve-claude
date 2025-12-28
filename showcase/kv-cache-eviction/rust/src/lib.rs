//! KV-Cache Eviction Policy Evolution
//!
//! Evolves token importance scoring functions for KV-cache eviction in LLM inference.
//! The goal is to determine which tokens to keep in the cache when memory is constrained.

pub mod baselines;
pub mod evolved;

/// Token metadata for scoring
#[derive(Clone, Debug)]
pub struct TokenInfo {
    /// Absolute position in sequence (0-indexed)
    pub position: usize,
    /// Distance from current generation position
    pub relative_pos: usize,
    /// Cumulative attention received (sum over all past queries)
    pub cumulative_attn: f64,
    /// Recent attention received (sum over last K queries)
    pub recent_attn: f64,
    /// L2 norm of the key vector
    pub key_norm: f64,
    /// Attention entropy (how spread out attention to this token is)
    pub attn_entropy: f64,
    /// Layer index (0 to num_layers-1)
    pub layer_idx: usize,
    /// Total number of layers
    pub num_layers: usize,
    /// Current sequence length
    pub sequence_len: usize,
    /// Is this a sink token (position < 4)?
    pub is_sink: bool,
    /// Token type (0=regular, 1=special, 2=punctuation)
    pub token_type: u8,
}

impl TokenInfo {
    /// Create a new TokenInfo with default values
    pub fn new(position: usize, sequence_len: usize) -> Self {
        Self {
            position,
            relative_pos: sequence_len.saturating_sub(position),
            cumulative_attn: 0.0,
            recent_attn: 0.0,
            key_norm: 1.0,
            attn_entropy: 0.0,
            layer_idx: 0,
            num_layers: 32,
            sequence_len,
            is_sink: position < 4,
            token_type: 0,
        }
    }
}

/// Trait for KV-cache eviction scoring functions.
///
/// The scorer assigns an importance value to each token in the cache.
/// Higher scores = more important = keep in cache.
/// Lower scores = less important = evict first.
pub trait EvictionScorer {
    /// Score a token's importance.
    ///
    /// # Arguments
    /// * `token` - Token metadata including position, attention scores, norms, etc.
    ///
    /// # Returns
    /// Importance score (higher = keep, lower = evict)
    fn score(&self, token: &TokenInfo) -> f64;

    /// Name of this scoring method
    fn name(&self) -> &'static str;
}

/// Attention pattern for a single sequence
#[derive(Clone, Debug)]
pub struct AttentionPattern {
    /// Sequence length
    pub seq_len: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Full attention scores [num_layers][seq_len][seq_len]
    /// attention[layer][query][key] = attention weight
    pub attention: Vec<Vec<Vec<f64>>>,
    /// Key norms [num_layers][seq_len]
    pub key_norms: Vec<Vec<f64>>,
    /// Token types [seq_len]
    pub token_types: Vec<u8>,
    /// Ground truth important tokens (for some patterns)
    pub important_positions: Vec<usize>,
}

impl AttentionPattern {
    /// Compute token info for all tokens at a given layer and query position
    pub fn get_token_infos(&self, layer: usize, query_pos: usize) -> Vec<TokenInfo> {
        let mut infos = Vec::with_capacity(query_pos + 1);

        // Compute cumulative attention for each key position
        for key_pos in 0..=query_pos {
            let mut cumulative = 0.0;
            let mut recent = 0.0;
            let recent_window = 32.min(query_pos);

            // Sum attention from all queries to this key
            for q in key_pos..=query_pos {
                let attn = self.attention[layer][q][key_pos];
                cumulative += attn;
                if q >= query_pos.saturating_sub(recent_window) {
                    recent += attn;
                }
            }

            // Compute attention entropy for this key
            let mut entropy = 0.0;
            for q in key_pos..=query_pos {
                let attn = self.attention[layer][q][key_pos];
                if attn > 1e-10 {
                    entropy -= attn * attn.ln();
                }
            }

            infos.push(TokenInfo {
                position: key_pos,
                relative_pos: query_pos - key_pos,
                cumulative_attn: cumulative,
                recent_attn: recent,
                key_norm: self.key_norms[layer][key_pos],
                attn_entropy: entropy,
                layer_idx: layer,
                num_layers: self.num_layers,
                sequence_len: query_pos + 1,
                is_sink: key_pos < 4,
                token_type: self.token_types[key_pos],
            });
        }

        infos
    }
}

/// Simulate eviction and compute reconstruction error
pub fn evaluate_eviction<S: EvictionScorer + ?Sized>(
    pattern: &AttentionPattern,
    scorer: &S,
    compression_ratio: f64,
) -> f64 {
    let mut total_error = 0.0;
    let mut count = 0;

    // Evaluate at multiple query positions (skip early positions with few tokens)
    for query_pos in (pattern.seq_len / 4)..pattern.seq_len {
        let budget = ((query_pos + 1) as f64 * compression_ratio).ceil() as usize;

        for layer in 0..pattern.num_layers {
            let token_infos = pattern.get_token_infos(layer, query_pos);

            // Score all tokens
            let mut scored: Vec<(usize, f64)> = token_infos
                .iter()
                .map(|t| (t.position, scorer.score(t)))
                .collect();

            // Sort by score descending (highest score = keep)
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Keep top `budget` tokens
            let kept: std::collections::HashSet<usize> =
                scored.iter().take(budget).map(|(pos, _)| *pos).collect();

            // Compute attention reconstruction error
            let full_attn = &pattern.attention[layer][query_pos];

            // Original attention sum (should be ~1.0 for softmax)
            let original_sum: f64 = full_attn.iter().take(query_pos + 1).sum();

            // Attention to kept tokens only
            let kept_sum: f64 = (0..=query_pos)
                .filter(|p| kept.contains(p))
                .map(|p| full_attn[p])
                .sum();

            // Error = fraction of attention lost to evicted tokens
            let error = (original_sum - kept_sum) / original_sum.max(1e-10);
            total_error += error;
            count += 1;
        }
    }

    total_error / count as f64
}

/// Benchmark result for a single scorer
#[derive(Clone, Debug, serde::Serialize)]
pub struct ScorerResult {
    pub name: String,
    pub avg_error: f64,
    pub error_at_25: f64,  // Error at 25% compression
    pub error_at_50: f64,  // Error at 50% compression
    pub error_at_75: f64,  // Error at 75% compression
}

/// Run benchmark across multiple patterns and compression ratios
pub fn benchmark_scorer<S: EvictionScorer + ?Sized>(
    scorer: &S,
    patterns: &[AttentionPattern],
) -> ScorerResult {
    let mut error_25 = 0.0;
    let mut error_50 = 0.0;
    let mut error_75 = 0.0;

    for pattern in patterns {
        error_25 += evaluate_eviction(pattern, scorer, 0.25);
        error_50 += evaluate_eviction(pattern, scorer, 0.50);
        error_75 += evaluate_eviction(pattern, scorer, 0.75);
    }

    let n = patterns.len() as f64;
    error_25 /= n;
    error_50 /= n;
    error_75 /= n;

    ScorerResult {
        name: scorer.name().to_string(),
        avg_error: (error_25 + error_50 + error_75) / 3.0,
        error_at_25: error_25,
        error_at_50: error_50,
        error_at_75: error_75,
    }
}

/// Fast sampled eviction evaluation - samples positions and layers
pub fn evaluate_eviction_fast<S: EvictionScorer + ?Sized>(
    pattern: &AttentionPattern,
    scorer: &S,
    compression_ratio: f64,
    position_samples: usize,
    layer_samples: usize,
) -> f64 {
    let mut total_error = 0.0;
    let mut count = 0;

    let start_pos = pattern.seq_len / 4;
    let end_pos = pattern.seq_len;
    let pos_step = ((end_pos - start_pos) / position_samples).max(1);

    let layer_step = (pattern.num_layers / layer_samples).max(1);

    // Sample query positions evenly
    for query_pos in (start_pos..end_pos).step_by(pos_step) {
        let budget = ((query_pos + 1) as f64 * compression_ratio).ceil() as usize;

        // Sample layers evenly
        for layer in (0..pattern.num_layers).step_by(layer_step) {
            let token_infos = pattern.get_token_infos(layer, query_pos);

            // Score all tokens
            let mut scored: Vec<(usize, f64)> = token_infos
                .iter()
                .map(|t| (t.position, scorer.score(t)))
                .collect();

            // Sort by score descending (highest score = keep)
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Keep top `budget` tokens
            let kept: std::collections::HashSet<usize> =
                scored.iter().take(budget).map(|(pos, _)| *pos).collect();

            // Compute attention reconstruction error
            let full_attn = &pattern.attention[layer][query_pos];

            // Original attention sum (should be ~1.0 for softmax)
            let original_sum: f64 = full_attn.iter().take(query_pos + 1).sum();

            // Attention to kept tokens only
            let kept_sum: f64 = (0..=query_pos)
                .filter(|p| kept.contains(p))
                .map(|p| full_attn[p])
                .sum();

            // Error = fraction of attention lost to evicted tokens
            let error = (original_sum - kept_sum) / original_sum.max(1e-10);
            total_error += error;
            count += 1;
        }
    }

    total_error / count as f64
}

/// Fast benchmark - samples fewer positions and layers
pub fn benchmark_scorer_fast<S: EvictionScorer + ?Sized>(
    scorer: &S,
    patterns: &[AttentionPattern],
    position_samples: usize,
    layer_samples: usize,
) -> ScorerResult {
    let mut error_25 = 0.0;
    let mut error_50 = 0.0;
    let mut error_75 = 0.0;

    for pattern in patterns {
        error_25 += evaluate_eviction_fast(pattern, scorer, 0.25, position_samples, layer_samples);
        error_50 += evaluate_eviction_fast(pattern, scorer, 0.50, position_samples, layer_samples);
        error_75 += evaluate_eviction_fast(pattern, scorer, 0.75, position_samples, layer_samples);
    }

    let n = patterns.len() as f64;
    error_25 /= n;
    error_50 /= n;
    error_75 /= n;

    ScorerResult {
        name: scorer.name().to_string(),
        avg_error: (error_25 + error_50 + error_75) / 3.0,
        error_at_25: error_25,
        error_at_50: error_50,
        error_at_75: error_75,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baselines::StreamingLLM;

    #[test]
    fn test_token_info() {
        let token = TokenInfo::new(5, 100);
        assert_eq!(token.position, 5);
        assert_eq!(token.relative_pos, 95);
        assert!(!token.is_sink);

        let sink = TokenInfo::new(2, 100);
        assert!(sink.is_sink);
    }

    #[test]
    fn test_streaming_llm_sink_priority() {
        let scorer = StreamingLLM::new(4, 64);

        // Sink token should have very high score
        let mut sink_token = TokenInfo::new(2, 100);
        sink_token.is_sink = true;
        let sink_score = scorer.score(&sink_token);

        // Non-sink token
        let mut regular = TokenInfo::new(50, 100);
        regular.is_sink = false;
        let regular_score = scorer.score(&regular);

        assert!(sink_score > regular_score);
    }
}
