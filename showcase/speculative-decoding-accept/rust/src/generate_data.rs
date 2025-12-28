//! Generate synthetic evaluation data for speculative decoding acceptance heuristics
//!
//! This creates realistic (draft_prob, target_prob) pairs that simulate
//! the distribution of values seen in real speculative decoding scenarios.
//!
//! Key characteristics we model:
//! 1. Draft and target often agree (top token match ~60-80%)
//! 2. When they agree, probabilities are correlated but not identical
//! 3. Entropy varies by position and content type
//! 4. Early positions tend to be more predictable

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde_json;
use speculative_decoding::TokenVerification;
use std::fs::File;
use std::io::BufWriter;

/// Configuration for data generation
struct DataConfig {
    /// Number of sequences to generate
    num_sequences: usize,
    /// Average sequence length
    avg_seq_len: usize,
    /// Probability that draft and target agree on top token
    top_token_match_rate: f64,
    /// Correlation between draft and target probs when they match
    prob_correlation: f64,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            num_sequences: 1000,
            avg_seq_len: 50,
            top_token_match_rate: 0.72, // Realistic for good draft models
            prob_correlation: 0.85,
        }
    }
}

/// Generate a single token verification instance
fn generate_token(
    rng: &mut ChaCha8Rng,
    position: usize,
    config: &DataConfig,
) -> TokenVerification {
    // Determine if top tokens match
    let top_token_match = rng.gen::<f64>() < config.top_token_match_rate;

    // Generate draft probability
    // Use beta distribution to get realistic probability shapes
    // Higher alpha,beta = more peaked around middle values
    // Lower = more extreme (very high or very low)
    let draft_alpha = if position < 5 { 3.0 } else { 2.0 }; // Early tokens more predictable
    let draft_beta = 1.5;

    // Sample from beta and transform to get draft_prob
    let beta_sample: f64 = rand_distr::Beta::new(draft_alpha, draft_beta)
        .map(|d| rng.sample(d))
        .unwrap_or(0.5);
    let draft_prob = 0.05 + beta_sample * 0.9; // Range [0.05, 0.95]

    // Generate target probability
    let target_prob = if top_token_match {
        // When matching, target prob is correlated with draft prob
        let noise: f64 = rng.gen::<f64>() * 0.3 - 0.15; // ±15%
        let correlated = draft_prob * config.prob_correlation
            + rng.gen::<f64>() * (1.0 - config.prob_correlation);
        (correlated + noise).clamp(0.01, 0.99)
    } else {
        // When not matching, target assigns lower prob to draft's token
        // Target chose a different token, so draft's token gets lower prob
        let reduction = 0.3 + rng.gen::<f64>() * 0.5; // Reduce by 30-80%
        (draft_prob * (1.0 - reduction)).clamp(0.01, 0.5)
    };

    // Generate entropy values
    // Lower entropy = more confident, higher entropy = more uncertain
    // Entropy typically ranges from 0 to ln(vocab_size) ≈ 10 for 32k vocab
    let base_entropy = 2.0 + rng.gen::<f64>() * 4.0; // Range [2, 6] typical

    // Draft entropy (small models often more uncertain)
    let draft_entropy = base_entropy * (1.0 + rng.gen::<f64>() * 0.3);

    // Target entropy (usually lower than draft for good tokens)
    let target_entropy = if top_token_match {
        base_entropy * (0.8 + rng.gen::<f64>() * 0.3) // 80-110% of base
    } else {
        base_entropy * (1.0 + rng.gen::<f64>() * 0.5) // 100-150% of base when disagreeing
    };

    // Pre-generate random value for rejection sampling
    let rand_value = rng.gen::<f64>();

    // Calculate baseline acceptance (standard rejection sampling)
    let baseline_threshold = (target_prob / draft_prob).min(1.0);
    let baseline_accepts = rand_value < baseline_threshold;

    TokenVerification {
        draft_prob,
        target_prob,
        position,
        draft_entropy,
        target_entropy,
        top_token_match,
        rand_value,
        baseline_accepts,
    }
}

/// Generate a full dataset
fn generate_dataset(seed: u64, config: &DataConfig) -> Vec<TokenVerification> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut data = Vec::new();

    for _ in 0..config.num_sequences {
        // Variable sequence length
        let seq_len = config.avg_seq_len / 2
            + (rng.gen::<f64>() * config.avg_seq_len as f64) as usize;

        for position in 0..seq_len {
            data.push(generate_token(&mut rng, position, config));
        }
    }

    data
}

fn main() {
    let config = DataConfig::default();

    println!("Generating speculative decoding evaluation datasets...\n");

    // Generate train, valid, and test sets with different seeds
    let datasets = [
        ("train", 42, 1.0),       // Full size
        ("valid", 123, 0.3),      // 30% of train size
        ("test", 456, 0.3),       // 30% of train size (holdout)
    ];

    for (name, seed, size_factor) in datasets {
        let mut adjusted_config = DataConfig::default();
        adjusted_config.num_sequences = (config.num_sequences as f64 * size_factor) as usize;

        let data = generate_dataset(seed, &adjusted_config);

        // Calculate statistics
        let total = data.len();
        let accepts = data.iter().filter(|t| t.baseline_accepts).count();
        let matches = data.iter().filter(|t| t.top_token_match).count();

        println!("{}:", name.to_uppercase());
        println!("  Tokens: {}", total);
        println!(
            "  Baseline acceptance rate: {:.1}%",
            100.0 * accepts as f64 / total as f64
        );
        println!(
            "  Top token match rate: {:.1}%",
            100.0 * matches as f64 / total as f64
        );
        println!();

        // Save to file
        let path = format!("data/{}.json", name);
        std::fs::create_dir_all("data").expect("Failed to create data directory");
        let file = File::create(&path).expect("Failed to create file");
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &data).expect("Failed to write JSON");
        println!("  Saved to {}", path);
        println!();
    }

    println!("Data generation complete!");
    println!("\nTo verify baseline performance, run:");
    println!("  cargo run --release --bin benchmark");
}
