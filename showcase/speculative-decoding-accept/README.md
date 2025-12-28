# Speculative Decoding Acceptance Heuristic

**Evolve smarter token acceptance criteria for speculative decoding**

## The Problem

In speculative decoding, a small **draft model** proposes K tokens that a larger **target model** verifies in parallel. The acceptance decision traditionally uses rejection sampling:

```
accept if: random() < min(1, p_target / p_draft)
```

This is **theoretically optimal for lossless** decoding—it guarantees the output distribution matches the target model exactly. But what if we could find heuristics that:

1. **Accept more aggressively** when we're confident the target will agree
2. **Maintain quality** by avoiding false accepts that degrade output
3. **Adapt to context** using entropy, position, and token-match signals

## Why This Matters for NVIDIA

Speculative decoding is a **core optimization** in modern LLM inference:

- **TensorRT-LLM**: NVIDIA's production inference engine uses speculative decoding
- **2-3x speedup potential**: Higher acceptance rate = fewer target model calls
- **Medusa, EAGLE**: Advanced speculative methods need smart acceptance criteria

A 5% improvement in acceptance rate (without quality loss) translates directly to **5% faster inference**.

## The Evolution Target

We evolve the `acceptance_threshold` function:

```rust
fn acceptance_threshold(
    draft_prob: f64,      // P(token) from draft model
    target_prob: f64,     // P(token) from target model
    position: usize,      // Token position in sequence
    draft_entropy: f64,   // Uncertainty of draft distribution
    target_entropy: f64,  // Uncertainty of target distribution
    top_token_match: bool // Do both models agree on top token?
) -> f64
```

**Accept the token if** `random_value < threshold`.

## Fitness Function

```
fitness = acceptance_rate × quality_score  (if quality ≥ 0.95)
fitness = quality_score × 0.5              (if quality < 0.95)
```

Where:
- **acceptance_rate**: Fraction of tokens accepted (higher = faster inference)
- **quality_score**: Penalizes false accepts (10×) more than false rejects (0.1×)

This heavily penalizes quality degradation while rewarding speed improvements.

## Baseline Performance

| Heuristic | Accept% | Accuracy | Fitness | Notes |
|-----------|---------|----------|---------|-------|
| **standard_rejection** | 78.5% | 100.0% | 0.785 | Theoretically optimal baseline |
| always_accept | 100.0% | 78.5% | 0.000 | Maximum speed, zero quality |
| conservative | 33.2% | 54.7% | 0.317 | Too aggressive rejection |
| entropy_aware | 61.6% | 83.2% | 0.606 | Adapts to uncertainty |
| ratio_floor_0.3 | 78.7% | 99.7% | 0.766 | Minimum 30% acceptance |

**Target**: Beat 0.785 fitness while maintaining near-100% accuracy.

## Evaluation Speed

The benchmark evaluates **49,000 tokens in 0.2ms** — enabling hundreds of generations in minutes.

| Dataset | Tokens | Eval Time |
|---------|--------|-----------|
| TRAIN | 49,185 | 0.21ms |
| VALID | 14,602 | 0.06ms |
| TEST | 14,855 | 0.06ms |

## Quick Start

```bash
cd showcase/speculative-decoding-accept/rust

# Generate evaluation data
cargo run --release --bin generate_data

# Run benchmark
cargo run --release --bin benchmark
```

## Data Characteristics

The synthetic data models realistic speculative decoding scenarios:

- **Top token match rate**: ~72% (draft and target agree on most likely token)
- **Baseline acceptance**: ~78.5% (via standard rejection sampling)
- **Entropy range**: 2-6 nats (varies by confidence)
- **Position effect**: Early tokens more predictable

## Evolution Strategy

Promising mutation directions:

1. **Entropy-weighted thresholds**: Be more aggressive when both models are confident
2. **Top-token boosting**: Higher acceptance when models agree on the winner
3. **Position-adaptive**: Different thresholds for early vs. late tokens
4. **Probability ratio transforms**: Non-linear mappings of p_target/p_draft
5. **Hybrid strategies**: Combine multiple signals with learned weights

## File Structure

```
showcase/speculative-decoding-accept/
├── rust/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs           # Trait + data structures
│       ├── baselines.rs     # Known algorithms to beat
│       ├── evolved.rs       # Champion (evolves here)
│       ├── benchmark.rs     # Evaluation harness
│       └── generate_data.rs # Synthetic data generator
├── data/
│   ├── train.json           # Training data
│   ├── valid.json           # Validation data
│   └── test.json            # Holdout test data
└── README.md
```

## References

- [Accelerating LLM Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) - Original paper
- [NVIDIA Speculative Decoding Blog](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)
- [Medusa: Simple LLM Inference Acceleration](https://arxiv.org/abs/2401.10774) - Multi-head speculation
- [EAGLE: Speculative Sampling with Draft Model](https://arxiv.org/abs/2401.15077) - Efficient draft modeling
