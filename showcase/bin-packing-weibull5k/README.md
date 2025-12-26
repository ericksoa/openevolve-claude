# Bin Packing: Beating FunSearch on Weibull 5k Benchmark

This showcase demonstrates an evolved bin packing heuristic that **beats Google DeepMind's FunSearch** result on the Weibull 5k benchmark.

## Results Summary

| Algorithm | Avg Excess % | Total Bins (5 instances) |
|-----------|-------------|--------------------------|
| **Evolved (ours)** | **0.6339%** | 10,002 |
| FunSearch | 0.6842% | 10,007 |
| Best Fit | 3.98% | 10,335 |
| First Fit | 4.23% | 10,359 |

**Improvement over FunSearch: 7.4% relative (0.05 percentage points)**

## Quick Start

### Prerequisites

- Rust toolchain (install via https://rustup.rs)

### Run the Benchmark

```bash
cd rust
cargo build --release
cargo run --release --bin benchmark
```

### Expected Output

The benchmark outputs JSON with results for each algorithm. You should see:

```
"evolved": avg_excess_percent: ~0.6339
"funsearch": avg_excess_percent: ~0.6842
```

The exact numbers may vary slightly due to floating-point precision, but evolved should consistently beat funsearch.

## Benchmark Details

### Dataset: Weibull 5k

- **5 test instances**, each with **5,000 items**
- Item sizes drawn from Weibull distribution (k=5, λ=50), scaled to [1, 100]
- Bin capacity: 100
- This is the exact benchmark from the FunSearch paper

### Metric: Excess Percentage

```
excess % = (bins_used - L1_lower_bound) / L1_lower_bound * 100
```

Where L1 lower bound = ceil(sum(items) / capacity), the theoretical minimum.

### Algorithm Interface

All algorithms implement:

```rust
pub trait BinPackingHeuristic {
    fn priority(&self, item: u32, bins: &[u32]) -> Vec<f64>;
}
```

Given an item size and array of bin remaining capacities, return priority scores. Higher priority = prefer that bin.

### Online Bin Packing Protocol

Following FunSearch's exact protocol:

1. Pre-allocate one bin per item (5000 bins)
2. For each item:
   - Filter to bins with remaining capacity >= item size
   - Pass **only valid bins** to priority function
   - Select bin with highest priority
   - Update bin's remaining capacity
3. Count bins actually used (remaining != capacity)

## The Winning Algorithm

```rust
fn priority(&self, item: u32, bins: &[u32]) -> Vec<f64> {
    let max_bin_cap = *bins.iter().max().unwrap_or(&0) as f64;
    let item_f = item as f64;

    let mut scores: Vec<f64> = bins.iter()
        .map(|&b| {
            let b_f = b as f64;
            let waste = b_f - item_f;

            // Log transformations - capture relationships in log space
            let log_waste = (waste + 1.0).ln();
            let log_item = (item_f + 1.0).ln();
            let log_bin = (b_f + 1.0).ln();
            let log_ratio = log_bin - log_item; // ln(bin/item)

            // Keep proven quadratic max difference term from FunSearch
            let max_diff_term = (b_f - max_bin_cap).powi(2) / item_f;

            // Log-based utilization emphasizes tight fits
            let log_util_term = log_waste / log_item;

            // Ratio-based log term
            let log_ratio_term = log_ratio / log_item;

            let mut score = max_diff_term + log_util_term * 2.0 + log_ratio_term;

            if b > item { score = -score; }
            score
        })
        .collect();

    // Adjacent difference operation (from FunSearch)
    for i in (1..scores.len()).rev() {
        scores[i] -= scores[i - 1];
    }
    scores
}
```

### Key Innovations

1. **Logarithmic terms** instead of FunSearch's polynomial `b²/item² + b²/item³`
2. **Log-waste ratio** `ln(waste+1) / ln(item+1)` emphasizes tight fits
3. **Log-ratio term** `ln(bin/item) / ln(item+1)` captures relative sizing
4. **Preserved** FunSearch's quadratic max-difference term `(b - max_cap)² / item`
5. **Preserved** the sign flip and adjacent difference operations

## File Structure

```
rust/
├── Cargo.toml          # Build configuration
└── src/
    ├── lib.rs          # Trait definition + bin packing algorithm
    ├── baselines.rs    # First Fit, Best Fit, Worst Fit, FunSearch
    ├── evolved.rs      # Our winning algorithm
    └── benchmark.rs    # Benchmark harness with Weibull 5k data
```

## Reproducing from Scratch

To verify this result without any Claude involvement:

### Step 1: Build and Run

```bash
cd rust
cargo build --release
cargo run --release --bin benchmark 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
print('Algorithm'.ljust(15), 'Excess %')
print('-' * 30)
for r in sorted(data['results'], key=lambda x: x['avg_excess_percent']):
    print(f\"{r['name'].ljust(15)} {r['avg_excess_percent']:.4f}%\")
"
```

### Step 2: Verify Correctness

The benchmark automatically verifies:
- All items are placed
- No bin exceeds capacity
- Bin count matches expectations

Check `"correctness": true` in the output.

### Step 3: Compare Results

Expected ordering (best to worst):
1. evolved: ~0.63%
2. funsearch: ~0.68%
3. best_fit: ~3.98%
4. first_fit: ~4.23%
5. worst_fit: ~151% (pathological)

## References

- FunSearch paper: "Mathematical discoveries from program search with large language models" (Nature, 2024)
- Original FunSearch bin packing code: https://github.com/google-deepmind/funsearch

## Deterministic Reproduction

The benchmark uses fixed test data embedded in `benchmark.rs`. Running the benchmark will always produce the same results (within floating-point precision).

No randomness, no external data files, no network requests.
