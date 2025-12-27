# TSP 2-opt: GPU-Accelerated Move Selection Evolution

Evolve priority functions for 2-opt local search in the Traveling Salesman Problem. Uses **Apple Metal GPU** for parallel evaluation of O(n²) potential moves.

## The Problem

**2-opt** is the most common TSP local search: it swaps two edges in the tour to reduce total distance. The question is: **which swap to try first?**

```
Before 2-opt:    A ─── B ─── C ─── D
                        ↓ swap (A-B) and (C-D)
After 2-opt:     A ─── C ─── B ─── D
```

With n cities, there are O(n²) possible swaps each iteration. The **priority function** determines which moves are evaluated first - and this matters enormously for solution quality.

## What We Evolve

A priority function that scores potential 2-opt moves:

```rust
pub trait TwoOptPriority {
    fn priority(
        &self,
        delta: f64,           // Tour length change (negative = improvement)
        edge1_len: f64,       // Length of first edge being removed
        edge2_len: f64,       // Length of second edge being removed
        new_edge1_len: f64,   // Length of first new edge
        new_edge2_len: f64,   // Length of second new edge
        tour_len: f64,        // Current total tour length
        n: usize,             // Number of cities
    ) -> f64;
}
```

Higher priority moves are tried first. A good priority function leads the search toward better solutions faster.

## GPU Acceleration

The **Metal GPU benchmark** evaluates all O(n²) potential 2-opt moves in parallel:

```
CPU: Evaluate moves sequentially, O(n²) per iteration
GPU: Evaluate ALL moves in parallel, O(1) per iteration (with n² threads)
```

For large instances (500+ cities), this provides 10-100x speedup.

## Benchmarks

Uses classic **TSPLIB** instances with known optimal solutions:

| Instance | Cities | Optimal | Best Baseline Gap |
|----------|--------|---------|-------------------|
| eil51 | 51 | 426 | 2.23% |
| berlin52 | 52 | 7,542 | 3.97% |
| kroA100 | 100 | 21,282 | 3.00% |

## Current Baselines

| Algorithm | Avg Gap | Description |
|-----------|---------|-------------|
| **greedy_delta** | 3.07% | Prioritize by improvement amount |
| **best_improvement** | 3.07% | Only consider improving moves |
| **relative_gain** | 3.07% | Prioritize by % improvement |
| **long_edge_removal** | 4.31% | Favor removing long edges |
| **edge_ratio** | 4.17% | Ratio of removed to added edges |
| **lk_inspired** | 4.56% | Lin-Kernighan style scoring |
| **balanced** | 3.22% | Combine improvement + structure |

**Target**: Evolve a priority function with <2% average gap.

## Quick Start

### Prerequisites

- Rust toolchain (install via https://rustup.rs)
- macOS with Apple Silicon (for Metal GPU)

### Run CPU Benchmark

```bash
cd showcase/tsp-2opt-gpu/rust
cargo build --release
cargo run --release --bin benchmark 2>/dev/null
```

### Run Metal GPU Benchmark

```bash
cargo run --release --bin benchmark_metal
```

## Evolution Strategy

The evolved priority function can combine multiple signals:

1. **Improvement magnitude**: How much does this move reduce tour length?
2. **Edge characteristics**: Are we removing long edges? Adding short ones?
3. **Relative values**: How do edges compare to average edge length?
4. **Geometric relationships**: Angles, ratios, normalized values

Example evolved formula (conceptual):
```
priority = -delta/tour_len                    // Relative improvement
         + 0.3 * (max_removed - avg_edge)     // Long edge bonus
         - 0.2 * (min_added - avg_edge)       // Short replacement bonus
         + 0.1 * (edge1_len * edge2_len).sqrt() // Geometric mean term
```

## File Structure

```
showcase/tsp-2opt-gpu/
├── README.md           # This file
└── rust/
    ├── Cargo.toml      # Build configuration
    └── src/
        ├── lib.rs          # Core 2-opt algorithm + TwoOptPriority trait
        ├── baselines.rs    # 7 baseline priority functions
        ├── evolved.rs      # Evolving solution (starts as greedy)
        ├── benchmark.rs    # CPU benchmark
        └── benchmark_metal.rs # Metal GPU benchmark
```

## References

- TSPLIB: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
- Lin-Kernighan heuristic: https://en.wikipedia.org/wiki/Lin–Kernighan_heuristic
- 2-opt: https://en.wikipedia.org/wiki/2-opt

## Why This Problem?

1. **Well-defined benchmarks**: TSPLIB has known optimal solutions
2. **GPU-friendly**: O(n²) parallel move evaluation
3. **Evolution potential**: Priority functions have many possible formulations
4. **Practical impact**: TSP appears in routing, logistics, manufacturing

## Deterministic Reproduction

- [x] No external data files required (coordinates embedded in benchmark.rs)
- [x] No network requests
- [x] No randomness (deterministic nearest-neighbor start, best-improvement search)
- [x] Same results every run
