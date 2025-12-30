# Santa 2025 - Christmas Tree Packing

Competing in the [Kaggle Santa 2025 Christmas Tree Packing Challenge](https://www.kaggle.com/competitions/santa-2025).

## Problem

Pack 1-200 Christmas tree-shaped polygons into the smallest square box.

**Scoring**: `score = Σ(side²/n)` for n=1 to 200 (lower is better)

**Leaderboard**: Top scores ~69, our current best ~100

## Tree Shape

The tree is a 15-vertex polygon:
- Height: 1.0 (tip at y=0.8, trunk bottom at y=-0.2)
- Width: 0.7 (at base)
- 3 tiers of branches + rectangular trunk

## Approach

### Baseline Algorithm
- **Incremental packing**: Build n-tree solution from (n-1)-tree solution
- **Greedy placement**: Place new tree at closest valid position from multiple directions
- **Simulated annealing**: Local optimization with moves:
  - Small translations
  - 90° and 45° rotations
  - Move toward center
  - Position swaps

### Multi-start Optimization
- Run multiple independent restarts
- Keep best solution for each n
- Score: ~100-110 (depending on parameters)

## Running

```bash
cd rust

# Build
cargo build --release

# Run benchmark
./target/release/benchmark

# Generate submission
./target/release/submit

# Submit to Kaggle
kaggle competitions submit -c santa-2025 -f submission.csv -m "message"
```

## File Structure

```
santa-2025-packing/
├── README.md
├── data/
│   └── sample_submission.csv
├── mutations/           # (for evolution)
└── rust/
    ├── Cargo.toml
    └── src/
        ├── lib.rs           # Core types (tree, packing)
        ├── baselines.rs     # Simple algorithms
        ├── incremental.rs   # Incremental packing
        ├── multistart.rs    # Multi-start optimizer
        ├── simulated_annealing.rs
        ├── benchmark.rs     # Benchmark runner
        ├── submit.rs        # Submission generator
        └── analyze.rs       # Score analysis
```

## Evolution Targets

Key parameters to evolve:
1. **SA temperature schedule**: `sa_temp`, `sa_cooling`
2. **Search depth**: `restarts`, `search_attempts`, `sa_iterations`
3. **Direction selection**: weighted angles, number of attempts
4. **Rotation choices**: which angles to try
5. **Move operators**: translation scale, rotation granularity

## Results

| Version | Score | Gap to Leader |
|---------|-------|---------------|
| Initial greedy | ~111 | +61% |
| Incremental | ~103 | +49% |
| Multi-start | ~100 | +45% |

**Status**: In progress. Competition deadline: January 30, 2026.

## References

- [Kaggle Competition](https://www.kaggle.com/competitions/santa-2025)
- [Getting Started Notebook](https://www.kaggle.com/code/inversion/santa-2025-getting-started)
