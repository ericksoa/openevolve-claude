# Santa 2025 - Christmas Tree Packing

Competing in the [Kaggle Santa 2025 Christmas Tree Packing Challenge](https://www.kaggle.com/competitions/santa-2025).

## Claude Code Instructions

When working on this problem, always:
1. **Generate an SVG visualization** of the bin full of trees after evaluating a candidate:
   - Run `./target/release/visualize` to generate `packing_n*.svg` files
   - Open with `open packing_n200.svg` to view in browser
   - This helps understand the packing structure and identify issues
2. Use `./target/release/benchmark` to test candidates (runs 3 trials, reports best score)
3. Save mutation candidates to `mutations/` directory before testing
4. Update `EVOLUTION_STATE.md` with results

## Problem

Pack 1-200 Christmas tree-shaped polygons into the smallest square box.

**Scoring**: `score = sum(side^2/n)` for n=1 to 200 (lower is better)

**Leaderboard**: Top scores ~69, our current best: **~87-88** (Gen91b)

## Tree Shape

The tree is a 15-vertex polygon:
- Height: 1.0 (tip at y=0.8, trunk bottom at y=-0.2)
- Width: 0.7 (at base)
- 3 tiers of branches + rectangular trunk

## Current Best Packing (n=200)

![Packing visualization for n=200 trees](packing_n200.svg)

*Gen91b packing of 200 trees. Green polygons are the tree shapes, blue box shows the bounding square.*

## Current Best Algorithm (Gen91b - ROTATION-FIRST OPTIMIZATION)

```rust
// 6 parallel placement strategies
strategies = [ClockwiseSpiral, CounterclockwiseSpiral, Grid,
              Random, BoundaryFirst, ConcentricRings]

// Gen91b KEY INNOVATION: Exhaustive rotation search at each position
for attempt in 0..200 {
    let dir = select_direction_for_strategy(n, strategy, attempt);

    // First find approximate valid distance (any rotation)
    let probe_dist = binary_search_any_rotation_valid(dir);

    // Then try ALL 8 rotations at this distance with fine-tuning
    for angle in [0, 45, 90, 135, 180, 225, 270, 315] {
        let pos = fine_tune_binary_search(dir, probe_dist, angle);
        if better_score(pos) { best = pos; }
    }
}

// SA optimization parameters:
// - 85% boundary-focused moves
// - 20% compression probability
// - center_pull_strength: 0.08
// - Hot restarts from elite pool
// - 28,000 iterations per pass

// Wave compaction: 5 passes with 4+1 bidirectional split
for wave in 0..5 {
    let tree_order = if wave < 4 {
        trees_sorted_by_distance_from_center.desc()  // Outside-in (4 waves)
    } else {
        trees_sorted_by_distance_from_center.asc()   // Inside-out (1 wave)
    };

    for tree in tree_order {
        // Phase 1-4: Cardinal directions (R->L->U->D)
        // Phase 5: Diagonal movement
        try_compaction_moves(tree, steps);
    }
}

// Greedy backtracking for boundary trees
for pass in 0..3 {
    for tree in boundary_defining_trees {
        try_aggressive_inward_move(tree);
        if_failed_try_with_rotation(tree);
    }
}
```

## What Works

1. **Exhaustive 8-rotation search at each position** (Gen91b) - Try all rotations for best placement
2. **ConcentricRings placement** - Structured > chaotic
3. **Gentle radius compression** - Pull trees toward center (20% prob, 0.08 strength)
4. **Bidirectional wave compaction** - First 4 waves outside-in, last 1 wave inside-out
5. **4-cardinal wave phases** - Compress in right->left->up->down->diagonal directions
6. **Hot restarts with elite pool** - Escape local optima
7. **Boundary-focused SA** (85% probability) - Move trees that define bbox
8. **Binary search for placement** - Fast, precise positioning
9. **Discrete 45 deg angles** - Maintains SA stability
10. **Greedy backtracking for boundary trees** - Post-wave optimization

## What Doesn't Work (Exhaustively Tested)

### Parameter Tuning (Gen92)
- More rotations (16), finer precision (0.0005), more iterations
- Different wave passes, SA temperatures, cooling rates

### Algorithmic Changes (Gen93)
- Relocate moves, coarse-to-fine, aspect ratio penalty
- Force-directed compression, combined parameters

### Paradigm Shifts within Greedy (Gen94)
- Multi-start optimization (high variance, unreliable)
- Hexagonal grid seeding
- Genetic algorithm (crossover creates overlaps)

### Global Optimization (Gen95)
- Annealing schedule overhaul (higher temp 2.0, slower cooling, 100k iters) - WORSE
- Full configuration SA (optimize all at once after placement) - WORSE
- Global rotation optimization (decouple position/rotation) - WORSE
- Center-first placement (re-center and compress) - WORSE

### Earlier Failures
- Finer angle granularity everywhere (15 deg or 30 deg) - 4x slower, worse results
- Continuous angles in SA - Hurts convergence badly
- Global rotation during SA - Destabilizes search
- NFP tangent placement - Misses good positions
- Post-SA global rotation - Doesn't help
- Chain moves, micro-rotations, adaptive density-based angles

## Evolution Journey

This project uses the `/evolve` skill to discover novel packing algorithms through evolutionary optimization.

### Key Milestones

| Gen | Score | Key Innovation |
|-----|-------|----------------|
| 1-10 | ~100-111 | Basic greedy + binary search |
| 11-28 | ~91-97 | Simulated annealing + boundary focus |
| 47 | 89.59 | ConcentricRings breakthrough |
| 62 | 88.22 | Radius compression moves |
| 72b | 89.46 | Wave compaction |
| 80b | 88.44 | 4-cardinal wave phases |
| 83a | 88.22 | Bidirectional wave crossover |
| 84c | 87.36 | Extreme 4+1 split |
| 91b | ~87-88 | Rotation-first optimization |
| 99-100 | - | ILP, Sparrow algorithm (all failed) |
| 101 | 89.59 | Combined strategies (+0.38%) |
| 102 | 86.55 | ML value function + Best-of-N discovery |
| **103** | **~86** | **Best-of-N optimization (+3.87%)** |

### Plateau and Breakthrough (Gen92-103)

**Gen92-98**: Algorithm plateau confirmed
- Parameter tuning, algorithmic changes, paradigm shifts - all failed
- Greedy incremental approach hit fundamental limit

**Gen99-100**: Alternative approaches explored
- ILP/constraint optimization - computationally intractable
- Sparrow algorithm (from research) - worse than evolved
- Global optimization - too many local minima

**Gen101**: Combined strategies
- Diamond/Hex patterns + Sparrow exploration + Wave compaction
- +0.38% improvement, mostly for small N

**Gen102**: ML value function
- Trained neural network to predict packing quality
- Model predicts well (MAE=0.04) but doesn't select well
- **Discovery**: Simple best-of-N beats ML re-ranking!

**Gen103**: Best-of-N optimization (breakthrough!)
- Run evolved multiple times, pick best per N
- Exploits stochastic variance in SA algorithm
- **Best-of-20: 85.89 (+3.87%)** - best result achieved!

| Approach | Score | Improvement |
|----------|-------|-------------|
| Single run | ~89 | baseline |
| Best-of-5 | 86.55 | +3.52% |
| **Best-of-20** | **85.89** | **+3.87%** |
| Stochastic-20 | 86.09 | +2.86% |
| Multi-strategy | 86.80 | +2.06% |

**Key insight**: The evolved algorithm's default parameters are already optimal. Parameter variation and multi-strategy approaches don't help. Simple multiple runs with selection is most effective.

## Running

```bash
cd rust

# Build
cargo build --release

# Run benchmark (3 runs, ~2-3 min each)
./target/release/benchmark 200 3

# Generate submission
./target/release/submit

# Submit to Kaggle
kaggle competitions submit -c santa-2025 -f submission.csv -m "message"
```

## File Structure

```
santa-2025-packing/
├── README.md
├── EVOLUTION_STATE.md      # Current evolution status
├── NEXT_STEPS.md           # Future directions
├── data/
│   └── sample_submission.csv
├── mutations/              # All generation variants
└── rust/
    ├── Cargo.toml
    └── src/
        ├── lib.rs           # Core types (tree, packing)
        ├── evolved.rs       # Current champion (Gen91b)
        ├── baselines.rs     # Simple algorithms
        ├── incremental.rs   # Incremental packing
        └── benchmark.rs     # Benchmark runner
```

## Results Summary

| Milestone | Score | Gap to Leader | Key Innovation |
|-----------|-------|---------------|----------------|
| Initial greedy | ~111 | +61% | - |
| SA optimization | ~91 | +32% | Simulated annealing |
| Gen47 ConcentricRings | 89.59 | +30% | Structured placement |
| Gen62 RadiusCompress | 88.22 | +28% | Compression moves |
| Gen84c ExtremeSplit | 87.36 | +27% | 4+1 wave split |
| **Gen91b RotationFirst** | **~87-88** | **~27%** | **Exhaustive rotation search** |
| *Target (top solution)* | *~69* | - | Unknown (likely ILP or different paradigm) |

**Note**: High run-to-run variance (1-2 points) due to stochastic SA. Scores shown are best of 3 runs.

**Status**: Evolution plateaued at Gen91b. Further progress requires fundamentally different algorithmic paradigm (ILP, constraint programming, etc.).

## References

- [Kaggle Competition](https://www.kaggle.com/competitions/santa-2025)
- [Getting Started Notebook](https://www.kaggle.com/code/inversion/santa-2025-getting-started)
- [70.1 Solution Analysis](https://github.com/berkaycamur/Santa-Competition) - Uses continuous angles + global rotation
