# Santa 2025 - Christmas Tree Packing

Competing in the [Kaggle Santa 2025 Christmas Tree Packing Challenge](https://www.kaggle.com/competitions/santa-2025).

## Problem

Pack 1-200 Christmas tree-shaped polygons into the smallest square box.

**Scoring**: `score = Σ(side²/n)` for n=1 to 200 (lower is better)

**Leaderboard**: Top scores ~69, our current best: **88.22** (Gen62)

## Tree Shape

The tree is a 15-vertex polygon:
- Height: 1.0 (tip at y=0.8, trunk bottom at y=-0.2)
- Width: 0.7 (at base)
- 3 tiers of branches + rectangular trunk

## Evolution Journey

This project uses the `/evolve` skill to discover novel packing algorithms through evolutionary optimization. Below is the complete journey with learnings from each generation.

### Phase 1: Foundation (Gen1-Gen10)
**Goal**: Establish baseline algorithms

| Gen | Strategy | Score | Learning |
|-----|----------|-------|----------|
| 1-5 | Simple greedy | ~111 | Need smarter placement |
| 6-10 | Multi-direction search | ~100-103 | Binary search helps find tight placements |

**Key insight**: Greedy placement alone isn't enough. Need local search.

### Phase 2: Simulated Annealing (Gen11-Gen28)
**Goal**: Add local optimization

| Gen | Strategy | Score | Learning |
|-----|----------|-------|----------|
| 11-15 | Basic SA | ~95-97 | SA helps but slow cooling needed |
| 16-20 | Boundary-focused moves | ~93-95 | Moving boundary trees is key |
| 21-25 | Multiple strategies | ~92-93 | Running 5+ strategies in parallel helps |
| 26-28 | Hot restarts | ~91 | Restarting from elite pool escapes local optima |

**Key insight**: 85% of SA moves should target boundary trees (trees that define the bounding box).

### Phase 3: Parameter Tuning (Gen29-Gen46)
**Goal**: Fine-tune SA parameters

| Gen | Strategy | Score | Learning |
|-----|----------|-------|----------|
| 29a-d | Hot restart variants | ~91 | Diminishing returns on tuning |
| 30a-d | Elite pool tracking | ~91 | Keep top 3 configurations |
| 31a-d | More angles, slower cooling | ~91 | 8 angles (45° steps) is optimal |
| 32-40 | Various parameter tweaks | ~91 | Little improvement from params alone |
| 41-46 | Aggressive optimization | ~90-91 | Too aggressive hurts stability |

**Key insight**: Parameter tuning plateaus quickly. Need new algorithmic ideas.

### Phase 4: Breakthrough - Concentric Rings (Gen47)
**Goal**: New placement strategy

| Gen | Strategy | Score | Learning |
|-----|----------|-------|----------|
| **47** | **ConcentricRings** | **89.59** | **BREAKTHROUGH!** First sub-90 |

**The winning formula**:
```rust
let ring = ((n as f64).sqrt() as usize).max(1);
let trees_in_ring = (ring * 6).max(1);  // Hexagonal-ish
let position_in_ring = n % trees_in_ring;
let base_angle = (position_in_ring as f64 / trees_in_ring as f64) * 2.0 * PI;
```

**Key insight**: Structured placement (concentric rings) >> chaotic placement (spirals, random).

### Phase 5: Post-Breakthrough Exploration (Gen48-Gen59)
**Goal**: Improve on ConcentricRings

| Gen | Strategy | Score | Learning |
|-----|----------|-------|----------|
| 48 | Concentric + diagonal | 90.7 | Adding complexity hurts |
| 49 | More SA iterations | 91.1 | More iterations ≠ better |
| 50 | Refined concentric | 90.6 | Marginal gains |
| 51 | 7th strategy (hexagonal) | 92.0 | More strategies = overhead |
| 52 | Adaptive temperature | 91.5 | Not helpful |
| 53 | Tight gap focus | 92.0 | Over-penalizing gaps hurts |
| 54 | 15° angle granularity | **97+** | MUCH WORSE + 4x slower |
| 55 | 45k SA iterations | 91.1 | Diminishing returns |
| 56 | 8 trees per ring | 90.6 | Comparable |
| 57 | Only 3 strategies | 91.2 | Need diversity |
| 58 | Compaction phase | 90.8 | Post-processing not enough |
| 59 | Multi-seed | 90.7 | 2x time, no improvement |

**Key insights**:
1. Finer angles (15°) are MUCH worse - stick to 45° multiples
2. More iterations without new ideas doesn't help
3. Need fundamentally different approach to break 89

### Phase 6: New Move Types (Gen60-Gen63)
**Goal**: Different SA neighborhood moves

| Gen | Strategy | Score | Learning |
|-----|----------|-------|----------|
| 60 | Swap moves | 89.65 | Swapping positions helps slightly |
| 61 | Greedy angle selection | **98.06** | MUCH WORSE - need all 8 angles |
| **62** | **Radius compression** | **88.22** | **NEW BEST!** Pull toward center |
| 63 | Double ring density | 90.50 | More trees per ring doesn't help |

**Key insight**: Radius-based compression moves (pull trees toward center proportional to distance) significantly improve packing density.

### Analysis of Top Solutions

Analyzed a [70.1 score solution](https://github.com/berkaycamur/Santa-Competition) to understand what we're missing:

| Aspect | Our Approach | Top Solution |
|--------|-------------|--------------|
| Angles | 45° multiples | Continuous angles (21°, 66°, etc.) |
| Rotation | None | Global rotation optimization (3° steps) |
| Compaction | In SA | Multiple dedicated passes |
| n=200 side | 9.18 | 7.81 (15% smaller!) |

**Next directions**:
1. Global rotation optimization (rotate entire packing)
2. Dedicated compaction/squeeze passes
3. Continuous angle optimization (careful - Gen54 showed pitfalls)

## Current Best Algorithm (Gen62)

```rust
// 6 parallel placement strategies
strategies = [ClockwiseSpiral, CounterclockwiseSpiral, Grid,
              Random, BoundaryFirst, ConcentricRings]

// Placement: Binary search along direction vectors
for attempt in 0..200 {
    let dir = select_direction_for_strategy(n, strategy, attempt);
    for angle in [0, 45, 90, 135, 180, 225, 270, 315] {
        let pos = binary_search_placement(dir, angle);
        if better_score(pos) { best = pos; }
    }
}

// SA optimization with:
// - 85% boundary-focused moves
// - 20% radius compression moves (NEW in Gen62)
// - Hot restarts from elite pool
// - 28,000 iterations per pass
```

## What Works

1. **ConcentricRings placement** - Structured > chaotic
2. **Radius compression** - Pull trees toward center based on distance
3. **Hot restarts with elite pool** - Escape local optima
4. **Boundary-focused SA** (85% probability) - Move trees that define bbox
5. **Binary search for placement** - Fast, precise positioning
6. **8 angles (45° steps)** - More or fewer is worse

## What Doesn't Work

1. **More iterations alone** - Diminishing returns without new ideas
2. **Finer angle granularity** (15°) - 4x slower, worse results
3. **Too many strategies** (7+) - Overhead > benefit
4. **Multi-seed approach** - 2x time, no improvement
5. **Post-processing compaction** - Should be in SA
6. **Greedy angle selection** - Need exhaustive 8-angle search

## Running

```bash
cd rust

# Build
cargo build --release

# Run benchmark (3 runs, ~2 min each)
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
├── mutations/           # All generation variants (Gen29-Gen63+)
│   ├── gen47_concentric.rs     # First sub-90
│   ├── gen62_radius_compress.rs # Current best (88.22)
│   └── ...
└── rust/
    ├── Cargo.toml
    └── src/
        ├── lib.rs           # Core types (tree, packing)
        ├── evolved.rs       # Current champion algorithm
        ├── baselines.rs     # Simple algorithms
        ├── incremental.rs   # Incremental packing
        └── benchmark.rs     # Benchmark runner
```

## Results Summary

| Milestone | Score | Gap to Leader | Key Innovation |
|-----------|-------|---------------|----------------|
| Initial greedy | ~111 | +61% | - |
| Multi-start | ~100 | +45% | Multiple restarts |
| SA optimization | ~91 | +32% | Simulated annealing |
| Gen47 ConcentricRings | 89.59 | +30% | Structured placement |
| Gen62 RadiusCompress | **88.22** | +28% | Compression moves |
| *Target (top solution)* | *~69* | - | Continuous angles + global rotation |

**Status**: Active evolution. Competition deadline: January 30, 2026.

## References

- [Kaggle Competition](https://www.kaggle.com/competitions/santa-2025)
- [Getting Started Notebook](https://www.kaggle.com/code/inversion/santa-2025-getting-started)
- [70.1 Solution Analysis](https://github.com/berkaycamur/Santa-Competition) - Uses continuous angles + global rotation
