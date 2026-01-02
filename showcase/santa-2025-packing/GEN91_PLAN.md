# Gen91+ Evolution Plan: Breaking the Plateau

## Current State
- **Champion**: Gen87d (greedy backtracking wave)
- **Best Score**: 88.03
- **Target**: ~69 (leaderboard top)
- **Gap**: ~27.6%
- **Plateau**: 3 generations (88-90) without improvement

## What's Been Tried (Don't Repeat)

### Gen88-89 (Greedy Variations)
- Pre-wave greedy (88.64) - REJECTED
- Adaptive greedy scaling (89.22) - REJECTED
- Density-ordered greedy (89.21) - REJECTED
- All-direction greedy (89.62) - REJECTED
- Rotation-first greedy (89.19) - REJECTED

### Gen90 (Orthogonal Mutations)
- Wave phase reversal U→D→L→R (89.25) - REJECTED
- Boundary tree swap (88.80) - REJECTED
- More SA iterations 28k→35k (88.50) - REJECTED
- Split ratio 3+2 instead of 4+1 (89.30) - REJECTED
- More greedy passes 3→5 (88.42) - REJECTED (inconsistent)
- Finer greedy step sizes (88.66) - REJECTED
- Center-directed greedy (89.02) - REJECTED

## Gen91 Strategy: Radical Paradigm Shifts

Since local modifications are exhausted, try fundamentally different approaches:

### Phase 1: Bottom-Left-Fill (BLF) Hybrid
**Rationale**: BLF is a classic 2D packing heuristic that places items in the lowest-leftmost valid position.

**Implementation**:
1. Create new placement strategy `BottomLeftFill`
2. For each tree, scan from bottom-left corner outward
3. Place at first valid position (no overlap)
4. Apply existing SA optimization after placement
5. Compare against spiral-based strategies

**Expected Complexity**: Medium - new placement logic, reuse existing optimization

### Phase 2: Size-Ordered Placement
**Rationale**: Current algorithm adds trees 1..N in order. Larger trees (higher N) are harder to place.

**Implementation**:
1. Pre-compute all tree sizes for N=1..200
2. Sort by decreasing size (place largest first)
3. Use greedy placement for each
4. Remap indices back to original order
5. This is a preprocessing step, compatible with all existing strategies

**Expected Complexity**: Low - sorting + remapping

### Phase 3: Genetic Algorithm for Placement Order
**Rationale**: The order of tree placement significantly affects final packing.

**Implementation**:
1. Chromosome = permutation of [1..N]
2. Fitness = negative bounding box side length
3. Crossover = order crossover (OX)
4. Mutation = swap two random positions
5. Population size: 50, generations: 100
6. Use champion packer as fitness evaluator

**Expected Complexity**: High - new module, many runs needed

### Phase 4: Rotation-First Optimization
**Rationale**: Current algorithm fixes rotation during placement, then adjusts in SA. Try optimizing rotation more aggressively.

**Implementation**:
1. During placement, try ALL 8 rotations for each candidate position
2. Score = bounding box + compactness metric
3. Keep best (position, rotation) pair
4. This makes placement O(8x) slower but potentially much better

**Expected Complexity**: Low-Medium - modify existing placement loop

### Phase 5: Annealing Schedule Tuning
**Rationale**: The SA parameters were manually tuned. Systematic search might find better values.

**Parameters to tune**:
- `sa_initial_temp`: currently 0.45
- `sa_cooling_rate`: currently 0.99993
- `sa_iterations`: currently 28000
- `translation_scale`: currently 0.055

**Method**: Grid search or Bayesian optimization over parameter space.

**Expected Complexity**: Medium - need to run many benchmarks

## Implementation Order

1. **Gen91a**: Size-ordered placement (easiest, quick test)
2. **Gen91b**: Rotation-first optimization (low-hanging fruit)
3. **Gen91c**: Bottom-left-fill hybrid (new paradigm)
4. **Gen91d**: SA parameter grid search (if others fail)
5. **Gen91e**: Genetic algorithm (last resort, expensive)

## Benchmark Protocol

For each candidate:
1. Build: `cargo build --release`
2. Test: `./target/release/benchmark 200 3`
3. If best score < 88.03, run 5 more times to verify
4. If consistently better, update champion

## File Locations
- Champion code: `rust/src/evolved.rs`
- Champion backup: `rust/src/evolved_champion.rs`
- Library: `rust/src/lib.rs`
- Benchmark: `rust/src/bin/benchmark.rs`

## Key Code Sections to Modify

### For Size-Ordered (Gen91a)
Modify `pack_all()` in `evolved.rs`:
- Add size computation loop
- Sort indices by size
- Remap at end

### For Rotation-First (Gen91b)
Modify `find_placement_with_strategy()` in `evolved.rs`:
- Inner loop over all 8 rotations for each direction
- Track best (position, rotation, score) tuple

### For BLF (Gen91c)
Add new variant to `PlacementStrategy` enum:
- `BottomLeftFill`
- Implement in `select_direction_for_strategy()`
- Scan from min bounds outward

## Success Criteria
- Score < 88.00: Minor improvement, continue exploring
- Score < 87.00: Significant improvement, new champion
- Score < 85.00: Major breakthrough, focus on refinement
- Score < 80.00: Exceptional, prepare submission
