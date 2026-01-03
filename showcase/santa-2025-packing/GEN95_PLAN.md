# Gen95 Plan: Global Optimization Approaches

## Current State
- **Champion**: Gen91b (rotation-first optimization)
- **Score**: 87.29 (lower is better)
- **Target**: ~69 (leaderboard top)
- **Gap**: 26.5%

## What's Been Exhaustively Tried (Don't Repeat)

### Gen92 (Parameter Tuning) - All Failed
- More rotations (16), finer precision (0.0005), more iterations
- Different wave passes, SA temperatures, cooling rates

### Gen93 (Algorithmic Changes) - All Failed
- Relocate moves, coarse-to-fine, aspect ratio penalty
- Force-directed compression, combined parameters

### Gen94 (Paradigm Shifts) - All Failed
- Multi-start (high variance, unreliable)
- Hexagonal grid seeding (no improvement)
- Genetic algorithm (crossover creates overlaps, slower)

## Gen95 Strategy: Global Optimization

The 26.5% gap suggests incremental approaches are fundamentally limited.
Gen95 tries **global optimization** - optimizing entire configurations at once.

### Gen95a: Full Configuration Simulated Annealing
**Rationale**: Instead of building incrementally and then refining, start with ALL trees placed and SA the entire configuration.

- For each n: initialize all n trees in a valid configuration (using current best)
- Run SA with much higher temperature and more iterations
- Move/rotate ANY tree, not just recently placed ones
- Accept moves that increase bounding box with SA probability

**Key change**: SA operates on complete configuration from the start.

### Gen95b: Tight Lattice Initialization
**Rationale**: Start from a theoretically dense arrangement.

- Calculate minimum possible bounding box (theoretical lower bound)
- Initialize trees on a tight rectangular lattice
- Trees may overlap initially
- Use repair + SA to resolve overlaps while minimizing expansion

**Key insight**: Start dense and expand, rather than start sparse and compact.

### Gen95c: Global Rotation Optimization
**Rationale**: Positions may be good but rotations suboptimal.

- Take champion's positions as fixed
- Try ALL 8^n rotation combinations for small n (n ≤ 8)
- For larger n, use SA on rotations only
- May find better rotation assignments without moving trees

**Key change**: Decouple position and rotation optimization.

### Gen95d: Center-First Spiral Placement
**Rationale**: Current approach places from origin outward. Try placing center trees first.

- Start with trees near center
- Spiral outward, placing outer trees last
- Outer trees have more room to adjust
- May result in more compact core

**Key change**: Reverse the placement order philosophy.

### Gen95e: Annealing Schedule Overhaul
**Rationale**: Current SA may cool too fast, missing global optima.

- Start at much higher temperature (2.0 vs 0.45)
- Use slower cooling (0.99998 vs 0.99993)
- Run 100k iterations (vs 28k)
- Accept larger moves at high temperature

**Key change**: More exploration before exploitation.

### Gen95f: Two-Phase Optimization
**Rationale**: Separate coarse placement from fine tuning.

Phase 1 (Coarse):
- Place trees with binary search precision 0.1 (not 0.001)
- Run fewer SA iterations
- Accept larger moves

Phase 2 (Fine):
- Take Phase 1 result
- Run SA with fine precision (0.001)
- Focus on boundary trees

**Key change**: Explicit coarse-to-fine phases with separate SA.

## Implementation Order

1. **Gen95e**: Annealing schedule overhaul (simplest change)
2. **Gen95a**: Full configuration SA (most promising)
3. **Gen95c**: Global rotation optimization (novel idea)
4. **Gen95d**: Center-first spiral (ordering experiment)

## Benchmark Protocol

For each candidate:
1. Build: `cargo build --release`
2. Test: `./target/release/benchmark 200 3`
3. If score < 87.29, verify with 5 runs
4. Update EVOLUTION_STATE.md with results

## Key Constraints (Don't Break)
- Keep discrete 45° angles
- Keep 6 parallel placement strategies
- Keep step sizes [0.10, 0.05, 0.02, 0.01, 0.005]
- Keep 5 wave passes total

## Success Criteria
Any candidate with score < 87.29 consistently (3+ runs) becomes new champion.
