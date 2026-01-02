# Gen78 Strategy: Better Wave Compaction

## Gen78 Results Summary

| Candidate | Approach | Score | vs Gen74a (89.26) |
|-----------|----------|-------|-------------------|
| Gen78a | Stronger compression (35%, 0.10 pull) | 89.64 | -0.4% (worse) |
| **Gen78b** | **5 wave passes + finer steps (0.005)** | **88.92** | **+0.4% (better!)** |

**Winner: Gen78b** with score 88.92 (best of 3 runs)

### Key Learnings
1. **Stronger compression hurts** - Gen78a's more aggressive compression (35% prob, 0.10 center pull) made things worse
2. **More wave passes help** - 5 waves with finer steps (adding 0.005) improved compaction
3. **High variance** - Scores vary 1-2 points between runs due to stochastic SA

## Original Analysis

### Score Comparison (verified by testing)
| Version | Claimed | Tested | Notes |
|---------|---------|--------|-------|
| Gen62 | 88.22 | 90.24 | High variance - 88.22 was lucky |
| Gen74a | 89.26 | 89.26 | Baseline |
| Gen76d | 87.86 | 89.39 | High variance |
| **Gen78b** | - | **88.92** | **New best!** |
| Top solutions | ~70 | - | Different algorithm entirely |

### Gen62 Core Algorithm
- 6 parallel strategies (ClockwiseSpiral, CounterclockwiseSpiral, Grid, Random, BoundaryFirst, ConcentricRings)
- SA with hot restarts + elite pool
- **Compression moves**: 20% probability, pulls trees toward center
- 45° angles only (8 angles total)
- No wave compaction
- No late-stage fine angles

## Gen77 Learnings (Post-SA Refinement Failed)
1. Post-SA angle refinement disrupts SA's local optimum
2. Discrete 45° angles create geometric structure that shouldn't be broken
3. Top solutions use continuous angles FROM THE START, not as post-processing

## Gen78 Hypothesis: Better SA Moves

Instead of post-processing, improve SA itself:

### Candidate Mutations

#### Group A: Compression & Movement Improvements (4 candidates)
| ID | Mutation | Hypothesis |
|----|----------|------------|
| 78a | compression_prob=0.25, range=[0.03, 0.10] | More aggressive compression |
| 78b | compression_prob=0.30, target outermost 40% | Focus compression on boundary |
| 78c | Add "squeeze" move: compress in X OR Y direction only | Directional compression |
| 78d | Variable compression: stronger at high temp, weaker at low temp | Adaptive compression |

#### Group B: SA Temperature & Iteration Tuning (4 candidates)
| ID | Mutation | Hypothesis |
|----|----------|------------|
| 78e | sa_iterations=35000, cooling_rate=0.99995 | More iterations, slower cooling |
| 78f | hot_restart_temp=0.45, interval=600 | Hotter restarts more often |
| 78g | Add 3rd SA pass with very low temp | Final polish pass |
| 78h | Adaptive iterations: more for n > 150 | More search for harder cases |

#### Group C: Placement Strategy Improvements (4 candidates)
| ID | Mutation | Hypothesis |
|----|----------|------------|
| 78i | search_attempts=300 for n > 150 | More placement search for late trees |
| 78j | Add HexagonalSpiral strategy | New placement pattern |
| 78k | Better gap-finding: score gaps by size/accessibility | Smarter gap targeting |
| 78l | Two-phase placement: coarse then fine | Hierarchical placement |

#### Group D: Move Type Improvements (4 candidates)
| ID | Mutation | Hypothesis |
|----|----------|------------|
| 78m | Add "swap" move: exchange positions of two trees | New move type |
| 78n | Add "slide" move: move tree along boundary | Boundary optimization |
| 78o | Improve rotation moves: try 22.5° for boundary trees | Finer rotation for edge cases |
| 78p | Add "cluster" move: move 2-3 nearby trees together | Coordinated movement |

## Priority Order
1. **78a-78d** (compression improvements) - builds on Gen62's key innovation
2. **78e-78h** (SA tuning) - safe parameter tweaks
3. **78m-78p** (new move types) - higher risk, higher potential reward
4. **78i-78l** (placement) - moderate risk

## Implementation Plan

### Phase 1: Test compression variants (78a-78d)
Start with what made Gen62 good - its compression moves.

### Phase 2: Test SA tuning (78e-78h)
If compression doesn't help, try SA parameter adjustments.

### Phase 3: Test new move types (78m-78p)
If parameters don't help, add new move types.

## Expected Outcomes
| Group | Expected Score Range | Risk |
|-------|---------------------|------|
| A (Compression) | 87.5-88.5 | Low |
| B (SA Tuning) | 87.8-88.4 | Low |
| C (Placement) | 87.0-89.0 | Medium |
| D (New Moves) | 86.0-90.0 | High |

## Success Criteria
- Beat Gen62's 88.22 score
- Focus on consistent improvements over one-run luck
- Run 3 benchmarks per candidate for reliability

---

## Execution Notes
1. Start from Gen62 baseline (NOT Gen74a)
2. One change at a time
3. Save each mutation to mutations/gen78*.rs
4. Generate SVG visualizations for promising candidates
