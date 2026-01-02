# Next Steps for Santa 2025 Packing Evolution

**Last Updated**: Gen84c (87.36 - ALL-TIME BEST)
**Previous Best**: Gen83a (88.22), Gen62 (88.22)

## Quick Start

```bash
cd rust
cargo build --release
./target/release/benchmark  # Test current champion
```

## Current Champion: Gen84c (87.36)

Key innovation: **Extreme 4+1 bidirectional wave split**
- First 4 waves: outside-in (far trees first)
- Last 1 wave: inside-out (close trees first)

## Gen85 Plan: Explore Split Ratio Limits

### Hypothesis
If 4+1 beat 3+2, would 5+0 with a post-wave inside-out pass be even better?

### Candidates to Test

#### 85a: Pure Outside-In + Final Inside-Out Pass
- 5 waves all outside-in (like Gen80b)
- Add a 6th wave that's inside-out
- Tests: Is 6 waves with 5+1 better than 5 waves with 4+1?

#### 85b: Aggressive 4+1 with Larger Steps
- Keep 4+1 split from Gen84c
- Increase step sizes: [0.15, 0.08, 0.04, 0.02, 0.01] instead of [0.10, 0.05, 0.02, 0.01, 0.005]
- Hypothesis: Larger initial steps may find bigger moves

#### 85c: 4+1 with More Wave Passes
- Increase wave_passes from 5 to 7
- Split as 6+1 (6 outside-in, 1 inside-out)
- Hypothesis: More iterations with same ratio

#### 85d: 4+1 with Stronger Center Pull
- Keep 4+1 split
- Increase center_pull_strength from 0.08 to 0.12
- Hypothesis: Stronger pull during SA + 4+1 wave = synergy

### Priority Order
1. **85a** - Tests if adding a 6th wave helps
2. **85b** - Quick parameter tweak to existing winner
3. **85c** - More waves with optimal ratio
4. **85d** - Parameter tuning on top of structural win

## Score Progression (for reference)

| Gen | Split | Score | Notes |
|-----|-------|-------|-------|
| 62 | N/A (radius compression) | 88.22 | Original best |
| 80b | 5+0 | 88.44 | All outside-in |
| 82a | 0+5 | 88.62 | All inside-out |
| 83a | 3+2 | 88.22 | First crossover success |
| **84c** | **4+1** | **87.36** | **CURRENT BEST** |

## What Works (Don't Break These)

1. **Discrete 45° angles in SA** - Continuous angles break the framework
2. **6 parallel placement strategies** - Diversity helps
3. **Hot restarts from elite pool** - Escape local optima
4. **Bidirectional wave processing** - Better than single direction
5. **Cardinal phase order** (R→L→U→D→diagonal) - Optimal sequence
6. **Outside-in dominant ratio** - 4+1 > 3+2 > 2+3

## What Doesn't Work (Avoid)

1. More SA iterations (diminishing returns)
2. Finer step sizes in wave compaction (too fine = noise)
3. More than 5 wave phases (undoes positioning)
4. Alternating O-I-O-I-O pattern (88.36, worse than 4+1)
5. Inverse split 2+3 (88.39, order matters)
6. Higher compression probability (88.98, 35% hurt)

## Files to Reference

- `mutations/gen84c_extreme_split.rs` - Current champion
- `mutations/gen83a_bidirectional_wave.rs` - Previous best crossover
- `GEN84_STRATEGY.md` - Full analysis of split ratios
- `GEN83_STRATEGY.md` - Crossover breakthrough details

## Commands Reference

```bash
# Test a mutation
cp mutations/gen85a_xxx.rs rust/src/evolved.rs
cd rust && cargo build --release && ./target/release/benchmark

# Generate visualization
./target/release/visualize
open packing_n200.svg

# Submit to Kaggle
./target/release/submit
kaggle competitions submit -c santa-2025 -f submission.csv -m "Gen85a: description"
```

## Target

- **Current**: 87.36
- **Leaderboard top**: ~69
- **Gap**: 26.6%

Each 1-point improvement is significant progress toward the leaderboard.
