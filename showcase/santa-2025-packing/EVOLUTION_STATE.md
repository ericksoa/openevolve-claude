# Evolution State - Gen91 Complete

## Current Champion
- **Gen91b** (rotation-first optimization)
- **Score: 87.29** (lower is better)
- Location: `rust/src/evolved.rs`

## Gen91 Results Summary

Generation 91 explored multiple paradigm shifts after Gen90 plateau. Found improvement with rotation-first optimization.

| Candidate | Score | Strategy | Result |
|-----------|-------|----------|--------|
| Gen91a (size-ordered) | N/A | Place larger trees first | SKIPPED - all trees identical |
| Gen91b (rotation-first) | **87.29** | Exhaustive 8-rotation search at each position | **NEW CHAMPION** |
| Gen91c (BLF hybrid) | 88.62 | Bottom-left-fill placement strategy | REJECTED |
| Gen91d (SA temp tuning) | 87.64 | Higher initial temp (0.45→0.60) | REJECTED |
| Gen91e (SA iterations) | 88.32 | More iterations (28k→35k) | REJECTED |
| Gen91f (more search) | 87.99 | More search attempts (200→300) | REJECTED |

## Gen91b Key Innovation

The rotation-first optimization fundamentally changes placement:

**Before (Gen87d):**
- For each direction, try each rotation separately
- Find closest valid position per rotation
- Compare results

**After (Gen91b):**
- For each direction, find approximate valid distance (any rotation)
- At that position, try ALL 8 rotations with fine-tuned positioning
- This finds better (position, rotation) pairs that fit tighter

## Performance Summary
- Previous champion (Gen87d): 88.03
- New champion (Gen91b): 87.29
- **Improvement: 0.74 points (0.8%)**
- Target (leaderboard top): ~69
- Gap to target: 26.5%

## What We Learned in Gen91
1. **Rotation-first helps**: Exhaustive rotation search at each position finds better fits
2. **BLF doesn't help**: Adding bottom-left-fill strategy increased variance and hurt score
3. **SA parameter tuning marginal**: Higher temp and more iterations didn't improve
4. **More search attempts**: Marginal or no improvement with increased compute

## Next Directions (Gen92+)

Consider:
1. **Fine-tune rotation-first**: Try 16 or 24 rotation angles instead of 8
2. **Position refinement passes**: After initial placement, try local position adjustments
3. **Improved gap filling**: Better detection and filling of internal gaps
4. **Specialized high-N handling**: Different strategy for n > 150

## File Locations
- Champion code: `rust/src/evolved.rs`
- Champion backup: `rust/src/evolved_champion.rs`
- Benchmark: `cargo build --release && ./target/release/benchmark 200 3`
