# Gen81 Strategy: Extending the Decomposition Trend

## Gen80b Baseline (88.44)
- 5-phase wave: right→left→up→down→diagonal
- Each phase only moves trees in one direction toward center
- Trend: More phases = better compaction

## Pattern Recognition

| Gen | Phases | Score | Improvement |
|-----|--------|-------|-------------|
| 78b | 1 (diagonal only) | 88.92 | baseline |
| 79b | 3 (X→Y→diagonal) | 88.57 | +0.35 |
| 80b | 5 (R→L→U→D→diag) | 88.44 | +0.13 |

**Trend**: Each decomposition step gives diminishing but real returns.

## Gen81 Candidates

### Group A: More Directions
| ID | Mutation | Hypothesis |
|----|----------|------------|
| 81a | 8-directional (add NE, NW, SE, SW) | 8 directions may find more opportunities |
| 81b | 6-directional (R→L→U→D→NE→SW) | Add 2 diagonals without full 8 |

### Group B: Order Experiments
| ID | Mutation | Hypothesis |
|----|----------|------------|
| 81c | Alternating order (R→U→L→D→diag) | Interlacing X/Y may help |
| 81d | Reverse order (D→U→L→R→diag) | Maybe opposite order is better |

### Group C: Strategy Variations
| ID | Mutation | Hypothesis |
|----|----------|------------|
| 81e | Double passes (2x through all 5 phases) | More iterations per wave |
| 81f | Progressive steps per phase | Aggressive→gentle within each phase |

## Priority Order
1. **81a** - 8-directional is the natural extension of decomposition trend
2. **81c** - Alternating order is low-risk variation
3. **81b, 81e** - Lower priority variants

## Risk Assessment
- 81a: Medium risk - more compute, unclear if diagonals add value after cardinal
- 81b: Low risk - incremental addition
- 81c: Low risk - just order change
- 81e: Medium risk - 2x compute

## Execution Plan
Test 81a first (8-directional), then 81c if time permits.

---

## Results

| Candidate | Score | vs Baseline (88.44) | Notes |
|-----------|-------|---------------------|-------|
| 81a | 89.15 | -0.71 (REGRESSED) | 9 phases too many, diagonal phases interfere |
| 81b | - | - | Not tested |
| 81c | 89.68 | -1.24 (REGRESSED) | Alternating order worse than consecutive |

## Key Learnings

### 81a: 8-Directional Failed
Adding 4 diagonal direction phases (NE, NW, SE, SW) after 4 cardinal phases **hurts** performance. The diagonal phases undo some of the precise cardinal positioning.

### 81c: Alternating Order Failed
Changing from R→L→U→D to R→U→L→D (interlacing X/Y) made it worse. Consecutive same-axis passes are better than interlaced.

## Conclusion

**The 5-phase cardinal wave (R→L→U→D→diagonal) is optimal for this approach.**

The decomposition trend:
- 1 phase (78b): 88.92
- 3 phases (79b): 88.57 (+0.35)
- 5 phases (80b): 88.44 (+0.13)
- 9 phases (81a): 89.15 (-0.71) - TOO MANY
- 5 phases alt (81c): 89.68 (-1.24) - WRONG ORDER

**Gen80b remains champion at 88.44.**

Further wave compaction mutations unlikely to help. Need paradigm shift for Gen82.
