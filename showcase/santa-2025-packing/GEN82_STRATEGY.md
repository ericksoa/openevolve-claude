# Gen82 Strategy: Beyond Wave Decomposition

## Gen80b Baseline (88.44)
- 5-phase cardinal wave: R→L→U→D→diagonal
- Gen81 proved: more phases hurt, different order hurts
- Wave decomposition is exhausted

## What We've Learned

### Successful Patterns
| Pattern | Impact | Notes |
|---------|--------|-------|
| Discrete 45° angles | Essential | Continuous angles break SA |
| 5-phase cardinal wave | +0.48 vs 1-phase | Optimal decomposition |
| 6 parallel strategies | Good | Diversity helps |
| Hot restarts | Good | Escape local optima |
| Late-stage fine angles | Marginal | Helps n >= 140 |

### Failed Patterns
| Pattern | Impact | Notes |
|---------|--------|-------|
| More SA iterations | None/negative | Diminishing returns |
| Finer step sizes (0.002) | Negative | Too fine = noise |
| 8+ wave phases | Negative | Undoes positioning |
| Alternating phase order | Negative | Consecutive better |
| Continuous angles | Very negative | Breaks SA framework |

## Gen82 Candidates - Novel Approaches

### Group A: Wave Strategy Variations (Untested from Gen80/81)
| ID | Mutation | Hypothesis |
|----|----------|------------|
| 82a | Inside-out wave | Move center trees first, create structure from inside |
| 82b | Double passes (2x all phases) | More settling opportunities |

### Group B: Rotation During Compaction
| ID | Mutation | Hypothesis |
|----|----------|------------|
| 82c | Rotate during wave | Try angle adjustments while compacting |
| 82d | Rotation-first wave | Rotate all trees, then compact |

### Group C: Adaptive Targeting
| ID | Mutation | Hypothesis |
|----|----------|------------|
| 82e | Center-of-mass target | Use CoM instead of geometric center |
| 82f | Tight-axis preference | Compress shorter axis first |

### Group D: Multi-Pass Strategies
| ID | Mutation | Hypothesis |
|----|----------|------------|
| 82g | Wave + jiggle | Add small random jiggle after wave |
| 82h | Aggressive then gentle | Big steps first, refine with small |

## Priority Order
1. **82a** - Inside-out is inverse of proven approach (high novelty)
2. **82b** - Double passes is low-risk extension
3. **82c** - Rotation during wave could unlock positions
4. **82e** - Center-of-mass is principled alternative

## Risk Assessment
- 82a: Medium risk - inverts successful pattern
- 82b: Low risk - just more iterations
- 82c: Medium risk - adds complexity
- 82e: Low risk - minor targeting change

## Execution Plan
Test 82a (inside-out) and 82b (double passes) first.

---

## Results

| Candidate | Score | vs Baseline (88.44) | Notes |
|-----------|-------|---------------------|-------|
| 82a | 88.62 | -0.18 (regressed) | Inside-out doesn't help |
| 82b | 89.39 | -0.95 (regressed) | Double passes hurt, slower too |
| 82c | - | - | Not tested |

## Key Learnings

### 82a: Inside-Out Processing Failed
Moving center trees first (instead of outer trees first) slightly hurt. The outside-in approach of Gen80b is optimal - moving outer trees first creates compaction opportunities for inner trees.

### 82b: Double Passes Failed
Increasing wave passes from 5 to 10 significantly hurt (+0.95 regression) and added ~20s runtime. The diminishing returns have turned negative - more iterations create more noise without finding better positions.

## Conclusion

**Gen80b remains champion at 88.44.**

Wave compaction strategies are fully exhausted:
- Phase count: 5 is optimal (more hurts)
- Phase order: R→L→U→D→diagonal is optimal
- Processing order: outside-in is optimal
- Iteration count: 5 passes is optimal (more hurts)

Need completely different paradigm for Gen83 - perhaps:
- SA parameter exploration (risky, past attempts hurt)
- Different placement strategy combinations
- Rotation during compaction
- Or accept current plateau

