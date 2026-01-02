# Gen77 Strategy: GA + AlphaEvolve + Online Research

## Gen77 Results Summary (COMPLETED)

**All post-SA refinement attempts FAILED to improve on baseline:**

| Candidate | Approach | Score | vs Baseline |
|-----------|----------|-------|-------------|
| Gen76d (baseline) | None | ~87.86 | - |
| Gen77i | Post-SA angle refinement (during strategy) | 89.05 | -1.4% |
| Gen77j | Post-SA global rotation | (not tested solo) | - |
| Gen77l | Combined (angle + rotation + repositioning) | 89.27 | -1.6% |
| Gen77m | Final-only angle refinement | 88.61 | -0.9% |

**Key Learnings:**
1. **Post-SA refinement disrupts optimization**: The SA finds a local optimum specific to 45° angles. Changing angles afterwards breaks the delicate geometric balance.
2. **Discrete angles are a feature, not a bug**: The 45° constraint during SA helps avoid local minima and creates consistent structure.
3. **Strategy isolation matters**: Modifying intermediate packings during incremental building hurts subsequent placements.
4. **Top solutions likely use different fundamental approaches**: Continuous angles work for them because they optimize differently from the start, not as post-processing.

**Conclusion**: Gen76d remains the champion at 87.86. Future work should focus on:
- Different SA move strategies (not post-SA changes)
- Better initial placement strategies
- More search attempts or iterations
- Fundamentally different algorithms (NFP, LP-based)

---

## Executive Summary

Gen76d achieved **87.86** using 3-way crossover. Target is **~69** (27% gap remaining).
This document outlines a comprehensive Gen77 strategy combining:
1. Genetic Algorithm best practices (crossover, mutation, selection)
2. AlphaEvolve techniques (LLM-guided evolution, diversity, ensemble)
3. Online research insights (continuous rotation, NFP, hybrid methods)

---

## Part 1: Analysis of Current State

### What's Working (Keep)
| Feature | Evidence | Gen |
|---------|----------|-----|
| 6 parallel strategies | Consistently best | Gen47+ |
| ConcentricRings placement | First sub-90 | Gen47 |
| Binary search positioning | Fast, precise | Gen10+ |
| 45° SA angles | Stability | Gen31+ |
| Late-stage fine angles (15°) | 88.90→87.86 | Gen73c-76d |
| Wave compaction post-SA | Consistent gains | Gen72b |
| Hot restarts + elite pool | Escapes local optima | Gen28+ |

### Current Champion Parameters (Gen76d)
```rust
late_stage_threshold: 150,     // Last 25% get fine angles
fine_angle_step: 15.0,         // 24 angles for late stage
compression_prob: 0.25,        // 25% compression moves
center_pull_strength: 0.09,    // Moderate pull
wave_passes: 3,
sa_iterations: 28000,
sa_passes: 2,
elite_pool_size: 3,
```

### What Doesn't Work (Avoid)
- Continuous angles in SA (97+ scores)
- Global rotation during SA
- More than 3 wave passes
- More SA iterations without new ideas
- Kitchen sink approaches
- Two-tier angle complexity

---

## Part 2: Research Insights

### Top Solution Analysis (~70.1 score)
From [GitHub analysis](https://github.com/berkaycamur/Santa-Competition):
- Uses `adaptive_continuous_optimizer.py`
- GPU acceleration (`santa_optimizer_gpu.py`)
- Continuous angles (21°, 66°, etc. - NOT 45° multiples)
- Global rotation optimization (3° steps)
- n=200 side: 7.81 vs our 9.14 (15% smaller!)

### Key Gap Analysis
| Aspect | Our Best | Top Solution | Gap |
|--------|----------|--------------|-----|
| Score | 87.86 | ~70 | 25% |
| n=200 side | 9.14 | 7.81 | 15% |
| Angles | Discrete 45°/15° | Continuous | Fundamental |
| Global rotation | None | Yes | Missing |

### AlphaEvolve Techniques (from [DeepMind](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/))
1. **Ensemble approach**: Multiple models (fast for breadth, powerful for depth)
2. **Evolutionary database**: Store successful programs, use as context
3. **Mutation diversity**: Extensive changes across multiple components
4. **Quantifiable evaluation**: Objective metrics for selection

### Irregular Packing Literature (from [academic research](https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2022.966691/pdf))
1. **No-Fit Polygon (NFP)**: Geometric tool for valid placements
2. **Hybrid GA + SA**: Combine global search with local optimization
3. **Continuous rotation**: Key for dense packing
4. **Linear programming**: For local position refinement

---

## Part 3: Gen77 Strategy

### Philosophy: Staged Approach
Instead of trying to jump to continuous angles (which breaks SA), use a **staged approach**:

1. **Stage A**: Optimize current SA framework to its limit
2. **Stage B**: Add post-SA continuous angle refinement
3. **Stage C**: Add post-SA global rotation refinement

### Gen77 Candidates (12 total)

#### Group A: Crossover Refinements (4 candidates)
Build on Gen76d's success with targeted crossovers:

| ID | Parents | Change | Hypothesis |
|----|---------|--------|------------|
| 77a | Gen76d × Gen76f | threshold=150, slower cooling (0.999945) | Combine best two from Gen76 |
| 77b | Gen76d × Gen72b | threshold=150, compression=25%, 4 waves but smaller steps | More gentle waves |
| 77c | Gen76d tuned | threshold=145 | Fine-tune threshold down |
| 77d | Gen76d tuned | threshold=155 | Fine-tune threshold up |

#### Group B: AlphaEvolve-Inspired (4 candidates)
Apply AlphaEvolve's mutation strategies:

| ID | Technique | Change | Hypothesis |
|----|-----------|--------|------------|
| 77e | Multi-component mutation | Change 3 params at once: iterations=32k, cooling=0.99994, temp=0.48 | Coordinated changes like AlphaEvolve |
| 77f | Diversity injection | Add 7th strategy: "DiagonalSpiral" | New search direction |
| 77g | Adaptive evaluation | More search_attempts for n>180 (300 vs 200) | Harder problems need more search |
| 77h | Elite ensemble | elite_pool_size=5, keep top from different algorithm families | Diversity in restarts |

#### Group C: Post-SA Refinement (4 candidates) - KEY INNOVATION
**This is the breakthrough attempt** - add optimization AFTER SA:

| ID | Technique | Change | Hypothesis |
|----|-----------|--------|------------|
| 77i | **Post-SA angle gradient descent** | After SA, try ±2°, ±4°, ±6° on each tree | Refine angles without breaking SA |
| 77j | **Post-SA global rotation** | After SA, try rotating entire packing 0-15° in 1° steps | Find better global orientation |
| 77k | **Post-SA local repositioning** | After SA, gradient descent on x,y for each tree | Fine-tune positions |
| 77l | **Combined post-SA** | 77i + 77j + 77k in sequence | Full post-processing pipeline |

---

## Part 4: Implementation Details

### Group C Implementation (Most Important)

#### 77i: Post-SA Angle Gradient Descent
```rust
fn post_sa_angle_refinement(&self, trees: &mut Vec<PlacedTree>) {
    for idx in 0..trees.len() {
        let old_angle = trees[idx].angle_deg;
        let old_side = compute_side_length(trees);

        // Try small angle adjustments
        for delta in [2.0, -2.0, 4.0, -4.0, 6.0, -6.0] {
            let new_angle = (old_angle + delta).rem_euclid(360.0);
            trees[idx] = PlacedTree::new(trees[idx].x, trees[idx].y, new_angle);

            if has_overlap(trees, idx) {
                trees[idx] = PlacedTree::new(trees[idx].x, trees[idx].y, old_angle);
                continue;
            }

            let new_side = compute_side_length(trees);
            if new_side < old_side {
                // Keep improvement
                break;
            } else {
                // Revert
                trees[idx] = PlacedTree::new(trees[idx].x, trees[idx].y, old_angle);
            }
        }
    }
}
```

#### 77j: Post-SA Global Rotation
```rust
fn post_sa_global_rotation(&self, trees: &mut Vec<PlacedTree>) -> f64 {
    let original = trees.clone();
    let mut best_angle = 0.0;
    let mut best_side = compute_side_length(trees);

    // Try rotating entire packing
    for angle_deg in (0..=15).map(|i| i as f64) {
        let rotated = self.rotate_all_trees(&original, angle_deg);
        let side = compute_side_length(&rotated);

        if side < best_side {
            best_side = side;
            best_angle = angle_deg;
        }
    }

    if best_angle > 0.0 {
        *trees = self.rotate_all_trees(&original, best_angle);
    }
    best_angle
}

fn rotate_all_trees(&self, trees: &[PlacedTree], angle_deg: f64) -> Vec<PlacedTree> {
    let rad = angle_deg * PI / 180.0;
    let cos_a = rad.cos();
    let sin_a = rad.sin();

    trees.iter().map(|t| {
        let new_x = t.x * cos_a - t.y * sin_a;
        let new_y = t.x * sin_a + t.y * cos_a;
        let new_angle = (t.angle_deg + angle_deg).rem_euclid(360.0);
        PlacedTree::new(new_x, new_y, new_angle)
    }).collect()
}
```

### Selection Strategy (GA Best Practice)

Use **tournament selection** with diversity bonus:
```python
def select_next_generation(candidates, population_size=4):
    # 1. Always keep champion (elitism)
    selected = [best_candidate]

    # 2. Tournament selection for remaining slots
    for _ in range(population_size - 1):
        # Pick 3 random candidates
        tournament = random.sample(candidates, 3)

        # Score = fitness + diversity_bonus
        scores = []
        for c in tournament:
            fitness = -c.score  # Lower is better
            diversity = min_distance_to_selected(c, selected)
            scores.append(fitness + 0.1 * diversity)

        winner = tournament[argmax(scores)]
        selected.append(winner)

    return selected
```

### Crossover Strategy (GA Best Practice)

For Group A, use **uniform crossover** on parameters:
```python
def crossover(parent1, parent2):
    child = {}
    for param in PARAMS:
        if random() < 0.5:
            child[param] = parent1[param]
        else:
            child[param] = parent2[param]
    return child
```

---

## Part 5: Evaluation Plan

### Metrics to Track
1. **Primary**: Best score (lower is better)
2. **Secondary**: n=200 side length (key indicator)
3. **Tertiary**: Runtime (efficiency)

### Expected Outcomes

| Group | Expected Best | Reasoning |
|-------|---------------|-----------|
| A (Crossover) | 87.5-88.0 | Incremental improvement |
| B (AlphaEvolve) | 88.0-89.0 | May help, may not |
| C (Post-SA) | **85.0-87.0** | **Breakthrough potential** |

### Priority Order
1. **77l** (combined post-SA) - highest potential
2. **77i** (angle refinement) - safest of Group C
3. **77j** (global rotation) - addresses known gap
4. **77a** (best crossover) - solid baseline

---

## Part 6: Execution Checklist

### Before Starting
- [ ] Read this document
- [ ] Verify Gen76d is current champion in evolved.rs
- [ ] Create mutations/gen77*.rs files

### Implementation Order
1. [ ] Implement Group A (77a-77d) - simple parameter changes
2. [ ] Implement Group B (77e-77h) - moderate changes
3. [ ] Implement Group C (77i-77l) - structural changes
4. [ ] Evaluate all 12 candidates
5. [ ] Select top 4 for population
6. [ ] If breakthrough, iterate on Group C

### After Evaluation
- [ ] Update README.md with results
- [ ] Commit and push
- [ ] Submit best to Kaggle
- [ ] Plan Gen78 based on learnings

---

## Part 7: Key Insights Summary

### Why Post-SA Refinement Should Work
1. **SA with 45° angles is stable** - we've proven this works
2. **Continuous angles hurt SA** - too much search space
3. **Post-SA is safe** - SA is done, won't destabilize
4. **Top solutions use continuous** - they must be doing post-processing
5. **Small angle changes are local** - won't break overlaps much

### The Missing Piece
Our SA finds good STRUCTURE but leaves ~15% density on the table because:
- We only use 45° multiples during optimization
- We never try global rotation
- We never fine-tune after SA

Gen77 Group C addresses all three.

---

## References

- [Kaggle Competition](https://www.kaggle.com/competitions/santa-2025)
- [AlphaEvolve Blog](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
- [2D Irregular Packing Review](https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2022.966691/pdf)
- [Top Solution Repo](https://github.com/berkaycamur/Santa-Competition)
- [NFP Implementation](https://github.com/seanys/2D-Irregular-Packing-Algorithm)
