# Gen83 Strategy: Crossover Combinations

## Parent Generations for Crossover

| Gen | Score | Key Innovation |
|-----|-------|----------------|
| 62 | 88.22 | Radius-based compression (all-time best) |
| 78b | 88.92 | Wave compaction with finer steps |
| 79b | 88.57 | Directional waves (X→Y→diagonal) |
| 80b | 88.44 | 5-phase cardinal (R→L→U→D→diagonal) |
| 82a | 88.62 | Inside-out processing (failed but interesting) |

## Crossover Candidates

### 83a: Bidirectional Waves (80b × 82a)
Combine outside-in (80b) and inside-out (82a) processing:
- First 3 waves: outside-in (far trees first)
- Last 2 waves: inside-out (close trees first)

Hypothesis: Outer trees settle first, then inner trees adjust to fill gaps.

### 83b: Wave + Rotation Jiggle (80b × 72c)
After each cardinal phase, try rotating boundary trees by ±45°:
- Do phase (e.g., move RIGHT)
- For trees that didn't move, try rotating ±45° and move again

Hypothesis: Rotation unlocks stuck positions.

### 83c: Alternating Phase Order (80b × 79b)
Vary the phase order each wave:
- Odd waves: R→L→U→D→diagonal (horizontal first)
- Even waves: U→D→R→L→diagonal (vertical first)

Hypothesis: Different orderings find different opportunities.

### 83d: Hybrid Phase Count (80b × 79b)
Combine 5-phase and 3-phase approaches:
- First 3 waves: 5-phase (R→L→U→D→diagonal)
- Last 2 waves: 3-phase (X→Y→diagonal) for final settling

Hypothesis: Different granularity at different stages helps.

## Priority Order
1. **83a** - Most unusual combination (bidirectional)
2. **83b** - Rotation is underexplored with wave compaction
3. **83c** - Low-risk order variation
4. **83d** - Hybrid granularity

## Execution Plan
Test 83a and 83b first (most novel crossovers).

---

## Results

| Candidate | Score | vs Baseline (88.44) | Notes |
|-----------|-------|---------------------|-------|
| **83a** | **88.22** | **+0.22 (IMPROVED!)** | **Bidirectional crossover works! Ties all-time best!** |
| 83b | 91.40 | -2.96 (regressed) | Rotation jiggle hurt significantly |
| 83c | - | - | Not tested |
| 83d | - | - | Not tested |

## Winner: Gen83a (BIDIRECTIONAL WAVE)

The crossover of Gen80b (outside-in) and Gen82a (inside-out) **achieved the all-time best score of 88.22!**

Key insight: Using outside-in for the first 3 waves settles outer trees, then switching to inside-out for the last 2 waves allows inner trees to adjust and fill gaps.

### Why Crossover Worked

- **Gen80b alone**: 88.44 (all outside-in)
- **Gen82a alone**: 88.62 (all inside-out)
- **Gen83a crossover**: 88.22 (3 outside-in + 2 inside-out)

Neither parent achieved this score individually. The combination is greater than the sum of its parts.

### Why Gen83b Failed

Adding rotation jiggle during wave compaction introduced too much instability. The rotation attempts likely disrupted the careful positioning achieved by the cardinal phases.

