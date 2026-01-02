# Gen84 Strategy: Crossover Experiments

## Parent Generations for Crossover

| Gen | Score | Key Innovation |
|-----|-------|----------------|
| 62 | 88.22 | Radius-based compression during SA (all-time best) |
| 83a | 88.22 | Bidirectional waves (3 out-in + 2 in-out, ties best) |
| 80b | 88.44 | 5-phase cardinal wave (R→L→U→D→diagonal) |
| 82a | 88.62 | Inside-out processing (failed alone, but useful in crossover) |

## Key Insight from Gen83

Gen83a proved that **partial strategies can be combined** more effectively than full strategies:
- Gen80b (100% outside-in): 88.44
- Gen82a (100% inside-out): 88.62
- **Gen83a (60% outside-in + 40% inside-out): 88.22**

This suggests exploring **different split ratios and orderings**.

## Gen84 Crossover Candidates

### 84a: Inverse Split (2+3 instead of 3+2)
Try inside-out first, then outside-in (inverse of 83a):
- First 2 waves: inside-out (close trees first)
- Last 3 waves: outside-in (far trees first)

Hypothesis: Maybe settling inner trees first, then outer trees, works even better.

### 84b: Alternating Pattern (O-I-O-I-O)
Alternate between outside-in and inside-out each wave:
- Wave 1: outside-in
- Wave 2: inside-out
- Wave 3: outside-in
- Wave 4: inside-out
- Wave 5: outside-in

Hypothesis: Interleaving creates better settling opportunities.

### 84c: 4+1 Extreme Split
Try more extreme ratio:
- First 4 waves: outside-in
- Last 1 wave: inside-out (final settling pass)

Hypothesis: One final inside-out pass for gap filling.

### 84d: Stronger Compression + Bidirectional
Increase compression probability from 20% to 35% combined with bidirectional wave:
- compression_prob: 0.35 (was 0.20)
- Keep bidirectional wave (3+2)

Hypothesis: More aggressive compression during SA + bidirectional wave = better.

## Priority Order
1. **84b** - Alternating is most novel combination
2. **84a** - Inverse split tests whether order matters
3. **84c** - Extreme ratio as control
4. **84d** - Parameter crossover with structural crossover

## Execution Plan
Test 84a and 84b first (structural crossovers), then 84c and 84d.

---

## Results

| Candidate | Score | vs Baseline (88.22) | Notes |
|-----------|-------|---------------------|-------|
| 84a | 88.39 | -0.17 (regressed) | Inverse split slightly worse |
| 84b | 88.36 | -0.14 (regressed) | Alternating didn't help |
| **84c** | **87.36** | **+0.86 (NEW BEST!)** | **Extreme 4+1 split wins!** |
| 84d | 88.98 | -0.76 (regressed) | Stronger compression hurt |

## Winner: Gen84c (EXTREME 4+1 SPLIT)

The 4+1 extreme split achieved a **new all-time best score of 87.36!**

### Why 4+1 Worked

The pattern suggests:
- **More outside-in waves are better**: 4 outside-in waves settle outer trees more thoroughly
- **One final inside-out pass is enough**: Just one wave for inner tree gap-filling
- **Extreme ratios beat balanced ratios**: 4+1 > 3+2 > 2+3

### Score Progression

| Gen | Split | Score | Improvement |
|-----|-------|-------|-------------|
| 80b | 5+0 (all outside-in) | 88.44 | baseline |
| 82a | 0+5 (all inside-out) | 88.62 | -0.18 |
| 83a | 3+2 | 88.22 | +0.22 |
| **84c** | **4+1** | **87.36** | **+1.08** |

### Key Insight

The extreme 4+1 split combines:
1. **4 waves of outside-in**: Settles outer trees thoroughly first
2. **1 wave of inside-out**: Final pass fills remaining gaps

This is more effective than the balanced 3+2 approach from Gen83a.
