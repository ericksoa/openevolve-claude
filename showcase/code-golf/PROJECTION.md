# Score Projection Tracker

## Current Status

| Metric | Value |
|--------|-------|
| Tasks Solved | 16 / 400 |
| Total Score | 35,153 |
| Avg Score/Task | 2,196 |

---

## Tasks by Difficulty

### Easy (< 100 bytes) - 5 tasks
| Task | Bytes | Score | Est. Winner | % of Winner |
|------|-------|-------|-------------|-------------|
| 0520fde7 | 57 | 2,443 | ~2,450 | 99.7% |
| 0d3d703e | 58 | 2,442 | ~2,450 | 99.7% |
| 007bbfb7 | 65 | 2,435 | ~2,450 | 99.4% |
| 1e0a9b12 | 69 | 2,431 | ~2,450 | 99.2% |
| 017c7c7b | 80 | 2,420 | ~2,445 | 99.0% |
| **Avg** | 66 | 2,434 | 2,449 | **99.4%** |

### Medium (100-300 bytes) - 6 tasks
| Task | Bytes | Score | Est. Winner | % of Winner |
|------|-------|-------|-------------|-------------|
| 05269061 | 113 | 2,387 | ~2,420 | 98.6% |
| 08ed6ac7 | 142 | 2,358 | ~2,400 | 98.3% |
| 0ca9ddb6 | 207 | 2,293 | ~2,380 | 96.3% |
| 00d62c1b | 219 | 2,281 | ~2,370 | 96.2% |
| 0962bcdd | 241 | 2,259 | ~2,370 | 95.3% |
| 025d127b | 266 | 2,234 | ~2,360 | 94.7% |
| **Avg** | 201 | 2,299 | 2,383 | **96.5%** |

### Hard (300-600 bytes) - 4 tasks
| Task | Bytes | Score | Est. Winner | % of Winner |
|------|-------|-------|-------------|-------------|
| 06df4c85 | 378 | 2,122 | ~2,300 | 92.3% |
| 0b148d64 | 454 | 2,046 | ~2,250 | 90.9% |
| a64e4611 | 523 | 1,977 | ~2,200 | 89.9% |
| 045e512c | 591 | 1,909 | ~2,150 | 88.8% |
| **Avg** | 487 | 2,014 | 2,225 | **90.5%** |

### Very Hard (600+ bytes) - 1 task
| Task | Bytes | Score | Est. Winner | % of Winner |
|------|-------|-------|-------------|-------------|
| 0e206a2e | 1384 | 1,116 | ~1,800 | 62.0% |
| **Avg** | 1384 | 1,116 | 1,800 | **62.0%** |

---

## Projection Model

Assuming task distribution: 200 easy, 120 medium, 60 hard, 20 very hard

| Tier | # Tasks | Our Avg | Projected | Winner Est. |
|------|---------|---------|-----------|-------------|
| Easy | 200 | 2,434 | 486,800 | 489,800 |
| Medium | 120 | 2,299 | 275,880 | 285,960 |
| Hard | 60 | 2,014 | 120,840 | 133,500 |
| V.Hard | 20 | 1,116 | 22,320 | 36,000 |
| **Total** | 400 | - | **905,840** | **945,260** |

**Projected Final: ~906,000 points (95.8% of winner)**

---

## Progress Over Time

```
Score Projection (thousands) vs Tasks Solved
│
960 ┤                                              ════════ Winner: 962k
    │
940 ┤
    │                                    ╭───────────────── Target: 906k
920 ┤                               ╭───╯
    │                          ╭───╯
900 ┤                     ╭───╯
    │                ╭───╯
880 ┤           ╭───╯
    │      ╭───╯
860 ┤ ●───╯
    │
    ┼────┬────┬────┬────┬────┬────┬────┬────┬────
    0    2    4    6    8   10   12   14   16  tasks

Current: 35,134 pts (16 tasks) → Projected: 906k
```

---

## Evolution Log

| # | Task | Difficulty | Bytes | Score | Running Total | Projection |
|---|------|------------|-------|-------|---------------|------------|
| 1 | 0520fde7 | Easy | 57 | 2,443 | 2,443 | 977k |
| 2 | 00d62c1b | Medium | 238 | 2,262 | 4,705 | 941k |
| 3 | a64e4611 | Hard | 523 | 1,977 | 6,682 | 890k |
| 4 | 017c7c7b | Easy | 80 | 2,420 | 9,102 | 912k |
| 5 | 007bbfb7 | Easy | 65 | 2,435 | 11,537 | 923k |
| 6 | 1e0a9b12 | Easy | 69 | 2,431 | 13,968 | 925k |
| 7 | 0ca9ddb6 | Medium | 207 | 2,293 | 16,261 | 927k |
| 8 | 0d3d703e | Easy | 58 | 2,442 | 18,703 | 934k |
| 9 | 05269061 | Medium | 113 | 2,387 | 21,090 | 936k |
| 10 | 08ed6ac7 | Medium | 142 | 2,358 | 23,448 | 935k |
| 11 | 0962bcdd | Medium | 241 | 2,259 | 25,707 | 929k |
| 12 | 025d127b | Medium | 266 | 2,234 | 27,941 | 924k |
| 13 | 06df4c85 | Hard | 378 | 2,122 | 30,063 | 916k |
| 14 | 0b148d64 | Hard | 454 | 2,046 | 32,109 | 910k |
| 15 | 045e512c | Hard | 591 | 1,909 | 34,018 | 902k |
| 16 | 0e206a2e | V.Hard | 1384 | 1,116 | 35,134 | 878k |

---

## ASCII Projection Graph

```
Projected Score vs Actual Progress
│
950k┤════════════════════════════════════════════════ Winner (962k)
    │
940k┤         ●
    │        ╱ ╲
930k┤   ●───●   ●───●
    │  ╱         ╲   ╲
920k┤ ●           ●   ●───●
    │╱                     ╲
910k●                       ●───●
    │                            ╲
900k┤                             ●
    │                              ╲
890k┤                               ●
    │                                ╲
880k┤                                 ● ← Current (16 tasks)
    │
    ┼──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬
    1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16

Note: Projection dropped after task 16 (very hard, 1384 bytes)
      Need more easy/medium tasks to recover projection
```

---

## Key Insights

1. **Easy tasks: 99.4%** - Nearly optimal, minimal room for improvement
2. **Medium tasks: 96.5%** - Good performance, 3-4% gap
3. **Hard tasks: 90.5%** - 10% gap, significant byte savings possible
4. **Very Hard tasks: 62%** - 0e206a2e at 1384 bytes is dragging down average

## Recommendations

1. **Focus on easy tasks** - Quick wins, high scores, stabilize projection
2. **Re-golf very hard task** - 0e206a2e needs major algorithm rework (1384→~600 bytes)
3. **Avoid very hard tasks** - They tank the projection; skip or defer
4. **Target medium tasks** - Best effort/reward ratio
