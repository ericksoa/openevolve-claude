# Score Projection Tracker

## Current Status

| Metric | Value |
|--------|-------|
| Tasks Solved | 41 / 400 |
| Total Score | 89,621 |
| Avg Score/Task | 2,186 |
| Projected Final | ~875,000 (90.9% of winner) |

---

## Tasks by Difficulty

### Easy (< 100 bytes) - 7 tasks
| Task | Bytes | Score | Est. Winner | % of Winner |
|------|-------|-------|-------------|-------------|
| 0520fde7 | 57 | 2,443 | ~2,450 | 99.7% |
| 0d3d703e | 58 | 2,442 | ~2,450 | 99.7% |
| 007bbfb7 | 65 | 2,435 | ~2,450 | 99.4% |
| 29c11459 | 68 | 2,432 | ~2,450 | 99.3% |
| 1e0a9b12 | 69 | 2,431 | ~2,450 | 99.2% |
| 27a28665 | 70 | 2,430 | ~2,450 | 99.2% |
| 017c7c7b | 80 | 2,420 | ~2,445 | 99.0% |
| **Avg** | **67** | **2,433** | **2,449** | **99.4%** |

### Medium (100-300 bytes) - 19 tasks
| Task | Bytes | Score | Est. Winner | % of Winner |
|------|-------|-------|-------------|-------------|
| 05269061 | 113 | 2,387 | ~2,420 | 98.6% |
| 137eaa0f | 130 | 2,370 | ~2,400 | 98.8% |
| 1cf80156 | 138 | 2,362 | ~2,400 | 98.4% |
| 08ed6ac7 | 142 | 2,358 | ~2,400 | 98.3% |
| 09629e4f | 170 | 2,330 | ~2,380 | 97.9% |
| 239be575 | 170 | 2,330 | ~2,380 | 97.9% |
| 1b2d62fb | 170 | 2,330 | ~2,380 | 97.9% |
| 10fcaaa3 | 176 | 2,324 | ~2,380 | 97.6% |
| 1190e5a7 | 188 | 2,312 | ~2,370 | 97.6% |
| 363442ee | 205 | 2,295 | ~2,360 | 97.2% |
| 0ca9ddb6 | 207 | 2,293 | ~2,360 | 97.2% |
| 1e32b0e9 | 207 | 2,293 | ~2,360 | 97.2% |
| 00d62c1b | 219 | 2,281 | ~2,350 | 97.1% |
| 0dfd9992 | 239 | 2,261 | ~2,340 | 96.6% |
| 0962bcdd | 241 | 2,259 | ~2,340 | 96.5% |
| 1c786137 | 249 | 2,251 | ~2,330 | 96.6% |
| 025d127b | 266 | 2,234 | ~2,320 | 96.3% |
| 32597951 | 274 | 2,226 | ~2,310 | 96.4% |
| 1caeab9d | 280 | 2,220 | ~2,300 | 96.5% |
| **Avg** | **199** | **2,301** | **2,357** | **97.6%** |

### Hard (300-600 bytes) - 10 tasks
| Task | Bytes | Score | Est. Winner | % of Winner |
|------|-------|-------|-------------|-------------|
| 178fcbfb | 304 | 2,196 | ~2,280 | 96.3% |
| 05f2a901 | 326 | 2,174 | ~2,260 | 96.2% |
| 11852cab | 333 | 2,167 | ~2,250 | 96.3% |
| 06df4c85 | 378 | 2,122 | ~2,220 | 95.6% |
| 1bfc4729 | 406 | 2,094 | ~2,200 | 95.2% |
| 0b148d64 | 454 | 2,046 | ~2,150 | 95.2% |
| 2bcee788 | 465 | 2,035 | ~2,140 | 95.1% |
| a64e4611 | 523 | 1,977 | ~2,100 | 94.1% |
| 0a938d79 | 539 | 1,961 | ~2,080 | 94.3% |
| 045e512c | 591 | 1,909 | ~2,050 | 93.1% |
| **Avg** | **432** | **2,068** | **2,173** | **95.1%** |

### Very Hard (600+ bytes) - 5 tasks
| Task | Bytes | Score | Est. Winner | % of Winner |
|------|-------|-------|-------------|-------------|
| 1a07d186 | 635 | 1,865 | ~2,000 | 93.3% |
| 2dd70a9a | 673 | 1,827 | ~1,950 | 93.7% |
| 150deff5 | 684 | 1,816 | ~1,950 | 93.1% |
| 1b60fb0c | 933 | 1,567 | ~1,800 | 87.1% |
| 0e206a2e | 1,384 | 1,116 | ~1,400 | 79.7% |
| **Avg** | **862** | **1,638** | **1,820** | **90.0%** |

---

## Projection Model

Based on 41 solved tasks with tier distribution:

| Tier | Solved | Our Avg | Assumed # | Projected | Winner Est. |
|------|--------|---------|-----------|-----------|-------------|
| Easy | 7 | 2,433 | 180 | 437,940 | 440,820 |
| Medium | 19 | 2,301 | 140 | 322,140 | 329,980 |
| Hard | 10 | 2,068 | 60 | 124,080 | 130,380 |
| V.Hard | 5 | 1,638 | 20 | 32,760 | 36,400 |
| **Total** | **41** | **2,186** | **400** | **916,920** | **937,580** |

**Conservative estimate (current avg × 400)**: 2,186 × 400 = **874,400 points**

**Optimistic estimate (tier-weighted)**: **916,920 points** (if we maintain tier averages)

---

## Improvement Opportunities

### Low-Hanging Fruit (re-golf candidates)

Tasks with byte counts significantly above tier average:

| Task | Current | Tier Avg | Gap | Potential Savings |
|------|---------|----------|-----|-------------------|
| 0e206a2e | 1,384 | 862 | +522 | High priority |
| 1b60fb0c | 933 | 862 | +71 | Medium priority |
| 2dd70a9a | 673 | 862 | -189 | ✓ Optimized! |
| 045e512c | 591 | 432 | +159 | Medium priority |
| 0a938d79 | 539 | 432 | +107 | Medium priority |
| a64e4611 | 523 | 432 | +91 | Medium priority |

### Score Impact of Re-golfing

If we could reduce:
- `0e206a2e`: 1384→600 bytes = +784 points
- `1b60fb0c`: 933→600 bytes = +333 points
- `045e512c`: 591→400 bytes = +191 points

**Total potential gain: ~1,300+ points**

### Recent Re-golf Wins
- `2dd70a9a`: 1163→673 bytes = **+490 points** (42% reduction)

---

## Key Insights

1. **Easy tasks: 99.4%** - Nearly optimal, minimal room for improvement
2. **Medium tasks: 97.6%** - Good performance, 2-3% gap
3. **Hard tasks: 95.1%** - 5% gap, some byte savings possible
4. **Very Hard tasks: 90.0%** - 10% gap, 2dd70a9a now optimized!

## Recommendations

1. **Re-golf very hard tasks** - 0e206a2e and 1b60fb0c need major rework
2. **Scan for low-hanging fruit** - Check if new tricks apply to older solutions
3. **Focus on medium difficulty new tasks** - Best effort/reward ratio
4. **Apply 2dd70a9a tricks** - Walrus, tuple indexing, __setitem__ tricks
