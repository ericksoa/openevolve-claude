# ARC Task 0e206a2e - Template Matching with Rotation

## Task Description

This ARC task involves pattern recognition and template matching with rotations/reflections.

**Pattern:**
1. The input contains multiple connected components made of a "filler" color (most common color)
2. Each component has corner markers of different colors at specific positions
3. Some corner markers appear isolated (not connected to filler cells)
4. The output places templates at the isolated marker positions, matching the orientation implied by the marker arrangement

## Solution Approach

1. **Find filler color**: Most frequently occurring non-zero color
2. **Find connected components**: BFS starting from filler cells, expanding to adjacent non-zero cells (8-connectivity)
3. **Identify templates**: Each connected component is a template with corner markers
4. **Find isolated markers**: Non-filler cells not part of any template
5. **Match templates to isolated markers**: For each isolated marker group, find which template orientation matches (8 possible rotations/reflections)
6. **Place transformed templates**: Apply the matching transformation and place at isolated marker positions

## Evolution Progress

| Generation | Bytes | Score |
|------------|-------|-------|
| 0 (working) | 4119 | 1 |
| 1 | 1968 | 532 |
| 2 | 1528 | 972 |
| 3 | 1460 | 1040 |
| 4 | 1469 | 1031 |
| 5 | 1454 | 1046 |
| 6 | 1409 | 1091 |
| 7 | 1404 | 1096 |
| 8 | 1406 | 1094 |
| 9 | 1398 | 1102 |
| 10 | 1384 | 1116 |

## Final Stats

- **Final bytes**: 1384
- **Final score**: 1116
- **Reduction**: 66.4% from working solution

## Key Golf Tricks

1. **Single-space indentation**: Using 1 space instead of 4 saves significant bytes
2. **Tuple unpacking**: `R,*_=K` instead of `R=list(K)[0]`
3. **Chained comparisons**: `h>n[0]>=0<=n[1]<w` for bounds checking
4. **List append shortcut**: `q+=n,` instead of `q.append(n)`
5. **Dict comprehension**: Inlining isolated marker detection
6. **Matrix transforms**: Using 4-tuples `(a,b,d,e)` for 8 rotation/reflection transformations
7. **Boolean multiplication**: `C+=[x]*bool(x)` to conditionally add
8. **set().union(*C)**: Faster than iterating with `|=`
9. **Short variable names**: Single letters for all variables
10. **Inline for loops**: Removing intermediate assignments where possible
