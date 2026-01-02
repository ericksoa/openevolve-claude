# ARC Task 045e512c - Pattern Replication with Direction Markers

## Task Description

Given a grid containing:
1. A main pattern (the color with the most cells)
2. Marker cells of different colors positioned around the pattern

The transformation replicates the main pattern in directions indicated by each marker cell:
- Each marker cell indicates a direction from the pattern center (up, down, left, right, or diagonal)
- The pattern is replicated in that direction using the marker's color
- Replication continues until reaching the grid edge, with gaps between copies

## Evolution Summary

| Gen | Bytes | Score | Key Changes |
|-----|-------|-------|-------------|
| 0 | 2370 | 130 | Initial working solution with comments |
| 1 | 1112 | 1388 | Remove comments, short variable names |
| 2 | 763 | 1737 | Use enumerate, semicolons, compact conditionals |
| 3 | 754 | 1746 | Walrus operator, chained comparisons |
| 4 | 700 | 1800 | zip(*M) for coord extraction, simplified direction calc |
| 5 | 670 | 1830 | Bitwise or for direction check |
| 6 | 621 | 1879 | for loop with range instead of while |
| 7 | 613 | 1887 | Inline setitem with and operator |
| 8 | 611 | 1889 | More compact direction calculation |
| 9 | 595 | 1905 | Shorter variable names throughout |
| 10 | 591 | 1909 | -~ph trick for (ph+1) |

## Final Solution

**591 bytes, Score: 1909**

## Key Golf Tricks

1. **Walrus operator** (`:=`): Assign and use in same expression
2. **Bitwise OR** (`|`): Shorter than `or` for truthy checks
3. **Chained comparisons**: `-ph<nr<h>-pw<nc<w` instead of multiple `and`
4. **-~x trick**: `-~x` equals `x+1` for integers, saves 2 chars
5. **zip(*M)**: Unpack list of tuples into separate lists
6. **v-m**: Shorter than `v!=m` (truthy when non-zero)
7. **__setitem__ with and**: `cond and(o[r].__setitem__(c,v))` avoids if statement
8. **Single-char names**: Use all single-letter variable names
9. **Semicolon chaining**: Multiple statements on one line
10. **Tuple unpacking in comprehensions**: `{(r-a,c-e)for r,c in M}`

## Algorithm

1. Group all non-zero cells by color
2. Find main pattern (color with most cells)
3. Extract bounding box and relative pattern positions
4. For each marker cell of non-main colors:
   - Determine direction from pattern center
   - Replicate pattern in that direction until edge
   - Use marker's color for the replicated copies
