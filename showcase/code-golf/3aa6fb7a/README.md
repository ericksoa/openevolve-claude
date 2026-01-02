# Task 3aa6fb7a

## Pattern

Find L-shaped patterns of 8s (3 cells) and add a 1 at the missing corner to complete a 2×2 square.

## Algorithm

For each cell with value 8 that forms the corner of an L-shape:
1. Check all 4 diagonal directions (d,e in ±1)
2. If orthogonal neighbors g[r+d][c] and g[r][c+e] are both 8
3. And the diagonal g[r+d][c+e] is 0 (not already 8)
4. Set the diagonal to 1

## Key Tricks

- `E=enumerate` - alias saves bytes when used twice
- `__setitem__` - modify grid in list comprehension
- `v>7` - shorter than `v==8` (grid only has 0s and 8s)
- `7>r+d>-1<c+e<7` - chained comparisons for bounds (hardcoded 7×7)
- `g[r+d][c]==8==R[c+e]>g[r+d][c+e]` - chain conditions with comparisons

## Byte History

| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 251 | Initial working solution |
| v2 | 224 | Use enumerate |
| v3 | 209 | Nested d,e loops |
| v4 | 194 | List comprehension with `__setitem__` |
| v5 | 188 | `v>7` instead of `v==8` |
| v6 | 178 | Hardcode grid size 7 |

## Solution (178 bytes)

```python
def solve(g):E=enumerate;[g[r+d].__setitem__(c+e,1)for r,R in E(g)for c,v in E(R)if v>7for d in(-1,1)for e in(-1,1)if 7>r+d>-1<c+e<7and g[r+d][c]==8==R[c+e]>g[r+d][c+e]];return g
```

Score: **2322 points** (2500 - 178)
