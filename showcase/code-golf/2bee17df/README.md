# Task 2bee17df

## Pattern
Fill cross lines from border intersection - where rows/columns have only border cells at edges.

## Algorithm
The grid has 8s and 2s forming L-shaped borders at opposite corners with 0s in the interior.
A row or column "qualifies" for filling if it has non-zero values ONLY at the first and last
positions (all interior cells are 0). For each cell, if it's 0 and either its row or column
qualifies, fill it with 3. This creates cross/line patterns extending from the boundary intersection.

## Key Tricks
- `l[0]*l[-1]` - check both endpoints are non-zero (product is truthy iff both are)
- `1>max(l[1:-1])` - check all interior values are 0 (max of 0s is 0 < 1)
- `[*zip(*g)][c]` - extract column c by transposing grid (shorter than list comprehension)
- `x or 3*(...)` - replace 0 with 3 if condition, else keep x
- `for r in g` - iterate rows directly instead of `range(len(g))`
- `for c,x in enumerate(r)` - get both index and value in one loop

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 221 | Initial approach checking first/last non-zero indices (wrong pattern) |
| v2 | 171 | Corrected to check only edge cells are non-zero |
| v3 | 167 | Used `max(l[1:-1])<1` instead of `sum()<1` |
| v4 | 166 | Used `1>max()` instead of `max()<1` |
| v5 | 135 | Iterate `for r in g` and use `[R[c]for R in g]` for columns |
| v6 | 132 | Use `[*zip(*g)][c]` for column extraction |
