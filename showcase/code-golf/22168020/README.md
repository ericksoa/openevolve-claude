# Task 22168020

## Pattern
Fill each row between the leftmost and rightmost occurrence of each non-zero color.

## Algorithm
For each row, find all unique non-zero colors. For each color, locate its first and
last occurrence using `r.index(c)` and `len(r)-r[::-1].index(c)`. Then use slice
assignment to fill that range with the color. The triangular diagonal patterns
become solid filled triangles.

## Key Tricks
- `{*r}-{0}` to get unique non-zero colors in one expression
- `r.index(c)` for first occurrence (cheaper than enumerate)
- `r[::-1].index(c)` to find last index via reversed list
- `len(r)-r[::-1].index(c)` gives last_index+1 (perfect for slice end)
- Slice assignment `r[a:b]=[c]*(b-a)` to fill in-place

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 114 | Walrus in slice (failed - RHS evaluated first) |
| v2 | 112 | Separate a,b assignment with semicolons |
| v3 | 118 | Tried for-loop (longer due to indentation) |
| v4 | 128 | Tried enumerate approach (too verbose) |
| v5 | 127 | Tried pure list comp (next() overhead) |

## Notes
- Each row can have multiple colors with non-overlapping spans
- The `r[::-1].index(c)` trick avoids expensive `max(i for i,x in enumerate(r)if x==c)`
- In-place modification is shorter than building a new grid
