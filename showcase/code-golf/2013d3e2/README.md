# Task 2013d3e2

## Pattern
Extract the upper-left quadrant of a point-symmetric pattern.

## Algorithm
Find the bounding box of all non-zero cells in the grid. The pattern has point
symmetry (identical when rotated 180 degrees), so we extract just the upper-left
quadrant. The quadrant size is half the bounding box dimensions in each direction.

## Key Tricks
- `e=enumerate` alias saves 4 bytes (used twice)
- `zip(*[...])` to transpose list of (row, col) pairs into (rows, cols)
- `-~max(R)-r>>1` for `(max(R)-r+1)//2`: `-~x` is `x+1`, `>>1` is `//2`
- One-liner with semicolons saves newline/indentation bytes

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 166 | Initial working solution with multi-line format |
| v2 | 162 | One-liner format |
| v3 | 158 | `e=enumerate` alias |
| v4 | 156 | `-~max(R)-r>>1` instead of `(max(R)-r+1)//2` |
| v5 | 152 | Inline P into zip() |
