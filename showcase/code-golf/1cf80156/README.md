# Task 1cf80156

## Pattern
Extract the minimal bounding box containing all non-zero cells from a sparse grid.

## Algorithm
Scan the grid to find all coordinates (row, col) where cells are non-zero. Compute the
min/max row indices and min/max column indices. Return the rectangular subgrid defined
by these bounds - essentially cropping the grid to just the region containing colored cells.

## Key Tricks
- `E=enumerate` alias saves 4 bytes when enumerate used twice
- `zip(*...)` to transpose list of (i,j) pairs into separate row/col sequences
- Enumerate with `if v` to filter non-zero cells
- Slice with `[min(b):max(b)+1]` for both rows and columns

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 138 | Initial solution with zip+enumerate approach |
| v2 | 135 | Semicolon one-liner (gen1d) |
| v3 | 131 | E=enumerate alias (gen5a) |
| v4 | 130 | Remove trailing newline (gen9b) |

## Evolution Summary (AlphaEvolve-Inspired)

10 generations, ~40 mutations tested. Final: **130 bytes** (-6%, -8 bytes)

### Key Discoveries
| Gen | Discovery | Bytes | Delta |
|-----|-----------|-------|-------|
| 1 | Semicolon one-liner | 135 | -3 |
| 5 | E=enumerate alias | 131 | -4 |
| 9 | No trailing newline | 130 | -1 |

### Failed Approaches
- `any(r)` row detection (longer than collecting all coords)
- `sorted()` with P[0]/P[-1] (longer than zip+min/max)
- Lambda functions (added overhead)
- Caching min/max results (variable cost exceeds repeat calls)
- Sets instead of lists (no benefit for this pattern)
