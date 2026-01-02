# Task 1caeab9d

## Pattern
Align all colored blocks vertically to the same row as block 1, preserving column positions.

## Algorithm
The grid contains several colored rectangular blocks (typically colors 1, 2, 4). Each block needs to
be shifted vertically so that its top row aligns with the top row of block 1 (the reference). Column
positions remain unchanged. This effectively "flattens" all blocks to the same horizontal band.

Steps:
1. Find all non-zero cells, group by color value
2. Use block 1's top row as the target row
3. For each color block: shift cells by (target_row - block_top_row)
4. Place cells in output grid

## Key Tricks
- `S.setdefault(v,[]).append((i,j))` groups cells by color in one pass
- `P[0][0]` gets top row of block (cells collected in row-order, so first = topmost)
- `S[1][0][0]` gets target row from block 1
- `len(r)` in comprehension reuses row variable vs `len(g[0])`
- Inline initialization `S={};o=...;[collect...]` saves newline bytes

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 280 | Initial solution with explicit min() |
| v2 | 213 | P[0][0] trick (row-order collection = sorted) |
| v3 | 208 | `len(r)` instead of `len(g[0])` |
| v4 | 207 | Inline initialization on one line |

## Evolution Summary (AlphaEvolve-Inspired)

10 generations, ~40 mutations tested. Final: **207 bytes** (-26%, -73 bytes)

### Key Discoveries
| Gen | Discovery | Bytes | Delta |
|-----|-----------|-------|-------|
| 1 | enumerate vs range | 263 | -17 |
| 2 | single loop comprehension | 250 | -30 |
| 3 | inline sr variable | 237 | -43 |
| 5 | **P[0][0] trick** (major breakthrough) | 213 | -67 |
| 6 | remove outer parens | 211 | -69 |
| 7 | `len(r)` instead of `len(g[0])` | 208 | -72 |
| 10 | inline initialization | 207 | -73 |

### Major Breakthrough: P[0][0] Trick (Gen 5)
The biggest savings came from realizing that cells are collected in row-order via enumerate(g),
so the first cell in each color's list `P[0][0]` is already the topmost row - no need for
`min(i for i,j in P)`. This saved 24 bytes in one mutation.

### Failed Approaches
- Lambda functions (adds significant overhead)
- Walrus operator for S initialization (scope issues)
- `r*0` for zero-filled rows (creates empty list, not zero-filled)
- Caching `S[1][0][0]` in variable (overhead exceeds savings)
