# Task 22233c11

## Pattern
Mark opposite diagonal corners of connected 3-blocks with 8s, extended by block size.

## Algorithm
Find connected groups of 3s using diagonal adjacency (BFS). For each group, compute the bounding
box. The 3s always form a diagonal pattern occupying two opposite corners (TL+BR or TR+BL).
Determine block size as half the bounding box dimensions. Place 8s at the OTHER two corners,
extended outward by block size, clipped to grid boundaries.

## Key Tricks
- `for s in S-V` eliminates `if s in V:continue` check (-19 bytes)
- `zip(*G)` extracts rows/cols for min/max in one line (-34 bytes)
- `b-a+1>>1` bit shift for divide by 2
- `o[t].__setitem__(k,8)` in list comp for assignment
- Single flat list comp with nested `for` loops
- `R>t>=0<=k<C` chained comparison for bounds check

## Evolution Summary (AlphaEvolve-Inspired)

10 generations, 40 mutations tested. Final: **474 bytes** (-15.7%, -88 bytes from initial 562)

### Key Discoveries
| Gen | Discovery | Bytes | Delta |
|-----|-----------|-------|-------|
| 4 | `for s in S-V` eliminates continue | 517 | -45 |
| 6 | `zip(*G)` for min/max extraction | 480 | -37 |
| 7 | Inline index ternary `[g[a][e]<1]` | 477 | -3 |
| 9 | Single flat list comprehension | 474 | -3 |

### Failed Approaches
- `in-1,0,1` without space (syntax error in Python 3.10+)
- Recursive flood fill (larger than BFS)
- `*_,=rs,cs=zip(*G)` unpack trick (unpack error)
- Direct `or` chaining for setitem (bounds check issues)

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 734 | Initial working solution |
| v2 | 562 | Basic golf (chained bounds, compact BFS) |
| v3 | 517 | Eliminate `if s in V:continue` |
| v4 | 480 | zip(*G) for min/max (-34 bytes breakthrough) |
| v5 | 474 | Single comprehension, inline ternary |
