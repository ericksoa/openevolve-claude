# Task 1f85a75f

## Pattern
Extract the bounding box region containing the least common non-zero color in the grid.

## Algorithm
Flatten the grid and find the rarest non-zero color using min with count. Then find all coordinates (i,j) where that color appears using enumerate. Unzip to get separate row and column indices, then slice the grid using min/max bounds on both axes.

## Key Tricks
- `E=enumerate` alias saves bytes when enumerate is used twice
- `[*filter(bool,sum(g,[]))]` flattens and filters zeros in one expression
- `sum(g,[])` to flatten 2D list instead of nested comprehension
- Walrus operator `:=` to assign and use in same expression
- `zip(*[(i,j)...])` to split coordinate pairs into R,C tuples
- List slicing `g[min(R):max(R)+1]` directly on input grid

## Evolution Summary (AlphaEvolve-Inspired)

10 generations, 40 mutations tested. Final: **182 bytes** (-25%, -61 bytes from initial 243)

### Key Discoveries
| Gen | Discovery | Bytes | Delta |
|-----|-----------|-------|-------|
| 1d | Slice-based extraction vs manual indexing | 217 | -26 |
| 2b | Combined P list with zip unpack | 200 | -17 |
| 3a | Inline zip in assignment | 195 | -5 |
| 4b | filter(bool,...) pattern | 190 | -5 |
| 7d | One-line def with semicolons | 186 | -4 |
| 9d | E=enumerate alias | 182 | -4 |

### Failed Approaches
- Lambda function (added overhead)
- sorted() for min/max (doesn't work for column bounds)
- Counter from collections (import overhead)
- min/max aliasing (not enough uses to justify)

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 243 | Initial working solution |
| v2 | 217 | Slice-based bounding box |
| v3 | 200 | zip(*P) unpack pattern |
| v4 | 195 | Inline zip assignment |
| v5 | 190 | filter(bool,...) pattern |
| v6 | 186 | One-line semicolon style |
| v7 | 182 | E=enumerate alias |
