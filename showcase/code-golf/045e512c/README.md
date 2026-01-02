# ARC Task 045e512c - Pattern Replication with Direction Markers

## Pattern

Replicate a main pattern in directions indicated by marker cells of different colors.

## Algorithm

1. Group all non-zero cells by color into dict C
2. Find main pattern (color with most cells)
3. Extract bounding box and relative pattern positions
4. For each marker (non-main color):
   - Calculate direction from pattern center
   - Replicate pattern in that direction using marker's color

## Key Golf Tricks

1. **Flat indexing**: `for i in range(h*w)` with `i//w, i%w` instead of nested enumerate
2. **eval(str(g))**: Deep copy in 12 bytes vs 16 for `[r[:]for r in g]`
3. **Walrus operator**: `:=` for inline assignment
4. **Integer center trick**: `(2*u-a-b>2)` avoids float division for center comparison
5. **List comprehension with side effects**: Replace nested loops with single comprehension
6. **Inline D,E**: `(b-a+2)` and `(f-e+2)` inlined instead of stored in variables
7. **Bitwise or**: `dr|dc` shorter than `dr or dc`
8. **Bounds in comprehension**: Eliminate outer bounds check, handle inside

## Evolution Summary (AlphaEvolve-Inspired)

15 generations, ~60 mutations tested. Final: **486 bytes** (-105 bytes, -17.8%)

### Key Breakthroughs

| Gen | Discovery | Bytes | Delta |
|-----|-----------|-------|-------|
| 1 | `>1` instead of `>1.4` for direction | 579 | -12 |
| 2 | Inline ph/pw as (b-a+2) | 572 | -7 |
| 5 | Flat indexing `range(h*w)` | 559 | -13 |
| 6 | Integer center `A,B=a+b,e+f` | 553 | -6 |
| 8 | Store D,E variables | 550 | -3 |
| 9 | List comprehension for inner loop | 545 | -5 |
| 10 | Combine no A,B with list comp | 541 | -4 |
| 11 | Use `range(1,h)` instead of `range(1,50)` | 540 | -1 |
| 12 | Remove outer bounds check | 522 | -18 |
| 13 | Nested list comprehension | 492 | -30 |
| 14 | Inline D,E in comprehension | 488 | -4 |
| 15 | Flat comprehension `for s... for p,q...` | 486 | -2 |

### Failed Approaches

- `setdefault` for dict building (syntax error)
- `min(M)` / `max(M)` for bounding box (wrong semantics)
- `divmod(i,w)` (actually longer)
- Storing S,T for center sums (no savings)

## Byte History

| Version | Bytes | Score | Change |
|---------|-------|-------|--------|
| v1 | 2370 | 130 | Initial working solution |
| v2 | 591 | 1909 | Manual golf (pre-evolution) |
| v3 | 486 | 2014 | AlphaEvolve-inspired evolution (-105 bytes) |
