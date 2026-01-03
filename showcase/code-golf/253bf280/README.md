# Task 253bf280: Connect 8s with 3s

## Pattern
Find pairs of 8s that share the same row or column and fill the cells between them with 3s.

## Algorithm
1. Find all positions of 8s in the grid
2. For each pair of 8s:
   - If same row (a==c): fill horizontal line between with 3s
   - If same column (b==d): fill vertical line between with 3s
3. Slicing handles directionality - only fills when first 8 comes before second

## Key Tricks
| Trick | Savings | Description |
|-------|---------|-------------|
| `eval(str(g))` | 1 byte | Shorter than `[*map(list,g)]` for deep copy |
| `d+~b` | 0 | Equivalent to `d-b-1` (complement trick) |
| `v>7` | 0 | Detects 8 (only non-zero value above 7) |
| `[*range()]*bool` | - | Conditional iteration via list multiplication |
| `E=enumerate` | 5 bytes | Alias for enumerate used 2x |

## Evolution Summary (8 generations)

| Gen | Attempt | Bytes | Result |
|-----|---------|-------|--------|
| 0 | Baseline | 208 | Working |
| 1c | `eval(str(g))` copy | **207** | **-1 byte** |
| 2-8 | Various optimizations | 207+ | No improvement |

**Plateau reached** after 3+ generations at 207 bytes.

### Mutations Tested
- Different copy methods: `eval(str(g))`, `[*map(list,g)]`, `[r[:]for r in g]`
- Different iteration: walrus operator, explicit assignment
- Different conditions: `v>7`, `v==8`, `7<v`
- exec tricks, __setitem__, nested comprehensions
- Semicolon chaining, ternary operators

## Byte History
| Version | Bytes | Score | Change |
|---------|-------|-------|--------|
| Initial | 208 | 2292 | - |
| Final | 207 | 2293 | -1 byte |
