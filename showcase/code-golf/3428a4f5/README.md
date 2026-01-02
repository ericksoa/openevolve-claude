# Task 3428a4f5

## Pattern
XOR halves by separator - find the row of 4s, compare top and bottom halves cell-by-cell, output 3 where values differ and 0 where same.

## Algorithm
The input grid has a separator row of all 4s in the middle. Split into top half (before separator) and bottom half (after separator). For each corresponding cell pair, output 3 if values differ (XOR-like behavior) or 0 if values match. The separator is always centered, so `len(g)//2` finds it.

## Key Tricks
- `len(g)//2` to find separator index (always centered)
- `g[-s:]` to get last s rows (shorter than `g[s+1:]`)
- `zip(*r)` to unpack paired rows in a single comprehension
- `(a!=b)*3` for conditional 3 output
- Double `zip` pattern: outer pairs rows, inner pairs cells

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 101 | Initial with `g.index(max(g))` |
| v2 | 98 | Semicolon one-liner |
| v3 | 95 | `zip(*r)` unpacking trick |
| v4 | 94 | `g[-s:]` instead of `g[s+1:]` |
| v5 | 88 | `len(g)//2` - separator always centered |
