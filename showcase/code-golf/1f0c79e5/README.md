# Task 1f0c79e5

## Pattern
Extend diagonal rays from a 2×2 seed block, where 2-valued cells indicate ray directions.

## Algorithm
Find the 2×2 region containing non-zero cells. The cells with value 2 are direction markers - each 2 indicates a diagonal direction based on its position relative to the top-left corner of the block. For each marker, extend the entire shape diagonally in that direction (up-left, up-right, down-left, or down-right) until hitting grid boundaries. The shape is painted with the non-2 color.

## Key Tricks
- `(a>r0)*2-1` computes direction: -1 for top row, +1 for bottom row
- One-liner list comprehension with side effects via `.__setitem__`
- Chain comparison `9>(x:=...)>-1<(y:=...)<9` for bounds check with walrus operators
- `min(P)` returns lexicographically smallest tuple giving (r0, c0)
- `max(g[r][c] for r,c in P)` finds the color (always > 2)
- Set comprehension `{...}` for automatic deduplication

## Evolution Summary (AlphaEvolve-Inspired)

12 generations, ~40 mutations tested. Final: **261 bytes** (-4.4%, -12 bytes from baseline)

### Key Discoveries
| Gen | Discovery | Bytes | Delta |
|-----|-----------|-------|-------|
| 2 | `(a>r0)*2-1` direction calc | 271 | -2 |
| 9 | One-liner list comprehension | 263 | -8 |
| 10 | Remove outer brackets `[[...]]` → `[...]` | 261 | -2 |

### Failed Approaches
- Lambda conversion (walrus in comprehensions causes syntax errors)
- `exec()` with string loops (added overhead)
- `a-r0 or-1` trick (requires space, same length)
- `any()` instead of `[...]` (adds 3 chars)

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 273 | Initial working solution |
| v2 (gen2a) | 271 | Direction calc `(a>r0)*2-1` |
| v3 (gen9d) | 263 | One-liner with `__setitem__` |
| v4 (gen10c) | 261 | Single brackets for list comp |
