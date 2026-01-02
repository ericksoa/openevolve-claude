# Task 1190e5a7

## Pattern
Count grid cells formed by separator lines and output uniform grid of that size.

## Algorithm
The input grid is divided by full-row and full-column "separator" lines (one color).
Find the separator color via `min(map(max,g))` (separator appears in every row).
The cell color is `sum({*g[0]})-d` since g[0] contains both colors and we subtract separator.
Output dimensions: (separator_rows+1) x (separator_cols+1), filled with cell color.

## Key Tricks
- `min(map(max,g))` finds separator (uniform rows have max=separator, all rows contain it)
- `{*g[0]}` gets both colors from first row (separator column runs through all rows)
- `sum({a,b})-d` extracts the other color when there are exactly 2
- `-~x` is shorter than `x+1` (bitwise not + negate)
- `g[:n]` for iteration instead of `range(n)` saves 5 bytes
- `len({*x})<2` detects uniform row/column (all same value)
- Lambda form saves 4 bytes over def

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 188 | Initial solution |
| v2 | 124 | AlphaEvolve-inspired evolution |

## Evolution Summary (AlphaEvolve-Inspired)

11 generations, ~44 mutations tested. Final: **124 bytes** (-34%, -64 bytes)

### Key Discoveries
| Gen | Discovery | Bytes | Delta |
|-----|-----------|-------|-------|
| 1b | `sum(g,[])` flatten + set difference | 177 | -11 |
| 2b | `min(map(max,g))` for separator color | 156 | -21 |
| 4d | `sum({*...})-d` for cell color | 151 | -5 |
| 5a | `min(map(max,g))` vs genexp | 145 | -6 |
| 6a | One-liner (inline d) | 138 | -7 |
| 7b | `-~` instead of `+1` | 136 | -2 |
| 8b | Lambda form | 132 | -4 |
| 9d | `g[:n]` instead of `range(n)` | 129 | -3 |
| 11d | `{*g[0]}` instead of flatten | 124 | -5 |

### Failed Approaches
- Lambda for uniform check (adds overhead)
- `max(r)==min(r)` longer than `len({*r})<2`
- List multiplication for height (wrong semantics)
- Variable extraction (adds overhead for single-use)
