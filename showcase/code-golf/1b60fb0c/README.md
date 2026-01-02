# Task 1b60fb0c

## Pattern
Horizontal mirroring of 1s to create 2s around a vertical axis, with complex rules for cap detection and body extent.

## Algorithm
The input contains a shape with "caps" (horizontal bars at top/bottom) and a "body" (rows between caps). The task mirrors 1s to the left of an axis, placing 2s. The axis position and mirroring rules depend on whether caps are symmetric (same pattern) or asymmetric (different patterns).

Key components:
1. **Caps**: Contiguous rows with 3+ cells that don't reach the rightmost column
2. **Widest row**: Contiguous row reaching rightmost with most cells
3. **Body**: Rows between caps that receive 2s (determined by connectivity and gap rules)
4. **Axis**: `widest_leftmost` for symmetric caps, `widest_leftmost - 1` for asymmetric

## Key Tricks
- `eval(str(g))` for deep copy
- `O=lambda r:[c for c in range(C)if g[r][c]]` - no sorted needed
- `S=lambda r:set(O(r))` - set wrapper
- `G=lambda X:(len(X)>1)*(X[-1]-X[0]+1-len(X))` - gap using multiplication
- `ax=wl-1+sy` handles both cap types in one expression
- `sy*rg>sy` instead of `sy and rg>1`
- Combined while loop with direction variable `d in-1,1`
- Chained comparisons: `X[-1]==rm>len(wo)-len(X)`

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| original | 933 | Broken solution (0/4 tests) |
| gen1 | 2211 | First correct solution (4/4) |
| gen8 | 1456 | eval(str(g)), lambda shortcuts |
| gen9f | 1266 | Remove segment detection |
| gen11c | 1148 | Combined while loop |
| gen14c | 1042 | Remove sorted() |
| gen17c | 1026 | sy*rg>sy trick - **Champion** |

## Challenges
- Complex pattern with 4+ different mirroring rules based on row type
- Symmetric vs asymmetric caps require different axis and body detection
- Original solution was broken; required complete algorithm redesign

## Evolution Summary (AlphaEvolve-Inspired)

18 generations, ~40 mutations tested. Final: **1026 bytes** (score: 1474)

### Key Breakthroughs
| Gen | Discovery | Bytes | Delta |
|-----|-----------|-------|-------|
| 2 | Symmetric caps need `row subset of ci` check | 1568 | -643 |
| 9f | Segment detection not needed (or 2 handles it) | 1266 | -302 |
| 11c | Combined while loop with direction variable | 1148 | -118 |
| 14c | sorted() unnecessary for column indices | 1042 | -106 |
| 17c | `sy*rg>sy` trick, th replaces cw | 1026 | -16 |

### Failed Approaches
- Over-aggressive inlining broke segment detection (gen9)
- Trying to simplify M branches broke asymmetric cases
- Flood fill approach was too simple for complex rules
