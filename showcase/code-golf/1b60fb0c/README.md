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
- `eval(str(g))` for deep copy (4 bytes shorter than `[r[:]for r in g]`)
- `O=lambda r:sorted(...)` to avoid repeated comprehensions
- `ax=wl-1+sy` handles both cap types in one expression
- `sy*(condition)` replaces `if sy and condition`

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| original | 933 | Broken solution (0/4 tests) |
| gen1 | 2211 | First correct solution (4/4) |
| gen2 | 1568 | Body detection fix |
| gen6 | 1491 | Major golf pass |
| gen8 | 1456 | eval(str(g)), lambda shortcuts |

## Challenges
- Complex pattern with 5 different mirroring rules based on row type
- Symmetric vs asymmetric caps require different axis and body detection
- Segment detection needed for gap > 1 mirroring
- Original solution was broken; required complete algorithm redesign

## Evolution Summary (AlphaEvolve-Inspired)

9 generations, ~20 mutations tested. Final: **1456 bytes** (score: 1044)

### Key Discoveries
| Gen | Discovery | Bytes | Delta |
|-----|-----------|-------|-------|
| 1 | Working solution | 2211 | baseline |
| 2 | Symmetric caps need `row ⊆ ci` check | 1568 | -643 |
| 6 | Inline lambdas, combine checks | 1491 | -77 |
| 8 | eval(str(g)), S=lambda | 1456 | -35 |

### Failed Approaches
- Over-aggressive inlining broke segment detection (gen9)
- Using `s>[])` instead of `if s:` was unreliable

## Potential Improvements
- Algorithm fundamentally complex (~1456 bytes minimum with current approach)
- May need radically different algorithm to reach ~600 bytes target
- Segment detection loop is verbose (could potentially use itertools.groupby)
- The 5-branch M calculation might be consolidatable
