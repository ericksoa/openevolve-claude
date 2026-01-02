# Task 2dd70a9a

## Pattern
Connect two colored marker pairs (2s and 3s) with a U-shaped path of 3s.

## Algorithm
Find two 2-cell marker segments (red=2, green=3) in the grid. Depending on their relative positions:
1. **G<3 branch (close/overlapping columns)**: Extend both markers horizontally to a common column V, then connect with a vertical spine
2. **else branch (far apart columns)**: Draw an L-shaped connection with horizontal spine at row K, calculated as 1/3 of the way between markers

The markers can be horizontal or vertical segments. The path avoids overwriting existing colored cells (8s).

## Key Tricks
- `[]in[p:=F(2),q:=F(3)]` - walrus operator + empty list check in one expression
- `[sorted({z[k]for z in t})for t in[p,q]for k in(0,1)]` - extract sorted rows/cols for both markers in single comprehension
- `o[i][j]or o[i].__setitem__(j,3)` - short-circuit assignment only when cell is 0
- `I=range;m=min;x=max` - alias builtins that are used multiple times
- `-~x` instead of `x+1` and `~-C` instead of `C-1`
- `x(*b,*d)` instead of `max(b[-1],d[-1])` - unpack to get max of all elements
- `(c[-1],a[-1])[u]` - tuple indexing for conditional selection (shorter than ternary)
- `G*(G>0)` - zero when G<=0, else G (avoids if/else)
- Reuse computed values: `X`, `Z` used in both K formula and loop ranges

## Challenges
- Two distinct branches with complex coordinate calculations
- Need to handle both horizontal and vertical marker orientations
- U-shape requires two corners, not just one L-corner
- Must avoid overwriting existing 8s in grid

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 1163 | Original working solution |
| v2 | 906 | Removed redundant F lambda, used set comprehension |
| v3 | 821 | Used `__setitem__` helper D, list comp loops |
| v4 | 774 | Shorter variable names (a,b,c,d) |
| v5 | 728 | Inlined T lambda into single comprehension |
| v6 | 714 | `x(*b,*d)` unpacking for max |
| v7 | 705 | Clamped V, simpler D with `or` |
| v8 | 702 | Walrus operator for p,q assignment |
| v9 | 685 | Tuple indexing `(x,y)[u]` instead of `[x,y][u]` |
| v10 | 672 | Reuse X,Z variables in else branch |
| v11 | 673 | Final (includes trailing newline) |

## Potential Improvements
- The two branches might be unifiable with more clever coordinate transforms
- The sorted() calls could potentially be avoided if we track min/max directly
- The K formula `X+(Z-X)//3` might have a shorter equivalent
