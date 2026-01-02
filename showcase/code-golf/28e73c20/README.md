# Task 28e73c20

## Pattern
Generate a rectangular spiral maze pattern with 3s on a variable-size square grid of zeros.

## Algorithm
For each cell at position (r, c), compute the distance from the nearest edge: d = min(r, c, L-1-r, L-1-c).
The pattern consists of nested rectangular frames where:
- Even layers (d=0, 2, 4...) are filled with 3s, except a gap at (d+1, d) connecting to inner layers
- Odd layers (d=1, 3, 5...) are empty, except a path cell at (d+1, d) connecting layers
- The innermost even layer has no gap (no inner layer to connect to)

## Key Tricks
- `L=len(g)` exploits that all grids are square (saves 19 bytes vs h,w)
- `L+~r` for `L-1-r` using bitwise complement
- `~d%2` returns 1 for even d, 0 for odd d (d is even check)
- `r==d+1==c+1` chain comparison for path position
- `(...)and 3` instead of `3if...else 0` saves 2 bytes
- `d>=L-1>>1` checks if we're at innermost layer (M = (L-1)//2)

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 175 | Initial working solution with h,w variables |
| v2 | 170 | Eliminated M by inlining min(h,w)-1>>1 |
| v3 | 151 | Used L=len(g) since all grids are square |
| v4 | 149 | Changed `3if...else 0` to `(...)and 3` |
