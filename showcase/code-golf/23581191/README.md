# Task 23581191

## Pattern
Draw cross lines (horizontal + vertical) through colored markers, with color 2 at intersections.

## Algorithm
Find the two markers (8 and 7) and draw full cross lines through each position.
For marker 8, fill its entire row and column with 8. For marker 7, fill its entire
row and column with 7. The four intersection points where the crosses meet are
replaced with color 2.

## Key Tricks
- `eval(str(g))` for deep copy (12 bytes)
- `v>7` to distinguish 8 from 7 (3 bytes vs `v==8` at 4 bytes)
- Chained assignment `o[i][k]=o[k][j]=v` fills row and column in one statement
- `I=enumerate` alias saves bytes with 4 uses

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 198 | Initial working solution |
