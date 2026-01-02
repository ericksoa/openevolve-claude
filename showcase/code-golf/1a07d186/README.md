# Task 1a07d186

## Pattern
Project isolated cells toward their matching main line (vertical or horizontal), placing them one step adjacent to the line.

## Algorithm
Detect "main lines" - vertical columns or horizontal rows where all cells are the same non-zero color. For each cell, if it's on a main line, copy it to output. If it's an isolated cell matching a line's color (but not on the line), place a new cell adjacent to the line on the side toward the isolated cell's position. Cells not matching any line are removed.

## Key Tricks
- `len({...})<2` instead of `==1` for set uniqueness (saves 1 byte)
- `g[0][c]and len(...)` uses first cell as representative for line detection
- `j>c or-1` gives direction (+1 or -1) toward isolated cell
- `exec('o[i][n]=v')` shorter than `o[i].__setitem__(n,v)` for conditional assignment
- Walrus operator `:=` for inline bound calculation in condition chain
- `-1<n<C` for bounds check (same length as `0<=n<C` but consistent style)
- `I=range` alias saves 5 bytes per additional range() call

## Challenges
- Lines can be vertical OR horizontal - need to detect and handle both
- Each color can have at most one line type (V xor H in practice)
- Isolated cells must be projected toward the correct adjacent position
- Bounds checking required for edge cases (line at column 0 or last column)

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 635 | Initial solution with explicit loops |
| v4 | 454 | Combined loops, walrus operator, `.get()` lookup |
| v7 | 450 | Changed `>R//2` to `==R` (lines are full rows/cols) |
| v12 | 438 | Used `exec()` instead of if-statement |
| v14 | 436 | Set-based line detection `len({...})==1` |
| v15 | 434 | Changed `==1` to `<2` |

## Potential Improvements
- Finding a way to merge V and H detection into one comprehension
- Alternative bounds handling without explicit check
- Different algorithm that processes by line instead of by cell
