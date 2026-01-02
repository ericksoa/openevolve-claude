# Task 28bf18c6

## Pattern
Extract the non-background shape's bounding box and tile it 2x horizontally.

## Algorithm
Transpose the grid to work with columns as rows, filter out all-zero columns using
`filter(any, ...)`, transpose back to get the horizontally-cropped shape, then
filter out all-zero rows and duplicate each remaining row with `*2`.

## Key Tricks
- `zip(*g)` to transpose grid (columns become rows)
- `filter(any, ...)` shorter than `[c for c in ... if any(c)]` (18 vs 30 chars)
- Double transpose to crop both dimensions
- `[*r]*2` to convert tuple to list and tile horizontally

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 108 | Initial solution with walrus operator for column bounds |
| v2 | 102 | Switch to lambda |
| v3 | 77 | Transpose-filter-transpose approach |
| v4 | 67 | Use `filter(any,...)` instead of list comprehension |
