# Task 3c9b0459

## Pattern

Rotate a 3×3 grid 180 degrees.

## Algorithm

180° rotation = reverse row order + reverse each row. Using slice notation `[::-1]`
twice achieves this in a single list comprehension.

## Key Tricks

- `g[::-1]` - reverse list of rows
- `r[::-1]` - reverse each row
- `lambda` - saves 4 bytes vs `def` for single expression

## Byte History

| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 44 | `def solve(g):return[r[::-1]for r in g[::-1]]` |
| v2 | 40 | Lambda: `solve=lambda g:[r[::-1]for r in g[::-1]]` |

## Solution (40 bytes)

```python
solve=lambda g:[r[::-1]for r in g[::-1]]
```

Score: **2460 points** (2500 - 40)
