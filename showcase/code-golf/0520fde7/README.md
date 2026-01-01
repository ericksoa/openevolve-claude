# Task 0520fde7: Grid Intersection

## Problem
Given a 3x7 grid divided by a column of 5s at position 4, compute the AND of the left (cols 0-2) and right (cols 5-6) halves. Output 2 where both halves have 1, otherwise 0.

## Solution Stats
- **Bytes**: 57
- **Score**: 2,443 points (2500 - 57)
- **Status**: Passing all tests

## Algorithm
For each cell (r,c) in the 3x3 output:
- output[r][c] = 2 if input[r][c] AND input[r][c+4] are both non-zero, else 0

## Solution
```python
solve=lambda g:[[2*r[c]*r[c+4]for c in(0,1,2)]for r in g]
```

## Key Golf Tricks Used
- Lambda instead of def (saves ~4 bytes)
- `(0,1,2)` tuple instead of `range(3)` (same length but clearer)
- `2*r[c]*r[c+4]` - multiplication gives 2 only when both are 1
- Direct row iteration `for r in g` instead of index-based
