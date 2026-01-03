# Task 44f52bb0: Horizontal Symmetry Classification

## Pattern
Classify a 3x3 grid based on horizontal (left-right) symmetry of red (2) cells.

**Output:**
- `[[1]]` if grid is horizontally symmetric (each row is a palindrome)
- `[[7]]` if grid is NOT horizontally symmetric

## Algorithm
Check if first and last elements of each row are equal. In a 3x3 grid, this is sufficient to determine row palindrome status (middle element doesn't affect symmetry).

1. For each row, compute `first_element - last_element`
2. If ANY difference is non-zero, grid is not symmetric → output 7
3. If ALL differences are zero, grid is symmetric → output 1

## Solution (46 bytes)
```python
solve=lambda g:[[1+6*any(a-c for a,_,c in g)]]
```

## Key Tricks

| Trick | Before | After | Savings |
|-------|--------|-------|---------|
| Tuple unpacking | `r[0]-r[2]` | `a-c for a,_,c in g` | 1 byte |
| Arithmetic vs conditional | `7 if ... else 1` | `1+6*any(...)` | 5 bytes |
| Difference instead of equality | `a!=c` (bool) | `a-c` (int, works with any()) | 1 byte |

## Byte History

| Version | Bytes | Change |
|---------|-------|--------|
| Initial | 54 | `1if all(r==r[::-1]for r in g)else 7` |
| Arithmetic | 48 | `7-6*all(r==r[::-1]for r in g)` |
| Fixed size | 48 | `7-6*all(r[0]==r[2]for r in g)` |
| any() with diff | 47 | `1+6*any(r[0]-r[2]for r in g)` |
| Tuple unpack | 46 | `1+6*any(a-c for a,_,c in g)` |

## Why It Works

- `any()` on integers: treats 0 as False, non-zero as True
- When `a == c`: difference is 0 (falsy)
- When `a != c`: difference is non-zero (truthy)
- `any(...)` returns True if any row is asymmetric
- `1 + 6*True = 7`, `1 + 6*False = 1`
