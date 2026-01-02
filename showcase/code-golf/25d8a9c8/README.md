# Task 25d8a9c8

## Pattern

For each row in a 3×3 grid, output `[5,5,5]` if all values are the same, else `[0,0,0]`.

## Algorithm

Check row uniformity using set cardinality: `len({*r})<2` returns True if all elements
are identical. Multiply 5 by this boolean (1 or 0) and repeat 3 times to form the output row.

## Key Tricks

- `{*r}` - unpack row into set (shorter than `set(r)`)
- `len({*r})<2` - True if all elements same (cardinality 1), False otherwise
- `5*(condition)` - multiply by boolean (True=1, False=0) gives 5 or 0
- `[x]*3` - replicate to form output row

## Byte History

| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 66 | `[[5,5,5]if len({*r})<2 else[0,0,0]for r in g]` |
| v2 | 57 | Index trick `[[0]*3,[5]*3][condition]` |
| v3 | 50 | Multiplication trick `[5*(condition)]*3` |

## Solution (50 bytes)

```python
def solve(g):return[[5*(len({*r})<2)]*3for r in g]
```

Score: **2450 points** (2500 - 50)
