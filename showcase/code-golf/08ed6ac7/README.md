# ARC Task 08ed6ac7 - Column Height Ranking

## Task Description

This ARC task involves ranking vertical columns of 5s by their height. The input grid contains columns of 5s that extend from different starting rows to the bottom. The output replaces each 5 with a color (1-4) based on the column's height rank:
- Color 1: Tallest column (most 5s)
- Color 2: Second tallest
- Color 3: Third tallest
- Color 4: Shortest column

## Evolution Summary

| Generation | Bytes | Score | Key Changes |
|------------|-------|-------|-------------|
| 0 (initial) | 529 | 1971 | Verbose working solution with comments |
| 1 | 223 | 2277 | Remove comments, single-char variables, compress whitespace |
| 2 | 198 | 2302 | Use zip(*g) to iterate columns, sum for counting |
| 3 | 195 | 2305 | Use c.get instead of lambda for sort key |
| 4 | 189 | 2311 | Merge statements with semicolons |
| 5 | 184 | 2316 | Use (v>0) multiplication trick |
| 6 | 178 | 2322 | Use m[x] instead of m.get(x,0) |
| 7 | 157 | 2343 | Use .index() instead of dict lookup |
| 8 | 156 | 2344 | Use -~ instead of +1 |
| 9 | 150 | 2350 | Remove unnecessary filter (if sum(c)) |
| 10 | 142 | 2358 | Use [*map(sum,zip(*g))] instead of list comprehension |

## Final Solution (142 bytes)

```python
def solve(g):
 c=[*map(sum,zip(*g))];s=sorted(range(len(c)),key=lambda i:-c[i])
 return[[v and-~s.index(x)for x,v in enumerate(r)]for r in g]
```

## Key Golfing Tricks Used

1. **`[*map(sum,zip(*g))]`** - Transpose grid and sum columns in one expression (saves bytes over list comprehension)

2. **`-~x` instead of `x+1`** - Bitwise NOT followed by negation equals increment, saves 1 byte

3. **`v and expr`** - Short-circuit evaluation returns 0 (falsy) when v=0, otherwise evaluates expr

4. **`sorted(range(len(c)),key=lambda i:-c[i])`** - Sort indices by column sum descending

5. **`.index(x)+1`** - Get 1-based rank directly from sorted list position

6. **Single-character variables** - `g`, `c`, `s`, `r`, `v`, `x` for maximum compression

7. **Semicolons** - Chain statements on one line to minimize newline overhead

## Algorithm

1. Transpose the grid using `zip(*g)` to get columns
2. Sum each column (counting 5s, since 0s contribute nothing)
3. Sort column indices by their sums in descending order
4. For each cell: if value is non-zero, look up its column's rank (index+1)
