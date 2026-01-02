# ARC Task 0b148d64 - Code Golf Solution

## Task Description
Given a grid divided into 4 quadrants by separator rows/columns of zeros, find the quadrant containing a unique color (one that doesn't appear in any other quadrant) and return it with zero-padding stripped.

## Pattern
1. Find the separator row (all zeros) and separator column (all zeros)
2. Extract 4 quadrants from the grid
3. Find the quadrant with a color unique to it (not in other quadrants)
4. Strip zero-padding from all edges of that quadrant
5. Return the trimmed quadrant

## Evolution History

| Gen | Bytes | Score | Key Change |
|-----|-------|-------|------------|
| 0   | 1210  | 1290  | Initial working solution |
| 1   | 647   | 1853  | Basic compression, single-char vars |
| 2   | 543   | 1957  | Use `sum` for zero detection, set operations |
| 3   | 539   | 1961  | Inline L=len |
| 4   | 533   | 1967  | Assign S=sum |
| 5   | 529   | 1971  | Remove redundant guards |
| 6   | 522   | 1978  | Simplify while conditions |
| 7   | 507   | 1993  | Combined L/S aliasing |
| 8   | 503   | 1997  | Use zip transpose for column stripping |
| 9   | 497   | 2003  | Star unpacking [*zip(*q)] and [[*x]] |
| 10  | 454   | 2046  | Recursive T function with ternary |

## Final Solution (454 bytes)
```python
def solve(g):
 S=sum;n=len(g);m=len(g[0])
 def T(q):
  while S(q[0])<1:q=q[1:]
  return q if S(q[-1])else T(q[:-1])
 r=[i for i in range(n)if S(g[i])<1][0]
 c=[j for j in range(m)if S(g[i][j]for i in range(n))<1][0]
 Q=[[x[:c]for x in g[:r]],[x[c+1:]for x in g[:r]],[x[:c]for x in g[r+1:]],[x[c+1:]for x in g[r+1:]]]
 for q in Q:
  if{x for y in q for x in y}-{0,*[x for p in Q if p!=q for y in p for x in y]}:return[[*x]for x in zip(*T([*zip(*T(q))]))]
```

## Key Golf Tricks

1. **sum() for zero detection**: `S(q[0])<1` instead of `all(c==0 for c in q[0])`
2. **Set difference for unique colors**: `{x for y in q for x in y}-{0,*[other colors]}`
3. **Transpose via zip for 2D stripping**: Apply T to rows, transpose, apply T again, transpose back
4. **Recursive function with ternary**: `return q if S(q[-1])else T(q[:-1])` saves bytes vs while loop
5. **Star unpacking**: `[*zip(*q)]` and `[[*x]for x in ...]` shorter than `list()`
6. **Single-char aliases**: `S=sum`, `n=len(g)`, `m=len(g[0])`
7. **Inline list indexing**: `[...][0]` to get first match

## Score Calculation
- Byte count: 454
- Score: max(1, 2500 - 454) = 2046
- Fitness: 2046 / 2500 = 0.8184
