# ARC Task 0962bcdd - Cross Expansion

## Task Description
Given a 12x12 grid containing one or more "cross" patterns (a center cell surrounded by 4 orthogonal neighbors of a different non-zero color), expand each cross by:
1. Extending the arm color by 1 cell in each cardinal direction
2. Filling diagonal positions at distances 1 and 2 from the center with the center color

## Evolution History

| Gen | Bytes | Score | Key Changes |
|-----|-------|-------|-------------|
| 0   | 1058  | 1442  | Initial working solution with bounds checks |
| 1   | 535   | 1965  | Remove whitespace, shorter variable names |
| 2   | 363   | 2137  | Use loops for coordinate pairs, fixed bounds |
| 3   | 291   | 2209  | Use x*x==y*y trick for diagonal detection |
| 4   | 260   | 2240  | Diagonal loop with +x/-x symmetry |
| 5   | 260   | 2240  | (No improvement) |
| 6   | 254   | 2246  | Cache range(2,10) in variable R |
| 7   | 252   | 2248  | Use [*map(list,g)] instead of [r[:]for r in g] |
| 8   | 248   | 2252  | c*u instead of c and u!=0 |
| 9   | 254   | 2246  | (Regression - removed R cache) |
| 10  | 241   | 2259  | Single loop with k//8+2, k%8+2 for coordinates |

## Final Solution (241 bytes)

```python
def solve(g):
 o=[*map(list,g)]
 for k in range(64):
  i,j=k//8+2,k%8+2;c=g[i][j];u=g[i-1][j]
  if c*u and u==g[i+1][j]==g[i][j-1]==g[i][j+1]!=c:
   for d in-2,2:o[i+d][j]=o[i][j+d]=u
   for x in-2,-1,1,2:o[i+x][j+x]=o[i+x][j-x]=c
 return o
```

## Key Golf Tricks

1. **Single loop with divmod**: `k//8+2, k%8+2` flattens nested loops into one
2. **Splat map**: `[*map(list,g)]` is shorter than `[r[:]for r in g]`
3. **Product condition**: `c*u` is shorter than `c and u` when both must be non-zero
4. **Chained comparison**: `u==...==...==...!=c` in one expression
5. **Symmetric assignment**: `o[i+x][j+x]=o[i+x][j-x]=c` handles both diagonals
6. **Negative-first tuple**: `-2,2` needs no parens after `in`

## Score
- **Bytes**: 241
- **Score**: 2259 (max(1, 2500 - 241))
- **Fitness**: 0.9036
