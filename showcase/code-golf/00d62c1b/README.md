# Task 00d62c1b: Fill Enclosed Regions

## Problem

Given a grid with 3s forming boundaries, fill enclosed regions (cells that cannot reach the edge) with 4s. Cells reachable from the edge remain 0.

```
Input:                    Output:
0 0 0 0 0 0              0 0 0 0 0 0
0 0 3 0 0 0              0 0 3 0 0 0
0 3 0 3 0 0    →         0 3 4 3 0 0
0 0 3 0 3 0              0 0 3 4 3 0
0 0 0 3 0 0              0 0 0 3 0 0
0 0 0 0 0 0              0 0 0 0 0 0
```

## Solution Stats

- **Bytes**: 238
- **Score**: 2,262 points (2500 - 238)
- **Status**: Passing all tests

---

## Evolution Journey

This medium-difficulty task demonstrates the full evolution pipeline.

### Evolution Progress

| Gen | Bytes | Technique | Insight |
|-----|-------|-----------|---------|
| 0 | 280 | Original edge-detection flood fill | Working baseline |
| 1 | 262 | **Padding approach** | Pad grid with 0s, flood from corner |
| 2 | 241 | Walrus + unpacking | Reuse padding row with `:=` |
| 3 | **238** | **Marker = 1** | Eliminates modulo in lookup |

### Key Breakthrough: Padding Approach

Instead of complex edge detection:
```python
# Old (42 bytes just for initialization)
s=[(i,j)for i in range(H)for j in range(W)if g[i][j]<1if i%~-H<1or j%~-W<1]
```

Pad the grid and start from corner:
```python
# New (~22 bytes, simpler)
g=[o:=[0]*w,*[[0,*r,0]for r in G],o];s=[(0,0)]
```

**Improvement**: 280 → 238 bytes (**-15%**, +42 competition points)

---

## Algorithm

1. Pad grid with zeros around edges
2. Flood fill from (0,0) marking all reachable cells as 1
3. Transform: 0→4 (enclosed), 1→0 (reachable), 3→3 (boundary)

## Key Golf Tricks Used

- Walrus operator `:=` for inline assignment
- Tuple unpacking in stack operations
- List comprehension with conditional mapping
- Boundary padding to simplify edge handling

## Champion Solution (238 bytes)

```python
def solve(G):
 w=len(G[0])+2;g=[o:=[0]*w,*[[0,*r,0]for r in G],o];s=[(0,0)]
 while s:
  a,b=s.pop()
  if len(g)>a>=0<=b<w and g[a][b]<1:g[a][b]=1;s+=(a+1,b),(a-1,b),(a,b+1),(a,b-1),
 return[[[4,0,0,3][c]for c in r[1:-1]]for r in g[1:-1]]
```
