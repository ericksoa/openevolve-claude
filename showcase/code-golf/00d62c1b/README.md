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

- **Bytes**: 219
- **Score**: 2,281 points (2500 - 219)
- **Status**: Passing all tests

---

## Evolution Journey

### Evolution Progress

| Gen | Bytes | Technique | Insight |
|-----|-------|-----------|---------|
| 0 | 280 | Original edge-detection flood fill | Working baseline |
| 1 | 262 | **Padding approach** | Pad grid with 0s, flood from corner |
| 2 | 241 | Walrus + unpacking | Reuse padding row with `:=` |
| 3 | 238 | Marker = 1 | Eliminates modulo in lookup |
| 4 | 223 | **Recursive flood fill** | Replace while loop with recursion |
| 5 | 219 | **Chained comparison + ~0 trick** | `a>~0` replaces `a>=0` |

### Key Breakthrough: Recursive Flood Fill

The iterative approach with a stack was replaced by direct recursion:
```python
# Old (iterative, ~80 bytes for loop)
while s:
 a,b=s.pop()
 if len(g)>a>=0<=b<w and g[a][b]<1:g[a][b]=1;s+=...

# New (recursive, ~77 bytes)
def f(a,b):
 if len(g)>a>~0<b<w>1>g[a][b]:g[a][b]=1;f(a+1,b);f(a-1,b);f(a,b+1);f(a,b-1)
```

### Key Golf Trick: ~0 for -1

`a>~0` is equivalent to `a>-1` which is `a>=0`, saving 1 byte in the chained comparison.

**Improvement**: 238 → 219 bytes (**-8%**, +19 competition points)

---

## Algorithm

1. Pad grid with zeros around edges
2. Recursive flood fill from (0,0) marking all reachable cells as 1
3. Transform: 0→4 (enclosed), 1→0 (reachable), 3→3 (boundary)

## Key Golf Tricks Used

- Walrus operator `:=` for inline assignment
- Recursive flood fill instead of iterative
- Chained comparison: `len(g)>a>~0<b<w>1>g[a][b]`
- `~0` equals `-1` for shorter bounds check
- List lookup `[4,0,0,3][c]` for value mapping

## Champion Solution (219 bytes)

```python
def solve(G):
 w=len(G[0])+2;g=[o:=[0]*w,*[[0,*r,0]for r in G],o]
 def f(a,b):
  if len(g)>a>~0<b<w>1>g[a][b]:g[a][b]=1;f(a+1,b);f(a-1,b);f(a,b+1);f(a,b-1)
 f(0,0);return[[[4,0,0,3][c]for c in r[1:-1]]for r in g[1:-1]]
```
