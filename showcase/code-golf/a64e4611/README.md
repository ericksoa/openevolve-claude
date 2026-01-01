# Task a64e4611: Largest Zero Rectangle with Extensions

## Problem

Find the largest rectangular region of zeros in a 30x30 grid, then fill with 3s using orientation-dependent shrinking and neighbor-conditional extensions to create a cross pattern.

```
Input (30x30):                Output (30x30):
[mostly 8s with a            [same but largest zero
 region of 0s]                region filled with 3s
                              in a cross pattern]
```

## Solution Stats

- **Bytes**: 523
- **Score**: 1,977 points (2500 - 523)
- **Status**: Passing all tests

---

## Evolution Journey

This hard-difficulty task demonstrates evolution on a complex algorithm with multiple phases.

### Algorithm Discovery

This task required significant reverse-engineering:
- **Initial hypothesis**: Simple maximal rectangle - WRONG (output is a cross, not rectangle)
- **Second hypothesis**: Flood fill - WRONG (connected components don't match)
- **Key insight**: Orientation-dependent shrinking + neighbor-conditional extensions

### Evolution Progress

| Gen | Bytes | Technique | Insight |
|-----|-------|-----------|---------|
| 0 | ~1200 | Verbose with helpers | Working baseline |
| 20 | 858 | Unified E lambda, max() | Eliminate redundant functions |
| 40 | 794 | Tuple indexing | `O[(v,i)[z]][(i,v)[z]]` for row/col swap |
| 50 | 763 | Merged loops | Histogram + max rect in single pass |
| 60 | 751 | Unified shrink formula | `H=j-f>g-e;e+=e>0;g-=H*(g<R-1);...` |
| 65 | 750 | `[*map(list,G)]` | Shorter deep copy |
| 70 | 718 | `I=range` alias | range used 11 times, saves 32 bytes |
| 75 | 717 | `r and X or Y` ternary | Shorter than `X if r else Y` |
| 80 | 712 | `c<C and h[r][c]` | Shorter ternary for zero fallback |
| **85** | **642** | **O(n⁴) brute-force** | **Algorithm swap: simpler is shorter** |
| 90 | 631 | List as tuple `[I(f),I(j+1,C)]` | Shorter than `[(I(f),f),...]` pairs |
| 95 | 619 | Range truthiness | `if L` works for empty ranges |
| 100 | 615 | `-~x` trick + `[0,]` | `(c-a+1)` → `-~(c-a)`, `[(0,)]` → `[0,]` |
| 105 | 578 | Merged extension check | Single `all()` with `I(i-(i>A),...)` range |
| 110 | 571 | Tuple iteration | `for A,B,P,M,z in(row_tuple),(col_tuple):` |
| 115 | 549 | Single list comp + `__setitem__` | Flatten extension into `[...for v in L]` |
| 120 | 547 | Tuple for neighbor range | `(i,i-(i>a),i+(i<b))` vs `I(i-(i>a),...)` |
| 125 | 544 | **1D array** | `O=sum(G,[])` + `O[r*C+j]` indexing |
| 130 | 542 | **~-any trick** | `~-any(...)` replaces `all(...<1)` |
| 135 | 541 | `[0]` fallback | `or[0]` instead of `or[0,]` |
| **140** | **523** | **Final optimizations** | Combined tricks, final tuning |

### Key Breakthrough: Algorithm Swap (Gen 85)

The biggest single improvement came from **switching algorithms**:

- **Old (O(n²) histogram)**: Complex stack-based maximal rectangle (712 bytes)
- **New (O(n⁴) brute-force)**: Simple nested loops checking all rectangles (642 bytes)

Despite being computationally slower, the brute-force approach is **70 bytes shorter** because:
1. No histogram construction or stack manipulation
2. Single `max()` comprehension vs complex while loop
3. All conditions inline in one expression

**Improvement**: ~1200 → 523 bytes (**-56%**, +677 competition points)

---

## Algorithm

1. O(n^4) brute force to find largest rectangle of zeros
2. Adjust boundaries based on rectangle orientation (taller vs wider)
3. Fill the core rectangle with 3s
4. Extend along rows and columns where contiguous zeros exist

## Key Golf Tricks Used

- `I=range` alias saves 12+ bytes
- `N=30` hardcoded grid size (all grids are 30x30)
- `O=sum(G,[])` for 1D array flattening
- `~-any()` to invert any() result (6 chars)
- `-~(x)` for x+1
- `(v,i)[z]` tuple indexing for conditional row/col swap
- `__setitem__` for in-expression mutation

## Champion Solution (523 bytes)

```python
def solve(G):
 I=range;N=30;O=sum(G,[])
 if(b:=max([-~(c-a)*-~(k-d),a,d,c,k]for a in I(N)for d in I(N)for c in I(a,N)for k in I(d,N)if~-any(O[r*N+j]for r in I(a,c+1)for j in I(d,k+1)))or[0])[0]<1:return G
 _,e,f,g,j=b;H=j-f>g-e;e+=e>0;g-=H*(g<29);f+=1-H;j-=1-H
 for r in I(e,g+1):G[r][f:j+1]=[3]*(j-f+1)
 [G[(v,i)[z]].__setitem__((i,v)[z],3)for a,b,*P,z in((e,g,I(f),I(j+1,N),1),(f,j,I(e),I(g+1,N),0))for i in I(a,b+1)for L in P if L and~-any(O[N*(v,w)[z]+(w,v)[z]]for w in(i,i-(i>a),i+(i<b))for v in L)for v in L];return G
```

## Notes

The extension logic (~170 bytes) is essential and cannot be simplified without breaking correctness. Attempts to use simpler flood-fill or threshold-based approaches all failed.
