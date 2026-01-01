# Evolution Log: ARC Fill Enclosed Regions

**Task ID**: `00d62c1b` (ARC-AGI public training set)
**Competition**: [NeurIPS 2025 - Google Code Golf Championship](https://www.kaggle.com/competitions/google-code-golf-2025)
**Prize Pool**: $100,000 | **Scoring**: `max(1, 2500 - bytes)` per task
**Difficulty**: Medium (flood fill algorithm required)

---

## Competition Context

The Google Code Golf Championship challenges participants to write the **shortest possible Python programs** that correctly solve 400 ARC-AGI tasks. This is a fundamentally different optimization target from typical code:

- **Normal code**: Optimize for readability, maintainability, performance
- **Code golf**: Optimize for **minimum byte count** while maintaining correctness

This task (`00d62c1b`) is a medium-difficulty flood fill problem - harder than simple grid transformations but with a well-known algorithmic solution that can be heavily golfed.

---

## Task Analysis

### Problem
Given a grid with 0s (empty) and 3s (boundaries), fill all enclosed regions with 4s.
- Cells connected to the grid edge remain 0
- Cells enclosed by 3s become 4
- Boundary cells (3s) remain unchanged

### Example
```
Input:                    Output:
0 0 0 0 0 0              0 0 0 0 0 0
0 0 3 0 0 0              0 0 3 0 0 0
0 3 0 3 0 0    →         0 3 4 3 0 0
0 0 3 0 3 0              0 0 3 4 3 0
0 0 0 3 0 0              0 0 0 3 0 0
0 0 0 0 0 0              0 0 0 0 0 0
```

### Algorithm Type
Flood fill / connected component detection - mark all cells connected to the boundary as "exterior", then convert remaining 0s to 4s.

---

## Evolution Results

### Original (Gen 0) - 280 bytes
```python
def solve(G):
 g=[*map(list,G)];H=len(g);W=len(g[0]);s=[(i,j)for i in range(H)for j in range(W)if g[i][j]<1if i%~-H<1or j%~-W<1]
 while s:
  i,j=s.pop()
  if-1<i<H and-1<j<W and g[i][j]<1:g[i][j]=9;s+=[(i+1,j),(i-1,j),(i,j+1),(i,j-1)]
 return[[[4,0,0,3][c%4]for c in r]for r in g]
```
- **Bytes**: 280 | **Score**: 2220
- **Approach**: Find edge cells, flood fill exterior with 9, use modulo lookup

### Champion (Gen 3) - 238 bytes
```python
def solve(G):
 w=len(G[0])+2;g=[o:=[0]*w,*[[0,*r,0]for r in G],o];s=[(0,0)]
 while s:
  a,b=s.pop()
  if len(g)>a>=0<=b<w and g[a][b]<1:g[a][b]=1;s+=(a+1,b),(a-1,b),(a,b+1),(a,b-1),
 return[[[4,0,0,3][c]for c in r[1:-1]]for r in g[1:-1]]
```
- **Bytes**: 238 | **Score**: 2262
- **Improvement**: **42 bytes saved (15% reduction)**

---

## Key Innovations

### 1. Padding Approach (~20 bytes saved)
**Insight**: Instead of complex edge detection, pad the grid with zeros and flood fill from corner.

```python
# Old: Complex edge detection (42 bytes)
s=[(i,j)for i in range(H)for j in range(W)if g[i][j]<1if i%~-H<1or j%~-W<1]

# New: Simple padding + corner start (~22 bytes)
g=[o:=[0]*w,*[[0,*r,0]for r in G],o];s=[(0,0)]
```

### 2. Marker Value Optimization (2 bytes saved)
**Insight**: Using 1 as marker (instead of 9) eliminates modulo in lookup.

```python
# Old: marker 9 needs %4
[4,0,0,3][c%4]  # 9%4=1 → 0

# New: marker 1, direct index
[4,0,0,3][c]    # 1 → 0
```

### 3. Walrus Operator (1 byte saved)
**Insight**: Reuse the padding row with `:=`

```python
g=[o:=[0]*w,*[[0,*r,0]for r in G],o]  # 'o' reused for top and bottom
```

### 4. Star Unpacking (1 byte saved)
```python
# Old: Concatenation
[[0]+r+[0]for r in G]

# New: Unpacking
[[0,*r,0]for r in G]
```

### 5. Trailing Comma Extension (1 byte saved)
```python
# Old: List literal
s+=[(a+1,b),(a-1,b),(a,b+1),(a,b-1)]

# New: Trailing comma creates tuple
s+=(a+1,b),(a-1,b),(a,b+1),(a,b-1),
```

---

## Evolution Journey

### Stage 1: Analysis
Identified the algorithm pattern and evaluated the original solution structure.

### Stage 2: Apply Known Tricks

| Step | Bytes | Change | Technique |
|------|-------|--------|-----------|
| Original | 280 | - | Starting point |
| Padding approach | 262 | -18 | Eliminate edge detection |
| Add `w` variable | 249 | -13 | Avoid repeated `len()` |
| Chain comparison | 244 | -5 | `len(g)>a>=0<=b<w` |
| Trailing comma | 243 | -1 | Tuple extension |
| `[0,*r,0]` unpacking | 242 | -1 | Star unpacking |
| List unpacking | 241 | -1 | `*[[...]]` syntax |
| Walrus operator | 240 | -1 | Reuse `o` |
| **Marker = 1** | **238** | **-2** | **No modulo needed** |

### Stage 3: Failed Attempts

| Attempt | Bytes | Result | Why It Failed |
|---------|-------|--------|---------------|
| Set-based stack | 241 | ✓ | `\|=` syntax overhead |
| Recursive DFS | 270 | ✓ | Needs `setrecursionlimit` |
| Fixed-point | 306 | ✓ | Triple nested loops |
| `w>g[a][b]` chain | 232 | ✗ | Infinite loop (wrong logic) |
| XOR formula | 235 | ✗ | Fails for marker value |
| String lookup | 235 | ✗ | Returns strings, not ints |

---

## Reusable Golf Tricks

### For Flood Fill Problems
| Trick | Savings | When to Use |
|-------|---------|-------------|
| Padding with zeros | ~20 bytes | Any boundary-connected problem |
| Smart marker values | 2-4 bytes | When mapping values at end |
| Tuple trailing comma | 1 byte | Extending lists/stacks |
| Chain comparisons | 3-5 bytes | Multiple bound checks |

### General Python Golf
| Trick | Example | Savings |
|-------|---------|---------|
| Walrus for reuse | `[o:=[0]*w,...,o]` | 1+ bytes |
| Star unpacking | `[0,*r,0]` vs `[0]+r+[0]` | 1 byte |
| Lambda vs def | `f=lambda:` vs `def f():` | ~6 bytes |
| Lookup vs formula | `[4,0,0,3][c]` vs arithmetic | Usually shorter |

---

## Summary

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Bytes** | 280 | 238 | -42 (15%) |
| **Score** | 2220 | 2262 | +42 |
| **Mutations tested** | - | 30+ | - |

**Key breakthrough**: The padding approach eliminated complex edge detection, providing the foundation for additional micro-optimizations.

---

## References

- [NeurIPS 2025 - Google Code Golf Championship](https://www.kaggle.com/competitions/google-code-golf-2025)
- [ARC-AGI Benchmark](https://arcprize.org/)
- [Competition Details](https://www.competehub.dev/en/competitions/kagglegoogle-code-golf-2025)
