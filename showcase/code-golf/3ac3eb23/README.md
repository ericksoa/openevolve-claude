# Task 3ac3eb23: Diagonal checkerboard propagation

## Pattern

Seeds on the first row expand diagonally downward in a checkerboard pattern:
- **Even rows**: Seed at original column j
- **Odd rows**: Seeds at columns j-1 and j+1

Each seed independently creates its own diagonal pattern extending through all rows.

## Algorithm

1. Copy input grid to output
2. For each non-zero value (seed) in the first row at column j:
   - For each row i:
     - If even row: set column j to seed color
     - If odd row: set columns j-1 and j+1 to seed color (if within bounds)

## Key Tricks

| Trick | Before | After | Saves |
|-------|--------|-------|-------|
| Lambda + walrus | `def solve(g):...return o` | `solve=lambda g:(o:=...,[])[0]` | ~6 bytes |
| `[*map(list,g)]` | `[r[:]for r in g]` | `[*map(list,g)]` | 2 bytes |
| Slice step `[~i%2::2]` | `[j] if i%2==0 else [j-1,j+1]` | `(j-1,j,j+1)[~i%2::2]` | ~15 bytes |
| `~i%2` | `1-i%2` | `~i%2` | 1 byte |
| `exec(f"r[{k}]=c")` | `r[k]=c` via nested loop | inline exec | ~5 bytes |
| `len(r)>k>=0<c` | `len(r)>k and k>=0 and c` | chained comparison | 5 bytes |
| Remove W variable | `W:=len(g[0])...W>k` | `len(r)>k` | 5 bytes |

## The Slice Trick Explained

`(j-1,j,j+1)[~i%2::2]` elegantly selects:
- When i=0 (even): `~0%2=1`, so `[1::2]` = `(j,)`
- When i=1 (odd): `~1%2=0`, so `[0::2]` = `(j-1, j+1)`

This replaces a conditional with a single slice expression!

## Byte History

| Stage | Bytes | Change |
|-------|-------|--------|
| Initial working | 205 | Baseline |
| Golf nested loops | 180 | -25 |
| Add exec trick | 171 | -9 |
| Slice `[~i%2::2]` | 164 | -7 |
| Lambda + walrus | 158 | -6 |
| Remove W variable | 153 | -5 |
| Use `len(r)` | **150** | -3 |

**Final: 150 bytes, Score: 2350 points**

## Solution

```python
solve=lambda g:(o:=[*map(list,g)],[exec(f"r[{k}]=c")for i,r in enumerate(o)for j,c in enumerate(g[0])for k in(j-1,j,j+1)[~i%2::2]if len(r)>k>=0<c])[0]
```
