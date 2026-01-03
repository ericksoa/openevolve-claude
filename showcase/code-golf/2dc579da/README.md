# Task 2dc579da: Extract Quadrant with Anomaly

## Pattern
Grid divided by cross divider (horizontal + vertical line). One quadrant contains an anomaly (different color). Extract that quadrant.

## Algorithm
1. Find center position (divider intersection): `v=h//2, u=w//2`
2. Divider color = `g[v][u]`, background = `g[0][0]`
3. Generate 4 quadrants via slicing
4. Return first quadrant containing a cell not in {divider, background}

## Key Tricks
- Divider always at center: `v,u=len(g)//2,len(g[0])//2`
- Background is always `g[0][0]` (corner never on divider)
- Nested generator: `for R in(g[:v],g[v+1:])for q in[[r[:u]for r in R],[r[u+1:]for r in R]]`
- `any(c not in S for r in q for c in r)` for anomaly detection

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 452 | Initial working solution |
| v10 | 241 | Set unpacking for S |
| v14 | 239 | Factor out A,B slices |
| v15 | 214 | Loop over row slices |
| v16 | 206 | Remove h,w variables |
| v17 | 201 | Use g[0][0] as background |
| v18 | 189 | Inline A,B, remove d variable |

## Solution
```python
def solve(g):
 v,u=len(g)//2,len(g[0])//2;S={g[v][u],g[0][0]}
 return next(q for R in(g[:v],g[v+1:])for q in[[r[:u]for r in R],[r[u+1:]for r in R]]if any(c not in S for r in q for c in r))
```
