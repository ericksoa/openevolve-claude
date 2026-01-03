# Task 3de23699: Extract marker-bounded region

## Pattern
Four corner markers of one color define a rectangular region. A pattern inside uses a different color. Extract the inner region and replace the pattern color with the marker color.

## Algorithm
1. Find all non-zero cells and their colors
2. Identify marker color (appears exactly 4 times) vs pattern color (appears more)
3. Get marker positions to define inner region bounds
4. Extract inner region (excluding marker row/columns)
5. Replace pattern color with marker color in output

## Key Tricks
| Trick | Bytes Saved | Description |
|-------|-------------|-------------|
| `E=enumerate` default arg | 3 | Alias for 3 uses |
| `sorted({*L},key=L.count)` | 5+ | Find marker (fewer occurrences) vs pattern |
| `(m,v)[v!=p]` | 2 | Tuple indexing for conditional |
| Row slicing `G[min(R)+1:max(R)]` | 6 | Direct slice instead of range |
| Single-line semicolons | 6 | Reduce newline overhead |

## Evolution Summary
| Gen | Bytes | Key Discovery |
|-----|-------|---------------|
| 0 | 233 | Baseline with default arg E=enumerate |
| 9 | 232 | Semicolon combining lines |
| 10 | 227 | All logic on single line |
| 15 | 225 | Inline L extraction, remove c variable |

## Byte History
| Version | Bytes | Score | Change |
|---------|-------|-------|--------|
| Initial | 425 | 2075 | Working solution with Counter |
| Golf v1 | 274 | 2226 | Remove Counter, use sorted |
| Golf v2 | 251 | 2249 | Walrus + tuple indexing |
| Golf v3 | 233 | 2267 | Default arg + row slicing |
| Evolved | 225 | 2275 | Single-line + inline L |

## Solution
```python
def solve(G,E=enumerate):L=[v for r,R in E(G)for j,v in E(R)if v];m,p=sorted({*L},key=L.count);R,C=zip(*[(r,j)for r,R in E(G)for j,v in E(R)if v==m]);return[[(m,v)[v!=p]for v in r[min(C)+1:max(C)]]for r in G[min(R)+1:max(R)]]
```