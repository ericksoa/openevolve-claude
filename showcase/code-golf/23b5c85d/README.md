# Task 23b5c85d

## Pattern
Find the smallest colored rectangle in the grid and return it filled with its color.

## Algorithm
1. Build a dictionary mapping each non-zero color to its list of (row, col) positions
2. Find the color with the minimum count (smallest rectangle)
3. Get the bounding box of that color using zip(*positions)
4. Return a filled rectangle of that color

## Key Tricks
- `C.get(v,[])+[(r,c)]` - build position lists
- `zip(*C[m])` - unpack positions into row/col tuples in one step
- `m:=min(...)` - walrus operator to capture min color
- `lambda _:len(C[_])` - underscore saves 1 byte vs `x`
- `-~(x)` = `x+1` - saves 1 byte when used with subtraction

## Solution (201 bytes)
```python
def solve(g):
 C={}
 for r,R in enumerate(g):
  for c,v in enumerate(R):
   if v:C[v]=C.get(v,[])+[(r,c)]
 a,b=zip(*C[m:=min(C,key=lambda _:len(C[_]))])
 return[[m]*-~(max(b)-min(b))]*-~(max(a)-min(a))
```

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| Initial | 286 | Working solution |
| v2 | 258 | Removed semicolons, simplified |
| v3 | 231 | Used list multiplication |
| v4 | 205 | Used zip(*P) |
| v5 | 201 | Walrus operator, underscore lambda |
