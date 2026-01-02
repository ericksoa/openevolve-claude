# ARC Task 0d3d703e - Code Golf Solution

## Task Description

Transform a 3x3 grid by applying a color mapping. Each cell value is swapped with its pair:
- 1 <-> 5
- 2 <-> 6
- 3 <-> 4
- 8 <-> 9

## Evolution Summary

| Generation | Bytes | Score | Technique |
|------------|-------|-------|-----------|
| 0 (initial) | 104 | 2396 | def with dictionary lookup |
| 1 | 75 | 2425 | Lambda with inline dict |
| 2 | 63 | 2437 | List lookup table |
| 3 | 59 | 2441 | String + int() conversion |
| 4 | 58 | 2442 | Bytes literal + subtraction |
| 5-10 | 58 | 2442 | Various attempts (no improvement) |

## Final Solution

```python
solve=lambda g:[[b"0564312098"[c]&15for c in r]for r in g]
```

**58 bytes, Score: 2442, Fitness: 0.9768**

## Key Golfing Tricks

1. **Lambda over def**: `solve=lambda g:` saves bytes over `def solve(g):\n    return`

2. **Bytes literal indexing**: `b"0564312098"[c]` returns the ASCII value directly as an integer, avoiding list/tuple overhead

3. **Bitwise AND for digit extraction**: `&15` extracts the last 4 bits, converting ASCII digits ('0'-'9', which are 48-57) to their numeric values (0-9). This works because:
   - ASCII '0' = 48 = 0b110000, and 48 & 15 = 0
   - ASCII '5' = 53 = 0b110101, and 53 & 15 = 5
   - etc.

4. **Lookup table encoding**: The string "0564312098" encodes the mapping where position `c` maps to the digit at that position (index 1 -> '5', index 2 -> '6', etc.)

5. **No spaces after digits**: Python allows `&15for` without a space since `15for` cannot be parsed as a single token

## Alternative Approaches Tried

- `%48` instead of `&15`: Same byte count (both 2 chars after the index)
- Integer bit-shifting `588445337168>>c*4&15`: 59 bytes (1 byte longer)
- XOR-based approach with lookup table: 60+ bytes
- Tuple lookup: 63 bytes
- Map/lambda combinations: All longer

## Verification

All 5 examples (4 train + 1 test) pass correctly.
