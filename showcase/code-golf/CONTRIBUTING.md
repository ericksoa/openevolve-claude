# Code Golf Solution Standards

## Required Files

Every solved task MUST have these files in its directory:

```
<task_id>/
├── solution.py    # The golfed Python solution
└── README.md      # Documentation (REQUIRED)
```

---

## README.md Requirements

**Every solution MUST have a README.md** documenting the approach. This is non-negotiable.

### Minimum Required Sections

```markdown
# Task <task_id>

## Pattern
[One sentence describing what the transformation does]

## Algorithm
[2-4 sentences explaining the approach taken]

## Key Tricks
- [List golf tricks used]
- [e.g., "walrus operator for inline assignment"]
- [e.g., "1D array indexing instead of 2D"]

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | XXX | Initial solution |
| v2 | YYY | [what changed] |
```

### Example README

```markdown
# Task 10fcaaa3

## Pattern
Tile input 2x2, then add 8s diagonally adjacent to colored cells.

## Algorithm
First tile the input grid by doubling in both dimensions. Then identify all
non-zero cells in the tiled output. For each empty cell, check if it's
diagonally adjacent (distance √2) to any colored cell - if so, fill with 8.

## Key Tricks
- `g*2` to repeat list (tile rows)
- `r*2` to repeat row (tile columns)
- `(r-i)**2+(c-j)**2==2` for diagonal adjacency check
- `2in{...}` shorter than `any(...==2 for...)`

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 198 | Initial working solution |
| v2 | 176 | Inline S computation, use `2in{...}` trick |
```

---

## Solution Workflow

When solving a new task, follow this order:

### 1. Solve First
```python
# Get it working, don't worry about bytes yet
def solve(g):
    # ... working solution
```

### 2. Initial Golf
```python
# Apply known tricks to minimize bytes
def solve(g):...  # one-liner if possible
```

### 3. Evolution (AlphaEvolve-Inspired) (REQUIRED for 200+ byte solutions)

We use techniques *inspired by* AlphaEvolve (mutation, crossover, selection) but
implemented via our `/evolve` skill - not the actual AlphaEvolve system.

For any solution over 200 bytes, run **at least 5-10 generations** of evolution:

```
mkdir -p .evolve/<task_id>/mutations
```

#### Evolution Process

**Each generation**: Create 3-4 mutations, test all, keep best:

```python
# Gen N mutations to try:
# a) Variable elimination - can we remove a variable?
# b) Expression rewriting - can we express X differently?
# c) Algorithm change - is there a fundamentally shorter approach?
# d) Crossover - combine tricks from previous best solutions
```

#### Mutation Categories

| Category | Examples | Typical Savings |
|----------|----------|-----------------|
| **Variable elimination** | `v=u-t` → inline `u-t` | 3-10 bytes |
| **Operator substitution** | `//2` → `>>1` | 2 bytes |
| **Loop restructuring** | nested loops → list comp | varies |
| **Algorithm change** | complex → integer math | 10-30 bytes |
| **Copy method** | `[r[:]for r in g]` → `eval(str(g))` | 4 bytes |

#### When to Stop Evolution

- 3 consecutive generations with no improvement (plateau)
- Solution is under 100 bytes (diminishing returns)
- All obvious mutation categories exhausted

#### Evolution Documentation

Add to README.md:

```markdown
## Evolution Summary (AlphaEvolve-Inspired)

X generations, Y mutations tested. Final: **Z bytes** (-N%, -M bytes)

### Key Discoveries
| Gen | Discovery | Bytes | Delta |
|-----|-----------|-------|-------|
| 4 | v=u-t insight | 284 | -5 |
| 6 | >>1 bit shift | 280 | -9 |

### Failed Approaches
- Lambda functions (added overhead)
- Dict-based storage (too verbose)
```

### 4. Document Immediately
Create README.md **before** committing. Do not commit solutions without documentation.

### 5. Verify
```bash
python3 evaluator.py <task_id> <task_id>/solution.py
```

### 6. Update Project Files (REQUIRED)

After every solve, update these files:

**README.md:**
- Progress Summary (solved count, total score, avg, % of winner)
- Solved Problems table (add entry sorted by bytes)
- Unsolved Problems (remove if listed)
- Competition Status table (including Est. Place)

**PROJECTION.md:**
- Current Status metrics
- Projected Final Standings (recalculate placement)
- Tasks by Difficulty tier (add task, update averages)
- Projection Model table

**Placement Formula:**
```
Rank 50 = 932,557 pts, ~500 pts/rank drop

Conservative place = 50 + (932,557 - (avg × 400)) / 500
Optimistic place = 50 + (932,557 - tier_weighted_score) / 500
```

### 7. Commit Together
```bash
git add <task_id>/ README.md PROJECTION.md
git commit -m "<task_id>: <pattern> (<bytes> bytes, +<score> pts)"
```

---

## Pre-Commit Checklist

Before committing any solution:

- [ ] `solution.py` passes all train AND test examples
- [ ] `README.md` exists with all required sections
- [ ] Pattern description is clear and concise
- [ ] Algorithm explanation is understandable
- [ ] Key tricks are documented for future reference
- [ ] Byte history shows evolution (even if just v1)
- [ ] **For 200+ byte solutions**: Evolution was attempted (min 5 generations)
- [ ] **Project README.md updated** (Progress Summary, Solved Problems, Competition Status)
- [ ] **PROJECTION.md updated** (Current Status, tier tables, Est. Place recalculated)

---

## Known Golf Tricks Library

### Deep Copy
| Trick | Bytes | Notes |
|-------|-------|-------|
| `eval(str(g))` | 12 | Best for 2D lists |
| `[*map(list,g)]` | 15 | Standard approach |
| `[r[:]for r in g]` | 16 | Verbose |

### Division by 2
| Trick | Bytes | Notes |
|-------|-------|-------|
| `x>>1` | 4 | Best (bit shift) |
| `x//2` | 4 | Same length but needs parens in expressions |
| `(x)//2` | 6 | With precedence |

### Variable Patterns
| Pattern | When to Use |
|---------|-------------|
| Keep variable | Used 3+ times |
| Inline expression | Used 1-2 times, or can be simplified |
| Eliminate via algebra | `v=u-t` type relationships |

### Loop Structures
| Structure | Best For |
|-----------|----------|
| Explicit `for` loops | Side effects, multiple statements |
| List comprehension | Building new lists, simple transforms |
| `map`/`filter` | Rarely shorter in Python 3 |

---

## Why Documentation Matters

1. **Context rot** - Without docs, we forget how solutions work
2. **Re-golf opportunities** - Documented tricks help identify improvements
3. **Knowledge transfer** - Tricks discovered apply to other tasks
4. **Debugging** - Easier to fix broken solutions with documented intent
5. **Evolution tracking** - Failed approaches inform future attempts

---

## Quick Reference: Common Sections

### For Simple Tasks (< 150 bytes)
- Pattern: 1 sentence
- Algorithm: 1-2 sentences
- Key Tricks: 2-3 bullets
- Byte History: 1-2 rows

### For Complex Tasks (200-400 bytes)
- Pattern: 1-2 sentences
- Algorithm: 3-5 sentences, may include pseudocode
- Key Tricks: 4-6 bullets with explanations
- Byte History: multiple iterations
- **Evolution Summary**: generations run, key discoveries
- Optional: "Failed Approaches" section

### For Very Hard Tasks (400+ bytes)
All of the above, plus:
- "Challenges" section explaining what made it hard
- "Potential Improvements" for future re-golf attempts
- Full evolution log in `.evolve/<task_id>/evolution.md`
