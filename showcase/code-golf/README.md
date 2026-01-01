# ARC-AGI Code Golf: Evolving the Shortest Programs

**Evolved Python solutions for the [NeurIPS 2025 Google Code Golf Championship](https://www.kaggle.com/competitions/google-code-golf-2025) using LLM-driven evolutionary optimization.**

---

## Results Summary

| Task | Pattern | Original | Evolved | Improvement |
|------|---------|----------|---------|-------------|
| `0520fde7` | Grid comparison | 80 bytes | **57 bytes** | -29% |
| `00d62c1b` | Fill enclosed regions | 280 bytes | **238 bytes** | -15% |
| `017c7c7b` | Simple transform | 54 bytes | **54 bytes** | baseline |

**Total score improvement**: +65 points across evolved tasks

---

## Why This Problem Matters

### The Competition

The [NeurIPS 2025 - Google Code Golf Championship](https://www.kaggle.com/competitions/google-code-golf-2025) challenged participants to write the **shortest possible Python programs** that correctly solve 400 [ARC-AGI](https://arcprize.org/) tasks.

| Detail | Value |
|--------|-------|
| **Prize Pool** | $100,000 |
| **Tasks** | 400 (ARC-AGI public training set) |
| **Scoring** | `max(1, 2500 - bytes)` per correct solution |
| **Maximum Score** | 1,000,000 (400 × 2500) |
| **Deadline** | October 30, 2025 |

### Why Code Golf is Hard

Code golf is a unique optimization challenge:

1. **Correctness is binary** - A solution that fails ANY test case scores 0.001
2. **Every byte matters** - Saving 1 byte = +1 point
3. **Semantic equivalence required** - Transformations must preserve behavior
4. **Language mastery needed** - Exploiting Python quirks and shortcuts
5. **Algorithm selection critical** - Sometimes a completely different approach is shorter

### Why This Matters for Evolution

Code golf is an ideal testbed for LLM-driven evolution because:

- **Clear fitness function**: Byte count (lower = better)
- **Automatic verification**: Run tests to check correctness
- **Rich mutation space**: Syntax tricks, algorithm changes, refactoring
- **Transferable learnings**: Tricks discovered on one task apply to others

---

## The Evolution Approach

Unlike performance optimization (where we measure ops/sec), code golf evolution optimizes for **minimum byte count**:

```
fitness = correctness × (2500 - bytes) / 2500
```

### Three-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  Code Golf Evolution Pipeline                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Stage 1:     │───▶│ Stage 2:     │───▶│ Stage 3:     │       │
│  │ Find Correct │    │ Apply Known  │    │ Discover New │       │
│  │ Solution     │    │ Tricks       │    │ Approaches   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
│  "Make it work"      "Make it short"     "Make it shorter"      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Evolution Journey: Fill Enclosed Regions (`00d62c1b`)

This medium-difficulty task demonstrates the full evolution pipeline.

### The Task

Fill enclosed regions (surrounded by 3s) with 4s. Cells connected to edges remain 0.

```
Input:                    Output:
0 0 0 0 0 0              0 0 0 0 0 0
0 0 3 0 0 0              0 0 3 0 0 0
0 3 0 3 0 0    →         0 3 4 3 0 0
0 0 3 0 3 0              0 0 3 4 3 0
0 0 0 3 0 0              0 0 0 3 0 0
0 0 0 0 0 0              0 0 0 0 0 0
```

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

### Champion Solution (238 bytes)

```python
def solve(G):
 w=len(G[0])+2;g=[o:=[0]*w,*[[0,*r,0]for r in G],o];s=[(0,0)]
 while s:
  a,b=s.pop()
  if len(g)>a>=0<=b<w and g[a][b]<1:g[a][b]=1;s+=(a+1,b),(a-1,b),(a,b+1),(a,b-1),
 return[[[4,0,0,3][c]for c in r[1:-1]]for r in g[1:-1]]
```

**Improvement**: 280 → 238 bytes (**-15%**, +42 competition points)

---

## Golf Tricks Library

Tricks discovered during evolution, applicable to other tasks:

### Structural Tricks

| Trick | Before | After | Saves |
|-------|--------|-------|-------|
| Lambda over def | `def f(x):\n return E` | `f=lambda x:E` | ~6 bytes |
| Star unpacking | `[0]+r+[0]` | `[0,*r,0]` | 1 byte |
| Walrus reuse | `a=[0]*w;g=[a,...,a]` | `g=[a:=[0]*w,...,a]` | 1 byte |
| Trailing comma | `s+=[(a,b),(c,d)]` | `s+=(a,b),(c,d),` | 1 byte |

### Comparison Tricks

| Trick | Before | After | Saves |
|-------|--------|-------|-------|
| Chain bounds | `0<=a and a<H` | `H>a>=0` | 5 bytes |
| Zero check | `x==0` | `x<1` | 1 byte |
| Nonzero check | `x!=0` | `x>0` | 1 byte |

### Algorithm Tricks

| Trick | Description | Savings |
|-------|-------------|---------|
| Padding for flood fill | Add border of 0s, start from corner | ~20 bytes |
| Smart marker values | Choose markers that simplify final lookup | 2-4 bytes |
| Direct row iteration | `for r in g` vs `for i in range(len(g))` | 7+ bytes |
| Tuple indices | `(0,1,2)` instead of `range(3)` | 2 bytes |

---

## Quick Start

### Prerequisites
- Python 3.8+

### Evaluate a Solution
```bash
cd showcase/code-golf

# Evaluate single task
python evaluator.py 00d62c1b solutions/00d62c1b.py

# Expected output:
# {
#   "task_id": "00d62c1b",
#   "fitness": 0.9048,
#   "score": 2262,
#   "byte_count": 238,
#   "correct": true
# }
```

### Evolve a Solution
```bash
# Use the /evolve-size skill
/evolve shortest Python solution for ARC task <task_id>
```

---

## Technical Details

### Scoring Formula

For each of the 400 tasks:
```python
score = max(1, 2500 - byte_count) if correct else 0.001
```

- Maximum per task: 2500 (0 bytes - impossible)
- Practical maximum: ~2450 (50-byte solution)
- Incorrect solutions: 0.001 (effectively zero)

### Solution Format

Each solution must define a `solve` function:
```python
def solve(grid):
    # grid: List[List[int]] - input grid
    # return: List[List[int]] - output grid
```

### Constraints
- Python Standard Library only (no numpy, scipy, etc.)
- Self-contained (no imports from other files)
- Must pass all train AND test examples

---

## File Structure

```
showcase/code-golf/
├── README.md                    # This file
├── evaluator.py                 # Scoring and validation harness
├── tasks/                       # 400 ARC-AGI task JSONs
│   ├── 00d62c1b.json
│   ├── 0520fde7.json
│   └── ...
├── solutions/                   # Evolved Python solutions
│   ├── 00d62c1b.py             # 238 bytes (champion)
│   ├── 0520fde7.py             # 57 bytes (champion)
│   └── 017c7c7b.py             # 54 bytes (baseline)
└── mutations/                   # Evolution logs
    ├── arc_fill_enclosed_regions.md
    └── 0520fde7_evolution.md
```

---

## Reproducing Results

### Step 1: Verify Existing Solutions
```bash
cd showcase/code-golf
python evaluator.py 00d62c1b solutions/00d62c1b.py
python evaluator.py 0520fde7 solutions/0520fde7.py
```

### Step 2: Evolve a New Task
```bash
# Pick an unsolved task
ls tasks/ | head -20

# Evolve it
/evolve shortest Python solution for ARC task <task_id>
```

### Step 3: Verify Improvement
```bash
python evaluator.py <task_id> solutions/<task_id>.py
```

---

## What Works

1. **Padding approach** for flood-fill problems - dramatically simplifies boundary logic
2. **Lambda over def** - saves 6+ bytes in most cases
3. **Direct iteration** (`for r in g`) over index iteration (`for i in range(len(g))`)
4. **Lookup tables** - usually shorter than arithmetic formulas
5. **Chain comparisons** - `H>a>=0<=b<W` saves multiple `and` operators
6. **Smart marker values** - choose values that simplify final mapping

## What Doesn't Work

1. **Recursion** - requires `setrecursionlimit`, adds overhead
2. **Sets for stacks** - `|=` syntax longer than tuple extension
3. **Bitwise tricks** - often need parentheses, same or longer
4. **String lookups** - return strings, not ints
5. **Complex formulas** - lookup tables usually shorter

---

## Competition Status

| Metric | Current | Target |
|--------|---------|--------|
| Tasks solved | 3 | 400 |
| Total score | ~6,700 | ~960,000+ |
| Competition status | Ended (Oct 2025) | - |

This showcase demonstrates the `/evolve-size` capability. The techniques transfer to any code golf challenge.

---

## References

- [NeurIPS 2025 - Google Code Golf Championship](https://www.kaggle.com/competitions/google-code-golf-2025)
- [ARC Prize](https://arcprize.org/) - The ARC-AGI benchmark
- [Competition Details](https://www.competehub.dev/en/competitions/kagglegoogle-code-golf-2025)
- [François Chollet's announcement](https://x.com/fchollet/status/1953493314323562922)

---

## Deterministic Reproduction

- [x] No external data files required (tasks embedded in `tasks/`)
- [x] No network requests during evaluation
- [x] Deterministic scoring (byte count is exact)
- [x] Same results every run
