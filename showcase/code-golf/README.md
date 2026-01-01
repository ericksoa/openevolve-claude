# ARC-AGI Code Golf: Evolving the Shortest Programs

**Evolved Python solutions for the [NeurIPS 2025 Google Code Golf Championship](https://www.kaggle.com/competitions/google-code-golf-2025) using LLM-driven evolutionary optimization.**

---

## Results Summary

| Task | Pattern | Original | Evolved | Improvement |
|------|---------|----------|---------|-------------|
| `0520fde7` | Grid comparison | 80 bytes | **57 bytes** | -29% |
| `00d62c1b` | Fill enclosed regions | 280 bytes | **238 bytes** | -15% |
| `a64e4611` | Largest rectangle + cross | ~1200 bytes | **547 bytes** | -54% |
| `017c7c7b` | Simple transform | 54 bytes | **54 bytes** | baseline |

**Total score improvement**: +718 points across evolved tasks

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

## Evolution Journey: Largest Rectangle Cross (`a64e4611`)

This hard-difficulty task demonstrates evolution on a complex algorithm with multiple phases.

### The Task

Find the largest rectangular region of zeros in a 30x30 grid, then fill with 3s using orientation-dependent shrinking and neighbor-conditional extensions to create a cross pattern.

```
Input (30x30):                Output (30x30):
[mostly 8s with a            [same but largest zero
 region of 0s]                region filled with 3s
                              in a cross pattern]
```

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
| **120** | **547** | **Tuple for neighbor range** | `(i,i-(i>a),i+(i<b))` vs `I(i-(i>a),...)` |

### Key Breakthrough: Algorithm Swap (Gen 85)

The biggest single improvement came from **switching algorithms**:

- **Old (O(n²) histogram)**: Complex stack-based maximal rectangle (712 bytes)
- **New (O(n⁴) brute-force)**: Simple nested loops checking all rectangles (642 bytes)

Despite being computationally slower, the brute-force approach is **70 bytes shorter** because:
1. No histogram construction or stack manipulation
2. Single `max()` comprehension vs complex while loop
3. All conditions inline in one expression

### Champion Solution (547 bytes)

```python
def solve(G):
 I=range;R,C=len(G),len(G[0]);O=[*map(list,G)]
 if(b:=max([-~(c-a)*-~(k-d),a,d,c,k]for a in I(R)for d in I(C)for c in I(a,R)for k in I(d,C)if all(O[r][j]<1for r in I(a,c+1)for j in I(d,k+1)))or[0,])[0]<1:return G
 _,e,f,g,j=b;H=j-f>g-e;e+=e>0;g-=H*(g<R-1);f+=1-H;j-=1-H
 for r in I(e,g+1):G[r][f:j+1]=[3]*(j-f+1)
 [G[(v,i)[z]].__setitem__((i,v)[z],3)for a,b,*P,z in((e,g,I(f),I(j+1,C),1),(f,j,I(e),I(g+1,R),0))for i in I(a,b+1)for L in P if L and all(O[(v,w)[z]][(w,v)[z]]<1for w in(i,i-(i>a),i+(i<b))for v in L)for v in L]
 return G
```

**Improvement**: ~1200 → 547 bytes (**-54%**, +653 competition points)

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
| max() for best | `b=max(b,new)` vs `if new>b:b=new` | 5+ bytes |
| Unified dimension swap | `O[(v,i)[z]][(i,v)[z]]` for row/col toggling | 10+ bytes |
| Merged loops | Combine histogram + rect finding in one pass | 15+ bytes |
| `[*map(list,G)]` | Shorter deep copy than `[r[:]for r in G]` | 2 bytes |
| `I=range` alias | When range used 5+ times, alias saves bytes | 3+ bytes/use |
| `x and Y or Z` | Shorter than `Y if x else Z` for truthy Y | 2 bytes |
| E with default | `E(i,L,z,d=0):d or F` for edge-case bypass | 2+ bytes |
| **Algorithm swap** | O(n⁴) brute-force can be shorter than O(n²) | 70+ bytes |
| `-~x` for `x+1` | Bitwise not trick: `-~(c-a)` = `c-a+1` | 1 byte |
| `[0,]` fallback | Shorter than `[(0,)]` for empty fallback | 2 bytes |
| Range truthiness | `if L` works for empty `range()` checks | 4+ bytes |
| Lists in tuples | `[I(f),I(j+1,C)]` vs `[(I(f),f),...]` pairs | 10+ bytes |
| Merged conditionals | `I(i-(i>A),i+(i<B)+1)` combines 3 checks | 37 bytes |
| Tuple iteration | `for A,B,P,M,z in(t1),(t2):` vs list concat | 7 bytes |
| **Single list comp** | `[f()for...for v in L]` flattens nested loops | 22 bytes |
| **Tuple vs range** | `(i,i-(i>a),i+(i<b))` vs `I(i-(i>a),...)` | 2 bytes |
| `*P` unpacking | `for a,b,*P,z in...` captures middle elements | 2 bytes |

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
│   ├── a64e4611.py             # 547 bytes (champion)
│   └── 017c7c7b.py             # 54 bytes (baseline)
└── mutations/                   # Evolution logs
    ├── arc_fill_enclosed_regions.md
    ├── 0520fde7_evolution.md
    └── a64e4611_evolution.md
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
| Tasks solved | 4 | 400 |
| Total score | ~8,450 | ~960,000+ |
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
