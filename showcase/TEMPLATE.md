# Showcase Template

Use this template when creating new showcase entries. Each showcase should be self-contained, reproducible, and educational.

---

## Directory Structure

```
showcase/<problem-name>/
├── README.md           # Main documentation (follow template below)
├── rust/               # Or python/, go/, etc.
│   ├── Cargo.toml      # Build configuration
│   ├── Cargo.lock      # Locked dependencies (for reproducibility)
│   └── src/
│       ├── lib.rs      # Core algorithm trait/interface
│       ├── baselines.rs # Known algorithms to compare against
│       ├── evolved.rs  # The winning evolved algorithm
│       └── benchmark.rs # Benchmark harness with embedded test data
└── (optional) mutations/ # Archive of evolution attempts
```

---

## README.md Template

Copy and adapt this structure:

```markdown
# [Problem Name]: [Achievement Summary]

One-sentence summary of what was achieved.

## Results Summary

| Algorithm | Metric | Comparison |
|-----------|--------|------------|
| **Evolved (ours)** | **X.XX** | baseline |
| Baseline 1 | Y.YY | +/-Z% |
| Baseline 2 | Z.ZZ | +/-W% |

**Improvement over [best baseline]: X.X% relative**

---

## Why This Problem Matters

### The Problem
- What is the problem?
- Why is it important?
- What are real-world applications?

### The Challenge
- Why is this problem hard?
- What makes it interesting for evolutionary approaches?

### Prior Art
- What were the best known solutions?
- What research exists on this problem?

### Why This Result Matters
- What does beating the baseline demonstrate?
- What new insights were gained?

---

## The Evolution Journey

### Generation 1: [Phase Name]

| Mutation | Result | Approach | Why It Failed/Succeeded |
|----------|--------|----------|------------------------|
| `name` | X.XX | Description | Explanation |

**Key Learning**: What did we learn from this generation?

### Generation 2: [Phase Name]
(Continue for each significant generation)

### The Winning Mutation

- What combination of ideas produced the winner?
- Why does it work better than alternatives?
- What's the key insight?

```code
// Show the key algorithmic insight
```

---

## Quick Start

### Prerequisites
- List required tools and versions

### Run the Benchmark
```bash
# Commands to build and run
```

### Expected Output
```
What the user should see
```

---

## Technical Details

### Dataset
- What data is used?
- How is it generated/sourced?
- Why this dataset?

### Metric
- How is performance measured?
- What does the metric mean?

### Algorithm Interface
```code
// The trait/interface that algorithms implement
```

---

## The Winning Algorithm

```code
// Complete winning algorithm with comments
```

### Key Innovations
1. First key innovation
2. Second key innovation
3. (etc.)

---

## Reproducing from Scratch

Step-by-step instructions that anyone can follow without Claude:

### Step 1: Build
```bash
commands
```

### Step 2: Run
```bash
commands
```

### Step 3: Verify
- How to confirm the result is correct
- What to check

---

## File Structure

```
directory/
└── tree
```

---

## References

- Links to papers, code, prior work

---

## Deterministic Reproduction

Confirm:
- [ ] No external data files required
- [ ] No network requests
- [ ] No randomness (or seeded randomness)
- [ ] Same results every run
```

---

## Checklist Before Committing

- [ ] README includes "Why This Problem Matters" section
- [ ] README includes "The Evolution Journey" with generation-by-generation breakdown
- [ ] README includes step-by-step reproduction instructions
- [ ] Benchmark runs without errors: `cargo build --release && cargo run --release --bin benchmark`
- [ ] Results are deterministic (run twice, same output)
- [ ] No external dependencies (data embedded in code)
- [ ] All baselines included for comparison
- [ ] Cargo.lock included for reproducible builds
- [ ] target/ directory NOT included (add to .gitignore)

---

## Style Guidelines

1. **Be specific about numbers**: "7.4% improvement" not "significant improvement"
2. **Explain failures**: Failed mutations are as educational as successes
3. **Show the math**: Include the actual formulas, not just descriptions
4. **Include code**: The winning algorithm should be fully visible in the README
5. **Be reproducible**: Anyone should be able to verify the result
6. **Credit prior work**: Link to papers and code you built upon
