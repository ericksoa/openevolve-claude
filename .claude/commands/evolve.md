---
description: Evolve novel algorithms through LLM-driven mutation, crossover, and selection
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Task, TodoWrite, WebSearch, WebFetch, AskUserQuestion
argument-hint: <problem description>
---

# /evolve - Evolutionary Algorithm Discovery

Evolve novel algorithms through LLM-driven mutation and selection with **true genetic recombination**. Automatically generates Rust benchmarks and evaluators for any problem.

## Core Innovation: Semantic Crossover

Unlike simple "make it faster" loops, this skill implements real genetic algorithm principles:

1. **Population-based**: Maintains top 4 diverse solutions, not just the winner
2. **Innovation extraction**: Identifies what makes each solution fast
3. **Crossover**: Combines innovations from multiple parents
4. **Diversity pressure**: Prevents convergence to a single algorithm family

This means more generations = genuinely better results through trait combination.

## Usage

```
/evolve <problem description>
```

Examples:
- `/evolve integer parsing - beat std library`
- `/evolve string search - beat Boyer-Moore`
- `/evolve sorting algorithm for nearly-sorted arrays`
- `/evolve hash function with minimal collisions`
- `/evolve fibonacci with matrix exponentiation`

## Execution

### Step -1: Bootstrap (First Run Only)

On first invocation, check and set up the environment. Skip this step if `.evolve/.bootstrapped` exists.

1. **Check Rust Toolchain**:
   ```bash
   ~/.cargo/bin/cargo --version 2>/dev/null || cargo --version
   ```

   If cargo is not found, use AskUserQuestion to offer installation:
   ```
   Rust toolchain is required but not found.

   Options:
   1. Install via rustup (recommended)
   2. I'll install it manually
   ```

   If user chooses rustup:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
   source ~/.cargo/env
   ```

2. **Check Python 3.10+**:
   ```bash
   python3 --version
   ```

   If Python < 3.10 or not found, inform user:
   "Python 3.10+ is required. Please install from python.org or via your package manager."

3. **Create local .evolve directory**:
   ```bash
   mkdir -p .evolve
   ```

4. **Mark bootstrap complete**:
   ```bash
   touch .evolve/.bootstrapped
   ```

   Report: "Bootstrap complete! Environment ready for evolution."

---

### Step 0-pre: Benchmark Discovery (Web Search)

Before generating synthetic benchmarks, search the web for existing high-quality benchmarks:

1. **Web Search for Benchmarks**:
   Use WebSearch to find relevant benchmarks. Construct queries like:
   - `"<algorithm> benchmark rust github"`
   - `"<algorithm> performance comparison rust"`
   - `"<algorithm> crate benchmark rust"`

   Look for:
   - GitHub repos with benchmark suites
   - Crates.io packages with `bench` directories
   - Published performance comparisons

2. **Evaluate Search Results**:
   For each promising result, assess:
   - Repository activity (recent commits, stars)
   - Benchmark quality (realistic test data, multiple implementations)
   - API compatibility (can we implement their trait?)

   Use WebFetch to examine promising repos if needed.

3. **Present Options to User**:
   Use AskUserQuestion:
   ```
   Found benchmark options for {problem}:

   1. {repo1} - {description} ({stars} stars)
   2. {repo2} - {description} ({stars} stars)
   3. Generate synthetic benchmark from scratch
   ```

4. **If User Selects External**:
   - Clone to `.evolve/<problem>/external/`
   - Analyze the benchmark interface
   - Generate compatible `evolved.rs`
   - Adapt `evaluator.py` to run their benchmark
   - Skip to Step 1 (Establish Baseline)

5. **If Synthetic or No Good Results**: Proceed to Step 0 (generate synthetic benchmark)

---

### Step 0: Generate Benchmark Infrastructure

If no external benchmark is used, create a complete Rust benchmark harness in `.evolve/<problem-name>/`:

#### 0a. Analyze the Problem

1. Parse the problem description to understand:
   - What function/algorithm to optimize
   - Input/output types
   - Success criteria (speed, memory, accuracy)
   - Known baseline algorithms to compare against

2. Create directory structure:
   ```
   .evolve/<problem-name>/
   ├── rust/
   │   ├── Cargo.toml
   │   └── src/
   │       ├── lib.rs        # Trait definition
   │       ├── baselines.rs  # Known algorithms to beat
   │       ├── evolved.rs    # Current best (mutations go here)
   │       └── benchmark.rs  # Benchmark binary
   ├── evaluator.py          # Fitness evaluation script
   ├── population.json       # Current population state
   └── mutations/            # Store all mutations by generation
   ```

#### 0b. Generate Cargo.toml

```toml
[package]
name = "<problem_name>"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
rand = "0.8"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"

[[bin]]
name = "benchmark"
path = "src/benchmark.rs"
```

#### 0c. Generate lib.rs with Trait

Define a trait that all implementations must satisfy:

```rust
//! <Problem> Benchmark
pub mod baselines;
pub mod evolved;

/// Trait for <problem> implementations
pub trait <TraitName> {
    fn <method>(&self, <inputs>) -> <output>;
}

#[cfg(test)]
mod tests {
    // Correctness tests for all implementations
}
```

#### 0d. Generate baselines.rs

Implement 2-4 known baseline algorithms:
- A naive/simple implementation (the "bad" algorithm to dramatically beat)
- The standard library approach (if applicable)
- 1-2 optimized known algorithms

These are the targets to beat.

#### 0e. Generate evolved.rs

Start with the naive/simple implementation that:
- Implements the trait correctly
- Passes all tests
- Is intentionally suboptimal (this is what we evolve FROM)

#### 0f. Generate benchmark.rs

Create a benchmark binary that:
1. Generates realistic test data (mix of edge cases and common cases)
2. Verifies correctness of all implementations
3. Times each implementation with warmup
4. Outputs JSON with results:

```rust
#[derive(Serialize)]
struct FullResults {
    results: Vec<BenchmarkResult>,
    correctness: bool,
}
```

#### 0g. Generate evaluator.py

Python script that:
1. Accepts an optional path to evolved code
2. Copies it to evolved.rs if provided
3. Runs `cargo build --release`
4. Runs the benchmark
5. Computes fitness score (0.0-1.0)
6. Outputs JSON:

```json
{
  "fitness": 0.85,
  "ops_per_second": 12345678,
  "vs_best_baseline": 15.2,
  "correctness": true,
  "all_results": {...}
}
```

#### 0h. Verify Setup

Run the evaluator to ensure everything compiles and works:
```bash
cd .evolve/<problem-name> && python3 evaluator.py
```

Report baseline results to user before proceeding.

### Step 1: Establish Baseline

1. Run evaluator on initial (naive) implementation
2. Report all baseline speeds to user
3. Identify the targets to beat

---

## Step 2: Evolution Loop (Population-Based with Crossover)

Run 3-5 generations with **true genetic recombination**.

### Generation 1: Divergent Exploration

#### 2a. Generate Initial Mutations (PARALLEL - 8 agents)

Spawn 8 mutation agents in parallel using the Task tool. Each agent receives:
- The current seed code
- The trait definition and requirements
- A specific mutation strategy

**Mutation strategies**:
- **tweak**: Micro-optimizations (cache values, reorder branches, inline)
- **unroll**: Loop unrolling, batch processing
- **specialize**: Fast paths for common cases (small inputs, specific values)
- **vectorize**: SIMD-friendly patterns, word-at-a-time processing
- **memoize**: Lookup tables, precomputation
- **restructure**: Different algorithmic approach entirely
- **hybrid**: Combine techniques from multiple known algorithms
- **alien**: Radically different approach (e.g., different complexity class)

**Mutation agent prompt template**:
```
You are an algorithm optimizer. Improve this Rust code for SPEED.

TRAIT TO IMPLEMENT:
<trait definition>

CURRENT CODE:
<code>

STRATEGY: <strategy>

Requirements:
- Must implement the trait exactly
- Must pass all correctness tests
- Focus purely on PERFORMANCE
- Use unsafe if it helps (with proper safety invariants)

Return ONLY the complete Rust code for evolved.rs, no explanations.
The code must start with the imports and struct definition.
```

#### 2b. Evaluate All Mutations

For each mutation:
1. Write to `.evolve/<problem>/mutations/gen1_<strategy>.rs`
2. Run evaluator: `python3 evaluator.py <path>`
3. Parse JSON result
4. Track: fitness, ops_per_second, correctness

#### 2c. Extract Innovations (CRITICAL FOR CROSSOVER)

For each successful mutation (correctness=true), spawn an analysis agent:

**Innovation extraction prompt**:
```
Analyze this algorithm implementation and extract its key innovations.

CODE:
<code>

PERFORMANCE: <ops_per_second> ops/sec

Respond in this exact JSON format:
{
  "algorithm_family": "<e.g., radix_sort, quicksort, lookup_table, simd, etc.>",
  "key_innovations": [
    "<specific technique 1>",
    "<specific technique 2>",
    "<specific technique 3>"
  ],
  "strengths": [
    "<what input types/sizes/patterns is this fast on>"
  ],
  "weaknesses": [
    "<what input types/sizes/patterns is this slow on>"
  ],
  "complexity": {
    "time": "<O(n), O(n log n), etc.>",
    "space": "<O(1), O(n), etc.>"
  }
}
```

#### 2d. Select Top 4 with Diversity

Rank by fitness, but enforce diversity:
- Cannot have more than 2 solutions from the same `algorithm_family`
- If top 4 would violate this, skip to next-best from different family

Store population state in `population.json`:
```json
{
  "generation": 1,
  "population": [
    {
      "id": "gen1_radix",
      "fitness": 0.89,
      "ops_per_second": 156000,
      "algorithm_family": "radix_sort",
      "key_innovations": ["11-bit buckets", "sign-bit flip"],
      "strengths": ["random data", "large arrays"],
      "weaknesses": ["nearly-sorted", "small arrays"],
      "code_path": "mutations/gen1_restructure.rs"
    },
    // ... 3 more
  ]
}
```

Report generation results to user:
```
Generation 1 Complete:
  Population:
    1. radix_sort     - 156K ops/sec (innovations: 11-bit buckets, sign-bit flip)
    2. quicksort      - 142K ops/sec (innovations: median-of-3, tail recursion)
    3. heapsort       - 98K ops/sec  (innovations: bottom-up heapify)
    4. shellsort      - 89K ops/sec  (innovations: Ciura gaps)

  Diversity: 4 algorithm families represented
  Best vs baseline: +7126% (71x faster than bubble sort)
```

---

### Generation 2+: Crossover + Mutation

#### 2e. Generate Crossover Offspring (PARALLEL - 4 agents)

Spawn 4 crossover agents, each combining 2 parents from the population.

**Crossover pairs** (ensure each parent participates at least once):
- Parent 1 × Parent 2
- Parent 1 × Parent 3
- Parent 2 × Parent 4
- Parent 3 × Parent 4

**Crossover agent prompt template**:
```
You are creating a HYBRID algorithm by combining two parent solutions.

TRAIT TO IMPLEMENT:
<trait definition>

PARENT A: <algorithm_family_a> (<ops_per_second_a> ops/sec)
Innovations: <key_innovations_a>
Strengths: <strengths_a>
Weaknesses: <weaknesses_a>

CODE A:
<code_a>

---

PARENT B: <algorithm_family_b> (<ops_per_second_b> ops/sec)
Innovations: <key_innovations_b>
Strengths: <strengths_b>
Weaknesses: <weaknesses_b>

CODE B:
<code_b>

---

Create a HYBRID solution that:
1. COMBINES key innovations from BOTH parents
2. Uses A's approach where A is strong, B's where B is strong
3. May dispatch based on input characteristics (size, pattern detection)
4. Inherits the best constants/thresholds from each

The goal is a solution FASTER than either parent by combining their strengths.

Requirements:
- Must implement the trait exactly
- Must pass all correctness tests
- Should incorporate specific techniques from BOTH parents

Return ONLY the complete Rust code for evolved.rs, no explanations.
```

#### 2f. Generate Mutation Offspring (PARALLEL - 4 agents)

Also spawn 4 mutation agents on the top 2 parents:
- Mutate Parent 1 with strategies: tweak, specialize
- Mutate Parent 2 with strategies: vectorize, unroll

Use the same mutation prompt as Generation 1.

#### 2g. Evaluate All Offspring (8 total)

Same as 2b - evaluate all 8 offspring.

#### 2h. Extract Innovations

Same as 2c - extract innovations from successful offspring.

#### 2i. Select New Population

Combine parents + offspring (12 solutions total), select top 4 with diversity.

**Elitism**: The best solution from the previous generation is always kept (cannot be displaced).

Report generation results.

---

### Step 3: Repeat for 3-5 Generations

Continue the crossover + mutation cycle. Each generation:
- 4 crossover offspring (combining parent innovations)
- 4 mutation offspring (refining existing solutions)
- Selection with diversity pressure

**Stopping criteria**:
- 5 generations completed, OR
- No improvement for 2 consecutive generations, OR
- Champion exceeds 2x best baseline performance

---

### Step 4: Finalize

1. Write champion (best overall) to `evolved.rs`
2. Run final benchmark
3. Report complete evolution history:

```
Evolution Complete!

Problem: <problem description>
Generations: N
Total mutations tested: M
Crossovers performed: C

Evolution History:
  Gen 1: radix_sort discovered (156K ops/sec)
  Gen 2: radix+quicksort hybrid (178K ops/sec) - combined bucket distribution + partition
  Gen 3: added insertion sort base case from shellsort (185K ops/sec)
  Gen 4: no improvement (plateau)
  Gen 5: breakthrough - SIMD-friendly radix (201K ops/sec)

Baselines:
  - bubble:       1.3K ops/sec (the naive starting point)
  - std:        115K ops/sec
  - std_unstable: 168K ops/sec

Champion:   201K ops/sec
Improvement: +15,369% vs bubble (154x faster)
            +20% vs std_unstable

Key Innovations in Champion:
  - 11-bit radix buckets (from Gen 1 radix)
  - Sign-bit flip for negative handling (from Gen 1 radix)
  - Insertion sort for n < 32 (from Gen 3 crossover with shellsort)
  - Cache-line aligned buffers (from Gen 5 mutation)

Champion saved to: .evolve/<problem>/rust/src/evolved.rs
```

---

## Population Data Structures

### population.json

```json
{
  "generation": 3,
  "champion": {
    "id": "gen2_crossover_radix_quick",
    "fitness": 0.92,
    "ops_per_second": 178000
  },
  "population": [
    {
      "id": "gen2_crossover_radix_quick",
      "fitness": 0.92,
      "ops_per_second": 178000,
      "algorithm_family": "hybrid_radix_quick",
      "key_innovations": [
        "11-bit radix for large arrays",
        "quicksort partition for medium",
        "insertion for small"
      ],
      "parents": ["gen1_radix", "gen1_quicksort"],
      "strengths": ["all sizes", "random data"],
      "weaknesses": ["nearly-sorted large arrays"],
      "code_path": "mutations/gen2_crossover_radix_quick.rs"
    }
    // ... 3 more
  ],
  "history": [
    {"generation": 1, "best_fitness": 0.89, "best_id": "gen1_radix"},
    {"generation": 2, "best_fitness": 0.92, "best_id": "gen2_crossover_radix_quick"}
  ]
}
```

---

## Fitness Function

```python
# Base: speed ratio to best baseline
speed_ratio = evolved_speed / best_baseline_speed

# Scale to 0-1, cap at 2x improvement
fitness = min(speed_ratio, 2.0) / 2.0

# Bonus for beating all baselines
if evolved_speed > best_baseline_speed:
    fitness = min(fitness + 0.1, 1.0)

# Correctness gate: 0 if tests fail
if not correctness:
    fitness = 0.0
```

---

## Key Principles

1. **Correctness First**: Any mutation/crossover that fails tests gets fitness = 0
2. **Parallel Execution**: Always spawn agents in parallel for speed
3. **True Recombination**: Crossover combines innovations, not just picks winner
4. **Diversity Pressure**: Population must represent multiple algorithm families
5. **Elitism**: Never lose the best solution found so far
6. **Innovation Tracking**: Know WHY each solution is fast, enabling intelligent crossover
7. **Realistic Benchmarks**: Test data should reflect real-world usage patterns

---

## Why This Works Better Than "Make It Faster" Loops

| Simple Loop | Population + Crossover |
|-------------|------------------------|
| Refines one lineage | Combines multiple lineages |
| Plateus quickly | Escapes local optima via recombination |
| Gen 5 ≈ Gen 2 with more attempts | Gen 5 has traits from 5 generations |
| No diversity | Enforced algorithm diversity |
| No memory of what worked | Innovations tracked and recombined |

Example of compounding benefits:
- **Gen 1**: Discovers radix sort (fast distribution) and quicksort (fast partition)
- **Gen 2**: Crossover creates radix+quick hybrid
- **Gen 3**: Adds shellsort's gap sequence for nearly-sorted detection
- **Gen 4**: Incorporates heapsort as depth-limit fallback
- **Gen 5**: Final hybrid has traits from 4 original algorithm families

Each generation COMBINES innovations rather than just refining one approach.
