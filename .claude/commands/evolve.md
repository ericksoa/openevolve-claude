---
description: Evolve novel algorithms through LLM-driven mutation and selection
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Task, TodoWrite
argument-hint: <problem description>
---

# /evolve - Evolutionary Algorithm Discovery

Evolve novel algorithms through LLM-driven mutation and selection. Automatically generates Rust benchmarks and evaluators for any problem.

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

### Step 0: Generate Benchmark Infrastructure

Before evolution can begin, create a complete Rust benchmark harness in `.evolve/<problem-name>/`:

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
- A naive/simple implementation
- The standard library approach (if applicable)
- 1-2 optimized known algorithms

These are the targets to beat.

#### 0e. Generate evolved.rs

Start with a simple working implementation that:
- Implements the trait correctly
- Passes all tests
- Serves as the starting point for evolution

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
3. Runs `cargo clean && cargo build --release`
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

1. Run evaluator on initial implementation
2. Report all baseline speeds to user
3. Identify the target to beat

### Step 2: Evolution Loop

Run 3-5 generations:

#### 2a. Generate Mutations (PARALLEL)

Spawn 8 mutation agents in parallel using the Task tool. Each agent receives:
- The current best code
- The trait definition and requirements
- A specific mutation strategy

Mutation strategies:
- **tweak**: Micro-optimizations (cache values, reorder branches, inline)
- **unroll**: Loop unrolling, batch processing
- **specialize**: Fast paths for common cases (small inputs, specific values)
- **vectorize**: SIMD-friendly patterns, word-at-a-time processing
- **memoize**: Lookup tables, precomputation
- **restructure**: Different algorithmic approach
- **hybrid**: Combine techniques from multiple baselines
- **alien**: Radically different approach

Agent prompt template:
```
You are an algorithm optimizer. Improve this Rust code for SPEED.

TRAIT TO IMPLEMENT:
<trait definition>

CURRENT BEST CODE:
<code>

STRATEGY: <strategy>

Requirements:
- Must implement the trait exactly
- Must pass all correctness tests
- Focus purely on PERFORMANCE
- Use unsafe if it helps (with bounds checking where needed)

Return ONLY the complete Rust code for evolved.rs, no explanations.
The code must start with the imports and struct definition.
```

#### 2b. Evaluate Each Mutation

For each mutation:
1. Write to `.evolve/<problem>/mutations/gen<N>_<strategy>.rs`
2. Run evaluator: `python3 evaluator.py <path>`
3. Parse JSON result
4. Track: fitness, ops_per_second, vs_best_baseline

#### 2c. Selection

- Rank mutations by fitness
- Keep top 3-5 as parents for next generation
- Report generation results to user

### Step 3: Finalize

1. Write champion to `evolved.rs`
2. Run final benchmark
3. Report results:

```
Evolution Complete!

Problem: <problem description>
Generations: N
Mutations tested: M

Baselines:
  - std:      X ops/sec
  - naive:    Y ops/sec
  - optimized: Z ops/sec

Champion:   W ops/sec
Improvement: +P% vs best baseline

Champion saved to: .evolve/<problem>/rust/src/evolved.rs
```

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

## Key Principles

1. **Correctness First**: Any mutation that fails tests gets fitness = 0
2. **Parallel Mutations**: Always spawn mutation agents in parallel for speed
3. **Preserve Winners**: Never lose the best solution found so far
4. **Diverse Strategies**: Use different mutation strategies to explore the space
5. **Realistic Benchmarks**: Test data should reflect real-world usage patterns
