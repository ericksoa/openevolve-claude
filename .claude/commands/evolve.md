---
description: Evolve novel algorithms through LLM-driven mutation, crossover, and selection
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Task, TodoWrite, WebSearch, WebFetch, AskUserQuestion
argument-hint: <problem description>
---

# /evolve - Evolutionary Algorithm Discovery

Evolve novel algorithms through LLM-driven mutation and selection with **true genetic recombination**. Runs adaptively—continuing while improvement is possible, stopping when plateaued.

## Core Features

1. **Population-based**: Maintains top 4 diverse solutions, not just the winner
2. **Semantic crossover**: Combines innovations from multiple parents
3. **Adaptive generations**: Continues while improving, stops on plateau
4. **Budget control**: User sets token/generation limits
5. **Checkpointing**: Resume evolution from where you left off

## Usage

```
/evolve <problem description>
/evolve <problem description> --budget <tokens|generations>
/evolve --resume  # Continue previous evolution
```

**Examples**:
```
/evolve sorting algorithm for integers
/evolve string search --budget 50k        # ~50,000 tokens max
/evolve integer parsing --budget 20gen    # Max 20 generations
/evolve hash function --budget unlimited  # Run until plateau
/evolve --resume                          # Continue last evolution
```

**Budget Options**:
| Budget | Meaning | Approx. Generations |
|--------|---------|---------------------|
| `10k` | 10,000 tokens | ~2-3 generations |
| `50k` | 50,000 tokens | ~10-12 generations |
| `100k` | 100,000 tokens | ~20-25 generations |
| `5gen` | 5 generations | Fixed count |
| `unlimited` | No limit | Until plateau |
| (none) | Default 50k | ~10-12 generations |

---

## Execution Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Step -1: Bootstrap (first run only)                            │
│  Step 0-pre: Search for existing benchmarks                     │
│  Step 0: Generate benchmark infrastructure                      │
│  Step 1: Establish baseline                                     │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  EVOLUTION LOOP (adaptive)                                 │  │
│  │                                                            │  │
│  │  while budget_remaining AND improving:                     │  │
│  │    - Generation N: crossover + mutation                    │  │
│  │    - Evaluate offspring                                    │  │
│  │    - Update population                                     │  │
│  │    - Check stopping criteria                               │  │
│  │    - Checkpoint state                                      │  │
│  │    - If plateau: ask user to continue?                     │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Step Final: Report results                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step -1: Bootstrap (First Run Only)

Skip if `.evolve/.bootstrapped` exists.

1. **Check Rust Toolchain**:
   ```bash
   ~/.cargo/bin/cargo --version 2>/dev/null || cargo --version
   ```
   If missing, offer installation via AskUserQuestion.

2. **Check Python 3.10+**:
   ```bash
   python3 --version
   ```

3. **Create directories and mark complete**:
   ```bash
   mkdir -p .evolve
   touch .evolve/.bootstrapped
   ```

---

## Step 0-pre: Benchmark Discovery

Search web for existing benchmarks before generating synthetic ones.

Use WebSearch: `"<algorithm> benchmark rust github"`

Present options via AskUserQuestion, or proceed to synthetic benchmark.

---

## Step 0: Generate Benchmark Infrastructure

Create in `.evolve/<problem-name>/`:

```
.evolve/<problem-name>/
├── rust/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs        # Trait definition
│       ├── baselines.rs  # Known algorithms to beat
│       ├── evolved.rs    # Current champion
│       └── benchmark.rs  # Benchmark binary
├── evaluator.py          # Fitness evaluation
├── evolution.json        # Full evolution state (for resume)
└── mutations/            # All tested mutations
```

### evolution.json (Master State File)

This file enables resumption and tracks all evolution state:

```json
{
  "problem": "sorting algorithm for integers",
  "created": "2024-01-15T10:30:00Z",
  "updated": "2024-01-15T11:45:00Z",

  "problem_analysis": {
    "algorithm_families": ["quicksort", "mergesort", "heapsort", "radix", "counting", "shell", "timsort", "introsort"],
    "optimization_dimensions": ["cache", "simd", "branch_prediction", "small_array", "nearly_sorted", "memory"],
    "viable_strategies": 26,
    "gen1_agents": 26,
    "gen2_agents": 16
  },

  "budget": {
    "type": "tokens",
    "limit": 50000,
    "used": 23450,
    "remaining": 26550
  },

  "generation": 5,
  "status": "running",

  "baseline": {
    "naive": 1289,
    "std": 114592,
    "std_unstable": 168417
  },

  "champion": {
    "id": "gen4_crossover_radix_shell",
    "fitness": 0.94,
    "ops_per_second": 185000,
    "generation_discovered": 4
  },

  "population": [
    {
      "id": "gen4_crossover_radix_shell",
      "fitness": 0.94,
      "ops_per_second": 185000,
      "algorithm_family": "hybrid_radix_shell",
      "key_innovations": ["11-bit radix", "insertion base case", "gap presort"],
      "parents": ["gen3_radix_quick", "gen2_shellsort"],
      "code_path": "mutations/gen4_crossover_radix_shell.rs"
    }
    // ... 3 more
  ],

  "history": [
    {
      "generation": 1,
      "best_fitness": 0.89,
      "best_ops": 156000,
      "best_id": "gen1_radix",
      "improvement": null,
      "tokens_used": 4500
    },
    {
      "generation": 2,
      "best_fitness": 0.91,
      "best_ops": 172000,
      "best_id": "gen2_crossover_radix_quick",
      "improvement": 0.02,
      "tokens_used": 4200
    },
    {
      "generation": 3,
      "best_fitness": 0.91,
      "best_ops": 173000,
      "best_id": "gen2_crossover_radix_quick",
      "improvement": 0.00,
      "plateau_count": 1,
      "tokens_used": 4100
    }
  ],

  "stopping": {
    "plateau_count": 0,
    "plateau_threshold": 3,
    "min_improvement": 0.005,
    "max_generations": null
  }
}
```

---

## Step 1: Establish Baseline & Analyze Problem

1. Run evaluator on naive implementation
2. Report baseline speeds
3. **Analyze problem to determine viable strategies**

### Problem Analysis

Analyze the problem to estimate:
- **Algorithm families**: How many fundamentally different approaches exist?
- **Optimization dimensions**: What can be optimized (cache, SIMD, branches, memory)?
- **Input characteristics**: What variations matter (size, distribution, patterns)?

```
Example analysis for "sorting integers":

Algorithm families (8+):
  - Comparison: quicksort, mergesort, heapsort, shellsort, timsort
  - Distribution: radix sort, counting sort, bucket sort
  - Hybrid: introsort, pdqsort, pattern-defeating

Optimization dimensions (6):
  - Cache efficiency, SIMD, branch prediction, memory allocation
  - Small-array specialization, nearly-sorted detection

Estimated viable strategies: 14
Recommended agents: 16 (Gen1), 12 (Gen2+)
```

### Agent Scaling Formula

```python
def estimate_agents(problem_analysis):
    # Base: number of distinct algorithm families
    algo_families = len(problem_analysis["algorithm_families"])

    # Add optimization dimensions (each can be applied to top algos)
    opt_dimensions = len(problem_analysis["optimization_dimensions"])

    # Viable strategies = families + (top_3_families × opt_dimensions)
    viable_strategies = algo_families + min(3, algo_families) * opt_dimensions

    # Gen1: Explore all viable strategies (cap at 32)
    gen1_agents = min(viable_strategies, 32)

    # Gen2+: Crossover pairs + mutations (scale with population)
    gen2_agents = min(viable_strategies // 2 + 4, 24)

    return {
        "gen1_agents": gen1_agents,
        "gen2_agents": gen2_agents,
        "viable_strategies": viable_strategies
    }
```

### Smart Budget Recommendation

Based on analysis, recommend budget and ask user:

```
Evolution ready for: sorting algorithm for integers

Baselines:
  bubble (naive):    1,289 ops/sec
  std:             114,592 ops/sec
  std_unstable:    168,417 ops/sec  ← target

Problem Analysis:
  Algorithm families: 8 (comparison, distribution, hybrid)
  Optimization dimensions: 6
  Viable strategies: 14

Recommended: 16 agents/gen, ~6k tokens/gen

Budget Options:
1. Quick (10k) - 2 gens, 8 agents each [minimal exploration]
2. Standard (50k) - 8 gens, 16 agents Gen1, 12 Gen2+ [Recommended]
3. Deep (100k) - 16 gens, 24 agents Gen1, 16 Gen2+ [thorough]
4. Maximum (200k) - 32 gens, 32 agents Gen1, 24 Gen2+ [exhaustive]
5. Unlimited - run until plateau

⚡ For this problem, Standard (50k) should explore most viable algorithms.
   Deep (100k) recommended if you want thorough hybrid combinations.
```

### Dynamic Scaling Examples

| Problem | Algo Families | Opt Dims | Viable | Gen1 Agents | Gen2+ Agents |
|---------|--------------|----------|--------|-------------|--------------|
| Fibonacci | 4 | 2 | 10 | 10 | 8 |
| Sorting | 8 | 6 | 26 | 26 | 16 |
| String search | 6 | 4 | 18 | 18 | 12 |
| Hash function | 10 | 5 | 25 | 25 | 16 |
| Integer parsing | 3 | 4 | 15 | 15 | 10 |

Store analysis and agent counts in `evolution.json`.

---

## Step 2: Evolution Loop (Adaptive)

### Token Estimation

Tokens scale with agent count:
```python
def estimate_tokens_per_gen(agent_count):
    return agent_count * 400 + 800  # ~400 tokens/agent + overhead

# Examples:
#   8 agents  → ~4,000 tokens/gen
#  16 agents  → ~7,200 tokens/gen
#  32 agents  → ~13,600 tokens/gen
```

### Generation 1: Divergent Exploration

Spawn **N parallel mutation agents** where N = `gen1_agents` from problem analysis.

Generate strategy list dynamically based on problem:
```python
def generate_strategies(problem_analysis, agent_count):
    strategies = []

    # 1. One agent per algorithm family
    for family in problem_analysis["algorithm_families"]:
        strategies.append(f"implement_{family}")

    # 2. Optimization variants of top families
    top_families = problem_analysis["algorithm_families"][:3]
    for family in top_families:
        for opt in problem_analysis["optimization_dimensions"]:
            strategies.append(f"{family}_{opt}")

    # 3. Fill remaining with general strategies
    general = ["tweak", "unroll", "specialize", "vectorize",
               "memoize", "restructure", "hybrid", "alien",
               "simd", "branch_free", "cache_friendly", "unsafe_opt"]

    while len(strategies) < agent_count:
        strategies.extend(general)

    return strategies[:agent_count]
```

Evaluate all, extract innovations, select top 4 with diversity.

**Update evolution.json** after each generation.

### Generation 2+: Crossover + Mutation

Each generation uses `gen2_agents` from problem analysis:

```python
def allocate_gen2_agents(agent_count, population):
    # Half crossover, half mutation
    crossover_count = agent_count // 2
    mutation_count = agent_count - crossover_count

    # Crossover: pair top performers
    crossover_pairs = [(pop[i], pop[j])
                       for i in range(len(pop))
                       for j in range(i+1, len(pop))][:crossover_count]

    # Mutation: apply diverse strategies to top performers
    mutation_targets = population[:mutation_count]

    return crossover_pairs, mutation_targets
```

Each generation:
1. **Budget check**: Is there budget remaining?
2. **N/2 crossover agents**: Combine parent pairs
3. **N/2 mutation agents**: Refine top performers
4. **Evaluate** all N offspring in parallel
5. **Extract innovations** from successful ones
6. **Select** new population with diversity + elitism
7. **Update** evolution.json with new state
8. **Check stopping criteria**

### Adaptive Stopping Criteria

After each generation, evaluate:

```python
def should_stop(evolution_state):
    # 1. Budget exhausted
    if budget_used >= budget_limit:
        return True, "budget_exhausted"

    # 2. Plateau detected (no improvement for N generations)
    if plateau_count >= plateau_threshold:
        return True, "plateau"

    # 3. Target achieved (if specified)
    if champion_fitness >= target_fitness:
        return True, "target_achieved"

    # 4. Max generations (if specified)
    if generation >= max_generations:
        return True, "max_generations"

    return False, None
```

### Plateau Detection

Track improvement across generations:

```python
def update_plateau_status(current_fitness, previous_fitness, min_improvement=0.005):
    improvement = current_fitness - previous_fitness

    if improvement < min_improvement:
        plateau_count += 1
    else:
        plateau_count = 0  # Reset on improvement

    return plateau_count
```

**Plateau threshold**: 3 generations without meaningful improvement (>0.5%)

### User Checkpoint (On Plateau or Every N Generations)

When plateau detected OR every 5 generations with `unlimited` budget:

```
Generation 8 Complete:
  Champion: 185K ops/sec (+14,254% vs bubble, +10% vs std_unstable)

  ⚠️  Plateau detected: No improvement for 3 generations

  Budget used: 34,200 / 50,000 tokens (68%)

  Options:
  1. Continue evolution (may find breakthrough)
  2. Stop and save champion
  3. Try radical mutations only (higher variance)
```

Use AskUserQuestion to let user decide.

### Breakthrough Detection

If improvement after plateau:

```
🎉 Breakthrough in Generation 9!
  Previous best: 185K ops/sec
  New champion:  201K ops/sec (+8.6%)

  Innovation: SIMD-friendly radix bucket distribution

  Plateau reset. Continuing evolution...
```

---

## Step 3: Checkpointing & Resume

### After Each Generation

Write complete state to `evolution.json`:
- Current population
- Champion
- History with fitness trajectory
- Budget usage
- Plateau count

### Resume Command

When user runs `/evolve --resume`:

1. Find most recent `evolution.json` in `.evolve/*/`
2. Load state
3. Report current status:

```
Resuming evolution: sorting algorithm for integers

Status: Paused at generation 5
Champion: 185K ops/sec (gen4_crossover_radix_shell)
Budget remaining: 26,550 / 50,000 tokens

Last 3 generations:
  Gen 3: 173K ops/sec (plateau 1)
  Gen 4: 185K ops/sec (breakthrough!)
  Gen 5: 185K ops/sec (plateau 1)

Continue evolution?
```

4. Resume from saved population state

---

## Step 4: Finalize

When evolution stops (any reason):

```
Evolution Complete!

Problem: sorting algorithm for integers
Generations: 12
Total tokens: 48,750
Stop reason: plateau (3 generations without improvement)

Evolution Trajectory:
  Gen  1: 156K ops/sec  radix_sort discovered
  Gen  2: 172K ops/sec  radix+quicksort hybrid (+10%)
  Gen  3: 173K ops/sec  minor refinement (+0.5%)
  Gen  4: 185K ops/sec  breakthrough: added shellsort presort (+7%)
  Gen  5: 185K ops/sec  plateau
  ...
  Gen 12: 189K ops/sec  final refinement

Baselines:
  bubble:        1.3K ops/sec (naive starting point)
  std:         115K ops/sec
  std_unstable: 168K ops/sec

Champion: 189K ops/sec
  vs bubble:       +14,538% (146x faster)
  vs std_unstable: +12.5%

Key Innovations in Champion:
  - 11-bit radix buckets (Gen 1)
  - Sign-bit flip for negatives (Gen 1)
  - Insertion sort for n < 32 (Gen 4)
  - Nearly-sorted detection (Gen 4)

Champion saved to: .evolve/sorting/rust/src/evolved.rs
State saved to: .evolve/sorting/evolution.json

To continue evolution later: /evolve --resume
```

---

## Mutation & Crossover Prompts

### Mutation Agent Prompt

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
```

### Crossover Agent Prompt

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

Return ONLY the complete Rust code for evolved.rs, no explanations.
```

### Innovation Extraction Prompt

```
Analyze this algorithm implementation and extract its key innovations.

CODE:
<code>

PERFORMANCE: <ops_per_second> ops/sec

Respond in this exact JSON format:
{
  "algorithm_family": "<e.g., radix_sort, quicksort, lookup_table, simd, etc.>",
  "key_innovations": ["<technique 1>", "<technique 2>", ...],
  "strengths": ["<fast on what>", ...],
  "weaknesses": ["<slow on what>", ...],
  "complexity": {"time": "<O(?)>", "space": "<O(?)>"}
}
```

---

## Radical Mutation Mode

When user chooses "radical mutations only" after plateau:

Spawn 8 agents with high-variance strategies:
- **alien**: Completely different algorithm family
- **alien2**: Another alien approach
- **restructure**: Fundamental reorganization
- **complexity_change**: Try different complexity class (e.g., O(n²) → O(n))
- **simd**: Explicit SIMD vectorization
- **unsafe_aggressive**: Aggressive unsafe optimizations
- **lookup_heavy**: Maximum precomputation
- **branch_free**: Eliminate all branches

This increases variance to escape local optima at the cost of more failed mutations.

---

## Key Principles

1. **Adaptive by default**: Continues while improving, stops on plateau
2. **User control**: Budget limits prevent runaway token usage
3. **Resumable**: Full state checkpointed after each generation
4. **Transparent**: Clear reporting of improvement trajectory
5. **Correctness first**: Failed tests = fitness 0
6. **Diversity maintained**: Population represents multiple algorithm families
7. **Elitism**: Never lose the best solution

---

## Token Budget Reference

Budget scales with problem complexity (agents × generations):

| Budget | Simple (10 agents) | Medium (16 agents) | Complex (26 agents) |
|--------|-------------------|-------------------|---------------------|
| 10k | 2 gens | 1 gen | 1 gen |
| 50k | 10 gens | 6 gens | 4 gens |
| 100k | 20 gens | 12 gens | 7 gens |
| 200k | 40 gens | 25 gens | 15 gens |

**Rule of thumb**: More agents = better Gen1 exploration, more generations = better refinement.

The system will recommend based on problem analysis:
- Simple problems (fibonacci): fewer agents, more generations
- Complex problems (sorting): more agents to explore algorithm space

Most improvements happen in first 5-10 generations. More agents help for:
- Large algorithm spaces (many valid approaches)
- Problems with multiple optimization dimensions
- Finding non-obvious hybrid combinations
