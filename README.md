# OpenEvolve

**Genetic algorithm optimization for any algorithmic problem.** Uses parallel Claude Code agents to evolve faster solutions through mutation, crossover, and selection—with Rust benchmarks for precise fitness measurement.

Works on: sorting, searching, parsing, hashing, compression, numeric algorithms, data structures, and any problem where performance can be measured.

## How It Works

The `/evolve` skill uses **parallel Claude Code agents** for true genetic algorithm evolution:

### Agent-Based Architecture
- **Dynamic agent scaling**: Analyzes problem to spawn 10-32 agents based on viable strategies
- **Semantic crossover**: Agents combine innovations from parent solutions
- **Adaptive stopping**: Runs until plateau (3 gens without >0.5% improvement)
- **Smart budget**: Recommends budget based on problem complexity

### Generation 1: Divergent Exploration
- Analyze problem → determine viable algorithm families and optimization dimensions
- Spawn N agents in parallel (one per viable strategy, typically 10-32)
- Extract **innovations** from each solution (what makes it fast?)
- Select top 4 with **diversity pressure** (max 2 from same algorithm family)

### Generation 2+: Crossover + Mutation
- **N/2 crossover agents**: Combine innovations from parent pairs
- **N/2 mutation agents**: Refine top performers
- Elitism: Never lose the champion
- Checkpoint state to `evolution.json` for resume

### Why Genetic Algorithms?

Unlike "make it faster" prompting, true genetic algorithms:
- **Explore diverse solutions** in parallel (not just refining one approach)
- **Combine innovations** from different algorithm families via crossover
- **Maintain population diversity** to avoid local optima
- **Preserve winners** while still exploring new territory

```
Gen 1: Discovers lookup tables (O(1)) + fast doubling (O(log n))
Gen 2: Crossover → hybrid with table for small n, doubling for large
Gen 3: Mutation adds unsafe access to eliminate bounds checks
```

## Quick Start

### 1. Install the Skill

```bash
# Option A: Clone and copy skill
git clone https://github.com/ericksoa/openevolve-claude
cp openevolve-claude/.claude/commands/evolve.md ~/.claude/commands/

# Option B: Direct download
curl -o ~/.claude/commands/evolve.md \
  https://raw.githubusercontent.com/ericksoa/openevolve-claude/main/.claude/commands/evolve.md
```

### 2. Use It

```bash
claude
> /evolve fibonacci sequence
> /evolve integer sorting
> /evolve substring search
> /evolve hash function for strings
> /evolve JSON parser
> /evolve LRU cache
```

The system will:
1. Analyze the problem → estimate viable algorithm families
2. Generate Rust benchmarks with baselines to beat
3. Spawn 10-32 agents in parallel (scales with problem complexity)
4. Run evolution until plateau or budget exhausted
5. Report champion with full innovation lineage

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         /evolve <problem>                        │
│                                                                  │
│  1. Analyze problem → identify algorithm families & optimizations│
│  2. Scale agents: 10-32 based on viable strategies              │
│  3. Generate Rust benchmark with trait + baselines              │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GENERATION 1: Exploration                     │
│                                                                  │
│  Spawn N agents in parallel (one per viable strategy)           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────┐       │
│  │ algo_1  │ │ algo_2  │ │ algo_3  │ │ algo_4  │ │ ... │       │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └──┬──┘       │
│       └───────────┴───────────┴───────────┴─────────┘           │
│                               │                                  │
│                    [Evaluate all in parallel]                   │
│                    [Extract innovations from each]               │
│                    [Select top 4 with diversity]                │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│               GENERATION 2+: Crossover + Mutation                │
│                                                                  │
│  CROSSOVER (N/2 agents):           MUTATION (N/2 agents):       │
│  ┌──────────────────┐              ┌──────────────────┐         │
│  │ parent1 × parent2│              │ tweak(best)      │         │
│  │ parent1 × parent3│              │ specialize(best) │         │
│  │ parent2 × parent4│              │ vectorize(best)  │         │
│  │       ...        │              │       ...        │         │
│  └────────┬─────────┘              └────────┬─────────┘         │
│           └──────────────┬──────────────────┘                   │
│                          ▼                                       │
│              [Evaluate N offspring]                             │
│              [Extract innovations]                               │
│              [Select top 4 + elitism]                           │
│                                                                  │
│  Repeat until: plateau (3 gens) OR budget exhausted             │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Rust Evaluator                               │
│                                                                  │
│  cargo build --release → benchmark → JSON fitness score         │
│  Correctness gate: tests fail = fitness 0                       │
└─────────────────────────────────────────────────────────────────┘
```

## Example Results

| Problem | What Evolved | Improvement |
|---------|--------------|-------------|
| **Fibonacci** | Lookup table + unsafe access | **834M ops/sec** (30x vs iterative) |
| **Sorting** | Radix + insertion hybrid | **71x** vs bubble sort |
| **Integer parsing** | SWAR + lookup | +51% vs std |
| **String search** | Rare-byte + memchr | +27% vs Boyer-Moore |

### Fibonacci Evolution (2 generations)

```
Baselines:
  naive (recursive):  4.6M ops/sec
  iterative:         27.4M ops/sec
  matrix exp:        94.8M ops/sec
  lookup table:     810.4M ops/sec

Gen 1: 10 agents → discovered lookup, matrix, fast-doubling
Gen 2: crossover → unsafe lookup with get_unchecked

Champion: 834.5M ops/sec (+3% vs lookup baseline, +18,142% vs naive)
Innovation: bounds-free array access via unsafe
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

## Why Rust Benchmarks?

- **No JIT warmup**: Consistent timing from first run
- **No GC pauses**: Predictable performance
- **Native speed**: Measure algorithmic improvements, not runtime overhead
- **LTO + codegen-units=1**: Maximum optimization for fair comparison

## Project Structure

```
.evolve/<problem>/           # Created per evolution
├── rust/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs           # Trait definition
│       ├── baselines.rs     # Known algorithms to beat
│       ├── evolved.rs       # Champion code
│       └── benchmark.rs     # Performance measurement
├── evaluator.py             # Fitness evaluation
├── evolution.json           # Checkpoint for resume
└── mutations/               # All tested mutations
```

## Credits

- Inspired by [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve/) concepts
- Built with [Claude Code](https://claude.ai/code) by Anthropic

## License

MIT
