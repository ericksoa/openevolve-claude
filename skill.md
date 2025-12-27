# Evolve Skill

Evolutionary algorithm discovery for Claude Code. Evolves novel solutions to hard problems through parallel mutation and selection, with Rust benchmarks for precise performance measurement.

## Usage

Invoke with `/evolve` followed by what you want to optimize:

```
/evolve "Optimize the string search in src/search.rs"
/evolve "Find a faster sorting algorithm for nearly-sorted arrays"
/evolve "Evolve a more efficient pathfinding heuristic"
```

## How It Works

```
┌───────────────────────────────────────────────────────────┐
│                         /evolve                           │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│                Claude Code Task Agents                    │
│           (8 parallel mutation strategies)                │
│                                                           │
│   tweak | unroll | specialize | vectorize                 │
│   memoize | restructure | hybrid | alien                  │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│                     Rust Evaluator                        │
│   • Compile with optimizations (LTO, release mode)        │
│   • Run comprehensive benchmarks                          │
│   • Compare against baselines                             │
│   • Return fitness score                                  │
└───────────────────────────────────────────────────────────┘
```

## Requirements

The skill automatically checks for and helps install:
- **Rust toolchain** - offers to install via rustup if missing
- **Python 3.10+** - guides you to install if needed

Just run `/evolve` and follow the prompts! No external dependencies required.

## Fitness Function

```
score = correctness_gate * (performance_score + baseline_bonus)
```

Where:
- `correctness_gate`: 0 if any test fails, 1 otherwise
- `performance_score`: log(ops_per_second) normalized
- `baseline_bonus`: 50% bonus for each 100% improvement over best baseline

## Output

Evolution artifacts in `.evolve/<problem>/`:

```
.evolve/<problem>/
├── rust/
│   └── src/
│       ├── lib.rs        # Trait definition
│       ├── baselines.rs  # Algorithms to beat
│       ├── evolved.rs    # Champion code
│       └── benchmark.rs  # Performance measurement
├── evaluator.py          # Fitness evaluation
└── mutations/            # All tested variants
```

## Example Results

| Problem | Champion | Improvement |
|---------|----------|-------------|
| Integer parsing | Custom parser | +51% vs std |
| Sorting | 11-bit radix sort | +14% vs std::sort_unstable |
| String search | Rarebyte+memchr | +27% vs Boyer-Moore |

## Tips

- **Start simple**: Begin with a basic implementation, let evolution discover complexity
- **Good tests**: Ensure your test suite covers edge cases; correctness is non-negotiable
- **Diverse corpus**: Use varied benchmark inputs to avoid overfitting
- **Patience**: Significant improvements often emerge after multiple generations
