# OpenEvolve Skill

Evolutionary algorithm discovery powered by OpenEvolve. Evolves novel solutions to hard problems through massively parallel mutation and selection, with Rust benchmarks for precise performance measurement.

## Usage

Invoke with `/evolve` followed by what you want to optimize:

```
/evolve "Optimize the string search in src/search.rs"
/evolve "Find a faster sorting algorithm for nearly-sorted arrays"
/evolve "Evolve a more efficient pathfinding heuristic"
```

## How It Works

This skill integrates [OpenEvolve](https://github.com/codelion/openevolve) with Claude Code:

```
┌─────────────────────────────────────────────────────────────┐
│                        /evolve                               │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     OpenEvolve                               │
│  • MAP-Elites quality-diversity evolution                   │
│  • Island-based parallel populations                        │
│  • LLM ensemble for intelligent mutations                   │
└─────────────────────────────┬───────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Island 1  │      │   Island 2  │      │   Island N  │
│             │      │             │      │             │
│ Population  │ ←──→ │ Population  │ ←──→ │ Population  │
│ of variants │      │ of variants │      │ of variants │
└──────┬──────┘      └──────┬──────┘      └──────┬──────┘
       │                    │                    │
       └────────────────────┼────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Rust Evaluator                             │
│  • Compile with optimizations (LTO, release mode)           │
│  • Run comprehensive benchmarks                              │
│  • Compare against baselines (Boyer-Moore, KMP, etc.)       │
│  • Return fitness score                                      │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

The skill automatically checks for and helps install:
- **Rust toolchain** - offers to install via rustup if missing
- **Python 3.10+** - guides you to install if needed

Just run `/evolve` and follow the prompts! No manual setup required.

## Configuration

Create `config.yaml` in your project:

```yaml
evolution:
  iterations: 200
  population_size: 50
  num_islands: 4

llm:
  primary:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.7

evaluation:
  timeout: 300
  parallel_evaluations: 4
```

## Running Evolution

```bash
# Install OpenEvolve
pip install openevolve

# Navigate to showcase
cd showcase/string-search

# Run evolution
openevolve-run initial_program.rs evaluator.py \
  --config config.yaml \
  --iterations 200
```

## Showcases

### String Search (Rust)

Evolves a string search algorithm to compete with Boyer-Moore:

```bash
cd showcase/string-search
pip install -r requirements.txt
cargo build --release  # Build Rust benchmarks
openevolve-run initial_program.rs evaluator.py --config config.yaml
```

The evaluator:
1. Copies evolved code to `rust/src/evolved.rs`
2. Compiles with `cargo build --release`
3. Runs benchmarks against naive, KMP, Boyer-Moore, Horspool, Two-Way
4. Returns fitness based on correctness + performance

## Fitness Function

Default fitness combines:

```
score = correctness_gate * (
    performance_score +
    baseline_bonus
)
```

Where:
- `correctness_gate`: 0 if any test fails, 1 otherwise
- `performance_score`: log(searches_per_second) normalized
- `baseline_bonus`: 50% bonus for each 100% improvement over best baseline

## Output

Evolution artifacts in `.evolve/`:

```
.evolve/
├── population.json      # All candidates with scores
├── best_program.rs      # Champion code
├── evolution_log.json   # Iteration history
└── pareto_front/        # Non-dominated solutions
```

## Example Output

```
╔════════════════════════════════════════════════════════════╗
║         OpenEvolve String Search Evolution                 ║
╚════════════════════════════════════════════════════════════╝

Baseline Performance:
  naive:       12,450 searches/sec
  kmp:         45,230 searches/sec
  boyer_moore: 78,900 searches/sec  (best baseline)
  horspool:    71,200 searches/sec
  two_way:     68,500 searches/sec

Evolution Progress:
  Iteration  10: score=0.42, 85,200/sec (+8% vs BM)
  Iteration  25: score=0.58, 112,000/sec (+42% vs BM)
  Iteration  50: score=0.71, 156,000/sec (+98% vs BM)
  Iteration 100: score=0.79, 189,000/sec (+140% vs BM)
  Iteration 150: score=0.82, 201,000/sec (+155% vs BM)
  Converged at iteration 178

Champion: 203,500 searches/sec (+158% vs Boyer-Moore)
Strategy: Hybrid skip table + SIMD-friendly inner loop
```

## Integration with Claude Code

When you run `/evolve`, Claude will:

1. **Analyze** the target code and identify optimization opportunities
2. **Configure** OpenEvolve with appropriate settings
3. **Launch** evolution as a background process
4. **Monitor** progress and report improvements
5. **Apply** the champion solution to your codebase

## Tips

- **Start simple**: Begin with a basic implementation, let evolution discover complexity
- **Good tests**: Ensure your test suite covers edge cases; correctness is non-negotiable
- **Diverse corpus**: Use varied benchmark inputs to avoid overfitting
- **Patience**: Significant improvements often emerge after 50+ iterations
