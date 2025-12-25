# Evolve

Evolutionary algorithm discovery for Claude Code. Evolves novel solutions to hard programming problems through LLM-driven mutation with Rust benchmarks for precise performance measurement.

## How It Works

The `/evolve` skill implements an evolutionary algorithm natively in Claude Code:

1. **Generate Mutations**: Claude spawns 8 parallel Task agents, each applying a different mutation strategy (tweak, unroll, specialize, vectorize, memoize, restructure, hybrid, alien)
2. **Evaluate**: Each mutation is compiled and benchmarked against baselines using Rust
3. **Select**: Best performers become parents for the next generation
4. **Repeat**: 3-5 generations of evolution

Claude Code itself acts as the LLM ensemble—no external dependencies required.

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
> /evolve sorting algorithm for integers
> /evolve string search - beat Boyer-Moore
> /evolve integer parsing - beat std library
```

That's it! The skill will:
- Check for Rust toolchain and offer to install via rustup
- Search the web for relevant benchmarks
- Generate Rust benchmark infrastructure
- Run 3-5 generations of evolution
- Report the champion algorithm

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Claude Code /evolve                          │
│  "Optimize the string search algorithm for DNA sequences"       │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Claude Code Task Agents                          │
│            (8 parallel mutation strategies)                      │
│                                                                  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐               │
│  │  tweak  │ │ unroll  │ │specialize│ │vectorize│               │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘               │
│       │           │           │           │                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐               │
│  │ memoize │ │restructure│ │ hybrid │ │  alien  │               │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘               │
│       │           │           │           │                      │
│       └───────────┴───────────┴───────────┘                      │
│                           │                                      │
│                     Collect Results                              │
│                    Select Top Performers                         │
│                     Repeat 3-5 Generations                       │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Rust Evaluator                               │
│                                                                  │
│  1. Copy evolved code → rust/src/evolved.rs                     │
│  2. cargo build --release (LTO, opt-level=3)                    │
│  3. Run benchmarks against baselines                            │
│  4. Return JSON: { fitness, ops_per_second, correctness }       │
└─────────────────────────────────────────────────────────────────┘
```

## Demo: Bubble Sort → 71x Faster

Watch `/evolve` transform a naive O(n²) bubble sort into a fast O(n) radix sort:

```
> /evolve sorting algorithm for integers

Starting evolution...

Baseline:
  bubble:       1,289 ops/sec  ← The slow algorithm
  std:        114,592 ops/sec
  std_unstable: 168,417 ops/sec

Generation 1: Spawning 8 mutations...
  ✓ quicksort:   89,234 ops/sec
  ✓ radix:      156,892 ops/sec  ← Winner!
  ✓ heapsort:    78,456 ops/sec
  ...

Champion: Radix sort with sign-bit handling
  91,835 ops/sec → 71x faster than bubble sort!
```

The evolved algorithm:
- Uses 11-bit radix sort for large arrays
- Falls back to insertion sort for small arrays (≤64)
- Handles signed integers via sign-bit flipping

See [`showcase/sort-demo/`](showcase/sort-demo/) for the full benchmark.

## Results

| Problem | Champion | Improvement |
|---------|----------|-------------|
| **Sorting** | Radix sort | **71x** faster than bubble sort |
| Integer parsing | Custom parser | +51% vs std |
| String search | Rarebyte+memchr | +27% vs Boyer-Moore (scalar) |

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
└── mutations/               # All tested mutations
```

## Credits

- Inspired by [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve/) concepts
- Built with [Claude Code](https://claude.ai/code) by Anthropic

## License

MIT
