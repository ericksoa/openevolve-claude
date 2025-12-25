# OpenEvolve-Claude

Evolutionary algorithm discovery for Claude Code, powered by [OpenEvolve](https://github.com/codelion/openevolve). Evolves novel solutions to hard programming problems through massively parallel LLM-driven mutation with Rust benchmarks for precise performance measurement.

## The Vision

Combine the best of both worlds:
- **OpenEvolve**: Battle-tested evolutionary framework with MAP-Elites, island populations, and LLM ensembles
- **Claude Code**: Intelligent agents that understand code context and can apply evolved solutions
- **Rust Benchmarks**: Nanosecond-precision performance measurement without JIT variance

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/openevolve-claude
cd openevolve-claude

# Install Python dependencies
pip install openevolve

# Build the Rust benchmarks
cd showcase/string-search/rust
cargo build --release
cd ..

# Run evolution
openevolve-run initial_program.rs evaluator.py \
  --config config.yaml \
  --iterations 200
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Claude Code /evolve                          │
│  "Optimize the string search algorithm for DNA sequences"       │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OpenEvolve                                │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Island 1   │  │   Island 2   │  │   Island 3   │          │
│  │              │  │              │  │              │          │
│  │ ┌──┐┌──┐┌──┐ │  │ ┌──┐┌──┐┌──┐ │  │ ┌──┐┌──┐┌──┐ │          │
│  │ │P1││P2││P3│ │  │ │P4││P5││P6│ │  │ │P7││P8││P9│ │          │
│  │ └──┘└──┘└──┘ │  │ └──┘└──┘└──┘ │  │ └──┘└──┘└──┘ │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └────────── Migration ──────────────┘                   │
│                           │                                      │
│  ┌────────────────────────┴────────────────────────────┐        │
│  │                  LLM Ensemble                        │        │
│  │  Claude Opus (creative) + Sonnet + Haiku (tweaks)   │        │
│  └─────────────────────────────────────────────────────┘        │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Rust Evaluator                               │
│                                                                  │
│  1. Copy evolved code → rust/src/evolved.rs                     │
│  2. cargo build --release (LTO, opt-level=3)                    │
│  3. Run benchmarks against baselines:                           │
│     • Naive O(nm)                                                │
│     • KMP O(n+m)                                                 │
│     • Boyer-Moore O(n/m) best                                   │
│     • Horspool                                                   │
│     • Two-Way (glibc memmem)                                    │
│  4. Return fitness: correctness × performance × baseline_bonus  │
└─────────────────────────────────────────────────────────────────┘
```

## Showcases

### String Search (The Awe-Inspiring Demo)

Everyone knows "find needle in haystack". But can we evolve an algorithm that beats 40-year-old classics like Boyer-Moore?

```
showcase/string-search/
├── initial_program.rs    # Seed: simple Horspool variant
├── evaluator.py          # Compiles Rust, runs benchmarks
├── config.yaml           # OpenEvolve settings
└── rust/
    ├── Cargo.toml
    └── src/
        ├── lib.rs        # Core traits
        ├── baselines.rs  # KMP, Boyer-Moore, Horspool, Two-Way
        ├── evolved.rs    # THE CODE BEING EVOLVED
        └── benchmark.rs  # Performance measurement
```

**Baselines we're competing against:**

| Algorithm | Complexity | Notes |
|-----------|------------|-------|
| Naive | O(nm) | Simple, slow |
| KMP | O(n+m) | Failure function |
| Boyer-Moore | O(n/m) best | Bad char + good suffix |
| Horspool | O(n/m) best | Simplified BM |
| Two-Way | O(n) | Used by glibc |

**Run it:**

```bash
cd showcase/string-search
pip install openevolve
cd rust && cargo build --release && cd ..
openevolve-run initial_program.rs evaluator.py --config config.yaml
```

## Configuration

```yaml
# config.yaml
evolution:
  iterations: 200           # More = better solutions
  population_size: 50       # Diversity vs compute
  num_islands: 4            # Parallel populations
  migration_interval: 10    # Cross-pollination frequency

llm:
  primary:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.7        # Balanced exploration

  secondary:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.9        # Creative mutations

  fast:
    provider: anthropic
    model: claude-haiku-4-20250514
    temperature: 0.3        # Quick tweaks

evaluation:
  timeout: 300              # Max seconds per candidate
  parallel_evaluations: 4   # Concurrent benchmarks
```

## Fitness Function

```python
def calculate_fitness(evolved_result, best_baseline):
    # Correctness is non-negotiable
    if not evolved_result.all_correct:
        return 0.0

    # Base score from raw performance
    perf_score = log(searches_per_second) / 20  # normalized

    # Bonus for beating baselines
    if evolved_result.speed > best_baseline:
        improvement = evolved_result.speed / best_baseline - 1
        baseline_bonus = improvement * 0.5  # 50% per 100% improvement
    else:
        baseline_bonus = 0

    return perf_score + baseline_bonus
```

## Installation as Claude Code Skill

```bash
# Copy skill definition
cp skill.md ~/.claude/skills/evolve.md

# Now use in Claude Code
> /evolve "Optimize the sorting algorithm in src/sort.rs"
```

## How OpenEvolve Works

1. **Prompt Sampler**: Creates context-rich prompts with past programs and scores
2. **LLM Ensemble**: Multiple models generate diverse mutations
3. **Evaluator Pool**: Tests candidates and assigns fitness scores
4. **Program Database**: Stores successful programs, guides evolution
5. **MAP-Elites**: Maintains diversity across quality dimensions

Key innovations:
- **Island model**: Separate populations prevent premature convergence
- **Quality-diversity**: Keeps diverse solutions, not just the best
- **Cascade evaluation**: Fast rejection of broken code before expensive benchmarks

## Why Rust Benchmarks?

- **No JIT warmup**: Consistent timing from first run
- **No GC pauses**: Predictable performance
- **Native speed**: Measure algorithmic improvements, not runtime overhead
- **LTO + codegen-units=1**: Maximum optimization for fair comparison

## Project Structure

```
openevolve-claude/
├── skill.md                  # Claude Code skill definition
├── README.md                 # This file
├── src/                      # TypeScript utilities (optional)
│   ├── types.ts
│   ├── orchestrator.ts
│   └── ...
└── showcase/
    └── string-search/
        ├── initial_program.rs
        ├── evaluator.py
        ├── config.yaml
        ├── requirements.txt
        └── rust/
            ├── Cargo.toml
            └── src/
                ├── lib.rs
                ├── baselines.rs
                ├── evolved.rs
                └── benchmark.rs
```

## Future Showcases

- **Sorting**: Evolve a sort that adapts to input characteristics
- **Pathfinding**: Discover novel A* heuristics
- **Compression**: Evolve domain-specific encoders
- **Regex**: Optimize pattern matching for specific workloads

## Credits

- [OpenEvolve](https://github.com/codelion/openevolve) by Asankhaya Sharma
- [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve/) by DeepMind
- [Claude Code](https://claude.ai/code) by Anthropic

## License

MIT

---

**Sources:**
- [OpenEvolve GitHub](https://github.com/codelion/openevolve)
- [OpenEvolve on Hugging Face](https://huggingface.co/blog/codelion/openevolve)
- [OpenEvolve PyPI](https://pypi.org/project/openevolve/)
