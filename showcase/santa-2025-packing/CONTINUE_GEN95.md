# Continuation Prompt for Gen95

Copy everything below the line and paste after clearing context:

---

/evolve-perf Continue from Gen94 plateau - Santa 2025 Kaggle packing competition

## Context
Read these files first:
- `EVOLUTION_STATE.md` - Current champion and history
- `GEN95_PLAN.md` - Plan for this generation
- `rust/src/evolved.rs` - Champion code (Gen91b)

## Quick Summary
- **CHAMPION**: Gen91b (rotation-first optimization) scored **87.29**
- **TARGET**: ~69 (leaderboard top), **gap: 26.5%**
- Gen92-94 exhaustively tried: parameter tuning, algorithmic changes, paradigm shifts
- All failed - we're at a fundamental plateau

## Gen95 Strategy: Global Optimization
Try approaches that optimize entire configurations at once, not incrementally:

1. **Gen95e: Annealing overhaul** (try first - simplest)
   - Higher initial temp: 2.0 (vs 0.45)
   - Slower cooling: 0.99998 (vs 0.99993)
   - More iterations: 100k (vs 28k)

2. **Gen95a: Full configuration SA**
   - Start with complete n-tree configuration
   - SA on entire configuration (not just last tree)
   - Higher acceptance of "worse" moves early

3. **Gen95c: Global rotation optimization**
   - Fix champion positions
   - Optimize rotations only
   - Decouple position/rotation search

4. **Gen95d: Center-first placement**
   - Place center trees first
   - Spiral outward (reverse of current)

## Key Constraints (Don't Change)
- Keep discrete 45° angles in SA
- Keep 6 parallel placement strategies
- Keep step sizes [0.10, 0.05, 0.02, 0.01, 0.005]
- Keep 5 wave passes total

## Benchmark Command
```bash
cd rust && cargo build --release && ./target/release/benchmark 200 3
```

## Execution Instructions
1. Read the files listed above
2. Start with Gen95e (annealing overhaul) - simplest change
3. Modify `rust/src/evolved.rs` with the mutation
4. Build and benchmark
5. If score < 87.29, run 5 times to verify
6. If worse, restore champion and try next mutation
7. Update EVOLUTION_STATE.md with results

## File Locations
- Champion: `rust/src/evolved.rs`
- Mutations archive: `mutations/`
- State tracking: `EVOLUTION_STATE.md`
