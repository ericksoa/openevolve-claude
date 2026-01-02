# Evolution State - Gen90 Complete (Plateau Extended)

## Current Champion
- **Gen87d** (greedy backtracking wave)
- **Score: 88.03** (lower is better)
- Location: `rust/src/evolved.rs`

## Gen90 Results Summary (All REJECTED)

Generation 90 exhaustively explored multiple orthogonal mutation strategies. None improved on the champion.

| Candidate | Score | Strategy | Result |
|-----------|-------|----------|--------|
| Gen90b (phase reversal) | 89.25 | U→D→L→R instead of R→L→U→D wave order | REJECTED |
| Gen90c (boundary swap) | 88.80 | 5% probability swap positions of boundary trees | REJECTED |
| Gen90d (SA boost) | 88.50 | Increase SA iterations 28000→35000 | REJECTED |
| Gen90e (3+2 split) | 89.30 | Wave split 3+2 instead of 4+1 | REJECTED |
| Gen90f (more greedy) | 88.42 | Increase greedy passes 3→5 (inconsistent) | REJECTED |
| Gen90g (finer steps) | 88.66 | Add finer greedy step sizes | REJECTED |
| Gen90h (center-directed) | 89.02 | All boundary trees move toward center | REJECTED |

## Plateau Analysis

After Gen88-90 (3 generations without improvement), the champion is at a robust local optimum:

### What We Learned:
1. **Wave order doesn't matter much**: Reversing phase order hurt performance
2. **Multi-tree operations hurt**: Swap operations added noise without improvement
3. **More compute doesn't help**: More SA iterations or greedy passes didn't improve
4. **Split ratios are sensitive**: 4+1 is better than 3+2
5. **Finer steps don't help**: The current step sizes are adequate
6. **Edge-specific > general**: Edge-specific greedy movement beats center-directed

### Champion Key Features:
- 4+1 wave split (4 outside-in, 1 inside-out)
- R→L→U→D→diagonal phase order
- Edge-specific greedy movement (3 passes)
- Rotation fallback when translation fails

## Performance Summary
- Champion score: 88.03
- Target (leaderboard top): ~69
- Gap: ~27.6%

## Next Directions (Gen91+)

Having exhausted local modifications, consider:

1. **Completely different algorithms**:
   - Bottom-left-fill heuristics
   - Genetic algorithms for tree placement
   - Machine learning-guided placement

2. **Problem reformulation**:
   - Focus on specific N ranges (e.g., optimize only high-N)
   - Specialized strategies for different tree counts

3. **Hybrid approaches**:
   - Use current algorithm as initialization, then apply different optimizer

4. **Parameter tuning with Bayesian optimization**:
   - Systematic hyperparameter search instead of manual tuning

## File Locations
- Champion code: `rust/src/evolved.rs`
- Champion backup: `rust/src/evolved_champion.rs`
- Benchmark: `cargo build --release && ./target/release/benchmark 200 3`
