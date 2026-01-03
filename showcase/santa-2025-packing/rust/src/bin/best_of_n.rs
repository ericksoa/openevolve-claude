//! Run evolved N times and pick best for each n
//!
//! Simple but effective: exploit variance in stochastic algorithm

use santa_packing::calculate_score;
use santa_packing::evolved::EvolvedPacker;
use santa_packing::Packing;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(50);
    let num_runs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);

    eprintln!("Best-of-{} selection (n=1..{})", num_runs, max_n);

    let start = Instant::now();

    // Collect all runs
    let mut all_packings: Vec<Vec<Packing>> = Vec::with_capacity(num_runs);

    for run in 0..num_runs {
        if run % 5 == 0 || run == num_runs - 1 {
            eprintln!("  Run {}/{}", run + 1, num_runs);
        }
        let packer = EvolvedPacker::default();
        let packings = packer.pack_all(max_n);
        all_packings.push(packings);
    }

    // Select best for each n
    let mut best_packings: Vec<Packing> = Vec::with_capacity(max_n);
    let mut improvements = 0;

    for n_idx in 0..max_n {
        let n = n_idx + 1;

        let mut best_side = f64::INFINITY;
        let mut best_packing: Option<&Packing> = None;
        let first_side = all_packings[0][n_idx].side_length();

        for run_packings in &all_packings {
            let side = run_packings[n_idx].side_length();
            if side < best_side && !run_packings[n_idx].has_overlaps() {
                best_side = side;
                best_packing = Some(&run_packings[n_idx]);
            }
        }

        let best = best_packing.unwrap_or(&all_packings[0][n_idx]);
        best_packings.push(best.clone());

        if best_side < first_side - 0.0001 {
            improvements += 1;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let best_score = calculate_score(&best_packings);
    let first_score = calculate_score(&all_packings[0]);

    eprintln!("\nResults:");
    eprintln!("  First run score:  {:.4}", first_score);
    eprintln!("  Best-of-{} score: {:.4}", num_runs, best_score);
    eprintln!("  Improvement: {:.2}%", (first_score - best_score) / first_score * 100.0);
    eprintln!("  N improved: {}/{}", improvements, max_n);
    eprintln!("  Time: {:.1}s", elapsed);

    // Output for comparison
    println!("[BEST_OF_N_SCORE={:.6}]", best_score);
}
