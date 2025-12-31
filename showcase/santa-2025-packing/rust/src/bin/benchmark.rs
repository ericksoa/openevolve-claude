//! Benchmark the evolved packer
//!
//! Measures score and validates correctness for evolution fitness.

use santa_packing::calculate_score;
use santa_packing::evolved::EvolvedPacker;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let runs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    println!("Benchmarking EvolvedPacker (n=1..{}, {} runs)", max_n, runs);

    let mut best_score = f64::INFINITY;
    let mut total_time = 0.0;

    for run in 1..=runs {
        let start = Instant::now();
        let packer = EvolvedPacker::default();
        let packings = packer.pack_all(max_n);
        let elapsed = start.elapsed().as_secs_f64();
        total_time += elapsed;

        // Validate
        let mut valid = true;
        for (i, packing) in packings.iter().enumerate() {
            if packing.trees.len() != i + 1 {
                eprintln!("ERROR: n={} has {} trees (expected {})", i + 1, packing.trees.len(), i + 1);
                valid = false;
            }
            if packing.has_overlaps() {
                eprintln!("ERROR: n={} has overlapping trees!", i + 1);
                valid = false;
            }
        }

        let score = calculate_score(&packings);

        if valid && score < best_score {
            best_score = score;
        }

        println!("  Run {}: score={:.4}, time={:.2}s, valid={}", run, score, elapsed, valid);
    }

    println!("\nResults:");
    println!("  Best score: {:.4}", best_score);
    println!("  Avg time: {:.2}s", total_time / runs as f64);

    // Output for evolution fitness parsing
    println!("\n[EVOLVED_SCORE={:.6}]", best_score);
}
