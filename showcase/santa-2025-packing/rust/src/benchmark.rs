//! Benchmark runner for packing algorithms

use santa_packing::calculate_score;
use santa_packing::multistart::MultiStartPacker;

fn main() {
    println!("Santa 2025 - Christmas Tree Packing Benchmark");
    println!("==============================================\n");

    let max_trees = 200;

    // High quality multi-start packer
    let packer = MultiStartPacker {
        restarts: 3,
        search_attempts: 50,
        sa_iterations: 3000,
        sa_temp: 0.3,
        sa_cooling: 0.999,
    };

    println!("Running multi-start packer...");
    let packings = packer.pack_all(max_trees);

    // Validate
    let mut all_valid = true;
    for (i, packing) in packings.iter().enumerate() {
        if packing.trees.len() != i + 1 {
            eprintln!("  Warning: n={} expected {} trees, got {}", i + 1, i + 1, packing.trees.len());
            all_valid = false;
        }
        if packing.has_overlaps() {
            eprintln!("  Warning: n={} has overlapping trees!", i + 1);
            all_valid = false;
        }
    }

    if all_valid {
        println!("  All packings valid!");
    }

    // Report scores at intervals
    for n in [10, 25, 50, 100, 150, 200] {
        if n <= max_trees {
            let partial = calculate_score(&packings[..n]);
            let side = packings[n - 1].side_length();
            println!("  n={:3}: score = {:.4}, side_length = {:.4}", n, partial, side);
        }
    }

    let total_score = calculate_score(&packings);
    println!("\nTOTAL SCORE: {:.4}", total_score);
    println!("\nLeaderboard comparison:");
    println!("  Top score: ~69");
    println!("  Our score: {:.2}", total_score);
    println!("  Gap: {:.1}%", (total_score / 69.0 - 1.0) * 100.0);
}
