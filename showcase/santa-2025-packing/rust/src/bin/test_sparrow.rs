//! Test the Sparrow-inspired packer

use santa_packing::calculate_score;
use santa_packing::sparrow::SparrowPacker;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);

    println!("Testing SparrowPacker (n=1..{})", max_n);
    println!("Key ideas: temporary overlap tolerance, guided local search, two-phase");

    let packer = SparrowPacker::default();

    let start = Instant::now();
    let packings = packer.pack_all(max_n);
    let elapsed = start.elapsed().as_secs_f64();

    // Validate
    let mut valid = true;
    let mut invalid_ns = Vec::new();
    for (i, packing) in packings.iter().enumerate() {
        if packing.trees.len() != i + 1 {
            eprintln!(
                "ERROR: n={} has {} trees (expected {})",
                i + 1,
                packing.trees.len(),
                i + 1
            );
            valid = false;
        }
        if packing.has_overlaps() {
            invalid_ns.push(i + 1);
            valid = false;
        }
    }

    if !invalid_ns.is_empty() {
        eprintln!("ERROR: Invalid packings at n={:?}", invalid_ns);
    }

    let score = calculate_score(&packings);

    println!("\nResults:");
    println!("  Score: {:.4}", score);
    println!("  Time: {:.2}s", elapsed);
    println!("  Valid: {}", valid);

    // Print individual n scores
    println!("\nBreakdown:");
    for (i, packing) in packings.iter().enumerate() {
        let n = i + 1;
        let side = packing.side_length();
        let contrib = side * side / n as f64;
        let status = if packing.has_overlaps() { " INVALID" } else { "" };
        println!("  n={:3}: side={:.4}, contrib={:.4}{}", n, side, contrib, status);
    }
}
