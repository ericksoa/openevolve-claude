//! Analyze score breakdown

use santa_packing::incremental::IncrementalPacker;

fn main() {
    println!("Santa 2025 - Score Analysis");
    println!("===========================\n");

    let packer = IncrementalPacker {
        search_attempts: 30,
        local_opt_iterations: 500,
    };

    let packings = packer.pack_all(200);

    println!("Per-n score breakdown (top contributors):");
    println!("{:>5} {:>10} {:>10} {:>12}", "n", "side", "side²/n", "cumulative");

    let mut scores: Vec<(usize, f64, f64)> = Vec::new();
    let mut cumulative = 0.0;

    for (i, packing) in packings.iter().enumerate() {
        let n = i + 1;
        let side = packing.side_length();
        let contribution = side * side / n as f64;
        cumulative += contribution;
        scores.push((n, side, contribution));
    }

    // Show largest contributors
    let mut sorted_scores = scores.clone();
    sorted_scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    println!("\nTop 20 score contributors:");
    for (n, side, contrib) in sorted_scores.iter().take(20) {
        println!("  n={:3}: side={:.4}, contribution={:.4}", n, side, contrib);
    }

    println!("\nScore by range:");
    let ranges = [(1, 10), (11, 25), (26, 50), (51, 100), (101, 150), (151, 200)];
    for (start, end) in ranges {
        let range_score: f64 = scores[start - 1..end]
            .iter()
            .map(|(_, _, c)| c)
            .sum();
        println!("  n={:3}-{:3}: {:.4}", start, end, range_score);
    }

    println!("\nTotal score: {:.4}", cumulative);

    // Compare side lengths to optimal estimates
    // Optimal packing density for circles is pi/(2*sqrt(3)) ≈ 0.9069
    // For our tree shape, estimate based on bounding box
    println!("\nSide length analysis (vs theoretical):");
    println!("{:>5} {:>10} {:>10} {:>10}", "n", "actual", "optimal*", "ratio");

    let tree_area = 0.7 * 1.0 * 0.5; // Rough estimate of tree area
    for n in [1, 4, 9, 16, 25, 50, 100, 200] {
        let optimal_side = (n as f64 * tree_area / 0.7).sqrt(); // 0.7 packing efficiency
        let actual = scores[n - 1].1;
        println!(
            "  {:3}: {:>10.4} {:>10.4} {:>10.2}x",
            n,
            actual,
            optimal_side,
            actual / optimal_side
        );
    }
}
