//! Generate high-quality submission file for Kaggle

use santa_packing::calculate_score;
use santa_packing::multistart::MultiStartPacker;
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() {
    println!("Generating high-quality submission...");

    // High quality settings
    let packer = MultiStartPacker {
        restarts: 10,
        search_attempts: 80,
        sa_iterations: 10000,
        sa_temp: 0.4,
        sa_cooling: 0.9995,
    };

    println!("  Running optimization (this will take a few minutes)...");
    let packings = packer.pack_all(200);

    let score = calculate_score(&packings);
    println!("  Score: {:.4}", score);

    // Validate
    for (i, packing) in packings.iter().enumerate() {
        if packing.trees.len() != i + 1 {
            eprintln!("  ERROR: n={} has wrong tree count!", i + 1);
        }
        if packing.has_overlaps() {
            eprintln!("  ERROR: n={} has overlapping trees!", i + 1);
        }
    }

    let mut writer = BufWriter::new(
        File::create("submission.csv").expect("Failed to create submission.csv"),
    );

    writeln!(writer, "id,x,y,deg").unwrap();

    for (n, packing) in packings.iter().enumerate() {
        let n = n + 1;
        for (i, tree) in packing.trees.iter().enumerate() {
            writeln!(
                writer,
                "{:03}_{},s{:.6},s{:.6},s{:.6}",
                n, i, tree.x, tree.y, tree.angle_deg
            )
            .unwrap();
        }
    }

    println!("Done! Created submission.csv with score {:.4}", score);
}
