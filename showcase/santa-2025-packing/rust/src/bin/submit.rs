//! Generate submission file for Kaggle using EvolvedPacker

use santa_packing::calculate_score;
use santa_packing::evolved::EvolvedPacker;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

fn main() {
    println!("Generating submission with EvolvedPacker...");

    let start = Instant::now();
    let packer = EvolvedPacker::default();
    let packings = packer.pack_all(200);
    let elapsed = start.elapsed().as_secs_f64();

    let score = calculate_score(&packings);
    println!("  Score: {:.4}", score);
    println!("  Time: {:.2}s", elapsed);

    // Validate
    let mut valid = true;
    for (i, packing) in packings.iter().enumerate() {
        if packing.trees.len() != i + 1 {
            eprintln!("  ERROR: n={} has wrong tree count!", i + 1);
            valid = false;
        }
        if packing.has_overlaps() {
            eprintln!("  ERROR: n={} has overlapping trees!", i + 1);
            valid = false;
        }
    }

    if !valid {
        eprintln!("Validation failed! Not creating submission.");
        std::process::exit(1);
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
