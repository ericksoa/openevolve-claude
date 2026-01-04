//! Final submission generator
//!
//! Best-of-N with the proven best configuration

use santa_packing::calculate_score;
use santa_packing::evolved::EvolvedPacker;
use santa_packing::Packing;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num_runs: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let output_file = args.get(2).map(|s| s.as_str()).unwrap_or("submission.csv");

    let max_n = 200;

    eprintln!("Final submission generator (best-of-{})", num_runs);

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

    for n_idx in 0..max_n {
        let mut best_side = f64::INFINITY;
        let mut best_packing: Option<&Packing> = None;

        for run_packings in &all_packings {
            let side = run_packings[n_idx].side_length();
            if side < best_side && !run_packings[n_idx].has_overlaps() {
                best_side = side;
                best_packing = Some(&run_packings[n_idx]);
            }
        }

        best_packings.push(best_packing.unwrap_or(&all_packings[0][n_idx]).clone());
    }

    // Validate
    let mut valid = true;
    for (i, packing) in best_packings.iter().enumerate() {
        if packing.trees.len() != i + 1 {
            eprintln!("ERROR: n={} has {} trees", i + 1, packing.trees.len());
            valid = false;
        }
        if packing.has_overlaps() {
            eprintln!("ERROR: n={} has overlaps", i + 1);
            valid = false;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let score = calculate_score(&best_packings);

    eprintln!("\nScore: {:.4}", score);
    eprintln!("Valid: {}", valid);
    eprintln!("Time: {:.1}min", elapsed / 60.0);

    if !valid {
        eprintln!("ERROR: Invalid!");
        std::process::exit(1);
    }

    // Write CSV
    eprintln!("Writing to {}...", output_file);
    let file = File::create(output_file).expect("Failed to create file");
    let mut writer = BufWriter::new(file);

    writeln!(writer, "row_id,x,y,angle").unwrap();

    let mut row_id = 0;
    for packing in &best_packings {
        for tree in &packing.trees {
            writeln!(writer, "{},{},{},{}", row_id, tree.x, tree.y, tree.angle_deg).unwrap();
            row_id += 1;
        }
    }

    eprintln!("Done!");
}
