//! Generate training data for ML value function
//!
//! For each N, runs evolved multiple times and records:
//! - Partial packing states at each step
//! - Final side length achieved
//!
//! Output format: JSON lines with state vectors and targets

use santa_packing::evolved::EvolvedPacker;
use santa_packing::Packing;
use std::fs::File;
use std::io::{BufWriter, Write};

/// Convert packing state to feature vector
/// Features: [n_trees, tree1_x, tree1_y, tree1_rot, tree2_x, ...]
/// Padded to max_n * 3 + 1 features
fn state_to_features(packing: &Packing, max_n: usize) -> Vec<f32> {
    let mut features = Vec::with_capacity(max_n * 3 + 1);

    // Number of trees (normalized)
    features.push(packing.trees.len() as f32 / max_n as f32);

    // Tree positions and rotations (normalized)
    for tree in &packing.trees {
        features.push(tree.x as f32 / 10.0);  // Normalize to ~[-1, 1]
        features.push(tree.y as f32 / 10.0);
        features.push(tree.angle_deg as f32 / 360.0);  // Normalize to [0, 1]
    }

    // Pad with zeros for missing trees
    while features.len() < max_n * 3 + 1 {
        features.push(0.0);
    }

    features
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(50);
    let num_runs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let output_file = args.get(3).map(|s| s.as_str()).unwrap_or("training_data.jsonl");

    eprintln!("Generating training data: max_n={}, runs={}", max_n, num_runs);

    let file = File::create(output_file).expect("Failed to create output file");
    let mut writer = BufWriter::new(file);

    let mut total_samples = 0;

    for run in 0..num_runs {
        if run % 10 == 0 {
            eprintln!("Run {}/{}", run + 1, num_runs);
        }

        // Run evolved and collect intermediate states
        let packer = EvolvedPacker::default();
        let packings = packer.pack_all(max_n);

        // For each n, we have a final packing
        // Record states at different completion percentages
        for (n_idx, final_packing) in packings.iter().enumerate() {
            let n = n_idx + 1;
            let final_side = final_packing.side_length() as f32;

            // Record partial states (25%, 50%, 75%, 100% complete)
            for frac in [0.25, 0.5, 0.75, 1.0] {
                let num_trees = ((n as f32 * frac).ceil() as usize).min(n);
                if num_trees == 0 { continue; }

                // Create partial packing
                let mut partial = Packing::new();
                for i in 0..num_trees {
                    partial.trees.push(final_packing.trees[i].clone());
                }

                let features = state_to_features(&partial, max_n);

                // Also include target n for context
                let target_n = n as f32 / max_n as f32;

                // Write as JSON line
                writeln!(
                    writer,
                    r#"{{"features": {:?}, "target_n": {}, "final_side": {}, "n": {}, "num_placed": {}}}"#,
                    features, target_n, final_side, n, num_trees
                ).unwrap();

                total_samples += 1;
            }
        }
    }

    writer.flush().unwrap();
    eprintln!("Generated {} samples to {}", total_samples, output_file);
}
