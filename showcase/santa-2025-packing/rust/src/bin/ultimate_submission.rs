//! Ultimate submission generator
//!
//! Combines all optimization approaches:
//! 1. Best-of-N with default config
//! 2. Stochastic parameter variation
//! 3. Multi-strategy configs
//!
//! Picks the best result for each n.

use santa_packing::calculate_score;
use santa_packing::evolved::{EvolvedConfig, EvolvedPacker};
use santa_packing::Packing;
use rand::Rng;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

fn random_config(rng: &mut impl Rng) -> EvolvedConfig {
    EvolvedConfig {
        search_attempts: rng.gen_range(150..250),
        direction_samples: rng.gen_range(48..80),
        sa_iterations: rng.gen_range(20000..40000),
        sa_initial_temp: rng.gen_range(0.35..0.55),
        sa_cooling_rate: rng.gen_range(0.99990..0.99996),
        sa_min_temp: 0.00001,
        translation_scale: rng.gen_range(0.045..0.065),
        rotation_granularity: 45.0,
        center_pull_strength: rng.gen_range(0.06..0.12),
        sa_passes: rng.gen_range(1..4),
        early_exit_threshold: rng.gen_range(2000..3000),
        boundary_focus_prob: rng.gen_range(0.80..0.95),
        num_strategies: 6,
        density_grid_resolution: 20,
        gap_penalty_weight: rng.gen_range(0.10..0.20),
        local_density_radius: 0.5,
        fill_move_prob: rng.gen_range(0.10..0.25),
        hot_restart_interval: rng.gen_range(600..1200),
        hot_restart_temp: rng.gen_range(0.30..0.40),
        elite_pool_size: rng.gen_range(2..5),
        compression_prob: rng.gen_range(0.15..0.30),
        wave_passes: rng.gen_range(4..8),
        late_stage_threshold: 140,
        fine_angle_step: 15.0,
        swap_prob: 0.0,
    }
}

fn strategy_configs() -> Vec<EvolvedConfig> {
    vec![
        // slow_cool - best performer
        EvolvedConfig {
            sa_iterations: 25000,
            sa_cooling_rate: 0.99997,
            hot_restart_interval: 1200,
            ..EvolvedConfig::default()
        },
        // center_pull - 2nd best
        EvolvedConfig {
            center_pull_strength: 0.12,
            fill_move_prob: 0.25,
            ..EvolvedConfig::default()
        },
        // more_waves - 3rd best
        EvolvedConfig {
            wave_passes: 8,
            compression_prob: 0.30,
            ..EvolvedConfig::default()
        },
        // boundary_focus
        EvolvedConfig {
            boundary_focus_prob: 0.95,
            search_attempts: 300,
            ..EvolvedConfig::default()
        },
    ]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num_default_runs: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);
    let num_stochastic_runs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
    let output_file = args.get(3).map(|s| s.as_str()).unwrap_or("submission_ultimate.csv");

    let max_n = 200;
    let strategies = strategy_configs();

    let total_runs = num_default_runs + num_stochastic_runs + strategies.len();
    eprintln!("Ultimate submission generator");
    eprintln!("  {} default runs + {} stochastic + {} strategies = {} total",
        num_default_runs, num_stochastic_runs, strategies.len(), total_runs);

    let start = Instant::now();
    let mut rng = rand::thread_rng();
    let mut all_packings: Vec<Vec<Packing>> = Vec::new();

    // Default config runs
    eprintln!("\nRunning {} default config runs...", num_default_runs);
    for run in 0..num_default_runs {
        if run % 5 == 0 { eprintln!("  Default run {}/{}", run + 1, num_default_runs); }
        let packer = EvolvedPacker::default();
        all_packings.push(packer.pack_all(max_n));
    }

    // Stochastic runs
    eprintln!("\nRunning {} stochastic runs...", num_stochastic_runs);
    for run in 0..num_stochastic_runs {
        if run % 5 == 0 { eprintln!("  Stochastic run {}/{}", run + 1, num_stochastic_runs); }
        let config = random_config(&mut rng);
        let packer = EvolvedPacker { config };
        all_packings.push(packer.pack_all(max_n));
    }

    // Strategy runs
    eprintln!("\nRunning {} strategy configs...", strategies.len());
    for (idx, config) in strategies.iter().enumerate() {
        eprintln!("  Strategy {}/{}", idx + 1, strategies.len());
        let packer = EvolvedPacker { config: EvolvedConfig {
            search_attempts: config.search_attempts,
            direction_samples: config.direction_samples,
            sa_iterations: config.sa_iterations,
            sa_initial_temp: config.sa_initial_temp,
            sa_cooling_rate: config.sa_cooling_rate,
            sa_min_temp: config.sa_min_temp,
            translation_scale: config.translation_scale,
            rotation_granularity: config.rotation_granularity,
            center_pull_strength: config.center_pull_strength,
            sa_passes: config.sa_passes,
            early_exit_threshold: config.early_exit_threshold,
            boundary_focus_prob: config.boundary_focus_prob,
            num_strategies: config.num_strategies,
            density_grid_resolution: config.density_grid_resolution,
            gap_penalty_weight: config.gap_penalty_weight,
            local_density_radius: config.local_density_radius,
            fill_move_prob: config.fill_move_prob,
            hot_restart_interval: config.hot_restart_interval,
            hot_restart_temp: config.hot_restart_temp,
            elite_pool_size: config.elite_pool_size,
            compression_prob: config.compression_prob,
            wave_passes: config.wave_passes,
            late_stage_threshold: config.late_stage_threshold,
            fine_angle_step: config.fine_angle_step,
            swap_prob: config.swap_prob,
        }};
        all_packings.push(packer.pack_all(max_n));
    }

    // Select best for each n
    eprintln!("\nSelecting best for each n...");
    let mut best_packings: Vec<Packing> = Vec::with_capacity(max_n);
    let mut improvements = 0;

    for n_idx in 0..max_n {
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
    let best_score = calculate_score(&best_packings);
    let first_score = calculate_score(&all_packings[0]);

    eprintln!("\nResults:");
    eprintln!("  First run:  {:.4}", first_score);
    eprintln!("  Ultimate:   {:.4}", best_score);
    eprintln!("  Improvement: {:.2}%", (first_score - best_score) / first_score * 100.0);
    eprintln!("  N improved: {}/{}", improvements, max_n);
    eprintln!("  Valid: {}", valid);
    eprintln!("  Time: {:.1}s ({:.1}min)", elapsed, elapsed / 60.0);

    if !valid {
        eprintln!("ERROR: Invalid submission!");
        std::process::exit(1);
    }

    // Write CSV
    eprintln!("\nWriting submission to {}...", output_file);
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

    eprintln!("Done! Score: {:.4}", best_score);
    println!("[ULTIMATE_SCORE={:.6}]", best_score);
}
