//! Fast KV-Cache Eviction Benchmark with Progress Feedback
//!
//! Optimizations:
//! - Sampled evaluation (fewer positions/layers in fast mode)
//! - Live progress updates
//! - Configurable dataset size
//! - Time estimates and ETA

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

mod generate_data;

use generate_data::generate_benchmark_patterns;
use kv_cache::{
    baselines::{
        H2O, HybridBaseline, KnormPress, PositionCorrectedH2O, PyramidKV,
        RandomEviction, SnapKVLite, StreamingLLM, TOVA,
    },
    benchmark_scorer, benchmark_scorer_fast,
    evolved::Evolved,
    EvictionScorer, ScorerResult, AttentionPattern,
};

/// Run benchmark for a single scorer across all splits
fn benchmark_scorer_full(
    scorer: &dyn EvictionScorer,
    train: &[AttentionPattern],
    valid: &[AttentionPattern],
    test: &[AttentionPattern],
    fast: bool,
) -> (ScorerResult, ScorerResult, ScorerResult) {
    if fast {
        // Sample 16 positions and 8 layers (vs full ~384 positions and 32 layers)
        let train_result = benchmark_scorer_fast(scorer, train, 16, 8);
        let valid_result = benchmark_scorer_fast(scorer, valid, 16, 8);
        let test_result = benchmark_scorer_fast(scorer, test, 16, 8);
        (train_result, valid_result, test_result)
    } else {
        let train_result = benchmark_scorer(scorer, train);
        let valid_result = benchmark_scorer(scorer, valid);
        let test_result = benchmark_scorer(scorer, test);
        (train_result, valid_result, test_result)
    }
}

/// Print a progress bar
fn print_progress(completed: usize, total: usize, elapsed: f64, current_name: &str) {
    let pct = (completed as f64 / total as f64 * 100.0) as usize;
    let bar_width = 30;
    let filled = (completed * bar_width) / total.max(1);
    let bar: String = (0..bar_width)
        .map(|i| if i < filled { '=' } else { ' ' })
        .collect();

    let eta = if completed > 0 {
        let rate = elapsed / completed as f64;
        let remaining = (total - completed) as f64 * rate;
        format!("{:.0}s", remaining)
    } else {
        "...".to_string()
    };

    eprint!("\r[{}] {}/{} ({:>3}%) | ETA: {:>5} | Testing: {:<25}",
        bar, completed, total, pct, eta, current_name);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let quick_mode = args.iter().any(|a| a == "--quick" || a == "-q");
    let evolved_only = args.iter().any(|a| a == "--evolved" || a == "-e");
    let full_eval = args.iter().any(|a| a == "--full" || a == "-f");

    eprintln!("KV-Cache Eviction Benchmark");
    eprintln!("===========================");
    if quick_mode {
        eprintln!("Dataset: QUICK (5/3/3 patterns)");
    } else {
        eprintln!("Dataset: FULL (20/10/10 patterns)");
    }
    if !full_eval {
        eprintln!("Eval: SAMPLED (16 positions, 8 layers per pattern)");
    } else {
        eprintln!("Eval: EXHAUSTIVE (all positions, all layers)");
    }
    if evolved_only {
        eprintln!("Focus: Evolved vs Hybrid only");
    }
    eprintln!();

    let start = Instant::now();

    // Generate benchmark data
    eprint!("Generating attention patterns...");
    let gen_start = Instant::now();
    let (train_full, valid_full, test_full) = generate_benchmark_patterns();

    // In quick mode, use smaller dataset
    let (train, valid, test): (Vec<_>, Vec<_>, Vec<_>) = if quick_mode {
        (
            train_full.into_iter().take(5).collect(),
            valid_full.into_iter().take(3).collect(),
            test_full.into_iter().take(3).collect(),
        )
    } else {
        (train_full, valid_full, test_full)
    };

    eprintln!(" done ({:.1}s)", gen_start.elapsed().as_secs_f64());
    eprintln!(
        "Dataset: {} train, {} valid, {} test patterns\n",
        train.len(),
        valid.len(),
        test.len()
    );

    // Create scorers
    let scorer_factories: Vec<(&str, Box<dyn Fn() -> Box<dyn EvictionScorer> + Send + Sync>)> = if evolved_only {
        vec![
            ("hybrid", Box::new(|| Box::new(HybridBaseline::new()))),
            ("evolved", Box::new(|| Box::new(Evolved))),
        ]
    } else {
        vec![
            ("streaming_llm", Box::new(|| Box::new(StreamingLLM::new(4, 64)))),
            ("h2o", Box::new(|| Box::new(H2O::new(32)))),
            ("snapkv_lite", Box::new(|| Box::new(SnapKVLite::new(32)))),
            ("knorm_press", Box::new(|| Box::new(KnormPress))),
            ("tova", Box::new(|| Box::new(TOVA::new(0.01)))),
            ("pyramid_kv", Box::new(|| Box::new(PyramidKV::new(48)))),
            ("hybrid", Box::new(|| Box::new(HybridBaseline::new()))),
            ("pos_corrected_h2o", Box::new(|| Box::new(PositionCorrectedH2O::new(32, 0.3)))),
            ("random", Box::new(|| Box::new(RandomEviction::new(42)))),
            ("evolved", Box::new(|| Box::new(Evolved))),
        ]
    };

    let total_scorers = scorer_factories.len();
    let completed = Arc::new(AtomicUsize::new(0));

    eprintln!("Running benchmarks with {} scorers...\n", total_scorers);

    // Sequential with progress
    let bench_start = Instant::now();
    let mut all_results: Vec<(ScorerResult, ScorerResult, ScorerResult, u64)> = Vec::new();

    for (name, factory) in &scorer_factories {
        print_progress(
            completed.load(Ordering::Relaxed),
            total_scorers,
            bench_start.elapsed().as_secs_f64(),
            name,
        );

        let scorer_start = Instant::now();
        let scorer = factory();
        let (train_r, valid_r, test_r) = benchmark_scorer_full(&*scorer, &train, &valid, &test, !full_eval);
        let duration = scorer_start.elapsed().as_millis() as u64;

        all_results.push((train_r, valid_r, test_r, duration));
        completed.fetch_add(1, Ordering::Relaxed);
    }

    // Final progress
    print_progress(total_scorers, total_scorers, bench_start.elapsed().as_secs_f64(), "complete");
    eprintln!("\n");

    // Sort by TRAIN performance
    let mut indexed: Vec<(usize, &ScorerResult, &ScorerResult, &ScorerResult, u64)> = all_results
        .iter()
        .enumerate()
        .map(|(i, (t, v, te, d))| (i, t, v, te, *d))
        .collect();
    indexed.sort_by(|a, b| {
        a.1.avg_error
            .partial_cmp(&b.1.avg_error)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Print results
    println!("\nResults (sorted by TRAIN avg_error, lower is better):");
    println!("------------------------------------------------------\n");

    println!(
        "{:<25} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "Scorer", "TRAIN", "VALID", "TEST", "Delta", "Time"
    );
    println!("{}", "-".repeat(75));

    let best_train = indexed[0].1.avg_error;
    let best_name = &indexed[0].1.name;

    for (_, train_r, valid_r, test_r, dur) in &indexed {
        let delta = if train_r.avg_error > best_train {
            format!("+{:.2}%", (train_r.avg_error - best_train) / best_train * 100.0)
        } else {
            "BEST".to_string()
        };

        let is_evolved = train_r.name.contains("gen") || train_r.name == "evolved";
        let marker = if is_evolved { "*" } else { " " };

        println!(
            "{}{:<24} {:>10.4} {:>10.4} {:>10.4} {:>10} {:>6}ms",
            marker, train_r.name, train_r.avg_error, valid_r.avg_error, test_r.avg_error, delta, dur
        );
    }

    println!("\n* = evolved candidate\n");

    // Find evolved result for comparison
    let evolved_idx = indexed.iter().position(|(_, t, _, _, _)| t.name.contains("gen") || t.name == "evolved");
    let hybrid_idx = indexed.iter().position(|(_, t, _, _, _)| t.name.contains("hybrid"));

    if let (Some(e_idx), Some(h_idx)) = (evolved_idx, hybrid_idx) {
        let (_, evolved_t, evolved_v, evolved_te, _) = &indexed[e_idx];
        let (_, hybrid_t, hybrid_v, hybrid_te, _) = &indexed[h_idx];

        println!("Evolution Progress:");
        println!("-------------------");
        let train_imp = (hybrid_t.avg_error - evolved_t.avg_error) / hybrid_t.avg_error * 100.0;
        let valid_imp = (hybrid_v.avg_error - evolved_v.avg_error) / hybrid_v.avg_error * 100.0;
        let test_imp = (hybrid_te.avg_error - evolved_te.avg_error) / hybrid_te.avg_error * 100.0;

        println!("  vs Hybrid Baseline:");
        println!("    TRAIN: {:>+7.2}% ({:.4} -> {:.4})", train_imp, hybrid_t.avg_error, evolved_t.avg_error);
        println!("    VALID: {:>+7.2}% ({:.4} -> {:.4})", valid_imp, hybrid_v.avg_error, evolved_v.avg_error);
        println!("    TEST:  {:>+7.2}% ({:.4} -> {:.4})", test_imp, hybrid_te.avg_error, evolved_te.avg_error);

        let rank = indexed.iter().position(|(_, t, _, _, _)| t.name == evolved_t.name).unwrap() + 1;
        println!("\n  Current Rank: {} of {} (best = {})", rank, total_scorers, best_name);
    }

    // Detailed results for top 3
    println!("\nDetailed Results (Top 3):");
    println!("-------------------------\n");

    for (i, (_, train_r, valid_r, test_r, _)) in indexed.iter().take(3).enumerate() {
        println!("{}. {}", i + 1, train_r.name);
        println!(
            "   25%: T={:.4} V={:.4} Te={:.4}",
            train_r.error_at_25, valid_r.error_at_25, test_r.error_at_25
        );
        println!(
            "   50%: T={:.4} V={:.4} Te={:.4}",
            train_r.error_at_50, valid_r.error_at_50, test_r.error_at_50
        );
        println!(
            "   75%: T={:.4} V={:.4} Te={:.4}",
            train_r.error_at_75, valid_r.error_at_75, test_r.error_at_75
        );
        println!();
    }

    // JSON output
    println!("JSON Output:");
    println!("------------");
    let evolved_result = indexed.iter().find(|(_, t, _, _, _)| t.name.contains("gen") || t.name == "evolved");
    let hybrid_result = indexed.iter().find(|(_, t, _, _, _)| t.name.contains("hybrid"));

    if let (Some((_, et, ev, ete, _)), Some((_, ht, hv, hte, _))) = (evolved_result, hybrid_result) {
        println!("{{");
        println!("  \"evolved_train\": {:.6},", et.avg_error);
        println!("  \"evolved_valid\": {:.6},", ev.avg_error);
        println!("  \"evolved_test\": {:.6},", ete.avg_error);
        println!("  \"hybrid_train\": {:.6},", ht.avg_error);
        println!("  \"hybrid_valid\": {:.6},", hv.avg_error);
        println!("  \"hybrid_test\": {:.6},", hte.avg_error);
        let train_imp = (ht.avg_error - et.avg_error) / ht.avg_error * 100.0;
        let valid_imp = (hv.avg_error - ev.avg_error) / hv.avg_error * 100.0;
        let test_imp = (hte.avg_error - ete.avg_error) / hte.avg_error * 100.0;
        println!("  \"improvement_train_pct\": {:.4},", train_imp);
        println!("  \"improvement_valid_pct\": {:.4},", valid_imp);
        println!("  \"improvement_test_pct\": {:.4}", test_imp);
        println!("}}");
    }

    let elapsed = start.elapsed();
    println!("\nBenchmark completed in {:.2}s", elapsed.as_secs_f64());

    // Usage hint
    eprintln!("\nUsage: fast_bench [OPTIONS]");
    eprintln!("  -q, --quick    Use smaller dataset (5/3/3 patterns)");
    eprintln!("  -e, --evolved  Only test Evolved vs Hybrid baseline");
    eprintln!("  -f, --full     Full exhaustive evaluation (slower, more accurate)");
}
