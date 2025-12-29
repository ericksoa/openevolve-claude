//! Real benchmark for quantization bit allocation using GPT-2
//!
//! Evaluates heuristics using actual quantization and perplexity measurement.
//! Requires Python with transformers and torch installed.

use quantization_bit_alloc::{
    baselines::*,
    eval_bridge::{evaluate, calculate_fitness, verify_consistency, AllocationPlan, EvalMode, EvalResult, GPT2_LAYER_PATTERNS},
};
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::time::Instant;

/// Result of evaluating a strategy
#[derive(Debug, Clone)]
struct StrategyResult {
    name: String,
    fast_result: EvalResult,
    verify_result: Option<EvalResult>,
    fitness: f64,
    is_valid: bool,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("baselines");

    // Setup paths
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let python_script = manifest_dir.join("../python/eval_gpt2.py");
    let work_dir = manifest_dir.clone();

    // Check if Python evaluator exists
    if !python_script.exists() {
        eprintln!("Error: Python evaluator not found at {:?}", python_script);
        eprintln!("Make sure python/eval_gpt2.py exists");
        std::process::exit(1);
    }

    // Check if corpus files exist
    let fast_corpus = manifest_dir.join("data/eval_fast.txt");
    let verify_corpus = manifest_dir.join("data/eval_verify.txt");
    if !fast_corpus.exists() || !verify_corpus.exists() {
        eprintln!("Error: Corpus files not found in data/");
        eprintln!("Expected: data/eval_fast.txt and data/eval_verify.txt");
        std::process::exit(1);
    }

    eprintln!("\n=== GPT-2 Real Quantization Benchmark ===\n");

    match mode {
        "baselines" => run_baselines(&python_script, &work_dir),
        "evolved" => run_evolved(&python_script, &work_dir),
        "full" => {
            run_baselines(&python_script, &work_dir);
            eprintln!("\n--- Evolved Strategy ---\n");
            run_evolved(&python_script, &work_dir);
        }
        _ => {
            eprintln!("Usage: benchmark_real [baselines|evolved|full]");
            std::process::exit(1);
        }
    }
}

fn run_baselines(python_script: &PathBuf, work_dir: &PathBuf) {
    let baselines = all_gpt2_baselines();

    eprintln!("Evaluating {} baseline strategies...\n", baselines.len());
    eprintln!("{:<25} {:>10} {:>12} {:>10} {:>10}",
        "Strategy", "Perplexity", "Size (MB)", "Compress", "Fitness");
    eprintln!("{}", "-".repeat(70));

    // First, get FP16 baseline for fitness calculation
    let fp16_plan = gpt2_all_fp16();
    eprintln!("Computing FP16 baseline...");
    let fp16_result = evaluate(&fp16_plan, EvalMode::Fast, python_script, work_dir)
        .expect("Failed to evaluate FP16 baseline");
    let baseline_perplexity = fp16_result.perplexity;
    let baseline_size = fp16_result.model_size_bytes;
    eprintln!("FP16 baseline: perplexity={:.2}, size={:.1}MB\n",
        baseline_perplexity, baseline_size as f64 / 1_000_000.0);

    let mut results: Vec<StrategyResult> = Vec::new();

    for (name, plan) in baselines {
        let start = Instant::now();

        match evaluate(&plan, EvalMode::Fast, python_script, work_dir) {
            Ok(result) => {
                let elapsed = start.elapsed();
                let fitness = calculate_fitness(&result, baseline_perplexity, baseline_size);
                let compression = baseline_size as f64 / result.model_size_bytes as f64;
                let size_mb = result.model_size_bytes as f64 / 1_000_000.0;

                eprintln!("{:<25} {:>10.2} {:>10.1}MB {:>9.2}x {:>10.4}  ({:.1}s)",
                    name, result.perplexity, size_mb, compression, fitness, elapsed.as_secs_f64());

                results.push(StrategyResult {
                    name: name.to_string(),
                    fast_result: result,
                    verify_result: None,
                    fitness,
                    is_valid: true,
                });
            }
            Err(e) => {
                eprintln!("{:<25} ERROR: {}", name, e);
            }
        }
    }

    // Sort by fitness and show summary
    results.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));

    eprintln!("\n=== Top Strategies ===");
    for (i, r) in results.iter().take(5).enumerate() {
        eprintln!("{}. {} (fitness: {:.4}, perplexity: {:.2})",
            i + 1, r.name, r.fitness, r.fast_result.perplexity);
    }

    // Verify top strategy
    if let Some(best) = results.first() {
        eprintln!("\n=== Verifying Best Strategy: {} ===", best.name);

        // Re-create the plan (we don't store it in the result)
        let plan = match best.name.as_str() {
            "all_fp32" => gpt2_all_fp32(),
            "all_fp16" => gpt2_all_fp16(),
            "all_int8" => gpt2_all_int8(),
            "ln_fp32_rest_int8" => gpt2_layernorm_fp32_rest_int8(),
            "emb_ln_fp32_rest_int8" => gpt2_emb_ln_fp32_rest_int8(),
            "all_int4" => gpt2_all_int4(),
            "ln_fp32_rest_int4" => gpt2_layernorm_fp32_rest_int4(),
            "edges_fp16_middle_int8" => gpt2_edges_fp16_middle_int8(),
            _ => return,
        };

        match evaluate(&plan, EvalMode::Verify, python_script, work_dir) {
            Ok(verify_result) => {
                let is_consistent = verify_consistency(&best.fast_result, &verify_result, 0.02);
                let status = if is_consistent { "PASS" } else { "FAIL" };

                eprintln!("FAST perplexity:   {:.4}", best.fast_result.perplexity);
                eprintln!("VERIFY perplexity: {:.4}", verify_result.perplexity);
                eprintln!("Consistency check: {} (tolerance 2%)", status);
            }
            Err(e) => {
                eprintln!("VERIFY failed: {}", e);
            }
        }
    }
}

fn run_evolved(python_script: &PathBuf, work_dir: &PathBuf) {
    // Get FP16 baseline
    let fp16_plan = gpt2_all_fp16();
    let fp16_result = evaluate(&fp16_plan, EvalMode::Fast, python_script, work_dir)
        .expect("Failed to evaluate FP16 baseline");
    let baseline_perplexity = fp16_result.perplexity;
    let baseline_size = fp16_result.model_size_bytes;

    // Create evolved strategy plan
    // This matches the wider_fp16_zone champion from synthetic evolution
    let evolved_plan = create_evolved_plan();

    eprintln!("Evaluating evolved strategy...\n");

    // FAST evaluation
    let start = Instant::now();
    let fast_result = evaluate(&evolved_plan, EvalMode::Fast, python_script, work_dir)
        .expect("Failed to evaluate evolved strategy");
    let fast_time = start.elapsed();

    let fitness = calculate_fitness(&fast_result, baseline_perplexity, baseline_size);
    let compression = baseline_size as f64 / fast_result.model_size_bytes as f64;

    eprintln!("=== FAST Evaluation ===");
    eprintln!("Perplexity:  {:.4}", fast_result.perplexity);
    eprintln!("Model size:  {:.1} MB", fast_result.model_size_bytes as f64 / 1_000_000.0);
    eprintln!("Compression: {:.2}x", compression);
    eprintln!("Fitness:     {:.4}", fitness);
    eprintln!("Eval time:   {:.1}s", fast_time.as_secs_f64());

    // Print bit histogram
    eprintln!("\nBit allocation histogram:");
    let total: u64 = fast_result.bit_histogram.values().sum();
    for (bw, count) in &fast_result.bit_histogram {
        let pct = (*count as f64 / total as f64) * 100.0;
        eprintln!("  {}: {} params ({:.1}%)", bw, count, pct);
    }

    // VERIFY evaluation
    eprintln!("\n=== VERIFY Evaluation ===");
    let start = Instant::now();
    let verify_result = evaluate(&evolved_plan, EvalMode::Verify, python_script, work_dir)
        .expect("Failed to verify evolved strategy");
    let verify_time = start.elapsed();

    eprintln!("Perplexity:  {:.4}", verify_result.perplexity);
    eprintln!("Eval time:   {:.1}s", verify_time.as_secs_f64());

    // Consistency check
    let is_consistent = verify_consistency(&fast_result, &verify_result, 0.02);
    let ppl_diff = ((verify_result.perplexity - fast_result.perplexity) / fast_result.perplexity * 100.0).abs();

    eprintln!("\n=== Consistency Check ===");
    eprintln!("FAST perplexity:   {:.4}", fast_result.perplexity);
    eprintln!("VERIFY perplexity: {:.4}", verify_result.perplexity);
    eprintln!("Difference:        {:.2}%", ppl_diff);
    eprintln!("Status:            {}", if is_consistent { "PASS" } else { "FAIL (>2% difference)" });

    // Final verdict
    eprintln!("\n=== Final Result ===");
    if is_consistent {
        eprintln!("Evolved strategy is VALID");
        eprintln!("Fitness: {:.4}", fitness);
    } else {
        eprintln!("Evolved strategy is INVALID (VERIFY failed)");
    }
}

/// Create the evolved allocation plan
/// Based on wider_fp16_zone champion from synthetic evolution
fn create_evolved_plan() -> AllocationPlan {
    let total = GPT2_LAYER_PATTERNS.len();
    let allocations: Vec<_> = GPT2_LAYER_PATTERNS
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let pos = i as f64 / (total - 1) as f64;

            // Layer norm always FP32
            if p.contains("ln_") || p == "ln_f" {
                return (p.to_string(), "fp32".to_string());
            }

            // Edge protection (first/last 7.5%)
            if pos < 0.075 || pos > 0.925 {
                return (p.to_string(), "fp32".to_string());
            }

            // Middle layers: use FP16 (we don't have sensitivity info in real mode)
            // In real mode, we can't check sensitivity, so use FP16 as default
            // Evolution in real mode will discover better policies
            (p.to_string(), "fp16".to_string())
        })
        .collect();

    AllocationPlan::from_iter(allocations)
}

/// Print the allocation plan for debugging
#[allow(dead_code)]
fn print_plan(plan: &AllocationPlan) {
    eprintln!("\nAllocation plan:");
    for (layer, bw) in &plan.allocations {
        eprintln!("  {}: {}", layer, bw);
    }
}
