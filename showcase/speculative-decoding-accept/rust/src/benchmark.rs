//! Benchmark harness for speculative decoding acceptance heuristics
//!
//! Evaluates all heuristics (baselines + evolved) on train/valid/test sets
//! and outputs results in JSON format.

use speculative_decoding::baselines::*;
use speculative_decoding::evolved::Evolved;
use speculative_decoding::{load_data, AcceptanceHeuristic, EvaluationResult, TokenVerification};
use std::collections::HashMap;
use std::time::Instant;

/// Benchmark a single heuristic and return timing + results
fn benchmark_heuristic<H: AcceptanceHeuristic + ?Sized>(
    heuristic: &H,
    data: &[TokenVerification],
) -> (EvaluationResult, f64) {
    let start = Instant::now();
    let result = heuristic.evaluate(data);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    (result, elapsed_ms)
}

/// Pretty print a result row
fn print_result_row(name: &str, result: &EvaluationResult, time_ms: f64) {
    println!(
        "  {:<25} {:>6.1}%  {:>6.1}%  {:>6.3}  {:>6.3}  {:>8.2}ms",
        name,
        result.acceptance_rate * 100.0,
        result.accuracy * 100.0,
        result.quality_score,
        result.fitness,
        time_ms
    );
}

fn main() {
    println!("=== Speculative Decoding Acceptance Heuristic Benchmark ===\n");

    // Load datasets
    let train_data = load_data("data/train.json").expect("Failed to load train data");
    let valid_data = load_data("data/valid.json").expect("Failed to load valid data");
    let test_data = load_data("data/test.json").expect("Failed to load test data");

    println!("Datasets loaded:");
    println!("  TRAIN: {} tokens", train_data.len());
    println!("  VALID: {} tokens", valid_data.len());
    println!("  TEST:  {} tokens", test_data.len());
    println!();

    // Define all heuristics to benchmark
    let baselines: Vec<(&str, Box<dyn AcceptanceHeuristic>)> = vec![
        ("standard_rejection", Box::new(StandardRejectionSampling)),
        ("always_accept", Box::new(AlwaysAccept)),
        ("conservative", Box::new(Conservative)),
        ("top_token_match", Box::new(TopTokenMatch)),
        ("entropy_aware", Box::new(EntropyAware)),
        ("position_aware", Box::new(PositionAware)),
        ("ratio_floor_0.3", Box::new(RatioWithFloor { floor: 0.3 })),
    ];

    // Results storage
    let mut all_results: HashMap<String, HashMap<String, serde_json::Value>> = HashMap::new();

    // Benchmark on each dataset
    for (dataset_name, data) in [("train", &train_data), ("valid", &valid_data), ("test", &test_data)] {
        println!("--- {} ({} tokens) ---", dataset_name.to_uppercase(), data.len());
        println!(
            "  {:<25} {:>7}  {:>7}  {:>7}  {:>7}  {:>10}",
            "Heuristic", "Accept%", "Accuracy", "Quality", "Fitness", "Time"
        );
        println!("  {}", "-".repeat(75));

        // Benchmark baselines
        for (name, heuristic) in &baselines {
            let (result, time_ms) = benchmark_heuristic(heuristic.as_ref(), data);
            print_result_row(name, &result, time_ms);

            // Store result
            let dataset_results = all_results
                .entry(name.to_string())
                .or_insert_with(HashMap::new);
            dataset_results.insert(
                format!("{}_acceptance_rate", dataset_name),
                serde_json::json!(result.acceptance_rate),
            );
            dataset_results.insert(
                format!("{}_accuracy", dataset_name),
                serde_json::json!(result.accuracy),
            );
            dataset_results.insert(
                format!("{}_fitness", dataset_name),
                serde_json::json!(result.fitness),
            );
        }

        // Benchmark evolved
        println!("  {}", "-".repeat(75));
        let evolved = Evolved;
        let (result, time_ms) = benchmark_heuristic(&evolved, data);
        print_result_row("EVOLVED", &result, time_ms);

        // Store evolved result
        let evolved_results = all_results
            .entry("evolved".to_string())
            .or_insert_with(HashMap::new);
        evolved_results.insert(
            format!("{}_acceptance_rate", dataset_name),
            serde_json::json!(result.acceptance_rate),
        );
        evolved_results.insert(
            format!("{}_accuracy", dataset_name),
            serde_json::json!(result.accuracy),
        );
        evolved_results.insert(
            format!("{}_fitness", dataset_name),
            serde_json::json!(result.fitness),
        );
        evolved_results.insert(
            format!("{}_quality_score", dataset_name),
            serde_json::json!(result.quality_score),
        );
        evolved_results.insert(
            format!("{}_false_accepts", dataset_name),
            serde_json::json!(result.false_accepts),
        );
        evolved_results.insert(
            format!("{}_false_rejects", dataset_name),
            serde_json::json!(result.false_rejects),
        );

        println!();
    }

    // Output JSON summary for the evaluator
    let evolved_results = all_results.get("evolved").unwrap();
    let baseline_results = all_results.get("standard_rejection").unwrap();

    let output = serde_json::json!({
        "evolved": {
            "train_fitness": evolved_results.get("train_fitness"),
            "valid_fitness": evolved_results.get("valid_fitness"),
            "test_fitness": evolved_results.get("test_fitness"),
            "train_acceptance_rate": evolved_results.get("train_acceptance_rate"),
            "valid_acceptance_rate": evolved_results.get("valid_acceptance_rate"),
            "train_accuracy": evolved_results.get("train_accuracy"),
            "valid_accuracy": evolved_results.get("valid_accuracy"),
            "train_quality_score": evolved_results.get("train_quality_score"),
            "valid_quality_score": evolved_results.get("valid_quality_score"),
        },
        "baseline": {
            "train_fitness": baseline_results.get("train_fitness"),
            "valid_fitness": baseline_results.get("valid_fitness"),
            "test_fitness": baseline_results.get("test_fitness"),
            "train_acceptance_rate": baseline_results.get("train_acceptance_rate"),
            "valid_acceptance_rate": baseline_results.get("valid_acceptance_rate"),
        },
        "comparison": {
            "train_improvement": evolved_results.get("train_fitness").and_then(|e| {
                baseline_results.get("train_fitness").map(|b| {
                    let e_val = e.as_f64().unwrap_or(0.0);
                    let b_val = b.as_f64().unwrap_or(0.0);
                    if b_val > 0.0 { (e_val - b_val) / b_val * 100.0 } else { 0.0 }
                })
            }),
            "valid_improvement": evolved_results.get("valid_fitness").and_then(|e| {
                baseline_results.get("valid_fitness").map(|b| {
                    let e_val = e.as_f64().unwrap_or(0.0);
                    let b_val = b.as_f64().unwrap_or(0.0);
                    if b_val > 0.0 { (e_val - b_val) / b_val * 100.0 } else { 0.0 }
                })
            }),
        }
    });

    println!("=== JSON Output ===");
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}
