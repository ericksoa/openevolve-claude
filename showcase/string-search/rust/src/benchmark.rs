//! Benchmark runner for string search algorithms
//!
//! Outputs JSON results for the evaluator

use std::time::{Duration, Instant};
use std::io::{self, Read};
use serde::{Deserialize, Serialize};
use string_search::{StringSearch, verify_search};
use string_search::evolved::EvolvedSearch;
use string_search::baselines::*;

#[derive(Deserialize)]
struct BenchmarkInput {
    texts: Vec<String>,
    patterns: Vec<String>,
    warmup_iterations: usize,
    measured_iterations: usize,
}

#[derive(Serialize)]
struct BenchmarkResult {
    algorithm: String,
    total_searches: usize,
    total_time_ns: u128,
    searches_per_second: f64,
    all_correct: bool,
    median_time_ns: u128,
}

#[derive(Serialize)]
struct BenchmarkOutput {
    results: Vec<BenchmarkResult>,
    evolved_vs_best_baseline: f64,  // >1 means evolved is faster
    score: f64,  // Main fitness score
}

fn benchmark_algorithm<S: StringSearch>(
    name: &str,
    searcher: &S,
    texts: &[Vec<u8>],
    patterns: &[Vec<u8>],
    warmup: usize,
    measured: usize,
) -> BenchmarkResult {
    let mut all_correct = true;
    let mut times: Vec<Duration> = Vec::new();

    // Warmup
    for _ in 0..warmup {
        for text in texts {
            for pattern in patterns {
                let _ = searcher.search(text, pattern);
            }
        }
    }

    // Measured runs
    for _ in 0..measured {
        let start = Instant::now();
        for text in texts {
            for pattern in patterns {
                let results = searcher.search(text, pattern);
                if !verify_search(text, pattern, &results) {
                    all_correct = false;
                }
            }
        }
        times.push(start.elapsed());
    }

    // Calculate statistics
    times.sort();
    let median_time = times[times.len() / 2];
    let total_time: Duration = times.iter().sum();
    let total_searches = texts.len() * patterns.len() * measured;

    let searches_per_second = if total_time.as_nanos() > 0 {
        (total_searches as f64) / total_time.as_secs_f64()
    } else {
        f64::INFINITY
    };

    BenchmarkResult {
        algorithm: name.to_string(),
        total_searches,
        total_time_ns: total_time.as_nanos(),
        searches_per_second,
        all_correct,
        median_time_ns: median_time.as_nanos(),
    }
}

fn main() {
    // Read benchmark configuration from stdin
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).expect("Failed to read input");

    let config: BenchmarkInput = serde_json::from_str(&input)
        .expect("Failed to parse benchmark config");

    // Convert to bytes
    let texts: Vec<Vec<u8>> = config.texts.iter().map(|s| s.as_bytes().to_vec()).collect();
    let patterns: Vec<Vec<u8>> = config.patterns.iter().map(|s| s.as_bytes().to_vec()).collect();

    let warmup = config.warmup_iterations;
    let measured = config.measured_iterations;

    // Benchmark all algorithms
    let mut results = Vec::new();

    // Baselines
    results.push(benchmark_algorithm(
        "naive", &NaiveSearch::new(), &texts, &patterns, warmup, measured
    ));
    results.push(benchmark_algorithm(
        "kmp", &KMPSearch::new(), &texts, &patterns, warmup, measured
    ));
    results.push(benchmark_algorithm(
        "boyer_moore", &BoyerMooreSearch::new(), &texts, &patterns, warmup, measured
    ));
    results.push(benchmark_algorithm(
        "horspool", &HorspoolSearch::new(), &texts, &patterns, warmup, measured
    ));
    // Note: TwoWaySearch omitted due to edge case bugs in implementation

    // Evolved algorithm
    let evolved_result = benchmark_algorithm(
        "evolved", &EvolvedSearch::new(), &texts, &patterns, warmup, measured
    );

    // Find best baseline
    let best_baseline = results.iter()
        .filter(|r| r.all_correct)
        .map(|r| r.searches_per_second)
        .fold(0.0f64, |a, b| a.max(b));

    // Calculate relative performance
    let evolved_vs_best = if best_baseline > 0.0 {
        evolved_result.searches_per_second / best_baseline
    } else {
        1.0
    };

    // Calculate fitness score
    // - Correctness is a hard requirement (0 if any test fails)
    // - Performance is the main metric
    // - Bonus for beating baselines
    let score = if evolved_result.all_correct {
        // Base score from raw performance (log scale for stability)
        let perf_score = (evolved_result.searches_per_second.ln() / 20.0).min(1.0).max(0.0);

        // Bonus for beating baselines
        let baseline_bonus = if evolved_vs_best > 1.0 {
            (evolved_vs_best - 1.0) * 0.5  // 50% bonus per 100% improvement
        } else {
            0.0
        };

        perf_score + baseline_bonus
    } else {
        0.0  // Failed correctness
    };

    results.push(evolved_result);

    let output = BenchmarkOutput {
        results,
        evolved_vs_best_baseline: evolved_vs_best,
        score,
    };

    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}
