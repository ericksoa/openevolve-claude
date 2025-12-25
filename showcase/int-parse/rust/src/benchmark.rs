//! Benchmark for integer parsing implementations

use int_parse::baselines::*;
use int_parse::evolved::EvolvedParser;
use int_parse::IntParse;
use rand::Rng;
use serde::Serialize;
use std::time::{Duration, Instant};

#[derive(Serialize)]
struct BenchmarkResult {
    algorithm: String,
    parses_per_second: f64,
    total_parses: u64,
    duration_ms: f64,
}

#[derive(Serialize)]
struct FullResults {
    results: Vec<BenchmarkResult>,
    correctness: bool,
}

fn generate_test_data() -> Vec<Vec<u8>> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(100_000);

    // Mix of different number lengths for realistic workload
    for _ in 0..20_000 {
        // Single digit (common case)
        data.push(format!("{}", rng.gen_range(0u64..10)).into_bytes());
    }
    for _ in 0..20_000 {
        // 2-3 digits
        data.push(format!("{}", rng.gen_range(10u64..1000)).into_bytes());
    }
    for _ in 0..20_000 {
        // 4-6 digits
        data.push(format!("{}", rng.gen_range(1000u64..1_000_000)).into_bytes());
    }
    for _ in 0..20_000 {
        // 7-10 digits
        data.push(format!("{}", rng.gen_range(1_000_000u64..10_000_000_000)).into_bytes());
    }
    for _ in 0..15_000 {
        // 11-15 digits
        data.push(format!("{}", rng.gen_range(10_000_000_000u64..1_000_000_000_000_000)).into_bytes());
    }
    for _ in 0..5_000 {
        // Large numbers near max
        data.push(format!("{}", rng.gen_range(1_000_000_000_000_000u64..u64::MAX)).into_bytes());
    }

    data
}

fn verify_correctness<P: IntParse>(parser: &P) -> bool {
    // Test cases that must pass
    let cases: Vec<(&[u8], Option<u64>)> = vec![
        (b"0", Some(0)),
        (b"1", Some(1)),
        (b"123", Some(123)),
        (b"18446744073709551615", Some(u64::MAX)),
        (b"", None),
        (b"abc", None),
        (b"12a34", None),
        (b"-1", None),
        (b" 123", None),
        (b"123 ", None),
        (b"18446744073709551616", None), // overflow
        (b"99999999999999999999", None), // overflow
        (b"007", Some(7)),
    ];

    for (input, expected) in cases {
        let result = parser.parse_u64(input);
        if result != expected {
            eprintln!(
                "FAIL: parse({:?}) = {:?}, expected {:?}",
                String::from_utf8_lossy(input),
                result,
                expected
            );
            return false;
        }
    }
    true
}

fn benchmark<P: IntParse>(name: &str, parser: &P, data: &[Vec<u8>], iterations: u32) -> BenchmarkResult {
    // Warmup
    for input in data.iter().take(1000) {
        let _ = parser.parse_u64(input);
    }

    let start = Instant::now();
    let mut total_parses = 0u64;

    for _ in 0..iterations {
        for input in data {
            let _ = std::hint::black_box(parser.parse_u64(std::hint::black_box(input)));
            total_parses += 1;
        }
    }

    let duration = start.elapsed();
    let duration_ms = duration.as_secs_f64() * 1000.0;
    let parses_per_second = total_parses as f64 / duration.as_secs_f64();

    BenchmarkResult {
        algorithm: name.to_string(),
        parses_per_second,
        total_parses,
        duration_ms,
    }
}

fn main() {
    let data = generate_test_data();
    let iterations = 20;

    // Verify correctness first
    let evolved = EvolvedParser::new();
    let correctness = verify_correctness(&evolved);

    if !correctness {
        let results = FullResults {
            results: vec![],
            correctness: false,
        };
        println!("{}", serde_json::to_string(&results).unwrap());
        return;
    }

    let mut results = Vec::new();

    // Benchmark all implementations
    results.push(benchmark("std", &StdParser, &data, iterations));
    results.push(benchmark("naive", &NaiveParser, &data, iterations));
    results.push(benchmark("unrolled", &UnrolledParser, &data, iterations));
    results.push(benchmark("evolved", &evolved, &data, iterations));

    let full_results = FullResults {
        results,
        correctness: true,
    };

    println!("{}", serde_json::to_string(&full_results).unwrap());
}
