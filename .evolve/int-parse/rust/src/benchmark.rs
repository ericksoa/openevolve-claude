//! Benchmark binary for integer parsing

use int_parse::baselines::{NaiveParser, StdParser, SwarParser, UnrolledParser};
use int_parse::evolved::EvolvedParser;
use int_parse::IntParser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::time::{Duration, Instant};

#[derive(Serialize)]
struct BenchmarkResult {
    name: String,
    ops_per_second: f64,
    total_ns: u64,
    iterations: u64,
}

#[derive(Serialize)]
struct FullResults {
    results: Vec<BenchmarkResult>,
    correctness: bool,
}

fn generate_test_data(rng: &mut StdRng) -> Vec<Vec<u8>> {
    let mut data = Vec::with_capacity(10000);

    // Small positive numbers (very common case)
    for _ in 0..2000 {
        let n: i64 = rng.gen_range(0..1000);
        data.push(n.to_string().into_bytes());
    }

    // Medium positive numbers
    for _ in 0..2000 {
        let n: i64 = rng.gen_range(1000..1000000);
        data.push(n.to_string().into_bytes());
    }

    // Large positive numbers
    for _ in 0..1500 {
        let n: i64 = rng.gen_range(1000000..1000000000000i64);
        data.push(n.to_string().into_bytes());
    }

    // Negative numbers
    for _ in 0..1500 {
        let n: i64 = rng.gen_range(-1000000000..0);
        data.push(n.to_string().into_bytes());
    }

    // Edge cases
    data.push(b"0".to_vec());
    data.push(b"-1".to_vec());
    data.push(b"9223372036854775807".to_vec()); // i64::MAX
    data.push(b"-9223372036854775808".to_vec()); // i64::MIN

    // Very large numbers
    for _ in 0..500 {
        let n: i64 = rng.gen_range(1000000000000i64..i64::MAX / 10);
        data.push(n.to_string().into_bytes());
    }

    // Numbers with leading zeros (less common)
    for _ in 0..100 {
        let n: i64 = rng.gen_range(1..1000);
        data.push(format!("{:06}", n).into_bytes());
    }

    data
}

fn verify_correctness<P: IntParser>(parser: &P, test_cases: &[(Vec<u8>, i64)]) -> bool {
    for (input, expected) in test_cases {
        match parser.parse(input) {
            Ok(result) if result == *expected => {}
            _ => return false,
        }
    }
    true
}

fn benchmark_parser<P: IntParser>(
    parser: &P,
    name: &str,
    data: &[Vec<u8>],
    warmup_iters: u64,
    bench_iters: u64,
) -> BenchmarkResult {
    // Warmup
    for _ in 0..warmup_iters {
        for input in data {
            std::hint::black_box(parser.parse(std::hint::black_box(input)));
        }
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..bench_iters {
        for input in data {
            std::hint::black_box(parser.parse(std::hint::black_box(input)));
        }
    }
    let elapsed = start.elapsed();

    let total_ops = bench_iters * data.len() as u64;
    let ops_per_second = total_ops as f64 / elapsed.as_secs_f64();

    BenchmarkResult {
        name: name.to_string(),
        ops_per_second,
        total_ns: elapsed.as_nanos() as u64,
        iterations: total_ops,
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(12345);
    let data = generate_test_data(&mut rng);

    // Generate correctness test cases
    let test_cases: Vec<(Vec<u8>, i64)> = vec![
        (b"0".to_vec(), 0),
        (b"1".to_vec(), 1),
        (b"123".to_vec(), 123),
        (b"-1".to_vec(), -1),
        (b"-123".to_vec(), -123),
        (b"9223372036854775807".to_vec(), i64::MAX),
        (b"-9223372036854775808".to_vec(), i64::MIN),
        (b"007".to_vec(), 7),
        (b"999999999".to_vec(), 999999999),
        (b"-999999999".to_vec(), -999999999),
    ];

    let std_parser = StdParser;
    let naive_parser = NaiveParser;
    let unrolled_parser = UnrolledParser;
    let swar_parser = SwarParser;
    let evolved_parser = EvolvedParser;

    // Verify correctness
    let all_correct = verify_correctness(&std_parser, &test_cases)
        && verify_correctness(&naive_parser, &test_cases)
        && verify_correctness(&unrolled_parser, &test_cases)
        && verify_correctness(&swar_parser, &test_cases)
        && verify_correctness(&evolved_parser, &test_cases);

    // Benchmark parameters
    let warmup_iters = 10;
    let bench_iters = 100;

    let mut results = Vec::new();

    results.push(benchmark_parser(
        &std_parser,
        "std",
        &data,
        warmup_iters,
        bench_iters,
    ));
    results.push(benchmark_parser(
        &naive_parser,
        "naive",
        &data,
        warmup_iters,
        bench_iters,
    ));
    results.push(benchmark_parser(
        &unrolled_parser,
        "unrolled",
        &data,
        warmup_iters,
        bench_iters,
    ));
    results.push(benchmark_parser(
        &swar_parser,
        "swar",
        &data,
        warmup_iters,
        bench_iters,
    ));
    results.push(benchmark_parser(
        &evolved_parser,
        "evolved",
        &data,
        warmup_iters,
        bench_iters,
    ));

    let full_results = FullResults {
        results,
        correctness: all_correct,
    };

    println!("{}", serde_json::to_string(&full_results).unwrap());
}
