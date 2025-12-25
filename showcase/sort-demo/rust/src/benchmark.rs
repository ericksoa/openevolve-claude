use sort_demo::{Sorter, baselines::*, evolved::EvolvedSorter};
use serde::Serialize;
use std::time::{Duration, Instant};
use rand::Rng;

#[derive(Serialize)]
struct BenchmarkResult {
    name: String,
    ops_per_second: f64,
    correct: bool,
}

#[derive(Serialize)]
struct FullResults {
    results: Vec<BenchmarkResult>,
    correctness: bool,
}

fn generate_test_data() -> Vec<Vec<i32>> {
    let mut rng = rand::thread_rng();
    let mut tests = Vec::new();

    // Various sizes
    for size in [100, 500, 1000, 2000] {
        // Random data
        for _ in 0..5 {
            let data: Vec<i32> = (0..size).map(|_| rng.gen_range(-10000..10000)).collect();
            tests.push(data);
        }

        // Nearly sorted (90% sorted)
        for _ in 0..2 {
            let mut data: Vec<i32> = (0..size as i32).collect();
            let swaps = size / 10;
            for _ in 0..swaps {
                let i = rng.gen_range(0..size);
                let j = rng.gen_range(0..size);
                data.swap(i, j);
            }
            tests.push(data);
        }

        // Reverse sorted
        let data: Vec<i32> = (0..size as i32).rev().collect();
        tests.push(data);
    }

    tests
}

fn is_sorted(data: &[i32]) -> bool {
    data.windows(2).all(|w| w[0] <= w[1])
}

fn verify_correctness<S: Sorter>(sorter: &S, tests: &[Vec<i32>]) -> bool {
    for test in tests {
        let mut data = test.clone();
        sorter.sort(&mut data);
        if !is_sorted(&data) {
            return false;
        }
    }
    true
}

fn benchmark<S: Sorter>(sorter: &S, tests: &[Vec<i32>], warmup_ms: u64, run_ms: u64) -> f64 {
    // Warmup
    let warmup_end = Instant::now() + Duration::from_millis(warmup_ms);
    while Instant::now() < warmup_end {
        for test in tests.iter().take(3) {
            let mut data = test.clone();
            std::hint::black_box(sorter.sort(&mut data));
        }
    }

    // Benchmark
    let mut ops = 0u64;
    let start = Instant::now();
    let end = start + Duration::from_millis(run_ms);
    while Instant::now() < end {
        for test in tests {
            let mut data = test.clone();
            std::hint::black_box(sorter.sort(&mut data));
            ops += 1;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    ops as f64 / elapsed
}

fn main() {
    let tests = generate_test_data();
    let mut results = Vec::new();
    let mut all_correct = true;

    // Benchmark bubble sort (the bad one)
    let bubble = BubbleSorter;
    let correct = verify_correctness(&bubble, &tests);
    all_correct &= correct;
    let ops = benchmark(&bubble, &tests, 50, 300);
    results.push(BenchmarkResult { name: "bubble".into(), ops_per_second: ops, correct });

    // Benchmark std sort
    let std = StdSorter;
    let correct = verify_correctness(&std, &tests);
    all_correct &= correct;
    let ops = benchmark(&std, &tests, 50, 300);
    results.push(BenchmarkResult { name: "std".into(), ops_per_second: ops, correct });

    // Benchmark std unstable
    let std_unstable = StdUnstableSorter;
    let correct = verify_correctness(&std_unstable, &tests);
    all_correct &= correct;
    let ops = benchmark(&std_unstable, &tests, 50, 300);
    results.push(BenchmarkResult { name: "std_unstable".into(), ops_per_second: ops, correct });

    // Benchmark evolved
    let evolved = EvolvedSorter;
    let correct = verify_correctness(&evolved, &tests);
    all_correct &= correct;
    let ops = benchmark(&evolved, &tests, 50, 300);
    results.push(BenchmarkResult { name: "evolved".into(), ops_per_second: ops, correct });

    let full = FullResults { results, correctness: all_correct };
    println!("{}", serde_json::to_string(&full).unwrap());
}
