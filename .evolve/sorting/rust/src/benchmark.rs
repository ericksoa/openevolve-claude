use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::Serialize;
use sorting::baselines::{HeapSort, QuickSort, RadixSort, StdSort, StdUnstable};
use sorting::evolved::EvolvedSorter;
use sorting::Sorter;
use std::time::{Duration, Instant};

#[derive(Serialize)]
struct BenchmarkResult {
    name: String,
    ops_per_second: f64,
    avg_ns: f64,
    correct: bool,
}

#[derive(Serialize)]
struct FullResults {
    results: Vec<BenchmarkResult>,
    correctness: bool,
}

const WARMUP_ITERATIONS: u32 = 10;
const BENCH_ITERATIONS: u32 = 100;
const ARRAY_SIZE: usize = 10000;

fn generate_random_data(seed: u64, size: usize) -> Vec<u64> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    (0..size).map(|_| rng.gen()).collect()
}

fn is_sorted(data: &[u64]) -> bool {
    data.windows(2).all(|w| w[0] <= w[1])
}

fn verify_sort<S: Sorter>(sorter: &S, seeds: &[u64]) -> bool {
    for &seed in seeds {
        let mut data = generate_random_data(seed, ARRAY_SIZE);
        sorter.sort(&mut data);
        if !is_sorted(&data) {
            return false;
        }
    }

    // Edge cases
    let mut empty: Vec<u64> = vec![];
    sorter.sort(&mut empty);

    let mut single = vec![42u64];
    sorter.sort(&mut single);

    let mut two = vec![5u64, 3u64];
    sorter.sort(&mut two);
    if two != vec![3u64, 5u64] {
        return false;
    }

    true
}

fn benchmark_sorter<S: Sorter>(sorter: &S, name: &str, seeds: &[u64]) -> BenchmarkResult {
    // Verify correctness first
    let correct = verify_sort(sorter, seeds);

    if !correct {
        return BenchmarkResult {
            name: name.to_string(),
            ops_per_second: 0.0,
            avg_ns: f64::INFINITY,
            correct: false,
        };
    }

    // Warmup
    for &seed in seeds.iter().take(WARMUP_ITERATIONS as usize) {
        let mut data = generate_random_data(seed, ARRAY_SIZE);
        sorter.sort(&mut data);
    }

    // Benchmark
    let mut total_time = Duration::ZERO;
    let mut iterations = 0u32;

    for &seed in seeds.iter().cycle().take(BENCH_ITERATIONS as usize) {
        let mut data = generate_random_data(seed, ARRAY_SIZE);

        let start = Instant::now();
        sorter.sort(&mut data);
        total_time += start.elapsed();
        iterations += 1;
    }

    let avg_ns = total_time.as_nanos() as f64 / iterations as f64;
    let ops_per_second = 1_000_000_000.0 / avg_ns;

    BenchmarkResult {
        name: name.to_string(),
        ops_per_second,
        avg_ns,
        correct,
    }
}

fn main() {
    // Generate seeds for reproducible benchmarks
    let seeds: Vec<u64> = (0..200).map(|i| 12345 + i * 7).collect();

    let mut results = Vec::new();
    let mut all_correct = true;

    // Benchmark all implementations
    let std_result = benchmark_sorter(&StdSort, "std_sort", &seeds);
    all_correct &= std_result.correct;
    results.push(std_result);

    let unstable_result = benchmark_sorter(&StdUnstable, "std_unstable", &seeds);
    all_correct &= unstable_result.correct;
    results.push(unstable_result);

    let heap_result = benchmark_sorter(&HeapSort, "heap_sort", &seeds);
    all_correct &= heap_result.correct;
    results.push(heap_result);

    let quick_result = benchmark_sorter(&QuickSort, "quick_sort", &seeds);
    all_correct &= quick_result.correct;
    results.push(quick_result);

    let radix_result = benchmark_sorter(&RadixSort, "radix_sort", &seeds);
    all_correct &= radix_result.correct;
    results.push(radix_result);

    let evolved_result = benchmark_sorter(&EvolvedSorter, "evolved", &seeds);
    all_correct &= evolved_result.correct;
    results.push(evolved_result);

    let full_results = FullResults {
        results,
        correctness: all_correct,
    };

    println!("{}", serde_json::to_string(&full_results).unwrap());
}
