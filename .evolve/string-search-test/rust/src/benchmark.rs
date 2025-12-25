use string_search_test::{SubstringSearcher, baselines::*, evolved::EvolvedSearcher};
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

fn generate_test_data() -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut rng = rand::thread_rng();
    let mut tests = Vec::new();

    // Random text with embedded patterns
    for _ in 0..50 {
        let len: usize = rng.gen_range(1000..10000);
        let haystack: Vec<u8> = (0..len).map(|_| rng.gen_range(b'a'..=b'z')).collect();
        let needle_len: usize = rng.gen_range(4..20);
        let start = rng.gen_range(0..len.saturating_sub(needle_len).max(1));
        let needle = haystack[start..start + needle_len.min(len - start)].to_vec();
        tests.push((haystack, needle));
    }

    // Worst case: repeating pattern
    for _ in 0..10 {
        let haystack = vec![b'a'; 5000];
        let mut needle = vec![b'a'; rng.gen_range(10..50)];
        needle.push(b'b'); // Never matches
        tests.push((haystack, needle));
    }

    // Best case: unique first char
    for _ in 0..20 {
        let len: usize = rng.gen_range(1000..5000);
        let haystack: Vec<u8> = (0..len).map(|_| rng.gen_range(b'a'..=b'z')).collect();
        let needle_len: usize = rng.gen_range(8..30);
        let needle: Vec<u8> = (0..needle_len).map(|_| rng.gen_range(b'A'..=b'Z')).collect();
        tests.push((haystack, needle));
    }

    // Real-world patterns
    let lorem = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";
    tests.push((lorem.to_vec(), b"tempor".to_vec()));
    tests.push((lorem.to_vec(), b"magna aliqua".to_vec()));
    tests.push((lorem.to_vec(), b"not found".to_vec()));

    // DNA-like sequences
    for _ in 0..10 {
        let alphabet = [b'A', b'C', b'G', b'T'];
        let len: usize = rng.gen_range(2000..8000);
        let haystack: Vec<u8> = (0..len).map(|_| alphabet[rng.gen_range(0..4)]).collect();
        let needle_len: usize = rng.gen_range(6..20);
        let start = rng.gen_range(0..len.saturating_sub(needle_len).max(1));
        let needle = haystack[start..start + needle_len.min(len - start)].to_vec();
        tests.push((haystack, needle));
    }

    tests
}

fn verify_correctness<S: SubstringSearcher>(searcher: &S, tests: &[(Vec<u8>, Vec<u8>)]) -> bool {
    let reference = NaiveSearcher;
    for (haystack, needle) in tests {
        let expected = reference.find(haystack, needle);
        let actual = searcher.find(haystack, needle);
        if expected != actual {
            return false;
        }
    }
    true
}

fn benchmark<S: SubstringSearcher>(searcher: &S, tests: &[(Vec<u8>, Vec<u8>)], warmup_ms: u64, run_ms: u64) -> f64 {
    // Warmup
    let warmup_end = Instant::now() + Duration::from_millis(warmup_ms);
    while Instant::now() < warmup_end {
        for (haystack, needle) in tests {
            std::hint::black_box(searcher.find(haystack, needle));
        }
    }

    // Benchmark
    let mut ops = 0u64;
    let start = Instant::now();
    let end = start + Duration::from_millis(run_ms);
    while Instant::now() < end {
        for (haystack, needle) in tests {
            std::hint::black_box(searcher.find(haystack, needle));
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

    // Benchmark each implementation
    let naive = NaiveSearcher;
    let correct = verify_correctness(&naive, &tests);
    all_correct &= correct;
    let ops = benchmark(&naive, &tests, 100, 500);
    results.push(BenchmarkResult { name: "naive".into(), ops_per_second: ops, correct });

    let std = StdSearcher;
    let correct = verify_correctness(&std, &tests);
    all_correct &= correct;
    let ops = benchmark(&std, &tests, 100, 500);
    results.push(BenchmarkResult { name: "std".into(), ops_per_second: ops, correct });

    let memchr = MemchrSearcher;
    let correct = verify_correctness(&memchr, &tests);
    all_correct &= correct;
    let ops = benchmark(&memchr, &tests, 100, 500);
    results.push(BenchmarkResult { name: "memchr".into(), ops_per_second: ops, correct });

    let bm = BoyerMooreSearcher;
    let correct = verify_correctness(&bm, &tests);
    all_correct &= correct;
    let ops = benchmark(&bm, &tests, 100, 500);
    results.push(BenchmarkResult { name: "boyer_moore".into(), ops_per_second: ops, correct });

    let evolved = EvolvedSearcher;
    let correct = verify_correctness(&evolved, &tests);
    all_correct &= correct;
    let ops = benchmark(&evolved, &tests, 100, 500);
    results.push(BenchmarkResult { name: "evolved".into(), ops_per_second: ops, correct });

    let full = FullResults { results, correctness: all_correct };
    println!("{}", serde_json::to_string(&full).unwrap());
}
