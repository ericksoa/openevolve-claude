use rand::Rng;
use serde::Serialize;
use std::time::{Duration, Instant};
use string_search::{baselines::*, evolved::EvolvedSearcher, SubstringSearcher};

#[derive(Serialize)]
struct BenchmarkResult {
    name: String,
    ops_per_second: f64,
    total_time_ns: u64,
    iterations: u64,
}

#[derive(Serialize)]
struct FullResults {
    results: Vec<BenchmarkResult>,
    correctness: bool,
}

fn generate_test_data() -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut rng = rand::thread_rng();
    let mut cases = Vec::new();

    // Large haystack with needle at various positions
    let large: Vec<u8> = (0..100_000).map(|_| rng.gen_range(b'a'..=b'z')).collect();

    // Needle at end
    let mut haystack1 = large.clone();
    let needle1 = b"MARKER".to_vec();
    haystack1.extend_from_slice(&needle1);
    cases.push((haystack1, needle1));

    // Needle at beginning
    let mut haystack2 = b"NEEDLE".to_vec();
    haystack2.extend_from_slice(&large);
    cases.push((haystack2, b"NEEDLE".to_vec()));

    // Needle in middle
    let mut haystack3 = large[..50_000].to_vec();
    haystack3.extend_from_slice(b"FINDME");
    haystack3.extend_from_slice(&large[50_000..]);
    cases.push((haystack3, b"FINDME".to_vec()));

    // Needle not present (worst case for some algorithms)
    cases.push((large.clone(), b"ZZZZZ".to_vec()));

    // Short haystack, short needle
    cases.push((b"hello world".to_vec(), b"wor".to_vec()));

    // Single byte needle (memchr's specialty)
    cases.push((large.clone(), b"x".to_vec()));

    // Medium haystack with common substring
    let medium: Vec<u8> = (0..10_000).map(|_| rng.gen_range(b'a'..=b'd')).collect();
    cases.push((medium.clone(), b"abcd".to_vec()));

    // Repeated pattern haystack
    let repeated: Vec<u8> = b"abcdefgh".repeat(10_000);
    cases.push((repeated, b"efgh".to_vec()));

    // Binary data
    let binary: Vec<u8> = (0..50_000).map(|_| rng.gen::<u8>()).collect();
    cases.push((binary, vec![0x00, 0x01, 0x02, 0x03]));

    // DNA-like sequence
    let dna: Vec<u8> = (0..100_000)
        .map(|_| *[b'A', b'C', b'G', b'T'].choose(&mut rng).unwrap())
        .collect();
    cases.push((dna, b"GATTACA".to_vec()));

    cases
}

fn verify_correctness<S: SubstringSearcher>(searcher: &S, name: &str) -> bool {
    // Basic correctness tests
    let tests = vec![
        (b"hello world".as_slice(), b"world".as_slice(), Some(6)),
        (b"hello", b"hello", Some(0)),
        (b"hello", b"ello", Some(1)),
        (b"hello", b"xyz", None),
        (b"hello", b"", Some(0)),
        (b"", b"a", None),
        (b"aaaa", b"aa", Some(0)),
        (b"abcabc", b"abc", Some(0)),
    ];

    for (haystack, needle, expected) in tests {
        let result = searcher.find(haystack, needle);
        if result != expected {
            eprintln!(
                "{}: find({:?}, {:?}) = {:?}, expected {:?}",
                name, haystack, needle, result, expected
            );
            return false;
        }
    }
    true
}

fn benchmark<S: SubstringSearcher>(
    searcher: &S,
    name: &str,
    test_data: &[(Vec<u8>, Vec<u8>)],
) -> BenchmarkResult {
    // Warmup
    for _ in 0..10 {
        for (haystack, needle) in test_data {
            let _ = searcher.find(haystack, needle);
        }
    }

    // Benchmark
    let target_time = Duration::from_millis(500);
    let mut total_iterations: u64 = 0;
    let start = Instant::now();

    while start.elapsed() < target_time {
        for (haystack, needle) in test_data {
            let _ = searcher.find(haystack, needle);
        }
        total_iterations += test_data.len() as u64;
    }

    let elapsed = start.elapsed();
    let ops_per_second = total_iterations as f64 / elapsed.as_secs_f64();

    BenchmarkResult {
        name: name.to_string(),
        ops_per_second,
        total_time_ns: elapsed.as_nanos() as u64,
        iterations: total_iterations,
    }
}

fn main() {
    let test_data = generate_test_data();

    // Verify correctness
    let naive_correct = verify_correctness(&NaiveSearcher, "naive");
    let std_correct = verify_correctness(&StdSearcher, "std");
    let memchr_correct = verify_correctness(&MemchrSearcher, "memchr");
    let bm_correct = verify_correctness(&BoyerMooreSearcher, "boyer-moore");
    let evolved_correct = verify_correctness(&EvolvedSearcher, "evolved");

    let all_correct =
        naive_correct && std_correct && memchr_correct && bm_correct && evolved_correct;

    if !all_correct {
        let results = FullResults {
            results: vec![],
            correctness: false,
        };
        println!("{}", serde_json::to_string(&results).unwrap());
        return;
    }

    // Run benchmarks
    let mut results = vec![];

    results.push(benchmark(&NaiveSearcher, "naive", &test_data));
    results.push(benchmark(&StdSearcher, "std", &test_data));
    results.push(benchmark(&MemchrSearcher, "memchr", &test_data));
    results.push(benchmark(&BoyerMooreSearcher, "boyer-moore", &test_data));
    results.push(benchmark(&EvolvedSearcher, "evolved", &test_data));

    let full_results = FullResults {
        results,
        correctness: true,
    };

    println!("{}", serde_json::to_string(&full_results).unwrap());
}

trait SliceExt<T> {
    fn choose(&self, rng: &mut impl Rng) -> Option<&T>;
}

impl<T> SliceExt<T> for [T] {
    fn choose(&self, rng: &mut impl Rng) -> Option<&T> {
        if self.is_empty() {
            None
        } else {
            Some(&self[rng.gen_range(0..self.len())])
        }
    }
}
