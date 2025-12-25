#!/usr/bin/env python3
"""
Evaluator for Rust String Search Evolution

This evaluator:
1. Copies the evolved code to the Rust project
2. Compiles the Rust project in release mode
3. Runs benchmarks comparing evolved vs baselines
4. Returns a fitness score

The score combines:
- Correctness (must pass all tests)
- Performance (searches per second)
- Improvement over baselines (bonus points)
"""

import subprocess
import json
import tempfile
import shutil
import os
import sys
import random
from pathlib import Path

# Configuration
RUST_PROJECT_DIR = Path(__file__).parent / "rust"
EVOLVED_FILE = RUST_PROJECT_DIR / "src" / "evolved.rs"
WARMUP_ITERATIONS = 3
MEASURED_ITERATIONS = 10

# Ensure cargo is in PATH
CARGO_HOME = Path.home() / ".cargo" / "bin"
if CARGO_HOME.exists():
    os.environ["PATH"] = f"{CARGO_HOME}:{os.environ.get('PATH', '')}"


def generate_test_corpus():
    """Generate diverse test data for benchmarking"""
    texts = []
    patterns = []

    # Random text (various sizes)
    chars = "abcdefghijklmnopqrstuvwxyz"
    for size in [1000, 10000, 50000]:
        text = ''.join(random.choice(chars) for _ in range(size))
        texts.append(text)

    # Repetitive text (stress test)
    texts.append("a" * 10000 + "b")
    texts.append("ab" * 5000)
    texts.append("abc" * 3333)

    # Natural language-like
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
             "lorem", "ipsum", "dolor", "sit", "amet", "consectetur"]
    natural = ' '.join(random.choice(words) for _ in range(2000))
    texts.append(natural)

    # DNA-like sequences
    dna_chars = "ACGT"
    dna = ''.join(random.choice(dna_chars) for _ in range(50000))
    texts.append(dna)

    # Binary-like (limited alphabet)
    binary = ''.join(random.choice("01") for _ in range(20000))
    texts.append(binary)

    # Patterns of varying lengths and characteristics
    patterns = [
        "abc",              # short
        "abcdef",           # medium
        "the quick",        # with space
        "aaaaaab",          # repetitive with mismatch
        "ACGTACGT",         # DNA motif
        "fox jumps over",   # longer natural
        "0101010101",       # binary pattern
        "z",                # single char (rare)
        "the",              # common word
        "xyz123abc",        # mixed
    ]

    return texts, patterns


def compile_rust_project(evolved_code: str) -> tuple[bool, str]:
    """Compile the Rust project with the evolved code"""

    # Backup original evolved.rs
    backup_path = EVOLVED_FILE.with_suffix('.rs.bak')
    if EVOLVED_FILE.exists():
        shutil.copy(EVOLVED_FILE, backup_path)

    try:
        # Write evolved code
        with open(EVOLVED_FILE, 'w') as f:
            f.write(evolved_code)

        # Compile in release mode
        result = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=RUST_PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            return False, f"Compilation failed:\n{result.stderr}"

        return True, "Compilation successful"

    except subprocess.TimeoutExpired:
        return False, "Compilation timeout"
    except Exception as e:
        return False, f"Compilation error: {str(e)}"


def run_benchmarks() -> dict:
    """Run the benchmark binary and return results"""

    texts, patterns = generate_test_corpus()

    benchmark_input = {
        "texts": texts,
        "patterns": patterns,
        "warmup_iterations": WARMUP_ITERATIONS,
        "measured_iterations": MEASURED_ITERATIONS,
    }

    # Run benchmark binary
    benchmark_binary = RUST_PROJECT_DIR / "target" / "release" / "benchmark"

    if sys.platform == "win32":
        benchmark_binary = benchmark_binary.with_suffix(".exe")

    try:
        result = subprocess.run(
            [str(benchmark_binary)],
            input=json.dumps(benchmark_input),
            capture_output=True,
            text=True,
            timeout=300,
            cwd=RUST_PROJECT_DIR
        )

        if result.returncode != 0:
            return {
                "error": f"Benchmark failed: {result.stderr}",
                "score": 0.0
            }

        return json.loads(result.stdout)

    except subprocess.TimeoutExpired:
        return {"error": "Benchmark timeout", "score": 0.0}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid benchmark output: {e}", "score": 0.0}
    except Exception as e:
        return {"error": str(e), "score": 0.0}


def run_tests() -> tuple[bool, str]:
    """Run Rust tests to verify correctness"""
    try:
        result = subprocess.run(
            ["cargo", "test", "--release"],
            cwd=RUST_PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            return False, f"Tests failed:\n{result.stdout}\n{result.stderr}"

        return True, "All tests passed"

    except subprocess.TimeoutExpired:
        return False, "Test timeout"
    except Exception as e:
        return False, f"Test error: {str(e)}"


def evaluate(program_path: str) -> dict:
    """
    Main evaluation function for evolution

    Args:
        program_path: Path to the evolved Rust code file

    Returns:
        Dictionary with 'score' and optional metadata
    """

    # Read evolved code
    try:
        with open(program_path, 'r') as f:
            evolved_code = f.read()
    except Exception as e:
        return {"score": 0.0, "error": f"Cannot read program: {e}"}

    # Validate basic structure
    if "impl StringSearch for EvolvedSearch" not in evolved_code:
        return {
            "score": 0.0,
            "error": "Invalid code: missing StringSearch implementation"
        }

    if "fn search(" not in evolved_code:
        return {
            "score": 0.0,
            "error": "Invalid code: missing search function"
        }

    # Compile
    success, message = compile_rust_project(evolved_code)
    if not success:
        return {"score": 0.0, "error": message, "stage": "compilation"}

    # Run tests
    success, message = run_tests()
    if not success:
        return {"score": 0.0, "error": message, "stage": "testing"}

    # Run benchmarks
    benchmark_results = run_benchmarks()

    if "error" in benchmark_results:
        return {
            "score": 0.0,
            "error": benchmark_results["error"],
            "stage": "benchmarking"
        }

    # Extract metrics
    evolved_result = next(
        (r for r in benchmark_results["results"] if r["algorithm"] == "evolved"),
        None
    )

    if not evolved_result:
        return {"score": 0.0, "error": "No evolved results found"}

    if not evolved_result["all_correct"]:
        return {
            "score": 0.0,
            "error": "Evolved algorithm produced incorrect results",
            "stage": "correctness"
        }

    # Return full results
    return {
        "score": benchmark_results["score"],
        "searches_per_second": evolved_result["searches_per_second"],
        "vs_best_baseline": benchmark_results["evolved_vs_best_baseline"],
        "all_results": benchmark_results["results"],
        "correctness": "passed",
    }


# For command-line usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <program_path>", file=sys.stderr)
        sys.exit(1)

    result = evaluate(sys.argv[1])
    print(json.dumps(result, indent=2))
