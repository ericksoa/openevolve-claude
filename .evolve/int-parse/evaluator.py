#!/usr/bin/env python3
"""
Evaluator for integer parsing evolution.
Runs the Rust benchmark and computes fitness scores.
"""

import json
import subprocess
import sys
import shutil
from pathlib import Path

RUST_DIR = Path(__file__).parent / "rust"
EVOLVED_RS = RUST_DIR / "src" / "evolved.rs"


def copy_mutation(mutation_path: str) -> bool:
    """Copy a mutation file to evolved.rs"""
    try:
        shutil.copy(mutation_path, EVOLVED_RS)
        return True
    except Exception as e:
        print(f"Error copying mutation: {e}", file=sys.stderr)
        return False


def build_release() -> bool:
    """Build the Rust project in release mode"""
    import os
    cargo_path = os.path.expanduser("~/.cargo/bin/cargo")

    try:
        # Clean first to ensure fresh build
        subprocess.run(
            [cargo_path, "clean"],
            cwd=RUST_DIR,
            capture_output=True,
            check=False,
        )

        # Build release
        result = subprocess.run(
            [cargo_path, "build", "--release"],
            cwd=RUST_DIR,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Build failed: {result.stderr}", file=sys.stderr)
            return False
        return True
    except Exception as e:
        print(f"Build error: {e}", file=sys.stderr)
        return False


def run_benchmark() -> dict:
    """Run the benchmark and return results"""
    import os
    cargo_path = os.path.expanduser("~/.cargo/bin/cargo")

    try:
        result = subprocess.run(
            [cargo_path, "run", "--release", "--bin", "benchmark"],
            cwd=RUST_DIR,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Benchmark failed: {result.stderr}", file=sys.stderr)
            return None

        return json.loads(result.stdout)
    except Exception as e:
        print(f"Benchmark error: {e}", file=sys.stderr)
        return None


def compute_fitness(benchmark_results: dict) -> dict:
    """Compute fitness score from benchmark results"""
    if not benchmark_results or not benchmark_results.get("correctness"):
        return {
            "fitness": 0.0,
            "ops_per_second": 0,
            "vs_best_baseline": 0.0,
            "correctness": False,
            "all_results": benchmark_results,
        }

    results = benchmark_results["results"]

    # Find evolved and baseline results
    evolved_result = None
    baseline_results = []

    for r in results:
        if r["name"] == "evolved":
            evolved_result = r
        else:
            baseline_results.append(r)

    if not evolved_result:
        return {
            "fitness": 0.0,
            "ops_per_second": 0,
            "vs_best_baseline": 0.0,
            "correctness": True,
            "all_results": benchmark_results,
        }

    evolved_speed = evolved_result["ops_per_second"]
    best_baseline_speed = max(r["ops_per_second"] for r in baseline_results)

    # Speed ratio
    speed_ratio = evolved_speed / best_baseline_speed if best_baseline_speed > 0 else 0

    # Scale to 0-1, cap at 2x improvement
    fitness = min(speed_ratio, 2.0) / 2.0

    # Bonus for beating all baselines
    if evolved_speed > best_baseline_speed:
        fitness = min(fitness + 0.1, 1.0)

    # Percentage improvement
    vs_best = ((evolved_speed - best_baseline_speed) / best_baseline_speed * 100) if best_baseline_speed > 0 else 0

    return {
        "fitness": round(fitness, 4),
        "ops_per_second": int(evolved_speed),
        "vs_best_baseline": round(vs_best, 2),
        "correctness": True,
        "all_results": benchmark_results,
    }


def main():
    # Check if a mutation path was provided
    mutation_path = sys.argv[1] if len(sys.argv) > 1 else None

    if mutation_path:
        if not copy_mutation(mutation_path):
            print(json.dumps({"fitness": 0.0, "error": "Failed to copy mutation"}))
            return

    # Build
    if not build_release():
        print(json.dumps({"fitness": 0.0, "error": "Build failed"}))
        return

    # Run benchmark
    benchmark_results = run_benchmark()
    if not benchmark_results:
        print(json.dumps({"fitness": 0.0, "error": "Benchmark failed"}))
        return

    # Compute and output fitness
    fitness_result = compute_fitness(benchmark_results)
    print(json.dumps(fitness_result))


if __name__ == "__main__":
    main()
