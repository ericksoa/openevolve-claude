#!/usr/bin/env python3
"""
Evaluator for sorting algorithm evolution.
Compiles Rust code, runs benchmarks, and computes fitness.
"""

import json
import subprocess
import sys
import shutil
import os
from pathlib import Path
import math

# Ensure cargo is in PATH
os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + os.environ.get("PATH", "")

RUST_DIR = Path(__file__).parent / "rust"
EVOLVED_RS = RUST_DIR / "src" / "evolved.rs"


def run_command(cmd: list[str], cwd: Path = None, timeout: int = 120) -> tuple[bool, str, str]:
    """Run a command and return (success, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)


def compile_rust() -> tuple[bool, str]:
    """Compile the Rust benchmark. Returns (success, error_message)."""
    # Clean first to ensure fresh build
    run_command(["cargo", "clean"], cwd=RUST_DIR, timeout=30)

    # Build
    success, stdout, stderr = run_command(
        ["cargo", "build", "--release"],
        cwd=RUST_DIR,
        timeout=120,
    )

    if not success:
        return False, stderr

    return True, ""


def run_benchmark() -> dict | None:
    """Run the benchmark and return results."""
    binary = RUST_DIR / "target" / "release" / "benchmark"

    if not binary.exists():
        return None

    success, stdout, stderr = run_command([str(binary)], timeout=60)

    if not success:
        return None

    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return None


def calculate_fitness(results: dict) -> dict:
    """Calculate fitness score from benchmark results."""
    if not results or not results.get("correctness", False):
        return {
            "fitness": 0.0,
            "ops_per_second": 0,
            "vs_best_baseline": 0.0,
            "correctness": False,
            "all_results": results,
        }

    # Find evolved and baseline results
    evolved_ops = 0.0
    best_baseline_ops = 0.0
    best_baseline_name = ""

    for r in results.get("results", []):
        if r["name"] == "evolved":
            evolved_ops = r["ops_per_second"]
        else:
            if r["ops_per_second"] > best_baseline_ops:
                best_baseline_ops = r["ops_per_second"]
                best_baseline_name = r["name"]

    if best_baseline_ops == 0:
        return {
            "fitness": 0.0,
            "ops_per_second": evolved_ops,
            "vs_best_baseline": 0.0,
            "correctness": True,
            "all_results": results,
        }

    # Calculate improvement ratio
    ratio = evolved_ops / best_baseline_ops
    vs_best_baseline = (ratio - 1.0) * 100  # percentage improvement

    # Fitness: scale ratio to 0-1, cap at 2x improvement
    fitness = min(ratio, 2.0) / 2.0

    # Bonus for beating best baseline
    if ratio > 1.0:
        fitness = min(fitness + 0.1, 1.0)

    return {
        "fitness": round(fitness, 4),
        "ops_per_second": round(evolved_ops, 1),
        "vs_best_baseline": round(vs_best_baseline, 2),
        "best_baseline": best_baseline_name,
        "best_baseline_ops": round(best_baseline_ops, 1),
        "correctness": True,
        "all_results": results,
    }


def main():
    # If a path is provided, copy it to evolved.rs
    if len(sys.argv) > 1:
        source = Path(sys.argv[1])
        if source.exists():
            shutil.copy(source, EVOLVED_RS)

    # Compile
    success, error = compile_rust()
    if not success:
        result = {
            "fitness": 0.0,
            "ops_per_second": 0,
            "vs_best_baseline": 0.0,
            "correctness": False,
            "error": f"Compilation failed: {error}",
        }
        print(json.dumps(result))
        return

    # Run benchmark
    bench_results = run_benchmark()
    if bench_results is None:
        result = {
            "fitness": 0.0,
            "ops_per_second": 0,
            "vs_best_baseline": 0.0,
            "correctness": False,
            "error": "Benchmark execution failed",
        }
        print(json.dumps(result))
        return

    # Calculate fitness
    fitness_result = calculate_fitness(bench_results)
    print(json.dumps(fitness_result))


if __name__ == "__main__":
    main()
