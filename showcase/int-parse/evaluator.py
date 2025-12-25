#!/usr/bin/env python3
"""
Evaluator for integer parsing evolution.
Compiles the Rust code and runs benchmarks to compute fitness.
"""

import json
import subprocess
import sys
import os
import shutil
from pathlib import Path

# Add cargo to PATH
os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + os.environ.get("PATH", "")

def evaluate(evolved_code_path: str = None) -> dict:
    """
    Evaluate an evolved integer parser implementation.

    Returns JSON with:
    - fitness: 0.0-1.0 score (higher is better)
    - parses_per_second: raw throughput
    - vs_best_baseline: percentage improvement over best baseline
    - correctness: whether all tests pass
    """
    script_dir = Path(__file__).parent
    rust_dir = script_dir / "rust"
    evolved_rs = rust_dir / "src" / "evolved.rs"

    # If a specific file is provided, copy it to evolved.rs
    if evolved_code_path:
        source = Path(evolved_code_path)
        if source.exists():
            shutil.copy(source, evolved_rs)
        else:
            return {
                "fitness": 0.0,
                "parses_per_second": 0,
                "vs_best_baseline": 0,
                "correctness": False,
                "error": f"Source file not found: {evolved_code_path}"
            }

    # Clean and build
    try:
        subprocess.run(
            ["cargo", "clean"],
            cwd=rust_dir,
            capture_output=True,
            timeout=30
        )

        result = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=rust_dir,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            return {
                "fitness": 0.0,
                "parses_per_second": 0,
                "vs_best_baseline": 0,
                "correctness": False,
                "error": f"Compilation failed: {result.stderr[:500]}"
            }
    except subprocess.TimeoutExpired:
        return {
            "fitness": 0.0,
            "parses_per_second": 0,
            "vs_best_baseline": 0,
            "correctness": False,
            "error": "Build timeout"
        }

    # Run benchmark
    try:
        result = subprocess.run(
            ["./target/release/benchmark"],
            cwd=rust_dir,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            return {
                "fitness": 0.0,
                "parses_per_second": 0,
                "vs_best_baseline": 0,
                "correctness": False,
                "error": f"Benchmark failed: {result.stderr[:500]}"
            }

        data = json.loads(result.stdout)

    except subprocess.TimeoutExpired:
        return {
            "fitness": 0.0,
            "parses_per_second": 0,
            "vs_best_baseline": 0,
            "correctness": False,
            "error": "Benchmark timeout"
        }
    except json.JSONDecodeError as e:
        return {
            "fitness": 0.0,
            "parses_per_second": 0,
            "vs_best_baseline": 0,
            "correctness": False,
            "error": f"Invalid JSON output: {e}"
        }

    # Check correctness
    if not data.get("correctness", False):
        return {
            "fitness": 0.0,
            "parses_per_second": 0,
            "vs_best_baseline": 0,
            "correctness": False,
            "error": "Correctness check failed"
        }

    # Extract results
    results = {r["algorithm"]: r["parses_per_second"] for r in data["results"]}

    evolved_speed = results.get("evolved", 0)
    std_speed = results.get("std", 1)
    naive_speed = results.get("naive", 1)
    unrolled_speed = results.get("unrolled", 1)

    best_baseline = max(std_speed, naive_speed, unrolled_speed)

    # Calculate improvement
    vs_best = ((evolved_speed - best_baseline) / best_baseline) * 100 if best_baseline > 0 else 0
    vs_std = ((evolved_speed - std_speed) / std_speed) * 100 if std_speed > 0 else 0

    # Fitness function:
    # - Base: normalized speed vs std library
    # - Bonus for beating all baselines
    # - Scale to 0-1 range

    speed_ratio = evolved_speed / std_speed if std_speed > 0 else 0

    # Fitness: ratio to std, capped and scaled
    # 1.0 = same as std, >1.0 = faster
    fitness = min(speed_ratio, 2.0) / 2.0  # Cap at 2x std, scale to 0-1

    # Bonus for beating best baseline
    if evolved_speed > best_baseline:
        fitness = min(fitness + 0.1, 1.0)

    return {
        "fitness": round(fitness, 4),
        "parses_per_second": int(evolved_speed),
        "vs_std": round(vs_std, 2),
        "vs_best_baseline": round(vs_best, 2),
        "correctness": True,
        "all_results": results
    }


if __name__ == "__main__":
    evolved_path = sys.argv[1] if len(sys.argv) > 1 else None
    result = evaluate(evolved_path)
    print(json.dumps(result, indent=2))
