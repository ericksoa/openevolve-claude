#!/usr/bin/env python3
"""
Evaluator for string search evolution.
Compiles and runs the Rust benchmark, returning fitness scores.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def evaluate(evolved_path: Path | None = None) -> dict:
    """
    Evaluate the current or provided evolved implementation.

    Returns a dict with:
      - fitness: 0.0-1.0 score
      - ops_per_second: operations per second for evolved
      - vs_best_baseline: percentage improvement over best baseline
      - correctness: whether all tests pass
      - all_results: full benchmark results
    """
    script_dir = Path(__file__).parent
    rust_dir = script_dir / "rust"
    evolved_rs = rust_dir / "src" / "evolved.rs"

    # Copy evolved code if provided
    if evolved_path:
        shutil.copy(evolved_path, evolved_rs)

    # Clean and build
    cargo = Path.home() / ".cargo" / "bin" / "cargo"
    if not cargo.exists():
        cargo = "cargo"

    result = run_command([str(cargo), "clean"], cwd=rust_dir)
    if result.returncode != 0:
        return {
            "fitness": 0.0,
            "error": f"cargo clean failed: {result.stderr}",
            "correctness": False,
        }

    result = run_command([str(cargo), "build", "--release"], cwd=rust_dir)
    if result.returncode != 0:
        return {
            "fitness": 0.0,
            "error": f"Build failed: {result.stderr}",
            "correctness": False,
        }

    # Run benchmark
    benchmark_bin = rust_dir / "target" / "release" / "benchmark"
    result = run_command([str(benchmark_bin)])

    if result.returncode != 0:
        return {
            "fitness": 0.0,
            "error": f"Benchmark failed: {result.stderr}",
            "correctness": False,
        }

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        return {
            "fitness": 0.0,
            "error": f"Invalid JSON output: {e}\nOutput: {result.stdout}",
            "correctness": False,
        }

    if not data.get("correctness", False):
        return {
            "fitness": 0.0,
            "error": "Correctness tests failed",
            "correctness": False,
            "all_results": data.get("results", []),
        }

    # Parse results
    results = {r["name"]: r["ops_per_second"] for r in data["results"]}

    evolved_speed = results.get("evolved", 0)
    memchr_speed = results.get("memchr", 1)  # The target to beat
    naive_speed = results.get("naive", 1)
    std_speed = results.get("std", 1)
    bm_speed = results.get("boyer-moore", 1)

    # Best baseline is memchr (the SIMD-optimized library we're trying to beat)
    best_baseline = memchr_speed

    # Calculate improvement over memchr
    if best_baseline > 0:
        speed_ratio = evolved_speed / best_baseline
        vs_best_baseline = (speed_ratio - 1) * 100  # percentage improvement
    else:
        speed_ratio = 1.0
        vs_best_baseline = 0.0

    # Fitness: scale to 0-1, cap at 2x improvement
    fitness = min(speed_ratio, 2.0) / 2.0

    # Bonus for beating memchr
    if evolved_speed > memchr_speed:
        fitness = min(fitness + 0.1, 1.0)

    return {
        "fitness": round(fitness, 4),
        "ops_per_second": round(evolved_speed, 2),
        "vs_best_baseline": round(vs_best_baseline, 2),
        "correctness": True,
        "all_results": {
            "evolved": round(evolved_speed, 2),
            "memchr": round(memchr_speed, 2),
            "naive": round(naive_speed, 2),
            "std": round(std_speed, 2),
            "boyer-moore": round(bm_speed, 2),
        },
    }


def main():
    evolved_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    result = evaluate(evolved_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
