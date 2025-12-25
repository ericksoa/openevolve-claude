#!/usr/bin/env python3
"""Evaluator for sort evolution demo."""

import json
import os
import subprocess
import sys
import shutil
from pathlib import Path

os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + os.environ.get("PATH", "")

RUST_DIR = Path(__file__).parent / "rust"

def evaluate(evolved_code_path: str = None) -> dict:
    """Evaluate the evolved sorting implementation."""

    if evolved_code_path:
        shutil.copy(evolved_code_path, RUST_DIR / "src" / "evolved.rs")

    # Build
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=RUST_DIR,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return {
            "fitness": 0.0,
            "error": f"Build failed: {result.stderr}",
            "correctness": False
        }

    # Run benchmark
    result = subprocess.run(
        ["cargo", "run", "--release", "--bin", "benchmark"],
        cwd=RUST_DIR,
        capture_output=True,
        text=True,
        timeout=120
    )

    if result.returncode != 0:
        return {
            "fitness": 0.0,
            "error": f"Benchmark failed: {result.stderr}",
            "correctness": False
        }

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "fitness": 0.0,
            "error": f"Invalid JSON: {result.stdout}",
            "correctness": False
        }

    if not data.get("correctness", False):
        return {
            "fitness": 0.0,
            "error": "Correctness check failed",
            "correctness": False,
            "all_results": data
        }

    results = {r["name"]: r["ops_per_second"] for r in data["results"]}
    evolved_ops = results.get("evolved", 0)
    bubble_ops = results.get("bubble", 1)
    std_ops = results.get("std", 1)

    # Calculate improvement over bubble sort
    vs_bubble = ((evolved_ops / bubble_ops) - 1) * 100 if bubble_ops > 0 else 0
    vs_std = ((evolved_ops / std_ops) - 1) * 100 if std_ops > 0 else 0

    # Fitness based on improvement over bubble sort
    speed_ratio = evolved_ops / bubble_ops if bubble_ops > 0 else 0
    fitness = min(speed_ratio / 100, 1.0)  # Cap at 100x improvement

    return {
        "fitness": round(fitness, 4),
        "ops_per_second": evolved_ops,
        "vs_bubble": round(vs_bubble, 1),
        "vs_std": round(vs_std, 1),
        "correctness": True,
        "all_results": results
    }

if __name__ == "__main__":
    evolved_path = sys.argv[1] if len(sys.argv) > 1 else None
    result = evaluate(evolved_path)
    print(json.dumps(result, indent=2))
