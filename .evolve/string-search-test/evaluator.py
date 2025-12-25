#!/usr/bin/env python3
"""Evaluator for string search evolution."""

import json
import os
import subprocess
import sys
import shutil
from pathlib import Path

# Ensure cargo is in PATH
os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + os.environ.get("PATH", "")

RUST_DIR = Path(__file__).parent / "rust"

def evaluate(evolved_code_path: str = None) -> dict:
    """Evaluate the evolved string search implementation."""

    # Copy evolved code if provided
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

    # Parse results
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "fitness": 0.0,
            "error": f"Invalid JSON: {result.stdout}",
            "correctness": False
        }

    # Check correctness
    if not data.get("correctness", False):
        return {
            "fitness": 0.0,
            "error": "Correctness check failed",
            "correctness": False,
            "all_results": data
        }

    # Find evolved and best baseline
    results = {r["name"]: r["ops_per_second"] for r in data["results"]}
    evolved_ops = results.get("evolved", 0)

    # Best baseline (excluding evolved)
    baselines = {k: v for k, v in results.items() if k != "evolved"}
    best_baseline = max(baselines.values()) if baselines else 1
    best_baseline_name = max(baselines, key=baselines.get) if baselines else "none"

    # Calculate fitness
    speed_ratio = evolved_ops / best_baseline if best_baseline > 0 else 0
    fitness = min(speed_ratio, 2.0) / 2.0

    # Bonus for beating all baselines
    if evolved_ops > best_baseline:
        fitness = min(fitness + 0.1, 1.0)

    vs_best = ((evolved_ops / best_baseline) - 1) * 100 if best_baseline > 0 else 0

    return {
        "fitness": round(fitness, 4),
        "ops_per_second": evolved_ops,
        "vs_best_baseline": round(vs_best, 2),
        "best_baseline": best_baseline_name,
        "best_baseline_ops": best_baseline,
        "correctness": True,
        "all_results": results
    }

if __name__ == "__main__":
    evolved_path = sys.argv[1] if len(sys.argv) > 1 else None
    result = evaluate(evolved_path)
    print(json.dumps(result, indent=2))
