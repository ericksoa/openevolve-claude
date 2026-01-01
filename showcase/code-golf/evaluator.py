#!/usr/bin/env python3
"""
Evaluator for ARC-AGI Code Golf evolution.

Scoring: max(1, 2500 - byte_count) for correct solutions, 0.001 for incorrect.
Goal: Evolve the shortest Python programs that correctly transform ARC-AGI grids.
"""

import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Any

TASKS_DIR = Path(__file__).parent / "tasks"
SOLUTIONS_DIR = Path(__file__).parent / "solutions"

def load_task(task_id: str) -> dict:
    """Load a task JSON file."""
    task_path = TASKS_DIR / f"{task_id}.json"
    if not task_path.exists():
        raise FileNotFoundError(f"Task {task_id} not found")
    with open(task_path) as f:
        return json.load(f)

def run_solution(solution_code: str, input_grid: list[list[int]], timeout: float = 5.0) -> tuple[Any, str | None]:
    """
    Run a solution on an input grid.
    Returns (output_grid, error_message).
    """
    # Create a wrapper that calls the solve function
    wrapper = f'''
import json
import sys

# Solution code
{solution_code}

# Run on input
input_grid = json.loads(sys.argv[1])
try:
    result = solve(input_grid)
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(wrapper)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path, json.dumps(input_grid)],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode != 0:
            return None, f"Runtime error: {result.stderr[:200]}"

        try:
            output = json.loads(result.stdout)
            if isinstance(output, dict) and "error" in output:
                return None, output["error"]
            return output, None
        except json.JSONDecodeError:
            return None, f"Invalid output: {result.stdout[:100]}"

    except subprocess.TimeoutExpired:
        return None, "Timeout"
    finally:
        os.unlink(temp_path)

def check_correctness(solution_code: str, task: dict) -> tuple[bool, list[dict]]:
    """
    Check if solution produces correct output for all train and test examples.
    Returns (all_correct, results_per_example).
    """
    results = []
    all_correct = True

    # Check all train examples
    for i, example in enumerate(task.get("train", [])):
        output, error = run_solution(solution_code, example["input"])
        correct = output == example["output"] if error is None else False
        results.append({
            "type": "train",
            "index": i,
            "correct": correct,
            "error": error
        })
        if not correct:
            all_correct = False

    # Check all test examples
    for i, example in enumerate(task.get("test", [])):
        output, error = run_solution(solution_code, example["input"])
        correct = output == example["output"] if error is None else False
        results.append({
            "type": "test",
            "index": i,
            "correct": correct,
            "error": error
        })
        if not correct:
            all_correct = False

    return all_correct, results

def calculate_score(byte_count: int, correct: bool) -> float:
    """Calculate competition score for a solution."""
    if not correct:
        return 0.001
    return max(1, 2500 - byte_count)

def evaluate_solution(task_id: str, solution_code: str) -> dict:
    """
    Evaluate a solution for a specific task.
    Returns fitness score and details.
    """
    task = load_task(task_id)
    byte_count = len(solution_code.encode('utf-8'))

    correct, check_results = check_correctness(solution_code, task)
    score = calculate_score(byte_count, correct)

    # Fitness normalized to 0-1 range (2500 max score)
    fitness = score / 2500.0

    return {
        "task_id": task_id,
        "fitness": round(fitness, 6),
        "score": score,
        "byte_count": byte_count,
        "correct": correct,
        "check_results": check_results
    }

def evaluate_all_solutions(solutions_dir: Path = None) -> dict:
    """Evaluate all solutions in the solutions directory."""
    solutions_dir = solutions_dir or SOLUTIONS_DIR

    total_score = 0
    results = {}
    solved = 0

    for task_file in sorted(TASKS_DIR.glob("*.json")):
        task_id = task_file.stem
        solution_file = solutions_dir / f"{task_id}.py"

        if solution_file.exists():
            with open(solution_file) as f:
                solution_code = f.read()
            result = evaluate_solution(task_id, solution_code)
            results[task_id] = result
            total_score += result["score"]
            if result["correct"]:
                solved += 1
        else:
            results[task_id] = {
                "task_id": task_id,
                "fitness": 0.0,
                "score": 0.001,
                "byte_count": 0,
                "correct": False,
                "error": "No solution file"
            }
            total_score += 0.001

    return {
        "total_score": round(total_score, 3),
        "tasks_solved": solved,
        "tasks_total": 400,
        "solve_rate": round(solved / 400 * 100, 1),
        "results": results
    }

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Evaluate all solutions
        result = evaluate_all_solutions()
        print(json.dumps({
            "total_score": result["total_score"],
            "tasks_solved": result["tasks_solved"],
            "solve_rate": result["solve_rate"]
        }, indent=2))
    elif len(sys.argv) == 2:
        # Evaluate single task with solution from stdin
        task_id = sys.argv[1]
        solution_code = sys.stdin.read()
        result = evaluate_solution(task_id, solution_code)
        print(json.dumps(result, indent=2))
    elif len(sys.argv) == 3:
        # Evaluate task_id with solution file
        task_id = sys.argv[1]
        solution_file = sys.argv[2]
        with open(solution_file) as f:
            solution_code = f.read()
        result = evaluate_solution(task_id, solution_code)
        print(json.dumps(result, indent=2))
