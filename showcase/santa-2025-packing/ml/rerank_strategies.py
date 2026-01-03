"""
Use trained value function to re-rank multi-run evolved solutions.

Instead of beam search (too slow in Python), run evolved multiple times
and use the model to predict which solutions will be best.

This is a simpler but practical integration.
"""

import argparse
import json
import subprocess
import torch
import numpy as np
from pathlib import Path
import time

from model import create_model


def run_evolved_once(rust_dir: str, max_n: int) -> dict:
    """Run evolved once and return packings as JSON."""
    cmd = f"{rust_dir}/target/release/generate_training_data {max_n} 1 /dev/stdout"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Parse the training data format to extract final solutions
    lines = result.stderr.strip().split('\n') if result.stderr else []
    return result.stdout


class ModelReranker:
    """Re-rank evolved solutions using learned value function."""

    def __init__(self, model_path: str, max_n: int = 50):
        self.max_n = max_n
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = create_model(checkpoint['model_type'], max_n=checkpoint['max_n'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded model (device={self.device})")

    def predict_from_features(self, features: list, target_n: float) -> float:
        """Predict final side length from feature vector."""
        features_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        target_n_t = torch.tensor([[target_n]], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pred = self.model(features_t, target_n_t)

        return pred.item()

    def score_solution(self, solution: dict) -> float:
        """Score a solution using the model."""
        features = solution['features']
        target_n = solution['target_n']
        return self.predict_from_features(features, target_n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--rust-dir', type=str, default='../rust')
    parser.add_argument('--max-n', type=int, default=30)
    parser.add_argument('--num-runs', type=int, default=10, help='Number of evolved runs to re-rank')
    args = parser.parse_args()

    reranker = ModelReranker(args.model, max_n=args.max_n)

    print(f"\nRunning {args.num_runs} evolved runs and re-ranking...")

    # Collect multiple runs
    all_solutions = {}  # n -> list of (final_side, features)

    for run in range(args.num_runs):
        print(f"Run {run + 1}/{args.num_runs}...")

        # Generate training data includes final_side for each n
        cmd = f"{args.rust_dir}/target/release/generate_training_data {args.max_n} 1 /tmp/run_{run}.jsonl"
        subprocess.run(cmd, shell=True, capture_output=True)

        with open(f'/tmp/run_{run}.jsonl', 'r') as f:
            for line in f:
                data = json.loads(line)
                n = data['n']
                # Only use 100% complete solutions
                if data['num_placed'] == n:
                    if n not in all_solutions:
                        all_solutions[n] = []
                    all_solutions[n].append({
                        'actual_side': data['final_side'],
                        'features': data['features'],
                        'target_n': data['target_n'],
                    })

    # For each n, re-rank using model
    print("\nRe-ranking with model predictions...")

    evolved_score = 0
    reranked_score = 0

    for n in range(1, args.max_n + 1):
        solutions = all_solutions.get(n, [])
        if not solutions:
            print(f"  n={n}: No solutions")
            continue

        # Get actual best
        actual_sides = [s['actual_side'] for s in solutions]
        best_actual = min(actual_sides)

        # Get model predictions
        model_preds = [reranker.score_solution(s) for s in solutions]
        best_pred_idx = np.argmin(model_preds)
        selected_side = solutions[best_pred_idx]['actual_side']

        # First solution (baseline - what evolved would give)
        first_side = solutions[0]['actual_side']

        evolved_contrib = first_side ** 2 / n
        reranked_contrib = selected_side ** 2 / n
        best_contrib = best_actual ** 2 / n

        evolved_score += evolved_contrib
        reranked_score += reranked_contrib

        improvement = "+" if selected_side < first_side else ("=" if selected_side == first_side else "-")

        if n <= 20 or n % 5 == 0:
            print(f"  n={n:3d}: first={first_side:.3f}, selected={selected_side:.3f}, best={best_actual:.3f} [{improvement}]")

    print(f"\nResults (n=1..{args.max_n}):")
    print(f"  Evolved (first run):  {evolved_score:.4f}")
    print(f"  Reranked (ML select): {reranked_score:.4f}")
    print(f"  Improvement: {(evolved_score - reranked_score) / evolved_score * 100:.2f}%")


if __name__ == '__main__':
    main()
