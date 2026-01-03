"""
ML-guided beam search for tree packing.

Uses trained value function to guide placement decisions.
"""

import argparse
import json
import torch
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import time

from model import create_model


# Tree shape (same as Rust code)
TREE_VERTICES = [
    (0.0, 0.0),      # Base center
    (0.10, 0.0),
    (0.30, 0.30),
    (0.20, 0.30),
    (0.35, 0.55),
    (0.20, 0.55),
    (0.25, 0.80),
    (0.10, 0.80),
    (0.00, 1.00),    # Top
    (-0.10, 0.80),
    (-0.25, 0.80),
    (-0.20, 0.55),
    (-0.35, 0.55),
    (-0.20, 0.30),
    (-0.30, 0.30),
    (-0.10, 0.0),
]


@dataclass
class PlacedTree:
    x: float
    y: float
    angle_deg: float

    def get_vertices(self) -> List[Tuple[float, float]]:
        """Get rotated and translated vertices."""
        angle_rad = np.radians(self.angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        vertices = []
        for vx, vy in TREE_VERTICES:
            rx = vx * cos_a - vy * sin_a + self.x
            ry = vx * sin_a + vy * cos_a + self.y
            vertices.append((rx, ry))
        return vertices

    def bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box (min_x, min_y, max_x, max_y)."""
        vertices = self.get_vertices()
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        return min(xs), min(ys), max(xs), max(ys)


def polygons_overlap(verts1: List[Tuple[float, float]],
                     verts2: List[Tuple[float, float]]) -> bool:
    """Check if two convex-ish polygons overlap using SAT."""
    def get_edges(verts):
        edges = []
        for i in range(len(verts)):
            p1, p2 = verts[i], verts[(i + 1) % len(verts)]
            edges.append((p2[0] - p1[0], p2[1] - p1[1]))
        return edges

    def project(verts, axis):
        dots = [v[0] * axis[0] + v[1] * axis[1] for v in verts]
        return min(dots), max(dots)

    def overlap_1d(min1, max1, min2, max2):
        return max1 >= min2 and max2 >= min1

    # Check all edges as potential separating axes
    for edge in get_edges(verts1) + get_edges(verts2):
        # Normal to edge
        axis = (-edge[1], edge[0])
        # Normalize
        length = np.sqrt(axis[0]**2 + axis[1]**2)
        if length < 1e-10:
            continue
        axis = (axis[0] / length, axis[1] / length)

        min1, max1 = project(verts1, axis)
        min2, max2 = project(verts2, axis)

        if not overlap_1d(min1, max1, min2, max2):
            return False  # Found separating axis

    return True  # No separating axis found


def trees_overlap(t1: PlacedTree, t2: PlacedTree) -> bool:
    """Check if two trees overlap."""
    # Quick bounding box check
    b1 = t1.bounds()
    b2 = t2.bounds()
    if b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1]:
        return False

    return polygons_overlap(t1.get_vertices(), t2.get_vertices())


@dataclass
class PackingState:
    trees: List[PlacedTree]

    def side_length(self) -> float:
        if not self.trees:
            return 0.0

        min_x = min(t.bounds()[0] for t in self.trees)
        min_y = min(t.bounds()[1] for t in self.trees)
        max_x = max(t.bounds()[2] for t in self.trees)
        max_y = max(t.bounds()[3] for t in self.trees)

        return max(max_x - min_x, max_y - min_y)

    def is_valid(self) -> bool:
        for i in range(len(self.trees)):
            for j in range(i + 1, len(self.trees)):
                if trees_overlap(self.trees[i], self.trees[j]):
                    return False
        return True

    def to_features(self, max_n: int) -> torch.Tensor:
        """Convert to feature vector for model."""
        features = [len(self.trees) / max_n]

        for tree in self.trees:
            features.extend([
                tree.x / 10.0,
                tree.y / 10.0,
                tree.angle_deg / 360.0,
            ])

        # Pad
        while len(features) < max_n * 3 + 1:
            features.append(0.0)

        return torch.tensor(features, dtype=torch.float32)

    def copy(self) -> 'PackingState':
        return PackingState(trees=[PlacedTree(t.x, t.y, t.angle_deg) for t in self.trees])


class MLGuidedPacker:
    """Beam search packer guided by learned value function."""

    def __init__(self, model_path: str, max_n: int = 50, beam_width: int = 10):
        self.max_n = max_n
        self.beam_width = beam_width
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = create_model(checkpoint['model_type'], max_n=checkpoint['max_n'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded model from {model_path}")

    def predict_value(self, states: List[PackingState], target_n: int) -> List[float]:
        """Predict final side length for multiple states."""
        if not states:
            return []

        features = torch.stack([s.to_features(self.max_n) for s in states]).to(self.device)
        target_n_tensor = torch.full((len(states), 1), target_n / self.max_n, device=self.device)

        with torch.no_grad():
            predictions = self.model(features, target_n_tensor)

        return predictions.squeeze(-1).cpu().tolist()

    def generate_candidates(self, state: PackingState, num_candidates: int = 50) -> List[PackingState]:
        """Generate candidate next states using greedy boundary placement."""
        candidates = []
        angles = [0, 45, 90, 135, 180, 225, 270, 315]

        if not state.trees:
            # First tree at origin with each rotation
            for angle in angles:
                candidates.append(PackingState(trees=[PlacedTree(0, 0, angle)]))
            return candidates

        # Current bounding box
        min_x = min(t.bounds()[0] for t in state.trees)
        min_y = min(t.bounds()[1] for t in state.trees)
        max_x = max(t.bounds()[2] for t in state.trees)
        max_y = max(t.bounds()[3] for t in state.trees)

        # Try placing on each boundary (like evolved's greedy approach)
        boundary_positions = []

        # Along top edge
        for x in np.linspace(min_x - 0.5, max_x + 0.5, 15):
            boundary_positions.append((x, max_y + 0.3))
            boundary_positions.append((x, max_y + 0.5))

        # Along bottom edge
        for x in np.linspace(min_x - 0.5, max_x + 0.5, 15):
            boundary_positions.append((x, min_y - 0.3))
            boundary_positions.append((x, min_y - 0.5))

        # Along right edge
        for y in np.linspace(min_y - 0.5, max_y + 0.5, 15):
            boundary_positions.append((max_x + 0.3, y))
            boundary_positions.append((max_x + 0.5, y))

        # Along left edge
        for y in np.linspace(min_y - 0.5, max_y + 0.5, 15):
            boundary_positions.append((min_x - 0.3, y))
            boundary_positions.append((min_x - 0.5, y))

        # Corner positions
        for dx in [-0.3, 0, 0.3]:
            for dy in [-0.3, 0, 0.3]:
                boundary_positions.append((max_x + dx, max_y + dy))
                boundary_positions.append((min_x + dx, min_y + dy))
                boundary_positions.append((max_x + dx, min_y + dy))
                boundary_positions.append((min_x + dx, max_y + dy))

        # Try each position with each rotation
        for x, y in boundary_positions:
            for angle in angles:
                new_tree = PlacedTree(x, y, angle)
                new_state = state.copy()
                new_state.trees.append(new_tree)

                if new_state.is_valid():
                    # Binary search to move closer
                    best_x, best_y = x, y
                    cx = (min_x + max_x) / 2
                    cy = (min_y + max_y) / 2

                    for _ in range(5):
                        # Try moving toward center
                        dx = (cx - best_x) * 0.3
                        dy = (cy - best_y) * 0.3

                        test_tree = PlacedTree(best_x + dx, best_y + dy, angle)
                        test_state = state.copy()
                        test_state.trees.append(test_tree)

                        if test_state.is_valid():
                            best_x += dx
                            best_y += dy

                    final_tree = PlacedTree(best_x, best_y, angle)
                    final_state = state.copy()
                    final_state.trees.append(final_tree)

                    if final_state.is_valid():
                        candidates.append(final_state)

        # Deduplicate similar candidates
        unique = []
        for c in candidates:
            is_dup = False
            for u in unique:
                if len(c.trees) == len(u.trees):
                    last_c = c.trees[-1]
                    last_u = u.trees[-1]
                    dist = ((last_c.x - last_u.x)**2 + (last_c.y - last_u.y)**2)**0.5
                    if dist < 0.05 and last_c.angle_deg == last_u.angle_deg:
                        is_dup = True
                        break
            if not is_dup:
                unique.append(c)

        return unique[:num_candidates]

    def pack(self, n: int) -> PackingState:
        """Pack n trees using ML-guided beam search."""
        if n == 0:
            return PackingState(trees=[])

        # Start with single tree at origin
        initial = PackingState(trees=[PlacedTree(0, 0, 45)])

        if n == 1:
            return initial

        # Beam search
        beam = [initial]

        for step in range(1, n):
            # Generate candidates for each state in beam
            all_candidates = []
            for state in beam:
                candidates = self.generate_candidates(state, num_candidates=100)
                all_candidates.extend(candidates)

            if not all_candidates:
                print(f"Warning: No valid candidates at step {step}")
                break

            # Score candidates using model
            values = self.predict_value(all_candidates, n)

            # Select top beam_width candidates (lower predicted side = better)
            scored = list(zip(all_candidates, values))
            scored.sort(key=lambda x: x[1])
            beam = [s for s, v in scored[:self.beam_width]]

        # Return best final state
        final_values = self.predict_value(beam, n)
        best_idx = np.argmin(final_values)
        return beam[best_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--max-n', type=int, default=50)
    parser.add_argument('--beam-width', type=int, default=10)
    parser.add_argument('--test-n', type=int, default=20, help='Test up to this N')
    args = parser.parse_args()

    packer = MLGuidedPacker(args.model, max_n=args.max_n, beam_width=args.beam_width)

    print(f"\nTesting ML-guided packing (n=1..{args.test_n})")

    total_score = 0
    start_time = time.time()

    for n in range(1, args.test_n + 1):
        state = packer.pack(n)
        side = state.side_length()
        score = side * side / n
        total_score += score
        print(f"  n={n:3d}: side={side:.4f}, score_contrib={score:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTotal score (n=1..{args.test_n}): {total_score:.4f}")
    print(f"Time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
