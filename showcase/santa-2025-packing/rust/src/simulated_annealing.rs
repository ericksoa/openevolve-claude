//! Simulated Annealing packing algorithm
//!
//! Uses simulated annealing to optimize tree placements.

use crate::{Packing, PackingAlgorithm, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Simulated Annealing packing optimizer
pub struct SimulatedAnnealing {
    pub iterations: usize,
    pub initial_temp: f64,
    pub cooling_rate: f64,
}

impl Default for SimulatedAnnealing {
    fn default() -> Self {
        Self {
            iterations: 10000,
            initial_temp: 1.0,
            cooling_rate: 0.9995,
        }
    }
}

impl PackingAlgorithm for SimulatedAnnealing {
    fn pack(&self, n: usize) -> Packing {
        if n == 0 {
            return Packing::new();
        }

        let mut rng = rand::thread_rng();

        // Start with greedy initial placement
        let mut trees = greedy_initial_placement(n, &mut rng);
        let mut current_side = compute_side_length(&trees);

        let mut temp = self.initial_temp;

        for _ in 0..self.iterations {
            // Pick a random tree to perturb
            let idx = rng.gen_range(0..n);

            // Try a random perturbation
            let old_tree = trees[idx].clone();
            let perturbation = rng.gen_range(0..3);

            match perturbation {
                0 => {
                    // Small translation
                    let dx = rng.gen_range(-0.1..0.1);
                    let dy = rng.gen_range(-0.1..0.1);
                    trees[idx] = PlacedTree::new(
                        old_tree.x + dx,
                        old_tree.y + dy,
                        old_tree.angle_deg,
                    );
                }
                1 => {
                    // Rotation
                    let dangle = rng.gen_range(-45.0..45.0);
                    trees[idx] = PlacedTree::new(
                        old_tree.x,
                        old_tree.y,
                        (old_tree.angle_deg + dangle).rem_euclid(360.0),
                    );
                }
                _ => {
                    // Larger translation
                    let dx = rng.gen_range(-0.3..0.3);
                    let dy = rng.gen_range(-0.3..0.3);
                    trees[idx] = PlacedTree::new(
                        old_tree.x + dx,
                        old_tree.y + dy,
                        old_tree.angle_deg,
                    );
                }
            }

            // Check if valid (no overlaps)
            let valid = !has_overlap(&trees, idx);

            if valid {
                let new_side = compute_side_length(&trees);
                let delta = new_side - current_side;

                // Accept if better, or probabilistically if worse
                if delta < 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                    current_side = new_side;
                } else {
                    trees[idx] = old_tree;
                }
            } else {
                trees[idx] = old_tree;
            }

            temp *= self.cooling_rate;
        }

        // Convert to Packing
        let mut packing = Packing::new();
        for tree in trees {
            packing.trees.push(tree);
        }
        packing
    }

    fn name(&self) -> &'static str {
        "simulated_annealing"
    }
}

/// Create initial greedy placement
fn greedy_initial_placement(n: usize, rng: &mut impl Rng) -> Vec<PlacedTree> {
    let mut trees = Vec::with_capacity(n);

    // First tree at origin
    trees.push(PlacedTree::new(0.0, 0.0, 90.0));

    for _ in 1..n {
        let tree_angle = [0.0, 90.0, 180.0, 270.0][rng.gen_range(0..4)];

        let mut best_tree = PlacedTree::new(0.0, 0.0, tree_angle);
        let mut best_radius = f64::INFINITY;

        // Try multiple directions
        for attempt in 0..20 {
            let dir_angle = (attempt as f64 / 20.0) * 2.0 * PI + rng.gen_range(-0.2..0.2);
            let vx = dir_angle.cos();
            let vy = dir_angle.sin();

            // Binary search for placement
            let mut low = 0.0;
            let mut high = 15.0;

            while high - low > 0.02 {
                let mid = (low + high) / 2.0;
                let candidate = PlacedTree::new(mid * vx, mid * vy, tree_angle);

                let mut overlaps = false;
                for existing in &trees {
                    if candidate.overlaps(existing) {
                        overlaps = true;
                        break;
                    }
                }

                if overlaps {
                    low = mid;
                } else {
                    high = mid;
                }
            }

            if high < best_radius {
                best_radius = high;
                best_tree = PlacedTree::new(high * vx, high * vy, tree_angle);
            }
        }

        trees.push(best_tree);
    }

    trees
}

/// Compute side length of bounding square
fn compute_side_length(trees: &[PlacedTree]) -> f64 {
    if trees.is_empty() {
        return 0.0;
    }

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for tree in trees {
        let (bmin_x, bmin_y, bmax_x, bmax_y) = tree.bounds();
        min_x = min_x.min(bmin_x);
        min_y = min_y.min(bmin_y);
        max_x = max_x.max(bmax_x);
        max_y = max_y.max(bmax_y);
    }

    (max_x - min_x).max(max_y - min_y)
}

/// Check if tree at index overlaps with any other tree
fn has_overlap(trees: &[PlacedTree], idx: usize) -> bool {
    for (i, tree) in trees.iter().enumerate() {
        if i != idx && trees[idx].overlaps(tree) {
            return true;
        }
    }
    false
}

/// High-quality simulated annealing with more iterations
pub struct SimulatedAnnealingHQ;

impl PackingAlgorithm for SimulatedAnnealingHQ {
    fn pack(&self, n: usize) -> Packing {
        // Scale iterations with problem size
        let sa = SimulatedAnnealing {
            iterations: 5000 + n * 100,
            initial_temp: 0.5,
            cooling_rate: 0.9998,
        };
        sa.pack(n)
    }

    fn name(&self) -> &'static str {
        "sa_high_quality"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sa_packing() {
        let algo = SimulatedAnnealing::default();
        let packing = algo.pack(10);
        assert_eq!(packing.trees.len(), 10);
        assert!(!packing.has_overlaps());
    }

    #[test]
    fn test_sa_quality() {
        let algo = SimulatedAnnealing::default();
        let packing = algo.pack(20);
        let side = packing.side_length();
        assert!(side < 8.0, "Side length {} is too large", side);
    }
}
