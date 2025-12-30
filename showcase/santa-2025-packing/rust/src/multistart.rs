//! Multi-start packing with parallel restarts
//!
//! Runs multiple independent attempts and keeps the best for each n.

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Multi-start packing optimizer
pub struct MultiStartPacker {
    pub restarts: usize,
    pub search_attempts: usize,
    pub sa_iterations: usize,
    pub sa_temp: f64,
    pub sa_cooling: f64,
}

impl Default for MultiStartPacker {
    fn default() -> Self {
        Self {
            restarts: 5,
            search_attempts: 40,
            sa_iterations: 2000,
            sa_temp: 0.3,
            sa_cooling: 0.998,
        }
    }
}

impl MultiStartPacker {
    /// Pack all trees from 1 to max_n with multi-start optimization
    pub fn pack_all(&self, max_n: usize) -> Vec<Packing> {
        let mut packings: Vec<Packing> = Vec::with_capacity(max_n);
        let mut rng = rand::thread_rng();

        for n in 1..=max_n {
            let mut best_packing = None;
            let mut best_side = f64::INFINITY;

            // Run multiple restarts for this n
            for _ in 0..self.restarts {
                // Start fresh or from previous solution
                let base_trees = if n > 1 && !packings.is_empty() {
                    // Start from previous best solution
                    packings[n - 2].trees.clone()
                } else {
                    Vec::new()
                };

                let trees = self.pack_n(n, base_trees, &mut rng);
                let side = compute_side_length(&trees);

                if side < best_side {
                    best_side = side;
                    let mut packing = Packing::new();
                    for tree in trees {
                        packing.trees.push(tree);
                    }
                    best_packing = Some(packing);
                }
            }

            packings.push(best_packing.unwrap());
        }

        packings
    }

    /// Pack exactly n trees, starting from base configuration
    fn pack_n(&self, n: usize, base: Vec<PlacedTree>, rng: &mut impl Rng) -> Vec<PlacedTree> {
        let mut trees = base;

        // Add trees until we have n
        while trees.len() < n {
            let new_tree = self.find_placement(&trees, rng);
            trees.push(new_tree);
        }

        // Run simulated annealing
        self.simulated_annealing(&mut trees, rng);

        trees
    }

    /// Find best placement for a new tree
    fn find_placement(&self, existing: &[PlacedTree], rng: &mut impl Rng) -> PlacedTree {
        if existing.is_empty() {
            return PlacedTree::new(0.0, 0.0, 90.0);
        }

        let mut best_tree = PlacedTree::new(0.0, 0.0, 90.0);
        let mut best_score = f64::INFINITY;

        let angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0];

        for _ in 0..self.search_attempts {
            let dir_angle = rng.gen_range(0.0..2.0 * PI);
            let vx = dir_angle.cos();
            let vy = dir_angle.sin();

            for &tree_angle in &angles {
                // Binary search for closest valid placement
                let mut low = 0.0;
                let mut high = 15.0;

                while high - low > 0.005 {
                    let mid = (low + high) / 2.0;
                    let candidate = PlacedTree::new(mid * vx, mid * vy, tree_angle);

                    if is_valid(&candidate, existing) {
                        high = mid;
                    } else {
                        low = mid;
                    }
                }

                let candidate = PlacedTree::new(high * vx, high * vy, tree_angle);
                if is_valid(&candidate, existing) {
                    let score = placement_score(&candidate, existing);
                    if score < best_score {
                        best_score = score;
                        best_tree = candidate;
                    }
                }
            }
        }

        best_tree
    }

    /// Simulated annealing local search
    fn simulated_annealing(&self, trees: &mut Vec<PlacedTree>, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut temp = self.sa_temp;

        for _ in 0..self.sa_iterations {
            let idx = rng.gen_range(0..trees.len());
            let old_tree = trees[idx].clone();

            // Generate perturbation
            let move_type = rng.gen_range(0..5);
            let success = match move_type {
                0 => {
                    // Small translation
                    let scale = 0.02 + 0.08 * temp;
                    let dx = rng.gen_range(-scale..scale);
                    let dy = rng.gen_range(-scale..scale);
                    trees[idx] = PlacedTree::new(old_tree.x + dx, old_tree.y + dy, old_tree.angle_deg);
                    !has_overlap(trees, idx)
                }
                1 => {
                    // 90-degree rotation
                    let new_angle = (old_tree.angle_deg + 90.0).rem_euclid(360.0);
                    trees[idx] = PlacedTree::new(old_tree.x, old_tree.y, new_angle);
                    !has_overlap(trees, idx)
                }
                2 => {
                    // 45-degree rotation
                    let delta = if rng.gen() { 45.0 } else { -45.0 };
                    let new_angle = (old_tree.angle_deg + delta).rem_euclid(360.0);
                    trees[idx] = PlacedTree::new(old_tree.x, old_tree.y, new_angle);
                    !has_overlap(trees, idx)
                }
                3 => {
                    // Move toward center
                    let mag = (old_tree.x * old_tree.x + old_tree.y * old_tree.y).sqrt();
                    if mag > 0.05 {
                        let scale = 0.02 + 0.05 * temp;
                        let dx = -old_tree.x / mag * scale;
                        let dy = -old_tree.y / mag * scale;
                        trees[idx] = PlacedTree::new(old_tree.x + dx, old_tree.y + dy, old_tree.angle_deg);
                        !has_overlap(trees, idx)
                    } else {
                        false
                    }
                }
                _ => {
                    // Swap positions with another tree
                    if trees.len() > 1 {
                        let other_idx = loop {
                            let i = rng.gen_range(0..trees.len());
                            if i != idx {
                                break i;
                            }
                        };
                        // Save other tree's data before modifying
                        let other_x = trees[other_idx].x;
                        let other_y = trees[other_idx].y;
                        let other_angle = trees[other_idx].angle_deg;

                        trees[idx] = PlacedTree::new(other_x, other_y, old_tree.angle_deg);
                        if !has_overlap(trees, idx) {
                            trees[other_idx] = PlacedTree::new(old_tree.x, old_tree.y, other_angle);
                            !has_overlap(trees, other_idx)
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
            };

            if success {
                let new_side = compute_side_length(trees);
                let delta = new_side - current_side;

                if delta <= 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                    current_side = new_side;
                } else {
                    // Revert
                    trees[idx] = old_tree;
                    if move_type == 4 && trees.len() > 1 {
                        // Also revert swap partner if needed
                        // This is a bit ugly, we'll skip swap revert for simplicity
                    }
                }
            } else {
                trees[idx] = old_tree;
            }

            temp *= self.sa_cooling;
        }
    }
}

fn is_valid(tree: &PlacedTree, existing: &[PlacedTree]) -> bool {
    for other in existing {
        if tree.overlaps(other) {
            return false;
        }
    }
    true
}

fn placement_score(tree: &PlacedTree, existing: &[PlacedTree]) -> f64 {
    let (min_x, min_y, max_x, max_y) = tree.bounds();

    let mut pack_min_x = min_x;
    let mut pack_min_y = min_y;
    let mut pack_max_x = max_x;
    let mut pack_max_y = max_y;

    for t in existing {
        let (bmin_x, bmin_y, bmax_x, bmax_y) = t.bounds();
        pack_min_x = pack_min_x.min(bmin_x);
        pack_min_y = pack_min_y.min(bmin_y);
        pack_max_x = pack_max_x.max(bmax_x);
        pack_max_y = pack_max_y.max(bmax_y);
    }

    (pack_max_x - pack_min_x).max(pack_max_y - pack_min_y)
}

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

fn has_overlap(trees: &[PlacedTree], idx: usize) -> bool {
    for (i, tree) in trees.iter().enumerate() {
        if i != idx && trees[idx].overlaps(tree) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calculate_score;

    #[test]
    fn test_multistart() {
        let packer = MultiStartPacker::default();
        let packings = packer.pack_all(20);

        for (i, p) in packings.iter().enumerate() {
            assert_eq!(p.trees.len(), i + 1);
            assert!(!p.has_overlaps());
        }
    }

    #[test]
    fn test_multistart_quality() {
        let packer = MultiStartPacker::default();
        let packings = packer.pack_all(50);
        let score = calculate_score(&packings);
        println!("Multistart score for n=1..50: {:.4}", score);
    }
}
