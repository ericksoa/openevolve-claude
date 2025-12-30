//! Incremental packing with local optimization
//!
//! Builds solutions incrementally, reusing n-1 solution for n.

use crate::{Packing, PackingAlgorithm, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Incremental packing with local search
pub struct IncrementalPacker {
    pub search_attempts: usize,
    pub local_opt_iterations: usize,
}

impl Default for IncrementalPacker {
    fn default() -> Self {
        Self {
            search_attempts: 30,
            local_opt_iterations: 500,
        }
    }
}

impl IncrementalPacker {
    /// Pack all trees from 1 to max_n, returning all packings
    pub fn pack_all(&self, max_n: usize) -> Vec<Packing> {
        let mut rng = rand::thread_rng();
        let mut packings = Vec::with_capacity(max_n);
        let mut prev_trees: Vec<PlacedTree> = Vec::new();

        for n in 1..=max_n {
            // Copy previous trees
            let mut trees = prev_trees.clone();

            // Add one new tree
            let new_tree = self.find_best_placement(&trees, &mut rng);
            trees.push(new_tree);

            // Local optimization
            self.local_optimize(&mut trees, &mut rng);

            // Create packing
            let mut packing = Packing::new();
            for tree in &trees {
                packing.trees.push(tree.clone());
            }
            packings.push(packing);

            // Save for next iteration
            prev_trees = trees;
        }

        packings
    }

    /// Find best placement for a new tree given existing trees
    fn find_best_placement(&self, existing: &[PlacedTree], rng: &mut impl Rng) -> PlacedTree {
        let mut best_tree = PlacedTree::new(0.0, 0.0, 90.0);
        let mut best_score = f64::INFINITY;

        // If first tree, place at origin
        if existing.is_empty() {
            return best_tree;
        }

        // Try multiple directions and rotations
        let angles = [0.0, 90.0, 180.0, 270.0];

        for _ in 0..self.search_attempts {
            // Random direction weighted toward corners
            let dir_angle = generate_weighted_angle(rng);
            let vx = dir_angle.cos();
            let vy = dir_angle.sin();

            for &tree_angle in &angles {
                // Binary search for closest valid placement
                let mut low = 0.0;
                let mut high = 20.0;

                while high - low > 0.01 {
                    let mid = (low + high) / 2.0;
                    let candidate = PlacedTree::new(mid * vx, mid * vy, tree_angle);

                    if is_valid_placement(&candidate, existing) {
                        high = mid;
                    } else {
                        low = mid;
                    }
                }

                if high < 19.0 {  // Found a valid placement
                    let candidate = PlacedTree::new(high * vx, high * vy, tree_angle);
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

    /// Local optimization using small perturbations
    fn local_optimize(&self, trees: &mut [PlacedTree], rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);

        for _ in 0..self.local_opt_iterations {
            let idx = rng.gen_range(0..trees.len());
            let old_tree = trees[idx].clone();

            // Choose perturbation type
            let perturbation = rng.gen_range(0..4);
            let success = match perturbation {
                0 => {
                    // Small translation
                    let dx = rng.gen_range(-0.05..0.05);
                    let dy = rng.gen_range(-0.05..0.05);
                    trees[idx] = PlacedTree::new(old_tree.x + dx, old_tree.y + dy, old_tree.angle_deg);
                    !has_overlap(trees, idx)
                }
                1 => {
                    // Rotation by 90 degrees
                    let new_angle = (old_tree.angle_deg + 90.0).rem_euclid(360.0);
                    trees[idx] = PlacedTree::new(old_tree.x, old_tree.y, new_angle);
                    !has_overlap(trees, idx)
                }
                2 => {
                    // Move toward center
                    let mag = (old_tree.x * old_tree.x + old_tree.y * old_tree.y).sqrt();
                    if mag > 0.1 {
                        let dx = -old_tree.x / mag * 0.03;
                        let dy = -old_tree.y / mag * 0.03;
                        trees[idx] = PlacedTree::new(old_tree.x + dx, old_tree.y + dy, old_tree.angle_deg);
                        !has_overlap(trees, idx)
                    } else {
                        false
                    }
                }
                _ => {
                    // Small rotation
                    let dangle = rng.gen_range(-15.0..15.0);
                    trees[idx] = PlacedTree::new(old_tree.x, old_tree.y, (old_tree.angle_deg + dangle).rem_euclid(360.0));
                    !has_overlap(trees, idx)
                }
            };

            if success {
                let new_side = compute_side_length(trees);
                if new_side <= current_side {
                    current_side = new_side;
                } else {
                    trees[idx] = old_tree;  // Revert
                }
            } else {
                trees[idx] = old_tree;  // Revert
            }
        }
    }
}

impl PackingAlgorithm for IncrementalPacker {
    fn pack(&self, n: usize) -> Packing {
        // This is inefficient for single n - use pack_all for the full benchmark
        let packings = self.pack_all(n);
        packings.into_iter().last().unwrap_or_else(Packing::new)
    }

    fn name(&self) -> &'static str {
        "incremental"
    }
}

/// Generate direction angle weighted toward corners
fn generate_weighted_angle(rng: &mut impl Rng) -> f64 {
    loop {
        let angle = rng.gen_range(0.0..2.0 * PI);
        if rng.gen::<f64>() < (2.0 * angle).sin().abs().max(0.3) {
            return angle;
        }
    }
}

/// Check if placement is valid (no overlaps)
fn is_valid_placement(tree: &PlacedTree, existing: &[PlacedTree]) -> bool {
    for other in existing {
        if tree.overlaps(other) {
            return false;
        }
    }
    true
}

/// Score a placement (lower is better) - returns resulting side length
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calculate_score;

    #[test]
    fn test_incremental_packing() {
        let packer = IncrementalPacker::default();
        let packings = packer.pack_all(20);

        assert_eq!(packings.len(), 20);

        for (i, packing) in packings.iter().enumerate() {
            assert_eq!(packing.trees.len(), i + 1);
            assert!(!packing.has_overlaps(), "Overlaps at n={}", i + 1);
        }
    }

    #[test]
    fn test_incremental_score() {
        let packer = IncrementalPacker::default();
        let packings = packer.pack_all(50);
        let score = calculate_score(&packings);
        println!("Score for n=1..50: {:.4}", score);
        assert!(score < 100.0);
    }
}
