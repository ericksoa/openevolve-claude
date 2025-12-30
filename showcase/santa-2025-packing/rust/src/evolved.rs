//! Evolved packing algorithm
//!
//! This module contains the champion packing algorithm discovered through evolution.

use crate::{Packing, PackingAlgorithm, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Evolved packing algorithm with optimized placement heuristics
pub struct Evolved;

impl PackingAlgorithm for Evolved {
    fn pack(&self, n: usize) -> Packing {
        let mut rng = rand::thread_rng();
        let mut packing = Packing::new();

        if n == 0 {
            return packing;
        }

        // Place first tree at origin with rotation for compact packing
        packing.try_add(PlacedTree::new(0.0, 0.0, 90.0));

        // Place remaining trees using evolved heuristics
        for tree_idx in 1..n {
            let best_tree = find_best_placement(&packing, tree_idx, n, &mut rng);
            packing.try_add(best_tree);
        }

        packing
    }

    fn name(&self) -> &'static str {
        "evolved_v1"
    }
}

/// Find the best placement for a new tree
fn find_best_placement(
    packing: &Packing,
    tree_idx: usize,
    total_trees: usize,
    rng: &mut impl Rng,
) -> PlacedTree {
    let mut best_tree = PlacedTree::new(0.0, 0.0, 0.0);
    let mut best_score = f64::INFINITY;

    // Evolved parameter: number of attempts scales with problem size
    let attempts = (20 + total_trees / 10).min(50);

    // Evolved parameter: preferred angles based on tree index
    let base_angles = select_rotation_angles(tree_idx, total_trees);

    for _ in 0..attempts {
        // Evolved: direction selection weighted for compact packing
        let dir_angle = select_direction_angle(tree_idx, total_trees, rng);
        let vx = dir_angle.cos();
        let vy = dir_angle.sin();

        // Try each preferred rotation
        for &tree_angle in &base_angles {
            // Binary search for placement radius
            let (px, py) = find_placement_radius(packing, vx, vy, tree_angle);

            let candidate = PlacedTree::new(px, py, tree_angle);
            if packing.can_place(&candidate) {
                // Score the placement
                let score = score_placement(&candidate, packing, total_trees);
                if score < best_score {
                    best_score = score;
                    best_tree = candidate;
                }
            }
        }
    }

    // Fallback if nothing found
    if best_score == f64::INFINITY {
        // Try grid search as fallback
        for ix in -100..=100 {
            for iy in -100..=100 {
                let x = ix as f64 * 0.1;
                let y = iy as f64 * 0.1;
                let candidate = PlacedTree::new(x, y, base_angles[0]);
                if packing.can_place(&candidate) {
                    return candidate;
                }
            }
        }
    }

    best_tree
}

/// Select preferred rotation angles based on tree index
fn select_rotation_angles(tree_idx: usize, _total: usize) -> Vec<f64> {
    // Evolved: use cardinal + diagonal rotations
    match tree_idx % 4 {
        0 => vec![0.0, 90.0, 180.0, 270.0],
        1 => vec![90.0, 270.0, 0.0, 180.0],
        2 => vec![180.0, 0.0, 90.0, 270.0],
        _ => vec![270.0, 90.0, 180.0, 0.0],
    }
}

/// Select direction angle for approaching center
fn select_direction_angle(tree_idx: usize, total: usize, rng: &mut impl Rng) -> f64 {
    // Evolved: mix of structured and random directions
    let structured_prob = 0.6;

    if rng.gen::<f64>() < structured_prob {
        // Evenly distributed directions
        let num_dirs = 8;
        let base_dir = (tree_idx % num_dirs) as f64 * 2.0 * PI / num_dirs as f64;
        base_dir + rng.gen_range(-0.2..0.2)
    } else {
        // Random direction weighted toward corners
        generate_weighted_angle(rng, total)
    }
}

/// Generate angle weighted for corner placement
fn generate_weighted_angle(rng: &mut impl Rng, _total: usize) -> f64 {
    loop {
        let angle = rng.gen_range(0.0..2.0 * PI);
        // Weight toward 45, 135, 225, 315 degrees
        let corner_weight = (2.0 * angle).sin().abs();
        if rng.gen::<f64>() < corner_weight.max(0.3) {
            return angle;
        }
    }
}

/// Find placement radius using binary search
fn find_placement_radius(
    packing: &Packing,
    vx: f64,
    vy: f64,
    tree_angle: f64,
) -> (f64, f64) {
    // Start far away
    let mut low = 0.0;
    let mut high = 20.0;

    // Find upper bound where placement is valid
    while high > low + 0.01 {
        let mid = (low + high) / 2.0;
        let px = mid * vx;
        let py = mid * vy;
        let candidate = PlacedTree::new(px, py, tree_angle);

        if packing.can_place(&candidate) {
            high = mid;
        } else {
            low = mid;
        }
    }

    (high * vx, high * vy)
}

/// Score a placement (lower is better)
fn score_placement(tree: &PlacedTree, packing: &Packing, _total: usize) -> f64 {
    let (min_x, min_y, max_x, max_y) = tree.bounds();

    // Calculate current packing bounds
    let mut pack_min_x = f64::INFINITY;
    let mut pack_min_y = f64::INFINITY;
    let mut pack_max_x = f64::NEG_INFINITY;
    let mut pack_max_y = f64::NEG_INFINITY;

    for t in &packing.trees {
        let (bmin_x, bmin_y, bmax_x, bmax_y) = t.bounds();
        pack_min_x = pack_min_x.min(bmin_x);
        pack_min_y = pack_min_y.min(bmin_y);
        pack_max_x = pack_max_x.max(bmax_x);
        pack_max_y = pack_max_y.max(bmax_y);
    }

    // New bounds if we add this tree
    let new_min_x = pack_min_x.min(min_x);
    let new_min_y = pack_min_y.min(min_y);
    let new_max_x = pack_max_x.max(max_x);
    let new_max_y = pack_max_y.max(max_y);

    let new_width = new_max_x - new_min_x;
    let new_height = new_max_y - new_min_y;
    let new_side = new_width.max(new_height);

    // Primary: minimize side length
    // Secondary: prefer symmetric expansion
    let balance_penalty = (new_width - new_height).abs() * 0.1;

    new_side + balance_penalty
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evolved_packing() {
        let algo = Evolved;
        let packing = algo.pack(10);
        assert_eq!(packing.trees.len(), 10);
        assert!(!packing.has_overlaps());
    }

    #[test]
    fn test_evolved_quality() {
        let algo = Evolved;
        let packing = algo.pack(20);
        let side = packing.side_length();
        // Should be reasonably compact
        assert!(side < 10.0, "Side length {} is too large", side);
    }
}
