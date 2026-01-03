//! Sparrow-inspired packing algorithm
//!
//! Key innovations from the Sparrow paper (arxiv.org/html/2509.13329):
//! 1. Temporary overlap tolerance with continuous collision metric
//! 2. Guided Local Search with dynamic penalty weights
//! 3. Two-phase architecture: Exploration then Compression
//!
//! The insight: "Feasible solutions are rare oases in a vast desert of infeasibility"
//! By temporarily permitting collisions, we can traverse otherwise inaccessible regions.

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Configuration for Sparrow algorithm
pub struct SparrowConfig {
    /// Number of exploration iterations
    pub exploration_iters: usize,
    /// Number of compression iterations
    pub compression_iters: usize,
    /// Initial penalty weight for overlaps
    pub base_penalty: f64,
    /// Penalty increase factor for persistent collisions
    pub penalty_growth: f64,
    /// Penalty decay factor for resolved collisions
    pub penalty_decay: f64,
    /// Maximum penalty weight
    pub max_penalty: f64,
    /// Initial strip shrink rate (exploration phase)
    pub shrink_rate: f64,
    /// Move step size
    pub step_size: f64,
}

impl Default for SparrowConfig {
    fn default() -> Self {
        Self {
            exploration_iters: 15000,
            compression_iters: 30000,
            base_penalty: 1.0,
            penalty_growth: 1.05,
            penalty_decay: 0.98,
            max_penalty: 50.0,
            shrink_rate: 0.01,
            step_size: 0.03,
        }
    }
}

/// Tree placement with continuous coordinates
#[derive(Clone, Debug)]
struct TreeState {
    x: f64,
    y: f64,
    rotation: usize, // 0-7 for 45 degree increments
}

impl TreeState {
    fn to_placed_tree(&self) -> PlacedTree {
        PlacedTree::new(self.x, self.y, self.rotation as f64 * 45.0)
    }
}

/// Compute bounding box of all trees
fn compute_bbox(trees: &[TreeState]) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for t in trees {
        let placed = t.to_placed_tree();
        let (bx1, by1, bx2, by2) = placed.bounds();
        min_x = min_x.min(bx1);
        min_y = min_y.min(by1);
        max_x = max_x.max(bx2);
        max_y = max_y.max(by2);
    }

    (min_x, min_y, max_x, max_y)
}

/// Compute side length of bounding box
fn bbox_side(trees: &[TreeState]) -> f64 {
    let (min_x, min_y, max_x, max_y) = compute_bbox(trees);
    (max_x - min_x).max(max_y - min_y)
}

/// Compute approximate penetration depth between two trees
/// Returns 0 if no overlap, positive value proportional to overlap severity
fn penetration_depth(t1: &PlacedTree, t2: &PlacedTree) -> f64 {
    // Fast bounding box check first
    let (ax1, ay1, ax2, ay2) = t1.bounds();
    let (bx1, by1, bx2, by2) = t2.bounds();

    // Calculate bounding box overlap
    let overlap_x = (ax2.min(bx2) - ax1.max(bx1)).max(0.0);
    let overlap_y = (ay2.min(by2) - ay1.max(by1)).max(0.0);

    if overlap_x == 0.0 || overlap_y == 0.0 {
        return 0.0;
    }

    // If bounding boxes overlap, check actual polygon overlap
    if !t1.overlaps(t2) {
        return 0.0;
    }

    // Approximate penetration using center distance vs minimum separation
    let c1x = (ax1 + ax2) / 2.0;
    let c1y = (ay1 + ay2) / 2.0;
    let c2x = (bx1 + bx2) / 2.0;
    let c2y = (by1 + by2) / 2.0;

    let dx = c2x - c1x;
    let dy = c2y - c1y;
    let dist = (dx * dx + dy * dy).sqrt();

    // Approximate minimum separation distance (based on bbox)
    let min_sep_x = (ax2 - ax1 + bx2 - bx1) / 2.0;
    let min_sep_y = (ay2 - ay1 + by2 - by1) / 2.0;
    let min_sep = (min_sep_x * min_sep_x + min_sep_y * min_sep_y).sqrt() * 0.7;

    // Penetration is how much closer they are than minimum separation
    (min_sep - dist).max(0.01)
}

/// Total overlap penalty for all pairs, with weighted penalties
fn total_overlap_penalty(trees: &[TreeState], weights: &[Vec<f64>]) -> f64 {
    let placed: Vec<PlacedTree> = trees.iter().map(|t| t.to_placed_tree()).collect();
    let mut total = 0.0;

    for i in 0..trees.len() {
        for j in (i + 1)..trees.len() {
            let depth = penetration_depth(&placed[i], &placed[j]);
            if depth > 0.0 {
                total += depth * weights[i][j];
            }
        }
    }

    total
}

/// Count number of overlapping pairs
fn count_overlaps(trees: &[TreeState]) -> usize {
    let placed: Vec<PlacedTree> = trees.iter().map(|t| t.to_placed_tree()).collect();
    let mut count = 0;

    for i in 0..trees.len() {
        for j in (i + 1)..trees.len() {
            if placed[i].overlaps(&placed[j]) {
                count += 1;
            }
        }
    }

    count
}

/// Check if configuration is feasible (no overlaps)
fn is_feasible(trees: &[TreeState]) -> bool {
    count_overlaps(trees) == 0
}

/// Initialize trees in a compact grid pattern
fn initialize_compact(n: usize, target_side: f64) -> Vec<TreeState> {
    let mut trees = Vec::with_capacity(n);

    // Arrange in rough grid
    let cols = ((n as f64).sqrt().ceil() as usize).max(1);
    let spacing = target_side / (cols as f64 + 1.0);

    for i in 0..n {
        let row = i / cols;
        let col = i % cols;

        let x = (col as f64 + 0.5 - cols as f64 / 2.0) * spacing;
        let y = (row as f64 + 0.5 - (n / cols) as f64 / 2.0) * spacing;

        // Alternate rotations for potential interlocking
        let rotation = ((row + col) % 8) as usize;

        trees.push(TreeState { x, y, rotation });
    }

    trees
}

/// Push overlapping trees apart
fn resolve_collision(
    trees: &mut [TreeState],
    i: usize,
    j: usize,
    step_size: f64,
    rng: &mut impl Rng,
) {
    let dx = trees[j].x - trees[i].x;
    let dy = trees[j].y - trees[i].y;
    let dist = (dx * dx + dy * dy).sqrt().max(0.01);

    // Push apart along separation vector
    let push = step_size * (1.0 + rng.gen::<f64>() * 0.5);
    trees[i].x -= push * dx / dist;
    trees[i].y -= push * dy / dist;
    trees[j].x += push * dx / dist;
    trees[j].y += push * dy / dist;

    // Sometimes try rotation change
    if rng.gen::<f64>() < 0.2 {
        trees[i].rotation = rng.gen_range(0..8);
    }
    if rng.gen::<f64>() < 0.2 {
        trees[j].rotation = rng.gen_range(0..8);
    }
}

/// Center trees around origin
fn center_trees(trees: &mut [TreeState]) {
    let (min_x, min_y, max_x, max_y) = compute_bbox(trees);
    let cx = (min_x + max_x) / 2.0;
    let cy = (min_y + max_y) / 2.0;

    for t in trees.iter_mut() {
        t.x -= cx;
        t.y -= cy;
    }
}

/// Compress trees toward center
fn compress_toward_center(trees: &mut [TreeState], factor: f64) {
    let (min_x, min_y, max_x, max_y) = compute_bbox(trees);
    let cx = (min_x + max_x) / 2.0;
    let cy = (min_y + max_y) / 2.0;

    for t in trees.iter_mut() {
        t.x = cx + (t.x - cx) * (1.0 - factor);
        t.y = cy + (t.y - cy) * (1.0 - factor);
    }
}

/// Sparrow-inspired packer
pub struct SparrowPacker {
    pub config: SparrowConfig,
}

impl Default for SparrowPacker {
    fn default() -> Self {
        Self {
            config: SparrowConfig::default(),
        }
    }
}

impl SparrowPacker {
    pub fn pack(&self, n: usize) -> Packing {
        if n == 0 {
            return Packing::new();
        }
        if n == 1 {
            let mut packing = Packing::new();
            packing.trees.push(PlacedTree::new(0.0, 0.0, 45.0));
            return packing;
        }

        let mut rng = rand::thread_rng();

        // Estimate target side length (based on tree area ~ 0.25)
        let tree_area = 0.25;
        let target_efficiency = 0.60;
        let target_side = ((n as f64 * tree_area / target_efficiency).sqrt()).max(1.0);

        // Initialize with compact configuration (will have overlaps)
        let mut trees = initialize_compact(n, target_side);

        // Initialize penalty weights
        let mut weights: Vec<Vec<f64>> = vec![vec![self.config.base_penalty; n]; n];

        let mut best_trees: Option<Vec<TreeState>> = None;
        let mut best_side = f64::INFINITY;

        // Phase 1: Exploration
        // Aggressively resolve overlaps while trying to shrink
        for iter in 0..self.config.exploration_iters {
            let progress = iter as f64 / self.config.exploration_iters as f64;
            let step = self.config.step_size * (1.0 - 0.5 * progress);

            // Find and resolve overlaps with guided local search
            let placed: Vec<PlacedTree> = trees.iter().map(|t| t.to_placed_tree()).collect();

            let mut any_overlap = false;
            for i in 0..n {
                for j in (i + 1)..n {
                    let depth = penetration_depth(&placed[i], &placed[j]);
                    if depth > 0.0 {
                        any_overlap = true;
                        // Increase penalty for this pair
                        weights[i][j] = (weights[i][j] * self.config.penalty_growth)
                            .min(self.config.max_penalty);
                        weights[j][i] = weights[i][j];

                        // Resolve collision
                        resolve_collision(&mut trees, i, j, step * weights[i][j].sqrt(), &mut rng);
                    } else {
                        // Decay penalty for resolved pairs
                        weights[i][j] *= self.config.penalty_decay;
                        weights[j][i] = weights[i][j];
                    }
                }
            }

            // If no overlaps, this is a feasible solution
            if !any_overlap {
                center_trees(&mut trees);
                let side = bbox_side(&trees);
                if side < best_side {
                    best_side = side;
                    best_trees = Some(trees.clone());
                }

                // Try to compress
                compress_toward_center(&mut trees, self.config.shrink_rate * (1.0 - progress));
            }

            // Periodically recenter
            if iter % 100 == 0 {
                center_trees(&mut trees);
            }
        }

        // Phase 2: Compression
        // Starting from best feasible solution, try to improve
        if let Some(ref best) = best_trees {
            trees = best.clone();
        }

        for iter in 0..self.config.compression_iters {
            let progress = iter as f64 / self.config.compression_iters as f64;

            // Random local move
            let idx = rng.gen_range(0..n);
            let old = trees[idx].clone();

            let move_type = rng.gen_range(0..5);
            match move_type {
                0 => {
                    // Small translation
                    let step = self.config.step_size * 0.5 * (1.0 - 0.5 * progress);
                    trees[idx].x += rng.gen_range(-step..step);
                    trees[idx].y += rng.gen_range(-step..step);
                }
                1 => {
                    // Move toward center
                    let (min_x, min_y, max_x, max_y) = compute_bbox(&trees);
                    let cx = (min_x + max_x) / 2.0;
                    let cy = (min_y + max_y) / 2.0;
                    let step = 0.02 * (1.0 - progress);
                    trees[idx].x += (cx - old.x) * step;
                    trees[idx].y += (cy - old.y) * step;
                }
                2 => {
                    // Rotation
                    trees[idx].rotation = rng.gen_range(0..8);
                }
                3 => {
                    // Move along axis
                    let step = self.config.step_size * (1.0 - 0.5 * progress);
                    if rng.gen() {
                        trees[idx].x += rng.gen_range(-step..step);
                    } else {
                        trees[idx].y += rng.gen_range(-step..step);
                    }
                }
                _ => {
                    // Combined move + rotation
                    let step = self.config.step_size * 0.3;
                    trees[idx].x += rng.gen_range(-step..step);
                    trees[idx].y += rng.gen_range(-step..step);
                    if rng.gen::<f64>() < 0.3 {
                        trees[idx].rotation = rng.gen_range(0..8);
                    }
                }
            }

            // Check if move is valid and improves solution
            if is_feasible(&trees) {
                let side = bbox_side(&trees);
                if side < best_side {
                    best_side = side;
                    best_trees = Some(trees.clone());
                } else if side > best_side * 1.01 {
                    // Reject moves that make things worse
                    trees[idx] = old;
                }
                // Accept neutral/slightly worse moves with some probability
            } else {
                // Reject infeasible moves
                trees[idx] = old;
            }
        }

        // Convert best solution to Packing
        let final_trees = best_trees.unwrap_or(trees);
        let mut packing = Packing::new();
        for t in final_trees {
            packing.trees.push(t.to_placed_tree());
        }
        packing
    }

    pub fn pack_all(&self, max_n: usize) -> Vec<Packing> {
        (1..=max_n).map(|n| self.pack(n)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparrow_small() {
        let packer = SparrowPacker::default();
        let packing = packer.pack(10);
        assert_eq!(packing.trees.len(), 10);
        assert!(!packing.has_overlaps());
    }
}
