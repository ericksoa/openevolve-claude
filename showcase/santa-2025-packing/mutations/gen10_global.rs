//! Evolved Packing Algorithm - Generation 10 GLOBAL OPTIMIZATION
//!
//! This module contains the evolved packing heuristics.
//! The code is designed to be mutated by LLM-guided evolution.
//!
//! Evolution targets:
//! - placement_score(): How to score candidate placements
//! - select_angles(): Which rotation angles to try
//! - select_direction(): How to choose placement directions
//! - sa_move(): Local search move operators
//!
//! MUTATION STRATEGY: GLOBAL OPTIMIZATION (Gen10)
//! Completely new paradigm: place ALL trees at once instead of incrementally:
//!
//! Key differences from incremental approach:
//! - Initial placement: distribute ALL n trees in a grid pattern at once
//! - No incremental building - all trees placed simultaneously
//! - Global SA: optimize all trees together to minimize bounding box
//! - Much longer SA (100000 iterations) since optimizing entire configuration
//! - Objective: minimize total bounding box directly from the start
//!
//! Target: Beat Gen6's 94.14 at n=200 with global optimization

use crate::{Packing, PlacedTree};
use rand::Rng;

/// Evolved packing configuration
/// These parameters are tuned through evolution
pub struct EvolvedConfig {
    // Simulated annealing - much longer for global optimization
    pub sa_iterations: usize,
    pub sa_initial_temp: f64,
    pub sa_cooling_rate: f64,
    pub sa_min_temp: f64,

    // Move parameters
    pub translation_scale: f64,
    pub center_pull_strength: f64,

    // Early exit threshold
    pub early_exit_threshold: usize,

    // Boundary focus probability
    pub boundary_focus_prob: f64,

    // Global optimization parameters
    pub grid_spacing: f64,           // Initial grid spacing for tree placement
    pub compaction_passes: usize,    // Number of compaction passes after SA
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen10 GLOBAL OPTIMIZATION: Configuration for global approach
        Self {
            sa_iterations: 100000,           // Much longer SA for global optimization
            sa_initial_temp: 0.8,            // Higher initial temp for more exploration
            sa_cooling_rate: 0.99997,        // Very slow cooling
            sa_min_temp: 0.00001,            // Lower minimum for fine-tuning
            translation_scale: 0.08,         // Moderate moves
            center_pull_strength: 0.1,       // Strong center pull for compaction
            early_exit_threshold: 5000,      // More patience for global search
            boundary_focus_prob: 0.8,        // Focus on boundary trees
            grid_spacing: 0.5,               // Initial grid spacing
            compaction_passes: 3,            // Final compaction passes
        }
    }
}

/// Track which boundary a tree is blocking
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BoundaryEdge {
    Left,
    Right,
    Top,
    Bottom,
    Corner,
    None,
}

/// Main evolved packer
pub struct EvolvedPacker {
    pub config: EvolvedConfig,
}

impl Default for EvolvedPacker {
    fn default() -> Self {
        Self { config: EvolvedConfig::default() }
    }
}

impl EvolvedPacker {
    /// Pack all n from 1 to max_n
    /// GLOBAL: For each n, place ALL trees at once and optimize globally
    pub fn pack_all(&self, max_n: usize) -> Vec<Packing> {
        let mut rng = rand::thread_rng();
        let mut packings: Vec<Packing> = Vec::with_capacity(max_n);

        for n in 1..=max_n {
            // GLOBAL: Place all n trees at once
            let mut trees = self.initial_global_placement(n, &mut rng);

            // GLOBAL: Run SA to optimize ALL trees simultaneously
            self.global_sa(&mut trees, n, &mut rng);

            // Final compaction passes
            for _ in 0..self.config.compaction_passes {
                self.compaction_pass(&mut trees, &mut rng);
            }

            // Store result
            let mut packing = Packing::new();
            for t in &trees {
                packing.trees.push(t.clone());
            }
            packings.push(packing);
        }

        packings
    }

    /// GLOBAL: Place all n trees at once in a grid pattern
    fn initial_global_placement(&self, n: usize, rng: &mut impl Rng) -> Vec<PlacedTree> {
        if n == 0 {
            return Vec::new();
        }

        if n == 1 {
            return vec![PlacedTree::new(0.0, 0.0, 90.0)];
        }

        let mut trees = Vec::with_capacity(n);

        // Calculate grid dimensions
        let grid_size = ((n as f64).sqrt().ceil()) as usize;
        let spacing = self.config.grid_spacing;

        // Try multiple grid arrangements and pick the best valid one
        let angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0];

        for attempt in 0..50 {
            trees.clear();
            let mut valid = true;

            // Offset the grid slightly on each attempt
            let offset_x = if attempt > 0 { rng.gen_range(-0.1..0.1) } else { 0.0 };
            let offset_y = if attempt > 0 { rng.gen_range(-0.1..0.1) } else { 0.0 };

            // Spacing adjustment based on attempt
            let adj_spacing = spacing * (1.0 + 0.05 * (attempt as f64 / 10.0));

            // Center the grid
            let grid_offset = -(grid_size as f64 * adj_spacing) / 2.0;

            for i in 0..n {
                let row = i / grid_size;
                let col = i % grid_size;

                // Position in grid
                let x = grid_offset + (col as f64) * adj_spacing + offset_x;
                let y = grid_offset + (row as f64) * adj_spacing + offset_y;

                // Choose angle - alternate or random
                let angle = if attempt == 0 {
                    angles[(i * 3) % angles.len()]
                } else {
                    angles[rng.gen_range(0..angles.len())]
                };

                let tree = PlacedTree::new(x, y, angle);

                // Check validity
                let mut overlaps = false;
                for existing in &trees {
                    if tree.overlaps(existing) {
                        overlaps = true;
                        break;
                    }
                }

                if overlaps {
                    valid = false;
                    break;
                }

                trees.push(tree);
            }

            if valid && trees.len() == n {
                return trees;
            }
        }

        // Fallback: place trees one by one with larger spacing
        trees.clear();
        let fallback_spacing = 1.2; // Larger spacing guaranteed to work

        for i in 0..n {
            let row = i / grid_size;
            let col = i % grid_size;
            let grid_offset = -(grid_size as f64 * fallback_spacing) / 2.0;

            let x = grid_offset + (col as f64) * fallback_spacing;
            let y = grid_offset + (row as f64) * fallback_spacing;
            let angle = angles[i % angles.len()];

            trees.push(PlacedTree::new(x, y, angle));
        }

        trees
    }

    /// GLOBAL: Simulated annealing that optimizes ALL trees simultaneously
    fn global_sa(&self, trees: &mut Vec<PlacedTree>, n: usize, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config: Vec<PlacedTree> = trees.clone();

        let mut temp = self.config.sa_initial_temp;

        // Scale iterations with problem size
        let iterations = self.config.sa_iterations + n * 500;

        let mut iterations_without_improvement = 0;

        // Cache boundary info
        let mut boundary_cache_iter = 0;
        let mut boundary_info: Vec<(usize, BoundaryEdge)> = Vec::new();

        for iter in 0..iterations {
            if iterations_without_improvement >= self.config.early_exit_threshold {
                break;
            }

            // Update boundary cache periodically
            if iter == 0 || iter - boundary_cache_iter >= 500 {
                boundary_info = self.find_boundary_trees_with_edges(trees);
                boundary_cache_iter = iter;
            }

            // Select tree to move - prefer boundary trees
            let (idx, edge) = if !boundary_info.is_empty() && rng.gen::<f64>() < self.config.boundary_focus_prob {
                let bi = &boundary_info[rng.gen_range(0..boundary_info.len())];
                (bi.0, bi.1)
            } else {
                (rng.gen_range(0..trees.len()), BoundaryEdge::None)
            };

            let old_tree = trees[idx].clone();

            // Apply move
            let success = self.global_move(trees, idx, temp, edge, rng);

            if success {
                let new_side = compute_side_length(trees);
                let delta = new_side - current_side;

                if delta <= 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                    current_side = new_side;
                    if current_side < best_side {
                        best_side = current_side;
                        best_config = trees.clone();
                        iterations_without_improvement = 0;
                    } else {
                        iterations_without_improvement += 1;
                    }
                } else {
                    trees[idx] = old_tree;
                    iterations_without_improvement += 1;
                }
            } else {
                trees[idx] = old_tree;
                iterations_without_improvement += 1;
            }

            temp = (temp * self.config.sa_cooling_rate).max(self.config.sa_min_temp);
        }

        if best_side < compute_side_length(trees) {
            *trees = best_config;
        }
    }

    /// GLOBAL: Move operator for global optimization
    #[inline]
    fn global_move(
        &self,
        trees: &mut [PlacedTree],
        idx: usize,
        temp: f64,
        edge: BoundaryEdge,
        rng: &mut impl Rng,
    ) -> bool {
        let old = &trees[idx];
        let old_x = old.x;
        let old_y = old.y;
        let old_angle = old.angle_deg;

        let scale = self.config.translation_scale * (0.5 + temp * 2.0);

        // Get current bounds for center calculation
        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
        let bbox_cx = (min_x + max_x) / 2.0;
        let bbox_cy = (min_y + max_y) / 2.0;

        let move_type = match edge {
            BoundaryEdge::Left => rng.gen_range(0..5),
            BoundaryEdge::Right => rng.gen_range(5..10),
            BoundaryEdge::Top => rng.gen_range(10..15),
            BoundaryEdge::Bottom => rng.gen_range(15..20),
            BoundaryEdge::Corner => rng.gen_range(20..25),
            BoundaryEdge::None => rng.gen_range(0..30),
        };

        match move_type % 10 {
            0 => {
                // Move toward center (compaction)
                let dx = (bbox_cx - old_x) * self.config.center_pull_strength * (0.5 + temp);
                let dy = (bbox_cy - old_y) * self.config.center_pull_strength * (0.5 + temp);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            1 => {
                // Move inward from boundary
                let dx = match edge {
                    BoundaryEdge::Left => rng.gen_range(scale * 0.3..scale),
                    BoundaryEdge::Right => rng.gen_range(-scale..-scale * 0.3),
                    _ => rng.gen_range(-scale * 0.3..scale * 0.3),
                };
                let dy = match edge {
                    BoundaryEdge::Top => rng.gen_range(-scale..-scale * 0.3),
                    BoundaryEdge::Bottom => rng.gen_range(scale * 0.3..scale),
                    _ => rng.gen_range(-scale * 0.3..scale * 0.3),
                };
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            2 => {
                // Rotation
                let angles = [45.0, 90.0, -45.0, -90.0, 30.0, -30.0, 15.0, -15.0];
                let delta = angles[rng.gen_range(0..angles.len())];
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            3 => {
                // Small random translation
                let dx = rng.gen_range(-scale * 0.5..scale * 0.5);
                let dy = rng.gen_range(-scale * 0.5..scale * 0.5);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            4 => {
                // Rotate and translate toward center
                let angles = [45.0, 90.0, -45.0, -90.0];
                let delta = angles[rng.gen_range(0..angles.len())];
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                let dx = (bbox_cx - old_x) * 0.05;
                let dy = (bbox_cy - old_y) * 0.05;
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, new_angle);
            }
            5 => {
                // Diagonal move
                let diag = rng.gen_range(-scale..scale);
                let sign = if rng.gen() { 1.0 } else { -1.0 };
                trees[idx] = PlacedTree::new(old_x + diag, old_y + sign * diag, old_angle);
            }
            6 => {
                // Horizontal move only
                let dx = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y, old_angle);
            }
            7 => {
                // Vertical move only
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x, old_y + dy, old_angle);
            }
            8 => {
                // Radial move (toward/away from origin)
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.05 {
                    let delta_r = rng.gen_range(-0.08..0.08) * (1.0 + temp);
                    let new_mag = (mag + delta_r).max(0.0);
                    let scale_r = new_mag / mag;
                    trees[idx] = PlacedTree::new(old_x * scale_r, old_y * scale_r, old_angle);
                } else {
                    return false;
                }
            }
            _ => {
                // Strong center pull for corner trees
                let dx = (bbox_cx - old_x) * self.config.center_pull_strength * 1.5 * (0.5 + temp);
                let dy = (bbox_cy - old_y) * self.config.center_pull_strength * 1.5 * (0.5 + temp);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
        }

        !has_overlap(trees, idx)
    }

    /// Compaction pass: try to move each tree toward center
    fn compaction_pass(&self, trees: &mut [PlacedTree], _rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
        let cx = (min_x + max_x) / 2.0;
        let cy = (min_y + max_y) / 2.0;

        // Try to move each tree toward center
        for idx in 0..trees.len() {
            let old = &trees[idx];
            let old_x = old.x;
            let old_y = old.y;
            let old_angle = old.angle_deg;

            // Try several step sizes
            for &step in &[0.1, 0.05, 0.02, 0.01] {
                let dx = (cx - old_x).signum() * step;
                let dy = (cy - old_y).signum() * step;

                let candidate = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                trees[idx] = candidate;

                if has_overlap(trees, idx) {
                    trees[idx] = PlacedTree::new(old_x, old_y, old_angle);
                } else {
                    // Check if this improved the side length
                    let new_side = compute_side_length(trees);
                    trees[idx] = PlacedTree::new(old_x, old_y, old_angle);
                    let old_side = compute_side_length(trees);

                    if new_side <= old_side {
                        trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                        break;
                    }
                }
            }
        }

        // Also try rotations
        let angles = [45.0, 90.0, -45.0, -90.0];
        for idx in 0..trees.len() {
            let old = &trees[idx];
            let old_x = old.x;
            let old_y = old.y;
            let old_angle = old.angle_deg;
            let old_side = compute_side_length(trees);

            for &delta in &angles {
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);

                if has_overlap(trees, idx) {
                    trees[idx] = PlacedTree::new(old_x, old_y, old_angle);
                } else {
                    let new_side = compute_side_length(trees);
                    if new_side >= old_side {
                        trees[idx] = PlacedTree::new(old_x, old_y, old_angle);
                    } else {
                        break; // Keep the improvement
                    }
                }
            }
        }
    }

    /// Find trees on the bounding box boundary
    #[inline]
    fn find_boundary_trees_with_edges(&self, trees: &[PlacedTree]) -> Vec<(usize, BoundaryEdge)> {
        if trees.is_empty() {
            return Vec::new();
        }

        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
        let eps = 0.02;

        let mut boundary_info: Vec<(usize, BoundaryEdge)> = Vec::new();

        for (i, tree) in trees.iter().enumerate() {
            let (bx1, by1, bx2, by2) = tree.bounds();

            let on_left = (bx1 - min_x).abs() < eps;
            let on_right = (bx2 - max_x).abs() < eps;
            let on_bottom = (by1 - min_y).abs() < eps;
            let on_top = (by2 - max_y).abs() < eps;

            let edge = match (on_left, on_right, on_top, on_bottom) {
                (true, true, _, _) | (_, _, true, true) => BoundaryEdge::Corner,
                (true, _, true, _) | (true, _, _, true) => BoundaryEdge::Corner,
                (_, true, true, _) | (_, true, _, true) => BoundaryEdge::Corner,
                (true, false, false, false) => BoundaryEdge::Left,
                (false, true, false, false) => BoundaryEdge::Right,
                (false, false, true, false) => BoundaryEdge::Top,
                (false, false, false, true) => BoundaryEdge::Bottom,
                _ => continue,
            };

            boundary_info.push((i, edge));
        }

        boundary_info
    }
}

// Helper functions
fn compute_side_length(trees: &[PlacedTree]) -> f64 {
    if trees.is_empty() {
        return 0.0;
    }

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for tree in trees {
        let (bx1, by1, bx2, by2) = tree.bounds();
        min_x = min_x.min(bx1);
        min_y = min_y.min(by1);
        max_x = max_x.max(bx2);
        max_y = max_y.max(by2);
    }

    (max_x - min_x).max(max_y - min_y)
}

fn compute_bounds(trees: &[PlacedTree]) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for tree in trees {
        let (bx1, by1, bx2, by2) = tree.bounds();
        min_x = min_x.min(bx1);
        min_y = min_y.min(by1);
        max_x = max_x.max(bx2);
        max_y = max_y.max(by2);
    }

    (min_x, min_y, max_x, max_y)
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
    fn test_evolved_packer() {
        let packer = EvolvedPacker::default();
        let packings = packer.pack_all(20);

        for (i, p) in packings.iter().enumerate() {
            assert_eq!(p.trees.len(), i + 1);
            assert!(!p.has_overlaps());
        }
    }

    #[test]
    fn test_evolved_score() {
        let packer = EvolvedPacker::default();
        let packings = packer.pack_all(50);
        let score = calculate_score(&packings);
        println!("Evolved score for n=1..50: {:.4}", score);
    }

    #[test]
    fn test_global_initial_placement() {
        let packer = EvolvedPacker::default();
        let mut rng = rand::thread_rng();

        // Test that initial placement creates valid configurations
        for n in 1..=20 {
            let trees = packer.initial_global_placement(n, &mut rng);
            assert_eq!(trees.len(), n, "Should place exactly {} trees", n);

            // Check no overlaps
            for i in 0..trees.len() {
                for j in (i + 1)..trees.len() {
                    assert!(!trees[i].overlaps(&trees[j]),
                            "Trees {} and {} should not overlap for n={}", i, j, n);
                }
            }
        }
    }
}
