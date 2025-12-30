//! Evolved Packing Algorithm - Generation 5 SIMPLER
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
//! MUTATION STRATEGY: EVEN SIMPLER (Gen5)
//! Hypothesis: Gen4 showed that simpler is better. Push simplification further:
//!
//! Parameter changes from Gen4:
//! - sa_iterations: 10000 + n*50 (50% of Gen4's 20000 + n*100)
//! - sa_passes: 1 (down from Gen4's 2 passes)
//! - search_attempts: 150 (down from Gen4's 250)
//! - direction_samples: 48 (down from Gen4's 64)
//! - Binary search precision: 0.002 (less precise, faster than Gen4's 0.001)
//! - Early exit: 500 iterations (more aggressive than Gen4's 1000)
//! - 4 rotation angles only (0, 90, 180, 270) instead of 8
//! - Simplified scoring function
//!
//! Goal: Maximum speed through aggressive simplification
//! Target: Match or beat Gen4's 98.37 with much faster execution

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Evolved packing configuration
/// These parameters are tuned through evolution
pub struct EvolvedConfig {
    // Search parameters
    pub search_attempts: usize,
    pub direction_samples: usize,

    // Simulated annealing
    pub sa_iterations: usize,
    pub sa_initial_temp: f64,
    pub sa_cooling_rate: f64,
    pub sa_min_temp: f64,

    // Move parameters
    pub translation_scale: f64,
    pub rotation_granularity: f64,
    pub center_pull_strength: f64,

    // Multi-pass settings
    pub sa_passes: usize,

    // SIMPLER: More aggressive early exit threshold
    pub early_exit_threshold: usize,
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen5 SIMPLER: Even more aggressive simplification
        Self {
            search_attempts: 150,            // Was 250 (Gen4) - 40% reduction
            direction_samples: 48,           // Was 64 (Gen4) - 25% reduction
            sa_iterations: 10000,            // Was 20000 (Gen4) - 50% reduction
            sa_initial_temp: 0.5,            // Was 0.6 (Gen4) - slightly lower
            sa_cooling_rate: 0.9998,         // Was 0.9999 (Gen4) - faster cooling
            sa_min_temp: 0.0001,             // Was 0.00001 (Gen4) - higher minimum
            translation_scale: 0.08,         // Unchanged
            rotation_granularity: 90.0,      // 4 angles (every 90 degrees) vs 45 in Gen4
            center_pull_strength: 0.06,      // Unchanged
            sa_passes: 1,                    // Was 2 (Gen4) - single pass only
            early_exit_threshold: 500,       // Was 1000 (Gen4) - more aggressive exit
        }
    }
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
    pub fn pack_all(&self, max_n: usize) -> Vec<Packing> {
        let mut rng = rand::thread_rng();
        let mut packings: Vec<Packing> = Vec::with_capacity(max_n);
        let mut prev_trees: Vec<PlacedTree> = Vec::new();

        for n in 1..=max_n {
            let mut trees = prev_trees.clone();

            // Place new tree using evolved heuristics
            let new_tree = self.find_placement(&trees, n, max_n, &mut rng);
            trees.push(new_tree);

            // SIMPLER: Single SA pass only
            for pass in 0..self.config.sa_passes {
                self.local_search(&mut trees, n, pass, &mut rng);
            }

            // Store result
            let mut packing = Packing::new();
            for t in &trees {
                packing.trees.push(t.clone());
            }
            packings.push(packing);
            prev_trees = trees;
        }

        packings
    }

    /// EVOLVED FUNCTION: Find best placement for new tree
    /// This function is a primary evolution target
    fn find_placement(
        &self,
        existing: &[PlacedTree],
        n: usize,
        _max_n: usize,
        rng: &mut impl Rng,
    ) -> PlacedTree {
        if existing.is_empty() {
            // First tree: place at origin with optimal rotation
            return PlacedTree::new(0.0, 0.0, 90.0);
        }

        let mut best_tree = PlacedTree::new(0.0, 0.0, 90.0);
        let mut best_score = f64::INFINITY;

        let angles = self.select_angles(n);

        for _ in 0..self.config.search_attempts {
            let dir = self.select_direction(n, rng);
            let vx = dir.cos();
            let vy = dir.sin();

            for &tree_angle in &angles {
                // Binary search for closest valid position
                // SIMPLER: Use coarser precision 0.002 (was 0.001 in Gen4)
                let mut low = 0.0;
                let mut high = 12.0;

                while high - low > 0.002 {
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
                    let score = self.placement_score(&candidate, existing, n);
                    if score < best_score {
                        best_score = score;
                        best_tree = candidate;
                    }
                }
            }
        }

        best_tree
    }

    /// EVOLVED FUNCTION: Score a placement (lower is better)
    /// Key evolution target - determines placement quality
    /// SIMPLER: Minimal scoring function
    #[inline]
    fn placement_score(&self, tree: &PlacedTree, existing: &[PlacedTree], n: usize) -> f64 {
        let (min_x, min_y, max_x, max_y) = tree.bounds();

        // Compute combined bounds
        let mut pack_min_x = min_x;
        let mut pack_min_y = min_y;
        let mut pack_max_x = max_x;
        let mut pack_max_y = max_y;

        for t in existing {
            let (bx1, by1, bx2, by2) = t.bounds();
            pack_min_x = pack_min_x.min(bx1);
            pack_min_y = pack_min_y.min(by1);
            pack_max_x = pack_max_x.max(bx2);
            pack_max_y = pack_max_y.max(by2);
        }

        let width = pack_max_x - pack_min_x;
        let height = pack_max_y - pack_min_y;
        let side = width.max(height);

        // SIMPLER: Just three scoring components
        // Primary: minimize side length (most important)
        let side_score = side;

        // Secondary: prefer balanced aspect ratio (simplified)
        let balance_penalty = (width - height).abs() * 0.12;

        // Tertiary: slight center preference (very simple)
        let center_penalty = (tree.x.abs() + tree.y.abs()) * 0.005 / (n as f64 + 1.0).sqrt();

        side_score + balance_penalty + center_penalty
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    /// Returns angles in priority order
    /// SIMPLER: 4 angles only (0, 90, 180, 270)
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
        // SIMPLER: Only 4 cardinal directions
        match n % 4 {
            0 => vec![0.0, 90.0, 180.0, 270.0],
            1 => vec![90.0, 270.0, 0.0, 180.0],
            2 => vec![180.0, 0.0, 270.0, 90.0],
            _ => vec![270.0, 90.0, 180.0, 0.0],
        }
    }

    /// EVOLVED FUNCTION: Select direction angle for placement search
    /// SIMPLER: Two-way mix of direction strategies
    #[inline]
    fn select_direction(&self, n: usize, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;

        // SIMPLER: Two-way mix (structured vs random)
        if rng.gen::<f64>() < 0.65 {
            // Structured: evenly spaced with small jitter
            let base_idx = rng.gen_range(0..num_dirs);
            let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
            base + rng.gen_range(-0.08..0.08)
        } else {
            // Golden angle spiral for coverage
            let golden_angle = PI * (3.0 - (5.0_f64).sqrt());
            let base = (n as f64 * golden_angle) % (2.0 * PI);
            let offset = rng.gen_range(0..6) as f64 * PI / 3.0;
            (base + offset + rng.gen_range(-0.12..0.12)) % (2.0 * PI)
        }
    }

    /// EVOLVED FUNCTION: Local search with simulated annealing
    /// SIMPLER: Single pass with aggressive early exit
    fn local_search(&self, trees: &mut Vec<PlacedTree>, n: usize, _pass: usize, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config: Vec<PlacedTree> = trees.clone();

        let mut temp = self.config.sa_initial_temp;

        // SIMPLER: 10000 + n*50 iterations
        let base_iterations = self.config.sa_iterations + n * 50;

        // SIMPLER: Track iterations without improvement for early exit
        let mut iterations_without_improvement = 0;

        // SIMPLER: Pre-compute boundary trees once per batch
        let mut boundary_cache_iter = 0;
        let mut boundary_indices: Vec<usize> = Vec::new();

        for iter in 0..base_iterations {
            // SIMPLER: Early exit when no improvement for 500 iterations
            if iterations_without_improvement >= self.config.early_exit_threshold {
                break;
            }

            // Update boundary cache every 400 iterations
            if iter == 0 || iter - boundary_cache_iter >= 400 {
                boundary_indices = self.find_boundary_trees(trees);
                boundary_cache_iter = iter;
            }

            // SIMPLER: 75% chance to pick boundary tree, 25% random
            let idx = if !boundary_indices.is_empty() && rng.gen::<f64>() < 0.75 {
                boundary_indices[rng.gen_range(0..boundary_indices.len())]
            } else {
                rng.gen_range(0..trees.len())
            };

            let old_tree = trees[idx].clone();

            // EVOLVED: Move operator selection
            let success = self.sa_move(trees, idx, temp, &boundary_indices, rng);

            if success {
                let new_side = compute_side_length(trees);
                let delta = new_side - current_side;

                // Metropolis criterion
                if delta <= 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                    current_side = new_side;
                    // Track best solution found
                    if current_side < best_side {
                        best_side = current_side;
                        best_config = trees.clone();
                        iterations_without_improvement = 0;  // Reset counter
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

        // Restore best configuration found during search
        if best_side < compute_side_length(trees) {
            *trees = best_config;
        }
    }

    /// SIMPLER: Find trees on the bounding box boundary
    #[inline]
    fn find_boundary_trees(&self, trees: &[PlacedTree]) -> Vec<usize> {
        if trees.is_empty() {
            return Vec::new();
        }

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for tree in trees.iter() {
            let (bx1, by1, bx2, by2) = tree.bounds();
            min_x = min_x.min(bx1);
            min_y = min_y.min(by1);
            max_x = max_x.max(bx2);
            max_y = max_y.max(by2);
        }

        let mut boundary_indices: Vec<usize> = Vec::new();
        let eps = 0.01;

        for (i, tree) in trees.iter().enumerate() {
            let (bx1, by1, bx2, by2) = tree.bounds();
            if (bx1 - min_x).abs() < eps || (bx2 - max_x).abs() < eps ||
               (by1 - min_y).abs() < eps || (by2 - max_y).abs() < eps {
                boundary_indices.push(i);
            }
        }

        boundary_indices
    }

    /// EVOLVED FUNCTION: SA move operator
    /// SIMPLER: 6 move types (reduced from 8)
    /// Returns true if move is valid (no overlap)
    #[inline]
    fn sa_move(
        &self,
        trees: &mut [PlacedTree],
        idx: usize,
        temp: f64,
        boundary_indices: &[usize],
        rng: &mut impl Rng,
    ) -> bool {
        let old = &trees[idx];
        let old_x = old.x;
        let old_y = old.y;
        let old_angle = old.angle_deg;

        let is_boundary = boundary_indices.contains(&idx);

        // SIMPLER: 6 move types with simpler distribution
        let move_type = if is_boundary {
            // Boundary trees: prefer inward moves
            match rng.gen_range(0..6) {
                0..=2 => 0,  // Inward translation (50%)
                3 => 1,      // Rotation (16.7%)
                4 => 2,      // Center pull (16.7%)
                _ => 3,      // Small nudge (16.7%)
            }
        } else {
            rng.gen_range(0..6)
        };

        match move_type {
            0 => {
                // SIMPLER: Inward translation (toward reducing bounding box)
                let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                let (bx1, by1, bx2, by2) = trees[idx].bounds();

                // Find which boundary we're on and move inward
                let scale = self.config.translation_scale * (0.3 + temp * 1.5);
                let mut dx = rng.gen_range(-scale * 0.3..scale * 0.3);
                let mut dy = rng.gen_range(-scale * 0.3..scale * 0.3);

                // Bias toward moving inward from boundary
                if (bx1 - min_x).abs() < 0.02 { dx += scale * 0.6; }
                if (bx2 - max_x).abs() < 0.02 { dx -= scale * 0.6; }
                if (by1 - min_y).abs() < 0.02 { dy += scale * 0.6; }
                if (by2 - max_y).abs() < 0.02 { dy -= scale * 0.6; }

                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            1 => {
                // 90-degree rotation
                let new_angle = (old_angle + 90.0).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            2 => {
                // Move toward center
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.05 {
                    let scale = self.config.center_pull_strength * (0.5 + temp * 1.2);
                    let dx = -old_x / mag * scale;
                    let dy = -old_y / mag * scale;
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                } else {
                    return false;
                }
            }
            3 => {
                // Small nudge for fine-tuning
                let scale = 0.02 * (0.5 + temp);
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            4 => {
                // Translate + rotate combo
                let scale = self.config.translation_scale * 0.5;
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                let new_angle = (old_angle + 90.0).rem_euclid(360.0);  // Only 90-degree rotations
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, new_angle);
            }
            _ => {
                // Polar move (radial in/out)
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.08 {
                    let delta_r = rng.gen_range(-0.06..0.06) * (1.0 + temp);
                    let new_mag = (mag + delta_r).max(0.0);
                    let scale = new_mag / mag;
                    trees[idx] = PlacedTree::new(old_x * scale, old_y * scale, old_angle);
                } else {
                    return false;
                }
            }
        }

        !has_overlap(trees, idx)
    }
}

// Helper functions
fn is_valid(tree: &PlacedTree, existing: &[PlacedTree]) -> bool {
    for other in existing {
        if tree.overlaps(other) {
            return false;
        }
    }
    true
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
}
