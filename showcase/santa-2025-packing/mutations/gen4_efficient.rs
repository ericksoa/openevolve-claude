//! Evolved Packing Algorithm - Generation 4 EFFICIENT
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
//! MUTATION STRATEGY: SIMPLER + FASTER (Gen4)
//! Hypothesis: Extreme parameters in Gen3 cause overfitting. Try reducing computation:
//!
//! Parameter changes from Gen3:
//! - sa_iterations: 20000 + n*100 (50% of Gen3's 40000 + n*200)
//! - sa_passes: 2 (reduced from Gen3's 3 passes)
//! - search_attempts: 250 (reduced from Gen3's 400)
//! - direction_samples: 64 (reduced from Gen3's 96)
//! - Early exit: Stop SA when no improvement for 1000 iterations (NEW)
//! - Boundary-focused: Only move trees that contribute to bounding box (NEW)
//! - Smarter moves: Direct bounding box reduction moves (NEW)
//! - Removed greedy compaction phase (unnecessary with smarter moves)
//! - 8 rotation angles (every 45 degrees) instead of 16
//!
//! Goal: Faster execution allows more exploration; avoid overfitting
//! Target: Match or beat Gen3's 101.90 with faster execution

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

    // EFFICIENT: Early exit threshold
    pub early_exit_threshold: usize,
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen4 EFFICIENT: Simpler, faster configuration
        Self {
            search_attempts: 250,            // Was 400 (Gen3) - 37.5% reduction
            direction_samples: 64,           // Was 96 (Gen3) - 33% reduction
            sa_iterations: 20000,            // Was 40000 (Gen3) - 50% reduction
            sa_initial_temp: 0.6,            // Was 0.7 (Gen3) - slightly lower
            sa_cooling_rate: 0.9999,         // Was 0.99998 (Gen3) - faster cooling
            sa_min_temp: 0.00001,            // Was 0.000001 (Gen3) - higher minimum
            translation_scale: 0.08,         // Unchanged
            rotation_granularity: 45.0,      // 8 angles (every 45 degrees) vs 22.5 in Gen3
            center_pull_strength: 0.06,      // Unchanged
            sa_passes: 2,                    // Was 3 (Gen3) - reduced passes
            early_exit_threshold: 1000,      // NEW: exit if no improvement for this many iterations
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

            // EFFICIENT: Run double SA passes (reduced from triple)
            for pass in 0..self.config.sa_passes {
                self.local_search(&mut trees, n, pass, &mut rng);
            }

            // NOTE: Removed greedy compaction phase - smarter SA moves should handle this

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
                // EFFICIENT: Use coarser precision 0.001 (was 0.0001 in Gen3)
                let mut low = 0.0;
                let mut high = 12.0;

                while high - low > 0.001 {
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
    /// EFFICIENT: Simplified scoring function
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

        // Primary: minimize side length (most important)
        let side_score = side;

        // Secondary: prefer balanced aspect ratio
        // EFFICIENT: Simpler, constant weight
        let balance_penalty = (width - height).abs() * 0.15;

        // Tertiary: slight preference for compact center (scaled by n)
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.008 / (n as f64).sqrt();

        // EFFICIENT: Simplified density bonus
        let area = width * height;
        let density_bonus = if area > 0.0 {
            -0.01 * (n as f64 / area).min(2.0)
        } else {
            0.0
        };

        side_score + balance_penalty + center_penalty + density_bonus
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    /// Returns angles in priority order
    /// EFFICIENT: 8 angles (every 45 degrees) for speed
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
        // EFFICIENT: Use 8 directions with n-dependent priority
        let base = match n % 4 {
            0 => vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0],
            1 => vec![90.0, 270.0, 0.0, 180.0, 135.0, 315.0, 45.0, 225.0],
            2 => vec![180.0, 0.0, 270.0, 90.0, 225.0, 45.0, 315.0, 135.0],
            _ => vec![270.0, 90.0, 180.0, 0.0, 315.0, 135.0, 225.0, 45.0],
        };
        base
    }

    /// EVOLVED FUNCTION: Select direction angle for placement search
    /// EFFICIENT: Simpler direction selection with 64 samples
    #[inline]
    fn select_direction(&self, n: usize, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;

        // EFFICIENT: Three-way mix of direction strategies (simplified from four)
        let strategy = rng.gen::<f64>();

        if strategy < 0.50 {
            // Structured: evenly spaced with small jitter
            let base_idx = rng.gen_range(0..num_dirs);
            let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
            base + rng.gen_range(-0.06..0.06)
        } else if strategy < 0.75 {
            // Weighted random: favor corners and edges
            loop {
                let angle = rng.gen_range(0.0..2.0 * PI);
                // Favor 45-degree increments
                let corner_weight = ((4.0 * angle).sin().abs() + (4.0 * angle).cos().abs()) / 2.0;
                let threshold = 0.2;
                if rng.gen::<f64>() < corner_weight.max(threshold) {
                    return angle;
                }
            }
        } else {
            // Golden angle spiral for good coverage
            let golden_angle = PI * (3.0 - (5.0_f64).sqrt());  // ~137.5 degrees
            let base = (n as f64 * golden_angle) % (2.0 * PI);
            let offset = rng.gen_range(0..8) as f64 * PI / 4.0;  // 8 offsets
            (base + offset + rng.gen_range(-0.1..0.1)) % (2.0 * PI)
        }
    }

    /// EVOLVED FUNCTION: Local search with simulated annealing
    /// EFFICIENT: Shorter search with early exit when stuck
    fn local_search(&self, trees: &mut Vec<PlacedTree>, n: usize, pass: usize, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config: Vec<PlacedTree> = trees.clone();

        // EFFICIENT: Adjust temperature based on pass number
        let temp_multiplier = match pass {
            0 => 1.0,
            _ => 0.4,  // Second pass: lower temp
        };
        let mut temp = self.config.sa_initial_temp * temp_multiplier;

        // EFFICIENT: 50% fewer iterations: 20000 + n*100
        let base_iterations = match pass {
            0 => self.config.sa_iterations + n * 100,
            _ => self.config.sa_iterations / 2 + n * 50,  // Second pass
        };

        // EFFICIENT: Track iterations without improvement for early exit
        let mut iterations_without_improvement = 0;

        // EFFICIENT: Pre-compute boundary trees once per batch
        let mut boundary_cache_iter = 0;
        let mut boundary_indices: Vec<usize> = Vec::new();

        for iter in 0..base_iterations {
            // EFFICIENT: Early exit when no improvement for threshold iterations
            if iterations_without_improvement >= self.config.early_exit_threshold {
                break;
            }

            // EFFICIENT: Update boundary cache every 500 iterations
            if iter == 0 || iter - boundary_cache_iter >= 500 {
                boundary_indices = self.find_boundary_trees(trees);
                boundary_cache_iter = iter;
            }

            // EFFICIENT: 70% chance to pick boundary tree, 30% random
            let idx = if !boundary_indices.is_empty() && rng.gen::<f64>() < 0.70 {
                boundary_indices[rng.gen_range(0..boundary_indices.len())]
            } else {
                rng.gen_range(0..trees.len())
            };

            let old_tree = trees[idx].clone();

            // EVOLVED: Move operator selection with bounding box focus
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

    /// EFFICIENT: Find trees on the bounding box boundary
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
    /// EFFICIENT: 8 move types focused on bounding box reduction
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

        // EFFICIENT: 8 move types (reduced from 12), with special boundary moves
        let move_type = if is_boundary {
            // Boundary trees: prefer inward moves
            match rng.gen_range(0..10) {
                0..=3 => 0,  // Inward translation (40%)
                4..=5 => 1,  // Rotation (20%)
                6..=7 => 2,  // Center pull (20%)
                8 => 3,      // Small nudge (10%)
                _ => 4,      // Translate + rotate (10%)
            }
        } else {
            rng.gen_range(0..8)
        };

        match move_type {
            0 => {
                // EFFICIENT: Inward translation (toward reducing bounding box)
                let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                let (bx1, by1, bx2, by2) = trees[idx].bounds();

                // Find which boundary we're on and move inward
                let scale = self.config.translation_scale * (0.2 + temp * 2.0);
                let mut dx = rng.gen_range(-scale * 0.3..scale * 0.3);
                let mut dy = rng.gen_range(-scale * 0.3..scale * 0.3);

                // Bias toward moving inward from boundary
                if (bx1 - min_x).abs() < 0.02 { dx += scale * 0.5; }
                if (bx2 - max_x).abs() < 0.02 { dx -= scale * 0.5; }
                if (by1 - min_y).abs() < 0.02 { dy += scale * 0.5; }
                if (by2 - max_y).abs() < 0.02 { dy -= scale * 0.5; }

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
                if mag > 0.04 {
                    let scale = self.config.center_pull_strength * (0.4 + temp * 1.5);
                    let dx = -old_x / mag * scale;
                    let dy = -old_y / mag * scale;
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                } else {
                    return false;
                }
            }
            3 => {
                // Small nudge for fine-tuning
                let scale = 0.015 * (0.5 + temp);
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            4 => {
                // Translate + rotate combo
                let scale = self.config.translation_scale * 0.4;
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                let dangle = rng.gen_range(-45.0..45.0);
                let new_angle = (old_angle + dangle).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, new_angle);
            }
            5 => {
                // Fine rotation (45 degrees)
                let delta = if rng.gen() { self.config.rotation_granularity }
                            else { -self.config.rotation_granularity };
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            6 => {
                // Polar move (radial in/out)
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.08 {
                    let delta_r = rng.gen_range(-0.08..0.08) * (1.0 + temp);
                    let new_mag = (mag + delta_r).max(0.0);
                    let scale = new_mag / mag;
                    trees[idx] = PlacedTree::new(old_x * scale, old_y * scale, old_angle);
                } else {
                    return false;
                }
            }
            _ => {
                // Angular orbit (move around center)
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.08 {
                    let current_angle = old_y.atan2(old_x);
                    let delta_angle = rng.gen_range(-0.2..0.2) * (1.0 + temp);
                    let new_ang = current_angle + delta_angle;
                    trees[idx] = PlacedTree::new(mag * new_ang.cos(), mag * new_ang.sin(), old_angle);
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
