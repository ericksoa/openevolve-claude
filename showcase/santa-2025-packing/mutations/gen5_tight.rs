//! Evolved Packing Algorithm - Generation 5 TIGHT PLACEMENT
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
//! MUTATION STRATEGY: TIGHT PLACEMENT (Gen5)
//! Hypothesis: Better initial placement reduces need for expensive SA optimization.
//!
//! Key changes from Gen4:
//! - search_attempts: 400 (increased from 250) with early-out for excellent placements
//! - Binary search precision: 0.0001 (finer than Gen4's 0.001)
//! - Score placements by gap they leave for future trees (forward-looking)
//! - Stronger preference for balanced placements (width ≈ height)
//! - sa_iterations: 8000 (reduced from 20000 - placement is good enough)
//! - Skip SA entirely if initial placement didn't increase bounding box
//! - Early placement exit: Stop searching if placement score < side_length * 1.001
//!
//! Goal: Invest more in placement quality, less in SA refinement
//! Target: Beat Gen4's 98.37 with faster execution due to reduced SA

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

    // Early exit threshold for SA
    pub early_exit_threshold: usize,

    // TIGHT PLACEMENT: Binary search precision
    pub binary_search_precision: f64,

    // TIGHT PLACEMENT: Early placement exit threshold (ratio to current side)
    pub placement_early_exit_ratio: f64,
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen5 TIGHT PLACEMENT: High-quality initial placement, minimal SA
        Self {
            search_attempts: 400,            // Increased from 250 - more placement search
            direction_samples: 64,           // Unchanged
            sa_iterations: 8000,             // Reduced from 20000 - placement is good
            sa_initial_temp: 0.5,            // Lower temp - less exploration needed
            sa_cooling_rate: 0.9998,         // Faster cooling
            sa_min_temp: 0.00001,            // Unchanged
            translation_scale: 0.08,         // Unchanged
            rotation_granularity: 45.0,      // 8 angles
            center_pull_strength: 0.06,      // Unchanged
            sa_passes: 2,                    // Keep 2 passes
            early_exit_threshold: 800,       // Shorter SA runs
            binary_search_precision: 0.0001, // TIGHT: Finer precision (was 0.001)
            placement_early_exit_ratio: 1.001, // Exit if placement is near-optimal
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

            // TIGHT PLACEMENT: Compute current side before placement
            let side_before = compute_side_length(&trees);

            // Place new tree using evolved heuristics
            let new_tree = self.find_placement(&trees, n, max_n, &mut rng);
            trees.push(new_tree);

            // TIGHT PLACEMENT: Check if bounding box increased
            let side_after = compute_side_length(&trees);
            let side_increased = side_after > side_before + 0.0001;

            // TIGHT PLACEMENT: Skip SA if placement didn't increase bounding box
            if side_increased {
                // Run SA passes only when needed
                for pass in 0..self.config.sa_passes {
                    self.local_search(&mut trees, n, pass, &mut rng);
                }
            }
            // If side didn't increase, placement was perfect - no SA needed

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
    /// TIGHT PLACEMENT: More attempts, finer precision, early exit for good placements
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

        // TIGHT PLACEMENT: Compute current side for early exit check
        let current_side = compute_side_length(existing);
        let early_exit_threshold = current_side * self.config.placement_early_exit_ratio;

        // TIGHT PLACEMENT: Track attempts for early exit
        let mut good_placements_found = 0;

        for attempt in 0..self.config.search_attempts {
            let dir = self.select_direction(n, rng);
            let vx = dir.cos();
            let vy = dir.sin();

            for &tree_angle in &angles {
                // Binary search for closest valid position
                // TIGHT PLACEMENT: Finer precision 0.0001
                let mut low = 0.0;
                let mut high = 12.0;

                while high - low > self.config.binary_search_precision {
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

                        // TIGHT PLACEMENT: Early exit if placement is near-optimal
                        if best_score < early_exit_threshold {
                            good_placements_found += 1;
                            // Exit after finding 3 good placements (ensure we have options)
                            if good_placements_found >= 3 && attempt > 100 {
                                return best_tree;
                            }
                        }
                    }
                }
            }
        }

        best_tree
    }

    /// EVOLVED FUNCTION: Score a placement (lower is better)
    /// TIGHT PLACEMENT: Forward-looking scoring that considers future placements
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

        // TIGHT PLACEMENT: Strong preference for balanced aspect ratio
        // A balanced packing leaves room for future trees in all directions
        let aspect_diff = (width - height).abs();
        let balance_penalty = aspect_diff * 0.25; // Increased from 0.15

        // TIGHT PLACEMENT: Compute gap efficiency - how much unused space
        // Lower gap = tighter packing = better
        let area = width * height;
        let tree_area_estimate = (n + 1) as f64 * 0.35; // Approximate area per tree
        let gap_ratio = if area > tree_area_estimate {
            (area - tree_area_estimate) / area
        } else {
            0.0
        };
        let gap_penalty = gap_ratio * 0.1;

        // TIGHT PLACEMENT: Prefer placements that leave space for future trees
        // Check distance to nearest edge of bounding box - we want balanced margins
        let margin_left = min_x - pack_min_x;
        let margin_right = pack_max_x - max_x;
        let margin_bottom = min_y - pack_min_y;
        let margin_top = pack_max_y - max_y;

        // Penalize unbalanced margins (tree pushed to one side)
        let h_margin_diff = (margin_left - margin_right).abs();
        let v_margin_diff = (margin_top - margin_bottom).abs();
        let margin_penalty = (h_margin_diff + v_margin_diff) * 0.03;

        // Tertiary: slight preference for compact center
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.006 / (n as f64).sqrt();

        // TIGHT PLACEMENT: Bonus for not increasing the side length
        let old_side = compute_side_length_from_bounds(existing);
        let side_increase_penalty = if side > old_side {
            (side - old_side) * 0.5 // Extra penalty for growing the box
        } else {
            -0.02 // Small bonus for fitting without growth
        };

        side_score + balance_penalty + gap_penalty + margin_penalty + center_penalty + side_increase_penalty
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    /// Returns angles in priority order
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
        // 8 directions with n-dependent priority
        let base = match n % 4 {
            0 => vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0],
            1 => vec![90.0, 270.0, 0.0, 180.0, 135.0, 315.0, 45.0, 225.0],
            2 => vec![180.0, 0.0, 270.0, 90.0, 225.0, 45.0, 315.0, 135.0],
            _ => vec![270.0, 90.0, 180.0, 0.0, 315.0, 135.0, 225.0, 45.0],
        };
        base
    }

    /// EVOLVED FUNCTION: Select direction angle for placement search
    #[inline]
    fn select_direction(&self, n: usize, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;

        // Three-way mix of direction strategies
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
    /// TIGHT PLACEMENT: Shorter SA since placement is high quality
    fn local_search(&self, trees: &mut Vec<PlacedTree>, n: usize, pass: usize, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config: Vec<PlacedTree> = trees.clone();

        // Adjust temperature based on pass number
        let temp_multiplier = match pass {
            0 => 1.0,
            _ => 0.4,  // Second pass: lower temp
        };
        let mut temp = self.config.sa_initial_temp * temp_multiplier;

        // TIGHT PLACEMENT: Reduced iterations since placement is good
        let base_iterations = match pass {
            0 => self.config.sa_iterations + n * 80,
            _ => self.config.sa_iterations / 2 + n * 40,
        };

        // Track iterations without improvement for early exit
        let mut iterations_without_improvement = 0;

        // Pre-compute boundary trees once per batch
        let mut boundary_cache_iter = 0;
        let mut boundary_indices: Vec<usize> = Vec::new();

        for iter in 0..base_iterations {
            // Early exit when no improvement for threshold iterations
            if iterations_without_improvement >= self.config.early_exit_threshold {
                break;
            }

            // Update boundary cache every 400 iterations
            if iter == 0 || iter - boundary_cache_iter >= 400 {
                boundary_indices = self.find_boundary_trees(trees);
                boundary_cache_iter = iter;
            }

            // 70% chance to pick boundary tree, 30% random
            let idx = if !boundary_indices.is_empty() && rng.gen::<f64>() < 0.70 {
                boundary_indices[rng.gen_range(0..boundary_indices.len())]
            } else {
                rng.gen_range(0..trees.len())
            };

            let old_tree = trees[idx].clone();

            // Move operator selection with bounding box focus
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

    /// Find trees on the bounding box boundary
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
    /// 8 move types focused on bounding box reduction
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

        // 8 move types with special boundary moves
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
                // Inward translation (toward reducing bounding box)
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

/// Compute side length from bounds (for existing trees only)
fn compute_side_length_from_bounds(trees: &[PlacedTree]) -> f64 {
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
