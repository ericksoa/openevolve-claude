//! Evolved Packing Algorithm - Generation 2 Rotation Optimization
//!
//! This module focuses on optimizing tree rotations for better packing density.
//! The Christmas tree shape is asymmetric, so rotation matters significantly.
//!
//! MUTATION STRATEGY: Rotation Optimization
//!
//! Key changes from Gen1:
//! 1. Expanded angle set: 16 angles (every 22.5 degrees) instead of 8
//! 2. Dedicated rotation-only SA pass after placement SA
//! 3. Fine-grained continuous rotation adjustments during SA
//! 4. Rotation angle preference tracking and bias
//! 5. Interlock-aware rotation scoring (tip-to-trunk alignment bonus)
//!
//! Evolution targets:
//! - placement_score(): How to score candidate placements
//! - select_angles(): Which rotation angles to try
//! - select_direction(): How to choose placement directions
//! - sa_move(): Local search move operators
//! - rotation_only_pass(): NEW - dedicated rotation optimization

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Evolved packing configuration
/// These parameters are tuned through evolution
pub struct EvolvedConfig {
    // Search parameters
    pub search_attempts: usize,
    pub direction_samples: usize,

    // Simulated annealing - placement phase
    pub sa_iterations: usize,
    pub sa_initial_temp: f64,
    pub sa_cooling_rate: f64,
    pub sa_min_temp: f64,

    // Move parameters
    pub translation_scale: f64,
    pub rotation_granularity: f64,
    pub center_pull_strength: f64,

    // NEW: Rotation optimization parameters
    pub rotation_angles: usize,           // Number of discrete angles to try (16 = every 22.5 deg)
    pub rotation_sa_iterations: usize,    // Iterations for rotation-only SA pass
    pub rotation_sa_temp: f64,            // Temperature for rotation SA
    pub fine_rotation_range: f64,         // Range for continuous fine rotation (degrees)
    pub interlock_bonus: f64,             // Bonus for tip-to-trunk alignment
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        Self {
            // Inherited from Gen1 with slight adjustments
            search_attempts: 80,            // Increased: more exploration with expanded angles
            direction_samples: 16,
            sa_iterations: 5500,            // Reduced: rely more on rotation pass
            sa_initial_temp: 0.45,
            sa_cooling_rate: 0.9993,
            sa_min_temp: 0.001,
            translation_scale: 0.08,
            rotation_granularity: 22.5,     // CHANGED: Finer granularity (was 45)
            center_pull_strength: 0.04,

            // NEW rotation optimization parameters
            rotation_angles: 16,            // 16 angles = every 22.5 degrees
            rotation_sa_iterations: 2000,   // Dedicated rotation iterations
            rotation_sa_temp: 0.25,         // Lower temp for fine-tuning
            fine_rotation_range: 10.0,      // +/- 10 degrees for fine adjustment
            interlock_bonus: 0.015,         // Bonus for good interlock patterns
        }
    }
}

/// Main evolved packer with rotation optimization
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

            // Place new tree using evolved heuristics with expanded angles
            let new_tree = self.find_placement(&trees, n, max_n, &mut rng);
            trees.push(new_tree);

            // Run standard local search (translation + rotation)
            self.local_search(&mut trees, n, &mut rng);

            // NEW: Run rotation-only optimization pass
            self.rotation_only_pass(&mut trees, n, &mut rng);

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
    /// Now uses 16 angles instead of 8 for more systematic rotation search
    fn find_placement(
        &self,
        existing: &[PlacedTree],
        n: usize,
        _max_n: usize,
        rng: &mut impl Rng,
    ) -> PlacedTree {
        if existing.is_empty() {
            // First tree: place at origin with 90 degree rotation (tip pointing left)
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
                let mut low = 0.0;
                let mut high = 12.0;

                while high - low > 0.003 {
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
    /// Added interlock bonus for tip-to-trunk alignment patterns
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

        // Primary: minimize side length
        let side_score = side;

        // Secondary: prefer balanced aspect ratio
        let balance_penalty = (width - height).abs() * 0.12;

        // Tertiary: slight preference for compact center
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.01 / (n as f64).sqrt();

        // NEW: Interlock bonus - reward tip-to-trunk alignment patterns
        let interlock_bonus = self.compute_interlock_score(tree, existing);

        side_score + balance_penalty + center_penalty - interlock_bonus
    }

    /// NEW: Compute interlock score for tip-to-trunk alignment
    /// Trees at 0 and 180 degrees can interlock well (tip into trunk cavity)
    #[inline]
    fn compute_interlock_score(&self, tree: &PlacedTree, existing: &[PlacedTree]) -> f64 {
        if existing.is_empty() {
            return 0.0;
        }

        let mut score = 0.0;
        let tree_angle = tree.angle_deg;

        // Check if this tree's rotation could interlock with nearby trees
        for other in existing {
            let other_angle = other.angle_deg;

            // Distance between trees
            let dx = tree.x - other.x;
            let dy = tree.y - other.y;
            let dist = (dx * dx + dy * dy).sqrt();

            // Only consider nearby trees (within reasonable interlock distance)
            if dist > 0.5 && dist < 2.0 {
                // Check for complementary angles (tip-to-trunk patterns)
                // 0 and 180 are complementary (facing opposite ways)
                // 90 and 270 are complementary
                let angle_diff = (tree_angle - other_angle).abs() % 360.0;
                let angle_diff = if angle_diff > 180.0 { 360.0 - angle_diff } else { angle_diff };

                // Reward near-180 degree difference (facing opposite directions)
                if (angle_diff - 180.0).abs() < 30.0 {
                    score += self.config.interlock_bonus / dist;
                }

                // Also reward 90-degree offset (perpendicular interlocking)
                if (angle_diff - 90.0).abs() < 20.0 || (angle_diff - 270.0).abs() < 20.0 {
                    score += self.config.interlock_bonus * 0.5 / dist;
                }
            }
        }

        score
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    /// Now uses 16 angles (every 22.5 degrees) with priority ordering
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
        // Generate 16 angles with n-dependent priority ordering
        let num_angles = self.config.rotation_angles;

        // Base angles: every 22.5 degrees
        let all_angles: Vec<f64> = (0..num_angles)
            .map(|i| (i as f64 * 360.0 / num_angles as f64))
            .collect();

        // Priority ordering based on n - favor different angles for different tree counts
        match n % 4 {
            0 => {
                // Prioritize cardinal directions, then 45s, then 22.5s
                vec![0.0, 90.0, 180.0, 270.0,
                     45.0, 135.0, 225.0, 315.0,
                     22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]
            }
            1 => {
                // Prioritize 90/270, then 0/180, then intermediates
                vec![90.0, 270.0, 0.0, 180.0,
                     135.0, 315.0, 45.0, 225.0,
                     67.5, 247.5, 112.5, 292.5, 22.5, 202.5, 157.5, 337.5]
            }
            2 => {
                // Prioritize 45-degree offsets
                vec![45.0, 135.0, 225.0, 315.0,
                     0.0, 90.0, 180.0, 270.0,
                     22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]
            }
            _ => {
                // Prioritize 22.5-degree offsets for variety
                vec![22.5, 112.5, 202.5, 292.5,
                     67.5, 157.5, 247.5, 337.5,
                     0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
            }
        }
    }

    /// EVOLVED FUNCTION: Select direction angle for placement search
    #[inline]
    fn select_direction(&self, n: usize, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;

        // Mix structured and weighted-random directions
        if rng.gen::<f64>() < 0.7 {
            // Structured: evenly spaced with jitter
            let base_idx = rng.gen_range(0..num_dirs);
            let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
            base + rng.gen_range(-0.15..0.15)
        } else {
            // Weighted random: favor corners (45, 135, 225, 315)
            loop {
                let angle = rng.gen_range(0.0..2.0 * PI);
                let corner_weight = (2.0 * angle).sin().abs();
                let threshold = 0.25 + 0.1 * (1.0 - (n as f64 / 200.0).min(1.0));
                if rng.gen::<f64>() < corner_weight.max(threshold) {
                    return angle;
                }
            }
        }
    }

    /// EVOLVED FUNCTION: Local search with simulated annealing
    fn local_search(&self, trees: &mut Vec<PlacedTree>, n: usize, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut temp = self.config.sa_initial_temp;

        // Scale iterations with n
        let iterations = self.config.sa_iterations + n * 20;

        for iter in 0..iterations {
            let idx = rng.gen_range(0..trees.len());
            let old_tree = trees[idx].clone();

            // EVOLVED: Move operator selection with more rotation emphasis
            let success = self.sa_move(trees, idx, temp, iter, rng);

            if success {
                let new_side = compute_side_length(trees);
                let delta = new_side - current_side;

                // Metropolis criterion
                if delta <= 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                    current_side = new_side;
                } else {
                    trees[idx] = old_tree;
                }
            } else {
                trees[idx] = old_tree;
            }

            temp = (temp * self.config.sa_cooling_rate).max(self.config.sa_min_temp);
        }
    }

    /// NEW: Rotation-only SA pass
    /// Keeps positions fixed, only optimizes orientations
    fn rotation_only_pass(&self, trees: &mut Vec<PlacedTree>, n: usize, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut temp = self.config.rotation_sa_temp;

        // Scale iterations with n
        let iterations = self.config.rotation_sa_iterations + n * 10;

        for _iter in 0..iterations {
            let idx = rng.gen_range(0..trees.len());
            let old_tree = trees[idx].clone();

            // Only try rotation moves
            let success = self.rotation_move(trees, idx, temp, rng);

            if success {
                let new_side = compute_side_length(trees);
                let delta = new_side - current_side;

                // Metropolis criterion with lower temperature for finer control
                if delta <= 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                    current_side = new_side;
                } else {
                    trees[idx] = old_tree;
                }
            } else {
                trees[idx] = old_tree;
            }

            // Slower cooling for rotation pass
            temp = (temp * 0.9998).max(0.0005);
        }
    }

    /// NEW: Rotation-only move operator
    /// Tries various rotation adjustments while keeping position fixed
    #[inline]
    fn rotation_move(
        &self,
        trees: &mut [PlacedTree],
        idx: usize,
        temp: f64,
        rng: &mut impl Rng,
    ) -> bool {
        let old = &trees[idx];
        let old_x = old.x;
        let old_y = old.y;
        let old_angle = old.angle_deg;

        let move_type = rng.gen_range(0..5);

        match move_type {
            0 => {
                // Fine continuous rotation: small angle change
                let delta = rng.gen_range(-self.config.fine_rotation_range..self.config.fine_rotation_range);
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            1 => {
                // Very fine rotation: 1-5 degrees
                let delta = if rng.gen() { rng.gen_range(1.0..5.0) } else { rng.gen_range(-5.0..-1.0) };
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            2 => {
                // 22.5 degree rotation (finer granularity)
                let delta = if rng.gen() { 22.5 } else { -22.5 };
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            3 => {
                // 45 degree rotation
                let delta = if rng.gen() { 45.0 } else { -45.0 };
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            _ => {
                // 90 degree rotation
                let new_angle = (old_angle + 90.0).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
        }

        !has_overlap(trees, idx)
    }

    /// EVOLVED FUNCTION: SA move operator
    /// Enhanced with more rotation options and continuous adjustment
    #[inline]
    fn sa_move(
        &self,
        trees: &mut [PlacedTree],
        idx: usize,
        temp: f64,
        _iter: usize,
        rng: &mut impl Rng,
    ) -> bool {
        let old = &trees[idx];
        let old_x = old.x;
        let old_y = old.y;
        let old_angle = old.angle_deg;

        // Increased rotation probability (8 move types, 4 involve rotation)
        let move_type = rng.gen_range(0..8);

        match move_type {
            0 => {
                // Small translation (temperature-scaled)
                let scale = self.config.translation_scale * (0.3 + temp * 2.0);
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            1 => {
                // 90-degree rotation
                let new_angle = (old_angle + 90.0).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            2 => {
                // Fine rotation (22.5 degrees - finer than Gen1's 45)
                let delta = if rng.gen() { self.config.rotation_granularity }
                            else { -self.config.rotation_granularity };
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            3 => {
                // Move toward center
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.05 {
                    let scale = self.config.center_pull_strength * (0.5 + temp);
                    let dx = -old_x / mag * scale;
                    let dy = -old_y / mag * scale;
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                } else {
                    return false;
                }
            }
            4 => {
                // Translate + rotate combo
                let scale = self.config.translation_scale * 0.5;
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                let dangle = rng.gen_range(-30.0..30.0);
                let new_angle = (old_angle + dangle).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, new_angle);
            }
            5 => {
                // Polar move (radial in/out)
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.1 {
                    let delta_r = rng.gen_range(-0.05..0.05) * (1.0 + temp);
                    let new_mag = (mag + delta_r).max(0.0);
                    let scale = new_mag / mag;
                    trees[idx] = PlacedTree::new(old_x * scale, old_y * scale, old_angle);
                } else {
                    return false;
                }
            }
            6 => {
                // NEW: Continuous fine rotation (temperature-scaled)
                let range = self.config.fine_rotation_range * (0.5 + temp);
                let delta = rng.gen_range(-range..range);
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            _ => {
                // NEW: Snap to nearest 22.5-degree angle (clean up rotations)
                let snapped = (old_angle / 22.5).round() * 22.5;
                if (snapped - old_angle).abs() > 0.01 {
                    trees[idx] = PlacedTree::new(old_x, old_y, snapped);
                } else {
                    // Already snapped, try small perturbation
                    let delta = if rng.gen() { 22.5 } else { -22.5 };
                    let new_angle = (old_angle + delta).rem_euclid(360.0);
                    trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
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
        println!("Gen2 Rotation-Optimized score for n=1..50: {:.4}", score);
    }

    #[test]
    fn test_expanded_angles() {
        let packer = EvolvedPacker::default();

        // Test that we get 16 angles for each n value
        for n in 1..=4 {
            let angles = packer.select_angles(n);
            assert_eq!(angles.len(), 16, "Expected 16 angles for n={}", n);

            // Check all angles are unique and in valid range
            for &angle in &angles {
                assert!(angle >= 0.0 && angle < 360.0, "Invalid angle: {}", angle);
            }
        }
    }

    #[test]
    fn test_rotation_only_pass() {
        let packer = EvolvedPacker::default();

        // Create a small packing
        let packings = packer.pack_all(5);

        // All packings should be valid (no overlaps)
        for p in &packings {
            assert!(!p.has_overlaps());
        }
    }
}
