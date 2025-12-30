//! Evolved Packing Algorithm - Generation 1 INTENSIFY Mutation
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
//! MUTATION STRATEGY: INTENSIFY
//! - Doubled SA iterations: 10000 + n*50 (was 5000 + n*20)
//! - More search attempts: 120 (was 60)
//! - More direction samples: 24 (was 16)
//! - Finer binary search: 0.001 (was 0.003)
//! - Lower min temperature: 0.0001 (was 0.001)
//! - Slower cooling: 0.9997 (was 0.9993)
//! - Higher initial temp: 0.5 (was 0.35)
//!
//! Current best score: 103.5 (target: < 69)

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
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // INTENSIFY: Dramatically increased computational effort
        Self {
            search_attempts: 120,           // Was 60 - doubled
            direction_samples: 24,          // Was 16 - 50% more samples
            sa_iterations: 10000,           // Was 5000 - doubled base iterations
            sa_initial_temp: 0.5,           // Was 0.35 - higher start for more exploration
            sa_cooling_rate: 0.9997,        // Was 0.9993 - slower cooling
            sa_min_temp: 0.0001,            // Was 0.001 - 10x lower minimum
            translation_scale: 0.08,        // Unchanged
            rotation_granularity: 45.0,     // Unchanged
            center_pull_strength: 0.04,     // Unchanged
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

            // Run evolved local search
            self.local_search(&mut trees, n, &mut rng);

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
                // INTENSIFY: Finer precision 0.001 (was 0.003)
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
        let balance_penalty = (width - height).abs() * 0.15;

        // Tertiary: slight preference for compact center (scaled by n)
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.01 / (n as f64).sqrt();

        side_score + balance_penalty + center_penalty
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    /// Returns angles in priority order
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
        // Evolved: use 8 directions with n-dependent priority
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
                // Adaptive threshold based on n
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

        // INTENSIFY: Scale iterations more aggressively with n
        // Was: 5000 + n * 20, now: 10000 + n * 50
        let iterations = self.config.sa_iterations + n * 50;

        for iter in 0..iterations {
            let idx = rng.gen_range(0..trees.len());
            let old_tree = trees[idx].clone();

            // EVOLVED: Move operator selection
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

    /// EVOLVED FUNCTION: SA move operator
    /// Returns true if move is valid (no overlap)
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

        let move_type = rng.gen_range(0..6);

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
                // Fine rotation (45 degrees)
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
            _ => {
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
        println!("Evolved score for n=1..50: {:.4}", score);
    }
}
