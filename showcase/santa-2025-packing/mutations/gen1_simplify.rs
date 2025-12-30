//! Evolved Packing Algorithm - Generation 1 Simplify Mutation
//!
//! MUTATION STRATEGY: Simplify
//! Changes made:
//! - select_angles(): Removed n%4 pattern, use single consistent order
//! - select_direction(): Removed corner_weight complexity, use uniform random with jitter
//! - sa_move(): Reduced from 6 to 4 move types (removed combo and polar moves)
//! - placement_score(): Removed weak center_penalty
//! - sa_move(): Removed temperature scaling from translation_scale
//!
//! Hypothesis: Simpler algorithms have fewer ways to go wrong

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
        // Gen1 parameters - evolved from baseline
        Self {
            search_attempts: 60,
            direction_samples: 16,
            sa_iterations: 5000,
            sa_initial_temp: 0.35,
            sa_cooling_rate: 0.9993,
            sa_min_temp: 0.001,
            translation_scale: 0.08,
            rotation_granularity: 45.0,
            center_pull_strength: 0.04,
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
    /// Key evolution target - determines placement quality
    /// SIMPLIFIED: Removed center_penalty (was too weak to matter)
    #[inline]
    fn placement_score(&self, tree: &PlacedTree, existing: &[PlacedTree], _n: usize) -> f64 {
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

        // SIMPLIFIED: Removed center_penalty - was too weak at 0.01 to matter
        side_score + balance_penalty
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    /// Returns angles in priority order
    /// SIMPLIFIED: Use one consistent order instead of n%4 pattern
    #[inline]
    fn select_angles(&self, _n: usize) -> Vec<f64> {
        // Simplified: single consistent order for all n
        // Cardinal directions first, then diagonals
        vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0]
    }

    /// EVOLVED FUNCTION: Select direction angle for placement search
    /// SIMPLIFIED: Removed corner weighting, use uniform random with structured sampling
    #[inline]
    fn select_direction(&self, _n: usize, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;

        // Simplified: evenly spaced directions with small jitter
        let base_idx = rng.gen_range(0..num_dirs);
        let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
        base + rng.gen_range(-0.15..0.15)
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
    /// SIMPLIFIED: Reduced from 6 to 4 move types (most effective ones)
    /// SIMPLIFIED: Removed temperature scaling from translation
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

        // SIMPLIFIED: Only 4 move types instead of 6
        let move_type = rng.gen_range(0..4);

        match move_type {
            0 => {
                // Small translation (SIMPLIFIED: fixed scale, no temperature)
                let scale = self.config.translation_scale;
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
            _ => {
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
