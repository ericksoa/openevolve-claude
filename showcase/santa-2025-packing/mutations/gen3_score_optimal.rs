//! Evolved Packing Algorithm - Generation 3 SCORE-OPTIMAL
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
//! MUTATION STRATEGY: SCORE-OPTIMAL (Gen3)
//!
//! KEY INSIGHT: The competition score is sum(side^2/n) for n=1 to 200.
//! This means:
//!   - n=1 has weight 1/1 = 1.0
//!   - n=10 has weight 1/10 = 0.1
//!   - n=50 has weight 1/50 = 0.02
//!   - n=100 has weight 1/100 = 0.01
//!   - n=200 has weight 1/200 = 0.005
//!
//! SMALL n VALUES MATTER DISPROPORTIONATELY!
//! - n=1 to n=20 accounts for ~50% of total score weight (harmonic series)
//! - n=1 to n=50 accounts for ~70% of total score weight
//!
//! Strategy: Allocate computation proportionally to score importance:
//! - n < 20:   search_attempts=500, sa_iterations=30000 (MAXIMUM effort)
//! - n 20-50:  search_attempts=300, sa_iterations=25000 (HIGH effort)
//! - n 50-100: search_attempts=200, sa_iterations=20000 (MEDIUM effort)
//! - n > 100:  search_attempts=150, sa_iterations=15000 (STANDARD effort)
//!
//! Additional optimizations:
//! - placement_score directly optimizes side^2/n
//! - Multi-restart for small n to find globally better solutions
//! - Finer binary search precision for small n
//! - More SA passes for small n

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Evolved packing configuration - parameters vary by n
/// These parameters are tuned through evolution
pub struct EvolvedConfig {
    // Base parameters (overridden by get_params_for_n)
    pub direction_samples: usize,

    // Move parameters
    pub translation_scale: f64,
    pub rotation_granularity: f64,
    pub center_pull_strength: f64,
}

/// N-dependent parameters for score-optimal computation allocation
#[derive(Clone, Copy)]
struct NParams {
    search_attempts: usize,
    sa_iterations: usize,
    sa_passes: usize,
    binary_precision: f64,
    sa_initial_temp: f64,
    sa_cooling_rate: f64,
    sa_min_temp: f64,
    restarts: usize,  // Number of complete restarts to try for small n
}

impl NParams {
    /// Get parameters tuned for the importance of n in the score
    fn for_n(n: usize) -> Self {
        if n < 20 {
            // CRITICAL: These are 50% of the score weight!
            // Use MAXIMUM computational effort
            NParams {
                search_attempts: 500,
                sa_iterations: 30000,
                sa_passes: 3,
                binary_precision: 0.0002,  // Ultra-fine precision
                sa_initial_temp: 0.8,
                sa_cooling_rate: 0.99995,  // Very slow cooling
                sa_min_temp: 0.000001,
                restarts: if n <= 5 { 3 } else if n <= 10 { 2 } else { 1 },
            }
        } else if n <= 50 {
            // HIGH importance: ~20% of score weight
            NParams {
                search_attempts: 300,
                sa_iterations: 25000,
                sa_passes: 2,
                binary_precision: 0.0003,
                sa_initial_temp: 0.7,
                sa_cooling_rate: 0.9999,
                sa_min_temp: 0.00001,
                restarts: 1,
            }
        } else if n <= 100 {
            // MEDIUM importance: ~15% of score weight
            NParams {
                search_attempts: 200,
                sa_iterations: 20000,
                sa_passes: 2,
                binary_precision: 0.0005,
                sa_initial_temp: 0.6,
                sa_cooling_rate: 0.9999,
                sa_min_temp: 0.00001,
                restarts: 1,
            }
        } else {
            // LOWER importance: ~15% of score weight for n>100
            NParams {
                search_attempts: 150,
                sa_iterations: 15000,
                sa_passes: 1,
                binary_precision: 0.001,
                sa_initial_temp: 0.5,
                sa_cooling_rate: 0.9998,
                sa_min_temp: 0.0001,
                restarts: 1,
            }
        }
    }
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        Self {
            direction_samples: 48,         // High base for all n
            translation_scale: 0.08,
            rotation_granularity: 22.5,    // 16 angles
            center_pull_strength: 0.05,
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
            let params = NParams::for_n(n);

            // For small n, try multiple complete restarts and keep the best
            let mut best_trees = self.pack_single_n(&prev_trees, n, max_n, &params, &mut rng);
            let mut best_side = compute_side_length(&best_trees);

            // Multi-restart optimization for critical small n values
            for _restart in 1..params.restarts {
                let candidate_trees = self.pack_single_n(&prev_trees, n, max_n, &params, &mut rng);
                let candidate_side = compute_side_length(&candidate_trees);

                if candidate_side < best_side {
                    best_side = candidate_side;
                    best_trees = candidate_trees;
                }
            }

            // Store result
            let mut packing = Packing::new();
            for t in &best_trees {
                packing.trees.push(t.clone());
            }
            packings.push(packing);
            prev_trees = best_trees;
        }

        packings
    }

    /// Pack a single n value with given parameters
    fn pack_single_n(
        &self,
        prev_trees: &[PlacedTree],
        n: usize,
        max_n: usize,
        params: &NParams,
        rng: &mut impl Rng,
    ) -> Vec<PlacedTree> {
        let mut trees = prev_trees.to_vec();

        // Place new tree using evolved heuristics
        let new_tree = self.find_placement(&trees, n, max_n, params, rng);
        trees.push(new_tree);

        // Run multiple SA passes for deeper optimization
        for pass in 0..params.sa_passes {
            self.local_search(&mut trees, n, pass, params, rng);
        }

        trees
    }

    /// EVOLVED FUNCTION: Find best placement for new tree
    /// Score-optimal version focuses on minimizing side^2/n
    fn find_placement(
        &self,
        existing: &[PlacedTree],
        n: usize,
        _max_n: usize,
        params: &NParams,
        rng: &mut impl Rng,
    ) -> PlacedTree {
        if existing.is_empty() {
            // First tree: place at origin with optimal rotation
            return PlacedTree::new(0.0, 0.0, 90.0);
        }

        let mut best_tree = PlacedTree::new(0.0, 0.0, 90.0);
        let mut best_score = f64::INFINITY;

        let angles = self.select_angles(n);

        for _ in 0..params.search_attempts {
            let dir = self.select_direction(n, rng);
            let vx = dir.cos();
            let vy = dir.sin();

            for &tree_angle in &angles {
                // Binary search for closest valid position
                // Precision varies by n importance
                let mut low = 0.0;
                let mut high = 12.0;

                while high - low > params.binary_precision {
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
    /// SCORE-OPTIMAL: Directly optimizes side^2/n, the actual competition metric
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

        // PRIMARY: Directly optimize competition score metric side^2/n
        // This is THE metric that matters for the leaderboard
        let score_contribution = (side * side) / (n as f64);

        // SECONDARY: Prefer balanced aspect ratio (helps future placements)
        // Weight this more for small n where we have more flexibility
        let n_factor = 1.0 / (1.0 + (n as f64).ln());
        let balance_penalty = (width - height).abs() * 0.08 * n_factor;

        // TERTIARY: Slight preference for compact center
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.005 * n_factor;

        // QUATERNARY: Density heuristic - prefer filling gaps
        let area = width * height;
        let density_bonus = if area > 0.0 {
            -0.003 * (n as f64 / area).min(2.0) * n_factor
        } else {
            0.0
        };

        score_contribution + balance_penalty + center_penalty + density_bonus
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    /// Returns angles in priority order
    /// 12 angles for fine granularity
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
        // Use 12 directions with n-dependent priority
        let base = match n % 6 {
            0 => vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0, 30.0, 60.0, 120.0, 150.0],
            1 => vec![90.0, 270.0, 0.0, 180.0, 135.0, 315.0, 45.0, 225.0, 60.0, 120.0, 240.0, 300.0],
            2 => vec![180.0, 0.0, 270.0, 90.0, 225.0, 45.0, 315.0, 135.0, 150.0, 210.0, 330.0, 30.0],
            3 => vec![270.0, 90.0, 180.0, 0.0, 315.0, 135.0, 225.0, 45.0, 240.0, 300.0, 60.0, 120.0],
            4 => vec![45.0, 225.0, 135.0, 315.0, 0.0, 90.0, 180.0, 270.0, 15.0, 75.0, 195.0, 255.0],
            _ => vec![135.0, 315.0, 45.0, 225.0, 90.0, 270.0, 0.0, 180.0, 105.0, 165.0, 285.0, 345.0],
        };
        base
    }

    /// EVOLVED FUNCTION: Select direction angle for placement search
    /// More sophisticated direction selection with golden angle coverage
    #[inline]
    fn select_direction(&self, n: usize, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;

        // Three-way mix of direction strategies
        let strategy = rng.gen::<f64>();

        if strategy < 0.5 {
            // Structured: evenly spaced with small jitter
            let base_idx = rng.gen_range(0..num_dirs);
            let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
            base + rng.gen_range(-0.08..0.08)
        } else if strategy < 0.75 {
            // Weighted random: favor corners and edges
            loop {
                let angle = rng.gen_range(0.0..2.0 * PI);
                // Favor 45-degree increments
                let corner_weight = ((4.0 * angle).sin().abs() + (4.0 * angle).cos().abs()) / 2.0;
                let threshold = 0.2 + 0.15 * (1.0 - (n as f64 / 200.0).min(1.0));
                if rng.gen::<f64>() < corner_weight.max(threshold) {
                    return angle;
                }
            }
        } else {
            // Golden angle spiral for good coverage
            let golden_angle = PI * (3.0 - (5.0_f64).sqrt());  // ~137.5 degrees
            let base = (n as f64 * golden_angle) % (2.0 * PI);
            let offset = rng.gen_range(0..8) as f64 * PI / 4.0;
            (base + offset + rng.gen_range(-0.1..0.1)) % (2.0 * PI)
        }
    }

    /// EVOLVED FUNCTION: Local search with simulated annealing
    /// Score-optimal version uses n-dependent parameters
    fn local_search(
        &self,
        trees: &mut Vec<PlacedTree>,
        n: usize,
        pass: usize,
        params: &NParams,
        rng: &mut impl Rng,
    ) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config: Vec<PlacedTree> = trees.clone();

        // Adjust temperature based on pass number
        let temp_multiplier = match pass {
            0 => 1.0,
            1 => 0.3,
            _ => 0.1,  // Third pass: very focused refinement
        };
        let mut temp = params.sa_initial_temp * temp_multiplier;

        // Iterations scale with n for later passes
        let iterations = if pass == 0 {
            params.sa_iterations + n * 100
        } else if pass == 1 {
            params.sa_iterations / 2 + n * 50
        } else {
            params.sa_iterations / 4 + n * 25
        };

        for iter in 0..iterations {
            // Prefer moving trees that contribute to bounding box
            let idx = self.select_tree_to_move(trees, rng);
            let old_tree = trees[idx].clone();

            // Move operator selection
            let success = self.sa_move(trees, idx, temp, iter, rng);

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
                    }
                } else {
                    trees[idx] = old_tree;
                }
            } else {
                trees[idx] = old_tree;
            }

            temp = (temp * params.sa_cooling_rate).max(params.sa_min_temp);
        }

        // Restore best configuration found during search
        if best_side < compute_side_length(trees) {
            *trees = best_config;
        }
    }

    /// Select tree to move with preference for boundary trees
    #[inline]
    fn select_tree_to_move(&self, trees: &[PlacedTree], rng: &mut impl Rng) -> usize {
        // 70% chance to pick randomly, 30% chance to pick boundary tree
        if trees.len() <= 2 || rng.gen::<f64>() < 0.7 {
            return rng.gen_range(0..trees.len());
        }

        // Find bounding box
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

        // Find trees touching the boundary
        let mut boundary_indices: Vec<usize> = Vec::new();
        let eps = 0.01;

        for (i, tree) in trees.iter().enumerate() {
            let (bx1, by1, bx2, by2) = tree.bounds();
            if (bx1 - min_x).abs() < eps || (bx2 - max_x).abs() < eps ||
               (by1 - min_y).abs() < eps || (by2 - max_y).abs() < eps {
                boundary_indices.push(i);
            }
        }

        if boundary_indices.is_empty() {
            rng.gen_range(0..trees.len())
        } else {
            boundary_indices[rng.gen_range(0..boundary_indices.len())]
        }
    }

    /// EVOLVED FUNCTION: SA move operator
    /// 10 move types with well-tuned parameters
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

        // 10 move types
        let move_type = rng.gen_range(0..10);

        match move_type {
            0 => {
                // Small translation (temperature-scaled)
                let scale = self.config.translation_scale * (0.2 + temp * 2.5);
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
                // Fine rotation (22.5 degrees)
                let delta = if rng.gen() { self.config.rotation_granularity }
                            else { -self.config.rotation_granularity };
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            3 => {
                // Move toward center
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.05 {
                    let scale = self.config.center_pull_strength * (0.4 + temp * 1.5);
                    let dx = -old_x / mag * scale;
                    let dy = -old_y / mag * scale;
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                } else {
                    return false;
                }
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
                // Polar move (radial in/out)
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.1 {
                    let delta_r = rng.gen_range(-0.06..0.06) * (1.0 + temp);
                    let new_mag = (mag + delta_r).max(0.0);
                    let scale = new_mag / mag;
                    trees[idx] = PlacedTree::new(old_x * scale, old_y * scale, old_angle);
                } else {
                    return false;
                }
            }
            6 => {
                // Angular orbit (move around center)
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.1 {
                    let current_angle = old_y.atan2(old_x);
                    let delta_angle = rng.gen_range(-0.15..0.15) * (1.0 + temp);
                    let new_ang = current_angle + delta_angle;
                    trees[idx] = PlacedTree::new(mag * new_ang.cos(), mag * new_ang.sin(), old_angle);
                } else {
                    return false;
                }
            }
            7 => {
                // Very small nudge for fine-tuning
                let scale = 0.015 * (0.5 + temp);
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            8 => {
                // 180-degree flip
                let new_angle = (old_angle + 180.0).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            _ => {
                // Directional slide (move toward one corner)
                let corner_idx = rng.gen_range(0..4);
                let (dir_x, dir_y) = match corner_idx {
                    0 => (-1.0, -1.0),
                    1 => (1.0, -1.0),
                    2 => (-1.0, 1.0),
                    _ => (1.0, 1.0),
                };
                let scale = 0.03 * (0.5 + temp * 1.5);
                let norm = (2.0_f64).sqrt();
                trees[idx] = PlacedTree::new(
                    old_x + dir_x * scale / norm,
                    old_y + dir_y * scale / norm,
                    old_angle
                );
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

    #[test]
    fn test_n_params() {
        // Verify parameter scaling is correct
        let p1 = NParams::for_n(1);
        let p50 = NParams::for_n(50);
        let p100 = NParams::for_n(100);
        let p150 = NParams::for_n(150);

        // Small n should have more iterations
        assert!(p1.sa_iterations > p50.sa_iterations);
        assert!(p50.sa_iterations > p100.sa_iterations);
        assert!(p100.sa_iterations > p150.sa_iterations);

        // Small n should have more search attempts
        assert!(p1.search_attempts > p50.search_attempts);
        assert!(p50.search_attempts > p100.search_attempts);
        assert!(p100.search_attempts > p150.search_attempts);

        // Very small n should have restarts
        let p5 = NParams::for_n(5);
        assert!(p5.restarts >= 2);
    }
}
