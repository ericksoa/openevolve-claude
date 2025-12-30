//! Evolved Packing Algorithm - Generation 3 ADAPTIVE Scaling
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
//! MUTATION STRATEGY: ADAPTIVE SCALING (Gen3)
//! Building on Gen2 champion with n-adaptive computation:
//!
//! Key insight: Focus computational effort where it matters most (large n).
//! Small problems (n < 30) are easy, large problems (n > 100) need maximum effort.
//!
//! Parameter scaling by problem size:
//! - sa_iterations: 5000 + n^1.5 * 20 (non-linear scaling)
//! - search_attempts: 50 + n * 2 (more attempts for harder problems)
//! - For n > 150: Run 3 SA passes instead of 2
//! - For n > 100: Use finer binary search (0.0002)
//! - Track best solution and periodically restart from it
//!
//! Small n (<30):    Moderate computation - these are easy
//! Medium n (30-100): Scale up computation significantly
//! Large n (100-200): Maximum computation - this is where the gap is

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Evolved packing configuration
/// These parameters are tuned through evolution
pub struct EvolvedConfig {
    // Base search parameters (will be scaled adaptively)
    pub base_search_attempts: usize,
    pub direction_samples: usize,

    // Base simulated annealing (will be scaled adaptively)
    pub base_sa_iterations: usize,
    pub sa_initial_temp: f64,
    pub sa_cooling_rate: f64,
    pub sa_min_temp: f64,

    // Move parameters
    pub translation_scale: f64,
    pub rotation_granularity: f64,
    pub center_pull_strength: f64,

    // ADAPTIVE: Restart tracking
    pub restart_interval: usize,
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen3 ADAPTIVE: n-dependent computational effort
        Self {
            base_search_attempts: 50,        // Base: 50 + n * 2 scaling
            direction_samples: 48,           // Keep from Gen2
            base_sa_iterations: 5000,        // Base: 5000 + n^1.5 * 20 scaling
            sa_initial_temp: 0.6,            // Keep from Gen2
            sa_cooling_rate: 0.9998,         // Slightly faster than Gen2 (balance with more iterations)
            sa_min_temp: 0.00001,            // Keep from Gen2
            translation_scale: 0.08,         // Unchanged
            rotation_granularity: 22.5,      // Keep from Gen2
            center_pull_strength: 0.05,      // Keep from Gen2
            restart_interval: 5000,          // Restart from best every 5000 iterations
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
    /// ADAPTIVE: Compute search attempts based on n
    #[inline]
    fn get_search_attempts(&self, n: usize) -> usize {
        // 50 + n * 2: scales linearly with problem size
        // n=10:  50 + 20  = 70
        // n=50:  50 + 100 = 150
        // n=100: 50 + 200 = 250
        // n=200: 50 + 400 = 450
        self.config.base_search_attempts + n * 2
    }

    /// ADAPTIVE: Compute SA iterations based on n
    #[inline]
    fn get_sa_iterations(&self, n: usize) -> usize {
        // 5000 + n^1.5 * 20: non-linear scaling
        // n=10:  5000 + 31.6 * 20  = 5632
        // n=30:  5000 + 164.3 * 20 = 8286
        // n=50:  5000 + 353.6 * 20 = 12072
        // n=100: 5000 + 1000 * 20  = 25000
        // n=150: 5000 + 1837 * 20  = 41748
        // n=200: 5000 + 2828 * 20  = 61569
        let n_factor = (n as f64).powf(1.5) * 20.0;
        self.config.base_sa_iterations + n_factor as usize
    }

    /// ADAPTIVE: Compute number of SA passes based on n
    #[inline]
    fn get_sa_passes(&self, n: usize) -> usize {
        if n > 150 {
            3  // Maximum passes for hardest problems
        } else if n > 50 {
            2  // Standard double pass for medium problems
        } else {
            1  // Single pass sufficient for easy problems
        }
    }

    /// ADAPTIVE: Get binary search precision based on n
    #[inline]
    fn get_binary_precision(&self, n: usize) -> f64 {
        if n > 100 {
            0.0002  // Ultra-fine for large n
        } else if n > 50 {
            0.0004  // Fine for medium n
        } else {
            0.0008  // Moderate for small n (faster)
        }
    }

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

            // ADAPTIVE: Run n-dependent SA passes for optimization
            let sa_passes = self.get_sa_passes(n);
            for pass in 0..sa_passes {
                self.local_search(&mut trees, n, pass, sa_passes, &mut rng);
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
        let search_attempts = self.get_search_attempts(n);
        let precision = self.get_binary_precision(n);

        for _ in 0..search_attempts {
            let dir = self.select_direction(n, rng);
            let vx = dir.cos();
            let vy = dir.sin();

            for &tree_angle in &angles {
                // Binary search for closest valid position
                // ADAPTIVE: precision varies by n
                let mut low = 0.0;
                let mut high = 12.0;

                while high - low > precision {
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

        // Primary: minimize side length (most important)
        let side_score = side;

        // Secondary: prefer balanced aspect ratio
        // ADAPTIVE: Increase balance weight for larger n (more important to keep things square)
        let balance_weight = 0.08 + 0.12 * ((n as f64 / 200.0).min(1.0));
        let balance_penalty = (width - height).abs() * balance_weight;

        // Tertiary: slight preference for compact center (scaled by n)
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.006 / (n as f64).sqrt();

        // ADAPTIVE: Density heuristic - more important for large n
        let area = width * height;
        let density_weight = 0.003 + 0.007 * ((n as f64 / 200.0).min(1.0));
        let density_bonus = if area > 0.0 {
            -density_weight * (n as f64 / area).min(2.0)
        } else {
            0.0
        };

        side_score + balance_penalty + center_penalty + density_bonus
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    /// Returns angles in priority order
    /// ADAPTIVE: More angles for larger n
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
        // ADAPTIVE: Use more angles for larger problems
        let base = match n % 6 {
            0 => vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0, 30.0, 60.0, 120.0, 150.0],
            1 => vec![90.0, 270.0, 0.0, 180.0, 135.0, 315.0, 45.0, 225.0, 60.0, 120.0, 240.0, 300.0],
            2 => vec![180.0, 0.0, 270.0, 90.0, 225.0, 45.0, 315.0, 135.0, 150.0, 210.0, 330.0, 30.0],
            3 => vec![270.0, 90.0, 180.0, 0.0, 315.0, 135.0, 225.0, 45.0, 240.0, 300.0, 60.0, 120.0],
            4 => vec![45.0, 225.0, 135.0, 315.0, 0.0, 90.0, 180.0, 270.0, 15.0, 75.0, 195.0, 255.0],
            _ => vec![135.0, 315.0, 45.0, 225.0, 90.0, 270.0, 0.0, 180.0, 105.0, 165.0, 285.0, 345.0],
        };

        // ADAPTIVE: Add finer angles for large n
        if n > 100 {
            let mut extended = base;
            // Add 15-degree increments for denser packing
            for i in 0..24 {
                let angle = i as f64 * 15.0;
                if !extended.contains(&angle) {
                    extended.push(angle);
                }
            }
            extended
        } else {
            base
        }
    }

    /// EVOLVED FUNCTION: Select direction angle for placement search
    /// ADAPTIVE: More sophisticated direction selection for large n
    #[inline]
    fn select_direction(&self, n: usize, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;

        // ADAPTIVE: Increase structured sampling for larger n
        let structured_prob = if n > 100 { 0.6 } else if n > 50 { 0.5 } else { 0.4 };

        let strategy = rng.gen::<f64>();

        if strategy < structured_prob {
            // Structured: evenly spaced with small jitter
            let base_idx = rng.gen_range(0..num_dirs);
            let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
            let jitter = if n > 100 { 0.05 } else { 0.08 };  // Tighter for large n
            base + rng.gen_range(-jitter..jitter)
        } else if strategy < structured_prob + 0.25 {
            // Weighted random: favor corners and edges
            loop {
                let angle = rng.gen_range(0.0..2.0 * PI);
                let corner_weight = ((4.0 * angle).sin().abs() + (4.0 * angle).cos().abs()) / 2.0;
                let threshold = 0.2 + 0.15 * (1.0 - (n as f64 / 200.0).min(1.0));
                if rng.gen::<f64>() < corner_weight.max(threshold) {
                    return angle;
                }
            }
        } else {
            // Golden angle spiral for good coverage
            let golden_angle = PI * (3.0 - (5.0_f64).sqrt());
            let base = (n as f64 * golden_angle) % (2.0 * PI);
            let offset = rng.gen_range(0..8) as f64 * PI / 4.0;
            (base + offset + rng.gen_range(-0.1..0.1)) % (2.0 * PI)
        }
    }

    /// EVOLVED FUNCTION: Local search with simulated annealing
    /// ADAPTIVE: n-dependent iterations with periodic restarts
    fn local_search(
        &self,
        trees: &mut Vec<PlacedTree>,
        n: usize,
        pass: usize,
        total_passes: usize,
        rng: &mut impl Rng,
    ) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config: Vec<PlacedTree> = trees.clone();

        // ADAPTIVE: Adjust temperature based on pass number and total passes
        let temp_multiplier = match pass {
            0 => 1.0,
            1 => 0.4,
            _ => 0.15,  // Third pass: very focused refinement
        };
        let mut temp = self.config.sa_initial_temp * temp_multiplier;

        // ADAPTIVE: Get n-dependent iterations
        let base_iterations = self.get_sa_iterations(n);
        let pass_iterations = if pass == 0 {
            base_iterations
        } else if pass == 1 {
            base_iterations / 2
        } else {
            base_iterations / 4  // Third pass: shorter
        };

        // Track iterations since improvement for restart
        let mut iters_since_improvement = 0;

        for iter in 0..pass_iterations {
            // ADAPTIVE: Periodically restart from best solution
            if iters_since_improvement > self.config.restart_interval {
                if best_side < current_side {
                    *trees = best_config.clone();
                    current_side = best_side;
                    // Reheat slightly for new exploration
                    temp = (temp * 2.0).min(self.config.sa_initial_temp * 0.2);
                }
                iters_since_improvement = 0;
            }

            // ADAPTIVE: Prefer moving boundary trees more for large n
            let idx = self.select_tree_to_move(trees, n, rng);
            let old_tree = trees[idx].clone();

            // EVOLVED: Move operator selection
            let success = self.sa_move(trees, idx, temp, iter, n, rng);

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
                        iters_since_improvement = 0;
                    } else {
                        iters_since_improvement += 1;
                    }
                } else {
                    trees[idx] = old_tree;
                    iters_since_improvement += 1;
                }
            } else {
                trees[idx] = old_tree;
                iters_since_improvement += 1;
            }

            temp = (temp * self.config.sa_cooling_rate).max(self.config.sa_min_temp);
        }

        // Restore best configuration found during search
        if best_side < compute_side_length(trees) {
            *trees = best_config;
        }
    }

    /// ADAPTIVE: Select tree to move with preference for boundary trees
    /// Increased boundary preference for large n
    #[inline]
    fn select_tree_to_move(&self, trees: &[PlacedTree], n: usize, rng: &mut impl Rng) -> usize {
        // ADAPTIVE: Higher boundary probability for larger n
        let boundary_prob = if n > 100 { 0.4 } else if n > 50 { 0.35 } else { 0.3 };

        if trees.len() <= 2 || rng.gen::<f64>() < (1.0 - boundary_prob) {
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
    /// ADAPTIVE: Move parameters scale with n and temperature
    /// Returns true if move is valid (no overlap)
    #[inline]
    fn sa_move(
        &self,
        trees: &mut [PlacedTree],
        idx: usize,
        temp: f64,
        iter: usize,
        n: usize,
        rng: &mut impl Rng,
    ) -> bool {
        let old = &trees[idx];
        let old_x = old.x;
        let old_y = old.y;
        let old_angle = old.angle_deg;

        // ADAPTIVE: More move types for large n, including finer moves
        let max_move_type = if n > 100 { 12 } else { 10 };
        let move_type = rng.gen_range(0..max_move_type);

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
            9 => {
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
            10 => {
                // ADAPTIVE: Micro-nudge for large n fine-tuning
                let scale = 0.005 * (0.3 + temp * 0.7);
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            _ => {
                // ADAPTIVE: 15-degree rotation for finer granularity
                let delta = if rng.gen() { 15.0 } else { -15.0 };
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
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
    fn test_adaptive_scaling() {
        let packer = EvolvedPacker::default();

        // Test search attempts scaling
        assert_eq!(packer.get_search_attempts(10), 70);   // 50 + 20
        assert_eq!(packer.get_search_attempts(100), 250); // 50 + 200

        // Test SA iterations scaling (approximate due to float)
        assert!(packer.get_sa_iterations(10) > 5500);     // ~5632
        assert!(packer.get_sa_iterations(100) > 24000);   // ~25000

        // Test SA passes scaling
        assert_eq!(packer.get_sa_passes(30), 1);
        assert_eq!(packer.get_sa_passes(80), 2);
        assert_eq!(packer.get_sa_passes(180), 3);
    }
}
