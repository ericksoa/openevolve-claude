//! Evolved Packing Algorithm - Generation 1 Specialize Mutation
//!
//! This mutation adds problem-size (n) dependent specialization:
//! - Small n (<20): More exploration, favor corners, less SA
//! - Medium n (20-100): Balanced approach with moderate SA
//! - Large n (>100): Heavy SA, focus on compaction, favor axis-aligned angles
//!
//! Key changes:
//! - Adaptive SA iterations: 2000 for n<20, 5000 for 20<=n<100, 15000 for n>=100
//! - n-dependent angle selection favoring vertical/horizontal for small n
//! - Different move operator weights based on problem size
//! - Size-adaptive translation scales and cooling rates

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Evolved packing configuration with n-dependent specialization
/// These parameters are tuned through evolution
pub struct EvolvedConfig {
    // Search parameters
    pub search_attempts: usize,
    pub direction_samples: usize,

    // Simulated annealing (base values - will be adapted per n)
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

/// Size category for n-dependent specialization
#[derive(Clone, Copy, PartialEq)]
enum SizeCategory {
    Small,   // n < 20
    Medium,  // 20 <= n < 100
    Large,   // n >= 100
}

impl SizeCategory {
    fn from_n(n: usize) -> Self {
        if n < 20 {
            SizeCategory::Small
        } else if n < 100 {
            SizeCategory::Medium
        } else {
            SizeCategory::Large
        }
    }
}

/// Main evolved packer with n-dependent specialization
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

            // Run evolved local search with n-dependent parameters
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

    /// Get n-dependent SA iterations
    #[inline]
    fn get_sa_iterations(&self, n: usize) -> usize {
        match SizeCategory::from_n(n) {
            SizeCategory::Small => 2000 + n * 50,      // 2000-3000 for n<20
            SizeCategory::Medium => 5000 + n * 30,     // 5600-7970 for 20<=n<100
            SizeCategory::Large => 15000 + n * 10,     // 16000+ for n>=100
        }
    }

    /// Get n-dependent search attempts
    #[inline]
    fn get_search_attempts(&self, n: usize) -> usize {
        match SizeCategory::from_n(n) {
            SizeCategory::Small => 100,   // More exploration for small n
            SizeCategory::Medium => 70,   // Moderate
            SizeCategory::Large => 50,    // Focus on SA instead
        }
    }

    /// Get n-dependent translation scale
    #[inline]
    fn get_translation_scale(&self, n: usize) -> f64 {
        match SizeCategory::from_n(n) {
            SizeCategory::Small => 0.12,   // Larger moves for small n
            SizeCategory::Medium => 0.08,  // Moderate
            SizeCategory::Large => 0.05,   // Fine-grained for large n
        }
    }

    /// Get n-dependent cooling rate
    #[inline]
    fn get_cooling_rate(&self, n: usize) -> f64 {
        match SizeCategory::from_n(n) {
            SizeCategory::Small => 0.998,   // Faster cooling, less iterations
            SizeCategory::Medium => 0.9993, // Moderate
            SizeCategory::Large => 0.9997,  // Slower cooling for more iterations
        }
    }

    /// Get n-dependent initial temperature
    #[inline]
    fn get_initial_temp(&self, n: usize) -> f64 {
        match SizeCategory::from_n(n) {
            SizeCategory::Small => 0.5,    // Higher temp for exploration
            SizeCategory::Medium => 0.35,  // Moderate
            SizeCategory::Large => 0.25,   // Lower temp, focus on refinement
        }
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

        for _ in 0..search_attempts {
            let dir = self.select_direction(n, rng);
            let vx = dir.cos();
            let vy = dir.sin();

            for &tree_angle in &angles {
                // Binary search for closest valid position
                let mut low = 0.0;
                let mut high = 12.0;

                // n-dependent precision: tighter for small n
                let precision = match SizeCategory::from_n(n) {
                    SizeCategory::Small => 0.001,
                    SizeCategory::Medium => 0.003,
                    SizeCategory::Large => 0.005,
                };

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

        // Primary: minimize side length
        let side_score = side;

        // n-dependent balance penalty
        let balance_weight = match SizeCategory::from_n(n) {
            SizeCategory::Small => 0.20,   // Stronger balance for small n
            SizeCategory::Medium => 0.15,  // Moderate
            SizeCategory::Large => 0.10,   // Less important for large n
        };
        let balance_penalty = (width - height).abs() * balance_weight;

        // n-dependent center pull
        let center_weight = match SizeCategory::from_n(n) {
            SizeCategory::Small => 0.02,   // Moderate center pull
            SizeCategory::Medium => 0.01,  // Standard
            SizeCategory::Large => 0.005,  // Less center focus for large
        };
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * center_weight / (n as f64).sqrt();

        // For large n, add density bonus (prefer filling gaps)
        let density_bonus = if n >= 100 {
            let area_used = (n as f64) * 0.5; // Approximate tree area
            let total_area = width * height;
            if total_area > 0.0 {
                (1.0 - area_used / total_area).max(0.0) * 0.05
            } else {
                0.0
            }
        } else {
            0.0
        };

        side_score + balance_penalty + center_penalty + density_bonus
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    /// Returns angles in priority order with n-dependent selection
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
        match SizeCategory::from_n(n) {
            SizeCategory::Small => {
                // Small n: favor axis-aligned (vertical/horizontal) for simpler packing
                match n % 4 {
                    0 => vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0],
                    1 => vec![90.0, 270.0, 0.0, 180.0, 45.0, 135.0, 225.0, 315.0],
                    2 => vec![180.0, 0.0, 90.0, 270.0, 45.0, 135.0, 225.0, 315.0],
                    _ => vec![270.0, 90.0, 0.0, 180.0, 45.0, 135.0, 225.0, 315.0],
                }
            }
            SizeCategory::Medium => {
                // Medium n: balanced with cycle
                match n % 4 {
                    0 => vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0],
                    1 => vec![90.0, 270.0, 0.0, 180.0, 135.0, 315.0, 45.0, 225.0],
                    2 => vec![180.0, 0.0, 270.0, 90.0, 225.0, 45.0, 315.0, 135.0],
                    _ => vec![270.0, 90.0, 180.0, 0.0, 315.0, 135.0, 225.0, 45.0],
                }
            }
            SizeCategory::Large => {
                // Large n: favor diagonals for better packing in tight spaces
                // Also add finer angles for more options
                match n % 6 {
                    0 => vec![45.0, 135.0, 225.0, 315.0, 0.0, 90.0, 180.0, 270.0, 22.5, 67.5],
                    1 => vec![135.0, 315.0, 45.0, 225.0, 90.0, 270.0, 0.0, 180.0, 112.5, 157.5],
                    2 => vec![225.0, 45.0, 315.0, 135.0, 180.0, 0.0, 270.0, 90.0, 202.5, 247.5],
                    3 => vec![315.0, 135.0, 225.0, 45.0, 270.0, 90.0, 180.0, 0.0, 292.5, 337.5],
                    4 => vec![0.0, 180.0, 45.0, 225.0, 90.0, 270.0, 135.0, 315.0, 15.0, 165.0],
                    _ => vec![90.0, 270.0, 135.0, 315.0, 0.0, 180.0, 45.0, 225.0, 75.0, 255.0],
                }
            }
        }
    }

    /// EVOLVED FUNCTION: Select direction angle for placement search
    /// n-dependent direction selection strategy
    #[inline]
    fn select_direction(&self, n: usize, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;
        let category = SizeCategory::from_n(n);

        match category {
            SizeCategory::Small => {
                // Small n: more structured, favor cardinal directions
                if rng.gen::<f64>() < 0.8 {
                    // Strongly structured
                    let base_idx = rng.gen_range(0..num_dirs);
                    let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
                    base + rng.gen_range(-0.1..0.1)
                } else {
                    // Cardinal direction with small jitter
                    let cardinal = [0.0, PI/2.0, PI, 3.0*PI/2.0];
                    cardinal[rng.gen_range(0..4)] + rng.gen_range(-0.2..0.2)
                }
            }
            SizeCategory::Medium => {
                // Medium n: balanced approach (original behavior)
                if rng.gen::<f64>() < 0.7 {
                    let base_idx = rng.gen_range(0..num_dirs);
                    let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
                    base + rng.gen_range(-0.15..0.15)
                } else {
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
            SizeCategory::Large => {
                // Large n: favor corners and diagonals for compact packing
                if rng.gen::<f64>() < 0.6 {
                    // Favor diagonal directions (45, 135, 225, 315 degrees)
                    let diagonals = [PI/4.0, 3.0*PI/4.0, 5.0*PI/4.0, 7.0*PI/4.0];
                    diagonals[rng.gen_range(0..4)] + rng.gen_range(-0.2..0.2)
                } else if rng.gen::<f64>() < 0.7 {
                    // Structured sampling
                    let base_idx = rng.gen_range(0..num_dirs);
                    let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
                    base + rng.gen_range(-0.1..0.1)
                } else {
                    // Random with corner bias
                    loop {
                        let angle = rng.gen_range(0.0..2.0 * PI);
                        let corner_weight = (2.0 * angle).sin().abs();
                        if rng.gen::<f64>() < corner_weight.max(0.3) {
                            return angle;
                        }
                    }
                }
            }
        }
    }

    /// EVOLVED FUNCTION: Local search with simulated annealing
    /// n-dependent SA parameters
    fn local_search(&self, trees: &mut Vec<PlacedTree>, n: usize, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut temp = self.get_initial_temp(n);
        let cooling_rate = self.get_cooling_rate(n);
        let iterations = self.get_sa_iterations(n);

        for iter in 0..iterations {
            let idx = rng.gen_range(0..trees.len());
            let old_tree = trees[idx].clone();

            // EVOLVED: Move operator selection with n-dependent weights
            let success = self.sa_move(trees, idx, temp, iter, n, rng);

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

            temp = (temp * cooling_rate).max(self.config.sa_min_temp);
        }
    }

    /// EVOLVED FUNCTION: SA move operator with n-dependent weights
    /// Returns true if move is valid (no overlap)
    #[inline]
    fn sa_move(
        &self,
        trees: &mut [PlacedTree],
        idx: usize,
        temp: f64,
        _iter: usize,
        n: usize,
        rng: &mut impl Rng,
    ) -> bool {
        let old = &trees[idx];
        let old_x = old.x;
        let old_y = old.y;
        let old_angle = old.angle_deg;

        let translation_scale = self.get_translation_scale(n);
        let category = SizeCategory::from_n(n);

        // n-dependent move type selection
        let move_type = match category {
            SizeCategory::Small => {
                // Small n: favor rotation and large moves
                let weights = [15, 25, 20, 15, 15, 10]; // translation, 90-rot, fine-rot, center, combo, polar
                weighted_choice(&weights, rng)
            }
            SizeCategory::Medium => {
                // Medium n: balanced (original)
                rng.gen_range(0..6)
            }
            SizeCategory::Large => {
                // Large n: favor center pull and fine adjustments
                let weights = [20, 15, 15, 25, 15, 10]; // more center pull
                weighted_choice(&weights, rng)
            }
        };

        match move_type {
            0 => {
                // Small translation (temperature-scaled)
                let scale = translation_scale * (0.3 + temp * 2.0);
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
                // Fine rotation (n-dependent granularity)
                let granularity = match category {
                    SizeCategory::Small => 45.0,
                    SizeCategory::Medium => 45.0,
                    SizeCategory::Large => 30.0, // Finer for large n
                };
                let delta = if rng.gen() { granularity } else { -granularity };
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            3 => {
                // Move toward center (n-dependent strength)
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.05 {
                    let center_strength = match category {
                        SizeCategory::Small => 0.06,
                        SizeCategory::Medium => 0.04,
                        SizeCategory::Large => 0.03, // Gentler for large n
                    };
                    let scale = center_strength * (0.5 + temp);
                    let dx = -old_x / mag * scale;
                    let dy = -old_y / mag * scale;
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                } else {
                    return false;
                }
            }
            4 => {
                // Translate + rotate combo
                let scale = translation_scale * 0.5;
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

/// Weighted random choice helper
#[inline]
fn weighted_choice(weights: &[usize], rng: &mut impl Rng) -> usize {
    let total: usize = weights.iter().sum();
    let mut roll = rng.gen_range(0..total);
    for (i, &w) in weights.iter().enumerate() {
        if roll < w {
            return i;
        }
        roll -= w;
    }
    weights.len() - 1
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
    fn test_size_categories() {
        // Verify size categorization
        assert_eq!(SizeCategory::from_n(1), SizeCategory::Small);
        assert_eq!(SizeCategory::from_n(19), SizeCategory::Small);
        assert_eq!(SizeCategory::from_n(20), SizeCategory::Medium);
        assert_eq!(SizeCategory::from_n(99), SizeCategory::Medium);
        assert_eq!(SizeCategory::from_n(100), SizeCategory::Large);
        assert_eq!(SizeCategory::from_n(200), SizeCategory::Large);
    }

    #[test]
    fn test_adaptive_iterations() {
        let packer = EvolvedPacker::default();

        // Small n should have fewer iterations
        assert!(packer.get_sa_iterations(10) < packer.get_sa_iterations(50));
        // Large n should have more iterations
        assert!(packer.get_sa_iterations(50) < packer.get_sa_iterations(150));
    }
}
