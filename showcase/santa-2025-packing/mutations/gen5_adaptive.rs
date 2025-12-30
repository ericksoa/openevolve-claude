//! Evolved Packing Algorithm - Generation 5 N-ADAPTIVE
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
//! MUTATION STRATEGY: N-ADAPTIVE (Gen5)
//! Key insight: Different problem sizes need different strategies.
//! Small n values contribute disproportionately to total score.
//!
//! Parameter changes from Gen4:
//! - Adaptive SA iterations based on n:
//!   * n < 20:   30000 iterations (heavy computation - these matter most!)
//!   * n 20-50:  15000 iterations (medium computation)
//!   * n 50-100: 8000 iterations (light computation)
//!   * n > 100:  4000 iterations (minimal computation)
//! - Adaptive binary search precision:
//!   * n < 20:   0.0001 (fine precision)
//!   * n < 50:   0.0005 (medium precision)
//!   * else:     0.001 (coarse precision)
//! - Adaptive search attempts:
//!   * n < 20:   500 attempts (more exploration)
//!   * n 20-50:  300 attempts
//!   * else:     200 attempts
//! - More rotation angles for small n (16 vs 8)
//! - 3 SA passes for small n, 2 for medium, 1 for large
//!
//! Goal: Allocate computation where it matters most for total score
//! Target: Beat Gen4's 98.37 by improving small-n packings

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Evolved packing configuration
/// These parameters are tuned through evolution
pub struct EvolvedConfig {
    // Search parameters (base values, adjusted by n)
    pub base_search_attempts: usize,
    pub base_direction_samples: usize,

    // Simulated annealing (base values, adjusted by n)
    pub base_sa_iterations: usize,
    pub sa_initial_temp: f64,
    pub sa_cooling_rate: f64,
    pub sa_min_temp: f64,

    // Move parameters
    pub translation_scale: f64,
    pub rotation_granularity: f64,
    pub center_pull_strength: f64,

    // ADAPTIVE: Early exit threshold (adjusted by n)
    pub base_early_exit_threshold: usize,
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen5 N-ADAPTIVE: Base configuration (will be scaled by n)
        Self {
            base_search_attempts: 300,       // Base, scaled by n
            base_direction_samples: 64,      // Base direction samples
            base_sa_iterations: 15000,       // Base, heavily scaled by n
            sa_initial_temp: 0.7,            // Slightly higher for more exploration
            sa_cooling_rate: 0.9999,         // Adjusted per n-tier
            sa_min_temp: 0.00001,            // Standard minimum
            translation_scale: 0.08,         // Unchanged
            rotation_granularity: 22.5,      // Finer for more angles when needed
            center_pull_strength: 0.06,      // Unchanged
            base_early_exit_threshold: 800,  // Base threshold, scaled by n
        }
    }
}

/// N-tier classification for adaptive parameters
#[derive(Clone, Copy, PartialEq)]
enum NTier {
    Critical,  // n < 20: These contribute most to score
    Important, // n 20-50: Still significant
    Medium,    // n 50-100: Moderate contribution
    Large,     // n > 100: Minimal individual contribution
}

impl NTier {
    fn from_n(n: usize) -> Self {
        if n < 20 {
            NTier::Critical
        } else if n < 50 {
            NTier::Important
        } else if n < 100 {
            NTier::Medium
        } else {
            NTier::Large
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
    /// Get adaptive SA iterations based on n
    #[inline]
    fn get_sa_iterations(&self, n: usize) -> usize {
        match NTier::from_n(n) {
            NTier::Critical => 30000 + n * 200,   // Heavy: 30000+ iterations
            NTier::Important => 15000 + n * 80,   // Medium: 15000+ iterations
            NTier::Medium => 8000 + n * 40,       // Light: 8000+ iterations
            NTier::Large => 4000 + n * 20,        // Minimal: 4000+ iterations
        }
    }

    /// Get adaptive search attempts based on n
    #[inline]
    fn get_search_attempts(&self, n: usize) -> usize {
        match NTier::from_n(n) {
            NTier::Critical => 500,   // More exploration for critical n
            NTier::Important => 300,  // Medium exploration
            NTier::Medium => 200,     // Standard
            NTier::Large => 150,      // Reduced for speed
        }
    }

    /// Get adaptive binary search precision based on n
    #[inline]
    fn get_binary_precision(&self, n: usize) -> f64 {
        match NTier::from_n(n) {
            NTier::Critical => 0.0001,   // Fine precision
            NTier::Important => 0.0005,  // Medium precision
            NTier::Medium => 0.001,      // Coarse
            NTier::Large => 0.002,       // Very coarse
        }
    }

    /// Get adaptive SA passes based on n
    #[inline]
    fn get_sa_passes(&self, n: usize) -> usize {
        match NTier::from_n(n) {
            NTier::Critical => 3,    // Triple pass for critical
            NTier::Important => 2,   // Double pass
            NTier::Medium => 2,      // Double pass
            NTier::Large => 1,       // Single pass
        }
    }

    /// Get adaptive early exit threshold based on n
    #[inline]
    fn get_early_exit_threshold(&self, n: usize) -> usize {
        match NTier::from_n(n) {
            NTier::Critical => 2000,  // Patient - wait longer for improvements
            NTier::Important => 1200, // Medium patience
            NTier::Medium => 800,     // Standard
            NTier::Large => 500,      // Quick exit
        }
    }

    /// Get number of rotation angles based on n
    #[inline]
    fn get_num_angles(&self, n: usize) -> usize {
        match NTier::from_n(n) {
            NTier::Critical => 16,   // Every 22.5 degrees
            NTier::Important => 12,  // Every 30 degrees
            NTier::Medium => 8,      // Every 45 degrees
            NTier::Large => 8,       // Every 45 degrees
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

            // ADAPTIVE: Run SA passes based on n-tier
            let passes = self.get_sa_passes(n);
            for pass in 0..passes {
                self.local_search(&mut trees, n, pass, passes, &mut rng);
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
                // ADAPTIVE: Use n-dependent precision
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
    /// ADAPTIVE: Scoring weights adjusted by n-tier
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

        // ADAPTIVE: Stronger balance penalty for small n
        let balance_weight = match NTier::from_n(n) {
            NTier::Critical => 0.20,   // Strong balance for small n
            NTier::Important => 0.15,
            NTier::Medium => 0.12,
            NTier::Large => 0.10,
        };
        let balance_penalty = (width - height).abs() * balance_weight;

        // Tertiary: preference for compact center (scaled by n)
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.008 / (n as f64).sqrt();

        // ADAPTIVE: Density bonus more important for small n
        let area = width * height;
        let density_weight = match NTier::from_n(n) {
            NTier::Critical => 0.02,
            NTier::Important => 0.015,
            NTier::Medium => 0.01,
            NTier::Large => 0.008,
        };
        let density_bonus = if area > 0.0 {
            -density_weight * (n as f64 / area).min(2.0)
        } else {
            0.0
        };

        side_score + balance_penalty + center_penalty + density_bonus
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    /// Returns angles in priority order
    /// ADAPTIVE: More angles for small n
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
        let num_angles = self.get_num_angles(n);

        match num_angles {
            16 => {
                // Every 22.5 degrees - full exploration for critical n
                let base: Vec<f64> = match n % 4 {
                    0 => vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0,
                              22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5],
                    1 => vec![90.0, 270.0, 0.0, 180.0, 135.0, 315.0, 45.0, 225.0,
                              112.5, 292.5, 22.5, 202.5, 157.5, 337.5, 67.5, 247.5],
                    2 => vec![180.0, 0.0, 270.0, 90.0, 225.0, 45.0, 315.0, 135.0,
                              202.5, 22.5, 292.5, 112.5, 247.5, 67.5, 337.5, 157.5],
                    _ => vec![270.0, 90.0, 180.0, 0.0, 315.0, 135.0, 225.0, 45.0,
                              292.5, 112.5, 202.5, 22.5, 337.5, 157.5, 247.5, 67.5],
                };
                base
            }
            12 => {
                // Every 30 degrees - good coverage
                let base: Vec<f64> = match n % 3 {
                    0 => vec![0.0, 90.0, 180.0, 270.0, 30.0, 60.0, 120.0, 150.0, 210.0, 240.0, 300.0, 330.0],
                    1 => vec![90.0, 180.0, 270.0, 0.0, 120.0, 150.0, 210.0, 240.0, 300.0, 330.0, 30.0, 60.0],
                    _ => vec![180.0, 270.0, 0.0, 90.0, 210.0, 240.0, 300.0, 330.0, 30.0, 60.0, 120.0, 150.0],
                };
                base
            }
            _ => {
                // 8 angles (every 45 degrees) - efficient for larger n
                let base = match n % 4 {
                    0 => vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0],
                    1 => vec![90.0, 270.0, 0.0, 180.0, 135.0, 315.0, 45.0, 225.0],
                    2 => vec![180.0, 0.0, 270.0, 90.0, 225.0, 45.0, 315.0, 135.0],
                    _ => vec![270.0, 90.0, 180.0, 0.0, 315.0, 135.0, 225.0, 45.0],
                };
                base
            }
        }
    }

    /// EVOLVED FUNCTION: Select direction angle for placement search
    /// ADAPTIVE: More sophisticated sampling for small n
    #[inline]
    fn select_direction(&self, n: usize, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.base_direction_samples;
        let tier = NTier::from_n(n);

        // ADAPTIVE: Strategy mix depends on n-tier
        let strategy = rng.gen::<f64>();

        match tier {
            NTier::Critical => {
                // For critical n: more structured exploration
                if strategy < 0.40 {
                    // Structured: evenly spaced with minimal jitter
                    let base_idx = rng.gen_range(0..num_dirs * 2);
                    let base = (base_idx as f64 / (num_dirs * 2) as f64) * 2.0 * PI;
                    base + rng.gen_range(-0.03..0.03)
                } else if strategy < 0.65 {
                    // Weighted random: favor corners and edges
                    loop {
                        let angle = rng.gen_range(0.0..2.0 * PI);
                        let corner_weight = ((4.0 * angle).sin().abs() + (4.0 * angle).cos().abs()) / 2.0;
                        if rng.gen::<f64>() < corner_weight.max(0.25) {
                            return angle;
                        }
                    }
                } else if strategy < 0.85 {
                    // Golden angle spiral
                    let golden_angle = PI * (3.0 - (5.0_f64).sqrt());
                    let base = (n as f64 * golden_angle) % (2.0 * PI);
                    let offset = rng.gen_range(0..16) as f64 * PI / 8.0;
                    (base + offset + rng.gen_range(-0.05..0.05)) % (2.0 * PI)
                } else {
                    // Fine-grained random for remaining exploration
                    rng.gen_range(0.0..2.0 * PI)
                }
            }
            _ => {
                // Standard strategy for larger n
                if strategy < 0.50 {
                    let base_idx = rng.gen_range(0..num_dirs);
                    let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
                    base + rng.gen_range(-0.06..0.06)
                } else if strategy < 0.75 {
                    loop {
                        let angle = rng.gen_range(0.0..2.0 * PI);
                        let corner_weight = ((4.0 * angle).sin().abs() + (4.0 * angle).cos().abs()) / 2.0;
                        if rng.gen::<f64>() < corner_weight.max(0.2) {
                            return angle;
                        }
                    }
                } else {
                    let golden_angle = PI * (3.0 - (5.0_f64).sqrt());
                    let base = (n as f64 * golden_angle) % (2.0 * PI);
                    let offset = rng.gen_range(0..8) as f64 * PI / 4.0;
                    (base + offset + rng.gen_range(-0.1..0.1)) % (2.0 * PI)
                }
            }
        }
    }

    /// EVOLVED FUNCTION: Local search with simulated annealing
    /// ADAPTIVE: Iterations and parameters scale with n-tier
    fn local_search(&self, trees: &mut Vec<PlacedTree>, n: usize, pass: usize, total_passes: usize, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config: Vec<PlacedTree> = trees.clone();

        let tier = NTier::from_n(n);

        // ADAPTIVE: Temperature and cooling based on tier and pass
        let temp_multiplier = match (tier, pass) {
            (NTier::Critical, 0) => 1.2,   // Higher temp for critical first pass
            (NTier::Critical, 1) => 0.6,
            (NTier::Critical, _) => 0.3,   // Fine-tuning on third pass
            (NTier::Important, 0) => 1.0,
            (NTier::Important, _) => 0.4,
            (_, 0) => 0.8,
            (_, _) => 0.4,
        };
        let mut temp = self.config.sa_initial_temp * temp_multiplier;

        // ADAPTIVE: Iterations based on n-tier and pass
        let base_iterations = self.get_sa_iterations(n);
        let iterations = match pass {
            0 => base_iterations,
            1 => base_iterations * 2 / 3,
            _ => base_iterations / 2,
        };

        // ADAPTIVE: Early exit threshold based on n-tier
        let early_exit_threshold = self.get_early_exit_threshold(n);

        // Track iterations without improvement
        let mut iterations_without_improvement = 0;

        // Pre-compute boundary trees once per batch
        let mut boundary_cache_iter = 0;
        let mut boundary_indices: Vec<usize> = Vec::new();

        // ADAPTIVE: Cooling rate based on tier
        let cooling_rate = match tier {
            NTier::Critical => 0.99995,   // Slower cooling for more exploration
            NTier::Important => 0.9999,
            NTier::Medium => 0.9998,
            NTier::Large => 0.9996,       // Faster cooling
        };

        for iter in 0..iterations {
            // Early exit when no improvement
            if iterations_without_improvement >= early_exit_threshold {
                break;
            }

            // Update boundary cache periodically
            let cache_update_freq = match tier {
                NTier::Critical => 300,   // More frequent updates
                NTier::Important => 400,
                NTier::Medium => 500,
                NTier::Large => 600,
            };
            if iter == 0 || iter - boundary_cache_iter >= cache_update_freq {
                boundary_indices = self.find_boundary_trees(trees);
                boundary_cache_iter = iter;
            }

            // ADAPTIVE: Boundary focus based on tier
            let boundary_prob = match tier {
                NTier::Critical => 0.75,  // Higher focus on boundary
                NTier::Important => 0.70,
                NTier::Medium => 0.65,
                NTier::Large => 0.60,
            };
            let idx = if !boundary_indices.is_empty() && rng.gen::<f64>() < boundary_prob {
                boundary_indices[rng.gen_range(0..boundary_indices.len())]
            } else {
                rng.gen_range(0..trees.len())
            };

            let old_tree = trees[idx].clone();

            // EVOLVED: Move operator with adaptive parameters
            let success = self.sa_move(trees, idx, temp, &boundary_indices, n, rng);

            if success {
                let new_side = compute_side_length(trees);
                let delta = new_side - current_side;

                // Metropolis criterion
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

            temp = (temp * cooling_rate).max(self.config.sa_min_temp);
        }

        // Restore best configuration found
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
    /// ADAPTIVE: Move types and scales adjusted by n-tier
    /// Returns true if move is valid (no overlap)
    #[inline]
    fn sa_move(
        &self,
        trees: &mut [PlacedTree],
        idx: usize,
        temp: f64,
        boundary_indices: &[usize],
        n: usize,
        rng: &mut impl Rng,
    ) -> bool {
        let old = &trees[idx];
        let old_x = old.x;
        let old_y = old.y;
        let old_angle = old.angle_deg;

        let is_boundary = boundary_indices.contains(&idx);
        let tier = NTier::from_n(n);

        // ADAPTIVE: Move type selection based on tier and boundary status
        let move_type = if is_boundary {
            match tier {
                NTier::Critical => {
                    // More varied moves for critical n
                    match rng.gen_range(0..12) {
                        0..=4 => 0,  // Inward translation (42%)
                        5..=6 => 1,  // Rotation (17%)
                        7..=8 => 2,  // Center pull (17%)
                        9 => 3,      // Small nudge (8%)
                        10 => 4,     // Translate + rotate (8%)
                        _ => 5,      // Fine rotation (8%)
                    }
                }
                _ => {
                    match rng.gen_range(0..10) {
                        0..=3 => 0,
                        4..=5 => 1,
                        6..=7 => 2,
                        8 => 3,
                        _ => 4,
                    }
                }
            }
        } else {
            rng.gen_range(0..8)
        };

        // ADAPTIVE: Move scale based on tier
        let scale_mult = match tier {
            NTier::Critical => 0.8,   // Finer moves for precision
            NTier::Important => 0.9,
            NTier::Medium => 1.0,
            NTier::Large => 1.1,      // Larger moves for speed
        };

        match move_type {
            0 => {
                // Inward translation (toward reducing bounding box)
                let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                let (bx1, by1, bx2, by2) = trees[idx].bounds();

                let scale = self.config.translation_scale * scale_mult * (0.2 + temp * 2.0);
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
                    let scale = self.config.center_pull_strength * scale_mult * (0.4 + temp * 1.5);
                    let dx = -old_x / mag * scale;
                    let dy = -old_y / mag * scale;
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                } else {
                    return false;
                }
            }
            3 => {
                // Small nudge for fine-tuning
                let nudge_scale = match tier {
                    NTier::Critical => 0.010,   // Finer nudge
                    NTier::Important => 0.012,
                    _ => 0.015,
                };
                let scale = nudge_scale * (0.5 + temp);
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            4 => {
                // Translate + rotate combo
                let scale = self.config.translation_scale * scale_mult * 0.4;
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                let dangle = rng.gen_range(-45.0..45.0);
                let new_angle = (old_angle + dangle).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, new_angle);
            }
            5 => {
                // Fine rotation
                let angle_step = match tier {
                    NTier::Critical => 22.5,   // Finer rotation
                    NTier::Important => 30.0,
                    _ => 45.0,
                };
                let delta = if rng.gen() { angle_step } else { -angle_step };
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            6 => {
                // Polar move (radial in/out)
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.08 {
                    let delta_r = rng.gen_range(-0.08..0.08) * scale_mult * (1.0 + temp);
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
                    let delta_angle = rng.gen_range(-0.2..0.2) * scale_mult * (1.0 + temp);
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
