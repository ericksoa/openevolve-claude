//! Evolved Packing Algorithm - Generation 3 Multi-Restart (Fixed)
//!
//! MUTATION STRATEGY: MULTI-RESTART WITH GEN2_EXTREME BASE
//! Combines the aggressive parameters from gen2_extreme with the multi-restart
//! exploration strategy from gen2_multistart (with borrow checker fix).
//!
//! Key innovations:
//! 1. Run 8 independent attempts per n with different seeds
//! 2. Each attempt uses varied temperature, iterations, angle priority
//! 3. Base parameters from gen2_extreme for quality
//! 4. Fixed perturb_solution to avoid borrow checker issues
//! 5. Keep the best result from all restarts
//!
//! Fix for gen2_multistart compile error:
//! - Original error: borrowing `perturbed` mutably while also cloning it
//! - Solution: Use indices and collect changes separately, then apply them

use crate::{Packing, PlacedTree};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::f64::consts::PI;

/// Multi-restart configuration with gen2_extreme base parameters
pub struct EvolvedConfig {
    // Multi-restart parameters
    pub num_restarts: usize,

    // Search parameters (from gen2_extreme)
    pub search_attempts: usize,
    pub direction_samples: usize,

    // Simulated annealing (base values from gen2_extreme)
    pub sa_iterations_base: usize,
    pub sa_initial_temp_base: f64,
    pub sa_cooling_rate: f64,
    pub sa_min_temp: f64,
    pub sa_passes: usize,

    // Move parameters
    pub translation_scale: f64,
    pub rotation_granularity: f64,
    pub center_pull_strength: f64,
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        Self {
            // Multi-restart: 8 attempts for exploration
            num_restarts: 8,

            // From gen2_extreme (slightly reduced since we have restarts)
            search_attempts: 150,        // Was 200, reduced for multi-restart budget
            direction_samples: 48,       // Keep 48 from gen2_extreme

            // SA parameters from gen2_extreme (base values)
            sa_iterations_base: 15000,   // Was 20000, reduced for budget
            sa_initial_temp_base: 0.6,   // From gen2_extreme
            sa_cooling_rate: 0.9999,     // From gen2_extreme (slow cooling)
            sa_min_temp: 0.00001,        // From gen2_extreme (100x lower)
            sa_passes: 2,                // From gen2_extreme (double pass)

            // Move parameters from gen2_extreme
            translation_scale: 0.08,
            rotation_granularity: 22.5,  // Finer rotations
            center_pull_strength: 0.05,
        }
    }
}

/// Restart variant parameters for diversity across attempts
struct RestartVariant {
    seed: u64,
    temp_multiplier: f64,
    iterations_multiplier: f64,
    angle_priority: usize,
    direction_bias: f64,
    fresh_start: bool,
}

impl RestartVariant {
    fn generate(restart_idx: usize, n: usize, master_seed: u64) -> Self {
        let seed = master_seed.wrapping_add(restart_idx as u64 * 12345 + n as u64 * 67890);

        // Diverse parameter combinations across 8 restarts
        let variants = [
            // (temp_mult, iter_mult, angle_prio, dir_bias, fresh)
            (1.0, 1.0, 0, 0.7, false),   // Standard, use previous best
            (1.3, 1.2, 1, 0.5, false),   // Higher temp, more iters, explore
            (0.8, 0.9, 2, 0.8, true),    // Lower temp, fresh start
            (1.5, 1.4, 3, 0.6, false),   // Aggressive exploration
            (0.7, 1.3, 0, 0.9, true),    // Cool start, many iters, fresh
            (1.2, 1.0, 1, 0.4, false),   // Slightly warm, corner bias
            (1.0, 1.5, 2, 0.7, true),    // Many iterations, fresh start
            (1.4, 0.8, 3, 0.5, false),   // Hot start, quick adapt
        ];

        let v = &variants[restart_idx % variants.len()];

        RestartVariant {
            seed,
            temp_multiplier: v.0,
            iterations_multiplier: v.1,
            angle_priority: v.2,
            direction_bias: v.3,
            fresh_start: v.4,
        }
    }
}

/// Main evolved packer with multi-restart optimization
pub struct EvolvedPacker {
    pub config: EvolvedConfig,
}

impl Default for EvolvedPacker {
    fn default() -> Self {
        Self { config: EvolvedConfig::default() }
    }
}

impl EvolvedPacker {
    /// Pack all n from 1 to max_n with multi-restart optimization
    pub fn pack_all(&self, max_n: usize) -> Vec<Packing> {
        let mut master_rng = rand::thread_rng();
        let master_seed: u64 = master_rng.gen();

        let mut packings: Vec<Packing> = Vec::with_capacity(max_n);
        let mut prev_best_trees: Vec<PlacedTree> = Vec::new();

        for n in 1..=max_n {
            let mut best_trees: Option<Vec<PlacedTree>> = None;
            let mut best_side = f64::INFINITY;

            // Run multiple independent restarts
            for restart_idx in 0..self.config.num_restarts {
                let variant = RestartVariant::generate(restart_idx, n, master_seed);
                let mut rng = StdRng::seed_from_u64(variant.seed);

                // Decide starting point
                let starting_trees = if n == 1 {
                    Vec::new()
                } else if variant.fresh_start {
                    // Fresh start: rebuild from previous packings
                    self.rebuild_from_packings(&packings, n - 1, &variant, &mut rng)
                } else {
                    // Use previous best (possibly perturbed)
                    if restart_idx > 0 && restart_idx % 3 == 0 {
                        // Every 3rd restart: perturb previous solution
                        self.perturb_solution(&prev_best_trees, &mut rng)
                    } else {
                        prev_best_trees.clone()
                    }
                };

                // Place new tree with variant-specific parameters
                let mut trees = starting_trees;
                let new_tree = self.find_placement(&trees, n, max_n, &variant, &mut rng);
                trees.push(new_tree);

                // Run SA with multi-pass (from gen2_extreme)
                for pass in 0..self.config.sa_passes {
                    self.local_search(&mut trees, n, &variant, pass, &mut rng);
                }

                // Evaluate and keep if best
                let side = compute_side_length(&trees);
                if side < best_side {
                    best_side = side;
                    best_trees = Some(trees);
                }
            }

            // Store the best result from all restarts
            let trees = best_trees.unwrap();
            let mut packing = Packing::new();
            for t in &trees {
                packing.trees.push(t.clone());
            }
            packings.push(packing);
            prev_best_trees = trees;
        }

        packings
    }

    /// Rebuild solution from previous packings (for fresh starts)
    fn rebuild_from_packings(
        &self,
        packings: &[Packing],
        up_to_n: usize,
        variant: &RestartVariant,
        rng: &mut impl Rng,
    ) -> Vec<PlacedTree> {
        if up_to_n == 0 || packings.is_empty() {
            return Vec::new();
        }

        let mut trees = Vec::new();

        for i in 1..=up_to_n {
            if i == 1 {
                trees.push(PlacedTree::new(0.0, 0.0, 90.0));
            } else {
                let new_tree = self.find_placement(&trees, i, up_to_n, variant, rng);
                trees.push(new_tree);
            }
        }

        // Quick SA pass to refine the rebuild
        let quick_variant = RestartVariant {
            iterations_multiplier: 0.3,
            ..*variant
        };
        self.local_search(&mut trees, up_to_n, &quick_variant, 0, rng);

        trees
    }

    /// Perturb an existing solution for diversity
    /// FIXED: Use indices to avoid borrow checker issues
    fn perturb_solution(&self, trees: &[PlacedTree], rng: &mut impl Rng) -> Vec<PlacedTree> {
        let mut perturbed: Vec<PlacedTree> = trees.to_vec();

        // First pass: collect which trees to perturb and how
        let mut changes: Vec<(usize, f64, f64)> = Vec::new();

        for i in 0..perturbed.len() {
            if rng.gen::<f64>() < 0.3 {
                // 30% chance to perturb each tree
                let dx = rng.gen_range(-0.02..0.02);
                let dy = rng.gen_range(-0.02..0.02);
                changes.push((i, dx, dy));
            }
        }

        // Second pass: apply valid changes
        for (idx, dx, dy) in changes {
            let old_tree = &perturbed[idx];
            let new_tree = PlacedTree::new(old_tree.x + dx, old_tree.y + dy, old_tree.angle_deg);

            // Check if new position is valid (no overlaps)
            let mut valid = true;
            for (j, other) in perturbed.iter().enumerate() {
                if j != idx && new_tree.overlaps(other) {
                    valid = false;
                    break;
                }
            }

            if valid {
                perturbed[idx] = new_tree;
            }
        }

        perturbed
    }

    /// Find best placement for new tree (variant-aware)
    fn find_placement(
        &self,
        existing: &[PlacedTree],
        n: usize,
        _max_n: usize,
        variant: &RestartVariant,
        rng: &mut impl Rng,
    ) -> PlacedTree {
        if existing.is_empty() {
            return PlacedTree::new(0.0, 0.0, 90.0);
        }

        let mut best_tree = PlacedTree::new(0.0, 0.0, 90.0);
        let mut best_score = f64::INFINITY;

        let angles = self.select_angles(n, variant.angle_priority);

        for _ in 0..self.config.search_attempts {
            let dir = self.select_direction(n, variant.direction_bias, rng);
            let vx = dir.cos();
            let vy = dir.sin();

            for &tree_angle in &angles {
                // Binary search for closest valid position
                // From gen2_extreme: 6x finer precision
                let mut low = 0.0;
                let mut high = 12.0;

                while high - low > 0.0005 {
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

    /// Score a placement (lower is better) - from gen2_extreme
    #[inline]
    fn placement_score(&self, tree: &PlacedTree, existing: &[PlacedTree], n: usize) -> f64 {
        let (min_x, min_y, max_x, max_y) = tree.bounds();

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

        // Secondary: prefer balanced aspect ratio (from gen2_extreme)
        let balance_weight = 0.10 + 0.05 * (1.0 - (n as f64 / 200.0).min(1.0));
        let balance_penalty = (width - height).abs() * balance_weight;

        // Tertiary: slight preference for compact center
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.008 / (n as f64).sqrt();

        // Density bonus from gen2_extreme
        let area = width * height;
        let density_bonus = if area > 0.0 {
            -0.005 * (n as f64 / area).min(2.0)
        } else {
            0.0
        };

        side_score + balance_penalty + center_penalty + density_bonus
    }

    /// Select rotation angles (12 angles from gen2_extreme with variant-based priority)
    #[inline]
    fn select_angles(&self, n: usize, priority: usize) -> Vec<f64> {
        // 12 angles from gen2_extreme with different orderings
        let orderings = [
            vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0, 30.0, 60.0, 120.0, 150.0],
            vec![90.0, 270.0, 0.0, 180.0, 135.0, 315.0, 45.0, 225.0, 60.0, 120.0, 240.0, 300.0],
            vec![180.0, 0.0, 270.0, 90.0, 225.0, 45.0, 315.0, 135.0, 150.0, 210.0, 330.0, 30.0],
            vec![270.0, 90.0, 180.0, 0.0, 315.0, 135.0, 225.0, 45.0, 240.0, 300.0, 60.0, 120.0],
        ];

        let idx = (n + priority) % orderings.len();
        orderings[idx].clone()
    }

    /// Select direction angle with configurable bias (from gen2_extreme strategies)
    #[inline]
    fn select_direction(&self, n: usize, corner_bias: f64, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;
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
                let corner_weight = ((4.0 * angle).sin().abs() + (4.0 * angle).cos().abs()) / 2.0;
                let threshold = 0.2 + 0.15 * (1.0 - (n as f64 / 200.0).min(1.0));
                if rng.gen::<f64>() < corner_weight.max(threshold) * corner_bias + (1.0 - corner_bias) {
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

    /// Local search with simulated annealing (variant-aware, multi-pass)
    fn local_search(
        &self,
        trees: &mut Vec<PlacedTree>,
        n: usize,
        variant: &RestartVariant,
        pass: usize,
        rng: &mut impl Rng,
    ) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config: Vec<PlacedTree> = trees.clone();

        // Adjust temperature based on pass number (from gen2_extreme)
        let pass_multiplier = if pass == 0 { 1.0 } else { 0.3 };
        let temp_multiplier = variant.temp_multiplier * pass_multiplier;
        let mut temp = self.config.sa_initial_temp_base * temp_multiplier;

        // Calculate iterations with variant scaling
        let base_iters = if pass == 0 {
            self.config.sa_iterations_base + n * 80
        } else {
            self.config.sa_iterations_base / 2 + n * 40
        };
        let iterations = ((base_iters as f64) * variant.iterations_multiplier) as usize;

        for iter in 0..iterations {
            // Prefer moving boundary trees (from gen2_extreme)
            let idx = self.select_tree_to_move(trees, rng);
            let old_tree = trees[idx].clone();

            let success = self.sa_move(trees, idx, temp, iter, rng);

            if success {
                let new_side = compute_side_length(trees);
                let delta = new_side - current_side;

                // Metropolis criterion
                if delta <= 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                    current_side = new_side;
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

            temp = (temp * self.config.sa_cooling_rate).max(self.config.sa_min_temp);
        }

        // Restore best configuration found (from gen2_extreme)
        if best_side < compute_side_length(trees) {
            *trees = best_config;
        }
    }

    /// Select tree to move with preference for boundary trees (from gen2_extreme)
    #[inline]
    fn select_tree_to_move(&self, trees: &[PlacedTree], rng: &mut impl Rng) -> usize {
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

    /// SA move operator with 10 move types (from gen2_extreme)
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
        println!("Gen3 Multi-restart score for n=1..50: {:.4}", score);
    }

    #[test]
    fn test_perturb_solution() {
        // Test that perturb_solution compiles and works
        let packer = EvolvedPacker::default();
        let mut rng = rand::thread_rng();

        // Create a small valid packing
        let trees = vec![
            PlacedTree::new(0.0, 0.0, 90.0),
            PlacedTree::new(1.5, 0.0, 0.0),
        ];

        // Perturb should not panic
        let perturbed = packer.perturb_solution(&trees, &mut rng);
        assert_eq!(perturbed.len(), trees.len());
    }

    #[test]
    fn test_restart_variants() {
        // Verify restart variants are diverse
        for n in [5, 10, 20] {
            let mut seeds = std::collections::HashSet::new();
            for restart_idx in 0..8 {
                let variant = RestartVariant::generate(restart_idx, n, 42);
                seeds.insert(variant.seed);
            }
            assert_eq!(seeds.len(), 8, "All restart variants should have unique seeds");
        }
    }
}
