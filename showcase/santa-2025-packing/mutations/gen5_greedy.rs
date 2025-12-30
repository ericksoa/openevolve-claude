//! Evolved Packing Algorithm - Generation 5 GREEDY HILL CLIMB
//!
//! This module contains the evolved packing heuristics.
//! The code is designed to be mutated by LLM-guided evolution.
//!
//! Evolution targets:
//! - placement_score(): How to score candidate placements
//! - select_angles(): Which rotation angles to try
//! - select_direction(): How to choose placement directions
//! - greedy_move(): Local search move operators
//!
//! MUTATION STRATEGY: GREEDY HILL CLIMB (Gen5)
//! Hypothesis: SA overhead may not be worth it. Pure greedy with restarts could be faster and equally effective:
//!
//! Key changes from Gen4:
//! - No simulated annealing - only accept strictly improving moves
//! - Multiple random restarts (5 restarts per optimization)
//! - Each restart: 5000 greedy hill climbing iterations
//! - Move operators: small translations, rotations, center pulls (same as Gen4)
//! - Take the best result across all restarts
//! - Should be much faster, allowing more exploration
//!
//! Goal: Match or beat Gen4's 98.37 with faster, simpler optimization
//! Target: < 98.37 at n=200

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Evolved packing configuration
/// These parameters are tuned through evolution
pub struct EvolvedConfig {
    // Search parameters
    pub search_attempts: usize,
    pub direction_samples: usize,

    // Greedy hill climbing parameters
    pub num_restarts: usize,
    pub iterations_per_restart: usize,

    // Move parameters
    pub translation_scale: f64,
    pub rotation_granularity: f64,
    pub center_pull_strength: f64,

    // Early exit threshold (no improvement for this many iterations)
    pub early_exit_threshold: usize,
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen5 GREEDY: Fast greedy hill climbing with restarts
        Self {
            search_attempts: 250,            // Same as Gen4
            direction_samples: 64,           // Same as Gen4
            num_restarts: 5,                 // 5 random restarts
            iterations_per_restart: 5000,    // 5000 greedy iterations per restart
            translation_scale: 0.08,         // Same as Gen4
            rotation_granularity: 45.0,      // 8 angles (every 45 degrees)
            center_pull_strength: 0.06,      // Same as Gen4
            early_exit_threshold: 500,       // Exit early if stuck
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

            // GREEDY: Run greedy hill climbing with multiple restarts
            trees = self.greedy_optimize(&trees, n, &mut rng);

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

        // Primary: minimize side length (most important)
        let side_score = side;

        // Secondary: prefer balanced aspect ratio
        let balance_penalty = (width - height).abs() * 0.15;

        // Tertiary: slight preference for compact center (scaled by n)
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.008 / (n as f64).sqrt();

        // Density bonus
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

    /// GREEDY HILL CLIMB: Optimize with multiple random restarts
    /// Key difference from SA: only accept strictly improving moves
    fn greedy_optimize(&self, trees: &[PlacedTree], _n: usize, rng: &mut impl Rng) -> Vec<PlacedTree> {
        if trees.len() <= 1 {
            return trees.to_vec();
        }

        let mut global_best = trees.to_vec();
        let mut global_best_side = compute_side_length(&global_best);

        // Run multiple restarts
        for restart in 0..self.config.num_restarts {
            // Start from slightly perturbed version (except first restart)
            let mut current = if restart == 0 {
                trees.to_vec()
            } else {
                self.perturb_configuration(trees, rng)
            };
            let mut current_side = compute_side_length(&current);

            // Track iterations without improvement for early exit
            let mut no_improvement_count = 0;

            // Greedy hill climbing iterations
            for _ in 0..self.config.iterations_per_restart {
                if no_improvement_count >= self.config.early_exit_threshold {
                    break;
                }

                // Find boundary trees (trees that contribute to bounding box)
                let boundary_indices = self.find_boundary_trees(&current);

                // Pick a tree to move (70% boundary, 30% random)
                let idx = if !boundary_indices.is_empty() && rng.gen::<f64>() < 0.70 {
                    boundary_indices[rng.gen_range(0..boundary_indices.len())]
                } else {
                    rng.gen_range(0..current.len())
                };

                let old_tree = current[idx].clone();

                // Try a greedy move
                let success = self.greedy_move(&mut current, idx, &boundary_indices, rng);

                if success {
                    let new_side = compute_side_length(&current);
                    // GREEDY: Only accept strictly improving moves
                    if new_side < current_side {
                        current_side = new_side;
                        no_improvement_count = 0;
                    } else {
                        // Reject non-improving move
                        current[idx] = old_tree;
                        no_improvement_count += 1;
                    }
                } else {
                    current[idx] = old_tree;
                    no_improvement_count += 1;
                }
            }

            // Update global best if this restart found better solution
            if current_side < global_best_side {
                global_best = current;
                global_best_side = current_side;
            }
        }

        global_best
    }

    /// Perturb a configuration for restart diversification
    fn perturb_configuration(&self, trees: &[PlacedTree], rng: &mut impl Rng) -> Vec<PlacedTree> {
        let mut perturbed = trees.to_vec();

        // Apply small random perturbations to a subset of trees
        let num_to_perturb = (trees.len() / 3).max(1);

        for _ in 0..num_to_perturb {
            let idx = rng.gen_range(0..perturbed.len());
            let old = &perturbed[idx];

            // Small random translation
            let scale = 0.05;
            let dx = rng.gen_range(-scale..scale);
            let dy = rng.gen_range(-scale..scale);

            // Optionally rotate by 90 degrees
            let new_angle = if rng.gen::<f64>() < 0.3 {
                (old.angle_deg + 90.0).rem_euclid(360.0)
            } else {
                old.angle_deg
            };

            let new_tree = PlacedTree::new(old.x + dx, old.y + dy, new_angle);

            // Only apply if valid
            perturbed[idx] = new_tree;
            if has_overlap(&perturbed, idx) {
                perturbed[idx] = old.clone();
            }
        }

        perturbed
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

    /// GREEDY move operator
    /// Similar to SA moves but simpler (no temperature dependency)
    /// Returns true if move is valid (no overlap)
    #[inline]
    fn greedy_move(
        &self,
        trees: &mut [PlacedTree],
        idx: usize,
        boundary_indices: &[usize],
        rng: &mut impl Rng,
    ) -> bool {
        let old = &trees[idx];
        let old_x = old.x;
        let old_y = old.y;
        let old_angle = old.angle_deg;

        let is_boundary = boundary_indices.contains(&idx);

        // 8 move types with special handling for boundary trees
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
                let scale = self.config.translation_scale;
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
                    let scale = self.config.center_pull_strength;
                    let dx = -old_x / mag * scale;
                    let dy = -old_y / mag * scale;
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                } else {
                    return false;
                }
            }
            3 => {
                // Small nudge for fine-tuning
                let scale = 0.015;
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
                    let delta_r = rng.gen_range(-0.08..0.08);
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
                    let delta_angle = rng.gen_range(-0.2..0.2);
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
