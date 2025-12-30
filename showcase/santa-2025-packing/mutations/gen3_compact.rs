//! Evolved Packing Algorithm - Generation 3: Greedy Compact
//!
//! This module combines Gen2 EXTREME intensification with post-SA greedy compaction.
//!
//! MUTATION STRATEGY: GREEDY COMPACTION + SHAKE (Gen3)
//! Building on Gen2 EXTREME with additional compaction phases:
//!
//! Key additions over Gen2 EXTREME:
//! 1. Greedy boundary compaction after SA:
//!    - Find trees on the boundary (touching min_x, max_x, min_y, max_y)
//!    - Binary search for furthest valid position toward center
//!    - Repeat until no improvement > 0.001
//!
//! 2. Shake operator during SA:
//!    - Every 10000 iterations, slightly perturb all trees
//!    - Run quick greedy compaction pass
//!    - Helps escape local optima
//!
//! 3. Final global compaction:
//!    - After SA completes, run boundary compaction iteratively
//!    - Use binary search to find optimal positions
//!
//! Evolution targets:
//! - greedy_boundary_compaction(): New - slides boundary trees toward center
//! - shake_and_compact(): New - perturb + compact to escape local optima
//! - find_boundary_tree_indices(): Helper to avoid borrow issues

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Evolved packing configuration - Gen3 with compaction parameters
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

    // Gen2 EXTREME: Multi-pass settings
    pub sa_passes: usize,

    // Gen3: Compaction parameters
    pub compaction_tolerance: f64,      // Improvement threshold to continue
    pub compaction_binary_precision: f64, // Binary search precision
    pub shake_interval: usize,          // SA iterations between shakes
    pub shake_magnitude: f64,           // Perturbation size
    pub max_compaction_rounds: usize,   // Max rounds of boundary compaction
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        Self {
            // Gen2 EXTREME base parameters
            search_attempts: 200,
            direction_samples: 48,
            sa_iterations: 20000,
            sa_initial_temp: 0.6,
            sa_cooling_rate: 0.9999,
            sa_min_temp: 0.00001,
            translation_scale: 0.08,
            rotation_granularity: 22.5,
            center_pull_strength: 0.05,
            sa_passes: 2,

            // Gen3: Compaction parameters
            compaction_tolerance: 0.001,        // Continue if improvement > this
            compaction_binary_precision: 0.001, // Binary search precision
            shake_interval: 10000,              // Shake every 10k iterations
            shake_magnitude: 0.02,              // Small perturbation
            max_compaction_rounds: 50,          // Max compaction iterations
        }
    }
}

/// Main evolved packer with greedy compaction
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

            // Gen2 EXTREME: Run multiple SA passes with shake integration
            for pass in 0..self.config.sa_passes {
                self.local_search_with_shake(&mut trees, n, pass, &mut rng);
            }

            // Gen3: Post-SA greedy boundary compaction
            self.greedy_boundary_compaction(&mut trees);

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

    /// Gen3 FUNCTION: Greedy boundary compaction
    /// Repeatedly moves boundary trees toward center until no improvement
    fn greedy_boundary_compaction(&self, trees: &mut Vec<PlacedTree>) {
        if trees.len() <= 1 {
            return;
        }

        for _round in 0..self.config.max_compaction_rounds {
            let initial_side = compute_side_length(trees);
            let mut improved = false;

            // Get bounding box
            let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
            let center_x = (min_x + max_x) / 2.0;
            let center_y = (min_y + max_y) / 2.0;

            // Find boundary tree indices (by index to avoid borrow issues)
            let boundary_indices = self.find_boundary_tree_indices(trees, min_x, min_y, max_x, max_y);

            // Try to move each boundary tree toward center
            for &idx in &boundary_indices {
                let old_tree = trees[idx].clone();

                // Direction toward center
                let dx = center_x - old_tree.x;
                let dy = center_y - old_tree.y;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist < 0.01 {
                    continue; // Already at center
                }

                let dir_x = dx / dist;
                let dir_y = dy / dist;

                // Binary search for furthest valid position toward center
                let improvement = self.binary_search_toward_center(
                    trees, idx, &old_tree, dir_x, dir_y, dist, initial_side
                );

                if improvement > self.config.compaction_tolerance {
                    improved = true;
                }
            }

            if !improved {
                break; // No more improvement possible
            }
        }
    }

    /// Find indices of trees touching the boundary
    /// Returns indices rather than references to avoid borrow issues
    #[inline]
    fn find_boundary_tree_indices(
        &self,
        trees: &[PlacedTree],
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
    ) -> Vec<usize> {
        let eps = 0.01;
        let mut indices = Vec::new();

        for (i, tree) in trees.iter().enumerate() {
            let (bx1, by1, bx2, by2) = tree.bounds();
            if (bx1 - min_x).abs() < eps
                || (bx2 - max_x).abs() < eps
                || (by1 - min_y).abs() < eps
                || (by2 - max_y).abs() < eps
            {
                indices.push(i);
            }
        }

        indices
    }

    /// Binary search for best valid position toward center
    /// Returns the improvement achieved (or 0 if no improvement)
    fn binary_search_toward_center(
        &self,
        trees: &mut [PlacedTree],
        idx: usize,
        old_tree: &PlacedTree,
        dir_x: f64,
        dir_y: f64,
        max_dist: f64,
        initial_side: f64,
    ) -> f64 {
        let mut best_tree = old_tree.clone();
        let mut best_side = initial_side;

        // Binary search for furthest valid position
        let mut low = 0.0;
        let mut high = max_dist;

        while high - low > self.config.compaction_binary_precision {
            let mid = (low + high) / 2.0;
            let new_x = old_tree.x + dir_x * mid;
            let new_y = old_tree.y + dir_y * mid;
            let candidate = PlacedTree::new(new_x, new_y, old_tree.angle_deg);

            trees[idx] = candidate.clone();

            if !has_overlap(trees, idx) {
                let new_side = compute_side_length(trees);
                if new_side < best_side {
                    best_side = new_side;
                    best_tree = candidate;
                }
                low = mid; // Can move further
            } else {
                high = mid; // Too far, back off
            }
        }

        // Apply best position found
        trees[idx] = best_tree;
        initial_side - best_side
    }

    /// Gen3 FUNCTION: Local search with integrated shake operator
    fn local_search_with_shake(
        &self,
        trees: &mut Vec<PlacedTree>,
        n: usize,
        pass: usize,
        rng: &mut impl Rng,
    ) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config: Vec<PlacedTree> = trees.clone();

        // Adjust temperature based on pass number
        let temp_multiplier = if pass == 0 { 1.0 } else { 0.3 };
        let mut temp = self.config.sa_initial_temp * temp_multiplier;

        let base_iterations = if pass == 0 {
            self.config.sa_iterations + n * 100
        } else {
            self.config.sa_iterations / 2 + n * 50
        };

        for iter in 0..base_iterations {
            // Gen3: Shake and compact at intervals
            if iter > 0 && iter % self.config.shake_interval == 0 {
                self.shake_and_compact(trees, rng);
                current_side = compute_side_length(trees);
                if current_side < best_side {
                    best_side = current_side;
                    best_config = trees.clone();
                }
            }

            // Select tree to move (prefer boundary trees)
            let idx = self.select_tree_to_move(trees, rng);
            let old_tree = trees[idx].clone();

            // Apply SA move
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

        // Restore best configuration
        if best_side < compute_side_length(trees) {
            *trees = best_config;
        }
    }

    /// Gen3 FUNCTION: Shake and compact to escape local optima
    fn shake_and_compact(&self, trees: &mut Vec<PlacedTree>, rng: &mut impl Rng) {
        if trees.len() <= 2 {
            return;
        }

        let initial_side = compute_side_length(trees);
        let backup = trees.clone();

        // Slightly perturb all trees
        for i in 0..trees.len() {
            let old = &trees[i];
            let dx = rng.gen_range(-self.config.shake_magnitude..self.config.shake_magnitude);
            let dy = rng.gen_range(-self.config.shake_magnitude..self.config.shake_magnitude);
            trees[i] = PlacedTree::new(old.x + dx, old.y + dy, old.angle_deg);
        }

        // Check if shake caused overlaps
        let mut valid = true;
        for i in 0..trees.len() {
            if has_overlap(trees, i) {
                valid = false;
                break;
            }
        }

        if !valid {
            *trees = backup;
            return;
        }

        // Quick greedy compaction pass
        self.quick_compaction_pass(trees);

        // Only keep if improved
        if compute_side_length(trees) > initial_side {
            *trees = backup;
        }
    }

    /// Quick compaction pass (single round, all trees)
    fn quick_compaction_pass(&self, trees: &mut Vec<PlacedTree>) {
        if trees.len() <= 1 {
            return;
        }

        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
        let center_x = (min_x + max_x) / 2.0;
        let center_y = (min_y + max_y) / 2.0;

        for idx in 0..trees.len() {
            let old_tree = trees[idx].clone();
            let dx = center_x - old_tree.x;
            let dy = center_y - old_tree.y;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < 0.01 {
                continue;
            }

            let dir_x = dx / dist;
            let dir_y = dy / dist;

            // Try a few step sizes
            let steps = [0.5, 0.3, 0.2, 0.1, 0.05];
            let mut best_tree = old_tree.clone();
            let initial_side = compute_side_length(trees);
            let mut best_side = initial_side;

            for &factor in &steps {
                let step = dist * factor;
                let new_x = old_tree.x + dir_x * step;
                let new_y = old_tree.y + dir_y * step;
                let candidate = PlacedTree::new(new_x, new_y, old_tree.angle_deg);

                trees[idx] = candidate.clone();
                if !has_overlap(trees, idx) {
                    let new_side = compute_side_length(trees);
                    if new_side < best_side {
                        best_side = new_side;
                        best_tree = candidate;
                    }
                }
            }

            trees[idx] = best_tree;
        }
    }

    /// Select tree to move with preference for boundary trees (from Gen2 EXTREME)
    #[inline]
    fn select_tree_to_move(&self, trees: &[PlacedTree], rng: &mut impl Rng) -> usize {
        if trees.len() <= 2 || rng.gen::<f64>() < 0.7 {
            return rng.gen_range(0..trees.len());
        }

        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
        let boundary_indices = self.find_boundary_tree_indices(trees, min_x, min_y, max_x, max_y);

        if boundary_indices.is_empty() {
            rng.gen_range(0..trees.len())
        } else {
            boundary_indices[rng.gen_range(0..boundary_indices.len())]
        }
    }

    /// Find best placement for new tree (from Gen2 EXTREME)
    fn find_placement(
        &self,
        existing: &[PlacedTree],
        n: usize,
        _max_n: usize,
        rng: &mut impl Rng,
    ) -> PlacedTree {
        if existing.is_empty() {
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

    /// Score a placement (lower is better) - from Gen2 EXTREME
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

        // Secondary: prefer balanced aspect ratio
        let balance_weight = 0.10 + 0.05 * (1.0 - (n as f64 / 200.0).min(1.0));
        let balance_penalty = (width - height).abs() * balance_weight;

        // Tertiary: slight preference for compact center
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.008 / (n as f64).sqrt();

        // Density heuristic
        let area = width * height;
        let density_bonus = if area > 0.0 {
            -0.005 * (n as f64 / area).min(2.0)
        } else {
            0.0
        };

        side_score + balance_penalty + center_penalty + density_bonus
    }

    /// Select rotation angles (12 angles from Gen2 EXTREME)
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
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

    /// Select direction angle for placement search (from Gen2 EXTREME)
    #[inline]
    fn select_direction(&self, n: usize, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;

        let strategy = rng.gen::<f64>();

        if strategy < 0.5 {
            let base_idx = rng.gen_range(0..num_dirs);
            let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
            base + rng.gen_range(-0.08..0.08)
        } else if strategy < 0.75 {
            loop {
                let angle = rng.gen_range(0.0..2.0 * PI);
                let corner_weight = ((4.0 * angle).sin().abs() + (4.0 * angle).cos().abs()) / 2.0;
                let threshold = 0.2 + 0.15 * (1.0 - (n as f64 / 200.0).min(1.0));
                if rng.gen::<f64>() < corner_weight.max(threshold) {
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

    /// SA move operator (10 move types from Gen2 EXTREME)
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
                // Fine rotation
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

    let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
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
        println!("Gen3 Compact score for n=1..50: {:.4}", score);
    }

    #[test]
    fn test_boundary_compaction() {
        let packer = EvolvedPacker::default();
        let packings = packer.pack_all(30);

        // Verify all packings are valid after compaction
        for p in &packings {
            assert!(!p.has_overlaps());
        }
    }
}
