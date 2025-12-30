//! Evolved Packing Algorithm - Generation 2: Greedy Compaction
//!
//! This mutation builds on Generation 1 by adding post-placement compaction phases:
//!
//! MUTATION STRATEGY - POST-PLACEMENT COMPACTION:
//! 1. Greedy compaction pass: slide trees toward bounding box center
//! 2. Re-centering step: translate all trees so centroid is at origin
//! 3. Shake and settle phase: perturb trees and run SA again
//! 4. Corner-filling strategy: pull outer trees into empty corners
//!
//! Evolution targets:
//! - compaction_pass(): New - slides trees toward center
//! - recenter_trees(): New - translates pack centroid to origin
//! - shake_and_settle(): New - perturbation with SA restart
//! - corner_fill(): New - identifies and fills empty corners

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Evolved packing configuration - Gen2 with compaction parameters
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

    // Gen2: Compaction parameters
    pub compaction_passes: usize,
    pub compaction_step_size: f64,
    pub shake_magnitude: f64,
    pub shake_settle_iterations: usize,
    pub corner_fill_attempts: usize,
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        Self {
            // Gen1 Tweak baseline parameters
            search_attempts: 75,
            direction_samples: 16,
            sa_iterations: 6500,
            sa_initial_temp: 0.45,
            sa_cooling_rate: 0.9993,
            sa_min_temp: 0.001,
            translation_scale: 0.08,
            rotation_granularity: 45.0,
            center_pull_strength: 0.04,

            // Gen2: Compaction parameters
            compaction_passes: 3,          // Multiple greedy passes
            compaction_step_size: 0.02,    // Binary search precision
            shake_magnitude: 0.03,         // Perturbation size
            shake_settle_iterations: 2000, // SA iterations after shake
            corner_fill_attempts: 20,      // Attempts per corner
        }
    }
}

/// Main evolved packer with compaction
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

            // Run evolved local search (SA)
            self.local_search(&mut trees, n, &mut rng);

            // GEN2: Post-placement compaction phases
            // Phase 1: Greedy compaction toward center
            for _ in 0..self.config.compaction_passes {
                self.compaction_pass(&mut trees);
            }

            // Phase 2: Re-center the packing
            self.recenter_trees(&mut trees);

            // Phase 3: Shake and settle (every 5th n or for larger packings)
            if n >= 10 && n % 5 == 0 {
                self.shake_and_settle(&mut trees, n, &mut rng);
            }

            // Phase 4: Corner filling
            self.corner_fill(&mut trees, &mut rng);

            // Final re-centering after all compaction
            self.recenter_trees(&mut trees);

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

    /// GEN2 FUNCTION: Greedy compaction pass
    /// Slides each tree toward the bounding box center, accepting moves that reduce size
    fn compaction_pass(&self, trees: &mut Vec<PlacedTree>) {
        if trees.len() <= 1 {
            return;
        }

        let initial_side = compute_side_length(trees);

        // Get bounding box center
        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
        let center_x = (min_x + max_x) / 2.0;
        let center_y = (min_y + max_y) / 2.0;

        // Try to slide each tree toward center
        for idx in 0..trees.len() {
            let old_tree = trees[idx].clone();
            let dx = center_x - old_tree.x;
            let dy = center_y - old_tree.y;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < 0.01 {
                continue; // Already at center
            }

            // Normalize direction
            let dir_x = dx / dist;
            let dir_y = dy / dist;

            // Binary search for best position along this direction
            let mut best_tree = old_tree.clone();
            let mut best_side = initial_side;

            // Try progressively closer positions
            let mut step = dist;
            while step > self.config.compaction_step_size {
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
                step *= 0.5;
            }

            trees[idx] = best_tree;
        }
    }

    /// GEN2 FUNCTION: Re-center the packing
    /// Translates all trees so the centroid is at the origin
    fn recenter_trees(&self, trees: &mut Vec<PlacedTree>) {
        if trees.is_empty() {
            return;
        }

        // Compute centroid of tree positions
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        for tree in trees.iter() {
            sum_x += tree.x;
            sum_y += tree.y;
        }
        let centroid_x = sum_x / trees.len() as f64;
        let centroid_y = sum_y / trees.len() as f64;

        // Translate all trees
        let new_trees: Vec<PlacedTree> = trees
            .iter()
            .map(|t| PlacedTree::new(t.x - centroid_x, t.y - centroid_y, t.angle_deg))
            .collect();

        *trees = new_trees;
    }

    /// GEN2 FUNCTION: Shake and settle phase
    /// Randomly perturb trees, then run SA to let them settle into better positions
    fn shake_and_settle(&self, trees: &mut Vec<PlacedTree>, n: usize, rng: &mut impl Rng) {
        if trees.len() <= 2 {
            return;
        }

        let initial_side = compute_side_length(trees);
        let backup = trees.clone();

        // Shake: randomly perturb each tree slightly
        for tree in trees.iter_mut() {
            let dx = rng.gen_range(-self.config.shake_magnitude..self.config.shake_magnitude);
            let dy = rng.gen_range(-self.config.shake_magnitude..self.config.shake_magnitude);
            *tree = PlacedTree::new(tree.x + dx, tree.y + dy, tree.angle_deg);
        }

        // Check validity after shake
        let mut valid = true;
        for i in 0..trees.len() {
            if has_overlap(trees, i) {
                valid = false;
                break;
            }
        }

        if !valid {
            // Revert if shake caused overlaps
            *trees = backup;
            return;
        }

        // Settle: run SA with temperature restart
        let mut current_side = compute_side_length(trees);
        let mut temp = self.config.sa_initial_temp * 0.5; // Restart at half temp

        for _ in 0..self.config.shake_settle_iterations {
            let idx = rng.gen_range(0..trees.len());
            let old_tree = trees[idx].clone();

            // Use existing SA move operator
            let success = self.sa_move(trees, idx, temp, 0, rng);

            if success {
                let new_side = compute_side_length(trees);
                let delta = new_side - current_side;

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

        // Only keep result if it improved
        let final_side = compute_side_length(trees);
        if final_side > initial_side {
            *trees = backup;
        }
    }

    /// GEN2 FUNCTION: Corner-filling strategy
    /// Identifies empty corners and tries to pull outer trees into them
    fn corner_fill(&self, trees: &mut Vec<PlacedTree>, rng: &mut impl Rng) {
        if trees.len() <= 3 {
            return;
        }

        let initial_side = compute_side_length(trees);
        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);

        // Define the four corners
        let corners = [
            (min_x, min_y), // bottom-left
            (max_x, min_y), // bottom-right
            (min_x, max_y), // top-left
            (max_x, max_y), // top-right
        ];

        // Find which corners are relatively empty (no tree center nearby)
        let corner_threshold = (max_x - min_x).max(max_y - min_y) * 0.25;

        for &(corner_x, corner_y) in &corners {
            // Check if corner is empty
            let mut corner_empty = true;
            for tree in trees.iter() {
                let dx = tree.x - corner_x;
                let dy = tree.y - corner_y;
                if (dx * dx + dy * dy).sqrt() < corner_threshold {
                    corner_empty = false;
                    break;
                }
            }

            if !corner_empty {
                continue;
            }

            // Try to pull an outer tree toward this corner
            // Find the tree farthest from center
            let center_x = (min_x + max_x) / 2.0;
            let center_y = (min_y + max_y) / 2.0;

            let mut best_improvement = 0.0;
            let mut best_tree_state: Option<(usize, PlacedTree)> = None;

            for idx in 0..trees.len() {
                let tree = &trees[idx];
                let dist_from_center = ((tree.x - center_x).powi(2) + (tree.y - center_y).powi(2)).sqrt();

                // Only consider outer trees
                if dist_from_center < corner_threshold {
                    continue;
                }

                // Direction toward corner
                let dx = corner_x - tree.x;
                let dy = corner_y - tree.y;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < 0.1 {
                    continue;
                }
                let dir_x = dx / dist;
                let dir_y = dy / dist;

                // Try pulling toward corner
                for attempt in 0..self.config.corner_fill_attempts {
                    let pull_factor = (attempt as f64 + 1.0) / self.config.corner_fill_attempts as f64;
                    let new_x = tree.x + dir_x * dist * pull_factor * 0.3;
                    let new_y = tree.y + dir_y * dist * pull_factor * 0.3;

                    // Also try some rotations
                    let angle_offset = rng.gen_range(-45.0..45.0);
                    let new_angle = (tree.angle_deg + angle_offset).rem_euclid(360.0);

                    let candidate = PlacedTree::new(new_x, new_y, new_angle);
                    let old_tree = trees[idx].clone();
                    trees[idx] = candidate.clone();

                    if !has_overlap(trees, idx) {
                        let new_side = compute_side_length(trees);
                        let improvement = initial_side - new_side;
                        if improvement > best_improvement {
                            best_improvement = improvement;
                            best_tree_state = Some((idx, candidate));
                        }
                    }

                    trees[idx] = old_tree;
                }
            }

            // Apply best improvement if found
            if let Some((idx, new_tree)) = best_tree_state {
                trees[idx] = new_tree;
            }
        }
    }

    /// EVOLVED FUNCTION: Find best placement for new tree (from Gen1)
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

        // Secondary: prefer balanced aspect ratio (Gen1 tweak)
        let balance_penalty = (width - height).abs() * 0.12;

        // Tertiary: slight preference for compact center
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.01 / (n as f64).sqrt();

        side_score + balance_penalty + center_penalty
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
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

    /// EVOLVED FUNCTION: Local search with simulated annealing
    fn local_search(&self, trees: &mut Vec<PlacedTree>, n: usize, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut temp = self.config.sa_initial_temp;

        let iterations = self.config.sa_iterations + n * 20;

        for iter in 0..iterations {
            let idx = rng.gen_range(0..trees.len());
            let old_tree = trees[idx].clone();

            let success = self.sa_move(trees, idx, temp, iter, rng);

            if success {
                let new_side = compute_side_length(trees);
                let delta = new_side - current_side;

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
                let scale = self.config.translation_scale * (0.3 + temp * 2.0);
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            1 => {
                let new_angle = (old_angle + 90.0).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            2 => {
                let delta = if rng.gen() { self.config.rotation_granularity }
                            else { -self.config.rotation_granularity };
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            3 => {
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
                let scale = self.config.translation_scale * 0.5;
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                let dangle = rng.gen_range(-30.0..30.0);
                let new_angle = (old_angle + dangle).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, new_angle);
            }
            _ => {
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
        println!("Gen2 Compact score for n=1..50: {:.4}", score);
    }

    #[test]
    fn test_compaction_improves() {
        let packer = EvolvedPacker::default();
        let packings = packer.pack_all(30);

        // Verify all packings are valid
        for p in &packings {
            assert!(!p.has_overlaps());
        }
    }
}
