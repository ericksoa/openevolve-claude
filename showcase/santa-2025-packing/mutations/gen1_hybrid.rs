//! Evolved Packing Algorithm - Generation 1 Hybrid
//!
//! This module contains the evolved packing heuristics.
//! The code is designed to be mutated by LLM-guided evolution.
//!
//! HYBRIDIZATION STRATEGY:
//! - Spiral placement pattern: place trees in an outward spiral from center
//! - Greedy compaction moves: after placement, try sliding trees inward
//! - Combined with existing radial search and corner-weighting heuristics
//!
//! Evolution targets:
//! - placement_score(): How to score candidate placements
//! - select_angles(): Which rotation angles to try
//! - select_direction(): How to choose placement directions
//! - sa_move(): Local search move operators
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

    // HYBRID: Spiral placement parameters
    pub spiral_weight: f64,           // Weight for spiral placement strategy
    pub spiral_arm_count: usize,      // Number of spiral arms to try
    pub spiral_tightness: f64,        // How tightly wound the spiral is

    // HYBRID: Compaction parameters
    pub compaction_passes: usize,     // Number of greedy compaction passes
    pub compaction_step: f64,         // Step size for compaction slides
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen1 Hybrid parameters - evolved from baseline + new hybrid features
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
            // HYBRID: Spiral parameters
            spiral_weight: 0.4,
            spiral_arm_count: 4,
            spiral_tightness: 0.3,
            // HYBRID: Compaction parameters
            compaction_passes: 3,
            compaction_step: 0.02,
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

            // HYBRID: Apply greedy compaction before local search
            self.greedy_compaction(&mut trees);

            // Run evolved local search
            self.local_search(&mut trees, n, &mut rng);

            // HYBRID: Final compaction pass after SA
            self.greedy_compaction(&mut trees);

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
    /// HYBRID: Combines radial search with spiral placement pattern
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

        // HYBRID: Mix of spiral and original radial search
        let spiral_attempts = (self.config.search_attempts as f64 * self.config.spiral_weight) as usize;
        let radial_attempts = self.config.search_attempts - spiral_attempts;

        // Try spiral placement pattern
        for attempt in 0..spiral_attempts {
            let dir = self.select_spiral_direction(n, attempt, existing);
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

        // Original radial search (with jitter)
        for _ in 0..radial_attempts {
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

    /// HYBRID FUNCTION: Select direction based on spiral pattern
    /// Generates directions that follow an outward spiral from the center of mass
    #[inline]
    fn select_spiral_direction(&self, n: usize, attempt: usize, existing: &[PlacedTree]) -> f64 {
        // Compute center of mass of existing trees
        let (cx, cy) = if existing.is_empty() {
            (0.0, 0.0)
        } else {
            let sum_x: f64 = existing.iter().map(|t| t.x).sum();
            let sum_y: f64 = existing.iter().map(|t| t.y).sum();
            (sum_x / existing.len() as f64, sum_y / existing.len() as f64)
        };

        // Spiral arm selection
        let arm = attempt % self.config.spiral_arm_count;
        let arm_base_angle = (arm as f64 / self.config.spiral_arm_count as f64) * 2.0 * PI;

        // Spiral outward: angle increases with n
        let spiral_offset = (n as f64) * self.config.spiral_tightness;

        // Final direction from center of mass
        let direction = arm_base_angle + spiral_offset;

        // Bias direction away from center of mass if we're off-center
        let cm_angle = cy.atan2(cx);
        let bias = cm_angle + PI; // Point away from CM

        // Blend spiral direction with anti-CM bias
        let blend_factor = 0.3;
        let blended = direction * (1.0 - blend_factor) + bias * blend_factor;

        blended
    }

    /// HYBRID FUNCTION: Greedy compaction - slide trees toward center
    /// Tries to reduce bounding box by moving trees inward
    fn greedy_compaction(&self, trees: &mut Vec<PlacedTree>) {
        if trees.len() <= 1 {
            return;
        }

        for _pass in 0..self.config.compaction_passes {
            let mut improved = false;

            for idx in 0..trees.len() {
                let old_side = compute_side_length(trees);
                let old_tree = trees[idx].clone();

                // Try sliding toward center
                let best_tree = self.try_slide_inward(trees, idx);

                if let Some(new_tree) = best_tree {
                    trees[idx] = new_tree;
                    let new_side = compute_side_length(trees);

                    if new_side < old_side - 0.0001 {
                        improved = true;
                    } else {
                        // Revert if no improvement
                        trees[idx] = old_tree;
                    }
                }
            }

            if !improved {
                break;
            }
        }
    }

    /// HYBRID FUNCTION: Try sliding a tree toward the center
    /// Returns the best valid position closer to center, or None
    fn try_slide_inward(&self, trees: &[PlacedTree], idx: usize) -> Option<PlacedTree> {
        let tree = &trees[idx];
        let (x, y, angle) = (tree.x, tree.y, tree.angle_deg);

        // Compute centroid of bounding box
        let (min_x, min_y, max_x, max_y) = self.compute_bounds(trees);
        let center_x = (min_x + max_x) / 2.0;
        let center_y = (min_y + max_y) / 2.0;

        // Direction toward center
        let dx = center_x - x;
        let dy = center_y - y;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist < 0.01 {
            return None;
        }

        let ux = dx / dist;
        let uy = dy / dist;

        // Try sliding inward by step increments
        let mut best_tree: Option<PlacedTree> = None;
        let mut step = self.config.compaction_step;

        while step <= dist {
            let new_x = x + ux * step;
            let new_y = y + uy * step;
            let candidate = PlacedTree::new(new_x, new_y, angle);

            // Check if valid (no overlap with others)
            let mut valid = true;
            for (i, other) in trees.iter().enumerate() {
                if i != idx && candidate.overlaps(other) {
                    valid = false;
                    break;
                }
            }

            if valid {
                best_tree = Some(candidate);
            }

            step += self.config.compaction_step;
        }

        best_tree
    }

    /// Helper: Compute bounding box of all trees
    fn compute_bounds(&self, trees: &[PlacedTree]) -> (f64, f64, f64, f64) {
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

    /// EVOLVED FUNCTION: Score a placement (lower is better)
    /// HYBRID: Enhanced with aspect ratio optimization
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

        // HYBRID: Stronger aspect ratio penalty - prefer square arrangements
        let aspect_ratio = if width > height { width / height } else { height / width };
        let aspect_penalty = (aspect_ratio - 1.0) * 0.25; // Increased from 0.15

        // Tertiary: slight preference for compact center (scaled by n)
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.01 / (n as f64).sqrt();

        // HYBRID: Bonus for filling "inner shells" - penalize if tree is far from center of mass
        let com_x: f64 = existing.iter().map(|t| t.x).sum::<f64>() / existing.len().max(1) as f64;
        let com_y: f64 = existing.iter().map(|t| t.y).sum::<f64>() / existing.len().max(1) as f64;
        let dist_from_com = ((tree.x - com_x).powi(2) + (tree.y - com_y).powi(2)).sqrt();
        let shell_penalty = dist_from_com * 0.02 / (n as f64).sqrt();

        side_score + aspect_penalty + center_penalty + shell_penalty
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
    /// HYBRID: Enhanced with compaction-aware moves
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
    /// HYBRID: Added slide-to-boundary move
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

        let move_type = rng.gen_range(0..8); // HYBRID: Increased from 6 to 8

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
            5 => {
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
            6 => {
                // HYBRID: Slide toward bounding box center
                let (min_x, min_y, max_x, max_y) = self.compute_bounds(trees);
                let cx = (min_x + max_x) / 2.0;
                let cy = (min_y + max_y) / 2.0;
                let dx = cx - old_x;
                let dy = cy - old_y;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist > 0.05 {
                    let step = self.config.compaction_step * (1.0 + temp * 3.0);
                    let new_x = old_x + dx / dist * step;
                    let new_y = old_y + dy / dist * step;
                    trees[idx] = PlacedTree::new(new_x, new_y, old_angle);
                } else {
                    return false;
                }
            }
            _ => {
                // HYBRID: Corner-seeking move - slide toward nearest corner
                let (min_x, min_y, max_x, max_y) = self.compute_bounds(trees);
                let corners = [
                    (min_x, min_y),
                    (min_x, max_y),
                    (max_x, min_y),
                    (max_x, max_y),
                ];
                // Find nearest corner
                let (corner_x, corner_y) = corners.iter()
                    .min_by(|(ax, ay), (bx, by)| {
                        let da = (ax - old_x).powi(2) + (ay - old_y).powi(2);
                        let db = (bx - old_x).powi(2) + (by - old_y).powi(2);
                        da.partial_cmp(&db).unwrap()
                    })
                    .copied()
                    .unwrap_or((old_x, old_y));

                let dx = corner_x - old_x;
                let dy = corner_y - old_y;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist > 0.05 {
                    let step = self.config.compaction_step * 2.0 * (1.0 + temp);
                    let new_x = old_x + dx / dist * step;
                    let new_y = old_y + dy / dist * step;
                    trees[idx] = PlacedTree::new(new_x, new_y, old_angle);
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
