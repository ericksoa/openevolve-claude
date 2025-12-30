//! Evolved Packing Algorithm - Generation 1 Geometric Mutation
//!
//! This module contains the evolved packing heuristics.
//! The code is designed to be mutated by LLM-guided evolution.
//!
//! GEOMETRIC STRATEGY:
//! Exploits the specific geometry of the Christmas tree polygon:
//! - Tree is 1.0 tall (y: -0.2 to 0.8), 0.7 wide at base
//! - Trunk region (y: -0.2 to 0.0) is only 0.15 wide - natural nesting area
//! - Tip-to-trunk interlocking: 180-degree rotated trees can nest together
//! - Hexagonal packing patterns for better space utilization
//! - Branch tier alignment for minimal overlap waste
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

    // Geometric parameters
    pub interlock_bonus: f64,        // Bonus for tip-to-trunk alignment
    pub trunk_gap_threshold: f64,    // Distance to consider for trunk nesting
    pub hexagonal_weight: f64,       // Weight for hexagonal direction preference
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen1 Geometric parameters - tuned for tree shape interlocking
        Self {
            search_attempts: 72,          // More attempts to find interlocking positions
            direction_samples: 24,        // Finer direction sampling
            sa_iterations: 6000,          // More iterations for geometric refinement
            sa_initial_temp: 0.40,
            sa_cooling_rate: 0.9992,
            sa_min_temp: 0.0008,
            translation_scale: 0.06,      // Finer movements for precise interlocking
            rotation_granularity: 30.0,   // 30-degree steps for hexagonal patterns
            center_pull_strength: 0.035,
            interlock_bonus: 0.08,        // Reward for good interlocking
            trunk_gap_threshold: 0.25,    // Trunk gap region consideration
            hexagonal_weight: 0.6,        // Preference for 60-degree directions
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
    /// GEOMETRIC: Considers interlocking potential and trunk nesting
    fn find_placement(
        &self,
        existing: &[PlacedTree],
        n: usize,
        _max_n: usize,
        rng: &mut impl Rng,
    ) -> PlacedTree {
        if existing.is_empty() {
            // First tree: place at origin pointing up
            return PlacedTree::new(0.0, 0.0, 0.0);
        }

        let mut best_tree = PlacedTree::new(0.0, 0.0, 0.0);
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

                while high - low > 0.002 {
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
    /// GEOMETRIC: Includes interlocking bonus for tip-to-trunk alignment
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

        // Secondary: prefer balanced aspect ratio
        let balance_penalty = (width - height).abs() * 0.12;

        // Tertiary: slight preference for compact center
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.008 / (n as f64).sqrt();

        // GEOMETRIC: Interlocking bonus - reward complementary angles
        let interlock_bonus = self.compute_interlock_bonus(tree, existing);

        // GEOMETRIC: Trunk nesting bonus - reward when tip is near trunk gap
        let nesting_bonus = self.compute_nesting_bonus(tree, existing);

        side_score + balance_penalty + center_penalty - interlock_bonus - nesting_bonus
    }

    /// Compute bonus for tip-to-trunk interlocking potential
    /// Trees at 180-degree angle difference can potentially nest together
    #[inline]
    fn compute_interlock_bonus(&self, tree: &PlacedTree, existing: &[PlacedTree]) -> f64 {
        let tree_angle_normalized = tree.angle_deg.rem_euclid(360.0);
        let mut bonus = 0.0;

        for other in existing {
            let other_angle_normalized = other.angle_deg.rem_euclid(360.0);
            let angle_diff = (tree_angle_normalized - other_angle_normalized).abs();

            // Check for complementary angles (180 degrees apart = tip-to-trunk potential)
            let is_complementary = (angle_diff - 180.0).abs() < 15.0 ||
                                   (angle_diff - 0.0).abs() < 15.0 ||
                                   (angle_diff - 360.0).abs() < 15.0;

            if is_complementary {
                // Distance-based bonus - closer trees benefit more from interlocking
                let dx = tree.x - other.x;
                let dy = tree.y - other.y;
                let dist = (dx * dx + dy * dy).sqrt();

                // Strong bonus for close complementary trees (within ~1.5 tree heights)
                if dist < 1.5 {
                    bonus += self.config.interlock_bonus * (1.5 - dist) / 1.5;
                }
            }
        }

        bonus
    }

    /// Compute bonus for nesting tip into trunk gap region
    /// The trunk gap (width 0.7 - 0.15 = 0.55 on each side) can fit a tip
    #[inline]
    fn compute_nesting_bonus(&self, tree: &PlacedTree, existing: &[PlacedTree]) -> f64 {
        // Get the tip position of the new tree
        let tip = tree.vertices()[0]; // First vertex is always the tip

        let mut bonus = 0.0;

        for other in existing {
            // Get the trunk region bounds of the other tree
            // Trunk vertices are at indices 6, 7, 8, 9 (right trunk, left trunk)
            let trunk_vertices = [other.vertices()[6], other.vertices()[7],
                                  other.vertices()[8], other.vertices()[9]];

            // Compute trunk bounding box
            let trunk_min_x = trunk_vertices.iter().map(|v| v.0).fold(f64::INFINITY, f64::min);
            let trunk_max_x = trunk_vertices.iter().map(|v| v.0).fold(f64::NEG_INFINITY, f64::max);
            let trunk_min_y = trunk_vertices.iter().map(|v| v.1).fold(f64::INFINITY, f64::min);
            let trunk_max_y = trunk_vertices.iter().map(|v| v.1).fold(f64::NEG_INFINITY, f64::max);

            // Check if tip is near the trunk gap region (the space beside the trunk)
            let base_vertices = [other.vertices()[5], other.vertices()[10]]; // Base tier edges
            let base_y = (base_vertices[0].1 + base_vertices[1].1) / 2.0;

            // The gap region is beside the trunk, below the base tier
            let gap_left_x = trunk_min_x - 0.3;
            let gap_right_x = trunk_max_x + 0.3;

            // Check if tip is in the gap region
            let in_gap_region = (tip.0 > gap_left_x && tip.0 < trunk_min_x - 0.05 ||
                                tip.0 < gap_right_x && tip.0 > trunk_max_x + 0.05) &&
                               tip.1 > trunk_min_y - 0.1 && tip.1 < trunk_max_y + 0.3;

            if in_gap_region {
                bonus += self.config.interlock_bonus * 0.5;
            }

            // Also reward if the tip is close to the base tier level (y ~ 0 in local coords)
            let dist_to_base = (tip.1 - base_y).abs();
            if dist_to_base < self.config.trunk_gap_threshold {
                bonus += self.config.interlock_bonus * 0.3 * (1.0 - dist_to_base / self.config.trunk_gap_threshold);
            }
        }

        bonus
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    /// GEOMETRIC: Includes 60-degree intervals for hexagonal patterns + complementary pairs
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
        // Combine octagonal (45-degree) and hexagonal (60-degree) patterns
        // Plus complementary pairs for tip-to-trunk interlocking
        match n % 6 {
            0 => vec![0.0, 180.0, 60.0, 240.0, 120.0, 300.0, 90.0, 270.0, 30.0, 210.0],
            1 => vec![0.0, 180.0, 90.0, 270.0, 45.0, 225.0, 135.0, 315.0, 60.0, 240.0],
            2 => vec![180.0, 0.0, 120.0, 300.0, 60.0, 240.0, 90.0, 270.0, 150.0, 330.0],
            3 => vec![90.0, 270.0, 0.0, 180.0, 60.0, 240.0, 120.0, 300.0, 45.0, 225.0],
            4 => vec![270.0, 90.0, 180.0, 0.0, 240.0, 60.0, 300.0, 120.0, 315.0, 135.0],
            _ => vec![60.0, 240.0, 0.0, 180.0, 120.0, 300.0, 90.0, 270.0, 30.0, 210.0],
        }
    }

    /// EVOLVED FUNCTION: Select direction angle for placement search
    /// GEOMETRIC: Favors hexagonal directions for better packing
    #[inline]
    fn select_direction(&self, n: usize, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;

        // Select direction strategy
        let strategy = rng.gen::<f64>();

        if strategy < self.config.hexagonal_weight {
            // Hexagonal directions: 0, 60, 120, 180, 240, 300 degrees
            let hex_base = (rng.gen_range(0..6) as f64) * 60.0;
            let jitter = rng.gen_range(-8.0..8.0);
            (hex_base + jitter) * PI / 180.0
        } else if strategy < 0.85 {
            // Structured: evenly spaced with jitter
            let base_idx = rng.gen_range(0..num_dirs);
            let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
            base + rng.gen_range(-0.12..0.12)
        } else {
            // Favor directions toward existing tree positions for interlocking
            // This helps find positions that can nest with existing trees
            loop {
                let angle = rng.gen_range(0.0..2.0 * PI);
                let corner_weight = (2.0 * angle).sin().abs();
                let hex_weight = (3.0 * angle).cos().abs(); // Peaks at 0, 60, 120, ...
                let combined_weight = 0.5 * corner_weight + 0.5 * hex_weight;
                let threshold = 0.2 + 0.15 * (1.0 - (n as f64 / 200.0).min(1.0));
                if rng.gen::<f64>() < combined_weight.max(threshold) {
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

        // Scale iterations with n
        let iterations = self.config.sa_iterations + n * 25;

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
    /// GEOMETRIC: Includes interlock adjustment and hexagonal rotation moves
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

        let move_type = rng.gen_range(0..8);

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
                // GEOMETRIC: 60-degree rotation for hexagonal patterns
                let delta = if rng.gen() { 60.0 } else { -60.0 };
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
                let dangle = rng.gen_range(-self.config.rotation_granularity..self.config.rotation_granularity);
                let new_angle = (old_angle + dangle).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, new_angle);
            }
            5 => {
                // Polar move (radial in/out)
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.1 {
                    let delta_r = rng.gen_range(-0.04..0.04) * (1.0 + temp);
                    let new_mag = (mag + delta_r).max(0.0);
                    let scale = new_mag / mag;
                    trees[idx] = PlacedTree::new(old_x * scale, old_y * scale, old_angle);
                } else {
                    return false;
                }
            }
            6 => {
                // GEOMETRIC: 180-degree flip for tip-to-trunk interlocking
                let new_angle = (old_angle + 180.0).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            _ => {
                // GEOMETRIC: Interlock slide - move along the direction perpendicular to tree's tip
                // This helps trees slide into interlocking positions
                let tip_dir = old_angle * PI / 180.0;
                let slide_dir = tip_dir + PI / 2.0; // Perpendicular to tip direction
                let slide_dist = rng.gen_range(-0.08..0.08) * (0.5 + temp);
                let dx = slide_dir.cos() * slide_dist;
                let dy = slide_dir.sin() * slide_dist;
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
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
        println!("Evolved Geometric score for n=1..50: {:.4}", score);
    }
}
