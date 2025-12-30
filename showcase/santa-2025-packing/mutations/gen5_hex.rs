//! Evolved Packing Algorithm - Generation 5 HEXAGONAL
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
//! MUTATION STRATEGY: HEXAGONAL FOCUS (Gen5)
//! Hypothesis: Christmas trees pack best in hexagonal arrangements due to triangular shape.
//!
//! Key changes from Gen4 efficient:
//! - Angle bias: Strongly prefer 0, 60, 120, 180, 240, 300 degrees (hexagonal)
//! - Direction bias: Search in hexagonal directions primarily
//! - Placement scoring: Bonus for trees aligned 60 degrees apart
//! - Tip-to-trunk nesting: Bonus for complementary orientations (180 apart)
//! - Computation kept similar to Gen4 efficient for speed
//!
//! Base: Gen4 champion (score 98.37 at n=200)
//! Goal: Exploit hexagonal packing geometry for better density

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Hexagonal angles in degrees (60-degree increments)
const HEX_ANGLES: [f64; 6] = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0];

/// Hexagonal angles in radians for direction search
const HEX_RADIANS: [f64; 6] = [
    0.0,
    PI / 3.0,
    2.0 * PI / 3.0,
    PI,
    4.0 * PI / 3.0,
    5.0 * PI / 3.0,
];

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

    // Multi-pass settings
    pub sa_passes: usize,

    // EFFICIENT: Early exit threshold
    pub early_exit_threshold: usize,

    // HEXAGONAL: Weighting for hexagonal alignment bonus
    pub hex_alignment_bonus: f64,
    pub tip_trunk_bonus: f64,
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen5 HEXAGONAL: Based on Gen4 efficient with hexagonal focus
        Self {
            search_attempts: 250,            // Same as Gen4
            direction_samples: 64,           // Same as Gen4
            sa_iterations: 20000,            // Same as Gen4
            sa_initial_temp: 0.6,            // Same as Gen4
            sa_cooling_rate: 0.9999,         // Same as Gen4
            sa_min_temp: 0.00001,            // Same as Gen4
            translation_scale: 0.08,         // Same as Gen4
            rotation_granularity: 60.0,      // Changed to 60 degrees (hexagonal)
            center_pull_strength: 0.06,      // Same as Gen4
            sa_passes: 2,                    // Same as Gen4
            early_exit_threshold: 1000,      // Same as Gen4
            hex_alignment_bonus: 0.03,       // NEW: Bonus for hexagonal alignment
            tip_trunk_bonus: 0.05,           // NEW: Bonus for tip-to-trunk nesting
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

            // EFFICIENT: Run double SA passes (same as Gen4)
            for pass in 0..self.config.sa_passes {
                self.local_search(&mut trees, n, pass, &mut rng);
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
    /// HEXAGONAL: Uses hexagonal angles and directions
    fn find_placement(
        &self,
        existing: &[PlacedTree],
        n: usize,
        _max_n: usize,
        rng: &mut impl Rng,
    ) -> PlacedTree {
        if existing.is_empty() {
            // First tree: place at origin with hexagonal rotation (0 degrees)
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
                // EFFICIENT: Use coarser precision 0.001 (same as Gen4)
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
    /// HEXAGONAL: Adds bonuses for hexagonal alignment and tip-to-trunk nesting
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

        // Density bonus (same as Gen4)
        let area = width * height;
        let density_bonus = if area > 0.0 {
            -0.01 * (n as f64 / area).min(2.0)
        } else {
            0.0
        };

        // HEXAGONAL: Bonus for hexagonal angle alignment
        let hex_bonus = self.compute_hex_alignment_bonus(tree, existing);

        // HEXAGONAL: Bonus for tip-to-trunk nesting (180 degrees apart)
        let nesting_bonus = self.compute_tip_trunk_bonus(tree, existing);

        side_score + balance_penalty + center_penalty + density_bonus - hex_bonus - nesting_bonus
    }

    /// HEXAGONAL: Compute bonus for trees aligned at hexagonal angles
    #[inline]
    fn compute_hex_alignment_bonus(&self, tree: &PlacedTree, existing: &[PlacedTree]) -> f64 {
        if existing.is_empty() {
            return 0.0;
        }

        let mut bonus = 0.0;
        let tree_angle = tree.angle_deg.rem_euclid(360.0);

        // Check if this tree's angle is close to a hexagonal angle
        let mut min_hex_dist = 360.0;
        for &hex in &HEX_ANGLES {
            let dist = angle_distance(tree_angle, hex);
            min_hex_dist = min_hex_dist.min(dist);
        }

        // Bonus if tree is at a hexagonal angle (within 10 degrees)
        if min_hex_dist < 10.0 {
            bonus += self.config.hex_alignment_bonus * (1.0 - min_hex_dist / 10.0);
        }

        // Bonus for each neighbor at a hexagonal angle relative to this tree
        for other in existing {
            let other_angle = other.angle_deg.rem_euclid(360.0);
            let angle_diff = angle_distance(tree_angle, other_angle);

            // Check if difference is close to 60 degrees (hexagonal)
            let hex_diff = (angle_diff - 60.0).abs().min((angle_diff - 120.0).abs());
            if hex_diff < 15.0 {
                bonus += self.config.hex_alignment_bonus * 0.5 * (1.0 - hex_diff / 15.0);
            }
        }

        bonus
    }

    /// HEXAGONAL: Compute bonus for tip-to-trunk nesting (180 degrees apart)
    #[inline]
    fn compute_tip_trunk_bonus(&self, tree: &PlacedTree, existing: &[PlacedTree]) -> f64 {
        if existing.is_empty() {
            return 0.0;
        }

        let mut bonus = 0.0;
        let tree_angle = tree.angle_deg.rem_euclid(360.0);

        // Check for complementary orientations (180 degrees apart)
        for other in existing {
            let other_angle = other.angle_deg.rem_euclid(360.0);
            let angle_diff = angle_distance(tree_angle, other_angle);

            // Bonus if trees are 180 degrees apart (tip-to-trunk nesting)
            if (angle_diff - 180.0).abs() < 20.0 {
                // Also check if they're reasonably close (within 1.5 units)
                let dx = tree.x - other.x;
                let dy = tree.y - other.y;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < 1.5 {
                    let proximity_factor = 1.0 - dist / 1.5;
                    let angle_factor = 1.0 - (angle_diff - 180.0).abs() / 20.0;
                    bonus += self.config.tip_trunk_bonus * proximity_factor * angle_factor;
                }
            }
        }

        bonus
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    /// HEXAGONAL: Prioritize 60-degree increments, then add intermediate angles
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
        // HEXAGONAL: Primary angles are 60-degree increments
        // Start with different base angles depending on n for variety
        let base_offset = ((n % 6) as f64) * 60.0;
        let mut angles: Vec<f64> = Vec::with_capacity(12);

        // First priority: hexagonal angles with offset
        for i in 0..6 {
            let angle = (base_offset + (i as f64) * 60.0).rem_euclid(360.0);
            angles.push(angle);
        }

        // Second priority: 30-degree offsets (between hexagonal angles)
        for i in 0..6 {
            let angle = (base_offset + 30.0 + (i as f64) * 60.0).rem_euclid(360.0);
            angles.push(angle);
        }

        angles
    }

    /// EVOLVED FUNCTION: Select direction angle for placement search
    /// HEXAGONAL: Strongly bias toward hexagonal directions
    #[inline]
    fn select_direction(&self, n: usize, rng: &mut impl Rng) -> f64 {
        let strategy = rng.gen::<f64>();

        if strategy < 0.55 {
            // HEXAGONAL: Pick a hexagonal direction with small jitter
            let hex_idx = rng.gen_range(0..6);
            let base = HEX_RADIANS[hex_idx];
            base + rng.gen_range(-0.08..0.08)
        } else if strategy < 0.75 {
            // Structured: evenly spaced with small jitter (from Gen4)
            let num_dirs = self.config.direction_samples;
            let base_idx = rng.gen_range(0..num_dirs);
            let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
            base + rng.gen_range(-0.06..0.06)
        } else if strategy < 0.90 {
            // HEXAGONAL: Intermediate angles (30 degrees offset from hex)
            let hex_idx = rng.gen_range(0..6);
            let base = HEX_RADIANS[hex_idx] + PI / 6.0; // 30 degree offset
            base + rng.gen_range(-0.08..0.08)
        } else {
            // Golden angle spiral for exploration (from Gen4)
            let golden_angle = PI * (3.0 - (5.0_f64).sqrt());
            let base = (n as f64 * golden_angle) % (2.0 * PI);
            let offset = rng.gen_range(0..6) as f64 * PI / 3.0;  // 6 offsets at 60 degrees
            (base + offset + rng.gen_range(-0.1..0.1)) % (2.0 * PI)
        }
    }

    /// EVOLVED FUNCTION: Local search with simulated annealing
    /// HEXAGONAL: SA moves biased toward hexagonal rotations
    fn local_search(&self, trees: &mut Vec<PlacedTree>, n: usize, pass: usize, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config: Vec<PlacedTree> = trees.clone();

        // Adjust temperature based on pass number (same as Gen4)
        let temp_multiplier = match pass {
            0 => 1.0,
            _ => 0.4,
        };
        let mut temp = self.config.sa_initial_temp * temp_multiplier;

        // Iterations (same as Gen4)
        let base_iterations = match pass {
            0 => self.config.sa_iterations + n * 100,
            _ => self.config.sa_iterations / 2 + n * 50,
        };

        // Track iterations without improvement for early exit
        let mut iterations_without_improvement = 0;

        // Pre-compute boundary trees once per batch
        let mut boundary_cache_iter = 0;
        let mut boundary_indices: Vec<usize> = Vec::new();

        for iter in 0..base_iterations {
            // Early exit when no improvement (same as Gen4)
            if iterations_without_improvement >= self.config.early_exit_threshold {
                break;
            }

            // Update boundary cache every 500 iterations
            if iter == 0 || iter - boundary_cache_iter >= 500 {
                boundary_indices = self.find_boundary_trees(trees);
                boundary_cache_iter = iter;
            }

            // 70% chance to pick boundary tree (same as Gen4)
            let idx = if !boundary_indices.is_empty() && rng.gen::<f64>() < 0.70 {
                boundary_indices[rng.gen_range(0..boundary_indices.len())]
            } else {
                rng.gen_range(0..trees.len())
            };

            let old_tree = trees[idx].clone();

            // HEXAGONAL: Move operator selection with hexagonal bias
            let success = self.sa_move(trees, idx, temp, &boundary_indices, rng);

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

            temp = (temp * self.config.sa_cooling_rate).max(self.config.sa_min_temp);
        }

        // Restore best configuration found during search
        if best_side < compute_side_length(trees) {
            *trees = best_config;
        }
    }

    /// Find trees on the bounding box boundary (same as Gen4)
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
    /// HEXAGONAL: Moves biased toward hexagonal rotations
    #[inline]
    fn sa_move(
        &self,
        trees: &mut [PlacedTree],
        idx: usize,
        temp: f64,
        boundary_indices: &[usize],
        rng: &mut impl Rng,
    ) -> bool {
        let old = &trees[idx];
        let old_x = old.x;
        let old_y = old.y;
        let old_angle = old.angle_deg;

        let is_boundary = boundary_indices.contains(&idx);

        // HEXAGONAL: 9 move types with hexagonal rotation bias
        let move_type = if is_boundary {
            // Boundary trees: prefer inward moves
            match rng.gen_range(0..10) {
                0..=3 => 0,  // Inward translation (40%)
                4..=5 => 1,  // Hexagonal rotation (20%)
                6..=7 => 2,  // Center pull (20%)
                8 => 3,      // Small nudge (10%)
                _ => 4,      // Translate + hex rotate (10%)
            }
        } else {
            rng.gen_range(0..9)
        };

        match move_type {
            0 => {
                // Inward translation (same as Gen4)
                let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                let (bx1, by1, bx2, by2) = trees[idx].bounds();

                let scale = self.config.translation_scale * (0.2 + temp * 2.0);
                let mut dx = rng.gen_range(-scale * 0.3..scale * 0.3);
                let mut dy = rng.gen_range(-scale * 0.3..scale * 0.3);

                if (bx1 - min_x).abs() < 0.02 { dx += scale * 0.5; }
                if (bx2 - max_x).abs() < 0.02 { dx -= scale * 0.5; }
                if (by1 - min_y).abs() < 0.02 { dy += scale * 0.5; }
                if (by2 - max_y).abs() < 0.02 { dy -= scale * 0.5; }

                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            1 => {
                // HEXAGONAL: 60-degree rotation
                let delta = if rng.gen() { 60.0 } else { -60.0 };
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            2 => {
                // Move toward center (same as Gen4)
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.04 {
                    let scale = self.config.center_pull_strength * (0.4 + temp * 1.5);
                    let dx = -old_x / mag * scale;
                    let dy = -old_y / mag * scale;
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                } else {
                    return false;
                }
            }
            3 => {
                // Small nudge (same as Gen4)
                let scale = 0.015 * (0.5 + temp);
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            4 => {
                // HEXAGONAL: Translate + 60-degree rotate combo
                let scale = self.config.translation_scale * 0.4;
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                let delta = if rng.gen() { 60.0 } else { -60.0 };
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, new_angle);
            }
            5 => {
                // HEXAGONAL: Snap to nearest hexagonal angle
                let current = old_angle.rem_euclid(360.0);
                let mut nearest_hex = HEX_ANGLES[0];
                let mut min_dist = angle_distance(current, HEX_ANGLES[0]);
                for &hex in &HEX_ANGLES[1..] {
                    let dist = angle_distance(current, hex);
                    if dist < min_dist {
                        min_dist = dist;
                        nearest_hex = hex;
                    }
                }
                if (nearest_hex - current).abs() > 0.1 {
                    trees[idx] = PlacedTree::new(old_x, old_y, nearest_hex);
                } else {
                    return false;
                }
            }
            6 => {
                // HEXAGONAL: 180-degree flip (for tip-to-trunk nesting)
                let new_angle = (old_angle + 180.0).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            7 => {
                // Polar move - radial (same as Gen4)
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.08 {
                    let delta_r = rng.gen_range(-0.08..0.08) * (1.0 + temp);
                    let new_mag = (mag + delta_r).max(0.0);
                    let scale = new_mag / mag;
                    trees[idx] = PlacedTree::new(old_x * scale, old_y * scale, old_angle);
                } else {
                    return false;
                }
            }
            _ => {
                // HEXAGONAL: Angular orbit in 60-degree steps
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.08 {
                    let current_angle = old_y.atan2(old_x);
                    // Move in 60-degree increments with small jitter
                    let delta_angle = if rng.gen() { PI / 3.0 } else { -PI / 3.0 };
                    let jitter = rng.gen_range(-0.1..0.1) * temp;
                    let new_ang = current_angle + delta_angle + jitter;
                    trees[idx] = PlacedTree::new(mag * new_ang.cos(), mag * new_ang.sin(), old_angle);
                } else {
                    return false;
                }
            }
        }

        !has_overlap(trees, idx)
    }
}

/// Calculate the angular distance between two angles (0-180)
#[inline]
fn angle_distance(a: f64, b: f64) -> f64 {
    let diff = (a - b).abs().rem_euclid(360.0);
    if diff > 180.0 { 360.0 - diff } else { diff }
}

// Helper functions (same as Gen4)
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

    #[test]
    fn test_hex_angles() {
        // Verify hexagonal angles are correct
        for (i, &angle) in HEX_ANGLES.iter().enumerate() {
            assert!((angle - (i as f64 * 60.0)).abs() < 0.001);
        }
    }

    #[test]
    fn test_angle_distance() {
        assert!((angle_distance(0.0, 60.0) - 60.0).abs() < 0.001);
        assert!((angle_distance(350.0, 10.0) - 20.0).abs() < 0.001);
        assert!((angle_distance(180.0, 0.0) - 180.0).abs() < 0.001);
    }
}
