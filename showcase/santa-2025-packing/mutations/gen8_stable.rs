//! Evolved Packing Algorithm - Generation 8 VARIANCE REDUCTION
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
//! MUTATION STRATEGY: VARIANCE REDUCTION (Gen8)
//! Reduce the 2% variance seen in multiple runs:
//!
//! Key improvements from Gen6:
//! - Deterministic seed based on n for reproducibility
//! - Run 3 internal attempts and take the best result
//! - More structured angle selection (less randomness)
//! - Tighter early exit (500 iterations) for faster exploration
//! - Consistent direction selection using golden ratio patterns
//!
//! Target: Beat Gen6's 94.14 at n=200 with more stable/reproducible results

use crate::{Packing, PlacedTree};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
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

    // Multi-pass settings
    pub sa_passes: usize,

    // Early exit threshold - reduced for faster exploration
    pub early_exit_threshold: usize,

    // Boundary focus probability
    pub boundary_focus_prob: f64,

    // VARIANCE REDUCTION: New parameters
    pub internal_attempts: usize,       // Number of internal attempts to run
    pub base_seed: u64,                 // Base seed for reproducibility
    pub structured_direction_prob: f64, // Probability of structured direction
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen8 VARIANCE REDUCTION: Stability-focused configuration
        Self {
            search_attempts: 280,            // Keep from Gen6
            direction_samples: 72,           // Keep from Gen6
            sa_iterations: 24000,            // Slightly reduced for faster attempts
            sa_initial_temp: 0.45,           // Keep from Gen6
            sa_cooling_rate: 0.99993,        // Slightly faster cooling
            sa_min_temp: 0.000008,           // Keep from Gen6
            translation_scale: 0.055,        // Keep from Gen6
            rotation_granularity: 45.0,      // Keep 8 angles
            center_pull_strength: 0.07,      // Keep from Gen6
            sa_passes: 2,                    // Keep 2 passes
            early_exit_threshold: 500,       // REDUCED for faster exploration
            boundary_focus_prob: 0.85,       // Keep from Gen6
            // VARIANCE REDUCTION parameters
            internal_attempts: 3,            // Run 3 attempts, take best
            base_seed: 2025_1225,            // Christmas-themed seed
            structured_direction_prob: 0.70, // 70% structured, 30% random
        }
    }
}

/// Track which boundary a tree is blocking
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BoundaryEdge {
    Left,
    Right,
    Top,
    Bottom,
    Corner,
    None,
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
    /// VARIANCE REDUCTION: Run multiple attempts with deterministic seeds
    pub fn pack_all(&self, max_n: usize) -> Vec<Packing> {
        // Run multiple internal attempts and take the best
        let mut best_packings: Option<Vec<Packing>> = None;
        let mut best_total_score = f64::INFINITY;

        for attempt in 0..self.config.internal_attempts {
            // Deterministic seed based on attempt number
            let seed = self.config.base_seed + attempt as u64;
            let mut rng = StdRng::seed_from_u64(seed);

            let packings = self.pack_all_with_rng(max_n, &mut rng);

            // Calculate total score for this attempt
            let total_score: f64 = packings.iter()
                .enumerate()
                .map(|(i, p)| {
                    let side = compute_side_length(&p.trees);
                    (side * side) / ((i + 1) as f64)
                })
                .sum();

            if total_score < best_total_score {
                best_total_score = total_score;
                best_packings = Some(packings);
            }
        }

        best_packings.unwrap_or_else(Vec::new)
    }

    /// Internal packing with provided RNG
    fn pack_all_with_rng(&self, max_n: usize, rng: &mut StdRng) -> Vec<Packing> {
        let mut packings: Vec<Packing> = Vec::with_capacity(max_n);
        let mut prev_trees: Vec<PlacedTree> = Vec::new();

        for n in 1..=max_n {
            let mut trees = prev_trees.clone();

            // Place new tree using structured heuristics
            let new_tree = self.find_placement(&trees, n, max_n, rng);
            trees.push(new_tree);

            // Run SA passes
            for pass in 0..self.config.sa_passes {
                self.local_search(&mut trees, n, pass, rng);
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
    /// VARIANCE REDUCTION: More structured direction selection
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

        // Compute current bounds
        let (min_x, min_y, max_x, max_y) = compute_bounds(existing);
        let current_width = max_x - min_x;
        let current_height = max_y - min_y;

        // VARIANCE REDUCTION: Use deterministic direction sequence
        let directions = self.generate_structured_directions(n, current_width, current_height, rng);

        for (attempt, dir) in directions.iter().enumerate() {
            if attempt >= self.config.search_attempts {
                break;
            }

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

    /// VARIANCE REDUCTION: Generate structured directions with minimal randomness
    fn generate_structured_directions(
        &self,
        n: usize,
        width: f64,
        height: f64,
        rng: &mut impl Rng,
    ) -> Vec<f64> {
        let num_dirs = self.config.search_attempts;
        let mut directions = Vec::with_capacity(num_dirs);

        // Golden angle for even distribution
        let golden_angle = PI * (3.0 - (5.0_f64).sqrt());

        // Phase 1: Evenly spaced angles (deterministic)
        let evenly_spaced_count = (num_dirs as f64 * 0.4) as usize;
        for i in 0..evenly_spaced_count {
            let angle = (i as f64 / evenly_spaced_count as f64) * 2.0 * PI;
            directions.push(angle);
        }

        // Phase 2: Golden ratio spiral (deterministic based on n)
        let golden_count = (num_dirs as f64 * 0.3) as usize;
        for i in 0..golden_count {
            let base = ((n + i) as f64 * golden_angle) % (2.0 * PI);
            directions.push(base);
        }

        // Phase 3: Dimension-biased directions (semi-deterministic)
        let biased_count = (num_dirs as f64 * 0.2) as usize;
        for i in 0..biased_count {
            let angle = if width < height {
                // Bias horizontal - pack in shorter dimension
                if i % 2 == 0 { 0.0 } else { PI }
            } else {
                // Bias vertical
                if i % 2 == 0 { PI / 2.0 } else { -PI / 2.0 }
            };
            // Add small deterministic offset based on i
            let offset = (i as f64 / biased_count as f64 - 0.5) * PI / 4.0;
            directions.push(angle + offset);
        }

        // Phase 4: Corner directions (deterministic)
        let corners = [PI / 4.0, 3.0 * PI / 4.0, 5.0 * PI / 4.0, 7.0 * PI / 4.0];
        for (i, &corner) in corners.iter().cycle().take(num_dirs - directions.len()).enumerate() {
            let offset = (i as f64 * 0.05) % 0.2 - 0.1;
            directions.push(corner + offset);
        }

        // Only add small jitter if random exploration is enabled
        if rng.gen::<f64>() > self.config.structured_direction_prob {
            // Add a few random directions for exploration
            for _ in 0..5 {
                directions.push(rng.gen_range(0.0..2.0 * PI));
            }
        }

        directions
    }

    /// EVOLVED FUNCTION: Score a placement (lower is better)
    /// Simplified scoring for consistency
    #[inline]
    fn placement_score(&self, tree: &PlacedTree, existing: &[PlacedTree], n: usize) -> f64 {
        let (tree_min_x, tree_min_y, tree_max_x, tree_max_y) = tree.bounds();

        // Compute combined bounds
        let mut pack_min_x = tree_min_x;
        let mut pack_min_y = tree_min_y;
        let mut pack_max_x = tree_max_x;
        let mut pack_max_y = tree_max_y;

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

        // Secondary: balance penalty (prefer square-ish bounds)
        let balance_penalty = (width - height).abs() * 0.10;

        // Extension penalty
        let (old_min_x, old_min_y, old_max_x, old_max_y) = if !existing.is_empty() {
            compute_bounds(existing)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        let x_extension = (pack_max_x - old_max_x).max(0.0) + (old_min_x - pack_min_x).max(0.0);
        let y_extension = (pack_max_y - old_max_y).max(0.0) + (old_min_y - pack_min_y).max(0.0);
        let extension_penalty = (x_extension + y_extension) * 0.08;

        // Neighbor proximity bonus
        let neighbor_bonus = self.neighbor_proximity_bonus(tree, existing);

        // Center penalty (mild preference for centered packing)
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.005 / (n as f64).sqrt();

        side_score + balance_penalty + extension_penalty + center_penalty - neighbor_bonus
    }

    /// Bonus for being close to existing trees (promotes compactness)
    #[inline]
    fn neighbor_proximity_bonus(&self, tree: &PlacedTree, existing: &[PlacedTree]) -> f64 {
        if existing.is_empty() {
            return 0.0;
        }

        let (tree_min_x, tree_min_y, tree_max_x, tree_max_y) = tree.bounds();
        let tree_cx = (tree_min_x + tree_max_x) / 2.0;
        let tree_cy = (tree_min_y + tree_max_y) / 2.0;

        let mut min_dist = f64::INFINITY;
        let mut close_neighbors = 0;

        for other in existing {
            let (ox1, oy1, ox2, oy2) = other.bounds();
            let other_cx = (ox1 + ox2) / 2.0;
            let other_cy = (oy1 + oy2) / 2.0;

            let dx = tree_cx - other_cx;
            let dy = tree_cy - other_cy;
            let dist = (dx * dx + dy * dy).sqrt();

            min_dist = min_dist.min(dist);
            if dist < 0.8 {
                close_neighbors += 1;
            }
        }

        // Bonus for close neighbors and minimum distance
        let dist_bonus = if min_dist < 1.5 { 0.02 * (1.5 - min_dist) } else { 0.0 };
        let neighbor_bonus = 0.005 * close_neighbors as f64;

        dist_bonus + neighbor_bonus
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    /// VARIANCE REDUCTION: Deterministic angle selection based on n
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
        // Deterministic rotation based on n mod 8 for variety
        let base_rotation = ((n % 8) as f64) * 45.0;
        vec![
            base_rotation,
            (base_rotation + 90.0) % 360.0,
            (base_rotation + 180.0) % 360.0,
            (base_rotation + 270.0) % 360.0,
            (base_rotation + 45.0) % 360.0,
            (base_rotation + 135.0) % 360.0,
            (base_rotation + 225.0) % 360.0,
            (base_rotation + 315.0) % 360.0,
        ]
    }

    /// EVOLVED FUNCTION: Local search with simulated annealing
    /// VARIANCE REDUCTION: Faster early exit, more deterministic moves
    fn local_search(&self, trees: &mut Vec<PlacedTree>, n: usize, pass: usize, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config: Vec<PlacedTree> = trees.clone();

        let temp_multiplier = match pass {
            0 => 1.0,
            _ => 0.35,
        };
        let mut temp = self.config.sa_initial_temp * temp_multiplier;

        let base_iterations = match pass {
            0 => self.config.sa_iterations + n * 100,
            _ => self.config.sa_iterations / 2 + n * 50,
        };

        let mut iterations_without_improvement = 0;

        // Cache boundary info
        let mut boundary_cache_iter = 0;
        let mut boundary_info: Vec<(usize, BoundaryEdge)> = Vec::new();

        for iter in 0..base_iterations {
            // VARIANCE REDUCTION: Tighter early exit
            if iterations_without_improvement >= self.config.early_exit_threshold {
                break;
            }

            // Update boundary cache every 300 iterations
            if iter == 0 || iter - boundary_cache_iter >= 300 {
                boundary_info = self.find_boundary_trees_with_edges(trees);
                boundary_cache_iter = iter;
            }

            // Select tree to move
            let (idx, edge) = if !boundary_info.is_empty() && rng.gen::<f64>() < self.config.boundary_focus_prob {
                let bi = &boundary_info[rng.gen_range(0..boundary_info.len())];
                (bi.0, bi.1)
            } else {
                (rng.gen_range(0..trees.len()), BoundaryEdge::None)
            };

            let old_tree = trees[idx].clone();

            let success = self.sa_move(trees, idx, temp, edge, rng);

            if success {
                let new_side = compute_side_length(trees);
                let delta = new_side - current_side;

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

        if best_side < compute_side_length(trees) {
            *trees = best_config;
        }
    }

    /// Find trees on the bounding box boundary
    #[inline]
    fn find_boundary_trees_with_edges(&self, trees: &[PlacedTree]) -> Vec<(usize, BoundaryEdge)> {
        if trees.is_empty() {
            return Vec::new();
        }

        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
        let eps = 0.015;

        let mut boundary_info: Vec<(usize, BoundaryEdge)> = Vec::new();

        for (i, tree) in trees.iter().enumerate() {
            let (bx1, by1, bx2, by2) = tree.bounds();

            let on_left = (bx1 - min_x).abs() < eps;
            let on_right = (bx2 - max_x).abs() < eps;
            let on_bottom = (by1 - min_y).abs() < eps;
            let on_top = (by2 - max_y).abs() < eps;

            let edge = match (on_left, on_right, on_top, on_bottom) {
                (true, true, _, _) | (_, _, true, true) => BoundaryEdge::Corner,
                (true, _, true, _) | (true, _, _, true) => BoundaryEdge::Corner,
                (_, true, true, _) | (_, true, _, true) => BoundaryEdge::Corner,
                (true, false, false, false) => BoundaryEdge::Left,
                (false, true, false, false) => BoundaryEdge::Right,
                (false, false, true, false) => BoundaryEdge::Top,
                (false, false, false, true) => BoundaryEdge::Bottom,
                _ => continue,
            };

            boundary_info.push((i, edge));
        }

        boundary_info
    }

    /// SA move operator
    #[inline]
    fn sa_move(
        &self,
        trees: &mut [PlacedTree],
        idx: usize,
        temp: f64,
        edge: BoundaryEdge,
        rng: &mut impl Rng,
    ) -> bool {
        let old = &trees[idx];
        let old_x = old.x;
        let old_y = old.y;
        let old_angle = old.angle_deg;

        let scale = self.config.translation_scale * (0.3 + temp * 1.5);

        // Boundary-aware moves
        let move_type = match edge {
            BoundaryEdge::Left => {
                match rng.gen_range(0..10) {
                    0..=4 => 0,  // Move right
                    5..=6 => 1,  // Move vertically
                    7..=8 => 2,  // Rotate
                    _ => 3,      // Small random
                }
            }
            BoundaryEdge::Right => {
                match rng.gen_range(0..10) {
                    0..=4 => 4,  // Move left
                    5..=6 => 1,  // Move vertically
                    7..=8 => 2,  // Rotate
                    _ => 3,      // Small random
                }
            }
            BoundaryEdge::Top => {
                match rng.gen_range(0..10) {
                    0..=4 => 5,  // Move down
                    5..=6 => 6,  // Move horizontally
                    7..=8 => 2,  // Rotate
                    _ => 3,      // Small random
                }
            }
            BoundaryEdge::Bottom => {
                match rng.gen_range(0..10) {
                    0..=4 => 7,  // Move up
                    5..=6 => 6,  // Move horizontally
                    7..=8 => 2,  // Rotate
                    _ => 3,      // Small random
                }
            }
            BoundaryEdge::Corner => {
                match rng.gen_range(0..10) {
                    0..=4 => 8,  // Move toward center
                    5..=6 => 2,  // Rotate
                    7..=8 => 9,  // Diagonal move
                    _ => 3,      // Small random
                }
            }
            BoundaryEdge::None => {
                rng.gen_range(0..10)
            }
        };

        match move_type {
            0 => {
                // Move right
                let dx = rng.gen_range(scale * 0.3..scale);
                let dy = rng.gen_range(-scale * 0.2..scale * 0.2);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            1 => {
                // Move vertically
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x, old_y + dy, old_angle);
            }
            2 => {
                // Rotate
                let angles = [45.0, 90.0, -45.0, -90.0];
                let delta = angles[rng.gen_range(0..angles.len())];
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            3 => {
                // Small random move
                let dx = rng.gen_range(-scale * 0.5..scale * 0.5);
                let dy = rng.gen_range(-scale * 0.5..scale * 0.5);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            4 => {
                // Move left
                let dx = rng.gen_range(-scale..-scale * 0.3);
                let dy = rng.gen_range(-scale * 0.2..scale * 0.2);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            5 => {
                // Move down
                let dx = rng.gen_range(-scale * 0.2..scale * 0.2);
                let dy = rng.gen_range(-scale..-scale * 0.3);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            6 => {
                // Move horizontally
                let dx = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y, old_angle);
            }
            7 => {
                // Move up
                let dx = rng.gen_range(-scale * 0.2..scale * 0.2);
                let dy = rng.gen_range(scale * 0.3..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            8 => {
                // Move toward center
                let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                let bbox_cx = (min_x + max_x) / 2.0;
                let bbox_cy = (min_y + max_y) / 2.0;

                let dx = (bbox_cx - old_x) * self.config.center_pull_strength * (0.5 + temp);
                let dy = (bbox_cy - old_y) * self.config.center_pull_strength * (0.5 + temp);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            9 => {
                // Diagonal move
                let diag = rng.gen_range(-scale..scale);
                let sign = if rng.gen() { 1.0 } else { -1.0 };
                trees[idx] = PlacedTree::new(old_x + diag, old_y + sign * diag, old_angle);
            }
            _ => {
                // Radial move
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.08 {
                    let delta_r = rng.gen_range(-0.06..0.06) * (1.0 + temp);
                    let new_mag = (mag + delta_r).max(0.0);
                    let scale_r = new_mag / mag;
                    trees[idx] = PlacedTree::new(old_x * scale_r, old_y * scale_r, old_angle);
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

    #[test]
    fn test_reproducibility() {
        // VARIANCE REDUCTION: Test that same config produces same results
        let packer = EvolvedPacker::default();
        let packings1 = packer.pack_all(30);
        let packings2 = packer.pack_all(30);

        let score1 = calculate_score(&packings1);
        let score2 = calculate_score(&packings2);

        // Should be identical due to deterministic seeding
        assert!((score1 - score2).abs() < 1e-10, "Scores should be identical: {} vs {}", score1, score2);
    }
}
