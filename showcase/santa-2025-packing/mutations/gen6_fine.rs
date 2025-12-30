//! Evolved Packing Algorithm - Generation 6 ULTRA-FINE PRECISION
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
//! MUTATION STRATEGY: ULTRA-FINE PRECISION (Gen6)
//! Focus on micro-optimizations for incremental improvements:
//!
//! Key improvements from Gen5:
//! - Binary search precision: 0.00005 (10x finer than Gen5's 0.001)
//! - Very small SA moves: translation_scale 0.02 (from 0.06)
//! - Fine rotation: 15-degree increments (24 angles vs Gen5's 8)
//! - More SA iterations: 50000 for thorough fine-tuning
//! - No early exit - explore fully for maximum improvement
//! - Extremely slow cooling for extended refinement
//!
//! Target: Beat Gen5's 95.69 at n=200 with ultra-fine precision

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

    // Multi-pass settings
    pub sa_passes: usize,

    // Binary search precision
    pub binary_search_precision: f64,

    // SMART MOVES: Boundary focus probability (90% boundary, 10% random)
    pub boundary_focus_prob: f64,
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen6 ULTRA-FINE PRECISION configuration
        Self {
            search_attempts: 250,            // Keep from Gen5
            direction_samples: 64,           // Keep from Gen5
            sa_iterations: 50000,            // 2x Gen5 for thorough exploration
            sa_initial_temp: 0.4,            // Slightly lower for focused exploitation
            sa_cooling_rate: 0.999975,       // Even slower cooling for extended refinement
            sa_min_temp: 0.000001,           // Lower minimum for fine-tuning at end
            translation_scale: 0.02,         // 3x smaller than Gen5 (0.06) for precision
            rotation_granularity: 15.0,      // 24 angles (vs Gen5's 8 at 45 degrees)
            center_pull_strength: 0.05,      // Gentler pull for precision
            sa_passes: 2,                    // Keep from Gen5
            binary_search_precision: 0.00005, // 10x finer than Gen5's 0.001
            boundary_focus_prob: 0.90,       // Keep from Gen5
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
    Corner, // Blocking two edges
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
    pub fn pack_all(&self, max_n: usize) -> Vec<Packing> {
        let mut rng = rand::thread_rng();
        let mut packings: Vec<Packing> = Vec::with_capacity(max_n);
        let mut prev_trees: Vec<PlacedTree> = Vec::new();

        for n in 1..=max_n {
            let mut trees = prev_trees.clone();

            // Place new tree using evolved heuristics
            let new_tree = self.find_placement(&trees, n, max_n, &mut rng);
            trees.push(new_tree);

            // Run SA passes with smart moves
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
    /// Gen6: Uses ultra-fine binary search precision (0.00005)
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
                // Gen6: Ultra-fine precision 0.00005 (10x finer than Gen5)
                let mut low = 0.0;
                let mut high = 12.0;

                while high - low > self.config.binary_search_precision {
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
    /// Gen6: Same scoring as Gen5, focus is on precision not scoring changes
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
        let balance_penalty = (width - height).abs() * 0.12;

        // Penalize extending the limiting dimension
        let limiting_dim_penalty = if width > height {
            // Width is limiting, penalize x extension
            let tree_extends_x = (max_x > pack_max_x - 0.01) || (min_x < pack_min_x + 0.01);
            if tree_extends_x { 0.05 } else { 0.0 }
        } else {
            // Height is limiting, penalize y extension
            let tree_extends_y = (max_y > pack_max_y - 0.01) || (min_y < pack_min_y + 0.01);
            if tree_extends_y { 0.05 } else { 0.0 }
        };

        // Tertiary: preference for compact center
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.006 / (n as f64).sqrt();

        // Density bonus
        let area = width * height;
        let density_bonus = if area > 0.0 {
            -0.01 * (n as f64 / area).min(2.0)
        } else {
            0.0
        };

        side_score + balance_penalty + limiting_dim_penalty + center_penalty + density_bonus
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    /// Gen6: 24 angles at 15-degree increments (vs Gen5's 8 at 45 degrees)
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
        // Generate all 24 angles at 15-degree increments
        let mut angles: Vec<f64> = (0..24).map(|i| i as f64 * 15.0).collect();

        // Prioritize based on n for variety
        let offset = (n % 24) as f64 * 15.0;
        angles.sort_by(|a, b| {
            let dist_a = ((a - offset).abs()).min((a - offset + 360.0).abs()).min((a - offset - 360.0).abs());
            let dist_b = ((b - offset).abs()).min((b - offset + 360.0).abs()).min((b - offset - 360.0).abs());
            dist_a.partial_cmp(&dist_b).unwrap()
        });

        angles
    }

    /// EVOLVED FUNCTION: Select direction angle for placement search
    #[inline]
    fn select_direction(&self, n: usize, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;

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
                let corner_weight = ((4.0 * angle).sin().abs() + (4.0 * angle).cos().abs()) / 2.0;
                let threshold = 0.2;
                if rng.gen::<f64>() < corner_weight.max(threshold) {
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

    /// EVOLVED FUNCTION: Local search with simulated annealing
    /// Gen6: No early exit, 50000 iterations for thorough exploration
    fn local_search(&self, trees: &mut Vec<PlacedTree>, n: usize, pass: usize, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config: Vec<PlacedTree> = trees.clone();

        // Adjust temperature based on pass number
        let temp_multiplier = match pass {
            0 => 1.0,
            _ => 0.35, // Lower temp for second pass
        };
        let mut temp = self.config.sa_initial_temp * temp_multiplier;

        // Gen6: More iterations for thorough exploration
        let base_iterations = match pass {
            0 => self.config.sa_iterations + n * 150, // Increased from n*100
            _ => self.config.sa_iterations / 2 + n * 75, // Increased from n*50
        };

        // Gen6: NO early exit - explore fully
        // (removed early_exit_threshold check)

        // Cache boundary info with edge tracking
        let mut boundary_cache_iter = 0;
        let mut boundary_info: Vec<(usize, BoundaryEdge)> = Vec::new();

        for iter in 0..base_iterations {
            // Update boundary cache every 400 iterations
            if iter == 0 || iter - boundary_cache_iter >= 400 {
                boundary_info = self.find_boundary_trees_with_edges(trees);
                boundary_cache_iter = iter;
            }

            // 90% chance to pick boundary tree, 10% random
            let (idx, edge) = if !boundary_info.is_empty() && rng.gen::<f64>() < self.config.boundary_focus_prob {
                let bi = &boundary_info[rng.gen_range(0..boundary_info.len())];
                (bi.0, bi.1)
            } else {
                (rng.gen_range(0..trees.len()), BoundaryEdge::None)
            };

            let old_tree = trees[idx].clone();

            // Use targeted move based on which boundary the tree is blocking
            let success = self.smart_move(trees, idx, temp, edge, rng);

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

        // Restore best configuration found during search
        if best_side < compute_side_length(trees) {
            *trees = best_config;
        }
    }

    /// Find trees on the bounding box boundary and which edge they block
    #[inline]
    fn find_boundary_trees_with_edges(&self, trees: &[PlacedTree]) -> Vec<(usize, BoundaryEdge)> {
        if trees.is_empty() {
            return Vec::new();
        }

        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
        let eps = 0.015; // Slightly larger epsilon for edge detection

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
                _ => continue, // Not on boundary
            };

            boundary_info.push((i, edge));
        }

        boundary_info
    }

    /// Gen6: Ultra-fine targeted move operator
    /// Very small moves (translation_scale 0.02) for precision
    #[inline]
    fn smart_move(
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

        // Choose move type based on which boundary we're blocking
        let move_type = match edge {
            BoundaryEdge::Left => {
                // Move right (toward center)
                match rng.gen_range(0..10) {
                    0..=4 => 0, // Move right
                    5..=6 => 1, // Slide along boundary (up/down)
                    7..=8 => 2, // Rotate to reduce footprint
                    _ => 3,     // General move
                }
            }
            BoundaryEdge::Right => {
                // Move left (toward center)
                match rng.gen_range(0..10) {
                    0..=4 => 4, // Move left
                    5..=6 => 1, // Slide along boundary
                    7..=8 => 2, // Rotate
                    _ => 3,     // General move
                }
            }
            BoundaryEdge::Top => {
                // Move down (toward center)
                match rng.gen_range(0..10) {
                    0..=4 => 5, // Move down
                    5..=6 => 6, // Slide along boundary (left/right)
                    7..=8 => 2, // Rotate
                    _ => 3,     // General move
                }
            }
            BoundaryEdge::Bottom => {
                // Move up (toward center)
                match rng.gen_range(0..10) {
                    0..=4 => 7, // Move up
                    5..=6 => 6, // Slide along boundary
                    7..=8 => 2, // Rotate
                    _ => 3,     // General move
                }
            }
            BoundaryEdge::Corner => {
                // Corner: need diagonal moves or rotation
                match rng.gen_range(0..10) {
                    0..=4 => 8, // Gradient descent toward bbox center
                    5..=6 => 2, // Rotate to reduce footprint
                    7..=8 => 9, // Diagonal slide
                    _ => 3,     // General move
                }
            }
            BoundaryEdge::None => {
                // Interior tree: any move
                rng.gen_range(0..10)
            }
        };

        // Gen6: Very small scale (0.02 base) for ultra-fine precision
        let scale = self.config.translation_scale * (0.3 + temp * 1.5);

        match move_type {
            0 => {
                // Move right (for left boundary trees)
                let dx = rng.gen_range(scale * 0.3..scale);
                let dy = rng.gen_range(-scale * 0.2..scale * 0.2);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            1 => {
                // Slide vertically along boundary
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x, old_y + dy, old_angle);
            }
            2 => {
                // Gen6: Rotate with finer granularity (15-degree increments)
                let angles = [15.0, 30.0, 45.0, 60.0, 90.0, -15.0, -30.0, -45.0, -60.0, -90.0];
                let delta = angles[rng.gen_range(0..angles.len())];
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            3 => {
                // General small move
                let dx = rng.gen_range(-scale * 0.5..scale * 0.5);
                let dy = rng.gen_range(-scale * 0.5..scale * 0.5);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            4 => {
                // Move left (for right boundary trees)
                let dx = rng.gen_range(-scale..-scale * 0.3);
                let dy = rng.gen_range(-scale * 0.2..scale * 0.2);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            5 => {
                // Move down (for top boundary trees)
                let dx = rng.gen_range(-scale * 0.2..scale * 0.2);
                let dy = rng.gen_range(-scale..-scale * 0.3);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            6 => {
                // Slide horizontally along boundary
                let dx = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y, old_angle);
            }
            7 => {
                // Move up (for bottom boundary trees)
                let dx = rng.gen_range(-scale * 0.2..scale * 0.2);
                let dy = rng.gen_range(scale * 0.3..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            8 => {
                // Gradient descent toward bounding box center (gentler pull)
                let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                let bbox_cx = (min_x + max_x) / 2.0;
                let bbox_cy = (min_y + max_y) / 2.0;

                let dx = (bbox_cx - old_x) * self.config.center_pull_strength * (0.5 + temp);
                let dy = (bbox_cy - old_y) * self.config.center_pull_strength * (0.5 + temp);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            9 => {
                // Diagonal slide (for corners)
                let diag = rng.gen_range(-scale..scale);
                let sign = if rng.gen() { 1.0 } else { -1.0 };
                trees[idx] = PlacedTree::new(old_x + diag, old_y + sign * diag, old_angle);
            }
            _ => {
                // Polar/orbit move
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.08 {
                    let delta_r = rng.gen_range(-0.04..0.04) * (1.0 + temp); // Smaller than Gen5
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
}
