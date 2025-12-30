//! Evolved Packing Algorithm - Generation 10 LOOK-BACK OPTIMIZATION
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
//! MUTATION STRATEGY: LOOK-BACK OPTIMIZATION (Gen10)
//! After placing n trees, go back and re-optimize earlier trees:
//!
//! Key improvements from Gen6:
//! - After placing tree n, re-run SA on all trees 1 to n
//! - Start with current positions but allow more drastic moves
//! - For n > 50: only re-optimize the last 30 trees to save time
//! - This allows escaping local minima created by incremental placement
//! - More aggressive re-optimization with higher initial temperature
//! - Periodic full re-optimization at key milestones
//!
//! Target: Beat Gen6's 94.14 at n=200 with look-back optimization

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

    // Early exit threshold
    pub early_exit_threshold: usize,

    // Boundary focus probability
    pub boundary_focus_prob: f64,

    // DENSITY MAXIMIZATION: Parameters from Gen6
    pub density_grid_resolution: usize,
    pub gap_penalty_weight: f64,
    pub local_density_radius: f64,
    pub fill_move_prob: f64,

    // LOOK-BACK OPTIMIZATION: New parameters
    pub lookback_window: usize,       // How many trees to re-optimize
    pub lookback_threshold: usize,    // Start lookback after this many trees
    pub lookback_temp_multiplier: f64, // Higher temp for more exploration
    pub lookback_iterations: usize,   // Iterations for lookback SA
    pub full_reopt_interval: usize,   // Full re-optimization every N trees
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen10 LOOK-BACK OPTIMIZATION: Enhanced configuration
        Self {
            search_attempts: 280,
            direction_samples: 72,
            sa_iterations: 22000,           // Reduced since we do lookback
            sa_initial_temp: 0.45,
            sa_cooling_rate: 0.99994,
            sa_min_temp: 0.000008,
            translation_scale: 0.055,
            rotation_granularity: 45.0,
            center_pull_strength: 0.07,
            sa_passes: 2,
            early_exit_threshold: 1500,     // Faster initial passes
            boundary_focus_prob: 0.85,
            // DENSITY parameters from Gen6
            density_grid_resolution: 20,
            gap_penalty_weight: 0.15,
            local_density_radius: 0.5,
            fill_move_prob: 0.15,
            // LOOK-BACK parameters
            lookback_window: 30,            // Re-optimize last 30 trees
            lookback_threshold: 10,         // Start after 10 trees placed
            lookback_temp_multiplier: 2.5,  // Much higher temp for exploration
            lookback_iterations: 15000,     // Dedicated lookback iterations
            full_reopt_interval: 25,        // Full re-opt every 25 trees
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
    pub fn pack_all(&self, max_n: usize) -> Vec<Packing> {
        let mut rng = rand::thread_rng();
        let mut packings: Vec<Packing> = Vec::with_capacity(max_n);
        let mut prev_trees: Vec<PlacedTree> = Vec::new();

        for n in 1..=max_n {
            let mut trees = prev_trees.clone();

            // Place new tree using density-aware heuristics
            let new_tree = self.find_placement(&trees, n, max_n, &mut rng);
            trees.push(new_tree);

            // Run standard SA passes
            for pass in 0..self.config.sa_passes {
                self.local_search(&mut trees, n, pass, &mut rng);
            }

            // LOOK-BACK OPTIMIZATION: Re-optimize earlier trees
            if n >= self.config.lookback_threshold {
                self.lookback_optimization(&mut trees, n, &mut rng);
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

    /// LOOK-BACK OPTIMIZATION: Re-optimize a window of trees
    fn lookback_optimization(
        &self,
        trees: &mut Vec<PlacedTree>,
        n: usize,
        rng: &mut impl Rng,
    ) {
        let num_trees = trees.len();
        if num_trees <= 1 {
            return;
        }

        // Determine which trees to re-optimize
        let (start_idx, do_full) = if n % self.config.full_reopt_interval == 0 {
            // Periodic full re-optimization
            (0, true)
        } else if num_trees > self.config.lookback_window {
            // Only re-optimize recent trees
            (num_trees - self.config.lookback_window, false)
        } else {
            // Re-optimize all trees
            (0, false)
        };

        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config = trees.clone();

        // Higher temperature for more exploration in lookback
        let temp_mult = if do_full {
            self.config.lookback_temp_multiplier * 1.5
        } else {
            self.config.lookback_temp_multiplier
        };
        let mut temp = self.config.sa_initial_temp * temp_mult;

        let iterations = if do_full {
            self.config.lookback_iterations * 2
        } else {
            self.config.lookback_iterations
        };

        let mut iterations_without_improvement = 0;
        let early_exit = self.config.early_exit_threshold * 2;

        // Cache boundary info
        let mut boundary_info = self.find_boundary_trees_with_edges(trees);
        let mut boundary_cache_iter = 0;

        for iter in 0..iterations {
            if iterations_without_improvement >= early_exit {
                break;
            }

            // Update boundary cache periodically
            if iter - boundary_cache_iter >= 500 {
                boundary_info = self.find_boundary_trees_with_edges(trees);
                boundary_cache_iter = iter;
            }

            // Select a tree to move from the lookback window
            let idx = if !boundary_info.is_empty() && rng.gen::<f64>() < 0.7 {
                // Prefer boundary trees within the window
                let window_boundary: Vec<_> = boundary_info.iter()
                    .filter(|(i, _)| *i >= start_idx)
                    .collect();
                if !window_boundary.is_empty() {
                    let bi = window_boundary[rng.gen_range(0..window_boundary.len())];
                    bi.0
                } else {
                    rng.gen_range(start_idx..num_trees)
                }
            } else {
                rng.gen_range(start_idx..num_trees)
            };

            let edge = boundary_info.iter()
                .find(|(i, _)| *i == idx)
                .map(|(_, e)| *e)
                .unwrap_or(BoundaryEdge::None);

            let old_tree = trees[idx].clone();

            // Use more aggressive moves for lookback
            let success = self.lookback_move(trees, idx, temp, edge, rng);

            if success {
                let new_side = compute_side_length(trees);
                let delta = new_side - current_side;

                // More accepting of uphill moves for exploration
                let accept_prob = if delta <= 0.0 {
                    1.0
                } else {
                    (-delta / temp).exp()
                };

                if rng.gen::<f64>() < accept_prob {
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

        // Restore best configuration found
        if best_side < compute_side_length(trees) {
            *trees = best_config;
        }
    }

    /// LOOK-BACK: More aggressive move operator for escaping local minima
    #[inline]
    fn lookback_move(
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

        // Larger scale for more drastic moves
        let scale = self.config.translation_scale * (0.5 + temp * 3.0);

        let move_type = rng.gen_range(0..12);

        match move_type {
            0 => {
                // Large translation toward center
                let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                let bbox_cx = (min_x + max_x) / 2.0;
                let bbox_cy = (min_y + max_y) / 2.0;
                let strength = self.config.center_pull_strength * (1.0 + temp * 2.0);
                let dx = (bbox_cx - old_x) * strength;
                let dy = (bbox_cy - old_y) * strength;
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            1 => {
                // Swap-like move: try to move to a gap
                let gaps = self.find_gaps(trees, idx);
                if !gaps.is_empty() {
                    let gap = &gaps[rng.gen_range(0..gaps.len())];
                    let gap_cx = (gap.0 + gap.2) / 2.0;
                    let gap_cy = (gap.1 + gap.3) / 2.0;
                    // Move partially toward the gap
                    let dx = (gap_cx - old_x) * 0.3;
                    let dy = (gap_cy - old_y) * 0.3;
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                } else {
                    return false;
                }
            }
            2 => {
                // Large rotation
                let angles = [45.0, 90.0, 135.0, 180.0, -45.0, -90.0, -135.0];
                let delta = angles[rng.gen_range(0..angles.len())];
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            3 => {
                // Combined rotation and translation
                let angles = [45.0, 90.0, -45.0, -90.0];
                let delta = angles[rng.gen_range(0..angles.len())];
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, new_angle);
            }
            4 | 5 => {
                // Edge-aware moves
                match edge {
                    BoundaryEdge::Left => {
                        let dx = rng.gen_range(scale * 0.5..scale * 1.5);
                        trees[idx] = PlacedTree::new(old_x + dx, old_y, old_angle);
                    }
                    BoundaryEdge::Right => {
                        let dx = rng.gen_range(-scale * 1.5..-scale * 0.5);
                        trees[idx] = PlacedTree::new(old_x + dx, old_y, old_angle);
                    }
                    BoundaryEdge::Top => {
                        let dy = rng.gen_range(-scale * 1.5..-scale * 0.5);
                        trees[idx] = PlacedTree::new(old_x, old_y + dy, old_angle);
                    }
                    BoundaryEdge::Bottom => {
                        let dy = rng.gen_range(scale * 0.5..scale * 1.5);
                        trees[idx] = PlacedTree::new(old_x, old_y + dy, old_angle);
                    }
                    BoundaryEdge::Corner => {
                        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                        let bbox_cx = (min_x + max_x) / 2.0;
                        let bbox_cy = (min_y + max_y) / 2.0;
                        let dx = (bbox_cx - old_x) * 0.15 * (1.0 + temp);
                        let dy = (bbox_cy - old_y) * 0.15 * (1.0 + temp);
                        trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                    }
                    BoundaryEdge::None => {
                        let dx = rng.gen_range(-scale..scale);
                        let dy = rng.gen_range(-scale..scale);
                        trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                    }
                }
            }
            6 => {
                // Diagonal move
                let diag = rng.gen_range(-scale * 1.5..scale * 1.5);
                let sign = if rng.gen() { 1.0 } else { -1.0 };
                trees[idx] = PlacedTree::new(old_x + diag, old_y + sign * diag, old_angle);
            }
            7 => {
                // Radial move (scale distance from center)
                let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                let bbox_cx = (min_x + max_x) / 2.0;
                let bbox_cy = (min_y + max_y) / 2.0;
                let dx = old_x - bbox_cx;
                let dy = old_y - bbox_cy;
                let mag = (dx * dx + dy * dy).sqrt();
                if mag > 0.1 {
                    let scale_factor = 1.0 + rng.gen_range(-0.15..0.15) * (1.0 + temp);
                    let new_x = bbox_cx + dx * scale_factor;
                    let new_y = bbox_cy + dy * scale_factor;
                    trees[idx] = PlacedTree::new(new_x, new_y, old_angle);
                } else {
                    return false;
                }
            }
            8 => {
                // Small precise move
                let dx = rng.gen_range(-scale * 0.3..scale * 0.3);
                let dy = rng.gen_range(-scale * 0.3..scale * 0.3);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            9 => {
                // Horizontal slide
                let dx = rng.gen_range(-scale * 2.0..scale * 2.0);
                trees[idx] = PlacedTree::new(old_x + dx, old_y, old_angle);
            }
            10 => {
                // Vertical slide
                let dy = rng.gen_range(-scale * 2.0..scale * 2.0);
                trees[idx] = PlacedTree::new(old_x, old_y + dy, old_angle);
            }
            _ => {
                // Random walk
                let angle = rng.gen_range(0.0..2.0 * PI);
                let dist = rng.gen_range(0.0..scale);
                let dx = dist * angle.cos();
                let dy = dist * angle.sin();
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
        }

        !has_overlap(trees, idx)
    }

    /// Find gaps, excluding the tree at idx from the calculation
    fn find_gaps(&self, trees: &[PlacedTree], exclude_idx: usize) -> Vec<(f64, f64, f64, f64)> {
        if trees.len() <= 1 {
            return Vec::new();
        }

        // Compute bounds excluding the tree at idx
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for (i, tree) in trees.iter().enumerate() {
            if i == exclude_idx {
                continue;
            }
            let (bx1, by1, bx2, by2) = tree.bounds();
            min_x = min_x.min(bx1);
            min_y = min_y.min(by1);
            max_x = max_x.max(bx2);
            max_y = max_y.max(by2);
        }

        if min_x >= max_x || min_y >= max_y {
            return Vec::new();
        }

        let mut gaps = Vec::new();
        let grid_res = self.config.density_grid_resolution;
        let cell_w = (max_x - min_x) / grid_res as f64;
        let cell_h = (max_y - min_y) / grid_res as f64;

        if cell_w <= 0.0 || cell_h <= 0.0 {
            return Vec::new();
        }

        // Create occupancy grid
        let mut occupied = vec![false; grid_res * grid_res];

        for (i, tree) in trees.iter().enumerate() {
            if i == exclude_idx {
                continue;
            }
            let (bx1, by1, bx2, by2) = tree.bounds();
            let i1 = ((bx1 - min_x) / cell_w).floor().max(0.0) as usize;
            let i2 = ((bx2 - min_x) / cell_w).ceil().min(grid_res as f64) as usize;
            let j1 = ((by1 - min_y) / cell_h).floor().max(0.0) as usize;
            let j2 = ((by2 - min_y) / cell_h).ceil().min(grid_res as f64) as usize;

            for ii in i1..i2.min(grid_res) {
                for jj in j1..j2.min(grid_res) {
                    occupied[jj * grid_res + ii] = true;
                }
            }
        }

        // Find empty cells surrounded by occupied cells
        for i in 1..grid_res - 1 {
            for j in 1..grid_res - 1 {
                let idx = j * grid_res + i;
                if !occupied[idx] {
                    let neighbors_occupied =
                        occupied[(j - 1) * grid_res + i] as i32 +
                        occupied[(j + 1) * grid_res + i] as i32 +
                        occupied[j * grid_res + i - 1] as i32 +
                        occupied[j * grid_res + i + 1] as i32;

                    if neighbors_occupied >= 2 {
                        let gx1 = min_x + i as f64 * cell_w;
                        let gy1 = min_y + j as f64 * cell_h;
                        let gx2 = gx1 + cell_w;
                        let gy2 = gy1 + cell_h;
                        gaps.push((gx1, gy1, gx2, gy2));
                    }
                }
            }
        }

        gaps
    }

    /// EVOLVED FUNCTION: Find best placement for new tree
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

        let (min_x, min_y, max_x, max_y) = compute_bounds(existing);
        let current_width = max_x - min_x;
        let current_height = max_y - min_y;

        // Find gaps for density-aware placement
        let gaps = self.find_placement_gaps(existing, min_x, min_y, max_x, max_y);

        for attempt in 0..self.config.search_attempts {
            // Sometimes target gaps directly
            let dir = if !gaps.is_empty() && attempt % 4 == 0 {
                let gap = &gaps[attempt % gaps.len()];
                let gap_cx = (gap.0 + gap.2) / 2.0;
                let gap_cy = (gap.1 + gap.3) / 2.0;
                gap_cy.atan2(gap_cx)
            } else {
                self.select_direction(n, current_width, current_height, rng)
            };

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

    /// Find gaps for placement scoring
    fn find_placement_gaps(&self, trees: &[PlacedTree], min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<(f64, f64, f64, f64)> {
        if trees.is_empty() {
            return Vec::new();
        }

        let mut gaps = Vec::new();
        let grid_res = self.config.density_grid_resolution;
        let cell_w = (max_x - min_x) / grid_res as f64;
        let cell_h = (max_y - min_y) / grid_res as f64;

        if cell_w <= 0.0 || cell_h <= 0.0 {
            return Vec::new();
        }

        let mut occupied = vec![false; grid_res * grid_res];

        for tree in trees {
            let (bx1, by1, bx2, by2) = tree.bounds();
            let i1 = ((bx1 - min_x) / cell_w).floor().max(0.0) as usize;
            let i2 = ((bx2 - min_x) / cell_w).ceil().min(grid_res as f64) as usize;
            let j1 = ((by1 - min_y) / cell_h).floor().max(0.0) as usize;
            let j2 = ((by2 - min_y) / cell_h).ceil().min(grid_res as f64) as usize;

            for i in i1..i2.min(grid_res) {
                for j in j1..j2.min(grid_res) {
                    occupied[j * grid_res + i] = true;
                }
            }
        }

        for i in 1..grid_res - 1 {
            for j in 1..grid_res - 1 {
                let idx = j * grid_res + i;
                if !occupied[idx] {
                    let neighbors_occupied =
                        occupied[(j - 1) * grid_res + i] as i32 +
                        occupied[(j + 1) * grid_res + i] as i32 +
                        occupied[j * grid_res + i - 1] as i32 +
                        occupied[j * grid_res + i + 1] as i32;

                    if neighbors_occupied >= 2 {
                        let gx1 = min_x + i as f64 * cell_w;
                        let gy1 = min_y + j as f64 * cell_h;
                        let gx2 = gx1 + cell_w;
                        let gy2 = gy1 + cell_h;
                        gaps.push((gx1, gy1, gx2, gy2));
                    }
                }
            }
        }

        gaps
    }

    /// EVOLVED FUNCTION: Score a placement (lower is better)
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

        // Secondary: balance penalty
        let balance_penalty = (width - height).abs() * 0.10;

        // Local density calculation
        let tree_cx = (tree_min_x + tree_max_x) / 2.0;
        let tree_cy = (tree_min_y + tree_max_y) / 2.0;
        let local_density = self.calculate_local_density(tree_cx, tree_cy, existing);
        let density_bonus = -self.config.gap_penalty_weight * local_density;

        // Extension penalty
        let (old_min_x, old_min_y, old_max_x, old_max_y) = if !existing.is_empty() {
            compute_bounds(existing)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        let x_extension = (pack_max_x - old_max_x).max(0.0) + (old_min_x - pack_min_x).max(0.0);
        let y_extension = (pack_max_y - old_max_y).max(0.0) + (old_min_y - pack_min_y).max(0.0);
        let extension_penalty = (x_extension + y_extension) * 0.08;

        // Gap penalty
        let gap_penalty = self.estimate_unusable_gap(tree, existing) * self.config.gap_penalty_weight;

        // Center penalty
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.005 / (n as f64).sqrt();

        // Neighbor bonus
        let neighbor_bonus = self.neighbor_proximity_bonus(tree, existing);

        side_score + balance_penalty + extension_penalty + gap_penalty + center_penalty + density_bonus - neighbor_bonus
    }

    /// Calculate local density around a point
    #[inline]
    fn calculate_local_density(&self, cx: f64, cy: f64, trees: &[PlacedTree]) -> f64 {
        let radius = self.config.local_density_radius;
        let radius_sq = radius * radius;
        let mut count = 0.0;

        for tree in trees {
            let (bx1, by1, bx2, by2) = tree.bounds();
            let tree_cx = (bx1 + bx2) / 2.0;
            let tree_cy = (by1 + by2) / 2.0;

            let dx = tree_cx - cx;
            let dy = tree_cy - cy;
            let dist_sq = dx * dx + dy * dy;

            if dist_sq < radius_sq {
                count += 1.0 - (dist_sq / radius_sq).sqrt();
            }
        }

        count
    }

    /// Estimate unusable gap penalty
    #[inline]
    fn estimate_unusable_gap(&self, tree: &PlacedTree, existing: &[PlacedTree]) -> f64 {
        if existing.is_empty() {
            return 0.0;
        }

        let (tree_min_x, tree_min_y, tree_max_x, tree_max_y) = tree.bounds();
        let mut gap_penalty = 0.0;
        let min_useful_gap = 0.15;
        let max_wasteful_gap = 0.4;

        for other in existing {
            let (ox1, oy1, ox2, oy2) = other.bounds();

            // Horizontal gap
            if tree_min_y < oy2 && tree_max_y > oy1 {
                if tree_min_x > ox2 {
                    let gap = tree_min_x - ox2;
                    if gap > min_useful_gap && gap < max_wasteful_gap {
                        gap_penalty += (max_wasteful_gap - gap) / max_wasteful_gap * 0.1;
                    }
                } else if tree_max_x < ox1 {
                    let gap = ox1 - tree_max_x;
                    if gap > min_useful_gap && gap < max_wasteful_gap {
                        gap_penalty += (max_wasteful_gap - gap) / max_wasteful_gap * 0.1;
                    }
                }
            }

            // Vertical gap
            if tree_min_x < ox2 && tree_max_x > ox1 {
                if tree_min_y > oy2 {
                    let gap = tree_min_y - oy2;
                    if gap > min_useful_gap && gap < max_wasteful_gap {
                        gap_penalty += (max_wasteful_gap - gap) / max_wasteful_gap * 0.1;
                    }
                } else if tree_max_y < oy1 {
                    let gap = oy1 - tree_max_y;
                    if gap > min_useful_gap && gap < max_wasteful_gap {
                        gap_penalty += (max_wasteful_gap - gap) / max_wasteful_gap * 0.1;
                    }
                }
            }
        }

        gap_penalty
    }

    /// Neighbor proximity bonus
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

        let dist_bonus = if min_dist < 1.5 { 0.02 * (1.5 - min_dist) } else { 0.0 };
        let neighbor_bonus = 0.005 * close_neighbors as f64;

        dist_bonus + neighbor_bonus
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
    fn select_direction(&self, n: usize, width: f64, height: f64, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;

        let strategy = rng.gen::<f64>();

        if strategy < 0.45 {
            let base_idx = rng.gen_range(0..num_dirs);
            let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
            base + rng.gen_range(-0.05..0.05)
        } else if strategy < 0.70 {
            if width < height {
                let angle = if rng.gen() { 0.0 } else { PI };
                angle + rng.gen_range(-PI / 4.0..PI / 4.0)
            } else {
                let angle = if rng.gen() { PI / 2.0 } else { -PI / 2.0 };
                angle + rng.gen_range(-PI / 4.0..PI / 4.0)
            }
        } else if strategy < 0.85 {
            let corners = [PI / 4.0, 3.0 * PI / 4.0, 5.0 * PI / 4.0, 7.0 * PI / 4.0];
            corners[rng.gen_range(0..4)] + rng.gen_range(-0.15..0.15)
        } else {
            let golden_angle = PI * (3.0 - (5.0_f64).sqrt());
            let base = (n as f64 * golden_angle) % (2.0 * PI);
            let offset = rng.gen_range(0..8) as f64 * PI / 4.0;
            (base + offset + rng.gen_range(-0.1..0.1)) % (2.0 * PI)
        }
    }

    /// EVOLVED FUNCTION: Local search with simulated annealing
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

        let mut boundary_cache_iter = 0;
        let mut boundary_info: Vec<(usize, BoundaryEdge)> = Vec::new();

        for iter in 0..base_iterations {
            if iterations_without_improvement >= self.config.early_exit_threshold {
                break;
            }

            if iter == 0 || iter - boundary_cache_iter >= 350 {
                boundary_info = self.find_boundary_trees_with_edges(trees);
                boundary_cache_iter = iter;
            }

            let do_fill_move = rng.gen::<f64>() < self.config.fill_move_prob;

            let (idx, edge) = if do_fill_move {
                let interior_trees: Vec<usize> = (0..trees.len())
                    .filter(|&i| !boundary_info.iter().any(|(bi, _)| *bi == i))
                    .collect();

                if !interior_trees.is_empty() && rng.gen::<f64>() < 0.5 {
                    (interior_trees[rng.gen_range(0..interior_trees.len())], BoundaryEdge::None)
                } else if !boundary_info.is_empty() {
                    let bi = &boundary_info[rng.gen_range(0..boundary_info.len())];
                    (bi.0, bi.1)
                } else {
                    (rng.gen_range(0..trees.len()), BoundaryEdge::None)
                }
            } else if !boundary_info.is_empty() && rng.gen::<f64>() < self.config.boundary_focus_prob {
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

        let move_type = match edge {
            BoundaryEdge::Left => {
                match rng.gen_range(0..10) {
                    0..=4 => 0,
                    5..=6 => 1,
                    7..=8 => 2,
                    _ => 3,
                }
            }
            BoundaryEdge::Right => {
                match rng.gen_range(0..10) {
                    0..=4 => 4,
                    5..=6 => 1,
                    7..=8 => 2,
                    _ => 3,
                }
            }
            BoundaryEdge::Top => {
                match rng.gen_range(0..10) {
                    0..=4 => 5,
                    5..=6 => 6,
                    7..=8 => 2,
                    _ => 3,
                }
            }
            BoundaryEdge::Bottom => {
                match rng.gen_range(0..10) {
                    0..=4 => 7,
                    5..=6 => 6,
                    7..=8 => 2,
                    _ => 3,
                }
            }
            BoundaryEdge::Corner => {
                match rng.gen_range(0..10) {
                    0..=4 => 8,
                    5..=6 => 2,
                    7..=8 => 9,
                    _ => 3,
                }
            }
            BoundaryEdge::None => {
                rng.gen_range(0..10)
            }
        };

        match move_type {
            0 => {
                let dx = rng.gen_range(scale * 0.3..scale);
                let dy = rng.gen_range(-scale * 0.2..scale * 0.2);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            1 => {
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x, old_y + dy, old_angle);
            }
            2 => {
                let angles = [45.0, 90.0, -45.0, -90.0];
                let delta = angles[rng.gen_range(0..angles.len())];
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            3 => {
                let dx = rng.gen_range(-scale * 0.5..scale * 0.5);
                let dy = rng.gen_range(-scale * 0.5..scale * 0.5);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            4 => {
                let dx = rng.gen_range(-scale..-scale * 0.3);
                let dy = rng.gen_range(-scale * 0.2..scale * 0.2);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            5 => {
                let dx = rng.gen_range(-scale * 0.2..scale * 0.2);
                let dy = rng.gen_range(-scale..-scale * 0.3);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            6 => {
                let dx = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y, old_angle);
            }
            7 => {
                let dx = rng.gen_range(-scale * 0.2..scale * 0.2);
                let dy = rng.gen_range(scale * 0.3..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            8 => {
                let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                let bbox_cx = (min_x + max_x) / 2.0;
                let bbox_cy = (min_y + max_y) / 2.0;

                let dx = (bbox_cx - old_x) * self.config.center_pull_strength * (0.5 + temp);
                let dy = (bbox_cy - old_y) * self.config.center_pull_strength * (0.5 + temp);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            9 => {
                let diag = rng.gen_range(-scale..scale);
                let sign = if rng.gen() { 1.0 } else { -1.0 };
                trees[idx] = PlacedTree::new(old_x + diag, old_y + sign * diag, old_angle);
            }
            _ => {
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
}
