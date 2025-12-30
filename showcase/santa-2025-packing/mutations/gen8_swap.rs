//! Evolved Packing Algorithm - Generation 8 TREE SWAPS
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
//! MUTATION STRATEGY: TREE SWAPS (Gen8)
//! Add swap operations between trees as new move operators:
//!
//! Key improvements from Gen6 (density champion):
//! 1. New move operator: swap positions of two trees
//! 2. Swap rotations between trees (keep positions, exchange angles)
//! 3. Swap boundary trees with interior trees
//! 4. Select swap candidates based on bounding box contribution
//! 5. Keep all density features from Gen6
//!
//! Rationale: Swapping trees can escape local optima that single-tree
//! moves cannot reach. A boundary tree might fit better in another position,
//! and an interior tree might reduce the bounding box if moved to the boundary.
//!
//! Target: Beat Gen6's 94.14 at n=200 with swap-based exploration

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

    // DENSITY MAXIMIZATION: Parameters (kept from Gen6)
    pub density_grid_resolution: usize,  // Grid cells for density tracking
    pub gap_penalty_weight: f64,         // Weight for penalizing gaps
    pub local_density_radius: f64,       // Radius for local density calculation
    pub fill_move_prob: f64,             // Probability of fill-gap move

    // TREE SWAPS: New parameters (Gen8)
    pub swap_prob: f64,                  // Probability of trying a swap move
    pub swap_boundary_interior_prob: f64, // Probability of boundary-interior swap vs random swap
    pub swap_rotation_prob: f64,         // Probability of rotation-only swap
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen8 TREE SWAPS: Configuration with swap operations
        Self {
            search_attempts: 280,            // From Gen6
            direction_samples: 72,           // From Gen6
            sa_iterations: 30000,            // Slightly more iterations for swaps
            sa_initial_temp: 0.48,           // Slightly higher for swap exploration
            sa_cooling_rate: 0.99994,        // From Gen6
            sa_min_temp: 0.000008,           // From Gen6
            translation_scale: 0.055,        // From Gen6
            rotation_granularity: 45.0,      // 8 angles
            center_pull_strength: 0.07,      // From Gen6
            sa_passes: 2,                    // From Gen6
            early_exit_threshold: 2000,      // Slightly more patience for swaps
            boundary_focus_prob: 0.82,       // Slightly lower to make room for swaps
            // DENSITY parameters (from Gen6)
            density_grid_resolution: 20,     // From Gen6
            gap_penalty_weight: 0.15,        // From Gen6
            local_density_radius: 0.5,       // From Gen6
            fill_move_prob: 0.12,            // Slightly lower to make room for swaps
            // SWAP parameters (new for Gen8)
            swap_prob: 0.15,                 // 15% chance of swap move
            swap_boundary_interior_prob: 0.6, // 60% of swaps are boundary-interior
            swap_rotation_prob: 0.25,        // 25% of swaps are rotation-only
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

            // Run SA passes with density-aware and swap moves
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
    /// DENSITY: Prefer placements that fill gaps and maximize local density
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

        // Compute current bounds and density info
        let (min_x, min_y, max_x, max_y) = compute_bounds(existing);
        let current_width = max_x - min_x;
        let current_height = max_y - min_y;

        // Find gaps in the packing for density-aware placement
        let gaps = self.find_gaps(existing, min_x, min_y, max_x, max_y);

        for attempt in 0..self.config.search_attempts {
            // DENSITY: Sometimes target gaps directly
            let dir = if !gaps.is_empty() && attempt % 4 == 0 {
                // Target a gap
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

    /// EVOLVED FUNCTION: Score a placement (lower is better)
    /// DENSITY MAXIMIZATION: Score based on gap filling and local density
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

        // DENSITY: Calculate local density around the new tree
        let tree_cx = (tree_min_x + tree_max_x) / 2.0;
        let tree_cy = (tree_min_y + tree_max_y) / 2.0;
        let local_density = self.calculate_local_density(tree_cx, tree_cy, existing);

        // Reward high local density (tree is filling a gap)
        // Higher density = lower penalty (negative reward)
        let density_bonus = -self.config.gap_penalty_weight * local_density;

        // DENSITY: Penalize placements that extend the bounding box into empty space
        let (old_min_x, old_min_y, old_max_x, old_max_y) = if !existing.is_empty() {
            compute_bounds(existing)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        // Calculate how much this placement extends the bounding box
        let x_extension = (pack_max_x - old_max_x).max(0.0) + (old_min_x - pack_min_x).max(0.0);
        let y_extension = (pack_max_y - old_max_y).max(0.0) + (old_min_y - pack_min_y).max(0.0);
        let extension_penalty = (x_extension + y_extension) * 0.08;

        // DENSITY: Penalize leaving unusable gaps
        let gap_penalty = self.estimate_unusable_gap(tree, existing) * self.config.gap_penalty_weight;

        // Center penalty (mild preference for centered packing)
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.005 / (n as f64).sqrt();

        // Neighbor bonus: reward being close to existing trees (filling gaps)
        let neighbor_bonus = self.neighbor_proximity_bonus(tree, existing);

        side_score + balance_penalty + extension_penalty + gap_penalty + center_penalty + density_bonus - neighbor_bonus
    }

    /// DENSITY: Calculate local density around a point
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
                // Weight by inverse distance (closer trees contribute more)
                count += 1.0 - (dist_sq / radius_sq).sqrt();
            }
        }

        count
    }

    /// DENSITY: Estimate if placement creates an unusable gap
    #[inline]
    fn estimate_unusable_gap(&self, tree: &PlacedTree, existing: &[PlacedTree]) -> f64 {
        if existing.is_empty() {
            return 0.0;
        }

        let (tree_min_x, tree_min_y, tree_max_x, tree_max_y) = tree.bounds();

        // Check for small gaps between this tree and neighbors
        let mut gap_penalty = 0.0;
        let min_useful_gap = 0.15; // Minimum gap that can fit another tree part
        let max_wasteful_gap = 0.4; // Gaps larger than this are potentially fillable

        for other in existing {
            let (ox1, oy1, ox2, oy2) = other.bounds();

            // Horizontal gap
            if tree_min_y < oy2 && tree_max_y > oy1 {
                // Overlapping in y
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
                // Overlapping in x
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

    /// DENSITY: Bonus for being close to existing trees (promotes compactness)
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

    /// DENSITY: Find gaps in the current packing
    fn find_gaps(&self, trees: &[PlacedTree], min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<(f64, f64, f64, f64)> {
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

        // Create occupancy grid
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

        // Find empty cells that are surrounded by occupied cells (gaps)
        for i in 1..grid_res - 1 {
            for j in 1..grid_res - 1 {
                let idx = j * grid_res + i;
                if !occupied[idx] {
                    // Check if surrounded by occupied cells
                    let neighbors_occupied =
                        occupied[(j - 1) * grid_res + i] as i32 +
                        occupied[(j + 1) * grid_res + i] as i32 +
                        occupied[j * grid_res + i - 1] as i32 +
                        occupied[j * grid_res + i + 1] as i32;

                    if neighbors_occupied >= 2 {
                        // This is a gap
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
    /// DENSITY: Bias toward less dense regions to fill gaps
    #[inline]
    fn select_direction(&self, n: usize, width: f64, height: f64, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;

        let strategy = rng.gen::<f64>();

        if strategy < 0.45 {
            // Structured: evenly spaced with small jitter
            let base_idx = rng.gen_range(0..num_dirs);
            let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
            base + rng.gen_range(-0.05..0.05)
        } else if strategy < 0.70 {
            // DENSITY: Bias toward the shorter dimension (to fill before expanding)
            if width < height {
                // Pack more horizontally
                let angle = if rng.gen() { 0.0 } else { PI };
                angle + rng.gen_range(-PI / 4.0..PI / 4.0)
            } else {
                // Pack more vertically
                let angle = if rng.gen() { PI / 2.0 } else { -PI / 2.0 };
                angle + rng.gen_range(-PI / 4.0..PI / 4.0)
            }
        } else if strategy < 0.85 {
            // Corner bias
            let corners = [PI / 4.0, 3.0 * PI / 4.0, 5.0 * PI / 4.0, 7.0 * PI / 4.0];
            corners[rng.gen_range(0..4)] + rng.gen_range(-0.15..0.15)
        } else {
            // Golden angle spiral
            let golden_angle = PI * (3.0 - (5.0_f64).sqrt());
            let base = (n as f64 * golden_angle) % (2.0 * PI);
            let offset = rng.gen_range(0..8) as f64 * PI / 4.0;
            (base + offset + rng.gen_range(-0.1..0.1)) % (2.0 * PI)
        }
    }

    /// EVOLVED FUNCTION: Local search with simulated annealing
    /// GEN8 SWAPS: Include swap moves between trees
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
            0 => self.config.sa_iterations + n * 110,
            _ => self.config.sa_iterations / 2 + n * 55,
        };

        let mut iterations_without_improvement = 0;

        // Cache boundary info
        let mut boundary_cache_iter = 0;
        let mut boundary_info: Vec<(usize, BoundaryEdge)> = Vec::new();
        let mut interior_trees: Vec<usize> = Vec::new();

        for iter in 0..base_iterations {
            if iterations_without_improvement >= self.config.early_exit_threshold {
                break;
            }

            // Update boundary cache every 350 iterations
            if iter == 0 || iter - boundary_cache_iter >= 350 {
                boundary_info = self.find_boundary_trees_with_edges(trees);
                interior_trees = (0..trees.len())
                    .filter(|&i| !boundary_info.iter().any(|(bi, _)| *bi == i))
                    .collect();
                boundary_cache_iter = iter;
            }

            // GEN8 SWAPS: Choose move type
            let move_choice = rng.gen::<f64>();

            if move_choice < self.config.swap_prob && trees.len() >= 2 {
                // GEN8: Try a swap move
                let old_trees = trees.clone();
                let success = self.try_swap_move(trees, &boundary_info, &interior_trees, temp, rng);

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
                        *trees = old_trees;
                        iterations_without_improvement += 1;
                    }
                } else {
                    *trees = old_trees;
                    iterations_without_improvement += 1;
                }
            } else {
                // Standard single-tree move (density-aware or boundary)
                let do_fill_move = rng.gen::<f64>() < self.config.fill_move_prob;

                let (idx, edge) = if do_fill_move {
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

                let success = self.density_aware_move(trees, idx, temp, edge, do_fill_move, rng);

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
            }

            temp = (temp * self.config.sa_cooling_rate).max(self.config.sa_min_temp);
        }

        if best_side < compute_side_length(trees) {
            *trees = best_config;
        }
    }

    /// GEN8 SWAPS: Try a swap move between two trees
    /// Returns true if the swap results in a valid configuration
    #[inline]
    fn try_swap_move(
        &self,
        trees: &mut [PlacedTree],
        boundary_info: &[(usize, BoundaryEdge)],
        interior_trees: &[usize],
        temp: f64,
        rng: &mut impl Rng,
    ) -> bool {
        let n = trees.len();
        if n < 2 {
            return false;
        }

        let swap_type = rng.gen::<f64>();

        if swap_type < self.config.swap_rotation_prob {
            // GEN8: Rotation swap - exchange angles between two trees
            self.try_rotation_swap(trees, rng)
        } else if swap_type < self.config.swap_rotation_prob + self.config.swap_boundary_interior_prob * (1.0 - self.config.swap_rotation_prob) {
            // GEN8: Boundary-interior swap
            self.try_boundary_interior_swap(trees, boundary_info, interior_trees, temp, rng)
        } else {
            // GEN8: Position swap - exchange positions of two trees
            self.try_position_swap(trees, boundary_info, temp, rng)
        }
    }

    /// GEN8: Swap rotations between two trees (keep positions, exchange angles)
    #[inline]
    fn try_rotation_swap(&self, trees: &mut [PlacedTree], rng: &mut impl Rng) -> bool {
        let n = trees.len();
        if n < 2 {
            return false;
        }

        let idx1 = rng.gen_range(0..n);
        let mut idx2 = rng.gen_range(0..n);
        while idx2 == idx1 {
            idx2 = rng.gen_range(0..n);
        }

        // Get current angles
        let angle1 = trees[idx1].angle_deg;
        let angle2 = trees[idx2].angle_deg;

        // Create new trees with swapped angles
        let new_tree1 = PlacedTree::new(trees[idx1].x, trees[idx1].y, angle2);
        let new_tree2 = PlacedTree::new(trees[idx2].x, trees[idx2].y, angle1);

        // Check if swap is valid (no overlaps)
        trees[idx1] = new_tree1;
        trees[idx2] = new_tree2;

        !has_any_overlap(trees, idx1) && !has_any_overlap(trees, idx2)
    }

    /// GEN8: Swap positions of a boundary tree with an interior tree
    #[inline]
    fn try_boundary_interior_swap(
        &self,
        trees: &mut [PlacedTree],
        boundary_info: &[(usize, BoundaryEdge)],
        interior_trees: &[usize],
        _temp: f64,
        rng: &mut impl Rng,
    ) -> bool {
        if boundary_info.is_empty() || interior_trees.is_empty() {
            return false;
        }

        // Select a boundary tree (prioritize those contributing most to bbox)
        let boundary_idx = self.select_high_impact_boundary_tree(trees, boundary_info, rng);

        // Select an interior tree
        let interior_idx = interior_trees[rng.gen_range(0..interior_trees.len())];

        // Get positions
        let (bx, by) = (trees[boundary_idx].x, trees[boundary_idx].y);
        let (ix, iy) = (trees[interior_idx].x, trees[interior_idx].y);
        let b_angle = trees[boundary_idx].angle_deg;
        let i_angle = trees[interior_idx].angle_deg;

        // Try swap with same angles first
        let new_boundary = PlacedTree::new(ix, iy, b_angle);
        let new_interior = PlacedTree::new(bx, by, i_angle);

        trees[boundary_idx] = new_boundary;
        trees[interior_idx] = new_interior;

        if !has_any_overlap(trees, boundary_idx) && !has_any_overlap(trees, interior_idx) {
            return true;
        }

        // Try swap with exchanged angles
        let new_boundary = PlacedTree::new(ix, iy, i_angle);
        let new_interior = PlacedTree::new(bx, by, b_angle);

        trees[boundary_idx] = new_boundary;
        trees[interior_idx] = new_interior;

        !has_any_overlap(trees, boundary_idx) && !has_any_overlap(trees, interior_idx)
    }

    /// GEN8: Swap positions of two trees (potentially with rotation adjustments)
    #[inline]
    fn try_position_swap(
        &self,
        trees: &mut [PlacedTree],
        boundary_info: &[(usize, BoundaryEdge)],
        _temp: f64,
        rng: &mut impl Rng,
    ) -> bool {
        let n = trees.len();
        if n < 2 {
            return false;
        }

        // Prefer swapping boundary trees
        let (idx1, idx2) = if !boundary_info.is_empty() && rng.gen::<f64>() < 0.7 {
            let bi = rng.gen_range(0..boundary_info.len());
            let idx1 = boundary_info[bi].0;
            let idx2 = if boundary_info.len() > 1 && rng.gen::<f64>() < 0.5 {
                // Swap two boundary trees
                let mut bi2 = rng.gen_range(0..boundary_info.len());
                while bi2 == bi {
                    bi2 = rng.gen_range(0..boundary_info.len());
                }
                boundary_info[bi2].0
            } else {
                // Swap boundary with random
                let mut idx2 = rng.gen_range(0..n);
                while idx2 == idx1 {
                    idx2 = rng.gen_range(0..n);
                }
                idx2
            };
            (idx1, idx2)
        } else {
            let idx1 = rng.gen_range(0..n);
            let mut idx2 = rng.gen_range(0..n);
            while idx2 == idx1 {
                idx2 = rng.gen_range(0..n);
            }
            (idx1, idx2)
        };

        // Get positions and angles
        let (x1, y1, a1) = (trees[idx1].x, trees[idx1].y, trees[idx1].angle_deg);
        let (x2, y2, a2) = (trees[idx2].x, trees[idx2].y, trees[idx2].angle_deg);

        // Try different swap variants
        // Variant 1: Swap positions, keep angles
        let new_tree1 = PlacedTree::new(x2, y2, a1);
        let new_tree2 = PlacedTree::new(x1, y1, a2);

        trees[idx1] = new_tree1;
        trees[idx2] = new_tree2;

        if !has_any_overlap(trees, idx1) && !has_any_overlap(trees, idx2) {
            return true;
        }

        // Variant 2: Swap both positions and angles
        let new_tree1 = PlacedTree::new(x2, y2, a2);
        let new_tree2 = PlacedTree::new(x1, y1, a1);

        trees[idx1] = new_tree1;
        trees[idx2] = new_tree2;

        if !has_any_overlap(trees, idx1) && !has_any_overlap(trees, idx2) {
            return true;
        }

        // Variant 3: Try with 90-degree rotation adjustments
        for delta in &[90.0, -90.0, 180.0] {
            let new_tree1 = PlacedTree::new(x2, y2, (a1 + delta).rem_euclid(360.0));
            let new_tree2 = PlacedTree::new(x1, y1, (a2 + delta).rem_euclid(360.0));

            trees[idx1] = new_tree1;
            trees[idx2] = new_tree2;

            if !has_any_overlap(trees, idx1) && !has_any_overlap(trees, idx2) {
                return true;
            }
        }

        false
    }

    /// GEN8: Select a boundary tree that contributes most to the bounding box
    #[inline]
    fn select_high_impact_boundary_tree(
        &self,
        trees: &[PlacedTree],
        boundary_info: &[(usize, BoundaryEdge)],
        rng: &mut impl Rng,
    ) -> usize {
        if boundary_info.is_empty() {
            return 0;
        }

        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
        let side = (max_x - min_x).max(max_y - min_y);

        // Calculate impact score for each boundary tree
        let mut impacts: Vec<(usize, f64)> = boundary_info
            .iter()
            .map(|(idx, edge)| {
                let (bx1, by1, bx2, by2) = trees[*idx].bounds();

                // How much does this tree extend the bounding box?
                let impact = match edge {
                    BoundaryEdge::Left => (min_x - bx1).abs() / side,
                    BoundaryEdge::Right => (bx2 - max_x).abs() / side,
                    BoundaryEdge::Top => (by2 - max_y).abs() / side,
                    BoundaryEdge::Bottom => (min_y - by1).abs() / side,
                    BoundaryEdge::Corner => {
                        // Corner trees have high impact
                        let x_impact = ((min_x - bx1).abs() + (bx2 - max_x).abs()) / side;
                        let y_impact = ((min_y - by1).abs() + (by2 - max_y).abs()) / side;
                        x_impact + y_impact
                    }
                    BoundaryEdge::None => 0.0,
                };

                (*idx, impact)
            })
            .collect();

        // Sort by impact (highest first)
        impacts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select from top 50% with bias toward highest impact
        let top_count = (impacts.len() / 2).max(1);
        let selected = rng.gen_range(0..top_count);
        impacts[selected].0
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

    /// DENSITY: Move operator with gap-filling awareness
    #[inline]
    fn density_aware_move(
        &self,
        trees: &mut [PlacedTree],
        idx: usize,
        temp: f64,
        edge: BoundaryEdge,
        is_fill_move: bool,
        rng: &mut impl Rng,
    ) -> bool {
        let old = &trees[idx];
        let old_x = old.x;
        let old_y = old.y;
        let old_angle = old.angle_deg;

        let scale = self.config.translation_scale * (0.3 + temp * 1.5);

        // DENSITY: If this is a fill move, try to move toward less dense areas within bbox
        if is_fill_move {
            let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
            let bbox_cx = (min_x + max_x) / 2.0;
            let bbox_cy = (min_y + max_y) / 2.0;

            // Find a less dense direction within the bounding box
            let move_type = rng.gen_range(0..4);
            match move_type {
                0 => {
                    // Move toward center of bbox (compacting)
                    let dx = (bbox_cx - old_x) * 0.1 * (0.5 + temp);
                    let dy = (bbox_cy - old_y) * 0.1 * (0.5 + temp);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
                1 => {
                    // Small random move to fill gaps
                    let dx = rng.gen_range(-scale * 0.4..scale * 0.4);
                    let dy = rng.gen_range(-scale * 0.4..scale * 0.4);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
                2 => {
                    // Rotate to potentially fill gap better
                    let angles = [45.0, 90.0, -45.0, -90.0, 30.0, -30.0];
                    let delta = angles[rng.gen_range(0..angles.len())];
                    let new_angle = (old_angle + delta).rem_euclid(360.0);
                    trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
                }
                _ => {
                    // Move toward nearest gap
                    let gaps = self.find_gaps(trees, min_x, min_y, max_x, max_y);
                    if !gaps.is_empty() {
                        let gap = &gaps[rng.gen_range(0..gaps.len())];
                        let gap_cx = (gap.0 + gap.2) / 2.0;
                        let gap_cy = (gap.1 + gap.3) / 2.0;
                        let dx = (gap_cx - old_x) * 0.05;
                        let dy = (gap_cy - old_y) * 0.05;
                        trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                    } else {
                        return false;
                    }
                }
            }
        } else {
            // Standard boundary-aware moves (from Gen5)
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

/// GEN8: Check if a tree at idx overlaps with any other tree
fn has_any_overlap(trees: &[PlacedTree], idx: usize) -> bool {
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
    fn test_swap_config() {
        let config = EvolvedConfig::default();
        // Verify swap probabilities sum to valid range
        assert!(config.swap_prob > 0.0 && config.swap_prob < 1.0);
        assert!(config.swap_boundary_interior_prob > 0.0 && config.swap_boundary_interior_prob < 1.0);
        assert!(config.swap_rotation_prob > 0.0 && config.swap_rotation_prob < 1.0);
    }
}
