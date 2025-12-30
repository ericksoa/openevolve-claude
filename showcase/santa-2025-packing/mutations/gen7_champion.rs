//! Evolved Packing Algorithm - Generation 7 TIERED N-SCALING
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
//! MUTATION STRATEGY: TIERED N-SCALING (Gen7)
//! Use different strategies based on n value:
//!
//! Key improvements from Gen6:
//! - n < 30: Maximum effort (50000 iterations, 500 attempts) - these matter most for score
//! - n 30-80: Medium effort (30000 iterations, 300 attempts)
//! - n 80-150: Standard (20000 iterations, 200 attempts)
//! - n > 150: Minimal (10000 iterations, 150 attempts)
//! - Keep all density features throughout
//!
//! Rationale: Small n packings contribute more to score (score += side^2/n)
//! so spending more compute on small n values gives better ROI
//!
//! Target: Beat Gen6's 93.23 at n=200 with tiered effort allocation

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Tiered configuration based on n value
#[derive(Clone)]
pub struct TieredConfig {
    pub search_attempts: usize,
    pub direction_samples: usize,
    pub sa_iterations: usize,
    pub sa_initial_temp: f64,
    pub sa_cooling_rate: f64,
    pub sa_min_temp: f64,
    pub translation_scale: f64,
    pub rotation_granularity: f64,
    pub center_pull_strength: f64,
    pub sa_passes: usize,
    pub early_exit_threshold: usize,
    pub boundary_focus_prob: f64,
    // DENSITY parameters (kept from Gen6)
    pub density_grid_resolution: usize,
    pub gap_penalty_weight: f64,
    pub local_density_radius: f64,
    pub fill_move_prob: f64,
}

impl TieredConfig {
    /// Create configuration based on current n value
    pub fn for_n(n: usize) -> Self {
        if n < 30 {
            // TIER 1: Maximum effort for small n (highest score impact)
            Self {
                search_attempts: 500,            // Maximum attempts
                direction_samples: 96,           // More directions
                sa_iterations: 50000,            // Maximum iterations
                sa_initial_temp: 0.50,           // Higher temp for exploration
                sa_cooling_rate: 0.99996,        // Very slow cooling
                sa_min_temp: 0.000005,           // Very low minimum
                translation_scale: 0.045,        // Precise moves
                rotation_granularity: 45.0,      // 8 angles
                center_pull_strength: 0.08,      // Strong pull
                sa_passes: 3,                    // Extra pass for small n
                early_exit_threshold: 3000,      // Very patient
                boundary_focus_prob: 0.88,       // High boundary focus
                // DENSITY parameters
                density_grid_resolution: 25,     // Higher resolution
                gap_penalty_weight: 0.18,        // Stronger gap penalty
                local_density_radius: 0.6,       // Larger radius
                fill_move_prob: 0.18,            // More fill moves
            }
        } else if n < 80 {
            // TIER 2: Medium effort
            Self {
                search_attempts: 300,            // Good attempts
                direction_samples: 80,           // Good direction coverage
                sa_iterations: 30000,            // Good iterations
                sa_initial_temp: 0.48,           // Moderate temp
                sa_cooling_rate: 0.99995,        // Slow cooling
                sa_min_temp: 0.000006,           // Low minimum
                translation_scale: 0.050,        // Moderate precision
                rotation_granularity: 45.0,      // 8 angles
                center_pull_strength: 0.075,     // Moderate pull
                sa_passes: 2,                    // Standard passes
                early_exit_threshold: 2200,      // Good patience
                boundary_focus_prob: 0.86,       // Good boundary focus
                // DENSITY parameters
                density_grid_resolution: 22,     // Good resolution
                gap_penalty_weight: 0.16,        // Good gap penalty
                local_density_radius: 0.55,      // Good radius
                fill_move_prob: 0.16,            // Moderate fill moves
            }
        } else if n < 150 {
            // TIER 3: Standard effort
            Self {
                search_attempts: 200,            // Standard attempts
                direction_samples: 72,           // Standard directions
                sa_iterations: 20000,            // Standard iterations
                sa_initial_temp: 0.45,           // Standard temp
                sa_cooling_rate: 0.99994,        // Standard cooling
                sa_min_temp: 0.000008,           // Standard minimum
                translation_scale: 0.055,        // Standard precision
                rotation_granularity: 45.0,      // 8 angles
                center_pull_strength: 0.07,      // Standard pull
                sa_passes: 2,                    // Standard passes
                early_exit_threshold: 1800,      // Standard patience
                boundary_focus_prob: 0.85,       // Standard boundary focus
                // DENSITY parameters
                density_grid_resolution: 20,     // Standard resolution
                gap_penalty_weight: 0.15,        // Standard gap penalty
                local_density_radius: 0.5,       // Standard radius
                fill_move_prob: 0.15,            // Standard fill moves
            }
        } else {
            // TIER 4: Minimal effort for large n (lowest score impact)
            Self {
                search_attempts: 150,            // Minimal attempts
                direction_samples: 60,           // Fewer directions
                sa_iterations: 10000,            // Minimal iterations
                sa_initial_temp: 0.42,           // Lower temp
                sa_cooling_rate: 0.99992,        // Faster cooling
                sa_min_temp: 0.00001,            // Higher minimum
                translation_scale: 0.060,        // Less precision needed
                rotation_granularity: 45.0,      // 8 angles
                center_pull_strength: 0.065,     // Weaker pull
                sa_passes: 2,                    // Standard passes
                early_exit_threshold: 1200,      // Less patience
                boundary_focus_prob: 0.82,       // Lower boundary focus
                // DENSITY parameters
                density_grid_resolution: 18,     // Lower resolution
                gap_penalty_weight: 0.12,        // Lower gap penalty
                local_density_radius: 0.45,      // Smaller radius
                fill_move_prob: 0.12,            // Fewer fill moves
            }
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

/// Main evolved packer with tiered effort
pub struct EvolvedPacker;

impl Default for EvolvedPacker {
    fn default() -> Self {
        Self
    }
}

impl EvolvedPacker {
    /// Pack all n from 1 to max_n with tiered effort
    pub fn pack_all(&self, max_n: usize) -> Vec<Packing> {
        let mut rng = rand::thread_rng();
        let mut packings: Vec<Packing> = Vec::with_capacity(max_n);
        let mut prev_trees: Vec<PlacedTree> = Vec::new();

        for n in 1..=max_n {
            // Get tiered configuration for this n
            let config = TieredConfig::for_n(n);

            let mut trees = prev_trees.clone();

            // Place new tree using density-aware heuristics
            let new_tree = self.find_placement(&trees, n, max_n, &config, &mut rng);
            trees.push(new_tree);

            // Run SA passes with density-aware moves
            for pass in 0..config.sa_passes {
                self.local_search(&mut trees, n, pass, &config, &mut rng);
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
        config: &TieredConfig,
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
        let gaps = self.find_gaps(existing, min_x, min_y, max_x, max_y, config);

        for attempt in 0..config.search_attempts {
            // DENSITY: Sometimes target gaps directly
            let dir = if !gaps.is_empty() && attempt % 4 == 0 {
                // Target a gap
                let gap = &gaps[attempt % gaps.len()];
                let gap_cx = (gap.0 + gap.2) / 2.0;
                let gap_cy = (gap.1 + gap.3) / 2.0;
                gap_cy.atan2(gap_cx)
            } else {
                self.select_direction(n, current_width, current_height, config, rng)
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
                    let score = self.placement_score(&candidate, existing, n, config);
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
    fn placement_score(&self, tree: &PlacedTree, existing: &[PlacedTree], n: usize, config: &TieredConfig) -> f64 {
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
        let local_density = self.calculate_local_density(tree_cx, tree_cy, existing, config);

        // Reward high local density (tree is filling a gap)
        // Higher density = lower penalty (negative reward)
        let density_bonus = -config.gap_penalty_weight * local_density;

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
        let gap_penalty = self.estimate_unusable_gap(tree, existing) * config.gap_penalty_weight;

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
    fn calculate_local_density(&self, cx: f64, cy: f64, trees: &[PlacedTree], config: &TieredConfig) -> f64 {
        let radius = config.local_density_radius;
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
    fn find_gaps(&self, trees: &[PlacedTree], min_x: f64, min_y: f64, max_x: f64, max_y: f64, config: &TieredConfig) -> Vec<(f64, f64, f64, f64)> {
        if trees.is_empty() {
            return Vec::new();
        }

        let mut gaps = Vec::new();
        let grid_res = config.density_grid_resolution;
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
    fn select_direction(&self, n: usize, width: f64, height: f64, config: &TieredConfig, rng: &mut impl Rng) -> f64 {
        let num_dirs = config.direction_samples;

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
    /// DENSITY: Include gap-filling moves
    fn local_search(&self, trees: &mut Vec<PlacedTree>, n: usize, pass: usize, config: &TieredConfig, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config: Vec<PlacedTree> = trees.clone();

        let temp_multiplier = match pass {
            0 => 1.0,
            1 => 0.35,
            _ => 0.20, // Third pass for small n
        };
        let mut temp = config.sa_initial_temp * temp_multiplier;

        let base_iterations = match pass {
            0 => config.sa_iterations + n * 110,
            1 => config.sa_iterations / 2 + n * 55,
            _ => config.sa_iterations / 3 + n * 35, // Third pass
        };

        let mut iterations_without_improvement = 0;

        // Cache boundary info
        let mut boundary_cache_iter = 0;
        let mut boundary_info: Vec<(usize, BoundaryEdge)> = Vec::new();

        for iter in 0..base_iterations {
            if iterations_without_improvement >= config.early_exit_threshold {
                break;
            }

            // Update boundary cache every 350 iterations
            if iter == 0 || iter - boundary_cache_iter >= 350 {
                boundary_info = self.find_boundary_trees_with_edges(trees);
                boundary_cache_iter = iter;
            }

            // DENSITY: Choose between boundary optimization and gap-filling
            let do_fill_move = rng.gen::<f64>() < config.fill_move_prob;

            let (idx, edge) = if do_fill_move {
                // DENSITY: Try to fill a gap by moving an interior tree
                let interior_trees: Vec<usize> = (0..trees.len())
                    .filter(|&i| !boundary_info.iter().any(|(bi, _)| *bi == i))
                    .collect();

                if !interior_trees.is_empty() && rng.gen::<f64>() < 0.5 {
                    (interior_trees[rng.gen_range(0..interior_trees.len())], BoundaryEdge::None)
                } else {
                    // Fall back to boundary tree
                    if !boundary_info.is_empty() {
                        let bi = &boundary_info[rng.gen_range(0..boundary_info.len())];
                        (bi.0, bi.1)
                    } else {
                        (rng.gen_range(0..trees.len()), BoundaryEdge::None)
                    }
                }
            } else if !boundary_info.is_empty() && rng.gen::<f64>() < config.boundary_focus_prob {
                let bi = &boundary_info[rng.gen_range(0..boundary_info.len())];
                (bi.0, bi.1)
            } else {
                (rng.gen_range(0..trees.len()), BoundaryEdge::None)
            };

            let old_tree = trees[idx].clone();

            let success = self.density_aware_move(trees, idx, temp, edge, do_fill_move, config, rng);

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

            temp = (temp * config.sa_cooling_rate).max(config.sa_min_temp);
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

    /// DENSITY: Move operator with gap-filling awareness
    #[inline]
    fn density_aware_move(
        &self,
        trees: &mut [PlacedTree],
        idx: usize,
        temp: f64,
        edge: BoundaryEdge,
        is_fill_move: bool,
        config: &TieredConfig,
        rng: &mut impl Rng,
    ) -> bool {
        let old = &trees[idx];
        let old_x = old.x;
        let old_y = old.y;
        let old_angle = old.angle_deg;

        let scale = config.translation_scale * (0.3 + temp * 1.5);

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
                    let gaps = self.find_gaps(trees, min_x, min_y, max_x, max_y, config);
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

                    let dx = (bbox_cx - old_x) * config.center_pull_strength * (0.5 + temp);
                    let dy = (bbox_cy - old_y) * config.center_pull_strength * (0.5 + temp);
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
    fn test_tiered_config() {
        // Verify tiered configs are created correctly
        let tier1 = TieredConfig::for_n(10);
        let tier2 = TieredConfig::for_n(50);
        let tier3 = TieredConfig::for_n(100);
        let tier4 = TieredConfig::for_n(180);

        // Tier 1 should have highest effort
        assert!(tier1.sa_iterations > tier2.sa_iterations);
        assert!(tier1.search_attempts > tier2.search_attempts);

        // Tier 2 should have more effort than tier 3
        assert!(tier2.sa_iterations > tier3.sa_iterations);
        assert!(tier2.search_attempts > tier3.search_attempts);

        // Tier 3 should have more effort than tier 4
        assert!(tier3.sa_iterations > tier4.sa_iterations);
        assert!(tier3.search_attempts > tier4.search_attempts);
    }
}
