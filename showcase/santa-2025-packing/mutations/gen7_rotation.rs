//! Evolved Packing Algorithm - Generation 7 ROTATION OPTIMIZATION
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
//! MUTATION STRATEGY: ROTATION OPTIMIZATION (Gen7)
//! Focus on finding optimal rotations for tighter packing:
//!
//! Key improvements from Gen6:
//! - Try 24 angles (15-degree increments) during placement for finer granularity
//! - During SA: dedicated rotation-only moves (50% of moves)
//! - Score placements by how rotation affects bounding box contribution
//! - Rotate toward angles that minimize tree's footprint contribution
//! - Keep density features from Gen6 for gap filling
//!
//! Target: Beat Gen6's 93.23 at n=200 with rotation-aware packing

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

    // ROTATION OPTIMIZATION: New parameters for Gen7
    pub num_placement_angles: usize,     // 24 angles (15-degree increments)
    pub rotation_move_prob: f64,         // 50% of SA moves are rotation-only
    pub rotation_footprint_weight: f64,  // Weight for footprint contribution scoring
    pub fine_rotation_search: bool,      // Enable fine rotation search during placement
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen7 ROTATION OPTIMIZATION: Rotation-aware configuration
        Self {
            search_attempts: 300,            // More attempts for rotation exploration
            direction_samples: 72,           // Keep direction coverage from Gen6
            sa_iterations: 30000,            // More iterations for rotation optimization
            sa_initial_temp: 0.42,           // Slightly lower for exploitation
            sa_cooling_rate: 0.99995,        // Slower cooling for rotation convergence
            sa_min_temp: 0.000006,           // Lower minimum for fine rotation tuning
            translation_scale: 0.050,        // Smaller moves, rely more on rotation
            rotation_granularity: 15.0,      // Finer rotation granularity (15 degrees)
            center_pull_strength: 0.065,     // Moderate pull
            sa_passes: 2,                    // Keep 2 passes
            early_exit_threshold: 2000,      // More patience for rotation moves
            boundary_focus_prob: 0.80,       // 80% boundary focus
            // DENSITY parameters from Gen6
            density_grid_resolution: 20,
            gap_penalty_weight: 0.12,        // Slightly reduced, rotation takes priority
            local_density_radius: 0.5,
            fill_move_prob: 0.10,            // Reduced to 10% for rotation focus
            // ROTATION parameters for Gen7
            num_placement_angles: 24,        // 24 angles = 15-degree increments
            rotation_move_prob: 0.50,        // 50% of moves are rotation-only
            rotation_footprint_weight: 0.20, // Weight for footprint scoring
            fine_rotation_search: true,      // Enable fine rotation search
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

            // Place new tree using rotation-aware heuristics
            let new_tree = self.find_placement(&trees, n, max_n, &mut rng);
            trees.push(new_tree);

            // Run SA passes with rotation-focused moves
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
    /// ROTATION OPTIMIZATION: Try 24 angles and score by footprint contribution
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

        // ROTATION: Use 24 angles (15-degree increments) for fine-grained rotation search
        let angles = self.select_angles(n);

        // Compute current bounds and density info
        let (min_x, min_y, max_x, max_y) = compute_bounds(existing);
        let current_width = max_x - min_x;
        let current_height = max_y - min_y;

        // Find gaps in the packing for density-aware placement
        let gaps = self.find_gaps(existing, min_x, min_y, max_x, max_y);

        for attempt in 0..self.config.search_attempts {
            // DENSITY: Sometimes target gaps directly
            let dir = if !gaps.is_empty() && attempt % 5 == 0 {
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

            // ROTATION: Try all 24 angles for each direction
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
                    // ROTATION: Score with rotation-aware scoring
                    let score = self.placement_score(&candidate, existing, n);
                    if score < best_score {
                        best_score = score;
                        best_tree = candidate;
                    }
                }
            }
        }

        // ROTATION: Fine-tune the best placement with local rotation search
        if self.config.fine_rotation_search {
            best_tree = self.fine_tune_rotation(best_tree, existing, n);
        }

        best_tree
    }

    /// ROTATION OPTIMIZATION: Fine-tune rotation angle around best found
    #[inline]
    fn fine_tune_rotation(&self, tree: PlacedTree, existing: &[PlacedTree], n: usize) -> PlacedTree {
        let mut best_tree = tree.clone();
        let mut best_score = self.placement_score(&tree, existing, n);

        // Try small angle adjustments around the current angle
        let fine_angles = [-7.5, -5.0, -2.5, 2.5, 5.0, 7.5];

        for &delta in &fine_angles {
            let new_angle = (tree.angle_deg + delta).rem_euclid(360.0);
            let candidate = PlacedTree::new(tree.x, tree.y, new_angle);

            if is_valid(&candidate, existing) {
                let score = self.placement_score(&candidate, existing, n);
                if score < best_score {
                    best_score = score;
                    best_tree = candidate;
                }
            }
        }

        best_tree
    }

    /// EVOLVED FUNCTION: Score a placement (lower is better)
    /// ROTATION OPTIMIZATION: Include footprint contribution scoring
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
        let balance_penalty = (width - height).abs() * 0.08;

        // ROTATION: Calculate footprint contribution from this tree's rotation
        // A tree's footprint is its bounding box size - smaller is better
        let tree_width = tree_max_x - tree_min_x;
        let tree_height = tree_max_y - tree_min_y;
        let tree_footprint = tree_width.max(tree_height);

        // Compute the "ideal" minimum footprint for any rotation
        // Tree at 45 degrees has smallest axis-aligned bounding box for this shape
        let min_footprint = 0.85; // Empirical minimum footprint

        // Penalty for sub-optimal rotation (larger footprint)
        let footprint_penalty = (tree_footprint - min_footprint).max(0.0) * self.config.rotation_footprint_weight;

        // ROTATION: Score how this rotation contributes to bounding box expansion
        let (old_min_x, old_min_y, old_max_x, old_max_y) = if !existing.is_empty() {
            compute_bounds(existing)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        // Calculate how much this placement extends the bounding box
        let x_extension = (pack_max_x - old_max_x).max(0.0) + (old_min_x - pack_min_x).max(0.0);
        let y_extension = (pack_max_y - old_max_y).max(0.0) + (old_min_y - pack_min_y).max(0.0);

        // ROTATION: Weight extension by tree's footprint in that direction
        let rotation_extension_penalty = if x_extension > 0.0 || y_extension > 0.0 {
            let x_footprint_contrib = if x_extension > 0.0 { tree_width / x_extension.max(0.01) } else { 0.0 };
            let y_footprint_contrib = if y_extension > 0.0 { tree_height / y_extension.max(0.01) } else { 0.0 };
            // Higher contribution = rotation is causing more of the extension
            (x_footprint_contrib + y_footprint_contrib) * 0.03
        } else {
            0.0
        };

        let extension_penalty = (x_extension + y_extension) * 0.10;

        // DENSITY: Calculate local density around the new tree
        let tree_cx = (tree_min_x + tree_max_x) / 2.0;
        let tree_cy = (tree_min_y + tree_max_y) / 2.0;
        let local_density = self.calculate_local_density(tree_cx, tree_cy, existing);

        // Reward high local density (tree is filling a gap)
        let density_bonus = -self.config.gap_penalty_weight * local_density;

        // DENSITY: Penalize leaving unusable gaps
        let gap_penalty = self.estimate_unusable_gap(tree, existing) * self.config.gap_penalty_weight;

        // Center penalty (mild preference for centered packing)
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.004 / (n as f64).sqrt();

        // Neighbor bonus: reward being close to existing trees
        let neighbor_bonus = self.neighbor_proximity_bonus(tree, existing);

        side_score + balance_penalty + extension_penalty + footprint_penalty +
            rotation_extension_penalty + gap_penalty + center_penalty + density_bonus - neighbor_bonus
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

    /// DENSITY: Bonus for being close to existing trees
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

    /// EVOLVED FUNCTION: Select rotation angles to try
    /// ROTATION OPTIMIZATION: 24 angles (15-degree increments) for fine-grained search
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
        // Gen7: 24 angles at 15-degree increments
        let mut angles: Vec<f64> = (0..self.config.num_placement_angles)
            .map(|i| (i as f64 * 360.0 / self.config.num_placement_angles as f64))
            .collect();

        // Prioritize certain angles based on tree index for diversity
        let priority_offset = (n % 6) as f64 * 15.0;
        angles.sort_by(|a, b| {
            let dist_a = ((a - priority_offset).abs() % 90.0).min(90.0 - (a - priority_offset).abs() % 90.0);
            let dist_b = ((b - priority_offset).abs() % 90.0).min(90.0 - (b - priority_offset).abs() % 90.0);
            dist_a.partial_cmp(&dist_b).unwrap()
        });

        angles
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
    /// ROTATION OPTIMIZATION: 50% of moves are rotation-only
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
            0 => self.config.sa_iterations + n * 120,
            _ => self.config.sa_iterations / 2 + n * 60,
        };

        let mut iterations_without_improvement = 0;

        let mut boundary_cache_iter = 0;
        let mut boundary_info: Vec<(usize, BoundaryEdge)> = Vec::new();

        for iter in 0..base_iterations {
            if iterations_without_improvement >= self.config.early_exit_threshold {
                break;
            }

            if iter == 0 || iter - boundary_cache_iter >= 400 {
                boundary_info = self.find_boundary_trees_with_edges(trees);
                boundary_cache_iter = iter;
            }

            // ROTATION OPTIMIZATION: Decide move type
            let move_choice = rng.gen::<f64>();
            let is_rotation_move = move_choice < self.config.rotation_move_prob;
            let is_fill_move = move_choice >= self.config.rotation_move_prob &&
                               move_choice < self.config.rotation_move_prob + self.config.fill_move_prob;

            // Select tree to move
            let (idx, edge) = if is_fill_move {
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

            // ROTATION OPTIMIZATION: Apply appropriate move type
            let success = if is_rotation_move {
                self.rotation_only_move(trees, idx, temp, edge, rng)
            } else if is_fill_move {
                self.density_aware_move(trees, idx, temp, edge, rng)
            } else {
                self.standard_move(trees, idx, temp, edge, rng)
            };

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

    /// ROTATION OPTIMIZATION: Rotation-only move operator
    /// 50% of SA moves - focuses purely on finding better rotation angles
    #[inline]
    fn rotation_only_move(
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

        // ROTATION: Choose rotation strategy based on edge and temperature
        let rotation_type = rng.gen_range(0..6);

        let new_angle = match rotation_type {
            0 => {
                // Small rotation: 15-degree increments
                let delta = if rng.gen() { 15.0 } else { -15.0 };
                (old_angle + delta).rem_euclid(360.0)
            }
            1 => {
                // Medium rotation: 30-45 degrees
                let options = [30.0, -30.0, 45.0, -45.0];
                (old_angle + options[rng.gen_range(0..4)]).rem_euclid(360.0)
            }
            2 => {
                // Large rotation: 90 degrees
                let options = [90.0, -90.0, 180.0];
                (old_angle + options[rng.gen_range(0..3)]).rem_euclid(360.0)
            }
            3 => {
                // Temperature-scaled rotation
                let max_delta = 15.0 + temp * 60.0;
                let delta = rng.gen_range(-max_delta..max_delta);
                (old_angle + delta).rem_euclid(360.0)
            }
            4 => {
                // Edge-aware rotation: rotate toward better edge alignment
                let align_angle = match edge {
                    BoundaryEdge::Left | BoundaryEdge::Right => {
                        // Align vertically
                        let targets = [0.0, 180.0];
                        targets[rng.gen_range(0..2)]
                    }
                    BoundaryEdge::Top | BoundaryEdge::Bottom => {
                        // Align horizontally
                        let targets = [90.0, 270.0];
                        targets[rng.gen_range(0..2)]
                    }
                    BoundaryEdge::Corner => {
                        // Diagonal alignment
                        let targets = [45.0, 135.0, 225.0, 315.0];
                        targets[rng.gen_range(0..4)]
                    }
                    BoundaryEdge::None => {
                        old_angle + rng.gen_range(-45.0..45.0)
                    }
                };
                // Interpolate toward alignment
                let blend = 0.3 + temp * 0.5;
                let delta = (align_angle - old_angle + 540.0) % 360.0 - 180.0;
                (old_angle + delta * blend).rem_euclid(360.0)
            }
            _ => {
                // Fine rotation: very small adjustments
                let delta = rng.gen_range(-7.5..7.5);
                (old_angle + delta).rem_euclid(360.0)
            }
        };

        trees[idx] = PlacedTree::new(old_x, old_y, new_angle);

        !has_overlap(trees, idx)
    }

    /// DENSITY: Gap-filling move operator
    #[inline]
    fn density_aware_move(
        &self,
        trees: &mut [PlacedTree],
        idx: usize,
        temp: f64,
        _edge: BoundaryEdge,
        rng: &mut impl Rng,
    ) -> bool {
        let old = &trees[idx];
        let old_x = old.x;
        let old_y = old.y;
        let old_angle = old.angle_deg;

        let scale = self.config.translation_scale * (0.3 + temp * 1.5);

        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
        let bbox_cx = (min_x + max_x) / 2.0;
        let bbox_cy = (min_y + max_y) / 2.0;

        let move_type = rng.gen_range(0..4);
        match move_type {
            0 => {
                // Move toward center
                let dx = (bbox_cx - old_x) * 0.1 * (0.5 + temp);
                let dy = (bbox_cy - old_y) * 0.1 * (0.5 + temp);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            1 => {
                // Small random move
                let dx = rng.gen_range(-scale * 0.4..scale * 0.4);
                let dy = rng.gen_range(-scale * 0.4..scale * 0.4);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            2 => {
                // Combined rotation and small translation
                let angles = [15.0, 30.0, -15.0, -30.0];
                let delta = angles[rng.gen_range(0..angles.len())];
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                let dx = rng.gen_range(-scale * 0.2..scale * 0.2);
                let dy = rng.gen_range(-scale * 0.2..scale * 0.2);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, new_angle);
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

        !has_overlap(trees, idx)
    }

    /// Standard boundary-aware move operator
    #[inline]
    fn standard_move(
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
                // Move right
                let dx = rng.gen_range(scale * 0.3..scale);
                let dy = rng.gen_range(-scale * 0.2..scale * 0.2);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            1 => {
                // Vertical slide
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x, old_y + dy, old_angle);
            }
            2 => {
                // Rotation (keep some rotation in standard moves too)
                let angles = [15.0, 30.0, 45.0, -15.0, -30.0, -45.0];
                let delta = angles[rng.gen_range(0..angles.len())];
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            3 => {
                // Small random translation
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
                // Horizontal slide
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
                // Pull toward center
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
                // Radial adjustment
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
