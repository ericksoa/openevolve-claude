//! Evolved Packing Algorithm - Generation 6 CORNER AVOIDANCE
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
//! MUTATION STRATEGY: CORNER AVOIDANCE (Gen6)
//! The bounding box is determined by the most extreme points. Avoid corners:
//!
//! Key improvements from Gen5 (Smart Moves):
//! 1. Placement scoring: strong penalty for corner placements
//! 2. SA moves: bias away from corners when possible
//! 3. Track which trees are "corner trees" (touching both edges)
//! 4. Prioritize moving corner trees over edge trees
//! 5. Angle selection: prefer angles that minimize corner footprint
//!
//! Target: Beat Gen5's 95.69 at n=200 by reducing corner congestion

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

    // CORNER AVOIDANCE: Corner penalty weight
    pub corner_penalty_weight: f64,

    // CORNER AVOIDANCE: Priority for moving corner trees (vs edge trees)
    pub corner_tree_priority: f64,
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen6 CORNER AVOIDANCE: Focus on reducing corner congestion
        Self {
            search_attempts: 250,            // Keep from Gen5
            direction_samples: 64,           // Keep from Gen5
            sa_iterations: 28000,            // Slightly more for corner optimization
            sa_initial_temp: 0.45,           // Slightly lower for exploitation
            sa_cooling_rate: 0.99995,        // Keep from Gen5
            sa_min_temp: 0.00001,            // Keep from Gen5
            translation_scale: 0.055,        // Slightly smaller for precision
            rotation_granularity: 45.0,      // Keep 8 angles
            center_pull_strength: 0.10,      // Stronger pull toward center
            sa_passes: 2,                    // Keep from Gen5
            early_exit_threshold: 1800,      // More patience for corner moves
            boundary_focus_prob: 0.92,       // Slightly more focus on boundary
            corner_penalty_weight: 0.15,     // Strong penalty for corners
            corner_tree_priority: 0.70,      // 70% chance to prioritize corner trees
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
    Corner, // Blocking two edges - HIGHEST PRIORITY
    None,
}

/// Corner position classification
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CornerType {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
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

            // Run SA passes with corner-aware moves
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
    /// CORNER AVOIDANCE: Penalize corner placements
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

        // Get current bounds for corner detection
        let (cur_min_x, cur_min_y, cur_max_x, cur_max_y) = compute_bounds(existing);

        let angles = self.select_angles(n, existing);

        for _ in 0..self.config.search_attempts {
            let dir = self.select_direction(n, existing, rng);
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
                    let score = self.placement_score(
                        &candidate, existing, n,
                        cur_min_x, cur_min_y, cur_max_x, cur_max_y
                    );
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
    /// CORNER AVOIDANCE: Strong penalty for corner placements
    #[inline]
    fn placement_score(
        &self,
        tree: &PlacedTree,
        existing: &[PlacedTree],
        n: usize,
        cur_min_x: f64,
        cur_min_y: f64,
        cur_max_x: f64,
        cur_max_y: f64,
    ) -> f64 {
        let (min_x, min_y, max_x, max_y) = tree.bounds();

        // Compute combined bounds
        let mut pack_min_x = min_x.min(cur_min_x);
        let mut pack_min_y = min_y.min(cur_min_y);
        let mut pack_max_x = max_x.max(cur_max_x);
        let mut pack_max_y = max_y.max(cur_max_y);

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

        // CORNER AVOIDANCE: Strong penalty for creating/extending corners
        let corner_penalty = self.calculate_corner_penalty(
            min_x, min_y, max_x, max_y,
            pack_min_x, pack_min_y, pack_max_x, pack_max_y,
        );

        // Penalty for extending the limiting dimension
        let limiting_dim_penalty = if width > height {
            let tree_extends_x = (max_x > cur_max_x - 0.01) || (min_x < cur_min_x + 0.01);
            if tree_extends_x { 0.05 } else { 0.0 }
        } else {
            let tree_extends_y = (max_y > cur_max_y - 0.01) || (min_y < cur_min_y + 0.01);
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

        side_score + balance_penalty + corner_penalty + limiting_dim_penalty + center_penalty + density_bonus
    }

    /// CORNER AVOIDANCE: Calculate penalty for corner placement
    #[inline]
    fn calculate_corner_penalty(
        &self,
        tree_min_x: f64, tree_min_y: f64, tree_max_x: f64, tree_max_y: f64,
        pack_min_x: f64, pack_min_y: f64, pack_max_x: f64, pack_max_y: f64,
    ) -> f64 {
        let eps = 0.02;

        // Check if tree is at each edge
        let at_left = (tree_min_x - pack_min_x).abs() < eps;
        let at_right = (tree_max_x - pack_max_x).abs() < eps;
        let at_bottom = (tree_min_y - pack_min_y).abs() < eps;
        let at_top = (tree_max_y - pack_max_y).abs() < eps;

        // Count how many edges the tree touches
        let edge_count = [at_left, at_right, at_bottom, at_top].iter().filter(|&&x| x).count();

        // CORNER AVOIDANCE: Strong penalty for corner positions (touching 2+ edges)
        if edge_count >= 2 {
            // Corner tree - highest penalty
            self.config.corner_penalty_weight * 1.5
        } else if edge_count == 1 {
            // Edge tree - smaller penalty
            self.config.corner_penalty_weight * 0.3
        } else {
            // Interior - no penalty (ideal)
            0.0
        }
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    /// CORNER AVOIDANCE: Prefer angles that minimize corner footprint
    #[inline]
    fn select_angles(&self, n: usize, existing: &[PlacedTree]) -> Vec<f64> {
        // Base angles with variation based on n
        let mut angles = match n % 4 {
            0 => vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0],
            1 => vec![90.0, 270.0, 0.0, 180.0, 135.0, 315.0, 45.0, 225.0],
            2 => vec![180.0, 0.0, 270.0, 90.0, 225.0, 45.0, 315.0, 135.0],
            _ => vec![270.0, 90.0, 180.0, 0.0, 315.0, 135.0, 225.0, 45.0],
        };

        // CORNER AVOIDANCE: Add intermediate angles for better corner-avoiding rotations
        // These 22.5-degree offsets can sometimes fit better along edges
        if existing.len() >= 10 {
            angles.extend_from_slice(&[22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]);
        }

        angles
    }

    /// EVOLVED FUNCTION: Select direction angle for placement search
    /// CORNER AVOIDANCE: Bias toward edges rather than corners
    #[inline]
    fn select_direction(&self, n: usize, _existing: &[PlacedTree], rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;

        let strategy = rng.gen::<f64>();

        if strategy < 0.45 {
            // Structured: evenly spaced with small jitter
            let base_idx = rng.gen_range(0..num_dirs);
            let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
            base + rng.gen_range(-0.06..0.06)
        } else if strategy < 0.70 {
            // CORNER AVOIDANCE: Favor cardinal directions (edges) over diagonals (corners)
            // Cardinal directions: 0, 90, 180, 270 degrees with jitter
            let cardinals = [0.0, PI / 2.0, PI, 3.0 * PI / 2.0];
            let base = cardinals[rng.gen_range(0..4)];
            base + rng.gen_range(-0.25..0.25) // Allow some spread but biased to edges
        } else if strategy < 0.85 {
            // Rejection sampling: avoid corner directions
            loop {
                let angle = rng.gen_range(0.0..2.0 * PI);
                // Corners are at 45, 135, 225, 315 degrees (PI/4, 3PI/4, etc)
                // Penalize angles near corners
                let corner_proximity = ((4.0 * angle + PI).sin()).abs(); // High near corners
                let accept_prob = 0.4 + 0.6 * (1.0 - corner_proximity); // Lower near corners
                if rng.gen::<f64>() < accept_prob {
                    return angle;
                }
            }
        } else {
            // Golden angle spiral for coverage
            let golden_angle = PI * (3.0 - (5.0_f64).sqrt());
            let base = (n as f64 * golden_angle) % (2.0 * PI);
            let offset = rng.gen_range(0..8) as f64 * PI / 4.0;
            (base + offset + rng.gen_range(-0.1..0.1)) % (2.0 * PI)
        }
    }

    /// EVOLVED FUNCTION: Local search with simulated annealing
    /// CORNER AVOIDANCE: Prioritize moving corner trees over edge trees
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
            _ => 0.35,
        };
        let mut temp = self.config.sa_initial_temp * temp_multiplier;

        let base_iterations = match pass {
            0 => self.config.sa_iterations + n * 100,
            _ => self.config.sa_iterations / 2 + n * 50,
        };

        let mut iterations_without_improvement = 0;

        // CORNER AVOIDANCE: Cache boundary info with corner tracking
        let mut boundary_cache_iter = 0;
        let mut boundary_info: Vec<(usize, BoundaryEdge, CornerType)> = Vec::new();
        let mut corner_trees: Vec<usize> = Vec::new();

        for iter in 0..base_iterations {
            if iterations_without_improvement >= self.config.early_exit_threshold {
                break;
            }

            // Update boundary cache every 350 iterations
            if iter == 0 || iter - boundary_cache_iter >= 350 {
                boundary_info = self.find_boundary_trees_with_corners(trees);
                corner_trees = boundary_info
                    .iter()
                    .filter(|(_, edge, _)| *edge == BoundaryEdge::Corner)
                    .map(|(idx, _, _)| *idx)
                    .collect();
                boundary_cache_iter = iter;
            }

            // CORNER AVOIDANCE: Prioritize corner trees
            let (idx, edge, corner_type) = if !corner_trees.is_empty()
                && rng.gen::<f64>() < self.config.corner_tree_priority
            {
                // Pick a corner tree with high probability
                let corner_idx = corner_trees[rng.gen_range(0..corner_trees.len())];
                let info = boundary_info.iter().find(|(i, _, _)| *i == corner_idx).unwrap();
                *info
            } else if !boundary_info.is_empty() && rng.gen::<f64>() < self.config.boundary_focus_prob {
                // Pick any boundary tree
                boundary_info[rng.gen_range(0..boundary_info.len())]
            } else {
                // Random tree
                (rng.gen_range(0..trees.len()), BoundaryEdge::None, CornerType::None)
            };

            let old_tree = trees[idx].clone();

            // CORNER AVOIDANCE: Use corner-aware move
            let success = self.corner_aware_move(trees, idx, temp, edge, corner_type, rng);

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

    /// CORNER AVOIDANCE: Find trees on boundary with corner classification
    #[inline]
    fn find_boundary_trees_with_corners(&self, trees: &[PlacedTree]) -> Vec<(usize, BoundaryEdge, CornerType)> {
        if trees.is_empty() {
            return Vec::new();
        }

        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
        let eps = 0.015;

        let mut boundary_info: Vec<(usize, BoundaryEdge, CornerType)> = Vec::new();

        for (i, tree) in trees.iter().enumerate() {
            let (bx1, by1, bx2, by2) = tree.bounds();

            let on_left = (bx1 - min_x).abs() < eps;
            let on_right = (bx2 - max_x).abs() < eps;
            let on_bottom = (by1 - min_y).abs() < eps;
            let on_top = (by2 - max_y).abs() < eps;

            let (edge, corner_type) = match (on_left, on_right, on_top, on_bottom) {
                (true, _, true, _) => (BoundaryEdge::Corner, CornerType::TopLeft),
                (_, true, true, _) => (BoundaryEdge::Corner, CornerType::TopRight),
                (true, _, _, true) => (BoundaryEdge::Corner, CornerType::BottomLeft),
                (_, true, _, true) => (BoundaryEdge::Corner, CornerType::BottomRight),
                (true, true, _, _) | (_, _, true, true) => (BoundaryEdge::Corner, CornerType::None),
                (true, false, false, false) => (BoundaryEdge::Left, CornerType::None),
                (false, true, false, false) => (BoundaryEdge::Right, CornerType::None),
                (false, false, true, false) => (BoundaryEdge::Top, CornerType::None),
                (false, false, false, true) => (BoundaryEdge::Bottom, CornerType::None),
                _ => continue, // Not on boundary
            };

            boundary_info.push((i, edge, corner_type));
        }

        boundary_info
    }

    /// CORNER AVOIDANCE: Move operator that specifically tries to reduce corner impact
    #[inline]
    fn corner_aware_move(
        &self,
        trees: &mut [PlacedTree],
        idx: usize,
        temp: f64,
        edge: BoundaryEdge,
        corner_type: CornerType,
        rng: &mut impl Rng,
    ) -> bool {
        let old = &trees[idx];
        let old_x = old.x;
        let old_y = old.y;
        let old_angle = old.angle_deg;

        let scale = self.config.translation_scale * (0.3 + temp * 1.5);

        // CORNER AVOIDANCE: Special moves for corner trees
        if edge == BoundaryEdge::Corner {
            // Move type specifically for corner trees
            let move_type = rng.gen_range(0..10);
            match move_type {
                0..=3 => {
                    // Diagonal move toward center (away from corner)
                    let (dx_sign, dy_sign) = match corner_type {
                        CornerType::TopLeft => (1.0, -1.0),     // Move right and down
                        CornerType::TopRight => (-1.0, -1.0),   // Move left and down
                        CornerType::BottomLeft => (1.0, 1.0),   // Move right and up
                        CornerType::BottomRight => (-1.0, 1.0), // Move left and up
                        CornerType::None => (0.0, 0.0),
                    };
                    let dx = dx_sign * rng.gen_range(scale * 0.3..scale);
                    let dy = dy_sign * rng.gen_range(scale * 0.3..scale);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
                4..=5 => {
                    // Rotate to potentially reduce corner footprint
                    let angles = [45.0, 90.0, -45.0, -90.0, 22.5, -22.5, 67.5, -67.5];
                    let delta = angles[rng.gen_range(0..angles.len())];
                    let new_angle = (old_angle + delta).rem_euclid(360.0);
                    trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
                }
                6..=7 => {
                    // Gradient descent toward bbox center
                    let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                    let bbox_cx = (min_x + max_x) / 2.0;
                    let bbox_cy = (min_y + max_y) / 2.0;

                    let dx = (bbox_cx - old_x) * self.config.center_pull_strength * (0.6 + temp);
                    let dy = (bbox_cy - old_y) * self.config.center_pull_strength * (0.6 + temp);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
                _ => {
                    // Slide along one edge to reduce corner impact
                    let (dx, dy) = match corner_type {
                        CornerType::TopLeft | CornerType::TopRight => {
                            (rng.gen_range(-scale..scale), -rng.gen_range(0.0..scale * 0.5))
                        }
                        CornerType::BottomLeft | CornerType::BottomRight => {
                            (rng.gen_range(-scale..scale), rng.gen_range(0.0..scale * 0.5))
                        }
                        CornerType::None => (rng.gen_range(-scale..scale), rng.gen_range(-scale..scale)),
                    };
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
            }
        } else {
            // Standard moves for edge/interior trees
            let move_type = match edge {
                BoundaryEdge::Left => {
                    match rng.gen_range(0..10) {
                        0..=4 => 0, // Move right
                        5..=6 => 1, // Slide vertically
                        7..=8 => 2, // Rotate
                        _ => 3,     // General
                    }
                }
                BoundaryEdge::Right => {
                    match rng.gen_range(0..10) {
                        0..=4 => 4, // Move left
                        5..=6 => 1, // Slide vertically
                        7..=8 => 2, // Rotate
                        _ => 3,     // General
                    }
                }
                BoundaryEdge::Top => {
                    match rng.gen_range(0..10) {
                        0..=4 => 5, // Move down
                        5..=6 => 6, // Slide horizontally
                        7..=8 => 2, // Rotate
                        _ => 3,     // General
                    }
                }
                BoundaryEdge::Bottom => {
                    match rng.gen_range(0..10) {
                        0..=4 => 7, // Move up
                        5..=6 => 6, // Slide horizontally
                        7..=8 => 2, // Rotate
                        _ => 3,     // General
                    }
                }
                _ => rng.gen_range(0..10),
            };

            match move_type {
                0 => {
                    // Move right
                    let dx = rng.gen_range(scale * 0.3..scale);
                    let dy = rng.gen_range(-scale * 0.2..scale * 0.2);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
                1 => {
                    // Slide vertically
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
                    // General small move
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
                    // Slide horizontally
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
                    // Gradient descent toward center
                    let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                    let bbox_cx = (min_x + max_x) / 2.0;
                    let bbox_cy = (min_y + max_y) / 2.0;

                    let dx = (bbox_cx - old_x) * self.config.center_pull_strength * (0.5 + temp);
                    let dy = (bbox_cy - old_y) * self.config.center_pull_strength * (0.5 + temp);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
                _ => {
                    // Polar/orbit move
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
}
