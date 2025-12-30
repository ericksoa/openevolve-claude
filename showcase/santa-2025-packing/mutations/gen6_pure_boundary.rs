//! Evolved Packing Algorithm - Generation 6 PURE BOUNDARY
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
//! MUTATION STRATEGY: PURE BOUNDARY FOCUS (Gen6)
//! 100% boundary focus - never move interior trees during SA.
//!
//! Key improvements from Gen5:
//! - 100% boundary trees only (up from 90%)
//! - ONLY move boundary trees, never waste time on interior trees
//! - For each boundary edge, identify the specific blocking tree(s)
//! - Move operators: inward slide only, rotate to reduce footprint
//! - Increased SA iterations to compensate for smaller search space
//! - Multi-angle rotations (12 angles) to find smallest footprint orientation
//! - More aggressive center pull for boundary trees
//! - Dedicated footprint-minimizing rotation search
//!
//! Target: Beat Gen5's 95.69 at n=200 with pure boundary focus

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

    // PURE BOUNDARY: Always 100% boundary focus
    pub boundary_focus_prob: f64,
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen6 PURE BOUNDARY: 100% boundary focus configuration
        Self {
            search_attempts: 280,            // More attempts for better initial placement
            direction_samples: 72,           // More directions (72 = 5 degree steps)
            sa_iterations: 35000,            // Increased iterations for smaller search space
            sa_initial_temp: 0.45,           // Slightly lower for exploitation
            sa_cooling_rate: 0.99997,        // Slower cooling for deeper search
            sa_min_temp: 0.000005,           // Lower minimum for fine-tuning
            translation_scale: 0.055,        // Smaller moves for precision
            rotation_granularity: 30.0,      // 12 angles (30 degree steps)
            center_pull_strength: 0.12,      // Stronger pull toward center
            sa_passes: 3,                    // More passes with smaller search space
            early_exit_threshold: 2000,      // More patience with focused search
            boundary_focus_prob: 1.0,        // 100% boundary trees only
        }
    }
}

/// Track which boundary a tree is blocking and how much
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BoundaryEdge {
    Left,
    Right,
    Top,
    Bottom,
    CornerLeftTop,
    CornerRightTop,
    CornerLeftBottom,
    CornerRightBottom,
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

            // Run SA passes with pure boundary focus
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
    /// PURE BOUNDARY: Focus on not extending bounding box
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

        // PURE BOUNDARY: Heavy penalty for extending the limiting dimension
        let limiting_dim_penalty = if width > height {
            // Width is limiting, strongly penalize x extension
            let extends_max_x = max_x > pack_max_x - 0.005;
            let extends_min_x = min_x < pack_min_x + 0.005;
            if extends_max_x || extends_min_x { 0.08 } else { 0.0 }
        } else {
            // Height is limiting, strongly penalize y extension
            let extends_max_y = max_y > pack_max_y - 0.005;
            let extends_min_y = min_y < pack_min_y + 0.005;
            if extends_max_y || extends_min_y { 0.08 } else { 0.0 }
        };

        // Tertiary: preference for compact center
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.008 / (n as f64).sqrt();

        // Density bonus - prefer tighter packing
        let area = width * height;
        let density_bonus = if area > 0.0 {
            -0.015 * (n as f64 / area).min(2.5)
        } else {
            0.0
        };

        // PURE BOUNDARY: Bonus for placing tree that doesn't become boundary
        let becomes_boundary_penalty = {
            let on_boundary = (max_x - pack_max_x).abs() < 0.01
                || (min_x - pack_min_x).abs() < 0.01
                || (max_y - pack_max_y).abs() < 0.01
                || (min_y - pack_min_y).abs() < 0.01;
            if on_boundary { 0.02 } else { 0.0 }
        };

        side_score + balance_penalty + limiting_dim_penalty + center_penalty + density_bonus + becomes_boundary_penalty
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    /// PURE BOUNDARY: 12 angles for finer rotation search
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
        // 12 angles at 30 degree intervals
        let base: Vec<f64> = (0..12).map(|i| (i as f64) * 30.0).collect();

        // Rotate starting point based on n for diversity
        let offset = (n % 12) as f64 * 30.0;
        base.iter().map(|&a| (a + offset).rem_euclid(360.0)).collect()
    }

    /// EVOLVED FUNCTION: Select direction angle for placement search
    #[inline]
    fn select_direction(&self, n: usize, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;

        let strategy = rng.gen::<f64>();

        if strategy < 0.55 {
            // Structured: evenly spaced with small jitter
            let base_idx = rng.gen_range(0..num_dirs);
            let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
            base + rng.gen_range(-0.04..0.04)
        } else if strategy < 0.80 {
            // Weighted random: favor corners and edges
            loop {
                let angle = rng.gen_range(0.0..2.0 * PI);
                let corner_weight = ((4.0 * angle).sin().abs() + (4.0 * angle).cos().abs()) / 2.0;
                let threshold = 0.15;
                if rng.gen::<f64>() < corner_weight.max(threshold) {
                    return angle;
                }
            }
        } else {
            // Golden angle spiral for good coverage
            let golden_angle = PI * (3.0 - (5.0_f64).sqrt());
            let base = (n as f64 * golden_angle) % (2.0 * PI);
            let offset = rng.gen_range(0..12) as f64 * PI / 6.0;
            (base + offset + rng.gen_range(-0.08..0.08)) % (2.0 * PI)
        }
    }

    /// EVOLVED FUNCTION: Local search with simulated annealing
    /// PURE BOUNDARY: Only move boundary trees, never interior trees
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
            1 => 0.4,
            _ => 0.15, // Lower temp for third pass - fine tuning
        };
        let mut temp = self.config.sa_initial_temp * temp_multiplier;

        let base_iterations = match pass {
            0 => self.config.sa_iterations + n * 120,
            1 => self.config.sa_iterations / 2 + n * 60,
            _ => self.config.sa_iterations / 4 + n * 30,
        };

        let mut iterations_without_improvement = 0;

        // PURE BOUNDARY: Cache boundary info with detailed edge tracking
        let mut boundary_cache_iter = 0;
        let mut boundary_info: Vec<(usize, BoundaryEdge)> = Vec::new();

        for iter in 0..base_iterations {
            // Early exit when no improvement for threshold iterations
            if iterations_without_improvement >= self.config.early_exit_threshold {
                break;
            }

            // Update boundary cache every 300 iterations (more frequent updates)
            if iter == 0 || iter - boundary_cache_iter >= 300 {
                boundary_info = self.find_boundary_trees_with_edges(trees);
                boundary_cache_iter = iter;
            }

            // PURE BOUNDARY: Only pick boundary trees - skip if none found
            if boundary_info.is_empty() {
                // No boundary trees identified, skip this iteration
                iterations_without_improvement += 1;
                temp = (temp * self.config.sa_cooling_rate).max(self.config.sa_min_temp);
                continue;
            }

            let (idx, edge) = {
                let bi = &boundary_info[rng.gen_range(0..boundary_info.len())];
                (bi.0, bi.1)
            };

            let old_tree = trees[idx].clone();

            // PURE BOUNDARY: Use targeted inward moves based on which boundary the tree is blocking
            let success = self.pure_boundary_move(trees, idx, temp, edge, rng);

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

    /// PURE BOUNDARY: Find trees on the bounding box boundary with detailed edge classification
    #[inline]
    fn find_boundary_trees_with_edges(&self, trees: &[PlacedTree]) -> Vec<(usize, BoundaryEdge)> {
        if trees.is_empty() {
            return Vec::new();
        }

        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
        let eps = 0.02; // Epsilon for edge detection

        let mut boundary_info: Vec<(usize, BoundaryEdge)> = Vec::new();

        for (i, tree) in trees.iter().enumerate() {
            let (bx1, by1, bx2, by2) = tree.bounds();

            let on_left = (bx1 - min_x).abs() < eps;
            let on_right = (bx2 - max_x).abs() < eps;
            let on_bottom = (by1 - min_y).abs() < eps;
            let on_top = (by2 - max_y).abs() < eps;

            let edge = match (on_left, on_right, on_top, on_bottom) {
                (true, _, true, _) => BoundaryEdge::CornerLeftTop,
                (true, _, _, true) => BoundaryEdge::CornerLeftBottom,
                (_, true, true, _) => BoundaryEdge::CornerRightTop,
                (_, true, _, true) => BoundaryEdge::CornerRightBottom,
                (true, false, false, false) => BoundaryEdge::Left,
                (false, true, false, false) => BoundaryEdge::Right,
                (false, false, true, false) => BoundaryEdge::Top,
                (false, false, false, true) => BoundaryEdge::Bottom,
                _ => continue, // Not on boundary - skip entirely
            };

            boundary_info.push((i, edge));
        }

        boundary_info
    }

    /// PURE BOUNDARY: Move operators that only move boundary trees inward
    /// Returns true if move is valid (no overlap)
    #[inline]
    fn pure_boundary_move(
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

        let scale = self.config.translation_scale * (0.25 + temp * 1.2);

        // PURE BOUNDARY: Choose move type based on which boundary we're blocking
        // All moves are designed to reduce footprint or move inward
        let move_type = match edge {
            BoundaryEdge::Left => {
                // Move right (inward)
                match rng.gen_range(0..10) {
                    0..=5 => 0, // Move right (inward)
                    6..=7 => 1, // Rotate to reduce x footprint
                    _ => 2,     // Slide along boundary
                }
            }
            BoundaryEdge::Right => {
                // Move left (inward)
                match rng.gen_range(0..10) {
                    0..=5 => 3, // Move left (inward)
                    6..=7 => 1, // Rotate to reduce x footprint
                    _ => 2,     // Slide along boundary
                }
            }
            BoundaryEdge::Top => {
                // Move down (inward)
                match rng.gen_range(0..10) {
                    0..=5 => 4, // Move down (inward)
                    6..=7 => 5, // Rotate to reduce y footprint
                    _ => 6,     // Slide along boundary
                }
            }
            BoundaryEdge::Bottom => {
                // Move up (inward)
                match rng.gen_range(0..10) {
                    0..=5 => 7, // Move up (inward)
                    6..=7 => 5, // Rotate to reduce y footprint
                    _ => 6,     // Slide along boundary
                }
            }
            BoundaryEdge::CornerLeftTop => {
                // Corner: move diagonally inward (right-down)
                match rng.gen_range(0..10) {
                    0..=4 => 8,  // Diagonal inward
                    5..=7 => 9,  // Best footprint rotation
                    _ => 10,     // Center pull
                }
            }
            BoundaryEdge::CornerRightTop => {
                // Corner: move diagonally inward (left-down)
                match rng.gen_range(0..10) {
                    0..=4 => 11, // Diagonal inward
                    5..=7 => 9,  // Best footprint rotation
                    _ => 10,     // Center pull
                }
            }
            BoundaryEdge::CornerLeftBottom => {
                // Corner: move diagonally inward (right-up)
                match rng.gen_range(0..10) {
                    0..=4 => 12, // Diagonal inward
                    5..=7 => 9,  // Best footprint rotation
                    _ => 10,     // Center pull
                }
            }
            BoundaryEdge::CornerRightBottom => {
                // Corner: move diagonally inward (left-up)
                match rng.gen_range(0..10) {
                    0..=4 => 13, // Diagonal inward
                    5..=7 => 9,  // Best footprint rotation
                    _ => 10,     // Center pull
                }
            }
        };

        match move_type {
            0 => {
                // Move right (inward from left boundary)
                let dx = rng.gen_range(scale * 0.3..scale);
                let dy = rng.gen_range(-scale * 0.15..scale * 0.15);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            1 => {
                // Rotate to reduce x footprint (for left/right boundary trees)
                // Try multiple angles and pick best x footprint
                let best_angle = self.find_best_rotation_for_x(old_x, old_y, old_angle, trees, idx);
                trees[idx] = PlacedTree::new(old_x, old_y, best_angle);
            }
            2 => {
                // Slide vertically along boundary
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x, old_y + dy, old_angle);
            }
            3 => {
                // Move left (inward from right boundary)
                let dx = rng.gen_range(-scale..-scale * 0.3);
                let dy = rng.gen_range(-scale * 0.15..scale * 0.15);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            4 => {
                // Move down (inward from top boundary)
                let dx = rng.gen_range(-scale * 0.15..scale * 0.15);
                let dy = rng.gen_range(-scale..-scale * 0.3);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            5 => {
                // Rotate to reduce y footprint (for top/bottom boundary trees)
                let best_angle = self.find_best_rotation_for_y(old_x, old_y, old_angle, trees, idx);
                trees[idx] = PlacedTree::new(old_x, old_y, best_angle);
            }
            6 => {
                // Slide horizontally along boundary
                let dx = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y, old_angle);
            }
            7 => {
                // Move up (inward from bottom boundary)
                let dx = rng.gen_range(-scale * 0.15..scale * 0.15);
                let dy = rng.gen_range(scale * 0.3..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            8 => {
                // Diagonal inward from left-top corner (move right-down)
                let d = rng.gen_range(scale * 0.3..scale);
                trees[idx] = PlacedTree::new(old_x + d, old_y - d, old_angle);
            }
            9 => {
                // Best footprint rotation - try all 12 angles
                let best_angle = self.find_smallest_footprint_rotation(old_x, old_y, trees, idx);
                trees[idx] = PlacedTree::new(old_x, old_y, best_angle);
            }
            10 => {
                // Strong center pull for corner trees
                let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                let bbox_cx = (min_x + max_x) / 2.0;
                let bbox_cy = (min_y + max_y) / 2.0;

                let dx = (bbox_cx - old_x) * self.config.center_pull_strength * (0.6 + temp);
                let dy = (bbox_cy - old_y) * self.config.center_pull_strength * (0.6 + temp);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            11 => {
                // Diagonal inward from right-top corner (move left-down)
                let d = rng.gen_range(scale * 0.3..scale);
                trees[idx] = PlacedTree::new(old_x - d, old_y - d, old_angle);
            }
            12 => {
                // Diagonal inward from left-bottom corner (move right-up)
                let d = rng.gen_range(scale * 0.3..scale);
                trees[idx] = PlacedTree::new(old_x + d, old_y + d, old_angle);
            }
            13 => {
                // Diagonal inward from right-bottom corner (move left-up)
                let d = rng.gen_range(scale * 0.3..scale);
                trees[idx] = PlacedTree::new(old_x - d, old_y + d, old_angle);
            }
            _ => {
                return false;
            }
        }

        !has_overlap(trees, idx)
    }

    /// Find the rotation angle that minimizes x footprint (for left/right boundary trees)
    #[inline]
    fn find_best_rotation_for_x(
        &self,
        x: f64,
        y: f64,
        current_angle: f64,
        trees: &[PlacedTree],
        idx: usize,
    ) -> f64 {
        let mut best_angle = current_angle;
        let mut best_width = f64::INFINITY;

        // Try 12 angles
        for i in 0..12 {
            let angle = (i as f64) * 30.0;
            let test_tree = PlacedTree::new(x, y, angle);

            // Check if valid first
            let mut valid = true;
            for (j, other) in trees.iter().enumerate() {
                if j != idx && test_tree.overlaps(other) {
                    valid = false;
                    break;
                }
            }

            if valid {
                let (min_x, _, max_x, _) = test_tree.bounds();
                let width = max_x - min_x;
                if width < best_width {
                    best_width = width;
                    best_angle = angle;
                }
            }
        }

        best_angle
    }

    /// Find the rotation angle that minimizes y footprint (for top/bottom boundary trees)
    #[inline]
    fn find_best_rotation_for_y(
        &self,
        x: f64,
        y: f64,
        current_angle: f64,
        trees: &[PlacedTree],
        idx: usize,
    ) -> f64 {
        let mut best_angle = current_angle;
        let mut best_height = f64::INFINITY;

        // Try 12 angles
        for i in 0..12 {
            let angle = (i as f64) * 30.0;
            let test_tree = PlacedTree::new(x, y, angle);

            // Check if valid first
            let mut valid = true;
            for (j, other) in trees.iter().enumerate() {
                if j != idx && test_tree.overlaps(other) {
                    valid = false;
                    break;
                }
            }

            if valid {
                let (_, min_y, _, max_y) = test_tree.bounds();
                let height = max_y - min_y;
                if height < best_height {
                    best_height = height;
                    best_angle = angle;
                }
            }
        }

        best_angle
    }

    /// Find the rotation angle that minimizes overall footprint (for corner trees)
    #[inline]
    fn find_smallest_footprint_rotation(
        &self,
        x: f64,
        y: f64,
        trees: &[PlacedTree],
        idx: usize,
    ) -> f64 {
        let mut best_angle = 0.0;
        let mut best_footprint = f64::INFINITY;

        // Try 12 angles
        for i in 0..12 {
            let angle = (i as f64) * 30.0;
            let test_tree = PlacedTree::new(x, y, angle);

            // Check if valid first
            let mut valid = true;
            for (j, other) in trees.iter().enumerate() {
                if j != idx && test_tree.overlaps(other) {
                    valid = false;
                    break;
                }
            }

            if valid {
                let (min_x, min_y, max_x, max_y) = test_tree.bounds();
                // Use max dimension as footprint (since we're minimizing square side)
                let footprint = (max_x - min_x).max(max_y - min_y);
                if footprint < best_footprint {
                    best_footprint = footprint;
                    best_angle = angle;
                }
            }
        }

        best_angle
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
