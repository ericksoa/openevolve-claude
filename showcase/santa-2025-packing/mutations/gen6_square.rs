//! Evolved Packing Algorithm - Generation 6 ASPECT RATIO PRIORITY
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
//! MUTATION STRATEGY: ASPECT RATIO PRIORITY (Gen6)
//! A perfect square minimizes the scoring metric side^2. Focus on balanced packing:
//!
//! Key improvements from Gen5:
//! - During placement: strong preference for width ≈ height
//! - Track "long dimension" at each step
//! - Bias new tree placement toward the shorter dimension
//! - SA moves: prioritize moves that balance the aspect ratio
//! - Early in packing: establish square shape
//! - Aspect ratio imbalance penalty increases with deviation
//! - Direction selection biased toward shorter dimension
//! - Move operators include "balance" moves that target aspect ratio
//!
//! Target: Beat Gen5's 95.69 at n=200 with square-focused packing

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

    // ASPECT RATIO: New parameters for square packing
    pub aspect_ratio_weight: f64,      // Weight for aspect ratio penalty
    pub short_dim_bias: f64,           // Bias toward shorter dimension in placement
    pub balance_move_prob: f64,        // Probability of balance-focused moves
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen6 ASPECT RATIO: Square-focused configuration
        Self {
            search_attempts: 280,            // Slightly more attempts for better placement
            direction_samples: 72,           // More directions for finer placement
            sa_iterations: 28000,            // More iterations for aspect balancing
            sa_initial_temp: 0.45,           // Similar to Gen5
            sa_cooling_rate: 0.99996,        // Slower cooling for better balance convergence
            sa_min_temp: 0.00001,            // Keep from Gen5
            translation_scale: 0.055,        // Slightly smaller for precision
            rotation_granularity: 45.0,      // Keep 8 angles
            center_pull_strength: 0.09,      // Stronger center pull for balance
            sa_passes: 2,                    // Keep from Gen5
            early_exit_threshold: 1800,      // More patience for balance moves
            boundary_focus_prob: 0.85,       // 85% boundary trees
            aspect_ratio_weight: 0.25,       // Strong weight for aspect ratio
            short_dim_bias: 0.70,            // 70% chance to bias toward short dimension
            balance_move_prob: 0.30,         // 30% chance of balance-focused moves
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

/// Track current aspect ratio state
#[derive(Clone, Copy, Debug)]
struct AspectState {
    width: f64,
    height: f64,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
}

impl AspectState {
    fn from_trees(trees: &[PlacedTree]) -> Self {
        if trees.is_empty() {
            return Self {
                width: 0.0,
                height: 0.0,
                min_x: 0.0,
                max_x: 0.0,
                min_y: 0.0,
                max_y: 0.0,
            };
        }

        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
        Self {
            width: max_x - min_x,
            height: max_y - min_y,
            min_x,
            max_x,
            min_y,
            max_y,
        }
    }

    /// Returns the ratio of the longer dimension to the shorter
    fn aspect_ratio(&self) -> f64 {
        if self.width <= 0.0 || self.height <= 0.0 {
            return 1.0;
        }
        self.width.max(self.height) / self.width.min(self.height)
    }

    /// Returns true if width > height (wider than tall)
    fn is_wide(&self) -> bool {
        self.width > self.height
    }

    /// Returns the side length (max of width/height)
    fn side(&self) -> f64 {
        self.width.max(self.height)
    }

    /// Returns how much the aspect ratio deviates from 1.0 (perfect square)
    fn imbalance(&self) -> f64 {
        (self.width - self.height).abs()
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

            // Place new tree using evolved heuristics with aspect awareness
            let new_tree = self.find_placement(&trees, n, max_n, &mut rng);
            trees.push(new_tree);

            // Run SA passes with aspect-aware moves
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
    /// ASPECT RATIO: Bias toward shorter dimension
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

        // Get current aspect state for direction biasing
        let aspect = AspectState::from_trees(existing);

        for _ in 0..self.config.search_attempts {
            let dir = self.select_direction_aspect(n, &aspect, rng);
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
                    let score = self.placement_score_aspect(&candidate, existing, n, &aspect);
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
    /// ASPECT RATIO: Heavy penalty for aspect ratio imbalance
    #[inline]
    fn placement_score_aspect(
        &self,
        tree: &PlacedTree,
        existing: &[PlacedTree],
        n: usize,
        current_aspect: &AspectState,
    ) -> f64 {
        let (min_x, min_y, max_x, max_y) = tree.bounds();

        // Compute combined bounds
        let new_min_x = current_aspect.min_x.min(min_x);
        let new_min_y = current_aspect.min_y.min(min_y);
        let new_max_x = current_aspect.max_x.max(max_x);
        let new_max_y = current_aspect.max_y.max(max_y);

        let new_width = new_max_x - new_min_x;
        let new_height = new_max_y - new_min_y;
        let new_side = new_width.max(new_height);

        // Primary: minimize side length (most important)
        let side_score = new_side;

        // ASPECT RATIO: Strong penalty for imbalance
        let imbalance = (new_width - new_height).abs();
        let aspect_penalty = imbalance * self.config.aspect_ratio_weight;

        // ASPECT RATIO: Progressive penalty - worse imbalance costs more
        let progressive_penalty = if imbalance > 0.1 {
            (imbalance - 0.1).powi(2) * 0.5
        } else {
            0.0
        };

        // ASPECT RATIO: Penalty for extending the already-long dimension
        let extends_long_dim = if current_aspect.is_wide() {
            // Width is already long, penalize x extension
            max_x > current_aspect.max_x || min_x < current_aspect.min_x
        } else {
            // Height is already long (or equal), penalize y extension
            max_y > current_aspect.max_y || min_y < current_aspect.min_y
        };
        let long_dim_penalty = if extends_long_dim { 0.08 } else { 0.0 };

        // ASPECT RATIO: Bonus for extending the SHORT dimension (helps balance)
        let extends_short_dim = if current_aspect.is_wide() {
            max_y > current_aspect.max_y || min_y < current_aspect.min_y
        } else {
            max_x > current_aspect.max_x || min_x < current_aspect.min_x
        };
        let balance_bonus = if extends_short_dim && !extends_long_dim { -0.03 } else { 0.0 };

        // Tertiary: preference for compact center
        let center_x = (new_min_x + new_max_x) / 2.0;
        let center_y = (new_min_y + new_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.005 / (n as f64).sqrt();

        // Density bonus
        let area = new_width * new_height;
        let density_bonus = if area > 0.0 {
            -0.01 * (n as f64 / area).min(2.0)
        } else {
            0.0
        };

        side_score + aspect_penalty + progressive_penalty + long_dim_penalty + balance_bonus + center_penalty + density_bonus
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
    /// ASPECT RATIO: Bias toward the shorter dimension
    #[inline]
    fn select_direction_aspect(&self, n: usize, aspect: &AspectState, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;

        // ASPECT RATIO: Bias toward shorter dimension
        if rng.gen::<f64>() < self.config.short_dim_bias && aspect.imbalance() > 0.05 {
            // Bias placement toward the shorter dimension
            if aspect.is_wide() {
                // Width > height: prefer vertical directions (up/down)
                let base_angle = if rng.gen() { PI / 2.0 } else { 3.0 * PI / 2.0 };
                return base_angle + rng.gen_range(-PI / 6.0..PI / 6.0);
            } else {
                // Height > width: prefer horizontal directions (left/right)
                let base_angle = if rng.gen() { 0.0 } else { PI };
                return base_angle + rng.gen_range(-PI / 6.0..PI / 6.0);
            }
        }

        let strategy = rng.gen::<f64>();

        if strategy < 0.45 {
            // Structured: evenly spaced with small jitter
            let base_idx = rng.gen_range(0..num_dirs);
            let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
            base + rng.gen_range(-0.06..0.06)
        } else if strategy < 0.70 {
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
    /// ASPECT RATIO: Include balance-focused moves
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

        // Cache boundary info with edge tracking
        let mut boundary_cache_iter = 0;
        let mut boundary_info: Vec<(usize, BoundaryEdge)> = Vec::new();
        let mut aspect_state = AspectState::from_trees(trees);

        for iter in 0..base_iterations {
            // Early exit when no improvement for threshold iterations
            if iterations_without_improvement >= self.config.early_exit_threshold {
                break;
            }

            // Update boundary cache and aspect state every 400 iterations
            if iter == 0 || iter - boundary_cache_iter >= 400 {
                boundary_info = self.find_boundary_trees_with_edges(trees);
                aspect_state = AspectState::from_trees(trees);
                boundary_cache_iter = iter;
            }

            // Choose between balance move and regular move
            let do_balance_move = rng.gen::<f64>() < self.config.balance_move_prob
                && aspect_state.imbalance() > 0.05;

            let (idx, edge) = if do_balance_move {
                // ASPECT RATIO: Select tree blocking the long dimension
                self.select_tree_for_balance(&boundary_info, &aspect_state, rng)
            } else if !boundary_info.is_empty() && rng.gen::<f64>() < self.config.boundary_focus_prob {
                let bi = &boundary_info[rng.gen_range(0..boundary_info.len())];
                (bi.0, bi.1)
            } else {
                (rng.gen_range(0..trees.len()), BoundaryEdge::None)
            };

            let old_tree = trees[idx].clone();

            // Use targeted move based on boundary and aspect ratio
            let success = if do_balance_move {
                self.balance_move(trees, idx, temp, &aspect_state, rng)
            } else {
                self.smart_move(trees, idx, temp, edge, rng)
            };

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

    /// ASPECT RATIO: Select a tree that is blocking the long dimension
    #[inline]
    fn select_tree_for_balance(
        &self,
        boundary_info: &[(usize, BoundaryEdge)],
        aspect: &AspectState,
        rng: &mut impl Rng,
    ) -> (usize, BoundaryEdge) {
        if boundary_info.is_empty() {
            return (0, BoundaryEdge::None);
        }

        // Find trees blocking the long dimension
        let long_dim_trees: Vec<_> = boundary_info
            .iter()
            .filter(|(_, edge)| {
                if aspect.is_wide() {
                    // Width is long: look for Left/Right boundary trees
                    matches!(edge, BoundaryEdge::Left | BoundaryEdge::Right | BoundaryEdge::Corner)
                } else {
                    // Height is long: look for Top/Bottom boundary trees
                    matches!(edge, BoundaryEdge::Top | BoundaryEdge::Bottom | BoundaryEdge::Corner)
                }
            })
            .collect();

        if long_dim_trees.is_empty() {
            // Fall back to any boundary tree
            let bi = &boundary_info[rng.gen_range(0..boundary_info.len())];
            return (bi.0, bi.1);
        }

        let selected = long_dim_trees[rng.gen_range(0..long_dim_trees.len())];
        (selected.0, selected.1)
    }

    /// ASPECT RATIO: Move designed to reduce aspect ratio imbalance
    #[inline]
    fn balance_move(
        &self,
        trees: &mut [PlacedTree],
        idx: usize,
        temp: f64,
        aspect: &AspectState,
        rng: &mut impl Rng,
    ) -> bool {
        let old = &trees[idx];
        let old_x = old.x;
        let old_y = old.y;
        let old_angle = old.angle_deg;

        let scale = self.config.translation_scale * (0.3 + temp * 1.5);

        // Move toward reducing the long dimension
        let move_type = rng.gen_range(0..5);

        match move_type {
            0 => {
                // Move toward center of bounding box
                let bbox_cx = (aspect.min_x + aspect.max_x) / 2.0;
                let bbox_cy = (aspect.min_y + aspect.max_y) / 2.0;

                let dx = (bbox_cx - old_x) * self.config.center_pull_strength * (0.5 + temp);
                let dy = (bbox_cy - old_y) * self.config.center_pull_strength * (0.5 + temp);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            1 => {
                // Move inward from the long dimension edges
                if aspect.is_wide() {
                    // Width is long: move horizontally toward center
                    let bbox_cx = (aspect.min_x + aspect.max_x) / 2.0;
                    let dx = (bbox_cx - old_x).signum() * rng.gen_range(scale * 0.3..scale);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y, old_angle);
                } else {
                    // Height is long: move vertically toward center
                    let bbox_cy = (aspect.min_y + aspect.max_y) / 2.0;
                    let dy = (bbox_cy - old_y).signum() * rng.gen_range(scale * 0.3..scale);
                    trees[idx] = PlacedTree::new(old_x, old_y + dy, old_angle);
                }
            }
            2 => {
                // Rotate to potentially reduce footprint in long dimension
                let angles = [45.0, 90.0, -45.0, -90.0];
                let delta = angles[rng.gen_range(0..angles.len())];
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            3 => {
                // Slide along short dimension edge (perpendicular to long)
                if aspect.is_wide() {
                    // Slide vertically
                    let dy = rng.gen_range(-scale..scale);
                    trees[idx] = PlacedTree::new(old_x, old_y + dy, old_angle);
                } else {
                    // Slide horizontally
                    let dx = rng.gen_range(-scale..scale);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y, old_angle);
                }
            }
            _ => {
                // Combined: move toward center and rotate
                let bbox_cx = (aspect.min_x + aspect.max_x) / 2.0;
                let bbox_cy = (aspect.min_y + aspect.max_y) / 2.0;

                let dx = (bbox_cx - old_x) * 0.05 * (0.5 + temp);
                let dy = (bbox_cy - old_y) * 0.05 * (0.5 + temp);
                let new_angle = (old_angle + 45.0).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, new_angle);
            }
        }

        !has_overlap(trees, idx)
    }

    /// Find trees on the bounding box boundary and which edge they block
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

    /// Targeted move operator based on which boundary the tree blocks
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

        let move_type = match edge {
            BoundaryEdge::Left => {
                match rng.gen_range(0..10) {
                    0..=4 => 0, // Move right
                    5..=6 => 1, // Slide along boundary
                    7..=8 => 2, // Rotate
                    _ => 3,     // General move
                }
            }
            BoundaryEdge::Right => {
                match rng.gen_range(0..10) {
                    0..=4 => 4, // Move left
                    5..=6 => 1, // Slide along boundary
                    7..=8 => 2, // Rotate
                    _ => 3,     // General move
                }
            }
            BoundaryEdge::Top => {
                match rng.gen_range(0..10) {
                    0..=4 => 5, // Move down
                    5..=6 => 6, // Slide along boundary
                    7..=8 => 2, // Rotate
                    _ => 3,     // General move
                }
            }
            BoundaryEdge::Bottom => {
                match rng.gen_range(0..10) {
                    0..=4 => 7, // Move up
                    5..=6 => 6, // Slide along boundary
                    7..=8 => 2, // Rotate
                    _ => 3,     // General move
                }
            }
            BoundaryEdge::Corner => {
                match rng.gen_range(0..10) {
                    0..=4 => 8, // Gradient descent
                    5..=6 => 2, // Rotate
                    7..=8 => 9, // Diagonal slide
                    _ => 3,     // General move
                }
            }
            BoundaryEdge::None => {
                rng.gen_range(0..10)
            }
        };

        let scale = self.config.translation_scale * (0.3 + temp * 1.5);

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

    #[test]
    fn test_aspect_state() {
        // Test aspect state calculation
        let tree1 = PlacedTree::new(0.0, 0.0, 0.0);
        let tree2 = PlacedTree::new(1.0, 0.0, 0.0);
        let trees = vec![tree1, tree2];
        let aspect = AspectState::from_trees(&trees);

        // Width should be larger than height for horizontally placed trees
        assert!(aspect.width > 0.0);
        assert!(aspect.height > 0.0);
    }
}
