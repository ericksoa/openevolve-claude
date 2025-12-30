//! Evolved Packing Algorithm - Generation 6 HYBRID
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
//! MUTATION STRATEGY: HYBRID PARAMETERS (Gen6)
//! Combine the best ideas from all previous generations:
//!
//! From Gen5 (champion, 95.69):
//! - 90% boundary focus for tree selection
//! - Smart edge-aware move operators
//! - BoundaryEdge tracking for targeted moves
//!
//! From Gen4 (98.37):
//! - Early exit threshold (exit when stuck)
//! - Efficient boundary caching
//! - Reasonable base iterations
//!
//! From Gen3 (101.90):
//! - N-adaptive computation (more effort for small n)
//! - Multiple SA passes with temperature adjustment
//! - Greedy compaction phase (selective)
//! - Golden angle direction sampling
//!
//! New in Gen6:
//! - N-adaptive SA passes: 3 passes for small n, 2 for medium, 1 for large
//! - N-adaptive search attempts: more for small n where each matters more
//! - Aspect ratio scoring that adapts to current limiting dimension
//! - Hybrid move selection combining Gen5's smart moves with Gen3's variety
//! - Progressive early exit threshold based on n
//!
//! Target: Beat Gen5's 95.69 at n=200 with hybrid approach

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Evolved packing configuration
/// These parameters are tuned through evolution
pub struct EvolvedConfig {
    // Search parameters
    pub search_attempts_base: usize,
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

    // Multi-pass settings (base, adjusted by n)
    pub sa_passes_small: usize,   // n <= 50
    pub sa_passes_medium: usize,  // 50 < n <= 120
    pub sa_passes_large: usize,   // n > 120

    // Early exit threshold (base, scaled by n)
    pub early_exit_base: usize,

    // HYBRID: Boundary focus probability
    pub boundary_focus_prob: f64,

    // HYBRID: Greedy compaction iterations (for important trees)
    pub compaction_iterations: usize,
    pub compaction_threshold_n: usize,  // Only compact for n <= this
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen6 HYBRID: Best of all generations
        Self {
            search_attempts_base: 200,        // Base attempts, scaled up for small n
            direction_samples: 72,            // Between Gen3's 96 and Gen4's 64
            sa_iterations: 22000,             // Balanced between Gen3 and Gen4
            sa_initial_temp: 0.55,            // Slightly lower for more exploitation
            sa_cooling_rate: 0.99993,         // Between Gen3's slow and Gen4's fast
            sa_min_temp: 0.000005,            // Between Gen3 and Gen4
            translation_scale: 0.07,          // Slightly smaller for precision
            rotation_granularity: 45.0,       // 8 angles like Gen4/5 (efficient)
            center_pull_strength: 0.07,       // Slightly stronger than Gen5
            sa_passes_small: 3,               // 3 passes for n <= 50 (like Gen3)
            sa_passes_medium: 2,              // 2 passes for 50 < n <= 120
            sa_passes_large: 2,               // 2 passes for n > 120
            early_exit_base: 1200,            // Base threshold, scaled by n
            boundary_focus_prob: 0.88,        // High boundary focus like Gen5
            compaction_iterations: 1500,      // Greedy compaction from Gen3
            compaction_threshold_n: 80,       // Only compact for small n
        }
    }
}

/// Track which boundary a tree is blocking (from Gen5)
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
    /// Get number of SA passes based on n
    #[inline]
    fn get_sa_passes(&self, n: usize) -> usize {
        if n <= 50 {
            self.config.sa_passes_small
        } else if n <= 120 {
            self.config.sa_passes_medium
        } else {
            self.config.sa_passes_large
        }
    }

    /// Get search attempts based on n (more for small n)
    #[inline]
    fn get_search_attempts(&self, n: usize) -> usize {
        if n <= 30 {
            self.config.search_attempts_base + 150  // 350 for tiny n
        } else if n <= 80 {
            self.config.search_attempts_base + 80   // 280 for small n
        } else if n <= 150 {
            self.config.search_attempts_base + 30   // 230 for medium n
        } else {
            self.config.search_attempts_base        // 200 for large n
        }
    }

    /// Get early exit threshold based on n
    #[inline]
    fn get_early_exit_threshold(&self, n: usize) -> usize {
        // More patience for small n, less for large n
        if n <= 50 {
            self.config.early_exit_base + 500
        } else if n <= 120 {
            self.config.early_exit_base + 200
        } else {
            self.config.early_exit_base
        }
    }

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

            // HYBRID: N-adaptive SA passes
            let num_passes = self.get_sa_passes(n);
            for pass in 0..num_passes {
                self.local_search(&mut trees, n, pass, num_passes, &mut rng);
            }

            // HYBRID: Greedy compaction only for small n (from Gen3)
            if n <= self.config.compaction_threshold_n {
                self.greedy_compaction(&mut trees, &mut rng);
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
        let search_attempts = self.get_search_attempts(n);

        for _ in 0..search_attempts {
            let dir = self.select_direction(n, rng);
            let vx = dir.cos();
            let vy = dir.sin();

            for &tree_angle in &angles {
                // Binary search for closest valid position
                // HYBRID: Finer precision for small n
                let precision = if n <= 60 { 0.0005 } else { 0.001 };
                let mut low = 0.0;
                let mut high = 12.0;

                while high - low > precision {
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
    /// HYBRID: Combines best scoring ideas from all generations
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
        // HYBRID: Adaptive weight from Gen3
        let balance_weight = 0.10 + 0.05 * (1.0 - (n as f64 / 200.0).min(1.0));
        let balance_penalty = (width - height).abs() * balance_weight;

        // HYBRID: From Gen5 - penalize extending the limiting dimension
        let limiting_dim_penalty = if width > height {
            let tree_extends_x = (max_x > pack_max_x - 0.01) || (min_x < pack_min_x + 0.01);
            if tree_extends_x { 0.04 } else { 0.0 }
        } else {
            let tree_extends_y = (max_y > pack_max_y - 0.01) || (min_y < pack_min_y + 0.01);
            if tree_extends_y { 0.04 } else { 0.0 }
        };

        // Tertiary: preference for compact center
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.007 / (n as f64).sqrt();

        // HYBRID: Enhanced density bonus from Gen3
        let area = width * height;
        let density_bonus = if area > 0.0 {
            -0.009 * (n as f64 / area).min(2.2)
        } else {
            0.0
        };

        // HYBRID: Perimeter bonus from Gen3
        let perimeter_bonus = -0.0015 * (2.0 * (width + height)) / (n as f64).sqrt();

        side_score + balance_penalty + limiting_dim_penalty + center_penalty + density_bonus + perimeter_bonus
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
        // HYBRID: 8 angles with n-dependent priority (efficient like Gen4/5)
        let base = match n % 4 {
            0 => vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0],
            1 => vec![90.0, 270.0, 0.0, 180.0, 135.0, 315.0, 45.0, 225.0],
            2 => vec![180.0, 0.0, 270.0, 90.0, 225.0, 45.0, 315.0, 135.0],
            _ => vec![270.0, 90.0, 180.0, 0.0, 315.0, 135.0, 225.0, 45.0],
        };
        base
    }

    /// EVOLVED FUNCTION: Select direction angle for placement search
    /// HYBRID: Combines best direction strategies
    #[inline]
    fn select_direction(&self, n: usize, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;

        let strategy = rng.gen::<f64>();

        if strategy < 0.45 {
            // Structured: evenly spaced with small jitter
            let base_idx = rng.gen_range(0..num_dirs);
            let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
            base + rng.gen_range(-0.055..0.055)
        } else if strategy < 0.70 {
            // Weighted random: favor corners and edges
            loop {
                let angle = rng.gen_range(0.0..2.0 * PI);
                let corner_weight = ((4.0 * angle).sin().abs() + (4.0 * angle).cos().abs()) / 2.0;
                let threshold = 0.18;
                if rng.gen::<f64>() < corner_weight.max(threshold) {
                    return angle;
                }
            }
        } else if strategy < 0.88 {
            // Golden angle spiral for good coverage (from Gen3)
            let golden_angle = PI * (3.0 - (5.0_f64).sqrt());
            let base = (n as f64 * golden_angle) % (2.0 * PI);
            let offset = rng.gen_range(0..10) as f64 * PI / 5.0;
            (base + offset + rng.gen_range(-0.08..0.08)) % (2.0 * PI)
        } else {
            // Fibonacci lattice directions (from Gen3)
            let idx = rng.gen_range(0..num_dirs);
            let golden_ratio = (1.0 + (5.0_f64).sqrt()) / 2.0;
            ((idx as f64 * 2.0 * PI / golden_ratio) % (2.0 * PI)) + rng.gen_range(-0.04..0.04)
        }
    }

    /// EVOLVED FUNCTION: Local search with simulated annealing
    /// HYBRID: Combines Gen5's smart boundary moves with Gen4's efficiency
    fn local_search(&self, trees: &mut Vec<PlacedTree>, n: usize, pass: usize, total_passes: usize, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config: Vec<PlacedTree> = trees.clone();

        // HYBRID: Temperature adjustment based on pass
        let temp_multiplier = match pass {
            0 => 1.0,
            1 => 0.38,
            _ => 0.18,
        };
        let mut temp = self.config.sa_initial_temp * temp_multiplier;

        // HYBRID: N-adaptive iterations
        let extra_iters = if n <= 60 { n * 150 } else if n <= 120 { n * 100 } else { n * 70 };
        let base_iterations = match pass {
            0 => self.config.sa_iterations + extra_iters,
            1 => self.config.sa_iterations / 2 + extra_iters / 2,
            _ => self.config.sa_iterations / 3 + extra_iters / 3,
        };

        let early_exit_threshold = self.get_early_exit_threshold(n);
        let mut iterations_without_improvement = 0;

        // HYBRID: Boundary cache with edge tracking (from Gen5)
        let mut boundary_cache_iter = 0;
        let mut boundary_info: Vec<(usize, BoundaryEdge)> = Vec::new();

        for iter in 0..base_iterations {
            // Early exit when no improvement (from Gen4)
            if iterations_without_improvement >= early_exit_threshold {
                break;
            }

            // Update boundary cache every 350 iterations
            if iter == 0 || iter - boundary_cache_iter >= 350 {
                boundary_info = self.find_boundary_trees_with_edges(trees);
                boundary_cache_iter = iter;
            }

            // HYBRID: High boundary focus from Gen5
            let (idx, edge) = if !boundary_info.is_empty() && rng.gen::<f64>() < self.config.boundary_focus_prob {
                let bi = &boundary_info[rng.gen_range(0..boundary_info.len())];
                (bi.0, bi.1)
            } else {
                (rng.gen_range(0..trees.len()), BoundaryEdge::None)
            };

            let old_tree = trees[idx].clone();

            // HYBRID: Smart moves from Gen5 combined with Gen3's variety
            let success = self.hybrid_move(trees, idx, temp, edge, rng);

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

        // Restore best configuration
        if best_side < compute_side_length(trees) {
            *trees = best_config;
        }
    }

    /// HYBRID: Greedy compaction from Gen3 (only for small n)
    fn greedy_compaction(&self, trees: &mut Vec<PlacedTree>, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut best_side = compute_side_length(trees);
        let mut improved = true;
        let mut iterations = 0;

        while improved && iterations < self.config.compaction_iterations {
            improved = false;
            iterations += 1;

            for idx in 0..trees.len() {
                let old_tree = trees[idx].clone();
                let (old_x, old_y, old_angle) = (old_tree.x, old_tree.y, old_tree.angle_deg);

                // Move toward center of bounding box
                let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                let center_x = (min_x + max_x) / 2.0;
                let center_y = (min_y + max_y) / 2.0;

                let dx = center_x - old_x;
                let dy = center_y - old_y;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist > 0.01 {
                    let step_sizes = [0.04, 0.018, 0.008, 0.004];

                    for &step in &step_sizes {
                        let new_x = old_x + dx / dist * step;
                        let new_y = old_y + dy / dist * step;

                        trees[idx] = PlacedTree::new(new_x, new_y, old_angle);

                        if !has_overlap(trees, idx) {
                            let new_side = compute_side_length(trees);
                            if new_side < best_side - 1e-9 {
                                best_side = new_side;
                                improved = true;
                                break;
                            }
                        }

                        trees[idx] = old_tree.clone();
                    }
                }

                // Rotation adjustments
                if !improved && rng.gen::<f64>() < 0.25 {
                    for delta_angle in &[45.0, -45.0, 90.0, -90.0] {
                        let new_angle = (old_angle + delta_angle).rem_euclid(360.0);
                        trees[idx] = PlacedTree::new(old_x, old_y, new_angle);

                        if !has_overlap(trees, idx) {
                            let new_side = compute_side_length(trees);
                            if new_side < best_side - 1e-9 {
                                best_side = new_side;
                                improved = true;
                                break;
                            }
                        }

                        trees[idx] = old_tree.clone();
                    }
                }
            }
        }
    }

    /// Find trees on boundary with edge information (from Gen5)
    #[inline]
    fn find_boundary_trees_with_edges(&self, trees: &[PlacedTree]) -> Vec<(usize, BoundaryEdge)> {
        if trees.is_empty() {
            return Vec::new();
        }

        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
        let eps = 0.012;

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

    /// HYBRID: Move operator combining Gen5's smart moves with Gen3's variety
    #[inline]
    fn hybrid_move(
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

        // HYBRID: Choose move type based on boundary edge (Gen5 strategy)
        // but with additional move types from Gen3
        let move_type = match edge {
            BoundaryEdge::Left => {
                match rng.gen_range(0..12) {
                    0..=4 => 0,   // Move right
                    5..=6 => 1,   // Slide vertically
                    7..=8 => 2,   // Rotate
                    9 => 10,      // Micro nudge (Gen3)
                    10 => 11,     // Center pull (Gen3)
                    _ => 3,
                }
            }
            BoundaryEdge::Right => {
                match rng.gen_range(0..12) {
                    0..=4 => 4,   // Move left
                    5..=6 => 1,   // Slide vertically
                    7..=8 => 2,   // Rotate
                    9 => 10,
                    10 => 11,
                    _ => 3,
                }
            }
            BoundaryEdge::Top => {
                match rng.gen_range(0..12) {
                    0..=4 => 5,   // Move down
                    5..=6 => 6,   // Slide horizontally
                    7..=8 => 2,   // Rotate
                    9 => 10,
                    10 => 11,
                    _ => 3,
                }
            }
            BoundaryEdge::Bottom => {
                match rng.gen_range(0..12) {
                    0..=4 => 7,   // Move up
                    5..=6 => 6,   // Slide horizontally
                    7..=8 => 2,   // Rotate
                    9 => 10,
                    10 => 11,
                    _ => 3,
                }
            }
            BoundaryEdge::Corner => {
                match rng.gen_range(0..12) {
                    0..=4 => 8,   // Gradient descent
                    5..=6 => 2,   // Rotate
                    7..=8 => 9,   // Diagonal slide
                    9 => 10,
                    10 => 11,
                    _ => 3,
                }
            }
            BoundaryEdge::None => {
                // Interior tree: full variety of moves
                rng.gen_range(0..14)
            }
        };

        let scale = self.config.translation_scale * (0.25 + temp * 1.8);

        match move_type {
            0 => {
                // Move right (for left boundary)
                let dx = rng.gen_range(scale * 0.25..scale);
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
                // Move left (for right boundary)
                let dx = rng.gen_range(-scale..-scale * 0.25);
                let dy = rng.gen_range(-scale * 0.2..scale * 0.2);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            5 => {
                // Move down (for top boundary)
                let dx = rng.gen_range(-scale * 0.2..scale * 0.2);
                let dy = rng.gen_range(-scale..-scale * 0.25);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            6 => {
                // Slide horizontally
                let dx = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y, old_angle);
            }
            7 => {
                // Move up (for bottom boundary)
                let dx = rng.gen_range(-scale * 0.2..scale * 0.2);
                let dy = rng.gen_range(scale * 0.25..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            8 => {
                // Gradient descent toward bounding box center
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
            10 => {
                // Micro nudge (from Gen3)
                let micro_scale = 0.012 * (0.4 + temp);
                let dx = rng.gen_range(-micro_scale..micro_scale);
                let dy = rng.gen_range(-micro_scale..micro_scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            11 => {
                // Center pull (from Gen3/4)
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.04 {
                    let pull_scale = self.config.center_pull_strength * (0.4 + temp * 1.5);
                    let dx = -old_x / mag * pull_scale;
                    let dy = -old_y / mag * pull_scale;
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                } else {
                    return false;
                }
            }
            12 => {
                // Polar move (from Gen3)
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
            13 => {
                // Angular orbit (from Gen3)
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.08 {
                    let current_angle = old_y.atan2(old_x);
                    let delta_angle = rng.gen_range(-0.15..0.15) * (1.0 + temp);
                    let new_ang = current_angle + delta_angle;
                    trees[idx] = PlacedTree::new(mag * new_ang.cos(), mag * new_ang.sin(), old_angle);
                } else {
                    return false;
                }
            }
            _ => {
                // Translate + rotate combo
                let dx = rng.gen_range(-scale * 0.4..scale * 0.4);
                let dy = rng.gen_range(-scale * 0.4..scale * 0.4);
                let dangle = rng.gen_range(-45.0..45.0);
                let new_angle = (old_angle + dangle).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, new_angle);
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
