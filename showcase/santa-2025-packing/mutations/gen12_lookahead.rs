//! Evolved Packing Algorithm - Generation 12 STRATEGY LOOKAHEAD
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
//! MUTATION STRATEGY: STRATEGY LOOKAHEAD (Gen12)
//! Building on Gen10's diverse strategies (~91.35 at n=200) with lookahead evaluation:
//!
//! Key insight: Gen10 tries 5 strategies independently and picks the best result
//! for each tree. But it only looks at the immediate result. Gen12 adds lookahead:
//!
//! 1. For each of the 5 strategies, simulate placing the NEXT 3 trees
//! 2. Score based on projected packing quality after those 3 trees
//! 3. Choose the strategy that looks best for future trees
//! 4. This helps avoid myopic decisions that look good now but hurt later
//!
//! Key changes from Gen10:
//! - When evaluating each strategy, simulate placing next 3 trees (reduced search)
//! - Score strategies by projected packing quality, not just immediate result
//! - Adaptive lookahead depth: 3 for small n, 2 for medium, 1 for large
//! - Cache strategy performance to inform future decisions
//!
//! Target: Beat Gen10's ~91.35 at n=200 by making globally-aware strategy choices

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Strategy for initial placement direction
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlacementStrategy {
    ClockwiseSpiral,
    CounterclockwiseSpiral,
    Grid,
    Random,
    BoundaryFirst,
}

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

    // DIVERSE STARTS: Number of independent attempts
    pub num_strategies: usize,

    // Density parameters (from Gen6)
    pub density_grid_resolution: usize,
    pub gap_penalty_weight: f64,
    pub local_density_radius: f64,
    pub fill_move_prob: f64,

    // LOOKAHEAD: New parameters for Gen12
    pub lookahead_depth_small_n: usize,    // Depth for small n (< 50)
    pub lookahead_depth_medium_n: usize,   // Depth for medium n (50-100)
    pub lookahead_depth_large_n: usize,    // Depth for large n (> 100)
    pub lookahead_search_attempts: usize,  // Reduced attempts for lookahead
    pub lookahead_discount_rate: f64,      // Discount for future scores
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen12 STRATEGY LOOKAHEAD: Configuration
        Self {
            search_attempts: 180,             // Slightly reduced for lookahead overhead
            direction_samples: 64,            // Good coverage per strategy
            sa_iterations: 20000,             // Balanced for multiple attempts + lookahead
            sa_initial_temp: 0.45,            // From Gen6
            sa_cooling_rate: 0.99993,         // Slightly faster for multi-attempt
            sa_min_temp: 0.00001,             // From Gen6
            translation_scale: 0.055,         // From Gen6
            rotation_granularity: 45.0,       // 8 angles
            center_pull_strength: 0.07,       // From Gen6
            sa_passes: 2,                     // Keep 2 passes
            early_exit_threshold: 1400,       // Slightly lower for efficiency
            boundary_focus_prob: 0.85,        // From Gen6
            // DIVERSE STARTS parameters
            num_strategies: 5,                // 5 different strategies
            // Density parameters from Gen6
            density_grid_resolution: 20,
            gap_penalty_weight: 0.15,
            local_density_radius: 0.5,
            fill_move_prob: 0.15,
            // LOOKAHEAD parameters for Gen12
            lookahead_depth_small_n: 3,       // Look 3 trees ahead for n < 50
            lookahead_depth_medium_n: 2,      // Look 2 trees ahead for 50 <= n < 100
            lookahead_depth_large_n: 1,       // Look 1 tree ahead for n >= 100
            lookahead_search_attempts: 40,    // Reduced attempts for speed
            lookahead_discount_rate: 0.75,    // Discount factor for future scores
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
    /// Pack all n from 1 to max_n using STRATEGY LOOKAHEAD
    pub fn pack_all(&self, max_n: usize) -> Vec<Packing> {
        let mut rng = rand::thread_rng();
        let mut packings: Vec<Packing> = Vec::with_capacity(max_n);

        // Track best configurations for each strategy
        let strategies = [
            PlacementStrategy::ClockwiseSpiral,
            PlacementStrategy::CounterclockwiseSpiral,
            PlacementStrategy::Grid,
            PlacementStrategy::Random,
            PlacementStrategy::BoundaryFirst,
        ];

        // Maintain separate tree configurations for each strategy
        let mut strategy_trees: Vec<Vec<PlacedTree>> = vec![Vec::new(); strategies.len()];

        for n in 1..=max_n {
            let mut best_trees: Option<Vec<PlacedTree>> = None;
            let mut best_score = f64::INFINITY;

            // Determine lookahead depth for this n
            let lookahead_depth = self.get_lookahead_depth(n, max_n);

            // Try each strategy independently with lookahead evaluation
            for (s_idx, &strategy) in strategies.iter().enumerate() {
                let mut trees = strategy_trees[s_idx].clone();

                // Place new tree using strategy-specific heuristics
                let new_tree = self.find_placement_with_strategy(&trees, n, max_n, strategy, &mut rng);
                trees.push(new_tree);

                // Run SA passes
                for pass in 0..self.config.sa_passes {
                    self.local_search(&mut trees, n, pass, strategy, &mut rng);
                }

                // LOOKAHEAD: Score this strategy by simulating future placements
                let strategy_score = if lookahead_depth > 0 {
                    self.score_strategy_with_lookahead(&trees, n, max_n, lookahead_depth, strategy, &mut rng)
                } else {
                    compute_side_length(&trees)
                };

                // Update strategy's best configuration
                strategy_trees[s_idx] = trees.clone();

                // Check if this is the best across all strategies
                if strategy_score < best_score {
                    best_score = strategy_score;
                    best_trees = Some(trees);
                }
            }

            // Store the best result
            let best = best_trees.unwrap();
            let mut packing = Packing::new();
            for t in &best {
                packing.trees.push(t.clone());
            }
            packings.push(packing);

            // Update all strategies to use the best configuration going forward
            // This helps propagate good solutions across strategies
            let best_side = compute_side_length(&best);
            for strat_trees in strategy_trees.iter_mut() {
                if compute_side_length(strat_trees) > best_side * 1.02 {
                    *strat_trees = best.clone();
                }
            }
        }

        packings
    }

    /// Determine lookahead depth based on n
    #[inline]
    fn get_lookahead_depth(&self, n: usize, max_n: usize) -> usize {
        // Don't look ahead past the final tree
        let remaining = max_n.saturating_sub(n);
        let base_depth = if n < 50 {
            self.config.lookahead_depth_small_n
        } else if n < 100 {
            self.config.lookahead_depth_medium_n
        } else {
            self.config.lookahead_depth_large_n
        };
        base_depth.min(remaining)
    }

    /// LOOKAHEAD: Score a strategy by simulating future tree placements
    fn score_strategy_with_lookahead(
        &self,
        trees: &[PlacedTree],
        n: usize,
        max_n: usize,
        depth: usize,
        strategy: PlacementStrategy,
        rng: &mut impl Rng,
    ) -> f64 {
        // Base score: current side length
        let mut total_score = compute_side_length(trees);

        // Simulate future placements
        let mut temp_trees: Vec<PlacedTree> = trees.to_vec();
        let mut discount = 1.0;

        for future_step in 1..=depth {
            let future_n = n + future_step;
            if future_n > max_n {
                break;
            }

            // Apply discount for future steps
            discount *= self.config.lookahead_discount_rate;

            // Find a quick placement for the future tree using this strategy
            let future_tree = self.find_placement_quick(&temp_trees, future_n, max_n, strategy, rng);
            temp_trees.push(future_tree);

            // Add discounted future side length to score
            let future_side = compute_side_length(&temp_trees);
            total_score += (future_side - compute_side_length(&temp_trees[..temp_trees.len()-1])) * discount;
        }

        // Final projected side length with heavier weight
        let final_side = compute_side_length(&temp_trees);
        total_score = total_score * 0.3 + final_side * 0.7;

        total_score
    }

    /// Quick placement search for lookahead simulation (reduced computation)
    fn find_placement_quick(
        &self,
        existing: &[PlacedTree],
        n: usize,
        _max_n: usize,
        strategy: PlacementStrategy,
        rng: &mut impl Rng,
    ) -> PlacedTree {
        if existing.is_empty() {
            let initial_angle = match strategy {
                PlacementStrategy::ClockwiseSpiral => 0.0,
                PlacementStrategy::CounterclockwiseSpiral => 90.0,
                PlacementStrategy::Grid => 45.0,
                PlacementStrategy::Random => rng.gen_range(0..8) as f64 * 45.0,
                PlacementStrategy::BoundaryFirst => 180.0,
            };
            return PlacedTree::new(0.0, 0.0, initial_angle);
        }

        let mut best_tree = PlacedTree::new(0.0, 0.0, 90.0);
        let mut best_score = f64::INFINITY;

        // Use reduced angles for speed
        let angles = vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0];

        let (min_x, min_y, max_x, max_y) = compute_bounds(existing);
        let current_width = max_x - min_x;
        let current_height = max_y - min_y;

        for attempt in 0..self.config.lookahead_search_attempts {
            // Quick strategy-specific direction
            let dir = self.select_direction_for_strategy(n, current_width, current_height, strategy, attempt, rng);
            let vx = dir.cos();
            let vy = dir.sin();

            for &tree_angle in &angles {
                // Binary search with coarser precision
                let mut low = 0.0;
                let mut high = 12.0;

                while high - low > 0.002 {  // Coarser than main search
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
                    let score = self.placement_score_quick(&candidate, existing);
                    if score < best_score {
                        best_score = score;
                        best_tree = candidate;
                    }
                }
            }
        }

        best_tree
    }

    /// Quick placement score for lookahead (simplified)
    #[inline]
    fn placement_score_quick(&self, tree: &PlacedTree, existing: &[PlacedTree]) -> f64 {
        let (tree_min_x, tree_min_y, tree_max_x, tree_max_y) = tree.bounds();

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

        // Simplified scoring for speed
        let balance_penalty = (width - height).abs() * 0.08;
        side + balance_penalty
    }

    /// Find best placement for new tree using strategy-specific approach
    fn find_placement_with_strategy(
        &self,
        existing: &[PlacedTree],
        n: usize,
        _max_n: usize,
        strategy: PlacementStrategy,
        rng: &mut impl Rng,
    ) -> PlacedTree {
        if existing.is_empty() {
            // Strategy-specific initial angle
            let initial_angle = match strategy {
                PlacementStrategy::ClockwiseSpiral => 0.0,
                PlacementStrategy::CounterclockwiseSpiral => 90.0,
                PlacementStrategy::Grid => 45.0,
                PlacementStrategy::Random => rng.gen_range(0..8) as f64 * 45.0,
                PlacementStrategy::BoundaryFirst => 180.0,
            };
            return PlacedTree::new(0.0, 0.0, initial_angle);
        }

        let mut best_tree = PlacedTree::new(0.0, 0.0, 90.0);
        let mut best_score = f64::INFINITY;

        let angles = self.select_angles_for_strategy(n, strategy);

        // Compute current bounds and density info
        let (min_x, min_y, max_x, max_y) = compute_bounds(existing);
        let current_width = max_x - min_x;
        let current_height = max_y - min_y;

        // Find gaps for density-aware placement
        let gaps = self.find_gaps(existing, min_x, min_y, max_x, max_y);

        for attempt in 0..self.config.search_attempts {
            // Strategy-specific direction selection
            let dir = if !gaps.is_empty() && attempt % 5 == 0 {
                // Sometimes target gaps directly
                let gap = &gaps[attempt % gaps.len()];
                let gap_cx = (gap.0 + gap.2) / 2.0;
                let gap_cy = (gap.1 + gap.3) / 2.0;
                gap_cy.atan2(gap_cx)
            } else {
                self.select_direction_for_strategy(n, current_width, current_height, strategy, attempt, rng)
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

    /// Select rotation angles based on strategy
    #[inline]
    fn select_angles_for_strategy(&self, n: usize, strategy: PlacementStrategy) -> Vec<f64> {
        match strategy {
            PlacementStrategy::ClockwiseSpiral => {
                // Favor angles that work well for clockwise packing
                vec![0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
            }
            PlacementStrategy::CounterclockwiseSpiral => {
                // Reverse order for counterclockwise
                vec![315.0, 270.0, 225.0, 180.0, 135.0, 90.0, 45.0, 0.0]
            }
            PlacementStrategy::Grid => {
                // Prefer axis-aligned for grid packing
                vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0]
            }
            PlacementStrategy::Random => {
                // Vary based on n for diversity
                match n % 4 {
                    0 => vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0],
                    1 => vec![90.0, 270.0, 0.0, 180.0, 135.0, 315.0, 45.0, 225.0],
                    2 => vec![180.0, 0.0, 270.0, 90.0, 225.0, 45.0, 315.0, 135.0],
                    _ => vec![270.0, 90.0, 180.0, 0.0, 315.0, 135.0, 225.0, 45.0],
                }
            }
            PlacementStrategy::BoundaryFirst => {
                // Angles that work well for boundary placement
                vec![45.0, 135.0, 225.0, 315.0, 0.0, 90.0, 180.0, 270.0]
            }
        }
    }

    /// Select direction based on strategy
    #[inline]
    fn select_direction_for_strategy(
        &self,
        n: usize,
        width: f64,
        height: f64,
        strategy: PlacementStrategy,
        attempt: usize,
        rng: &mut impl Rng,
    ) -> f64 {
        match strategy {
            PlacementStrategy::ClockwiseSpiral => {
                // Clockwise spiral: start at top, go right, down, left, up...
                let golden_angle = PI * (3.0 - (5.0_f64).sqrt());
                let base = (n as f64 * golden_angle) % (2.0 * PI);
                let offset = (attempt as f64 / self.config.search_attempts as f64) * 2.0 * PI;
                (base + offset) % (2.0 * PI)
            }
            PlacementStrategy::CounterclockwiseSpiral => {
                // Counterclockwise: opposite direction
                let golden_angle = -PI * (3.0 - (5.0_f64).sqrt());
                let base = (n as f64 * golden_angle).rem_euclid(2.0 * PI);
                let offset = (attempt as f64 / self.config.search_attempts as f64) * 2.0 * PI;
                (base - offset).rem_euclid(2.0 * PI)
            }
            PlacementStrategy::Grid => {
                // Grid pattern: structured directions
                let num_dirs = 16;
                let base_idx = attempt % num_dirs;
                let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
                // Add slight jitter for variation
                base + rng.gen_range(-0.03..0.03)
            }
            PlacementStrategy::Random => {
                // Pure random with some structure
                let mix = rng.gen::<f64>();
                if mix < 0.5 {
                    rng.gen_range(0.0..2.0 * PI)
                } else {
                    // Bias toward shorter dimension
                    if width < height {
                        let angle = if rng.gen() { 0.0 } else { PI };
                        angle + rng.gen_range(-PI / 3.0..PI / 3.0)
                    } else {
                        let angle = if rng.gen() { PI / 2.0 } else { -PI / 2.0 };
                        angle + rng.gen_range(-PI / 3.0..PI / 3.0)
                    }
                }
            }
            PlacementStrategy::BoundaryFirst => {
                // Prioritize corners and edges
                let prob = rng.gen::<f64>();
                if prob < 0.4 {
                    // Corners
                    let corners = [PI / 4.0, 3.0 * PI / 4.0, 5.0 * PI / 4.0, 7.0 * PI / 4.0];
                    corners[attempt % 4] + rng.gen_range(-0.1..0.1)
                } else if prob < 0.8 {
                    // Edges
                    let edges = [0.0, PI / 2.0, PI, 3.0 * PI / 2.0];
                    edges[attempt % 4] + rng.gen_range(-0.2..0.2)
                } else {
                    // Random for coverage
                    rng.gen_range(0.0..2.0 * PI)
                }
            }
        }
    }

    /// Score a placement (lower is better) - from Gen6 with density awareness
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

        // Calculate local density around the new tree
        let tree_cx = (tree_min_x + tree_max_x) / 2.0;
        let tree_cy = (tree_min_y + tree_max_y) / 2.0;
        let local_density = self.calculate_local_density(tree_cx, tree_cy, existing);

        // Reward high local density (tree is filling a gap)
        let density_bonus = -self.config.gap_penalty_weight * local_density;

        // Penalize placements that extend the bounding box
        let (old_min_x, old_min_y, old_max_x, old_max_y) = if !existing.is_empty() {
            compute_bounds(existing)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        let x_extension = (pack_max_x - old_max_x).max(0.0) + (old_min_x - pack_min_x).max(0.0);
        let y_extension = (pack_max_y - old_max_y).max(0.0) + (old_min_y - pack_min_y).max(0.0);
        let extension_penalty = (x_extension + y_extension) * 0.08;

        // Penalize leaving unusable gaps
        let gap_penalty = self.estimate_unusable_gap(tree, existing) * self.config.gap_penalty_weight;

        // Center penalty
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.005 / (n as f64).sqrt();

        // Neighbor proximity bonus
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

    /// Estimate if placement creates an unusable gap
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

    /// Bonus for being close to existing trees
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

    /// Find gaps in the current packing
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

    /// Local search with simulated annealing
    fn local_search(
        &self,
        trees: &mut Vec<PlacedTree>,
        n: usize,
        pass: usize,
        _strategy: PlacementStrategy,
        rng: &mut impl Rng,
    ) {
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
            if iterations_without_improvement >= self.config.early_exit_threshold {
                break;
            }

            // Update boundary cache every 300 iterations
            if iter == 0 || iter - boundary_cache_iter >= 300 {
                boundary_info = self.find_boundary_trees_with_edges(trees);
                boundary_cache_iter = iter;
            }

            // Choose between boundary optimization and gap-filling
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

            let success = self.sa_move(trees, idx, temp, edge, do_fill_move, rng);

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

    /// SA move operator with gap-filling awareness
    #[inline]
    fn sa_move(
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

        if is_fill_move {
            let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
            let bbox_cx = (min_x + max_x) / 2.0;
            let bbox_cy = (min_y + max_y) / 2.0;

            let move_type = rng.gen_range(0..4);
            match move_type {
                0 => {
                    // Move toward center of bbox
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
                    // Rotate
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
            // Standard boundary-aware moves
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
    fn test_lookahead_depth() {
        let packer = EvolvedPacker::default();

        // Small n should use depth 3
        assert_eq!(packer.get_lookahead_depth(10, 200), 3);
        assert_eq!(packer.get_lookahead_depth(49, 200), 3);

        // Medium n should use depth 2
        assert_eq!(packer.get_lookahead_depth(50, 200), 2);
        assert_eq!(packer.get_lookahead_depth(99, 200), 2);

        // Large n should use depth 1
        assert_eq!(packer.get_lookahead_depth(100, 200), 1);
        assert_eq!(packer.get_lookahead_depth(150, 200), 1);

        // Near end should be limited by remaining trees
        assert_eq!(packer.get_lookahead_depth(199, 200), 1);
        assert_eq!(packer.get_lookahead_depth(200, 200), 0);
    }

    #[test]
    fn test_strategy_lookahead() {
        // Test that lookahead produces different strategy scores
        let packer = EvolvedPacker::default();
        let packings = packer.pack_all(10);

        // Just verify it works and produces valid packings
        for (i, p) in packings.iter().enumerate() {
            assert_eq!(p.trees.len(), i + 1);
            assert!(!p.has_overlaps());
        }
    }
}
