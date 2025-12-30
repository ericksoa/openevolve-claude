//! Evolved Packing Algorithm - Generation 12 FINE-TUNED ANGLES
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
//! MUTATION STRATEGY: FINE-TUNED ANGLES (Gen12)
//! Better angle selection for tree packing:
//!
//! Key improvements from Gen10:
//! - Use 12 specific angles instead of 8 (45-degree increments)
//! - Primary angles: 0, 90, 180, 270 (cardinal directions)
//! - Secondary angles: 30, 60, 120, 150, 210, 240, 300, 330 (hexagonal-related)
//! - Try all 12 angles during placement search
//! - Weight angles based on existing tree orientations for better meshing
//! - Maintains diverse starts strategy from Gen10
//!
//! Target: Beat Gen10's ~91.35 at n=200 with optimized angle selection

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Strategy for initial placement direction
#[derive(Clone, Copy, Debug)]
pub enum PlacementStrategy {
    ClockwiseSpiral,
    CounterclockwiseSpiral,
    Grid,
    Random,
    BoundaryFirst,
}

/// All 12 fine-tuned angles for tree packing
/// Cardinal angles (primary): 0, 90, 180, 270
/// Hexagonal-related angles (secondary): 30, 60, 120, 150, 210, 240, 300, 330
const FINE_TUNED_ANGLES: [f64; 12] = [
    0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0,
];

/// Cardinal angles only (4 angles)
const CARDINAL_ANGLES: [f64; 4] = [0.0, 90.0, 180.0, 270.0];

/// Hexagonal-related angles only (8 angles)
const HEXAGONAL_ANGLES: [f64; 8] = [30.0, 60.0, 120.0, 150.0, 210.0, 240.0, 300.0, 330.0];

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

    // FINE-TUNED ANGLES parameters (Gen12)
    pub angle_weight_cardinal: f64,      // Weight for cardinal angles (0, 90, 180, 270)
    pub angle_weight_hexagonal: f64,     // Weight for hexagonal angles (30, 60, etc.)
    pub angle_adaptation_strength: f64,  // How strongly to adapt to existing orientations
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen12 FINE-TUNED ANGLES: Enhanced angle selection
        Self {
            search_attempts: 200,            // Same as Gen10
            direction_samples: 64,           // Good coverage per strategy
            sa_iterations: 22000,            // Balanced for multiple attempts
            sa_initial_temp: 0.45,           // From Gen6
            sa_cooling_rate: 0.99993,        // Slightly faster for multi-attempt
            sa_min_temp: 0.00001,            // From Gen6
            translation_scale: 0.055,        // From Gen6
            rotation_granularity: 30.0,      // 12 angles (finer than Gen10's 45.0)
            center_pull_strength: 0.07,      // From Gen6
            sa_passes: 2,                    // Keep 2 passes
            early_exit_threshold: 1500,      // Slightly lower for efficiency
            // DIVERSE STARTS parameters
            num_strategies: 5,               // 5 different strategies
            // Density parameters from Gen6
            density_grid_resolution: 20,
            gap_penalty_weight: 0.15,
            local_density_radius: 0.5,
            fill_move_prob: 0.15,
            // FINE-TUNED ANGLES parameters (Gen12)
            angle_weight_cardinal: 1.2,      // Slight preference for cardinal angles
            angle_weight_hexagonal: 1.0,     // Base weight for hexagonal angles
            angle_adaptation_strength: 0.3,  // Moderate adaptation to existing orientations
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
    /// Pack all n from 1 to max_n using DIVERSE STARTS strategy with FINE-TUNED ANGLES
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
            let mut best_side = f64::INFINITY;

            // Try each strategy independently
            for (s_idx, &strategy) in strategies.iter().enumerate() {
                let mut trees = strategy_trees[s_idx].clone();

                // Place new tree using strategy-specific heuristics with fine-tuned angles
                let new_tree = self.find_placement_with_strategy(&trees, n, max_n, strategy, &mut rng);
                trees.push(new_tree);

                // Run SA passes
                for pass in 0..self.config.sa_passes {
                    self.local_search(&mut trees, n, pass, strategy, &mut rng);
                }

                let side = compute_side_length(&trees);

                // Update strategy's best configuration
                strategy_trees[s_idx] = trees.clone();

                // Check if this is the best across all strategies
                if side < best_side {
                    best_side = side;
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
            for strat_trees in strategy_trees.iter_mut() {
                if compute_side_length(strat_trees) > best_side * 1.02 {
                    *strat_trees = best.clone();
                }
            }
        }

        packings
    }

    /// Analyze existing tree orientations and compute angle weights
    fn compute_angle_weights(&self, existing: &[PlacedTree]) -> Vec<(f64, f64)> {
        let mut angle_weights: Vec<(f64, f64)> = Vec::with_capacity(12);

        // Count how many trees are near each angle
        let mut angle_counts: [f64; 12] = [0.0; 12];

        for tree in existing {
            let tree_angle = tree.angle_deg.rem_euclid(360.0);
            for (i, &angle) in FINE_TUNED_ANGLES.iter().enumerate() {
                let diff = (tree_angle - angle).abs();
                let diff = diff.min(360.0 - diff);
                if diff < 15.0 {
                    // Count trees within 15 degrees of this angle
                    angle_counts[i] += 1.0 - diff / 15.0;
                }
            }
        }

        // Normalize and compute weights
        let max_count = angle_counts.iter().cloned().fold(0.0_f64, f64::max).max(1.0);

        for (i, &angle) in FINE_TUNED_ANGLES.iter().enumerate() {
            // Base weight depends on whether it's cardinal or hexagonal
            let base_weight = if CARDINAL_ANGLES.contains(&angle) {
                self.config.angle_weight_cardinal
            } else {
                self.config.angle_weight_hexagonal
            };

            // Adapt weight based on existing orientations
            // Trees tend to mesh better when at complementary angles
            let popularity = angle_counts[i] / max_count;

            // Weight complementary angles higher (angles 180 degrees apart)
            let complementary_idx = (i + 6) % 12;
            let complementary_popularity = angle_counts[complementary_idx] / max_count;

            // Also consider adjacent angles for smooth transitions
            let prev_idx = if i == 0 { 11 } else { i - 1 };
            let next_idx = (i + 1) % 12;
            let adjacent_popularity = (angle_counts[prev_idx] + angle_counts[next_idx]) / (2.0 * max_count);

            // Final weight: base + adaptation bonus
            let adaptation_bonus = self.config.angle_adaptation_strength *
                (complementary_popularity * 0.5 + adjacent_popularity * 0.3 + (1.0 - popularity) * 0.2);

            let final_weight = base_weight + adaptation_bonus;
            angle_weights.push((angle, final_weight));
        }

        // Sort by weight (highest first) for priority during search
        angle_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        angle_weights
    }

    /// Find best placement for new tree using strategy-specific approach with fine-tuned angles
    fn find_placement_with_strategy(
        &self,
        existing: &[PlacedTree],
        n: usize,
        _max_n: usize,
        strategy: PlacementStrategy,
        rng: &mut impl Rng,
    ) -> PlacedTree {
        if existing.is_empty() {
            // Strategy-specific initial angle from fine-tuned set
            let initial_angle = match strategy {
                PlacementStrategy::ClockwiseSpiral => 0.0,
                PlacementStrategy::CounterclockwiseSpiral => 90.0,
                PlacementStrategy::Grid => 0.0,  // Grid prefers cardinal angles
                PlacementStrategy::Random => FINE_TUNED_ANGLES[rng.gen_range(0..12)],
                PlacementStrategy::BoundaryFirst => 180.0,
            };
            return PlacedTree::new(0.0, 0.0, initial_angle);
        }

        let mut best_tree = PlacedTree::new(0.0, 0.0, 90.0);
        let mut best_score = f64::INFINITY;

        // Get weighted angles based on existing tree orientations
        let weighted_angles = self.compute_angle_weights(existing);
        let angles = self.select_angles_for_strategy(n, strategy, &weighted_angles);

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

            // Try all 12 fine-tuned angles (prioritized by weight)
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

    /// Select rotation angles based on strategy with fine-tuned 12-angle system
    #[inline]
    fn select_angles_for_strategy(&self, n: usize, strategy: PlacementStrategy, weighted_angles: &[(f64, f64)]) -> Vec<f64> {
        // Extract just the angles from weighted pairs (already sorted by weight)
        let prioritized: Vec<f64> = weighted_angles.iter().map(|(a, _)| *a).collect();

        match strategy {
            PlacementStrategy::ClockwiseSpiral => {
                // Start with cardinal angles, then hexagonal
                let mut angles = CARDINAL_ANGLES.to_vec();
                angles.extend_from_slice(&HEXAGONAL_ANGLES);
                angles
            }
            PlacementStrategy::CounterclockwiseSpiral => {
                // Reverse order - prioritize weighted angles
                let mut angles: Vec<f64> = prioritized.clone();
                angles.reverse();
                angles
            }
            PlacementStrategy::Grid => {
                // Grid packing prefers cardinal angles strongly
                let mut angles = CARDINAL_ANGLES.to_vec();
                // Add hexagonal angles sorted by weight
                for angle in &prioritized {
                    if !CARDINAL_ANGLES.contains(angle) {
                        angles.push(*angle);
                    }
                }
                angles
            }
            PlacementStrategy::Random => {
                // Use weighted order with some variation based on n
                let shift = n % 12;
                let mut angles = Vec::with_capacity(12);
                for i in 0..12 {
                    let idx = (i + shift) % 12;
                    angles.push(FINE_TUNED_ANGLES[idx]);
                }
                angles
            }
            PlacementStrategy::BoundaryFirst => {
                // Boundary placement: use weight-prioritized order
                prioritized
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
                // Grid pattern: structured directions aligned with fine-tuned angles
                let num_dirs = 24;  // Increased for finer resolution (matching 30-degree increments)
                let base_idx = attempt % num_dirs;
                let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
                // Add slight jitter for variation
                base + rng.gen_range(-0.02..0.02)
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
                // Prioritize corners and edges with 30-degree aligned directions
                let prob = rng.gen::<f64>();
                if prob < 0.4 {
                    // Corners at 30-degree related angles
                    let corners = [PI / 6.0, PI / 3.0, 2.0 * PI / 3.0, 5.0 * PI / 6.0,
                                   7.0 * PI / 6.0, 4.0 * PI / 3.0, 5.0 * PI / 3.0, 11.0 * PI / 6.0];
                    corners[attempt % 8] + rng.gen_range(-0.08..0.08)
                } else if prob < 0.8 {
                    // Edges (cardinal directions)
                    let edges = [0.0, PI / 2.0, PI, 3.0 * PI / 2.0];
                    edges[attempt % 4] + rng.gen_range(-0.15..0.15)
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

        // Angle alignment bonus (Gen12): reward angles that mesh well with neighbors
        let angle_bonus = self.angle_alignment_bonus(tree, existing);

        side_score + balance_penalty + extension_penalty + gap_penalty + center_penalty + density_bonus - neighbor_bonus - angle_bonus
    }

    /// Bonus for angle alignment with nearby trees (Gen12 addition)
    #[inline]
    fn angle_alignment_bonus(&self, tree: &PlacedTree, existing: &[PlacedTree]) -> f64 {
        if existing.is_empty() {
            return 0.0;
        }

        let (tree_min_x, tree_min_y, tree_max_x, tree_max_y) = tree.bounds();
        let tree_cx = (tree_min_x + tree_max_x) / 2.0;
        let tree_cy = (tree_min_y + tree_max_y) / 2.0;
        let tree_angle = tree.angle_deg.rem_euclid(360.0);

        let mut bonus = 0.0;
        let proximity_threshold = 1.0;

        for other in existing {
            let (ox1, oy1, ox2, oy2) = other.bounds();
            let other_cx = (ox1 + ox2) / 2.0;
            let other_cy = (oy1 + oy2) / 2.0;

            let dx = tree_cx - other_cx;
            let dy = tree_cy - other_cy;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < proximity_threshold {
                let other_angle = other.angle_deg.rem_euclid(360.0);
                let angle_diff = (tree_angle - other_angle).abs();
                let angle_diff = angle_diff.min(360.0 - angle_diff);

                // Reward complementary angles (90 or 180 degrees apart)
                let complementary_90 = (angle_diff - 90.0).abs();
                let complementary_180 = (angle_diff - 180.0).abs();
                let min_complementary = complementary_90.min(complementary_180);

                if min_complementary < 15.0 {
                    let proximity_factor = 1.0 - dist / proximity_threshold;
                    bonus += 0.01 * proximity_factor * (1.0 - min_complementary / 15.0);
                }
            }
        }

        bonus
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

    /// Local search with simulated annealing (with fine-tuned angle rotations)
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

    /// SA move operator with gap-filling awareness and fine-tuned angle rotations
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
                    // Rotate using fine-tuned angles (30-degree increments)
                    let angles = [30.0, 60.0, 90.0, -30.0, -60.0, -90.0, 120.0, -120.0];
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
            // Standard boundary-aware moves with fine-tuned angles
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
                    // Fine-tuned angle rotations (30-degree increments instead of 45)
                    let angles = [30.0, 60.0, 90.0, -30.0, -60.0, -90.0];
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
    fn test_fine_tuned_angles() {
        // Test that all 12 angles are used
        assert_eq!(FINE_TUNED_ANGLES.len(), 12);
        assert_eq!(CARDINAL_ANGLES.len(), 4);
        assert_eq!(HEXAGONAL_ANGLES.len(), 8);

        // Test that cardinal angles are in the fine-tuned set
        for angle in &CARDINAL_ANGLES {
            assert!(FINE_TUNED_ANGLES.contains(angle));
        }

        // Test that hexagonal angles are in the fine-tuned set
        for angle in &HEXAGONAL_ANGLES {
            assert!(FINE_TUNED_ANGLES.contains(angle));
        }
    }

    #[test]
    fn test_angle_weights() {
        let packer = EvolvedPacker::default();

        // Test with empty trees
        let empty: Vec<PlacedTree> = Vec::new();
        let weights = packer.compute_angle_weights(&empty);
        assert_eq!(weights.len(), 12);

        // Test with some trees
        let trees = vec![
            PlacedTree::new(0.0, 0.0, 0.0),
            PlacedTree::new(1.0, 0.0, 90.0),
            PlacedTree::new(0.0, 1.0, 180.0),
        ];
        let weights = packer.compute_angle_weights(&trees);
        assert_eq!(weights.len(), 12);
    }

    #[test]
    fn test_diverse_strategies_with_angles() {
        // Test that different strategies produce valid packings
        let packer = EvolvedPacker::default();
        let packings = packer.pack_all(10);

        for (i, p) in packings.iter().enumerate() {
            assert_eq!(p.trees.len(), i + 1);
            assert!(!p.has_overlaps());
        }
    }
}
