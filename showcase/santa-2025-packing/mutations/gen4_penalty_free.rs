//! Evolved Packing Algorithm - Generation 4 PENALTY-FREE ZONES
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
//! MUTATION STRATEGY: PENALTY-FREE ZONES (Gen4)
//! Building on Gen3 champion (score 101.90) with geometric awareness:
//!
//! Key innovations:
//! - Detect "penalty-free" configurations (tip-to-trunk, interlocking)
//! - Special move operators that preserve penalty-free zones
//! - Reward placements that create new penalty-free arrangements
//! - Track and exploit successful geometric patterns
//!
//! Hypothesis: Certain geometric arrangements allow trees to pack tighter
//! with minimal wasted space. By identifying and preserving these patterns,
//! we can achieve better overall density.

use crate::{PlacedTree, Packing, TREE_VERTICES};
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

    // Multi-pass settings (from Gen3)
    pub sa_passes: usize,
    pub restart_threshold: usize,
    pub reheat_temp: f64,
    pub compaction_iterations: usize,

    // PENALTY-FREE ZONE: New parameters
    pub penalty_free_bonus: f64,        // Reward for penalty-free placements
    pub interlocking_threshold: f64,    // Distance threshold for interlocking detection
    pub pattern_preservation_prob: f64, // Probability to use pattern-preserving moves
    pub tip_trunk_reward: f64,          // Extra bonus for tip-to-trunk arrangements
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        Self {
            // Inherit optimized search parameters from Gen3
            search_attempts: 400,
            direction_samples: 96,
            sa_iterations: 40000,
            sa_initial_temp: 0.7,
            sa_cooling_rate: 0.99998,
            sa_min_temp: 0.000001,
            translation_scale: 0.08,
            rotation_granularity: 22.5,
            center_pull_strength: 0.06,
            sa_passes: 3,
            restart_threshold: 5000,
            reheat_temp: 0.4,
            compaction_iterations: 2000,

            // PENALTY-FREE ZONE: New tuned parameters
            penalty_free_bonus: 0.015,       // Significant bonus for penalty-free zones
            interlocking_threshold: 0.25,    // Trees within this distance may interlock
            pattern_preservation_prob: 0.35, // 35% of moves preserve patterns
            tip_trunk_reward: 0.02,          // Extra reward for tip-to-trunk
        }
    }
}

/// Main evolved packer with penalty-free zone awareness
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

            // Place new tree using evolved heuristics with penalty-free awareness
            let new_tree = self.find_placement(&trees, n, max_n, &mut rng);
            trees.push(new_tree);

            // Run SA passes with pattern-aware optimization
            for pass in 0..self.config.sa_passes {
                self.local_search(&mut trees, n, pass, &mut rng);
            }

            // Greedy compaction with pattern preservation
            self.greedy_compaction(&mut trees, &mut rng);

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
    /// Now includes penalty-free zone awareness
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

        for _ in 0..self.config.search_attempts {
            let dir = self.select_direction(n, rng);
            let vx = dir.cos();
            let vy = dir.sin();

            for &tree_angle in &angles {
                // Binary search for closest valid position
                let mut low = 0.0;
                let mut high = 12.0;

                while high - low > 0.0001 {
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
                    // PENALTY-FREE: Enhanced scoring with pattern detection
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
    /// Enhanced with penalty-free zone detection and rewards
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

        // Primary: minimize side length
        let side_score = side;

        // Secondary: balanced aspect ratio (from Gen3)
        let balance_weight = 0.12 + 0.06 * (1.0 - (n as f64 / 200.0).min(1.0));
        let balance_penalty = (width - height).abs() * balance_weight;

        // Tertiary: center compactness (from Gen3)
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.010 / (n as f64).sqrt();

        // Gen3 density heuristic
        let area = width * height;
        let density_bonus = if area > 0.0 {
            -0.008 * (n as f64 / area).min(2.5)
        } else {
            0.0
        };

        // Gen3 perimeter bonus
        let perimeter_bonus = -0.002 * (2.0 * (width + height)) / (n as f64).sqrt();

        // PENALTY-FREE ZONE: Detect and reward good geometric patterns
        let pattern_bonus = self.compute_pattern_bonus(tree, existing, n);

        // PENALTY-FREE ZONE: Reward tight neighbor relationships
        let neighbor_bonus = self.compute_neighbor_bonus(tree, existing);

        side_score + balance_penalty + center_penalty + density_bonus + perimeter_bonus
            + pattern_bonus + neighbor_bonus
    }

    /// PENALTY-FREE ZONE: Compute bonus for favorable geometric patterns
    #[inline]
    fn compute_pattern_bonus(&self, tree: &PlacedTree, existing: &[PlacedTree], n: usize) -> f64 {
        let mut bonus = 0.0;

        // Get tree's tip and trunk positions
        let (tip_x, tip_y) = self.get_tip_position(tree);
        let (trunk_x, trunk_y) = self.get_trunk_center(tree);

        for other in existing {
            let (other_tip_x, other_tip_y) = self.get_tip_position(other);
            let (other_trunk_x, other_trunk_y) = self.get_trunk_center(other);

            // Check tip-to-trunk arrangement
            let tip_to_trunk_dist = ((tip_x - other_trunk_x).powi(2)
                                   + (tip_y - other_trunk_y).powi(2)).sqrt();
            let trunk_to_tip_dist = ((trunk_x - other_tip_x).powi(2)
                                   + (trunk_y - other_tip_y).powi(2)).sqrt();

            if tip_to_trunk_dist < self.config.interlocking_threshold {
                // Tip nestles into trunk area - excellent arrangement
                bonus -= self.config.tip_trunk_reward * (1.0 - tip_to_trunk_dist / self.config.interlocking_threshold);
            }
            if trunk_to_tip_dist < self.config.interlocking_threshold {
                bonus -= self.config.tip_trunk_reward * (1.0 - trunk_to_tip_dist / self.config.interlocking_threshold);
            }

            // Check anti-parallel arrangement (trees pointing opposite ways)
            let angle_diff = (tree.angle_deg - other.angle_deg).abs();
            let is_anti_parallel = (angle_diff - 180.0).abs() < 30.0;

            if is_anti_parallel {
                let center_dist = ((tree.x - other.x).powi(2) + (tree.y - other.y).powi(2)).sqrt();
                if center_dist < 0.8 {
                    // Anti-parallel trees close together - good interlocking potential
                    bonus -= self.config.penalty_free_bonus * (1.0 - center_dist / 0.8);
                }
            }

            // Check parallel nesting (same direction, offset)
            let is_parallel = angle_diff < 30.0 || (angle_diff - 360.0).abs() < 30.0;
            if is_parallel {
                // Perpendicular offset calculation
                let angle_rad = tree.angle_deg * PI / 180.0;
                let perp_x = -(tree.y - other.y) * angle_rad.cos() + (tree.x - other.x) * angle_rad.sin();
                let perp_offset = perp_x.abs();

                if perp_offset > 0.15 && perp_offset < 0.5 {
                    // Good parallel nesting - branches can interleave
                    bonus -= self.config.penalty_free_bonus * 0.7;
                }
            }
        }

        // Scale bonus by tree count (more important when dense)
        bonus * (1.0 + (n as f64 / 100.0).min(1.0))
    }

    /// PENALTY-FREE ZONE: Bonus for having close non-overlapping neighbors
    #[inline]
    fn compute_neighbor_bonus(&self, tree: &PlacedTree, existing: &[PlacedTree]) -> f64 {
        let mut close_neighbors = 0;
        let mut total_gap = 0.0;

        for other in existing {
            let dist = ((tree.x - other.x).powi(2) + (tree.y - other.y).powi(2)).sqrt();

            if dist < 1.0 {
                // Trees are close - measure tightness
                close_neighbors += 1;

                // Compute minimum gap between vertices (approximation)
                let min_gap = self.estimate_min_gap(tree, other);
                if min_gap < 0.1 {
                    // Very tight packing - reward
                    total_gap += 0.1 - min_gap;
                }
            }
        }

        // Reward having close neighbors with tight gaps
        if close_neighbors > 0 {
            -self.config.penalty_free_bonus * (total_gap / close_neighbors as f64)
        } else {
            0.0
        }
    }

    /// Estimate minimum gap between two trees
    #[inline]
    fn estimate_min_gap(&self, tree1: &PlacedTree, tree2: &PlacedTree) -> f64 {
        let v1 = tree1.vertices();
        let v2 = tree2.vertices();

        let mut min_dist = f64::INFINITY;

        // Sample vertex-to-vertex distances
        for &(x1, y1) in v1 {
            for &(x2, y2) in v2 {
                let dist = ((x1 - x2).powi(2) + (y1 - y2).powi(2)).sqrt();
                min_dist = min_dist.min(dist);
            }
        }

        min_dist
    }

    /// Get tip position of a tree (vertex 0 after rotation)
    #[inline]
    fn get_tip_position(&self, tree: &PlacedTree) -> (f64, f64) {
        let angle_rad = tree.angle_deg * PI / 180.0;
        let (vx, vy) = TREE_VERTICES[0]; // Tip at (0, 0.8)
        let rx = vx * angle_rad.cos() - vy * angle_rad.sin();
        let ry = vx * angle_rad.sin() + vy * angle_rad.cos();
        (tree.x + rx, tree.y + ry)
    }

    /// Get trunk center position
    #[inline]
    fn get_trunk_center(&self, tree: &PlacedTree) -> (f64, f64) {
        let angle_rad = tree.angle_deg * PI / 180.0;
        // Trunk center is approximately at (0, -0.1)
        let (vx, vy) = (0.0, -0.1);
        let rx = vx * angle_rad.cos() - vy * angle_rad.sin();
        let ry = vx * angle_rad.sin() + vy * angle_rad.cos();
        (tree.x + rx, tree.y + ry)
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    /// Prioritize angles that create penalty-free arrangements
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
        // PENALTY-FREE: Include complementary angles (180 apart) early
        // for anti-parallel arrangements
        let base = match n % 8 {
            0 => vec![0.0, 180.0, 90.0, 270.0, 45.0, 225.0, 135.0, 315.0,
                      22.5, 202.5, 67.5, 247.5, 112.5, 292.5, 157.5, 337.5],
            1 => vec![90.0, 270.0, 0.0, 180.0, 135.0, 315.0, 45.0, 225.0,
                      67.5, 247.5, 112.5, 292.5, 22.5, 202.5, 157.5, 337.5],
            2 => vec![180.0, 0.0, 270.0, 90.0, 225.0, 45.0, 315.0, 135.0,
                      157.5, 337.5, 202.5, 22.5, 67.5, 247.5, 112.5, 292.5],
            3 => vec![270.0, 90.0, 180.0, 0.0, 315.0, 135.0, 225.0, 45.0,
                      247.5, 67.5, 292.5, 112.5, 157.5, 337.5, 202.5, 22.5],
            4 => vec![45.0, 225.0, 135.0, 315.0, 0.0, 180.0, 90.0, 270.0,
                      22.5, 202.5, 67.5, 247.5, 112.5, 292.5, 157.5, 337.5],
            5 => vec![135.0, 315.0, 45.0, 225.0, 90.0, 270.0, 0.0, 180.0,
                      112.5, 292.5, 157.5, 337.5, 22.5, 202.5, 67.5, 247.5],
            6 => vec![22.5, 202.5, 67.5, 247.5, 112.5, 292.5, 157.5, 337.5,
                      0.0, 180.0, 45.0, 225.0, 90.0, 270.0, 135.0, 315.0],
            _ => vec![67.5, 247.5, 22.5, 202.5, 112.5, 292.5, 157.5, 337.5,
                      45.0, 225.0, 135.0, 315.0, 0.0, 180.0, 90.0, 270.0],
        };
        base
    }

    /// EVOLVED FUNCTION: Select direction angle for placement search
    #[inline]
    fn select_direction(&self, n: usize, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;
        let strategy = rng.gen::<f64>();

        if strategy < 0.45 {
            // Structured: evenly spaced with small jitter
            let base_idx = rng.gen_range(0..num_dirs);
            let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
            base + rng.gen_range(-0.05..0.05)
        } else if strategy < 0.65 {
            // Weighted random: favor corners and edges
            loop {
                let angle = rng.gen_range(0.0..2.0 * PI);
                let corner_weight = ((4.0 * angle).sin().abs() + (4.0 * angle).cos().abs()) / 2.0;
                let threshold = 0.15 + 0.12 * (1.0 - (n as f64 / 200.0).min(1.0));
                if rng.gen::<f64>() < corner_weight.max(threshold) {
                    return angle;
                }
            }
        } else if strategy < 0.85 {
            // Golden angle spiral
            let golden_angle = PI * (3.0 - (5.0_f64).sqrt());
            let base = (n as f64 * golden_angle) % (2.0 * PI);
            let offset = rng.gen_range(0..16) as f64 * PI / 8.0;
            (base + offset + rng.gen_range(-0.08..0.08)) % (2.0 * PI)
        } else {
            // Fibonacci lattice directions
            let idx = rng.gen_range(0..num_dirs);
            let golden_ratio = (1.0 + (5.0_f64).sqrt()) / 2.0;
            ((idx as f64 * 2.0 * PI / golden_ratio) % (2.0 * PI)) + rng.gen_range(-0.03..0.03)
        }
    }

    /// EVOLVED FUNCTION: Local search with pattern-aware moves
    fn local_search(&self, trees: &mut Vec<PlacedTree>, n: usize, pass: usize, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config: Vec<PlacedTree> = trees.clone();

        // PENALTY-FREE: Track pattern statistics (used for potential future adaptive tuning)
        let mut _pattern_success_count = 0usize;
        let mut _regular_success_count = 0usize;

        let temp_multiplier = match pass {
            0 => 1.0,
            1 => 0.4,
            _ => 0.2,
        };
        let mut temp = self.config.sa_initial_temp * temp_multiplier;

        let base_iterations = match pass {
            0 => self.config.sa_iterations + n * 200,
            1 => self.config.sa_iterations / 2 + n * 100,
            _ => self.config.sa_iterations / 4 + n * 50,
        };

        let mut iterations_without_improvement = 0;

        for _iter in 0..base_iterations {
            // PENALTY-FREE: Detect trees in penalty-free zones
            let pattern_trees = self.find_pattern_trees(trees);

            // Choose move strategy based on patterns
            let use_pattern_move = !pattern_trees.is_empty()
                && rng.gen::<f64>() < self.config.pattern_preservation_prob;

            let idx = if use_pattern_move {
                // Prefer moving non-pattern trees to preserve good arrangements
                self.select_non_pattern_tree(trees, &pattern_trees, rng)
            } else {
                self.select_tree_to_move(trees, rng)
            };

            let old_tree = trees[idx].clone();

            // PENALTY-FREE: Use pattern-preserving moves when appropriate
            let success = if use_pattern_move && pattern_trees.contains(&idx) {
                self.pattern_preserving_move(trees, idx, temp, rng)
            } else {
                self.sa_move(trees, idx, temp, rng)
            };

            if success {
                let new_side = compute_side_length(trees);
                let delta = new_side - current_side;

                if delta <= 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                    current_side = new_side;

                    // Track which strategy worked
                    if use_pattern_move {
                        _pattern_success_count += 1;
                    } else {
                        _regular_success_count += 1;
                    }

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

            // Restart mechanism
            if iterations_without_improvement >= self.config.restart_threshold {
                temp = self.config.reheat_temp * temp_multiplier;
                iterations_without_improvement = 0;
                *trees = best_config.clone();
                current_side = best_side;
            }

            temp = (temp * self.config.sa_cooling_rate).max(self.config.sa_min_temp);
        }

        if best_side < compute_side_length(trees) {
            *trees = best_config;
        }
    }

    /// PENALTY-FREE: Find trees that are in penalty-free patterns
    fn find_pattern_trees(&self, trees: &[PlacedTree]) -> Vec<usize> {
        let mut pattern_trees = Vec::new();

        for i in 0..trees.len() {
            let tree = &trees[i];
            let (tip_x, tip_y) = self.get_tip_position(tree);
            let (trunk_x, trunk_y) = self.get_trunk_center(tree);

            for (j, other) in trees.iter().enumerate() {
                if i == j { continue; }

                let (other_tip_x, other_tip_y) = self.get_tip_position(other);
                let (other_trunk_x, other_trunk_y) = self.get_trunk_center(other);

                // Check for tip-to-trunk
                let tip_to_trunk = ((tip_x - other_trunk_x).powi(2)
                                  + (tip_y - other_trunk_y).powi(2)).sqrt();
                let trunk_to_tip = ((trunk_x - other_tip_x).powi(2)
                                  + (trunk_y - other_tip_y).powi(2)).sqrt();

                if tip_to_trunk < self.config.interlocking_threshold
                   || trunk_to_tip < self.config.interlocking_threshold {
                    if !pattern_trees.contains(&i) {
                        pattern_trees.push(i);
                    }
                    break;
                }

                // Check anti-parallel
                let angle_diff = (tree.angle_deg - other.angle_deg).abs();
                let center_dist = ((tree.x - other.x).powi(2) + (tree.y - other.y).powi(2)).sqrt();
                if (angle_diff - 180.0).abs() < 30.0 && center_dist < 0.6 {
                    if !pattern_trees.contains(&i) {
                        pattern_trees.push(i);
                    }
                    break;
                }
            }
        }

        pattern_trees
    }

    /// PENALTY-FREE: Select a tree not in a pattern (to preserve patterns)
    fn select_non_pattern_tree(&self, trees: &[PlacedTree], pattern_trees: &[usize], rng: &mut impl Rng) -> usize {
        let non_pattern: Vec<usize> = (0..trees.len())
            .filter(|i| !pattern_trees.contains(i))
            .collect();

        if non_pattern.is_empty() {
            // All trees are in patterns, pick least important pattern tree
            rng.gen_range(0..trees.len())
        } else {
            non_pattern[rng.gen_range(0..non_pattern.len())]
        }
    }

    /// PENALTY-FREE: Move that tries to preserve the pattern relationship
    fn pattern_preserving_move(
        &self,
        trees: &mut [PlacedTree],
        idx: usize,
        temp: f64,
        rng: &mut impl Rng,
    ) -> bool {
        let old = &trees[idx];
        let old_x = old.x;
        let old_y = old.y;
        let old_angle = old.angle_deg;

        // Pattern-preserving moves: small adjustments that maintain relationships
        let move_type = rng.gen_range(0..6);

        match move_type {
            0 => {
                // Very small translation (preserve relative position)
                let scale = 0.015 * (0.3 + temp);
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            1 => {
                // Micro rotation (preserve interlocking)
                let delta = rng.gen_range(-5.0..5.0) * (0.5 + temp);
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            2 => {
                // Slide along pattern axis
                let (tip_x, tip_y) = self.get_tip_position(&trees[idx]);
                let dir_x = tip_x - old_x;
                let dir_y = tip_y - old_y;
                let len = (dir_x * dir_x + dir_y * dir_y).sqrt();
                if len > 0.01 {
                    let scale = 0.02 * (0.4 + temp);
                    let sign = if rng.gen() { 1.0 } else { -1.0 };
                    trees[idx] = PlacedTree::new(
                        old_x + sign * dir_x / len * scale,
                        old_y + sign * dir_y / len * scale,
                        old_angle
                    );
                } else {
                    return false;
                }
            }
            3 => {
                // Perpendicular slide
                let (tip_x, tip_y) = self.get_tip_position(&trees[idx]);
                let dir_x = tip_x - old_x;
                let dir_y = tip_y - old_y;
                let len = (dir_x * dir_x + dir_y * dir_y).sqrt();
                if len > 0.01 {
                    // Perpendicular direction
                    let perp_x = -dir_y / len;
                    let perp_y = dir_x / len;
                    let scale = 0.015 * (0.4 + temp);
                    let sign = if rng.gen() { 1.0 } else { -1.0 };
                    trees[idx] = PlacedTree::new(
                        old_x + sign * perp_x * scale,
                        old_y + sign * perp_y * scale,
                        old_angle
                    );
                } else {
                    return false;
                }
            }
            4 => {
                // Tiny combined move
                let scale = 0.01 * (0.3 + temp);
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                let dangle = rng.gen_range(-3.0..3.0);
                let new_angle = (old_angle + dangle).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, new_angle);
            }
            _ => {
                // Radial micro-adjustment
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.05 {
                    let delta_r = rng.gen_range(-0.02..0.02) * (0.5 + temp);
                    let new_mag = (mag + delta_r).max(0.0);
                    let scale = new_mag / mag;
                    trees[idx] = PlacedTree::new(old_x * scale, old_y * scale, old_angle);
                } else {
                    return false;
                }
            }
        }

        !has_overlap(trees, idx)
    }

    /// Select tree to move with preference for boundary trees (from Gen3)
    #[inline]
    fn select_tree_to_move(&self, trees: &[PlacedTree], rng: &mut impl Rng) -> usize {
        if trees.len() <= 2 || rng.gen::<f64>() < 0.6 {
            return rng.gen_range(0..trees.len());
        }

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for tree in trees.iter() {
            let (bx1, by1, bx2, by2) = tree.bounds();
            min_x = min_x.min(bx1);
            min_y = min_y.min(by1);
            max_x = max_x.max(bx2);
            max_y = max_y.max(by2);
        }

        let mut boundary_indices: Vec<usize> = Vec::new();
        let eps = 0.02;

        for (i, tree) in trees.iter().enumerate() {
            let (bx1, by1, bx2, by2) = tree.bounds();
            if (bx1 - min_x).abs() < eps || (bx2 - max_x).abs() < eps ||
               (by1 - min_y).abs() < eps || (by2 - max_y).abs() < eps {
                boundary_indices.push(i);
            }
        }

        if boundary_indices.is_empty() {
            rng.gen_range(0..trees.len())
        } else {
            boundary_indices[rng.gen_range(0..boundary_indices.len())]
        }
    }

    /// EVOLVED FUNCTION: SA move operator (from Gen3 with enhancements)
    #[inline]
    fn sa_move(
        &self,
        trees: &mut [PlacedTree],
        idx: usize,
        temp: f64,
        rng: &mut impl Rng,
    ) -> bool {
        let old = &trees[idx];
        let old_x = old.x;
        let old_y = old.y;
        let old_angle = old.angle_deg;

        // 14 move types (12 from Gen3 + 2 new pattern-seeking moves)
        let move_type = rng.gen_range(0..14);

        match move_type {
            0 => {
                // Small translation
                let scale = self.config.translation_scale * (0.15 + temp * 2.8);
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            1 => {
                // 90-degree rotation
                let new_angle = (old_angle + 90.0).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            2 => {
                // Fine rotation (22.5 degrees)
                let delta = if rng.gen() { self.config.rotation_granularity }
                            else { -self.config.rotation_granularity };
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            3 => {
                // Move toward center
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.04 {
                    let scale = self.config.center_pull_strength * (0.35 + temp * 1.8);
                    let dx = -old_x / mag * scale;
                    let dy = -old_y / mag * scale;
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                } else {
                    return false;
                }
            }
            4 => {
                // Translate + rotate combo
                let scale = self.config.translation_scale * 0.35;
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                let dangle = rng.gen_range(-45.0..45.0);
                let new_angle = (old_angle + dangle).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, new_angle);
            }
            5 => {
                // Polar move (radial in/out)
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.08 {
                    let delta_r = rng.gen_range(-0.07..0.07) * (1.0 + temp);
                    let new_mag = (mag + delta_r).max(0.0);
                    let scale = new_mag / mag;
                    trees[idx] = PlacedTree::new(old_x * scale, old_y * scale, old_angle);
                } else {
                    return false;
                }
            }
            6 => {
                // Angular orbit
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.08 {
                    let current_angle = old_y.atan2(old_x);
                    let delta_angle = rng.gen_range(-0.18..0.18) * (1.0 + temp);
                    let new_ang = current_angle + delta_angle;
                    trees[idx] = PlacedTree::new(mag * new_ang.cos(), mag * new_ang.sin(), old_angle);
                } else {
                    return false;
                }
            }
            7 => {
                // Very small nudge
                let scale = 0.012 * (0.4 + temp);
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            8 => {
                // 180-degree flip
                let new_angle = (old_angle + 180.0).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            9 => {
                // Directional slide
                let corner_idx = rng.gen_range(0..4);
                let (dir_x, dir_y) = match corner_idx {
                    0 => (-1.0, -1.0),
                    1 => (1.0, -1.0),
                    2 => (-1.0, 1.0),
                    _ => (1.0, 1.0),
                };
                let scale = 0.025 * (0.4 + temp * 1.6);
                let norm = (2.0_f64).sqrt();
                trees[idx] = PlacedTree::new(
                    old_x + dir_x * scale / norm,
                    old_y + dir_y * scale / norm,
                    old_angle
                );
            }
            10 => {
                // Micro rotation (11.25 degrees)
                let delta = if rng.gen() { 11.25 } else { -11.25 };
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            11 => {
                // Combined radial + angular
                let mag = (old_x * old_x + old_y * old_y).sqrt();
                if mag > 0.1 {
                    let current_angle = old_y.atan2(old_x);
                    let delta_r = rng.gen_range(-0.04..0.04) * (1.0 + temp);
                    let delta_angle = rng.gen_range(-0.1..0.1) * (1.0 + temp);
                    let new_mag = (mag + delta_r).max(0.0);
                    let new_ang = current_angle + delta_angle;
                    trees[idx] = PlacedTree::new(new_mag * new_ang.cos(), new_mag * new_ang.sin(), old_angle);
                } else {
                    return false;
                }
            }
            12 => {
                // PENALTY-FREE: Seek anti-parallel arrangement
                // Flip to opposite direction with small position adjustment
                let new_angle = (old_angle + 180.0).rem_euclid(360.0);
                let scale = 0.03 * (0.5 + temp);
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, new_angle);
            }
            _ => {
                // PENALTY-FREE: Move toward nearest neighbor (seek interlocking)
                let nearest = self.find_nearest_neighbor(trees, idx);
                if let Some(neighbor_idx) = nearest {
                    let neighbor = &trees[neighbor_idx];
                    let dx = neighbor.x - old_x;
                    let dy = neighbor.y - old_y;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist > 0.2 {
                        let scale = 0.04 * (0.4 + temp);
                        trees[idx] = PlacedTree::new(
                            old_x + dx / dist * scale,
                            old_y + dy / dist * scale,
                            old_angle
                        );
                    } else {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }

        !has_overlap(trees, idx)
    }

    /// Find nearest neighbor tree
    fn find_nearest_neighbor(&self, trees: &[PlacedTree], idx: usize) -> Option<usize> {
        let tree = &trees[idx];
        let mut min_dist = f64::INFINITY;
        let mut nearest = None;

        for (i, other) in trees.iter().enumerate() {
            if i == idx { continue; }
            let dist = ((tree.x - other.x).powi(2) + (tree.y - other.y).powi(2)).sqrt();
            if dist < min_dist {
                min_dist = dist;
                nearest = Some(i);
            }
        }

        nearest
    }

    /// Greedy compaction with pattern awareness
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

                // Try moving toward center
                let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                let center_x = (min_x + max_x) / 2.0;
                let center_y = (min_y + max_y) / 2.0;

                let dx = center_x - old_x;
                let dy = center_y - old_y;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist > 0.01 {
                    let step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002];

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

                // Try rotation adjustments
                if !improved && rng.gen::<f64>() < 0.3 {
                    for delta_angle in &[22.5, -22.5, 45.0, -45.0, 11.25, -11.25, 180.0] {
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
