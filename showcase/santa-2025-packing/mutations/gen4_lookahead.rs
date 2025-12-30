//! Evolved Packing Algorithm - Generation 4 LOOKAHEAD
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
//! MUTATION STRATEGY: LOOKAHEAD PLACEMENT (Gen4)
//! Building on Gen3 champion (101.90) with predictive placement:
//!
//! Hypothesis: Greedy tree-by-tree placement misses globally better configurations.
//! Solution: When placing tree n, also simulate placing n+1 and n+2 to score the
//! combined quality of future placements.
//!
//! Key changes from Gen3:
//! - When placing tree n, simulate placing n+1 and n+2 as well
//! - Score based on combined quality of all lookahead placements
//! - Choose placement for n that leads to best future packing
//! - Reduce computation for large n where lookahead is expensive
//! - Adaptive lookahead depth: 2 for small n, 1 for medium n, 0 for large n
//!
//! Goal: Make globally-aware placement decisions
//! Target: Beat Gen3's 101.90 score

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

    // Restart mechanism
    pub restart_threshold: usize,
    pub reheat_temp: f64,

    // Greedy compaction
    pub compaction_iterations: usize,

    // LOOKAHEAD: New parameters for lookahead search
    pub lookahead_depth_small_n: usize,    // Depth for small n (< 50)
    pub lookahead_depth_medium_n: usize,   // Depth for medium n (50-100)
    pub lookahead_depth_large_n: usize,    // Depth for large n (> 100)
    pub lookahead_search_attempts: usize,  // Reduced attempts for lookahead
    pub lookahead_direction_samples: usize, // Reduced samples for lookahead
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        Self {
            // Same base parameters as Gen3 Ultra
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

            // LOOKAHEAD: New parameters
            lookahead_depth_small_n: 2,     // Look 2 trees ahead for n < 50
            lookahead_depth_medium_n: 1,    // Look 1 tree ahead for 50 <= n < 100
            lookahead_depth_large_n: 0,     // No lookahead for n >= 100
            lookahead_search_attempts: 50,  // Reduced from 400 for speed
            lookahead_direction_samples: 24, // Reduced from 96 for speed
        }
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

            // LOOKAHEAD: Place new tree using lookahead-aware heuristics
            let new_tree = self.find_placement_with_lookahead(&trees, n, max_n, &mut rng);
            trees.push(new_tree);

            // Run triple SA passes for maximum optimization
            for pass in 0..self.config.sa_passes {
                self.local_search(&mut trees, n, pass, &mut rng);
            }

            // Final greedy compaction phase
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

    /// EVOLVED FUNCTION: Find best placement with lookahead simulation
    /// This is the key innovation of Gen4
    fn find_placement_with_lookahead(
        &self,
        existing: &[PlacedTree],
        n: usize,
        max_n: usize,
        rng: &mut impl Rng,
    ) -> PlacedTree {
        if existing.is_empty() {
            // First tree: place at origin with optimal rotation
            return PlacedTree::new(0.0, 0.0, 90.0);
        }

        let lookahead_depth = self.get_lookahead_depth(n, max_n);

        if lookahead_depth == 0 {
            // No lookahead: fall back to standard placement
            return self.find_placement(existing, n, max_n, rng);
        }

        let mut best_tree = PlacedTree::new(0.0, 0.0, 90.0);
        let mut best_score = f64::INFINITY;

        let angles = self.select_angles(n);

        // Use full search attempts for main tree
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
                    // LOOKAHEAD: Score includes simulation of future placements
                    let score = self.score_with_lookahead(
                        &candidate,
                        existing,
                        n,
                        max_n,
                        lookahead_depth,
                        rng,
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

    /// Score a placement by simulating future tree placements
    fn score_with_lookahead(
        &self,
        tree: &PlacedTree,
        existing: &[PlacedTree],
        n: usize,
        max_n: usize,
        depth: usize,
        rng: &mut impl Rng,
    ) -> f64 {
        // Build a temporary configuration with the candidate tree
        let mut temp_trees: Vec<PlacedTree> = existing.iter().cloned().collect();
        temp_trees.push(tree.clone());

        // Base score for current placement
        let mut total_score = self.placement_score(tree, existing, n);

        // Simulate future placements with reduced search
        for future_step in 1..=depth {
            let future_n = n + future_step;
            if future_n > max_n {
                break;
            }

            // Find a reasonable placement for the future tree (reduced search)
            let future_tree = self.find_placement_quick(&temp_trees, future_n, max_n, rng);

            // Add penalty based on how far out the future tree is placed
            // (further = worse packing potential)
            let future_score = self.placement_score(&future_tree, &temp_trees, future_n);

            // Weight future scores less than current (discounting)
            let discount = match future_step {
                1 => 0.7,  // First lookahead tree: 70% weight
                2 => 0.4,  // Second lookahead tree: 40% weight
                _ => 0.2,  // Further: 20% weight
            };

            total_score += future_score * discount;
            temp_trees.push(future_tree);
        }

        total_score
    }

    /// Quick placement search for lookahead simulation (reduced computation)
    fn find_placement_quick(
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

        // Use reduced angles for speed
        let angles = vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0];

        for _ in 0..self.config.lookahead_search_attempts {
            // Quick direction selection (simpler than full version)
            let base_idx = rng.gen_range(0..self.config.lookahead_direction_samples);
            let dir = (base_idx as f64 / self.config.lookahead_direction_samples as f64) * 2.0 * PI;

            let vx = dir.cos();
            let vy = dir.sin();

            for &tree_angle in &angles {
                // Binary search with coarser precision
                let mut low = 0.0;
                let mut high = 12.0;

                while high - low > 0.001 {  // 10x coarser than main search
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

    /// Standard placement without lookahead (used for large n)
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
    /// Key evolution target - determines placement quality
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
        let balance_weight = 0.12 + 0.06 * (1.0 - (n as f64 / 200.0).min(1.0));
        let balance_penalty = (width - height).abs() * balance_weight;

        // Tertiary: slight preference for compact center
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.010 / (n as f64).sqrt();

        // Enhanced density heuristic
        let area = width * height;
        let density_bonus = if area > 0.0 {
            -0.008 * (n as f64 / area).min(2.5)
        } else {
            0.0
        };

        // Perimeter minimization bonus
        let perimeter_bonus = -0.002 * (2.0 * (width + height)) / (n as f64).sqrt();

        // LOOKAHEAD: Additional heuristic - penalize placements that leave awkward gaps
        // This encourages placements that leave room for future trees
        let gap_penalty = self.compute_gap_penalty(tree, existing, n);

        side_score + balance_penalty + center_penalty + density_bonus + perimeter_bonus + gap_penalty
    }

    /// Compute penalty for awkward gaps that might be hard to fill later
    #[inline]
    fn compute_gap_penalty(&self, tree: &PlacedTree, existing: &[PlacedTree], n: usize) -> f64 {
        if existing.is_empty() || n > 100 {
            return 0.0;
        }

        // Find minimum distance to existing trees
        let mut min_dist = f64::INFINITY;
        for other in existing {
            let dx = tree.x - other.x;
            let dy = tree.y - other.y;
            let dist = (dx * dx + dy * dy).sqrt();
            min_dist = min_dist.min(dist);
        }

        // Penalize placements that are too far from any existing tree
        // (leaves gaps that are hard to fill)
        let gap_threshold = 1.5;  // Trees should ideally be closer than this
        if min_dist > gap_threshold {
            0.03 * (min_dist - gap_threshold)
        } else {
            0.0
        }
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    /// 16 angles (every 22.5 degrees) for finest granularity
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
        let base = match n % 8 {
            0 => vec![0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0,
                      22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5],
            1 => vec![90.0, 270.0, 0.0, 180.0, 135.0, 315.0, 45.0, 225.0,
                      67.5, 112.5, 247.5, 292.5, 22.5, 157.5, 202.5, 337.5],
            2 => vec![180.0, 0.0, 270.0, 90.0, 225.0, 45.0, 315.0, 135.0,
                      157.5, 202.5, 337.5, 22.5, 67.5, 112.5, 247.5, 292.5],
            3 => vec![270.0, 90.0, 180.0, 0.0, 315.0, 135.0, 225.0, 45.0,
                      247.5, 292.5, 67.5, 112.5, 157.5, 202.5, 337.5, 22.5],
            4 => vec![45.0, 225.0, 135.0, 315.0, 0.0, 90.0, 180.0, 270.0,
                      22.5, 67.5, 202.5, 247.5, 112.5, 157.5, 292.5, 337.5],
            5 => vec![135.0, 315.0, 45.0, 225.0, 90.0, 270.0, 0.0, 180.0,
                      112.5, 157.5, 292.5, 337.5, 22.5, 67.5, 202.5, 247.5],
            6 => vec![22.5, 202.5, 67.5, 247.5, 112.5, 292.5, 157.5, 337.5,
                      0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
            _ => vec![67.5, 247.5, 22.5, 202.5, 112.5, 292.5, 157.5, 337.5,
                      45.0, 135.0, 225.0, 315.0, 0.0, 90.0, 180.0, 270.0],
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
            // Golden angle spiral for good coverage
            let golden_angle = PI * (3.0 - (5.0_f64).sqrt());
            let base = (n as f64 * golden_angle) % (2.0 * PI);
            let offset = rng.gen_range(0..16) as f64 * PI / 8.0;
            (base + offset + rng.gen_range(-0.08..0.08)) % (2.0 * PI)
        } else {
            // Fibonacci lattice directions for uniform coverage
            let idx = rng.gen_range(0..num_dirs);
            let golden_ratio = (1.0 + (5.0_f64).sqrt()) / 2.0;
            ((idx as f64 * 2.0 * PI / golden_ratio) % (2.0 * PI)) + rng.gen_range(-0.03..0.03)
        }
    }

    /// EVOLVED FUNCTION: Local search with simulated annealing
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
            _ => 0.2,
        };
        let mut temp = self.config.sa_initial_temp * temp_multiplier;

        let base_iterations = match pass {
            0 => self.config.sa_iterations + n * 200,
            1 => self.config.sa_iterations / 2 + n * 100,
            _ => self.config.sa_iterations / 4 + n * 50,
        };

        let mut iterations_without_improvement = 0;

        for iter in 0..base_iterations {
            let idx = self.select_tree_to_move(trees, rng);
            let old_tree = trees[idx].clone();

            let success = self.sa_move(trees, idx, temp, iter, rng);

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

            // Restart mechanism - reheat if stuck
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

    /// Greedy compaction phase after SA
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

                if !improved && rng.gen::<f64>() < 0.3 {
                    for delta_angle in &[22.5, -22.5, 45.0, -45.0, 11.25, -11.25] {
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

    /// Select tree to move with preference for boundary trees
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

    /// EVOLVED FUNCTION: SA move operator
    /// 12 move types with fine control
    #[inline]
    fn sa_move(
        &self,
        trees: &mut [PlacedTree],
        idx: usize,
        temp: f64,
        _iter: usize,
        rng: &mut impl Rng,
    ) -> bool {
        let old = &trees[idx];
        let old_x = old.x;
        let old_y = old.y;
        let old_angle = old.angle_deg;

        let move_type = rng.gen_range(0..12);

        match move_type {
            0 => {
                // Small translation (temperature-scaled)
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
                // Angular orbit (move around center)
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
                // Very small nudge for fine-tuning
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
                // Directional slide (move toward one corner)
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
                // Micro rotation (11.25 degrees = half of 22.5)
                let delta = if rng.gen() { 11.25 } else { -11.25 };
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            _ => {
                // Combined radial + angular move
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

        // Small n should use depth 2
        assert_eq!(packer.get_lookahead_depth(10, 200), 2);
        assert_eq!(packer.get_lookahead_depth(49, 200), 2);

        // Medium n should use depth 1
        assert_eq!(packer.get_lookahead_depth(50, 200), 1);
        assert_eq!(packer.get_lookahead_depth(99, 200), 1);

        // Large n should use depth 0
        assert_eq!(packer.get_lookahead_depth(100, 200), 0);
        assert_eq!(packer.get_lookahead_depth(150, 200), 0);

        // Near end should be limited by remaining trees
        assert_eq!(packer.get_lookahead_depth(199, 200), 1);
        assert_eq!(packer.get_lookahead_depth(200, 200), 0);
    }
}
