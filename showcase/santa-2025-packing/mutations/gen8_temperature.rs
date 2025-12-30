//! Evolved Packing Algorithm - Generation 8 TEMPERATURE SCHEDULE
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
//! MUTATION STRATEGY: TEMPERATURE SCHEDULE (Gen8)
//! Advanced simulated annealing temperature management:
//!
//! Key improvements from Gen6:
//! - Reheat mechanism: when stuck for 2000 iterations, reheat to 0.3
//! - Multi-temperature stages: 0.6 -> 0.3 -> 0.1 -> 0.01
//! - Proportional temperature to problem size (larger n = higher initial temp)
//! - Final very cold stage for fine-tuning (temp 0.001)
//! - Adaptive cooling rate based on improvement frequency
//!
//! Target: Beat Gen6's 94.14 at n=200 with better temperature management

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

/// Evolved packing configuration
/// These parameters are tuned through evolution
pub struct EvolvedConfig {
    // Search parameters
    pub search_attempts: usize,
    pub direction_samples: usize,

    // Simulated annealing - Multi-stage temperature schedule
    pub sa_iterations: usize,
    // Stage temperatures: 0.6 -> 0.3 -> 0.1 -> 0.01 -> 0.001
    pub temp_stages: [f64; 5],
    pub stage_fractions: [f64; 5],  // Fraction of iterations for each stage
    pub sa_cooling_rate: f64,
    pub sa_min_temp: f64,

    // Reheat mechanism
    pub reheat_threshold: usize,    // Iterations without improvement before reheat
    pub reheat_temperature: f64,    // Temperature to reheat to
    pub max_reheats: usize,         // Maximum number of reheats per pass

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
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen8 TEMPERATURE SCHEDULE: Advanced temperature management
        Self {
            search_attempts: 280,
            direction_samples: 72,
            sa_iterations: 32000,            // More iterations for staged cooling
            // Multi-stage temperatures: 0.6 -> 0.3 -> 0.1 -> 0.01 -> 0.001
            temp_stages: [0.6, 0.3, 0.1, 0.01, 0.001],
            stage_fractions: [0.25, 0.25, 0.25, 0.15, 0.10],  // Time in each stage
            sa_cooling_rate: 0.99996,        // Slower cooling within stages
            sa_min_temp: 0.0005,             // Higher min for final stage to use
            // Reheat mechanism
            reheat_threshold: 2000,          // Reheat after 2000 iterations without improvement
            reheat_temperature: 0.3,         // Reheat to 0.3
            max_reheats: 3,                  // Up to 3 reheats per pass
            translation_scale: 0.055,
            rotation_granularity: 45.0,
            center_pull_strength: 0.07,
            sa_passes: 2,
            early_exit_threshold: 2500,      // Higher patience with reheating
            boundary_focus_prob: 0.85,
            // DENSITY parameters from Gen6
            density_grid_resolution: 20,
            gap_penalty_weight: 0.15,
            local_density_radius: 0.5,
            fill_move_prob: 0.15,
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

/// Temperature stage tracker
struct TemperatureScheduler {
    stages: [f64; 5],
    stage_fractions: [f64; 5],
    current_stage: usize,
    iterations_in_stage: usize,
    stage_iterations: usize,
    cooling_rate: f64,
    current_temp: f64,
    min_temp: f64,
}

impl TemperatureScheduler {
    fn new(stages: [f64; 5], fractions: [f64; 5], total_iterations: usize, cooling_rate: f64, min_temp: f64, size_factor: f64) -> Self {
        // Scale initial temperature by problem size
        let scaled_stages: [f64; 5] = [
            stages[0] * size_factor,
            stages[1] * size_factor,
            stages[2],
            stages[3],
            stages[4],
        ];

        let stage_iterations = (total_iterations as f64 * fractions[0]) as usize;

        Self {
            stages: scaled_stages,
            stage_fractions: fractions,
            current_stage: 0,
            iterations_in_stage: 0,
            stage_iterations,
            cooling_rate,
            current_temp: scaled_stages[0],
            min_temp,
        }
    }

    fn step(&mut self, total_iterations: usize) {
        self.iterations_in_stage += 1;

        // Check if we should move to next stage
        if self.current_stage < 4 && self.iterations_in_stage >= self.stage_iterations {
            self.current_stage += 1;
            self.iterations_in_stage = 0;
            self.current_temp = self.stages[self.current_stage];
            self.stage_iterations = (total_iterations as f64 * self.stage_fractions[self.current_stage]) as usize;
        } else {
            // Cool within stage toward next stage temperature
            let target = if self.current_stage < 4 {
                self.stages[self.current_stage + 1]
            } else {
                self.min_temp
            };
            self.current_temp = (self.current_temp * self.cooling_rate).max(target);
        }
    }

    fn reheat(&mut self, reheat_temp: f64) {
        // Reheat to specified temperature, but respect current stage minimum
        let stage_temp = self.stages[self.current_stage];
        self.current_temp = reheat_temp.min(stage_temp * 1.5);
    }

    fn temperature(&self) -> f64 {
        self.current_temp
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

            // Place new tree using density-aware heuristics
            let new_tree = self.find_placement(&trees, n, max_n, &mut rng);
            trees.push(new_tree);

            // Run SA passes with advanced temperature scheduling
            for pass in 0..self.config.sa_passes {
                self.local_search(&mut trees, n, max_n, pass, &mut rng);
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
        let gaps = self.find_gaps(existing, min_x, min_y, max_x, max_y);

        for attempt in 0..self.config.search_attempts {
            // DENSITY: Sometimes target gaps directly
            let dir = if !gaps.is_empty() && attempt % 4 == 0 {
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
    /// DENSITY MAXIMIZATION: Score based on gap filling and local density
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

        // DENSITY: Calculate local density around the new tree
        let tree_cx = (tree_min_x + tree_max_x) / 2.0;
        let tree_cy = (tree_min_y + tree_max_y) / 2.0;
        let local_density = self.calculate_local_density(tree_cx, tree_cy, existing);

        // Reward high local density (tree is filling a gap)
        let density_bonus = -self.config.gap_penalty_weight * local_density;

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
        let gap_penalty = self.estimate_unusable_gap(tree, existing) * self.config.gap_penalty_weight;

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

    /// EVOLVED FUNCTION: Local search with advanced temperature scheduling
    /// TEMPERATURE SCHEDULE: Multi-stage with reheat mechanism
    fn local_search(&self, trees: &mut Vec<PlacedTree>, n: usize, max_n: usize, pass: usize, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config: Vec<PlacedTree> = trees.clone();

        // TEMPERATURE SCHEDULE: Scale initial temperature by problem size
        // Larger problems need more exploration
        let size_factor = 1.0 + 0.3 * ((n as f64) / (max_n as f64)).sqrt();

        let pass_multiplier = match pass {
            0 => 1.0,
            _ => 0.4,  // Lower temperature for refinement passes
        };

        let base_iterations = match pass {
            0 => self.config.sa_iterations + n * 120,
            _ => self.config.sa_iterations / 2 + n * 60,
        };

        // Initialize temperature scheduler with proportional scaling
        let mut scheduler = TemperatureScheduler::new(
            self.config.temp_stages,
            self.config.stage_fractions,
            base_iterations,
            self.config.sa_cooling_rate,
            self.config.sa_min_temp,
            size_factor * pass_multiplier,
        );

        let mut iterations_without_improvement = 0;
        let mut reheats_used = 0;
        #[allow(unused_variables)]
        let mut total_improvements = 0;

        // Cache boundary info
        let mut boundary_cache_iter = 0;
        let mut boundary_info: Vec<(usize, BoundaryEdge)> = Vec::new();

        for iter in 0..base_iterations {
            // TEMPERATURE SCHEDULE: Reheat mechanism
            if iterations_without_improvement >= self.config.reheat_threshold
               && reheats_used < self.config.max_reheats {
                scheduler.reheat(self.config.reheat_temperature * pass_multiplier);
                reheats_used += 1;
                iterations_without_improvement = 0;
            }

            // Check early exit
            if iterations_without_improvement >= self.config.early_exit_threshold {
                break;
            }

            // Update boundary cache every 350 iterations
            if iter == 0 || iter - boundary_cache_iter >= 350 {
                boundary_info = self.find_boundary_trees_with_edges(trees);
                boundary_cache_iter = iter;
            }

            let temp = scheduler.temperature();

            // Choose between boundary optimization and gap-filling
            let do_fill_move = rng.gen::<f64>() < self.config.fill_move_prob;

            let (idx, edge) = if do_fill_move {
                let interior_trees: Vec<usize> = (0..trees.len())
                    .filter(|&i| !boundary_info.iter().any(|(bi, _)| *bi == i))
                    .collect();

                if !interior_trees.is_empty() && rng.gen::<f64>() < 0.5 {
                    (interior_trees[rng.gen_range(0..interior_trees.len())], BoundaryEdge::None)
                } else {
                    if !boundary_info.is_empty() {
                        let bi = &boundary_info[rng.gen_range(0..boundary_info.len())];
                        (bi.0, bi.1)
                    } else {
                        (rng.gen_range(0..trees.len()), BoundaryEdge::None)
                    }
                }
            } else if !boundary_info.is_empty() && rng.gen::<f64>() < self.config.boundary_focus_prob {
                let bi = &boundary_info[rng.gen_range(0..boundary_info.len())];
                (bi.0, bi.1)
            } else {
                (rng.gen_range(0..trees.len()), BoundaryEdge::None)
            };

            let old_tree = trees[idx].clone();

            let success = self.temperature_aware_move(trees, idx, temp, edge, do_fill_move, rng);

            if success {
                let new_side = compute_side_length(trees);
                let delta = new_side - current_side;

                if delta <= 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                    current_side = new_side;
                    if current_side < best_side {
                        best_side = current_side;
                        best_config = trees.clone();
                        iterations_without_improvement = 0;
                        total_improvements += 1;
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

            // Step the temperature scheduler
            scheduler.step(base_iterations);
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

    /// TEMPERATURE SCHEDULE: Move operator with temperature awareness
    /// At high temps: more exploration, larger moves
    /// At low temps: fine-tuning, smaller precise moves
    #[inline]
    fn temperature_aware_move(
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

        // TEMPERATURE SCHEDULE: Scale moves by temperature
        // High temp = large exploration moves, low temp = fine-tuning
        let temp_scale = if temp > 0.1 {
            1.0 + temp * 2.0  // Aggressive at high temp
        } else if temp > 0.01 {
            0.5 + temp * 5.0  // Moderate at medium temp
        } else {
            0.2 + temp * 10.0  // Fine-tuning at low temp
        };

        let scale = self.config.translation_scale * temp_scale;

        if is_fill_move {
            let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
            let bbox_cx = (min_x + max_x) / 2.0;
            let bbox_cy = (min_y + max_y) / 2.0;

            let move_type = rng.gen_range(0..4);
            match move_type {
                0 => {
                    let dx = (bbox_cx - old_x) * 0.1 * temp_scale;
                    let dy = (bbox_cy - old_y) * 0.1 * temp_scale;
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
                1 => {
                    let dx = rng.gen_range(-scale * 0.4..scale * 0.4);
                    let dy = rng.gen_range(-scale * 0.4..scale * 0.4);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
                2 => {
                    // TEMPERATURE SCHEDULE: At low temp, use smaller angle increments
                    let angles = if temp > 0.05 {
                        [45.0, 90.0, -45.0, -90.0, 30.0, -30.0]
                    } else {
                        [15.0, -15.0, 10.0, -10.0, 5.0, -5.0]
                    };
                    let delta = angles[rng.gen_range(0..angles.len())];
                    let new_angle = (old_angle + delta).rem_euclid(360.0);
                    trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
                }
                _ => {
                    let gaps = self.find_gaps(trees, min_x, min_y, max_x, max_y);
                    if !gaps.is_empty() {
                        let gap = &gaps[rng.gen_range(0..gaps.len())];
                        let gap_cx = (gap.0 + gap.2) / 2.0;
                        let gap_cy = (gap.1 + gap.3) / 2.0;
                        let dx = (gap_cx - old_x) * 0.05 * temp_scale;
                        let dy = (gap_cy - old_y) * 0.05 * temp_scale;
                        trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                    } else {
                        return false;
                    }
                }
            }
        } else {
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
                    // TEMPERATURE SCHEDULE: At low temp, use smaller angle increments
                    let angles = if temp > 0.05 {
                        [45.0, 90.0, -45.0, -90.0]
                    } else {
                        [15.0, -15.0, 10.0, -10.0]
                    };
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

                    let dx = (bbox_cx - old_x) * self.config.center_pull_strength * temp_scale;
                    let dy = (bbox_cy - old_y) * self.config.center_pull_strength * temp_scale;
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
                        let delta_r = rng.gen_range(-0.06..0.06) * temp_scale;
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
