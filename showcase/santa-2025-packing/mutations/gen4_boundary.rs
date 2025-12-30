//! Evolved Packing Algorithm - Generation 4 BOUNDARY SURGERY
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
//! MUTATION STRATEGY: BOUNDARY SURGERY (Gen4)
//! Building on Gen3 champion (score 101.90) with surgical boundary optimization:
//!
//! Key innovations:
//! 1. Identify which trees define each edge of the bounding box (min_x, max_x, min_y, max_y)
//! 2. Use specialized moves that try to pull boundary trees inward
//! 3. Allow small overlaps during boundary surgery, then repair via repair phase
//! 4. Iterative: shrink bounding box, re-optimize interior, repeat
//!
//! New phases:
//! - boundary_surgery(): Aggressive moves on boundary-defining trees
//! - repair_overlaps(): Fix any overlaps created during surgery
//! - iterative_shrink(): Alternate between boundary surgery and interior optimization
//!
//! Goal: Surgically reduce bounding box by targeting the specific trees that define it
//! Target: Break below 101 score at n=200

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

    // BOUNDARY SURGERY: New parameters
    pub boundary_surgery_passes: usize,      // Number of boundary surgery iterations
    pub boundary_move_attempts: usize,       // Attempts per boundary tree
    pub boundary_inward_steps: Vec<f64>,     // Step sizes for inward moves
    pub max_overlap_tolerance: f64,          // Max overlap to allow during surgery
    pub repair_iterations: usize,            // Iterations to repair overlaps
    pub shrink_optimize_cycles: usize,       // Shrink-optimize cycle count
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        Self {
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

            // BOUNDARY SURGERY parameters
            boundary_surgery_passes: 5,
            boundary_move_attempts: 50,
            boundary_inward_steps: vec![0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001],
            max_overlap_tolerance: 0.03,  // Allow small overlaps during surgery
            repair_iterations: 500,
            shrink_optimize_cycles: 3,
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

/// Which edge a tree defines on the bounding box
#[derive(Clone, Copy, Debug, PartialEq)]
enum BoundaryEdge {
    MinX,
    MaxX,
    MinY,
    MaxY,
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

            // Run SA passes for optimization
            for pass in 0..self.config.sa_passes {
                self.local_search(&mut trees, n, pass, &mut rng);
            }

            // Greedy compaction phase
            self.greedy_compaction(&mut trees, &mut rng);

            // BOUNDARY SURGERY: Iterative shrink-optimize cycles
            for _cycle in 0..self.config.shrink_optimize_cycles {
                let old_side = compute_side_length(&trees);

                // Perform boundary surgery
                self.boundary_surgery(&mut trees, &mut rng);

                // Repair any overlaps from surgery
                self.repair_overlaps(&mut trees, &mut rng);

                // Quick interior optimization after surgery
                self.interior_optimization(&mut trees, &mut rng);

                let new_side = compute_side_length(&trees);

                // Stop if no improvement
                if new_side >= old_side - 1e-9 {
                    break;
                }
            }

            // Final greedy compaction
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

        // Secondary: prefer balanced aspect ratio
        let balance_weight = 0.12 + 0.06 * (1.0 - (n as f64 / 200.0).min(1.0));
        let balance_penalty = (width - height).abs() * balance_weight;

        // Tertiary: slight preference for compact center
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.010 / (n as f64).sqrt();

        // Density heuristic
        let area = width * height;
        let density_bonus = if area > 0.0 {
            -0.008 * (n as f64 / area).min(2.5)
        } else {
            0.0
        };

        // Perimeter minimization bonus
        let perimeter_bonus = -0.002 * (2.0 * (width + height)) / (n as f64).sqrt();

        // BOUNDARY SURGERY: Bonus for not extending bounding box
        let boundary_penalty = if n > 1 {
            let (ex_min_x, ex_min_y, ex_max_x, ex_max_y) = compute_bounds_slice(existing);
            let extends_x = (min_x < ex_min_x) as i32 + (max_x > ex_max_x) as i32;
            let extends_y = (min_y < ex_min_y) as i32 + (max_y > ex_max_y) as i32;
            (extends_x + extends_y) as f64 * 0.015
        } else {
            0.0
        };

        side_score + balance_penalty + center_penalty + density_bonus + perimeter_bonus + boundary_penalty
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
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
            let base_idx = rng.gen_range(0..num_dirs);
            let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
            base + rng.gen_range(-0.05..0.05)
        } else if strategy < 0.65 {
            loop {
                let angle = rng.gen_range(0.0..2.0 * PI);
                let corner_weight = ((4.0 * angle).sin().abs() + (4.0 * angle).cos().abs()) / 2.0;
                let threshold = 0.15 + 0.12 * (1.0 - (n as f64 / 200.0).min(1.0));
                if rng.gen::<f64>() < corner_weight.max(threshold) {
                    return angle;
                }
            }
        } else if strategy < 0.85 {
            let golden_angle = PI * (3.0 - (5.0_f64).sqrt());
            let base = (n as f64 * golden_angle) % (2.0 * PI);
            let offset = rng.gen_range(0..16) as f64 * PI / 8.0;
            (base + offset + rng.gen_range(-0.08..0.08)) % (2.0 * PI)
        } else {
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

    /// BOUNDARY SURGERY: Identify trees that define each edge of the bounding box
    fn identify_boundary_trees(&self, trees: &[PlacedTree]) -> Vec<(usize, BoundaryEdge)> {
        if trees.is_empty() {
            return Vec::new();
        }

        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
        let eps = 0.001;  // Tight epsilon for boundary detection

        let mut boundary_trees = Vec::new();

        for (i, tree) in trees.iter().enumerate() {
            let (bx1, by1, bx2, by2) = tree.bounds();

            // Check which edges this tree defines
            if (bx1 - min_x).abs() < eps {
                boundary_trees.push((i, BoundaryEdge::MinX));
            }
            if (bx2 - max_x).abs() < eps {
                boundary_trees.push((i, BoundaryEdge::MaxX));
            }
            if (by1 - min_y).abs() < eps {
                boundary_trees.push((i, BoundaryEdge::MinY));
            }
            if (by2 - max_y).abs() < eps {
                boundary_trees.push((i, BoundaryEdge::MaxY));
            }
        }

        boundary_trees
    }

    /// BOUNDARY SURGERY: Main surgical optimization on boundary trees
    fn boundary_surgery(&self, trees: &mut Vec<PlacedTree>, rng: &mut impl Rng) {
        if trees.len() <= 2 {
            return;
        }

        for _pass in 0..self.config.boundary_surgery_passes {
            let boundary_trees = self.identify_boundary_trees(trees);

            if boundary_trees.is_empty() {
                break;
            }

            let mut best_side = compute_side_length(trees);
            let mut any_improved = false;

            // Try to move each boundary tree inward
            for (idx, edge) in &boundary_trees {
                let old_tree = trees[*idx].clone();

                // Determine inward direction based on which edge this tree defines
                let (inward_dx, inward_dy) = match edge {
                    BoundaryEdge::MinX => (1.0, 0.0),   // Move right
                    BoundaryEdge::MaxX => (-1.0, 0.0),  // Move left
                    BoundaryEdge::MinY => (0.0, 1.0),   // Move up
                    BoundaryEdge::MaxY => (0.0, -1.0),  // Move down
                };

                // Try various step sizes
                for &step in &self.config.boundary_inward_steps {
                    let new_x = old_tree.x + inward_dx * step;
                    let new_y = old_tree.y + inward_dy * step;

                    // Try with current angle and nearby rotations
                    for angle_delta in &[0.0, 22.5, -22.5, 45.0, -45.0, 11.25, -11.25] {
                        let new_angle = (old_tree.angle_deg + angle_delta).rem_euclid(360.0);
                        trees[*idx] = PlacedTree::new(new_x, new_y, new_angle);

                        // Check for overlaps - allow small ones for later repair
                        let overlap_amount = compute_max_overlap(trees, *idx);

                        if overlap_amount <= self.config.max_overlap_tolerance {
                            let new_side = compute_side_length(trees);

                            // Accept if improves bounding box (even with small overlap)
                            if new_side < best_side - 1e-9 {
                                best_side = new_side;
                                any_improved = true;
                                break;  // Keep this move
                            }
                        }

                        trees[*idx] = old_tree.clone();
                    }
                }

                // BOUNDARY SURGERY: Try diagonal moves (corner trees)
                let diagonal_dirs = [
                    (inward_dx + 0.5 * inward_dy.signum(), inward_dy + 0.5 * inward_dx.signum()),
                    (inward_dx - 0.5 * inward_dy.signum(), inward_dy - 0.5 * inward_dx.signum()),
                ];

                for (diag_dx, diag_dy) in &diagonal_dirs {
                    let norm = (diag_dx * diag_dx + diag_dy * diag_dy).sqrt();
                    if norm < 0.01 {
                        continue;
                    }

                    for &step in &self.config.boundary_inward_steps[..4] {
                        let new_x = old_tree.x + diag_dx / norm * step;
                        let new_y = old_tree.y + diag_dy / norm * step;

                        trees[*idx] = PlacedTree::new(new_x, new_y, old_tree.angle_deg);

                        let overlap_amount = compute_max_overlap(trees, *idx);

                        if overlap_amount <= self.config.max_overlap_tolerance {
                            let new_side = compute_side_length(trees);
                            if new_side < best_side - 1e-9 {
                                best_side = new_side;
                                any_improved = true;
                                break;
                            }
                        }

                        trees[*idx] = old_tree.clone();
                    }
                }

                // BOUNDARY SURGERY: Try rotation-only moves to reduce extent
                for angle_delta in &[90.0, -90.0, 180.0, 45.0, -45.0, 135.0, -135.0] {
                    let new_angle = (old_tree.angle_deg + angle_delta).rem_euclid(360.0);
                    trees[*idx] = PlacedTree::new(old_tree.x, old_tree.y, new_angle);

                    if !has_overlap(trees, *idx) {
                        let new_side = compute_side_length(trees);
                        if new_side < best_side - 1e-9 {
                            best_side = new_side;
                            any_improved = true;
                            continue;  // Keep this rotation
                        }
                    }

                    trees[*idx] = old_tree.clone();
                }
            }

            // Stop if no improvement in this pass
            if !any_improved {
                break;
            }
        }
    }

    /// BOUNDARY SURGERY: Repair any overlaps created during surgery
    fn repair_overlaps(&self, trees: &mut Vec<PlacedTree>, rng: &mut impl Rng) {
        let mut iterations = 0;

        while iterations < self.config.repair_iterations {
            iterations += 1;

            // Find all overlapping pairs
            let mut overlapping_indices: Vec<usize> = Vec::new();
            for i in 0..trees.len() {
                if has_overlap(trees, i) {
                    overlapping_indices.push(i);
                }
            }

            if overlapping_indices.is_empty() {
                break;  // No overlaps to repair
            }

            // Try to fix each overlapping tree
            for &idx in &overlapping_indices {
                let old_tree = trees[idx].clone();

                // Try small moves to resolve overlap
                let directions = [
                    (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
                    (0.707, 0.707), (-0.707, 0.707), (0.707, -0.707), (-0.707, -0.707),
                ];

                let step_sizes = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1];

                let mut fixed = false;
                'outer: for &step in &step_sizes {
                    for &(dx, dy) in &directions {
                        let new_x = old_tree.x + dx * step;
                        let new_y = old_tree.y + dy * step;
                        trees[idx] = PlacedTree::new(new_x, new_y, old_tree.angle_deg);

                        if !has_overlap(trees, idx) {
                            fixed = true;
                            break 'outer;
                        }
                    }

                    // Also try rotations
                    for angle_delta in &[22.5, -22.5, 45.0, -45.0, 90.0, -90.0] {
                        let new_angle = (old_tree.angle_deg + angle_delta).rem_euclid(360.0);
                        trees[idx] = PlacedTree::new(old_tree.x, old_tree.y, new_angle);

                        if !has_overlap(trees, idx) {
                            fixed = true;
                            break 'outer;
                        }
                    }
                }

                if !fixed {
                    // Try combined translation + rotation
                    for &step in &step_sizes {
                        for &(dx, dy) in &directions {
                            for angle_delta in &[22.5, -22.5, 45.0, -45.0] {
                                let new_x = old_tree.x + dx * step;
                                let new_y = old_tree.y + dy * step;
                                let new_angle = (old_tree.angle_deg + angle_delta).rem_euclid(360.0);
                                trees[idx] = PlacedTree::new(new_x, new_y, new_angle);

                                if !has_overlap(trees, idx) {
                                    fixed = true;
                                    break;
                                }
                            }
                            if fixed {
                                break;
                            }
                        }
                        if fixed {
                            break;
                        }
                    }
                }

                if !fixed {
                    // Revert to original if can't fix
                    trees[idx] = old_tree;
                }
            }
        }
    }

    /// BOUNDARY SURGERY: Quick interior optimization after surgery
    fn interior_optimization(&self, trees: &mut Vec<PlacedTree>, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut best_side = compute_side_length(trees);

        // Fewer iterations than full SA, focused on interior trees
        for _ in 0..1000 {
            // Pick a non-boundary tree preferentially
            let boundary_trees = self.identify_boundary_trees(trees);
            let boundary_indices: Vec<usize> = boundary_trees.iter().map(|(i, _)| *i).collect();

            let idx = if rng.gen::<f64>() < 0.7 {
                // Prefer interior trees
                let mut attempts = 0;
                loop {
                    let candidate = rng.gen_range(0..trees.len());
                    if !boundary_indices.contains(&candidate) || attempts > 10 {
                        break candidate;
                    }
                    attempts += 1;
                }
            } else {
                rng.gen_range(0..trees.len())
            };

            let old_tree = trees[idx].clone();

            // Small adjustments
            let move_type = rng.gen_range(0..4);
            match move_type {
                0 => {
                    // Small translation
                    let scale = 0.03;
                    let dx = rng.gen_range(-scale..scale);
                    let dy = rng.gen_range(-scale..scale);
                    trees[idx] = PlacedTree::new(old_tree.x + dx, old_tree.y + dy, old_tree.angle_deg);
                }
                1 => {
                    // Small rotation
                    let delta = if rng.gen() { 22.5 } else { -22.5 };
                    let new_angle = (old_tree.angle_deg + delta).rem_euclid(360.0);
                    trees[idx] = PlacedTree::new(old_tree.x, old_tree.y, new_angle);
                }
                2 => {
                    // Move toward center
                    let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                    let center_x = (min_x + max_x) / 2.0;
                    let center_y = (min_y + max_y) / 2.0;
                    let dx = center_x - old_tree.x;
                    let dy = center_y - old_tree.y;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist > 0.01 {
                        let step = 0.02;
                        trees[idx] = PlacedTree::new(
                            old_tree.x + dx / dist * step,
                            old_tree.y + dy / dist * step,
                            old_tree.angle_deg
                        );
                    }
                }
                _ => {
                    // Combined small move
                    let scale = 0.02;
                    let dx = rng.gen_range(-scale..scale);
                    let dy = rng.gen_range(-scale..scale);
                    let delta = rng.gen_range(-22.5..22.5);
                    let new_angle = (old_tree.angle_deg + delta).rem_euclid(360.0);
                    trees[idx] = PlacedTree::new(old_tree.x + dx, old_tree.y + dy, new_angle);
                }
            }

            if !has_overlap(trees, idx) {
                let new_side = compute_side_length(trees);
                if new_side < best_side - 1e-9 {
                    best_side = new_side;
                } else {
                    trees[idx] = old_tree;
                }
            } else {
                trees[idx] = old_tree;
            }
        }
    }

    /// Select tree to move with preference for boundary trees
    #[inline]
    fn select_tree_to_move(&self, trees: &[PlacedTree], rng: &mut impl Rng) -> usize {
        if trees.len() <= 2 || rng.gen::<f64>() < 0.5 {
            return rng.gen_range(0..trees.len());
        }

        // BOUNDARY SURGERY: Higher preference for boundary trees (50% vs 40%)
        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
        let eps = 0.02;

        let mut boundary_indices: Vec<usize> = Vec::new();

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

        // BOUNDARY SURGERY: 14 move types including boundary-aware moves
        let move_type = rng.gen_range(0..14);

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
                // Micro rotation (11.25 degrees)
                let delta = if rng.gen() { 11.25 } else { -11.25 };
                let new_angle = (old_angle + delta).rem_euclid(360.0);
                trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
            }
            11 => {
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
            12 => {
                // BOUNDARY SURGERY: Move toward bounding box center
                let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                let bbox_center_x = (min_x + max_x) / 2.0;
                let bbox_center_y = (min_y + max_y) / 2.0;
                let dx = bbox_center_x - old_x;
                let dy = bbox_center_y - old_y;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist > 0.01 {
                    let scale = 0.04 * (0.5 + temp);
                    trees[idx] = PlacedTree::new(
                        old_x + dx / dist * scale,
                        old_y + dy / dist * scale,
                        old_angle
                    );
                } else {
                    return false;
                }
            }
            _ => {
                // BOUNDARY SURGERY: Move away from nearest boundary
                let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                let (bx1, by1, bx2, by2) = trees[idx].bounds();

                // Find which boundary is closest
                let dist_min_x = (bx1 - min_x).abs();
                let dist_max_x = (bx2 - max_x).abs();
                let dist_min_y = (by1 - min_y).abs();
                let dist_max_y = (by2 - max_y).abs();

                let min_dist = dist_min_x.min(dist_max_x).min(dist_min_y).min(dist_max_y);
                let scale = 0.03 * (0.5 + temp);

                let (dx, dy) = if (dist_min_x - min_dist).abs() < 1e-9 {
                    (scale, 0.0)  // Move right
                } else if (dist_max_x - min_dist).abs() < 1e-9 {
                    (-scale, 0.0)  // Move left
                } else if (dist_min_y - min_dist).abs() < 1e-9 {
                    (0.0, scale)  // Move up
                } else {
                    (0.0, -scale)  // Move down
                };

                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
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

    let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
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

fn compute_bounds_slice(trees: &[PlacedTree]) -> (f64, f64, f64, f64) {
    compute_bounds(trees)
}

fn has_overlap(trees: &[PlacedTree], idx: usize) -> bool {
    for (i, tree) in trees.iter().enumerate() {
        if i != idx && trees[idx].overlaps(tree) {
            return true;
        }
    }
    false
}

/// Compute the maximum overlap amount for a tree with all others
/// Returns 0.0 if no overlap, positive value representing overlap extent
fn compute_max_overlap(trees: &[PlacedTree], idx: usize) -> f64 {
    let tree = &trees[idx];
    let (t_min_x, t_min_y, t_max_x, t_max_y) = tree.bounds();

    let mut max_overlap = 0.0;

    for (i, other) in trees.iter().enumerate() {
        if i == idx {
            continue;
        }

        // First check bounding box overlap as proxy for actual overlap
        let (o_min_x, o_min_y, o_max_x, o_max_y) = other.bounds();

        // Compute bounding box overlap
        let overlap_x = (t_max_x.min(o_max_x) - t_min_x.max(o_min_x)).max(0.0);
        let overlap_y = (t_max_y.min(o_max_y) - t_min_y.max(o_min_y)).max(0.0);

        if overlap_x > 0.0 && overlap_y > 0.0 {
            // Bounding boxes overlap, check actual polygon overlap
            if tree.overlaps(other) {
                // Use minimum of overlap dimensions as proxy for overlap amount
                let overlap_amount = overlap_x.min(overlap_y);
                max_overlap = max_overlap.max(overlap_amount);
            }
        }
    }

    max_overlap
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
    fn test_boundary_identification() {
        let packer = EvolvedPacker::default();
        let trees = vec![
            PlacedTree::new(0.0, 0.0, 0.0),
            PlacedTree::new(1.0, 0.0, 0.0),
            PlacedTree::new(0.5, 1.0, 0.0),
        ];
        let boundary_trees = packer.identify_boundary_trees(&trees);
        assert!(!boundary_trees.is_empty());
    }
}
