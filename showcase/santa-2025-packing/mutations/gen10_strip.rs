//! Evolved Packing Algorithm - Generation 10 STRIP PACKING
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
//! MUTATION STRATEGY: STRIP PACKING (Gen10)
//! Use strip packing instead of square packing:
//!
//! Key improvements from Gen6:
//! - Pack trees into horizontal strips/rows first
//! - Each strip: trees packed side by side
//! - Stack strips vertically
//! - Optimize to minimize total height
//! - Different placement order: by tree size/angle
//! - Strip-aware boundary moves
//!
//! Target: Beat Gen6's 94.14 at n=200 with strip-based packing

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

    // STRIP PACKING: New parameters
    pub strip_height_target: f64,     // Target height for each strip
    pub strip_gap: f64,               // Minimum gap between strips
    pub horizontal_bias: f64,         // Bias toward horizontal placement
    pub strip_compact_prob: f64,      // Probability of strip compaction move
    pub height_weight: f64,           // Weight for height in scoring (vs width)
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen10 STRIP PACKING: Strip-based configuration
        Self {
            search_attempts: 300,             // More attempts for strip placement
            direction_samples: 80,            // More directions for horizontal coverage
            sa_iterations: 30000,             // More iterations for strip optimization
            sa_initial_temp: 0.40,            // Lower temp for exploitation
            sa_cooling_rate: 0.99993,         // Slower cooling for convergence
            sa_min_temp: 0.000006,            // Very low minimum
            translation_scale: 0.050,         // Small moves for precision
            rotation_granularity: 45.0,       // Keep 8 angles
            center_pull_strength: 0.06,       // Moderate pull
            sa_passes: 2,                     // Keep 2 passes
            early_exit_threshold: 2000,       // More patience
            boundary_focus_prob: 0.80,        // 80% boundary focus
            // STRIP parameters
            strip_height_target: 0.95,        // Target strip height (close to tree height)
            strip_gap: 0.02,                  // Small gap between strips
            horizontal_bias: 0.65,            // Strong horizontal placement bias
            strip_compact_prob: 0.20,         // 20% chance of strip compaction
            height_weight: 1.15,              // Slightly favor reducing height over width
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

/// Strip information for tracking rows
#[derive(Clone, Debug)]
struct Strip {
    y_min: f64,
    y_max: f64,
    trees: Vec<usize>,  // Indices of trees in this strip
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

            // Place new tree using strip-aware heuristics
            let new_tree = self.find_strip_placement(&trees, n, max_n, &mut rng);
            trees.push(new_tree);

            // Run SA passes with strip-aware moves
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

    /// EVOLVED FUNCTION: Find best placement using strip packing strategy
    /// STRIP: Prefer placements that extend strips horizontally first
    fn find_strip_placement(
        &self,
        existing: &[PlacedTree],
        n: usize,
        _max_n: usize,
        rng: &mut impl Rng,
    ) -> PlacedTree {
        if existing.is_empty() {
            // Start first tree with angle that's wide horizontally
            return PlacedTree::new(0.0, 0.0, 90.0);
        }

        let mut best_tree = PlacedTree::new(0.0, 0.0, 90.0);
        let mut best_score = f64::INFINITY;

        let angles = self.select_angles(n);

        // Compute current bounds and strip info
        let (min_x, min_y, max_x, max_y) = compute_bounds(existing);
        let current_width = max_x - min_x;
        let current_height = max_y - min_y;

        // Identify current strips
        let strips = self.identify_strips(existing);

        for attempt in 0..self.config.search_attempts {
            // STRIP: Bias toward horizontal directions to extend strips
            let dir = self.select_strip_direction(n, current_width, current_height, &strips, attempt, rng);

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
                    let score = self.strip_placement_score(&candidate, existing, n, &strips);
                    if score < best_score {
                        best_score = score;
                        best_tree = candidate;
                    }
                }
            }
        }

        best_tree
    }

    /// STRIP: Identify strips (horizontal bands) in current packing
    fn identify_strips(&self, trees: &[PlacedTree]) -> Vec<Strip> {
        if trees.is_empty() {
            return Vec::new();
        }

        // Sort trees by their y-center
        let mut tree_y: Vec<(usize, f64, f64, f64)> = trees.iter().enumerate()
            .map(|(i, t)| {
                let (_, y_min, _, y_max) = t.bounds();
                let y_center = (y_min + y_max) / 2.0;
                (i, y_center, y_min, y_max)
            })
            .collect();
        tree_y.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Group into strips based on y-overlap
        let mut strips: Vec<Strip> = Vec::new();
        let merge_threshold = self.config.strip_height_target * 0.5;

        for (idx, _y_center, y_min, y_max) in tree_y {
            let mut added = false;
            for strip in &mut strips {
                // Check if this tree overlaps with the strip's y-range
                if y_max > strip.y_min - merge_threshold && y_min < strip.y_max + merge_threshold {
                    strip.y_min = strip.y_min.min(y_min);
                    strip.y_max = strip.y_max.max(y_max);
                    strip.trees.push(idx);
                    added = true;
                    break;
                }
            }
            if !added {
                strips.push(Strip {
                    y_min,
                    y_max,
                    trees: vec![idx],
                });
            }
        }

        strips
    }

    /// STRIP: Select direction with horizontal bias for strip extension
    #[inline]
    fn select_strip_direction(
        &self,
        n: usize,
        width: f64,
        height: f64,
        strips: &[Strip],
        attempt: usize,
        rng: &mut impl Rng,
    ) -> f64 {
        let strategy = rng.gen::<f64>();

        if strategy < self.config.horizontal_bias {
            // STRIP: Strongly bias toward horizontal directions to extend strips
            let horizontal_angle = if rng.gen() { 0.0 } else { PI };
            horizontal_angle + rng.gen_range(-PI / 6.0..PI / 6.0)
        } else if strategy < 0.85 {
            // Try to fill gaps within existing strips
            if !strips.is_empty() && attempt % 3 == 0 {
                let strip = &strips[attempt % strips.len()];
                let strip_center_y = (strip.y_min + strip.y_max) / 2.0;
                // Point toward strip center
                strip_center_y.atan2(if rng.gen() { 1.0 } else { -1.0 })
            } else {
                // Structured: evenly spaced with jitter
                let num_dirs = self.config.direction_samples;
                let base_idx = rng.gen_range(0..num_dirs);
                let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
                base + rng.gen_range(-0.05..0.05)
            }
        } else if strategy < 0.92 {
            // STRIP: Add new strip above or below
            if height < width * 0.9 {
                // Prefer vertical extension (new strips)
                let angle = if rng.gen() { PI / 2.0 } else { -PI / 2.0 };
                angle + rng.gen_range(-PI / 8.0..PI / 8.0)
            } else {
                // Prefer horizontal extension
                let angle = if rng.gen() { 0.0 } else { PI };
                angle + rng.gen_range(-PI / 6.0..PI / 6.0)
            }
        } else {
            // Golden angle for coverage
            let golden_angle = PI * (3.0 - (5.0_f64).sqrt());
            let base = (n as f64 * golden_angle) % (2.0 * PI);
            let offset = rng.gen_range(0..8) as f64 * PI / 4.0;
            (base + offset + rng.gen_range(-0.1..0.1)) % (2.0 * PI)
        }
    }

    /// EVOLVED FUNCTION: Score a placement (lower is better)
    /// STRIP PACKING: Score based on strip alignment and height minimization
    #[inline]
    fn strip_placement_score(
        &self,
        tree: &PlacedTree,
        existing: &[PlacedTree],
        n: usize,
        strips: &[Strip],
    ) -> f64 {
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

        // STRIP: Asymmetric scoring - penalize height more than width
        let side = width.max(height * self.config.height_weight);

        // Primary: minimize side length
        let side_score = side;

        // Secondary: balance penalty (but prefer wider over taller)
        let balance_penalty = if height > width {
            (height - width) * 0.15  // Penalize tall more
        } else {
            (width - height) * 0.05  // Mild penalty for wide
        };

        // STRIP: Bonus for aligning with existing strips
        let strip_alignment_bonus = self.calculate_strip_alignment(tree, strips);

        // STRIP: Penalize creating new strips vs extending existing
        let strip_extension_bonus = self.calculate_strip_extension_bonus(tree, existing);

        // Penalize vertical extension more than horizontal
        let (old_min_x, old_min_y, old_max_x, old_max_y) = if !existing.is_empty() {
            compute_bounds(existing)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        let x_extension = (pack_max_x - old_max_x).max(0.0) + (old_min_x - pack_min_x).max(0.0);
        let y_extension = (pack_max_y - old_max_y).max(0.0) + (old_min_y - pack_min_y).max(0.0);
        // STRIP: Penalize vertical extension more heavily
        let extension_penalty = x_extension * 0.05 + y_extension * 0.12;

        // Center penalty (mild preference for centered packing)
        let center_x = (pack_min_x + pack_max_x) / 2.0;
        let center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (center_x.abs() + center_y.abs()) * 0.004 / (n as f64).sqrt();

        // Neighbor bonus: reward being close horizontally (same strip)
        let neighbor_bonus = self.horizontal_neighbor_bonus(tree, existing);

        side_score + balance_penalty + extension_penalty + center_penalty
            - strip_alignment_bonus - strip_extension_bonus - neighbor_bonus
    }

    /// STRIP: Calculate bonus for aligning with existing strip y-coordinates
    #[inline]
    fn calculate_strip_alignment(&self, tree: &PlacedTree, strips: &[Strip]) -> f64 {
        if strips.is_empty() {
            return 0.0;
        }

        let (_, tree_min_y, _, tree_max_y) = tree.bounds();
        let tree_center_y = (tree_min_y + tree_max_y) / 2.0;

        let mut best_alignment = 0.0;
        for strip in strips {
            let strip_center_y = (strip.y_min + strip.y_max) / 2.0;
            let y_diff = (tree_center_y - strip_center_y).abs();

            // Strong bonus if tree's y-center is close to strip's y-center
            if y_diff < 0.3 {
                let alignment = 0.05 * (1.0 - y_diff / 0.3);
                best_alignment = best_alignment.max(alignment);
            }
        }

        best_alignment
    }

    /// STRIP: Bonus for extending a strip rather than starting new one
    #[inline]
    fn calculate_strip_extension_bonus(&self, tree: &PlacedTree, existing: &[PlacedTree]) -> f64 {
        if existing.is_empty() {
            return 0.0;
        }

        let (_, tree_min_y, _, tree_max_y) = tree.bounds();

        // Check if this tree's y-range overlaps with existing trees
        let mut overlap_count = 0;
        for other in existing {
            let (_, oy1, _, oy2) = other.bounds();
            // Check y-overlap
            if tree_max_y > oy1 && tree_min_y < oy2 {
                overlap_count += 1;
            }
        }

        // More overlap = better extension of existing strip
        0.01 * (overlap_count as f64).min(5.0)
    }

    /// STRIP: Bonus for being close horizontally to existing trees (same strip)
    #[inline]
    fn horizontal_neighbor_bonus(&self, tree: &PlacedTree, existing: &[PlacedTree]) -> f64 {
        if existing.is_empty() {
            return 0.0;
        }

        let (tree_min_x, tree_min_y, tree_max_x, tree_max_y) = tree.bounds();
        let tree_cx = (tree_min_x + tree_max_x) / 2.0;

        let mut bonus = 0.0;
        for other in existing {
            let (ox1, oy1, ox2, oy2) = other.bounds();

            // Check if in same y-band (same strip)
            if tree_max_y > oy1 && tree_min_y < oy2 {
                let other_cx = (ox1 + ox2) / 2.0;
                let x_dist = (tree_cx - other_cx).abs();

                // Bonus for being close horizontally within same strip
                if x_dist < 1.5 {
                    bonus += 0.015 * (1.5 - x_dist) / 1.5;
                }
            }
        }

        bonus.min(0.06)  // Cap the bonus
    }

    /// EVOLVED FUNCTION: Select rotation angles to try
    /// STRIP: Prefer angles that make trees wider (for strip packing)
    #[inline]
    fn select_angles(&self, n: usize) -> Vec<f64> {
        // 90 and 270 make tree horizontal (wider footprint)
        // 0 and 180 make tree vertical (taller footprint)
        // For strip packing, prefer horizontal orientations
        let base = match n % 4 {
            0 => vec![90.0, 270.0, 45.0, 135.0, 315.0, 225.0, 0.0, 180.0],
            1 => vec![270.0, 90.0, 135.0, 45.0, 225.0, 315.0, 180.0, 0.0],
            2 => vec![90.0, 270.0, 315.0, 225.0, 45.0, 135.0, 0.0, 180.0],
            _ => vec![270.0, 90.0, 225.0, 315.0, 135.0, 45.0, 180.0, 0.0],
        };
        base
    }

    /// EVOLVED FUNCTION: Local search with simulated annealing
    /// STRIP: Include strip compaction moves
    fn local_search(&self, trees: &mut Vec<PlacedTree>, n: usize, pass: usize, rng: &mut impl Rng) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length_strip(trees, self.config.height_weight);
        let mut best_side = current_side;
        let mut best_config: Vec<PlacedTree> = trees.clone();

        let temp_multiplier = match pass {
            0 => 1.0,
            _ => 0.35,
        };
        let mut temp = self.config.sa_initial_temp * temp_multiplier;

        let base_iterations = match pass {
            0 => self.config.sa_iterations + n * 120,
            _ => self.config.sa_iterations / 2 + n * 60,
        };

        let mut iterations_without_improvement = 0;

        // Cache boundary info
        let mut boundary_cache_iter = 0;
        let mut boundary_info: Vec<(usize, BoundaryEdge)> = Vec::new();

        for iter in 0..base_iterations {
            if iterations_without_improvement >= self.config.early_exit_threshold {
                break;
            }

            // Update boundary cache every 400 iterations
            if iter == 0 || iter - boundary_cache_iter >= 400 {
                boundary_info = self.find_boundary_trees_with_edges(trees);
                boundary_cache_iter = iter;
            }

            // STRIP: Choose between boundary optimization and strip compaction
            let do_strip_move = rng.gen::<f64>() < self.config.strip_compact_prob;

            let (idx, edge) = if do_strip_move {
                // STRIP: Try to compact within a strip
                let strips = self.identify_strips(trees);
                if !strips.is_empty() {
                    let strip = &strips[rng.gen_range(0..strips.len())];
                    if !strip.trees.is_empty() {
                        let tree_idx = strip.trees[rng.gen_range(0..strip.trees.len())];
                        (tree_idx, BoundaryEdge::None)
                    } else {
                        (rng.gen_range(0..trees.len()), BoundaryEdge::None)
                    }
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

            let success = self.strip_aware_move(trees, idx, temp, edge, do_strip_move, rng);

            if success {
                let new_side = compute_side_length_strip(trees, self.config.height_weight);
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

        if best_side < compute_side_length_strip(trees, self.config.height_weight) {
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

    /// STRIP: Move operator with strip compaction awareness
    #[inline]
    fn strip_aware_move(
        &self,
        trees: &mut [PlacedTree],
        idx: usize,
        temp: f64,
        edge: BoundaryEdge,
        is_strip_move: bool,
        rng: &mut impl Rng,
    ) -> bool {
        let old = &trees[idx];
        let old_x = old.x;
        let old_y = old.y;
        let old_angle = old.angle_deg;

        let scale = self.config.translation_scale * (0.3 + temp * 1.5);

        // STRIP: If this is a strip compaction move, focus on horizontal compression
        if is_strip_move {
            let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
            let bbox_cx = (min_x + max_x) / 2.0;

            let move_type = rng.gen_range(0..5);
            match move_type {
                0 => {
                    // Move horizontally toward center (compress strip)
                    let dx = (bbox_cx - old_x) * 0.08 * (0.5 + temp);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y, old_angle);
                }
                1 => {
                    // Small horizontal move
                    let dx = rng.gen_range(-scale * 0.6..scale * 0.6);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y, old_angle);
                }
                2 => {
                    // Rotate to potentially pack tighter in strip
                    let angles = [45.0, 90.0, -45.0, -90.0];
                    let delta = angles[rng.gen_range(0..angles.len())];
                    let new_angle = (old_angle + delta).rem_euclid(360.0);
                    trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
                }
                3 => {
                    // Small vertical adjustment within strip
                    let dy = rng.gen_range(-scale * 0.3..scale * 0.3);
                    trees[idx] = PlacedTree::new(old_x, old_y + dy, old_angle);
                }
                _ => {
                    // Combined small move
                    let dx = rng.gen_range(-scale * 0.4..scale * 0.4);
                    let dy = rng.gen_range(-scale * 0.2..scale * 0.2);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
            }
        } else {
            // Standard boundary-aware moves with strip bias
            let move_type = match edge {
                BoundaryEdge::Left => {
                    match rng.gen_range(0..10) {
                        0..=4 => 0,  // Move right (compress)
                        5..=6 => 1,
                        7..=8 => 2,
                        _ => 3,
                    }
                }
                BoundaryEdge::Right => {
                    match rng.gen_range(0..10) {
                        0..=4 => 4,  // Move left (compress)
                        5..=6 => 1,
                        7..=8 => 2,
                        _ => 3,
                    }
                }
                BoundaryEdge::Top => {
                    match rng.gen_range(0..10) {
                        0..=5 => 5,  // Move down (reduce height - important for strips!)
                        6..=7 => 6,
                        8 => 2,
                        _ => 3,
                    }
                }
                BoundaryEdge::Bottom => {
                    match rng.gen_range(0..10) {
                        0..=5 => 7,  // Move up (reduce height - important for strips!)
                        6..=7 => 6,
                        8 => 2,
                        _ => 3,
                    }
                }
                BoundaryEdge::Corner => {
                    match rng.gen_range(0..10) {
                        0..=4 => 8,  // Move toward center
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
                    // Move right
                    let dx = rng.gen_range(scale * 0.3..scale);
                    let dy = rng.gen_range(-scale * 0.15..scale * 0.15);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
                1 => {
                    // Move vertically
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
                    // Small random move
                    let dx = rng.gen_range(-scale * 0.5..scale * 0.5);
                    let dy = rng.gen_range(-scale * 0.5..scale * 0.5);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
                4 => {
                    // Move left
                    let dx = rng.gen_range(-scale..-scale * 0.3);
                    let dy = rng.gen_range(-scale * 0.15..scale * 0.15);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
                5 => {
                    // Move down (reduce height)
                    let dx = rng.gen_range(-scale * 0.15..scale * 0.15);
                    let dy = rng.gen_range(-scale..-scale * 0.3);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
                6 => {
                    // Move horizontally only
                    let dx = rng.gen_range(-scale..scale);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y, old_angle);
                }
                7 => {
                    // Move up (reduce height)
                    let dx = rng.gen_range(-scale * 0.15..scale * 0.15);
                    let dy = rng.gen_range(scale * 0.3..scale);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
                8 => {
                    // Move toward center
                    let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                    let bbox_cx = (min_x + max_x) / 2.0;
                    let bbox_cy = (min_y + max_y) / 2.0;

                    let dx = (bbox_cx - old_x) * self.config.center_pull_strength * (0.5 + temp);
                    let dy = (bbox_cy - old_y) * self.config.center_pull_strength * (0.5 + temp);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
                9 => {
                    // Diagonal move
                    let diag = rng.gen_range(-scale..scale);
                    let sign = if rng.gen() { 1.0 } else { -1.0 };
                    trees[idx] = PlacedTree::new(old_x + diag, old_y + sign * diag, old_angle);
                }
                _ => {
                    // Radial move
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

/// Compute side length with height weight for strip packing
fn compute_side_length_strip(trees: &[PlacedTree], height_weight: f64) -> f64 {
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

    let width = max_x - min_x;
    let height = max_y - min_y;

    // Apply height weight to favor reducing height
    width.max(height * height_weight)
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
