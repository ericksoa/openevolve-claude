//! Evolved Packing Algorithm - Generation 7 RING PLACEMENT
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
//! MUTATION STRATEGY: RING PLACEMENT (Gen7)
//! Place trees in concentric rings from center outward:
//!
//! Key improvements from Gen6:
//! - Spiral/ring-based placement: organize trees in concentric rings
//! - Direction selection: bias toward filling current ring before expanding
//! - Minimize "spokes": penalize placements that extend radially
//! - Score based on radial distance: prefer compact circular packing
//! - Keep density and boundary optimization from Gen6
//!
//! Target: Beat Gen6's 93.23 at n=200 with ring-aware packing

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

    // RING PLACEMENT: New parameters
    pub ring_width: f64,              // Width of each concentric ring
    pub ring_fill_bias: f64,          // Bias toward filling current ring (0-1)
    pub radial_penalty_weight: f64,   // Penalty for radial "spokes"
    pub angular_spacing_target: f64,  // Target angular spacing between trees in ring

    // Keep density parameters from Gen6
    pub density_grid_resolution: usize,
    pub gap_penalty_weight: f64,
    pub local_density_radius: f64,
    pub fill_move_prob: f64,
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        // Gen7 RING PLACEMENT: Ring-aware configuration
        Self {
            search_attempts: 300,            // More attempts for ring placement
            direction_samples: 80,           // More directions for angular coverage
            sa_iterations: 30000,            // More iterations for ring optimization
            sa_initial_temp: 0.42,           // Slightly lower for ring exploitation
            sa_cooling_rate: 0.99993,        // Slower cooling for ring convergence
            sa_min_temp: 0.000006,           // Lower minimum for fine-tuning
            translation_scale: 0.050,        // Smaller moves for ring precision
            rotation_granularity: 45.0,      // Keep 8 angles
            center_pull_strength: 0.08,      // Moderate pull toward center
            sa_passes: 2,                    // Keep 2 passes
            early_exit_threshold: 2000,      // More patience for ring moves
            boundary_focus_prob: 0.80,       // 80% boundary, 20% ring optimization
            // RING parameters
            ring_width: 0.45,                // Ring width ~half tree width
            ring_fill_bias: 0.65,            // 65% bias toward filling current ring
            radial_penalty_weight: 0.12,     // Penalty for radial extensions
            angular_spacing_target: PI / 6.0, // ~30 degrees between trees
            // Keep density parameters
            density_grid_resolution: 20,
            gap_penalty_weight: 0.12,
            local_density_radius: 0.55,
            fill_move_prob: 0.12,
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

/// RING: Track which ring a tree belongs to
#[derive(Clone, Copy, Debug)]
struct RingInfo {
    ring_index: usize,
    angle: f64,        // Angular position in ring (0 to 2*PI)
    radius: f64,       // Radial distance from center
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

            // Place new tree using ring-aware heuristics
            let new_tree = self.find_placement(&trees, n, max_n, &mut rng);
            trees.push(new_tree);

            // Run SA passes with ring-aware moves
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
    /// RING: Prefer placements that complete current ring before starting new one
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

        // Compute current bounds and center
        let (min_x, min_y, max_x, max_y) = compute_bounds(existing);
        let center_x = (min_x + max_x) / 2.0;
        let center_y = (min_y + max_y) / 2.0;
        let current_width = max_x - min_x;
        let current_height = max_y - min_y;

        // RING: Analyze current ring structure
        let ring_info = self.analyze_rings(existing, center_x, center_y);
        let (current_ring, incomplete_angles) = self.find_incomplete_ring(&ring_info);

        for attempt in 0..self.config.search_attempts {
            // RING: Sometimes target incomplete parts of current ring
            let dir = if !incomplete_angles.is_empty() && rng.gen::<f64>() < self.config.ring_fill_bias {
                // Target an incomplete angle in current ring
                let target_angle = incomplete_angles[attempt % incomplete_angles.len()];
                target_angle + rng.gen_range(-0.15..0.15)
            } else {
                self.select_direction(n, current_width, current_height, center_x, center_y, current_ring, rng)
            };

            let vx = dir.cos();
            let vy = dir.sin();

            for &tree_angle in &angles {
                // Binary search for closest valid position
                let mut low = 0.0;
                let mut high = 12.0;

                while high - low > 0.001 {
                    let mid = (low + high) / 2.0;
                    let candidate = PlacedTree::new(center_x + mid * vx, center_y + mid * vy, tree_angle);

                    if is_valid(&candidate, existing) {
                        high = mid;
                    } else {
                        low = mid;
                    }
                }

                let candidate = PlacedTree::new(center_x + high * vx, center_y + high * vy, tree_angle);
                if is_valid(&candidate, existing) {
                    let score = self.placement_score(&candidate, existing, n, center_x, center_y, current_ring);
                    if score < best_score {
                        best_score = score;
                        best_tree = candidate;
                    }
                }
            }
        }

        best_tree
    }

    /// RING: Analyze the ring structure of existing trees
    fn analyze_rings(&self, trees: &[PlacedTree], center_x: f64, center_y: f64) -> Vec<RingInfo> {
        trees.iter().map(|tree| {
            let (bx1, by1, bx2, by2) = tree.bounds();
            let tree_cx = (bx1 + bx2) / 2.0;
            let tree_cy = (by1 + by2) / 2.0;

            let dx = tree_cx - center_x;
            let dy = tree_cy - center_y;
            let radius = (dx * dx + dy * dy).sqrt();
            let angle = dy.atan2(dx);

            let ring_index = (radius / self.config.ring_width).floor() as usize;

            RingInfo { ring_index, angle, radius }
        }).collect()
    }

    /// RING: Find the current incomplete ring and its gaps
    fn find_incomplete_ring(&self, ring_info: &[RingInfo]) -> (usize, Vec<f64>) {
        if ring_info.is_empty() {
            return (0, vec![0.0, PI / 2.0, PI, -PI / 2.0]);
        }

        // Find maximum ring index
        let max_ring = ring_info.iter().map(|r| r.ring_index).max().unwrap_or(0);

        // Count trees in each ring
        let mut ring_counts: Vec<usize> = vec![0; max_ring + 2];
        for r in ring_info {
            if r.ring_index < ring_counts.len() {
                ring_counts[r.ring_index] += 1;
            }
        }

        // Find the outermost ring that isn't "full"
        // Expected trees per ring: circumference / tree_spacing
        let current_ring = max_ring;
        let expected_in_ring = ((current_ring as f64 + 0.5) * self.config.ring_width * 2.0 * PI /
            self.config.angular_spacing_target.max(0.3)).ceil() as usize;

        // Get angles already occupied in current ring
        let occupied_angles: Vec<f64> = ring_info.iter()
            .filter(|r| r.ring_index == current_ring)
            .map(|r| r.angle)
            .collect();

        // Find gap angles (incomplete parts of the ring)
        let mut incomplete_angles = Vec::new();
        let num_checks = 16;
        for i in 0..num_checks {
            let check_angle = -PI + (i as f64 / num_checks as f64) * 2.0 * PI;

            // Check if this angle is far enough from all occupied angles
            let min_dist = occupied_angles.iter()
                .map(|&occ| {
                    let diff = (check_angle - occ).abs();
                    diff.min(2.0 * PI - diff)
                })
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(PI);

            if min_dist > self.config.angular_spacing_target * 0.7 {
                incomplete_angles.push(check_angle);
            }
        }

        // If current ring is mostly full, suggest expanding
        if ring_counts.get(current_ring).unwrap_or(&0) >= &expected_in_ring.max(1) {
            // Start next ring
            let next_ring = current_ring + 1;
            return (next_ring, vec![0.0, PI / 4.0, PI / 2.0, 3.0 * PI / 4.0, PI, -3.0 * PI / 4.0, -PI / 2.0, -PI / 4.0]);
        }

        (current_ring, incomplete_angles)
    }

    /// EVOLVED FUNCTION: Score a placement (lower is better)
    /// RING: Score based on radial compactness and ring completion
    #[inline]
    fn placement_score(&self, tree: &PlacedTree, existing: &[PlacedTree], n: usize, center_x: f64, center_y: f64, current_ring: usize) -> f64 {
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
        let balance_penalty = (width - height).abs() * 0.08;

        // RING: Calculate radial distance from center
        let tree_cx = (tree_min_x + tree_max_x) / 2.0;
        let tree_cy = (tree_min_y + tree_max_y) / 2.0;
        let dx = tree_cx - center_x;
        let dy = tree_cy - center_y;
        let radius = (dx * dx + dy * dy).sqrt();

        // RING: Determine which ring this placement would be in
        let placement_ring = (radius / self.config.ring_width).floor() as usize;

        // RING: Penalty for "spoke" placements that extend radially
        // Prefer placements that stay within current ring
        let ring_penalty = if placement_ring > current_ring {
            // Penalize jumping to new ring if current is incomplete
            (placement_ring - current_ring) as f64 * self.config.radial_penalty_weight
        } else {
            0.0
        };

        // RING: Reward placements with good angular spacing from neighbors
        let angular_spacing_bonus = self.calculate_angular_spacing_bonus(tree_cx, tree_cy, center_x, center_y, existing);

        // DENSITY: Calculate local density
        let local_density = self.calculate_local_density(tree_cx, tree_cy, existing);
        let density_bonus = -self.config.gap_penalty_weight * local_density;

        // Extension penalty from Gen6
        let (old_min_x, old_min_y, old_max_x, old_max_y) = if !existing.is_empty() {
            compute_bounds(existing)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        let x_extension = (pack_max_x - old_max_x).max(0.0) + (old_min_x - pack_min_x).max(0.0);
        let y_extension = (pack_max_y - old_max_y).max(0.0) + (old_min_y - pack_min_y).max(0.0);
        let extension_penalty = (x_extension + y_extension) * 0.07;

        // RING: Penalize asymmetric extensions (creates non-circular packing)
        let asymmetry_penalty = (x_extension - y_extension).abs() * 0.05;

        // Center penalty (mild preference for centered packing)
        let new_center_x = (pack_min_x + pack_max_x) / 2.0;
        let new_center_y = (pack_min_y + pack_max_y) / 2.0;
        let center_penalty = (new_center_x.abs() + new_center_y.abs()) * 0.004 / (n as f64).sqrt();

        // Neighbor proximity bonus from Gen6
        let neighbor_bonus = self.neighbor_proximity_bonus(tree, existing);

        side_score + balance_penalty + extension_penalty + ring_penalty + asymmetry_penalty +
            center_penalty + density_bonus - neighbor_bonus - angular_spacing_bonus
    }

    /// RING: Calculate bonus for good angular spacing in current ring
    #[inline]
    fn calculate_angular_spacing_bonus(&self, tree_cx: f64, tree_cy: f64, center_x: f64, center_y: f64, trees: &[PlacedTree]) -> f64 {
        let dx = tree_cx - center_x;
        let dy = tree_cy - center_y;
        let tree_angle = dy.atan2(dx);
        let tree_radius = (dx * dx + dy * dy).sqrt();
        let tree_ring = (tree_radius / self.config.ring_width).floor() as usize;

        // Find trees in the same ring
        let mut same_ring_angles: Vec<f64> = Vec::new();
        for other in trees {
            let (ox1, oy1, ox2, oy2) = other.bounds();
            let other_cx = (ox1 + ox2) / 2.0;
            let other_cy = (oy1 + oy2) / 2.0;
            let odx = other_cx - center_x;
            let ody = other_cy - center_y;
            let other_radius = (odx * odx + ody * ody).sqrt();
            let other_ring = (other_radius / self.config.ring_width).floor() as usize;

            if other_ring == tree_ring {
                same_ring_angles.push(ody.atan2(odx));
            }
        }

        if same_ring_angles.is_empty() {
            return 0.02; // Small bonus for being first in ring
        }

        // Calculate minimum angular distance to any tree in same ring
        let min_angular_dist = same_ring_angles.iter()
            .map(|&other_angle| {
                let diff = (tree_angle - other_angle).abs();
                diff.min(2.0 * PI - diff)
            })
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(PI);

        // Bonus for being close to target spacing (not too close, not too far)
        let target = self.config.angular_spacing_target;
        if min_angular_dist >= target * 0.8 && min_angular_dist <= target * 1.5 {
            0.015 // Good spacing
        } else if min_angular_dist < target * 0.5 {
            -0.01 // Too close
        } else {
            0.005 // Acceptable
        }
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
    /// RING: Bias toward filling current ring before expanding
    #[inline]
    fn select_direction(&self, n: usize, width: f64, height: f64, _center_x: f64, _center_y: f64, current_ring: usize, rng: &mut impl Rng) -> f64 {
        let num_dirs = self.config.direction_samples;

        let strategy = rng.gen::<f64>();

        if strategy < 0.40 {
            // RING: Spiral placement - use golden angle for even angular distribution
            let golden_angle = PI * (3.0 - (5.0_f64).sqrt());
            let base = (n as f64 * golden_angle) % (2.0 * PI);
            base + rng.gen_range(-0.1..0.1)
        } else if strategy < 0.60 {
            // RING: Target specific angular positions in current ring
            let positions_per_ring = 8 + current_ring * 4;
            let target_idx = rng.gen_range(0..positions_per_ring);
            let angle = (target_idx as f64 / positions_per_ring as f64) * 2.0 * PI - PI;
            angle + rng.gen_range(-0.2..0.2)
        } else if strategy < 0.75 {
            // Structured: evenly spaced with small jitter (from Gen6)
            let base_idx = rng.gen_range(0..num_dirs);
            let base = (base_idx as f64 / num_dirs as f64) * 2.0 * PI;
            base + rng.gen_range(-0.05..0.05)
        } else if strategy < 0.88 {
            // RING: Bias toward the shorter dimension to maintain circularity
            if width < height {
                let angle = if rng.gen() { 0.0 } else { PI };
                angle + rng.gen_range(-PI / 4.0..PI / 4.0)
            } else {
                let angle = if rng.gen() { PI / 2.0 } else { -PI / 2.0 };
                angle + rng.gen_range(-PI / 4.0..PI / 4.0)
            }
        } else {
            // Corner bias (from Gen6)
            let corners = [PI / 4.0, 3.0 * PI / 4.0, 5.0 * PI / 4.0, 7.0 * PI / 4.0];
            corners[rng.gen_range(0..4)] + rng.gen_range(-0.15..0.15)
        }
    }

    /// EVOLVED FUNCTION: Local search with simulated annealing
    /// RING: Include ring-aware moves
    fn local_search(&self, trees: &mut Vec<PlacedTree>, n: usize, pass: usize, rng: &mut impl Rng) {
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

            // Choose between boundary optimization, ring optimization, and gap-filling
            let move_choice = rng.gen::<f64>();
            let do_ring_move = move_choice < 0.15;
            let do_fill_move = move_choice >= 0.15 && move_choice < 0.25;

            let (idx, edge) = if do_ring_move || do_fill_move {
                // RING: Try to optimize ring structure
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

            let success = self.ring_aware_move(trees, idx, temp, edge, do_ring_move, rng);

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

    /// RING: Move operator with ring-awareness
    #[inline]
    fn ring_aware_move(
        &self,
        trees: &mut [PlacedTree],
        idx: usize,
        temp: f64,
        edge: BoundaryEdge,
        is_ring_move: bool,
        rng: &mut impl Rng,
    ) -> bool {
        let old = &trees[idx];
        let old_x = old.x;
        let old_y = old.y;
        let old_angle = old.angle_deg;

        let scale = self.config.translation_scale * (0.3 + temp * 1.5);

        if is_ring_move {
            // RING: Ring-specific moves
            let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
            let center_x = (min_x + max_x) / 2.0;
            let center_y = (min_y + max_y) / 2.0;

            let dx_from_center = old_x - center_x;
            let dy_from_center = old_y - center_y;
            let current_radius = (dx_from_center * dx_from_center + dy_from_center * dy_from_center).sqrt();
            let current_angle = dy_from_center.atan2(dx_from_center);

            let move_type = rng.gen_range(0..5);
            match move_type {
                0 => {
                    // RING: Orbital move - slide along current ring
                    let delta_angle = rng.gen_range(-0.15..0.15) * (0.5 + temp);
                    let new_angle = current_angle + delta_angle;
                    let new_x = center_x + current_radius * new_angle.cos();
                    let new_y = center_y + current_radius * new_angle.sin();
                    trees[idx] = PlacedTree::new(new_x, new_y, old_angle);
                }
                1 => {
                    // RING: Radial move - adjust distance from center
                    let delta_r = rng.gen_range(-0.08..0.08) * (0.5 + temp);
                    let new_radius = (current_radius + delta_r).max(0.0);
                    if current_radius > 0.05 {
                        let new_x = center_x + new_radius * dx_from_center / current_radius;
                        let new_y = center_y + new_radius * dy_from_center / current_radius;
                        trees[idx] = PlacedTree::new(new_x, new_y, old_angle);
                    } else {
                        return false;
                    }
                }
                2 => {
                    // RING: Pull toward center (compacting)
                    let pull = self.config.center_pull_strength * (0.5 + temp);
                    let new_x = old_x - dx_from_center * pull;
                    let new_y = old_y - dy_from_center * pull;
                    trees[idx] = PlacedTree::new(new_x, new_y, old_angle);
                }
                3 => {
                    // Rotation to fit better in ring
                    let angles = [45.0, 90.0, -45.0, -90.0, 30.0, -30.0];
                    let delta = angles[rng.gen_range(0..angles.len())];
                    let new_angle = (old_angle + delta).rem_euclid(360.0);
                    trees[idx] = PlacedTree::new(old_x, old_y, new_angle);
                }
                _ => {
                    // Small perturbation
                    let dx = rng.gen_range(-scale * 0.4..scale * 0.4);
                    let dy = rng.gen_range(-scale * 0.4..scale * 0.4);
                    trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
                }
            }
        } else {
            // Standard boundary-aware moves (from Gen6)
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
}
