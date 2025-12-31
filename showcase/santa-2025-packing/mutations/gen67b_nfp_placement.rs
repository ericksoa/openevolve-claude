//! Evolved Packing Algorithm - Generation 67b NFP-BASED PLACEMENT
//!
//! RADICAL MUTATION: NO-FIT POLYGON (NFP) APPROACH
//!
//! Instead of binary search along rays, use NFP concepts to find
//! the optimal position that minimizes bounding box directly.
//!
//! Key insight: For each existing tree, compute positions where
//! new tree can be placed tangent to it. Then evaluate all such
//! positions to find the one minimizing bounding box.
//!
//! This is a fundamental change from "search along direction" to
//! "evaluate all tangent positions".
//!
//! Changes:
//! 1. Compute tangent positions for each existing tree
//! 2. Evaluate all tangent positions across all angles
//! 3. Pick position that minimizes bounding box increase
//!
//! Target: Break the 88 barrier with geometry-aware placement

use crate::{Packing, PlacedTree};
use rand::Rng;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub enum PlacementStrategy {
    ClockwiseSpiral,
    CounterclockwiseSpiral,
    Grid,
    Random,
    BoundaryFirst,
    ConcentricRings,
}

pub struct EvolvedConfig {
    pub search_attempts: usize,
    pub sa_iterations: usize,
    pub sa_initial_temp: f64,
    pub sa_cooling_rate: f64,
    pub sa_min_temp: f64,
    pub translation_scale: f64,
    pub center_pull_strength: f64,
    pub sa_passes: usize,
    pub early_exit_threshold: usize,
    pub boundary_focus_prob: f64,
    pub hot_restart_interval: usize,
    pub hot_restart_temp: f64,
    pub elite_pool_size: usize,
    pub compression_prob: f64,
    pub tangent_samples: usize,  // Number of tangent positions per tree
    pub angle_steps: usize,      // Number of angles to try
}

impl Default for EvolvedConfig {
    fn default() -> Self {
        Self {
            search_attempts: 150,  // Fewer ray searches, more tangent sampling
            sa_iterations: 28000,
            sa_initial_temp: 0.45,
            sa_cooling_rate: 0.99993,
            sa_min_temp: 0.00001,
            translation_scale: 0.055,
            center_pull_strength: 0.07,
            sa_passes: 2,
            early_exit_threshold: 2500,
            boundary_focus_prob: 0.85,
            hot_restart_interval: 800,
            hot_restart_temp: 0.35,
            elite_pool_size: 3,
            compression_prob: 0.20,
            tangent_samples: 16,  // Sample 16 tangent positions per neighbor
            angle_steps: 12,      // 12 angles = 30° steps
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BoundaryEdge {
    Left, Right, Top, Bottom, Corner, None,
}

pub struct EvolvedPacker {
    pub config: EvolvedConfig,
}

impl Default for EvolvedPacker {
    fn default() -> Self {
        Self { config: EvolvedConfig::default() }
    }
}

impl EvolvedPacker {
    pub fn pack_all(&self, max_n: usize) -> Vec<Packing> {
        let mut rng = rand::thread_rng();
        let mut packings: Vec<Packing> = Vec::with_capacity(max_n);

        let strategies = [
            PlacementStrategy::ClockwiseSpiral,
            PlacementStrategy::CounterclockwiseSpiral,
            PlacementStrategy::Grid,
            PlacementStrategy::Random,
            PlacementStrategy::BoundaryFirst,
            PlacementStrategy::ConcentricRings,
        ];

        let mut strategy_trees: Vec<Vec<PlacedTree>> = vec![Vec::new(); strategies.len()];

        for n in 1..=max_n {
            let mut best_trees: Option<Vec<PlacedTree>> = None;
            let mut best_side = f64::INFINITY;

            for (s_idx, &strategy) in strategies.iter().enumerate() {
                let mut trees = strategy_trees[s_idx].clone();
                let new_tree = self.find_placement_nfp(&trees, n, strategy, &mut rng);
                trees.push(new_tree);

                for pass in 0..self.config.sa_passes {
                    self.local_search(&mut trees, n, pass, &mut rng);
                }

                let side = compute_side_length(&trees);
                strategy_trees[s_idx] = trees.clone();

                if side < best_side {
                    best_side = side;
                    best_trees = Some(trees);
                }
            }

            let best = best_trees.unwrap();
            let mut packing = Packing::new();
            for t in &best {
                packing.trees.push(t.clone());
            }
            packings.push(packing);

            for strat_trees in strategy_trees.iter_mut() {
                if compute_side_length(strat_trees) > best_side * 1.02 {
                    *strat_trees = best.clone();
                }
            }
        }

        packings
    }

    /// NFP-inspired placement: find positions tangent to existing trees
    fn find_placement_nfp(
        &self,
        existing: &[PlacedTree],
        n: usize,
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
                PlacementStrategy::ConcentricRings => 45.0,
            };
            return PlacedTree::new(0.0, 0.0, initial_angle);
        }

        let mut best_tree = PlacedTree::new(0.0, 0.0, 0.0);
        let mut best_score = f64::INFINITY;

        // Generate angles to try
        let angles: Vec<f64> = (0..self.config.angle_steps)
            .map(|i| (i as f64 / self.config.angle_steps as f64) * 360.0)
            .collect();

        let (min_x, min_y, max_x, max_y) = compute_bounds(existing);
        let center_x = (min_x + max_x) / 2.0;
        let center_y = (min_y + max_y) / 2.0;

        // For each existing tree, try placing tangent to it
        for pivot_tree in existing.iter() {
            let (px1, py1, px2, py2) = pivot_tree.bounds();
            let pivot_cx = (px1 + px2) / 2.0;
            let pivot_cy = (py1 + py2) / 2.0;

            // Sample positions around this tree
            for sample_idx in 0..self.config.tangent_samples {
                let theta = (sample_idx as f64 / self.config.tangent_samples as f64) * 2.0 * PI;

                // Try different distances (find closest valid)
                for &angle in &angles {
                    // Binary search for valid tangent distance
                    let dir_x = theta.cos();
                    let dir_y = theta.sin();

                    let mut low = 0.3;  // Min reasonable distance
                    let mut high = 2.0;  // Max reasonable distance

                    while high - low > 0.005 {
                        let mid = (low + high) / 2.0;
                        let cand_x = pivot_cx + dir_x * mid;
                        let cand_y = pivot_cy + dir_y * mid;
                        let candidate = PlacedTree::new(cand_x, cand_y, angle);

                        if is_valid(&candidate, existing) {
                            high = mid;
                        } else {
                            low = mid;
                        }
                    }

                    let cand_x = pivot_cx + dir_x * high;
                    let cand_y = pivot_cy + dir_y * high;
                    let candidate = PlacedTree::new(cand_x, cand_y, angle);

                    if is_valid(&candidate, existing) {
                        let score = self.placement_score_nfp(&candidate, existing, center_x, center_y, n);
                        if score < best_score {
                            best_score = score;
                            best_tree = candidate;
                        }
                    }
                }
            }
        }

        // Also try standard ray-based placement from center
        for attempt in 0..self.config.search_attempts {
            let dir = self.select_direction(n, strategy, attempt, rng);
            let vx = dir.cos();
            let vy = dir.sin();

            for &angle in &angles {
                let mut low = 0.0;
                let mut high = 12.0;

                while high - low > 0.001 {
                    let mid = (low + high) / 2.0;
                    let candidate = PlacedTree::new(mid * vx, mid * vy, angle);

                    if is_valid(&candidate, existing) {
                        high = mid;
                    } else {
                        low = mid;
                    }
                }

                let candidate = PlacedTree::new(high * vx, high * vy, angle);
                if is_valid(&candidate, existing) {
                    let score = self.placement_score_nfp(&candidate, existing, center_x, center_y, n);
                    if score < best_score {
                        best_score = score;
                        best_tree = candidate;
                    }
                }
            }
        }

        best_tree
    }

    fn select_direction(&self, n: usize, strategy: PlacementStrategy, attempt: usize, rng: &mut impl Rng) -> f64 {
        match strategy {
            PlacementStrategy::ClockwiseSpiral => {
                let golden_angle = PI * (3.0 - (5.0_f64).sqrt());
                let base = (n as f64 * golden_angle) % (2.0 * PI);
                let offset = (attempt as f64 / self.config.search_attempts as f64) * 2.0 * PI;
                (base + offset) % (2.0 * PI)
            }
            PlacementStrategy::CounterclockwiseSpiral => {
                let golden_angle = -PI * (3.0 - (5.0_f64).sqrt());
                let base = (n as f64 * golden_angle).rem_euclid(2.0 * PI);
                let offset = (attempt as f64 / self.config.search_attempts as f64) * 2.0 * PI;
                (base - offset).rem_euclid(2.0 * PI)
            }
            PlacementStrategy::Grid => {
                let num_dirs = 16;
                let base_idx = attempt % num_dirs;
                (base_idx as f64 / num_dirs as f64) * 2.0 * PI
            }
            PlacementStrategy::Random => rng.gen_range(0.0..2.0 * PI),
            PlacementStrategy::BoundaryFirst => {
                let corners = [PI / 4.0, 3.0 * PI / 4.0, 5.0 * PI / 4.0, 7.0 * PI / 4.0];
                corners[attempt % 4] + rng.gen_range(-0.2..0.2)
            }
            PlacementStrategy::ConcentricRings => {
                let ring = ((n as f64).sqrt() as usize).max(1);
                let trees_in_ring = (ring * 6).max(1);
                let position_in_ring = n % trees_in_ring;
                let base_angle = (position_in_ring as f64 / trees_in_ring as f64) * 2.0 * PI;
                let offset = (attempt as f64 / self.config.search_attempts as f64) * 0.5 * PI;
                (base_angle + offset).rem_euclid(2.0 * PI)
            }
        }
    }

    /// NFP-aware scoring: emphasize tight packing
    fn placement_score_nfp(
        &self,
        tree: &PlacedTree,
        existing: &[PlacedTree],
        center_x: f64,
        center_y: f64,
        n: usize,
    ) -> f64 {
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

        // Primary: minimize side
        let side_score = side;

        // Secondary: prefer square aspect ratio
        let balance_penalty = (width - height).abs() * 0.08;

        // Tertiary: prefer positions closer to center
        let tree_cx = (tree_min_x + tree_max_x) / 2.0;
        let tree_cy = (tree_min_y + tree_max_y) / 2.0;
        let dist_to_center = ((tree_cx - center_x).powi(2) + (tree_cy - center_y).powi(2)).sqrt();
        let center_penalty = dist_to_center * 0.02 / (n as f64).sqrt();

        // Bonus: close to neighbors (tight packing)
        let mut min_neighbor_dist = f64::INFINITY;
        for other in existing {
            let (ox1, oy1, ox2, oy2) = other.bounds();
            let other_cx = (ox1 + ox2) / 2.0;
            let other_cy = (oy1 + oy2) / 2.0;
            let dist = ((tree_cx - other_cx).powi(2) + (tree_cy - other_cy).powi(2)).sqrt();
            min_neighbor_dist = min_neighbor_dist.min(dist);
        }
        let neighbor_bonus = if min_neighbor_dist < 1.0 { 0.03 * (1.0 - min_neighbor_dist) } else { 0.0 };

        side_score + balance_penalty + center_penalty - neighbor_bonus
    }

    fn local_search(
        &self,
        trees: &mut Vec<PlacedTree>,
        n: usize,
        pass: usize,
        rng: &mut impl Rng,
    ) {
        if trees.len() <= 1 {
            return;
        }

        let mut current_side = compute_side_length(trees);
        let mut best_side = current_side;
        let mut best_config: Vec<PlacedTree> = trees.clone();

        let mut elite_pool: Vec<(f64, Vec<PlacedTree>)> = vec![(current_side, trees.clone())];

        let temp_multiplier = match pass { 0 => 1.0, _ => 0.35 };
        let mut temp = self.config.sa_initial_temp * temp_multiplier;

        let base_iterations = match pass {
            0 => self.config.sa_iterations + n * 100,
            _ => self.config.sa_iterations / 2 + n * 50,
        };

        let mut iterations_without_improvement = 0;
        let mut total_restarts = 0;
        let max_restarts = 4;

        let mut boundary_cache_iter = 0;
        let mut boundary_info: Vec<(usize, BoundaryEdge)> = Vec::new();

        for iter in 0..base_iterations {
            if iterations_without_improvement >= self.config.hot_restart_interval && total_restarts < max_restarts {
                let elite_idx = rng.gen_range(0..elite_pool.len());
                *trees = elite_pool[elite_idx].1.clone();
                current_side = elite_pool[elite_idx].0;
                temp = self.config.hot_restart_temp;
                iterations_without_improvement = 0;
                total_restarts += 1;
                boundary_cache_iter = 0;
            }

            if iterations_without_improvement >= self.config.early_exit_threshold && total_restarts >= max_restarts {
                break;
            }

            if iter == 0 || iter - boundary_cache_iter >= 300 {
                boundary_info = self.find_boundary_trees_with_edges(trees);
                boundary_cache_iter = iter;
            }

            let do_compression = rng.gen::<f64>() < self.config.compression_prob;

            if do_compression {
                let old_trees = trees.clone();
                let success = self.compression_move(trees, rng);

                if success {
                    let new_side = compute_side_length(trees);
                    let delta = new_side - current_side;

                    if delta <= 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                        current_side = new_side;
                        if current_side < best_side {
                            best_side = current_side;
                            best_config = trees.clone();
                            iterations_without_improvement = 0;
                            self.update_elite_pool(&mut elite_pool, current_side, trees.clone());
                        } else {
                            iterations_without_improvement += 1;
                        }
                    } else {
                        *trees = old_trees;
                        iterations_without_improvement += 1;
                    }
                } else {
                    *trees = old_trees;
                    iterations_without_improvement += 1;
                }
            } else {
                let (idx, edge) = if !boundary_info.is_empty() && rng.gen::<f64>() < self.config.boundary_focus_prob {
                    let bi = &boundary_info[rng.gen_range(0..boundary_info.len())];
                    (bi.0, bi.1)
                } else {
                    (rng.gen_range(0..trees.len()), BoundaryEdge::None)
                };

                let old_tree = trees[idx].clone();
                let success = self.sa_move(trees, idx, temp, edge, rng);

                if success {
                    let new_side = compute_side_length(trees);
                    let delta = new_side - current_side;

                    if delta <= 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                        current_side = new_side;
                        if current_side < best_side {
                            best_side = current_side;
                            best_config = trees.clone();
                            iterations_without_improvement = 0;
                            self.update_elite_pool(&mut elite_pool, current_side, trees.clone());
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
            }

            temp = (temp * self.config.sa_cooling_rate).max(self.config.sa_min_temp);
        }

        if best_side < compute_side_length(trees) {
            *trees = best_config;
        }
    }

    fn compression_move(&self, trees: &mut [PlacedTree], rng: &mut impl Rng) -> bool {
        if trees.is_empty() {
            return false;
        }

        let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
        let center_x = (min_x + max_x) / 2.0;
        let center_y = (min_y + max_y) / 2.0;

        let idx = if rng.gen::<f64>() < 0.7 {
            let mut max_dist = 0.0;
            let mut max_idx = 0;
            for (i, tree) in trees.iter().enumerate() {
                let dx = tree.x - center_x;
                let dy = tree.y - center_y;
                let dist = dx * dx + dy * dy;
                if dist > max_dist {
                    max_dist = dist;
                    max_idx = i;
                }
            }
            max_idx
        } else {
            rng.gen_range(0..trees.len())
        };

        let old_x = trees[idx].x;
        let old_y = trees[idx].y;
        let old_angle = trees[idx].angle_deg;

        let dx = center_x - old_x;
        let dy = center_y - old_y;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist < 0.01 {
            return false;
        }

        let compression_factor = rng.gen_range(0.02..0.08);
        let new_x = old_x + dx * compression_factor;
        let new_y = old_y + dy * compression_factor;

        trees[idx] = PlacedTree::new(new_x, new_y, old_angle);

        !has_overlap(trees, idx)
    }

    fn update_elite_pool(&self, pool: &mut Vec<(f64, Vec<PlacedTree>)>, score: f64, config: Vec<PlacedTree>) {
        let mut dominated = false;
        for (elite_score, _) in pool.iter() {
            if *elite_score <= score {
                dominated = true;
                break;
            }
        }

        if !dominated {
            pool.push((score, config));
            pool.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            pool.truncate(self.config.elite_pool_size);
        } else if pool.len() < self.config.elite_pool_size {
            pool.push((score, config));
        }
    }

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

    fn sa_move(
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

        let scale = self.config.translation_scale * (0.3 + temp * 1.5);

        let move_type = match edge {
            BoundaryEdge::Left => 0,
            BoundaryEdge::Right => 1,
            BoundaryEdge::Top => 2,
            BoundaryEdge::Bottom => 3,
            BoundaryEdge::Corner => 4,
            BoundaryEdge::None => rng.gen_range(0..6),
        };

        match move_type {
            0 => {
                let dx = rng.gen_range(scale * 0.3..scale);
                let dy = rng.gen_range(-scale * 0.2..scale * 0.2);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            1 => {
                let dx = rng.gen_range(-scale..-scale * 0.3);
                let dy = rng.gen_range(-scale * 0.2..scale * 0.2);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            2 => {
                let dx = rng.gen_range(-scale * 0.2..scale * 0.2);
                let dy = rng.gen_range(-scale..-scale * 0.3);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            3 => {
                let dx = rng.gen_range(-scale * 0.2..scale * 0.2);
                let dy = rng.gen_range(scale * 0.3..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            4 => {
                let (min_x, min_y, max_x, max_y) = compute_bounds(trees);
                let bbox_cx = (min_x + max_x) / 2.0;
                let bbox_cy = (min_y + max_y) / 2.0;
                let dx = (bbox_cx - old_x) * self.config.center_pull_strength * (0.5 + temp);
                let dy = (bbox_cy - old_y) * self.config.center_pull_strength * (0.5 + temp);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
            5 => {
                // Angle adjustment with small translation
                let angle_delta = rng.gen_range(-15.0..15.0);
                let new_angle = (old_angle + angle_delta).rem_euclid(360.0);
                let dx = rng.gen_range(-scale * 0.3..scale * 0.3);
                let dy = rng.gen_range(-scale * 0.3..scale * 0.3);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, new_angle);
            }
            _ => {
                let dx = rng.gen_range(-scale..scale);
                let dy = rng.gen_range(-scale..scale);
                trees[idx] = PlacedTree::new(old_x + dx, old_y + dy, old_angle);
            }
        }

        !has_overlap(trees, idx)
    }
}

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
